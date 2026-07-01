# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft model for DeepSeek-V4 (semi-autoregressive speculative decoding).

See: qwen3_dspark.py for base architecture. This one is specialized to the DSV4 DSpark,
which reuses the target model's architecture similarly to MTP.

To implement non-causal attention, we leverage the sparse attention implementation to
include the future query tokens in the top-k indices for each query token.
"""

import typing
from collections.abc import Callable, Iterable

import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import get_draft_quant_config
from vllm.models.deepseek_v4.common.ops import fused_q_kv_rmsnorm
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.models.deepseek_v4.nvidia.ops.fp8_einsum import (
    deepseek_v4_fp8_einsum,
    deepseek_v4_sm12x_fp8_einsum_quant,
)
from vllm.v1.attention.backends.mla.sparse_mla_env import (
    triton_sparse_mla_head_block_size,
)
from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    matmul_sparse_mla_attention_with_sink,
)
from vllm.model_executor.models.utils import maybe_prefix

from .model import (
    DeepseekV4DecoderLayer,
    DeepseekV4Model,
    make_deepseek_v4_expert_params_mapping,
)
from .dspark_triton import (
    dspark_context_kv_store,
    dspark_qkv_postprocess,
    dspark_triton_attention,
)

logger = init_logger(__name__)

# MoE expert scale suffix differs by expert dtype (mirrors deepseek_v4 loaders):
# fp4 experts register ``.weight_scale``; block-fp8 experts ``.weight_scale_inv``.
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


def _spec_bool(vllm_config: VllmConfig, attr: str, default: bool = True) -> bool:
    spec_config = vllm_config.speculative_config
    return bool(getattr(spec_config, attr, default))


def _linear_output(output: torch.Tensor | tuple[torch.Tensor, object]) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def _rmsnorm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    return x_float.mul(torch.rsqrt(x_float.square().mean(-1, keepdim=True) + eps))


def _apply_rope_gptj_last(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    """Apply DS4 RoPE to the last rope_dim features.

    This mirrors the DS4 fused qnorm/kv-insert kernels' PyTorch reference:
    GPT-J/interleaved pairs, cos||sin cache layout, and RoPE on the tail of the
    head dimension.
    """
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    head_dim = x.shape[-1]
    nope_dim = head_dim - rope_dim

    cs = cos_sin_cache.index_select(0, positions.long()).to(torch.float32)
    cos = cs[..., :half]
    sin = cs[..., half:]

    rope = x[..., nope_dim:].float()
    shape = rope.shape
    rope = rope.reshape(*shape[:-1], half, 2)
    even = rope[..., 0]
    odd = rope[..., 1]

    while cos.dim() < even.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    new_even = torch.addcmul(-odd * sin, even, cos)
    new_odd = torch.addcmul(odd * cos, even, sin)
    rope_rotated = torch.stack((new_even, new_odd), dim=-1).reshape(shape)

    out = x.clone()
    out[..., nope_dim:] = rope_rotated.to(out.dtype)
    return out


class DeepSeekV4DSparkLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        dspark_layer_idx: int,
        prefix: str,
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        self.block_size = int(getattr(config, "dspark_block_size", 0))
        runtime_layer_idx = config.num_hidden_layers + dspark_layer_idx
        runtime_prefix = (
            f"{prefix}.layers.{runtime_layer_idx}"
            if prefix
            else f"layers.{runtime_layer_idx}"
        )

        self.attn = _select_dsv4_attn_cls(vllm_config)(
            vllm_config,
            prefix=f"{runtime_prefix}.attn",
            topk_indices_buffer=None,
            aux_stream_list=None,
        )
        self.ffn = DeepseekV4MoE(vllm_config, prefix=f"{runtime_prefix}.ffn")
        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )

        max_batch = max(1, int(vllm_config.scheduler_config.max_num_seqs))
        window = int(config.sliding_window)
        self.register_buffer(
            "_main_kv_cache",
            torch.zeros(
                max_batch,
                window,
                config.head_dim,
                dtype=vllm_config.model_config.dtype,
            ),
            persistent=False,
        )
        self.use_materialized_attention = _spec_bool(
            vllm_config,
            "dspark_materialized_attention",
        )
        self.use_triton_attention = _spec_bool(
            vllm_config,
            "dspark_triton_attention",
        )
        self.use_triton_qkv_postprocess = _spec_bool(
            vllm_config,
            "dspark_triton_qkv_postprocess",
        )
        self.use_triton_context_kv_store = _spec_bool(
            vllm_config,
            "dspark_triton_context_kv_store",
        )
        self.use_fused_o_proj_quant = _spec_bool(
            vllm_config,
            "dspark_fused_o_proj_quant",
        )
        self.materialized_head_block_size = triton_sparse_mla_head_block_size() or 1
        self.register_buffer(
            "_score_buffer",
            torch.empty(
                max_batch * self.block_size,
                self.attn.n_local_heads,
                window + self.block_size,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    @property
    def window_size(self) -> int:
        return int(self.attn.window_size)

    def _wo_b_output(self, wo_b_input: torch.Tensor) -> torch.Tensor:
        return _linear_output(self.attn.wo_b(wo_b_input))

        # Heads: final norm + hc_head, and the Markov head
        # Loaded from the "final" MTP layer weights (mtp.*) in the target checkpoint
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """main_x = main_norm(main_proj(concat of target aux hidden states)).

        ``aux_hidden_states`` is [T, hidden_size * len(target_layer_ids)].
        """
        return self.main_norm(self.main_proj(aux_hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        """Insert the sliding-window context KV for every draft layer.

        Mirrors the reference DSparkAttention: each layer derives its context KV
        from the SAME projected target hidden ``main_x``, via that layer's own
        ``wkv`` + ``kv_norm`` + RoPE + quant, then writes it at the
        layer's context slots.

        ``context_slot_mappings`` is a per-layer list (each entry is the context
        slot mapping for that layer's kv-cache group, since the hybrid manager may
        place draft layers in different groups). ``None`` (or a ``None`` entry)
        runs the projection to reserve workspace but writes nothing (profiling).
        """
        for i, layer in enumerate(self.layers):
            slot_mapping = (
                None if context_slot_mappings is None else context_slot_mappings[i]
            )
            attn = layer.attn
            # Optimized DSV4 MLA path: wkv part of the fused wq_a|wkv projection
            # (q_lora part discarded), then RoPE/quant/insert via the fused op.
            qr_kv, _ = attn.fused_wqa_wkv(main_x)
            kv = qr_kv[..., attn.q_lora_rank :]
            kv = attn.kv_norm(kv)
            if slot_mapping is None:
                continue
            _insert_context_kv(attn, kv, context_positions, slot_mapping)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del main_x
        block_size = self.block_size
        q, draft_kv = self._project_draft_q_kv(x, positions)
        q = q.view(batch_size, block_size, self.attn.n_local_heads, self.attn.head_dim)
        draft_kv = draft_kv.view(batch_size, block_size, self.attn.head_dim)

        cache_kv = self._main_kv_cache[:batch_size].to(draft_kv.dtype)
        if self.use_triton_attention:
            o = dspark_triton_attention(
                q,
                cache_kv,
                draft_kv,
                main_positions.long(),
                self.attn.attn_sink[: self.attn.n_local_heads],
                float(self.attn.scale),
            ).reshape(
                batch_size * block_size,
                self.attn.n_local_heads,
                self.attn.head_dim,
            )
        else:
            kv = torch.cat([cache_kv, draft_kv], dim=1)

            cache_arange = torch.arange(
                self.window_size,
                device=positions.device,
                dtype=main_positions.dtype,
            )
            valid_cache = cache_arange.unsqueeze(0) <= torch.minimum(
                main_positions.long().unsqueeze(1),
                torch.full_like(
                    main_positions.long().unsqueeze(1), self.window_size - 1
                ),
            )
            valid = torch.cat(
                [
                    valid_cache,
                    torch.ones(
                        batch_size,
                        block_size,
                        dtype=torch.bool,
                        device=positions.device,
                    ),
                ],
                dim=1,
            )
            if self.use_materialized_attention:
                q_tokens = q.reshape(
                    batch_size * block_size,
                    self.attn.n_local_heads,
                    self.attn.head_dim,
                )
                kv_tokens = (
                    kv.unsqueeze(1)
                    .expand(batch_size, block_size, kv.shape[1], kv.shape[2])
                    .reshape(batch_size * block_size, kv.shape[1], kv.shape[2])
                    .contiguous()
                )
                valid_tokens = (
                    valid.unsqueeze(1)
                    .expand(batch_size, block_size, valid.shape[1])
                    .reshape(batch_size * block_size, valid.shape[1])
                    .contiguous()
                )
                o = torch.empty_like(q_tokens)
                score_buffer = self._score_buffer[
                    : q_tokens.shape[0],
                    : self.attn.n_local_heads,
                    : kv.shape[1],
                ]
                matmul_sparse_mla_attention_with_sink(
                    q_tokens,
                    kv_tokens,
                    valid_tokens,
                    float(self.attn.scale),
                    self.attn.attn_sink,
                    o,
                    num_heads=self.attn.n_local_heads,
                    score_buffer=score_buffer,
                    head_block_size=self.materialized_head_block_size,
                )
            else:
                scores = torch.einsum("bqhd,bnd->bqhn", q.float(), kv.float())
                scores *= float(self.attn.scale)
                scores = scores.masked_fill(
                    ~valid[:, None, None, :], float("-inf")
                )

                sink = self.attn.attn_sink[: self.attn.n_local_heads].view(
                    1, 1, -1, 1
                )
                max_score = torch.maximum(
                    scores.max(dim=-1, keepdim=True).values, sink
                )
                exp_scores = torch.exp(scores - max_score).masked_fill(
                    ~valid[:, None, None, :], 0.0
                )
                denom = exp_scores.sum(dim=-1, keepdim=True) + torch.exp(
                    sink - max_score
                )
                o = torch.einsum(
                    "bqhn,bnd->bqhd", exp_scores / denom, kv.float()
                )
                o = o.to(x.dtype).reshape(
                    batch_size * block_size,
                    self.attn.n_local_heads,
                    self.attn.head_dim,
                )
        return self._o_proj(o, positions)

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        attn = self.attn
        if not self.use_fused_o_proj_quant:
            return _linear_output(attn._o_proj(o, positions))

        required_attrs = (
            "_einsum_recipe",
            "_tma_aligned_scales",
            "wo_a",
            "wo_b",
            "rotary_emb",
            "n_local_groups",
            "n_local_heads",
            "nope_head_dim",
            "rope_head_dim",
            "o_lora_rank",
        )
        if not all(hasattr(attn, attr) for attr in required_attrs):
            return _linear_output(attn._o_proj(o, positions))

        if (
            self.use_fused_o_proj_quant
            and tuple(attn._einsum_recipe) == (1, 128, 128)
            and not attn._tma_aligned_scales
        ):
            fused_output = self._o_proj_fused_quant(attn, o, positions)
            if fused_output is not None:
                return fused_output

        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            attn.rotary_emb.cos_sin_cache,
            n_groups=attn.n_local_groups,
            heads_per_group=attn.n_local_heads // attn.n_local_groups,
            nope_dim=attn.nope_head_dim,
            rope_dim=attn.rope_head_dim,
            tma_aligned_scales=attn._tma_aligned_scales,
        )
        z = torch.empty(
            (o.shape[0], attn.n_local_groups, attn.o_lora_rank),
            device=o.device,
            dtype=torch.bfloat16,
        )
        wo_a_scale = getattr(attn.wo_a, "weight_scale_inv", None)
        if wo_a_scale is None:
            wo_a_scale = attn.wo_a.weight_scale
        deepseek_v4_fp8_einsum(
            o_fp8,
            o_scale,
            attn.wo_a.weight,
            wo_a_scale,
            z,
            "bhr,hdr->bhd",
            list(attn._einsum_recipe),
        )
        wo_b_input = z.flatten(1)
        output = self._wo_b_output(wo_b_input)
        return output

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        main_x: torch.Tensor,
        main_positions: torch.Tensor,
        batch_size: int,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_norm_weight = self.attn_norm.weight.data
        attn_norm_eps = self.attn_norm.variance_epsilon
        if residual is None:
            residual = x
            post_mix, res_mix, x = mhc_pre_tilelang(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )
        else:
            residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
                x,
                residual,
                post_mix,
                res_mix,
                residual,
            )
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        # hc_head reduces the hc copies; return the PRE-norm head hidden
        hidden_states = hc_head_fused_kernel_tilelang(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        return hidden_states


def _insert_context_kv(
    attn: nn.Module,
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """RoPE + quant + paged-cache insert of (already kv_norm'd) context KV.

class DeepSeekV4DSpark(nn.Module):
    """DSpark draft model for fixed-block speculative decoding."""

    uses_query_start_loc_context_kv = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.block_size = int(getattr(config, "dspark_block_size", 0))
        self.target_layer_ids = tuple(getattr(config, "dspark_target_layer_ids", ()))
        if self.block_size <= 0:
            raise ValueError("DSpark requires dspark_block_size > 0")
        if not self.target_layer_ids:
            raise ValueError("DSpark requires dspark_target_layer_ids")

        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.num_dspark_layers = len(self.target_layer_ids)
        self.runtime_num_dspark_layers = self.num_dspark_layers
        self.max_batch = max(1, int(vllm_config.scheduler_config.max_num_seqs))
        hidden_size = int(config.hidden_size)
        dtype = vllm_config.model_config.dtype
        quant_config = get_draft_quant_config(vllm_config)
        self.dspark_aux_hidden_size = hidden_size * len(self.target_layer_ids)

class DSparkDeepseekV4ForCausalLM(nn.Module):
    # Draft weights ship in the target checkpoint (mtp.*) without embed/head, so
    # load_dspark_model always aliases the target's.
    has_own_embed_tokens = False
    has_own_lm_head = False
    # Full-vocab draft: draft ids are target ids, no remapping needed.
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.pad_shared_expert = (
            getattr(self.quant_config, "weight_block_size", None) is not None
            and not vllm_config.parallel_config.use_sequence_parallel_moe
        )
        self.model = DSparkDeepseekV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Shared with the target (aliased by the speculator's load utility).
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

        markov_rank = int(getattr(config, "dspark_markov_rank", 256))
        self.markov_w1 = VocabParallelEmbedding(
            config.vocab_size,
            markov_rank,
            prefix=f"{prefix}mtp.2.markov_head.markov_w1",
        )
        self.markov_w2 = ReplicatedLinear(
            markov_rank,
            config.vocab_size,
            bias=False,
            params_dtype=dtype,
            prefix=f"{prefix}mtp.2.markov_head.markov_w2",
            return_bias=False,
        )
        self.use_markov_inplace_add = _spec_bool(
            vllm_config,
            "dspark_markov_inplace_add",
        )
        self.confidence_head = ReplicatedLinear(
            hidden_size + markov_rank,
            1,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}mtp.2.confidence_head.proj",
            return_bias=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        # DSV4 MLA path: each draft layer's sliding-window cache is a separate
        # layer, named by its prefix.
        return [layer.attn.swa_cache_layer.prefix for layer in self.model.layers]

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.main_norm(_linear_output(self.main_proj(aux_hidden_states)))

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
        *,
        query_start_loc: torch.Tensor | None = None,
        batch_size: int | None = None,
        num_rejected_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        del context_slot_mapping
        if query_start_loc is None or context_states.numel() == 0:
            return None
        if batch_size is None:
            batch_size = int(query_start_loc.shape[0] - 1)
        main_x = (
            self.combine_hidden_states(context_states)
            if context_states.shape[-1] == self.dspark_aux_hidden_size
            else context_states
        )
        for idx, layer in enumerate(self.layers.values()):
            if idx >= self.runtime_num_dspark_layers:
                break
            layer.store_main_kv(
                main_x,
                context_positions,
                query_start_loc,
                batch_size,
                num_rejected_tokens,
            )
        return main_x

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Returns the pre-norm hc_head hidden ([T, hidden_size]).
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Base logits U_k = lm_head(norm(head_hidden))."""
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Full-vocab draft: base logits, no d2t scatter.
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids  # full-vocab: draft ids are target ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._markov_w1_embedding(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return _linear_output(self.markov_w2(markov_embed))

    def apply_dspark_markov_bias(
        self,
        base_logits: torch.Tensor,
        prev_token_ids: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        del step_idx
        markov_embed = self._markov_w1_embedding(prev_token_ids)
        markov_bias = _linear_output(self.markov_w2(markov_embed))
        if self.use_markov_inplace_add and markov_bias.dtype == base_logits.dtype:
            markov_bias.add_(base_logits)
            return markov_bias
        return base_logits + markov_bias

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the ``mtp.{0,1,2}.*`` draft weights from the target checkpoint.

        Non-mtp weights (embed/head/main layers) belong to the target model and
        are skipped here. ``embed_tokens``/``lm_head`` are aliased from the target.
        """
        first_layer = self.model.layers[0]
        use_mega_moe = first_layer.ffn.use_mega_moe
        if use_mega_moe:
            expert_mapping = make_deepseek_v4_expert_params_mapping(
                self.config.n_routed_experts
            )
        else:
            expert_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.n_routed_experts,
            )
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        # (param_name, ckpt_shard_name, shard_id) for non-expert stacked params.
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)

        for name, loaded_weight in weights:
            mapped = self._remap_dspark_name(name)
            if mapped is None:
                continue
            name = mapped

            # ``.scale`` -> per-method scale suffix.
            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix
            if ".shared_experts.w2" in name:
                name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
            if self.pad_shared_expert and ".shared_experts." in name:
                loaded_weight = DeepseekV4Model._pad_shared_expert_weight(
                    self.quant_config, name, loaded_weight
                )

            # E8M0 expert scales: keep raw exponent bytes.
            if ".experts." in name:
                if (
                    "weight_scale" in name
                    and loaded_weight.dtype == torch.float8_e8m0fnu
                ):
                    loaded_weight = loaded_weight.view(torch.uint8)
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    param = params_dict[name_mapped]
                    success = param.weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                continue

            # Stacked rules only apply to decoder-layer weights. Head-stack params
            # (main_proj/norm/hc_head/markov_head) load directly — otherwise e.g.
            # "markov_w1" would collide with the "w1" shard rule.
            is_layer_param = name.startswith("model.layers.")
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if not is_layer_param or weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(name)
                break
            else:
                if "attn_sink" in name:
                    narrow = loaded_weight[head_start:head_end]
                    params_dict[name][: narrow.shape[0]].copy_(narrow)
                    loaded_params.add(name)
                    continue
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        for layer_idx in range(self.num_dspark_layers):
            layer_prefix = f"layers.{layer_idx}."
            if not any(name.startswith(layer_prefix) for name in loaded_params):
                raise ValueError(
                    f"DSpark layer mtp.{layer_idx} weights missing from checkpoint."
                )
        self.finalize_mega_moe_weights()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def finalize_mega_moe_weights(self) -> None:
        for layer in self.layers.values():
            layer.finalize_mega_moe_weights()
