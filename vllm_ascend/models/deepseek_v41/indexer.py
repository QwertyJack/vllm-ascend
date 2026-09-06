# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness-first V4.1 Indexer and two-level sparse selection."""

import torch
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheLayer,
    paged_prefix,
    scatter_cache,
    select_candidate_blocks,
    select_index_topk,
)
from vllm_ascend.core.deepseek_v41 import DeepseekV41FullSpec

from .compressor import DeepseekV41RMSNorm, _read


class DeepseekV41Indexer(nn.Module):
    """Small side attention that selects compressed KV positions.

    All index heads are replicated on each TP rank for the correctness path,
    so every rank produces identical sparse indices without an all-reduce.
    """

    def __init__(
        self,
        config,
        owns_k,
        vllm_config,
        prefix,
        compress_ratio,
        quant_config=None,
    ):
        super().__init__()
        self.owns_k = owns_k
        self.compress_ratio = compress_ratio
        self.n_heads = int(_read(config, "index_n_heads"))
        self.width = int(_read(config, "index_head_dim"))
        self.rope_width = int(_read(config, "qk_rope_head_dim"))
        self.index_topk = int(_read(config, "index_topk"))
        self.softmax_scale = self.width**-0.5
        self.wq_b = ReplicatedLinear(
            _read(config, "q_lora_rank"),
            self.n_heads * self.width,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
        )
        self.weights_proj = ReplicatedLinear(
            _read(config, "hidden_size"),
            self.n_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
            return_bias=False,
        )
        if owns_k:
            self.wk = nn.Linear(
                _read(config, "head_dim"),
                self.width,
                bias=False,
                dtype=torch.bfloat16,
            )
            self.k_norm = DeepseekV41RMSNorm(
                self.width, _read(config, "rms_norm_eps")
            )
            self.k_cache = DeepseekV41CacheLayer(
                vllm_config,
                f"{prefix}.k_cache",
                DeepseekV41FullSpec(
                    block_size=vllm_config.cache_config.block_size,
                    num_kv_heads=1,
                    head_size=self.width,
                    dtype=torch.bfloat16,
                    compress_ratio=compress_ratio,
                ),
            )

    @staticmethod
    def _output(linear, value):
        output = linear(value)
        return output[0] if isinstance(output, tuple) else output

    def update_keys(self, latent, slots, cos, sin):
        """Publish source-owned index K before latent is RoPE'd as long KV."""
        if not self.owns_k or latent.shape[0] == 0:
            return
        key = self.k_norm(self.wk(latent)).view(-1, 1, self.width)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            key.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.width - self.rope_width, self.width],
        )
        scatter_cache(
            self.k_cache.kv_cache[0],
            slots,
            key.squeeze(1),
        )

    def select(
        self,
        hidden_states,
        qr,
        positions,
        cos,
        sin,
        source_cache,
        source_metadata,
        *,
        is_candidate_source,
        uses_candidate_filter,
        candidate_topk_blocks,
        candidate_block_size,
        candidates,
    ):
        """Score index K, optionally filter blocks, then return position TopK."""
        query = self._output(self.wq_b, qr).unflatten(
            -1, (self.n_heads, self.width)
        )
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            query.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.width - self.rope_width, self.width],
        )
        weights = self._output(self.weights_proj, hidden_states)
        weights = weights.float() * (self.softmax_scale * self.n_heads**-0.5)

        starts = source_metadata.query_start_loc.tolist()
        seq_lens = source_metadata.seq_lens.tolist()
        selected_per_request = []
        next_candidates = [] if is_candidate_source else candidates
        for req_idx, (q_start, q_end) in enumerate(zip(starts[:-1], starts[1:])):
            compressed_len = int(seq_lens[req_idx]) // self.compress_ratio
            if compressed_len == 0:
                selected_per_request.append(
                    torch.empty(
                        (q_end - q_start, 0),
                        dtype=torch.int32,
                        device=query.device,
                    )
                )
                if is_candidate_source:
                    next_candidates.append(
                        torch.empty(
                            (q_end - q_start, 0),
                            dtype=torch.bool,
                            device=query.device,
                        )
                    )
                continue

            key = paged_prefix(
                source_cache,
                source_metadata.block_table[req_idx],
                compressed_len,
                source_metadata.storage_block_size,
            )
            score = torch.einsum(
                "qhd,kd->qhk",
                query[q_start:q_end].float(),
                key.float(),
            )
            score = (score.relu_() * weights[q_start:q_end].unsqueeze(-1)).sum(1)
            visible = (
                (positions[q_start:q_end].long() + 1) // self.compress_ratio
            ).clamp_max(compressed_len).unsqueeze(-1)
            score.masked_fill_(
                torch.arange(compressed_len, device=score.device) >= visible,
                -torch.inf,
            )

            if is_candidate_source:
                candidate = select_candidate_blocks(
                    score,
                    visible,
                    candidate_topk_blocks,
                    candidate_block_size,
                )
                next_candidates.append(candidate)
            elif uses_candidate_filter:
                if candidates is None:
                    raise RuntimeError(
                        "V4.1 candidate-filtering indexer ran before its source"
                    )
                score.masked_fill_(~candidates[req_idx], -torch.inf)

            selected_per_request.append(
                select_index_topk(score, visible, self.index_topk)
            )

        max_topk = max((item.shape[-1] for item in selected_per_request), default=0)
        padded = []
        for item in selected_per_request:
            if item.shape[-1] < max_topk:
                item = torch.nn.functional.pad(
                    item, (0, max_topk - item.shape[-1]), value=-1
                )
            padded.append(item)
        result = (
            torch.cat(padded, 0)
            if padded
            else torch.empty((0, 0), dtype=torch.int32, device=query.device)
        )
        return result, next_candidates
