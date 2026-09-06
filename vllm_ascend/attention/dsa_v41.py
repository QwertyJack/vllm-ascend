# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 cache metadata. Sparse-attention execution is a separate next step."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata, AttentionMetadataBuilder

from vllm_ascend.core.deepseek_v41 import DeepseekV41CompressorStateSpec


@dataclass
class DeepseekV41Metadata(AttentionMetadata):
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    compress_ratio: int
    storage_block_size: int
    is_compressor_state: bool


def compressed_slot_mapping(slot_mapping: torch.Tensor, ratio: int) -> torch.Tensor:
    """Convert original-token physical slots to completed compressed slots.

    Logical block sizes must be divisible by ratio. Negative/padded slots and
    incomplete compression groups never produce a write.
    """
    if ratio not in (1, 2):
        raise ValueError("V4.1 only supports ratio 1 or 2")
    valid = (slot_mapping >= 0) & ((slot_mapping + 1) % ratio == 0)
    return torch.where(valid, slot_mapping // ratio, -1)


def scatter_cache(cache: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> None:
    """Write valid rows into one V4.1 paged cache using ordinary tensor ops."""
    cache = cache.squeeze(-2)
    slots = slots[: values.shape[0]].long()
    valid = slots >= 0
    if valid.any():
        cache.view(-1, cache.shape[-1]).index_copy_(
            0, slots[valid], values[valid].to(cache.dtype)
        )


def paged_prefix(cache, block_table, length, block_size):
    """Materialize one request's logical prefix from a paged cache."""
    if length <= 0:
        return cache.new_empty((0, cache.shape[-1]))
    blocks = (length + block_size - 1) // block_size
    page_ids = block_table[:blocks].long()
    return cache.squeeze(-2).index_select(0, page_ids).flatten(0, 1)[:length]


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    """Return the level-one candidate-position mask used by V4.1.

    A block is scored by its best position.  The newest, partially populated
    block is pinned so recent compressed tokens cannot be dropped merely
    because their block has fewer populated positions.
    """
    width = logits.shape[-1]
    if width == 0:
        return torch.zeros_like(logits, dtype=torch.bool)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(-1)
    num_blocks = scores.shape[-1]
    if not torch.is_tensor(compress_lens):
        compress_lens = torch.tensor(compress_lens, device=logits.device)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last,
        torch.inf,
    )
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def select_index_topk(logits, compress_lens, index_topk):
    """Level-two position TopK, sorted back into chronological order."""
    width = logits.shape[-1]
    if width == 0:
        return torch.empty(
            (*logits.shape[:-1], 0), dtype=torch.int32, device=logits.device
        )
    topk = min(index_topk, width)
    indices = logits.topk(topk, dim=-1, sorted=False).indices.sort(-1).values
    return torch.where(indices < compress_lens, indices, -1).int()


def small_op_attention(
    q,
    positions,
    swa_cache,
    swa_metadata,
    *,
    source_cache=None,
    source_metadata=None,
    compress_ratio=0,
    window_size=128,
    index_topk=512,
    compressed_indices=None,
    sinks=None,
    softmax_scale=1.0,
):
    """Unfused eager sparse attention over SWA plus a shared compressed KV.

    ``compressed_indices`` contains the real Indexer result in compressed-KV
    coordinates.  When absent, the bounded newest-row selection remains as a
    diagnostic fallback for local-only tests.
    """
    query_starts = swa_metadata.query_start_loc.tolist()
    seq_lens = swa_metadata.seq_lens.tolist()
    outputs = []
    for req_idx, (q_start, q_end) in enumerate(zip(query_starts[:-1], query_starts[1:])):
        seq_len = int(seq_lens[req_idx])
        local = paged_prefix(
            swa_cache,
            swa_metadata.block_table[req_idx],
            seq_len,
            swa_metadata.storage_block_size,
        )
        compressed = None
        if source_cache is not None:
            compressed_len = seq_len // compress_ratio
            compressed = paged_prefix(
                source_cache,
                source_metadata.block_table[req_idx],
                compressed_len,
                source_metadata.storage_block_size,
            )
        for token_idx in range(q_start, q_end):
            position = int(positions[token_idx])
            local_start = max(0, position - window_size + 1)
            keys = local[local_start : position + 1]
            if compressed is not None:
                visible = (position + 1) // compress_ratio
                if compressed_indices is None:
                    selected = compressed[max(0, visible - index_topk) : visible]
                else:
                    indices = compressed_indices[token_idx].long()
                    indices = indices[(indices >= 0) & (indices < visible)]
                    selected = compressed.index_select(0, indices)
                keys = torch.cat((keys, selected))
            logits = torch.einsum("hd,kd->hk", q[token_idx].float(), keys.float())
            logits *= softmax_scale
            if sinks is not None:
                logits = torch.cat((logits, sinks.float().unsqueeze(-1)), -1)
                probs = logits.softmax(-1)[..., :-1]
            else:
                probs = logits.softmax(-1)
            outputs.append(torch.einsum("hk,kd->hd", probs, keys.float()))
    return torch.stack(outputs).to(q.dtype)


class DeepseekV41MetadataBuilder(AttentionMetadataBuilder[DeepseekV41Metadata]):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        if common_prefix_len:
            raise NotImplementedError("V4.1 prefix caching is not implemented")
        spec = self.kv_cache_spec
        common = common_attn_metadata
        is_compressor_state = isinstance(spec, DeepseekV41CompressorStateSpec)
        ratio = getattr(spec, "compress_ratio", 1)
        # State rows use ordinary SWA token slots, not compressed or circular slots.
        slots = common.slot_mapping if is_compressor_state else compressed_slot_mapping(common.slot_mapping, ratio)
        return DeepseekV41Metadata(
            common.block_table_tensor,
            common.query_start_loc,
            common.seq_lens,
            slots,
            ratio,
            spec.storage_block_size,
            is_compressor_state,
        )


class DeepseekV41CacheBackend(AttentionBackend):
    """Cache-only backend: supplies layout and metadata, not an AttentionImpl."""

    @staticmethod
    def get_name():
        return "ASCEND_DSA_V41_CACHE"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("V4.1 sparse-attention execution is not implemented yet")

    @staticmethod
    def get_builder_cls():
        return DeepseekV41MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
        return num_blocks, block_size, num_kv_heads, head_size


class DeepseekV41CacheLayer(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(self, vllm_config, prefix, spec):
        super().__init__()
        self.prefix = prefix
        self.spec = spec
        self.kv_cache = [torch.empty(0)]
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate V4.1 cache prefix: {prefix}")
        context[prefix] = self

    def get_kv_cache_spec(self, vllm_config):
        return self.spec

    def get_attn_backend(self):
        return DeepseekV41CacheBackend
