import torch
from typing import NamedTuple, get_args, cast

import vllm
import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.selector import _cached_get_attn_backend

class AttentionSelectorConfig(NamedTuple):
    head_size: int
    dtype: torch.dtype
    kv_cache_dtype: CacheDType | None
    block_size: int | None
    use_mla: bool = False
    has_sink: bool = False
    use_compress: bool = False
    use_sparse: bool = False
    use_mm_prefix: bool = False
    use_per_head_quant_scales: bool = False
    attn_type: str = AttentionType.DECODER
    use_non_causal: bool = False
    use_batch_invariant: bool = False

    def __repr__(self):
        return (f"AttentionSelectorConfig(head_size={self.head_size}, "
                f"dtype={self.dtype}, "
                f"kv_cache_dtype={self.kv_cache_dtype}, "
                f"block_size={self.block_size}, "
                f"use_mla={self.use_mla}, "
                f"has_sink={self.has_sink}, "
                f"use_compress={self.use_compress}, "
                f"use_sparse={self.use_sparse}, "
                f"use_mm_prefix={self.use_mm_prefix}, "
                f"use_per_head_quant_scales={self.use_per_head_quant_scales}, "
                f"attn_type={self.attn_type}, "
                f"use_non_causal={self.use_non_causal}, "
                f"use_batch_invariant={self.use_batch_invariant})")


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    block_size: int | None = None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_compress: bool = False,
    use_sparse: bool = False,
    use_mm_prefix: bool = False,
    use_per_head_quant_scales: bool = False,
    attn_type: str | None = None,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""

    if kv_cache_dtype is not None:
        valid_cache_dtypes = get_args(CacheDType)
        assert kv_cache_dtype in valid_cache_dtypes, (
            f"Invalid kv_cache_dtype: {kv_cache_dtype}. "
            f"Valid values are: {valid_cache_dtypes}")

    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    backend_enum = vllm_config.attention_config.backend
    speculative_config = vllm_config.speculative_config
    use_non_causal = (speculative_config is not None
                      and speculative_config.method == "dflash")

    attn_selector_config = AttentionSelectorConfig(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=cast(CacheDType | None, kv_cache_dtype),
        block_size=block_size,
        use_mla=use_mla,
        has_sink=has_sink,
        use_compress=use_compress,
        use_sparse=use_sparse,
        use_mm_prefix=use_mm_prefix,
        use_per_head_quant_scales=use_per_head_quant_scales,
        attn_type=attn_type or AttentionType.DECODER,
        use_non_causal=use_non_causal,
        use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
    )

    return _cached_get_attn_backend(
        backend=backend_enum,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )


vllm.v1.attention.selector.AttentionSelectorConfig = AttentionSelectorConfig
vllm.v1.attention.selector.get_attn_backend = get_attn_backend
