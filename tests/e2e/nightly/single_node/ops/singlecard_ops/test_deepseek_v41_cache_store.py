# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate V4.1 fused stores against the PyTorch cache-write reference."""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.deepseek_v41_cache_utils import allocate_cache_views, make_cache_config
from vllm_ascend.attention.dsa_v41 import fused_scatter_cache, scatter_cache
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


@pytest.mark.parametrize(
    "name,rows,width,dtype",
    [
        ("model.layers.3.self_attn.swa_cache", 128, 512, torch.bfloat16),
        ("model.layers.2.self_attn.long_kv_cache", 64, 512, torch.bfloat16),
        ("model.layers.20.self_attn.long_kv_cache", 128, 512, torch.bfloat16),
    ],
)
def test_fused_store_matches_reference_in_layer_slots(name, rows, width, dtype):
    torch.manual_seed(47)
    config = make_cache_config(7)
    expected_backing, expected = allocate_cache_views(config, "npu")
    actual_backing, actual = allocate_cache_views(config, "npu")
    slots = torch.tensor(
        [-1, rows + 3, 5 * rows + rows - 1],
        dtype=torch.int64,
        device="npu",
    )
    values = torch.randn(3, width, dtype=dtype, device="npu")

    scatter_cache(expected[name], slots, values)
    fused_scatter_cache(actual[name], slots, values)
    torch.npu.synchronize()

    for expected_raw, actual_raw in zip(expected_backing, actual_backing):
        torch.testing.assert_close(actual_raw.cpu(), expected_raw.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["random", "zero", "tiny"])
def test_indexer_dynamic_quant_and_fused_store_match_reference(kind):
    torch.manual_seed(53)
    config = make_cache_config(7)
    expected_backing, expected = allocate_cache_views(config, "npu")
    actual_backing, actual = allocate_cache_views(config, "npu")
    name = "model.layers.2.self_attn.indexer.k_cache"
    expected_key, expected_scale = expected[name]
    actual_key, actual_scale = actual[name]
    rows = expected_key.shape[1]
    slots = torch.tensor(
        [-1, rows + 1, 3 * rows + rows - 1],
        dtype=torch.int64,
        device="npu",
    )
    key = torch.randn(3, 128, dtype=torch.bfloat16, device="npu")
    if kind == "zero":
        key.zero_()
    elif kind == "tiny":
        key.mul_(1e-7)

    reference_scale = key.float().abs().amax(-1, keepdim=True).clamp_min_(1e-12) / 127.0
    reference_key = (
        (key.float() / reference_scale)
        .round_()
        .clamp_(-127, 127)
        .to(torch.int8)
    )
    actual_quant, actual_quant_scale = torch_npu.npu_dynamic_quant(
        key, dst_type=torch.int8
    )
    torch.testing.assert_close(actual_quant.cpu(), reference_key.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(
        actual_quant_scale.float().cpu(),
        reference_scale.squeeze(-1).cpu(),
        rtol=1e-5,
        atol=1e-8,
    )

    scatter_cache(expected_key, slots, reference_key)
    scatter_cache(expected_scale, slots, reference_scale.to(torch.float16))
    fused_scatter_cache(actual_key, slots, actual_quant)
    fused_scatter_cache(
        actual_scale,
        slots,
        actual_quant_scale.unsqueeze(-1).to(torch.float16),
    )
    torch.npu.synchronize()

    for expected_raw, actual_raw in zip(expected_backing, actual_backing):
        torch.testing.assert_close(actual_raw.cpu(), expected_raw.cpu(), rtol=0, atol=0)
