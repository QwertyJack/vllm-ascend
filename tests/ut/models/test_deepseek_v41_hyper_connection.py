# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.models.deepseek_v41.model import DeepseekV41DecoderLayer


def _layer() -> DeepseekV41DecoderLayer:
    layer = DeepseekV41DecoderLayer.__new__(DeepseekV41DecoderLayer)
    layer.hc_mult = 4
    layer.hc_sinkhorn_iters = 3
    layer.norm_eps = 1e-6
    layer.hc_eps = 1e-6
    return layer


def test_v41_hc_small_ops_support_hidden_size_5120():
    torch.manual_seed(7)
    layer = _layer()
    x = torch.randn(2, 4, 5120, dtype=torch.bfloat16)
    hc_fn = torch.randn(24, 4 * 5120, dtype=torch.float32) / 5120
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(24, dtype=torch.float32)

    y, post, comb = layer.hc_pre(x, hc_fn, hc_scale, hc_base)

    assert y.shape == (2, 5120)
    assert y.dtype == torch.bfloat16
    assert post.shape == (2, 4)
    assert post.dtype == torch.float32
    assert comb.shape == (2, 4, 4)
    assert comb.dtype == torch.float32
    torch.testing.assert_close(comb.sum(-2), torch.ones(2, 4), atol=2e-5, rtol=2e-5)

    restored = layer.hc_post(y, x, post, comb)
    assert restored.shape == x.shape
    assert restored.dtype == x.dtype


def test_v41_hc_post_matches_reference_equation():
    torch.manual_seed(11)
    layer = _layer()
    x = torch.randn(3, 5, dtype=torch.bfloat16)
    residual = torch.randn(3, 4, 5, dtype=torch.bfloat16)
    post = torch.randn(3, 4, dtype=torch.float32)
    comb = torch.randn(3, 4, 4, dtype=torch.float32)

    actual = layer.hc_post(x, residual, post, comb)
    expected = (
        post.unsqueeze(-1) * x.unsqueeze(-2)
        + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=-3)
    ).to(x.dtype)
    torch.testing.assert_close(actual, expected)
