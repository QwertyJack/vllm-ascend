# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("dp_size", [2, 4, 8])
@pytest.mark.parametrize("live_rows", [0, 2])
def test_dp_finalize_casts_after_collective_with_uneven_or_idle_ranks(dp_size, live_rows):
    padded = 3
    generator = torch.Generator().manual_seed(1777 + dp_size)
    partials = torch.randint(-1024, 1024, (dp_size, dp_size * padded, 7), generator=generator).float() / 512
    golden = partials.double().sum(0).to(torch.bfloat16)
    premature = partials.to(torch.bfloat16).float().sum(0).to(torch.bfloat16)
    assert not torch.equal(premature, golden), "Fixture must catch a cast before DP reduction"
    for rank in range(dp_size):

        def reduce_scatter(value, dim, rank=rank):
            assert dim == 0 and value.dtype == torch.float32
            assert torch.equal(value, partials[rank])
            return partials.sum(0)[rank * padded : (rank + 1) * padded]

        state = SimpleNamespace(
            moe_config=SimpleNamespace(pcp_size=1, dp_size=dp_size),
            num_tokens=live_rows,
            _dots3_fp32_finalize=True,
            _dots3_output_dtype=torch.bfloat16,
        )
        with patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_dp_group",
            return_value=SimpleNamespace(reduce_scatter=reduce_scatter),
        ):
            actual = PrepareAndFinalizeWithAllGather._finalize_with_dp_group(state, partials[rank], False)
        assert actual.dtype == torch.bfloat16
        assert torch.equal(actual, golden[rank * padded : rank * padded + live_rows])


@pytest.mark.parametrize("quant_type", [QuantType.NONE, QuantType.W8A8MXFP])
@pytest.mark.parametrize("enabled", [False, True])
def test_prepare_fp32_finalize_is_opt_in_and_mxfp8_only(quant_type, enabled):
    state = SimpleNamespace(
        _use_ep_sequence_parallel=lambda: False,
        _prepare_with_dp_group=lambda *args: "prepared",
    )
    with patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_ascend_config",
        return_value=SimpleNamespace(moe_allgather_fp32_combine=enabled),
    ):
        result = PrepareAndFinalizeWithAllGather.prepare(
            state, torch.zeros(2, 7, dtype=torch.bfloat16), torch.zeros(2, 3), quant_type=quant_type
        )
    assert result == "prepared"
    assert state._dots3_output_dtype == torch.bfloat16
    assert state._dots3_fp32_finalize == (enabled and quant_type == QuantType.W8A8MXFP)


@pytest.mark.parametrize("dp_size", [1, 2, 4, 8, 3])
@pytest.mark.parametrize("sp,pcp", [(False, 1), (True, 1), (False, 2)])
def test_fp32_combine_guard_preserves_sp_and_context_parallel_restrictions(dp_size, sp, pcp):
    state = SimpleNamespace(moe_config=SimpleNamespace(dp_size=dp_size, pcp_size=pcp, is_sequence_parallel=sp))
    with patch(
        "vllm_ascend.ops.fused_moe.fused_moe.get_current_hardware_profile",
        return_value=SimpleNamespace(supports=lambda capability: True),
    ):
        if dp_size in (1, 2, 4, 8) and not sp and pcp == 1:
            AscendMoERunner._validate_allgather_fp32_combine(state)
        else:
            with pytest.raises(ValueError):
                AscendMoERunner._validate_allgather_fp32_combine(state)
