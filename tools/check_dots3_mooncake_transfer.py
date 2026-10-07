# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Byte-exact cross-node NPU Mooncake check before expensive model loading."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from mooncake.engine import TransferEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role", choices=("source", "reader"), required=True)
    parser.add_argument("--ip", required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    tensors = [torch.empty(1048576, dtype=dtype, device="npu") for dtype in (torch.int8, torch.bfloat16, torch.float32)]
    engine = TransferEngine()
    assert engine.initialize(args.ip, "P2PHANDSHAKE", "ascend", "") == 0
    for tensor in tensors:
        assert engine.register_memory(tensor.data_ptr(), tensor.numel() * tensor.element_size()) == 0
    deadline = time.monotonic() + 180

    def wait_for(name):
        while not (args.root / name).exists():
            assert time.monotonic() < deadline, name
            time.sleep(0.1)
        return json.loads((args.root / name).read_text())

    if args.role == "source":
        for index, tensor in enumerate(tensors):
            tensor.view(torch.uint8).copy_(
                torch.arange(tensor.numel() * tensor.element_size(), dtype=torch.int64, device="npu")
                .add(index * 37)
                .remainder(256)
                .to(torch.uint8)
            )
        torch.npu.synchronize()
        ticket = {
            "session": f"{args.ip}:{engine.get_rpc_port()}",
            "addresses": [tensor.data_ptr() for tensor in tensors],
            "lengths": [tensor.numel() * tensor.element_size() for tensor in tensors],
        }
        temporary = args.root / "source.tmp"
        temporary.write_text(json.dumps(ticket))
        temporary.replace(args.root / "source.json")
        wait_for("reader.json")
    else:
        ticket = wait_for("source.json")
        for tensor in tensors:
            tensor.zero_()
        torch.npu.synchronize()
        started = time.perf_counter()
        result = engine.batch_transfer_sync_read(
            ticket["session"], [tensor.data_ptr() for tensor in tensors], ticket["addresses"], ticket["lengths"]
        )
        seconds = time.perf_counter() - started
        assert result >= 0, result
        torch.npu.synchronize()
        for index, tensor in enumerate(tensors):
            actual = tensor.view(torch.uint8).cpu()
            expected = torch.arange(len(actual), dtype=torch.int64).add(index * 37).remainder(256).to(torch.uint8)
            assert torch.equal(actual, expected), f"byte mismatch in tensor {index}"
        record = {
            "bytes": sum(ticket["lengths"]),
            "seconds": seconds,
            "return_code": result,
            "byte_exact": True,
            "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
        }
        (args.root / "reader.json").write_text(json.dumps(record))
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
