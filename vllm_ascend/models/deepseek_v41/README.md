# DeepSeek V4.1 eager bring-up

The V4.1 backbone is registered through `model.py`, matching the package
structure used by DeepSeek V4. There is no separate `modeling.py` and no
model-side KV-cache owner.

## Current execution path

- `model.py` owns the V4.1 backbone, delayed mHC handoff, attention projections,
  source references, compressor invocation and the correctness-first eager
  attention path.
- `attention/dsa_v41.py` owns paged-cache registration and metadata plus the
  unfused SWA/long-KV gather, attention and cache-scatter operations.
- `core/deepseek_v41.py` owns cache specs, hybrid grouping, sizing, allocation
  and reshape.
- `compressor.py` owns ratio1/ratio2 compressor parameters and the ratio2
  FP32 state cache. `indexer.py` owns Index-K cache updates, Indexer scoring,
  candidate-block filtering and chronological TopK selection.

The model reuses DeepSeek V4's quantization-aware projection, MoE and output
projection implementations. Small operators replace the fused DSA kernel for
the initial eager milestone.

## Hybrid cache layout

All resource planes use one global block-ID lifecycle but have independent
physical tensors and exact byte accounting:

- one BF16 sliding-window-128 KV plane for every backbone attention layer;
- ratio2 compressed long-KV and Index-K planes owned by source layers 2, 8 and
  14, shared by consumers 2-7, 8-13 and 14-19 respectively;
- ratio1 long-KV and Index-K planes owned by source layer 20 and shared by
  consumers 20-39;
- one FP32 window-2 KV/score state plane at each ratio2 source. It stores one
  uncompressed row per original token and emits one compressed latent per pair.

Index source layers 24, 28, 32 and 36 compute new selections in the reference
architecture but do not own another copy of the long KV or Index K. Candidate
blocks originate at layer 20. Consumers retain the source prefix and retrieve
the source cache from `static_forward_context`; shared modules are never
re-registered under consumer layers.

## Supported milestone and remaining accuracy work

The validated milestone is model runner V1, eager mode, BF16 cache, hybrid KV
management, PP/DCP/PCP equal to one, and tensor/data/expert parallel serving.
Prefix caching, speculative decoding, KV transfer and graph mode fail closed.

The fallback attends over local SWA plus the compressed rows selected by the
Indexer/Candidate path. Engram execution is intentionally disabled: its two
roughly 196 GB embedding tables require a distributed HBM layout, while the
temporary CPU/NFS mmap implementation was both prohibitively slow and
numerically unverified. Full accuracy still requires HBM-sharded Engram at
layers 1 and 14 and reference FP8/FP4 rounding. These omissions must not be
interpreted as full model accuracy.

## Validation

On the A3 remote container the W8A8 checkpoint loads under TP4/DP4/EP, the
service becomes healthy, and greedy eager smoke requests return coherent
answers (`2+2 -> 4`, Chinese capital question -> Beijing). The focused cache and
mHC suite passes 30 tests. Formal task-level and long-context accuracy remain
follow-up gates.
