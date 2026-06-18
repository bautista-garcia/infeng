# Architecture Log

This file tracks the implementation and optimization path for the Qwen3.5 4B runtime. Keep entries chronological, compact, and biased toward decisions, evidence, and next implications.

## 2026-06-17 - Qwen3.5 4B Runtime And Metal Optimization Recap

### 1. PyTorch Baseline

We first implemented Qwen3.5 4B in PyTorch. `model/qwen35/layers.py` defines the core layers as `nn.Module` components, with the model stack built from those pieces.

To make the first version usable end to end, we added `model/qwen35/weights.py` to load BF16 `.gguf` weights and map GGUF tensor names/shapes into the local PyTorch modules. This was the v0 bridge from downloaded model artifacts into our implementation.

We then added `runtime/inference.py` as the basic inference surface. It instantiates the PyTorch model, loads weights, owns tokenization, applies sampling, and gives us a direct local generation path.

Correctness was checked against Hugging Face Transformers in `test/test_qwen35.py`. The PyTorch path is the reference integration layer: easy to inspect, easy to compare, and useful for catching regressions while lower-level kernels evolve.

### 2. Benchmark Harness

Once the baseline was correct, we added `benchmarks/benchmark_qwen35.py` to measure the runtime. The current report includes TTFT, TPOT, TTL, aggregate `tok/s`, `prefill_tok/s`, `decode_tok/s`, peak memory, MBU, and MFU.

The throughput metrics are direct timing measurements:

- `tok/s`: `(prefill tokens + decode tokens) / total time`
- `prefill_tok/s`: `prefill tokens / prefill time`
- `decode_tok/s`: `decode tokens / decode time`

MBU and MFU are useful directional metrics, but the formulas still need a more careful validation pass before treating them as precise hardware-utilization numbers.

### 3. Torch Compile Attempt

The first optimization attempt was `torch.compile` over model layers. This failed to become a stable speedup path because the model exposed variable shapes that triggered recompilation, especially across prefill/decode behavior and cache growth.

We dropped `torch.compile` as the main direction after confirming the issue was compile stability and graph shape churn, not a single isolated layer bug.

### 4. Metal Kernel Work

After `torch.compile`, the next step was moving hot operations into `backend/metal/`. We started with the delta-token-rule kernel because it had the highest expected payoff.

Decode reached a fairly optimized state first. Current evidence suggests decode is memory-bound, so further gains are likely limited unless we materially reduce memory traffic or change the state/layout strategy.

Prefill took most of the optimization work. The kernel follows the chunkwise parallel prefill approach from the literature: decompose the recurrence into GEMMs plus lower-triangular solves, then optimize those pieces for the GPU.

The profiling loop used `profile/profile_qwen35.py` with `MTL_CAPTURE_ENABLED=1`, then Xcode GPU captures to inspect bottlenecks in each kernel version. The detailed optimization trail lives in `KERNELS.md`.

Important optimization themes so far:

- coalesced global-memory access and vectorized writes
- reduced threadgroup-memory bank conflicts
- hot-path specialization around fixed chunk/tile shapes
- `mma_32x32`-style tiled GEMMs via `simdgroup_matrix` / `simdgroup_multiply_accumulate`
- running decay products computed with SIMD shuffles
- reductions using `simd_shuffle_xor` instead of threadgroup memory where possible

The main architectural conclusion is that gated delta-net prefill should keep using the chunkwise path. A sequential implementation can look attractive while the chunkwise kernel is immature, but chunkwise parallelism is the scalable target.

### 5. Current Result

Current BF16 benchmark result:

```text
TTFT=1073.64ms TPOT=59.77ms TTL=8.72s tok/s=73.36 prefill_tok/s=476.88 decode_tok/s=16.73 peak_mem=9.24GiB MBU=70.37% MFU=29.49%
```

This is about 76 tok/s behind llama.cpp on prefill and about 2 tok/s behind on decode for Qwen3.5 4B BF16. That is a strong current result: the custom path is close while still preserving a clear PyTorch reference path and an optimization log for the Metal kernels.


