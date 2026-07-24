# M2 Pro occupancy model
- M2 Pro has 19 GPU cores. Treat one Apple GPU core as the scheduling/resource analogue of an NVIDIA SM.
- A threadgroup's simdgroups reside on one GPU core; resident simdgroups interleave to hide memory and execution latency.
- Per-core limiting pools observed/assumed for occupancy accounting: 208 KiB register file and 60 KiB threadgroup/shared memory. The first exhausted pool limits residency.
- SIMD width is 32 lanes, so `simdgroups_per_threadgroup = threads_per_threadgroup / 32`.
- For kernels without threadgroup memory, register residency is approximately `floor(208 KiB / (threads_per_threadgroup * registers_per_thread * register_bytes))`.
- Xcode's `# Allocated Registers` / `High Register` is best read as 16-bit register slots on Apple GPU captures. The disassembly report's `r0..rN` counts full 32-bit AGX registers; `r0..r29` is 30 full registers, roughly 60 16-bit slots.
- Occupancy examples for `TG=128` with no threadgroup memory:
  - 60 half-register slots/thread = 120 B/thread = 15 KiB/TG => 13 TG/core, 52 simdgroups/core.
- GPU-wide resident threadgroups are `resident_tg_per_core * 19`; approximate wave count is `total_dispatched_threadgroups / gpu_wide_resident_threadgroups`.
- Occupancy is a means, not the objective: lowering registers only matters when it crosses a residency cliff, and lower-register variants must still be timed because extra loops/control/waits can dominate.

# Applying optimizations
Your workflow has to be:
1. Measuring current performance (time)
2. Trying the optimization (threadgroup, registers, unrolling, etc)
3. Verify correctness (matches same output as before)
4. Measure optimized performance (time)
5.a For speed optimizations if  Speedup > 1.1 keep optimization, otherwise discard it.
5.b For optimizations that attempt to reduce lines of code or reduce the memory footprint (threadgroup or register memory) then if speedup is  0.99 < S < 1.01 you can keep optimization.
6. If there are more optimizations to try go back to 2., otherwise finsih the job


# List of optimizations
## General
- Doing `simdgroup_load(transpose=true, ...)` gives us a free transpose operation, instead of doing it on the pytorch side.

## Prefill Kernels
- Specialize hot kernels around fixed chunk/tile shapes so loop bounds, launch geometry, and SRAM layouts are compile-time constants.
- Use `simd_shuffle_up` for associative intra-SIMD prefix scans, e.g. cumulative decay/gating products.
- Use `simd_shuffle_xor`/SIMD reductions for vector norms and partial sums instead of threadgroup-memory reductions.
- Use `simdgroup_matrix` + `simdgroup_multiply_accumulate` for SRAM-resident GEMMs instead of scalar thread loops.
- Keep global memory writes vectorized/coalesced and use SRAM staging for transposes that feed matrix fragments repeatedly.

## Decode Kernels
- Unroll loops.
- Use threadgroup memory as much as possible.
- Fuse kernels to increase arithmetic intensity of decode kernels..



# Xcode .gputrace
- Purple: ALU; Orange: Memory; Red: Barrier/Synchronization; Yellow: Control Flow; 

## 2026-07-23 - `q4_k_k4096_n4096_decode`: one row per SIMD

- Gate for this loop: cache-pressured speedup `>= 1.05x` in two fresh processes.
- Rejected screens: streamed two-row body `0.755x`; incremental `uint` addressing `0.986x`;
  fixed rolled loop `0.999x`; full unroll `0.590x`; metadata broadcast `0.785x`; TG32 `1.013x`;
  TG128 `1.008x`.
- Accepted: specialize only `<K=4096, N=4096>` from two rows/SIMD to one row/SIMD. TG64 remains
  unchanged; dispatch grows from 65,536 to 131,072 threads, exposing twice as many independent SIMDgroups.
  Weight bytes and per-row math are unchanged; input reads double, but the input is only 8 KiB.
- Exact-final-source cache-pressured A/B: `65.27 -> 62.08 us` (`1.051x`, paired `1.060x`)
  and `60.75 -> 56.88 us` (`1.068x`, paired `1.067x`). Warm speedups were `1.033x` and `1.044x`.
- Correctness: exact FP16 equality for the deterministic case and ten randomized valid Q4_K matrices;
  real-model llama.cpp greedy parity passed.
- Macro: Qwen3.5-9B TPOT `46.85 -> 46.19 ms`, decode `21.35 -> 21.65 tok/s`; peak memory unchanged.
- AIR for this function: SSA instructions `267 -> 253`, loads `21 -> 19`, branches `14 -> 9`,
  allocas `3 -> 2`. The accepted source keeps the outer loop rolled.
- M2 Pro native code: static instructions `412 -> 240`, final instruction address `0xAC4 -> 0x63E`,
  highest register `r39 -> r35`, device-load sites `22 -> 15`, waits `3 -> 2`, and SIMD reductions
  `2 -> 1`. Neither version emits `async_load`.
