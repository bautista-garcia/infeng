
# Applying Optimizations:
Your workflow has to be:
1. Measuring current performance (time)
2. Trying the optimization (threadgroup, registers, unrolling, etc)
3. Verify correctness (matches same output as before)
4. Measure optimized performance (time)
5.a For speed optimizations if  Speedup > 1.1 keep optimization, otherwise discard it.
5.b For optimizations that attempt to reduce lines of code or reduce the memory footprint (threadgroup or register memory) then if speedup is  0.99 < S < 1.01 you can keep optimization.
6. If there are more optimizations to try go back to 2., otherwise finsih the job

# Optimizations
## Prefill
- Specialize hot kernels around fixed chunk/tile shapes so loop bounds, launch geometry, and SRAM layouts are compile-time constants.
- Use `simd_shuffle_up` for associative intra-SIMD prefix scans, e.g. cumulative decay/gating products.
- Use `simd_shuffle_xor`/SIMD reductions for vector norms and partial sums instead of threadgroup-memory reductions.
- Use `simdgroup_matrix` + `simdgroup_multiply_accumulate` for SRAM-resident GEMMs instead of scalar thread loops.
- Keep global memory writes vectorized/coalesced and use SRAM staging for transposes that feed matrix fragments repeatedly.

