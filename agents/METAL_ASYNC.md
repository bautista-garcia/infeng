# Metal Async Copy Notes

This repo's async-copy path uses Apple's undocumented AIR intrinsic for copying from device memory to
threadgroup memory. The hardware instruction shows up in Dougall Johnson's Apple GPU disassembler as
`async_load copy_2d`.

## What It Is

The useful primitive is a SIMD-group scoped copy:

```llvm
%struct._simdgroup_event_t = type opaque

declare %struct._simdgroup_event_t*
@"air.simdgroup_async_copy_2d.read.p3i8.p1i8"(
  i64 element_size,
  i64 element_align,
  i8 addrspace(3)* threadgroup_dst,
  i64 dst_elements_per_row,
  i64 unknown_dst_stride,
  <2 x i64> tile_cols_rows,
  i8 addrspace(1)* device_src,
  i64 src_elements_per_row,
  i64 unknown_src_stride,
  <2 x i64> tile_cols_rows_again,
  <2 x i64> src_offset,
  i32 flags)

declare void @air.wait_simdgroup_events(i32 count, %struct._simdgroup_event_t** events)
```

Address spaces matter:

- `addrspace(1)` is device memory.
- `addrspace(3)` is threadgroup memory.
- The intrinsic spelling used here is `.read.p3i8.p1i8`: read from device pointer to threadgroup pointer.

The event returned by `air.simdgroup_async_copy_2d` must be waited on before any thread reads the destination
threadgroup memory. After waiting, use a `threadgroup_barrier(mem_flags::mem_threadgroup)` so all SIMD groups
in the threadgroup see the loaded tile.

## Why We Patch AIR

The Percisely post shows direct MSL declarations with `__asm("air.simdgroup_async_copy_2d.p3i8.p1i8")`.
On the current local toolchain (`metalfe-32023.917`), direct `air.*` asm labels fail in the Metal frontend
with `illegal string literal in 'asm'`. The workaround that works here is:

1. Write normal Metal source with fallback helper functions.
2. Compile to LLVM AIR text:

   ```sh
   xcrun metal -std=metal3.2 -O3 -emit-llvm -S backend/metal/quant_linear_async.metal -o kernel.ll
   ```

3. Patch the helper call sites in `kernel.ll` to call `air.simdgroup_async_copy_2d.read.p3i8.p1i8` and
   `air.wait_simdgroup_events`.
4. Assemble and package:

   ```sh
   xcrun metal-as kernel_patched.ll -o=kernel.air
   xcrun metallib kernel.air -o kernel.metallib
   ```

5. Optionally native-translate and disassemble:

   ```sh
   xcrun metal-nt -arch "$(xcrun metal-arch --default)" -platform_version macos <min> <sdk> \
     -N pipeline.mtlp-json -o kernel.gpuexec kernel.metallib
   ```

The benchmark script automates this for `quant_linear_async.metal`:

```sh
python benchmarks/benchmark_q4k_async.py --shapes 4096x4096 --m-values 32,128
```

## Current Kernel Structure

`backend/metal/quant_linear_async.metal` keeps the source readable and valid MSL. It has fallback helpers:

```metal
static ulong q4k_copy_block_tile_start(threadgroup uchar* dst, device const uchar* src,
                                       ulong src_stride, uint simd_lane);
static void q4k_copy_block_tile_wait(ulong event, threadgroup ulong* sink);
```

The benchmark patches calls to those helpers:

- `q4k_copy_block_tile_start(...)` becomes an AIR async-copy start.
- The returned event pointer is converted to `ulong` with `ptrtoint`.
- `q4k_copy_block_tile_wait(...)` converts the `ulong` back with `inttoptr`, stores it in a local one-element
  event array, and calls `air.wait_simdgroup_events`.

The Q4_K async prefill uses double-buffered compressed weight tiles:

- `threadgroup uchar w_tile[2 * 16 * 144]`
- tile shape copied by AIR: `144 x 16` bytes
- only `simd_group == 0` issues the copy, matching the Percisely benchmark observation that one SIMD group
  issuing the async load is faster than spreading it across SIMD groups.

The pipeline is:

1. Start async copy for compressed tile `kb=0`.
2. For each `kb`:
   - wait for current tile,
   - start async copy for `kb+1` into the other raw-tile buffer,
   - barrier,
   - dequantize the current raw tile into `b_tile`,
   - matrix multiply using `b_tile` and direct device loads from `x`.

Starting `kb+1` before dequant+MMA is what creates overlap. Waiting immediately after starting the copy
correctly emits `async_load`, but does not hide copy latency.

## Verification

A successful run should show exact correctness and nonzero `async_load` in the async variant:

```text
M,K,N,variant,median_ms,speedup_vs_v2,max_abs,max_rel,tflops,async_load,device_load
32,4096,1024,async_bn16,...,0,0,...,2,4
```

`async_load=2` means there are two static async-copy sites in the kernel: the initial prefetch and the loop
prefetch. `max_abs=0` and `max_rel=0` confirm the patched AIR path matches `quant_linearv2.metal`.

A direct toy proof is also in:

```sh
python benchmarks/benchmark_async_copy.py
```

That script builds a tiny AIR-only copy kernel, runs it, and checks that Dougall's disassembler prints
`async_load copy_2d`.

## Failed Or Risky Paths

- Direct MSL `__asm("air...")` declarations do not compile on the current toolchain.
- Starting an async copy and waiting inside the same helper works, but leaves no useful overlap.
- Moving the wait/start helpers out of line works, but adds helper-call overhead. The benchmark now patches
  start/wait call sites inline.
- Double-buffering a full `x` tile (`2 * 32 * 256` halfs) plus the existing raw weight tile, dequant tile, and
  scratch exceeded the native translator's threadgroup memory limit:

  ```text
  Threadgroup memory size (47616) exceeds the maximum threadgroup memory allowed (32768)
  ```

  So the current viable experiment only async-copies the compressed Q4_K weight tile. `x` is still loaded
  directly by `simdgroup_load` from device memory.

## Sources

- Percisely, "Fast Matrix Multiply on an Apple GPU": documents the `simdgroup_async_copy` idea, the event wait,
  and the observation that one SIMD group issuing the async copy can be faster than many.
  https://percisely.xyz/gemm
- Dougall Johnson's Apple GPU tools and docs: used to disassemble native GPU code and verify `async_load`.
  https://github.com/dougallj/applegpu
- dougallj/applegpu issue #28: discussion of the undocumented async device-to-threadgroup copy.
  https://github.com/dougallj/applegpu/issues/28
- dougallj/applegpu PR #29: adds async load/store disassembly support.
  https://github.com/dougallj/applegpu/pull/29
- Philip Turner's `metal-benchmarks`: background on Apple GPU microarchitecture and SIMD behavior.
  https://github.com/philipturner/metal-benchmarks
