#!/usr/bin/env python3
from __future__ import annotations

import argparse, ctypes, json, math, objc, os, random, re, statistics, struct, subprocess, tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


# Keep this table in sync with the explicit instantiations at the bottom of
# backend/metal/quant_linear.metal.  The profiler deliberately compiles the
# source itself instead of going through torch.mps.compile_shader(), so the
# captured function has the same name as the source kernel.
KERNELS = {
    ("Q4_K", 4096, 1024): ("q4_k_k4096_n1024", 64, 4, 144),
    ("Q4_K", 4096, 4096): ("q4_k_k4096_n4096", 64, 2, 144),
    ("Q4_K", 12288, 4096): ("q4_k_k12288_n4096", 64, 4, 144),
    ("Q4_K", 4096, 8192): ("q4_k_k4096_n8192", 64, 4, 144),
    ("Q4_K", 4096, 12288): ("q4_k_k4096_n12288", 64, 4, 144),
    ("Q5_K", 4096, 1024): ("q5_k_k4096_n1024", 64, 2, 176),
    ("Q5_K", 4096, 4096): ("q5_k_k4096_n4096", 64, 2, 176),
    ("Q5_K", 12288, 4096): ("q5_k_k12288_n4096", 64, 2, 176),
    ("Q5_K", 4096, 8192): ("q5_k_k4096_n8192", 64, 2, 176),
    ("Q5_K", 4096, 12288): ("q5_k_k4096_n12288", 64, 2, 176),
    ("Q6_K", 4096, 1024): ("q6_k_k4096_n1024", 64, 4, 210),
    ("Q6_K", 12288, 4096): ("q6_k_k12288_n4096", 64, 4, 210),
    ("Q6_K", 4096, 248320): ("q6_k_k4096_n248320", 64, 4, 210),
    ("Q8_0", 4096, 4096): ("q8_0_k4096_n4096", 128, 2, 34),
    ("IQ4_XS", 4096, 12288): ("iq4_xs_k4096_n12288", 64, 4, 136),
}
BLOCK_SIZE = {"Q4_K": 256, "Q5_K": 256, "Q6_K": 256, "Q8_0": 32, "IQ4_XS": 256}


def parse_args():
    p = argparse.ArgumentParser(description="Capture a quant_linear.metal kernel in Xcode's GPU capture format.")
    p.add_argument("--source", type=Path, default=ROOT / "backend/metal/quant_linear.metal")
    p.add_argument("--kernel", help="Full exported kernel name; overrides --quant-type/--k/--n.")
    p.add_argument("--quant-type", choices=sorted({x[0] for x in KERNELS}), default="Q4_K")
    p.add_argument("--k", type=int, default=4096, help="Input/features dimension.")
    p.add_argument("--n", type=int, default=4096, help="Output/features dimension.")
    p.add_argument("--kind", choices=("decode", "prefill"), default="decode")
    p.add_argument("--out-dir", type=Path, default=ROOT / "profile/out")
    p.add_argument("--workdir", type=Path)
    p.add_argument("--metal-std", default="metal3.2")
    p.add_argument("--m", type=int, default=32, help="Token count for prefill kernels.")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--batches", type=int, default=5)
    p.add_argument("--pressure-mib", type=int, default=64)
    p.add_argument("--random-seed", type=int)
    p.add_argument("--baseline-metallib", type=Path)
    p.add_argument("--decode-tg", type=int, choices=(32, 64, 128))
    p.add_argument("--decode-rows", type=int)
    p.add_argument("--baseline-decode-tg", type=int, choices=(32, 64, 128))
    p.add_argument("--baseline-decode-rows", type=int)
    p.add_argument("--json-out", type=Path)
    return p.parse_args()


def metal_device():
    bundle = objc.loadBundle("Metal", globals(), bundle_path="/System/Library/Frameworks/Metal.framework")
    objc.loadBundleFunctions(bundle, globals(), [("MTLCreateSystemDefaultDevice", b"@")])
    return MTLCreateSystemDefaultDevice()


def fill_buffer(device, data):
    buf = device.newBufferWithLength_options_(len(data), 0)
    ctypes.memmove(buf.contents(), data, len(data))
    return buf


def make_buffers(device, k, n):
    x, y = struct.pack("e", 1.0) * k, device.newBufferWithLength_options_(n * 2, 0)
    w, one, zero = bytearray(n * (k // 256) * 144), struct.pack("e", 1.0), struct.pack("e", 0.0)
    for o in range(0, len(w), 144):
        w[o:o+2] = one; w[o+2:o+4] = zero; w[o+4:o+12] = bytes([1, 0, 1, 0, 1, 0, 1, 0]); w[o+16:o+144] = bytes([1]) * 128
    return y, fill_buffer(device, x), fill_buffer(device, bytes(w))


def make_quant_buffers(device, quant_type, k, n, block_bytes, m=1, random_seed=None):
    """Return buffers for one decode dispatch using harmless valid quant data."""
    blocks = n * (k // BLOCK_SIZE[quant_type])
    w = bytearray(blocks * block_bytes)
    if random_seed is not None:
        if quant_type != "Q4_K": raise ValueError("--random-seed currently supports Q4_K only")
        rng = random.Random(random_seed)
        x = b"".join(struct.pack("e", rng.uniform(-1.0, 1.0)) for _ in range(m * k))
        for o in range(0, len(w), block_bytes):
            w[o:o + 4] = struct.pack("ee", rng.uniform(0.001, 0.01), rng.uniform(0.001, 0.01))
            w[o + 4:o + 16] = rng.randbytes(12); w[o + 16:o + block_bytes] = rng.randbytes(block_bytes - 16)
        return device.newBufferWithLength_options_(m * n * 2, 0), fill_buffer(device, x), fill_buffer(device, bytes(w))
    one, zero = struct.pack("e", 1.0), struct.pack("e", 0.0)
    for o in range(0, len(w), block_bytes):
        w[o:o + 2], w[o + 2:o + 4] = one, zero
        if quant_type in ("Q4_K", "Q5_K"):
            w[o + 4:o + 16] = bytes([1]) * 12
            w[o + 16:o + block_bytes] = bytes([1]) * (block_bytes - 16)
        elif quant_type == "Q6_K":
            w[o + 192:o + 208] = bytes([1]) * 16
            w[o + 208:o + 210] = one
            w[o:o + 192] = bytes([1]) * 192
        else:
            w[o + 2:o + block_bytes] = bytes([1]) * (block_bytes - 2)
    return (device.newBufferWithLength_options_(m * n * 2, 0),
            fill_buffer(device, struct.pack("e", 1.0) * (m * k)),
            fill_buffer(device, bytes(w)))


def dispatch(queue, pipe, buffers, threads, tg, m=None):
    cb, enc = queue.commandBuffer(), None
    enc = cb.computeCommandEncoder(); enc.setComputePipelineState_(pipe)
    for i, b in enumerate(buffers): enc.setBuffer_offset_atIndex_(b, 0, i)
    if m is not None:
        enc.setBytes_length_atIndex_(struct.pack("q", m), 8, 3)
    enc.dispatchThreads_threadsPerThreadgroup_((threads, 1, 1), (tg, 1, 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if cb.error(): raise RuntimeError(cb.error())
    return (cb.GPUEndTime() - cb.GPUStartTime()) * 1e6


def cache_pressure(queue, src, dst, size):
    cb = queue.commandBuffer(); enc = cb.blitCommandEncoder()
    enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(src, 0, dst, 0, size)
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if cb.error(): raise RuntimeError(cb.error())


def summarize(samples):
    s = sorted(samples)
    at = lambda q: s[min(len(s) - 1, round(q * (len(s) - 1)))]
    return {"min": s[0], "p10": at(0.10), "median": statistics.median(s), "mean": statistics.mean(s),
            "p90": at(0.90), "p99": at(0.99), "max": s[-1], "variance": statistics.pvariance(s)}


def compare_outputs(baseline, candidate, elements):
    a = struct.unpack(f"{elements}e", ctypes.string_at(baseline.contents(), elements * 2))
    b = struct.unpack(f"{elements}e", ctypes.string_at(candidate.contents(), elements * 2))
    diffs = [abs(float(x) - float(y)) for x, y in zip(a, b)]
    rel = [d / max(abs(float(x)), 1e-12) for d, x in zip(diffs, a)]
    return {"max_abs": max(diffs), "max_rel": max(rel), "different": sum(x != y for x, y in zip(a, b)),
            "nonfinite": sum(not math.isfinite(float(x)) for x in b)}


def benchmark(queue, candidate, baseline, candidate_buffers, baseline_buffers, candidate_launch, baseline_launch,
              warmup, samples, batches, pressure_mib, m=None):
    pipes = [("candidate", candidate, candidate_buffers, *candidate_launch)]
    if baseline is not None: pipes.insert(0, ("baseline", baseline, baseline_buffers, *baseline_launch))
    pressure_size = pressure_mib * 2**20
    pressure = ((queue.device().newBufferWithLength_options_(pressure_size, 0),
                 queue.device().newBufferWithLength_options_(pressure_size, 0)) if pressure_size else None)
    result = {}
    for regime in ("warm", "pressured"):
        for sample in range(warmup):
            for _, pipe, buffers, threads, tg in (pipes if sample % 2 == 0 else list(reversed(pipes))):
                if regime == "pressured" and pressure is not None:
                    cache_pressure(queue, pressure[0], pressure[1], pressure_size)
                dispatch(queue, pipe, buffers, threads, tg, m)
        medians = {name: [] for name, *_ in pipes}; batch_stats = {name: [] for name, *_ in pipes}
        paired_speedups = []
        for batch in range(batches):
            values = {name: [] for name, *_ in pipes}
            for sample in range(samples):
                ordered = pipes if (sample + batch) % 2 == 0 else list(reversed(pipes))
                for name, pipe, buffers, threads, tg in ordered:
                    if regime == "pressured" and pressure is not None:
                        cache_pressure(queue, pressure[0], pressure[1], pressure_size)
                    values[name].append(dispatch(queue, pipe, buffers, threads, tg, m))
            for name in values:
                stats = summarize(values[name]); batch_stats[name].append(stats); medians[name].append(stats["median"])
            if baseline is not None:
                paired_speedups.append(statistics.median(a / b for a, b in zip(values["baseline"],
                                                                               values["candidate"])))
        result[regime] = {name: {"median_of_medians": statistics.median(medians[name]),
                                 "batches": batch_stats[name]} for name, *_ in pipes}
        if baseline is not None:
            result[regime]["speedup"] = (result[regime]["baseline"]["median_of_medians"] /
                                          result[regime]["candidate"]["median_of_medians"])
            result[regime]["paired_speedup_median"] = statistics.median(paired_speedups)
            result[regime]["paired_speedup_batches"] = paired_speedups
    return result


@contextmanager
def metal_capture(path):
    if not torch.mps.profiler.is_metal_capture_enabled():
        yield None; return
    path.parent.mkdir(parents=True, exist_ok=True)
    old, actual = Path.cwd(), path.parent / f"0000-{path.stem}.gputrace"
    os.chdir(path.parent)
    cm = torch.mps.profiler.metal_capture(path.stem)
    try:
        cm.__enter__()
    except RuntimeError as e:
        os.chdir(old); print(f"[warn] metal_capture unavailable: {e}", flush=True); yield None; return
    try:
        yield actual
    finally:
        try: cm.__exit__(None, None, None)
        finally: os.chdir(old)


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def compile_kernel(a, kernel, work):
    source = a.source.resolve()
    work.mkdir(parents=True, exist_ok=True)
    air, metallib = work / "kernel.air", work / "kernel.metallib"
    run(["xcrun", "metal", f"-std={a.metal_std}", "-O3", "-c", source, "-o", air])
    run(["xcrun", "metallib", air, "-o", metallib])
    return metallib, work


def main():
    a = parse_args()
    if min(a.warmup, a.samples, a.batches, a.pressure_mib) < 0 or not a.samples or not a.batches:
        raise ValueError("--warmup/--pressure-mib must be nonnegative; --samples/--batches must be positive")
    if a.kernel:
        match = re.match(r"(.*)_k(\d+)_n(\d+)_(decode|prefill)$", a.kernel)
        if not match:
            raise ValueError("--kernel must end in _k<K>_n<N>_(decode|prefill)")
        _, k, n, kind = match.groups()
        kernel, tg, rows, block_bytes = a.kernel[:-len(kind) - 1], None, None, None
        for (qt, kk, nn), (base, dtg, drows, bbytes) in KERNELS.items():
            if base == kernel:
                a.quant_type, a.k, a.n, a.kind = qt, int(k), int(n), kind
                tg, rows, block_bytes = dtg, drows, bbytes
                break
        if block_bytes is None:
            raise ValueError(f"kernel is not an exported quant_linear.metal kernel: {a.kernel}")
    else:
        try:
            base, tg, rows, block_bytes = KERNELS[(a.quant_type, a.k, a.n)]
        except KeyError as e:
            raise ValueError(f"unsupported quant type/shape: {a.quant_type}, K={a.k}, N={a.n}") from e
        a.kernel = f"{base}_{a.kind}"
    work = (a.workdir or Path(tempfile.mkdtemp(prefix=f"{a.kernel}-profile-"))).resolve()
    metallib, work = compile_kernel(a, a.kernel, work)
    device = metal_device(); lib = device.newLibraryWithFile_error_(str(metallib), None)
    fn = lib.newFunctionWithName_(a.kernel)
    pipe = device.newComputePipelineStateWithFunction_error_(fn, None)
    if a.kind == "decode":
        baseline_tg, baseline_rows = a.baseline_decode_tg or tg, a.baseline_decode_rows or rows
        tg, rows = a.decode_tg or tg, a.decode_rows or rows
        threads, baseline_threads = ((a.n + rows - 1) // rows) * tg, (
            (a.n + baseline_rows - 1) // baseline_rows) * baseline_tg
        dispatch_m = None
    else:
        tg, threads, dispatch_m = 128, 128 * (a.n // 16), a.m
        baseline_tg, baseline_threads = tg, threads
        if a.m % 32:
            raise ValueError("--m must be divisible by 32 for prefill kernels")
    buffers = (make_quant_buffers(device, a.quant_type, a.k, a.n, block_bytes, a.m, a.random_seed)
               if a.kind == "prefill" else make_quant_buffers(device, a.quant_type, a.k, a.n, block_bytes,
                                                               random_seed=a.random_seed))
    queue = device.newCommandQueue()
    stem = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{a.kernel}"
    trace, results = None, None
    if torch.mps.profiler.is_metal_capture_enabled():
        with metal_capture(a.out_dir / f"{stem}.gputrace") as trace:
            for _ in range(a.repeat): dispatch(queue, pipe, buffers, threads, tg, dispatch_m)
    else:
        baseline_pipe = baseline_buffers = correctness = None
        if a.baseline_metallib:
            baseline_lib = device.newLibraryWithFile_error_(str(a.baseline_metallib.resolve()), None)
            baseline_pipe = device.newComputePipelineStateWithFunction_error_(
                baseline_lib.newFunctionWithName_(a.kernel), None)
            output_elements = a.n if a.kind == "decode" else a.m * a.n
            baseline_y = device.newBufferWithLength_options_(output_elements * 2, 0)
            baseline_buffers = (baseline_y, *buffers[1:])
            dispatch(queue, baseline_pipe, baseline_buffers, baseline_threads, baseline_tg, dispatch_m)
            dispatch(queue, pipe, buffers, threads, tg, dispatch_m)
            correctness = compare_outputs(baseline_y, buffers[0], output_elements)
        timing = benchmark(queue, pipe, baseline_pipe, buffers, baseline_buffers, (threads, tg),
                           (baseline_threads, baseline_tg), a.warmup, a.samples, a.batches, a.pressure_mib,
                           dispatch_m)
        results = {"kernel": a.kernel, "dispatch_threads": [threads, 1, 1], "threads_per_threadgroup": [tg, 1, 1],
                   "baseline_dispatch_threads": [baseline_threads, 1, 1],
                   "baseline_threads_per_threadgroup": [baseline_tg, 1, 1],
                   "warmup": a.warmup, "samples_per_batch": a.samples, "batches": a.batches,
                   "pressure_mib": a.pressure_mib, "random_seed": a.random_seed, "correctness": correctness,
                   "timing_us": timing, "pipeline_max_threads": pipe.maxTotalThreadsPerThreadgroup(),
                   "thread_execution_width": pipe.threadExecutionWidth()}
        json_out = (a.json_out or work / "benchmark.json").resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True); json_out.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))
        print(f"benchmark_json={json_out}")
    print(f"kernel={a.kernel}")
    print(f"compile=xcrun metal -std={a.metal_std} -O3 -c {a.source.resolve()} -o {work / 'kernel.air'}")
    print(f"source={a.source.resolve()}")
    print(f"dispatchThreads={{{threads}, 1, 1}} threadsPerThreadgroup={{{tg}, 1, 1}} repeat={a.repeat}")
    print(f"pipeline_max_threads={pipe.maxTotalThreadsPerThreadgroup()}")
    print(f"thread_execution_width={pipe.threadExecutionWidth()}")
    print(f"workdir={work}")
    if trace and trace.exists(): print(f"metal_capture={trace}\nopen_xcode=open {trace}")
    elif results is None:
        print(f"metal_capture=disabled; example=MTL_CAPTURE_ENABLED=1 python {Path(__file__).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
