#!/usr/bin/env python3
from __future__ import annotations

import argparse, ctypes, json, objc, os, struct, sys, tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "asm"))
import disassemble_metal_kernel as dmk


def parse_args():
    p = argparse.ArgumentParser(description="Capture the exact offline-built Q4_K decode kernel used by the asm report.")
    p.add_argument("--source", type=Path, default=dmk.DEFAULT_SOURCE)
    p.add_argument("--kernel", default=dmk.DEFAULT_KERNEL)
    p.add_argument("--out-dir", type=Path, default=ROOT / "profile/out")
    p.add_argument("--workdir", type=Path)
    p.add_argument("--metal-std", default="metal3.2")
    p.add_argument("--platform", default="macos")
    p.add_argument("--min-version", default="")
    p.add_argument("--sdk-version", default="")
    p.add_argument("--repeat", type=int, default=1)
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


def dispatch(queue, pipe, buffers, threads, tg):
    cb, enc = queue.commandBuffer(), None
    enc = cb.computeCommandEncoder(); enc.setComputePipelineState_(pipe)
    for i, b in enumerate(buffers): enc.setBuffer_offset_atIndex_(b, 0, i)
    enc.dispatchThreads_threadsPerThreadgroup_((threads, 1, 1), (tg, 1, 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if cb.error(): raise RuntimeError(cb.error())


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


def compile_like_disassembler(a, params):
    source, work = a.source.resolve(), (a.workdir or Path(tempfile.mkdtemp(prefix=f"{a.kernel}-profile-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    air, metallib = work / "kernel.air", work / "kernel.metallib"
    script, gpuexec = work / "pipeline.mtlp-json", work / "kernel.gpuexec"
    arch = dmk.maybe_run(["xcrun", "metal-arch", "--default"]) or "applegpu_g14s"
    sdk = a.sdk_version or dmk.version_pair(dmk.maybe_run(["xcrun", "--sdk", "macosx", "--show-sdk-version"]), "26.5")
    minv = a.min_version or dmk.version_pair(__import__("platform").mac_ver()[0], sdk.split(".")[0] + ".0")
    dmk.run(["xcrun", "metal", f"-std={a.metal_std}", "-O3", "-c", source, "-o", air])
    dmk.run(["xcrun", "metallib", air, "-o", metallib])
    script.write_text(json.dumps(
        {"libraries": {"paths": [{"label": "lib", "path": str(metallib)}]},
         "pipelines": {"compute_pipelines": [{"compute_function": f"alias:lib#{a.kernel}",
                                              "threadgroup_size_is_multiple_of_thread_execution_width": True,
                                              "max_total_threads_per_threadgroup": params["TG"]}]}}, indent=2))
    dmk.run(["xcrun", "metal-nt", "-arch", arch, "-platform_version", a.platform, minv, sdk,
             "-N", script, "-o", gpuexec, metallib])
    return metallib, work, arch, minv, sdk


def main():
    a, stem = parse_args(), datetime.now().strftime("%Y%m%d_%H%M%S") + "_q4_k_k4096_n4096_decode_v2_tg128_reg"
    params = dmk.kernel_params(a.source.resolve(), a.kernel)
    if not all(params[k] for k in ("K", "N", "TG", "NSIMD")):
        raise RuntimeError(f"could not parse DECODE_Q4K_V2_REG_TG params for {a.kernel}")
    metallib, work, arch, minv, sdk = compile_like_disassembler(a, params)
    device = metal_device(); lib = device.newLibraryWithFile_error_(str(metallib), None)
    fn = lib.newFunctionWithName_(a.kernel)
    pipe = device.newComputePipelineStateWithFunction_error_(fn, None)
    threads, tg = ((params["N"] + params["NSIMD"] - 1) // params["NSIMD"]) * params["TG"], params["TG"]
    buffers, queue = make_buffers(device, params["K"], params["N"]), device.newCommandQueue()
    with metal_capture(a.out_dir / f"{stem}.gputrace") as trace:
        for _ in range(a.repeat): dispatch(queue, pipe, buffers, threads, tg)
    print(f"kernel={a.kernel}")
    print(f"compile=xcrun metal -std={a.metal_std} -O3 -c {a.source.resolve()} -o {work / 'kernel.air'}")
    print(f"metal_nt_arch={arch} platform={a.platform} min={minv} sdk={sdk}")
    print(f"dispatchThreads={{{threads}, 1, 1}} threadsPerThreadgroup={{{tg}, 1, 1}} repeat={a.repeat}")
    print(f"pipeline_max_threads={pipe.maxTotalThreadsPerThreadgroup()}")
    print(f"thread_execution_width={pipe.threadExecutionWidth()}")
    print(f"workdir={work}")
    if trace and trace.exists(): print(f"metal_capture={trace}\nopen_xcode=open {trace}")
    else: print(f"metal_capture=disabled; example=MTL_CAPTURE_ENABLED=1 python {Path(__file__).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
