from __future__ import annotations

import argparse
import gc
import os
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from backend.metal import get_delta_rule_kernels, get_quant_linear_kernels
from model.qwen35.weights import GGUF_BLOCK


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3.5 MPS kernel profiler")
    p.add_argument("--target", default="delta")
    p.add_argument("--seq-len", "--seq_len", dest="seq_len", type=int, default=1)
    p.add_argument("--quant-type", "--quant_type", dest="quant_type", default="Q4_K",
                   choices=("Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ4_XS"))
    p.add_argument("--in-features", "--in_features", dest="in_features", type=int, default=4096)
    p.add_argument("--out-features", "--out_features", dest="out_features", type=int, default=4096)
    return p.parse_args()


def sync():
    gc.collect(); torch.mps.empty_cache(); torch.mps.synchronize()


@contextmanager
def metal_capture(path: Path):
    if not torch.mps.profiler.is_metal_capture_enabled():
        yield None; return
    path.parent.mkdir(parents=True, exist_ok=True)
    old_cwd, actual = Path.cwd(), path.parent / f"0000-{path.stem}.gputrace"
    os.chdir(path.parent)
    cm = torch.mps.profiler.metal_capture(path.stem)
    try:
        cm.__enter__()
    except RuntimeError as e:
        os.chdir(old_cwd)
        print(f"[warn] metal_capture unavailable: {e}", flush=True)
        yield None; return
    try:
        yield actual
    finally:
        try:
            cm.__exit__(None, None, None)
        finally:
            os.chdir(old_cwd)


def capture(name: str):
    out = ROOT / "profile/out"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}"
    gputrace = out / f"{stem}.gputrace"
    stack = ExitStack()
    stack.enter_context(torch.mps.profiler.profile("interval,event"))
    gputrace = stack.enter_context(metal_capture(gputrace))
    return stack, gputrace


@dataclass
class QuantWeight:
    shape: tuple[int, int]
    type_name: str
    data: torch.Tensor


def profile_delta(a):
    device, dtype, batch, heads, dim = torch.device("mps"), torch.float16, 1, 32, 128
    q = torch.randn(batch, a.seq_len, heads, dim, device=device, dtype=dtype).contiguous()
    k, v = torch.randn_like(q), torch.randn_like(q)
    g = (torch.randn(batch, a.seq_len, heads, device=device, dtype=torch.float32) * -0.01).contiguous()
    beta = torch.rand(batch, a.seq_len, heads, device=device, dtype=dtype).contiguous()
    initial = torch.randn(batch, heads, dim, dim, device=device, dtype=torch.float32) if a.seq_len == 1 else None
    kernels = get_delta_rule_kernels()
    with torch.inference_mode():
        _, _ = kernels(q, k, v, g, beta, initial.clone() if initial is not None else None)
        sync()
        before = torch.mps.driver_allocated_memory()
        stack, gputrace = capture(f"delta_{'decode' if a.seq_len == 1 else 'prefill'}_L{a.seq_len}")
        start = perf_counter()
        with stack, torch.profiler.record_function(f"delta_{'decode' if a.seq_len == 1 else 'prefill'}"):
            out, state = kernels(q, k, v, g, beta, initial.clone() if initial is not None else None)
            torch.mps.synchronize()
        elapsed, after = perf_counter() - start, torch.mps.driver_allocated_memory()
    return {"target": "delta", "kind": "decode" if a.seq_len == 1 else "prefill", "shape": tuple(q.shape),
            "state_shape": tuple(state.shape), "output_shape": tuple(out.shape), "dtype": str(dtype),
            "elapsed": elapsed, "mem_before": before, "mem_after": after,
            "gputrace": gputrace if gputrace and gputrace.exists() else None}


def profile_quant_linear(a):
    device, dtype, m, k, n = torch.device("mps"), torch.float16, a.seq_len, a.in_features, a.out_features
    block_size, block_bytes = GGUF_BLOCK[a.quant_type]
    if k % block_size:
        raise ValueError(f"{a.quant_type} requires --in-features divisible by {block_size}")
    blocks, half_one = n * (k // block_size), torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)
    data = torch.randint(0, 16, (blocks * block_bytes,), dtype=torch.uint8)
    for o in range(0, data.numel(), block_bytes):
        data[o:o + 2] = half_one
        if a.quant_type in ("Q4_K", "Q5_K"):
            data[o + 2:o + 4] = 0; data[o + 4:o + 16] = 1
        elif a.quant_type == "Q6_K":
            data[o + 208:o + 210] = half_one; data[o + 192:o + 208] = 1
    x = torch.randn((m, k) if m > 1 else (k,), device=device, dtype=dtype).contiguous()
    weight = QuantWeight((n, k), a.quant_type, data.to(device))
    kernels = get_quant_linear_kernels()
    with torch.inference_mode():
        _ = kernels(x, weight)
        sync()
        before = torch.mps.driver_allocated_memory()
        stack, gputrace = capture(f"quant_linear_{a.quant_type.lower()}_"
                                  f"{'decode' if m == 1 else 'prefill'}_M{m}_K{k}_N{n}")
        start = perf_counter()
        with stack, torch.profiler.record_function(f"quant_linear_{a.quant_type}_{'decode' if m == 1 else 'prefill'}"):
            y = kernels(x, weight)
            torch.mps.synchronize()
        elapsed, after = perf_counter() - start, torch.mps.driver_allocated_memory()
    return {"target": "quant-linear", "kind": "decode" if m == 1 else "prefill", "shape": tuple(x.shape),
            "weight_shape": weight.shape, "output_shape": tuple(y.shape), "quant_type": a.quant_type,
            "dtype": str(dtype), "elapsed": elapsed, "mem_before": before, "mem_after": after,
            "gputrace": gputrace if gputrace and gputrace.exists() else None}


def print_summary(r, a):
    print("\n# profile summary")
    for k in ("target", "kind", "shape", "weight_shape", "output_shape", "state_shape", "quant_type",
              "dtype", "elapsed"):
        if k in r:
            print(f"{k}={r[k]}")
    print(f"mem_before={r['mem_before'] / 2**20:.2f}MiB mem_after={r['mem_after'] / 2**20:.2f}MiB")
    if r["gputrace"]:
        print(f"metal_capture={r['gputrace']}")
        print(f"open_xcode=open {r['gputrace']}")
    elif os.getenv("MTL_CAPTURE_ENABLED"):
        print("metal_capture=unavailable_or_not_persisted")
    else:
        extra = (f" --quant-type {a.quant_type} --in-features {a.in_features}"
                 f" --out-features {a.out_features}") if r["target"] == "quant-linear" else ""
        cmd = f"MTL_CAPTURE_ENABLED=1 python profile/profile_qwen35.py --target {a.target} --seq-len {a.seq_len}"
        print(f"metal_capture=disabled; example={cmd}{extra}")


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    a = parse_args()
    registry = {"delta": profile_delta, "quant-linear": profile_quant_linear, "quant": profile_quant_linear}
    if a.target not in registry:
        raise ValueError(f"unknown target {a.target!r}; expected one of {', '.join(registry)}")
    print_summary(registry[a.target](a), a)


if __name__ == "__main__":
    main()
