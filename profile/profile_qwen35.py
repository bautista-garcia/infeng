from __future__ import annotations

import argparse
import gc
import os
import sys
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from backend.metal import get_delta_rule_kernels


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3.5 MPS kernel profiler")
    p.add_argument("--target", default="delta")
    p.add_argument("--seq-len", "--seq_len", dest="seq_len", type=int, default=1)
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


def print_summary(r, a):
    print("\n# profile summary")
    for k in ("target", "kind", "shape", "output_shape", "state_shape", "dtype", "elapsed"):
        print(f"{k}={r[k]}")
    print(f"mem_before={r['mem_before'] / 2**20:.2f}MiB mem_after={r['mem_after'] / 2**20:.2f}MiB")
    if r["gputrace"]:
        print(f"metal_capture={r['gputrace']}")
        print(f"open_xcode=open {r['gputrace']}")
    elif os.getenv("MTL_CAPTURE_ENABLED"):
        print("metal_capture=unavailable_or_not_persisted")
    else:
        print(f"metal_capture=disabled; example=MTL_CAPTURE_ENABLED=1 python profile/profile_qwen35.py --target {a.target} --seq-len {a.seq_len}")


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    a = parse_args()
    registry = {"delta": profile_delta}
    if a.target not in registry:
        raise ValueError(f"unknown target {a.target!r}; expected one of {', '.join(registry)}")
    print_summary(registry[a.target](a), a)


if __name__ == "__main__":
    main()
