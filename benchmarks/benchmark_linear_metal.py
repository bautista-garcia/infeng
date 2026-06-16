from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from backend.metal import get_linear_kernels


def bench(fn, warmup: int, iters: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.mps.synchronize()
        start = perf_counter()
        for _ in range(iters):
            fn()
        torch.mps.synchronize()
    return (perf_counter() - start) / iters


def main():
    p = argparse.ArgumentParser(description="Metal linear kernel vs torch.nn.Linear on MPS")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=1)
    p.add_argument("--in-features", type=int, default=4096)
    p.add_argument("--out-features", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    a = p.parse_args()
    assert torch.backends.mps.is_available(), "MPS is required"
    torch.manual_seed(0)
    x = torch.randn(a.batch, a.seq, a.in_features, device="mps", dtype=torch.float16).contiguous()
    linear = torch.nn.Linear(a.in_features, a.out_features, bias=False, device="mps", dtype=torch.float16)
    kernels = get_linear_kernels()
    mine = lambda: kernels(x, linear.weight)
    torch_linear = lambda: linear(x)
    torch.mps.synchronize()
    torch.testing.assert_close(mine().cpu(), torch_linear().cpu(), atol=2e-2, rtol=2e-2)
    metal_s, torch_s = bench(mine, a.warmup, a.iters), bench(torch_linear, a.warmup, a.iters)
    mode = "decode" if a.seq == 1 else "prefill"
    print(f"mode={mode} batch={a.batch} seq={a.seq} in={a.in_features} out={a.out_features} dtype=fp16")
    print(f"metal={metal_s * 1000:.4f}ms torch={torch_s * 1000:.4f}ms speedup={torch_s / metal_s:.3f}x")


if __name__ == "__main__":
    main()
