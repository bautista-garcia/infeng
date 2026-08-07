from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.metal.runtime import NativeModel

WEIGHTS = ROOT / "weights/Qwen3.5-9B-UD-Q4_K_XL.gguf"
BANDWIDTH_BPS, FLOPS = 200e9, 13.6e12

# For a tinygrad comparison:
# JITBEAM=2 python -c 'import sys, runpy; sys.setrecursionlimit(100000); import sys as s; s.argv=["tinygrad/llm/cli.py","--model","qwen3.5:4b","--warmup","--benchmark"]; runpy.run_path("tinygrad/llm/cli.py", run_name="__main__")'

# For a llama.cpp comparison:
# llama-bench -m weights/Qwen3.5-9B-UD-Q4_K_XL.gguf -r 3 -pg 32,32 -ngl 999


def run(model, tokens, decode):
    session = model.session(); start = perf_counter(); token = session.forward(tokens); ttft = perf_counter() - start
    start = perf_counter()
    for _ in range(decode): token = session.forward([token])
    elapsed, mapped = perf_counter() - start, session.mapped_bytes
    session.close(); return ttft, elapsed, mapped


def main():
    p = argparse.ArgumentParser(description="Qwen3.5 native Metal 4 macro benchmark")
    p.add_argument("--weights", type=Path, default=WEIGHTS)
    p.add_argument("--prefill", type=int, default=128)
    p.add_argument("--decode", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=5)
    a = p.parse_args()
    print(f"[load] weights={a.weights}", flush=True); model = NativeModel(a.weights)
    params, model_bytes = model.parameter_count, model.weight_bytes
    rng = random.Random(1); tokens = [rng.randrange(model.vocab_size) for _ in range(a.prefill)]
    print(f"# Qwen3.5 Metal 4 Benchmark\nweights={a.weights} dtype=float16 "
          f"batch=1 prefill={a.prefill} decode={a.decode} warmup={a.warmup} iters={a.iters}", flush=True)
    for i in range(a.warmup): print(f"warmup={i + 1}/{a.warmup}", flush=True); run(model, tokens, a.decode)
    ttft, decode, mapped = [], [], 0
    for i in range(a.iters):
        print(f"iter={i + 1}/{a.iters}", flush=True); ptime, dtime, mapped = run(model, tokens, a.decode)
        ttft.append(ptime); decode.append(dtime)
    ttft_s, decode_s = sum(ttft) / len(ttft), sum(decode) / len(decode)
    tpot_s, ttl_s = decode_s / a.decode, ttft_s + decode_s
    print(f"TTFT={ttft_s * 1000:.2f}ms TPOT={tpot_s * 1000:.2f}ms TTL={ttl_s:.2f}s "
          f"tok/s={(a.prefill + a.decode) / ttl_s:.2f} prefill_tok/s={a.prefill / ttft_s:.2f} "
          f"decode_tok/s={1 / tpot_s:.2f} mapped_kv={mapped / 2**20:.2f}MiB "
          f"MBU={(model_bytes / tpot_s) / BANDWIDTH_BPS * 100:.2f}% "
          f"MFU={(2 * params * a.prefill / ttft_s) / FLOPS * 100:.2f}%", flush=True)
    model.close()


if __name__ == "__main__": main()
