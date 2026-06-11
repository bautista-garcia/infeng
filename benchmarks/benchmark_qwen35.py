from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from runtime.inference import InferenceEngine

WEIGHTS = ROOT / "weights/unsloth-Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-BF16.gguf"
BANDWIDTH_BPS, FLOPS = 200e9, 13.6e12

# For a tinygrad comparison:
# JITBEAM=2 python -c 'import sys, runpy; sys.setrecursionlimit(100000); import sys as s; s.argv=["tinygrad/llm/cli.py","--model","qwen3.5:4b","--warmup","--benchmark"]; runpy.run_path("tinygrad/llm/cli.py", run_name="__main__")'


def main():
    p = argparse.ArgumentParser(description="Qwen3.5 MPS macro benchmark")
    p.add_argument("--config", type=Path, default=ROOT / "config.json")
    p.add_argument("--weights", type=Path, default=WEIGHTS)
    p.add_argument("--tokenizer", default=os.getenv("QWEN35_TOKENIZER", "Qwen/Qwen3.5-4B"))
    p.add_argument("--dtype", default="auto")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--prefill", type=int, default=128)
    p.add_argument("--decode", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=5)
    a = p.parse_args()
    if os.getenv("TORCH_COMPILE") == "1":
        os.environ.setdefault("INFENG_STATIC_KV_LEN", str(a.prefill + a.decode))

    print(f"[load] weights={a.weights}", flush=True)
    engine = InferenceEngine(a.config, a.weights, a.tokenizer, device="mps", dtype=a.dtype)
    model, device, dtype = engine.model, engine.device, next(engine.model.parameters()).dtype
    assert device.type == "mps", f"this benchmark is MPS-only, got {device}"
    params, model_bytes = sum(p.numel() for p in model.parameters()), sum(p.numel() * p.element_size() for p in model.parameters())
    ids = torch.randint(0, model.config["vocab_size"], (a.batch, a.prefill), device=device)
    print(f"# Qwen3.5 MPS Benchmark\nconfig={a.config} tokenizer={a.tokenizer} dtype={dtype} "
          f"batch={a.batch} prefill={a.prefill} decode={a.decode} warmup={a.warmup} iters={a.iters}", flush=True)

    with torch.inference_mode():
        for i in range(a.warmup):
            print(f"warmup={i + 1}/{a.warmup}", flush=True)
            logits, cache = model(ids, use_cache=True)
            nxt = logits[:, -1].float().argmax(-1, keepdim=True)
            for _ in range(a.decode):
                logits, cache = model(nxt, cache=cache, use_cache=True)
                nxt = logits[:, -1].float().argmax(-1, keepdim=True)
            del logits, cache, nxt
            gc.collect(); torch.mps.empty_cache(); torch.mps.synchronize()

        ttft, decode, peak = [], [], torch.mps.driver_allocated_memory()
        for i in range(a.iters):
            print(f"iter={i + 1}/{a.iters} prefill", flush=True)
            torch.mps.synchronize(); start = perf_counter()
            logits, cache = model(ids, use_cache=True)
            box = logits[:, -1].float().argmax(-1, keepdim=True)
            torch.mps.synchronize(); ttft.append(perf_counter() - start)

            print(f"iter={i + 1}/{a.iters} decode={a.decode}", flush=True)
            torch.mps.synchronize(); start = perf_counter()
            for _ in range(a.decode):
                logits, cache = model(box, cache=cache, use_cache=True)
                box = logits[:, -1].float().argmax(-1, keepdim=True)
            torch.mps.synchronize(); decode.append(perf_counter() - start)
            peak = max(peak, torch.mps.driver_allocated_memory())
            del logits, cache, box
            gc.collect(); torch.mps.empty_cache(); torch.mps.synchronize()

    ttft_s, decode_s = sum(ttft) / len(ttft), sum(decode) / len(decode)
    tpot_s, ttl_s = decode_s / a.decode, ttft_s + decode_s
    print(f"TTFT={ttft_s * 1000:.2f}ms TPOT={tpot_s * 1000:.2f}ms TTL={ttl_s:.2f}s "
          f"tok/s={a.batch * (a.prefill + a.decode) / ttl_s:.2f} decode_tok/s={a.batch / tpot_s:.2f} "
          f"peak_mem={peak / 2**30:.2f}GiB MBU={(model_bytes * a.batch / tpot_s) / BANDWIDTH_BPS * 100:.2f}% "
          f"MFU={(2 * params * a.batch * a.prefill / ttft_s) / FLOPS * 100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
