from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from model.qwen35.layers import DeltaNetCache, GatedDeltaNet
from runtime.inference import InferenceEngine

WEIGHTS = ROOT / "weights/unsloth-Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-BF16.gguf"
HARDWARE_CEILINGS = {
    "mps": {"name": "MacBook M2 Pro", "bandwidth_Bps": 200e9, "flops": 13.6e12},
    "cuda": {"name": None, "bandwidth_Bps": None, "flops": None},
    "cpu": {"name": None, "bandwidth_Bps": None, "flops": None},
}
MACRO = (("Pure Decode Stress", 1, 128, 1024), ("Deep Prefill Stress", 1, 2048, 128),
         ("Batch Scaling Sweep", 1, 512, 512), ("Batch Scaling Sweep", 4, 512, 512),
         ("Batch Scaling Sweep", 16, 512, 512))


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timed(fn: Callable[[], None], device: torch.device) -> float:
    if device.type == "cuda":
        start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        torch.cuda.synchronize(device)
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(stop) / 1000
    sync(device)
    start = perf_counter()
    fn()
    sync(device)
    return perf_counter() - start


def fmt(x: float | int | None, suffix: str = "", precision: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{precision}f}{suffix}"


def log(message: str):
    print(message, flush=True)


def mem(device: torch.device) -> int | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    return torch.mps.driver_allocated_memory() if device.type == "mps" else None


def cleanup(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    sync(device)


def reset_mem(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def table(title: str, headers: tuple[str, ...], rows: list[tuple[object, ...]]):
    print(f"\n## {title}")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(map(str, row)) + " |")


@torch.no_grad()
def macro(engine: InferenceEngine, scenarios, warmup: int, iters: int, decode_steps: int):
    model, device, vocab = engine.model, engine.device, engine.model.config["vocab_size"]
    params = sum(p.numel() for p in model.parameters())
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    rows, ceiling = [], HARDWARE_CEILINGS.get(device.type, {})
    for scenario_idx, (name, batch, prompt_len, out_len) in enumerate(scenarios, 1):
        ids = torch.randint(0, vocab, (batch, prompt_len), device=device)
        sample_tokens = min(max(out_len - 1, 1), decode_steps)
        log(f"[macro {scenario_idx}/{len(scenarios)}] {name}: B={batch} input={prompt_len} output={out_len} "
            f"measured_decode={sample_tokens}")
        for i in range(warmup):
            log(f"  warmup {i + 1}/{warmup}")
            logits, cache = model(ids, use_cache=True)
            nxt = logits[:, -1].float().argmax(-1, keepdim=True)
            for _ in range(sample_tokens):
                logits, cache = model(nxt, cache=cache, use_cache=True)
                nxt = logits[:, -1].float().argmax(-1, keepdim=True)
            del logits, cache, nxt
            cleanup(device)
        reset_mem(device)
        ttft, decode, peak = [], [], mem(device) or 0
        for i in range(iters):
            cache = box = None
            log(f"  iter {i + 1}/{iters}: prefill")

            def prefill():
                nonlocal cache, box
                logits, cache = model(ids, use_cache=True)
                box = logits[:, -1].float().argmax(-1, keepdim=True)

            ttft.append(timed(prefill, device))
            log(f"  iter {i + 1}/{iters}: decode {sample_tokens} steps")

            def decode_loop():
                nonlocal cache, box
                for _ in range(sample_tokens):
                    logits, cache = model(box, cache=cache, use_cache=True)
                    box = logits[:, -1].float().argmax(-1, keepdim=True)

            decode.append(timed(decode_loop, device))
            peak = max(peak, mem(device) or 0)
            del cache, box
            cleanup(device)
        ttft_s, dec_s, dec_tokens = sum(ttft) / len(ttft), sum(decode) / len(decode), sample_tokens
        tpot_s = dec_s / dec_tokens
        total_s = ttft_s + tpot_s * max(out_len - 1, 1)
        mbu = (model_bytes * batch / tpot_s) / ceiling["bandwidth_Bps"] * 100 if ceiling.get("bandwidth_Bps") else None
        mfu = (2 * params * batch * prompt_len / ttft_s) / ceiling["flops"] * 100 if ceiling.get("flops") else None
        rows.append((name, batch, prompt_len, out_len, dec_tokens, fmt(ttft_s * 1000), fmt(tpot_s * 1000),
                     fmt(batch * out_len / total_s), fmt(batch / tpot_s), fmt(peak / 2**30, " GiB"), fmt(mbu, "%"),
                     fmt(mfu, "%")))
        log(f"  done: TTFT={fmt(ttft_s * 1000)}ms TPOT={fmt(tpot_s * 1000)}ms peak={fmt(peak / 2**30, ' GiB')}")
    return rows


@torch.no_grad()
def micro(config: dict, device: torch.device, dtype: torch.dtype, scenarios, warmup: int, iters: int):
    layer = GatedDeltaNet(config, 0).to(device=device, dtype=dtype).eval()
    rows = []
    for scenario_idx, (_, batch, seq_len, _) in enumerate(scenarios, 1):
        log(f"[micro {scenario_idx}/{len(scenarios)}] GatedDeltaNet.forward: B={batch} seq={seq_len}")
        x = torch.randn(batch, seq_len, config["hidden_size"], device=device, dtype=dtype)
        for i in range(warmup):
            log(f"  warmup {i + 1}/{warmup}")
            layer(x, cache=DeltaNetCache())
        cleanup(device)
        lat = []
        for i in range(iters):
            log(f"  iter {i + 1}/{iters}")
            lat.append(timed(lambda: layer(x, cache=DeltaNetCache()), device))
        ms = sum(lat) / len(lat) * 1000
        rows.append(("GatedDeltaNet.forward", batch, seq_len, fmt(ms), fmt(ms * 1000 / (batch * seq_len)),
                     device.type, str(dtype).removeprefix("torch.")))
        log(f"  done: latency={fmt(ms)}ms")
        del x
        cleanup(device)
    return rows


def args():
    p = argparse.ArgumentParser(description="Qwen3.5 macro/micro benchmark")
    p.add_argument("--config", type=Path, default=ROOT / "config.json")
    p.add_argument("--weights", type=Path, default=WEIGHTS)
    p.add_argument("--tokenizer", default=os.getenv("QWEN35_TOKENIZER", "Qwen/Qwen3.5-4B"))
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--decode-steps", type=int, default=16, help="bounded measured decode window per macro scenario")
    p.add_argument("--macro-out", type=int, help="override output tokens for smoke tests")
    p.add_argument("--max-batch", type=int, help="drop scenarios above this batch size")
    p.add_argument("--skip-macro", action="store_true")
    p.add_argument("--skip-micro", action="store_true")
    return p.parse_args()


def main():
    a = args()
    warmup = a.warmup
    scenarios = tuple((n, b, i, a.macro_out or o) for n, b, i, o in MACRO if a.max_batch is None or b <= a.max_batch)
    log(f"[load] building InferenceEngine from {a.weights}")
    engine = InferenceEngine(a.config, a.weights, a.tokenizer, device=a.device, dtype=a.dtype)
    dtype = next(engine.model.parameters()).dtype
    with open(a.config) as f:
        raw_config = json.load(f)
    config = raw_config.get("text_config", raw_config)
    log(f"# Qwen3.5 Benchmark\nconfig: {a.config}\nweights: {a.weights}\ntokenizer: {a.tokenizer}\n"
        f"device: {engine.device} dtype: {dtype} warmup: {warmup} iters: {a.iters} decode_steps: {a.decode_steps}")
    if not a.skip_macro:
        headers = ("scenario", "B", "input", "output", "measured decode", "TTFT ms", "TPOT ms", "tok/s", "decode tok/s",
                   "peak mem", "MBU", "MFU")
        table("Macro Benchmark", headers, macro(engine, scenarios, warmup, a.iters, a.decode_steps))
    if not a.skip_micro:
        table("Micro Benchmark", ("layer", "B", "seq", "latency ms", "us/token", "device", "dtype"),
              micro(config, engine.device, dtype, scenarios, warmup, a.iters))


if __name__ == "__main__":
    main()
