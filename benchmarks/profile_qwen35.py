from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.metal.runtime import NativeModel

WEIGHTS = ROOT / "weights/Qwen3.5-9B-UD-Q4_K_XL.gguf"


def snapshot(model):
    return {"counters": model.counters(),
            "kernels": {(counter["phase"], counter["name"]): counter for counter in model.kernel_counters()}}


def delta(before, after):
    kernels = []
    for key, current in after["kernels"].items():
        previous = before["kernels"].get(key, {"gpu_time_ns": 0, "launches": 0})
        launches = current["launches"] - previous["launches"]
        if launches: kernels.append({"phase": current["phase"], "name": current["name"],
                                      "gpu_time_ns": current["gpu_time_ns"] - previous["gpu_time_ns"],
                                      "launches": launches})
    return {"gpu_time_ns": after["counters"]["gpu_time_ns"] - before["counters"]["gpu_time_ns"],
            "passes": after["counters"]["passes"] - before["counters"]["passes"], "kernels": kernels}


def merge(target, samples):
    for sample in samples:
        key = sample["phase"], sample["name"]
        current = target.setdefault(key, {"phase": sample["phase"], "name": sample["name"],
                                          "gpu_time_ns": 0, "launches": 0})
        current["gpu_time_ns"] += sample["gpu_time_ns"]; current["launches"] += sample["launches"]


def report(label, totals, gpu_time_ns, forwards):
    rows = sorted(totals.values(), key=lambda sample: sample["gpu_time_ns"], reverse=True)
    whole_ms = gpu_time_ns / forwards / 1e6
    covered_ms = sum(sample["gpu_time_ns"] for sample in rows) / forwards / 1e6
    print(f"\n[{label}] complete command-buffer GPU time={whole_ms:.3f} ms/forward")
    print("kernel                                      launches/forward  ms/forward  % of forward")
    for sample in rows:
        milliseconds = sample["gpu_time_ns"] / forwards / 1e6
        share = milliseconds / whole_ms * 100 if whole_ms else 0
        print(f"{sample['name']:<44} {sample['launches'] / forwards:>16.2f} {milliseconds:>11.3f} {share:>12.2f}%")
    if covered_ms < whole_ms:
        print(f"{'unattributed gaps/barriers':<44} {'':>16} {whole_ms - covered_ms:>11.3f} "
              f"{(whole_ms - covered_ms) / whole_ms * 100:>12.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Profile Qwen3.5 kernels within complete Metal forward passes")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--prefill", type=int, default=128)
    parser.add_argument("--decode", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()
    model = NativeModel(args.weights, profile=True)
    rng = random.Random(1); tokens = [rng.randrange(model.vocab_size) for _ in range(args.prefill)]
    for _ in range(args.warmup):
        session = model.session(); token = session.forward(tokens)
        for _ in range(args.decode): token = session.forward([token])
        session.close()
    prefill, decode, prefill_gpu, decode_gpu = {}, {}, 0, 0
    for _ in range(args.iters):
        session = model.session(); before = snapshot(model); token = session.forward(tokens)
        middle = snapshot(model); prefill_delta = delta(before, middle)
        for _ in range(args.decode): token = session.forward([token])
        after = snapshot(model); decode_delta = delta(middle, after); session.close()
        merge(prefill, prefill_delta["kernels"]); merge(decode, decode_delta["kernels"])
        prefill_gpu += prefill_delta["gpu_time_ns"]; decode_gpu += decode_delta["gpu_time_ns"]
    report("prefill", prefill, prefill_gpu, args.iters)
    report("decode", decode, decode_gpu, args.iters * args.decode)
    model.close()


if __name__ == "__main__": main()
