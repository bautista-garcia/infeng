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
    return {(counter["phase"], counter["name"]): counter for counter in model.kernel_counters()}


def delta(before, after):
    result = []
    for key, current in after.items():
        previous = before.get(key, {"gpu_time_ns": 0, "launches": 0})
        result.append({"phase": current["phase"], "name": current["name"],
                       "gpu_time_ns": current["gpu_time_ns"] - previous["gpu_time_ns"],
                       "launches": current["launches"] - previous["launches"]})
    return result


def merge(target, samples):
    for sample in samples:
        key = sample["phase"], sample["name"]
        current = target.setdefault(key, {"phase": sample["phase"], "name": sample["name"],
                                          "gpu_time_ns": 0, "launches": 0})
        current["gpu_time_ns"] += sample["gpu_time_ns"]; current["launches"] += sample["launches"]


def report(label, totals, forwards):
    rows = sorted(totals.values(), key=lambda sample: sample["gpu_time_ns"], reverse=True)
    total = sum(sample["gpu_time_ns"] for sample in rows)
    print(f"\n[{label}] kernel GPU time={total / forwards / 1e6:.3f} ms/forward")
    print("kernel                                      launches/forward  ms/forward  share")
    for sample in rows:
        milliseconds = sample["gpu_time_ns"] / forwards / 1e6
        share = sample["gpu_time_ns"] / total * 100 if total else 0
        print(f"{sample['name']:<44} {sample['launches'] / forwards:>16.2f} {milliseconds:>11.3f} {share:>6.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Profile every Qwen3.5 Metal kernel in prefill and decode")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--prefill", type=int, default=128)
    parser.add_argument("--decode", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()
    model = NativeModel(args.weights, profile=True)
    rng = random.Random(1); tokens = [rng.randrange(model.vocab_size) for _ in range(args.prefill)]
    for _ in range(args.warmup):
        session = model.session(); token = session.forward(tokens)
        for _ in range(args.decode): token = session.forward([token])
        session.close()
    prefill, decode = {}, {}
    for _ in range(args.iters):
        session = model.session(); before = snapshot(model); token = session.forward(tokens)
        middle = snapshot(model)
        for _ in range(args.decode): token = session.forward([token])
        after = snapshot(model); session.close()
        merge(prefill, delta(before, middle)); merge(decode, delta(middle, after))
    report("prefill", prefill, args.iters); report("decode", decode, args.iters * args.decode)
    model.close()


if __name__ == "__main__": main()
