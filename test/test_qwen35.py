from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from runtime.inference import InferenceEngine

# Llama.cpp via:
# llama-cli -m weights/Qwen3.5-9B-UD-Q4_K_XL.gguf -ngl 99 --temp 0 --reasoning off

WEIGHTS = ROOT / "weights/Qwen3.5-9B-UD-Q4_K_XL.gguf"
FIRST_TURN_RESPONSE = (
    "Hello! It seems like your message got cut off. Could you please share your "
    "name or let me know how I can assist you today? \U0001f60a"
)
LLAMA_CPP_GREEDY_MULTITURN_STRIPPED = (
    "Hello, Bautista! It's nice to meet you. How can I help you today?"
)

def test_greedy_multiturn_generation_matches_llama_cpp():
    messages = [
        {"role": "user", "content": "Hello, my name is"},
        {"role": "assistant", "content": FIRST_TURN_RESPONSE},
        {"role": "user", "content": "Bautista"},
    ]
    engine = InferenceEngine(WEIGHTS, os.getenv("QWEN35_TOKENIZER", "Qwen/Qwen3.5-9B"))
    ours = engine.tokenizer.decode(
        list(engine.generate(messages, max_new_tokens=64, thinking=False, temperature=0.0)),
        skip_special_tokens=False,
    )
    del engine; gc.collect(); torch.mps.empty_cache(); torch.mps.synchronize()
    assert ours.strip() == LLAMA_CPP_GREEDY_MULTITURN_STRIPPED, repr(ours)
