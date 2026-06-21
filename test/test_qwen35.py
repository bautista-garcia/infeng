from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from runtime.inference import InferenceEngine

WEIGHTS = ROOT / "weights/Qwen3.5-9B-UD-Q4_K_XL.gguf"
LLAMA_CPP_GREEDY_STRIPPED = "Hello! It seems"

def test_greedy_generation_matches_llama_cpp():
    prompt, n = "Hello, my name is", 4
    engine = InferenceEngine(WEIGHTS, os.getenv("QWEN35_TOKENIZER", "Qwen/Qwen3.5-9B"))
    ours = engine.tokenizer.decode(list(engine.generate(prompt, max_new_tokens=n, temperature=0.0)), skip_special_tokens=False)
    del engine; gc.collect(); torch.mps.empty_cache(); torch.mps.synchronize()
    assert ours.strip() == LLAMA_CPP_GREEDY_STRIPPED, repr(ours)
