from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from runtime.inference import InferenceEngine

WEIGHTS = ROOT / "weights/Qwen3.5-9B-UD-Q4_K_XL.gguf"
FIRST_TURN_RESPONSE = (
    "Hello! It seems like your message got cut off. Could you please share your "
    "name or let me know how I can assist you today? \U0001f60a"
)
# This intentionally differs from llama.cpp: the append-only session preserves the exact first-turn prompt tokens in
# its KV cache, including Qwen's empty thinking block, while llama.cpp rerenders history without that historical block.
NATIVE_GREEDY_MULTITURN = (
    "Nice to meet you, Bautista! How can I help you today? Whether you have questions, need assistance with a task, "
    "or just want to chat, feel free to ask! \U0001f60a"
)

def test_greedy_multiturn_generation():
    engine = InferenceEngine(WEIGHTS, os.getenv("QWEN35_TOKENIZER", "Qwen/Qwen3.5-9B"))
    session = engine.session()
    first = engine.tokenizer.decode(list(session.generate("Hello, my name is", max_new_tokens=64, thinking=False,
                                                          temperature=0.0)), skip_special_tokens=False)
    assert first.strip() == FIRST_TURN_RESPONSE, repr(first)
    ours = engine.tokenizer.decode(list(session.generate("Bautista", max_new_tokens=64, thinking=False,
                                                          temperature=0.0)), skip_special_tokens=False)
    session.close(); engine.close()
    assert ours.strip() == NATIVE_GREEDY_MULTITURN, repr(ours)
