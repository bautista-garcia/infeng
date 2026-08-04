from __future__ import annotations

from pathlib import Path
import struct
import time

from backend.metal.ops import Ops
from backend.metal.runtime import Runtime
from .layers import DecoderLayer
from .weights import config_from_gguf, load_weights


class ForCausalLM:
    def __init__(self, weights: str | Path, runtime: Runtime, max_context: int):
        path = Path(weights)
        if path.suffix != ".gguf": raise ValueError(f"ForCausalLM only accepts .gguf weights, got {path}")
        self.runtime, self.config, self.ops = runtime, config_from_gguf(path), Ops(runtime)
        self.weights, self.weight_arena = load_weights(path, runtime)
        self.layers = [DecoderLayer(self.config, i, self.weights, self.ops)
                       for i in range(self.config["num_hidden_layers"])]
        self.embedding = self.weights["model.embed_tokens.weight"]
        self.norm = self.weights["model.norm.weight"]
        self.head = self.weights["lm_head.weight"]
        self.rope = runtime.empty((max_context, 32, 2), "f16")
        self.token = runtime.empty((1,), "i32", shared=True)
        self.rng = runtime.upload(struct.pack("<Q", time.time_ns() & 0xffffffffffffffff), (1,), "i64")
        self.ops.begin()
        self.ops.p.dispatch("init_rope", [self.rope], [("I", max_context), ("f", 10_000_000.0)],
                            (((max_context * 32 + 255) // 256) * 256, 1, 1), (256, 1, 1))
        self.ops.end()

    def __call__(self, ids, memory, temperature=0.0, top_p=1.0, top_k=None):
        seq, start = ids.numel, memory.length
        self.ops.begin(); hidden = self.ops.embed(ids, self.embedding)
        for layer, state in zip(self.layers, memory.layers): hidden = layer(hidden, state, start, self.rope)
        last = hidden.view((1, 4096), offset=(seq - 1) * 4096)
        last = self.ops.rms(last, self.norm, 1e-6, "model.norm")
        logits = self.ops.linear(last, self.head, "model.logits")
        self.ops.sample(logits, self.token, self.rng, temperature, top_p, top_k); self.ops.end()
        memory.length += seq
        return self.runtime.read_i32(self.token)
