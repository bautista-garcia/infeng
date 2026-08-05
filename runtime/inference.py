from __future__ import annotations

import os
import weakref
from array import array
from pathlib import Path

os.environ.setdefault("USE_TORCH", "0")
from transformers import AutoTokenizer

from backend.metal.runtime import Runtime
from model.qwen35.model import ForCausalLM
from .memory import GDNState, KVCache, Memory


class Session:
    def __init__(self, engine: InferenceEngine):
        self.engine, self.sparse = engine, engine.runtime.sparse_cache(engine.max_context)
        layers, kv = [], 0
        for index, kind in enumerate(engine.model.config["layer_types"]):
            if kind == "full_attention":
                layers.append(KVCache(self.sparse.buffers[2 * kv], self.sparse.buffers[2 * kv + 1],
                                      engine.max_context)); kv += 1
            else: layers.append(GDNState(index))
        self.memory, self.transcript, self.closed, self.sealed = Memory(layers), [], False, True
        self.ids = engine.runtime.empty((engine.max_context,), "i32", shared=True)

    def _tokens(self, message, thinking):
        text = self.engine.tokenizer.apply_chat_template([{"role": "user", "content": message}], tokenize=False,
                                                         add_generation_prompt=True, enable_thinking=thinking)
        prefix = "" if not self.transcript else ("\n" if self.sealed else "<|im_end|>\n")
        return list(self.engine.tokenizer.encode(prefix + text, add_special_tokens=False))

    def generate(self, message: str, max_new_tokens: int | None = None, thinking: bool = False,
                 stop_token_ids: list[int] | None = None, temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int | None = None):
        if self.closed: raise RuntimeError("session is closed")
        turn = self._tokens(message, thinking); self.transcript.extend(turn)
        suffix = self.transcript[self.memory.length:]
        remaining = self.engine.max_context - len(self.transcript)
        count = remaining if max_new_tokens is None else min(max_new_tokens, remaining)
        if count < 0: raise ValueError(f"context exceeds {self.engine.max_context} tokens")
        im_end = self.engine.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if stop_token_ids is None: stop_token_ids = [x for x in (self.engine.tokenizer.eos_token_id, im_end) if x is not None]
        pending = suffix
        for step in range(count):
            end = self.memory.length + len(pending)
            if end > self.engine.max_context:
                raise ValueError(f"context requires {end} tokens, but max_context is {self.engine.max_context}")
            self.sparse.ensure(end)
            payload = array("i", pending); ids = self.ids.view((len(pending),))
            self.engine.runtime.write(ids, payload)
            token = self.engine.model(ids, self.memory, temperature, top_p, top_k); self.transcript.append(token)
            if token in stop_token_ids: self.sealed = token == im_end; break
            yield token
            pending = [token]
        else: self.sealed = False

    @property
    def mapped_kv_bytes(self): return self.sparse.mapped_bytes

    def close(self):
        if not self.closed: self.sparse.close(); self.closed = True

    def __del__(self): self.close()


class InferenceEngine:
    def __init__(self, weights: str | Path, tokenizer: str, *, max_context: int = 65536, profile: bool = False):
        if not 0 < max_context <= 65536: raise ValueError(f"max_context must be between 1 and 65536, got {max_context}")
        self.runtime, self.max_context, self.profile = Runtime(profile), max_context, profile
        self.device, self.dtype = "metal4", "float16"
        self.model = ForCausalLM(weights, self.runtime, max_context)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self._session = None

    def session(self):
        live = self._session() if self._session else None
        if live is not None and not live.closed: raise RuntimeError("only one live session is supported")
        session = Session(self); self._session = weakref.ref(session); return session

    def generate(self, *args, **kwargs):
        live = self._session() if self._session else None
        if live is None or live.closed: live = self.session()
        return live.generate(*args, **kwargs)

    def close(self):
        live = self._session() if self._session else None
        if live is not None: live.close()
        if self.runtime: self.runtime.close(); self.runtime = None

    def profile_counters(self):
        counters = self.runtime.counters()
        live = self._session() if self._session else None
        if counters and live is not None and not live.closed: counters["mapped_kv_bytes"] = live.mapped_kv_bytes
        return counters

    def __del__(self): self.close()
