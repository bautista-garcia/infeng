from __future__ import annotations

import os
import weakref
from pathlib import Path

os.environ.setdefault("USE_TORCH", "0")
from transformers import AutoTokenizer

from backend.metal.runtime import NativeModel


class Session:
    def __init__(self, engine: InferenceEngine):
        self.engine, self.native = engine, engine.native.session()
        self.pending, self.closed, self.sealed = [], False, True

    def _tokens(self, message, thinking):
        text = self.engine.tokenizer.apply_chat_template([{"role": "user", "content": message}], tokenize=False,
                                                         add_generation_prompt=True, enable_thinking=thinking)
        prefix = "" if not self.pending else ("\n" if self.sealed else "<|im_end|>\n")
        return list(self.engine.tokenizer.encode(prefix + text, add_special_tokens=False))

    def generate(self, message: str, max_new_tokens: int | None = None, thinking: bool = False,
                 stop_token_ids: list[int] | None = None, temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int | None = None):
        if self.closed: raise RuntimeError("session is closed")
        pending = self.pending + self._tokens(message, thinking); self.pending = pending
        remaining = self.engine.max_context - self.native.length - len(pending)
        count = remaining if max_new_tokens is None else min(max_new_tokens, remaining)
        if count < 0: raise ValueError(f"context exceeds {self.engine.max_context} tokens")
        im_end = self.engine.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if stop_token_ids is None: stop_token_ids = [x for x in (self.engine.tokenizer.eos_token_id, im_end) if x is not None]
        for _ in range(count):
            token = self.native.forward(pending, temperature, top_p, top_k); self.pending = pending = [token]
            if token in stop_token_ids: self.sealed = token == im_end; break
            yield token
        else: self.sealed = False

    @property
    def mapped_kv_bytes(self): return self.native.mapped_bytes

    def close(self):
        if not self.closed: self.native.close(); self.closed = True

    def __del__(self): self.close()


class InferenceEngine:
    def __init__(self, weights: str | Path, tokenizer: str, *, max_context: int = 65536, profile: bool = False):
        if not 0 < max_context <= 65536: raise ValueError(f"max_context must be between 1 and 65536, got {max_context}")
        self._session = None
        self.max_context, self.profile, self.device, self.dtype = max_context, profile, "metal4", "float16"
        self.native = NativeModel(weights, max_context=max_context, profile=profile)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def session(self):
        session = Session(self); self._session = weakref.ref(session); return session

    def generate(self, *args, **kwargs):
        live = self._session() if self._session else None
        if live is None or live.closed: live = self.session()
        return live.generate(*args, **kwargs)

    def close(self):
        live = self._session() if self._session else None
        if live is not None: live.close()
        if self.native: self.native.close(); self.native = None

    def profile_counters(self):
        if not self.profile: return {}
        counters = self.native.counters(); live = self._session() if self._session else None
        if live is not None and not live.closed: counters["mapped_kv_bytes"] = live.mapped_kv_bytes
        return counters

    def __del__(self): self.close()
