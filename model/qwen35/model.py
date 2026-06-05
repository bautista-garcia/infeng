from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .layers import Cache, DecoderLayer, Linear, RMSNorm


class TextModel(nn.Module):
    def __init__(self, config: str | Path | dict[str, Any]):
        super().__init__()
        if isinstance(config, str | Path):
            with open(config) as f:
                config = json.load(f)
        self.config = config["text_config"] if "text_config" in config else config
        # (L, d_model)
        self.embed_tokens = nn.Embedding(self.config["vocab_size"], self.config["hidden_size"])
        self.layers = nn.ModuleList(DecoderLayer(self.config, i) for i in range(self.config["num_hidden_layers"]))
        self.norm = RMSNorm(self.config["hidden_size"], eps=self.config.get("rms_norm_eps", 1e-6))

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, cache: Cache | None = None,
                use_cache: bool = False) -> tuple[torch.Tensor, Cache | None]:
        if use_cache and not cache:
            cache = Cache.from_config(self.config)

        # Scan for KV cache to know current seq_len
        past_len = 0
        for l_cache in cache.layers if cache else ():
            if getattr(l_cache, "length", None) is not None:
                past_len = int(l_cache.length.item()); break
            if getattr(l_cache, "keys", None) is not None:
                past_len = l_cache.keys.shape[2]; break

        # Build position_ids for new prompt
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + input_ids.shape[1], device=input_ids.device)
            position_ids = position_ids[None, :].expand(input_ids.shape[0], -1)

        # Embedding + 32 decoder layers + RMSNorm
        hidden_states = self.embed_tokens(input_ids)
        if cache:
            cache.allocate(self.config, hidden_states.shape[0], hidden_states.dtype, hidden_states.device)
        layer_caches = cache.layers if cache else [None] * len(self.layers)

        for layer, l_cache in zip(self.layers, layer_caches):
            hidden_states = layer(hidden_states, position_ids, attention_mask=attention_mask, cache=l_cache)

        return self.norm(hidden_states), cache


class ForCausalLM(nn.Module):
    def __init__(self, config: str | Path | dict[str, Any] = "config.json"):
        super().__init__()
        self.model = TextModel(config)
        self.config = self.model.config
        if self.config.get("tie_word_embeddings", True):
            self.lm_head = None
        else:
            self.lm_head = Linear(self.config["hidden_size"], self.config["vocab_size"], bias=False)

    @classmethod
    def build(cls, config: str | Path | dict[str, Any] = "config.json", device: torch.device | str | None = None,
              dtype: torch.dtype | None = None) -> "ForCausalLM":
        old_dtype = torch.get_default_dtype()
        if dtype is not None:
            torch.set_default_dtype(dtype)
        try:
            with torch.device(device) if device is not None else nullcontext():
                return cls(config)
        finally:
            torch.set_default_dtype(old_dtype)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, cache: Cache | None = None,
                use_cache: bool = False) -> tuple[torch.Tensor, Cache | None]:
        hidden_states, cache = self.model(input_ids, position_ids=position_ids, attention_mask=attention_mask,
                                          cache=cache, use_cache=use_cache)
        if self.lm_head is None:
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        return logits, cache
