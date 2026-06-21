from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .layers import Cache, DecoderLayer, Embedding, Linear, RMSNorm
from .weights import config_from_gguf


class TextModel(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(self.config["vocab_size"], self.config["hidden_size"])
        self.layers = nn.ModuleList(DecoderLayer(self.config, i) for i in range(self.config["num_hidden_layers"]))
        self.norm = RMSNorm(self.config["hidden_size"], eps=self.config.get("rms_norm_eps", 1e-6))

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, cache: Cache | None = None,
                use_cache: bool = False) -> tuple[torch.Tensor, Cache | None]:
        if use_cache and not cache:
            cache = Cache.from_config(self.config)

        past_len = 0
        for l_cache in cache.layers if cache else ():
            if getattr(l_cache, "length", None) is not None:
                past_len = l_cache.length; break
            if getattr(l_cache, "keys", None) is not None:
                past_len = l_cache.keys.shape[2]; break

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + input_ids.shape[1], device=input_ids.device)
            position_ids = position_ids[None, :].expand(input_ids.shape[0], -1)

        hidden_states = self.embed_tokens(input_ids)
        if cache:
            cache.allocate(self.config, hidden_states.shape[0], hidden_states.dtype, hidden_states.device)
        layer_caches = cache.layers if cache else [None] * len(self.layers)

        for layer, l_cache in zip(self.layers, layer_caches):
            hidden_states = layer(hidden_states, position_ids, attention_mask=attention_mask, cache=l_cache)

        return self.norm(hidden_states), cache


class ForCausalLM(nn.Module):
    def __init__(self, weights: str | Path = "weights/Qwen3.5-9B-UD-Q4_K_XL.gguf"):
        super().__init__()
        self.device, self.dtype = torch.device("mps"), torch.float16
        torch.set_default_dtype(self.dtype)
        weights = Path(weights)
        if weights.suffix != ".gguf":
            raise ValueError(f"ForCausalLM only accepts .gguf weights, got {weights}")
        self.config = config_from_gguf(weights)
        with torch.device(self.device):
            self.model = TextModel(self.config)
            self.lm_head = Linear(self.config["hidden_size"], self.config["vocab_size"], bias=False)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, cache: Cache | None = None,
                use_cache: bool = False) -> tuple[torch.Tensor, Cache | None]:
        hidden_states, cache = self.model(input_ids, position_ids=position_ids, attention_mask=attention_mask,
                                          cache=cache, use_cache=use_cache)
        return self.lm_head(hidden_states[:, -1:] if use_cache else hidden_states), cache
