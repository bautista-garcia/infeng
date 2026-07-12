from __future__ import annotations

from functools import partialmethod
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .layers import DecoderLayer, Embedding, LMHead, RMSNorm
from .weights import config_from_gguf


class TextModel(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(self.config["vocab_size"], self.config["hidden_size"])
        self.layers = nn.ModuleList(DecoderLayer(self.config, i) for i in range(self.config["num_hidden_layers"]))
        self.norm = RMSNorm(self.config["hidden_size"], eps=self.config.get("rms_norm_eps", 1e-6))

    def _run(self, input_ids: torch.Tensor, position_ids: torch.Tensor, memory: Any,
             attention_mask: torch.Tensor | None = None, decode: bool = False) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        memory.allocate(hidden_states.shape[0], hidden_states.dtype, hidden_states.device)
        for layer, layer_memory in zip(self.layers, memory.layers):
            hidden_states = (layer.decode if decode else layer.prefill)(
                hidden_states, position_ids, layer_memory, attention_mask=attention_mask)
        return self.norm(hidden_states)

    prefill = partialmethod(_run, decode=False)
    decode = partialmethod(_run, decode=True)


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
            self.lm_head = LMHead(self.config["hidden_size"], self.config["vocab_size"])

    def _run(self, input_ids: torch.Tensor, position_ids: torch.Tensor, memory: Any,
             attention_mask: torch.Tensor | None = None, decode: bool = False) -> torch.Tensor:
        hidden_states = (self.model.decode if decode else self.model.prefill)(
            input_ids, position_ids, memory, attention_mask=attention_mask)
        return self.lm_head.decode(hidden_states if decode else hidden_states[:, -1:])

    prefill = partialmethod(_run, decode=False)
    decode = partialmethod(_run, decode=True)
