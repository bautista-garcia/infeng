from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FullAttentionMemory:
    max_len: int | None = None
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
    length: int | None = None

    def allocate(self, batch_size: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device):
        if self.max_len and self.keys is None:
            shape = (batch_size, num_heads, self.max_len, head_dim)
            self.keys = torch.empty(shape, dtype=dtype, device=device)
            self.values = torch.empty(shape, dtype=dtype, device=device)
            self.length = 0

    def update(self, keys: torch.Tensor,
               values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.max_len:
            self.allocate(keys.shape[0], keys.shape[1], keys.shape[3], keys.dtype, keys.device)
            idx = self.length + torch.arange(keys.shape[2], device=keys.device)
            self.keys.index_copy_(2, idx, keys); self.values.index_copy_(2, idx, values); self.length += keys.shape[2]
            return self.keys, self.values, torch.arange(self.max_len, device=keys.device) < self.length
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys, self.values = torch.cat((self.keys, keys), dim=2), torch.cat((self.values, values), dim=2)
        return self.keys, self.values, None


@dataclass
class DeltaNetMemory:
    conv_state: torch.Tensor | None = None
    recurrent_state: torch.Tensor | None = None


@dataclass
class RequestMemory:
    config: dict[str, Any]
    layers: list[FullAttentionMemory | DeltaNetMemory]

    @property
    def length(self) -> int:
        for layer in self.layers:
            if isinstance(layer, FullAttentionMemory):
                return layer.length if layer.length is not None else 0 if layer.keys is None else layer.keys.shape[2]
        return 0

    def position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = torch.arange(self.length, self.length + input_ids.shape[1], device=input_ids.device)
        return ids[None, :].expand(input_ids.shape[0], -1)

    def allocate(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        for layer in self.layers:
            if isinstance(layer, FullAttentionMemory):
                layer.allocate(batch_size, self.config["num_key_value_heads"], self.config["head_dim"], dtype, device)


class Memory:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.max_len = int(v) if (v := os.getenv("INFENG_STATIC_KV_LEN")) else None

    def new_request(self) -> RequestMemory:
        return RequestMemory(self.config, [
            FullAttentionMemory(self.max_len) if t == "full_attention" else DeltaNetMemory()
            for t in self.config["layer_types"]])
