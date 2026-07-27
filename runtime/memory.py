from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FullAttentionMemory:
    max_length: int
    initial_capacity: int = 0
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
    length: int = 0
    capacity: int = 0

    def _ensure_capacity(self, required: int, batch_size: int, num_heads: int, head_dim: int,
                         dtype: torch.dtype, device: torch.device):
        if required > self.max_length:
            raise ValueError(
                f"full-attention cache requires {required} tokens, "
                f"but max_length is {self.max_length}"
            )
        if self.keys is not None and required <= self.capacity:
            return

        new_capacity = max(required, self.initial_capacity, self.capacity * 2, 1)
        new_capacity = min(new_capacity, self.max_length)
        shape = (batch_size, num_heads, new_capacity, head_dim)
        keys = torch.empty(shape, dtype=dtype, device=device)
        values = torch.empty(shape, dtype=dtype, device=device)
        if self.keys is not None and self.length:
            keys[:, :, :self.length, :].copy_(self.keys[:, :, :self.length, :])
            values[:, :, :self.length, :].copy_(self.values[:, :, :self.length, :])
        self.keys, self.values, self.capacity = keys, values, new_capacity

    def allocate(self, batch_size: int, num_heads: int, head_dim: int, dtype: torch.dtype,
                 device: torch.device, initial_length: int = 0):
        self._ensure_capacity(initial_length, batch_size, num_heads, head_dim, dtype, device)

    def update(self, keys: torch.Tensor,
               values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_heads, token_count, head_dim = keys.shape
        was_empty = self.length == 0
        end = self.length + token_count
        self._ensure_capacity(end, batch_size, num_heads, head_dim, keys.dtype, keys.device)
        self.keys[:, :, self.length:end, :].copy_(keys)
        self.values[:, :, self.length:end, :].copy_(values)
        self.length = end
        if was_empty:
            # Preserve the contiguous tensors produced by prefill for the
            # current attention operation. The cache has already been filled
            # for subsequent decode steps.
            return keys, values
        return self.keys[:, :, :self.length, :], self.values[:, :, :self.length, :]


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
                return layer.length
        return 0

    def position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = torch.arange(self.length, self.length + input_ids.shape[1], device=input_ids.device)
        return ids[None, :].expand(input_ids.shape[0], -1)

    def allocate(self, batch_size: int, dtype: torch.dtype, device: torch.device, initial_length: int = 0):
        for layer in self.layers:
            if isinstance(layer, FullAttentionMemory):
                layer.allocate(batch_size, self.config["num_key_value_heads"], self.config["head_dim"],
                               dtype, device, initial_length)


class Memory:
    def __init__(self, config: dict[str, Any], max_length: int | None = None):
        self.config = config
        self.max_length = int(config["max_position_embeddings"] if max_length is None else max_length)
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")

    def new_request(self, initial_capacity: int = 0) -> RequestMemory:
        return RequestMemory(self.config, [
            FullAttentionMemory(self.max_length, initial_capacity) if t == "full_attention" else DeltaNetMemory()
            for t in self.config["layer_types"]])
