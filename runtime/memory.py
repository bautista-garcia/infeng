from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class KVCache:
    max_context: int
    buffer: torch.Tensor | None = None
    length: int = 0


@dataclass(slots=True)
class GDNState:
    conv: torch.Tensor | None = None
    recurrent: torch.Tensor | None = None


@dataclass(slots=True)
class Memory:
    layers: list[KVCache | GDNState]
