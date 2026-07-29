from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class KVCache:
    max_context: int
    buffer: torch.Tensor | None = None
    length: int = 0
    qg: torch.Tensor | None = None
    raw_k: torch.Tensor | None = None
    attention: torch.Tensor | None = None
    partials: torch.Tensor | None = None


@dataclass(slots=True)
class GDNState:
    conv: torch.Tensor | None = None
    recurrent: torch.Tensor | None = None


@dataclass(slots=True)
class Memory:
    layers: list[KVCache | GDNState]
