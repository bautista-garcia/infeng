from __future__ import annotations

from dataclasses import dataclass

from backend.metal.runtime import Tensor


@dataclass(slots=True)
class KVCache:
    k: Tensor
    v: Tensor
    max_context: int


@dataclass(slots=True)
class GDNState:
    layer: int
    conv: Tensor | None = None
    recurrent: Tensor | None = None
    conv_slot: int = 1
    initialized: bool = False


@dataclass(slots=True)
class Memory:
    layers: list[KVCache | GDNState]
    length: int = 0
