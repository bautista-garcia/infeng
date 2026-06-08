from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


_DELTA_RULE_KERNELS: "DeltaRuleKernels | None" = None


@dataclass
class DeltaRuleKernels:
    lib: object

    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, g: torch.Tensor,
                 beta: torch.Tensor, initial_state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if query.device.type != "mps" or key.device.type != "mps" or value.device.type != "mps":
            raise RuntimeError("Metal delta rule requires query/key/value on mps")
        if query.dtype != torch.float16 or key.dtype != torch.float16 or value.dtype != torch.float16 or beta.dtype != torch.float16:
            raise RuntimeError("Metal delta rule requires fp16 query/key/value/beta")
        if g.dtype != torch.float32:
            raise RuntimeError("Metal delta rule requires fp32 g")
        if query.shape != key.shape or query.ndim != 4 or value.ndim != 4 or g.ndim != 3 or beta.ndim != 3:
            raise RuntimeError(f"Metal delta rule expects q/k/v=(B,L,H,D), g/beta=(B,L,H); got {query.shape}, {key.shape}, {value.shape}, {g.shape}, {beta.shape}")

        batch_size, seq_len, num_heads, key_dim = query.shape
        if value.shape != (batch_size, seq_len, num_heads, 128) or g.shape != (batch_size, seq_len, num_heads) or beta.shape != g.shape:
            raise RuntimeError(f"Metal delta rule expects matching Qwen3.5 value/g/beta shapes, got value={value.shape}, g={g.shape}, beta={beta.shape}")
        if (num_heads, key_dim) != (32, 128):
            raise RuntimeError(f"Metal delta rule only supports H=32, Dk=Dv=128, got H={num_heads}, Dk={key_dim}")
        if not query.is_contiguous() or not key.is_contiguous() or not g.is_contiguous() or not beta.is_contiguous():
            raise RuntimeError("Metal delta rule expects contiguous query/key/g/beta")
        if initial_state is not None and (initial_state.device.type != "mps" or initial_state.dtype != torch.float32 or initial_state.shape != (batch_size, num_heads, 128, 128)):
            raise RuntimeError(f"Metal delta rule initial_state must be fp32 mps with shape {(batch_size, num_heads, 128, 128)}")

        output = torch.empty_like(value, memory_format=torch.contiguous_format)
        state = torch.empty((batch_size, num_heads, 128, 128), dtype=torch.float32, device=query.device)
        launch = torch.empty(batch_size * num_heads, dtype=torch.int32, device=query.device)
        state_in = state if initial_state is None else initial_state
        args = (launch, output, state, query, key, value, g, beta, state_in, batch_size, seq_len, num_heads,
                value.stride(0), value.stride(1), value.stride(2), value.stride(3), initial_state is not None)
        (self.lib.delta_rule_decode if seq_len == 1 else self.lib.delta_rule_prefill)(*args)
        return output, state


def get_delta_rule_kernels() -> DeltaRuleKernels:
    global _DELTA_RULE_KERNELS
    if _DELTA_RULE_KERNELS is None:
        source = (Path(__file__).with_name("delta_rule.metal")).read_text()
        _DELTA_RULE_KERNELS = DeltaRuleKernels(torch.mps.compile_shader(source))
    return _DELTA_RULE_KERNELS
