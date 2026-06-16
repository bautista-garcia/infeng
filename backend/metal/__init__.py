from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


_DELTA_RULE_KERNELS: "DeltaRuleKernels | None" = None
_LINEAR_KERNELS: "LinearKernels | None" = None


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
        state = initial_state if initial_state is not None else torch.empty((batch_size, num_heads, 128, 128),
                                                                            dtype=torch.float32, device=query.device)
        args = (output, state, query, key, value, g, beta, batch_size, seq_len, num_heads, value.stride(0),
                value.stride(1), value.stride(2), value.stride(3), initial_state is not None)
        if seq_len == 1:
            self.lib.delta_rule_decode(*args, threads=[batch_size * num_heads * 512, 1, 1], group_size=[512, 1, 1])
        else:
            for start in range(0, seq_len, 32):
                end = min(start + 32, seq_len)
                q, k, v, gg, bb = (x[:, start:end].contiguous() for x in (query, key, value, g, beta))
                out = torch.empty_like(v, memory_format=torch.contiguous_format)
                chunk_args = (out, state, q, k, v, gg, bb, batch_size, end - start, num_heads, v.stride(0),
                              v.stride(1), v.stride(2), v.stride(3), initial_state is not None or start > 0)
                self.lib.delta_rule_prefill(*chunk_args, threads=[128, batch_size * num_heads, 1],
                                            group_size=[128, 1, 1])
                output[:, start:end].copy_(out)
        return output, state


def get_delta_rule_kernels() -> DeltaRuleKernels:
    global _DELTA_RULE_KERNELS
    if _DELTA_RULE_KERNELS is None:
        source = (Path(__file__).with_name("delta_rule.metal")).read_text()
        _DELTA_RULE_KERNELS = DeltaRuleKernels(torch.mps.compile_shader(source))
    return _DELTA_RULE_KERNELS


@dataclass
class LinearKernels:
    lib: object

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, prefill: bool | None = None) -> torch.Tensor:
        if x.device.type != "mps" or weight.device.type != "mps" or x.dtype != torch.float16 or weight.dtype != torch.float16:
            raise RuntimeError("Metal linear requires fp16 x/weight on mps")
        if x.ndim < 2 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
            raise RuntimeError(f"Metal linear expects x=(..., K), weight=(N, K); got {x.shape}, {weight.shape}")
        if not x.is_contiguous() or not weight.is_contiguous():
            raise RuntimeError("Metal linear expects contiguous x and weight")
        shape, m, k, n = (*x.shape[:-1], weight.shape[0]), x.numel() // x.shape[-1], x.shape[-1], weight.shape[0]
        y = torch.empty(shape, dtype=x.dtype, device=x.device)
        use_prefill = m > 1 if prefill is None else prefill
        if use_prefill:
            self.lib.linear_prefill(y, x, weight, m, k, n, threads=[(n + 31) // 32 * 512, (m + 31) // 32, 1],
                                    group_size=[512, 1, 1])
        else:
            if k >= 4096 and k % 1024 == 0 and n % 4 == 0:
                self.lib.linear_decode_aligned(y, x, weight, m, k, n, threads=[(n + 3) // 4 * 256, m, 1],
                                               group_size=[256, 1, 1])
            elif k % 512 == 0 and n % 4 == 0:
                self.lib.linear_decode_aligned128(y, x, weight, m, k, n, threads=[(n + 3) // 4 * 128, m, 1],
                                                  group_size=[128, 1, 1])
            else:
                self.lib.linear_decode(y, x, weight, m, k, n, threads=[(n + 3) // 4 * 256, m, 1], group_size=[256, 1, 1])
        return y


def get_linear_kernels() -> LinearKernels:
    global _LINEAR_KERNELS
    if _LINEAR_KERNELS is None:
        source = (Path(__file__).with_name("linear.metal")).read_text()
        _LINEAR_KERNELS = LinearKernels(torch.mps.compile_shader(source))
    return _LINEAR_KERNELS
