from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


_DELTA_RULE_KERNELS: "DeltaRuleKernels | None" = None
_QUANT_LINEAR_KERNELS: "QuantLinearKernels | None" = None
# The 32-token Metal prefill solve can become unstable on real Qwen3.5 activations.
_DELTA_PREFILL_CHUNK = 16


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
            for start in range(0, seq_len, _DELTA_PREFILL_CHUNK):
                end = min(start + _DELTA_PREFILL_CHUNK, seq_len)
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
class QuantLinearKernels:
    lib: object

    def __call__(self, x: torch.Tensor, weight: Any) -> torch.Tensor:
        m, k, n = x.numel() // x.shape[-1], x.shape[-1], weight.shape[0]
        if k != weight.shape[1]:
            raise RuntimeError(f"quant linear expects x=(...,K), weight=(N,K); got {x.shape}, {weight.shape}")
        registry = {
            ("Q4_K", 4096, 1024): (self.lib.q4_k_k4096_n1024_decode_v2_tg256_reg,
                                    self.lib.q4_k_k4096_n1024_prefill_v2_bn16, 256, 8),
            ("Q4_K", 4096, 4096): (self.lib.q4_k_k4096_n4096_decode_v2_tg128_reg,
                                    self.lib.q4_k_k4096_n4096_prefill_v2_bn16, 128, 4),
            ("Q4_K", 12288, 4096): (self.lib.q4_k_k12288_n4096_decode_v2_tg128_reg,
                                     self.lib.q4_k_k12288_n4096_prefill_v2_bn16, 128, 4),
            ("Q4_K", 4096, 8192): (self.lib.q4_k_k4096_n8192_decode_v2_tg128_reg,
                                    self.lib.q4_k_k4096_n8192_prefill_v2_bn16, 128, 4),
            ("Q4_K", 4096, 12288): (self.lib.q4_k_k4096_n12288_decode_v2_tg128_reg,
                                     self.lib.q4_k_k4096_n12288_prefill_v2_bn16, 128, 4),
            ("Q5_K", 4096, 1024): (self.lib.q5_k_k4096_n1024_decode_v2_tg256_reg,
                                    self.lib.q5_k_k4096_n1024_prefill_v2_bn16, 256, 8),
            ("Q5_K", 4096, 4096): (self.lib.q5_k_k4096_n4096_decode_v2_tg128_reg,
                                    self.lib.q5_k_k4096_n4096_prefill_v2_bn16, 128, 4),
            ("Q5_K", 12288, 4096): (self.lib.q5_k_k12288_n4096_decode_v2_tg128_reg,
                                     self.lib.q5_k_k12288_n4096_prefill_v2_bn16, 128, 4),
            ("Q5_K", 4096, 8192): (self.lib.q5_k_k4096_n8192_decode_v2_tg128_reg,
                                    self.lib.q5_k_k4096_n8192_prefill_v2_bn16, 128, 4),
            ("Q5_K", 4096, 12288): (self.lib.q5_k_k4096_n12288_decode_v2_tg128_reg,
                                     self.lib.q5_k_k4096_n12288_prefill_v2_bn16, 128, 4),
            ("Q6_K", 4096, 1024): (self.lib.q6_k_k4096_n1024_decode_v2_tg256_reg,
                                    self.lib.q6_k_k4096_n1024_prefill_v2_bn16, 256, 8),
            ("Q6_K", 12288, 4096): (self.lib.q6_k_k12288_n4096_decode_v2_tg128_reg,
                                     self.lib.q6_k_k12288_n4096_prefill_v2_bn16, 128, 4),
            ("Q6_K", 4096, 248320): (self.lib.q6_k_k4096_n248320_decode_v2_tg128_reg,
                                      self.lib.q6_k_k4096_n248320_prefill_v2_bn16, 128, 4),
            ("Q8_0", 4096, 4096): (self.lib.q8_0_k4096_n4096_decode_v2_tg128_reg,
                                    self.lib.q8_0_k4096_n4096_prefill_v2_bn16, 128, 4),
            ("IQ4_XS", 4096, 12288): (self.lib.iq4_xs_k4096_n12288_decode,
                                       self.lib.iq4_xs_k4096_n12288_prefill, 128, 4),
        }
        if (weight.type_name, k, n) not in registry:
            raise RuntimeError(f"unsupported quant linear specialization {(weight.type_name, k, n)}")
        decode, prefill, decode_tg, decode_rows = registry[(weight.type_name, k, n)]
        if m != 1:
            x2, mpad = x.reshape(m, k), (m + 31) // 32 * 32
            x2 = x2 if mpad == m else torch.nn.functional.pad(x2, (0, 0, 0, mpad - m))
            y2 = torch.empty((mpad, n), dtype=x.dtype, device=x.device)
            prefill(y2, x2, weight.data, mpad, threads=[128 * (n // 16), mpad // 32, 1], group_size=[128, 1, 1])
            return y2[:m].reshape(*x.shape[:-1], n)

        y = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
        threadgroups = (n + decode_rows - 1) // decode_rows
        decode(y, x, weight.data, threads=[threadgroups * decode_tg, 1, 1], group_size=[decode_tg, 1, 1])
        return y

    def embed_q4(self, ids: torch.Tensor, weight: Any, dtype: torch.dtype) -> torch.Tensor:
        y = torch.empty((*ids.shape, weight.shape[1]), dtype=dtype, device=ids.device)
        tokens, k = ids.numel(), weight.shape[1]
        self.lib.q4_k_embed(y, ids, weight.data, tokens, k,
                            threads=[((k + 255) // 256) * 256, tokens, 1], group_size=[256, 1, 1])
        return y


def get_quant_linear_kernels() -> QuantLinearKernels:
    global _QUANT_LINEAR_KERNELS
    if _QUANT_LINEAR_KERNELS is None:
        source = (Path(__file__).with_name("quant_linear.metal")).read_text()
        _QUANT_LINEAR_KERNELS = QuantLinearKernels(torch.mps.compile_shader(source))
    return _QUANT_LINEAR_KERNELS
