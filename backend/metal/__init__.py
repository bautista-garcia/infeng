from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


_DELTA_RULE_KERNELS: "DeltaRuleKernels | None" = None
_QUANT_LINEAR_KERNELS: "QuantLinearKernels | None" = None
_FUSED_LAYER_KERNELS: "FusedLayerKernels | None" = None
# The 32-token Metal prefill solve can become unstable on real Qwen3.5 activations.
_DELTA_PREFILL_CHUNK = 16
QUANT_LINEAR_SPECS = {spec: ((256, 8) if spec[2] == 1024 else (128, 4)) for spec in (
    *[(q, 4096, n) for q in ("Q4_K", "Q5_K") for n in (1024, 4096, 8192, 12288)],
    ("Q4_K", 12288, 4096), ("Q5_K", 12288, 4096), ("Q6_K", 4096, 1024),
    ("Q6_K", 12288, 4096), ("Q6_K", 4096, 248320), ("Q8_0", 4096, 4096),
    ("IQ4_XS", 4096, 12288))}


def _compile(name: str):
    return torch.mps.compile_shader(Path(__file__).with_name(name).read_text())


@dataclass
class DeltaRuleKernels:
    lib: object

    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, g: torch.Tensor,
                 beta: torch.Tensor, initial_state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.decode if query.shape[1] == 1 else self.prefill)(query, key, value, g, beta, initial_state)

    def prefill(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, g: torch.Tensor,
                beta: torch.Tensor, initial_state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_heads, _ = query.shape
        output = torch.empty_like(value, memory_format=torch.contiguous_format)
        state = initial_state if initial_state is not None else torch.empty((batch_size, num_heads, 128, 128),
                                                                            dtype=torch.float32, device=query.device)
        for start in range(0, seq_len, _DELTA_PREFILL_CHUNK):
            end = min(start + _DELTA_PREFILL_CHUNK, seq_len)
            q, k, v, gg, bb = (x[:, start:end].contiguous() for x in (query, key, value, g, beta))
            out = torch.empty_like(v, memory_format=torch.contiguous_format)
            chunk_args = (out, state, q, k, v, gg, bb, batch_size, end - start, num_heads, v.stride(0),
                          v.stride(1), v.stride(2), v.stride(3), initial_state is not None or start > 0)
            self.lib.delta_rule_prefill(*chunk_args, threads=[128, batch_size * num_heads, 1], group_size=[128, 1, 1])
            output[:, start:end].copy_(out)
        return output, state

    def decode(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, g: torch.Tensor,
               beta: torch.Tensor, initial_state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_heads, _ = query.shape
        output = torch.empty_like(value, memory_format=torch.contiguous_format)
        state = initial_state if initial_state is not None else torch.empty((batch_size, num_heads, 128, 128),
                                                                            dtype=torch.float32, device=query.device)
        args = (output, state, query, key, value, g, beta, batch_size, seq_len, num_heads, value.stride(0),
                value.stride(1), value.stride(2), value.stride(3), initial_state is not None)
        self.lib.delta_rule_decode(*args, threads=[batch_size * num_heads * 512, 1, 1], group_size=[512, 1, 1])
        return output, state


def get_delta_rule_kernels() -> DeltaRuleKernels:
    global _DELTA_RULE_KERNELS
    if _DELTA_RULE_KERNELS is None:
        _DELTA_RULE_KERNELS = DeltaRuleKernels(_compile("delta_rule.metal"))
    return _DELTA_RULE_KERNELS


@dataclass
class QuantLinearKernels:
    lib: object

    def _spec(self, x: torch.Tensor, weight: Any):
        k, n = x.shape[-1], weight.shape[0]
        decode_tg, decode_rows = QUANT_LINEAR_SPECS[(weight.type_name, k, n)]
        base = f"{weight.type_name.lower()}_k{k}_n{n}"
        if weight.type_name == "IQ4_XS":
            return (getattr(self.lib, base + "_decode"), getattr(self.lib, base + "_prefill"),
                    decode_tg, decode_rows), k, n
        return (getattr(self.lib, f"{base}_decode_v2_tg{decode_tg}_reg"),
                getattr(self.lib, base + "_prefill_v2_bn16"), decode_tg, decode_rows), k, n

    def linear(self, x: torch.Tensor, weight: Any) -> torch.Tensor:
        return (self.decode if x.numel() // x.shape[-1] == 1 else self.prefill)(x, weight)

    def prefill(self, x: torch.Tensor, weight: Any) -> torch.Tensor:
        (_, prefill, _, _), k, n = self._spec(x, weight)
        m, x2 = x.numel() // k, x.reshape(x.numel() // k, k)
        mpad = (m + 31) // 32 * 32
        x2 = x2 if mpad == m else torch.nn.functional.pad(x2, (0, 0, 0, mpad - m))
        y2 = torch.empty((mpad, n), dtype=x.dtype, device=x.device)
        prefill(y2, x2, weight.data, mpad, threads=[128 * (n // 16), mpad // 32, 1], group_size=[128, 1, 1])
        return y2[:m].reshape(*x.shape[:-1], n)

    def decode(self, x: torch.Tensor, weight: Any) -> torch.Tensor:
        (decode, _, decode_tg, decode_rows), k, n = self._spec(x, weight)
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
        _QUANT_LINEAR_KERNELS = QuantLinearKernels(_compile("quant_linear.metal"))
    return _QUANT_LINEAR_KERNELS


@dataclass
class FusedLayerKernels:
    lib: object

    def gdn_in_proj_decode(self, x: torch.Tensor, w_qkv: Any, w_z: Any, w_b: Any, w_a: Any) -> tuple[torch.Tensor, ...]:
        lead = x.shape[:-1]
        yq = torch.empty((*lead, 8192), dtype=x.dtype, device=x.device)
        yz = torch.empty((*lead, 4096), dtype=x.dtype, device=x.device)
        yb = torch.empty((*lead, 32), dtype=x.dtype, device=x.device)
        ya = torch.empty_like(yb)
        self.lib.gdn_in_proj_decode(yq, yz, yb, ya, x, w_qkv.data, w_z.data, w_b.data, w_a.data,
                                    threads=[((12352 + 3) // 4) * 128, 1, 1], group_size=[128, 1, 1])
        return yq, yz, yb, ya

    def causal_conv_silu(self, x: torch.Tensor, weight: torch.Tensor,
                         prev_state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        b, seq_len, _ = x.shape
        has_prev = prev_state is not None
        prev = prev_state if has_prev else torch.empty((b, 8192, 4), dtype=x.dtype, device=x.device)
        y, state = torch.empty_like(x), torch.empty((b, 8192, 4), dtype=x.dtype, device=x.device)
        self.lib.gdn_causal_conv_silu(y, state, x, weight, prev, b, seq_len, has_prev,
                                      threads=[b * 8192 * max(seq_len, 4), 1, 1], group_size=[256, 1, 1])
        return y, state

    def rmsnorm_gated(self, x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        y, rows = torch.empty_like(x), x.numel() // 128
        self.lib.rmsnorm_gated_128(y, x, gate, weight, float(eps), threads=[128, rows, 1], group_size=[128, 1, 1])
        return y


def get_fused_layer_kernels() -> FusedLayerKernels:
    global _FUSED_LAYER_KERNELS
    if _FUSED_LAYER_KERNELS is None:
        _FUSED_LAYER_KERNELS = FusedLayerKernels(_compile("fused_kernels.metal"))
    return _FUSED_LAYER_KERNELS
