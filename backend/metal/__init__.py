from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


_DELTA_RULE_KERNELS: DeltaRuleKernels | None = None
_QUANT_LINEAR_KERNELS: QuantLinearKernels | None = None
_FUSED_LAYER_KERNELS: FusedLayerKernels | None = None
_ATTENTION_KERNELS: AttentionKernels | None = None
# The 32-token Metal prefill solve can become unstable on real Qwen3.5 activations.
_DELTA_PREFILL_CHUNK = 16
_ATTENTION_SPLITS = 8
MLP_HIDDEN_SIZE = 4096
MLP_INTERMEDIATE_SIZE = 12288
MLP_GATE_UP_KERNELS = {
    ("Q4_K", "Q4_K"): ("mlp_gate_up_q4_k_decode", 64, 4),
    ("Q5_K", "Q5_K"): ("mlp_gate_up_q5_k_decode", 64, 2),
    ("IQ4_XS", "IQ4_XS"): ("mlp_gate_up_iq4_xs_decode", 64, 4),
}
QUANT_LINEAR_KERNELS = {
    ("Q4_K", 4096, 1024): ("q4_k_k4096_n1024_decode", "q4_k_k4096_n1024_prefill", 64, 4),
    ("Q4_K", 4096, 4096): ("q4_k_k4096_n4096_decode", "q4_k_k4096_n4096_prefill", 64, 2),
    ("Q4_K", 4096, 8192): ("q4_k_k4096_n8192_decode", "q4_k_k4096_n8192_prefill", 64, 4),
    ("Q4_K", 4096, 12288): ("q4_k_k4096_n12288_decode", "q4_k_k4096_n12288_prefill", 64, 4),
    ("Q4_K", 12288, 4096): ("q4_k_k12288_n4096_decode", "q4_k_k12288_n4096_prefill", 64, 4),
    ("Q5_K", 4096, 1024): ("q5_k_k4096_n1024_decode", "q5_k_k4096_n1024_prefill", 64, 2),
    ("Q5_K", 4096, 4096): ("q5_k_k4096_n4096_decode", "q5_k_k4096_n4096_prefill", 64, 2),
    ("Q5_K", 4096, 8192): ("q5_k_k4096_n8192_decode", "q5_k_k4096_n8192_prefill", 64, 2),
    ("Q5_K", 4096, 12288): ("q5_k_k4096_n12288_decode", "q5_k_k4096_n12288_prefill", 64, 2),
    ("Q5_K", 12288, 4096): ("q5_k_k12288_n4096_decode", "q5_k_k12288_n4096_prefill", 64, 2),
    ("Q6_K", 4096, 1024): ("q6_k_k4096_n1024_decode", "q6_k_k4096_n1024_prefill", 64, 4),
    ("Q6_K", 12288, 4096): ("q6_k_k12288_n4096_decode", "q6_k_k12288_n4096_prefill", 64, 4),
    ("Q6_K", 4096, 248320): ("q6_k_k4096_n248320_decode", "q6_k_k4096_n248320_prefill", 64, 4),
    ("Q8_0", 4096, 4096): ("q8_0_k4096_n4096_decode", "q8_0_k4096_n4096_prefill", 128, 2),
    ("IQ4_XS", 4096, 12288): ("iq4_xs_k4096_n12288_decode", "iq4_xs_k4096_n12288_prefill", 64, 4),
}


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
        has_initial = initial_state is not None
        state = initial_state if has_initial else query.new_empty((batch_size, num_heads, 128, 128),
                                                                  dtype=torch.float32)
        for start in range(0, seq_len, _DELTA_PREFILL_CHUNK):
            end = min(start + _DELTA_PREFILL_CHUNK, seq_len)
            q, k, v, gg, bb = (x[:, start:end].contiguous() for x in (query, key, value, g, beta))
            out = torch.empty_like(v, memory_format=torch.contiguous_format)
            chunk_args = (out, state, q, k, v, gg, bb, batch_size, end - start, num_heads, v.stride(0),
                          v.stride(1), v.stride(2), v.stride(3), has_initial or start > 0)
            self.lib.delta_rule_prefill(*chunk_args, threads=[128, batch_size * num_heads, 1], group_size=[128, 1, 1])
            output[:, start:end].copy_(out)
        return output, state

    def decode(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, g: torch.Tensor,
               beta: torch.Tensor, initial_state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_heads, _ = query.shape
        output = torch.empty_like(value, memory_format=torch.contiguous_format)
        has_initial = initial_state is not None
        state = initial_state if has_initial else query.new_empty((batch_size, num_heads, 128, 128),
                                                                  dtype=torch.float32)
        args = (output, state, query, key, value, g, beta, batch_size, seq_len, num_heads, value.stride(0),
                value.stride(1), value.stride(2), value.stride(3), has_initial)
        self.lib.delta_rule_decode(*args, threads=[batch_size * num_heads * 512, 1, 1], group_size=[512, 1, 1])
        return output, state


def get_delta_rule_kernels() -> DeltaRuleKernels:
    global _DELTA_RULE_KERNELS
    if _DELTA_RULE_KERNELS is None:
        _DELTA_RULE_KERNELS = DeltaRuleKernels(_compile("delta_rule.metal"))
    return _DELTA_RULE_KERNELS


@dataclass
class AttentionKernels:
    lib: object
    rope: torch.Tensor | None = None

    def prefill(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                cache_keys: torch.Tensor, cache_values: torch.Tensor, context_length: int) -> torch.Tensor:
        output = torch.empty_like(query, memory_format=torch.contiguous_format)
        batch_size, seq_len = query.shape[:2]
        self.lib.attention_prefill(output, query, key, value, cache_keys, cache_values, batch_size, seq_len,
                                   context_length, cache_keys.shape[2], threads=[batch_size * seq_len * 16 * 128, 1, 1],
                                   group_size=[128, 1, 1])
        return output

    def decode(self, x: torch.Tensor, residual: torch.Tensor, memory: Any, q_weight: Any, k_weight: Any,
                     v_weight: Any, o_weight: Any, q_norm: torch.Tensor, k_norm: torch.Tensor) -> torch.Tensor:
        assert x.shape == (1, 1, 4096)
        if memory.buffer is None:
            memory.buffer = x.new_empty((2, 1, 4, memory.max_context, 256))
        if memory.qg is None:
            memory.qg = x.new_empty(8192)
            memory.raw_k = x.new_empty(1024)
            memory.attention = x.new_empty(4096)
            memory.partials = torch.empty((16, 16, 258), dtype=torch.float32, device=x.device)
        cache_k, cache_v = memory.buffer
        quant = get_quant_linear_kernels()
        quant.decode(x, q_weight, y=memory.qg)
        quant.decode(x, k_weight, y=memory.raw_k)
        quant.decode(x, v_weight, y=cache_v, mode=1, context=memory.length, capacity=memory.max_context)
        args = (memory.qg, memory.raw_k, cache_k, cache_v, q_norm, k_norm, self._rope(x.device, memory.max_context),
                memory.length, memory.max_context)
        splits = min(_ATTENTION_SPLITS, memory.length + 1)
        self.lib.attention_decode_scan(memory.partials, *args, splits,
                                       threads=[16 * splits * 128, 1, 1], group_size=[128, 1, 1])
        self.lib.attention_decode_reduce(memory.attention, memory.partials, memory.qg, splits,
                                         threads=[16 * 128, 1, 1], group_size=[128, 1, 1])
        output = torch.empty_like(x)
        quant.decode(memory.attention, o_weight, y=output, mode=2, aux=residual)
        memory.length += 1
        return output

    def _rope(self, device: torch.device, capacity: int) -> torch.Tensor:
        if self.rope is None or self.rope.shape[0] < capacity:
            dims = torch.arange(0, 64, 2, dtype=torch.float32)
            inv_freq = 1.0 / (10_000_000.0 ** (dims / 64.0))
            angles = torch.arange(capacity, dtype=torch.float32)[:, None] * inv_freq[None, :]
            self.rope = torch.stack((angles.cos(), angles.sin()), dim=-1).half().to(device)
        return self.rope


def get_attention_kernels() -> AttentionKernels:
    global _ATTENTION_KERNELS
    if _ATTENTION_KERNELS is None:
        _ATTENTION_KERNELS = AttentionKernels(_compile("attention.metal"))
    return _ATTENTION_KERNELS


@dataclass
class QuantLinearKernels:
    lib: object

    def __call__(self, x: torch.Tensor, weight: Any) -> torch.Tensor:
        return (self.decode if x.numel() // x.shape[-1] == 1 else self.prefill)(x, weight)

    def prefill(self, x: torch.Tensor, weight: Any) -> torch.Tensor:
        k, n = x.shape[-1], weight.shape[0]
        _, prefill_name, _, _ = QUANT_LINEAR_KERNELS[(weight.type_name, k, n)]
        m = x.numel() // k
        x2 = x.reshape(m, k)
        mpad = (m + 31) // 32 * 32
        x2 = x2 if mpad == m else torch.nn.functional.pad(x2, (0, 0, 0, mpad - m))
        y2 = torch.empty((mpad, n), dtype=x.dtype, device=x.device)
        getattr(self.lib, prefill_name)(y2, x2, weight.data, mpad, threads=[128 * (n // 16), mpad // 32, 1],
                                        group_size=[128, 1, 1])
        return y2[:m].reshape(*x.shape[:-1], n)

    def decode(self, x: torch.Tensor, weight: Any, y: torch.Tensor | None = None, *, mode: int = 0,
               aux: torch.Tensor | None = None, context: int = 0, capacity: int = 0) -> torch.Tensor:
        k, n = x.shape[-1], weight.shape[0]
        if y is None:
            y = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
        decode_name, _, decode_tg, decode_rows = QUANT_LINEAR_KERNELS[(weight.type_name, k, n)]
        getattr(self.lib, decode_name)(y, x, weight.data, y if aux is None else aux, mode, context, capacity,
                                       threads=[(n + decode_rows - 1) // decode_rows * decode_tg, 1, 1],
                                       group_size=[decode_tg, 1, 1])
        return y

    def embed_q4(self, ids: torch.Tensor, weight: Any, dtype: torch.dtype) -> torch.Tensor:
        tokens, k = ids.numel(), weight.shape[1]
        y = torch.empty((*ids.shape, k), dtype=dtype, device=ids.device)
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

    def mlp_gate_up(self, x: torch.Tensor, gate_weight: Any, up_weight: Any) -> torch.Tensor:
        seq_len = x.shape[1]
        x2 = x.reshape(seq_len, MLP_HIDDEN_SIZE).contiguous()
        y2 = torch.empty((seq_len, MLP_INTERMEDIATE_SIZE), dtype=x.dtype, device=x.device)
        kernel_name, threadgroup_size, rows = MLP_GATE_UP_KERNELS[(gate_weight.type_name, up_weight.type_name)]
        getattr(self.lib, kernel_name)(y2, x2, gate_weight.data, up_weight.data, seq_len,
                                       threads=[MLP_INTERMEDIATE_SIZE // rows * threadgroup_size, seq_len, 1],
                                       group_size=[threadgroup_size, 1, 1])
        return y2.reshape(*x.shape[:-1], MLP_INTERMEDIATE_SIZE)

    def causal_conv_silu(self, x: torch.Tensor, weight: torch.Tensor,
                         prev_state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        b, seq_len, _ = x.shape
        has_prev = prev_state is not None
        prev = prev_state if has_prev else x.new_empty((b, 8192, 4))
        y, state = torch.empty_like(x), x.new_empty((b, 8192, 4))
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
