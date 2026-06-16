from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from backend.metal import get_delta_rule_kernels


def _get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x

    batch, kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, kv_heads * n_rep, seq_len, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@dataclass
class FullAttentionCache:
    max_len: int | None = None
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
    length: torch.Tensor | None = None

    def allocate(self, batch_size: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device):
        if self.max_len and self.keys is None:
            shape = (batch_size, num_heads, self.max_len, head_dim)
            self.keys = torch.empty(shape, dtype=dtype, device=device)
            self.values = torch.empty(shape, dtype=dtype, device=device)
            self.length = torch.zeros((), dtype=torch.long, device=device)

    def update(self, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.max_len:
            self.allocate(keys.shape[0], keys.shape[1], keys.shape[3], keys.dtype, keys.device)
            idx = self.length + torch.arange(keys.shape[2], device=keys.device)
            self.keys.index_copy_(2, idx, keys)
            self.values.index_copy_(2, idx, values)
            self.length.add_(keys.shape[2])
            return self.keys, self.values, torch.arange(self.max_len, device=keys.device) < self.length
        elif self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # (B, H, L, d) so we append on the token dimension
            self.keys = torch.cat((self.keys, keys), dim=2)
            self.values = torch.cat((self.values, values), dim=2)
        return self.keys, self.values, None


@dataclass
class DeltaNetCache:
    conv_state: torch.Tensor | None = None
    recurrent_state: torch.Tensor | None = None


@dataclass
class Cache:
    layers: list[FullAttentionCache | DeltaNetCache]

    def allocate(self, config: Any, batch_size: int, dtype: torch.dtype, device: torch.device):
        for layer in self.layers:
            if isinstance(layer, FullAttentionCache):
                layer.allocate(batch_size, _get(config, "num_key_value_heads"), _get(config, "head_dim"), dtype, device)

    @classmethod
    def from_config(cls, config: Any) -> "Cache":
        layer_types = _get(config, "layer_types")
        max_len = int(v) if (v := os.getenv("INFENG_STATIC_KV_LEN")) else None
        layers: list[FullAttentionCache | DeltaNetCache] = []
        for layer_type in layer_types:
            if layer_type == "full_attention":
                layers.append(FullAttentionCache(max_len))
            elif layer_type == "linear_attention":
                layers.append(DeltaNetCache())
            else:
                raise ValueError(f"unknown Qwen3.5 layer type: {layer_type}")
        return cls(layers)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    # Normalization factor over hidden dimension; gamma = 1.0 + weight
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * (1.0 + self.weight.float())
        return x.to(input_dtype)


# Gate is not related to RMSNorm, their joint appearance is for fusing both ops
class RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = self.weight.float() * x
        x = x * F.silu(
            gate.float()
        )  # gate modulates: SiLU(z) ≈ 0 → suppress, SiLU(z) > 1 → amplify
        return x.to(input_dtype)


class MLP(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        hidden_size = _get(config, "hidden_size")
        intermediate_size = _get(config, "intermediate_size")

        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        head_dim = _get(config, "head_dim")
        rope_parameters = _get(config, "rope_parameters", {})
        theta = rope_parameters.get("rope_theta", 10000.0)
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        rotary_dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))

        self.rotary_dim = rotary_dim
        self.mrope_section = rope_parameters.get("mrope_section", [11, 11, 10])
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, x: torch.Tensor,
                position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 1:
            position_ids = position_ids[None, :]
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        inv_freq = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        pos = position_ids[:, :, None, :].float()
        freqs = (inv_freq.to(x.device) @ pos.to(x.device)).transpose(2, 3)
        freqs = self._apply_interleaved_mrope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype).unsqueeze(1)
        sin = emb.sin().to(dtype=x.dtype).unsqueeze(1)

        q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
        k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
        q = torch.cat((q_rot * cos + _rotate_half(q_rot) * sin, q_pass), dim=-1)
        k = torch.cat((k_rot * cos + _rotate_half(k_rot) * sin, k_pass), dim=-1)
        return q, k

    def _apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            freqs_t[..., offset:length:3] = freqs[dim, ..., offset:length:3]
        return freqs_t


class FullAttention(nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = _get(config, "hidden_size")
        self.num_heads = _get(config, "num_attention_heads")
        self.num_key_value_heads = _get(config, "num_key_value_heads")
        self.head_dim = _get(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        attention_bias = _get(config, "attention_bias", False)
        eps = _get(config, "rms_norm_eps", 1e-6)

        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim * 2, bias=attention_bias)
        self.k_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias=attention_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None, cache: FullAttentionCache | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        # Q = x @ Wq, gate = x @ W_gate; It does a fused matmul (due to both sharing x)
        q_and_gate = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim * 2)
        # (B, L, H, 2D) into two (B, L, H, D): H is num_heads
        query_states, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        # RMSNorm(Q)
        query_states = self.q_norm(query_states).transpose(1, 2)

        # K = RMSNorm(x @ Wk)
        key_states = self.k_norm(self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads,
                                                                 self.head_dim)).transpose(1, 2)

        # Transposing goes to a (B, H, L, D) so that B and H are fully independent (parallel)

        # V = x @ Wv (Value does not go through RoPE)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads,
                                                       self.head_dim).transpose(1, 2)

        # RoPE(Q, K)
        query_states, key_states = self.rotary_emb(query_states, key_states, hidden_states, position_ids)

        key_mask = None
        if cache is not None:
            key_states, value_states, key_mask = cache.update(key_states, value_states)

        # GQA: K & V from (B, H=4, L, D) into (B, H=16, L, D) by repeating each K, V head 4 times
        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        q_len, k_len = query_states.shape[-2], key_states.shape[-2]
        attn_mask, is_causal = None, False
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attn_mask = attention_mask[:, None, None, :].to(torch.bool)
            else:
                attn_mask = attention_mask
        elif key_mask is not None:
            pos_ids = position_ids[1:] if position_ids.ndim == 3 and position_ids.shape[0] == 4 else position_ids
            pos_ids = pos_ids[0] if pos_ids.ndim == 3 else pos_ids
            attn_mask = (torch.arange(k_len, device=key_states.device)[None, None, None, :] <= pos_ids[:, None, :, None]
                         ) & key_mask[None, None, None, :]
        elif q_len == k_len:
            is_causal = True
        else:
            attn_mask = torch.ones(q_len, k_len, dtype=torch.bool, device=query_states.device).tril(diagonal=k_len - q_len)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attn_mask,
                                                     dropout_p=0.0, is_causal=is_causal)
        # Back to (B, L, D)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        # Gating/modulation with the gate vector
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


class GatedDeltaNet(nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = _get(config, "hidden_size")
        self.num_k_heads = _get(config, "linear_num_key_heads")
        self.num_v_heads = _get(config, "linear_num_value_heads")
        self.head_k_dim = _get(config, "linear_key_head_dim")
        self.head_v_dim = _get(config, "linear_value_head_dim")
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = _get(config, "linear_conv_kernel_dim")
        eps = _get(config, "rms_norm_eps", 1e-6)

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=self.conv_kernel_size,
                                groups=self.conv_dim, padding=self.conv_kernel_size - 1, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads).uniform_(0, 16).log_())
        self.norm = RMSNormGated(self.head_v_dim, eps=eps)
        self.out_proj = Linear(self.value_dim, self.hidden_size, bias=False)
        self.delta_rule_metal = get_delta_rule_kernels()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
                cache: DeltaNetCache | None = None) -> torch.Tensor:
        if attention_mask is not None and attention_mask.ndim == 2:
            hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)

        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        # Local Learning rate (0,1)
        beta = torch.sigmoid(self.in_proj_b(hidden_states))
        # Global decay: gt = exp(g), so a highly negative value reduces the impact of St-1 (short term memory)
        #, a lower negative value amplifies the impact of St-1 (long term memory)
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)

        mixed_qkv = self._causal_conv(mixed_qkv, cache).transpose(1, 2)

        # (B, L, dk + dq + dv) into (B, L, dk), (B, L, dk), (B, L, dv)
        query, key, value = torch.split(mixed_qkv, (self.key_dim, self.key_dim, self.value_dim), dim=-1)
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        # Grouped Query Attention (GQA) 
        if self.num_v_heads != self.num_k_heads:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        initial_state = None if cache is None else cache.recurrent_state
        output, recurrent_state = self._recurrent_delta_rule(query, key, value, g, beta, initial_state)
        if cache is not None:
            cache.recurrent_state = recurrent_state

        output = output.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        output = self.norm(output, z).reshape(batch_size, seq_len, self.value_dim)
        return self.out_proj(output)

    def _causal_conv(self, x: torch.Tensor, cache: DeltaNetCache | None) -> torch.Tensor:
        seq_len = x.shape[-1]
        if cache is None:
            return F.silu(self.conv1d(x)[:, :, :seq_len])

        prev = cache.conv_state
        conv_input = x if prev is None else torch.cat((prev, x), dim=-1)
        cache.conv_state = F.pad(conv_input, (self.conv_kernel_size - conv_input.shape[-1], 0))[
            :, :, -self.conv_kernel_size :
        ].detach()

        output = F.silu(F.conv1d(conv_input, self.conv1d.weight, groups=self.conv_dim)
                        if prev is not None else self.conv1d(conv_input)[:, :, : conv_input.shape[-1]])
        return output[:, :, -seq_len:]

    def _recurrent_delta_rule(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                              g: torch.Tensor, beta: torch.Tensor,
                              initial_state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.delta_rule_metal(query, key, value, g, beta, initial_state)


class DecoderLayer(nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = _get(config, "layer_types")[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, layer_idx)
            self.self_attn = None
        elif self.layer_type == "full_attention":
            self.self_attn = FullAttention(config, layer_idx)
            self.linear_attn = None
        else:
            raise ValueError(f"unknown Qwen3.5 layer type: {self.layer_type}")

        hidden_size = _get(config, "hidden_size")
        eps = _get(config, "rms_norm_eps", 1e-6)
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.mlp = MLP(config)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                cache: FullAttentionCache | DeltaNetCache | None = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, attention_mask=attention_mask, cache=cache)
        else:
            hidden_states = self.self_attn(hidden_states, position_ids=position_ids, attention_mask=attention_mask,
                                           cache=cache)

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
