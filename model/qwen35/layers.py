from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from backend.metal import get_delta_rule_kernels, get_fused_layer_kernels, get_quant_linear_kernels


def _get(config: Any, name: str, default: Any = None) -> Any:
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x

    batch, kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, kv_heads * n_rep, seq_len, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = None
        self.bias = None
        self.kernels = get_quant_linear_kernels()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        y = F.linear(x, w.data) if w.torch_dtype else self.kernels.linear(x.contiguous(), w)
        return y if self.bias is None else y + self.bias.data


LMHead = Linear


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.weight = None
        self.dtype = dtype
        self.kernels = get_quant_linear_kernels()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        w = self.weight
        return F.embedding(ids, w.data) if w.torch_dtype else self.kernels.embed_q4(ids.contiguous(), w, self.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) *
                (1.0 if self.weight is None else self.weight.data)).type(dtype)


class RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = None


class MLP(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        hidden_size = _get(config, "hidden_size")
        intermediate_size = _get(config, "intermediate_size")

        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.fused_layers = get_fused_layer_kernels()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1: # Decode
            # Fused (silu(gate) * up)
            fused = self.fused_layers.mlp_gate_up(x, self.gate_proj.weight, self.up_proj.weight)
            return self.down_proj(fused)
        gate, up = (proj(x) for proj in (self.gate_proj, self.up_proj))
        return self.down_proj(F.silu(gate) * up)


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
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        pos = position_ids[:, :, None, :].float()
        freqs = (inv_freq @ pos).transpose(2, 3)
        freqs = self._apply_interleaved_mrope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().type(x.dtype).unsqueeze(1)
        sin = emb.sin().type(x.dtype).unsqueeze(1)
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

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, memory: Any,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q_and_gate = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim * 2)
        query_states, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads,
                                                                 self.head_dim)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads,
                                                       self.head_dim).transpose(1, 2)
        query_states, key_states = self.rotary_emb(query_states, key_states, hidden_states, position_ids)

        if memory is not None:
            key_states, value_states = memory.update(key_states, value_states)

        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        q_len, k_len = query_states.shape[-2], key_states.shape[-2]
        attn_mask, is_causal = None, False
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attn_mask = attention_mask[:, None, None, :].bool()
            else:
                attn_mask = attention_mask
        elif q_len == k_len:
            is_causal = True
        elif q_len == 1:
            # A single decode query is the newest cache entry and can attend
            # to every valid key, so no mask is needed.
            pass
        else:
            attn_mask = torch.ones(q_len, k_len, dtype=torch.bool, device=query_states.device).tril(
                diagonal=k_len - q_len)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attn_mask,
                                                     dropout_p=0.0, is_causal=is_causal)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
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
        eps = _get(config, "rms_norm_eps", 1e-6)

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.conv1d_weight = None
        self.dt_bias = None
        self.A_log = None
        self.norm = RMSNormGated(self.head_v_dim, eps=eps)
        self.out_proj = Linear(self.value_dim, self.hidden_size, bias=False)
        self.delta_rule_metal = get_delta_rule_kernels()
        self.fused_layers = get_fused_layer_kernels()

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, memory: Any,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is not None and attention_mask.ndim == 2:
            hidden_states = hidden_states * attention_mask[:, :, None].type(hidden_states.dtype)

        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv, z, b, a = (proj(hidden_states) for proj in (
            self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a))
        z = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        beta = torch.sigmoid(b)
        g = -self.A_log.data.float().exp() * F.softplus(a.float() + self.dt_bias.data)
        return self._delta(mixed_qkv, z, beta, g, batch_size, seq_len, memory)

    def _delta(self, mixed_qkv: torch.Tensor, z: torch.Tensor, beta: torch.Tensor, g: torch.Tensor,
               batch_size: int, seq_len: int, memory: Any) -> torch.Tensor:
        mixed_qkv, conv_state = self.fused_layers.causal_conv_silu(mixed_qkv, self.conv1d_weight.data,
                                                                    None if memory is None else memory.conv_state)
        if memory is not None:
            memory.conv_state = conv_state

        query, key, value = torch.split(mixed_qkv, (self.key_dim, self.key_dim, self.value_dim), dim=-1)
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        if self.num_v_heads != self.num_k_heads:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat(1, 1, repeat, 1)
            key = key.repeat(1, 1, repeat, 1)

        initial_state = None if memory is None else memory.recurrent_state
        output, recurrent_state = self.delta_rule_metal(query, key, value, g, beta, initial_state)
        if memory is not None:
            memory.recurrent_state = recurrent_state

        output = output.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        output = self.fused_layers.rmsnorm_gated(output, z, self.norm.weight.data, self.norm.eps).reshape(
            batch_size, seq_len, self.value_dim)
        return self.out_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self._attn_name, attn_cls = {"linear_attention": ("linear_attn", GatedDeltaNet),
                                     "full_attention": ("self_attn", FullAttention)}[
            _get(config, "layer_types")[layer_idx]]
        self.linear_attn = self.self_attn = None
        setattr(self, self._attn_name, attn_cls(config, layer_idx))

        hidden_size = _get(config, "hidden_size")
        eps = _get(config, "rms_norm_eps", 1e-6)
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.mlp = MLP(config)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, memory: Any,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn = getattr(self, self._attn_name)
        hidden_states = attn(hidden_states, position_ids, memory, attention_mask=attention_mask)

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
