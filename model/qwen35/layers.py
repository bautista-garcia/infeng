from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


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


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


@dataclass
class FullAttentionCache:
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    def update(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = torch.cat((self.keys, keys), dim=2)
            self.values = torch.cat((self.values, values), dim=2)
        return self.keys, self.values


@dataclass
class DeltaNetCache:
    conv_state: torch.Tensor | None = None
    recurrent_state: torch.Tensor | None = None


@dataclass
class Cache:
    layers: list[FullAttentionCache | DeltaNetCache]

    @classmethod
    def from_config(cls, config: Any) -> "Cache":
        layer_types = _get(config, "layer_types")
        layers: list[FullAttentionCache | DeltaNetCache] = []
        for layer_type in layer_types:
            if layer_type == "full_attention":
                layers.append(FullAttentionCache())
            elif layer_type == "linear_attention":
                layers.append(DeltaNetCache())
            else:
                raise ValueError(f"unknown Qwen3.5 layer type: {layer_type}")
        return cls(layers)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    # Normalization factor over hidden dimension; gamma = 1.0 + weight
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.to(input_dtype)


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

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

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
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )

        self.rotary_dim = rotary_dim
        self.mrope_section = rope_parameters.get("mrope_section", [11, 11, 10])
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 1:
            position_ids = position_ids[None, :]
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        inv_freq = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        pos = position_ids[:, :, None, :].float()
        freqs = (inv_freq.to(x.device) @ pos.to(x.device)).transpose(2, 3)
        freqs = self._apply_interleaved_mrope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)

    def _apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            freqs_t[..., offset:length:3] = freqs[dim, ..., offset:length:3]
        return freqs_t

    @staticmethod
    def apply(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        rotary_dim = cos.shape[-1]

        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q = torch.cat((q_rot * cos + _rotate_half(q_rot) * sin, q_pass), dim=-1)
        k = torch.cat((k_rot * cos + _rotate_half(k_rot) * sin, k_pass), dim=-1)
        return q, k


class FullAttention(nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = _get(config, "hidden_size")
        self.num_heads = _get(config, "num_attention_heads")
        self.num_key_value_heads = _get(config, "num_key_value_heads")
        self.head_dim = _get(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = _get(config, "attention_dropout", 0.0)
        attention_bias = _get(config, "attention_bias", False)
        eps = _get(config, "rms_norm_eps", 1e-6)

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim * 2, bias=attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=attention_bias
        )
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        cache: FullAttentionCache | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q_and_gate = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim * 2
        )
        query_states, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(
                batch_size, seq_len, self.num_key_value_heads, self.head_dim
            )
        ).transpose(1, 2)
        value_states = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = position_embeddings
        query_states, key_states = RotaryEmbedding.apply(
            query_states, key_states, cos, sin
        )

        if cache is not None:
            key_states, value_states = cache.update(key_states, value_states)

        attn_output = self._attention(
            query_states, key_states, value_states, attention_mask
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                scores = scores.masked_fill(
                    attention_mask[:, None, None, :].to(torch.bool).logical_not(),
                    -torch.inf,
                )
            else:
                scores = scores + attention_mask
        else:
            q_len, k_len = query.shape[-2], key.shape[-2]
            causal_mask = torch.ones(
                q_len, k_len, dtype=torch.bool, device=query.device
            ).tril(diagonal=k_len - q_len)
            scores = scores.masked_fill(~causal_mask, -torch.inf)

        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        probs = F.dropout(probs, p=self.attention_dropout, training=self.training)
        return torch.matmul(probs, value)


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
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads).uniform_(0, 16).log_())
        self.norm = RMSNormGated(self.head_v_dim, eps=eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: DeltaNetCache | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None and attention_mask.ndim == 2:
            hidden_states = hidden_states * attention_mask[:, :, None].to(
                hidden_states.dtype
            )

        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )
        beta = torch.sigmoid(self.in_proj_b(hidden_states))
        g = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias
        )

        mixed_qkv = self._causal_conv(mixed_qkv, cache).transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, (self.key_dim, self.key_dim, self.value_dim), dim=-1
        )
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        if self.num_v_heads != self.num_k_heads:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        initial_state = None if cache is None else cache.recurrent_state
        output, recurrent_state = self._recurrent_delta_rule(
            query, key, value, g, beta, initial_state
        )
        if cache is not None:
            cache.recurrent_state = recurrent_state

        output = output.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        output = self.norm(output, z).reshape(batch_size, seq_len, self.value_dim)
        return self.out_proj(output)

    def _causal_conv(
        self, x: torch.Tensor, cache: DeltaNetCache | None
    ) -> torch.Tensor:
        seq_len = x.shape[-1]
        if cache is None:
            return F.silu(self.conv1d(x)[:, :, :seq_len])

        prev = cache.conv_state
        conv_input = x if prev is None else torch.cat((prev, x), dim=-1)
        state_len = self.conv_kernel_size - 1
        cache.conv_state = conv_input[:, :, -state_len:].detach()

        output = F.silu(self.conv1d(conv_input)[:, :, : conv_input.shape[-1]])
        return output[:, :, -seq_len:]

    def _recurrent_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        initial_dtype = query.dtype
        query = _l2norm(query, dim=-1).transpose(1, 2).float()
        key = _l2norm(key, dim=-1).transpose(1, 2).float()
        value = value.transpose(1, 2).float()
        g = g.transpose(1, 2).float()
        beta = beta.transpose(1, 2).float()

        batch_size, num_heads, seq_len, key_dim = key.shape
        value_dim = value.shape[-1]
        state = (
            torch.zeros(
                batch_size,
                num_heads,
                key_dim,
                value_dim,
                dtype=value.dtype,
                device=value.device,
            )
            if initial_state is None
            else initial_state.to(value)
        )
        output = torch.empty(
            batch_size,
            num_heads,
            seq_len,
            value_dim,
            dtype=value.dtype,
            device=value.device,
        )
        scale = key_dim**-0.5

        for token_idx in range(seq_len):
            q_t = query[:, :, token_idx] * scale
            k_t = key[:, :, token_idx]
            v_t = value[:, :, token_idx]
            g_t = g[:, :, token_idx].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, token_idx].unsqueeze(-1)

            state = state * g_t
            prediction = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - prediction) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            output[:, :, token_idx] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

        return output.transpose(1, 2).to(initial_dtype), state


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        cache: FullAttentionCache | DeltaNetCache | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states, attention_mask=attention_mask, cache=cache
            )
        else:
            hidden_states = self.self_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                cache=cache,
            )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
