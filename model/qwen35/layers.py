from __future__ import annotations

from backend.metal.ops import Ops


class DecoderLayer:
    def __init__(self, config, index, weights, ops: Ops):
        self.index, self.kind, self.ops = index, config["layer_types"][index], ops
        p = f"model.layers.{index}."
        self.input_norm = weights[p + "input_layernorm.weight"]
        self.post_norm = weights[p + "post_attention_layernorm.weight"]
        self.mlp = {name: weights[p + f"mlp.{name}_proj.weight"] for name in ("gate", "up", "down")}
        if self.kind == "full_attention":
            a = p + "self_attn."
            self.attn = {name: weights[a + key] for name, key in {
                "q": "q_proj.weight", "k": "k_proj.weight", "v": "v_proj.weight", "o": "o_proj.weight",
                "qn": "q_norm.weight", "kn": "k_norm.weight"}.items()}
        else:
            a = p + "linear_attn."
            self.attn = {name: weights[a + key] for name, key in {
                "qkv": "in_proj_qkv.weight", "z": "in_proj_z.weight", "b": "in_proj_b.weight",
                "a": "in_proj_a.weight", "conv": "conv1d_weight", "out": "out_proj.weight",
                "norm": "norm.weight", "dt": "dt_bias", "A": "A_log"}.items()}

    def __call__(self, hidden, memory, start, rope):
        residual = hidden
        x = self.ops.rms(hidden, self.input_norm, 1e-6, "layer.input_norm")
        if self.kind == "full_attention": hidden = self.ops.full_attention(x, residual, self.attn, memory, start, rope)
        else: hidden = self.ops.add(residual, self.ops.gdn(x, self.attn, memory), "hidden.mid")
        x = self.ops.rms(hidden, self.post_norm, 1e-6, "layer.post_norm")
        return self.ops.add(hidden, self.ops.mlp(x, self.mlp["gate"], self.mlp["up"], self.mlp["down"]),
                            f"hidden{(self.index + 1) & 1}")
