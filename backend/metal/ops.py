from __future__ import annotations

from .runtime import Runtime, Tensor

QUANT = {
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
MLP = {("Q4_K", "Q4_K"): ("mlp_gate_up_q4_k_decode", 64, 4),
       ("Q5_K", "Q5_K"): ("mlp_gate_up_q5_k_decode", 64, 2),
       ("IQ4_XS", "IQ4_XS"): ("mlp_gate_up_iq4_xs_decode", 64, 4)}


class Ops:
    def __init__(self, runtime: Runtime): self.runtime, self.cache, self.p = runtime, {}, None

    def begin(self): self.p = self.runtime.begin()
    def end(self, wait=True): self.p.commit(wait); self.p = None

    def empty(self, key, shape, dtype="f16", shared=False):
        shape = tuple(shape); item = self.cache.get((key, dtype))
        if item is None or item.numel < self._numel(shape):
            item = self.runtime.empty(shape, dtype, shared); self.cache[(key, dtype)] = item
        return item.view(shape)

    @staticmethod
    def _numel(shape):
        n = 1
        for d in shape: n *= d
        return n

    def linear(self, x, w, key, *, mode=0, aux=None, context=0, output=None):
        rows, k, n = x.numel // x.shape[-1], x.shape[-1], w.shape[0]
        y = output or self.empty(key, (*x.shape[:-1], n)); typ = w.type_name
        if w.native_dtype:
            if (k, n, typ) != (4096, 32, "F16"): raise ValueError(f"unsupported dense linear {typ} {k}x{n}")
            self.p.dispatch("dense_4096x32", [y, x, w.data], [("I", rows)], (1024, rows, 1), (32, 1, 1)); return y
        decode, prefill, tg, per = QUANT[(typ, k, n)]
        if rows == 1:
            self.p.dispatch(decode, [y, x, w.data, aux or y], [("I", mode), ("I", context), ("I", 0)],
                            (((n + per - 1) // per) * tg, 1, 1), (tg, 1, 1)); return y
        if mode: raise ValueError("quant prefill has no fused store mode")
        mpad = (rows + 31) // 32 * 32
        source, target = x, y
        if mpad != rows:
            source = self.empty(key + ".pad_in", (mpad, k))
            target = self.empty(key + ".pad_out", (mpad, n))
            total = mpad * k
            self.p.dispatch("pad_rows", [source, x], [("I", rows), ("I", mpad), ("I", k)],
                            ((total + 255) // 256 * 256, 1, 1), (256, 1, 1))
        self.p.dispatch(prefill, [target, source, w.data], [("q", mpad)],
                        (128 * (n // 16), mpad // 32, 1), (128, 1, 1))
        return target.view((*x.shape[:-1], n))

    def embed(self, ids, w):
        tokens, dim = ids.numel, w.shape[1]; y = self.empty("hidden0", (tokens, dim))
        self.p.dispatch("q4_k_embed", [y, ids, w.data], [("q", tokens), ("q", dim)],
                        (((dim + 255) // 256) * 256, tokens, 1), (256, 1, 1)); return y

    def rms(self, x, w, eps, key, dim=None):
        dim = dim or x.shape[-1]; rows = x.numel // dim; y = self.empty(key, x.shape)
        self.p.dispatch("rmsnorm", [y, x, w.data], [("I", rows), ("I", dim), ("f", eps)],
                        (256, rows, 1), (256, 1, 1)); return y

    def add(self, a, b, key):
        y = self.empty(key, a.shape); n = a.numel
        self.p.dispatch("add_half", [y, a, b], [("I", n)], ((n + 255) // 256 * 256, 1, 1), (256, 1, 1)); return y

    def mlp(self, x, gate, up, down):
        rows = x.numel // 4096
        if rows == 1:
            name, tg, per = MLP[(gate.type_name, up.type_name)]; mixed = self.empty("mlp.mixed", (1, 12288))
            self.p.dispatch(name, [mixed, x, gate.data, up.data], [("q", 1)],
                            (12288 // per * tg, 1, 1), (tg, 1, 1))
        else:
            g = self.linear(x, gate, "mlp.gate"); u = self.linear(x, up, "mlp.up")
            mixed = self.empty("mlp.mixed", g.shape); n = g.numel
            self.p.dispatch("silu_mul", [mixed, g, u], [("I", n)], ((n + 255) // 256 * 256, 1, 1), (256, 1, 1))
        return self.linear(mixed, down, "mlp.down")

    def full_attention(self, x, residual, weights, cache, start, rope):
        seq = x.shape[0]
        if seq == 1:
            qg = self.linear(x, weights["q"], "attn.qg")
            raw_k = self.linear(x, weights["k"], "attn.k")
            self.linear(x, weights["v"], "attn.v", mode=1, context=start, output=cache.v)
            partials = self.empty("attn.partials", (16, 16, 258), "f32")
            attention = self.empty("attn.output", (1, 4096)); splits = min(8, start + 1)
            self.p.dispatch("attention_decode_scan", [partials, qg, raw_k, cache.k, cache.v,
                            weights["qn"].data, weights["kn"].data, rope],
                            [("I", start), ("I", cache.max_context), ("I", splits)],
                            (16 * splits * 128, 1, 1), (128, 1, 1))
            self.p.dispatch("attention_decode_reduce", [attention, partials, qg], [("I", splits)],
                            (16 * 128, 1, 1), (128, 1, 1))
            return self.linear(attention, weights["o"], "hidden.mid", mode=2, aux=residual)
        qg = self.linear(x, weights["q"], "attn.qg")
        q = self.empty("attn.q", (seq, 16, 256)); gate = self.empty("attn.gate", q.shape); n = q.numel
        self.p.dispatch("unpack_attention", [q, gate, qg], [("I", n)],
                        ((n + 255) // 256 * 256, 1, 1), (256, 1, 1))
        q = self.rms(q, weights["qn"], 1e-6, "attn.qnorm", 256)
        k = self.linear(x, weights["k"], "attn.k").view((seq, 4, 256))
        k = self.rms(k, weights["kn"], 1e-6, "attn.knorm", 256)
        v = self.linear(x, weights["v"], "attn.v").view((seq, 4, 256))
        qr, kr = self.empty("attn.qrope", q.shape), self.empty("attn.krope", k.shape)
        total = max(q.numel, k.numel)
        self.p.dispatch("rope_qk", [qr, kr, q, k, rope], [("I", seq), ("I", start)],
                        ((total + 255) // 256 * 256, 1, 1), (256, 1, 1))
        out = self.empty("attn.output", q.shape)
        self.p.dispatch("attention_prefill", [out, qr, kr, v, cache.k, cache.v],
                        [("q", 1), ("q", seq), ("q", start), ("q", cache.max_context)],
                        (seq * 16 * 128, 1, 1), (128, 1, 1))
        gated = self.empty("attn.gated", (seq, 4096))
        self.p.dispatch("attention_gate", [gated, out, qg], [("I", seq * 4096)],
                        (((seq * 4096 + 255) // 256) * 256, 1, 1), (256, 1, 1))
        return self.add(residual, self.linear(gated, weights["o"], "attn.projected"), "hidden.mid")

    def gdn(self, x, weights, state):
        seq = x.shape[0]; mixed = self.linear(x, weights["qkv"], "gdn.mixed")
        z = self.linear(x, weights["z"], "gdn.z")
        b = self.linear(x, weights["b"], "gdn.b"); a = self.linear(x, weights["a"], "gdn.a")
        beta = self.empty("gdn.beta", (seq, 32)); g = self.empty("gdn.g", (seq, 32), "f32")
        self.p.dispatch("gdn_prepare", [beta, g, b, a, weights["A"].data, weights["dt"].data],
                        [("I", seq * 32)], (((seq * 32 + 255) // 256) * 256, 1, 1), (256, 1, 1))
        slot = 1 - state.conv_slot; conv = self.empty(f"gdn.{state.layer}.conv{slot}", (8192, 4))
        prev = state.conv or conv; convolved = self.empty("gdn.convolved", mixed.shape)
        self.p.dispatch("gdn_causal_conv_silu", [convolved, conv, mixed, weights["conv"].data, prev],
                        [("q", 1), ("q", seq), ("?", state.conv is not None)],
                        (8192 * max(seq, 4), 1, 1), (256, 1, 1))
        state.conv, state.conv_slot = conv, slot
        q, k, v = (self.empty("gdn.q", (seq, 32, 128)), self.empty("gdn.k", (seq, 32, 128)),
                   self.empty("gdn.v", (seq, 32, 128)))
        total = seq * 4096
        self.p.dispatch("split_repeat_qk", [q, k, v, convolved], [("I", seq)],
                        ((total + 255) // 256 * 256, 1, 1), (256, 1, 1))
        if state.recurrent is None: state.recurrent = self.empty(f"gdn.{state.layer}.recurrent", (32, 128, 128), "f32")
        out = self.empty("gdn.delta", v.shape)
        if seq == 1:
            self.p.dispatch("delta_rule_decode", [out, state.recurrent, q, k, v, g, beta],
                            [("q", 1), ("q", 1), ("q", 32), ("q", 4096), ("q", 4096), ("q", 128),
                             ("q", 1), ("?", state.initialized)], (32 * 512, 1, 1), (512, 1, 1))
        else:
            for start in range(0, seq, 16):
                count = min(16, seq - start); offset = start * 4096
                tensors = [out.view((count, 32, 128), offset=offset), state.recurrent,
                           q.view((count, 32, 128), offset=offset), k.view((count, 32, 128), offset=offset),
                           v.view((count, 32, 128), offset=offset), g.view((count, 32), offset=start * 32),
                           beta.view((count, 32), offset=start * 32)]
                self.p.dispatch("delta_rule_prefill", tensors,
                                [("q", 1), ("q", count), ("q", 32), ("q", count * 4096),
                                 ("q", 4096), ("q", 128), ("q", 1), ("?", state.initialized or start > 0)],
                                (128, 32, 1), (128, 1, 1))
        state.initialized = True
        normed = self.empty("gdn.normed", v.shape)
        self.p.dispatch("rmsnorm_gated_128", [normed, out, z, weights["norm"].data], [("f", 1e-6)],
                        (128, seq * 32, 1), (128, 1, 1))
        return self.linear(normed.view((seq, 4096)), weights["out"], "gdn.projected")

    def sample(self, logits, token, rng, temperature=0.0, top_p=1.0, top_k=None):
        if temperature <= 0:
            self.p.dispatch("argmax_logits", [token, logits], [("I", logits.numel)], (256, 1, 1), (256, 1, 1)); return
        if top_k is not None and not 0 < top_k <= 64: raise ValueError("GPU top_k must be between 1 and 64")
        if top_k is None and top_p < 1: raise ValueError("top_p below 1 requires top_k on this specialized runtime")
        self.p.dispatch("sample_logits", [token, rng, logits],
                        [("I", logits.numel), ("f", temperature), ("f", top_p), ("I", top_k or 0)],
                        (1, 1, 1), (1, 1, 1))
