#include <metal_stdlib>
using namespace metal;

constant uint GDN_C = 8192;
constant uint GDN_D = 128;
constant uint MLP_K = 4096;
constant uint MLP_N = 12288;
constant uint MLP_NSIMD = 16;

constant float iq4nl[16] = {-127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
                              1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f};

static inline half h16(device const uchar* p) {
    ushort v = ushort(p[0]) | (ushort(p[1]) << 8);
    return as_type<half>(v);
}

static inline void scale_min_k4(uint j, device const uchar* q, thread uchar& d, thread uchar& m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 15) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static inline void q4k_decode_block(thread float& acc, device const half* x, device const uchar* w,
                                    uint k0, uint kb, uint row, uint simd_lane, uint K) {
    long o = long(row) * (K / 256) * 144 + kb * 144;
    float d = float(h16(w + o)), dm = float(h16(w + o + 2));
    uchar sc, mn, q = w[o + 16 + simd_lane];
    scale_min_k4(0, w + o + 4, sc, mn); acc = fma(float(x[k0 + simd_lane]), d * float(sc) * float(q & 15) - dm * float(mn), acc);
    scale_min_k4(1, w + o + 4, sc, mn); acc = fma(float(x[k0 + simd_lane + 32]), d * float(sc) * float(q >> 4) - dm * float(mn), acc);
    q = w[o + 48 + simd_lane];
    scale_min_k4(2, w + o + 4, sc, mn); acc = fma(float(x[k0 + simd_lane + 64]), d * float(sc) * float(q & 15) - dm * float(mn), acc);
    scale_min_k4(3, w + o + 4, sc, mn); acc = fma(float(x[k0 + simd_lane + 96]), d * float(sc) * float(q >> 4) - dm * float(mn), acc);
    q = w[o + 80 + simd_lane];
    scale_min_k4(4, w + o + 4, sc, mn); acc = fma(float(x[k0 + simd_lane + 128]), d * float(sc) * float(q & 15) - dm * float(mn), acc);
    scale_min_k4(5, w + o + 4, sc, mn); acc = fma(float(x[k0 + simd_lane + 160]), d * float(sc) * float(q >> 4) - dm * float(mn), acc);
    q = w[o + 112 + simd_lane];
    scale_min_k4(6, w + o + 4, sc, mn); acc = fma(float(x[k0 + simd_lane + 192]), d * float(sc) * float(q & 15) - dm * float(mn), acc);
    scale_min_k4(7, w + o + 4, sc, mn); acc = fma(float(x[k0 + simd_lane + 224]), d * float(sc) * float(q >> 4) - dm * float(mn), acc);
}

static inline void q5k_decode_block(thread float& acc, device const half* x, device const uchar* w,
                                    uint k0, uint kb, uint row, uint simd_lane, uint K) {
    long o = long(row) * (K / 256) * 176 + kb * 176;
    float d = float(h16(w + o)), dm = float(h16(w + o + 2));
    uchar hm = w[o + 16 + simd_lane];
    for (uint t = 0; t < 4; ++t) {
        uchar sc, mn, q = w[o + 48 + t * 32 + simd_lane];
        uint r = t * 64 + simd_lane, j = t * 2;
        scale_min_k4(j, w + o + 4, sc, mn);
        acc = fma(float(x[k0 + r]), d * float(sc) * float((q & 15) + ((hm & (1 << j)) ? 16 : 0)) - dm * float(mn), acc);
        scale_min_k4(j + 1, w + o + 4, sc, mn);
        acc = fma(float(x[k0 + r + 32]), d * float(sc) * float((q >> 4) + ((hm & (1 << (j + 1))) ? 16 : 0)) - dm * float(mn), acc);
    }
}

static inline void iq4xs_decode_block(thread float& acc, device const half* x, device const uchar* w,
                                      uint k0, uint kb, uint row, uint simd_lane, uint K) {
    long o = long(row) * (K / 256) * 136 + kb * 136;
    float d = float(h16(w + o));
    ushort sh = ushort(w[o + 2]) | (ushort(w[o + 3]) << 8);
    for (uint j = 0; j < 8; ++j) {
        uint r = j * 32 + simd_lane, qj = j * 16 + (simd_lane & 15);
        int ls = int((w[o + 4 + (j >> 1)] >> (4 * (j & 1))) & 15) | int(((sh >> (2 * j)) & 3) << 4);
        uchar q = w[o + 8 + qj], v = (simd_lane & 16) ? (q >> 4) : (q & 15);
        acc = fma(float(x[k0 + r]), d * float(ls - 32) * iq4nl[v], acc);
    }
}

static inline half gdn_load_context(device const half* x, device const half* prev, long b, long c, long L,
                                    long pos, bool has_prev) {
    if (has_prev) return pos < 4 ? prev[(b * GDN_C + c) * 4 + pos] : x[(b * L + pos - 4) * GDN_C + c];
    return pos < 0 || pos >= L ? half(0.0) : x[(b * L + pos) * GDN_C + c];
}

#define MLP_GATE_UP_DECODE(name, decode_block) \
[[max_total_threads_per_threadgroup(512)]] \
kernel void name(device half* y [[buffer(0)]], device const half* x [[buffer(1)]], \
                 device const uchar* wg [[buffer(2)]], device const uchar* wu [[buffer(3)]], \
                 constant long& S [[buffer(4)]], uint simd_lane [[thread_index_in_simdgroup]], \
                 uint simd_group [[simdgroup_index_in_threadgroup]], \
                 uint3 group [[threadgroup_position_in_grid]]) { \
    uint row = group.x * MLP_NSIMD + simd_group; \
    long s = long(group.y); \
    float gate = 0.0f, up = 0.0f; \
    device const half* xr = x + s * MLP_K; \
    for (uint kb = 0; kb < MLP_K / 256; ++kb) { \
        uint k0 = kb * 256; \
        decode_block(gate, xr, wg, k0, kb, row, simd_lane, MLP_K); \
        decode_block(up, xr, wu, k0, kb, row, simd_lane, MLP_K); \
    } \
    gate = simd_sum(gate); up = simd_sum(up); \
    if (s < S && row < MLP_N && simd_lane == 0) { \
        float gh = float(half(gate)), uh = float(half(up)); \
        y[s * MLP_N + row] = half((gh / (1.0f + exp(-gh))) * uh); \
    } \
}

MLP_GATE_UP_DECODE(mlp_gate_up_q4_k_decode, q4k_decode_block)
MLP_GATE_UP_DECODE(mlp_gate_up_q5_k_decode, q5k_decode_block)
MLP_GATE_UP_DECODE(mlp_gate_up_iq4_xs_decode, iq4xs_decode_block)

[[max_total_threads_per_threadgroup(256)]]
kernel void gdn_causal_conv_silu(device half* y [[buffer(0)]], device half* state [[buffer(1)]],
                                 device const half* x [[buffer(2)]], device const half* w [[buffer(3)]],
                                 device const half* prev [[buffer(4)]], constant long& B [[buffer(5)]],
                                 constant long& L [[buffer(6)]], constant bool& has_prev [[buffer(7)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    long span = L > 4 ? L : 4, idx = gid.x, b = idx / (GDN_C * span), rem = idx - b * GDN_C * span;
    if (b >= B) return;
    long p = rem / GDN_C, c = rem - p * GDN_C;
    if (p < L) {
        float acc = 0.0f;
        for (long r = 0; r < 4; ++r) acc = fma(float(gdn_load_context(x, prev, b, c, L, has_prev ? p + 1 + r : p + r - 3, has_prev)),
                                               float(w[c * 4 + r]), acc);
        float hv = float(half(acc));
        y[(b * L + p) * GDN_C + c] = half(hv / (1.0f + exp(-hv)));
    }
    if (p < 4) {
        long pos = (has_prev ? L + p : L - 4 + p);
        state[(b * GDN_C + c) * 4 + p] = gdn_load_context(x, prev, b, c, L, pos, has_prev);
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void rmsnorm_gated_128(device half* y [[buffer(0)]], device const half* x [[buffer(1)]],
                              device const half* gate [[buffer(2)]], device const float* w [[buffer(3)]],
                              constant float& eps [[buffer(4)]], uint3 lane3 [[thread_position_in_threadgroup]],
                              uint simd_lane [[thread_index_in_simdgroup]],
                              uint simd_group [[simdgroup_index_in_threadgroup]],
                              uint3 group [[threadgroup_position_in_grid]]) {
    uint d = lane3.x, row = group.y;
    threadgroup float scratch[5];
    float xv = float(x[row * GDN_D + d]), ss = simd_sum(xv * xv);
    if (simd_lane == 0) scratch[simd_group] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float s = simd_lane < 4 ? scratch[simd_lane] : 0.0f;
        s = simd_sum(s);
        if (simd_lane == 0) scratch[4] = rsqrt(s / float(GDN_D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gv = float(gate[row * GDN_D + d]);
    y[row * GDN_D + d] = half(xv * scratch[4] * w[d] * (gv / (1.0f + exp(-gv))));
}
