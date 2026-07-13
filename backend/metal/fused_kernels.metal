#include <metal_stdlib>
using namespace metal;

constant uint GDN_C = 8192;
constant uint GDN_D = 128;

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

static inline void q5k_decode_block(thread float& acc, device const half* x, device const uchar* w,
                                    uint k0, uint kb, uint row, uint simd_lane) {
    long o = long(row) * 16 * 176 + kb * 176;
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

static inline half gdn_load_context(device const half* x, device const half* prev, long b, long c, long L,
                                    long pos, bool has_prev) {
    if (has_prev) return pos < 4 ? prev[(b * GDN_C + c) * 4 + pos] : x[(b * L + pos - 4) * GDN_C + c];
    return pos < 0 || pos >= L ? half(0.0) : x[(b * L + pos) * GDN_C + c];
}

static inline void gdn_store_half(device half* yq, device half* yz, device half* yb, device half* ya,
                                  uint row, half v) {
    if (row < 8192) yq[row] = v;
    else if (row < 12288) yz[row - 8192] = v;
    else if (row < 12320) yb[row - 12288] = v;
    else ya[row - 12320] = v;
}

[[max_total_threads_per_threadgroup(128)]]
kernel void gdn_in_proj_decode(device half* yq [[buffer(0)]], device half* yz [[buffer(1)]],
                               device half* yb [[buffer(2)]], device half* ya [[buffer(3)]],
                               device const half* x [[buffer(4)]], device const uchar* wq [[buffer(5)]],
                               device const uchar* wz [[buffer(6)]], device const half* wb [[buffer(7)]],
                               device const half* wa [[buffer(8)]],
                               uint simd_lane [[thread_index_in_simdgroup]],
                               uint simd_group [[simdgroup_index_in_threadgroup]],
                               uint3 group [[threadgroup_position_in_grid]]) {
    uint row = group.x * 4 + simd_group;
    if (row >= 12352) return;
    float acc = 0.0f;
    if (row < 12288) {
        device const uchar* w = row < 8192 ? wq : wz;
        uint local = row < 8192 ? row : row - 8192;
        for (uint kb = 0; kb < 16; ++kb) q5k_decode_block(acc, x, w, kb * 256, kb, local, simd_lane);
    } else {
        device const half* w = row < 12320 ? wb : wa;
        uint local = row < 12320 ? row - 12288 : row - 12320;
        for (uint k = simd_lane; k < 4096; k += 32) acc = fma(float(x[k]), float(w[local * 4096 + k]), acc);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0) gdn_store_half(yq, yz, yb, ya, row, half(acc));
}

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
