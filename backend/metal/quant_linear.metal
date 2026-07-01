#include <metal_stdlib>
using namespace metal;

constant float iq4nl[16] = {-127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
                              1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f};

static inline half h16(device const uchar* p) {
    ushort v = ushort(p[0]) | (ushort(p[1]) << 8);
    return as_type<half>(v);
}

static inline float iq4_dot32(device const half* x, device const uchar* w, long K, uint row, uint lane) {
    long nb = K / 256, base = row * nb * 136;
    float acc = 0.0f;
    for (long b = 0; b < nb; ++b) {
        long o = base + b * 136, k = b * 256;
        float d = float(h16(w + o));
        ushort sh = ushort(w[o + 2]) | (ushort(w[o + 3]) << 8);
        for (uint j = 0; j < 8; ++j) {
            uint r = j * 32 + lane, qj = j * 16 + (lane & 15);
            int ls = int((w[o + 4 + (j >> 1)] >> (4 * (j & 1))) & 15) | int(((sh >> (2 * j)) & 3) << 4);
            uchar q = w[o + 8 + qj], v = (lane & 16) ? (q >> 4) : (q & 15);
            acc += float(x[k + r]) * d * float(ls - 32) * iq4nl[v];
        }
    }
    return acc;
}

static inline void store4_32(device half* y, float a0, float a1, float a2, float a3, uint row, long N,
                             uint simd_lane) {
    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    if (simd_lane == 0) {
        if (row + 0 < N) y[row + 0] = half(a0);
        if (row + 1 < N) y[row + 1] = half(a1);
        if (row + 2 < N) y[row + 2] = half(a2);
        if (row + 3 < N) y[row + 3] = half(a3);
    }
}

#define DECODE_KN(name, dotfn, K, N) \
[[max_total_threads_per_threadgroup(32)]] \
kernel void name(device half* y [[buffer(0)]], device const half* x [[buffer(1)]], \
                 device const uchar* w [[buffer(2)]], uint3 lane3 [[thread_position_in_threadgroup]], \
                 uint simd_lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) { \
    uint lane = lane3.x, row = group.y * 4; \
    store4_32(y, row + 0 < N ? dotfn(x, w, K, row + 0, lane) : 0.0f, \
                 row + 1 < N ? dotfn(x, w, K, row + 1, lane) : 0.0f, \
                 row + 2 < N ? dotfn(x, w, K, row + 2, lane) : 0.0f, \
                 row + 3 < N ? dotfn(x, w, K, row + 3, lane) : 0.0f, row, N, simd_lane); \
}

#define PREFILL_SCALAR_KN(name, dotfn, K, N) \
[[max_total_threads_per_threadgroup(32)]] \
kernel void name(device half* y [[buffer(0)]], device const half* x [[buffer(1)]], \
                 device const uchar* w [[buffer(2)]], constant long& M [[buffer(3)]], \
                 uint3 lane3 [[thread_position_in_threadgroup]], \
                 uint simd_lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) { \
    uint lane = lane3.x, row = group.y * 4; \
    long token = group.z; \
    if (token >= M) return; \
    store4_32(y + token * N, row + 0 < N ? dotfn(x + token * K, w, K, row + 0, lane) : 0.0f, \
                            row + 1 < N ? dotfn(x + token * K, w, K, row + 1, lane) : 0.0f, \
                            row + 2 < N ? dotfn(x + token * K, w, K, row + 2, lane) : 0.0f, \
                            row + 3 < N ? dotfn(x + token * K, w, K, row + 3, lane) : 0.0f, row, N, simd_lane); \
}

DECODE_KN(iq4_xs_k4096_n12288_decode, iq4_dot32, 4096, 12288)
PREFILL_SCALAR_KN(iq4_xs_k4096_n12288_prefill, iq4_dot32, 4096, 12288)
