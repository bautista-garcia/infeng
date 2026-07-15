#include <metal_stdlib>
using namespace metal;

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

static inline void q4k_row_fma(thread float& acc, float xv, device const uchar* sc_base,
                               float d, float dm, uchar q, uint j) {
    uchar sc, mn;
    scale_min_k4(j, sc_base, sc, mn);
    acc = fma(xv, d * float(sc) * float(q) - dm * float(mn), acc);
}

static inline void q5k_row_fma(thread float& acc, float xv, device const uchar* sc_base,
                               float d, float dm, uchar hm, uchar q, uint j) {
    uchar sc, mn;
    scale_min_k4(j, sc_base, sc, mn);
    acc = fma(xv, d * float(sc) * float(q + ((hm & (1 << j)) ? 16 : 0)) - dm * float(mn), acc);
}

static inline void q6k_row_fma(thread float& acc, float xv, device const uchar* w, long o, float d,
                               uint h, uint p, uint rr, uint qlo, uint l) {
    uchar q = w[o + qlo], lo = (rr & 64) ? (q >> 4) : (q & 15);
    uchar hi = (w[o + 128 + h * 32 + l] >> (2 * p)) & 3;
    char sc = char(w[o + 192 + h * 8 + (rr >> 4)]);
    acc = fma(xv, d * float(sc) * (float((hi << 4) | lo) - 32.0f), acc);
}

static inline void q8_0_row_fma(thread float& acc, float xv, device const uchar* w, long o, uint simd_lane) {
    acc = fma(xv, float(h16(w + o)) * float(char(w[o + 2 + simd_lane])), acc);
}

static inline void iq4xs_row_fma(thread float& acc, float xv, device const uchar* w, long o, float d,
                                 ushort sh, uint j, uint qj, uint simd_lane) {
    int ls = int((w[o + 4 + (j >> 1)] >> (4 * (j & 1))) & 15) | int(((sh >> (2 * j)) & 3) << 4);
    uchar q = w[o + 8 + qj], v = (simd_lane & 16) ? (q >> 4) : (q & 15);
    acc = fma(xv, d * float(ls - 32) * iq4nl[v], acc);
}

static inline void q4k_decode_block_r2(thread float& acc0, thread float& acc1, device const half* x,
                                       device const uchar* w, uint k0, uint kb, uint row, uint simd_lane, uint K) {
    long stride = long(K / 256) * 144;
    long o0 = long(row) * stride + kb * 144, o1 = long(row + 1) * stride + kb * 144;
    float d0 = float(h16(w + o0)), dm0 = float(h16(w + o0 + 2));
    float d1 = float(h16(w + o1)), dm1 = float(h16(w + o1 + 2));
    for (uint t = 0; t < 4; ++t) {
        uint r = t * 64 + simd_lane, j = t * 2;
        uchar q0 = w[o0 + 16 + t * 32 + simd_lane], q1 = w[o1 + 16 + t * 32 + simd_lane];
        float xv = float(x[k0 + r]);
        q4k_row_fma(acc0, xv, w + o0 + 4, d0, dm0, q0 & 15, j);
        q4k_row_fma(acc1, xv, w + o1 + 4, d1, dm1, q1 & 15, j);
        xv = float(x[k0 + r + 32]);
        q4k_row_fma(acc0, xv, w + o0 + 4, d0, dm0, q0 >> 4, j + 1);
        q4k_row_fma(acc1, xv, w + o1 + 4, d1, dm1, q1 >> 4, j + 1);
    }
}

static inline void q4k_decode_block_r4(thread float& acc0, thread float& acc1, thread float& acc2,
                                       thread float& acc3, device const half* x, device const uchar* w,
                                       uint k0, uint kb, uint row, uint simd_lane, uint K) {
    long stride = long(K / 256) * 144;
    long o0 = long(row) * stride + kb * 144, o1 = long(row + 1) * stride + kb * 144;
    long o2 = long(row + 2) * stride + kb * 144, o3 = long(row + 3) * stride + kb * 144;
    float d0 = float(h16(w + o0)), dm0 = float(h16(w + o0 + 2));
    float d1 = float(h16(w + o1)), dm1 = float(h16(w + o1 + 2));
    float d2 = float(h16(w + o2)), dm2 = float(h16(w + o2 + 2));
    float d3 = float(h16(w + o3)), dm3 = float(h16(w + o3 + 2));
    for (uint t = 0; t < 4; ++t) {
        uint r = t * 64 + simd_lane, j = t * 2;
        uchar q0 = w[o0 + 16 + t * 32 + simd_lane], q1 = w[o1 + 16 + t * 32 + simd_lane];
        uchar q2 = w[o2 + 16 + t * 32 + simd_lane], q3 = w[o3 + 16 + t * 32 + simd_lane];
        float xv = float(x[k0 + r]);
        q4k_row_fma(acc0, xv, w + o0 + 4, d0, dm0, q0 & 15, j);
        q4k_row_fma(acc1, xv, w + o1 + 4, d1, dm1, q1 & 15, j);
        q4k_row_fma(acc2, xv, w + o2 + 4, d2, dm2, q2 & 15, j);
        q4k_row_fma(acc3, xv, w + o3 + 4, d3, dm3, q3 & 15, j);
        xv = float(x[k0 + r + 32]);
        q4k_row_fma(acc0, xv, w + o0 + 4, d0, dm0, q0 >> 4, j + 1);
        q4k_row_fma(acc1, xv, w + o1 + 4, d1, dm1, q1 >> 4, j + 1);
        q4k_row_fma(acc2, xv, w + o2 + 4, d2, dm2, q2 >> 4, j + 1);
        q4k_row_fma(acc3, xv, w + o3 + 4, d3, dm3, q3 >> 4, j + 1);
    }
}

static inline void q5k_decode_block_r2(thread float& acc0, thread float& acc1, device const half* x,
                                       device const uchar* w, uint k0, uint kb, uint row, uint simd_lane, uint K) {
    long stride = long(K / 256) * 176;
    long o0 = long(row) * stride + kb * 176, o1 = long(row + 1) * stride + kb * 176;
    float d0 = float(h16(w + o0)), dm0 = float(h16(w + o0 + 2));
    float d1 = float(h16(w + o1)), dm1 = float(h16(w + o1 + 2));
    uchar hm0 = w[o0 + 16 + simd_lane], hm1 = w[o1 + 16 + simd_lane];
    for (uint t = 0; t < 4; ++t) {
        uint r = t * 64 + simd_lane, j = t * 2;
        uchar q0 = w[o0 + 48 + t * 32 + simd_lane], q1 = w[o1 + 48 + t * 32 + simd_lane];
        float xv = float(x[k0 + r]);
        q5k_row_fma(acc0, xv, w + o0 + 4, d0, dm0, hm0, q0 & 15, j);
        q5k_row_fma(acc1, xv, w + o1 + 4, d1, dm1, hm1, q1 & 15, j);
        xv = float(x[k0 + r + 32]);
        q5k_row_fma(acc0, xv, w + o0 + 4, d0, dm0, hm0, q0 >> 4, j + 1);
        q5k_row_fma(acc1, xv, w + o1 + 4, d1, dm1, hm1, q1 >> 4, j + 1);
    }
}

static inline void q5k_decode_block_r4(thread float& acc0, thread float& acc1, thread float& acc2,
                                       thread float& acc3, device const half* x, device const uchar* w,
                                       uint k0, uint kb, uint row, uint simd_lane, uint K) {
    long stride = long(K / 256) * 176;
    long o0 = long(row) * stride + kb * 176, o1 = long(row + 1) * stride + kb * 176;
    long o2 = long(row + 2) * stride + kb * 176, o3 = long(row + 3) * stride + kb * 176;
    float d0 = float(h16(w + o0)), dm0 = float(h16(w + o0 + 2));
    float d1 = float(h16(w + o1)), dm1 = float(h16(w + o1 + 2));
    float d2 = float(h16(w + o2)), dm2 = float(h16(w + o2 + 2));
    float d3 = float(h16(w + o3)), dm3 = float(h16(w + o3 + 2));
    uchar hm0 = w[o0 + 16 + simd_lane], hm1 = w[o1 + 16 + simd_lane];
    uchar hm2 = w[o2 + 16 + simd_lane], hm3 = w[o3 + 16 + simd_lane];
    for (uint t = 0; t < 4; ++t) {
        uint r = t * 64 + simd_lane, j = t * 2;
        uchar q0 = w[o0 + 48 + t * 32 + simd_lane], q1 = w[o1 + 48 + t * 32 + simd_lane];
        uchar q2 = w[o2 + 48 + t * 32 + simd_lane], q3 = w[o3 + 48 + t * 32 + simd_lane];
        float xv = float(x[k0 + r]);
        q5k_row_fma(acc0, xv, w + o0 + 4, d0, dm0, hm0, q0 & 15, j);
        q5k_row_fma(acc1, xv, w + o1 + 4, d1, dm1, hm1, q1 & 15, j);
        q5k_row_fma(acc2, xv, w + o2 + 4, d2, dm2, hm2, q2 & 15, j);
        q5k_row_fma(acc3, xv, w + o3 + 4, d3, dm3, hm3, q3 & 15, j);
        xv = float(x[k0 + r + 32]);
        q5k_row_fma(acc0, xv, w + o0 + 4, d0, dm0, hm0, q0 >> 4, j + 1);
        q5k_row_fma(acc1, xv, w + o1 + 4, d1, dm1, hm1, q1 >> 4, j + 1);
        q5k_row_fma(acc2, xv, w + o2 + 4, d2, dm2, hm2, q2 >> 4, j + 1);
        q5k_row_fma(acc3, xv, w + o3 + 4, d3, dm3, hm3, q3 >> 4, j + 1);
    }
}

static inline void q6k_decode_block_r2(thread float& acc0, thread float& acc1, device const half* x,
                                       device const uchar* w, uint k0, uint kb, uint row, uint simd_lane, uint K) {
    uint l = simd_lane;
    long stride = long(K / 256) * 210;
    long o0 = long(row) * stride + kb * 210, o1 = long(row + 1) * stride + kb * 210;
    float d0 = float(h16(w + o0 + 208)), d1 = float(h16(w + o1 + 208));
    for (uint h = 0; h < 2; ++h) for (uint p = 0; p < 4; ++p) {
        uint rr = p * 32 + l, qlo = h * 64 + ((rr & 32) ? 32 : 0) + l, r = h * 128 + rr;
        float xv = float(x[k0 + r]);
        q6k_row_fma(acc0, xv, w, o0, d0, h, p, rr, qlo, l);
        q6k_row_fma(acc1, xv, w, o1, d1, h, p, rr, qlo, l);
    }
}

static inline void q6k_decode_block_r4(thread float& acc0, thread float& acc1, thread float& acc2,
                                       thread float& acc3, device const half* x, device const uchar* w,
                                       uint k0, uint kb, uint row, uint simd_lane, uint K) {
    uint l = simd_lane;
    long stride = long(K / 256) * 210;
    long o0 = long(row) * stride + kb * 210, o1 = long(row + 1) * stride + kb * 210;
    long o2 = long(row + 2) * stride + kb * 210, o3 = long(row + 3) * stride + kb * 210;
    float d0 = float(h16(w + o0 + 208)), d1 = float(h16(w + o1 + 208));
    float d2 = float(h16(w + o2 + 208)), d3 = float(h16(w + o3 + 208));
    for (uint h = 0; h < 2; ++h) for (uint p = 0; p < 4; ++p) {
        uint rr = p * 32 + l, qlo = h * 64 + ((rr & 32) ? 32 : 0) + l, r = h * 128 + rr;
        float xv = float(x[k0 + r]);
        q6k_row_fma(acc0, xv, w, o0, d0, h, p, rr, qlo, l);
        q6k_row_fma(acc1, xv, w, o1, d1, h, p, rr, qlo, l);
        q6k_row_fma(acc2, xv, w, o2, d2, h, p, rr, qlo, l);
        q6k_row_fma(acc3, xv, w, o3, d3, h, p, rr, qlo, l);
    }
}

static inline void q8_0_decode_block_r2(thread float& acc0, thread float& acc1, device const half* x,
                                        device const uchar* w, uint k0, uint kb, uint row, uint simd_lane, uint K) {
    long stride = long(K / 32) * 34;
    long base0 = long(row) * stride + kb * 8 * 34;
    long base1 = long(row + 1) * stride + kb * 8 * 34;
    for (uint t = 0; t < 8; ++t) {
        float xv = float(x[k0 + t * 32 + simd_lane]);
        q8_0_row_fma(acc0, xv, w, base0 + t * 34, simd_lane);
        q8_0_row_fma(acc1, xv, w, base1 + t * 34, simd_lane);
    }
}

static inline void q8_0_decode_block_r4(thread float& acc0, thread float& acc1, thread float& acc2,
                                        thread float& acc3, device const half* x, device const uchar* w,
                                        uint k0, uint kb, uint row, uint simd_lane, uint K) {
    long stride = long(K / 32) * 34;
    long base0 = long(row) * stride + kb * 8 * 34;
    long base1 = long(row + 1) * stride + kb * 8 * 34;
    long base2 = long(row + 2) * stride + kb * 8 * 34;
    long base3 = long(row + 3) * stride + kb * 8 * 34;
    for (uint t = 0; t < 8; ++t) {
        float xv = float(x[k0 + t * 32 + simd_lane]);
        q8_0_row_fma(acc0, xv, w, base0 + t * 34, simd_lane);
        q8_0_row_fma(acc1, xv, w, base1 + t * 34, simd_lane);
        q8_0_row_fma(acc2, xv, w, base2 + t * 34, simd_lane);
        q8_0_row_fma(acc3, xv, w, base3 + t * 34, simd_lane);
    }
}

static inline void iq4xs_decode_block_r2(thread float& acc0, thread float& acc1, device const half* x,
                                         device const uchar* w, uint k0, uint kb, uint row, uint simd_lane, uint K) {
    long stride = long(K / 256) * 136;
    long o0 = long(row) * stride + kb * 136, o1 = long(row + 1) * stride + kb * 136;
    float d0 = float(h16(w + o0)), d1 = float(h16(w + o1));
    ushort sh0 = ushort(w[o0 + 2]) | (ushort(w[o0 + 3]) << 8);
    ushort sh1 = ushort(w[o1 + 2]) | (ushort(w[o1 + 3]) << 8);
    for (uint j = 0; j < 8; ++j) {
        uint r = j * 32 + simd_lane, qj = j * 16 + (simd_lane & 15);
        float xv = float(x[k0 + r]);
        iq4xs_row_fma(acc0, xv, w, o0, d0, sh0, j, qj, simd_lane);
        iq4xs_row_fma(acc1, xv, w, o1, d1, sh1, j, qj, simd_lane);
    }
}

static inline void iq4xs_decode_block_r4(thread float& acc0, thread float& acc1, thread float& acc2,
                                         thread float& acc3, device const half* x, device const uchar* w,
                                         uint k0, uint kb, uint row, uint simd_lane, uint K) {
    long stride = long(K / 256) * 136;
    long o0 = long(row) * stride + kb * 136, o1 = long(row + 1) * stride + kb * 136;
    long o2 = long(row + 2) * stride + kb * 136, o3 = long(row + 3) * stride + kb * 136;
    float d0 = float(h16(w + o0)), d1 = float(h16(w + o1));
    float d2 = float(h16(w + o2)), d3 = float(h16(w + o3));
    ushort sh0 = ushort(w[o0 + 2]) | (ushort(w[o0 + 3]) << 8);
    ushort sh1 = ushort(w[o1 + 2]) | (ushort(w[o1 + 3]) << 8);
    ushort sh2 = ushort(w[o2 + 2]) | (ushort(w[o2 + 3]) << 8);
    ushort sh3 = ushort(w[o3 + 2]) | (ushort(w[o3 + 3]) << 8);
    for (uint j = 0; j < 8; ++j) {
        uint r = j * 32 + simd_lane, qj = j * 16 + (simd_lane & 15);
        float xv = float(x[k0 + r]);
        iq4xs_row_fma(acc0, xv, w, o0, d0, sh0, j, qj, simd_lane);
        iq4xs_row_fma(acc1, xv, w, o1, d1, sh1, j, qj, simd_lane);
        iq4xs_row_fma(acc2, xv, w, o2, d2, sh2, j, qj, simd_lane);
        iq4xs_row_fma(acc3, xv, w, o3, d3, sh3, j, qj, simd_lane);
    }
}

#define DECODE_QK_R2(name, decode_block, TG, NSIMD, K, N) \
[[max_total_threads_per_threadgroup(TG)]] \
kernel void name(device half* y [[buffer(0)]], device const half* x [[buffer(1)]], \
                 device const uchar* w [[buffer(2)]], uint simd_lane [[thread_index_in_simdgroup]], \
                 uint simd_group [[simdgroup_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) { \
    uint row = group.x * (NSIMD * 2) + simd_group * 2; \
    float acc0 = 0.0f, acc1 = 0.0f; \
    for (uint kb = 0; kb < K / 256; ++kb) { \
        uint k0 = kb * 256; \
        decode_block(acc0, acc1, x, w, k0, kb, row, simd_lane, K); \
    } \
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1); \
    if (simd_lane == 0) { \
        if (row < N) y[row] = half(acc0); \
        if (row + 1 < N) y[row + 1] = half(acc1); \
    } \
}

#define DECODE_QK_R4(name, decode_block, TG, NSIMD, K, N) \
[[max_total_threads_per_threadgroup(TG)]] \
kernel void name(device half* y [[buffer(0)]], device const half* x [[buffer(1)]], \
                 device const uchar* w [[buffer(2)]], uint simd_lane [[thread_index_in_simdgroup]], \
                 uint simd_group [[simdgroup_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) { \
    uint row = group.x * (NSIMD * 4) + simd_group * 4; \
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f; \
    for (uint kb = 0; kb < K / 256; ++kb) { \
        uint k0 = kb * 256; \
        decode_block(acc0, acc1, acc2, acc3, x, w, k0, kb, row, simd_lane, K); \
    } \
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1); acc2 = simd_sum(acc2); acc3 = simd_sum(acc3); \
    if (simd_lane == 0) { \
        if (row < N) y[row] = half(acc0); \
        if (row + 1 < N) y[row + 1] = half(acc1); \
        if (row + 2 < N) y[row + 2] = half(acc2); \
        if (row + 3 < N) y[row + 3] = half(acc3); \
    } \
}

#define DECODE_QK_ROWBLOCKED(prefix, block_r2, block_r4, K, N) \
DECODE_QK_R2(prefix##_r2_tg64, block_r2, 64, 2, K, N) \
DECODE_QK_R2(prefix##_r2_tg128, block_r2, 128, 4, K, N) \
DECODE_QK_R4(prefix##_r4_tg64, block_r4, 64, 2, K, N)

DECODE_QK_ROWBLOCKED(q4_k_k4096_n1024_decode, q4k_decode_block_r2, q4k_decode_block_r4, 4096, 1024)
DECODE_QK_ROWBLOCKED(q4_k_k4096_n4096_decode, q4k_decode_block_r2, q4k_decode_block_r4, 4096, 4096)
DECODE_QK_ROWBLOCKED(q4_k_k12288_n4096_decode, q4k_decode_block_r2, q4k_decode_block_r4, 12288, 4096)
DECODE_QK_ROWBLOCKED(q4_k_k4096_n8192_decode, q4k_decode_block_r2, q4k_decode_block_r4, 4096, 8192)
DECODE_QK_ROWBLOCKED(q4_k_k4096_n12288_decode, q4k_decode_block_r2, q4k_decode_block_r4, 4096, 12288)

DECODE_QK_ROWBLOCKED(q5_k_k4096_n1024_decode, q5k_decode_block_r2, q5k_decode_block_r4, 4096, 1024)
DECODE_QK_ROWBLOCKED(q5_k_k4096_n4096_decode, q5k_decode_block_r2, q5k_decode_block_r4, 4096, 4096)
DECODE_QK_ROWBLOCKED(q5_k_k12288_n4096_decode, q5k_decode_block_r2, q5k_decode_block_r4, 12288, 4096)
DECODE_QK_ROWBLOCKED(q5_k_k4096_n8192_decode, q5k_decode_block_r2, q5k_decode_block_r4, 4096, 8192)
DECODE_QK_ROWBLOCKED(q5_k_k4096_n12288_decode, q5k_decode_block_r2, q5k_decode_block_r4, 4096, 12288)

DECODE_QK_ROWBLOCKED(q6_k_k4096_n1024_decode, q6k_decode_block_r2, q6k_decode_block_r4, 4096, 1024)
DECODE_QK_ROWBLOCKED(q6_k_k12288_n4096_decode, q6k_decode_block_r2, q6k_decode_block_r4, 12288, 4096)
DECODE_QK_ROWBLOCKED(q6_k_k4096_n248320_decode, q6k_decode_block_r2, q6k_decode_block_r4, 4096, 248320)

DECODE_QK_ROWBLOCKED(q8_0_k4096_n4096_decode, q8_0_decode_block_r2, q8_0_decode_block_r4, 4096, 4096)

DECODE_QK_ROWBLOCKED(iq4_xs_k4096_n12288_decode, iq4xs_decode_block_r2, iq4xs_decode_block_r4, 4096, 12288)
