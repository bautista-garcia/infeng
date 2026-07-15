#include <metal_stdlib>
#include <metal_matrix>
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

#define Q4K_DEQUANT_TILE(BN, K) \
    for (uint p = 0; p < BN / 4; ++p) { \
        uint n = simd_group * (BN / 4) + p; \
        long o = long(n0 + n) * (K / 256) * 144 + kb * 144; \
        float d = float(h16(w + o)), dm = float(h16(w + o + 2)); \
        for (uint t = 0; t < 4; ++t) { \
            uchar sc, mn, q = w[o + 16 + t * 32 + simd_lane]; \
            uint r = t * 64 + simd_lane, j = t * 2; \
            scale_min_k4(j, w + o + 4, sc, mn); \
            b_tile[r * BN + n] = half(d * float(sc) * float(q & 15) - dm * float(mn)); \
            scale_min_k4(j + 1, w + o + 4, sc, mn); \
            b_tile[(r + 32) * BN + n] = half(d * float(sc) * float(q >> 4) - dm * float(mn)); \
        } \
    }

#define Q5K_DEQUANT_TILE(BN, K) \
    for (uint p = 0; p < BN / 4; ++p) { \
        uint n = simd_group * (BN / 4) + p; \
        long o = long(n0 + n) * (K / 256) * 176 + kb * 176; \
        float d = float(h16(w + o)), dm = float(h16(w + o + 2)); \
        uchar hm = w[o + 16 + simd_lane]; \
        for (uint t = 0; t < 4; ++t) { \
            uchar sc, mn, q = w[o + 48 + t * 32 + simd_lane]; \
            uint r = t * 64 + simd_lane, j = t * 2; \
            scale_min_k4(j, w + o + 4, sc, mn); \
            b_tile[r * BN + n] = half(d * float(sc) * float((q & 15) + ((hm & (1 << j)) ? 16 : 0)) - dm * float(mn)); \
            scale_min_k4(j + 1, w + o + 4, sc, mn); \
            b_tile[(r + 32) * BN + n] = half(d * float(sc) * float((q >> 4) + ((hm & (1 << (j + 1))) ? 16 : 0)) - dm * float(mn)); \
        } \
    }

#define Q6K_DEQUANT_TILE(BN, K) \
    for (uint p = 0; p < BN / 4; ++p) { \
        uint n = simd_group * (BN / 4) + p, l = simd_lane; \
        long o = long(n0 + n) * (K / 256) * 210 + kb * 210; \
        float d = float(h16(w + o + 208)); \
        for (uint h = 0; h < 2; ++h) for (uint s = 0; s < 4; ++s) { \
            uint rr = s * 32 + l, qlo = h * 64 + ((rr & 32) ? 32 : 0) + l, r = h * 128 + rr; \
            uchar q = w[o + qlo], lo = (rr & 64) ? (q >> 4) : (q & 15); \
            uchar hi = (w[o + 128 + h * 32 + l] >> (2 * s)) & 3; \
            char sc = char(w[o + 192 + h * 8 + (rr >> 4)]); \
            b_tile[r * BN + n] = half(d * float(sc) * (float((hi << 4) | lo) - 32.0f)); \
        } \
    }

#define Q8_0_DEQUANT_TILE(BN, K) \
    for (uint p = 0; p < BN / 4; ++p) { \
        uint n = simd_group * (BN / 4) + p; \
        long base = long(n0 + n) * (K / 32) * 34 + kb * 8 * 34; \
        for (uint t = 0; t < 8; ++t) { \
            long o = base + t * 34; \
            b_tile[(t * 32 + simd_lane) * BN + n] = half(float(h16(w + o)) * float(char(w[o + 2 + simd_lane]))); \
        } \
    }

#define IQ4_XS_DEQUANT_TILE(BN, K) \
    for (uint p = 0; p < BN / 4; ++p) { \
        uint n = simd_group * (BN / 4) + p; \
        long o = long(n0 + n) * (K / 256) * 136 + kb * 136; \
        float d = float(h16(w + o)); \
        ushort sh = ushort(w[o + 2]) | (ushort(w[o + 3]) << 8); \
        for (uint j = 0; j < 8; ++j) { \
            uint r = j * 32 + simd_lane, qj = j * 16 + (simd_lane & 15); \
            int ls = int((w[o + 4 + (j >> 1)] >> (4 * (j & 1))) & 15) | int(((sh >> (2 * j)) & 3) << 4); \
            uchar q = w[o + 8 + qj], v = (simd_lane & 16) ? (q >> 4) : (q & 15); \
            b_tile[r * BN + n] = half(d * float(ls - 32) * iq4nl[v]); \
        } \
    }

#define PREFILL_QK(name, dequant, K, N) \
[[max_total_threads_per_threadgroup(128)]] \
kernel void name(device half* y [[buffer(0)]], device const half* x [[buffer(1)]], \
                 device const uchar* w [[buffer(2)]], constant long& M [[buffer(3)]], \
                 uint3 lane3 [[thread_position_in_threadgroup]], \
                 uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]], \
                 uint3 group [[threadgroup_position_in_grid]]) { \
    uint lane = lane3.x, rb = simd_group * 8; \
    long n0 = long(group.x) * 16, m0 = long(group.y) * 32; \
    threadgroup half b_tile[256 * 16]; \
    threadgroup float scratch[4 * 2 * 64]; \
    simdgroup_matrix<float, 8, 8> c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
    simdgroup_matrix<float, 8, 8> c1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
    for (long k0 = 0; k0 < K; k0 += 256) { \
        long kb = k0 / 256; \
        device const half* x_ptr = x + m0 * K + k0; \
        dequant(16, K) \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        simdgroup_matrix<half, 8, 8> a, b; \
        for (uint ko = 0; ko < 256; ko += 8) { \
            simdgroup_load(a, x_ptr, K, ulong2(ko, rb)); \
            simdgroup_load(b, b_tile, 16, ulong2(0, ko)); simdgroup_multiply_accumulate(c0, a, b, c0); \
            simdgroup_load(b, b_tile, 16, ulong2(8, ko)); simdgroup_multiply_accumulate(c1, a, b, c1); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    simdgroup_store(c0, scratch + simd_group * 128, 16); \
    simdgroup_store(c1, scratch + simd_group * 128 + 8, 16); \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    device half2* y2 = reinterpret_cast<device half2*>(y); \
    for (uint idx = lane; idx < 32 * 8; idx += 128) { \
        uint r = idx >> 3, cp = idx & 7, e = ((r & 7) << 4) + (cp << 1); \
        y2[((m0 + r) * N + n0 + (cp << 1)) >> 1] = half2(half(scratch[(r >> 3) * 128 + e]), \
                                                          half(scratch[(r >> 3) * 128 + e + 1])); \
    } \
}

template<uint N_ROWS>
static inline void decode_reduce(thread float& acc0, thread float& acc1) {
    acc0 = simd_sum(acc0);
    if (N_ROWS == 2) acc1 = simd_sum(acc1);
}

template<uint N_ROWS>
static inline void decode_store(device half* y, uint row, uint simd_lane, uint N,
                                float acc0, float acc1) {
    if (simd_lane == 0) {
        if (row < N) y[row] = half(acc0);
        if (N_ROWS == 2 && row + 1 < N) y[row + 1] = half(acc1);
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

template<uint N_ROWS>
static inline void q4k_pair(thread float& acc0, thread float& acc1, device const half* x, uint k0,
                            uint r, uint j, device const uchar* sc0, device const uchar* sc1,
                            float d0, float dm0, float d1, float dm1, uchar q0, uchar q1) {
    float xv = float(x[k0 + r]);
    q4k_row_fma(acc0, xv, sc0, d0, dm0, q0 & 15, j);
    if (N_ROWS == 2) q4k_row_fma(acc1, xv, sc1, d1, dm1, q1 & 15, j);
    xv = float(x[k0 + r + 32]);
    q4k_row_fma(acc0, xv, sc0, d0, dm0, q0 >> 4, j + 1);
    if (N_ROWS == 2) q4k_row_fma(acc1, xv, sc1, d1, dm1, q1 >> 4, j + 1);
}

template<uint N_ROWS>
static inline void q4k_decode_block(thread float& acc0, thread float& acc1, device const half* x,
                                    device const uchar* w, uint k0, uint kb, uint row, uint simd_lane, uint K) {
    long stride = long(K / 256) * 144;
    long o0 = long(row) * stride + kb * 144, o1 = o0;
    if (N_ROWS == 2) o1 = long(row + 1) * stride + kb * 144;
    float d0 = float(h16(w + o0)), dm0 = float(h16(w + o0 + 2));
    float d1 = d0, dm1 = dm0;
    if (N_ROWS == 2) { d1 = float(h16(w + o1)); dm1 = float(h16(w + o1 + 2)); }
    device const uchar* sc0 = w + o0 + 4;
    device const uchar* sc1 = w + o1 + 4;
    uchar q0 = w[o0 + 16 + simd_lane], q1 = q0;
    if (N_ROWS == 2) q1 = w[o1 + 16 + simd_lane];
    q4k_pair<N_ROWS>(acc0, acc1, x, k0, simd_lane, 0, sc0, sc1, d0, dm0, d1, dm1, q0, q1);
    q0 = w[o0 + 48 + simd_lane]; q1 = q0;
    if (N_ROWS == 2) q1 = w[o1 + 48 + simd_lane];
    q4k_pair<N_ROWS>(acc0, acc1, x, k0, simd_lane + 64, 2, sc0, sc1, d0, dm0, d1, dm1, q0, q1);
    q0 = w[o0 + 80 + simd_lane]; q1 = q0;
    if (N_ROWS == 2) q1 = w[o1 + 80 + simd_lane];
    q4k_pair<N_ROWS>(acc0, acc1, x, k0, simd_lane + 128, 4, sc0, sc1, d0, dm0, d1, dm1, q0, q1);
    q0 = w[o0 + 112 + simd_lane]; q1 = q0;
    if (N_ROWS == 2) q1 = w[o1 + 112 + simd_lane];
    q4k_pair<N_ROWS>(acc0, acc1, x, k0, simd_lane + 192, 6, sc0, sc1, d0, dm0, d1, dm1, q0, q1);
}

template<uint N_ROWS>
static inline void q5k_decode_block(thread float& acc0, thread float& acc1, device const half* x,
                                    device const uchar* w, uint k0, uint kb, uint row, uint simd_lane, uint K) {
    long stride = long(K / 256) * 176;
    long o0 = long(row) * stride + kb * 176, o1 = o0;
    if (N_ROWS == 2) o1 = long(row + 1) * stride + kb * 176;
    float d0 = float(h16(w + o0)), dm0 = float(h16(w + o0 + 2));
    float d1 = d0, dm1 = dm0;
    if (N_ROWS == 2) { d1 = float(h16(w + o1)); dm1 = float(h16(w + o1 + 2)); }
    uchar hm0 = w[o0 + 16 + simd_lane], hm1 = hm0;
    if (N_ROWS == 2) hm1 = w[o1 + 16 + simd_lane];
    device const uchar* sc0 = w + o0 + 4;
    device const uchar* sc1 = w + o1 + 4;
    for (uint t = 0; t < 4; ++t) {
        uint r = t * 64 + simd_lane, j = t * 2;
        uchar q0 = w[o0 + 48 + t * 32 + simd_lane], q1 = q0;
        if (N_ROWS == 2) q1 = w[o1 + 48 + t * 32 + simd_lane];
        float xv = float(x[k0 + r]);
        q5k_row_fma(acc0, xv, sc0, d0, dm0, hm0, q0 & 15, j);
        if (N_ROWS == 2) q5k_row_fma(acc1, xv, sc1, d1, dm1, hm1, q1 & 15, j);
        xv = float(x[k0 + r + 32]);
        q5k_row_fma(acc0, xv, sc0, d0, dm0, hm0, q0 >> 4, j + 1);
        if (N_ROWS == 2) q5k_row_fma(acc1, xv, sc1, d1, dm1, hm1, q1 >> 4, j + 1);
    }
}

template<uint N_ROWS>
static inline void q6k_decode_block(thread float& acc0, thread float& acc1, device const half* x,
                                    device const uchar* w, uint k0, uint kb, uint row, uint simd_lane, uint K) {
    uint l = simd_lane;
    long stride = long(K / 256) * 210;
    long o0 = long(row) * stride + kb * 210, o1 = o0;
    if (N_ROWS == 2) o1 = long(row + 1) * stride + kb * 210;
    float d0 = float(h16(w + o0 + 208)), d1 = d0;
    if (N_ROWS == 2) d1 = float(h16(w + o1 + 208));
    for (uint h = 0; h < 2; ++h) for (uint p = 0; p < 4; ++p) {
        uint rr = p * 32 + l, qlo = h * 64 + ((rr & 32) ? 32 : 0) + l, r = h * 128 + rr;
        float xv = float(x[k0 + r]);
        q6k_row_fma(acc0, xv, w, o0, d0, h, p, rr, qlo, l);
        if (N_ROWS == 2) q6k_row_fma(acc1, xv, w, o1, d1, h, p, rr, qlo, l);
    }
}

template<uint N_ROWS>
static inline void q8_0_decode_block(thread float& acc0, thread float& acc1, device const half* x,
                                     device const uchar* w, uint k0, uint kb, uint row, uint simd_lane, uint K) {
    long stride = long(K / 32) * 34;
    long base0 = long(row) * stride + kb * 8 * 34;
    long base1 = base0;
    if (N_ROWS == 2) base1 = long(row + 1) * stride + kb * 8 * 34;
    for (uint t = 0; t < 8; ++t) {
        float xv = float(x[k0 + t * 32 + simd_lane]);
        q8_0_row_fma(acc0, xv, w, base0 + t * 34, simd_lane);
        if (N_ROWS == 2) q8_0_row_fma(acc1, xv, w, base1 + t * 34, simd_lane);
    }
}

template<uint N_ROWS>
static inline void iq4xs_decode_block(thread float& acc0, thread float& acc1, device const half* x,
                                      device const uchar* w, uint k0, uint kb, uint row, uint simd_lane, uint K) {
    long stride = long(K / 256) * 136;
    long o0 = long(row) * stride + kb * 136, o1 = o0;
    if (N_ROWS == 2) o1 = long(row + 1) * stride + kb * 136;
    float d0 = float(h16(w + o0)), d1 = d0;
    if (N_ROWS == 2) d1 = float(h16(w + o1));
    ushort sh0 = ushort(w[o0 + 2]) | (ushort(w[o0 + 3]) << 8);
    ushort sh1 = sh0;
    if (N_ROWS == 2) sh1 = ushort(w[o1 + 2]) | (ushort(w[o1 + 3]) << 8);
    for (uint j = 0; j < 8; ++j) {
        uint r = j * 32 + simd_lane, qj = j * 16 + (simd_lane & 15);
        float xv = float(x[k0 + r]);
        iq4xs_row_fma(acc0, xv, w, o0, d0, sh0, j, qj, simd_lane);
        if (N_ROWS == 2) iq4xs_row_fma(acc1, xv, w, o1, d1, sh1, j, qj, simd_lane);
    }
}

#define DECODE_QK(name, decode_block, TG, NSIMD, N_ROWS, K, N) \
[[max_total_threads_per_threadgroup(TG)]] \
kernel void name(device half* y [[buffer(0)]], device const half* x [[buffer(1)]], \
                 device const uchar* w [[buffer(2)]], uint simd_lane [[thread_index_in_simdgroup]], \
                 uint simd_group [[simdgroup_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) { \
    uint row = group.x * (NSIMD * N_ROWS) + simd_group * N_ROWS; \
    float acc0 = 0.0f, acc1 = 0.0f; \
    for (uint kb = 0; kb < K / 256; ++kb) { \
        uint k0 = kb * 256; \
        decode_block<N_ROWS>(acc0, acc1, x, w, k0, kb, row, simd_lane, K); \
    } \
    decode_reduce<N_ROWS>(acc0, acc1); \
    decode_store<N_ROWS>(y, row, simd_lane, N, acc0, acc1); \
}

[[max_total_threads_per_threadgroup(256)]]
kernel void q4_k_embed(device half* y [[buffer(0)]], device const long* ids [[buffer(1)]],
                       device const uchar* w [[buffer(2)]], constant long& T [[buffer(3)]],
                       constant long& K [[buffer(4)]], uint3 lane3 [[thread_position_in_threadgroup]],
                       uint3 group [[threadgroup_position_in_grid]]) {
    uint col = group.x * 256 + lane3.x, t = group.y;
    if (t >= T || col >= K) return;
    long row = ids[t], nb = K / 256, o = row * nb * 144 + (col >> 8) * 144;
    uint r = col & 255, j = r >> 5, qj = (r >> 6) * 32 + (r & 31);
    uchar sc, mn; scale_min_k4(j, w + o + 4, sc, mn);
    uchar q = w[o + 16 + qj], v = (r & 32) ? (q >> 4) : (q & 15);
    y[t * K + col] = half(float(h16(w + o)) * float(sc) * float(v) - float(h16(w + o + 2)) * float(mn));
}

PREFILL_QK(q4_k_k4096_n1024_prefill, Q4K_DEQUANT_TILE, 4096, 1024)
PREFILL_QK(q4_k_k4096_n4096_prefill, Q4K_DEQUANT_TILE, 4096, 4096)
PREFILL_QK(q4_k_k12288_n4096_prefill, Q4K_DEQUANT_TILE, 12288, 4096)
PREFILL_QK(q4_k_k4096_n8192_prefill, Q4K_DEQUANT_TILE, 4096, 8192)
PREFILL_QK(q4_k_k4096_n12288_prefill, Q4K_DEQUANT_TILE, 4096, 12288)

DECODE_QK(q4_k_k4096_n1024_decode, q4k_decode_block, 256, 8, 1, 4096, 1024)
DECODE_QK(q4_k_k4096_n1024_decode_r2_tg64, q4k_decode_block, 64, 2, 2, 4096, 1024)
DECODE_QK(q4_k_k4096_n1024_decode_r2_tg128, q4k_decode_block, 128, 4, 2, 4096, 1024)
DECODE_QK(q4_k_k4096_n4096_decode, q4k_decode_block, 128, 4, 1, 4096, 4096)
DECODE_QK(q4_k_k4096_n4096_decode_r2_tg64, q4k_decode_block, 64, 2, 2, 4096, 4096)
DECODE_QK(q4_k_k4096_n4096_decode_r2_tg128, q4k_decode_block, 128, 4, 2, 4096, 4096)
DECODE_QK(q4_k_k12288_n4096_decode, q4k_decode_block, 128, 4, 1, 12288, 4096)
DECODE_QK(q4_k_k12288_n4096_decode_r2_tg64, q4k_decode_block, 64, 2, 2, 12288, 4096)
DECODE_QK(q4_k_k12288_n4096_decode_r2_tg128, q4k_decode_block, 128, 4, 2, 12288, 4096)
DECODE_QK(q4_k_k4096_n8192_decode, q4k_decode_block, 128, 4, 1, 4096, 8192)
DECODE_QK(q4_k_k4096_n8192_decode_r2_tg64, q4k_decode_block, 64, 2, 2, 4096, 8192)
DECODE_QK(q4_k_k4096_n8192_decode_r2_tg128, q4k_decode_block, 128, 4, 2, 4096, 8192)
DECODE_QK(q4_k_k4096_n12288_decode, q4k_decode_block, 128, 4, 1, 4096, 12288)
DECODE_QK(q4_k_k4096_n12288_decode_r2_tg64, q4k_decode_block, 64, 2, 2, 4096, 12288)
DECODE_QK(q4_k_k4096_n12288_decode_r2_tg128, q4k_decode_block, 128, 4, 2, 4096, 12288)

PREFILL_QK(q5_k_k4096_n1024_prefill, Q5K_DEQUANT_TILE, 4096, 1024)
PREFILL_QK(q5_k_k4096_n4096_prefill, Q5K_DEQUANT_TILE, 4096, 4096)
PREFILL_QK(q5_k_k12288_n4096_prefill, Q5K_DEQUANT_TILE, 12288, 4096)
PREFILL_QK(q5_k_k4096_n8192_prefill, Q5K_DEQUANT_TILE, 4096, 8192)
PREFILL_QK(q5_k_k4096_n12288_prefill, Q5K_DEQUANT_TILE, 4096, 12288)

DECODE_QK(q5_k_k4096_n1024_decode, q5k_decode_block, 256, 8, 1, 4096, 1024)
DECODE_QK(q5_k_k4096_n1024_decode_r2_tg64, q5k_decode_block, 64, 2, 2, 4096, 1024)
DECODE_QK(q5_k_k4096_n1024_decode_r2_tg128, q5k_decode_block, 128, 4, 2, 4096, 1024)
DECODE_QK(q5_k_k4096_n4096_decode, q5k_decode_block, 128, 4, 1, 4096, 4096)
DECODE_QK(q5_k_k4096_n4096_decode_r2_tg64, q5k_decode_block, 64, 2, 2, 4096, 4096)
DECODE_QK(q5_k_k4096_n4096_decode_r2_tg128, q5k_decode_block, 128, 4, 2, 4096, 4096)
DECODE_QK(q5_k_k12288_n4096_decode, q5k_decode_block, 128, 4, 1, 12288, 4096)
DECODE_QK(q5_k_k12288_n4096_decode_r2_tg64, q5k_decode_block, 64, 2, 2, 12288, 4096)
DECODE_QK(q5_k_k12288_n4096_decode_r2_tg128, q5k_decode_block, 128, 4, 2, 12288, 4096)
DECODE_QK(q5_k_k4096_n8192_decode, q5k_decode_block, 128, 4, 1, 4096, 8192)
DECODE_QK(q5_k_k4096_n8192_decode_r2_tg64, q5k_decode_block, 64, 2, 2, 4096, 8192)
DECODE_QK(q5_k_k4096_n8192_decode_r2_tg128, q5k_decode_block, 128, 4, 2, 4096, 8192)
DECODE_QK(q5_k_k4096_n12288_decode, q5k_decode_block, 128, 4, 1, 4096, 12288)
DECODE_QK(q5_k_k4096_n12288_decode_r2_tg64, q5k_decode_block, 64, 2, 2, 4096, 12288)
DECODE_QK(q5_k_k4096_n12288_decode_r2_tg128, q5k_decode_block, 128, 4, 2, 4096, 12288)

PREFILL_QK(q6_k_k4096_n1024_prefill, Q6K_DEQUANT_TILE, 4096, 1024)
PREFILL_QK(q6_k_k12288_n4096_prefill, Q6K_DEQUANT_TILE, 12288, 4096)
PREFILL_QK(q6_k_k4096_n248320_prefill, Q6K_DEQUANT_TILE, 4096, 248320)

DECODE_QK(q6_k_k4096_n1024_decode, q6k_decode_block, 256, 8, 1, 4096, 1024)
DECODE_QK(q6_k_k4096_n1024_decode_r2_tg64, q6k_decode_block, 64, 2, 2, 4096, 1024)
DECODE_QK(q6_k_k4096_n1024_decode_r2_tg128, q6k_decode_block, 128, 4, 2, 4096, 1024)
DECODE_QK(q6_k_k12288_n4096_decode, q6k_decode_block, 128, 4, 1, 12288, 4096)
DECODE_QK(q6_k_k12288_n4096_decode_r2_tg64, q6k_decode_block, 64, 2, 2, 12288, 4096)
DECODE_QK(q6_k_k12288_n4096_decode_r2_tg128, q6k_decode_block, 128, 4, 2, 12288, 4096)
DECODE_QK(q6_k_k4096_n248320_decode, q6k_decode_block, 128, 4, 1, 4096, 248320)
DECODE_QK(q6_k_k4096_n248320_decode_r2_tg64, q6k_decode_block, 64, 2, 2, 4096, 248320)
DECODE_QK(q6_k_k4096_n248320_decode_r2_tg128, q6k_decode_block, 128, 4, 2, 4096, 248320)

PREFILL_QK(q8_0_k4096_n4096_prefill, Q8_0_DEQUANT_TILE, 4096, 4096)
DECODE_QK(q8_0_k4096_n4096_decode, q8_0_decode_block, 128, 4, 1, 4096, 4096)
DECODE_QK(q8_0_k4096_n4096_decode_r2_tg64, q8_0_decode_block, 64, 2, 2, 4096, 4096)
DECODE_QK(q8_0_k4096_n4096_decode_r2_tg128, q8_0_decode_block, 128, 4, 2, 4096, 4096)

PREFILL_QK(iq4_xs_k4096_n12288_prefill, IQ4_XS_DEQUANT_TILE, 4096, 12288)
DECODE_QK(iq4_xs_k4096_n12288_decode, iq4xs_decode_block, 128, 4, 1, 4096, 12288)
DECODE_QK(iq4_xs_k4096_n12288_decode_r2_tg64, iq4xs_decode_block, 64, 2, 2, 4096, 12288)
DECODE_QK(iq4_xs_k4096_n12288_decode_r2_tg128, iq4xs_decode_block, 128, 4, 2, 4096, 12288)
