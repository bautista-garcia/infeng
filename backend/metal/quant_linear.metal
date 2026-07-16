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

struct q4k_tag {}; struct q5k_tag {}; struct q6k_tag {}; struct q8_0_tag {}; struct iq4xs_tag {};

template<uint K>
static inline __attribute__((always_inline)) void dequant_tile(q4k_tag, threadgroup half* b_tile,
                                                               device const uchar* w, long n0, long kb,
                                                               uint simd_lane, uint simd_group) {
    for (uint p = 0; p < 4; ++p) {
        uint n = simd_group * 4 + p;
        long o = long(n0 + n) * (K / 256) * 144 + kb * 144;
        float d = float(h16(w + o)), dm = float(h16(w + o + 2));
        for (uint t = 0; t < 4; ++t) {
            uchar sc, mn, q = w[o + 16 + t * 32 + simd_lane];
            uint r = t * 64 + simd_lane, j = t * 2;
            scale_min_k4(j, w + o + 4, sc, mn);
            b_tile[r * 16 + n] = half(d * float(sc) * float(q & 15) - dm * float(mn));
            scale_min_k4(j + 1, w + o + 4, sc, mn);
            b_tile[(r + 32) * 16 + n] = half(d * float(sc) * float(q >> 4) - dm * float(mn));
        }
    }
}

template<uint K>
static inline __attribute__((always_inline)) void dequant_tile(q5k_tag, threadgroup half* b_tile,
                                                               device const uchar* w, long n0, long kb,
                                                               uint simd_lane, uint simd_group) {
    for (uint p = 0; p < 4; ++p) {
        uint n = simd_group * 4 + p;
        long o = long(n0 + n) * (K / 256) * 176 + kb * 176;
        float d = float(h16(w + o)), dm = float(h16(w + o + 2));
        uchar hm = w[o + 16 + simd_lane];
        for (uint t = 0; t < 4; ++t) {
            uchar sc, mn, q = w[o + 48 + t * 32 + simd_lane];
            uint r = t * 64 + simd_lane, j = t * 2;
            scale_min_k4(j, w + o + 4, sc, mn);
            b_tile[r * 16 + n] = half(d * float(sc) * float((q & 15) + ((hm & (1 << j)) ? 16 : 0)) - dm * float(mn));
            scale_min_k4(j + 1, w + o + 4, sc, mn);
            b_tile[(r + 32) * 16 + n] = half(d * float(sc) * float((q >> 4) + ((hm & (1 << (j + 1))) ? 16 : 0)) - dm * float(mn));
        }
    }
}

template<uint K>
static inline __attribute__((always_inline)) void dequant_tile(q6k_tag, threadgroup half* b_tile,
                                                               device const uchar* w, long n0, long kb,
                                                               uint simd_lane, uint simd_group) {
    for (uint p = 0; p < 4; ++p) {
        uint n = simd_group * 4 + p, l = simd_lane;
        long o = long(n0 + n) * (K / 256) * 210 + kb * 210;
        float d = float(h16(w + o + 208));
        for (uint h = 0; h < 2; ++h) for (uint s = 0; s < 4; ++s) {
            uint rr = s * 32 + l, qlo = h * 64 + ((rr & 32) ? 32 : 0) + l, r = h * 128 + rr;
            uchar q = w[o + qlo], lo = (rr & 64) ? (q >> 4) : (q & 15);
            uchar hi = (w[o + 128 + h * 32 + l] >> (2 * s)) & 3;
            char sc = char(w[o + 192 + h * 8 + (rr >> 4)]);
            b_tile[r * 16 + n] = half(d * float(sc) * (float((hi << 4) | lo) - 32.0f));
        }
    }
}

template<uint K>
static inline __attribute__((always_inline)) void dequant_tile(q8_0_tag, threadgroup half* b_tile,
                                                               device const uchar* w, long n0, long kb,
                                                               uint simd_lane, uint simd_group) {
    for (uint p = 0; p < 4; ++p) {
        uint n = simd_group * 4 + p;
        long base = long(n0 + n) * (K / 32) * 34 + kb * 8 * 34;
        for (uint t = 0; t < 8; ++t) {
            long o = base + t * 34;
            b_tile[(t * 32 + simd_lane) * 16 + n] = half(float(h16(w + o)) * float(char(w[o + 2 + simd_lane])));
        }
    }
}

template<uint K>
static inline __attribute__((always_inline)) void dequant_tile(iq4xs_tag, threadgroup half* b_tile,
                                                               device const uchar* w, long n0, long kb,
                                                               uint simd_lane, uint simd_group) {
    for (uint p = 0; p < 4; ++p) {
        uint n = simd_group * 4 + p;
        long o = long(n0 + n) * (K / 256) * 136 + kb * 136;
        float d = float(h16(w + o));
        ushort sh = ushort(w[o + 2]) | (ushort(w[o + 3]) << 8);
        for (uint j = 0; j < 8; ++j) {
            uint r = j * 32 + simd_lane, qj = j * 16 + (simd_lane & 15);
            int ls = int((w[o + 4 + (j >> 1)] >> (4 * (j & 1))) & 15) | int(((sh >> (2 * j)) & 3) << 4);
            uchar q = w[o + 8 + qj], v = (simd_lane & 16) ? (q >> 4) : (q & 15);
            b_tile[r * 16 + n] = half(d * float(ls - 32) * iq4nl[v]);
        }
    }
}

template<typename Q, uint K, uint N>
[[max_total_threads_per_threadgroup(128)]]
kernel void prefill_qk(device half* y [[buffer(0)]], device const half* x [[buffer(1)]],
                       device const uchar* w [[buffer(2)]], constant long& M [[buffer(3)]],
                       uint3 lane3 [[thread_position_in_threadgroup]],
                       uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
                       uint3 group [[threadgroup_position_in_grid]]) {
    uint lane = lane3.x, rb = simd_group * 8;
    long n0 = long(group.x) * 16, m0 = long(group.y) * 32;
    threadgroup half b_tile[256 * 16];
    threadgroup float scratch[4 * 2 * 64];
    simdgroup_matrix<float, 8, 8> c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (long k0 = 0, kb = 0; k0 < K; k0 += 256, ++kb) {
        device const half* x_ptr = x + m0 * K + k0;
        dequant_tile<K>(Q(), b_tile, w, n0, kb, simd_lane, simd_group);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<half, 8, 8> a, b;
        for (uint ko = 0; ko < 256; ko += 8) {
            simdgroup_load(a, x_ptr, K, ulong2(ko, rb));
            simdgroup_load(b, b_tile, 16, ulong2(0, ko)); simdgroup_multiply_accumulate(c0, a, b, c0);
            simdgroup_load(b, b_tile, 16, ulong2(8, ko)); simdgroup_multiply_accumulate(c1, a, b, c1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c0, scratch + simd_group * 128, 16);
    simdgroup_store(c1, scratch + simd_group * 128 + 8, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half2* y2 = reinterpret_cast<device half2*>(y);
    for (uint idx = lane; idx < 32 * 8; idx += 128) {
        uint r = idx >> 3, cp = idx & 7, e = ((r & 7) << 4) + (cp << 1);
        y2[((m0 + r) * N + n0 + (cp << 1)) >> 1] = half2(half(scratch[(r >> 3) * 128 + e]),
                                                          half(scratch[(r >> 3) * 128 + e + 1]));
    }
}

// DECODE KERNELS

static inline __attribute__((always_inline)) void q4k_row_fma(thread float& acc, float xv, device const uchar* sc_base, float d, float dm, uchar q, uint j) {
    uchar sc, mn; scale_min_k4(j, sc_base, sc, mn);
    acc = fma(xv, d * float(sc) * float(q) - dm * float(mn), acc);
}

template<uint N_ROWS>
static inline __attribute__((always_inline)) void decode_block(q4k_tag, thread float& acc0, thread float& acc1,
                                                               device const half* x, device const uchar* w, uint k0,
                                                               uint kb, uint row, uint simd_lane, uint K) {
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
    float xv = float(x[k0 + simd_lane]); q4k_row_fma(acc0, xv, sc0, d0, dm0, q0 & 15, 0); if (N_ROWS == 2) q4k_row_fma(acc1, xv, sc1, d1, dm1, q1 & 15, 0);
    xv = float(x[k0 + simd_lane + 32]); q4k_row_fma(acc0, xv, sc0, d0, dm0, q0 >> 4, 1); if (N_ROWS == 2) q4k_row_fma(acc1, xv, sc1, d1, dm1, q1 >> 4, 1);
    q0 = w[o0 + 48 + simd_lane]; q1 = q0;
    if (N_ROWS == 2) q1 = w[o1 + 48 + simd_lane];
    xv = float(x[k0 + simd_lane + 64]); q4k_row_fma(acc0, xv, sc0, d0, dm0, q0 & 15, 2); if (N_ROWS == 2) q4k_row_fma(acc1, xv, sc1, d1, dm1, q1 & 15, 2);
    xv = float(x[k0 + simd_lane + 96]); q4k_row_fma(acc0, xv, sc0, d0, dm0, q0 >> 4, 3); if (N_ROWS == 2) q4k_row_fma(acc1, xv, sc1, d1, dm1, q1 >> 4, 3);
    q0 = w[o0 + 80 + simd_lane]; q1 = q0;
    if (N_ROWS == 2) q1 = w[o1 + 80 + simd_lane];
    xv = float(x[k0 + simd_lane + 128]); q4k_row_fma(acc0, xv, sc0, d0, dm0, q0 & 15, 4); if (N_ROWS == 2) q4k_row_fma(acc1, xv, sc1, d1, dm1, q1 & 15, 4);
    xv = float(x[k0 + simd_lane + 160]); q4k_row_fma(acc0, xv, sc0, d0, dm0, q0 >> 4, 5); if (N_ROWS == 2) q4k_row_fma(acc1, xv, sc1, d1, dm1, q1 >> 4, 5);
    q0 = w[o0 + 112 + simd_lane]; q1 = q0;
    if (N_ROWS == 2) q1 = w[o1 + 112 + simd_lane];
    xv = float(x[k0 + simd_lane + 192]); q4k_row_fma(acc0, xv, sc0, d0, dm0, q0 & 15, 6); if (N_ROWS == 2) q4k_row_fma(acc1, xv, sc1, d1, dm1, q1 & 15, 6);
    xv = float(x[k0 + simd_lane + 224]); q4k_row_fma(acc0, xv, sc0, d0, dm0, q0 >> 4, 7); if (N_ROWS == 2) q4k_row_fma(acc1, xv, sc1, d1, dm1, q1 >> 4, 7);
}

static inline __attribute__((always_inline)) void q5k_row_fma(thread float& acc, float xv, device const uchar* sc_base, float d, float dm, uchar hm, uchar q, uint j) {
    uchar sc, mn; scale_min_k4(j, sc_base, sc, mn);
    acc = fma(xv, d * float(sc) * float(q + ((hm & (1 << j)) ? 16 : 0)) - dm * float(mn), acc);
}

template<uint N_ROWS>
static inline __attribute__((always_inline)) void decode_block(q5k_tag, thread float& acc0, thread float& acc1,
                                                               device const half* x, device const uchar* w, uint k0,
                                                               uint kb, uint row, uint simd_lane, uint K) {
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
    uchar q0 = w[o0 + 48 + simd_lane], q1 = q0;
    if (N_ROWS == 2) q1 = w[o1 + 48 + simd_lane];
    float xv = float(x[k0 + simd_lane]); q5k_row_fma(acc0, xv, sc0, d0, dm0, hm0, q0 & 15, 0); if (N_ROWS == 2) q5k_row_fma(acc1, xv, sc1, d1, dm1, hm1, q1 & 15, 0);
    xv = float(x[k0 + simd_lane + 32]); q5k_row_fma(acc0, xv, sc0, d0, dm0, hm0, q0 >> 4, 1); if (N_ROWS == 2) q5k_row_fma(acc1, xv, sc1, d1, dm1, hm1, q1 >> 4, 1);
    q0 = w[o0 + 80 + simd_lane]; q1 = q0;
    if (N_ROWS == 2) q1 = w[o1 + 80 + simd_lane];
    xv = float(x[k0 + simd_lane + 64]); q5k_row_fma(acc0, xv, sc0, d0, dm0, hm0, q0 & 15, 2); if (N_ROWS == 2) q5k_row_fma(acc1, xv, sc1, d1, dm1, hm1, q1 & 15, 2);
    xv = float(x[k0 + simd_lane + 96]); q5k_row_fma(acc0, xv, sc0, d0, dm0, hm0, q0 >> 4, 3); if (N_ROWS == 2) q5k_row_fma(acc1, xv, sc1, d1, dm1, hm1, q1 >> 4, 3);
    q0 = w[o0 + 112 + simd_lane]; q1 = q0;
    if (N_ROWS == 2) q1 = w[o1 + 112 + simd_lane];
    xv = float(x[k0 + simd_lane + 128]); q5k_row_fma(acc0, xv, sc0, d0, dm0, hm0, q0 & 15, 4); if (N_ROWS == 2) q5k_row_fma(acc1, xv, sc1, d1, dm1, hm1, q1 & 15, 4);
    xv = float(x[k0 + simd_lane + 160]); q5k_row_fma(acc0, xv, sc0, d0, dm0, hm0, q0 >> 4, 5); if (N_ROWS == 2) q5k_row_fma(acc1, xv, sc1, d1, dm1, hm1, q1 >> 4, 5);
    q0 = w[o0 + 144 + simd_lane]; q1 = q0;
    if (N_ROWS == 2) q1 = w[o1 + 144 + simd_lane];
    xv = float(x[k0 + simd_lane + 192]); q5k_row_fma(acc0, xv, sc0, d0, dm0, hm0, q0 & 15, 6); if (N_ROWS == 2) q5k_row_fma(acc1, xv, sc1, d1, dm1, hm1, q1 & 15, 6);
    xv = float(x[k0 + simd_lane + 224]); q5k_row_fma(acc0, xv, sc0, d0, dm0, hm0, q0 >> 4, 7); if (N_ROWS == 2) q5k_row_fma(acc1, xv, sc1, d1, dm1, hm1, q1 >> 4, 7);
}

template<uint N_ROWS>
static inline __attribute__((always_inline)) void decode_block(q6k_tag, thread float& acc0, thread float& acc1,
                                                               device const half* x, device const uchar* w, uint k0,
                                                               uint kb, uint row, uint simd_lane, uint K) {
    uint l = simd_lane;
    long stride = long(K / 256) * 210;
    long o0 = long(row) * stride + kb * 210, o1 = o0;
    if (N_ROWS == 2) o1 = long(row + 1) * stride + kb * 210;
    float d0 = float(h16(w + o0 + 208)), d1 = d0;
    if (N_ROWS == 2) d1 = float(h16(w + o1 + 208));
    uint rr = l, qlo = l; uchar q = w[o0 + qlo], lo = q & 15, hi = w[o0 + 128 + l] & 3; char sc = char(w[o0 + 192 + (rr >> 4)]); float xv = float(x[k0 + rr]);
    acc0 = fma(xv, d0 * float(sc) * (float((hi << 4) | lo) - 32.0f), acc0); if (N_ROWS == 2) { q = w[o1 + qlo]; lo = q & 15; hi = w[o1 + 128 + l] & 3; sc = char(w[o1 + 192 + (rr >> 4)]); acc1 = fma(xv, d1 * float(sc) * (float((hi << 4) | lo) - 32.0f), acc1); }
    rr = 32 + l; qlo = 32 + l; q = w[o0 + qlo]; lo = q & 15; hi = (w[o0 + 128 + l] >> 2) & 3; sc = char(w[o0 + 192 + (rr >> 4)]); xv = float(x[k0 + rr]);
    acc0 = fma(xv, d0 * float(sc) * (float((hi << 4) | lo) - 32.0f), acc0); if (N_ROWS == 2) { q = w[o1 + qlo]; lo = q & 15; hi = (w[o1 + 128 + l] >> 2) & 3; sc = char(w[o1 + 192 + (rr >> 4)]); acc1 = fma(xv, d1 * float(sc) * (float((hi << 4) | lo) - 32.0f), acc1); }
    rr = 64 + l; qlo = l; q = w[o0 + qlo]; lo = q >> 4; hi = (w[o0 + 128 + l] >> 4) & 3; sc = char(w[o0 + 192 + (rr >> 4)]); xv = float(x[k0 + rr]);
    acc0 = fma(xv, d0 * float(sc) * (float((hi << 4) | lo) - 32.0f), acc0); if (N_ROWS == 2) { q = w[o1 + qlo]; lo = q >> 4; hi = (w[o1 + 128 + l] >> 4) & 3; sc = char(w[o1 + 192 + (rr >> 4)]); acc1 = fma(xv, d1 * float(sc) * (float((hi << 4) | lo) - 32.0f), acc1); }
    rr = 96 + l; qlo = 32 + l; q = w[o0 + qlo]; lo = q >> 4; hi = (w[o0 + 128 + l] >> 6) & 3; sc = char(w[o0 + 192 + (rr >> 4)]); xv = float(x[k0 + rr]);
    acc0 = fma(xv, d0 * float(sc) * (float((hi << 4) | lo) - 32.0f), acc0); if (N_ROWS == 2) { q = w[o1 + qlo]; lo = q >> 4; hi = (w[o1 + 128 + l] >> 6) & 3; sc = char(w[o1 + 192 + (rr >> 4)]); acc1 = fma(xv, d1 * float(sc) * (float((hi << 4) | lo) - 32.0f), acc1); }
    rr = l; qlo = 64 + l; q = w[o0 + qlo]; lo = q & 15; hi = w[o0 + 160 + l] & 3; sc = char(w[o0 + 200 + (rr >> 4)]); xv = float(x[k0 + 128 + rr]);
    acc0 = fma(xv, d0 * float(sc) * (float((hi << 4) | lo) - 32.0f), acc0); if (N_ROWS == 2) { q = w[o1 + qlo]; lo = q & 15; hi = w[o1 + 160 + l] & 3; sc = char(w[o1 + 200 + (rr >> 4)]); acc1 = fma(xv, d1 * float(sc) * (float((hi << 4) | lo) - 32.0f), acc1); }
    rr = 32 + l; qlo = 96 + l; q = w[o0 + qlo]; lo = q & 15; hi = (w[o0 + 160 + l] >> 2) & 3; sc = char(w[o0 + 200 + (rr >> 4)]); xv = float(x[k0 + 128 + rr]);
    acc0 = fma(xv, d0 * float(sc) * (float((hi << 4) | lo) - 32.0f), acc0); if (N_ROWS == 2) { q = w[o1 + qlo]; lo = q & 15; hi = (w[o1 + 160 + l] >> 2) & 3; sc = char(w[o1 + 200 + (rr >> 4)]); acc1 = fma(xv, d1 * float(sc) * (float((hi << 4) | lo) - 32.0f), acc1); }
    rr = 64 + l; qlo = 64 + l; q = w[o0 + qlo]; lo = q >> 4; hi = (w[o0 + 160 + l] >> 4) & 3; sc = char(w[o0 + 200 + (rr >> 4)]); xv = float(x[k0 + 128 + rr]);
    acc0 = fma(xv, d0 * float(sc) * (float((hi << 4) | lo) - 32.0f), acc0); if (N_ROWS == 2) { q = w[o1 + qlo]; lo = q >> 4; hi = (w[o1 + 160 + l] >> 4) & 3; sc = char(w[o1 + 200 + (rr >> 4)]); acc1 = fma(xv, d1 * float(sc) * (float((hi << 4) | lo) - 32.0f), acc1); }
    rr = 96 + l; qlo = 96 + l; q = w[o0 + qlo]; lo = q >> 4; hi = (w[o0 + 160 + l] >> 6) & 3; sc = char(w[o0 + 200 + (rr >> 4)]); xv = float(x[k0 + 128 + rr]);
    acc0 = fma(xv, d0 * float(sc) * (float((hi << 4) | lo) - 32.0f), acc0); if (N_ROWS == 2) { q = w[o1 + qlo]; lo = q >> 4; hi = (w[o1 + 160 + l] >> 6) & 3; sc = char(w[o1 + 200 + (rr >> 4)]); acc1 = fma(xv, d1 * float(sc) * (float((hi << 4) | lo) - 32.0f), acc1); }
}

template<uint N_ROWS>
static inline __attribute__((always_inline)) void decode_block(q8_0_tag, thread float& acc0, thread float& acc1,
                                                               device const half* x, device const uchar* w, uint k0,
                                                               uint kb, uint row, uint simd_lane, uint K) {
    long stride = long(K / 32) * 34;
    long base0 = long(row) * stride + kb * 8 * 34;
    long base1 = base0;
    if (N_ROWS == 2) base1 = long(row + 1) * stride + kb * 8 * 34;
    float xv = float(x[k0 + simd_lane]); acc0 = fma(xv, float(h16(w + base0)) * float(char(w[base0 + 2 + simd_lane])), acc0); if (N_ROWS == 2) acc1 = fma(xv, float(h16(w + base1)) * float(char(w[base1 + 2 + simd_lane])), acc1);
    xv = float(x[k0 + 32 + simd_lane]); acc0 = fma(xv, float(h16(w + base0 + 34)) * float(char(w[base0 + 36 + simd_lane])), acc0); if (N_ROWS == 2) acc1 = fma(xv, float(h16(w + base1 + 34)) * float(char(w[base1 + 36 + simd_lane])), acc1);
    xv = float(x[k0 + 64 + simd_lane]); acc0 = fma(xv, float(h16(w + base0 + 68)) * float(char(w[base0 + 70 + simd_lane])), acc0); if (N_ROWS == 2) acc1 = fma(xv, float(h16(w + base1 + 68)) * float(char(w[base1 + 70 + simd_lane])), acc1);
    xv = float(x[k0 + 96 + simd_lane]); acc0 = fma(xv, float(h16(w + base0 + 102)) * float(char(w[base0 + 104 + simd_lane])), acc0); if (N_ROWS == 2) acc1 = fma(xv, float(h16(w + base1 + 102)) * float(char(w[base1 + 104 + simd_lane])), acc1);
    xv = float(x[k0 + 128 + simd_lane]); acc0 = fma(xv, float(h16(w + base0 + 136)) * float(char(w[base0 + 138 + simd_lane])), acc0); if (N_ROWS == 2) acc1 = fma(xv, float(h16(w + base1 + 136)) * float(char(w[base1 + 138 + simd_lane])), acc1);
    xv = float(x[k0 + 160 + simd_lane]); acc0 = fma(xv, float(h16(w + base0 + 170)) * float(char(w[base0 + 172 + simd_lane])), acc0); if (N_ROWS == 2) acc1 = fma(xv, float(h16(w + base1 + 170)) * float(char(w[base1 + 172 + simd_lane])), acc1);
    xv = float(x[k0 + 192 + simd_lane]); acc0 = fma(xv, float(h16(w + base0 + 204)) * float(char(w[base0 + 206 + simd_lane])), acc0); if (N_ROWS == 2) acc1 = fma(xv, float(h16(w + base1 + 204)) * float(char(w[base1 + 206 + simd_lane])), acc1);
    xv = float(x[k0 + 224 + simd_lane]); acc0 = fma(xv, float(h16(w + base0 + 238)) * float(char(w[base0 + 240 + simd_lane])), acc0); if (N_ROWS == 2) acc1 = fma(xv, float(h16(w + base1 + 238)) * float(char(w[base1 + 240 + simd_lane])), acc1);
}

static inline __attribute__((always_inline)) void iq4xs_row_fma(thread float& acc, float xv, device const uchar* w, long o, float d, ushort sh, uint j, uint qj, uint simd_lane) {
    int ls = int((w[o + 4 + (j >> 1)] >> (4 * (j & 1))) & 15) | int(((sh >> (2 * j)) & 3) << 4);
    uchar q = w[o + 8 + qj], v = (simd_lane & 16) ? (q >> 4) : (q & 15);
    acc = fma(xv, d * float(ls - 32) * iq4nl[v], acc);
}

template<uint N_ROWS>
static inline __attribute__((always_inline)) void decode_block(iq4xs_tag, thread float& acc0, thread float& acc1,
                                                               device const half* x, device const uchar* w, uint k0,
                                                               uint kb, uint row, uint simd_lane, uint K) {
    long stride = long(K / 256) * 136;
    long o0 = long(row) * stride + kb * 136, o1 = o0;
    if (N_ROWS == 2) o1 = long(row + 1) * stride + kb * 136;
    float d0 = float(h16(w + o0)), d1 = d0;
    if (N_ROWS == 2) d1 = float(h16(w + o1));
    ushort sh0 = ushort(w[o0 + 2]) | (ushort(w[o0 + 3]) << 8);
    ushort sh1 = sh0;
    if (N_ROWS == 2) sh1 = ushort(w[o1 + 2]) | (ushort(w[o1 + 3]) << 8);
    uint ql = simd_lane & 15;
    float xv = float(x[k0 + simd_lane]); iq4xs_row_fma(acc0, xv, w, o0, d0, sh0, 0, ql, simd_lane); if (N_ROWS == 2) iq4xs_row_fma(acc1, xv, w, o1, d1, sh1, 0, ql, simd_lane);
    xv = float(x[k0 + 32 + simd_lane]); iq4xs_row_fma(acc0, xv, w, o0, d0, sh0, 1, 16 + ql, simd_lane); if (N_ROWS == 2) iq4xs_row_fma(acc1, xv, w, o1, d1, sh1, 1, 16 + ql, simd_lane);
    xv = float(x[k0 + 64 + simd_lane]); iq4xs_row_fma(acc0, xv, w, o0, d0, sh0, 2, 32 + ql, simd_lane); if (N_ROWS == 2) iq4xs_row_fma(acc1, xv, w, o1, d1, sh1, 2, 32 + ql, simd_lane);
    xv = float(x[k0 + 96 + simd_lane]); iq4xs_row_fma(acc0, xv, w, o0, d0, sh0, 3, 48 + ql, simd_lane); if (N_ROWS == 2) iq4xs_row_fma(acc1, xv, w, o1, d1, sh1, 3, 48 + ql, simd_lane);
    xv = float(x[k0 + 128 + simd_lane]); iq4xs_row_fma(acc0, xv, w, o0, d0, sh0, 4, 64 + ql, simd_lane); if (N_ROWS == 2) iq4xs_row_fma(acc1, xv, w, o1, d1, sh1, 4, 64 + ql, simd_lane);
    xv = float(x[k0 + 160 + simd_lane]); iq4xs_row_fma(acc0, xv, w, o0, d0, sh0, 5, 80 + ql, simd_lane); if (N_ROWS == 2) iq4xs_row_fma(acc1, xv, w, o1, d1, sh1, 5, 80 + ql, simd_lane);
    xv = float(x[k0 + 192 + simd_lane]); iq4xs_row_fma(acc0, xv, w, o0, d0, sh0, 6, 96 + ql, simd_lane); if (N_ROWS == 2) iq4xs_row_fma(acc1, xv, w, o1, d1, sh1, 6, 96 + ql, simd_lane);
    xv = float(x[k0 + 224 + simd_lane]); iq4xs_row_fma(acc0, xv, w, o0, d0, sh0, 7, 112 + ql, simd_lane); if (N_ROWS == 2) iq4xs_row_fma(acc1, xv, w, o1, d1, sh1, 7, 112 + ql, simd_lane);
}

template<typename Q, uint TG, uint NSIMD, uint N_ROWS, uint K, uint N>
[[max_total_threads_per_threadgroup(TG)]]
kernel void decode_qk(device half* y [[buffer(0)]], device const half* x [[buffer(1)]],
                      device const uchar* w [[buffer(2)]], uint simd_lane [[thread_index_in_simdgroup]],
                      uint simd_group [[simdgroup_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
    uint row = group.x * (NSIMD * N_ROWS) + simd_group * N_ROWS;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint kb = 0; kb < K / 256; ++kb) decode_block<N_ROWS>(Q(), acc0, acc1, x, w, kb * 256, kb, row, simd_lane, K);
    acc0 = simd_sum(acc0);
    if (N_ROWS == 2) acc1 = simd_sum(acc1);
    if (simd_lane == 0) {
        if (row < N) y[row] = half(acc0);
        if (N_ROWS == 2 && row + 1 < N) y[row + 1] = half(acc1);
    }
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

template [[host_name("q4_k_k4096_n1024_prefill")]] kernel void prefill_qk<q4k_tag, 4096, 1024>(device half*, device const half*, device const uchar*, constant long&, uint3, uint, uint, uint3);
template [[host_name("q4_k_k4096_n4096_prefill")]] kernel void prefill_qk<q4k_tag, 4096, 4096>(device half*, device const half*, device const uchar*, constant long&, uint3, uint, uint, uint3);
template [[host_name("q4_k_k12288_n4096_prefill")]] kernel void prefill_qk<q4k_tag, 12288, 4096>(device half*, device const half*, device const uchar*, constant long&, uint3, uint, uint, uint3);
template [[host_name("q4_k_k4096_n8192_prefill")]] kernel void prefill_qk<q4k_tag, 4096, 8192>(device half*, device const half*, device const uchar*, constant long&, uint3, uint, uint, uint3);
template [[host_name("q4_k_k4096_n12288_prefill")]] kernel void prefill_qk<q4k_tag, 4096, 12288>(device half*, device const half*, device const uchar*, constant long&, uint3, uint, uint, uint3);
template [[host_name("q4_k_k4096_n1024_decode")]] kernel void decode_qk<q4k_tag, 256, 8, 1, 4096, 1024>(device half*, device const half*, device const uchar*, uint, uint, uint3);
template [[host_name("q4_k_k4096_n4096_decode_r2_tg128")]] kernel void decode_qk<q4k_tag, 128, 4, 2, 4096, 4096>(device half*, device const half*, device const uchar*, uint, uint, uint3);
template [[host_name("q4_k_k12288_n4096_decode_r2_tg64")]] kernel void decode_qk<q4k_tag, 64, 2, 2, 12288, 4096>(device half*, device const half*, device const uchar*, uint, uint, uint3);
template [[host_name("q4_k_k4096_n8192_decode_r2_tg64")]] kernel void decode_qk<q4k_tag, 64, 2, 2, 4096, 8192>(device half*, device const half*, device const uchar*, uint, uint, uint3);
template [[host_name("q4_k_k4096_n12288_decode_r2_tg128")]] kernel void decode_qk<q4k_tag, 128, 4, 2, 4096, 12288>(device half*, device const half*, device const uchar*, uint, uint, uint3);

template [[host_name("q5_k_k4096_n1024_prefill")]] kernel void prefill_qk<q5k_tag, 4096, 1024>(device half*, device const half*, device const uchar*, constant long&, uint3, uint, uint, uint3);
template [[host_name("q5_k_k4096_n4096_prefill")]] kernel void prefill_qk<q5k_tag, 4096, 4096>(device half*, device const half*, device const uchar*, constant long&, uint3, uint, uint, uint3);
template [[host_name("q5_k_k12288_n4096_prefill")]] kernel void prefill_qk<q5k_tag, 12288, 4096>(device half*, device const half*, device const uchar*, constant long&, uint3, uint, uint, uint3);
template [[host_name("q5_k_k4096_n8192_prefill")]] kernel void prefill_qk<q5k_tag, 4096, 8192>(device half*, device const half*, device const uchar*, constant long&, uint3, uint, uint, uint3);
template [[host_name("q5_k_k4096_n12288_prefill")]] kernel void prefill_qk<q5k_tag, 4096, 12288>(device half*, device const half*, device const uchar*, constant long&, uint3, uint, uint, uint3);
template [[host_name("q5_k_k4096_n1024_decode_r2_tg128")]] kernel void decode_qk<q5k_tag, 128, 4, 2, 4096, 1024>(device half*, device const half*, device const uchar*, uint, uint, uint3);
template [[host_name("q5_k_k4096_n4096_decode_r2_tg64")]] kernel void decode_qk<q5k_tag, 64, 2, 2, 4096, 4096>(device half*, device const half*, device const uchar*, uint, uint, uint3);
template [[host_name("q5_k_k12288_n4096_decode_r2_tg128")]] kernel void decode_qk<q5k_tag, 128, 4, 2, 12288, 4096>(device half*, device const half*, device const uchar*, uint, uint, uint3);
template [[host_name("q5_k_k4096_n8192_decode_r2_tg128")]] kernel void decode_qk<q5k_tag, 128, 4, 2, 4096, 8192>(device half*, device const half*, device const uchar*, uint, uint, uint3);
template [[host_name("q5_k_k4096_n12288_decode_r2_tg64")]] kernel void decode_qk<q5k_tag, 64, 2, 2, 4096, 12288>(device half*, device const half*, device const uchar*, uint, uint, uint3);

template [[host_name("q6_k_k4096_n1024_prefill")]] kernel void prefill_qk<q6k_tag, 4096, 1024>(device half*, device const half*, device const uchar*, constant long&, uint3, uint, uint, uint3);
template [[host_name("q6_k_k12288_n4096_prefill")]] kernel void prefill_qk<q6k_tag, 12288, 4096>(device half*, device const half*, device const uchar*, constant long&, uint3, uint, uint, uint3);
template [[host_name("q6_k_k4096_n248320_prefill")]] kernel void prefill_qk<q6k_tag, 4096, 248320>(device half*, device const half*, device const uchar*, constant long&, uint3, uint, uint, uint3);
template [[host_name("q6_k_k4096_n1024_decode")]] kernel void decode_qk<q6k_tag, 256, 8, 1, 4096, 1024>(device half*, device const half*, device const uchar*, uint, uint, uint3);
template [[host_name("q6_k_k12288_n4096_decode")]] kernel void decode_qk<q6k_tag, 128, 4, 1, 12288, 4096>(device half*, device const half*, device const uchar*, uint, uint, uint3);
template [[host_name("q6_k_k4096_n248320_decode")]] kernel void decode_qk<q6k_tag, 128, 4, 1, 4096, 248320>(device half*, device const half*, device const uchar*, uint, uint, uint3);

template [[host_name("q8_0_k4096_n4096_prefill")]] kernel void prefill_qk<q8_0_tag, 4096, 4096>(device half*, device const half*, device const uchar*, constant long&, uint3, uint, uint, uint3);
template [[host_name("q8_0_k4096_n4096_decode")]] kernel void decode_qk<q8_0_tag, 128, 4, 1, 4096, 4096>(device half*, device const half*, device const uchar*, uint, uint, uint3);

template [[host_name("iq4_xs_k4096_n12288_prefill")]] kernel void prefill_qk<iq4xs_tag, 4096, 12288>(device half*, device const half*, device const uchar*, constant long&, uint3, uint, uint, uint3);
template [[host_name("iq4_xs_k4096_n12288_decode")]] kernel void decode_qk<iq4xs_tag, 128, 4, 1, 4096, 12288>(device half*, device const half*, device const uchar*, uint, uint, uint3);
