#include <metal_stdlib>
#include <metal_matrix>
using namespace metal;

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

#define PREFILL_QK_V2_BN16(name, dequant, K, N) \
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

#define DECODE_Q4K_V2_REG_TG(name, TG, NSIMD, K, N) \
[[max_total_threads_per_threadgroup(TG)]] \
kernel void name(device half* y [[buffer(0)]], device const half* x [[buffer(1)]], \
                 device const uchar* w [[buffer(2)]], uint simd_lane [[thread_index_in_simdgroup]], \
                 uint simd_group [[simdgroup_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) { \
    uint row = group.x * NSIMD + simd_group; \
    float acc = 0.0f; \
    for (long kb = 0; kb < K / 256; ++kb) { \
        long k0 = kb * 256, o = long(row) * (K / 256) * 144 + kb * 144; \
        float d = float(h16(w + o)), dm = float(h16(w + o + 2)); \
        half x0 = x[k0 + simd_lane], x1 = x[k0 + simd_lane + 32], x2 = x[k0 + simd_lane + 64], x3 = x[k0 + simd_lane + 96]; \
        half x4 = x[k0 + simd_lane + 128], x5 = x[k0 + simd_lane + 160], x6 = x[k0 + simd_lane + 192], x7 = x[k0 + simd_lane + 224]; \
        uchar sc, mn, q0 = w[o + 16 + simd_lane], q1 = w[o + 48 + simd_lane], q2 = w[o + 80 + simd_lane], q3 = w[o + 112 + simd_lane]; \
        scale_min_k4(0, w + o + 4, sc, mn); acc = fma(float(x0), d * float(sc) * float(q0 & 15) - dm * float(mn), acc); \
        scale_min_k4(1, w + o + 4, sc, mn); acc = fma(float(x1), d * float(sc) * float(q0 >> 4) - dm * float(mn), acc); \
        scale_min_k4(2, w + o + 4, sc, mn); acc = fma(float(x2), d * float(sc) * float(q1 & 15) - dm * float(mn), acc); \
        scale_min_k4(3, w + o + 4, sc, mn); acc = fma(float(x3), d * float(sc) * float(q1 >> 4) - dm * float(mn), acc); \
        scale_min_k4(4, w + o + 4, sc, mn); acc = fma(float(x4), d * float(sc) * float(q2 & 15) - dm * float(mn), acc); \
        scale_min_k4(5, w + o + 4, sc, mn); acc = fma(float(x5), d * float(sc) * float(q2 >> 4) - dm * float(mn), acc); \
        scale_min_k4(6, w + o + 4, sc, mn); acc = fma(float(x6), d * float(sc) * float(q3 & 15) - dm * float(mn), acc); \
        scale_min_k4(7, w + o + 4, sc, mn); acc = fma(float(x7), d * float(sc) * float(q3 >> 4) - dm * float(mn), acc); \
    } \
    acc = simd_sum(acc); \
    if (row < N && simd_lane == 0) y[row] = half(acc); \
}

PREFILL_QK_V2_BN16(q4_k_k4096_n1024_prefill_v2_bn16, Q4K_DEQUANT_TILE, 4096, 1024)
PREFILL_QK_V2_BN16(q4_k_k4096_n4096_prefill_v2_bn16, Q4K_DEQUANT_TILE, 4096, 4096)
PREFILL_QK_V2_BN16(q4_k_k12288_n4096_prefill_v2_bn16, Q4K_DEQUANT_TILE, 12288, 4096)
PREFILL_QK_V2_BN16(q4_k_k4096_n8192_prefill_v2_bn16, Q4K_DEQUANT_TILE, 4096, 8192)
PREFILL_QK_V2_BN16(q4_k_k4096_n12288_prefill_v2_bn16, Q4K_DEQUANT_TILE, 4096, 12288)

DECODE_Q4K_V2_REG_TG(q4_k_k4096_n1024_decode_v2_tg256_reg, 256, 8, 4096, 1024)
DECODE_Q4K_V2_REG_TG(q4_k_k4096_n4096_decode_v2_tg128_reg, 128, 4, 4096, 4096)
DECODE_Q4K_V2_REG_TG(q4_k_k12288_n4096_decode_v2_tg128_reg, 128, 4, 12288, 4096)
DECODE_Q4K_V2_REG_TG(q4_k_k4096_n8192_decode_v2_tg128_reg, 128, 4, 4096, 8192)
DECODE_Q4K_V2_REG_TG(q4_k_k4096_n12288_decode_v2_tg128_reg, 128, 4, 4096, 12288)

PREFILL_QK_V2_BN16(q5_k_k4096_n1024_prefill_v2_bn16, Q5K_DEQUANT_TILE, 4096, 1024)
PREFILL_QK_V2_BN16(q5_k_k4096_n4096_prefill_v2_bn16, Q5K_DEQUANT_TILE, 4096, 4096)
PREFILL_QK_V2_BN16(q5_k_k12288_n4096_prefill_v2_bn16, Q5K_DEQUANT_TILE, 12288, 4096)
PREFILL_QK_V2_BN16(q5_k_k4096_n8192_prefill_v2_bn16, Q5K_DEQUANT_TILE, 4096, 8192)
PREFILL_QK_V2_BN16(q5_k_k4096_n12288_prefill_v2_bn16, Q5K_DEQUANT_TILE, 4096, 12288)

PREFILL_QK_V2_BN16(q6_k_k4096_n1024_prefill_v2_bn16, Q6K_DEQUANT_TILE, 4096, 1024)
PREFILL_QK_V2_BN16(q6_k_k12288_n4096_prefill_v2_bn16, Q6K_DEQUANT_TILE, 12288, 4096)
PREFILL_QK_V2_BN16(q6_k_k4096_n248320_prefill_v2_bn16, Q6K_DEQUANT_TILE, 4096, 248320)

PREFILL_QK_V2_BN16(q8_0_k4096_n4096_prefill_v2_bn16, Q8_0_DEQUANT_TILE, 4096, 4096)
