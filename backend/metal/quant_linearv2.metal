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

#define PREFILL_Q4K_V2_BN16(name, K, N) \
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
        Q4K_DEQUANT_TILE(16, K) \
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

PREFILL_Q4K_V2_BN16(q4_k_k4096_n1024_prefill_v2_bn16, 4096, 1024)
PREFILL_Q4K_V2_BN16(q4_k_k4096_n4096_prefill_v2_bn16, 4096, 4096)
PREFILL_Q4K_V2_BN16(q4_k_k12288_n4096_prefill_v2_bn16, 12288, 4096)
PREFILL_Q4K_V2_BN16(q4_k_k4096_n8192_prefill_v2_bn16, 4096, 8192)
PREFILL_Q4K_V2_BN16(q4_k_k4096_n12288_prefill_v2_bn16, 4096, 12288)
