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

static inline void q6k_decode_block(thread float& acc, device const half* x, device const uchar* w,
                                    uint k0, uint kb, uint row, uint simd_lane, uint K) {
    uint l = simd_lane;
    long o = long(row) * (K / 256) * 210 + kb * 210;
    float d = float(h16(w + o + 208));
    for (uint h = 0; h < 2; ++h) for (uint p = 0; p < 4; ++p) {
        uint rr = p * 32 + l, qlo = h * 64 + ((rr & 32) ? 32 : 0) + l, r = h * 128 + rr;
        uchar q = w[o + qlo], lo = (rr & 64) ? (q >> 4) : (q & 15);
        uchar hi = (w[o + 128 + h * 32 + l] >> (2 * p)) & 3;
        char sc = char(w[o + 192 + h * 8 + (rr >> 4)]);
        acc = fma(float(x[k0 + r]), d * float(sc) * (float((hi << 4) | lo) - 32.0f), acc);
    }
}

static inline void q8_0_decode_block(thread float& acc, device const half* x, device const uchar* w,
                                     uint k0, uint kb, uint row, uint simd_lane, uint K) {
    long base = long(row) * (K / 32) * 34 + kb * 8 * 34;
    for (uint t = 0; t < 8; ++t) {
        long o = base + t * 34;
        acc = fma(float(x[k0 + t * 32 + simd_lane]), float(h16(w + o)) * float(char(w[o + 2 + simd_lane])), acc);
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

#define DECODE_QK_V2_REG_TG(name, decode_block, TG, NSIMD, K, N) \
[[max_total_threads_per_threadgroup(TG)]] \
kernel void name(device half* y [[buffer(0)]], device const half* x [[buffer(1)]], \
                 device const uchar* w [[buffer(2)]], uint simd_lane [[thread_index_in_simdgroup]], \
                 uint simd_group [[simdgroup_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) { \
    uint row = group.x * NSIMD + simd_group; \
    float acc = 0.0f; \
    for (uint kb = 0; kb < K / 256; ++kb) { \
        uint k0 = kb * 256; \
        decode_block(acc, x, w, k0, kb, row, simd_lane, K); \
    } \
    acc = simd_sum(acc); \
    if (row < N && simd_lane == 0) y[row] = half(acc); \
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

static inline void gdn_store_half2(device half* yq, device half* yz, device half* yb, device half* ya,
                                   long m, uint row, half2 v) {
    if (row < 8192) reinterpret_cast<device half2*>(yq)[(m * 8192 + row) >> 1] = v;
    else if (row < 12288) reinterpret_cast<device half2*>(yz)[(m * 4096 + row - 8192) >> 1] = v;
    else if (row < 12320) reinterpret_cast<device half2*>(yb)[(m * 32 + row - 12288) >> 1] = v;
    else reinterpret_cast<device half2*>(ya)[(m * 32 + row - 12320) >> 1] = v;
}

static inline void gdn_store_half(device half* yq, device half* yz, device half* yb, device half* ya,
                                  uint row, half v) {
    if (row < 8192) yq[row] = v;
    else if (row < 12288) yz[row - 8192] = v;
    else if (row < 12320) yb[row - 12288] = v;
    else ya[row - 12320] = v;
}

static inline void gdn_q5_f16_tile(threadgroup half* b_tile, device const uchar* wq, device const uchar* wz,
                                   device const half* wb, device const half* wa, uint n0, uint kb,
                                   uint simd_lane, uint simd_group) {
    for (uint p = 0; p < 4; ++p) {
        uint n = simd_group * 4 + p, row = n0 + n;
        if (row < 12288) {
            device const uchar* w = row < 8192 ? wq : wz;
            uint local = row < 8192 ? row : row - 8192;
            long o = long(local) * 16 * 176 + kb * 176;
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
        } else {
            device const half* w = row < 12320 ? wb : wa;
            uint local = row < 12320 ? row - 12288 : row - 12320;
            for (uint t = 0; t < 8; ++t) {
                uint r = t * 32 + simd_lane;
                b_tile[r * 16 + n] = w[local * 4096 + kb * 256 + r];
            }
        }
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void gdn_in_proj_prefill(device half* yq [[buffer(0)]], device half* yz [[buffer(1)]],
                                device half* yb [[buffer(2)]], device half* ya [[buffer(3)]],
                                device const half* x [[buffer(4)]], device const uchar* wq [[buffer(5)]],
                                device const uchar* wz [[buffer(6)]], device const half* wb [[buffer(7)]],
                                device const half* wa [[buffer(8)]], constant long& M [[buffer(9)]],
                                uint3 lane3 [[thread_position_in_threadgroup]],
                                uint simd_lane [[thread_index_in_simdgroup]],
                                uint simd_group [[simdgroup_index_in_threadgroup]],
                                uint3 group [[threadgroup_position_in_grid]]) {
    uint lane = lane3.x, rb = simd_group * 8;
    long n0 = long(group.x) * 16, m0 = long(group.y) * 32;
    threadgroup half b_tile[256 * 16];
    threadgroup float scratch[4 * 2 * 64];
    simdgroup_matrix<float, 8, 8> c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (long k0 = 0; k0 < 4096; k0 += 256) {
        device const half* x_ptr = x + m0 * 4096 + k0;
        gdn_q5_f16_tile(b_tile, wq, wz, wb, wa, uint(n0), uint(k0 / 256), simd_lane, simd_group);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<half, 8, 8> a, b;
        for (uint ko = 0; ko < 256; ko += 8) {
            simdgroup_load(a, x_ptr, 4096, ulong2(ko, rb));
            simdgroup_load(b, b_tile, 16, ulong2(0, ko)); simdgroup_multiply_accumulate(c0, a, b, c0);
            simdgroup_load(b, b_tile, 16, ulong2(8, ko)); simdgroup_multiply_accumulate(c1, a, b, c1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c0, scratch + simd_group * 128, 16);
    simdgroup_store(c1, scratch + simd_group * 128 + 8, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lane; idx < 32 * 8; idx += 128) {
        uint r = idx >> 3, cp = idx & 7, e = ((r & 7) << 4) + (cp << 1);
        gdn_store_half2(yq, yz, yb, ya, m0 + r, uint(n0) + (cp << 1),
                        half2(half(scratch[(r >> 3) * 128 + e]), half(scratch[(r >> 3) * 128 + e + 1])));
    }
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
        for (uint kb = 0; kb < 16; ++kb) q5k_decode_block(acc, x, w, kb * 256, kb, local, simd_lane, 4096);
    } else {
        device const half* w = row < 12320 ? wb : wa;
        uint local = row < 12320 ? row - 12288 : row - 12320;
        for (uint k = simd_lane; k < 4096; k += 32) acc = fma(float(x[k]), float(w[local * 4096 + k]), acc);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0) gdn_store_half(yq, yz, yb, ya, row, half(acc));
}

PREFILL_QK_V2_BN16(q4_k_k4096_n1024_prefill_v2_bn16, Q4K_DEQUANT_TILE, 4096, 1024)
PREFILL_QK_V2_BN16(q4_k_k4096_n4096_prefill_v2_bn16, Q4K_DEQUANT_TILE, 4096, 4096)
PREFILL_QK_V2_BN16(q4_k_k12288_n4096_prefill_v2_bn16, Q4K_DEQUANT_TILE, 12288, 4096)
PREFILL_QK_V2_BN16(q4_k_k4096_n8192_prefill_v2_bn16, Q4K_DEQUANT_TILE, 4096, 8192)
PREFILL_QK_V2_BN16(q4_k_k4096_n12288_prefill_v2_bn16, Q4K_DEQUANT_TILE, 4096, 12288)

DECODE_QK_V2_REG_TG(q4_k_k4096_n1024_decode_v2_tg256_reg, q4k_decode_block, 256, 8, 4096, 1024)
DECODE_QK_V2_REG_TG(q4_k_k4096_n4096_decode_v2_tg128_reg, q4k_decode_block, 128, 4, 4096, 4096)
DECODE_QK_V2_REG_TG(q4_k_k12288_n4096_decode_v2_tg128_reg, q4k_decode_block, 128, 4, 12288, 4096)
DECODE_QK_V2_REG_TG(q4_k_k4096_n8192_decode_v2_tg128_reg, q4k_decode_block, 128, 4, 4096, 8192)
DECODE_QK_V2_REG_TG(q4_k_k4096_n12288_decode_v2_tg128_reg, q4k_decode_block, 128, 4, 4096, 12288)

PREFILL_QK_V2_BN16(q5_k_k4096_n1024_prefill_v2_bn16, Q5K_DEQUANT_TILE, 4096, 1024)
PREFILL_QK_V2_BN16(q5_k_k4096_n4096_prefill_v2_bn16, Q5K_DEQUANT_TILE, 4096, 4096)
PREFILL_QK_V2_BN16(q5_k_k12288_n4096_prefill_v2_bn16, Q5K_DEQUANT_TILE, 12288, 4096)
PREFILL_QK_V2_BN16(q5_k_k4096_n8192_prefill_v2_bn16, Q5K_DEQUANT_TILE, 4096, 8192)
PREFILL_QK_V2_BN16(q5_k_k4096_n12288_prefill_v2_bn16, Q5K_DEQUANT_TILE, 4096, 12288)

DECODE_QK_V2_REG_TG(q5_k_k4096_n1024_decode_v2_tg256_reg, q5k_decode_block, 256, 8, 4096, 1024)
DECODE_QK_V2_REG_TG(q5_k_k4096_n4096_decode_v2_tg128_reg, q5k_decode_block, 128, 4, 4096, 4096)
DECODE_QK_V2_REG_TG(q5_k_k12288_n4096_decode_v2_tg128_reg, q5k_decode_block, 128, 4, 12288, 4096)
DECODE_QK_V2_REG_TG(q5_k_k4096_n8192_decode_v2_tg128_reg, q5k_decode_block, 128, 4, 4096, 8192)
DECODE_QK_V2_REG_TG(q5_k_k4096_n12288_decode_v2_tg128_reg, q5k_decode_block, 128, 4, 4096, 12288)

PREFILL_QK_V2_BN16(q6_k_k4096_n1024_prefill_v2_bn16, Q6K_DEQUANT_TILE, 4096, 1024)
PREFILL_QK_V2_BN16(q6_k_k12288_n4096_prefill_v2_bn16, Q6K_DEQUANT_TILE, 12288, 4096)
PREFILL_QK_V2_BN16(q6_k_k4096_n248320_prefill_v2_bn16, Q6K_DEQUANT_TILE, 4096, 248320)

DECODE_QK_V2_REG_TG(q6_k_k4096_n1024_decode_v2_tg256_reg, q6k_decode_block, 256, 8, 4096, 1024)
DECODE_QK_V2_REG_TG(q6_k_k12288_n4096_decode_v2_tg128_reg, q6k_decode_block, 128, 4, 12288, 4096)
DECODE_QK_V2_REG_TG(q6_k_k4096_n248320_decode_v2_tg128_reg, q6k_decode_block, 128, 4, 4096, 248320)

PREFILL_QK_V2_BN16(q8_0_k4096_n4096_prefill_v2_bn16, Q8_0_DEQUANT_TILE, 4096, 4096)
DECODE_QK_V2_REG_TG(q8_0_k4096_n4096_decode_v2_tg128_reg, q8_0_decode_block, 128, 4, 4096, 4096)

PREFILL_QK_V2_BN16(iq4_xs_k4096_n12288_prefill, IQ4_XS_DEQUANT_TILE, 4096, 12288)
DECODE_QK_V2_REG_TG(iq4_xs_k4096_n12288_decode, iq4xs_decode_block, 128, 4, 4096, 12288)
