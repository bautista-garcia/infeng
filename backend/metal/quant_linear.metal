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
        // Extract sj and mj (63 = 0b00111111) from the s0..s3 and m0..m3 group
        d = q[j] & 63; m = q[j + 4] & 63; 
    } else {
        // Extract sj and mj from the s4..s7 and m4..m7 group
        d = (q[j + 4] & 15) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static inline float q4_dot32(device const half* x, device const uchar* w, long K, uint row, uint lane) {
    long nb = K / 256, base = row * nb * 144;
    float acc = 0.0f;
    for (long b = 0; b < nb; ++b) {
        long o = base + b * 144, k = b * 256;
        float d = float(h16(w + o)), dm = float(h16(w + o + 2));
        for (uint t = 0; t < 4; ++t) {
            uint r = t * 64 + lane, j = t * 2;
            uchar q = w[o + 16 + t * 32 + lane], sc, mn;
            scale_min_k4(j, w + o + 4, sc, mn);
            acc += float(x[k + r]) * (d * float(sc) * float(q & 15) - dm * float(mn));
            scale_min_k4(j + 1, w + o + 4, sc, mn);
            acc += float(x[k + r + 32]) * (d * float(sc) * float(q >> 4) - dm * float(mn));
        }
    }
    return acc;
}

static inline half q4_val(device const uchar* w, long K, uint row, uint col) {
    // Indexing from (row,col) into (blk)
    long nb = K / 256, b = col >> 8, r = col & 255, o = row * nb * 144 + b * 144;
    float d = float(h16(w + o)), dm = float(h16(w + o + 2));
    uint j = r >> 5, qj = (r >> 6) * 32 + (r & 31);
    uchar sc, mn; scale_min_k4(j, w + o + 4, sc, mn);
    uchar q = w[o + 16 + qj], v = (r & 32) ? (q >> 4) : (q & 15);
    return half(d * float(sc) * float(v) - dm * float(mn));
}

static inline half q5_val(device const uchar* w, long K, uint row, uint col) {
    long nb = K / 256, b = col >> 8, r = col & 255, o = row * nb * 176 + b * 176;
    float d = float(h16(w + o)), dm = float(h16(w + o + 2));
    uint j = r >> 5, qj = (r >> 6) * 32 + (r & 31);
    uchar sc, mn; scale_min_k4(j, w + o + 4, sc, mn);
    uchar q = w[o + 48 + qj], hi = (w[o + 16 + (r & 31)] & (1 << j)) ? 16 : 0;
    uchar v = ((r & 32) ? (q >> 4) : (q & 15)) + hi;
    return half(d * float(sc) * float(v) - dm * float(mn));
}

static inline half q6_val(device const uchar* w, long K, uint row, uint col) {
    long nb = K / 256, b = col >> 8, r = col & 255, o = row * nb * 210 + b * 210;
    uint h = r >> 7, rr = r & 127, l = rr & 31, qlo = h * 64 + ((rr & 32) ? 32 : 0) + l;
    uchar q = w[o + qlo], lo = (rr & 64) ? (q >> 4) : (q & 15);
    uchar hi = (w[o + 128 + h * 32 + l] >> (2 * ((rr >> 5) & 3))) & 3;
    return half(float(h16(w + o + 208)) * float(char(w[o + 192 + h * 8 + (rr >> 4)])) *
                (float((hi << 4) | lo) - 32.0f));
}

static inline half q8_val(device const uchar* w, long K, uint row, uint col) {
    long nb = K / 32, b = col >> 5, r = col & 31, o = row * nb * 34 + b * 34;
    return half(float(h16(w + o)) * float(char(w[o + 2 + r])));
}

static inline float q5_dot32(device const half* x, device const uchar* w, long K, uint row, uint lane) {
    long nb = K / 256, base = row * nb * 176;
    float acc = 0.0f;
    for (long b = 0; b < nb; ++b) {
        long o = base + b * 176, k = b * 256;
        float d = float(h16(w + o)), dm = float(h16(w + o + 2));
        uchar hm = w[o + 16 + lane];
        for (uint t = 0; t < 4; ++t) {
            uint r = t * 64 + lane, j = t * 2;
            uchar q = w[o + 48 + t * 32 + lane], sc, mn;
            scale_min_k4(j, w + o + 4, sc, mn);
            acc += float(x[k + r]) * (d * float(sc) * float((q & 15) + ((hm & (1 << j)) ? 16 : 0)) - dm * float(mn));
            scale_min_k4(j + 1, w + o + 4, sc, mn);
            acc += float(x[k + r + 32]) * (d * float(sc) * float((q >> 4) + ((hm & (1 << (j + 1))) ? 16 : 0)) - dm * float(mn));
        }
    }
    return acc;
}

static inline float q6_dot32(device const half* x, device const uchar* w, long K, uint row, uint lane) {
    long nb = K / 256, base = row * nb * 210;
    uint l = lane;
    float acc = 0.0f;
    for (long b = 0; b < nb; ++b) {
        long o = base + b * 210, k = b * 256;
        float d = float(h16(w + o + 208));
        for (uint h = 0; h < 2; ++h) for (uint p = 0; p < 4; ++p) {
            uint rr = p * 32 + l, qlo = h * 64 + ((rr & 32) ? 32 : 0) + l, r = h * 128 + rr;
            uchar q = w[o + qlo], lo = (rr & 64) ? (q >> 4) : (q & 15);
            uchar hi = (w[o + 128 + h * 32 + l] >> (2 * p)) & 3;
            char sc = char(w[o + 192 + h * 8 + (rr >> 4)]);
            acc += float(x[k + r]) * d * float(sc) * (float((hi << 4) | lo) - 32.0f);
        }
    }
    return acc;
}

static inline float q8_dot32(device const half* x, device const uchar* w, long K, uint row, uint lane) {
    long nb = K / 32, base = row * nb * 34;
    float acc = 0.0f;
    for (long k = lane; k < K; k += 32) {
        long b = k >> 5, r = k & 31, o = base + b * 34;
        acc += float(x[k]) * float(h16(w + o)) * float(char(w[o + 2 + r]));
    }
    return acc;
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

#define PREFILL_MMA32_KN(name, valfn, K, N) \
[[max_total_threads_per_threadgroup(128)]] \
kernel void name(device half* y [[buffer(0)]], device const half* x [[buffer(1)]], \
                 device const uchar* w [[buffer(2)]], constant long& M [[buffer(3)]], \
                 uint3 lane3 [[thread_position_in_threadgroup]], \
                 uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]], \
                 uint3 group [[threadgroup_position_in_grid]]) { \
    uint lane = lane3.x, rb = simd_group * 8; \
    long n0 = long(group.y) * 32, m0 = long(group.z) * 32; \
    threadgroup half a_tile[32 * 32], b_tile[32 * 32]; \
    threadgroup float scratch[4 * 4 * 64]; \
    simdgroup_matrix<float, 8, 8> c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
    simdgroup_matrix<float, 8, 8> c1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
    simdgroup_matrix<float, 8, 8> c2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
    simdgroup_matrix<float, 8, 8> c3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
    for (long k0 = 0; k0 < K; k0 += 32) { \
        for (uint idx = lane; idx < 32 * 32; idx += 128) { \
            uint r = idx >> 5, c = idx & 31; \
            a_tile[idx] = m0 + r < M ? x[(m0 + r) * K + k0 + c] : half(0.0f); \
            b_tile[idx] = valfn(w, K, uint(n0 + c), uint(k0 + r)); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        simdgroup_matrix<half, 8, 8> a, b; \
        for (uint ko = 0; ko < 32; ko += 8) { \
            simdgroup_load(a, a_tile, 32, ulong2(ko, rb)); \
            simdgroup_load(b, b_tile, 32, ulong2(0, ko)); simdgroup_multiply_accumulate(c0, a, b, c0); \
            simdgroup_load(b, b_tile, 32, ulong2(8, ko)); simdgroup_multiply_accumulate(c1, a, b, c1); \
            simdgroup_load(b, b_tile, 32, ulong2(16, ko)); simdgroup_multiply_accumulate(c2, a, b, c2); \
            simdgroup_load(b, b_tile, 32, ulong2(24, ko)); simdgroup_multiply_accumulate(c3, a, b, c3); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    simdgroup_store(c0, scratch + simd_group * 256 + 0, 8); \
    simdgroup_store(c1, scratch + simd_group * 256 + 64, 8); \
    simdgroup_store(c2, scratch + simd_group * 256 + 128, 8); \
    simdgroup_store(c3, scratch + simd_group * 256 + 192, 8); \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint idx = lane; idx < 32 * 32; idx += 128) { \
        uint r = idx >> 5, c = idx & 31, g = r >> 3, e = ((r & 7) << 3) + (c & 7); \
        if (m0 + r < M) y[(m0 + r) * N + n0 + c] = half(scratch[g * 256 + (c >> 3) * 64 + e]); \
    } \
}

DECODE_KN(q4_k_k4096_n1024_decode, q4_dot32, 4096, 1024)
DECODE_KN(q4_k_k4096_n4096_decode, q4_dot32, 4096, 4096)
DECODE_KN(q4_k_k12288_n4096_decode, q4_dot32, 12288, 4096)
DECODE_KN(q4_k_k4096_n8192_decode, q4_dot32, 4096, 8192)
DECODE_KN(q4_k_k4096_n12288_decode, q4_dot32, 4096, 12288)
DECODE_KN(q5_k_k4096_n1024_decode, q5_dot32, 4096, 1024)
DECODE_KN(q5_k_k4096_n4096_decode, q5_dot32, 4096, 4096)
DECODE_KN(q5_k_k12288_n4096_decode, q5_dot32, 12288, 4096)
DECODE_KN(q5_k_k4096_n8192_decode, q5_dot32, 4096, 8192)
DECODE_KN(q5_k_k4096_n12288_decode, q5_dot32, 4096, 12288)
DECODE_KN(q6_k_k4096_n1024_decode, q6_dot32, 4096, 1024)
DECODE_KN(q6_k_k12288_n4096_decode, q6_dot32, 12288, 4096)
DECODE_KN(q6_k_k4096_n248320_decode, q6_dot32, 4096, 248320)
DECODE_KN(q8_0_k4096_n4096_decode, q8_dot32, 4096, 4096)
DECODE_KN(iq4_xs_k4096_n12288_decode, iq4_dot32, 4096, 12288)

PREFILL_MMA32_KN(q4_k_k4096_n1024_prefill, q4_val, 4096, 1024)
PREFILL_MMA32_KN(q4_k_k4096_n4096_prefill, q4_val, 4096, 4096)
PREFILL_MMA32_KN(q4_k_k12288_n4096_prefill, q4_val, 12288, 4096)
PREFILL_MMA32_KN(q4_k_k4096_n8192_prefill, q4_val, 4096, 8192)
PREFILL_MMA32_KN(q4_k_k4096_n12288_prefill, q4_val, 4096, 12288)
PREFILL_MMA32_KN(q5_k_k4096_n1024_prefill, q5_val, 4096, 1024)
PREFILL_MMA32_KN(q5_k_k4096_n4096_prefill, q5_val, 4096, 4096)
PREFILL_MMA32_KN(q5_k_k12288_n4096_prefill, q5_val, 12288, 4096)
PREFILL_MMA32_KN(q5_k_k4096_n8192_prefill, q5_val, 4096, 8192)
PREFILL_MMA32_KN(q5_k_k4096_n12288_prefill, q5_val, 4096, 12288)
PREFILL_MMA32_KN(q6_k_k4096_n1024_prefill, q6_val, 4096, 1024)
PREFILL_MMA32_KN(q6_k_k12288_n4096_prefill, q6_val, 12288, 4096)
PREFILL_MMA32_KN(q6_k_k4096_n248320_prefill, q6_val, 4096, 248320)
PREFILL_MMA32_KN(q8_0_k4096_n4096_prefill, q8_val, 4096, 4096)
PREFILL_SCALAR_KN(iq4_xs_k4096_n12288_prefill, iq4_dot32, 4096, 12288)

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
