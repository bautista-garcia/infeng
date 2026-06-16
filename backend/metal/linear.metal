#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_N = 4;
constant uint TILE = 32;

static inline void linear_block4(device half* y, device const half* x, device const half* w, long K, long N,
                                 uint lane, uint simd_lane, uint simd_group, uint row, uint n0,
                                 threadgroup float* partial) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long k0 = lane; k0 < K; k0 += 1024) {
        half x0 = x[row * K + k0];
        a0 += n0 + 0 < N ? float(x0) * float(w[(n0 + 0) * K + k0]) : 0.0f;
        a1 += n0 + 1 < N ? float(x0) * float(w[(n0 + 1) * K + k0]) : 0.0f;
        a2 += n0 + 2 < N ? float(x0) * float(w[(n0 + 2) * K + k0]) : 0.0f;
        a3 += n0 + 3 < N ? float(x0) * float(w[(n0 + 3) * K + k0]) : 0.0f;
        if (k0 + 256 < K) {
            half x1 = x[row * K + k0 + 256];
            a0 += n0 + 0 < N ? float(x1) * float(w[(n0 + 0) * K + k0 + 256]) : 0.0f;
            a1 += n0 + 1 < N ? float(x1) * float(w[(n0 + 1) * K + k0 + 256]) : 0.0f;
            a2 += n0 + 2 < N ? float(x1) * float(w[(n0 + 2) * K + k0 + 256]) : 0.0f;
            a3 += n0 + 3 < N ? float(x1) * float(w[(n0 + 3) * K + k0 + 256]) : 0.0f;
        }
        if (k0 + 512 < K) {
            half x2 = x[row * K + k0 + 512];
            a0 += n0 + 0 < N ? float(x2) * float(w[(n0 + 0) * K + k0 + 512]) : 0.0f;
            a1 += n0 + 1 < N ? float(x2) * float(w[(n0 + 1) * K + k0 + 512]) : 0.0f;
            a2 += n0 + 2 < N ? float(x2) * float(w[(n0 + 2) * K + k0 + 512]) : 0.0f;
            a3 += n0 + 3 < N ? float(x2) * float(w[(n0 + 3) * K + k0 + 512]) : 0.0f;
        }
        if (k0 + 768 < K) {
            half x3 = x[row * K + k0 + 768];
            a0 += n0 + 0 < N ? float(x3) * float(w[(n0 + 0) * K + k0 + 768]) : 0.0f;
            a1 += n0 + 1 < N ? float(x3) * float(w[(n0 + 1) * K + k0 + 768]) : 0.0f;
            a2 += n0 + 2 < N ? float(x3) * float(w[(n0 + 2) * K + k0 + 768]) : 0.0f;
            a3 += n0 + 3 < N ? float(x3) * float(w[(n0 + 3) * K + k0 + 768]) : 0.0f;
        }
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    if (simd_lane == 0) {
        partial[simd_group * BLOCK_N + 0] = a0; partial[simd_group * BLOCK_N + 1] = a1;
        partial[simd_group * BLOCK_N + 2] = a2; partial[simd_group * BLOCK_N + 3] = a3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        a0 = simd_sum(simd_lane < 8 ? partial[simd_lane * BLOCK_N + 0] : 0.0f);
        a1 = simd_sum(simd_lane < 8 ? partial[simd_lane * BLOCK_N + 1] : 0.0f);
        a2 = simd_sum(simd_lane < 8 ? partial[simd_lane * BLOCK_N + 2] : 0.0f);
        a3 = simd_sum(simd_lane < 8 ? partial[simd_lane * BLOCK_N + 3] : 0.0f);
        if (simd_lane == 0) {
            if (n0 + 0 < N) y[row * N + n0 + 0] = half(a0);
            if (n0 + 1 < N) y[row * N + n0 + 1] = half(a1);
            if (n0 + 2 < N) y[row * N + n0 + 2] = half(a2);
            if (n0 + 3 < N) y[row * N + n0 + 3] = half(a3);
        }
    }
}

static inline void linear_block4_aligned(device half* y, device const half* x, device const half* w, long K, long N,
                                         uint lane, uint simd_lane, uint simd_group, uint row, uint n0,
                                         threadgroup float* partial) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long k0 = lane; k0 < K; k0 += 1024) {
        half x0 = x[row * K + k0], x1 = x[row * K + k0 + 256], x2 = x[row * K + k0 + 512], x3 = x[row * K + k0 + 768];
        a0 += float(x0) * float(w[(n0 + 0) * K + k0]) + float(x1) * float(w[(n0 + 0) * K + k0 + 256]) +
              float(x2) * float(w[(n0 + 0) * K + k0 + 512]) + float(x3) * float(w[(n0 + 0) * K + k0 + 768]);
        a1 += float(x0) * float(w[(n0 + 1) * K + k0]) + float(x1) * float(w[(n0 + 1) * K + k0 + 256]) +
              float(x2) * float(w[(n0 + 1) * K + k0 + 512]) + float(x3) * float(w[(n0 + 1) * K + k0 + 768]);
        a2 += float(x0) * float(w[(n0 + 2) * K + k0]) + float(x1) * float(w[(n0 + 2) * K + k0 + 256]) +
              float(x2) * float(w[(n0 + 2) * K + k0 + 512]) + float(x3) * float(w[(n0 + 2) * K + k0 + 768]);
        a3 += float(x0) * float(w[(n0 + 3) * K + k0]) + float(x1) * float(w[(n0 + 3) * K + k0 + 256]) +
              float(x2) * float(w[(n0 + 3) * K + k0 + 512]) + float(x3) * float(w[(n0 + 3) * K + k0 + 768]);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    if (simd_lane == 0) {
        partial[simd_group * BLOCK_N + 0] = a0; partial[simd_group * BLOCK_N + 1] = a1;
        partial[simd_group * BLOCK_N + 2] = a2; partial[simd_group * BLOCK_N + 3] = a3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        a0 = simd_sum(simd_lane < 8 ? partial[simd_lane * BLOCK_N + 0] : 0.0f);
        a1 = simd_sum(simd_lane < 8 ? partial[simd_lane * BLOCK_N + 1] : 0.0f);
        a2 = simd_sum(simd_lane < 8 ? partial[simd_lane * BLOCK_N + 2] : 0.0f);
        a3 = simd_sum(simd_lane < 8 ? partial[simd_lane * BLOCK_N + 3] : 0.0f);
        if (simd_lane == 0) {
            y[row * N + n0 + 0] = half(a0); y[row * N + n0 + 1] = half(a1);
            y[row * N + n0 + 2] = half(a2); y[row * N + n0 + 3] = half(a3);
        }
    }
}

static inline void linear_block4_aligned128(device half* y, device const half* x, device const half* w, long K, long N,
                                            uint lane, uint simd_lane, uint simd_group, uint row, uint n0,
                                            threadgroup float* partial) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long k0 = lane; k0 < K; k0 += 512) {
        half x0 = x[row * K + k0], x1 = x[row * K + k0 + 128], x2 = x[row * K + k0 + 256], x3 = x[row * K + k0 + 384];
        a0 += float(x0) * float(w[(n0 + 0) * K + k0]) + float(x1) * float(w[(n0 + 0) * K + k0 + 128]) +
              float(x2) * float(w[(n0 + 0) * K + k0 + 256]) + float(x3) * float(w[(n0 + 0) * K + k0 + 384]);
        a1 += float(x0) * float(w[(n0 + 1) * K + k0]) + float(x1) * float(w[(n0 + 1) * K + k0 + 128]) +
              float(x2) * float(w[(n0 + 1) * K + k0 + 256]) + float(x3) * float(w[(n0 + 1) * K + k0 + 384]);
        a2 += float(x0) * float(w[(n0 + 2) * K + k0]) + float(x1) * float(w[(n0 + 2) * K + k0 + 128]) +
              float(x2) * float(w[(n0 + 2) * K + k0 + 256]) + float(x3) * float(w[(n0 + 2) * K + k0 + 384]);
        a3 += float(x0) * float(w[(n0 + 3) * K + k0]) + float(x1) * float(w[(n0 + 3) * K + k0 + 128]) +
              float(x2) * float(w[(n0 + 3) * K + k0 + 256]) + float(x3) * float(w[(n0 + 3) * K + k0 + 384]);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    if (simd_lane == 0) {
        partial[simd_group * BLOCK_N + 0] = a0; partial[simd_group * BLOCK_N + 1] = a1;
        partial[simd_group * BLOCK_N + 2] = a2; partial[simd_group * BLOCK_N + 3] = a3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        a0 = simd_sum(simd_lane < 4 ? partial[simd_lane * BLOCK_N + 0] : 0.0f);
        a1 = simd_sum(simd_lane < 4 ? partial[simd_lane * BLOCK_N + 1] : 0.0f);
        a2 = simd_sum(simd_lane < 4 ? partial[simd_lane * BLOCK_N + 2] : 0.0f);
        a3 = simd_sum(simd_lane < 4 ? partial[simd_lane * BLOCK_N + 3] : 0.0f);
        if (simd_lane == 0) {
            y[row * N + n0 + 0] = half(a0); y[row * N + n0 + 1] = half(a1);
            y[row * N + n0 + 2] = half(a2); y[row * N + n0 + 3] = half(a3);
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void linear_decode(device half* y [[buffer(0)]], device const half* x [[buffer(1)]],
                          device const half* w [[buffer(2)]], constant long& M [[buffer(3)]],
                          constant long& K [[buffer(4)]], constant long& N [[buffer(5)]],
                          uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
                          uint3 lane3 [[thread_position_in_threadgroup]], uint3 group3 [[threadgroup_position_in_grid]]) {
    if (group3.y >= M) return;
    threadgroup float partial[32];
    linear_block4(y, x, w, K, N, lane3.x, simd_lane, simd_group, group3.y, group3.x * BLOCK_N, partial);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void linear_decode_aligned(device half* y [[buffer(0)]], device const half* x [[buffer(1)]],
                                  device const half* w [[buffer(2)]], constant long& M [[buffer(3)]],
                                  constant long& K [[buffer(4)]], constant long& N [[buffer(5)]],
                                  uint simd_lane [[thread_index_in_simdgroup]],
                                  uint simd_group [[simdgroup_index_in_threadgroup]],
                                  uint3 lane3 [[thread_position_in_threadgroup]], uint3 group3 [[threadgroup_position_in_grid]]) {
    if (group3.y >= M) return;
    threadgroup float partial[32];
    linear_block4_aligned(y, x, w, K, N, lane3.x, simd_lane, simd_group, group3.y, group3.x * BLOCK_N, partial);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void linear_decode_aligned128(device half* y [[buffer(0)]], device const half* x [[buffer(1)]],
                                     device const half* w [[buffer(2)]], constant long& M [[buffer(3)]],
                                     constant long& K [[buffer(4)]], constant long& N [[buffer(5)]],
                                     uint simd_lane [[thread_index_in_simdgroup]],
                                     uint simd_group [[simdgroup_index_in_threadgroup]],
                                     uint3 lane3 [[thread_position_in_threadgroup]], uint3 group3 [[threadgroup_position_in_grid]]) {
    if (group3.y >= M) return;
    threadgroup float partial[16];
    linear_block4_aligned128(y, x, w, K, N, lane3.x, simd_lane, simd_group, group3.y, group3.x * BLOCK_N, partial);
}

[[max_total_threads_per_threadgroup(512)]]
kernel void linear_prefill(device half* y [[buffer(0)]], device const half* x [[buffer(1)]],
                           device const half* w [[buffer(2)]], constant long& M [[buffer(3)]],
                           constant long& K [[buffer(4)]], constant long& N [[buffer(5)]],
                           uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
                           uint3 lane3 [[thread_position_in_threadgroup]], uint3 group3 [[threadgroup_position_in_grid]]) {
    uint lane = lane3.x, row0 = group3.y * TILE, n0 = group3.x * TILE;
    threadgroup half xt[TILE * TILE], wt[TILE * TILE];
    threadgroup float out[TILE * TILE];
    uint rb = (simd_group >> 2) << 3, cb = (simd_group & 3) << 3;
    simdgroup_matrix<half, 8, 8> a, b;
    simdgroup_matrix<float, 8, 8> c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (long k0 = 0; k0 < K; k0 += TILE) {
        for (uint i = lane; i < TILE * TILE; i += 512) {
            uint r = i / TILE, col = i % TILE;
            xt[i] = row0 + r < M && k0 + col < K ? x[(row0 + r) * K + k0 + col] : half(0.0f);
            wt[i] = n0 + col < N && k0 + r < K ? w[(n0 + col) * K + k0 + r] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ko = 0; ko < TILE; ko += 8) {
            simdgroup_load(a, xt, TILE, ulong2(ko, rb));
            simdgroup_load(b, wt, TILE, ulong2(cb, ko));
            simdgroup_multiply_accumulate(c, a, b, c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c, out, TILE, ulong2(cb, rb));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < TILE * TILE; i += 512) {
        uint r = i / TILE, col = i % TILE;
        if (row0 + r < M && n0 + col < N) y[(row0 + r) * N + n0 + col] = half(out[i]);
    }
}
