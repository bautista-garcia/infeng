#include <metal_stdlib>
using namespace metal;

constant uint GDN_C = 8192;
constant uint GDN_D = 128;

static inline half gdn_load_context(device const half* x, device const half* prev, long b, long c, long L,
                                    long pos, bool has_prev) {
    if (has_prev) return pos < 4 ? prev[(b * GDN_C + c) * 4 + pos] : x[(b * L + pos - 4) * GDN_C + c];
    return pos < 0 || pos >= L ? half(0.0) : x[(b * L + pos) * GDN_C + c];
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
