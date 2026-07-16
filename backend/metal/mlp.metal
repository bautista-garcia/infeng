#include <metal_stdlib>
using namespace metal;

constant uint MLP_K = 4096;
constant uint MLP_N = 12288;
constant uint MLP_NSIMD = 16;

static inline half h16(device const uchar* p) {
    ushort bits = ushort(p[0]) | (ushort(p[1]) << 8);
    return as_type<half>(bits);
}

static inline __attribute__((always_inline)) void q4k_fma(thread float& acc, float x,
                                                          float ds, float mm, uchar q) {
    acc = fma(x, ds * float(q) - mm, acc);
}

static inline __attribute__((always_inline)) void q4k_block(thread float& acc,
                                                            device const half* x,
                                                            device const uchar* weight,
                                                            uint k0, uint row_base,
                                                            uint block, uint lane) {
    uint offset = row_base + block * 144;
    device const uchar* scales = weight + offset + 4;

    // Decode metadata once per lane/block, with no runtime j branch in the
    // inner FMAs. Precompute the scale/min products so the hot path only
    // unpacks q nibbles and performs the float accumulation.
    float d = float(h16(weight + offset));
    float dm = float(h16(weight + offset + 2));
    uchar s0 = scales[0] & 63, s1 = scales[1] & 63;
    uchar s2 = scales[2] & 63, s3 = scales[3] & 63;
    uchar m0 = scales[4] & 63, m1 = scales[5] & 63;
    uchar m2 = scales[6] & 63, m3 = scales[7] & 63;
    uchar s4 = (scales[8] & 15) | ((scales[0] >> 6) << 4);
    uchar s5 = (scales[9] & 15) | ((scales[1] >> 6) << 4);
    uchar s6 = (scales[10] & 15) | ((scales[2] >> 6) << 4);
    uchar s7 = (scales[11] & 15) | ((scales[3] >> 6) << 4);
    uchar m4 = (scales[8] >> 4) | ((scales[4] >> 6) << 4);
    uchar m5 = (scales[9] >> 4) | ((scales[5] >> 6) << 4);
    uchar m6 = (scales[10] >> 4) | ((scales[6] >> 6) << 4);
    uchar m7 = (scales[11] >> 4) | ((scales[7] >> 6) << 4);

    float ds0 = d * float(s0), ds1 = d * float(s1), ds2 = d * float(s2), ds3 = d * float(s3);
    float mm0 = dm * float(m0), mm1 = dm * float(m1), mm2 = dm * float(m2), mm3 = dm * float(m3);

    uchar q = weight[offset + 16 + lane];
    q4k_fma(acc, float(x[k0 + lane]), ds0, mm0, q & 15);
    q4k_fma(acc, float(x[k0 + lane + 32]), ds1, mm1, q >> 4);
    q = weight[offset + 48 + lane];
    q4k_fma(acc, float(x[k0 + lane + 64]), ds2, mm2, q & 15);
    q4k_fma(acc, float(x[k0 + lane + 96]), ds3, mm3, q >> 4);

    ds0 = d * float(s4), ds1 = d * float(s5), ds2 = d * float(s6), ds3 = d * float(s7);
    mm0 = dm * float(m4), mm1 = dm * float(m5), mm2 = dm * float(m6), mm3 = dm * float(m7);
    q = weight[offset + 80 + lane];
    q4k_fma(acc, float(x[k0 + lane + 128]), ds0, mm0, q & 15);
    q4k_fma(acc, float(x[k0 + lane + 160]), ds1, mm1, q >> 4);
    q = weight[offset + 112 + lane];
    q4k_fma(acc, float(x[k0 + lane + 192]), ds2, mm2, q & 15);
    q4k_fma(acc, float(x[k0 + lane + 224]), ds3, mm3, q >> 4);
}

// Experimental representation: q contains only the 128 packed nibbles per
// block, while meta contains eight precomputed d*scale values followed by
// eight precomputed dmin*minimum values. This removes metadata bit unpacking
// and scale/min multiplication from the decode hot loop.
static inline __attribute__((always_inline)) void q4k_predecoded_block(thread float& acc,
                                                                       device const half* x,
                                                                       device const uchar* weight,
                                                                       device const float* meta,
                                                                       uint k0, uint q_row_base,
                                                                       uint meta_row_base, uint block,
                                                                       uint lane) {
    uint qo = q_row_base + block * 128;
    uint mo = meta_row_base + block * 16;
    float ds0 = meta[mo + 0], ds1 = meta[mo + 1], ds2 = meta[mo + 2], ds3 = meta[mo + 3];
    float ds4 = meta[mo + 4], ds5 = meta[mo + 5], ds6 = meta[mo + 6], ds7 = meta[mo + 7];
    float mm0 = meta[mo + 8], mm1 = meta[mo + 9], mm2 = meta[mo + 10], mm3 = meta[mo + 11];
    float mm4 = meta[mo + 12], mm5 = meta[mo + 13], mm6 = meta[mo + 14], mm7 = meta[mo + 15];
    uchar q = weight[qo + lane];
    q4k_fma(acc, float(x[k0 + lane]), ds0, mm0, q & 15);
    q4k_fma(acc, float(x[k0 + lane + 32]), ds1, mm1, q >> 4);
    q = weight[qo + 32 + lane];
    q4k_fma(acc, float(x[k0 + lane + 64]), ds2, mm2, q & 15);
    q4k_fma(acc, float(x[k0 + lane + 96]), ds3, mm3, q >> 4);
    q = weight[qo + 64 + lane];
    q4k_fma(acc, float(x[k0 + lane + 128]), ds4, mm4, q & 15);
    q4k_fma(acc, float(x[k0 + lane + 160]), ds5, mm5, q >> 4);
    q = weight[qo + 96 + lane];
    q4k_fma(acc, float(x[k0 + lane + 192]), ds6, mm6, q & 15);
    q4k_fma(acc, float(x[k0 + lane + 224]), ds7, mm7, q >> 4);
}

[[max_total_threads_per_threadgroup(512)]]
kernel void mlp_gate_up_q4_k_decode(
    device half* output [[buffer(0)]], device const half* input [[buffer(1)]],
    device const uchar* gate_weight [[buffer(2)]], device const uchar* up_weight [[buffer(3)]],
    constant long& sequence_length [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint row = group.x * MLP_NSIMD + simd_group;
    uint row_base = row * (MLP_K / 256) * 144;
    long token = long(group.y);
    float gate = 0.0f;
    float up = 0.0f;
    device const half* x = input + token * MLP_K;
#pragma unroll
    for (uint block = 0; block < MLP_K / 256; ++block) {
        uint k0 = block * 256;
        q4k_block(gate, x, gate_weight, k0, row_base, block, lane);
        q4k_block(up, x, up_weight, k0, row_base, block, lane);
    }
    gate = simd_sum(gate);
    up = simd_sum(up);
    if (token < sequence_length && row < MLP_N && lane == 0) {
        float gate_half = float(half(gate));
        float up_half = float(half(up));
        output[token * MLP_N + row] = half((gate_half / (1.0f + exp(-gate_half))) * up_half);
    }
}

[[max_total_threads_per_threadgroup(512)]]
kernel void mlp_gate_up_q4_k_decode_predecoded(
    device half* output [[buffer(0)]], device const half* input [[buffer(1)]],
    device const uchar* gate_weight [[buffer(2)]], device const uchar* up_weight [[buffer(3)]],
    device const float* gate_meta [[buffer(4)]], device const float* up_meta [[buffer(5)]],
    constant long& sequence_length [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint row = group.x * MLP_NSIMD + simd_group;
    uint q_row_base = row * (MLP_K / 256) * 128;
    uint meta_row_base = row * (MLP_K / 256) * 16;
    long token = long(group.y);
    float gate = 0.0f;
    float up = 0.0f;
    device const half* x = input + token * MLP_K;
#pragma unroll
    for (uint block = 0; block < MLP_K / 256; ++block) {
        uint k0 = block * 256;
        q4k_predecoded_block(gate, x, gate_weight, gate_meta,
                             k0, q_row_base, meta_row_base, block, lane);
        q4k_predecoded_block(up, x, up_weight, up_meta,
                             k0, q_row_base, meta_row_base, block, lane);
    }
    gate = simd_sum(gate);
    up = simd_sum(up);
    if (token < sequence_length && row < MLP_N && lane == 0) {
        float gate_half = float(half(gate));
        float up_half = float(half(up));
        output[token * MLP_N + row] = half((gate_half / (1.0f + exp(-gate_half))) * up_half);
    }
}
