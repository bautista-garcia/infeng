#include <metal_stdlib>
using namespace metal;

constant int D = 128;

static inline long qkv_offset(long b, long t, long h, long d, long seq_len, long num_heads) {
    return ((b * seq_len + t) * num_heads + h) * D + d;
}

static inline long state_offset(long b, long h, long k, long v, long num_heads) {
    return ((b * num_heads + h) * D + k) * D + v;
}

static inline long value_offset(long b, long t, long h, long d, long s0, long s1, long s2, long s3) {
    return b * s0 + t * s1 + h * s2 + d * s3;
}

static inline void run_delta_rule_token(device half* output, device float* state, device const half* query,
                                        device const half* key, device const half* value, device const float* g,
                                        device const half* beta, long b, long t, long h, long seq_len, long num_heads,
                                        long vs0, long vs1, long vs2, long vs3, uint lane, threadgroup float* q,
                                        threadgroup float* k, threadgroup float* delta, threadgroup float* scratch) {
    float local = 0.0f;
    if (lane < D) {
        float qv = float(query[qkv_offset(b, t, h, lane, seq_len, num_heads)]);
        float kv = float(key[qkv_offset(b, t, h, lane, seq_len, num_heads)]);
        q[lane] = qv; k[lane] = kv; local = qv * qv;
    }
    scratch[lane] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512; stride; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float q_norm = rsqrt(scratch[0] + 1.0e-6f) * 0.08838834764831845f;

    local = lane < D ? k[lane] * k[lane] : 0.0f;
    scratch[lane] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512; stride; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float k_norm = rsqrt(scratch[0] + 1.0e-6f);
    if (lane < D) {
        q[lane] *= q_norm;
        k[lane] *= k_norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint vv = lane & 127, part = lane >> 7;
    float prediction = 0.0f, decay = exp(g[(b * seq_len + t) * num_heads + h]);
    for (uint kk = part; kk < D; kk += 8) {
            long off = state_offset(b, h, kk, vv, num_heads);
            float s = state[off] * decay;
            state[off] = s;
            prediction += s * k[kk];
    }
    scratch[lane] = prediction;
    threadgroup_barrier(mem_flags::mem_device);

    if (part == 0) {
        float pred = 0.0f;
        for (uint p = 0; p < 8; ++p) pred += scratch[p * D + vv];
        delta[vv] = (float(value[value_offset(b, t, h, vv, vs0, vs1, vs2, vs3)]) - pred) * float(beta[(b * seq_len + t) * num_heads + h]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = part; kk < D; kk += 8)
        state[state_offset(b, h, kk, vv, num_heads)] += k[kk] * delta[vv];
    threadgroup_barrier(mem_flags::mem_device);

    float out = 0.0f;
    for (uint kk = part; kk < D; kk += 8)
        out += state[state_offset(b, h, kk, vv, num_heads)] * q[kk];
    scratch[lane] = out;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0) {
        float total = 0.0f;
        for (uint p = 0; p < 8; ++p) total += scratch[p * D + vv];
        output[qkv_offset(b, t, h, vv, seq_len, num_heads)] = half(total);
    }
    threadgroup_barrier(mem_flags::mem_device);
}

kernel void delta_rule_prefill(device int* launch [[buffer(0)]], device half* output [[buffer(1)]],
                               device float* state [[buffer(2)]], device const half* query [[buffer(3)]],
                               device const half* key [[buffer(4)]], device const half* value [[buffer(5)]],
                               device const float* g [[buffer(6)]], device const half* beta [[buffer(7)]],
                               device const float* state_in [[buffer(8)]], constant long& batch_size [[buffer(9)]],
                               constant long& seq_len [[buffer(10)]], constant long& num_heads [[buffer(11)]],
                               constant long& vs0 [[buffer(12)]], constant long& vs1 [[buffer(13)]],
                               constant long& vs2 [[buffer(14)]], constant long& vs3 [[buffer(15)]],
                               constant bool& has_initial_state [[buffer(16)]], uint3 gid [[thread_position_in_grid]],
                               uint3 lane3 [[thread_position_in_threadgroup]], uint3 group3 [[threadgroup_position_in_grid]]) {
    uint lane = lane3.x;
    long group = group3.x;
    if (group >= batch_size * num_heads) return;
    launch[gid.x] = int(gid.x);
    long b = group / num_heads, h = group - b * num_heads;
    threadgroup float q[D], k[D], delta[D], scratch[1024];
    for (uint i = lane; i < D * D; i += 1024) {
        long off = state_offset(b, h, i / D, i % D, num_heads);
        state[off] = has_initial_state ? state_in[off] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (long t = 0; t < seq_len; ++t)
        run_delta_rule_token(output, state, query, key, value, g, beta, b, t, h, seq_len, num_heads, vs0, vs1, vs2, vs3, lane, q, k, delta, scratch);
}

// One threadgroup per (B, n_heads)
kernel void delta_rule_decode(device int* launch [[buffer(0)]], device half* output [[buffer(1)]],
                              device float* state [[buffer(2)]], device const half* query [[buffer(3)]],
                              device const half* key [[buffer(4)]], device const half* value [[buffer(5)]],
                              device const float* g [[buffer(6)]], device const half* beta [[buffer(7)]],
                              device const float* state_in [[buffer(8)]], constant long& batch_size [[buffer(9)]],
                              constant long& seq_len [[buffer(10)]], constant long& num_heads [[buffer(11)]],
                              constant long& vs0 [[buffer(12)]], constant long& vs1 [[buffer(13)]],
                              constant long& vs2 [[buffer(14)]], constant long& vs3 [[buffer(15)]],
                              constant bool& has_initial_state [[buffer(16)]], uint3 gid [[thread_position_in_grid]],
                              uint3 lane3 [[thread_position_in_threadgroup]], uint3 group3 [[threadgroup_position_in_grid]]) {
    uint lane = lane3.x;
    long group = group3.x;
    if (group >= batch_size * num_heads) return;
    launch[gid.x] = int(gid.x);
    long b = group / num_heads, h = group - b * num_heads;
    threadgroup float q[D], k[D], delta[D], scratch[1024];
    for (uint i = lane; i < D * D; i += 1024) {
        long off = state_offset(b, h, i / D, i % D, num_heads);
        state[off] = has_initial_state ? state_in[off] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_device);
    run_delta_rule_token(output, state, query, key, value, g, beta, b, 0, h, seq_len, num_heads, vs0, vs1, vs2, vs3, lane, q, k, delta, scratch);
}
