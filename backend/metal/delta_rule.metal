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
                                        long vs0, long vs1, long vs2, long vs3, uint lane, uint simd_lane,
                                        uint simd_group, threadgroup float* q, threadgroup float* k,
                                        threadgroup float* delta, threadgroup float* scratch) {
    if (simd_group < D / 32) {
        float qv = float(query[qkv_offset(b, t, h, lane, seq_len, num_heads)]);
        float kv = float(key[qkv_offset(b, t, h, lane, seq_len, num_heads)]);
        q[lane] = qv; k[lane] = kv;
        float q_partial = simd_sum(qv * qv), k_partial = simd_sum(kv * kv);
        if (simd_lane == 0) {
            scratch[simd_group] = q_partial;
            scratch[D / 32 + simd_group] = k_partial;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float q_total = simd_sum(simd_lane < D / 32 ? scratch[simd_lane] : 0.0f);
        float k_total = simd_sum(simd_lane < D / 32 ? scratch[D / 32 + simd_lane] : 0.0f);
        if (simd_lane == 0) {
            scratch[0] = q_total;
            scratch[1] = k_total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group < D / 32) {
        float q_norm = rsqrt(scratch[0] + 1.0e-6f) * 0.08838834764831845f;
        float k_norm = rsqrt(scratch[1] + 1.0e-6f);
        q[lane] *= q_norm;
        k[lane] *= k_norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Decay + Prediction (v = kt @ St-1 * decay)
    uint vv = lane & 127, part = lane >> 7;
    long base_offset = state_offset(b, h, 0, vv, num_heads);
    long row_stride  = state_offset(b, h, 1, vv, num_heads) - base_offset;
    long loop_stride = 8 * row_stride; 
    
    long off = base_offset + (part * row_stride);
    float prediction = 0.0f;
    float decay = exp(g[(b * seq_len + t) * num_heads + h]);
    
    for (uint kk = part; kk < D; kk += 8) {
        float s = state[off] * decay;
        state[off] = s;             
        prediction += s * k[kk];
        off += loop_stride;
    }
    scratch[lane] = prediction;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The first row of 128 threads reduce the 8 partials that summed give v[lane]
    if (part == 0) {
        float pred = scratch[0 * 128 + vv] + scratch[1 * 128 + vv] + 
                     scratch[2 * 128 + vv] + scratch[3 * 128 + vv] +
                     scratch[4 * 128 + vv] + scratch[5 * 128 + vv] + 
                     scratch[6 * 128 + vv] + scratch[7 * 128 + vv];
                     
        delta[vv] = (float(value[value_offset(b, t, h, vv, vs0, vs1, vs2, vs3)]) - pred) * float(beta[(b * seq_len + t) * num_heads + h]);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out = 0.0f;
    float cached_delta = delta[vv]; // Cache into local register execution space
    off = base_offset + (part * row_stride); // Reset memory pointer to the top row

    for (uint kk = part; kk < D; kk += 8) {
        // Outer product
        float s = state[off]; 
        s += k[kk] * cached_delta;
        state[off] = s; 

        // Output projection matmul
        out += s * q[kk]; 
        off += loop_stride; // (1024) for (8,128) blocks
    }
    scratch[lane] = out;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Same reduction as above
    if (part == 0) {
        float total = scratch[0 * 128 + vv] + scratch[1 * 128 + vv] + 
                      scratch[2 * 128 + vv] + scratch[3 * 128 + vv] +
                      scratch[4 * 128 + vv] + scratch[5 * 128 + vv] + 
                      scratch[6 * 128 + vv] + scratch[7 * 128 + vv];
                      
        output[qkv_offset(b, t, h, vv, seq_len, num_heads)] = half(total);
    }
    
}

[[max_total_threads_per_threadgroup(128)]]
kernel void delta_rule_prefill(device half* output [[buffer(0)]],
                               device float* state [[buffer(1)]], device const half* query [[buffer(2)]],
                               device const half* key [[buffer(3)]], device const half* value [[buffer(4)]],
                               device const float* g [[buffer(5)]], device const half* beta [[buffer(6)]],
                               constant long& batch_size [[buffer(7)]], constant long& seq_len [[buffer(8)]],
                               constant long& num_heads [[buffer(9)]], constant long& vs0 [[buffer(10)]],
                               constant long& vs1 [[buffer(11)]], constant long& vs2 [[buffer(12)]],
                               constant long& vs3 [[buffer(13)]], constant bool& has_initial_state [[buffer(14)]],
                               uint3 gid [[thread_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                               uint simd_group [[simdgroup_index_in_threadgroup]],
                               uint3 lane3 [[thread_position_in_threadgroup]], uint3 group3 [[threadgroup_position_in_grid]]) {
    uint lane = lane3.x;
    long group = group3.x;
    if (group >= batch_size * num_heads) return;
    long b = group / num_heads, h = group - b * num_heads;
    threadgroup float q[D], k[D], delta[D], scratch[D];
    if (!has_initial_state) {
        for (uint i = lane; i < D * D; i += D)
            state[state_offset(b, h, i / D, i % D, num_heads)] = 0.0f;
        threadgroup_barrier(mem_flags::mem_device);
    }
    for (long chunk = 0; chunk < seq_len; chunk += 64) {
        long end = min(chunk + 64, seq_len);
        for (long t = chunk; t < end; ++t) {
            float qv = float(query[qkv_offset(b, t, h, lane, seq_len, num_heads)]);
            float kv = float(key[qkv_offset(b, t, h, lane, seq_len, num_heads)]);
            q[lane] = qv; k[lane] = kv;
            float q_partial = simd_sum(qv * qv), k_partial = simd_sum(kv * kv);
            if (simd_lane == 0) {
                scratch[simd_group] = q_partial;
                scratch[D / 32 + simd_group] = k_partial;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                float q_total = simd_sum(simd_lane < D / 32 ? scratch[simd_lane] : 0.0f);
                float k_total = simd_sum(simd_lane < D / 32 ? scratch[D / 32 + simd_lane] : 0.0f);
                if (simd_lane == 0) {
                    scratch[0] = q_total;
                    scratch[1] = k_total;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float q_norm = rsqrt(scratch[0] + 1.0e-6f) * 0.08838834764831845f;
            float k_norm = rsqrt(scratch[1] + 1.0e-6f);
            q[lane] *= q_norm; k[lane] *= k_norm;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            long base_offset = state_offset(b, h, 0, lane, num_heads);
            float prediction = 0.0f, decay = exp(g[(b * seq_len + t) * num_heads + h]);
            for (uint kk = 0; kk < D; ++kk) {
                long off = base_offset + kk * D;
                float s = state[off] * decay;
                state[off] = s;
                prediction += s * k[kk];
            }
            delta[lane] = (float(value[value_offset(b, t, h, lane, vs0, vs1, vs2, vs3)]) - prediction) * float(beta[(b * seq_len + t) * num_heads + h]);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float out = 0.0f, cached_delta = delta[lane];
            for (uint kk = 0; kk < D; ++kk) {
                long off = base_offset + kk * D;
                float s = state[off] + k[kk] * cached_delta;
                state[off] = s;
                out += s * q[kk];
            }
            output[qkv_offset(b, t, h, lane, seq_len, num_heads)] = half(out);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// One threadgroup per (B, n_heads)
[[max_total_threads_per_threadgroup(1024)]]
kernel void delta_rule_decode(device half* output [[buffer(0)]],
                              device float* state [[buffer(1)]], device const half* query [[buffer(2)]],
                              device const half* key [[buffer(3)]], device const half* value [[buffer(4)]],
                              device const float* g [[buffer(5)]], device const half* beta [[buffer(6)]],
                              constant long& batch_size [[buffer(7)]], constant long& seq_len [[buffer(8)]],
                              constant long& num_heads [[buffer(9)]], constant long& vs0 [[buffer(10)]],
                              constant long& vs1 [[buffer(11)]], constant long& vs2 [[buffer(12)]],
                              constant long& vs3 [[buffer(13)]], constant bool& has_initial_state [[buffer(14)]],
                              uint3 gid [[thread_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                              uint simd_group [[simdgroup_index_in_threadgroup]],
                              uint3 lane3 [[thread_position_in_threadgroup]], uint3 group3 [[threadgroup_position_in_grid]]) {
    uint lane = lane3.x;
    long group = group3.x;
    if (group >= batch_size * num_heads) return;
    long b = group / num_heads, h = group - b * num_heads;
    threadgroup float q[D], k[D], delta[D], scratch[1024];
    // No divergence: all threads evaluate to same (no risk in barrier inside if)
    if (!has_initial_state) {
        for (uint i = lane; i < D * D; i += 1024)
            state[state_offset(b, h, i / D, i % D, num_heads)] = 0.0f;
        threadgroup_barrier(mem_flags::mem_device);
    }
    run_delta_rule_token(output, state, query, key, value, g, beta, b, 0, h, seq_len, num_heads, vs0, vs1, vs2, vs3, lane, simd_lane, simd_group, q, k, delta, scratch);
}
