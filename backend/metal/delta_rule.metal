#include <metal_stdlib>
#include <metal_matrix>
using namespace metal;

constant int D = 128;
constant int C_PREFILL = 64;
constant int D_PREFILL_TILE = 16;

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
    long chunk = long(group3.x) * C_PREFILL, h = group3.y % num_heads, b = group3.y / num_heads;
    if (b >= batch_size || h >= num_heads || chunk >= seq_len || lane >= 128) return;
    long C = min(long(C_PREFILL), seq_len - chunk);
    threadgroup half k_tile[C_PREFILL * D_PREFILL_TILE], w_tile[C_PREFILL * D_PREFILL_TILE], u_tile[C_PREFILL * D_PREFILL_TILE];
    threadgroup half l_tile[C_PREFILL * C_PREFILL], qk_tile[C_PREFILL * C_PREFILL];
    threadgroup float gamma[C_PREFILL], beta_tile[C_PREFILL], q_norm[C_PREFILL], k_norm[C_PREFILL], scratch[256];

    if (lane < C_PREFILL) {
        float x = lane < C ? exp(g[(b * seq_len + chunk + lane) * num_heads + h]) : 1.0f;
        for (uint o = 1; o < 32; o <<= 1) { float y = simd_shuffle_up(x, o); if (simd_lane >= o) x *= y; }
        gamma[lane] = x;
        beta_tile[lane] = lane < C ? float(beta[(b * seq_len + chunk + lane) * num_heads + h]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 1) gamma[lane] *= gamma[31];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        uint t = lane & 63, p = lane >> 6;
        float qs = 0.0f, ks = 0.0f;
        for (uint d = p; d < D; d += 2) {
            float qv = t < C ? float(query[qkv_offset(b, chunk + t, h, d, seq_len, num_heads)]) : 0.0f;
            float kv = t < C ? float(key[qkv_offset(b, chunk + t, h, d, seq_len, num_heads)]) : 0.0f;
            qs += qv * qv; ks += kv * kv;
        }
        scratch[p * C_PREFILL + t] = qs; scratch[128 + p * C_PREFILL + t] = ks;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < C_PREFILL) {
        q_norm[lane] = rsqrt(scratch[lane] + scratch[C_PREFILL + lane] + 1.0e-6f) * 0.08838834764831845f;
        k_norm[lane] = rsqrt(scratch[128 + lane] + scratch[128 + C_PREFILL + lane] + 1.0e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lane; idx < C_PREFILL * C_PREFILL; idx += 128) { l_tile[idx] = half(0.0f); qk_tile[idx] = half(0.0f); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d0 = 0; d0 < D; d0 += D_PREFILL_TILE) {
        for (uint idx = lane; idx < C_PREFILL * D_PREFILL_TILE; idx += 128) {
            uint i = idx / D_PREFILL_TILE, d = d0 + idx % D_PREFILL_TILE;
            k_tile[idx] = half(i < C ? float(key[qkv_offset(b, chunk + i, h, d, seq_len, num_heads)]) * k_norm[i] : 0.0f);
            u_tile[idx] = half(i < C ? float(query[qkv_offset(b, chunk + i, h, d, seq_len, num_heads)]) * q_norm[i] : 0.0f);
            w_tile[(idx % D_PREFILL_TILE) * C_PREFILL + i] = k_tile[idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ko = 0; ko < D_PREFILL_TILE; ko += 8) {
            for (uint block = simd_group; block < 64; block += 4) {
                uint ib = (block >> 3) << 3, jb = (block & 7) << 3, base = simd_group << 6;
                simdgroup_matrix<half, 8, 8> a, b;
                simdgroup_matrix<float, 8, 8> c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_load(a, k_tile, D_PREFILL_TILE, ulong2(ko, ib));
                simdgroup_load(b, w_tile, C_PREFILL, ulong2(jb, ko));
                simdgroup_multiply_accumulate(c, a, b, c);
                simdgroup_store(c, scratch + base, 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = simd_lane; e < 64; e += 32) {
                    uint r = e >> 3, col = e & 7, dst = (ib + r) * C_PREFILL + jb + col;
                    l_tile[dst] = half(float(l_tile[dst]) + scratch[base + e]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(a, u_tile, D_PREFILL_TILE, ulong2(ko, ib));
                c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_multiply_accumulate(c, a, b, c);
                simdgroup_store(c, scratch + base, 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = simd_lane; e < 64; e += 32) {
                    uint r = e >> 3, col = e & 7, dst = (ib + r) * C_PREFILL + jb + col;
                    qk_tile[dst] = half(float(qk_tile[dst]) + scratch[base + e]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint idx = lane; idx < C_PREFILL * C_PREFILL; idx += 128) {
        uint i = idx >> 6, j = idx & 63;
        l_tile[idx] = half(j < i && i < C && j < C ? -beta_tile[i] * float(l_tile[idx]) : 0.0f);
        qk_tile[idx] = half(j <= i && i < C && j < C ? gamma[i] / gamma[j] * float(qk_tile[idx]) : 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint v0 = 0; v0 < D; v0 += D_PREFILL_TILE) {
        for (uint idx = lane; idx < C_PREFILL * D_PREFILL_TILE; idx += 128) u_tile[idx] = half(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k0 = 0; k0 < D; k0 += D_PREFILL_TILE) {
            for (uint idx = lane; idx < C_PREFILL * D_PREFILL_TILE; idx += 128) {
                uint i = idx / D_PREFILL_TILE, kd = idx % D_PREFILL_TILE, k = k0 + kd;
                float kv = i < C ? float(key[qkv_offset(b, chunk + i, h, k, seq_len, num_heads)]) * k_norm[i] : 0.0f;
                w_tile[idx] = half(beta_tile[i] * kv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = 0; s < 32; ++s) {
                if (simd_group < 2) for (uint idx = lane; idx < 32 * D_PREFILL_TILE; idx += 64) {
                    uint i = idx / D_PREFILL_TILE, kd = idx % D_PREFILL_TILE, k = k0 + kd;
                    float kv = i < C ? float(key[qkv_offset(b, chunk + i, h, k, seq_len, num_heads)]) * k_norm[i] : 0.0f;
                    float acc = beta_tile[i] * kv;
                    for (uint j = 0; j < i; ++j) acc += float(l_tile[i * C_PREFILL + j]) * float(w_tile[j * D_PREFILL_TILE + kd]);
                    k_tile[idx] = half(acc);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group < 2) for (uint idx = lane; idx < 32 * D_PREFILL_TILE; idx += 64) w_tile[idx] = k_tile[idx];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint s = 0; s < 32; ++s) {
                if (simd_group >= 2) for (uint idx = lane - 64; idx < 32 * D_PREFILL_TILE; idx += 64) {
                    uint i = 32 + idx / D_PREFILL_TILE, kd = idx % D_PREFILL_TILE, k = k0 + kd;
                    float kv = i < C ? float(key[qkv_offset(b, chunk + i, h, k, seq_len, num_heads)]) * k_norm[i] : 0.0f;
                    float acc = beta_tile[i] * kv;
                    for (uint j = 0; j < 32; ++j) acc += float(l_tile[i * C_PREFILL + j]) * float(w_tile[j * D_PREFILL_TILE + kd]);
                    for (uint j = 32; j < i; ++j) acc += float(l_tile[i * C_PREFILL + j]) * float(w_tile[j * D_PREFILL_TILE + kd]);
                    k_tile[i * D_PREFILL_TILE + kd] = half(acc);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group >= 2) for (uint idx = lane - 64; idx < 32 * D_PREFILL_TILE; idx += 64) {
                    uint i = 32 + idx / D_PREFILL_TILE, kd = idx % D_PREFILL_TILE;
                    w_tile[i * D_PREFILL_TILE + kd] = k_tile[i * D_PREFILL_TILE + kd];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint idx = lane; idx < C_PREFILL * D_PREFILL_TILE; idx += 128) {
                uint i = idx / D_PREFILL_TILE, vd = idx % D_PREFILL_TILE, v = v0 + vd;
                float ws = float(u_tile[idx]);
                for (uint kd = 0; kd < D_PREFILL_TILE; ++kd) ws += float(w_tile[i * D_PREFILL_TILE + kd]) * (has_initial_state ? state[state_offset(b, h, k0 + kd, v, num_heads)] : 0.0f);
                u_tile[idx] = half(ws);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint idx = lane; idx < C_PREFILL * D_PREFILL_TILE; idx += 128) {
            uint i = idx / D_PREFILL_TILE, vd = idx % D_PREFILL_TILE, v = v0 + vd;
            w_tile[idx] = half(i < C ? beta_tile[i] * float(value[value_offset(b, chunk + i, h, v, vs0, vs1, vs2, vs3)]) : 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 0; s < 32; ++s) {
            if (simd_group < 2) for (uint idx = lane; idx < 32 * D_PREFILL_TILE; idx += 64) {
                uint i = idx / D_PREFILL_TILE, vd = idx % D_PREFILL_TILE, v = v0 + vd;
                float acc = i < C ? beta_tile[i] * float(value[value_offset(b, chunk + i, h, v, vs0, vs1, vs2, vs3)]) : 0.0f;
                for (uint j = 0; j < i; ++j) acc += float(l_tile[i * C_PREFILL + j]) * gamma[i] / gamma[j] * float(w_tile[j * D_PREFILL_TILE + vd]);
                k_tile[idx] = half(acc);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group < 2) for (uint idx = lane; idx < 32 * D_PREFILL_TILE; idx += 64) w_tile[idx] = k_tile[idx];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint s = 0; s < 32; ++s) {
            if (simd_group >= 2) for (uint idx = lane - 64; idx < 32 * D_PREFILL_TILE; idx += 64) {
                uint i = 32 + idx / D_PREFILL_TILE, vd = idx % D_PREFILL_TILE, v = v0 + vd;
                float acc = i < C ? beta_tile[i] * float(value[value_offset(b, chunk + i, h, v, vs0, vs1, vs2, vs3)]) : 0.0f;
                for (uint j = 0; j < 32; ++j) acc += float(l_tile[i * C_PREFILL + j]) * gamma[i] / gamma[j] * float(w_tile[j * D_PREFILL_TILE + vd]);
                for (uint j = 32; j < i; ++j) acc += float(l_tile[i * C_PREFILL + j]) * gamma[i] / gamma[j] * float(w_tile[j * D_PREFILL_TILE + vd]);
                k_tile[i * D_PREFILL_TILE + vd] = half(acc);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group >= 2) for (uint idx = lane - 64; idx < 32 * D_PREFILL_TILE; idx += 64) {
                uint i = 32 + idx / D_PREFILL_TILE, vd = idx % D_PREFILL_TILE;
                w_tile[i * D_PREFILL_TILE + vd] = k_tile[i * D_PREFILL_TILE + vd];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint idx = lane; idx < C_PREFILL * D_PREFILL_TILE; idx += 128) {
            uint i = idx / D_PREFILL_TILE;
            u_tile[idx] = half(float(w_tile[idx]) - gamma[i] * float(u_tile[idx]));
            k_tile[idx] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ko = 0; ko < C_PREFILL; ko += 8) {
            for (uint block = simd_group; block < 16; block += 4) {
                uint ib = (block >> 1) << 3, vb = (block & 1) << 3, base = simd_group << 6;
                simdgroup_matrix<half, 8, 8> a, dv;
                simdgroup_matrix<float, 8, 8> c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_load(a, qk_tile, C_PREFILL, ulong2(ko, ib));
                simdgroup_load(dv, u_tile, D_PREFILL_TILE, ulong2(vb, ko));
                simdgroup_multiply_accumulate(c, a, dv, c);
                simdgroup_store(c, scratch + base, 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = simd_lane; e < 64; e += 32) {
                    uint r = e >> 3, col = e & 7, dst = (ib + r) * D_PREFILL_TILE + vb + col;
                    k_tile[dst] = half(float(k_tile[dst]) + scratch[base + e]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        for (uint idx = lane; idx < C_PREFILL * D_PREFILL_TILE; idx += 128) {
            uint i = idx / D_PREFILL_TILE, vd = idx % D_PREFILL_TILE, v = v0 + vd;
            if (i < C) {
                float hist = 0.0f;
                for (uint k = 0; k < D; ++k) hist += float(query[qkv_offset(b, chunk + i, h, k, seq_len, num_heads)]) * q_norm[i] * (has_initial_state ? state[state_offset(b, h, k, v, num_heads)] : 0.0f);
                output[qkv_offset(b, chunk + i, h, v, seq_len, num_heads)] = half(gamma[i] * hist + float(k_tile[idx]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float gamma_c = gamma[C - 1];
        for (uint k0 = 0; k0 < D; k0 += D_PREFILL_TILE) {
            for (uint idx = lane; idx < C_PREFILL * D_PREFILL_TILE; idx += 128) {
                uint i = idx / D_PREFILL_TILE, kd = idx % D_PREFILL_TILE, k = k0 + kd;
                k_tile[idx] = half(i < C ? float(key[qkv_offset(b, chunk + i, h, k, seq_len, num_heads)]) * k_norm[i] : 0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = lane; idx < D_PREFILL_TILE * D_PREFILL_TILE; idx += 128) {
                uint kd = idx / D_PREFILL_TILE, vd = idx % D_PREFILL_TILE, k = k0 + kd, v = v0 + vd;
                float acc = gamma_c * (has_initial_state ? state[state_offset(b, h, k, v, num_heads)] : 0.0f);
                for (uint i = 0; i < C_PREFILL; ++i) {
                    acc += (i < C ? gamma_c / gamma[i] * float(u_tile[i * D_PREFILL_TILE + vd]) * float(k_tile[i * D_PREFILL_TILE + kd]) : 0.0f);
                }
                state[state_offset(b, h, k, v, num_heads)] = acc;
            }
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
