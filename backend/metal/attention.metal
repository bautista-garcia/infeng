#include <metal_stdlib>
using namespace metal;

constant uint Q_HEADS = 16;
constant uint KV_HEADS = 4;
constant uint GROUPS = Q_HEADS / KV_HEADS;
constant uint HEAD_DIM = 256;
constant uint SCORE_GROUPS = 4;

[[max_total_threads_per_threadgroup(128)]]
kernel void attention_prefill(
        device half* output [[buffer(0)]], device const half* query [[buffer(1)]],
        device const half* current_k [[buffer(2)]], device const half* current_v [[buffer(3)]],
        device half* cache_k [[buffer(4)]], device half* cache_v [[buffer(5)]],
        constant long& batch_size [[buffer(6)]], constant long& seq_len [[buffer(7)]],
        constant long& context_length [[buffer(8)]], constant long& cache_capacity [[buffer(9)]],
        uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
        uint3 group3 [[threadgroup_position_in_grid]]) {
    uint group = group3.x, h = group % Q_HEADS, row = group / Q_HEADS;
    uint qpos = row % uint(seq_len), b = row / uint(seq_len), kvh = h / GROUPS;
    if (b >= uint(batch_size)) return;
    uint old_length = uint(context_length), total_length = old_length + qpos + 1;
    uint q_offset = ((b * uint(seq_len) + qpos) * Q_HEADS + h) * HEAD_DIM;
    threadgroup half q_shared[HEAD_DIM];
    threadgroup float partial[SCORE_GROUPS][HEAD_DIM];
    threadgroup float local_max[SCORE_GROUPS], local_norm[SCORE_GROUPS], global_max, global_norm;

    if (simd_group == 0) {
        for (uint d = simd_lane; d < HEAD_DIM; d += 32) q_shared[d] = query[q_offset + d];
        if (h % GROUPS == 0) {
            uint cache_offset = ((b * KV_HEADS + kvh) * uint(cache_capacity) + old_length + qpos) * HEAD_DIM;
            uint current_offset = ((b * uint(seq_len) + qpos) * KV_HEADS + kvh) * HEAD_DIM;
            for (uint d = simd_lane; d < HEAD_DIM; d += 32) {
                cache_k[cache_offset + d] = current_k[current_offset + d];
                cache_v[cache_offset + d] = current_v[current_offset + d];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max_value = -1.0e30f, norm = 0.0f, acc[HEAD_DIM / 32] = {0.0f};
    for (uint token = simd_group; token < total_length; token += SCORE_GROUPS) {
        bool current = token >= old_length;
        uint kv_offset = current
            ? ((b * uint(seq_len) + token - old_length) * KV_HEADS + kvh) * HEAD_DIM
            : ((b * KV_HEADS + kvh) * uint(cache_capacity) + token) * HEAD_DIM;
        float dot_qk = 0.0f;
        for (uint d = simd_lane; d < HEAD_DIM; d += 32)
            dot_qk = fma(float(q_shared[d]), float(current ? current_k[kv_offset + d] : cache_k[kv_offset + d]),
                         dot_qk);
        dot_qk = simd_sum(dot_qk) * (1.0f / 16.0f);
        float next_max = max(max_value, dot_qk), scale = exp(max_value - next_max), weight = exp(dot_qk - next_max);
        norm = norm * scale + weight;
        for (uint d = simd_lane, i = 0; d < HEAD_DIM; d += 32, ++i) {
            float v = float(current ? current_v[kv_offset + d] : cache_v[kv_offset + d]);
            acc[i] = fma(acc[i], scale, weight * v);
        }
        max_value = next_max;
    }
    if (simd_lane == 0) {
        local_max[simd_group] = max_value;
        local_norm[simd_group] = norm;
    }
    for (uint d = simd_lane, i = 0; d < HEAD_DIM; d += 32, ++i) partial[simd_group][d] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane == 0) {
        global_max = local_max[0];
        for (uint s = 1; s < SCORE_GROUPS; ++s) global_max = max(global_max, local_max[s]);
        global_norm = 0.0f;
        for (uint s = 0; s < SCORE_GROUPS; ++s) global_norm += local_norm[s] * exp(local_max[s] - global_max);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        for (uint d = simd_lane; d < HEAD_DIM; d += 32) {
            float value = 0.0f;
            for (uint s = 0; s < SCORE_GROUPS; ++s)
                value += partial[s][d] * exp(local_max[s] - global_max);
            output[q_offset + d] = half(value / global_norm);
        }
    }
}

static inline __attribute__((always_inline)) void prepare_decode_qk(
        device const half* qg, device const half* raw_k, device half* cache_k,
        device const float* q_weight, device const float* k_weight, device const half2* rope,
        threadgroup half* q, threadgroup half* current_k, threadgroup float* inv_rms,
        uint h, uint kvh, uint context, uint capacity, bool write_k, uint lane, uint simd_group) {
    if (simd_group == 0) {
        float q2 = 0.0f, k2 = 0.0f;
        for (uint d = lane; d < HEAD_DIM; d += 32) {
            float qv = float(qg[h * HEAD_DIM * 2 + d]), kv = float(raw_k[kvh * HEAD_DIM + d]);
            q2 = fma(qv, qv, q2);
            k2 = fma(kv, kv, k2);
        }
        q2 = simd_sum(q2);
        k2 = simd_sum(k2);
        if (lane == 0) {
            inv_rms[0] = rsqrt(q2 / float(HEAD_DIM) + 1.0e-6f);
            inv_rms[1] = rsqrt(k2 / float(HEAD_DIM) + 1.0e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        for (uint d = lane; d < HEAD_DIM; d += 32) {
            q[d] = half(float(qg[h * HEAD_DIM * 2 + d]) * inv_rms[0] * q_weight[d]);
            current_k[d] = half(float(raw_k[kvh * HEAD_DIM + d]) * inv_rms[1] * k_weight[d]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0 && lane < 32) {
        half2 cs = rope[context * 32 + lane];
        float q0 = float(q[lane]), q1 = float(q[lane + 32]);
        float k0 = float(current_k[lane]), k1 = float(current_k[lane + 32]);
        q[lane] = half(fma(q0, float(cs.x), -q1 * float(cs.y)));
        q[lane + 32] = half(fma(q1, float(cs.x), q0 * float(cs.y)));
        current_k[lane] = half(fma(k0, float(cs.x), -k1 * float(cs.y)));
        current_k[lane + 32] = half(fma(k1, float(cs.x), k0 * float(cs.y)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (write_k && simd_group == 0)
        for (uint d = lane; d < HEAD_DIM; d += 32)
            cache_k[(kvh * capacity + context) * HEAD_DIM + d] = current_k[d];
}

[[max_total_threads_per_threadgroup(128)]]
kernel void attention_decode_scan(
        device float* output [[buffer(0)]], device const half* qg [[buffer(1)]],
        device const half* raw_k [[buffer(2)]], device half* cache_k [[buffer(3)]],
        device const half* cache_v [[buffer(4)]], device const float* q_weight [[buffer(5)]],
        device const float* k_weight [[buffer(6)]], device const half2* rope [[buffer(7)]],
        constant uint& context [[buffer(8)]], constant uint& capacity [[buffer(9)]],
        constant uint& splits [[buffer(10)]], uint lane [[thread_index_in_simdgroup]],
        uint simd_group [[simdgroup_index_in_threadgroup]],
        uint3 group3 [[threadgroup_position_in_grid]]) {
    uint h = group3.x / splits, split = group3.x % splits, kvh = h / GROUPS, total = context + 1;
    uint begin = (total * split) / splits, end = (total * (split + 1)) / splits;
    threadgroup half q[HEAD_DIM], current_k[HEAD_DIM];
    threadgroup float partial[SCORE_GROUPS][HEAD_DIM], local_max[SCORE_GROUPS], local_norm[SCORE_GROUPS];
    threadgroup float global_max, global_norm, inv_rms[2];
    prepare_decode_qk(qg, raw_k, cache_k, q_weight, k_weight, rope, q, current_k, inv_rms,
                      h, kvh, context, capacity, split == 0 && h % GROUPS == 0, lane, simd_group);

    float max_value = -INFINITY, norm = 0.0f, acc[HEAD_DIM / 32] = {0.0f};
    for (uint token = begin + simd_group; token < end; token += SCORE_GROUPS) {
        uint offset = (kvh * capacity + token) * HEAD_DIM;
        float dot_qk = 0.0f;
        for (uint d = lane; d < HEAD_DIM; d += 32) {
            float k = token == context ? float(current_k[d]) : float(cache_k[offset + d]);
            dot_qk = fma(float(q[d]), k, dot_qk);
        }
        dot_qk = simd_sum(dot_qk) * (1.0f / 16.0f);
        float next_max = max(max_value, dot_qk), scale = exp(max_value - next_max);
        float weight = exp(dot_qk - next_max);
        norm = norm * scale + weight;
        for (uint d = lane, i = 0; d < HEAD_DIM; d += 32, ++i)
            acc[i] = fma(acc[i], scale, weight * float(cache_v[offset + d]));
        max_value = next_max;
    }
    if (lane == 0) {
        local_max[simd_group] = max_value;
        local_norm[simd_group] = norm;
    }
    for (uint d = lane, i = 0; d < HEAD_DIM; d += 32, ++i) partial[simd_group][d] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0 && lane == 0) {
        global_max = local_max[0];
        for (uint s = 1; s < SCORE_GROUPS; ++s) global_max = max(global_max, local_max[s]);
        global_norm = 0.0f;
        for (uint s = 0; s < SCORE_GROUPS; ++s) global_norm += local_norm[s] * exp(local_max[s] - global_max);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = (h * splits + split) * (HEAD_DIM + 2);
    if (simd_group == 0) {
        for (uint d = lane; d < HEAD_DIM; d += 32) {
            float value = 0.0f;
            for (uint s = 0; s < SCORE_GROUPS; ++s) value += partial[s][d] * exp(local_max[s] - global_max);
            output[base + d] = value;
        }
        if (lane == 0) {
            output[base + HEAD_DIM] = global_max;
            output[base + HEAD_DIM + 1] = global_norm;
        }
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void attention_decode_reduce(
        device half* output [[buffer(0)]], device const float* partial [[buffer(1)]],
        device const half* qg [[buffer(2)]], constant uint& splits [[buffer(3)]],
        uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
        uint3 group3 [[threadgroup_position_in_grid]]) {
    uint h = group3.x;
    threadgroup float split_max[16], global_max, global_norm;
    if (simd_group == 0 && lane < splits)
        split_max[lane] = partial[(h * splits + lane) * (HEAD_DIM + 2) + HEAD_DIM];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0 && lane == 0) {
        global_max = split_max[0];
        for (uint s = 1; s < splits; ++s) global_max = max(global_max, split_max[s]);
        global_norm = 0.0f;
        for (uint s = 0; s < splits; ++s) {
            uint base = (h * splits + s) * (HEAD_DIM + 2);
            global_norm += partial[base + HEAD_DIM + 1] * exp(split_max[s] - global_max);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = simd_group * 32 + lane; d < HEAD_DIM; d += 128) {
        float value = 0.0f;
        for (uint s = 0; s < splits; ++s) {
            uint base = (h * splits + s) * (HEAD_DIM + 2);
            value += partial[base + d] * exp(split_max[s] - global_max);
        }
        float gate = float(qg[h * HEAD_DIM * 2 + HEAD_DIM + d]);
        output[h * HEAD_DIM + d] = half((value / global_norm) / (1.0f + exp(-gate)));
    }
}
