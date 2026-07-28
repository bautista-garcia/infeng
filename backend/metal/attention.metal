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

[[max_total_threads_per_threadgroup(128)]]
kernel void attention_decode(
        device half* output [[buffer(0)]], device const half* query [[buffer(1)]],
        device const half* current_k [[buffer(2)]], device const half* current_v [[buffer(3)]],
        device half* cache_k [[buffer(4)]], device half* cache_v [[buffer(5)]],
        constant long& batch_size [[buffer(6)]], constant long& context_length [[buffer(7)]],
        constant long& cache_capacity [[buffer(8)]],
        uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
        uint3 group3 [[threadgroup_position_in_grid]]) {
    uint group = group3.x, h = group % Q_HEADS, b = group / Q_HEADS, kvh = h / GROUPS;
    if (b >= uint(batch_size)) return;
    uint old_length = uint(context_length), total_length = old_length + 1;
    uint q_offset = (b * Q_HEADS + h) * HEAD_DIM;
    threadgroup half q_shared[HEAD_DIM];
    threadgroup float partial[SCORE_GROUPS][HEAD_DIM];
    threadgroup float local_max[SCORE_GROUPS], local_norm[SCORE_GROUPS], global_max, global_norm;

    if (simd_group == 0) {
        for (uint d = simd_lane; d < HEAD_DIM; d += 32) q_shared[d] = query[q_offset + d];
        if (h % GROUPS == 0) {
            uint cache_offset = ((b * KV_HEADS + kvh) * uint(cache_capacity) + old_length) * HEAD_DIM;
            uint current_offset = (b * KV_HEADS + kvh) * HEAD_DIM;
            for (uint d = simd_lane; d < HEAD_DIM; d += 32) {
                cache_k[cache_offset + d] = current_k[current_offset + d];
                cache_v[cache_offset + d] = current_v[current_offset + d];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max_value = -1.0e30f, norm = 0.0f, acc[HEAD_DIM / 32] = {0.0f};
    for (uint token = simd_group; token < total_length; token += SCORE_GROUPS) {
        bool current = token == old_length;
        uint kv_offset = current
            ? (b * KV_HEADS + kvh) * HEAD_DIM
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
