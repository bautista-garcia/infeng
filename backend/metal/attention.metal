#include <metal_stdlib>
using namespace metal;

constant uint Q_HEADS = 16;
constant uint KV_HEADS = 4;
constant uint GROUPS = Q_HEADS / KV_HEADS;
constant uint HEAD_DIM = 256;
constant uint SCORE_GROUPS = 4;
constant uint MAX_CONTEXT = 4096;

[[max_total_threads_per_threadgroup(128)]]
kernel void attention_decode(
        device half* output [[buffer(0)]], device const half* query [[buffer(1)]],
        device const half* current_k [[buffer(2)]], device const half* current_v [[buffer(3)]],
        device half* cache_k [[buffer(4)]], device half* cache_v [[buffer(5)]],
        constant long& batch_size [[buffer(6)]], constant long& context_length [[buffer(7)]],
        constant long& cache_capacity [[buffer(8)]],
        uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
        uint3 group3 [[threadgroup_position_in_grid]]) {
    uint group = group3.x;
    if (group >= uint(batch_size) * Q_HEADS) return;
    uint b = group / Q_HEADS, h = group % Q_HEADS, kvh = h / GROUPS;
    uint old_length = uint(context_length), total_length = old_length + 1;
    uint q_offset = (b * Q_HEADS + h) * HEAD_DIM;
    threadgroup half q_shared[HEAD_DIM];
    threadgroup float scores[MAX_CONTEXT], partial[SCORE_GROUPS][HEAD_DIM];
    threadgroup float local_max[SCORE_GROUPS], local_norm[SCORE_GROUPS], global_max, global_norm;

    for (uint d = simd_lane; d < HEAD_DIM; d += 32) q_shared[d] = query[q_offset + d];
    if (simd_group == 0 && h % GROUPS == 0) {
        uint cache_offset = ((b * KV_HEADS + kvh) * uint(cache_capacity) + old_length) * HEAD_DIM;
        uint current_offset = (b * KV_HEADS + kvh) * HEAD_DIM;
        for (uint d = simd_lane; d < HEAD_DIM; d += 32) {
            cache_k[cache_offset + d] = current_k[current_offset + d];
            cache_v[cache_offset + d] = current_v[current_offset + d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint token = simd_group; token < total_length; token += SCORE_GROUPS) {
        bool is_current = token == old_length;
        uint kv_offset = is_current ? (b * KV_HEADS + kvh) * HEAD_DIM
                                    : ((b * KV_HEADS + kvh) * uint(cache_capacity) + token) * HEAD_DIM;
        float dot_qk = 0.0f;
        for (uint d = simd_lane; d < HEAD_DIM; d += 32) {
            float k = float(is_current ? current_k[kv_offset + d] : cache_k[kv_offset + d]);
            dot_qk = fma(float(q_shared[d]), k, dot_qk);
        }
        dot_qk = simd_sum(dot_qk) * (1.0f / 16.0f);
        if (simd_lane == 0) scores[token] = dot_qk;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane == 0) {
        float value = -1.0e30f;
        for (uint token = simd_group; token < total_length; token += SCORE_GROUPS) value = max(value, scores[token]);
        local_max[simd_group] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0 && simd_lane == 0) {
        float value = local_max[0];
        for (uint s = 1; s < SCORE_GROUPS; ++s) value = max(value, local_max[s]);
        global_max = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane == 0) {
        float value = 0.0f;
        for (uint token = simd_group; token < total_length; token += SCORE_GROUPS) {
            float weight = exp(scores[token] - global_max);
            scores[token] = weight;
            value += weight;
        }
        local_norm[simd_group] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0 && simd_lane == 0) {
        global_norm = 0.0f;
        for (uint s = 0; s < SCORE_GROUPS; ++s) global_norm += local_norm[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = simd_lane; d < HEAD_DIM; d += 32) {
        float acc = 0.0f;
        for (uint token = simd_group; token < total_length; token += SCORE_GROUPS) {
            bool is_current = token == old_length;
            uint kv_offset = is_current ? (b * KV_HEADS + kvh) * HEAD_DIM
                                        : ((b * KV_HEADS + kvh) * uint(cache_capacity) + token) * HEAD_DIM;
            float v = float(is_current ? current_v[kv_offset + d] : cache_v[kv_offset + d]);
            acc = fma(scores[token], v, acc);
        }
        partial[simd_group][d] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint output_offset = (b * Q_HEADS + h) * HEAD_DIM;
        for (uint d = simd_lane; d < HEAD_DIM; d += 32) {
            float acc = partial[0][d] + partial[1][d] + partial[2][d] + partial[3][d];
            output[output_offset + d] = half(acc / global_norm);
        }
    }
}
