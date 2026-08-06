#include <metal_stdlib>
using namespace metal;

constant uint Q_HEADS = 16;
constant uint KV_HEADS = 4;
constant uint Q_HEADS_PER_KV_HEAD = Q_HEADS / KV_HEADS;
constant uint HEAD_DIM = 256;
constant uint SIMDGROUPS_PER_THREADGROUP = 4;

static inline uint kv_offset(uint head, uint token) {
    return (((token >> 7) * KV_HEADS + head) * 128 + (token & 127)) * HEAD_DIM;
}

[[max_total_threads_per_threadgroup(128)]]
kernel void attention_prefill(
        device half* output [[buffer(0)]], device const half* q [[buffer(1)]],
        device const half* k [[buffer(2)]], device const half* v [[buffer(3)]],
        device half* cache_k [[buffer(4)]], device half* cache_v [[buffer(5)]],
        constant long& batch_size [[buffer(6)]], constant long& seq_len [[buffer(7)]],
        constant long& context_length [[buffer(8)]], constant long& cache_capacity [[buffer(9)]],
        uint simd_lane [[thread_index_in_simdgroup]], uint simd_index [[simdgroup_index_in_threadgroup]],
        uint3 threadgroup_position [[threadgroup_position_in_grid]]) {
    uint q_head = threadgroup_position.x % Q_HEADS, row = threadgroup_position.x / Q_HEADS;
    uint q_position = row % uint(seq_len), batch = row / uint(seq_len);
    uint kv_head = q_head / Q_HEADS_PER_KV_HEAD;
    if (batch >= uint(batch_size)) return;
    uint cached_tokens = uint(context_length), attended_tokens = cached_tokens + q_position + 1;
    uint q_offset = ((batch * uint(seq_len) + q_position) * Q_HEADS + q_head) * HEAD_DIM;
    threadgroup half q_shared[HEAD_DIM];
    threadgroup float simd_numerator[SIMDGROUPS_PER_THREADGROUP][HEAD_DIM];
    threadgroup float simd_max[SIMDGROUPS_PER_THREADGROUP], simd_norm[SIMDGROUPS_PER_THREADGROUP];
    threadgroup float attention_max, attention_norm;

    if (simd_index == 0) {
        for (uint dim = simd_lane; dim < HEAD_DIM; dim += 32) q_shared[dim] = q[q_offset + dim];
        if (q_head % Q_HEADS_PER_KV_HEAD == 0) {
            uint cache_offset = kv_offset(kv_head, cached_tokens + q_position);
            uint token_offset = ((batch * uint(seq_len) + q_position) * KV_HEADS + kv_head) * HEAD_DIM;
            for (uint dim = simd_lane; dim < HEAD_DIM; dim += 32) {
                cache_k[cache_offset + dim] = k[token_offset + dim];
                cache_v[cache_offset + dim] = v[token_offset + dim];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float running_max = -1.0e30f, running_norm = 0.0f, numerator[HEAD_DIM / 32] = {0.0f};
    for (uint token = simd_index; token < attended_tokens; token += SIMDGROUPS_PER_THREADGROUP) {
        bool current_chunk = token >= cached_tokens;
        uint item_offset = current_chunk
            ? ((batch * uint(seq_len) + token - cached_tokens) * KV_HEADS + kv_head) * HEAD_DIM
            : kv_offset(kv_head, token);
        float score = 0.0f;
        for (uint dim = simd_lane; dim < HEAD_DIM; dim += 32)
            score = fma(float(q_shared[dim]), float(current_chunk ? k[item_offset + dim] : cache_k[item_offset + dim]),
                        score);
        score = simd_sum(score) * (1.0f / 16.0f);
        float next_max = max(running_max, score), scale = exp(running_max - next_max);
        float weight = exp(score - next_max);
        running_norm = running_norm * scale + weight;
        for (uint dim = simd_lane, i = 0; dim < HEAD_DIM; dim += 32, ++i) {
            float value = float(current_chunk ? v[item_offset + dim] : cache_v[item_offset + dim]);
            numerator[i] = fma(numerator[i], scale, weight * value);
        }
        running_max = next_max;
    }
    if (simd_lane == 0) {
        simd_max[simd_index] = running_max;
        simd_norm[simd_index] = running_norm;
    }
    for (uint dim = simd_lane, i = 0; dim < HEAD_DIM; dim += 32, ++i)
        simd_numerator[simd_index][dim] = numerator[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_index == 0 && simd_lane == 0) {
        attention_max = simd_max[0];
        for (uint s = 1; s < SIMDGROUPS_PER_THREADGROUP; ++s) attention_max = max(attention_max, simd_max[s]);
        attention_norm = 0.0f;
        for (uint s = 0; s < SIMDGROUPS_PER_THREADGROUP; ++s)
            attention_norm += simd_norm[s] * exp(simd_max[s] - attention_max);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_index == 0) {
        for (uint dim = simd_lane; dim < HEAD_DIM; dim += 32) {
            float value = 0.0f;
            for (uint s = 0; s < SIMDGROUPS_PER_THREADGROUP; ++s)
                value += simd_numerator[s][dim] * exp(simd_max[s] - attention_max);
            output[q_offset + dim] = half(value / attention_norm);
        }
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void attention_decode_scan(
        device float* split_partials [[buffer(0)]], device const half* projected_q_and_gate [[buffer(1)]],
        device const half* projected_k [[buffer(2)]], device half* cache_k [[buffer(3)]],
        device const half* cache_v [[buffer(4)]], device const float* q_norm_weight [[buffer(5)]],
        device const float* k_norm_weight [[buffer(6)]], device const half2* rope_table [[buffer(7)]],
        constant uint& current_position [[buffer(8)]], constant uint& cache_capacity [[buffer(9)]],
        constant uint& num_splits [[buffer(10)]], uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_index [[simdgroup_index_in_threadgroup]],
        uint3 threadgroup_position [[threadgroup_position_in_grid]]) {
    uint q_head = threadgroup_position.x / num_splits, split = threadgroup_position.x % num_splits;
    uint kv_head = q_head / Q_HEADS_PER_KV_HEAD, attended_tokens = current_position + 1;
    uint split_begin = (attended_tokens * split) / num_splits;
    uint split_end = (attended_tokens * (split + 1)) / num_splits;
    threadgroup half q[HEAD_DIM], k[HEAD_DIM];
    threadgroup float simd_numerator[SIMDGROUPS_PER_THREADGROUP][HEAD_DIM];
    threadgroup float simd_max[SIMDGROUPS_PER_THREADGROUP], simd_norm[SIMDGROUPS_PER_THREADGROUP];
    threadgroup float split_max, split_norm;

    if (simd_index < 2) {
        // simd 0 does (q,gate) and simd 1 does k
        device const half* projected = simd_index == 0
            ? projected_q_and_gate + q_head * HEAD_DIM * 2
            : projected_k + kv_head * HEAD_DIM;
        device const float* norm_weight = simd_index == 0 ? q_norm_weight : k_norm_weight;
        threadgroup half* prepared = simd_index == 0 ? q : k;
        float sum_squared = 0.0f;
        // compute norm factors
        for (uint dim = simd_lane; dim < HEAD_DIM; dim += 32)
            sum_squared = fma(float(projected[dim]), float(projected[dim]), sum_squared);
        float inv_rms = rsqrt(simd_sum(sum_squared) / float(HEAD_DIM) + 1.0e-6f);
        // rmsnorm(q or k)
        for (uint dim = simd_lane; dim < HEAD_DIM; dim += 32)
            prepared[dim] = half(float(projected[dim]) * inv_rms * norm_weight[dim]);
        // rope(q or k)
        half2 cos_sin = rope_table[current_position * 32 + simd_lane];
        float x0 = float(prepared[simd_lane]), x1 = float(prepared[simd_lane + 32]);
        prepared[simd_lane] = half(fma(x0, float(cos_sin.x), -x1 * float(cos_sin.y)));
        prepared[simd_lane + 32] = half(fma(x1, float(cos_sin.x), x0 * float(cos_sin.y)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float running_max = -INFINITY, running_norm = 0.0f, numerator[HEAD_DIM / 32] = {0.0f};
    for (uint token = split_begin + simd_index; token < split_end; token += SIMDGROUPS_PER_THREADGROUP) {
        uint cache_offset = kv_offset(kv_head, token);
        // calculate score:  Q @ K [threadgroup_id * T/8 + (simd_id mod SIMDGROUPS_PER_THREADGROUP)]
        float score = 0.0f;
        for (uint dim = simd_lane; dim < HEAD_DIM; dim += 32) {
            float key = token == current_position ? float(k[dim]) : float(cache_k[cache_offset + dim]);
            score = fma(float(q[dim]), key, score);
        }
        score = simd_sum(score) * (1.0f / 16.0f);
        // online softmax with the scores each simd owns (T/8)/SIMDGROUPS_PER_THREADGROUP
        float next_max = max(running_max, score), scale = exp(running_max - next_max);
        float weight = exp(score - next_max);
        running_norm = running_norm * scale + weight;
        // online_softmax(QK) @ V
        for (uint dim = simd_lane, i = 0; dim < HEAD_DIM; dim += 32, ++i)
            numerator[i] = fma(numerator[i], scale, weight * float(cache_v[cache_offset + dim]));
        running_max = next_max;
    }
    // write the running max and norm for each simdgroup
    if (simd_lane == 0) {
        simd_max[simd_index] = running_max;
        simd_norm[simd_index] = running_norm;
    }
    // cooperative write of simd max numerators (you need to adjust them with threadgroup's max)
    for (uint dim = simd_lane, i = 0; dim < HEAD_DIM; dim += 32, ++i)
        simd_numerator[simd_index][dim] = numerator[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_index == 0 && simd_lane == 0) {
        // find threadgroups max -> max(ms0, ms1, ms2, ms3)
        split_max = simd_max[0];
        for (uint s = 1; s < SIMDGROUPS_PER_THREADGROUP; ++s) split_max = max(split_max, simd_max[s]);
        // merge simd denominators (+correction -> inter simd)
        split_norm = 0.0f;
        for (uint s = 0; s < SIMDGROUPS_PER_THREADGROUP; ++s)
            split_norm += simd_norm[s] * exp(simd_max[s] - split_max);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint split_offset = (q_head * num_splits + split) * (HEAD_DIM + 2);
    if (simd_index == 0) {
        for (uint dim = simd_lane; dim < HEAD_DIM; dim += 32) {
            // merge simd numerators
            float split_numerator = 0.0f;
            for (uint s = 0; s < SIMDGROUPS_PER_THREADGROUP; ++s)
                split_numerator += simd_numerator[s][dim] * exp(simd_max[s] - split_max);
            split_partials[split_offset + dim] = split_numerator;
        }
        if (simd_lane == 0) {
            split_partials[split_offset + HEAD_DIM] = split_max;
            split_partials[split_offset + HEAD_DIM + 1] = split_norm;
        }
    }
    if (split == 0 && q_head % Q_HEADS_PER_KV_HEAD == 0 && simd_index == 0)
        for (uint dim = simd_lane; dim < HEAD_DIM; dim += 32)
            cache_k[kv_offset(kv_head, current_position) + dim] = k[dim];
}

[[max_total_threads_per_threadgroup(128)]]
kernel void attention_decode_reduce(
        device half* gated_attention [[buffer(0)]], device const float* split_partials [[buffer(1)]],
        device const half* projected_q_and_gate [[buffer(2)]], constant uint& num_splits [[buffer(3)]],
        uint simd_lane [[thread_index_in_simdgroup]], uint simd_index [[simdgroup_index_in_threadgroup]],
        uint3 threadgroup_position [[threadgroup_position_in_grid]]) {
    uint q_head = threadgroup_position.x;
    threadgroup float split_max[16], attention_max, attention_norm;
    if (simd_index == 0 && simd_lane < num_splits)
        split_max[simd_lane] = split_partials[(q_head * num_splits + simd_lane) * (HEAD_DIM + 2) + HEAD_DIM];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_index == 0 && simd_lane == 0) {
        attention_max = split_max[0];
        for (uint split = 1; split < num_splits; ++split)
            attention_max = max(attention_max, split_max[split]);
        attention_norm = 0.0f;
        for (uint split = 0; split < num_splits; ++split) {
            uint split_offset = (q_head * num_splits + split) * (HEAD_DIM + 2);
            // merge denominators (+ correction -> interthreadgroup)
            attention_norm += split_partials[split_offset + HEAD_DIM + 1]
                              * exp(split_max[split] - attention_max);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // correction to numerators of each query head
    for (uint dim = simd_index * 32 + simd_lane; dim < HEAD_DIM; dim += 128) {
        float numerator = 0.0f;
        for (uint split = 0; split < num_splits; ++split) {
            uint split_offset = (q_head * num_splits + split) * (HEAD_DIM + 2);
            numerator += split_partials[split_offset + dim] * exp(split_max[split] - attention_max);
        }
        // gate and write (output projection is a standard quant decode)
        float gate = float(projected_q_and_gate[q_head * HEAD_DIM * 2 + HEAD_DIM + dim]);
        gated_attention[q_head * HEAD_DIM + dim] = half((numerator / attention_norm) / (1.0f + exp(-gate)));
    }
}
