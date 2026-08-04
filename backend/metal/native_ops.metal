#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm(device half* y [[buffer(0)]], device const half* x [[buffer(1)]],
                    device const float* w [[buffer(2)]], constant uint& rows [[buffer(3)]],
                    constant uint& dim [[buffer(4)]], constant float& eps [[buffer(5)]],
                    uint lane [[thread_index_in_threadgroup]], uint2 pos [[threadgroup_position_in_grid]]) {
    uint row = pos.y;
    if (row >= rows) return;
    threadgroup float sums[256];
    float sum = 0.0f;
    for (uint d = lane; d < dim; d += 256) { float v = float(x[row * dim + d]); sum += v * v; }
    sums[lane] = sum; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s; s >>= 1) { if (lane < s) sums[lane] += sums[lane + s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float scale = rsqrt(sums[0] / float(dim) + eps);
    for (uint d = lane; d < dim; d += 256) y[row * dim + d] = half(float(x[row * dim + d]) * scale * w[d]);
}

kernel void add_half(device half* y [[buffer(0)]], device const half* a [[buffer(1)]],
                     device const half* b [[buffer(2)]], constant uint& n [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) { if (i < n) y[i] = a[i] + b[i]; }

kernel void silu_mul(device half* y [[buffer(0)]], device const half* gate [[buffer(1)]],
                     device const half* up [[buffer(2)]], constant uint& n [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) { float g = float(gate[i]); y[i] = half((g / (1.0f + exp(-g))) * float(up[i])); }
}

kernel void attention_gate(device half* y [[buffer(0)]], device const half* x [[buffer(1)]],
                           device const half* qg [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = half(float(x[i]) / (1.0f + exp(-float(qg[(i / 256) * 512 + 256 + i % 256]))));
}

kernel void dense_4096x32(device half* y [[buffer(0)]], device const half* x [[buffer(1)]],
                          device const half* w [[buffer(2)]], constant uint& rows [[buffer(3)]],
                          uint lane [[thread_index_in_simdgroup]], uint2 pos [[threadgroup_position_in_grid]]) {
    uint row = pos.y, out = pos.x; if (row >= rows || out >= 32) return;
    float sum = 0.0f;
    for (uint k = lane; k < 4096; k += 32) sum += float(x[row * 4096 + k]) * float(w[out * 4096 + k]);
    sum = simd_sum(sum);
    if (lane == 0) y[row * 32 + out] = half(sum);
}

kernel void gdn_prepare(device half* beta [[buffer(0)]], device float* g [[buffer(1)]],
                        device const half* b [[buffer(2)]], device const half* a [[buffer(3)]],
                        device const float* A [[buffer(4)]], device const float* dt [[buffer(5)]],
                        constant uint& n [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float av = float(a[i]) + dt[i % 32];
    beta[i] = half(1.0f / (1.0f + exp(-float(b[i]))));
    g[i] = -exp(A[i % 32]) * log(1.0f + exp(av));
}

kernel void split_repeat_qk(device half* q [[buffer(0)]], device half* k [[buffer(1)]],
                            device half* v [[buffer(2)]], device const half* mixed [[buffer(3)]],
                            constant uint& rows [[buffer(4)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= rows * 4096) return;
    uint row = i / 4096, d = i % 4096, source = (d % 2048);
    q[i] = mixed[row * 8192 + source]; k[i] = mixed[row * 8192 + 2048 + source];
    v[i] = mixed[row * 8192 + 4096 + d];
}

kernel void unpack_attention(device half* q [[buffer(0)]], device half* gate [[buffer(1)]],
                             device const half* qg [[buffer(2)]], constant uint& n [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= n) return; uint row = i >> 8, d = i & 255;
    q[i] = qg[row * 512 + d]; gate[i] = qg[row * 512 + 256 + d];
}

kernel void rope_qk(device half* qout [[buffer(0)]], device half* kout [[buffer(1)]],
                    device const half* q [[buffer(2)]], device const half* k [[buffer(3)]],
                    device const half2* rope [[buffer(4)]], constant uint& seq [[buffer(5)]],
                    constant uint& start [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    uint qn = seq * 16 * 256, kn = seq * 4 * 256;
    if (i < qn) {
        uint d = i & 255, row = i >> 8, t = row / 16, base = row << 8;
        if (d < 64) { uint pair = d & 31, other = d < 32 ? d + 32 : d - 32; half2 cs = rope[(start + t) * 32 + pair];
            float a = float(q[base + d]), b = float(q[base + other]); qout[i] = half(a * float(cs.x) + (d < 32 ? -b : b) * float(cs.y)); }
        else qout[i] = q[i];
    }
    if (i < kn) {
        uint d = i & 255, row = i >> 8, t = row / 4, base = row << 8;
        if (d < 64) { uint pair = d & 31, other = d < 32 ? d + 32 : d - 32; half2 cs = rope[(start + t) * 32 + pair];
            float a = float(k[base + d]), b = float(k[base + other]); kout[i] = half(a * float(cs.x) + (d < 32 ? -b : b) * float(cs.y)); }
        else kout[i] = k[i];
    }
}

kernel void init_rope(device half2* rope [[buffer(0)]], constant uint& capacity [[buffer(1)]],
                      constant float& theta [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= capacity * 32) return; uint p = i / 32, d = i % 32;
    float angle = float(p) / pow(theta, float(2 * d) / 64.0f); rope[i] = half2(half(cos(angle)), half(sin(angle)));
}

kernel void pad_rows(device half* y [[buffer(0)]], device const half* x [[buffer(1)]],
                     constant uint& rows [[buffer(2)]], constant uint& padded [[buffer(3)]],
                     constant uint& dim [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i < padded * dim) y[i] = i < rows * dim ? x[i] : half(0.0);
}

kernel void argmax_logits(device uint* token [[buffer(0)]], device const half* logits [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float values[256]; threadgroup uint indices[256];
    float best = -INFINITY; uint index = 0;
    for (uint i = lane; i < n; i += 256) { float v = float(logits[i]); if (v > best) { best = v; index = i; } }
    values[lane] = best; indices[lane] = index; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s; s >>= 1) { if (lane < s && values[lane + s] > values[lane]) { values[lane] = values[lane + s]; indices[lane] = indices[lane + s]; } threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (lane == 0) token[0] = indices[0];
}

kernel void sample_logits(device int* token [[buffer(0)]], device ulong* rng [[buffer(1)]],
                          device const half* logits [[buffer(2)]], constant uint& n [[buffer(3)]],
                          constant float& temperature [[buffer(4)]], constant float& top_p [[buffer(5)]],
                          constant uint& top_k [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if (i) return;
    ulong state = rng[0]; state ^= state << 13; state ^= state >> 7; state ^= state << 17; rng[0] = state;
    float random = float(state >> 40) * (1.0f / 16777216.0f);
    if (!top_k) {
        float maximum = -INFINITY;
        for (uint j = 0; j < n; ++j) maximum = max(maximum, float(logits[j]) / temperature);
        float total = 0.0f;
        for (uint j = 0; j < n; ++j) total += exp(float(logits[j]) / temperature - maximum);
        float target = random * total, cumulative = 0.0f;
        for (uint j = 0; j < n; ++j) { cumulative += exp(float(logits[j]) / temperature - maximum);
            if (cumulative >= target) { token[0] = int(j); return; } }
        token[0] = int(n - 1); return;
    }
    float values[64]; uint indices[64], count = min(top_k, 64u);
    for (uint j = 0; j < count; ++j) { values[j] = -INFINITY; indices[j] = 0; }
    for (uint j = 0; j < n; ++j) {
        float value = float(logits[j]) / temperature;
        if (value <= values[count - 1]) continue;
        uint p = count - 1;
        while (p && value > values[p - 1]) { values[p] = values[p - 1]; indices[p] = indices[p - 1]; --p; }
        values[p] = value; indices[p] = j;
    }
    float total = 0.0f;
    for (uint j = 0; j < count; ++j) total += exp(values[j] - values[0]);
    float limit = top_p * total, nucleus = 0.0f; uint keep = 0;
    do { nucleus += exp(values[keep] - values[0]); ++keep; } while (keep < count && nucleus < limit);
    float target = random * nucleus, cumulative = 0.0f;
    for (uint j = 0; j < keep; ++j) { cumulative += exp(values[j] - values[0]);
        if (cumulative >= target) { token[0] = int(indices[j]); return; } }
    token[0] = int(indices[keep - 1]);
}
