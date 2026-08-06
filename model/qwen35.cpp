#include "qwen35.hpp"
#include <chrono>
#include <stdexcept>
#include <vector>
namespace infeng::qwen35 {
namespace {
constexpr uint64_t halfBytes = 2;
MTL::Size size(uint64_t x, uint64_t y = 1, uint64_t z = 1) { return MTL::Size(x, y, z); }
uint64_t roundUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }
Tensor linear(Pass& pass, const Tensor& x, const Linear& weight, uint32_t rows, Tensor output, Scratch& scratch,
              uint32_t mode = 0, const Tensor* aux = nullptr, uint32_t context = 0) {
    if (weight.type == QuantType::F16) {
        pass.dispatch(weight.decode, size(1024, rows), size(32), {output, x, weight.weight}, rows); return output;
    }
    if (rows == 1) {
        const Tensor& extra = aux ? *aux : output;
        pass.dispatch(weight.decode, size(((weight.n + weight.outputsPerGroup - 1) / weight.outputsPerGroup) * weight.decodeThreads),
                      size(weight.decodeThreads), {output, x, weight.weight, extra}, mode, context, uint32_t(0));
        return output;
    }
    if (mode) throw std::runtime_error("quant prefill has no fused store mode");
    uint32_t padded = roundUp(rows, 32); Tensor source = x, target = output;
    if (padded != rows) {
        source = scratch.padInput.view(0, uint64_t(padded) * weight.k * halfBytes);
        uint64_t total = uint64_t(padded) * weight.k;
        pass.dispatch(scratch.padRows,
                      size(roundUp(total, 256)), size(256), {source, x}, rows, padded, weight.k);
    }
    pass.dispatch(weight.prefill, size(128 * (weight.n / 16), padded / 32), size(128),
                  {target, source, weight.weight}, int64_t(padded));
    return target.view(0, uint64_t(rows) * weight.n * halfBytes);
}
Tensor rms(Pass& pass, Model& model, const Tensor& x, const Tensor& weight, Tensor output, uint32_t rows, uint32_t dim) {
    pass.dispatch(model.kernels.rms, size(256, rows), size(256), {output, x, weight}, rows, dim, 1e-6f); return output;
}
Tensor add(Pass& pass, Model& model, const Tensor& a, const Tensor& b, Tensor output, uint64_t elements) {
    pass.dispatch(model.kernels.add, size(roundUp(elements, 256)), size(256), {output, a, b}, uint32_t(elements)); return output;
}
Tensor mlp(Pass& pass, Model& model, const Tensor& x, const MlpWeights& weights, Tensor output,
           Scratch& scratch, uint32_t seq) {
    if (seq == 1) {
        pass.dispatch(weights.fusedDecode, size(12288 / weights.outputsPerGroup * weights.decodeThreads),
                      size(weights.decodeThreads), {scratch.mlpMixed, x, weights.gate.weight, weights.up.weight}, int64_t(1));
    } else {
        Tensor gate = linear(pass, x, weights.gate, seq, scratch.mlpGate, scratch);
        Tensor up = linear(pass, x, weights.up, seq, scratch.mlpUp, scratch); uint64_t elements = uint64_t(seq) * 12288;
        pass.dispatch(model.kernels.siluMul, size(roundUp(elements, 256)), size(256),
                      {scratch.mlpMixed, gate, up}, uint32_t(elements));
    }
    return linear(pass, scratch.mlpMixed, weights.down, seq, output, scratch);
}
Tensor attention(Pass& pass, Model& model, const Layer& layer, const Tensor& x, const Tensor& residual,
                 Tensor destination, Scratch& scratch, Session& session, uint32_t seq) {
    const AttentionWeights& weights = layer.attention; uint32_t start = session.length;
    Tensor cacheK = session.kv.key(layer.kvIndex), cacheV = session.kv.value(layer.kvIndex);
    if (seq == 1) {
        Tensor qg = linear(pass, x, weights.q, 1, scratch.attnQG, scratch);
        Tensor rawK = linear(pass, x, weights.k, 1, scratch.attnK, scratch);
        linear(pass, x, weights.v, 1, cacheV, scratch, 1, nullptr, start);
        uint32_t splits = std::min(8u, start + 1);
        pass.dispatch(model.kernels.attentionDecodeScan, size(16 * splits * 128), size(128),
                      {scratch.attnPartials, qg, rawK, cacheK, cacheV, weights.qNorm, weights.kNorm, model.rope},
                      start, model.maxContext, splits);
        pass.dispatch(model.kernels.attentionDecodeReduce, size(16 * 128), size(128),
                      {scratch.attnOut, scratch.attnPartials, qg}, splits);
        return linear(pass, scratch.attnOut, weights.out, 1, destination, scratch, 2, &residual);
    }
    Tensor qg = linear(pass, x, weights.q, seq, scratch.attnQG, scratch); uint64_t qElements = uint64_t(seq) * 4096;
    pass.dispatch(model.kernels.unpackAttention, size(roundUp(qElements, 256)), size(256),
                  {scratch.attnQ, scratch.attnGate, qg}, uint32_t(qElements));
    Tensor q = rms(pass, model, scratch.attnQ, weights.qNorm, scratch.attnQNorm, seq * 16, 256);
    Tensor k = linear(pass, x, weights.k, seq, scratch.attnK, scratch);
    k = rms(pass, model, k, weights.kNorm, scratch.attnKNorm, seq * 4, 256);
    Tensor v = linear(pass, x, weights.v, seq, scratch.attnV, scratch);
    pass.dispatch(model.kernels.ropeQk, size(roundUp(qElements, 256)), size(256),
                  {scratch.attnQRope, scratch.attnKRope, q, k, model.rope}, seq, start);
    pass.dispatch(model.kernels.attentionPrefill, size(uint64_t(seq) * 16 * 128), size(128),
                  {scratch.attnOut, scratch.attnQRope, scratch.attnKRope, v, cacheK, cacheV},
                  int64_t(1), int64_t(seq), int64_t(start), int64_t(model.maxContext));
    pass.dispatch(model.kernels.attentionGate, size(roundUp(qElements, 256)), size(256),
                  {scratch.attnGated, scratch.attnOut, qg}, uint32_t(qElements));
    Tensor projected = linear(pass, scratch.attnGated, weights.out, seq, scratch.projected, scratch);
    return add(pass, model, residual, projected, destination, qElements);
}
Tensor gdn(Pass& pass, Model& model, const Layer& layer, const Tensor& x, const Tensor& residual,
           Tensor destination, Scratch& scratch, LayerState& state, uint32_t seq) {
    const GdnWeights& weights = layer.gdn;
    Tensor mixed = linear(pass, x, weights.qkv, seq, scratch.gdnMixed, scratch);
    Tensor z = linear(pass, x, weights.z, seq, scratch.gdnZ, scratch);
    Tensor b = linear(pass, x, weights.b, seq, scratch.gdnB, scratch);
    Tensor a = linear(pass, x, weights.a, seq, scratch.gdnA, scratch); uint64_t heads = uint64_t(seq) * 32;
    pass.dispatch(model.kernels.gdnPrepare, size(roundUp(heads, 256)), size(256),
                  {scratch.gdnBeta, scratch.gdnG, b, a, weights.A, weights.dt}, uint32_t(heads));
    uint8_t slot = 1 - state.convSlot; const Tensor& previous = state.initialized ? state.conv[state.convSlot] : state.conv[slot];
    pass.dispatch(model.kernels.gdnConv, size(uint64_t(8192) * std::max(seq, 4u)), size(256),
                  {scratch.gdnConvolved, state.conv[slot], mixed, weights.conv, previous},
                  int64_t(1), int64_t(seq), state.initialized); state.convSlot = slot;
    uint64_t elements = uint64_t(seq) * 4096;
    pass.dispatch(model.kernels.splitQk, size(roundUp(elements, 256)), size(256),
                  {scratch.gdnQ, scratch.gdnK, scratch.gdnV, scratch.gdnConvolved}, seq);
    if (seq == 1) {
        pass.dispatch(model.kernels.deltaDecode, size(32 * 512), size(512),
                      {scratch.gdnDelta, state.recurrent, scratch.gdnQ, scratch.gdnK, scratch.gdnV,
                       scratch.gdnG, scratch.gdnBeta},
                      int64_t(1), int64_t(1), int64_t(32), int64_t(4096), int64_t(4096), int64_t(128),
                      int64_t(1), state.initialized);
    } else for (uint32_t start = 0; start < seq; start += 16) {
        uint32_t count = std::min(16u, seq - start); uint64_t halfOffset = uint64_t(start) * 4096 * halfBytes;
        pass.dispatch(model.kernels.deltaPrefill, size(128, 32), size(128),
                      {scratch.gdnDelta.view(halfOffset, uint64_t(count) * 4096 * halfBytes), state.recurrent,
                       scratch.gdnQ.view(halfOffset, uint64_t(count) * 4096 * halfBytes),
                       scratch.gdnK.view(halfOffset, uint64_t(count) * 4096 * halfBytes),
                       scratch.gdnV.view(halfOffset, uint64_t(count) * 4096 * halfBytes),
                       scratch.gdnG.view(uint64_t(start) * 32 * 4, uint64_t(count) * 32 * 4),
                       scratch.gdnBeta.view(uint64_t(start) * 32 * 2, uint64_t(count) * 32 * 2)},
                      int64_t(1), int64_t(count), int64_t(32), int64_t(count) * 4096, int64_t(4096),
                      int64_t(128), int64_t(1), state.initialized || start > 0);
    }
    state.initialized = true;
    pass.dispatch(model.kernels.gdnNorm, size(128, uint64_t(seq) * 32), size(128),
                  {scratch.gdnNormed, scratch.gdnDelta, z, weights.norm}, 1e-6f);
    Tensor projected = linear(pass, scratch.gdnNormed, weights.out, seq, scratch.projected, scratch);
    return add(pass, model, residual, projected, destination, elements);
}
Tensor decoderLayer(Pass& pass, Model& model, const Layer& layer, LayerState& state, const Tensor& hidden,
                    Tensor destination, Scratch& scratch, Session& session, uint32_t seq) {
    Tensor x = rms(pass, model, hidden, layer.inputNorm, scratch.inputNorm, seq, 4096);
    Tensor mid = layer.fullAttention ? attention(pass, model, layer, x, hidden, scratch.mid, scratch, session, seq)
                                     : gdn(pass, model, layer, x, hidden, scratch.mid, scratch, state, seq);
    x = rms(pass, model, mid, layer.postNorm, scratch.postNorm, seq, 4096);
    Tensor projected = mlp(pass, model, x, layer.mlp, scratch.projected, scratch, seq);
    return add(pass, model, mid, projected, destination, uint64_t(seq) * 4096);
}
void sample(Pass& pass, Model& model, Session& session, const Tensor& logits,
            float temperature, float topP, int32_t topK) {
    if (temperature <= 0) {
        pass.dispatch(model.kernels.argmax, size(256), size(256), {session.token, logits}, uint32_t(248320)); return;
    }
    if (topK < 0 || topK > 64) throw std::runtime_error("GPU top_k must be between 1 and 64");
    if (!topK && topP < 1) throw std::runtime_error("top_p below 1 requires top_k on this specialized runtime");
    pass.dispatch(model.kernels.sample, size(1), size(1), {session.token, session.rng, logits},
                  uint32_t(248320), temperature, topP, topK);
}
}  // namespace
void Scratch::ensure(Device& device, uint32_t seq) {
    uint32_t target = roundUp(seq, 32); if (target <= capacity) return; capacity = target;
    uint64_t rows = target, hiddenBytes = rows * 4096 * 2, mlpBytes = rows * 12288 * 2;
    std::vector<std::pair<Tensor*, uint64_t>> slots = {
        {&hidden[0], hiddenBytes}, {&hidden[1], hiddenBytes}, {&inputNorm, hiddenBytes}, {&postNorm, hiddenBytes},
        {&padInput, rows * 12288 * 2},
        {&mlpGate, mlpBytes}, {&mlpUp, mlpBytes}, {&mlpMixed, mlpBytes},
        {&attnQG, rows * 8192 * 2}, {&attnK, rows * 1024 * 2}, {&attnV, rows * 1024 * 2},
        {&attnQ, hiddenBytes}, {&attnGate, hiddenBytes}, {&attnQNorm, hiddenBytes}, {&attnKNorm, rows * 1024 * 2},
        {&attnQRope, hiddenBytes}, {&attnKRope, rows * 1024 * 2}, {&attnOut, hiddenBytes},
        {&attnGated, hiddenBytes}, {&attnPartials, 16 * 16 * 258 * 4},
        {&gdnMixed, rows * 8192 * 2}, {&gdnZ, hiddenBytes}, {&gdnB, rows * 32 * 2}, {&gdnA, rows * 32 * 2},
        {&gdnBeta, rows * 32 * 2}, {&gdnG, rows * 32 * 4}, {&gdnConvolved, rows * 8192 * 2},
        {&gdnQ, hiddenBytes}, {&gdnK, hiddenBytes}, {&gdnV, hiddenBytes}, {&gdnDelta, hiddenBytes},
        {&gdnNormed, hiddenBytes}, {&mid, hiddenBytes}, {&projected, hiddenBytes}, {&modelNorm, 4096 * 2},
        {&logits, 248320 * 2}};
    uint64_t bytes = 0; for (const auto& [_, count] : slots) bytes = roundUp(bytes, 256) + count;
    arena = device.empty(bytes); uint64_t offset = 0;
    for (auto& [tensor, count] : slots) { offset = roundUp(offset, 256); *tensor = arena.view(offset, count); offset += count; }
}
Session::Session(Model& model) : model(model), kv(model.device, model.maxContext) {
    if (model.sessionLive) throw std::runtime_error("only one live session is supported");
    inputIds = model.device.empty(uint64_t(model.maxContext) * sizeof(int32_t), true);
    token = model.device.empty(sizeof(int32_t), true);
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count(); rng = model.device.upload(&seed, sizeof(seed));
    for (uint32_t i = 0; i < model.layers.size(); ++i) if (!model.layers[i].fullAttention) {
        layers[i].conv[0] = model.device.empty(8192 * 4 * 2); layers[i].conv[1] = model.device.empty(8192 * 4 * 2);
        layers[i].recurrent = model.device.empty(uint64_t(32) * 128 * 128 * 4);
    }
    model.sessionLive = true;
}
Session::~Session() { model.sessionLive = false; }
int32_t forward(Model& model, Session& session, const int32_t* ids, uint32_t seq,
                float temperature, float topP, int32_t topK) {
    if (!seq || session.length + seq > model.maxContext) throw std::runtime_error("forward exceeds maximum context");
    model.device.write(session.inputIds, ids, uint64_t(seq) * sizeof(int32_t)); session.kv.ensure(session.length + seq);
    Scratch& scratch = model.scratch(seq); scratch.padRows = model.kernels.padRows; uint32_t chunks = (seq + 15) / 16;
    model.device.preparePass(uint64_t(512 + 24 * chunks) * 128, 512 + 24 * chunks);
    Pass pass(model.device);
    pass.dispatch(model.kernels.embed, size(4096, seq), size(256),
                  {scratch.hidden[0], session.inputIds, model.embedding}, int64_t(seq), int64_t(4096));
    Tensor hidden = scratch.hidden[0];
    for (uint32_t i = 0; i < model.layers.size(); ++i)
        hidden = decoderLayer(pass, model, model.layers[i], session.layers[i], hidden, scratch.hidden[(i + 1) & 1],
                              scratch, session, seq);
    Tensor last = hidden.view(uint64_t(seq - 1) * 4096 * 2, 4096 * 2);
    last = rms(pass, model, last, model.norm, scratch.modelNorm, 1, 4096);
    Tensor logits = linear(pass, last, model.head, 1, scratch.logits, scratch); sample(pass, model, session, logits, temperature, topP, topK);
    pass.commit(); session.length += seq;
    return *static_cast<int32_t*>(session.token.buffer->value->contents());
}
}  // namespace infeng::qwen35
namespace {
thread_local std::string lastError;
template <class F> int32_t status(F&& function) {
    try { function(); return 0; } catch (const std::exception& error) { lastError = error.what(); return -1; }
}
}
extern "C" {
const char* infeng_last_error() { return lastError.c_str(); }
void* infeng_model_create(const char* weights, const char* kernels, uint32_t context, int32_t profile) {
    try { return new infeng::qwen35::Model(weights, kernels, context, profile != 0); }
    catch (const std::exception& error) { lastError = error.what(); return nullptr; }
}
void infeng_model_release(void* model) { delete static_cast<infeng::qwen35::Model*>(model); }
void* infeng_session_create(void* model) {
    try { return new infeng::qwen35::Session(*static_cast<infeng::qwen35::Model*>(model)); }
    catch (const std::exception& error) { lastError = error.what(); return nullptr; }
}
void infeng_session_release(void* session) { delete static_cast<infeng::qwen35::Session*>(session); }
int32_t infeng_forward(void* pointer, const int32_t* ids, uint32_t count, float temperature, float topP,
                       int32_t topK, int32_t* output) {
    return status([&] { auto& session = *static_cast<infeng::qwen35::Session*>(pointer);
        *output = infeng::qwen35::forward(session.model, session, ids, count, temperature, topP, topK); });
}
uint64_t infeng_session_length(void* session) { return static_cast<infeng::qwen35::Session*>(session)->length; }
uint64_t infeng_session_mapped_bytes(void* session) { return static_cast<infeng::qwen35::Session*>(session)->kv.mappedBytes(); }
uint64_t infeng_model_parameter_count(void* model) { return static_cast<infeng::qwen35::Model*>(model)->parameterCount; }
uint64_t infeng_model_weight_bytes(void* model) { return static_cast<infeng::qwen35::Model*>(model)->modelBytes; }
uint64_t infeng_model_vocab_size(void*) { return 248320; }
int32_t infeng_model_counters(void* pointer, InfengCounters* output) {
    return status([&] { const auto& counters = static_cast<infeng::qwen35::Model*>(pointer)->device.counters();
        *output = {counters.gpuTimeNs, counters.passes, counters.dispatches, counters.allocations, counters.allocatedBytes}; });
}
}
