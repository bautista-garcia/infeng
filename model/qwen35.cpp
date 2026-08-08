#include "qwen35.hpp"
#include <chrono>
#include <stdexcept>
#include <vector>
namespace infeng::qwen35 {
namespace {
constexpr uint64_t halfBytes = 2;
MTL::Size size(uint64_t x, uint64_t y = 1, uint64_t z = 1) { return MTL::Size(x, y, z); }
uint64_t roundUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }
Tensor linear(CommandBuffer& commandBuffer, const Tensor& x, const Linear& weight, uint32_t rows, Tensor output, Scratch& scratch,
              uint32_t mode = 0, const Tensor* aux = nullptr, uint32_t context = 0, bool synchronize = true) {
    if (weight.type == QuantType::F16) {
        if (synchronize) commandBuffer.dispatch(weight.decode, size(1024, rows), size(32), {output, x, weight.weight}, rows);
        else commandBuffer.dispatchConcurrent(weight.decode, size(1024, rows), size(32), {output, x, weight.weight}, rows);
        return output;
    }
    if (rows == 1) {
        const Tensor& extra = aux ? *aux : output;
        MTL::Size threads = size(((weight.n + weight.outputsPerGroup - 1) / weight.outputsPerGroup) * weight.decodeThreads);
        if (synchronize) commandBuffer.dispatch(weight.decode, threads, size(weight.decodeThreads),
                                                {output, x, weight.weight, extra}, mode, context, uint32_t(0));
        else commandBuffer.dispatchConcurrent(weight.decode, threads, size(weight.decodeThreads),
                                              {output, x, weight.weight, extra}, mode, context, uint32_t(0));
        return output;
    }
    if (mode) throw std::runtime_error("quant prefill has no fused store mode");
    uint32_t padded = roundUp(rows, 8); Tensor source = x, target = output;
    if (padded != rows) {
        source = scratch.padInput.view(0, uint64_t(padded) * weight.k * halfBytes);
        uint64_t total = uint64_t(padded) * weight.k;
        commandBuffer.dispatch(scratch.padRows,
                      size(roundUp(total, 256)), size(256), {source, x}, rows, padded, weight.k);
    }
    uint32_t rowGroups = (padded + 127) / 128;
    if (synchronize) commandBuffer.dispatch(weight.prefill, size(512 * (weight.n / 32), rowGroups), size(512),
                                            {target, source, weight.weight}, int64_t(padded));
    else commandBuffer.dispatchConcurrent(weight.prefill, size(512 * (weight.n / 32), rowGroups), size(512),
                                          {target, source, weight.weight}, int64_t(padded));
    return target.view(0, uint64_t(rows) * weight.n * halfBytes);
}
Tensor rms(CommandBuffer& commandBuffer, Model& model, const Tensor& x, const Tensor& weight, Tensor output, uint32_t rows, uint32_t dim) {
    commandBuffer.dispatch(model.kernels.rms, size(256, rows), size(256), {output, x, weight}, rows, dim, 1e-6f); return output;
}
Tensor add(CommandBuffer& commandBuffer, Model& model, const Tensor& a, const Tensor& b, Tensor output, uint64_t elements) {
    commandBuffer.dispatch(model.kernels.add, size(roundUp(elements, 256)), size(256), {output, a, b}, uint32_t(elements)); return output;
}
Tensor mlp(CommandBuffer& commandBuffer, Model& model, const Tensor& x, const MlpWeights& weights, Tensor output,
           Scratch& scratch, uint32_t seq) {
    if (seq == 1) {
        commandBuffer.dispatch(weights.fusedDecode, size(12288 / weights.outputsPerGroup * weights.decodeThreads),
                      size(weights.decodeThreads), {scratch.mlpMixed, x, weights.gate.weight, weights.up.weight}, int64_t(1));
    } else {
        Tensor gate = linear(commandBuffer, x, weights.gate, seq, scratch.mlpGate, scratch, 0, nullptr, 0, false);
        Tensor up = linear(commandBuffer, x, weights.up, seq, scratch.mlpUp, scratch); uint64_t elements = uint64_t(seq) * 12288;
        commandBuffer.dispatch(model.kernels.siluMul, size(roundUp(elements, 256)), size(256),
                      {scratch.mlpMixed, gate, up}, uint32_t(elements));
    }
    return linear(commandBuffer, scratch.mlpMixed, weights.down, seq, output, scratch);
}
Tensor attention(CommandBuffer& commandBuffer, Model& model, const Layer& layer, const Tensor& x, const Tensor& residual,
                 Tensor destination, Scratch& scratch, Session& session, uint32_t seq) {
    const AttentionWeights& weights = layer.attention; uint32_t start = session.length;
    Tensor cacheK = session.kv.key(layer.kvIndex), cacheV = session.kv.value(layer.kvIndex);
    if (seq == 1) {
        Tensor qg = linear(commandBuffer, x, weights.q, 1, scratch.attnQG, scratch, 0, nullptr, 0, false);
        Tensor rawK = linear(commandBuffer, x, weights.k, 1, scratch.attnK, scratch, 0, nullptr, 0, false);
        linear(commandBuffer, x, weights.v, 1, cacheV, scratch, 1, nullptr, start);
        uint32_t splits = std::min(8u, start + 1);
        commandBuffer.dispatch(model.kernels.attentionDecodeScan, size(16 * splits * 128), size(128),
                      {scratch.attnPartials, qg, rawK, cacheK, cacheV, weights.qNorm, weights.kNorm, model.rope},
                      start, model.maxContext, splits);
        commandBuffer.dispatch(model.kernels.attentionDecodeReduce, size(16 * 128), size(128),
                      {scratch.attnOut, scratch.attnPartials, qg}, splits);
        return linear(commandBuffer, scratch.attnOut, weights.out, 1, destination, scratch, 2, &residual);
    }
    Tensor qg = linear(commandBuffer, x, weights.q, seq, scratch.attnQG, scratch, 0, nullptr, 0, false);
    Tensor k = linear(commandBuffer, x, weights.k, seq, scratch.attnK, scratch, 0, nullptr, 0, false);
    Tensor v = linear(commandBuffer, x, weights.v, seq, scratch.attnV, scratch);
    uint64_t qElements = uint64_t(seq) * 4096;
    commandBuffer.dispatchConcurrent(model.kernels.unpackAttention, size(roundUp(qElements, 256)), size(256),
                                     {scratch.attnQ, scratch.attnGate, qg}, uint32_t(qElements));
    k = rms(commandBuffer, model, k, weights.kNorm, scratch.attnKNorm, seq * 4, 256);
    Tensor q = rms(commandBuffer, model, scratch.attnQ, weights.qNorm, scratch.attnQNorm, seq * 16, 256);
    commandBuffer.dispatch(model.kernels.ropeQk, size(roundUp(qElements, 256)), size(256),
                  {scratch.attnQRope, scratch.attnKRope, q, k, model.rope}, seq, start);
    commandBuffer.dispatch(model.kernels.attentionPrefill, size(uint64_t(seq) * 16 * 128), size(128),
                  {scratch.attnOut, scratch.attnQRope, scratch.attnKRope, v, cacheK, cacheV},
                  int64_t(1), int64_t(seq), int64_t(start), int64_t(model.maxContext));
    commandBuffer.dispatch(model.kernels.attentionGate, size(roundUp(qElements, 256)), size(256),
                  {scratch.attnGated, scratch.attnOut, qg}, uint32_t(qElements));
    Tensor projected = linear(commandBuffer, scratch.attnGated, weights.out, seq, scratch.projected, scratch);
    return add(commandBuffer, model, residual, projected, destination, qElements);
}
Tensor gdn(CommandBuffer& commandBuffer, Model& model, const Layer& layer, const Tensor& x, const Tensor& residual,
           Tensor destination, Scratch& scratch, LayerState& state, uint32_t seq) {
    const GdnWeights& weights = layer.gdn;
    Tensor mixed = linear(commandBuffer, x, weights.qkv, seq, scratch.gdnMixed, scratch, 0, nullptr, 0, false);
    Tensor z = linear(commandBuffer, x, weights.z, seq, scratch.gdnZ, scratch, 0, nullptr, 0, false);
    Tensor b = linear(commandBuffer, x, weights.b, seq, scratch.gdnB, scratch, 0, nullptr, 0, false);
    Tensor a = linear(commandBuffer, x, weights.a, seq, scratch.gdnA, scratch); uint64_t heads = uint64_t(seq) * 32;
    commandBuffer.dispatchConcurrent(model.kernels.gdnPrepare, size(roundUp(heads, 256)), size(256),
                                     {scratch.gdnBeta, scratch.gdnG, b, a, weights.A, weights.dt}, uint32_t(heads));
    uint8_t slot = 1 - state.convSlot; const Tensor& previous = state.initialized ? state.conv[state.convSlot] : state.conv[slot];
    commandBuffer.dispatch(model.kernels.gdnConv, size(uint64_t(8192) * std::max(seq, 4u)), size(256),
                  {scratch.gdnConvolved, state.conv[slot], mixed, weights.conv, previous},
                  int64_t(1), int64_t(seq), state.initialized); state.convSlot = slot;
    uint64_t elements = uint64_t(seq) * 4096;
    commandBuffer.dispatch(model.kernels.splitQk, size(roundUp(elements, 256)), size(256),
                  {scratch.gdnQ, scratch.gdnK, scratch.gdnV, scratch.gdnConvolved}, seq);
    if (seq == 1) {
        commandBuffer.dispatch(model.kernels.deltaDecode, size(32 * 512), size(512),
                      {scratch.gdnDelta, state.recurrent, scratch.gdnQ, scratch.gdnK, scratch.gdnV,
                       scratch.gdnG, scratch.gdnBeta},
                      int64_t(1), int64_t(1), int64_t(32), int64_t(4096), int64_t(4096), int64_t(128),
                      int64_t(1), state.initialized);
    } else {
        commandBuffer.dispatch(model.kernels.deltaPrefill, size(128, 32), size(128),
                      {scratch.gdnDelta, state.recurrent, scratch.gdnQ, scratch.gdnK, scratch.gdnV,
                       scratch.gdnG, scratch.gdnBeta},
                      int64_t(1), int64_t(seq), int64_t(32), int64_t(seq) * 4096, int64_t(4096),
                      int64_t(128), int64_t(1), state.initialized);
    }
    state.initialized = true;
    commandBuffer.dispatch(model.kernels.gdnNorm, size(128, uint64_t(seq) * 32), size(128),
                  {scratch.gdnNormed, scratch.gdnDelta, z, weights.norm}, 1e-6f);
    Tensor projected = linear(commandBuffer, scratch.gdnNormed, weights.out, seq, scratch.projected, scratch);
    return add(commandBuffer, model, residual, projected, destination, elements);
}
Tensor decoderLayer(CommandBuffer& commandBuffer, Model& model, const Layer& layer, LayerState& state, const Tensor& hidden,
                    Tensor destination, Scratch& scratch, Session& session, uint32_t seq) {
    Tensor x = rms(commandBuffer, model, hidden, layer.inputNorm, scratch.inputNorm, seq, 4096);
    Tensor mid = layer.fullAttention ? attention(commandBuffer, model, layer, x, hidden, scratch.mid, scratch, session, seq)
                                     : gdn(commandBuffer, model, layer, x, hidden, scratch.mid, scratch, state, seq);
    x = rms(commandBuffer, model, mid, layer.postNorm, scratch.postNorm, seq, 4096);
    Tensor projected = mlp(commandBuffer, model, x, layer.mlp, scratch.projected, scratch, seq);
    return add(commandBuffer, model, mid, projected, destination, uint64_t(seq) * 4096);
}
void sample(CommandBuffer& commandBuffer, Model& model, Session& session, const Tensor& logits,
            float temperature, float topP, int32_t topK) {
    if (temperature <= 0) {
        commandBuffer.dispatch(model.kernels.argmax, size(256), size(256), {session.token, logits}, uint32_t(248320)); return;
    }
    if (topK < 0 || topK > 64) throw std::runtime_error("GPU top_k must be between 1 and 64");
    if (!topK && topP < 1) throw std::runtime_error("top_p below 1 requires top_k on this specialized runtime");
    commandBuffer.dispatch(model.kernels.sample, size(1), size(1), {session.token, session.rng, logits},
                  uint32_t(248320), temperature, topP, topK);
}
}  // namespace
void Scratch::ensure(Device& device, uint32_t seq) {
    uint32_t target = seq == 1 ? 32 : roundUp(seq, 128); if (target <= capacity) return; capacity = target;
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
    // ensure acts like a page fault if there isn't enough kv allocated
    model.device.write(session.inputIds, ids, uint64_t(seq) * sizeof(int32_t)); session.kv.ensure(session.length + seq);
    Scratch& scratch = model.scratch(seq); scratch.padRows = model.kernels.padRows; uint32_t chunks = (seq + 31) / 32;
    CommandBuffer commands(model.device, uint64_t(512 + 24 * chunks) * 128, true, 1024 + 24 * chunks,
                           seq == 1 ? "decode" : "prefill");
    commands.dispatch(model.kernels.embed, size(4096, seq), size(256),
                  {scratch.hidden[0], session.inputIds, model.embedding}, int64_t(seq), int64_t(4096));
    Tensor hidden = scratch.hidden[0];
    for (uint32_t i = 0; i < model.layers.size(); ++i)
        hidden = decoderLayer(commands, model, model.layers[i], session.layers[i], hidden, scratch.hidden[(i + 1) & 1],
                              scratch, session, seq);
    Tensor last = hidden.view(uint64_t(seq - 1) * 4096 * 2, 4096 * 2);
    last = rms(commands, model, last, model.norm, scratch.modelNorm, 1, 4096);
    Tensor logits = linear(commands, last, model.head, 1, scratch.logits, scratch); sample(commands, model, session, logits, temperature, topP, topK);
    commands.commit(); session.length += seq;
    return *static_cast<int32_t*>(session.token.buffer->metalBuffer->contents());
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
        *output = {counters.gpuTimeNs, counters.passes}; });
}
uint32_t infeng_model_kernel_counter_count(void* pointer) {
    return static_cast<uint32_t>(static_cast<infeng::qwen35::Model*>(pointer)->device.kernelCounters().size());
}
int32_t infeng_model_kernel_counter(void* pointer, uint32_t index, const char** phase, const char** name,
                                     uint64_t* gpuTimeNs, uint64_t* launches) {
    return status([&] {
        const auto& counters = static_cast<infeng::qwen35::Model*>(pointer)->device.kernelCounters();
        if (index >= counters.size()) throw std::runtime_error("kernel counter index out of range");
        const auto& counter = counters[index]; *phase = counter.phase.c_str(); *name = counter.name.c_str();
        *gpuTimeNs = counter.gpuTimeNs; *launches = counter.launches;
    });
}
}
