#pragma once
#include "backend/metal/device.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
namespace infeng::qwen35 {
using metal::Device;
using metal::CommandBuffer;
using metal::Pipeline;
using metal::SparseBuffers;
using metal::Tensor;
enum class QuantType : uint32_t { F32 = 0, F16 = 1, Q8_0 = 8, Q4_K = 12, Q5_K = 13, Q6_K = 14, IQ4_XS = 23 };
struct Linear {
    Tensor weight;
    Pipeline* decode = nullptr;
    Pipeline* prefill = nullptr;
    uint32_t k = 0, n = 0;
    uint16_t decodeThreads = 0;
    uint8_t outputsPerGroup = 0;
    QuantType type{};
};
struct MlpWeights {
    Linear gate, up, down;
    Pipeline* fusedDecode = nullptr;
    uint16_t decodeThreads = 0;
    uint8_t outputsPerGroup = 0;
};
struct AttentionWeights { Linear q, k, v, out; Tensor qNorm, kNorm; };
struct GdnWeights { Linear qkv, z, b, a, out; Tensor conv, norm, dt, A; };
struct Layer {
    bool fullAttention = false;
    uint8_t kvIndex = 0;
    Tensor inputNorm, postNorm;
    MlpWeights mlp;
    AttentionWeights attention;
    GdnWeights gdn;
};
struct Kernels {
    Pipeline *embed, *rms, *add, *siluMul, *padRows, *initRope, *argmax, *sample;
    Pipeline *attentionPrefill, *attentionDecodeScan, *attentionDecodeReduce, *attentionGate, *unpackAttention, *ropeQk;
    Pipeline *gdnPrepare, *gdnConv, *splitQk, *deltaPrefill, *deltaDecode, *gdnNorm;
};
struct Scratch {
    uint32_t capacity = 0;
    Pipeline* padRows = nullptr;
    Tensor arena, hidden[2], inputNorm, postNorm, padInput;
    Tensor mlpGate, mlpUp, mlpMixed;
    Tensor attnQG, attnK, attnV, attnQ, attnGate, attnQNorm, attnKNorm, attnQRope, attnKRope;
    Tensor attnOut, attnGated, attnPartials;
    Tensor gdnMixed, gdnZ, gdnB, gdnA, gdnBeta, gdnG, gdnConvolved, gdnQ, gdnK, gdnV, gdnDelta, gdnNormed;
    Tensor mid, projected, modelNorm, logits;
    void ensure(Device& device, uint32_t seq);
};
struct Model {
    Device device;
    uint32_t maxContext;
    Tensor weightArena, embedding, norm, rope;
    Linear head;
    std::array<Layer, 32> layers;
    Kernels kernels{};
    Scratch decode, prefill;
    uint64_t parameterCount = 0, modelBytes = 0;
    bool sessionLive = false;
    Model(const std::filesystem::path& weights, const std::filesystem::path& kernels, uint32_t maxContext);
    Scratch& scratch(uint32_t seq) { Scratch& value = seq == 1 ? decode : prefill; value.ensure(device, seq); return value; }
};
struct LayerState {
    Tensor conv[2], recurrent;
    uint8_t convSlot = 1;
    bool initialized = false;
};
struct Session {
    Model& model;
    SparseBuffers kv;
    std::array<LayerState, 32> layers;
    Tensor inputIds, token, rng;
    uint32_t length = 0;
    Session(Model& model);
    ~Session();
};
int32_t forward(Model& model, Session& session, const int32_t* ids, uint32_t seq,
                float temperature, float topP, int32_t topK);
}  // namespace infeng::qwen35
extern "C" {
#define INFENG_EXPORT __attribute__((visibility("default")))
INFENG_EXPORT const char* infeng_last_error();
INFENG_EXPORT void* infeng_model_create(const char* weights_path, const char* kernels_path, uint32_t max_context);
INFENG_EXPORT void infeng_model_release(void* model);
INFENG_EXPORT void* infeng_session_create(void* model);
INFENG_EXPORT void infeng_session_release(void* session);
INFENG_EXPORT int32_t infeng_forward(void* session, const int32_t* ids, uint32_t count, float temperature, float top_p,
                                     int32_t top_k, int32_t* output_token);
INFENG_EXPORT uint64_t infeng_session_length(void* session);
INFENG_EXPORT uint64_t infeng_session_mapped_bytes(void* session);
INFENG_EXPORT uint64_t infeng_model_parameter_count(void* model);
INFENG_EXPORT uint64_t infeng_model_weight_bytes(void* model);
INFENG_EXPORT uint64_t infeng_model_vocab_size(void* model);
}
