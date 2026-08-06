#pragma once
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
namespace infeng::metal {
struct Counters {
    uint64_t gpuTimeNs = 0, passes = 0, dispatches = 0, allocations = 0, allocatedBytes = 0;
};
class Device;
struct Buffer {
    Device* device;
    MTL::Buffer* value;
    uint64_t bytes;
    Buffer(Device* device, MTL::Buffer* value, uint64_t bytes) : device(device), value(value), bytes(bytes) {}
    ~Buffer();
};
struct Tensor {
    std::shared_ptr<Buffer> buffer;
    uint64_t offset = 0, bytes = 0;
    Tensor view(uint64_t byteOffset, uint64_t byteCount) const {
        return {buffer, offset + byteOffset, byteCount};
    }
    MTL::GPUAddress address() const { return buffer->value->gpuAddress() + offset; }
};
using Pipeline = MTL::ComputePipelineState;
class Pass {
    Device& device;
    MTL4::ComputeCommandEncoder* encoder;
    uint64_t constantOffset = 0;
    uint32_t tableIndex = 0;
    template <class T> void scalar(MTL4::ArgumentTable* table, uint32_t index, const T& value);
public:
    explicit Pass(Device& device);
    Pass(const Pass&) = delete;
    ~Pass();
    template <class... Scalars>
    void dispatch(Pipeline* pipeline, MTL::Size threads, MTL::Size group, std::initializer_list<Tensor> tensors,
                  const Scalars&... scalars);
    void commit();
};
class Device {
    friend struct Buffer;
    friend class Pass;
    friend class SparseBuffers;
    MTL::Device* value = nullptr;
    MTL4::CommandQueue* queue = nullptr;
    MTL4::CommandAllocator* allocator = nullptr;
    MTL4::CommandBuffer* commandBuffer = nullptr;
    MTL::ResidencySet* residency = nullptr;
    MTL::SharedEvent* event = nullptr;
    MTL4::CounterHeap* counterHeap = nullptr;
    std::vector<MTL::Library*> libraries;
    std::unordered_map<std::string, Pipeline*> pipelines;
    std::vector<MTL4::ArgumentTable*> tables;
    Tensor constants;
    uint64_t eventValue = 0;
    void add(MTL::Allocation* allocation);
    void remove(MTL::Allocation* allocation);
    MTL4::ArgumentTable* table(uint32_t index);
    void finish(bool profilePass);
public:
    Device(const std::filesystem::path& kernels, bool profile);
    Device(const Device&) = delete;
    ~Device();
    Tensor empty(uint64_t bytes, bool shared = false);
    Tensor upload(const void* source, uint64_t bytes);
    void write(const Tensor& destination, const void* source, uint64_t bytes);
    Pipeline* pipeline(const std::string& name);
    void preparePass(uint64_t constantBytes, uint32_t tableCount);
    const Counters& counters() const { return stats; }
    Counters stats;
};
class SparseBuffers {
    static constexpr uint32_t count = 16, pageTokens = 128;
    static constexpr uint64_t pageBytes = 256ull << 10, heapBytes = 64ull << 20;
    static constexpr uint32_t pagesPerHeap = heapBytes / (pageBytes * count);
    Device& device;
    uint32_t maxPages, mappedPages = 0;
    std::vector<Tensor> buffers;
    std::vector<MTL::Heap*> heaps;
    void addHeap();
public:
    SparseBuffers(Device& device, uint32_t maxContext);
    SparseBuffers(const SparseBuffers&) = delete;
    ~SparseBuffers();
    void ensure(uint32_t tokens);
    Tensor key(uint32_t attentionLayer) const { return buffers[2 * attentionLayer]; }
    Tensor value(uint32_t attentionLayer) const { return buffers[2 * attentionLayer + 1]; }
    uint64_t mappedBytes() const { return uint64_t(mappedPages) * count * pageBytes; }
};
template <class T> void Pass::scalar(MTL4::ArgumentTable* table, uint32_t index, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    constantOffset = (constantOffset + 15) & ~15ull;
    if (constantOffset + sizeof(T) > device.constants.bytes) throw std::runtime_error("pass constant storage exceeded");
    std::memcpy(static_cast<uint8_t*>(device.constants.buffer->value->contents()) + constantOffset, &value, sizeof(T));
    table->setAddress(device.constants.address() + constantOffset, index); constantOffset += sizeof(T);
}
template <class... Scalars>
void Pass::dispatch(Pipeline* pipeline, MTL::Size threads, MTL::Size group, std::initializer_list<Tensor> tensors,
                    const Scalars&... scalars) {
    auto* table = device.table(tableIndex++); uint32_t index = 0;
    for (const Tensor& tensor : tensors) table->setAddress(tensor.address(), index++);
    (scalar(table, index++, scalars), ...);
    encoder->setComputePipelineState(pipeline); encoder->setArgumentTable(table); encoder->dispatchThreads(threads, group);
    encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    ++device.stats.dispatches;
}
}  // namespace infeng::metal
