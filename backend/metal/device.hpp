#pragma once
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>
namespace infeng::metal {
struct Counters {
    uint64_t gpuTimeNs = 0, passes = 0, dispatches = 0, allocations = 0, allocatedBytes = 0;
};
struct KernelCounter {
    std::string phase, name;
    uint64_t gpuTimeNs = 0, launches = 0;
};
class Device;
struct Buffer {
    Device* device;
    MTL::Buffer* metalBuffer;
    uint64_t bytes;
    Buffer(Device* device, MTL::Buffer* metalBuffer, uint64_t bytes)
        : device(device), metalBuffer(metalBuffer), bytes(bytes) {}
    ~Buffer();
};
struct Tensor {
    std::shared_ptr<Buffer> buffer;
    uint64_t offset = 0, bytes = 0;
    Tensor view(uint64_t byteOffset, uint64_t byteCount) const {
        return {buffer, offset + byteOffset, byteCount};
    }
    MTL::GPUAddress address() const { return buffer->metalBuffer->gpuAddress() + offset; }
};
using Pipeline = MTL::ComputePipelineState;
class CommandBuffer {
    friend class Device;
    Device& device;
    MTL4::CommandBuffer* metalCommandBuffer = nullptr;
    MTL4::ComputeCommandEncoder* encoder = nullptr;
    MTL4::CounterHeap* counterHeap = nullptr;
    Tensor constants;
    std::vector<MTL4::ArgumentTable*> tables;
    struct KernelSample { std::string name; uint32_t start, end; };
    std::vector<KernelSample> profiles;
    std::string phase;
    bool profiling = false;
    uint64_t constantOffset = 0;
    uint32_t tableIndex = 0, counterIndex = 0, counterLimit = 0;
    template <class T> void scalar(MTL4::ArgumentTable* table, uint32_t index, const T& value);
    MTL4::ArgumentTable* table(uint32_t index);
    void copy(MTL::Buffer* source, uint64_t sourceOffset, MTL::Buffer* destination, uint64_t destinationOffset,
              uint64_t bytes);
    void writeTimestamp(uint32_t index);
    void submit();
public:
    explicit CommandBuffer(Device& device, uint64_t constantBytes = 1 << 20, bool profile = true,
                           uint32_t dispatchCapacity = 1024, const char* phase = "unknown");
    CommandBuffer(const CommandBuffer&) = delete;
    ~CommandBuffer();
    template <class... Scalars>
    void dispatch(Pipeline* pipeline, MTL::Size threads, MTL::Size group, std::initializer_list<Tensor> tensors,
                  const Scalars&... scalars);
    void commit();
};
class Device {
    friend struct Buffer;
    friend class CommandBuffer;
    friend class SparseBuffers;
    static constexpr uint32_t counterHeapEntries = 4096;
    MTL::Device* metalDevice = nullptr;
    MTL4::CommandQueue* queue = nullptr;
    MTL4::CommandAllocator* allocator = nullptr;
    MTL::ResidencySet* residency = nullptr;
    MTL::SharedEvent* event = nullptr;
    MTL4::CounterHeap* counterHeap = nullptr;
    std::vector<MTL::Library*> libraries;
    std::unordered_map<std::string, Pipeline*> pipelines;
    std::unordered_map<Pipeline*, std::string> pipelineNames;
    std::vector<KernelCounter> kernelStats;
    uint64_t eventValue = 0;
    bool profileEnabled = false;
    void add(MTL::Allocation* allocation);
    void remove(MTL::Allocation* allocation);
    void recordKernel(const std::string& phase, const std::string& name, uint64_t gpuTimeNs);
    const std::string& pipelineName(Pipeline* pipeline) const { return pipelineNames.at(pipeline); }
public:
    Device(const std::filesystem::path& kernels, bool profile);
    Device(const Device&) = delete;
    ~Device();
    Tensor empty(uint64_t bytes, bool shared = false);
    Tensor upload(const void* source, uint64_t bytes);
    void write(const Tensor& destination, const void* source, uint64_t bytes);
    Pipeline* pipeline(const std::string& name);
    const Counters& counters() const { return stats; }
    const std::vector<KernelCounter>& kernelCounters() const { return kernelStats; }
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
template <class T> void CommandBuffer::scalar(MTL4::ArgumentTable* table, uint32_t index, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    constantOffset = (constantOffset + 15) & ~15ull;
    if (constantOffset + sizeof(T) > constants.bytes)
        throw std::runtime_error("command buffer constant storage exceeded");
    std::memcpy(static_cast<uint8_t*>(constants.buffer->metalBuffer->contents()) + constantOffset, &value, sizeof(T));
    table->setAddress(constants.address() + constantOffset, index); constantOffset += sizeof(T);
}
template <class... Scalars>
void CommandBuffer::dispatch(Pipeline* pipeline, MTL::Size threads, MTL::Size group,
                             std::initializer_list<Tensor> tensors, const Scalars&... scalars) {
    // bracket each dispatch with GPU timestamps and retain its pipeline name for aggregation
    uint32_t start = 0, end = 0;
    if (counterHeap) {
        if (counterIndex + 2 > counterLimit) throw std::runtime_error("command buffer counter storage exceeded");
        start = counterIndex++; writeTimestamp(start);
    }
    auto* argumentTable = table(tableIndex++); uint32_t index = 0;
    for (const Tensor& tensor : tensors) argumentTable->setAddress(tensor.address(), index++);
    (scalar(argumentTable, index++, scalars), ...);
    encoder->setComputePipelineState(pipeline); encoder->setArgumentTable(argumentTable);
    encoder->dispatchThreads(threads, group);
    encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    if (counterHeap) {
        end = counterIndex++; writeTimestamp(end);
        profiles.push_back({device.pipelineName(pipeline), start, end});
    }
    ++device.stats.dispatches;
}
}  // namespace infeng::metal
