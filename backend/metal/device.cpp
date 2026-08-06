#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "device.hpp"
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
namespace infeng::metal {
static std::string message(NS::Error* error) {
    return error && error->localizedDescription() ? error->localizedDescription()->utf8String() : "Metal operation failed";
}
Buffer::~Buffer() { if (value) { device->remove(value); value->release(); } }
Device::Device(const std::filesystem::path& kernels, bool profile) {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    value = MTL::CreateSystemDefaultDevice();
    if (!value) throw std::runtime_error("Metal device unavailable");
    if (!value->supportsPlacementSparse()) throw std::runtime_error("placement sparse buffers unsupported");
    queue = value->newMTL4CommandQueue(); allocator = value->newCommandAllocator(); commandBuffer = value->newCommandBuffer();
    if (!queue || !allocator || !commandBuffer) throw std::runtime_error("MTL4 command resources unavailable");
    auto descriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init()); descriptor->setInitialCapacity(512);
    NS::Error* error = nullptr; residency = value->newResidencySet(descriptor.get(), &error);
    if (!residency) throw std::runtime_error(message(error));
    event = value->newSharedEvent();
    if (!event) throw std::runtime_error("shared event creation failed");
    queue->addResidencySet(residency);
    if (profile) {
        auto counters = NS::TransferPtr(MTL4::CounterHeapDescriptor::alloc()->init());
        counters->setType(MTL4::CounterHeapTypeTimestamp); counters->setCount(2);
        counterHeap = value->newCounterHeap(counters.get(), &error);
        if (!counterHeap) throw std::runtime_error(message(error));
    }
    std::vector<std::filesystem::path> sources;
    for (const auto& entry : std::filesystem::directory_iterator(kernels))
        if (entry.path().extension() == ".metal") sources.push_back(entry.path());
    std::sort(sources.begin(), sources.end());
    for (const auto& path : sources) {
        std::ifstream file(path); std::stringstream stream; stream << file.rdbuf();
        auto source = NS::String::string(stream.str().c_str(), NS::UTF8StringEncoding);
        MTL::Library* library = value->newLibrary(source, nullptr, &error);
        if (!library) throw std::runtime_error(path.filename().string() + ": " + message(error));
        libraries.push_back(library);
    }
    constants = empty(1 << 20, true); tables.reserve(512);
}
Device::~Device() {
    constants = {}; for (auto* table : tables) table->release();
    for (auto& [_, pipeline] : pipelines) pipeline->release();
    for (auto* library : libraries) library->release();
    if (counterHeap) counterHeap->release();
    if (event) event->release();
    if (residency) { queue->removeResidencySet(residency); residency->release(); }
    if (commandBuffer) commandBuffer->release(); if (allocator) allocator->release(); if (queue) queue->release();
    if (value) value->release();
}
void Device::add(MTL::Allocation* allocation) { residency->addAllocation(allocation); residency->commit(); }
void Device::remove(MTL::Allocation* allocation) { residency->removeAllocation(allocation); residency->commit(); }
Tensor Device::empty(uint64_t bytes, bool shared) {
    MTL::Buffer* buffer = value->newBuffer(bytes, shared ? MTL::ResourceStorageModeShared : MTL::ResourceStorageModePrivate);
    if (!buffer) throw std::runtime_error("buffer allocation failed");
    add(buffer); ++stats.allocations; stats.allocatedBytes += bytes;
    return {std::make_shared<Buffer>(this, buffer, bytes), 0, bytes};
}
Tensor Device::upload(const void* source, uint64_t bytes) { Tensor target = empty(bytes); write(target, source, bytes); return target; }
void Device::write(const Tensor& destination, const void* source, uint64_t bytes) {
    if (bytes > destination.bytes) throw std::runtime_error("buffer write exceeds destination");
    if (destination.buffer->value->storageMode() == MTL::StorageModeShared) {
        std::memcpy(static_cast<uint8_t*>(destination.buffer->value->contents()) + destination.offset, source, bytes); return;
    }
    MTL::Buffer* staging = value->newBuffer(source, bytes, MTL::ResourceStorageModeShared);
    if (!staging) throw std::runtime_error("staging allocation failed"); add(staging);
    commandBuffer->beginCommandBuffer(allocator); commandBuffer->useResidencySet(residency);
    auto* encoder = commandBuffer->computeCommandEncoder();
    encoder->copyFromBuffer(staging, 0, destination.buffer->value, destination.offset, bytes);
    encoder->endEncoding(); finish(false); remove(staging); staging->release();
}
Pipeline* Device::pipeline(const std::string& name) {
    if (auto found = pipelines.find(name); found != pipelines.end()) return found->second;
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init()); MTL::Function* function = nullptr;
    auto string = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
    for (auto* library : libraries) if ((function = library->newFunction(string))) break;
    if (!function) throw std::runtime_error("missing kernel " + name);
    NS::Error* error = nullptr; Pipeline* result = value->newComputePipelineState(function, &error); function->release();
    if (!result) throw std::runtime_error(name + ": " + message(error));
    pipelines.emplace(name, result); return result;
}
void Device::preparePass(uint64_t constantBytes, uint32_t tableCount) {
    if (constantBytes > constants.bytes) constants = empty(std::max(constantBytes, constants.bytes * 2), true);
    while (tables.size() < tableCount) table(tables.size());
}
MTL4::ArgumentTable* Device::table(uint32_t index) {
    if (index < tables.size()) return tables[index];
    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(16); descriptor->setInitializeBindings(true);
    NS::Error* error = nullptr; auto* result = value->newArgumentTable(descriptor.get(), &error);
    if (!result) throw std::runtime_error(message(error)); tables.push_back(result); return result;
}
void Device::finish(bool profilePass) {
    if (profilePass && counterHeap) commandBuffer->writeTimestampIntoHeap(counterHeap, 1);
    commandBuffer->endCommandBuffer(); const MTL4::CommandBuffer* commands[] = {commandBuffer}; queue->commit(commands, 1);
    uint64_t signal = ++eventValue; queue->signalEvent(event, signal);
    if (!event->waitUntilSignaledValue(signal, std::numeric_limits<uint64_t>::max()))
        throw std::runtime_error("Metal event wait timed out");
    if (profilePass && counterHeap) {
        NS::Data* data = counterHeap->resolveCounterRange(NS::Range::Make(0, 2));
        auto* timestamps = static_cast<const MTL4::TimestampHeapEntry*>(data->bytes());
        stats.gpuTimeNs += timestamps[1].timestamp - timestamps[0].timestamp; ++stats.passes;
    }
    allocator->reset();
}
Pass::Pass(Device& device) : device(device) {
    device.commandBuffer->beginCommandBuffer(device.allocator); device.commandBuffer->useResidencySet(device.residency);
    if (device.counterHeap) {
        device.counterHeap->invalidateCounterRange(NS::Range::Make(0, 2));
        device.commandBuffer->writeTimestampIntoHeap(device.counterHeap, 0);
    }
    encoder = device.commandBuffer->computeCommandEncoder();
    encoder->barrierAfterQueueStages(MTL::StageResourceState, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
}
Pass::~Pass() {
    if (encoder) try { encoder->endEncoding(); device.finish(false); } catch (...) {}
}
void Pass::commit() { encoder->endEncoding(); encoder = nullptr; device.finish(true); }
SparseBuffers::SparseBuffers(Device& device, uint32_t maxContext)
    : device(device), maxPages(uint32_t((uint64_t(maxContext) + pageTokens - 1) / pageTokens)) {
    uint64_t bytes = uint64_t(maxContext) * 4 * 256 * 2; buffers.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        MTL::Buffer* buffer = device.value->newBuffer(bytes, MTL::ResourceStorageModePrivate, MTL::SparsePageSize256);
        if (!buffer) throw std::runtime_error("sparse KV buffer allocation failed");
        device.add(buffer); buffers.push_back({std::make_shared<Buffer>(&device, buffer, bytes), 0, bytes});
    }
}
void SparseBuffers::addHeap() {
    auto descriptor = NS::TransferPtr(MTL::HeapDescriptor::alloc()->init());
    descriptor->setType(MTL::HeapTypePlacement); descriptor->setStorageMode(MTL::StorageModePrivate);
    descriptor->setSize(heapBytes); descriptor->setSparsePageSize(MTL::SparsePageSize256);
    descriptor->setMaxCompatiblePlacementSparsePageSize(MTL::SparsePageSize256);
    MTL::Heap* heap = device.value->newHeap(descriptor.get());
    if (!heap) throw std::runtime_error("KV heap allocation failed"); device.add(heap); heaps.push_back(heap);
    // Allocate 64 MiB once, then issue sparse mappings per heap across all 16 buffers.
    uint32_t pages = std::min(pagesPerHeap, maxPages - mappedPages);
    MTL::Heap* mappingHeap = heaps.back();
    for (uint32_t page = 0; page < pages; ++page) for (uint32_t i = 0; i < count; ++i) {
        MTL4::UpdateSparseBufferMappingOperation operation{MTL::SparseTextureMappingModeMap,
                                                           NS::Range::Make(mappedPages + page, 1), page * count + i};
        device.queue->updateBufferMappings(buffers[i].buffer->value, mappingHeap, &operation, 1);
    }
    mappedPages += pages;
}
void SparseBuffers::ensure(uint32_t tokens) {
    uint32_t target = std::min(maxPages, uint32_t((uint64_t(tokens) + pageTokens - 1) / pageTokens));
    while (mappedPages < target) addHeap();
}
SparseBuffers::~SparseBuffers() {
    for (uint32_t page = 0; page < mappedPages; ++page) for (uint32_t i = 0; i < count; ++i) {
        MTL4::UpdateSparseBufferMappingOperation operation{MTL::SparseTextureMappingModeUnmap, NS::Range::Make(page, 1), 0};
        device.queue->updateBufferMappings(buffers[i].buffer->value, nullptr, &operation, 1);
    }
    buffers.clear(); for (auto* heap : heaps) { device.remove(heap); heap->release(); }
}
}  // namespace infeng::metal
