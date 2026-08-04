import Foundation
import Metal

nonisolated(unsafe) private var lastError = ""
private let pageSize = 256 * 1024

private func retained<T: AnyObject>(_ value: T) -> UnsafeMutableRawPointer {
    Unmanaged.passRetained(value).toOpaque()
}

private func object<T: AnyObject>(_ pointer: UnsafeMutableRawPointer?) -> T {
    Unmanaged<T>.fromOpaque(pointer!).takeUnretainedValue()
}

@available(macOS 26.4, *)
private final class Buffer {
    let value: MTLBuffer
    weak var runtime: Runtime?
    init(_ value: MTLBuffer, runtime: Runtime) { self.value = value; self.runtime = runtime }
    deinit { runtime?.residency.removeAllocation(value); runtime?.residency.commit() }
}

@available(macOS 26.4, *)
private final class SparseCache {
    static let count = 16, heapSize = 64 * 1024 * 1024, pagesPerHeap = heapSize / pageSize
    let runtime: Runtime
    let buffers: [Buffer]
    var heaps: [MTLHeap] = []
    var mappedPages = 0

    init(_ runtime: Runtime, maxContext: Int) throws {
        self.runtime = runtime
        let length = maxContext * 4 * 256 * 2
        buffers = try (0..<Self.count).map { _ in
            guard let value = runtime.device.makeBuffer(length: length, options: .storageModePrivate,
                                                        placementSparsePageSize: .size256) else {
                throw Failure("sparse KV buffer allocation failed")
            }
            runtime.add(value)
            return Buffer(value, runtime: runtime)
        }
    }

    func ensure(tokens: Int) throws {
        let target = (tokens + 127) / 128
        while mappedPages < target {
            for buffer in buffers {
                let tile = mappedPages * Self.count + buffers.firstIndex { $0 === buffer }!
                let heapIndex = tile / Self.pagesPerHeap, heapOffset = tile % Self.pagesPerHeap
                while heaps.count <= heapIndex { try addHeap() }
                let operation = MTL4UpdateSparseBufferMappingOperation(mode: .map,
                    bufferRange: NSRange(location: mappedPages, length: 1), heapOffset: heapOffset)
                [operation].withUnsafeBufferPointer {
                    runtime.queue.__updateBufferMappings(buffer.value, heap: heaps[heapIndex], operations: $0.baseAddress!, count: 1)
                }
            }
            mappedPages += 1
        }
    }

    private func addHeap() throws {
        let descriptor = MTLHeapDescriptor()
        descriptor.type = .placement; descriptor.storageMode = .private; descriptor.size = Self.heapSize
        descriptor.sparsePageSize = .size256; descriptor.maxCompatiblePlacementSparsePageSize = .size256
        guard let heap = runtime.device.makeHeap(descriptor: descriptor) else { throw Failure("KV heap allocation failed") }
        heaps.append(heap); runtime.add(heap)
    }

    deinit { for heap in heaps { runtime.residency.removeAllocation(heap) }; runtime.residency.commit() }
}

@available(macOS 26.4, *)
private final class Runtime {
    let device: MTLDevice
    let queue: MTL4CommandQueue
    let allocator: MTL4CommandAllocator
    let commandBuffer: MTL4CommandBuffer
    let residency: MTLResidencySet
    let event: MTLSharedEvent
    let counterHeap: MTL4CounterHeap?
    let listener = MTLSharedEventListener()
    var libraries: [MTLLibrary] = []
    var pipelines: [String: MTLComputePipelineState] = [:]
    var eventValue: UInt64 = 0
    var gpuNanoseconds: UInt64 = 0
    var profiledPasses: UInt64 = 0

    init(profile: Bool) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw Failure("Metal device unavailable") }
        guard device.supportsPlacementSparse else { throw Failure("placement sparse buffers unsupported") }
        guard let queue = device.makeMTL4CommandQueue() else { throw Failure("MTL4 queue creation failed") }
        guard let allocator = device.makeCommandAllocator(), let commandBuffer = device.makeCommandBuffer() else {
            throw Failure("MTL4 command resources unavailable")
        }
        let descriptor = MTLResidencySetDescriptor(); descriptor.initialCapacity = 512
        let residency = try device.makeResidencySet(descriptor: descriptor)
        self.device = device; self.queue = queue; self.allocator = allocator; self.commandBuffer = commandBuffer
        self.residency = residency; self.event = device.makeSharedEvent()!
        if profile {
            let counters = MTL4CounterHeapDescriptor(); counters.type = .timestamp; counters.count = 2
            self.counterHeap = try device.makeCounterHeap(descriptor: counters)
        } else { self.counterHeap = nil }
        queue.addResidencySet(residency)
    }

    func add(_ allocation: MTLAllocation) {
        residency.addAllocation(allocation); residency.commit()
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        if let pipeline = pipelines[name] { return pipeline }
        guard let function = libraries.lazy.compactMap({ $0.makeFunction(name: name) }).first else {
            throw Failure("missing kernel \(name)")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        pipelines[name] = pipeline
        return pipeline
    }

    func finish(_ wait: Bool, profilePass: Bool = false) {
        if profilePass, let counterHeap { commandBuffer.writeTimestamp(counterHeap: counterHeap, index: 1) }
        commandBuffer.endCommandBuffer(); queue.commit([commandBuffer]); eventValue += 1
        let value = eventValue; queue.signalEvent(event, value: value)
        if wait {
            let semaphore = DispatchSemaphore(value: 0)
            event.notify(listener, atValue: value) { _, _ in semaphore.signal() }
            semaphore.wait()
            if profilePass, let counterHeap, let data = try? counterHeap.resolveCounterRange(0..<2) {
                let values = data.withUnsafeBytes { bytes in
                    (bytes.loadUnaligned(fromByteOffset: 0, as: UInt64.self),
                     bytes.loadUnaligned(fromByteOffset: 8, as: UInt64.self))
                }
                gpuNanoseconds += values.1 - values.0; profiledPasses += 1
            }
            allocator.reset()
        }
    }
}

@available(macOS 26.4, *)
private final class Pass {
    let runtime: Runtime
    let encoder: MTL4ComputeCommandEncoder
    let constants: MTLBuffer
    var tables: [MTL4ArgumentTable] = []
    var constantOffset = 0

    init(_ runtime: Runtime) throws {
        self.runtime = runtime
        guard let constants = runtime.device.makeBuffer(length: 1 << 20, options: .storageModeShared) else {
            throw Failure("constant buffer allocation failed")
        }
        self.constants = constants; runtime.add(constants)
        runtime.commandBuffer.beginCommandBuffer(allocator: runtime.allocator)
        runtime.commandBuffer.useResidencySet(runtime.residency)
        if let counterHeap = runtime.counterHeap {
            counterHeap.invalidateCounterRange(0..<2)
            runtime.commandBuffer.writeTimestamp(counterHeap: counterHeap, index: 0)
        }
        guard let encoder = runtime.commandBuffer.makeComputeCommandEncoder() else {
            throw Failure("compute encoder creation failed")
        }
        self.encoder = encoder
        encoder.barrier(afterQueueStages: .resourceState, beforeStages: .dispatch, visibilityOptions: .device)
    }

    func inline(_ bytes: UnsafeRawPointer, _ count: Int) -> UInt64 {
        constantOffset = (constantOffset + 15) & ~15
        constants.contents().advanced(by: constantOffset).copyMemory(from: bytes, byteCount: count)
        defer { constantOffset += count }
        return constants.gpuAddress + UInt64(constantOffset)
    }
}

private struct Failure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

@_cdecl("infeng_last_error")
public func infengLastError() -> UnsafePointer<CChar>? {
    (lastError as NSString).utf8String
}

@_cdecl("infeng_runtime_create")
public func infengRuntimeCreate(_ profile: Int32) -> UnsafeMutableRawPointer? {
    guard #available(macOS 26.4, *) else { lastError = "macOS 26.4 is required"; return nil }
    do { return retained(try Runtime(profile: profile != 0)) } catch { lastError = String(describing: error); return nil }
}

@_cdecl("infeng_runtime_release")
@available(macOS 26.4, *)
public func infengRuntimeRelease(_ pointer: UnsafeMutableRawPointer?) {
    if let pointer { Unmanaged<Runtime>.fromOpaque(pointer).release() }
}

@_cdecl("infeng_compile")
@available(macOS 26.4, *)
public func infengCompile(_ runtimePointer: UnsafeMutableRawPointer?, _ source: UnsafePointer<CChar>?) -> Int32 {
    do {
        let runtime: Runtime = object(runtimePointer)
        runtime.libraries.append(try runtime.device.makeLibrary(source: String(cString: source!), options: nil))
        return 0
    } catch { lastError = String(describing: error); return -1 }
}

@_cdecl("infeng_buffer_create")
@available(macOS 26.4, *)
public func infengBufferCreate(_ runtimePointer: UnsafeMutableRawPointer?, _ length: UInt64,
                               _ shared: Int32) -> UnsafeMutableRawPointer? {
    let runtime: Runtime = object(runtimePointer)
    guard let buffer = runtime.device.makeBuffer(length: Int(length), options: shared != 0 ? .storageModeShared : .storageModePrivate)
    else { lastError = "buffer allocation failed"; return nil }
    runtime.add(buffer)
    return retained(Buffer(buffer, runtime: runtime))
}

@_cdecl("infeng_buffer_upload")
@available(macOS 26.4, *)
public func infengBufferUpload(_ runtimePointer: UnsafeMutableRawPointer?, _ source: UnsafeRawPointer?,
                               _ length: UInt64) -> UnsafeMutableRawPointer? {
    let runtime: Runtime = object(runtimePointer)
    guard let staging = runtime.device.makeBuffer(bytes: source!, length: Int(length), options: .storageModeShared),
          let buffer = runtime.device.makeBuffer(length: Int(length), options: .storageModePrivate)
    else { lastError = "upload allocation failed"; return nil }
    runtime.add(staging); runtime.add(buffer)
    runtime.commandBuffer.beginCommandBuffer(allocator: runtime.allocator)
    runtime.commandBuffer.useResidencySet(runtime.residency)
    guard let encoder = runtime.commandBuffer.makeComputeCommandEncoder() else { lastError = "upload encoder failed"; return nil }
    encoder.copy(sourceBuffer: staging, sourceOffset: 0, destinationBuffer: buffer,
                 destinationOffset: 0, size: Int(length))
    encoder.endEncoding(); runtime.finish(true)
    runtime.residency.removeAllocation(staging); runtime.residency.commit()
    return retained(Buffer(buffer, runtime: runtime))
}

@_cdecl("infeng_buffer_write")
@available(macOS 26.4, *)
public func infengBufferWrite(_ runtimePointer: UnsafeMutableRawPointer?, _ bufferPointer: UnsafeMutableRawPointer?,
                              _ offset: UInt64, _ source: UnsafeRawPointer?, _ length: UInt64) -> Int32 {
    let runtime: Runtime = object(runtimePointer), destination: Buffer = object(bufferPointer)
    guard let staging = runtime.device.makeBuffer(bytes: source!, length: Int(length), options: .storageModeShared)
    else { lastError = "staging allocation failed"; return -1 }
    runtime.add(staging); runtime.commandBuffer.beginCommandBuffer(allocator: runtime.allocator)
    runtime.commandBuffer.useResidencySet(runtime.residency)
    guard let encoder = runtime.commandBuffer.makeComputeCommandEncoder() else { lastError = "upload encoder failed"; return -1 }
    encoder.copy(sourceBuffer: staging, sourceOffset: 0, destinationBuffer: destination.value,
                 destinationOffset: Int(offset), size: Int(length))
    encoder.endEncoding(); runtime.finish(true)
    runtime.residency.removeAllocation(staging); runtime.residency.commit(); return 0
}

@_cdecl("infeng_buffer_contents")
@available(macOS 26.4, *)
public func infengBufferContents(_ pointer: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    let buffer: Buffer = object(pointer); return buffer.value.contents()
}

@_cdecl("infeng_buffer_read")
@available(macOS 26.4, *)
public func infengBufferRead(_ runtimePointer: UnsafeMutableRawPointer?, _ bufferPointer: UnsafeMutableRawPointer?,
                             _ offset: UInt64, _ destination: UnsafeMutableRawPointer?, _ length: UInt64) -> Int32 {
    let runtime: Runtime = object(runtimePointer), source: Buffer = object(bufferPointer)
    guard let staging = runtime.device.makeBuffer(length: Int(length), options: .storageModeShared) else {
        lastError = "readback allocation failed"; return -1
    }
    runtime.add(staging); runtime.commandBuffer.beginCommandBuffer(allocator: runtime.allocator)
    runtime.commandBuffer.useResidencySet(runtime.residency)
    guard let encoder = runtime.commandBuffer.makeComputeCommandEncoder() else { lastError = "readback encoder failed"; return -1 }
    encoder.copy(sourceBuffer: source.value, sourceOffset: Int(offset), destinationBuffer: staging,
                 destinationOffset: 0, size: Int(length))
    encoder.endEncoding(); runtime.finish(true)
    destination!.copyMemory(from: staging.contents(), byteCount: Int(length))
    runtime.residency.removeAllocation(staging); runtime.residency.commit(); return 0
}

@_cdecl("infeng_buffer_release")
@available(macOS 26.4, *)
public func infengBufferRelease(_ pointer: UnsafeMutableRawPointer?) {
    if let pointer { Unmanaged<Buffer>.fromOpaque(pointer).release() }
}

@_cdecl("infeng_sparse_create")
@available(macOS 26.4, *)
public func infengSparseCreate(_ runtimePointer: UnsafeMutableRawPointer?, _ maxContext: Int32) -> UnsafeMutableRawPointer? {
    do { let runtime: Runtime = object(runtimePointer); return retained(try SparseCache(runtime, maxContext: Int(maxContext))) }
    catch { lastError = String(describing: error); return nil }
}

@_cdecl("infeng_sparse_buffer")
@available(macOS 26.4, *)
public func infengSparseBuffer(_ cachePointer: UnsafeMutableRawPointer?, _ index: Int32) -> UnsafeMutableRawPointer? {
    let cache: SparseCache = object(cachePointer)
    return Unmanaged.passUnretained(cache.buffers[Int(index)]).toOpaque()
}

@_cdecl("infeng_sparse_ensure")
@available(macOS 26.4, *)
public func infengSparseEnsure(_ cachePointer: UnsafeMutableRawPointer?, _ tokens: Int32) -> Int32 {
    do { let cache: SparseCache = object(cachePointer); try cache.ensure(tokens: Int(tokens)); return 0 }
    catch { lastError = String(describing: error); return -1 }
}

@_cdecl("infeng_sparse_mapped_bytes")
@available(macOS 26.4, *)
public func infengSparseMappedBytes(_ cachePointer: UnsafeMutableRawPointer?) -> UInt64 {
    let cache: SparseCache = object(cachePointer); return UInt64(cache.mappedPages * SparseCache.count * pageSize)
}

@_cdecl("infeng_profile_gpu_ns")
@available(macOS 26.4, *)
public func infengProfileGPUNanoseconds(_ runtimePointer: UnsafeMutableRawPointer?) -> UInt64 {
    let runtime: Runtime = object(runtimePointer); return runtime.gpuNanoseconds
}

@_cdecl("infeng_profile_passes")
@available(macOS 26.4, *)
public func infengProfilePasses(_ runtimePointer: UnsafeMutableRawPointer?) -> UInt64 {
    let runtime: Runtime = object(runtimePointer); return runtime.profiledPasses
}

@_cdecl("infeng_sparse_release")
@available(macOS 26.4, *)
public func infengSparseRelease(_ pointer: UnsafeMutableRawPointer?) {
    if let pointer { Unmanaged<SparseCache>.fromOpaque(pointer).release() }
}

@_cdecl("infeng_pass_begin")
@available(macOS 26.4, *)
public func infengPassBegin(_ runtimePointer: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    do { let runtime: Runtime = object(runtimePointer); return retained(try Pass(runtime)) }
    catch { lastError = String(describing: error); return nil }
}

@_cdecl("infeng_dispatch")
@available(macOS 26.4, *)
public func infengDispatch(_ passPointer: UnsafeMutableRawPointer?, _ name: UnsafePointer<CChar>?,
                           _ buffers: UnsafePointer<UnsafeMutableRawPointer?>?, _ offsets: UnsafePointer<UInt64>?,
                           _ count: Int32, _ inlineBytes: UnsafeRawPointer?, _ inlineSizes: UnsafePointer<UInt32>?,
                           _ inlineCount: Int32, _ tx: UInt64, _ ty: UInt64, _ tz: UInt64,
                           _ gx: UInt64, _ gy: UInt64, _ gz: UInt64) -> Int32 {
    do {
        let pass: Pass = object(passPointer), descriptor = MTL4ArgumentTableDescriptor()
        descriptor.maxBufferBindCount = Int(count + inlineCount); descriptor.initializeBindings = true
        let table = try pass.runtime.device.makeArgumentTable(descriptor: descriptor)
        for i in 0..<Int(count) {
            let buffer: Buffer = object(buffers![i]); table.setAddress(buffer.value.gpuAddress + offsets![i], index: i)
        }
        var cursor = 0
        for i in 0..<Int(inlineCount) {
            let size = Int(inlineSizes![i]), address = pass.inline(inlineBytes!.advanced(by: cursor), size)
            table.setAddress(address, index: Int(count) + i); cursor += size
        }
        pass.encoder.setComputePipelineState(try pass.runtime.pipeline(String(cString: name!)))
        pass.encoder.setArgumentTable(table)
        pass.tables.append(table)
        pass.encoder.dispatchThreads(threadsPerGrid: MTLSize(width: Int(tx), height: Int(ty), depth: Int(tz)),
                                     threadsPerThreadgroup: MTLSize(width: Int(gx), height: Int(gy), depth: Int(gz)))
        pass.encoder.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: .device)
        return 0
    } catch { lastError = String(describing: error); return -1 }
}

@_cdecl("infeng_pass_commit")
@available(macOS 26.4, *)
public func infengPassCommit(_ passPointer: UnsafeMutableRawPointer?, _ wait: Int32) -> Int32 {
    let pass: Pass = object(passPointer), runtime = pass.runtime
    pass.encoder.endEncoding(); runtime.finish(wait != 0, profilePass: true)
    if wait != 0 { runtime.residency.removeAllocation(pass.constants); runtime.residency.commit() }
    Unmanaged<Pass>.fromOpaque(passPointer!).release()
    return 0
}
