from __future__ import annotations

import ctypes
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / ".build/release/libInfengMetal.dylib"


def _load():
    if not LIB.exists():
        subprocess.run(["swift", "build", "-c", "release"], cwd=ROOT, check=True)
    lib = ctypes.CDLL(LIB)
    lib.infeng_last_error.restype = ctypes.c_char_p
    lib.infeng_runtime_create.argtypes = (ctypes.c_int32,)
    lib.infeng_runtime_create.restype = ctypes.c_void_p
    lib.infeng_runtime_release.argtypes = (ctypes.c_void_p,)
    lib.infeng_compile.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
    lib.infeng_buffer_create.argtypes = (ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int32)
    lib.infeng_buffer_create.restype = ctypes.c_void_p
    lib.infeng_buffer_upload.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64)
    lib.infeng_buffer_upload.restype = ctypes.c_void_p
    lib.infeng_buffer_write.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p,
                                        ctypes.c_uint64)
    lib.infeng_buffer_contents.argtypes = (ctypes.c_void_p,)
    lib.infeng_buffer_contents.restype = ctypes.c_void_p
    lib.infeng_buffer_read.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p,
                                       ctypes.c_uint64)
    lib.infeng_buffer_release.argtypes = (ctypes.c_void_p,)
    lib.infeng_sparse_create.argtypes = (ctypes.c_void_p, ctypes.c_int32)
    lib.infeng_sparse_create.restype = ctypes.c_void_p
    lib.infeng_sparse_buffer.argtypes = (ctypes.c_void_p, ctypes.c_int32)
    lib.infeng_sparse_buffer.restype = ctypes.c_void_p
    lib.infeng_sparse_ensure.argtypes = (ctypes.c_void_p, ctypes.c_int32)
    lib.infeng_sparse_mapped_bytes.argtypes = (ctypes.c_void_p,)
    lib.infeng_sparse_mapped_bytes.restype = ctypes.c_uint64
    lib.infeng_sparse_release.argtypes = (ctypes.c_void_p,)
    lib.infeng_profile_gpu_ns.argtypes = (ctypes.c_void_p,)
    lib.infeng_profile_gpu_ns.restype = ctypes.c_uint64
    lib.infeng_profile_passes.argtypes = (ctypes.c_void_p,)
    lib.infeng_profile_passes.restype = ctypes.c_uint64
    lib.infeng_pass_begin.argtypes = (ctypes.c_void_p,)
    lib.infeng_pass_begin.restype = ctypes.c_void_p
    lib.infeng_dispatch.argtypes = (
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int32,
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
    )
    lib.infeng_pass_commit.argtypes = (ctypes.c_void_p, ctypes.c_int32)
    return lib


_LIB = _load()


def _check(value):
    if value in (None, 0):
        raise RuntimeError((_LIB.infeng_last_error() or b"Metal operation failed").decode())
    return value


def _status(value):
    if value != 0: raise RuntimeError((_LIB.infeng_last_error() or b"Metal operation failed").decode())


DTYPE_SIZE = {"u8": 1, "i32": 4, "i64": 8, "f16": 2, "f32": 4}


@dataclass(slots=True)
class Tensor:
    runtime: Runtime
    handle: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    offset: int = 0
    owner: bool = True

    @property
    def numel(self):
        n = 1
        for dim in self.shape: n *= dim
        return n

    @property
    def nbytes(self): return self.numel * DTYPE_SIZE[self.dtype]

    def view(self, shape, strides=None, offset=0):
        shape = tuple(shape)
        if strides is None:
            stride, strides = 1, []
            for dim in reversed(shape): strides.append(stride); stride *= dim
            strides = tuple(reversed(strides))
        return Tensor(self.runtime, self.handle, shape, tuple(strides), self.dtype,
                      self.offset + offset * DTYPE_SIZE[self.dtype], False)

    def __del__(self):
        if self.owner and getattr(self, "handle", 0): _LIB.infeng_buffer_release(ctypes.c_void_p(self.handle))


class Runtime:
    def __init__(self, profile=False):
        self.profile, self.allocations, self.allocated_bytes, self.dispatches = profile, 0, 0, 0
        self.handle = _check(_LIB.infeng_runtime_create(profile))
        for path in sorted((ROOT / "backend/metal").glob("*.metal")):
            _status(_LIB.infeng_compile(self.handle, path.read_bytes()))

    def empty(self, shape, dtype="f16", shared=False):
        shape = tuple(shape); n = DTYPE_SIZE[dtype]
        for dim in shape: n *= dim
        handle = _check(_LIB.infeng_buffer_create(self.handle, n, shared))
        self.allocations += 1; self.allocated_bytes += n
        return Tensor(self, handle, shape, self._strides(shape), dtype)

    def upload(self, data, shape, dtype):
        view = memoryview(data).cast("B")
        source = (ctypes.c_ubyte * len(view)).from_buffer_copy(view)
        handle = _check(_LIB.infeng_buffer_upload(self.handle, source, len(view)))
        self.allocations += 1; self.allocated_bytes += len(view)
        return Tensor(self, handle, tuple(shape), self._strides(shape), dtype)

    def shared(self, data, shape, dtype):
        tensor = self.empty(shape, dtype, shared=True)
        view = memoryview(data).cast("B")
        if len(view) != tensor.nbytes: raise ValueError(f"expected {tensor.nbytes} bytes, got {len(view)}")
        ctypes.memmove(_check(_LIB.infeng_buffer_contents(tensor.handle)), bytes(view), len(view))
        return tensor

    def upload_into(self, tensor, offset, data):
        view = memoryview(data).cast("B"); source = (ctypes.c_ubyte * len(view)).from_buffer_copy(view)
        if offset + len(view) > tensor.nbytes: raise ValueError("arena upload exceeds buffer")
        _status(_LIB.infeng_buffer_write(self.handle, tensor.handle, tensor.offset + offset, source, len(view)))

    def read_i32(self, tensor):
        if tensor.dtype != "i32" or tensor.numel != 1: raise ValueError("readback must be one i32")
        return ctypes.c_int32.from_address(_check(_LIB.infeng_buffer_contents(tensor.handle))).value

    def write(self, tensor, data):
        view = memoryview(data).cast("B")
        if len(view) > tensor.nbytes: raise ValueError(f"write requires {len(view)} bytes, buffer has {tensor.nbytes}")
        ctypes.memmove(_check(_LIB.infeng_buffer_contents(tensor.handle)) + tensor.offset, bytes(view), len(view))

    def read(self, tensor):
        output = (ctypes.c_ubyte * tensor.nbytes)()
        _status(_LIB.infeng_buffer_read(self.handle, tensor.handle, tensor.offset, output, tensor.nbytes))
        return bytes(output)

    def begin(self): return Pass(self, _check(_LIB.infeng_pass_begin(self.handle)))

    def sparse_cache(self, max_context=65536): return SparseCache(self, max_context)

    @staticmethod
    def _strides(shape):
        stride, out = 1, []
        for dim in reversed(shape): out.append(stride); stride *= dim
        return tuple(reversed(out))

    def close(self):
        if self.handle: _LIB.infeng_runtime_release(self.handle); self.handle = 0

    def counters(self):
        if not self.profile: return {}
        return {"gpu_time_ns": _LIB.infeng_profile_gpu_ns(self.handle),
                "passes": _LIB.infeng_profile_passes(self.handle), "dispatches": self.dispatches,
                "allocations": self.allocations, "allocated_bytes": self.allocated_bytes}


class Pass:
    FORMATS = {"i": "i", "I": "I", "q": "q", "Q": "Q", "f": "f", "?": "?"}

    def __init__(self, runtime, handle): self.runtime, self.handle = runtime, handle

    def dispatch(self, name, tensors, scalars, threads, group):
        self.runtime.dispatches += 1
        buffers = (ctypes.c_void_p * len(tensors))(*(x.handle for x in tensors))
        offsets = (ctypes.c_uint64 * len(tensors))(*(x.offset for x in tensors))
        raw = b"".join(struct.pack("<" + fmt, value) for fmt, value in scalars)
        sizes = (ctypes.c_uint32 * len(scalars))(*(struct.calcsize("<" + fmt) for fmt, _ in scalars))
        payload = ctypes.create_string_buffer(raw)
        _status(_LIB.infeng_dispatch(self.handle, name.encode(), buffers, offsets, len(tensors), payload, sizes,
                                     len(scalars), *threads, *group))

    def commit(self, wait=True): _status(_LIB.infeng_pass_commit(self.handle, wait)); self.handle = 0


class SparseCache:
    def __init__(self, runtime, max_context):
        self.runtime, self.max_context = runtime, max_context
        self.handle = _check(_LIB.infeng_sparse_create(runtime.handle, max_context))
        shape = (max_context // 128, 4, 128, 256)
        self.buffers = [Tensor(runtime, _LIB.infeng_sparse_buffer(self.handle, i), shape,
                               Runtime._strides(shape), "f16", owner=False) for i in range(16)]

    def ensure(self, tokens): _status(_LIB.infeng_sparse_ensure(self.handle, tokens))

    @property
    def mapped_bytes(self): return _LIB.infeng_sparse_mapped_bytes(self.handle)

    def close(self):
        if self.handle: _LIB.infeng_sparse_release(self.handle); self.handle = 0
