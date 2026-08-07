from __future__ import annotations

import ctypes
import subprocess
import weakref
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / ".build/libinfeng.dylib"
KERNELS = ROOT / "backend/metal/kernel"
SOURCES = [ROOT / "backend/metal/device.cpp", ROOT / "model/qwen35_weights.cpp", ROOT / "model/qwen35.cpp"]


def _build():
    headers = [*ROOT.glob("backend/metal/*.hpp"), *ROOT.glob("model/*.hpp"),
               *(ROOT / "third_party/metal-cpp").rglob("*.hpp")]
    if LIB.exists() and all(path.stat().st_mtime <= LIB.stat().st_mtime for path in [*SOURCES, *headers]): return
    LIB.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["xcrun", "clang++", "-std=c++17", "-O3", "-DNDEBUG", "-fblocks", "-fvisibility=hidden",
                    "-DMETALCPP_SYMBOL_VISIBILITY_HIDDEN", "-dynamiclib",
                    *(str(path) for path in SOURCES), "-I", str(ROOT / "third_party/metal-cpp"), "-I", str(ROOT),
                    "-framework", "Foundation", "-framework", "Metal", "-o", str(LIB)], check=True)


class _Counters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("gpu_time_ns", "passes")]


def _load():
    _build(); lib = ctypes.CDLL(LIB)
    lib.infeng_last_error.restype = ctypes.c_char_p
    lib.infeng_model_create.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_int32)
    lib.infeng_model_create.restype = ctypes.c_void_p
    lib.infeng_model_release.argtypes = (ctypes.c_void_p,)
    lib.infeng_session_create.argtypes = (ctypes.c_void_p,); lib.infeng_session_create.restype = ctypes.c_void_p
    lib.infeng_session_release.argtypes = (ctypes.c_void_p,)
    lib.infeng_forward.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32,
                                   ctypes.c_float, ctypes.c_float, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32))
    lib.infeng_session_length.argtypes = (ctypes.c_void_p,); lib.infeng_session_length.restype = ctypes.c_uint64
    lib.infeng_session_mapped_bytes.argtypes = (ctypes.c_void_p,); lib.infeng_session_mapped_bytes.restype = ctypes.c_uint64
    for name in ("infeng_model_parameter_count", "infeng_model_weight_bytes", "infeng_model_vocab_size"):
        function = getattr(lib, name); function.argtypes = (ctypes.c_void_p,); function.restype = ctypes.c_uint64
    lib.infeng_model_counters.argtypes = (ctypes.c_void_p, ctypes.POINTER(_Counters))
    lib.infeng_model_kernel_counter_count.argtypes = (ctypes.c_void_p,); lib.infeng_model_kernel_counter_count.restype = ctypes.c_uint32
    lib.infeng_model_kernel_counter.argtypes = (ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char_p),
                                                ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_uint64),
                                                ctypes.POINTER(ctypes.c_uint64))
    return lib


_LIB = _load()


def _error(): return (_LIB.infeng_last_error() or b"native Metal operation failed").decode()
def _check(value):
    if not value: raise RuntimeError(_error())
    return value
def _status(value):
    if value: raise RuntimeError(_error())


class NativeSession:
    def __init__(self, model: NativeModel):
        self.handle, self.closed = _check(_LIB.infeng_session_create(model.handle)), False

    def forward(self, ids, temperature=0.0, top_p=1.0, top_k=None):
        payload = array("i", ids); values = (ctypes.c_int32 * len(payload)).from_buffer(payload); output = ctypes.c_int32()
        _status(_LIB.infeng_forward(self.handle, values, len(payload), temperature, top_p, top_k or 0,
                                    ctypes.byref(output)))
        return output.value

    @property
    def length(self): return _LIB.infeng_session_length(self.handle)

    @property
    def mapped_bytes(self): return _LIB.infeng_session_mapped_bytes(self.handle)

    def close(self):
        if not self.closed: _LIB.infeng_session_release(self.handle); self.handle, self.closed = 0, True

    def __del__(self): self.close()


class NativeModel:
    def __init__(self, weights: str | Path, *, max_context=65536, profile=False):
        self._session = None
        self.handle = 0
        self.handle = _check(_LIB.infeng_model_create(str(weights).encode(), str(KERNELS).encode(), max_context, profile))

    def session(self):
        session = NativeSession(self); self._session = weakref.ref(session); return session

    @property
    def parameter_count(self): return _LIB.infeng_model_parameter_count(self.handle)

    @property
    def weight_bytes(self): return _LIB.infeng_model_weight_bytes(self.handle)

    @property
    def vocab_size(self): return _LIB.infeng_model_vocab_size(self.handle)

    def counters(self):
        output = _Counters(); _status(_LIB.infeng_model_counters(self.handle, ctypes.byref(output)))
        return {name: getattr(output, name) for name, _ in output._fields_}

    def kernel_counters(self):
        counters = []
        for index in range(_LIB.infeng_model_kernel_counter_count(self.handle)):
            phase = ctypes.c_char_p(); name = ctypes.c_char_p(); gpu_time_ns = ctypes.c_uint64(); launches = ctypes.c_uint64()
            _status(_LIB.infeng_model_kernel_counter(self.handle, index, ctypes.byref(phase), ctypes.byref(name),
                                                     ctypes.byref(gpu_time_ns), ctypes.byref(launches)))
            counters.append({"phase": phase.value.decode(), "name": name.value.decode(),
                             "gpu_time_ns": gpu_time_ns.value, "launches": launches.value})
        return counters

    def close(self):
        live = self._session() if self._session else None
        if live is not None: live.close()
        if getattr(self, "handle", 0): _LIB.infeng_model_release(self.handle); self.handle = 0

    def __del__(self): self.close()
