from __future__ import annotations

import mmap
import math
import struct
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.metal.runtime import Runtime, Tensor

GGUF_TYPE_NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 23: "IQ4_XS", 30: "BF16"}
GGUF_NATIVE_DTYPES = {0: "f32", 1: "f16"}
GGUF_BLOCK = {"Q8_0": (32, 34), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "IQ4_XS": (256, 136)}
GGUF_VALUE_FORMATS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
                      10: "<Q", 11: "<q", 12: "<d"}
GGUF_TARGETS = {"token_embd.weight": "model.embed_tokens.weight", "output.weight": "lm_head.weight",
                "output_norm.weight": "model.norm.weight", "model.output_norm.weight": "model.norm.weight"}
GGUF_BLOCK_TARGETS = {
    "attn_norm.weight": "input_layernorm.weight", "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight", "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight", "ffn_down.weight": "mlp.down_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight", "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight", "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d_weight", "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight", "ssm_dt.bias": "linear_attn.dt_bias", "ssm_a": "linear_attn.A_log",
}
GGUF_TRANSFORMS = {"linear_attn.conv1d_weight": "conv1d", "linear_attn.A_log": "neg_log"}


def _string(f) -> str:
    size = struct.unpack("<Q", f.read(8))[0]
    return f.read(size).decode()


def _value(f, typ: int):
    if fmt := GGUF_VALUE_FORMATS.get(typ):
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if typ == 8:
        return _string(f)
    if typ == 9:
        item_type, size = struct.unpack("<IQ", f.read(12))
        return [_value(f, item_type) for _ in range(size)]
    raise ValueError(f"unknown GGUF metadata type: {typ}")


def _skip_value(f, typ: int):
    if fmt := GGUF_VALUE_FORMATS.get(typ):
        f.seek(struct.calcsize(fmt), 1)
    elif typ == 8:
        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
    elif typ == 9:
        item_type, size = struct.unpack("<IQ", f.read(12))
        for _ in range(size):
            _skip_value(f, item_type)
    else:
        raise ValueError(f"unknown GGUF metadata type: {typ}")


def _gguf(file: Path):
    f = open(file, "rb")
    if f.read(4) != b"GGUF":
        raise ValueError(f"{file} is not a GGUF file")
    version, tensor_count, metadata_count = struct.unpack("<IQQ", f.read(20))
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version: {version}")
    metadata, alignment = {}, 32
    for _ in range(metadata_count):
        key, typ = _string(f), struct.unpack("<I", f.read(4))[0]
        if key == "tokenizer.ggml.tokens":
            metadata[key] = None; _skip_value(f, typ)
        else:
            metadata[key] = _value(f, typ)
        if key == "general.alignment":
            alignment = int(metadata[key])
    infos = []
    for _ in range(tensor_count):
        name, n_dims = _string(f), struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims))
        typ, offset = struct.unpack("<IQ", f.read(12))
        infos.append((name, tuple(reversed(dims)), typ, offset))
    return f, metadata, (f.tell() + alignment - 1) // alignment * alignment, infos


def config_from_gguf(path: str | Path) -> dict[str, Any]:
    f, m, _, infos = _gguf(Path(path)); f.close()
    prefix, n = "qwen35.", int(m["qwen35.block_count"])
    interval = int(m.get(prefix + "full_attention_interval", 4))
    return {
        "attention_bias": False, "attention_dropout": 0.0, "attn_output_gate": True, "dtype": "float16",
        "eos_token_id": int(m.get("tokenizer.ggml.eos_token_id", 248044)), "full_attention_interval": interval,
        "head_dim": int(m[prefix + "attention.key_length"]), "hidden_act": "silu",
        "hidden_size": int(m[prefix + "embedding_length"]), "intermediate_size": int(m[prefix + "feed_forward_length"]),
        "layer_types": ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)],
        "linear_conv_kernel_dim": int(m[prefix + "ssm.conv_kernel"]), "linear_key_head_dim": int(m[prefix + "ssm.state_size"]),
        "linear_num_key_heads": int(m[prefix + "ssm.group_count"]), "linear_num_value_heads": int(m[prefix + "ssm.time_step_rank"]),
        "linear_value_head_dim": int(m[prefix + "ssm.state_size"]), "max_position_embeddings": int(m[prefix + "context_length"]),
        "mlp_only_layers": [], "model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False, "num_attention_heads": int(m[prefix + "attention.head_count"]),
        "num_hidden_layers": n, "num_key_value_heads": int(m[prefix + "attention.head_count_kv"]),
        "rms_norm_eps": float(m[prefix + "attention.layer_norm_rms_epsilon"]), "tie_word_embeddings": False,
        "use_cache": True, "vocab_size": next(shape[0] for name, shape, _, _ in infos if name == "token_embd.weight"),
        "mamba_ssm_dtype": "float32", "rope_parameters": {"mrope_interleaved": True,
        "mrope_section": list(m[prefix + "rope.dimension_sections"][:3]), "rope_type": "default",
        "rope_theta": float(m[prefix + "rope.freq_base"]),
        "partial_rotary_factor": float(m[prefix + "rope.dimension_count"]) / float(m[prefix + "attention.key_length"])},
    }


@dataclass
class QuantWeight:
    name: str
    shape: tuple[int, ...]
    gguf_type: int
    data: Tensor
    offset: int
    nbytes: int
    block_size: int | None = None
    block_bytes: int | None = None
    transform: str | None = None

    @property
    def type_name(self) -> str:
        return GGUF_TYPE_NAMES.get(self.gguf_type, str(self.gguf_type))

    @property
    def native_dtype(self) -> str | None:
        return GGUF_NATIVE_DTYPES.get(self.gguf_type)


def _target(key: str) -> tuple[str, str | None] | None:
    if key in GGUF_TARGETS:
        return GGUF_TARGETS[key], None
    parts = key.split(".")
    if len(parts) < 3 or parts[0] != "blk":
        return None
    attr = GGUF_BLOCK_TARGETS.get(".".join(parts[2:]))
    return (f"model.layers.{parts[1]}.{attr}", GGUF_TRANSFORMS.get(attr)) if attr else None


def load_weights(path: str | Path, runtime: Runtime) -> tuple[dict[str, QuantWeight], Tensor]:
    load_t0 = time.perf_counter()
    f, _, data_start, infos = _gguf(Path(path))
    mm, weights = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY), {}
    selected, arena_size = [], 0
    for source_key, shape, typ, offset in infos:
        mapped = _target(source_key)
        if mapped is None: continue
        target_key, transform = mapped
        if typ in GGUF_NATIVE_DTYPES:
            nbytes = math.prod(shape) * (2 if transform == "conv1d" or typ == 1 else 4)
        else:
            block_size, block_bytes = GGUF_BLOCK[GGUF_TYPE_NAMES[typ]]
            nbytes = math.prod(shape[:-1]) * (shape[-1] // block_size) * block_bytes
        arena_size = (arena_size + 255) & ~255; selected.append((source_key, shape, typ, offset, target_key,
                                                                 transform, nbytes, arena_size))
        arena_size += nbytes
    arena = runtime.empty((arena_size,), "u8")
    for source_key, shape, typ, offset, target_key, transform, expected_nbytes, arena_offset in selected:
        dtype = GGUF_NATIVE_DTYPES.get(typ)
        start = data_start + offset
        if dtype is not None:
            numel, itemsize = math.prod(shape), 4 if dtype == "f32" else 2
            view = memoryview(mm)[start:start + numel * itemsize]
            if transform == "neg_log":
                values = array("f"); values.frombytes(view); values = array("f", (math.log(-x) for x in values))
                view = memoryview(values)
            if transform == "conv1d":
                values = struct.unpack(f"<{numel}f", view)
                view = memoryview(struct.pack(f"<{numel}e", *values)); dtype, itemsize = "f16", 2
                shape = (shape[0], 1, shape[1])
            nbytes, block_size, block_bytes = numel * itemsize, None, None
        else:
            block_size, block_bytes = GGUF_BLOCK[GGUF_TYPE_NAMES[typ]]
            nbytes = math.prod(shape[:-1]) * (shape[-1] // block_size) * block_bytes
            view = memoryview(mm)[start:start + nbytes]
        if nbytes != expected_nbytes: raise ValueError(f"storage size mismatch for {source_key}")
        runtime.upload_into(arena, arena_offset, view)
        data = arena.view((nbytes,), offset=arena_offset)
        weights[target_key] = QuantWeight(source_key, shape, typ, data, start, nbytes,
                                          block_size, block_bytes, transform)
        del view
    mm.close(); f.close()
    print(f"GGUF weights loaded in {time.perf_counter() - load_t0:.3f}s")
    return weights, arena
