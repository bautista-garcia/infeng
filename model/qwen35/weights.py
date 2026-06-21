from __future__ import annotations

import mmap
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


GGUF_TYPE_NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 23: "IQ4_XS", 30: "BF16"}
GGUF_NATIVE_DTYPES = {0: torch.float32, 1: torch.float16, 30: torch.bfloat16}
GGUF_BLOCK = {"Q8_0": (32, 34), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "IQ4_XS": (256, 136)}


def _string(f) -> str:
    size = struct.unpack("<Q", f.read(8))[0]
    return f.read(size).decode()


def _value(f, typ: int):
    if typ in (0, 1):
        return struct.unpack("<B" if typ == 0 else "<b", f.read(1))[0]
    if typ in (2, 3):
        return struct.unpack("<H" if typ == 2 else "<h", f.read(2))[0]
    if typ in (4, 5, 6, 7):
        return struct.unpack(("<I", "<i", "<f", "<?")[typ - 4], f.read(4 if typ != 7 else 1))[0]
    if typ == 8:
        return _string(f)
    if typ in (10, 11, 12):
        return struct.unpack(("<Q", "<q", "<d")[typ - 10], f.read(8))[0]
    if typ == 9:
        item_type, size = struct.unpack("<IQ", f.read(12))
        return [_value(f, item_type) for _ in range(size)]
    raise ValueError(f"unknown GGUF metadata type: {typ}")


def _skip_value(f, typ: int):
    if typ in (0, 1, 7):
        f.seek(1, 1)
    elif typ in (2, 3):
        f.seek(2, 1)
    elif typ in (4, 5, 6):
        f.seek(4, 1)
    elif typ == 8:
        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
    elif typ in (10, 11, 12):
        f.seek(8, 1)
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
    data: torch.Tensor
    offset: int
    nbytes: int
    block_size: int | None = None
    block_bytes: int | None = None
    transform: str | None = None

    @property
    def type_name(self) -> str:
        return GGUF_TYPE_NAMES.get(self.gguf_type, str(self.gguf_type))

    @property
    def torch_dtype(self) -> torch.dtype | None:
        return GGUF_NATIVE_DTYPES.get(self.gguf_type)


def _target(key: str) -> tuple[str, str | None] | None:
    if key == "token_embd.weight":
        return "model.embed_tokens.weight", None
    if key == "output.weight":
        return "lm_head.weight", None
    if key in ("output_norm.weight", "model.output_norm.weight"):
        return "model.norm.weight", None
    parts = key.split(".")
    if len(parts) < 3 or parts[0] != "blk":
        return None
    names = {
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
    attr = names.get(".".join(parts[2:]))
    return (f"model.layers.{parts[1]}.{attr}", "conv1d" if attr == "linear_attn.conv1d_weight" else "neg_log" if attr == "linear_attn.A_log" else None) if attr else None


def _set(root: torch.nn.Module, dotted: str, value: QuantWeight):
    obj = root
    for part in dotted.split(".")[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    setattr(obj, dotted.rsplit(".", 1)[1], value)


def load_weights(model: torch.nn.Module, path: str | Path, strict: bool = False) -> dict[str, Any]:
    f, _, data_start, infos = _gguf(Path(path))
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    handles, loaded, unexpected = getattr(model, "_gguf_handles", []), set(), []
    handles.append((f, mm)); model._gguf_handles = handles
    device, model_dtype = getattr(model, "device", None), getattr(model, "dtype", None)
    for source_key, shape, typ, offset in infos:
        mapped = _target(source_key)
        if mapped is None:
            unexpected.append(source_key); continue
        target_key, transform = mapped
        if not hasattr(model.get_submodule(target_key.rsplit(".", 1)[0]), target_key.rsplit(".", 1)[1]):
            unexpected.append(source_key); continue
        dtype = GGUF_NATIVE_DTYPES.get(typ)
        if dtype:
            numel = int(np.prod(shape))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                data = torch.frombuffer(mm, dtype=dtype, count=numel, offset=data_start + offset)
            data = data.reshape(shape)
            if transform == "neg_log":
                data = torch.log(-data.float())
            elif transform == "conv1d":
                data = data[:, None, :].to(dtype=model_dtype or data.dtype)
            shape = tuple(data.shape)
            nbytes, block_size, block_bytes = numel * data.element_size(), None, None
        else:
            block_size, block_bytes = GGUF_BLOCK[GGUF_TYPE_NAMES[typ]]
            nbytes = int(np.prod(shape[:-1])) * (shape[-1] // block_size) * block_bytes
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                data = torch.frombuffer(mm, dtype=torch.uint8, count=nbytes, offset=data_start + offset)
        _set(model, target_key, QuantWeight(source_key, shape, typ, data.to(device) if device else data,
                                            data_start + offset, nbytes, block_size, block_bytes, transform))
        loaded.add(target_key)
    missing = sorted(k for k in _expected(model) if k not in loaded)
    if strict and (missing or unexpected):
        raise RuntimeError(f"missing={missing[:20]} unexpected={unexpected[:20]}")
    return {"missing": missing, "unexpected": unexpected}


def _expected(model: torch.nn.Module) -> set[str]:
    out = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i, layer in enumerate(model.model.layers):
        p = f"model.layers.{i}."
        out |= {p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
                p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight", p + "mlp.down_proj.weight"}
        if layer.layer_type == "full_attention":
            out |= {p + f"self_attn.{x}.weight" for x in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")}
        else:
            out |= {p + f"linear_attn.{x}" for x in ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight",
                                                      "in_proj_a.weight", "conv1d_weight", "dt_bias", "A_log",
                                                      "norm.weight", "out_proj.weight")}
    return out
