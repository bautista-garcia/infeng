from __future__ import annotations

import json
import mmap
import struct
import warnings
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


GGUF_TYPES = {0: torch.float32, 1: torch.float16, 30: torch.bfloat16}


def _files(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = path / name
        if index.exists():
            with open(index) as f:
                return sorted({path / file for file in json.load(f)["weight_map"].values()})
    return sorted(path.glob("*.safetensors")) or sorted(path.glob("*.bin")) or sorted(path.glob("*.gguf"))


def _key(key: str, target: set[str]) -> str | None:
    for prefix in ("language_model.", "model.language_model.", "text_model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    candidates = [key]
    if key.startswith("model."):
        candidates.append(key[6:])
    if key.startswith("lm_head.") and "lm_head.linear." + key[8:] in target:
        candidates.append("lm_head.linear." + key[8:])
    suffixes = ("proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                "out_proj", "lm_head")
    for suffix in suffixes:
        candidates.append(key.replace(f".{suffix}.weight", f".{suffix}.linear.weight"))
        candidates.append(key.replace(f".{suffix}.bias", f".{suffix}.linear.bias"))
    for candidate in candidates:
        if candidate in target:
            return candidate
    return None


def _string(f) -> str:
    size = struct.unpack("<Q", f.read(8))[0]
    return f.read(size).decode()


def _skip_value(f, typ: int):
    if typ in (0, 1):
        f.seek(1, 1)
    elif typ in (2, 3):
        f.seek(2, 1)
    elif typ in (4, 5, 6):
        f.seek(4, 1)
    elif typ in (7,):
        f.seek(1, 1)
    elif typ in (8,):
        _string(f)
    elif typ in (10, 11, 12):
        f.seek(8, 1)
    elif typ == 9:
        item_type, size = struct.unpack("<IQ", f.read(12))
        for _ in range(size):
            _skip_value(f, item_type)
    else:
        raise ValueError(f"unknown GGUF metadata type: {typ}")


def _gguf_tensors(file: Path) -> dict[str, torch.Tensor]:
    with open(file, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{file} is not a GGUF file")
        version, tensor_count, metadata_count = struct.unpack("<IQQ", f.read(20))
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version: {version}")
        alignment = 32
        for _ in range(metadata_count):
            key, typ = _string(f), struct.unpack("<I", f.read(4))[0]
            if key == "general.alignment" and typ in (4, 10):
                alignment = struct.unpack("<I" if typ == 4 else "<Q", f.read(4 if typ == 4 else 8))[0]
            else:
                _skip_value(f, typ)
        infos = []
        for _ in range(tensor_count):
            name, n_dims = _string(f), struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims))
            typ, offset = struct.unpack("<IQ", f.read(12))
            infos.append((name, tuple(reversed(dims)), typ, offset))
        data_start = (f.tell() + alignment - 1) // alignment * alignment
        tensors = {}
        for name, shape, typ, offset in infos:
            if typ not in GGUF_TYPES:
                continue
            dtype = GGUF_TYPES[typ]
            size = torch.empty(shape, dtype=dtype).numel() * torch.empty((), dtype=dtype).element_size()
            f.seek(data_start + offset)
            tensors[name] = torch.frombuffer(f.read(size), dtype=dtype).clone().reshape(shape)
        return tensors


def _gguf_infos(file: Path):
    f = open(file, "rb")
    if f.read(4) != b"GGUF":
        raise ValueError(f"{file} is not a GGUF file")
    version, tensor_count, metadata_count = struct.unpack("<IQQ", f.read(20))
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version: {version}")
    alignment = 32
    for _ in range(metadata_count):
        key, typ = _string(f), struct.unpack("<I", f.read(4))[0]
        if key == "general.alignment" and typ in (4, 10):
            alignment = struct.unpack("<I" if typ == 4 else "<Q", f.read(4 if typ == 4 else 8))[0]
        else:
            _skip_value(f, typ)
    infos = []
    for _ in range(tensor_count):
        name, n_dims = _string(f), struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims))
        typ, offset = struct.unpack("<IQ", f.read(12))
        infos.append((name, tuple(reversed(dims)), typ, offset))
    return f, (f.tell() + alignment - 1) // alignment * alignment, infos


def _gguf_key(key: str, target: set[str]) -> str | None:
    parts = key.split(".")
    if key == "token_embd.weight":
        return "model.embed_tokens.weight"
    if key in ("output_norm.weight", "model.output_norm.weight"):
        return "model.norm.weight"
    if key == "output.weight":
        return None
    if len(parts) >= 3 and parts[0] == "blk":
        layer, name = parts[1], ".".join(parts[2:])
        names = {
            "attn_norm.weight": "input_layernorm.weight",
            "post_attention_norm.weight": "post_attention_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "ffn_gate.weight": "mlp.gate_proj.linear.weight",
            "ffn_up.weight": "mlp.up_proj.linear.weight",
            "ffn_down.weight": "mlp.down_proj.linear.weight",
            "attn_q.weight": "self_attn.q_proj.linear.weight",
            "attn_k.weight": "self_attn.k_proj.linear.weight",
            "attn_v.weight": "self_attn.v_proj.linear.weight",
            "attn_output.weight": "self_attn.o_proj.linear.weight",
            "attn_q_norm.weight": "self_attn.q_norm.weight",
            "attn_k_norm.weight": "self_attn.k_norm.weight",
            "attn_qkv.weight": "linear_attn.in_proj_qkv.linear.weight",
            "attn_gate.weight": "linear_attn.in_proj_z.linear.weight",
            "ssm_beta.weight": "linear_attn.in_proj_b.linear.weight",
            "ssm_alpha.weight": "linear_attn.in_proj_a.linear.weight",
            "ssm_conv1d.weight": "linear_attn.conv1d.weight",
            "ssm_out.weight": "linear_attn.out_proj.linear.weight",
            "ssm_norm.weight": "linear_attn.norm.weight",
            "ssm_dt.bias": "linear_attn.dt_bias",
            "ssm_a": "linear_attn.A_log",
        }
        candidate = f"model.layers.{layer}.{names.get(name, '')}"
        return candidate if candidate in target else None
    return _key(key, target)


def _undo_reorder_v_heads(tensor: torch.Tensor, dim: int, num_k_heads: int, num_v_per_k: int,
                          head_dim: int) -> torch.Tensor:
    if dim < 0:
        dim += tensor.ndim
    shape = list(tensor.shape)
    tensor = tensor.reshape(*shape[:dim], num_v_per_k, num_k_heads, head_dim, *shape[dim + 1 :])
    perm = list(range(tensor.ndim))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(shape)


def _undo_qwen35_gguf_reorder(target_key: str, tensor: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    cfg = model.config
    num_k_heads = cfg["linear_num_key_heads"]
    num_v_heads = cfg["linear_num_value_heads"]
    num_v_per_k = num_v_heads // num_k_heads
    head_k_dim = cfg["linear_key_head_dim"]
    head_v_dim = cfg["linear_value_head_dim"]
    if num_k_heads == num_v_heads or ".linear_attn." not in target_key:
        return tensor
    if target_key.endswith("in_proj_qkv.linear.weight"):
        q_dim = head_k_dim * num_k_heads
        k_dim = head_k_dim * num_k_heads
        q, k, v = tensor[:q_dim], tensor[q_dim : q_dim + k_dim], tensor[q_dim + k_dim :]
        return torch.cat((q, k, _undo_reorder_v_heads(v, 0, num_k_heads, num_v_per_k, head_v_dim)), dim=0)
    if target_key.endswith("in_proj_z.linear.weight"):
        return _undo_reorder_v_heads(tensor, 0, num_k_heads, num_v_per_k, head_v_dim)
    if target_key.endswith(("in_proj_a.linear.weight", "in_proj_b.linear.weight")):
        return _undo_reorder_v_heads(tensor, 0, num_k_heads, num_v_per_k, 1)
    if target_key.endswith(("A_log", "dt_bias")):
        return _undo_reorder_v_heads(tensor[:, None], 0, num_k_heads, num_v_per_k, 1).squeeze(1)
    if target_key.endswith("conv1d.weight"):
        data = tensor.squeeze(1)
        qk_dim = head_k_dim * num_k_heads * 2
        v = _undo_reorder_v_heads(data[qk_dim:], 0, num_k_heads, num_v_per_k, head_v_dim)
        return torch.cat((data[:qk_dim], v), dim=0)[:, None, :]
    if target_key.endswith("out_proj.linear.weight"):
        return _undo_reorder_v_heads(tensor, 1, num_k_heads, num_v_per_k, head_v_dim)
    return tensor


def _copy_tensor(model: torch.nn.Module, target: dict[str, torch.Tensor], target_key: str, tensor: torch.Tensor,
                 is_gguf: bool):
    if tensor.shape != target[target_key].shape and target[target_key].ndim == 3:
        tensor = tensor[:, None, :]
    if tensor.shape != target[target_key].shape and tensor.ndim == 2:
        tensor = tensor.T
    if is_gguf and target_key.endswith("A_log"):
        tensor = torch.log(-tensor.float())
    if is_gguf:
        tensor = _undo_qwen35_gguf_reorder(target_key, tensor, model)
    norm_weights = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight")
    if is_gguf and (target_key.endswith(norm_weights) or target_key == "model.norm.weight"):
        tensor = tensor.float() - 1
    target[target_key].copy_(tensor.to(dtype=target[target_key].dtype))


def load_weights(model: torch.nn.Module, path: str | Path, strict: bool = False) -> dict[str, Any]:
    target = model.state_dict()
    loaded, unexpected = set(), []
    for file in _files(path):
        if file.suffix == ".gguf":
            f, data_start, infos = _gguf_infos(file)
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            tensor = None
            for source_key, shape, typ, offset in infos:
                target_key = _gguf_key(source_key, set(target))
                if target_key is None:
                    unexpected.append(source_key)
                elif typ in GGUF_TYPES:
                    numel = torch.empty(shape, dtype=GGUF_TYPES[typ]).numel()
                    dtype = GGUF_TYPES[typ]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        tensor = torch.frombuffer(mm, dtype=dtype, count=numel,
                                                  offset=data_start + offset).reshape(shape)
                    _copy_tensor(model, target, target_key, tensor, True)
                    loaded.add(target_key)
            if any(t.device.type == "mps" for t in target.values()):
                torch.mps.synchronize()
            del tensor
            mm.close()
            f.close()
            continue
        tensors = load_file(file) if file.suffix == ".safetensors" else torch.load(file, map_location="cpu")
        for source_key, tensor in tensors.items():
            target_key = _key(source_key, set(target))
            if target_key is None:
                unexpected.append(source_key)
            else:
                _copy_tensor(model, target, target_key, tensor, False)
                loaded.add(target_key)
    missing = sorted(set(target) - loaded)
    if strict and (missing or unexpected):
        raise RuntimeError(f"missing={missing[:20]} unexpected={unexpected[:20]}")
    return {"missing": missing, "unexpected": unexpected}
