from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from model.qwen35.model import ForCausalLM
from model.qwen35.weights import load_weights

if os.getenv("SKIP_MODEL") is not None:
    pytest.skip("SKIP_MODEL is set", allow_module_level=True)

import transformers

WEIGHTS = ROOT / "weights/unsloth-Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-BF16.gguf"
CONFIG = ROOT / "config.json"
MIN_TRANSFORMERS = (5, 2, 0)
assert tuple(map(int, transformers.__version__.split("+", 1)[0].split(".")[:3])) >= MIN_TRANSFORMERS, f"transformers>={'.'.join(map(str, MIN_TRANSFORMERS))} required for Qwen3.5, found {transformers.__version__}"
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig


@pytest.fixture(scope="session")
def local_model():
    assert WEIGHTS.exists(), f"missing model weights: {WEIGHTS}"
    device = torch.device(os.getenv("QWEN35_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu"))
    dtype_name = os.getenv("QWEN35_DTYPE")
    dtype = getattr(torch, dtype_name) if dtype_name else (torch.float16 if device.type == "mps" else torch.bfloat16)
    model = ForCausalLM.build(CONFIG, device=device, dtype=dtype).eval()
    report = load_weights(model, WEIGHTS)
    assert not report["missing"]
    assert all(k == "output.weight" or k.startswith("blk.32.") for k in report["unexpected"])
    return model


@pytest.fixture(scope="session")
def hf_model(local_model):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(next(local_model.parameters()).dtype)
    try:
        with torch.device("meta"):
            ref = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**json.load(open(CONFIG))["text_config"]))
    finally:
        torch.set_default_dtype(old_dtype)

    ref_params = dict(ref.named_parameters())
    local_state = local_model.state_dict()
    set_param = lambda name, tensor: setattr(ref.get_submodule(name.rsplit(".", 1)[0]), name.rsplit(".", 1)[1], nn.Parameter(tensor, requires_grad=False))
    for name, tensor in local_state.items():
        target = name.replace(".linear.weight", ".weight").replace(".linear.bias", ".bias")
        if target in ref_params and ref_params[target].shape == tensor.shape:
            set_param(target, tensor)
            continue
        if target.endswith(".self_attn.q_proj.weight"):
            q, gate = tensor.chunk(2)
            base = target[: -len("q_proj.weight")]
            if target in ref_params and ref_params[target].shape == q.shape:
                set_param(target, q)
            for gate_name in (base + "q_gate_proj.weight", base + "gate_proj.weight", base + "q_gate.weight"):
                if gate_name in ref_params and ref_params[gate_name].shape == gate.shape:
                    set_param(gate_name, gate)
                    break

    if getattr(ref, "lm_head", None) is not None:
        ref.lm_head.weight = nn.Parameter(local_state["model.embed_tokens.weight"], requires_grad=False)
    inv_freq = local_model.model.layers[3].self_attn.rotary_emb.inv_freq
    for name, buffer in list(ref.named_buffers()):
        if buffer.device.type == "meta" and name.endswith(("inv_freq", "original_inv_freq")):
            module, attr = ref.get_submodule(name.rsplit(".", 1)[0]), name.rsplit(".", 1)[1]
            module.register_buffer(attr, inv_freq.clone(), persistent=False)

    assert not (missing := [name for name, param in ref.named_parameters() if param.device.type == "meta"]), f"unbound HF parameters: {missing[:20]}"
    return ref.eval()


def test_logits_match_reference(local_model, hf_model):
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=next(local_model.parameters()).device)
    with torch.no_grad():
        actual, expected = local_model(input_ids)[0].float(), hf_model(input_ids, use_cache=False).logits.float()
    torch.testing.assert_close(actual, expected, atol=float(os.getenv("QWEN35_ATOL", "3e-2")), rtol=float(os.getenv("QWEN35_RTOL", "3e-2")))
