from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from backend.metal import get_delta_rule_kernels


pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is required")


def _prefill(kernels, q, k, v, g, beta, initial_state):
    batch_size, seq_len, num_heads, _ = q.shape
    output = torch.empty_like(v, memory_format=torch.contiguous_format)
    state = initial_state if initial_state is not None else torch.empty((batch_size, num_heads, 128, 128),
                                                                        dtype=torch.float32, device=q.device)
    args = (output, state, q, k, v, g, beta, batch_size, seq_len, num_heads, v.stride(0), v.stride(1), v.stride(2),
            v.stride(3), initial_state is not None)
    kernels.lib.delta_rule_prefill(*args, threads=[batch_size * num_heads * 128, 1, 1], group_size=[128, 1, 1])
    return output, state


def _decode_tokens(kernels, q, k, v, g, beta, initial_state):
    outs, state = [], initial_state
    for t in range(q.shape[1]):
        out, state = kernels(q[:, t:t + 1].contiguous(), k[:, t:t + 1].contiguous(), v[:, t:t + 1].contiguous(),
                             g[:, t:t + 1].contiguous(), beta[:, t:t + 1].contiguous(), state)
        outs.append(out)
    return torch.cat(outs, dim=1), state


@pytest.mark.parametrize("seq_len", [1, 8, 63, 64, 65, 128])
@pytest.mark.parametrize("has_initial_state", [False, True])
def test_delta_rule_prefill_matches_decode(seq_len: int, has_initial_state: bool):
    torch.manual_seed(1000 + seq_len + has_initial_state)
    batch_size, num_heads, dim = 1, 32, 128
    q = torch.randn(batch_size, seq_len, num_heads, dim, device="mps", dtype=torch.float16).contiguous()
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = (torch.randn(batch_size, seq_len, num_heads, device="mps", dtype=torch.float32) * -0.01).contiguous()
    beta = torch.rand(batch_size, seq_len, num_heads, device="mps", dtype=torch.float16).contiguous()
    initial = torch.randn(batch_size, num_heads, dim, dim, device="mps",
                          dtype=torch.float32) if has_initial_state else None
    kernels = get_delta_rule_kernels()
    prefill_out, prefill_state = _prefill(kernels, q, k, v, g, beta, None if initial is None else initial.clone())
    decode_out, decode_state = _decode_tokens(kernels, q, k, v, g, beta, None if initial is None else initial.clone())
    torch.mps.synchronize()
    torch.testing.assert_close(prefill_out.cpu(), decode_out.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(prefill_state.cpu(), decode_state.cpu(), atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("seq_len", [8, 65])
def test_delta_rule_prefill_matches_decode_batch_two(seq_len: int):
    torch.manual_seed(2000 + seq_len)
    batch_size, num_heads, dim = 2, 32, 128
    q = torch.randn(batch_size, seq_len, num_heads, dim, device="mps", dtype=torch.float16).contiguous()
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = (torch.randn(batch_size, seq_len, num_heads, device="mps", dtype=torch.float32) * -0.01).contiguous()
    beta = torch.rand(batch_size, seq_len, num_heads, device="mps", dtype=torch.float16).contiguous()
    kernels = get_delta_rule_kernels()
    prefill_out, prefill_state = _prefill(kernels, q, k, v, g, beta, None)
    decode_out, decode_state = _decode_tokens(kernels, q, k, v, g, beta, None)
    torch.mps.synchronize()
    torch.testing.assert_close(prefill_out.cpu(), decode_out.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(prefill_state.cpu(), decode_state.cpu(), atol=5e-3, rtol=5e-3)
