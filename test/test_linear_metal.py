from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from backend.metal import get_linear_kernels

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is required")


@pytest.mark.parametrize(("batch", "seq", "k", "n"), [(1, 1, 128, 128), (1, 17, 256, 257), (2, 32, 512, 384)])
def test_linear_metal_matches_torch(batch: int, seq: int, k: int, n: int):
    torch.manual_seed(3000 + batch + seq + k + n)
    x = torch.randn(batch, seq, k, device="mps", dtype=torch.float16).contiguous()
    weight = torch.randn(n, k, device="mps", dtype=torch.float16).contiguous()
    y = get_linear_kernels()(x, weight)
    ref = torch.nn.functional.linear(x, weight)
    torch.mps.synchronize()
    torch.testing.assert_close(y.cpu(), ref.cpu(), atol=2e-2, rtol=2e-2)
