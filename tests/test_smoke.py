# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: triangle_attention vs. an eager reference.

Asserts that the Triton kernel's output matches a stock-PyTorch reference
within bf16 tolerance on a small representative shape. Skips automatically
when a Triton-capable GPU is not available.
"""

import math

import pytest
import torch

triton = pytest.importorskip("triton")
from geometric_nn_ops.triangle import triangle_attention  # noqa: E402

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU + Triton")


def _eager_triangle_attention(q, k, v, bias, mask, scale):
    """Reference implementation in stock PyTorch (no Triton).

    Shapes:
        q, k, v: [B, N_seq, N_res, H, D]
        bias:    [B, 1, H, N_res, N_res]
        mask:    [B, N_seq, 1, 1, N_res], bool (True = keep) or additive
    """
    qs = q.transpose(-2, -3)  # [B, N_seq, H, N_res, D]
    ks = k.transpose(-2, -3)
    vs = v.transpose(-2, -3)
    # [B, N_seq, H, N_res, N_res]
    logits = torch.einsum("bshid,bshjd->bshij", qs * scale, ks)
    logits = logits + bias  # bias [B,1,H,N,N] broadcasts over s
    if mask.dtype == torch.bool:
        logits = logits.masked_fill(~mask, float("-inf"))
    else:
        logits = logits + mask
    probs = torch.softmax(logits, dim=-1)
    o = torch.einsum("bshij,bshjd->bshid", probs, vs)  # [B, N_seq, H, N_res, D]
    return o.transpose(-2, -3)  # [B, N_seq, N_res, H, D]


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_triangle_attention_matches_eager(dtype):
    torch.manual_seed(0)
    device = "cuda"
    B, N_seq, N_res, H, D = 1, 4, 32, 4, 32
    q = torch.randn(B, N_seq, N_res, H, D, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bias = torch.randn(B, 1, H, N_res, N_res, device=device, dtype=dtype) * 0.1
    mask = torch.ones(B, N_seq, 1, 1, N_res, device=device, dtype=torch.bool)
    # mask out the last column
    mask[..., -1] = False
    scale = D ** -0.5

    o_triton = triangle_attention(q, k, v, bias=bias, mask=mask, scale=scale).to(torch.float32)
    o_ref = _eager_triangle_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32),
        bias.to(torch.float32), mask, scale,
    )

    # bf16 tolerance is loose (~1e-2 abs is realistic for accumulated softmax);
    # fp32 runs natively through the kernel so it should match tightly.
    atol = {torch.bfloat16: 5e-2, torch.float16: 1e-2, torch.float32: 1e-3}[dtype]
    assert torch.allclose(o_triton, o_ref, atol=atol, rtol=1e-2), (
        f"max abs diff: {(o_triton - o_ref).abs().max().item():.4f}"
    )


@requires_gpu
def test_scale_override():
    """Custom scale should be honoured via the kernel's softmax_scale arg."""
    torch.manual_seed(0)
    device = "cuda"
    B, N_seq, N_res, H, D = 1, 2, 16, 4, 32
    q = torch.randn(B, N_seq, N_res, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bias = torch.zeros(B, 1, H, N_res, N_res, device=device, dtype=torch.bfloat16)
    mask = torch.ones(B, N_seq, 1, 1, N_res, device=device, dtype=torch.bool)

    default = D ** -0.5
    custom = default * 2.0

    o_default = triangle_attention(q, k, v, bias=bias, mask=mask, scale=default).to(torch.float32)
    o_custom = triangle_attention(q, k, v, bias=bias, mask=mask, scale=custom).to(torch.float32)
    o_ref = _eager_triangle_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32),
        bias.to(torch.float32), mask, custom,
    )
    assert not torch.allclose(o_default, o_custom, atol=1e-2), "scale override had no effect"
    assert torch.allclose(o_custom, o_ref, atol=5e-2, rtol=1e-2)
