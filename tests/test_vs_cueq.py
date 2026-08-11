# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Numerical equivalence test: geometric-nn-ops vs. cuequivariance.

Skips automatically when cuequivariance_torch is not importable. Intended to be
run on a machine that has both Triton and cuEquivariance installed, to validate
that the two implementations produce numerically equivalent outputs on the
same inputs.

Run with:
    pytest tests/test_vs_cueq.py -v
"""

# The kernel imports run after `importorskip` / `skip` so the module skips
# cleanly without the dependencies; E402 (import not at top) is disabled here.
# ruff: noqa: E402

import pytest
import torch

triton = pytest.importorskip("triton")
cuequivariance_torch = pytest.importorskip(
    "cuequivariance_torch",
    reason="cuequivariance_torch not installed; skipping equivalence test.",
)

if not torch.cuda.is_available():
    pytest.skip("needs CUDA for the cuequivariance reference", allow_module_level=True)

from cuequivariance_torch.primitives.triangle import (
    triangle_attention as cueq_triangle_attention,
)

from geometric_nn_ops_torch.triangle import triangle_attention as hip_triangle_attention

DEVICE = "cuda"


def _make_inputs(B, N_seq, N_res, H, D, dtype, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, N_seq, N_res, H, D, device=DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bias = torch.randn(B, 1, H, N_res, N_res, device=DEVICE, dtype=dtype) * 0.1
    mask = torch.ones(B, N_seq, 1, 1, N_res, device=DEVICE, dtype=torch.bool)
    # mask out a few trailing positions so the boolean path actually does work
    mask[..., -3:] = False
    return q, k, v, bias, mask


def _cueq(q, k, v, bias, mask, scale):
    """Call cueq's triangle_attention for comparison with our direct API.

    geometric-nn-ops uses q/k/v layout ``(B, N, S, H, D)``; cuequivariance uses
    ``(B, N, H, S, D)``. Transpose the H/S axes on the way in and transpose the
    output back so results are comparable in our layout. (bias/mask share the
    same shape in both.)
    """
    o = cueq_triangle_attention(
        q.transpose(-2, -3),
        k.transpose(-2, -3),
        v.transpose(-2, -3),
        bias=bias,
        mask=mask,
        scale=scale,
    )
    return o.transpose(-2, -3)


# Representative shapes chosen to cover:
#  - small monomer-like (D=32, common pairformer attention call)
#  - MSA-att-like (D=8, the most common kernel call in OF3)
#  - bigger pairformer chunk (D=48, atom transformer)
@pytest.mark.parametrize(
    "shape",
    [
        # (B, N_seq, N_res, H, D)
        (1, 1, 64, 4, 32),
        (1, 4, 32, 8, 8),
        (1, 1, 32, 16, 48),
        (1, 2, 128, 4, 32),
    ],
    ids=["pairformer_64_h4_d32", "msa_att_d8", "atom_d48", "pairformer_128_h4_d32"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_matches_cueq(shape, dtype):
    """Triton adapter and cueq's triangle_attention should agree within fp tolerance."""
    B, N_seq, N_res, H, D = shape
    q, k, v, bias, mask = _make_inputs(B, N_seq, N_res, H, D, dtype)
    scale = D**-0.5

    o_triton = hip_triangle_attention(q, k, v, bias=bias, mask=mask, scale=scale)
    o_cueq = _cueq(q, k, v, bias=bias, mask=mask, scale=scale)

    # Both run in low precision; tolerances chosen to be loose enough to
    # accommodate different reduction orders inside the two kernels but tight
    # enough to catch any real divergence.
    atol = 5e-2 if dtype == torch.bfloat16 else 1e-2
    diff = (o_triton.to(torch.float32) - o_cueq.to(torch.float32)).abs()
    assert torch.allclose(
        o_triton.to(torch.float32), o_cueq.to(torch.float32), atol=atol, rtol=1e-2
    ), (
        f"Disagreement on {shape} {dtype}: "
        f"max abs diff = {diff.max().item():.4f}, mean = {diff.mean().item():.4f}"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_matches_cueq_custom_scale(dtype):
    """Custom (non-default) scale should still match cueq."""
    B, N_seq, N_res, H, D = 1, 2, 32, 4, 32
    q, k, v, bias, mask = _make_inputs(B, N_seq, N_res, H, D, dtype, seed=1)
    scale = (D**-0.5) * 1.5  # arbitrary non-default

    o_triton = hip_triangle_attention(q, k, v, bias=bias, mask=mask, scale=scale)
    o_cueq = _cueq(q, k, v, bias=bias, mask=mask, scale=scale)

    atol = 5e-2 if dtype == torch.bfloat16 else 1e-2
    assert torch.allclose(
        o_triton.to(torch.float32), o_cueq.to(torch.float32), atol=atol, rtol=1e-2
    ), f"max abs diff = {(o_triton - o_cueq).abs().max().item():.4f}"


def test_matches_cueq_backward():
    """Gradient check: both kernels should produce equivalent gradients."""
    dtype = torch.bfloat16
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 32
    q, k, v, bias, mask = _make_inputs(B, N_seq, N_res, H, D, dtype, seed=2)
    scale = D**-0.5

    # Need separate copies because both adapters call .backward()
    q_a, k_a, v_a, bias_a = (
        t.clone().detach().requires_grad_(True) for t in (q, k, v, bias)
    )
    q_c, k_c, v_c, bias_c = (
        t.clone().detach().requires_grad_(True) for t in (q, k, v, bias)
    )

    o_triton = hip_triangle_attention(
        q_a, k_a, v_a, bias=bias_a, mask=mask, scale=scale
    )
    o_cueq = _cueq(q_c, k_c, v_c, bias=bias_c, mask=mask, scale=scale)

    # Random (but matched) upstream gradient
    torch.manual_seed(99)
    grad_o = torch.randn_like(o_triton)
    o_triton.backward(grad_o)
    o_cueq.backward(grad_o)

    atol = 1e-1  # gradients have wider noise than forward outputs
    for name, ga, gc in [
        ("dq", q_a.grad, q_c.grad),
        ("dk", k_a.grad, k_c.grad),
        ("dv", v_a.grad, v_c.grad),
        ("dbias", bias_a.grad, bias_c.grad),
    ]:
        diff = (ga.to(torch.float32) - gc.to(torch.float32)).abs()
        assert torch.allclose(
            ga.to(torch.float32), gc.to(torch.float32), atol=atol, rtol=1e-1
        ), f"{name} disagreement: max abs diff = {diff.max().item():.4f}"
