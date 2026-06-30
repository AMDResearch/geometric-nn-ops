# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for triangle_multiplicative_update.

(1) Triton fast path matches the pure-torch reference (GPU).
(2) Pure-torch reference matches OpenFold-3's TriangleMultiplicativeUpdate (parity).
(3) The torch fallback is autograd-differentiable (grads flow).
"""

import pytest
import torch

triton = pytest.importorskip("triton")

from geometric_nn_ops.triangle import _trimul as T  # noqa: E402
from geometric_nn_ops.triangle import triangle_multiplicative_update  # noqa: E402

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs GPU + Triton"
)


def _weights(C_z, C_h, device, dtype, *, requires_grad=False):
    g = requires_grad

    def t(*shape, scale=1.0):
        return (torch.randn(*shape, device=device, dtype=dtype) * scale).requires_grad_(g)

    return dict(
        norm_in_weight=torch.ones(C_z, device=device, dtype=dtype).requires_grad_(g),
        norm_in_bias=torch.zeros(C_z, device=device, dtype=dtype).requires_grad_(g),
        p_in_weight=t(2 * C_h, C_z, scale=0.05),
        p_in_bias=t(2 * C_h, scale=0.05),
        g_in_weight=t(2 * C_h, C_z, scale=0.05),
        g_in_bias=t(2 * C_h, scale=0.05),
        norm_out_weight=torch.ones(C_h, device=device, dtype=dtype).requires_grad_(g),
        norm_out_bias=torch.zeros(C_h, device=device, dtype=dtype).requires_grad_(g),
        p_out_weight=t(C_z, C_h, scale=0.05),
        p_out_bias=t(C_z, scale=0.05),
        g_out_weight=t(C_z, C_z, scale=0.05),
        g_out_bias=t(C_z, scale=0.05),
    )


@requires_gpu
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_triton_matches_torch_reference(direction, dtype, monkeypatch):
    torch.manual_seed(0)
    N, C_z, C_h = 32, 32, 24
    x = torch.randn(N, N, C_z, device="cuda", dtype=dtype)
    mask = torch.ones(N, N, device="cuda", dtype=dtype)
    w = _weights(C_z, C_h, "cuda", dtype)

    out_triton = triangle_multiplicative_update(x, direction=direction, mask=mask, **w)

    # Force the torch reference path.
    monkeypatch.setattr(T, "TRITON_AVAILABLE", False)
    out_ref = triangle_multiplicative_update(x, direction=direction, mask=mask, **w)

    atol = 5e-2 if dtype == torch.bfloat16 else 1e-2
    diff = (out_triton.float() - out_ref.float()).abs()
    assert torch.allclose(out_triton.float(), out_ref.float(), atol=atol, rtol=1e-2), (
        f"max abs diff = {diff.max().item():.4f}"
    )


@requires_gpu
@pytest.mark.parametrize("outgoing", [True, False])
def test_matches_openfold3_module(outgoing):
    """Pure-torch reference must match OpenFold-3's module for the same weights."""
    of3 = pytest.importorskip("openfold3.core.model.layers.triangular_multiplicative_update")
    torch.manual_seed(1)
    N, C_z, C_h = 24, 32, 24
    device, dtype = "cuda", torch.float32

    module = of3.TriangleMultiplicativeUpdate(c_z=C_z, c_hidden=C_h, _outgoing=outgoing)
    module = module.to(device=device, dtype=dtype).eval()

    z = torch.randn(N, N, C_z, device=device, dtype=dtype)
    mask = torch.ones(N, N, device=device, dtype=dtype)

    with torch.no_grad():
        ref = module.forward(z, mask=mask)  # standard torch path

        def _b(linear):
            return getattr(linear, "bias", None)

        p_in_bias = (
            torch.cat([module.linear_a_p.bias, module.linear_b_p.bias])
            if _b(module.linear_a_p) is not None
            else None
        )
        g_in_bias = (
            torch.cat([module.linear_a_g.bias, module.linear_b_g.bias])
            if _b(module.linear_a_g) is not None
            else None
        )
        ours = triangle_multiplicative_update(
            z,
            direction="outgoing" if outgoing else "incoming",
            mask=mask,
            norm_in_weight=module.layer_norm_in.weight,
            norm_in_bias=module.layer_norm_in.bias,
            p_in_weight=torch.cat([module.linear_a_p.weight, module.linear_b_p.weight]),
            p_in_bias=p_in_bias,
            g_in_weight=torch.cat([module.linear_a_g.weight, module.linear_b_g.weight]),
            g_in_bias=g_in_bias,
            norm_out_weight=module.layer_norm_out.weight,
            norm_out_bias=module.layer_norm_out.bias,
            p_out_weight=module.linear_z.weight,
            p_out_bias=_b(module.linear_z),
            g_out_weight=module.linear_g.weight,
            g_out_bias=_b(module.linear_g),
            eps=module.layer_norm_in.eps,
        )

    diff = (ours.float() - ref.float()).abs()
    assert torch.allclose(ours.float(), ref.float(), atol=1e-4, rtol=1e-4), (
        f"max abs diff vs OF3 = {diff.max().item():.6f}"
    )


def test_torch_fallback_is_differentiable():
    """On CPU (no Triton path), grads must flow through the op."""
    torch.manual_seed(2)
    N, C_z, C_h = 8, 16, 12
    x = torch.randn(N, N, C_z, dtype=torch.float32, requires_grad=True)
    mask = torch.ones(N, N, dtype=torch.float32)
    w = _weights(C_z, C_h, "cpu", torch.float32, requires_grad=True)

    out = triangle_multiplicative_update(x, direction="outgoing", mask=mask, **w)
    out.sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert w["p_in_weight"].grad is not None
    assert w["g_out_weight"].grad is not None
