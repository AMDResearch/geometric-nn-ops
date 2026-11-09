# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bias-free Triton LayerNorm used by attention_pair_bias."""

import pytest
import torch
import torch.nn.functional as F

triton = pytest.importorskip("triton")
# ruff: noqa: E402

from geometric_nn_ops_torch.triangle._kernels.layernorm import triton_layer_norm
from testutil import generator, seeded_randn
from testutil.metrics import Tolerance, assert_matches_oracle
from utils import DTYPES_PARAM, requires_gpu

LAYERNORM_TOL = {
    torch.float32: Tolerance(1.0 - 1e-6, 1e-4),
    torch.float16: Tolerance(1.0 - 1e-5, 2e-3),
    torch.bfloat16: Tolerance(1.0 - 1e-4, 1e-2),
}


def make_inputs(shape, dtype, *, seed, gamma="learned"):
    g = generator("cuda", seed)
    x = torch.randn(*shape, device="cuda", generator=g) * 2.0 + 0.5
    C = shape[-1]
    if gamma == "learned":
        weight = torch.randn(C, device="cuda", generator=g) * 0.5 + 1.0
    else:
        weight = torch.ones(C, device="cuda")
        weight[: C // 4] = {"zero": 0.0, "tiny": 1e-4}[gamma]
    return x.to(dtype).requires_grad_(), weight.to(dtype).requires_grad_()


def assert_agrees_with_the_oracle(x, weight, *, eps=1e-5, label=""):
    """Forward and both gradients against ``F.layer_norm`` in fp64."""
    grad_out = seeded_randn(*x.shape, device="cuda", dtype=x.dtype, seed=11)
    out = triton_layer_norm(x, weight, eps)
    grads = torch.autograd.grad(out, [x, weight], grad_out)

    x64 = x.detach().double().requires_grad_()
    w64 = weight.detach().double().requires_grad_()
    out64 = F.layer_norm(x64, (x.shape[-1],), w64, None, eps)
    grads64 = torch.autograd.grad(out64, [x64, w64], grad_out.double())

    assert out.dtype == x.dtype
    assert_matches_oracle(
        [
            ("out", out, out64),
            ("grad_x", grads[0], grads64[0]),
            ("grad_weight", grads[1], grads64[1]),
        ],
        LAYERNORM_TOL[x.dtype],
        label=f"{label}/{x.dtype}",
    )


@requires_gpu
@DTYPES_PARAM
@pytest.mark.parametrize(
    "shape",
    [(1, 32, 32, 128), (2, 17, 23, 128), (3, 64)],
    ids=["pair", "ragged", "2d"],
)
def test_accuracy(dtype, shape):
    x, weight = make_inputs(shape, dtype, seed=3)
    assert_agrees_with_the_oracle(x, weight, label=str(shape))


@requires_gpu
@pytest.mark.parametrize("C", [16, 48, 64, 96], ids=["c16", "c48", "c64", "c96"])
def test_channel_widths(C):
    """Widths below the block, including non-powers of two, so the masked
    lanes have to drop out of the reductions."""
    x, weight = make_inputs((1, 16, 16, C), torch.float32, seed=5)
    assert_agrees_with_the_oracle(x, weight, label=f"C{C}")


@requires_gpu
@pytest.mark.parametrize("gamma", ["zero", "tiny"])
def test_degenerate_weights(gamma):
    x, weight = make_inputs((1, 16, 16, 128), torch.float32, seed=7, gamma=gamma)
    assert_agrees_with_the_oracle(x, weight, label=f"{gamma}_gamma")


@requires_gpu
def test_autocast():
    """Under autocast fp32 inputs run, and come back, in the autocast dtype."""
    x, weight = make_inputs((1, 16, 16, 128), torch.float32, seed=9)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = triton_layer_norm(x, weight, 1e-5)
    out.float().sum().backward()
    assert out.dtype == torch.bfloat16
    assert x.grad is not None and x.grad.dtype == torch.float32
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
