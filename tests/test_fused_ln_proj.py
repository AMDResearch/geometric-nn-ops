# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fused z-LayerNorm + pair-bias projection kernel."""

import pytest
import torch
import torch.nn.functional as F

triton = pytest.importorskip("triton")
# ruff: noqa: E402

from geometric_nn_ops_torch.triangle._kernels.fused_ln_proj import fused_ln_proj_bias
from testutil import generator
from testutil.metrics import ATTENTION_PAIR_BIAS_TOL, assert_matches_oracle
from testutil.precision import torch_matmul_precision
from utils import DTYPE_PRECISION_PARAM, requires_gpu


def make_inputs(B, U, V, Zc, H, dtype, *, seed):
    g = generator("cuda", seed)
    z = torch.randn(B, U, V, Zc, device="cuda", generator=g)
    ln_w = torch.randn(Zc, device="cuda", generator=g) * 0.5 + 1.0
    proj_w = torch.randn(H, Zc, device="cuda", generator=g) * Zc**-0.5
    return z.to(dtype), ln_w.to(dtype), proj_w.to(dtype)


def composed(z, ln_w, proj_w, eps=1e-5):
    zc = F.layer_norm(z, (z.shape[-1],), ln_w, None, eps)
    return F.linear(zc, proj_w).permute(0, 3, 1, 2)


def assert_agrees_with_the_oracle(z, ln_w, proj_w, *, precision="ieee", label=""):
    out = fused_ln_proj_bias(z, ln_w, proj_w, 1e-5)
    oracle = composed(z.double(), ln_w.double(), proj_w.double())
    assert out.shape == oracle.shape and out.is_contiguous()
    assert out.dtype == z.dtype
    assert_matches_oracle(
        [("bias", out, oracle)],
        ATTENTION_PAIR_BIAS_TOL[z.dtype, precision],
        label=f"{label}/{z.dtype}/{precision}",
    )


@requires_gpu
@DTYPE_PRECISION_PARAM
@pytest.mark.parametrize(
    "shape", [(1, 32, 32), (2, 40, 40), (1, 24, 56)], ids=["square", "B2", "rect"]
)
def test_accuracy(dtype, precision, shape):
    """Including row counts that are not a multiple of the row block."""
    B, U, V = shape
    z, ln_w, proj_w = make_inputs(B, U, V, 128, 16, dtype, seed=3)
    with torch_matmul_precision(precision):
        assert_agrees_with_the_oracle(
            z, ln_w, proj_w, precision=precision, label=str(shape)
        )


@requires_gpu
@pytest.mark.parametrize("H", [1, 4, 12, 16], ids=["h1", "h4", "h12", "h16"])
@pytest.mark.parametrize("Zc", [16, 64, 96, 128], ids=["zc16", "zc64", "zc96", "zc128"])
def test_widths(H, Zc):
    """Head counts and channel widths below their blocks, which have to drop
    out of the LayerNorm reductions and the dot."""
    z, ln_w, proj_w = make_inputs(1, 24, 24, Zc, H, torch.float32, seed=5)
    assert_agrees_with_the_oracle(z, ln_w, proj_w, label=f"H{H}/Zc{Zc}")


@requires_gpu
def test_autocast():
    z, ln_w, proj_w = make_inputs(1, 16, 16, 128, 8, torch.float32, seed=7)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = fused_ln_proj_bias(z, ln_w, proj_w, 1e-5)
    assert out.dtype == torch.bfloat16


@requires_gpu
@pytest.mark.parametrize("which", ["z", "ln_w", "proj_w"])
def test_grad_inputs_are_rejected(which):
    """Autograd cannot see the Triton write, so a grad-requiring input would
    otherwise get no gradient."""
    z, ln_w, proj_w = make_inputs(1, 8, 8, 16, 4, torch.float32, seed=9)
    tensors = {"z": z, "ln_w": ln_w, "proj_w": proj_w}
    tensors[which].requires_grad_()
    with pytest.raises(RuntimeError, match="forward-only"):
        fused_ln_proj_bias(**tensors, eps=1e-5)
    with torch.no_grad():
        fused_ln_proj_bias(**tensors, eps=1e-5)
