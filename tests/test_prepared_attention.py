# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for prepared triangle attention."""

import os

import pytest
import torch

from geometric_nn_ops_torch.triangle import PreparedTriangleAttention

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"
pytestmark = pytest.mark.skipif(
    not INTERPRET
    and (
        not torch.cuda.is_available()
        or getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(":")[0]
        != "gfx950"
    ),
    reason="Requires explicit Triton interpreter or gfx950",
)


@pytest.fixture(autouse=True)
def inference_context():
    torch.manual_seed(42)
    with torch.no_grad():
        if INTERPRET:
            yield
        else:
            previous = torch.backends.cuda.matmul.fp32_precision
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            try:
                yield
            finally:
                torch.backends.cuda.matmul.fp32_precision = previous


@pytest.mark.parametrize("n", [17, 64, 65])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize(
    ("heads", "dim"), [(1, 16), (3, 24), (4, 16), (4, 32), (4, 64)]
)
def test_attention_reference_mask_tails_gate_and_scale(n, gated, heads, dim):
    rows = 2
    bias = torch.randn(1, 1, heads, n, n, device=DEVICE) * 0.1
    mask = torch.rand(1, n, 1, 1, n, device=DEVICE) > 0.2
    mask[..., 0] = True
    prepared = PreparedTriangleAttention(bias, mask)
    q, k, v = [torch.randn(1, rows, n, heads, dim, device=DEVICE) for _ in range(3)]
    gate = torch.randn_like(q).sigmoid() if gated else None
    scale, start = 0.2, 1
    actual, m = prepared(
        q, k, v, start=start, gate=gate, scale=scale, return_logsumexp=True
    )
    qc, kc, vc = [t.transpose(2, 3) for t in (q, k, v)]
    logits = (qc * scale) @ kc.transpose(-1, -2)
    logits += 1e9 * (mask[:, start : start + rows].float() - 1)
    logits += bias
    expected = (logits.softmax(-1) @ vc).transpose(2, 3)
    if gate is not None:
        expected *= gate
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(m, logits.logsumexp(-1), atol=1e-5, rtol=1e-4)
    assert actual.is_contiguous()


def test_preparation_snapshots_and_fully_masked_rows():
    n = 17
    bias = torch.randn(1, 1, 4, n, n, device=DEVICE)
    mask = torch.zeros(1, n, 1, 1, n, dtype=torch.bool, device=DEVICE)
    prepared = PreparedTriangleAttention(bias, mask)
    saved_bias, saved_mask = prepared.bias.clone(), prepared.mask.clone()
    bias.zero_()
    mask.fill_(True)
    torch.testing.assert_close(prepared.bias, saved_bias, atol=0, rtol=0)
    torch.testing.assert_close(prepared.mask, saved_mask, atol=0, rtol=0)
    q, k, v = [torch.randn(1, 1, n, 4, 64, device=DEVICE) for _ in range(3)]
    actual, m = prepared(q, k, v, start=0, return_logsumexp=True)
    # Finite -1e9 masking matches the existing API, not an all-zero result.
    expected = v.mean(dim=2, keepdim=True).expand_as(v)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert torch.isfinite(m).all()


def test_attention_strided_chunks_and_default_scale():
    n, rows = 17, 2
    bias = torch.randn(1, 1, 4, n, n, device=DEVICE)
    keep = torch.ones(1, n, 1, 1, n, device=DEVICE, dtype=torch.bool)
    prepared = PreparedTriangleAttention(bias.transpose(-1, -2), keep)
    tensors = [
        torch.randn(1, 4, 4, n + 3, 64, device=DEVICE)[:, 1:3, :, :n].transpose(2, 3)
        for _ in range(3)
    ]
    q, k, v = tensors
    actual = prepared(q, k, v, start=3)
    qc, kc, vc = [t.transpose(2, 3) for t in tensors]
    logits = (qc * (64**-0.5)) @ kc.transpose(-1, -2)
    logits += bias.transpose(-1, -2)
    expected = (logits.softmax(-1) @ vc).transpose(2, 3)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert actual.shape == (1, rows, n, 4, 64)


def test_preparation_is_reused_across_chunks():
    n = 17
    bias = torch.randn(1, 1, 4, n, n, device=DEVICE) * 0.1
    mask = torch.rand(1, n, 1, 1, n, device=DEVICE) > 0.2
    mask[..., 0] = True
    prepared = PreparedTriangleAttention(bias, mask)
    bias_ptr, mask_ptr = prepared.bias.data_ptr(), prepared.mask.data_ptr()
    q, k, v = [torch.randn(1, 3, n, 4, 64, device=DEVICE) for _ in range(3)]
    for start in (0, 7):
        actual = prepared(q, k, v, start=start)
        qc, kc, vc = [t.transpose(2, 3) for t in (q, k, v)]
        logits = (qc * (64**-0.5)) @ kc.transpose(-1, -2)
        logits += 1e9 * (mask[:, start : start + 3].float() - 1)
        logits += bias
        expected = (logits.softmax(-1) @ vc).transpose(2, 3)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        assert prepared.bias.data_ptr() == bias_ptr
        assert prepared.mask.data_ptr() == mask_ptr


def test_validation_rejects_unsupported_inputs():
    n = 17
    bias = torch.zeros(1, 1, 4, n, n, device=DEVICE)
    mask = torch.ones(1, n, 1, 1, n, device=DEVICE, dtype=torch.bool)
    prepared = PreparedTriangleAttention(bias, mask)
    q = torch.randn(1, 1, n, 4, 64, device=DEVICE)
    with pytest.raises(ValueError, match="start"):
        prepared(q, q, q, start=n)
    with pytest.raises(ValueError, match="gate"):
        prepared(q, q, q, start=0, gate=q[..., :1])
    with pytest.raises(ValueError, match="finite"):
        prepared(q, q, q, start=0, scale=float("inf"))
    with pytest.raises(ValueError, match="mask must be bool"):
        PreparedTriangleAttention(bias, mask.float())
    with pytest.raises(ValueError, match="head dimension"):
        too_wide = torch.randn(1, 1, n, 4, 129, device=DEVICE)
        prepared(too_wide, too_wide, too_wide, start=0)
    with pytest.raises(ValueError, match="q must be"):
        wrong_heads = torch.randn(1, 1, n, 3, 64, device=DEVICE)
        prepared(wrong_heads, wrong_heads, wrong_heads, start=0)
    with pytest.raises(ValueError, match="at least one head"):
        PreparedTriangleAttention(torch.empty(1, 1, 0, n, n, device=DEVICE), mask)


@pytest.mark.parametrize("dim", [1, 128])
def test_head_dimension_boundaries(dim):
    n, heads = 17, 2
    bias = torch.randn(1, 1, heads, n, n, device=DEVICE) * 0.1
    mask = torch.ones(1, n, 1, 1, n, dtype=torch.bool, device=DEVICE)
    prepared = PreparedTriangleAttention(bias, mask)
    q, k, v = [torch.randn(1, 1, n, heads, dim, device=DEVICE) for _ in range(3)]
    actual, lse = prepared(q, k, v, start=0, return_logsumexp=True)
    qc, kc, vc = [t.transpose(2, 3).double() for t in (q, k, v)]
    logits = (qc * dim**-0.5) @ kc.transpose(-1, -2) + bias.double()
    expected = (logits.softmax(-1) @ vc).transpose(2, 3)
    torch.testing.assert_close(actual.double(), expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(lse.double(), logits.logsumexp(-1), atol=1e-5, rtol=1e-4)
