# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from test_trimul_variants import weights

from geometric_nn_ops_torch.triangle.experimental import layout_aware, run_variant
from geometric_nn_ops_torch.triangle.experimental.layout_aware_norm import (
    channel_major_norm,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA/HIP"
)


@pytest.fixture(autouse=True)
def ieee_precision():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("h,m", [(1, 1), (5, 37), (128, 65536), (129, 67), (384, 257)])
@pytest.mark.parametrize("affine", [False, True])
def test_channel_major_norm(h, m, affine):
    torch.manual_seed(42)
    x = torch.randn(h, m * 2, device="cuda")[:, ::2]
    before = x.clone()
    w = torch.rand(h, device="cuda") + 0.5 if affine else None
    b = torch.randn(h, device="cuda") * 0.1 if affine else None
    with torch.no_grad():
        expected = torch.nn.functional.layer_norm(x.t(), (h,), w, b, 1e-4).t()
        actual = channel_major_norm(x, w, b, 1e-4)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert actual.is_contiguous()
    assert torch.equal(x, before)


@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
@pytest.mark.parametrize("variant", ["layout_aware_norm", "layout_aware_norm_input"])
@pytest.mark.parametrize("biases", [False, True])
@pytest.mark.parametrize("use_mask", [False, True])
def test_complete_update(direction, variant, biases, use_mask, monkeypatch):
    torch.manual_seed(42)
    x = torch.randn(2, 17, 17, 8, device="cuda").transpose(1, 2)
    parameters = {k: v.cuda() for k, v in weights(8, 5, biases).items()}
    if not biases:
        parameters = {k: v for k, v in parameters.items() if not k.startswith("norm_")}
    mask = torch.rand(1, 17, 17, device="cuda") > 0.2 if use_mask else None
    before = x.clone()
    saved = {k: v.clone() for k, v in parameters.items()}
    monkeypatch.setattr(layout_aware, "BUDGET_BYTES", 12000)
    monkeypatch.setattr(layout_aware, "LN_MAX_ELEMS", 1024)
    with torch.no_grad():
        expected = run_variant("reference", x, direction, mask, **parameters)
        actual = run_variant(variant, x, direction, mask, **parameters)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert torch.equal(x, before)
    for k, v in parameters.items():
        assert torch.equal(v, saved[k])


@pytest.mark.parametrize("variant", ["layout_aware_norm", "layout_aware_norm_input"])
def test_rejects_training(variant):
    x = torch.randn(3, 3, 8, device="cuda", requires_grad=True)
    parameters = {k: v.cuda() for k, v in weights(8, 5, False).items()}
    with pytest.raises(ValueError, match="forward-only"):
        run_variant(variant, x, **parameters)


@pytest.mark.parametrize("h", [1, 129, 4096])
def test_constant_channels(h):
    x = torch.full((h, 33), 7.0, device="cuda")
    with torch.no_grad():
        actual = channel_major_norm(x, None, None, 1e-5)
    assert torch.equal(actual, torch.zeros_like(actual))


def test_rejects_unsupported_width_and_precision():
    with pytest.raises(ValueError, match="4096"):
        channel_major_norm(torch.empty(4097, 1, device="cuda"), None, None, 1e-5)
    x = torch.randn(3, 3, 8, device="cuda")
    parameters = {k: v.cuda() for k, v in weights(8, 5, False).items()}
    torch.set_float32_matmul_precision("high")
    with pytest.raises(ValueError, match="highest/IEEE"):
        run_variant("layout_aware_norm", x, **parameters)
