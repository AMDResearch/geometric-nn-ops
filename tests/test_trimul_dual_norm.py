# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from test_trimul_variants import weights

from geometric_nn_ops_torch.triangle.experimental import run_variant
from geometric_nn_ops_torch.triangle.experimental.dual_projection_norm import row_norm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA/HIP"
)


@pytest.fixture(autouse=True)
def ieee_precision():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("c", [1, 5, 64, 128, 129, 256, 384, 4096])
@pytest.mark.parametrize("layout", ["row", "channel"])
@pytest.mark.parametrize("affine", [False, True])
def test_row_norm(c, layout, affine):
    torch.manual_seed(42)
    x = torch.randn(19, c, device="cuda")
    if layout == "channel":
        x = x.t().contiguous().t()
    before = x.clone()
    w = torch.rand(c, device="cuda") + 0.5 if affine else None
    b = torch.randn(c, device="cuda") * 0.1 if affine else None
    with torch.no_grad():
        expected = torch.nn.functional.layer_norm(x, (c,), w, b, 1e-4)
        actual = row_norm(x, w, b, 1e-4)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert torch.equal(x, before)


@pytest.mark.parametrize("c,h", [(8, 5), (128, 128), (256, 256), (384, 384)])
@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
@pytest.mark.parametrize("biases", [False, True])
def test_complete_update(c, h, direction, biases):
    torch.manual_seed(42)
    x = torch.randn(2, 17, 17, c, device="cuda").transpose(1, 2)
    mask = torch.rand(1, 17, 17, device="cuda") > 0.2
    parameters = {k: v.cuda() for k, v in weights(c, h, biases).items()}
    if not biases:
        parameters = {k: v for k, v in parameters.items() if not k.startswith("norm_")}
    saved = {k: v.clone() for k, v in parameters.items()}
    before = x.clone()
    with torch.no_grad():
        expected = run_variant("reference", x, direction, mask, **parameters)
        actual = run_variant("dual_projection_norm", x, direction, mask, **parameters)
    assert torch.isfinite(actual).all()
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert torch.equal(x, before)
    for k, v in parameters.items():
        assert torch.equal(v, saved[k])


def test_centered_statistics():
    x = torch.full((7, 129), 10000.0, device="cuda")
    with torch.no_grad():
        actual = row_norm(x, None, None, 1e-5)
    assert torch.equal(actual, torch.zeros_like(actual))


def test_rejects_training_and_reduced_precision():
    x = torch.randn(3, 3, 8, device="cuda")
    parameters = {k: v.cuda() for k, v in weights(8, 5, False).items()}
    with pytest.raises(ValueError, match="forward-only"):
        run_variant("dual_projection_norm", x.requires_grad_(), **parameters)
    torch.set_float32_matmul_precision("high")
    with torch.no_grad(), pytest.raises(ValueError, match="highest/IEEE"):
        run_variant("dual_projection_norm", x.detach(), **parameters)
