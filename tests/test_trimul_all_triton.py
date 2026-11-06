# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from test_trimul_variants import weights

from geometric_nn_ops_torch.triangle.experimental import run_variant
from geometric_nn_ops_torch.triangle.experimental.all_triton import combine

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA/HIP"
)


@pytest.mark.parametrize("n", [1, 17, 32, 65, 127, 128, 129, 256])
@pytest.mark.parametrize("incoming", [False, True])
def test_contraction_tails(n, incoming):
    torch.manual_seed(42)
    a = torch.randn(3, n, n, device="cuda")
    b = torch.randn_like(a)
    if incoming:
        a = a.transpose(-1, -2)
    else:
        b = b.transpose(-1, -2)
    before_a, before_b = a.clone(), b.clone()
    expected = (a.double() @ b.double()).float()
    actual = combine(a, b)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=1e-4)
    assert torch.equal(a, before_a)
    assert torch.equal(b, before_b)


@pytest.mark.parametrize("c", [1, 5, 64, 129, 384])
@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize(
    "variant", ["all_triton_fused", "layout_aware_pipeline_mfma16"]
)
def test_fused_producer_tails_and_strided_parameters(c, direction, affine, variant):
    torch.manual_seed(42)
    x = torch.randn(2, 7, 7, c, device="cuda").transpose(1, 2)
    parameters = {}
    for key, value in weights(c, c, True).items():
        if not affine and key.startswith("norm_"):
            continue
        storage = torch.empty((*value.shape, 2), device="cuda")
        storage[..., 0] = value.cuda()
        parameters[key] = storage[..., 0]
    mask = torch.rand(1, 7, 7, device="cuda") > 0.2
    with torch.no_grad():
        expected = run_variant("reference", x, direction, mask, **parameters)
        actual = run_variant(variant, x, direction, mask, **parameters)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_fused_producer_constant_rows():
    x = torch.full((5, 5, 129), 10_000.0, device="cuda")
    parameters = {k: v.cuda() for k, v in weights(129, 129, True).items()}
    with torch.no_grad():
        expected = run_variant("reference", x, **parameters)
        actual = run_variant("all_triton_fused", x, **parameters)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_fused_producer_explicit_limits():
    x = torch.randn(3, 3, 513, device="cuda")
    parameters = {k: v.cuda() for k, v in weights(513, 513, False).items()}
    with torch.no_grad(), pytest.raises(ValueError, match="<= 512"):
        run_variant("all_triton_fused", x, **parameters)
    x = torch.randn(3, 3, 32, device="cuda")
    parameters = {k: v.cuda() for k, v in weights(32, 32, False).items()}
    with pytest.raises(ValueError, match="forward-only"):
        run_variant("all_triton_fused", x.requires_grad_(), **parameters)
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        with torch.no_grad(), pytest.raises(ValueError, match="highest/IEEE"):
            run_variant("all_triton_fused", x, **parameters)
    finally:
        torch.set_float32_matmul_precision(previous)
