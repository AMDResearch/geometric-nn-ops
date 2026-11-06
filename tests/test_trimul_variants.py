# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from geometric_nn_ops_torch.triangle.experimental import (
    VARIANTS,
    layout_aware,
    run_variant,
)


def weights(c, h, biases):
    result = {
        "p_in_weight": torch.randn(2 * h, c) / c**0.5,
        "g_in_weight": torch.randn(2 * h, c) / c**0.5,
        "p_out_weight": torch.randn(c, h) / h**0.5,
        "g_out_weight": torch.randn(c, c) / c**0.5,
        "norm_in_weight": torch.rand(c) + 0.5,
        "norm_in_bias": torch.randn(c) * 0.1,
        "norm_out_weight": torch.rand(h) + 0.5,
        "norm_out_bias": torch.randn(h) * 0.1,
    }
    if biases:
        for key, width in (
            ("p_in_bias", 2 * h),
            ("g_in_bias", 2 * h),
            ("p_out_bias", c),
            ("g_out_bias", c),
        ):
            result[key] = torch.randn(width) * 0.1
    return result


@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
@pytest.mark.parametrize("biases", [False, True])
@pytest.mark.parametrize("use_mask", [False, True])
def test_layout_aware_matches_reference(direction, biases, use_mask, monkeypatch):
    torch.manual_seed(42)
    x = torch.randn(2, 7, 7, 8).transpose(1, 2)
    mask = (torch.rand(1, 7, 7) > 0.2) if use_mask else None
    parameters = weights(8, 5, biases)
    before = x.clone()
    monkeypatch.setattr(layout_aware, "BUDGET_BYTES", 1500)
    monkeypatch.setattr(layout_aware, "LN_MAX_ELEMS", 64)
    with torch.no_grad():
        reference = run_variant("reference", x, direction, mask, eps=1e-4, **parameters)
        candidate = run_variant(
            "layout_aware", x, direction, mask, eps=1e-4, **parameters
        )
    torch.testing.assert_close(candidate, reference, atol=2e-5, rtol=2e-4)
    assert torch.equal(before, x)
    assert candidate.shape == x.shape


def test_explicit_selection_rejects_invalid_and_unsupported_inputs():
    x, parameters = torch.randn(3, 3, 8), weights(8, 5, False)
    with pytest.raises(ValueError, match="Unknown variant"):
        run_variant("nonexistent", x, **parameters)
    with pytest.raises(ValueError, match="direction"):
        run_variant("reference", x, "invalid", **parameters)
    with pytest.raises(ValueError, match="CUDA/HIP"):
        run_variant("pointwise", x, **parameters)
    with pytest.raises(ValueError, match="forward-only"):
        run_variant("layout_aware", x.requires_grad_(), **parameters)


def test_all_levels_are_independently_named():
    assert {
        "layout_aware_pipeline_mfma16",
        "layout_aware_pipeline",
        "all_triton_fused",
        "all_triton",
        "original_channel_major",
        "original_channel_major_norm",
        "pointwise",
        "projection_epilogue",
        "projection_epilogue_300x",
        "dual_projection",
        "dual_projection_fp32",
        "dual_projection_norm",
        "dual_normalization",
        "unified_projection",
        "channel_major",
        "layout_aware",
        "layout_aware_norm",
        "layout_aware_norm_input",
        "reference",
        "reference_bounded",
        "geometric",
    } == set(VARIANTS)


@pytest.mark.parametrize("epsilon", [float("nan"), float("inf"), 0, -1])
def test_invalid_epsilon(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        run_variant(
            "reference", torch.randn(3, 3, 8), eps=epsilon, **weights(8, 5, False)
        )


def test_mask_cannot_expand_pair_dimensions():
    with pytest.raises(ValueError, match="Mask"):
        run_variant(
            "reference",
            torch.randn(3, 3, 8),
            mask=torch.ones(2, 3, 3),
            **weights(8, 5, False),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/HIP")
@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
@pytest.mark.parametrize("biases", [False, True])
@pytest.mark.parametrize("use_mask", [False, True])
def test_gpu_complete_update(name, direction, biases, use_mask):
    torch.manual_seed(42)
    x = torch.randn(2, 17, 17, 32, device="cuda").transpose(1, 2)
    mask = (torch.rand(1, 17, 17, device="cuda") > 0.2) if use_mask else None
    parameters = {key: value.cuda() for key, value in weights(32, 32, biases).items()}
    before = x.clone()
    parameters_before = {key: value.clone() for key, value in parameters.items()}
    with torch.no_grad():
        if name == "dual_projection" and biases:
            with pytest.raises(ValueError, match="bias support"):
                run_variant(name, x, direction, mask, **parameters)
            return
        expected = run_variant("reference", x, direction, mask, **parameters)
        output = run_variant(name, x, direction, mask, **parameters)
    assert torch.isfinite(output).all()
    if name in ("dual_projection", "dual_normalization"):
        # Historical BF16 arithmetic is not a strict-FP32 qualification.
        torch.testing.assert_close(output, expected, atol=0.025, rtol=0.025)
    else:
        torch.testing.assert_close(output, expected, atol=3e-5, rtol=2e-4)
    assert torch.equal(x, before)
    for key, value in parameters.items():
        assert torch.equal(value, parameters_before[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/HIP")
@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
def test_gpu_unequal_hidden_channels(name, direction):
    torch.manual_seed(42)
    x = torch.randn(7, 7, 32, device="cuda")
    parameters = {
        key: value.cuda()
        for key, value in weights(32, 16, False).items()
        if not key.startswith("norm_")
    }
    with torch.no_grad():
        if name in (
            "layout_aware_pipeline_mfma16",
            "layout_aware_pipeline",
            "all_triton_fused",
            "all_triton",
            "channel_major",
            "unified_projection",
            "original_channel_major",
            "original_channel_major_norm",
        ):
            with pytest.raises(ValueError, match="C_hidden == C"):
                run_variant(name, x, direction, **parameters)
            return
        expected = run_variant("reference", x, direction, **parameters)
        output = run_variant(name, x, direction, **parameters)
    assert torch.isfinite(output).all()
    atol, rtol = (
        (0.025, 0.025)
        if name in ("dual_projection", "dual_normalization")
        else (3e-5, 2e-4)
    )
    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/HIP")
@pytest.mark.parametrize("n", [64, 65, 96, 127, 128, 129, 193])
@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
def test_dual_projection_contraction_reduction_tails(n, direction):
    from geometric_nn_ops_torch.triangle.experimental.dual_projection import (
        triton_batched_matmul,
    )

    torch.manual_seed(42)
    # Exactly BF16-representable inputs isolate scheduling from rounding error.
    a = torch.randn(2, 3, n, n, device="cuda").bfloat16().float()
    b = torch.randn_like(a).bfloat16().float()
    if direction == "incoming":
        a = a.transpose(-1, -2)
    else:
        b = b.transpose(-1, -2)
    with torch.no_grad():
        expected = a @ b
        output = triton_batched_matmul(a, b)
    torch.testing.assert_close(output, expected, atol=3e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/HIP")
@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
def test_dual_projection_full_update_n65(direction):
    torch.manual_seed(42)
    x = torch.randn(1, 65, 65, 32, device="cuda")
    parameters = {key: value.cuda() for key, value in weights(32, 32, False).items()}
    mask = torch.rand(1, 65, 65, device="cuda") > 0.2
    with torch.no_grad():
        expected = run_variant("reference", x, direction, mask, **parameters)
        output = run_variant("dual_projection", x, direction, mask, **parameters)
    # The retained BF16 projections/contraction are approximate, not FP32-qualified.
    torch.testing.assert_close(output, expected, atol=0.025, rtol=0.025)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/HIP")
@pytest.mark.parametrize("n", [1, 17, 65, 127, 129, 257])
@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
@pytest.mark.parametrize("c,h", [(8, 5), (64, 64)])
def test_fp32_dual_projection_strict_tails(n, direction, c, h):
    torch.manual_seed(42)
    x = torch.randn(2, n, n, c, device="cuda").transpose(1, 2)
    mask = torch.rand(1, n, n, device="cuda") > 0.2
    parameters = {key: value.cuda() for key, value in weights(c, h, True).items()}
    with torch.no_grad():
        expected = run_variant("reference", x, direction, mask, **parameters)
        output = run_variant("dual_projection_fp32", x, direction, mask, **parameters)
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/HIP")
@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
def test_bounded_reference_forced_chunks(direction, monkeypatch):
    from geometric_nn_ops_torch.triangle.experimental import reference_bounded

    monkeypatch.setattr(reference_bounded, "LIBRARY_MAX_ELEMS", 600)
    torch.manual_seed(42)
    x = torch.randn(2, 17, 17, 8, device="cuda").transpose(1, 2)
    parameters = {key: value.cuda() for key, value in weights(8, 5, True).items()}
    with torch.no_grad():
        expected = run_variant("reference", x, direction, **parameters)
        for name in ("reference_bounded", "dual_projection_fp32"):
            output = run_variant(name, x, direction, **parameters)
            torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/HIP")
@pytest.mark.parametrize("name", ["reference_bounded", "dual_projection_fp32"])
def test_precision_safe_variants_reject_reduced_precision(name, monkeypatch):
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "high")
    x = torch.randn(3, 3, 8, device="cuda")
    parameters = {key: value.cuda() for key, value in weights(8, 5, False).items()}
    with torch.no_grad(), pytest.raises(ValueError, match="highest/IEEE"):
        run_variant(name, x, **parameters)
