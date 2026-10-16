# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for attention_pair_bias.

There is no Triton implementation yet, so every test is just against the Torch
reference.
"""

import pytest
import torch

from geometric_nn_ops_torch.triangle._reference.attention_pair_bias import (
    attention_pair_bias_torch,
)
from testutil import cast_tree, masked_noise, seeded_randn
from testutil.attention_pair_bias import (
    make_key_mask,
    make_op_tensors,
    make_projection_biases,
)
from testutil.metrics import (
    ATTENTION_PAIR_BIAS_TOL,
    assert_matches_oracle,
)
from testutil.precision import torch_matmul_precision
from utils import DTYPE_PRECISION_PARAM, DTYPES_PARAM, requires_gpu

ACTIVATIONS = ("s", "q", "k", "v", "z")
PARAMETERS = (
    "proj_g_weight",
    "proj_o_weight",
    "proj_z_weight",
    "ln_z_weight",
    "proj_g_bias",
    "proj_o_bias",
)


def grad_targets(tensors):
    """Every input that will receive a gradient, in a fixed order."""
    return [
        key
        for key in ACTIVATIONS + PARAMETERS
        if tensors.get(key) is not None and tensors[key].requires_grad
    ]


def fwd_bwd(implementation, tensors, grad_out, **options):
    out, _ = implementation(**tensors, **options)
    keys = grad_targets(tensors)
    grads = torch.autograd.grad(
        out, [tensors[key] for key in keys], grad_outputs=grad_out
    )
    return out, keys, grads


def comparisons(out, keys, grads, out_oracle, grads_oracle):
    names = ["out", *(f"grad_{key}" for key in keys)]
    return list(zip(names, [out, *grads], [out_oracle, *grads_oracle], strict=True))


def assert_agrees_with_the_oracle(
    tensors, dtype, *, impl, label, precision="ieee", grad_seed=11, **options
):
    """Run the ``impl`` at ``dtype`` and the reference in fp64, compare the
    output and every gradient.

    ``precision`` only selects the tolerance; the caller sets it on torch. The
    fp64 oracle is unaffected by it.
    """
    BM, _, U, DH = tensors["q"].shape
    D = tensors["s"].shape[-1]
    grad_out = seeded_randn(BM, U, D, device="cuda", dtype=dtype, seed=grad_seed)
    mask = options.pop("mask", None)

    out, keys, grads = fwd_bwd(
        impl,
        cast_tree(tensors, dtype),
        grad_out,
        mask=mask,
        scale=DH**-0.5,
        **options,
    )
    out_oracle, _, grads_oracle = fwd_bwd(
        attention_pair_bias_torch,
        cast_tree(tensors, torch.float64),
        grad_out.double(),
        mask=None if mask is None else mask.double(),
        scale=DH**-0.5,
        **options,
    )

    assert_matches_oracle(
        comparisons(out, keys, grads, out_oracle, grads_oracle),
        ATTENTION_PAIR_BIAS_TOL[dtype, precision],
        label=f"{label}/{dtype}",
    )


# Structural tests: These use contrived scenarios where we know the answer
# analytically.


@requires_gpu
@DTYPES_PARAM
def test_masked_keys_are_ignored(dtype):
    """A masked key's ``v`` reaches the output only through a zero softmax
    weight, so it should leave the output bit-identical however large it is."""
    B, M, H, U, V, DH, Zc = 1, 1, 4, 32, 32, 16, 16
    tensors = cast_tree(
        make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=3),
        dtype,
        requires_grad=False,
    )
    mask = make_key_mask(B, V, "cuda", dtype, seed=5)
    dropped = (mask == 0).reshape(B, 1, V, 1)
    assert dropped.any(), "the fixture should mask at least one key"
    perturbed = {**tensors, "v": masked_noise(tensors["v"], dropped, 7)}

    with torch.no_grad():
        out, _ = attention_pair_bias_torch(**tensors, mask=mask, scale=DH**-0.5)
        out_perturbed, _ = attention_pair_bias_torch(
            **perturbed, mask=mask, scale=DH**-0.5
        )
        unmasked, _ = attention_pair_bias_torch(**tensors, scale=DH**-0.5)
        unmasked_perturbed, _ = attention_pair_bias_torch(**perturbed, scale=DH**-0.5)

    torch.testing.assert_close(out, out_perturbed, rtol=0, atol=0)
    assert not torch.allclose(unmasked, unmasked_perturbed), (
        "the perturbation does not change the result even unmasked"
    )


@requires_gpu
@DTYPES_PARAM
def test_masked_logits_are_suppressed(dtype):
    """``k`` and the pair rep at masked keys move those logits, and the masking
    magnitude has to bury them whatever they are."""
    B, M, H, U, V, DH, Zc = 1, 1, 4, 32, 32, 16, 16
    tensors = cast_tree(
        make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=3),
        dtype,
        requires_grad=False,
    )
    mask = make_key_mask(B, V, "cuda", dtype, seed=5)
    dropped = (mask == 0).reshape(B, 1, V, 1)
    perturbed = {
        **tensors,
        "k": masked_noise(tensors["k"], dropped, 11),
        "z": masked_noise(tensors["z"], (mask == 0).reshape(B, 1, V, 1), 13),
    }

    with torch.no_grad():
        out, _ = attention_pair_bias_torch(**tensors, mask=mask, scale=DH**-0.5)
        out_perturbed, _ = attention_pair_bias_torch(
            **perturbed, mask=mask, scale=DH**-0.5
        )
        unmasked, _ = attention_pair_bias_torch(**tensors, scale=DH**-0.5)
        unmasked_perturbed, _ = attention_pair_bias_torch(**perturbed, scale=DH**-0.5)

    torch.testing.assert_close(out, out_perturbed, rtol=0, atol=0)
    assert not torch.allclose(unmasked, unmasked_perturbed), (
        "the perturbation does not change the result even unmasked"
    )


@requires_gpu
@DTYPES_PARAM
@pytest.mark.parametrize("multiplicity", [1, 3], ids=["m1", "m3"])
def test_cached_pair_bias(dtype, multiplicity):
    """``return_z_proj`` hands back the projected bias so inference can cache
    it; feeding it back with ``z_is_projected`` skips the LayerNorm and the
    projection and changes nothing else.

    The bias is per batch element, so it carries no multiplicity axis and stays
    reusable across ``M``. It is not masked, so the cached call takes the same
    mask the uncached one did.
    """
    B, M, H, U, V, DH, Zc = 2, multiplicity, 4, 32, 32, 16, 16
    tensors = cast_tree(
        make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=31),
        dtype,
        requires_grad=False,
    )
    mask = make_key_mask(B, V, "cuda", seed=55, frac_keep=0.6)
    assert (mask == 0).any(), "the fixture should mask at least one key"

    with torch.no_grad():
        full, projected = attention_pair_bias_torch(
            **tensors, mask=mask, scale=DH**-0.5, return_z_proj=True
        )
        cached, _ = attention_pair_bias_torch(
            **{**tensors, "z": projected},
            mask=mask,
            scale=DH**-0.5,
            z_is_projected=True,
        )

    assert projected.shape == (B, H, U, V)
    torch.testing.assert_close(full, cached, rtol=0, atol=0)


@requires_gpu
@DTYPES_PARAM
def test_returned_pair_bias_is_not_masked(dtype):
    """The returned bias is the projection of ``z``. Contrary to the
    cuequivariance docstring, it does not have the mask folded in."""
    B, M, H, U, V, DH, Zc = 2, 1, 4, 32, 32, 16, 16
    tensors = cast_tree(
        make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=31),
        dtype,
        requires_grad=False,
    )
    mask = make_key_mask(B, V, "cuda", seed=55, frac_keep=0.6)
    assert (mask == 0).any(), "the fixture should mask at least one key"

    with torch.no_grad():
        _, masked = attention_pair_bias_torch(
            **tensors, mask=mask, scale=DH**-0.5, return_z_proj=True
        )
        _, unmasked = attention_pair_bias_torch(
            **tensors, scale=DH**-0.5, return_z_proj=True
        )

    torch.testing.assert_close(masked, unmasked, rtol=0, atol=0)
    assert torch.isfinite(masked).all()


@requires_gpu
@pytest.mark.parametrize("multiplicity", [1, 3], ids=["m1", "m3"])
def test_per_copy_mask_with_a_cached_bias(multiplicity):
    """A ``(BM, V)`` mask names a key per copy."""
    B, M, H, U, V, DH, Zc = 2, multiplicity, 4, 32, 32, 16, 16
    tensors = make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=41)
    per_copy = make_key_mask(B * M, V, "cuda", seed=43, frac_keep=0.6)

    with torch.no_grad():
        out, projected = attention_pair_bias_torch(
            **tensors, mask=per_copy, scale=DH**-0.5, return_z_proj=True
        )

    assert projected.shape == (B, H, U, V)
    assert torch.isfinite(out).all()


@requires_gpu
def test_inf_argument_is_honoured():
    """``inf`` is how far masked keys are pushed below the real ones. It is an
    argument, not a constant: a caller can weaken it, and a weak one leaves
    masked keys contributing.
    """
    B, M, H, U, V, DH, Zc = 1, 1, 4, 32, 32, 16, 16
    tensors = cast_tree(
        make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=57),
        torch.float32,
        requires_grad=False,
    )
    mask = make_key_mask(B, V, "cuda", seed=59, frac_keep=0.5)

    with torch.no_grad():
        suppressed, _ = attention_pair_bias_torch(
            **tensors, mask=mask, scale=DH**-0.5, inf=1e6
        )
        barely, _ = attention_pair_bias_torch(
            **tensors, mask=mask, scale=DH**-0.5, inf=1.0
        )

    assert not torch.allclose(suppressed, barely, rtol=1e-2, atol=1e-2), (
        "the inf argument had no effect on the logits"
    )


@requires_gpu
@DTYPES_PARAM
def test_fully_masked_rows(dtype):
    """A row with every key masked can lead to NaNs if the mask val overflows."""
    B, M, H, U, V, DH, Zc = 1, 1, 4, 32, 32, 16, 16
    tensors = cast_tree(
        make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=61),
        dtype,
        requires_grad=False,
    )
    mask = torch.ones(B, V, device="cuda", dtype=dtype)
    mask[:, :] = 0.0

    with torch.no_grad():
        out, projected = attention_pair_bias_torch(
            **tensors, mask=mask, scale=DH**-0.5, return_z_proj=True
        )

    assert torch.isfinite(out).all(), "a fully masked row must not produce NaN"
    assert torch.isfinite(projected).all(), "the cached bias must stay finite"


@requires_gpu
def test_mask_layouts_agree():
    """A ``(B, V)`` mask covers every copy on the multiplicity axis, so it must
    match the ``(BM, V)`` mask that repeats it. The two take different branches
    through the mask handling."""
    B, M, H, U, V, DH, Zc = 2, 3, 4, 16, 16, 8, 16
    tensors = make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=3)
    per_batch = make_key_mask(B, V, "cuda", seed=5, frac_keep=0.6)
    per_copy = per_batch.repeat_interleave(M, dim=0)
    assert (per_batch == 0).any(), "the fixture should mask at least one key"

    with torch.no_grad():
        from_batch, _ = attention_pair_bias_torch(
            **tensors, mask=per_batch, scale=DH**-0.5
        )
        from_copy, _ = attention_pair_bias_torch(
            **tensors, mask=per_copy, scale=DH**-0.5
        )

    torch.testing.assert_close(from_batch, from_copy, rtol=0, atol=0)


@requires_gpu
def test_pair_bias_is_shared_across_the_multiplicity_axis():
    """``z`` carries no multiplicity axis, so the same bias reaches all ``M``
    copies of a batch element. Copies given identical inputs must come back
    identical, and different batch elements must not."""
    B, M, H, U, V, DH, Zc = 2, 3, 4, 16, 16, 8, 16
    tensors = make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=7)
    for key in ("s", "q", "k", "v"):
        grouped = tensors[key].reshape(B, M, *tensors[key].shape[1:])
        tensors[key] = (
            grouped[:, :1].expand_as(grouped).reshape(tensors[key].shape).contiguous()
        )

    with torch.no_grad():
        out, _ = attention_pair_bias_torch(**tensors, scale=DH**-0.5)

    out = out.reshape(B, M, *out.shape[1:])
    for copy in range(1, M):
        torch.testing.assert_close(out[:, copy], out[:, 0], rtol=0, atol=0)
    assert not torch.equal(out[0, 0], out[1, 0]), (
        "different batch elements share a pair bias; the broadcast is wrong"
    )


# Precision: with no Triton path yet both sides are the reference, so this only
# bounds how much precision is lost at each dtype.


@requires_gpu
@DTYPE_PRECISION_PARAM
def test_reference_precision(dtype, precision):
    """The reference at ``dtype`` and ``precision`` against the same math in fp64."""
    B, M, H, U, V, DH, Zc = 1, 1, 4, 32, 32, 16, 16
    tensors = make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=3)
    with torch_matmul_precision(precision):
        assert_agrees_with_the_oracle(
            tensors,
            dtype,
            precision=precision,
            impl=attention_pair_bias_torch,
            mask=make_key_mask(B, V, "cuda", seed=9),
            label="base",
        )


@requires_gpu
@DTYPES_PARAM
def test_reference_precision_with_optional_projection_biases(dtype):
    """The same fp64-vs-``dtype`` check with the gate and output biases present,
    which are otherwise unexercised. Both sides are the reference."""
    B, M, H, U, V, DH, Zc = 1, 1, 4, 32, 32, 16, 16
    tensors = make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=3)
    tensors |= make_projection_biases(H, DH, "cuda", torch.float32, seed=5)

    assert_agrees_with_the_oracle(
        tensors, dtype, impl=attention_pair_bias_torch, label="projection-biases"
    )
