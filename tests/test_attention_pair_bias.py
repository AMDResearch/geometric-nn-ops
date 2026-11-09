# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for attention_pair_bias."""

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

triton = pytest.importorskip("triton")
# ruff: noqa: E402

from geometric_nn_ops_torch.triangle import _attention_pair_bias, attention_pair_bias
from geometric_nn_ops_torch.triangle._kernels.attention_pair_bias import (
    attention_pair_bias_triton,
)
from geometric_nn_ops_torch.triangle._reference.attention_pair_bias import (
    attention_pair_bias_torch,
)
from testutil import cast_tree, masked_noise
from testutil.attention_pair_bias import (
    assert_agrees_with_the_oracle,
    assert_forward_agrees_with_the_oracle,
    make_key_mask,
    make_op_tensors,
    make_projection_biases,
)
from testutil.metrics import cos, rel_fro
from testutil.precision import torch_matmul_precision
from utils import (
    DTYPE_PRECISION_PARAM,
    DTYPES_PARAM,
    gpu_arch,
    require_device_memory,
    requires_gpu,
    skip_without_reduced_precision_dot,
)

# The structural tests hold for both implementations. The accuracy tests take
# the reference as their oracle, so there it is not also a candidate.
IMPLEMENTATION_PARAM = pytest.mark.parametrize(
    "impl",
    [attention_pair_bias_torch, attention_pair_bias_triton],
    ids=["reference", "triton"],
)

# Structural tests: These use contrived scenarios where we know the answer
# analytically.


@requires_gpu
@DTYPES_PARAM
@IMPLEMENTATION_PARAM
def test_masked_keys_are_ignored(request, impl, dtype):
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
        out, _ = impl(**tensors, mask=mask, scale=DH**-0.5)
        out_perturbed, _ = impl(**perturbed, mask=mask, scale=DH**-0.5)
        unmasked, _ = impl(**tensors, scale=DH**-0.5)
        unmasked_perturbed, _ = impl(**perturbed, scale=DH**-0.5)

    # As in the triangle attention test of the same name, 16-bit Triton on
    # gfx1151 is off by one ulp. Do a close test first and then xfail that
    # combination before doing the exact test.
    torch.testing.assert_close(out, out_perturbed)
    assert not torch.allclose(unmasked, unmasked_perturbed), (
        "the perturbation does not change the result even unmasked"
    )

    if (
        impl is attention_pair_bias_triton
        and dtype in (torch.float16, torch.bfloat16)
        and gpu_arch() == "gfx1151"
    ):
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason="gfx1151 16-bit: masked v perturbs the output by one ulp",
            )
        )

    torch.testing.assert_close(out, out_perturbed, rtol=0, atol=0)


@requires_gpu
@DTYPES_PARAM
@IMPLEMENTATION_PARAM
def test_masked_logits_are_suppressed(impl, dtype):
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
        out, _ = impl(**tensors, mask=mask, scale=DH**-0.5)
        out_perturbed, _ = impl(**perturbed, mask=mask, scale=DH**-0.5)
        unmasked, _ = impl(**tensors, scale=DH**-0.5)
        unmasked_perturbed, _ = impl(**perturbed, scale=DH**-0.5)

    torch.testing.assert_close(out, out_perturbed, rtol=0, atol=0)
    assert not torch.allclose(unmasked, unmasked_perturbed), (
        "the perturbation does not change the result even unmasked"
    )


@requires_gpu
@DTYPES_PARAM
@pytest.mark.parametrize("multiplicity", [1, 3], ids=["m1", "m3"])
@IMPLEMENTATION_PARAM
def test_cached_pair_bias(impl, dtype, multiplicity):
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
        full, projected = impl(**tensors, mask=mask, scale=DH**-0.5, return_z_proj=True)
        cached, _ = impl(
            **{**tensors, "z": projected},
            mask=mask,
            scale=DH**-0.5,
            z_is_projected=True,
        )

    assert projected.shape == (B, H, U, V)
    torch.testing.assert_close(full, cached, rtol=0, atol=0)


@requires_gpu
@DTYPES_PARAM
@IMPLEMENTATION_PARAM
def test_returned_pair_bias_is_not_masked(impl, dtype):
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
        _, masked = impl(**tensors, mask=mask, scale=DH**-0.5, return_z_proj=True)
        _, unmasked = impl(**tensors, scale=DH**-0.5, return_z_proj=True)

    torch.testing.assert_close(masked, unmasked, rtol=0, atol=0)
    assert torch.isfinite(masked).all()


@requires_gpu
@pytest.mark.parametrize("multiplicity", [1, 3], ids=["m1", "m3"])
@IMPLEMENTATION_PARAM
def test_per_copy_mask_with_a_cached_bias(impl, multiplicity):
    """A ``(BM, V)`` mask names a key per copy."""
    B, M, H, U, V, DH, Zc = 2, multiplicity, 4, 32, 32, 16, 16
    tensors = make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=41)
    per_copy = make_key_mask(B * M, V, "cuda", seed=43, frac_keep=0.6)

    with torch.no_grad():
        out, projected = impl(
            **tensors, mask=per_copy, scale=DH**-0.5, return_z_proj=True
        )

    assert projected.shape == (B, H, U, V)
    assert torch.isfinite(out).all()


@requires_gpu
@IMPLEMENTATION_PARAM
def test_inf_argument_is_honoured(impl):
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
        suppressed, _ = impl(**tensors, mask=mask, scale=DH**-0.5, inf=1e6)
        barely, _ = impl(**tensors, mask=mask, scale=DH**-0.5, inf=1.0)

    assert not torch.allclose(suppressed, barely, rtol=1e-2, atol=1e-2), (
        "the inf argument had no effect on the logits"
    )


@requires_gpu
@DTYPES_PARAM
@IMPLEMENTATION_PARAM
def test_fully_masked_rows(impl, dtype):
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
        out, projected = impl(**tensors, mask=mask, scale=DH**-0.5, return_z_proj=True)

    assert torch.isfinite(out).all(), "a fully masked row must not produce NaN"
    assert torch.isfinite(projected).all(), "the cached bias must stay finite"


@requires_gpu
@IMPLEMENTATION_PARAM
def test_mask_layouts_agree(impl):
    """A ``(B, V)`` mask covers every copy on the multiplicity axis, so it must
    match the ``(BM, V)`` mask that repeats it. The two take different branches
    through the mask handling."""
    B, M, H, U, V, DH, Zc = 2, 3, 4, 16, 16, 8, 16
    tensors = make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=3)
    per_batch = make_key_mask(B, V, "cuda", seed=5, frac_keep=0.6)
    per_copy = per_batch.repeat_interleave(M, dim=0)
    assert (per_batch == 0).any(), "the fixture should mask at least one key"

    with torch.no_grad():
        from_batch, _ = impl(**tensors, mask=per_batch, scale=DH**-0.5)
        from_copy, _ = impl(**tensors, mask=per_copy, scale=DH**-0.5)

    torch.testing.assert_close(from_batch, from_copy, rtol=0, atol=0)


@requires_gpu
@IMPLEMENTATION_PARAM
def test_pair_bias_is_shared_across_the_multiplicity_axis(impl):
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
        out, _ = impl(**tensors, scale=DH**-0.5)

    out = out.reshape(B, M, *out.shape[1:])
    for copy in range(1, M):
        torch.testing.assert_close(out[:, copy], out[:, 0], rtol=0, atol=0)
    assert not torch.equal(out[0, 0], out[1, 0]), (
        "different batch elements share a pair bias; the broadcast is wrong"
    )


@requires_gpu
@pytest.mark.parametrize("grads", [True, False], ids=["train", "inference"])
def test_tf32_changes_the_result(grads):
    """Make sure the reduced precision actually affects the output."""
    skip_without_reduced_precision_dot()
    B, M, H, U, V, DH, Zc = 1, 1, 4, 32, 32, 16, 32
    tensors = make_op_tensors(
        B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=3, requires_grad=grads
    )

    outputs = {}
    for precision in ("ieee", "tf32"):
        with torch_matmul_precision(precision), torch.set_grad_enabled(grads):
            outputs[precision], _ = attention_pair_bias_triton(**tensors)

    assert not torch.allclose(outputs["ieee"], outputs["tf32"], rtol=0, atol=0), (
        "bit-identical output for ieee and tf32"
    )


# Reference tests: these compare the implementations to the pure Torch reference
# in fp64.


@requires_gpu
@DTYPE_PRECISION_PARAM
@pytest.mark.parametrize("multiplicity", [1, 3], ids=["m1", "m3"])
def test_triton_accuracy(dtype, precision, multiplicity):
    """The Triton path at ``dtype`` and ``precision``, forward and backward.
    With gradients, the pair bias takes the composed LayerNorm + projection."""
    B, M, H, U, V, DH, Zc = 2, multiplicity, 4, 32, 32, 16, 16
    tensors = make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=3)
    with torch_matmul_precision(precision):
        assert_agrees_with_the_oracle(
            tensors,
            dtype,
            precision=precision,
            impl=attention_pair_bias_triton,
            mask=make_key_mask(B, V, "cuda", seed=9),
            label=f"triton/m{multiplicity}/{precision}",
        )


@requires_gpu
@DTYPE_PRECISION_PARAM
@pytest.mark.parametrize("multiplicity", [1, 3], ids=["m1", "m3"])
def test_triton_accuracy_under_inference_mode(dtype, precision, multiplicity):
    """Without gradients the pair bias takes the fused LayerNorm + projection
    kernel."""
    B, M, H, U, V, DH, Zc = 2, multiplicity, 4, 32, 32, 16, 16
    tensors = make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=3)
    with torch_matmul_precision(precision):
        assert_forward_agrees_with_the_oracle(
            tensors,
            dtype,
            precision=precision,
            impl=attention_pair_bias_triton,
            mask=make_key_mask(B, V, "cuda", seed=9),
            label=f"inference/m{multiplicity}/{precision}",
        )


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


@requires_gpu
@pytest.mark.parametrize("grads", [True, False], ids=["train", "inference"])
def test_triton_without_layernorm_weight(grads):
    """cuequivariance's ``w_ln_z`` defaults to ``None``. The fused kernel needs
    the weight, so this takes the composed path even without gradients."""
    B, M, H, U, V, DH, Zc = 1, 1, 4, 32, 32, 16, 16
    tensors = make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=3)
    tensors["ln_z_weight"] = None
    check = (
        assert_agrees_with_the_oracle
        if grads
        else (assert_forward_agrees_with_the_oracle)
    )
    check(
        tensors,
        torch.bfloat16,
        impl=attention_pair_bias_triton,
        mask=make_key_mask(B, V, "cuda", seed=9),
        label="no-ln-weight",
    )


@requires_gpu
@pytest.mark.parametrize("multiplicity", [1, 3], ids=["m1", "m3"])
def test_triton_per_copy_mask(multiplicity):
    """A ``(BM, V)`` mask that differs between the copies."""
    B, M, H, U, V, DH, Zc = 2, multiplicity, 4, 32, 32, 16, 16
    tensors = make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=3)
    assert_agrees_with_the_oracle(
        tensors,
        torch.bfloat16,
        impl=attention_pair_bias_triton,
        mask=make_key_mask(B * M, V, "cuda", seed=13, frac_keep=0.6),
        label=f"per-copy/m{multiplicity}",
    )


@requires_gpu
@DTYPES_PARAM
def test_triton_with_optional_projection_biases(dtype):
    """The gate and output biases, which are otherwise unexercised."""
    B, M, H, U, V, DH, Zc = 1, 1, 4, 32, 32, 16, 16
    tensors = make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=3)
    tensors |= make_projection_biases(H, DH, "cuda", torch.float32, seed=5)

    assert_agrees_with_the_oracle(
        tensors, dtype, impl=attention_pair_bias_triton, label="projection-biases"
    )


@requires_gpu
@pytest.mark.parametrize(
    "H, DH, Zc",
    [(3, 16, 16), (2, 32, 96), (8, 48, 128), (16, 24, 64)],
    ids=["h3", "zc96", "of3", "h16"],
)
@pytest.mark.parametrize("grads", [True, False], ids=["train", "inference"])
def test_triton_widths(H, DH, Zc, grads):
    """Head counts, head dims and pair channel widths, including widths that
    are not a power of two and so leave masked lanes in the kernels' blocks."""
    B, M, U, V = 1, 2, 48, 48
    tensors = make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=17)
    check = (
        assert_agrees_with_the_oracle
        if grads
        else (assert_forward_agrees_with_the_oracle)
    )
    check(
        tensors,
        torch.bfloat16,
        impl=attention_pair_bias_triton,
        mask=make_key_mask(B, V, "cuda", seed=19),
        label=f"H{H}/DH{DH}/Zc{Zc}",
    )


@requires_gpu
@pytest.mark.parametrize("gamma", ["zero", "tiny", "learned"])
@pytest.mark.parametrize("grads", [True, False], ids=["train", "inference"])
def test_triton_layernorm_weight(gamma, grads):
    """Nontrivial z LayerNorm weight, including zero and near-zero entries."""
    B, M, H, U, V, DH, Zc = 1, 1, 4, 32, 32, 16, 64
    tensors = make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=21)
    weight = torch.ones(Zc, device="cuda")
    if gamma == "zero":
        weight[0] = weight[5] = 0.0
    elif gamma == "tiny":
        weight[:3] = torch.tensor([1e-2, 1e-3, 1e-4], device="cuda")
    else:
        g = torch.Generator(device="cuda").manual_seed(23)
        weight = torch.randn(Zc, device="cuda", generator=g) * 0.5 + 1.0
    tensors["ln_z_weight"] = weight
    check = (
        assert_agrees_with_the_oracle
        if grads
        else (assert_forward_agrees_with_the_oracle)
    )
    check(
        tensors,
        torch.bfloat16,
        impl=attention_pair_bias_triton,
        mask=make_key_mask(B, V, "cuda", seed=25),
        label=f"{gamma}_gamma",
    )


@requires_gpu
@pytest.mark.parametrize("true_length", [32, 48], ids=["L32", "L48"])
def test_triton_block_padded(true_length):
    """Real inputs are padded to a block: the first L residues are live and the
    rest is masked, with the live block itself partly masked."""
    B, M, H, U, V, DH, Zc = 2, 1, 4, 64, 64, 16, 32
    tensors = make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=27)
    generator = torch.Generator(device="cuda").manual_seed(29)
    mask = torch.zeros(B, V, device="cuda")
    for index in range(B):
        # Distinct per-batch true lengths, so the longest is not the length of
        # every element.
        live = max(1, true_length - 8 * index)
        mask[index, :live] = (
            torch.rand(live, device="cuda", generator=generator) < 0.9
        ).float()
        mask[index, 0] = 1.0
    assert_agrees_with_the_oracle(
        tensors,
        torch.bfloat16,
        impl=attention_pair_bias_triton,
        mask=mask,
        label=f"padded/L{true_length}",
    )


@requires_gpu
@pytest.mark.parametrize(
    "autocast_dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"]
)
def test_triton_autocast(autocast_dtype):
    """Under autocast the caller passes fp32 tensors while the compute dtype is
    narrower, and every Triton kernel has to agree on one dtype."""
    B, M, H, U, V, DH, Zc = 1, 1, 4, 32, 32, 16, 16
    tensors = make_op_tensors(
        B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=3, requires_grad=True
    )
    mask = make_key_mask(B, V, "cuda", seed=9)

    with torch.autocast("cuda", dtype=autocast_dtype):
        out, _ = attention_pair_bias_triton(**tensors, mask=mask)
        with torch.no_grad():
            inferred, _ = attention_pair_bias_triton(**tensors, mask=mask)
    out.float().sum().backward()

    assert out.dtype == autocast_dtype
    assert inferred.dtype == autocast_dtype
    for key, tensor in tensors.items():
        assert tensor.grad is not None, f"{key} received no gradient"
        assert torch.isfinite(tensor.grad).all(), f"{key} has a non-finite gradient"


@pytest.mark.large
@requires_gpu
@pytest.mark.parametrize("multiplicity", [1, 3], ids=["m1", "m3"])
def test_large(multiplicity):
    """N=1024 at the OpenFold-3 head and pair widths, where the reductions are
    far deeper than the small shapes. bf16 only to keep memory down."""
    # Peaks at ~7.5 GiB allocated, mostly the fp64 oracle.
    require_device_memory(gib=12)
    B, M, H, U, V, DH, Zc = 1, multiplicity, 16, 1024, 1024, 48, 128
    tensors = make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=31)
    assert_agrees_with_the_oracle(
        tensors,
        torch.bfloat16,
        impl=attention_pair_bias_triton,
        mask=make_key_mask(B, V, "cuda", seed=33),
        label=f"large/m{multiplicity}",
    )


@pytest.mark.large
@requires_gpu
@pytest.mark.parametrize("grads", [True, False], ids=["train", "inference"])
def test_large_n_no_int32_overflow(grads):
    """At N=5120 with Zc=128 the pair rep has more than 2**31 elements, where a
    32-bit offset would wrap and give a finite but wrong result. bf16, with the
    metrics accumulated in fp32, to keep memory down."""
    B, M, H, U, V, DH, Zc = 1, 1, 4, 5120, 5120, 16, 128
    require_device_memory(gib=48)
    tensors = cast_tree(
        make_op_tensors(B, M, H, U, V, DH, Zc, "cuda", torch.float32, seed=3),
        torch.bfloat16,
        requires_grad=False,
    )
    mask = make_key_mask(B, V, "cuda", torch.bfloat16, seed=9)

    # With a gradient required the pair bias takes the Triton LayerNorm,
    # without one the fused kernel.
    tensors["z"].requires_grad_(grads)
    with torch.inference_mode(not grads):
        out, projected = attention_pair_bias_triton(
            **tensors, mask=mask, return_z_proj=True
        )
    out, projected = out.detach().float(), projected.detach().float()
    torch.cuda.empty_cache()
    with torch.no_grad():
        out_reference, projected_reference = attention_pair_bias_torch(
            **tensors, mask=mask, return_z_proj=True
        )

    for name, actual, expected in (
        ("out", out, out_reference.float()),
        ("pair bias", projected, projected_reference.float()),
    ):
        relative = rel_fro(actual, expected, accumulate=torch.float32)
        similarity = cos(actual, expected, accumulate=torch.float32)
        assert similarity >= 1.0 - 1e-3 and relative <= 3e-2, (
            f"{name} diverges from the reference at N={U} (cos={similarity:.6f}, "
            f"rel-fro={relative:.3e}), which is what an int32 offset overflow "
            f"looks like"
        )


# Dispatch tests: conditions the Triton path cannot serve. The public op routes
# each to the reference. Adding Triton support for one of these breaks that,
# which is the signal to move the case to a full Triton test.


@contextlib.contextmanager
def dispatch_spy():
    """Record which implementation the public op calls."""
    with (
        mock.patch.object(
            _attention_pair_bias,
            "attention_pair_bias_torch",
            wraps=attention_pair_bias_torch,
        ) as reference,
        mock.patch.object(
            _attention_pair_bias,
            "attention_pair_bias_triton",
            wraps=attention_pair_bias_triton,
        ) as kernel,
    ):
        yield SimpleNamespace(reference=reference, kernel=kernel)


@pytest.fixture
def no_size_gate(monkeypatch):
    """Route every call the Triton path can serve to it, whatever its size, so
    that a dispatch test checks the condition it names and not the size gate."""
    monkeypatch.setattr(_attention_pair_bias, "_TRAIN_TRITON_MIN_WORK", 0)
    monkeypatch.setattr(_attention_pair_bias, "_INFER_TRITON_MIN_WORK", 0)


# Configs the Triton path does not implement, which the public op has to route
# to the torch path.
UNSUPPORTED_CONFIGS = {
    "cpu": {"device": "cpu"},
    "float64": {"dtype": torch.float64},
    "non_square": {"v": 48},
}


def build_config(
    *, device="cuda", dtype=torch.float32, u=32, v=32, requires_grad=False
):
    B, M, H, DH, Zc = 1, 1, 4, 16, 16
    return make_op_tensors(
        B, M, H, u, v, DH, Zc, device, dtype, seed=3, requires_grad=requires_grad
    )


@pytest.mark.parametrize(
    "config", list(UNSUPPORTED_CONFIGS.values()), ids=list(UNSUPPORTED_CONFIGS)
)
def test_unsupported_configs_use_the_reference(config, no_size_gate):
    if config.get("device", "cuda") == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs GPU")
    tensors = build_config(**config)

    with dispatch_spy() as spy:
        attention_pair_bias(**tensors)

    assert spy.reference.called
    assert not spy.kernel.called


@requires_gpu
def test_triton_rejects_non_square():
    tensors = build_config(v=48)
    with pytest.raises(ValueError, match="U == V"):
        attention_pair_bias_triton(**tensors)


def test_cpu_is_differentiable():
    """The public op on CPU, where there is no kernel to route to. Gradients
    still have to flow to every parameter."""
    tensors = build_config(device="cpu", requires_grad=True)
    tensors |= make_projection_biases(4, 16, "cpu", torch.float32, requires_grad=True)

    out, _ = attention_pair_bias(**tensors)
    out.sum().backward()

    for key, tensor in tensors.items():
        assert tensor.grad is not None, f"{key} received no gradient"
        assert torch.isfinite(tensor.grad).all(), f"{key} has a non-finite gradient"


# Size gate tests: the Triton path serves these inputs, but below a size the
# public op routes them to the reference because its launch overhead dominates.


@requires_gpu
@pytest.mark.parametrize("requires_grad", [False, True], ids=["inference", "train"])
def test_small_inputs_use_the_reference(requires_grad):
    tensors = build_config(requires_grad=requires_grad)
    with dispatch_spy() as spy:
        attention_pair_bias(**tensors)
    assert spy.reference.called
    assert not spy.kernel.called


@requires_gpu
@pytest.mark.parametrize("requires_grad", [False, True], ids=["inference", "train"])
def test_large_inputs_use_triton(requires_grad, monkeypatch):
    """The gate compares a work estimate with a threshold per regime, so
    lowering only this regime's threshold below a small input's estimate routes
    it to Triton."""
    tensors = build_config(requires_grad=requires_grad)
    this, other = ("_TRAIN", "_INFER") if requires_grad else ("_INFER", "_TRAIN")
    monkeypatch.setattr(_attention_pair_bias, f"{this}_TRITON_MIN_WORK", 0)
    monkeypatch.setattr(_attention_pair_bias, f"{other}_TRITON_MIN_WORK", 2**62)
    with dispatch_spy() as spy:
        attention_pair_bias(**tensors)
    assert spy.kernel.called
    assert not spy.reference.called
