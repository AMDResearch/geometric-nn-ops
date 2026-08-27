# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for triangle_multiplicative_update.

The Triton path is forward-only, so it is compared to the reference on the
forward alone. The reference carries the backward, and the public op routes to
it whenever gradients are needed.
"""

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

triton = pytest.importorskip("triton")
# ruff: noqa E402


from geometric_nn_ops_torch.triangle import _trimul, triangle_multiplicative_update
from geometric_nn_ops_torch.triangle._kernels.trimul import trimul_triton
from geometric_nn_ops_torch.triangle._reference.trimul import trimul_torch
from testutil import cast_tree
from testutil.metrics import (
    TRIMUL_TOL,
    Tolerance,
    assert_matches_oracle,
    cos,
    rel_fro,
)
from testutil.trimul import (
    BIAS_KEYS,
    assert_forward_agrees_with_the_oracle,
    make_pair_input,
    make_projection_biases,
    make_weights,
)
from utils import DTYPES_PARAM, skip_without_device_memory

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs GPU + Triton"
)

DIRECTIONS_PARAM = ["outgoing", "incoming"]


# The structural tests hold for both implementations. The accuracy tests take
# the reference as their oracle, so there it is not also a candidate.
IMPLEMENTATION_PARAM = pytest.mark.parametrize(
    "impl", [trimul_torch, trimul_triton], ids=["reference", "triton"]
)


# Structural tests: These use contrived scenarios where we know the answer
# analytically. We test both the Triton and Torch reference implementation.


def make_masked_weights(c_z, c_hidden, *, biased):
    """Weights whose gate is fixed, so only the combine can vary by position."""
    weights = make_weights(c_z, c_hidden, "cuda", torch.float32, seed=3)
    if biased:
        weights |= make_projection_biases(c_z, c_hidden, "cuda", torch.float32, seed=5)
    weights["norm_out_bias"] = torch.full((c_hidden,), 0.25, device="cuda")
    weights["g_out_weight"] = torch.zeros(c_z, c_z, device="cuda")
    return weights


@requires_gpu
@IMPLEMENTATION_PARAM
@pytest.mark.parametrize("biased", [False, True], ids=["bias_free", "biased"])
@pytest.mark.parametrize("direction", DIRECTIONS_PARAM)
def test_fully_masked_input(impl, biased, direction):
    """An all-zero mask zeroes the combine and a zeroed ``g_out_weight`` fixes
    the gate, so every output position holds the same vector."""
    N, C = 32, 128
    weights = make_masked_weights(C, C, biased=biased)
    x, _ = make_pair_input(1, N, C, "cuda", torch.float32, seed=4)
    mask = torch.zeros(1, N, N, device="cuda")

    with torch.no_grad():
        out = impl(x, direction, mask, **weights)

    # Not bit-exact: the output projection is a GEMM, and its tiling makes even
    # a constant input vary in the last place. A mask that reached the wrong
    # rows leaves real data behind, which is orders of magnitude larger.
    constant = out[:, :1, :1].expand_as(out)
    torch.testing.assert_close(out, constant, rtol=0, atol=1e-6)
    assert out.abs().max() > 0.01, "the constant field should not be trivially zero"


@requires_gpu
@IMPLEMENTATION_PARAM
@pytest.mark.parametrize("which", BIAS_KEYS)
@pytest.mark.parametrize("direction", DIRECTIONS_PARAM)
def test_each_projection_bias_contributes(impl, which, direction):
    """Zeroing any one bias changes the output, so none is silently dropped."""
    N, C = 16, 32
    weights = make_weights(C, C, "cuda", torch.float32, seed=47) | (
        make_projection_biases(C, C, "cuda", torch.float32, seed=48)
    )
    x, mask = make_pair_input(1, N, C, "cuda", torch.float32, seed=49)

    with torch.no_grad():
        out = impl(x, direction, mask, **weights)
        zeroed = impl(
            x, direction, mask, **{**weights, which: torch.zeros_like(weights[which])}
        )

    assert not torch.allclose(out, zeroed, atol=1e-5), f"{which} has no effect"


@requires_gpu
@IMPLEMENTATION_PARAM
def test_direction_changes_the_result(impl):
    """``direction`` has to reach the math: the two directions contract
    different index pairs, so on an asymmetric input they cannot agree."""
    N, C = 32, 128
    weights = make_weights(C, C, "cuda", torch.float32, seed=7)
    x, mask = make_pair_input(1, N, C, "cuda", torch.float32, seed=8)

    with torch.no_grad():
        outgoing = impl(x, "outgoing", mask, **weights)
        incoming = impl(x, "incoming", mask, **weights)

    assert not torch.allclose(outgoing, incoming, rtol=1e-2, atol=1e-2), (
        "the two directions should give different answers on an asymmetric input"
    )


@requires_gpu
@pytest.mark.parametrize(
    "impl",
    [trimul_torch, trimul_triton, triangle_multiplicative_update],
    ids=["reference", "triton", "public"],
)
def test_unknown_direction_is_rejected(impl):
    """Every entry point, since a direction that is merely misspelled would
    otherwise compute the other one."""
    N, C = 12, 32
    weights = make_weights(C, C, "cuda", torch.float32, seed=7)
    x, mask = make_pair_input(1, N, C, "cuda", torch.float32, seed=8)

    with pytest.raises(ValueError, match="direction must be one of"):
        impl(x, "outgoin", mask, **weights)


@requires_gpu
@IMPLEMENTATION_PARAM
@pytest.mark.parametrize("direction", DIRECTIONS_PARAM)
def test_mask_none(impl, direction):
    """``mask=None`` matches an explicit all-ones mask."""
    N, C = 12, 32
    weights = make_weights(C, C, "cuda", torch.float32, seed=37)
    x, _ = make_pair_input(1, N, C, "cuda", torch.float32, seed=38)

    with torch.no_grad():
        out = impl(x, direction, None, **weights)
        expected = impl(x, direction, torch.ones(1, N, N, device="cuda"), **weights)

    torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)


@requires_gpu
@IMPLEMENTATION_PARAM
@pytest.mark.parametrize("direction", DIRECTIONS_PARAM)
def test_broadcast_mask(impl, direction):
    """A mask with fewer leading dims than the input matches the expanded one."""
    N, C, batch = 12, 32, 2
    weights = make_weights(C, C, "cuda", torch.float32, seed=31)
    x, mask = make_pair_input(batch, N, C, "cuda", torch.float32, seed=32)
    broadcast = mask[0]

    with torch.no_grad():
        out = impl(x, direction, broadcast, **weights)
        expected = impl(x, direction, broadcast.expand(x.shape[:-1]), **weights)

    assert out.shape == x.shape
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


# Reference tests: these compare the Triton kernel to the pure Torch reference
# in fp64.


@requires_gpu
@pytest.mark.parametrize("direction", DIRECTIONS_PARAM)
@DTYPES_PARAM
def test_triton_accuracy(direction, dtype):
    """The kernel at ``dtype`` against the reference in fp64."""
    N, C = 32, 32
    weights = make_weights(C, C, "cuda", dtype, seed=3)
    x0, mask = make_pair_input(1, N, C, "cuda", dtype, seed=4)
    assert_forward_agrees_with_the_oracle(
        x0,
        mask,
        weights,
        direction,
        impl=trimul_triton,
        label=f"{direction}/{dtype}",
    )


@requires_gpu
@DTYPES_PARAM
def test_reference_precision(dtype):
    """How much the reference loses at ``dtype``, both sides being the
    reference. The reference is the op's fallback for inputs the kernel cannot
    serve. This is mostly just a sanity check."""
    N, C = 32, 32
    weights = make_weights(C, C, "cuda", dtype, seed=3)
    x0, mask = make_pair_input(1, N, C, "cuda", dtype, seed=4)
    assert_forward_agrees_with_the_oracle(
        x0,
        mask,
        weights,
        "outgoing",
        impl=trimul_torch,
        label=f"reference/{dtype}",
    )


@requires_gpu
@pytest.mark.parametrize("direction", DIRECTIONS_PARAM)
@pytest.mark.parametrize(
    "shape", [(2, 48, 48, 128), (2, 3, 40, 40, 128)], ids=["batched", "5D"]
)
def test_batched(direction, shape):
    """Bflat > 1, and more than one leading dim."""
    *batch, N, _, C = shape
    dtype = torch.bfloat16
    weights = make_weights(C, C, "cuda", dtype, seed=7)
    x, mask = make_pair_input(batch, N, C, "cuda", dtype, seed=11)
    assert_forward_agrees_with_the_oracle(
        x,
        mask,
        weights,
        direction,
        impl=trimul_triton,
        tol=Tolerance(1.0 - 1e-4, 1.5e-2),
        label=f"{shape}/{direction}",
    )


@requires_gpu
@pytest.mark.parametrize("direction", DIRECTIONS_PARAM)
@pytest.mark.parametrize("true_length", [32, 48], ids=["L32", "L48"])
def test_block_padded(direction, true_length):
    """Real inputs are padded to a block: the first L residues are live and the
    rest is masked, with the live block itself partly masked."""
    N, C, batch = 64, 128, 2
    dtype = torch.bfloat16
    weights = make_weights(C, C, "cuda", dtype, seed=11)
    x, _ = make_pair_input(batch, N, C, "cuda", dtype, seed=13)

    generator = torch.Generator(device="cuda").manual_seed(13)
    mask = torch.zeros(batch, N, N, device="cuda", dtype=dtype)
    for index in range(batch):
        # Distinct per-batch true lengths, so the max-L slice is not the length
        # of every element.
        live = max(1, true_length - 8 * index)
        inner = torch.rand(live, live, device="cuda", generator=generator) < 0.9
        mask[index, :live, :live] = inner.to(dtype)
    assert_forward_agrees_with_the_oracle(
        x,
        mask,
        weights,
        direction,
        impl=trimul_triton,
        tol=Tolerance(1.0 - 1e-4, 1.5e-2),
        label=f"padded/{direction}/L{true_length}",
    )


@requires_gpu
@pytest.mark.parametrize("direction", DIRECTIONS_PARAM)
@pytest.mark.parametrize("channels", [96, 128, 256], ids=["c96", "c128", "c256"])
def test_channel_widths(direction, channels):
    """Non-standard channel widths."""
    N = 24
    dtype = torch.bfloat16
    weights = make_weights(channels, channels, "cuda", dtype, seed=17)
    x0, mask = make_pair_input(1, N, channels, "cuda", dtype, seed=18)
    assert_forward_agrees_with_the_oracle(
        x0,
        mask,
        weights,
        direction,
        impl=trimul_triton,
        tol=Tolerance(1.0 - 1e-4, 2e-2),
        label=f"C{channels}/{direction}",
    )


@requires_gpu
@pytest.mark.parametrize("direction", DIRECTIONS_PARAM)
def test_hidden_differs_from_channels(direction):
    """``c_hidden`` narrower than ``c_z``, so the projections are not square."""
    N, C_z, C_hidden = 24, 64, 32
    dtype = torch.bfloat16
    weights = make_weights(C_z, C_hidden, "cuda", dtype, seed=19)
    x0, mask = make_pair_input(1, N, C_z, "cuda", dtype, seed=20)
    assert_forward_agrees_with_the_oracle(
        x0,
        mask,
        weights,
        direction,
        impl=trimul_triton,
        tol=Tolerance(1.0 - 1e-4, 2e-2),
        label=f"C{C_z}->{C_hidden}/{direction}",
    )


@requires_gpu
@pytest.mark.parametrize("direction", DIRECTIONS_PARAM)
@pytest.mark.parametrize("gamma", ["zero", "tiny", "learned"])
def test_layernorm_affine(direction, gamma):
    """Nontrivial LayerNorm affine, including zero/near-zero gamma."""
    N, C = 24, 128
    dtype = torch.bfloat16
    weights = make_weights(C, C, "cuda", dtype, seed=19)
    generator = torch.Generator(device="cuda").manual_seed(23)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

    gamma_in = torch.ones(C, device="cuda", dtype=dtype)
    gamma_out = torch.ones(C, device="cuda", dtype=dtype)
    if gamma == "zero":
        gamma_in[0] = gamma_in[5] = 0.0
        gamma_out[3] = 0.0
    elif gamma == "tiny":
        gamma_in[:3] = torch.tensor([1e-2, 1e-3, 1e-4], device="cuda")
        gamma_out[0] = 1e-3
    else:
        gamma_in = randn(C) * 0.5 + 1.0
        gamma_out = randn(C) * 0.5 + 1.0
    weights["norm_in_weight"] = gamma_in
    weights["norm_out_weight"] = gamma_out
    weights["norm_in_bias"] = randn(C) * 0.1
    weights["norm_out_bias"] = randn(C) * 0.1

    x0, mask = make_pair_input(1, N, C, "cuda", dtype, seed=20)
    assert_forward_agrees_with_the_oracle(
        x0,
        mask,
        weights,
        direction,
        impl=trimul_triton,
        tol=Tolerance(1.0 - 1e-4, 2e-2),
        label=f"{gamma}_gamma/{direction}",
    )


@requires_gpu
@pytest.mark.parametrize("direction", DIRECTIONS_PARAM)
def test_large(direction):
    """N=1024, where the reductions are far deeper than the small shapes. bf16
    only to keep memory down."""
    N, C = 1024, 128
    dtype = torch.bfloat16
    weights = make_weights(C, C, "cuda", dtype, seed=5)
    x0, mask = make_pair_input(1, N, C, "cuda", dtype, seed=6)
    assert_forward_agrees_with_the_oracle(
        x0,
        mask,
        weights,
        direction,
        impl=trimul_triton,
        tol=Tolerance(1.0 - 1e-4, 1.5e-2),
        label=f"large/{direction}",
    )


@pytest.mark.large
@requires_gpu
@pytest.mark.parametrize("direction", DIRECTIONS_PARAM)
def test_large_n_no_int32_overflow(direction):
    """At N=4096, C=128 the flattened element count reaches 2**31, where a
    32-bit offset would wrap and give a finite but wrong result. Kept in bf16
    with fp32 metrics calculations to reduce memory usage. Overflow would create
    huge differences."""
    N, C = 4096, 128
    skip_without_device_memory(gib=48)
    dtype = torch.bfloat16
    weights = cast_tree(
        make_weights(C, C, "cuda", dtype, seed=13), dtype, requires_grad=False
    )
    x, _ = make_pair_input((), N, C, "cuda", dtype, seed=47)
    mask = torch.ones(N, N, device="cuda", dtype=dtype)

    with torch.inference_mode():
        out = trimul_triton(x, direction, mask, **weights).float()
        torch.cuda.empty_cache()
        out_reference = trimul_torch(x, direction, mask, **weights).float()

    relative = rel_fro(out, out_reference, accumulate=torch.float32)
    similarity = cos(out, out_reference, accumulate=torch.float32)
    assert similarity >= 1.0 - 1e-3 and relative <= 3e-2, (
        f"the kernel diverges from the reference at N={N} "
        f"(cos={similarity:.6f}, rel-fro={relative:.3e}), which is what an "
        f"int32 offset overflow at {N * N * C} elements looks like"
    )


# Dispatch tests: conditions the Triton path cannot serve. The public op routes
# each to the reference. Adding Triton support for one of these breaks that,
# which is the signal to move the case to a full Triton test.


UNSUPPORTED_CONFIGS = {
    "cpu": {"device": "cpu"},
    "float64": {"dtype": torch.float64},
    "input_requires_grad": {"input_requires_grad": True},
    "weights_require_grad": {"weights_require_grad": True},
}


def build_config(
    *,
    device="cuda",
    dtype=torch.float32,
    input_requires_grad=False,
    weights_require_grad=False,
):
    weights = make_weights(
        32, 32, device, dtype, seed=1, requires_grad=weights_require_grad
    )
    x, mask = make_pair_input(
        1, 12, 32, device, dtype, seed=3, requires_grad=input_requires_grad
    )
    return x, mask, weights


@contextlib.contextmanager
def dispatch_spy():
    """Record which implementation the public op calls."""
    with (
        mock.patch.object(_trimul, "trimul_torch", wraps=trimul_torch) as reference,
        mock.patch.object(_trimul, "trimul_triton", wraps=trimul_triton) as kernel,
    ):
        yield SimpleNamespace(reference=reference, kernel=kernel)


@pytest.mark.parametrize(
    "config", list(UNSUPPORTED_CONFIGS.values()), ids=list(UNSUPPORTED_CONFIGS)
)
def test_unsupported_configs_use_the_reference(config):
    if config.get("device", "cuda") == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs GPU")
    x, mask, weights = build_config(**config)

    with dispatch_spy() as spy:
        triangle_multiplicative_update(x, "outgoing", mask, **weights)

    assert spy.reference.called
    assert not spy.kernel.called


@requires_gpu
@pytest.mark.parametrize(
    "no_backward", [torch.no_grad, torch.inference_mode], ids=["no_grad", "inference"]
)
def test_no_grad_modes_use_triton(no_backward):
    """Inputs keep ``requires_grad`` inside ``no_grad`` and ``inference_mode``, but no
    graph is recorded there, so the forward-only kernel is safe."""
    x, mask, weights = build_config(input_requires_grad=True, weights_require_grad=True)

    with no_backward(), dispatch_spy() as spy:
        triangle_multiplicative_update(x, "outgoing", mask, **weights)

    assert spy.kernel.called
    assert not spy.reference.called


def test_cpu_is_differentiable():
    """The public op on CPU, where there is no kernel to route to. Gradients
    still have to flow to every parameter."""
    N, C_z, C_hidden = 8, 16, 12
    weights = make_weights(
        C_z, C_hidden, "cpu", torch.float32, seed=2, requires_grad=True
    ) | make_projection_biases(
        C_z, C_hidden, "cpu", torch.float32, seed=3, requires_grad=True
    )
    x, _ = make_pair_input(
        (), N, C_z, "cpu", torch.float32, seed=49, requires_grad=True
    )
    mask = torch.ones(N, N)

    triangle_multiplicative_update(x, "outgoing", mask, **weights).sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for key, weight in weights.items():
        assert weight.grad is not None, f"{key} received no gradient"
