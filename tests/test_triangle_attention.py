# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for triangle_attention."""

# The kernel imports run after `importorskip` so the module skips cleanly when
# Triton is missing; E402 (import not at top of file) is disabled accordingly.
# ruff: noqa: E402

import pytest
import torch

triton = pytest.importorskip("triton")

from geometric_nn_ops_torch.triangle import triangle_attention
from geometric_nn_ops_torch.triangle._reference.triangle_attention import (
    triangle_attention_torch,
)
from geometric_nn_ops_torch.triangle._triattn import NEG_INF
from testutil import cast_tree, generator, make_grad_output
from testutil.metrics import (
    TRIANGLE_ATTENTION_TOL,
    Tolerance,
    assert_matches_oracle,
)
from testutil.triangle_attention import make_inputs, make_key_mask
from utils import DTYPES_PARAM

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs GPU + Triton"
)

SHAPES = [
    pytest.param((1, 1, 32, 4, 32), id="pairformer_32"),
    pytest.param((1, 4, 32, 8, 8), id="msa_d8"),
    pytest.param((1, 1, 64, 16, 48), id="atom_d48"),
    pytest.param((2, 2, 48, 4, 32), id="batched"),
]

# There is no torch fallback: the public entrypoint raises when Triton is
# unavailable, so it is the Triton path. The reference takes the canonical 5D
# layout only, and is the oracle rather than something the op can run.
IMPLEMENTATIONS = {
    "reference": triangle_attention_torch,
    "triton": triangle_attention,
}
IMPLEMENTATION_PARAM = pytest.mark.parametrize(
    "impl", list(IMPLEMENTATIONS.values()), ids=list(IMPLEMENTATIONS.keys())
)


def fwd_bwd(run, tensors, mask, scale, grad_out):
    """Forward plus gradients wrt q, k, v and, when present, the bias."""
    out = run(**tensors, mask=mask, scale=scale)
    targets = [tensors[k] for k in ("q", "k", "v")]
    if "bias" in tensors:
        targets.append(tensors["bias"])
    return out, torch.autograd.grad(out, targets, grad_outputs=grad_out)


def comparisons(out, grads, out_oracle, grads_oracle):
    """Name every (candidate, oracle) pair the accuracy assertion should cover."""
    names = ["out", "grad_q", "grad_k", "grad_v", "grad_bias"][: 1 + len(grads)]
    return list(zip(names, [out, *grads], [out_oracle, *grads_oracle], strict=True))


def assert_agrees_with_the_oracle(
    tensors, mask, scale, *, impl=triangle_attention, tol=None, label=""
):
    """Run the ``impl`` on the inputs as given and the reference in fp64,
    compare the output and every input gradient."""
    dtype = tensors["q"].dtype
    grad_out = make_grad_output(
        *tensors["q"].shape, device=tensors["q"].device, dtype=dtype, seed=31
    )

    out, grads = fwd_bwd(impl, tensors, mask, scale, grad_out)
    out_oracle, grads_oracle = fwd_bwd(
        triangle_attention_torch,
        cast_tree(tensors, torch.float64),
        mask,
        scale,
        grad_out.double(),
    )

    assert_matches_oracle(
        comparisons(out, grads, out_oracle, grads_oracle),
        tol or TRIANGLE_ATTENTION_TOL[dtype],
        label=label,
    )


# Structural tests: These use contrived scenarios where we know the answer
# analytically. We test both the Triton and Torch reference implementation,
# except where the case is the adapter in the public entrypoint, which the
# reference does not have.


def uniform_attention_inputs(b, n_seq, n_res, h, d, dtype, *, seed=0):
    """Zero q, so every logit is equal and softmax is exactly uniform.

    ``n_res`` is a power of two, making each probability exactly ``1/n_res``,
    and the values are small integers so the sums stay inside a bf16 mantissa.
    """
    g = generator("cuda", seed)
    return {
        "q": torch.zeros(b, n_seq, n_res, h, d, device="cuda", dtype=dtype),
        "k": torch.zeros(b, n_seq, n_res, h, d, device="cuda", dtype=dtype),
        "v": torch.randint(
            -4, 5, (b, n_seq, n_res, h, d), device="cuda", dtype=dtype, generator=g
        ),
        "bias": torch.zeros(b, 1, h, n_res, n_res, device="cuda", dtype=dtype),
    }


@requires_gpu
@IMPLEMENTATION_PARAM
@DTYPES_PARAM
@pytest.mark.parametrize("kept_key", [0, 7, 31], ids=["first", "interior", "last"])
def test_one_unmasked_key_returns_that_key(request, impl, dtype, kept_key):
    """One unmasked key saturates the softmax: its probability is exactly 1 and
    every other exactly 0, so the output is exactly ``v`` at that key."""
    # -1e9 is not representable in fp16, so the sentinel becomes -inf and the
    # online softmax evaluates -inf - -inf on a key tile that is masked in its
    # entirety. Only reachable when such a tile precedes every live key, so the
    # usual trailing-padding layout is unaffected.
    if impl is triangle_attention and dtype is torch.float16 and kept_key == 31:
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason="inf sentinel unrepresentable in fp16 causing NaNs",
            )
        )
    B, N_seq, N_res, H, D = 1, 2, 32, 4, 32
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", dtype, seed=3)
    mask = torch.zeros(B, N_seq, 1, 1, N_res, device="cuda", dtype=torch.bool)
    mask[..., kept_key] = True

    with torch.no_grad():
        out = impl(**tensors, mask=mask, scale=D**-0.5)

    expected = tensors["v"][:, :, kept_key : kept_key + 1, :, :].expand_as(out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@requires_gpu
@IMPLEMENTATION_PARAM
@DTYPES_PARAM
def test_uniform_logits(impl, dtype):
    """Uniform probabilities over an exactly representable mean.

    The reference is exact in every dtype. The kernel is exact in fp32 and lands
    within one fp32 epsilon of the mean in bf16 and fp16.
    """
    # TODO(gmnngeoffrey): figure out why there's a residual error in bf16/fp16
    # even with exactly representable (small int) inputs.
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 16
    tensors = uniform_attention_inputs(B, N_seq, N_res, H, D, dtype, seed=5)

    with torch.no_grad():
        out = impl(**tensors, mask=None, scale=D**-0.5)

    expected = tensors["v"].double().mean(dim=2, keepdim=True).expand_as(out)
    accumulator_ulp = torch.finfo(torch.float32).eps * expected.abs().max().item()
    torch.testing.assert_close(out.double(), expected, rtol=0, atol=accumulator_ulp)


@requires_gpu
@IMPLEMENTATION_PARAM
def test_mask_none(impl):
    """``mask=None`` matches an all-keep mask."""
    B, N_seq, N_res, H, D = 1, 2, 32, 4, 32
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=13)
    all_keep = torch.ones(B, N_seq, 1, 1, N_res, device="cuda", dtype=torch.bool)

    with torch.no_grad():
        out = impl(**tensors, mask=None, scale=D**-0.5)
        expected = impl(**tensors, mask=all_keep, scale=D**-0.5)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@requires_gpu
@IMPLEMENTATION_PARAM
def test_no_bias(impl):
    """Omitting the bias matches an explicit zero bias."""
    B, N_seq, N_res, H, D = 1, 2, 32, 4, 32
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=15)
    without = {k: v for k, v in tensors.items() if k != "bias"}
    zeroed = {**tensors, "bias": torch.zeros_like(tensors["bias"])}

    with torch.no_grad():
        out = impl(**without, mask=None, scale=D**-0.5)
        expected = impl(**zeroed, mask=None, scale=D**-0.5)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@requires_gpu
@IMPLEMENTATION_PARAM
def test_additive_mask(impl):
    """A boolean mask matches the additive mask it is converted to."""
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 32
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=11)
    boolean = make_key_mask(B, N_seq, N_res, "cuda", seed=4)
    additive = (~boolean).to(torch.float32) * NEG_INF

    with torch.no_grad():
        from_bool = impl(**tensors, mask=boolean, scale=D**-0.5)
        from_additive = impl(**tensors, mask=additive, scale=D**-0.5)

    torch.testing.assert_close(from_bool, from_additive, rtol=0, atol=0)


@requires_gpu
@IMPLEMENTATION_PARAM
def test_default_scale(impl):
    """``scale=None`` matches an explicit ``D ** -0.5``, and the argument has an
    effect at all."""
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 32
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", torch.float32, seed=13)

    with torch.no_grad():
        default = impl(**tensors, mask=None, scale=None)
        explicit = impl(**tensors, mask=None, scale=D**-0.5)
        other = impl(**tensors, mask=None, scale=D**-0.5 * 1.75)

    torch.testing.assert_close(default, explicit, rtol=0, atol=0)
    assert not torch.equal(default, other), "the scale argument had no effect"


@requires_gpu
def test_4d_input():
    """Ranks below 5 are padded onto the canonical layout and come back in the
    input shape. The reference takes 5D only, so this is Triton only."""
    N_seq, N_res, H, D = 2, 32, 4, 32
    unbatched = make_inputs((), N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=7)
    batched = make_inputs(1, N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=7)
    mask = make_key_mask(1, N_seq, N_res, "cuda", seed=9)

    with torch.no_grad():
        out_4d = triangle_attention(**unbatched, mask=mask, scale=D**-0.5)
        out_5d = triangle_attention(**batched, mask=mask, scale=D**-0.5)

    assert out_4d.shape == unbatched["q"].shape
    torch.testing.assert_close(out_4d, out_5d[0], rtol=0, atol=0)


@requires_gpu
def test_6d_input():
    """Ranks above 5 fold their leading dims into the batch axis."""
    B0, B1, N_seq, N_res, H, D = 2, 3, 1, 32, 4, 32
    nested = make_inputs((B0, B1), N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=11)
    flat = make_inputs(B0 * B1, N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=11)

    with torch.no_grad():
        out_6d = triangle_attention(**nested, mask=None, scale=D**-0.5)
        out_5d = triangle_attention(**flat, mask=None, scale=D**-0.5)

    assert out_6d.shape == nested["q"].shape
    torch.testing.assert_close(out_6d.reshape_as(out_5d), out_5d, rtol=0, atol=0)


@requires_gpu
def test_dynamic_entry_point():
    """The variable-shape entry trades a per-call cost for no per-size
    recompile. Its output is identical, not merely close."""
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 32
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", torch.bfloat16, seed=19)
    mask = make_key_mask(B, N_seq, N_res, "cuda", seed=21)

    with torch.no_grad():
        specialized = triangle_attention(
            **tensors, mask=mask, scale=D**-0.5, dynamic=False
        )
        dynamic = triangle_attention(**tensors, mask=mask, scale=D**-0.5, dynamic=True)

    torch.testing.assert_close(specialized, dynamic, rtol=0, atol=0)


# Reference tests: these compare the Triton kernel to the pure Torch reference
# in fp64.


@requires_gpu
@pytest.mark.parametrize(
    "inf_sentinel",
    [9e2, 1e3, 1e6, 1e9],
)
def test_fully_masked_row_gradient(request, inf_sentinel):
    """A fully-masked row across a range of magnitudes for the mask "inf" magnitude.

    The backward rebuilds probabilities as ``exp(QK - M)`` from the stored
    logsumexp. For a fully masked row, once the "inf" magnitude is large enough,
    the ``m + log(l)`` in the forward loses the ``log(l)`` term to rounding,
    leaving the reconstruction short by a factor of ``l``. The kernel currently
    works around this by dividing by ``SEQ_LEN`` to compensate. But whether or
    not the term was lost depends on the magnitude of "inf", the accumulator
    dtype (fortunately fixed in our kernel), and the sequence length (weakly
    since it scales with log).

    We compare against the reference at fp32 rather than the usual fp64 oracle.
    Folding the inf sentinel into the logits is the one place in this op where a
    huge constant is added to O(1) values. At inf magnitude 1e9 fp32 ulp is 64,
    far wider than the spread of the logits, so the whole row collapses to a
    single value and softmax is exactly uniform. In fp64 the logits remain
    intact (basically the mask is *removed* due to softmax scale invariance).
    This is a separate weirdness with all-masked rows that doesn't apply to the
    Triton path. Using fp32 for the reference precision avoids conflating these
    two issues.
    """
    if 1e3 <= inf_sentinel < 1e8:
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason="backward divides by SEQ_LEN having corrected nothing",
            )
        )
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 32
    tensors = make_inputs(
        B, N_seq, N_res, H, D, "cuda", torch.float32, seed=3, requires_grad=True
    )
    mask = torch.full((B, N_seq, 1, 1, N_res), -inf_sentinel, device="cuda")
    grad_out = make_grad_output(
        *tensors["q"].shape, device="cuda", dtype=torch.float32, seed=31
    )

    out, grads = fwd_bwd(triangle_attention, tensors, mask, D**-0.5, grad_out)
    out_ref, grads_ref = fwd_bwd(
        triangle_attention_torch, tensors, mask, D**-0.5, grad_out
    )

    assert_matches_oracle(
        comparisons(out, grads, out_ref, grads_ref),
        TRIANGLE_ATTENTION_TOL[torch.float32],
        label=f"fully_masked/{inf_sentinel:.0e}",
    )


@requires_gpu
@pytest.mark.parametrize("shape", SHAPES)
@DTYPES_PARAM
def test_triton_accuracy(shape, dtype):
    """The kernel at ``dtype`` against the reference in fp64."""
    B, N_seq, N_res, H, D = shape
    tensors = make_inputs(
        B, N_seq, N_res, H, D, "cuda", dtype, seed=3, requires_grad=True
    )
    mask = make_key_mask(B, N_seq, N_res, "cuda", seed=1)
    assert_agrees_with_the_oracle(tensors, mask, D**-0.5, label=f"{shape}/{dtype}")


@requires_gpu
@pytest.mark.parametrize(
    "case", ["no_bias", "no_mask", "four_dimensional_bias", "custom_scale"]
)
def test_optional_arguments(case):
    """One case per branch through the adapter's optional arguments."""
    B, N_seq, N_res, H, D = 1, 2, 32, 4, 32
    dtype = torch.bfloat16
    tensors = make_inputs(
        B,
        N_seq,
        N_res,
        H,
        D,
        "cuda",
        dtype,
        seed=5,
        with_bias=case != "no_bias",
        requires_grad=True,
    )
    if case == "four_dimensional_bias":
        tensors["bias"] = tensors["bias"][:, 0].detach().requires_grad_(True)
    mask = None if case == "no_mask" else make_key_mask(B, N_seq, N_res, "cuda")
    scale = D**-0.5 * (1.75 if case == "custom_scale" else 1.0)

    assert_agrees_with_the_oracle(tensors, mask, scale, label=case)


@requires_gpu
def test_pairformer_shape():
    """N=384, H=16, D=48 -- the pairformer operating point."""
    B, N_seq, N_res, H, D = 1, 1, 384, 16, 48
    dtype = torch.bfloat16
    tensors = make_inputs(B, N_seq, N_res, H, D, "cuda", dtype, seed=21)

    with torch.no_grad():
        out = triangle_attention(**tensors, scale=None)
        out_oracle = triangle_attention_torch(
            **cast_tree(tensors, torch.float64, requires_grad=False), scale=None
        )

    assert_matches_oracle(
        [("out", out, out_oracle)],
        Tolerance(1.0 - 1e-4, 5e-2),
        label="pairformer_384",
    )
