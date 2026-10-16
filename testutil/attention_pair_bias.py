# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Input builders and oracle comparison for attention_pair_bias."""

import torch

from geometric_nn_ops_torch.triangle._reference.attention_pair_bias import (
    attention_pair_bias_torch,
)

from . import cast_tree, generator, grad_targets, seeded_randn
from .metrics import ATTENTION_PAIR_BIAS_TOL, assert_matches_oracle


def make_op_tensors(
    b: int,
    m: int,
    h: int,
    u: int,
    v: int,
    dh: int,
    zc: int,
    device,
    dtype,
    *,
    seed: int = 0,
    requires_grad: bool = False,
) -> dict:
    """Op-boundary tensors: pre-projected q/k/v, single ``s``, pair ``z``, weights.

    One flat dict keyed by the op's argument names. ``m`` is the diffusion
    multiplicity, across which the pair rep ``z`` is shared.
    """
    d = h * dh
    g = generator(device, seed)

    def rn(*shape, scale=1.0):
        t = torch.randn(*shape, device=device, dtype=dtype, generator=g) * scale
        return t.requires_grad_(requires_grad)

    def const(fill, n):
        t = torch.full((n,), fill, device=device, dtype=dtype)
        return t.requires_grad_(requires_grad)

    return {
        "s": rn(b * m, u, d),
        "q": rn(b * m, h, u, dh),
        "k": rn(b * m, h, v, dh),
        "v": rn(b * m, h, v, dh),
        "z": rn(b, u, v, zc),
        "proj_z_weight": rn(h, zc, scale=zc**-0.5),
        "ln_z_weight": const(1.0, zc),
        "proj_g_weight": rn(d, d, scale=d**-0.5),
        "proj_o_weight": rn(d, d, scale=d**-0.5),
    }


def make_projection_biases(
    h: int, dh: int, device, dtype, *, seed: int = 0, requires_grad: bool = False
) -> dict:
    """Build optional gate and output projection biases."""
    g = generator(device, seed)

    def rn(*shape):
        t = torch.randn(*shape, device=device, dtype=dtype, generator=g) * 0.1
        return t.requires_grad_(requires_grad)

    return {"proj_g_bias": rn(h * dh), "proj_o_bias": rn(h * dh)}


def make_key_mask(
    batch: int, v: int, device, dtype=torch.float32, *, seed: int = 0, frac_keep=0.9
) -> torch.Tensor:
    """Key mask ``[batch, V]``, ``1`` = keep.

    Masked keys are scattered. Key 0 is always kept, since a fully masked row
    leaves softmax undefined.
    """
    g = generator(device, seed)
    mask = torch.rand(batch, v, device=device, generator=g) < frac_keep
    mask[:, 0] = True
    return mask.to(dtype)


def fwd_bwd(implementation, tensors, grad_out, **options):
    out, _ = implementation(**tensors, **options)
    targets = grad_targets(tensors)
    grads = torch.autograd.grad(out, list(targets.values()), grad_outputs=grad_out)
    return out, list(targets), grads


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
