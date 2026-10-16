# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Input builders for attention_pair_bias."""

import torch

from . import generator

# Grad-bearing activations, in the order tests compare gradients.
ACTIVATION_KEYS = ("s", "q", "k", "v", "z")

# Standard-config parameters (LN affine present, projections bias-free).
WEIGHT_KEYS = (
    "proj_z_weight",
    "ln_z_weight",
    "proj_g_weight",
    "proj_o_weight",
)


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

    One flat dict keyed by the op's argument names; ``ACTIVATION_KEYS`` and
    ``WEIGHT_KEYS`` split it into activations and parameters. ``m`` is the
    diffusion multiplicity, across which the pair rep ``z`` is shared.
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
