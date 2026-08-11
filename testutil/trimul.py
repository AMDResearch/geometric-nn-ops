# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Input builders for triangle_multiplicative_update."""

from collections.abc import Sequence

import torch

from . import generator

# Bias-free weight names, in the order tests compare gradients.
WEIGHT_KEYS = (
    "norm_in_weight",
    "norm_in_bias",
    "p_in_weight",
    "g_in_weight",
    "norm_out_weight",
    "norm_out_bias",
    "p_out_weight",
    "g_out_weight",
)

# Projection biases.
BIAS_KEYS = ("p_in_bias", "g_in_bias", "p_out_bias", "g_out_bias")


def make_weights(
    c_z: int,
    c_hidden: int,
    device,
    dtype,
    *,
    seed: int = 0,
    requires_grad: bool = False,
) -> dict:
    """TMU LayerNorm + projection random parameters, bias-free.

    This is the standard config: LayerNorms carry weight and bias, projections
    do not. ``c_z == c_hidden`` is the square case; the projections are sized
    for the concatenated ``[a; b]`` layout. Use :func:`make_projection_biases`
    to get projection biases.
    """
    g = generator(device, seed)

    def rn(*shape, scale=1.0):
        t = torch.randn(*shape, device=device, dtype=dtype, generator=g) * scale
        return t.requires_grad_(requires_grad)

    def ones(n):
        return torch.ones(n, device=device, dtype=dtype).requires_grad_(requires_grad)

    def zeros(n):
        return torch.zeros(n, device=device, dtype=dtype).requires_grad_(requires_grad)

    return {
        "norm_in_weight": ones(c_z),
        "norm_in_bias": zeros(c_z),
        "p_in_weight": rn(2 * c_hidden, c_z, scale=0.05),
        "g_in_weight": rn(2 * c_hidden, c_z, scale=0.05),
        "norm_out_weight": ones(c_hidden),
        "norm_out_bias": zeros(c_hidden),
        "p_out_weight": rn(c_z, c_hidden, scale=0.05),
        "g_out_weight": rn(c_z, c_z, scale=0.05),
    }


def make_projection_biases(
    c_z: int,
    c_hidden: int,
    device,
    dtype,
    *,
    seed: int = 0,
    requires_grad: bool = False,
) -> dict:
    """Four random projection biases."""
    g = generator(device, seed)

    def rn(*shape):
        t = torch.randn(*shape, device=device, dtype=dtype, generator=g) * 0.05
        return t.requires_grad_(requires_grad)

    return {
        "p_in_bias": rn(2 * c_hidden),
        "g_in_bias": rn(2 * c_hidden),
        "p_out_bias": rn(c_z),
        "g_out_bias": rn(c_z),
    }


def make_pair_input(
    batch: int | Sequence[int],
    n: int,
    c_z: int,
    device,
    dtype,
    *,
    seed: int = 0,
    requires_grad: bool = False,
    frac_keep: float = 0.9,
):
    """Pair representation ``[*batch, N, N, C_z]`` and a mask ``[*batch, N, N]``.

    ``batch`` is a single leading dim or a tuple of them; ``()`` gives an
    unbatched input.
    """
    leading = (batch,) if isinstance(batch, int) else batch
    g = generator(device, seed)
    x = torch.randn(*leading, n, n, c_z, device=device, dtype=dtype, generator=g) * 0.5
    mask = (torch.rand(*leading, n, n, device=device, generator=g) < frac_keep).to(
        dtype
    )
    return x.requires_grad_(requires_grad), mask
