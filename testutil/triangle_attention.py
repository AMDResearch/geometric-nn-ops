# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Input builders for triangle_attention."""

from collections.abc import Sequence

import torch

from . import generator


def make_inputs(
    batch: int | Sequence[int],
    n_seq: int,
    n_res: int,
    h: int,
    d: int,
    device,
    dtype,
    *,
    seed: int = 0,
    with_bias: bool = True,
    requires_grad: bool = False,
) -> dict:
    """q/k/v ``[*batch, N_seq, N_res, H, D]`` and a bias ``[*batch, 1, H, N, N]``.

    ``batch`` is a single leading dim or a tuple of them; ``()`` gives an
    unbatched input. A given seed draws the same values at any batch shape of
    the same size, so two ranks can be compared elementwise.
    """
    leading = (batch,) if isinstance(batch, int) else tuple(batch)
    g = generator(device, seed)

    def rn(*shape, scale=1.0):
        t = torch.randn(*shape, device=device, dtype=dtype, generator=g) * scale
        return t.requires_grad_(requires_grad)

    tensors = {
        "q": rn(*leading, n_seq, n_res, h, d),
        "k": rn(*leading, n_seq, n_res, h, d),
        "v": rn(*leading, n_seq, n_res, h, d),
    }
    if with_bias:
        tensors["bias"] = rn(*leading, 1, h, n_res, n_res, scale=0.1)
    return tensors


def make_key_mask(
    b: int, n_seq: int, n_res: int, device, *, seed: int = 0, frac_keep: float = 0.75
) -> torch.Tensor:
    """Random boolean key mask ``[B, N_seq, 1, 1, N_res]``, ``True`` = keep.

    Key 0 is always kept, since a fully masked row leaves softmax undefined.
    """
    g = generator(device, seed)
    mask = torch.rand(b, n_seq, 1, 1, n_res, device=device, generator=g) < frac_keep
    mask[..., 0] = True
    return mask
