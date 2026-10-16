# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Input builders and oracle comparison for triangle_attention."""

from collections.abc import Sequence

import torch

from geometric_nn_ops_torch.triangle._reference.triangle_attention import (
    triangle_attention_torch,
)

from . import cast_tree, generator, grad_targets, seeded_randn
from .metrics import TRIANGLE_ATTENTION_TOL, assert_matches_oracle


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


def fwd_bwd(run, tensors, mask, scale, grad_out):
    """Forward plus gradients wrt every input carrying one."""
    out = run(**tensors, mask=mask, scale=scale)
    targets = grad_targets(tensors)
    grads = torch.autograd.grad(out, list(targets.values()), grad_outputs=grad_out)
    return out, list(targets), grads


def comparisons(out, keys, grads, out_oracle, grads_oracle):
    """Name every (candidate, oracle) pair the accuracy assertion should cover."""
    names = ["out", *(f"grad_{key}" for key in keys)]
    return list(zip(names, [out, *grads], [out_oracle, *grads_oracle], strict=True))


def assert_agrees_with_the_oracle(
    tensors, mask, scale, *, impl, precision="ieee", tol=None, label=""
):
    """Run the ``impl`` on the inputs as given and the reference in fp64,
    compare the output and every input gradient.

    ``precision`` only selects the tolerance; the caller sets it on torch. The
    fp64 oracle is unaffected by it.
    """
    dtype = tensors["q"].dtype
    grad_out = seeded_randn(
        *tensors["q"].shape, device=tensors["q"].device, dtype=dtype, seed=31
    )

    out, keys, grads = fwd_bwd(impl, tensors, mask, scale, grad_out)
    out_oracle, _, grads_oracle = fwd_bwd(
        triangle_attention_torch,
        cast_tree(tensors, torch.float64),
        mask,
        scale,
        grad_out.double(),
    )

    assert_matches_oracle(
        comparisons(out, keys, grads, out_oracle, grads_oracle),
        tol or TRIANGLE_ATTENTION_TOL[dtype, precision],
        label=label,
    )
