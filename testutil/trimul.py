# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Input builders for triangle_multiplicative_update."""

from collections.abc import Sequence

import torch

from geometric_nn_ops_torch.triangle._reference.trimul import trimul_torch

from . import generator, grad_targets, oracle_tree, seeded_randn, to_oracle
from .metrics import TRIMUL_TOL, assert_matches_oracle


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


class TrimulLayer(torch.nn.Module):
    """One triangle multiplicative update, holding its own parameters.

    The implementation is passed in rather than dispatched to, so a caller
    measuring this module knows which one it is measuring. The public entry
    point is a valid choice when the dispatch decision is what you want
    included.
    """

    def __init__(self, impl, c_z, c_hidden, direction, dtype, *, seed=0, biases=False):
        super().__init__()
        self.impl = impl
        self.direction = direction
        parameters = make_weights(c_z, c_hidden, "cuda", dtype, seed=seed)
        if biases:
            parameters |= make_projection_biases(
                c_z, c_hidden, "cuda", dtype, seed=seed + 1
            )
        self.parameter_names = tuple(parameters)
        for name, tensor in parameters.items():
            self.register_parameter(name, torch.nn.Parameter(tensor))

    def forward(self, x, mask):
        weights = {name: getattr(self, name) for name in self.parameter_names}
        return self.impl(x, self.direction, mask, **weights)


class TrimulStack(torch.nn.Module):
    """Residual stack of :class:`TrimulLayer`, as a pairformer trunk applies it.

    ``direction`` may be ``"alternate"``, which is how the trunk actually runs:
    every pair block applies the outgoing update then the incoming one, so the
    flattened sequence alternates.
    """

    def __init__(self, impl, depth, c_z, c_hidden, direction, dtype, *, biases=False):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            TrimulLayer(
                impl,
                c_z,
                c_hidden,
                self._direction_for(direction, index),
                dtype,
                seed=2 * index,
                biases=biases,
            )
            for index in range(depth)
        )

    @staticmethod
    def _direction_for(direction, index):
        if direction != "alternate":
            return direction
        return "outgoing" if index % 2 == 0 else "incoming"

    def forward(self, x, mask):
        for layer in self.layers:
            x = x + layer(x, mask)
        return x


def assert_forward_agrees_with_the_oracle(
    x, mask, weights, direction, *, impl, precision="ieee", tol=None, label=""
):
    """Run the ``impl`` on the inputs as given and the reference in fp64,
    compare the forward output.

    ``precision`` only selects the tolerance; the caller sets it on torch. The
    fp64 oracle is unaffected by it.
    """
    dtype = x.dtype
    with torch.no_grad():
        out = impl(x, direction, mask, **weights)
        out_oracle = trimul_torch(
            to_oracle(x),
            direction,
            to_oracle(mask),
            **oracle_tree(weights, requires_grad=False),
        )

    assert_matches_oracle(
        [("out", out, out_oracle)],
        tol or TRIMUL_TOL[dtype, precision],
        label=label,
    )


def fwd_bwd(impl, x, mask, weights, direction, grad_out):
    """Forward plus gradients wrt ``x`` and every weight carrying one."""
    out = impl(x, direction, mask, **weights)
    targets = grad_targets(weights)
    grads = torch.autograd.grad(out, [x, *targets.values()], grad_outputs=grad_out)
    return out, ["x", *targets], grads


def comparisons(out, keys, grads, out_oracle, grads_oracle):
    """Name every (candidate, oracle) pair the accuracy assertion should cover."""
    names = ["out", *(f"grad_{key}" for key in keys)]
    return list(zip(names, [out, *grads], [out_oracle, *grads_oracle], strict=True))


def assert_agrees_with_the_oracle(
    x, mask, weights, direction, *, impl, precision="ieee", label=""
):
    """Run the ``impl`` on the inputs as given and the reference in fp64,
    compare the output and every gradient.

    ``precision`` only selects the tolerance; the caller sets it on torch. The
    fp64 oracle is unaffected by it.
    """
    grad_out = seeded_randn(*x.shape, device=x.device, dtype=x.dtype, seed=11)
    out, keys, grads = fwd_bwd(impl, x, mask, weights, direction, grad_out)
    out_oracle, _, grads_oracle = fwd_bwd(
        trimul_torch,
        to_oracle(x.detach()).requires_grad_(),
        to_oracle(mask),
        oracle_tree(weights),
        direction,
        to_oracle(grad_out),
    )

    assert_matches_oracle(
        comparisons(out, keys, grads, out_oracle, grads_oracle),
        TRIMUL_TOL[x.dtype, precision],
        label=label,
    )
