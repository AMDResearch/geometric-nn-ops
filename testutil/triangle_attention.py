# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Input builders and oracle comparison for triangle_attention."""

from collections.abc import Sequence

import torch
from torch.utils.checkpoint import checkpoint

from geometric_nn_ops_torch.triangle._reference.triangle_attention import (
    triangle_attention_torch,
)

from . import cast_tree, generator, grad_targets, seeded_randn
from .metrics import TRIANGLE_ATTENTION_TOL, assert_matches_oracle

# Inputs with an N_seq axis. `Oracle` can check a subset of their rows.
ROW_KEYS = ("q", "k", "v")


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


def checked_rows(n_seq: int, count: int, *, seed: int = 0) -> list[int] | None:
    """Choose ``count`` rows of N_seq for :class:`Oracle` to check: the first,
    the last and random ones. Returns ``None``, meaning all rows, if ``count``
    is at least the number of rows.
    """
    if count < 2:
        raise ValueError(f"count must be at least 2 (first and last), got {count}")
    if count >= n_seq:
        return None
    g = torch.Generator().manual_seed(seed)
    rows = {0, n_seq - 1}
    for row in torch.randperm(n_seq - 2, generator=g)[: max(0, count - 2)]:
        rows.add(int(row) + 1)
    return sorted(rows)


class Oracle:
    """Computes fp64 reference results for one input and checks implementations
    against them.

    With ``rows``, only those rows of N_seq are checked, which makes the oracle
    affordable at long sequences. The implementation under test still runs on
    the whole input. The output gradient is zero on the other rows, so their
    contribution to the gradients is exactly zero. Their q/k/v gradients are
    checked to be exactly zero. The bias gradient, a sum over all rows, becomes
    a sum over the checked rows, which the oracle can compute from those rows
    alone. ``backward=False`` computes only the forward, for
    :meth:`check_forward`.

    ``oracle`` is the reference to run in fp64. :func:`chunked_reference`
    handles inputs too large to hold all the scores at once.
    """

    def __init__(
        self,
        tensors,
        mask,
        scale,
        *,
        rows=None,
        backward=True,
        oracle=triangle_attention_torch,
    ):
        q = tensors["q"]
        self.tensors = tensors
        self.mask = mask
        self.scale = scale
        self.rows = None if rows is None else torch.tensor(rows, device=q.device)
        checked = cast_tree(
            {k: self.select(t) if k in ROW_KEYS else t for k, t in tensors.items()},
            torch.float64,
            requires_grad=False,
        )
        if self.rows is not None and mask is not None and mask.shape[1] > 1:
            mask = self.select(mask)
        self.grad_out = None
        if not backward:
            with torch.no_grad():
                self.out = oracle(**checked, mask=mask, scale=scale)
            return
        for key in grad_targets(tensors):
            checked[key].requires_grad_()
        self.grad_out = seeded_randn(*q.shape, device=q.device, dtype=q.dtype, seed=31)
        if self.rows is not None:
            keep = torch.zeros(q.shape[1], device=q.device, dtype=q.dtype)
            keep[self.rows] = 1
            self.grad_out.mul_(keep.view(1, -1, 1, 1, 1))
        out, _, self.grads = fwd_bwd(
            oracle, checked, mask, scale, self.select(self.grad_out).double()
        )
        self.out = out.detach()

    def select(self, t):
        """The checked rows of a ``[B, N_seq, ...]`` tensor."""
        return t if self.rows is None else t.index_select(1, self.rows)

    def check(self, impl, tol, *, label=""):
        """Check ``impl``'s output and input gradients."""
        out, keys, grads = fwd_bwd(
            impl, self.tensors, self.mask, self.scale, self.grad_out
        )
        if self.rows is not None:
            self.assert_unchecked_rows_have_no_gradient(keys, grads, label)
        grads = [
            self.select(g) if k in ROW_KEYS else g
            for k, g in zip(keys, grads, strict=True)
        ]
        assert_matches_oracle(
            comparisons(self.select(out), keys, grads, self.out, self.grads),
            tol,
            label=label,
        )

    def assert_unchecked_rows_have_no_gradient(self, keys, grads, label):
        for key, grad in zip(keys, grads, strict=True):
            if key not in ROW_KEYS:
                continue
            # Subtract the checked rows' count rather than indexing the
            # unchecked rows, which would copy nearly the whole gradient.
            leaked = (grad.count_nonzero() - self.select(grad).count_nonzero()).item()
            assert leaked == 0, (
                f"[{label}] grad_{key}: {leaked} nonzero entries on rows with "
                f"a zero output gradient"
            )

    def check_forward(self, impl, tensors, tol, *, label=""):
        """Check ``impl``'s output under ``torch.inference_mode`` on
        ``tensors``. They must have the values the reference results were
        computed from, but need not require gradients."""
        with torch.inference_mode():
            out = impl(**tensors, mask=self.mask, scale=self.scale)
        assert_matches_oracle([("out", self.select(out), self.out)], tol, label=label)


def assert_agrees_with_oracle(
    tensors, mask, scale, *, impl, precision="ieee", tol=None, label=""
):
    """Run the ``impl`` on the inputs as given and the reference in fp64,
    compare the output and every input gradient.

    ``precision`` only selects the tolerance; the caller sets it on torch. The
    fp64 oracle is unaffected by it.
    """
    tol = tol or TRIANGLE_ATTENTION_TOL[tensors["q"].dtype, precision]
    Oracle(tensors, mask, scale).check(impl, tol, label=label)


def chunked_reference(
    q, k, v, bias=None, mask=None, scale=None, *, max_slice_bytes=256 * 1024**2
):
    """The reference over slices of N_seq, each recomputed in the backward.

    Rows of N_seq share only the bias, so the output is the same as a single
    call and autograd sums the bias gradient across the slices. Checkpointing
    keeps one slice's scores alive at a time rather than all ``N_seq * N_res^2``
    of them, making an fp64 oracle affordable at benchmark sizes. A slice's
    scores take at most ``max_slice_bytes``, or one row of N_seq if that is
    larger. Takes the canonical 5D layout only.
    """
    batch, n_seq, n_res, heads, _ = q.shape
    row_bytes = batch * heads * n_res * n_res * q.element_size()
    rows = max(1, max_slice_bytes // row_bytes)
    outs = []
    for start in range(0, n_seq, rows):
        rows_slice = slice(start, start + rows)
        mask_slice = mask if mask is None or mask.shape[1] == 1 else mask[:, rows_slice]
        outs.append(
            checkpoint(
                triangle_attention_torch,
                q[:, rows_slice],
                k[:, rows_slice],
                v[:, rows_slice],
                bias,
                mask_slice,
                scale,
                use_reentrant=False,
            )
        )
    return torch.cat(outs, dim=1)
