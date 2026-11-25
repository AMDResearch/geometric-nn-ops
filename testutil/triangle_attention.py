# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Input builders and oracle comparison for triangle_attention."""

import copy
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from geometric_nn_ops_torch.triangle._reference.triangle_attention import (
    triangle_attention_torch,
)

from . import cast_tree, generator, grad_targets, seeded_randn
from .metrics import (
    TRIANGLE_ATTENTION_TOL,
    assert_matches_oracle,
    assert_module_agrees_with_oracle,
)

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


# There are at least three bugs in the Torch layer norm kernel for very large
# inputs:
#
# - A: https://github.com/pytorch/pytorch/issues/181555: when numel > 2^32 and
#   the channel dimension is a multiple of 4 (so the vectorized kernel is used)
#   there is a uint32 overflow in the row offset. Fixed in
#   https://github.com/pytorch/pytorch/pull/181600, first released in torch
#   2.13.0.
# - B: https://github.com/pytorch/pytorch/issues/184826: when rows is >= 2^23
#   and the channel dimension is not a multiple of 4 (so the non-vectorized
#   kernel is used), some outputs are never addressed and are filled with
#   garbage values. Fixed in https://github.com/pytorch/pytorch/pull/186956,
#   first released in Torch 2.14.0.
# - C: https://github.com/pytorch/pytorch/issues/199037: when rows is >
#   (2^32 - 1) / warp_size (so in practice ~2^26 or ~2^27) and the channel
#   dimension is a multiple of 4 (so the vectorized kernel is used), there is an
#   integer overflow and some rows get garbage values.
#
# `layer_norm` chunks over rows to keep every kernel call below all cliffs:
# chunk_size * C < 2^32 AND chunk_size < 2^23 (AND chunk_size < 2^26).
LAYER_NORM_MAX_ELEMENTS = 2**32 - 1
LAYER_NORM_MAX_ROWS = 2**23 - 1


def layer_norm(x, weight, bias):
    """``F.layer_norm`` over the last dim, in calls small enough to avoid the
    torch bugs above."""
    channels = x.shape[-1]
    n_rows = x.numel() // channels
    chunk = min(LAYER_NORM_MAX_ROWS, LAYER_NORM_MAX_ELEMENTS // channels)
    if n_rows <= chunk:
        return F.layer_norm(x, (channels,), weight, bias)
    rows = x.reshape(n_rows, channels)
    out = torch.empty_like(rows)
    for start in range(0, n_rows, chunk):
        out[start : start + chunk] = F.layer_norm(
            rows[start : start + chunk], (channels,), weight, bias
        )
    return out.view(x.shape)


class TriangleAttentionLayer(torch.nn.Module):
    """Gated triangle attention over the pair representation, as in AF3.

    Modeled after OF3, but set up for testing purposes:
     - the specific triangle attention implementation function is passed in as
       an init argument.
     - weights are randn initialized instead of zero, so that we can test real
       non-degenerate inputs without a full trained checkpoint.

    All projections are bias-free. The layer attends along rows of the pair rep.
    As in OF3, the ending node is this layer applied to the transposed pair rep
    with ``transpose_bias`` (see :class:`TriangleAttentionStack`).
    """

    def __init__(
        self, attention, c_z, heads, head_dim, dtype, *, transpose_bias, seed=0
    ):
        super().__init__()
        self.attention = attention
        self.heads = heads
        self.head_dim = head_dim
        self.transpose_bias = transpose_bias
        g = generator("cuda", seed)
        hidden = heads * head_dim

        def randn(*shape):
            return torch.randn(*shape, device="cuda", dtype=dtype, generator=g)

        def projection(out_features, in_features):
            # Fan-in scaling: unit-variance outputs for unit-variance inputs.
            w = randn(out_features, in_features) * in_features**-0.5
            return torch.nn.Parameter(w)

        # Close to the default ones and zeros, but with noise, so checks aren't
        # degenerate.
        self.norm_weight = torch.nn.Parameter(1 + 0.1 * randn(c_z))
        self.norm_bias = torch.nn.Parameter(0.1 * randn(c_z))
        self.bias_weight = projection(heads, c_z)
        self.q_weight = projection(hidden, c_z)
        self.k_weight = projection(hidden, c_z)
        self.v_weight = projection(hidden, c_z)
        self.gate_weight = projection(hidden, c_z)
        self.out_weight = projection(c_z, hidden)

    def norm(self, z):
        """LayerNorm of ``z``, converted to this layer's dtype first."""
        return layer_norm(
            z.to(self.norm_weight.dtype), self.norm_weight, self.norm_bias
        )

    def pair_bias(self, x):
        """The per-head bias ``[B, H, I, J]`` from the normalized pair rep."""
        return F.linear(x, self.bias_weight).permute(0, 3, 1, 2)

    def attention_bias(self, bias):
        """Lay out ``bias`` for the attention as ``[B, 1, H, I, J]``. With
        ``transpose_bias``, its last two dims are transposed."""
        if self.transpose_bias:
            bias = bias.transpose(-1, -2)
        return bias.unsqueeze(1)

    def attend(self, x, bias, key_mask):
        """The update for the rows of the normalized pair rep ``x``."""

        def heads(weight):
            return F.linear(x, weight).unflatten(-1, (self.heads, self.head_dim))

        q, k, v = heads(self.q_weight), heads(self.k_weight), heads(self.v_weight)
        o = self.attention(q, k, v, bias=bias, mask=key_mask, scale=self.head_dim**-0.5)
        gated = o * torch.sigmoid(heads(self.gate_weight))
        return F.linear(gated.flatten(-2), self.out_weight)

    def forward(self, z, mask, chunk_size=None):
        """The update for ``z`` ``[B, I, J, C_z]``, with ``mask`` boolean
        ``[B, I, J]`` or ``None``.

        ``chunk_size`` runs the projections and the attention that many rows of
        ``I`` at a time, as openfold-3 does in inference. The bias is computed
        once from the whole pair rep.
        """
        x = self.norm(z)
        bias = self.attention_bias(self.pair_bias(x))
        key_mask = None if mask is None else mask[:, :, None, None, :]
        if chunk_size is None:
            return self.attend(x, bias, key_mask)
        o = torch.empty_like(x)
        for start in range(0, z.shape[1], chunk_size):
            rows = slice(start, start + chunk_size)
            chunk_mask = None if key_mask is None else key_mask[:, rows]
            o[:, rows] = self.attend(x[:, rows], bias, chunk_mask)
        return o

    def update_rows(self, z, mask, rows, *, pair_chunk_size=2**22):
        """The update :meth:`forward` gives for ``rows`` of ``I``.

        Limits compute and memory by only calculating some rows.

        The bias needs the whole pair rep, so it is computed ``pair_chunk_size``
        pair entries at a time to bound memory. ``z`` may be in a narrower dtype
        than the layer, and is converted a chunk at a time too. Everything else
        is computed only for ``rows``.
        """
        batch, n_i, n_j, _ = z.shape
        bias = torch.empty(
            batch, self.heads, n_i, n_j, device=z.device, dtype=self.bias_weight.dtype
        )
        chunk_size = max(1, pair_chunk_size // (batch * n_j))
        for start in range(0, n_i, chunk_size):
            chunk = slice(start, start + chunk_size)
            bias[:, :, chunk] = self.pair_bias(self.norm(z[:, chunk]))
        index = torch.tensor(rows, device=z.device)
        key_mask = None
        if mask is not None:
            key_mask = mask.index_select(1, index)[:, :, None, None, :]
        x = self.norm(z.index_select(1, index))
        return self.attend(x, self.attention_bias(bias), key_mask)


def transpose_pair(z, mask):
    """The pair rep ``[B, I, J, C]`` and its mask ``[B, I, J]`` transposed."""
    return z.transpose(-2, -3), None if mask is None else mask.transpose(-1, -2)


class TriangleAttentionStack(torch.nn.Module):
    """Residual stack of :class:`TriangleAttentionLayer`.

    ``node`` is ``"starting"``, ``"ending"`` or ``"alternate"``. A pairformer
    block applies the starting node and then the ending one, so its flattened
    sequence alternates. As in OF3's pairformer, an ending-node layer runs on
    the transposed pair rep and mask with ``transpose_bias``, and its residual
    is added in that orientation.
    """

    def __init__(self, attention, depth, c_z, heads, head_dim, dtype, *, node):
        super().__init__()
        self.ending = [
            node == "ending" or (node == "alternate" and index % 2 == 1)
            for index in range(depth)
        ]
        self.layers = torch.nn.ModuleList(
            TriangleAttentionLayer(
                attention,
                c_z,
                heads,
                head_dim,
                dtype,
                transpose_bias=ending,
                seed=index,
            )
            for index, ending in enumerate(self.ending)
        )

    def forward(self, z, mask, chunk_size=None):
        for layer, ending in zip(self.layers, self.ending, strict=True):
            if ending:
                z, mask = transpose_pair(z, mask)
            z = z + layer(z, mask, chunk_size)
            if ending:
                z, mask = transpose_pair(z, mask)
        return z


def reference_copy(module):
    """An fp64 copy of ``module`` whose triangle attention layers run
    :func:`chunked_reference`."""
    copied = copy.deepcopy(module).to(torch.float64)
    for layer in copied.modules():
        if isinstance(layer, TriangleAttentionLayer):
            layer.attention = chunked_reference
    return copied


def assert_stack_agrees_with_oracle(
    model,
    z,
    mask,
    *,
    training,
    inference,
    chunk_size=None,
    grad_out=None,
    precision="ieee",
    tol=None,
    label="",
):
    """``model`` against an fp64 copy of itself running :func:`chunked_reference`.

    With ``training``, compares the output and the gradients wrt ``z`` and every
    parameter, with ``grad_out`` as the output gradient (seeded random by
    default). With ``inference``, compares the output under
    ``torch.inference_mode``, run in ``chunk_size`` chunks of N_seq.
    """
    if training and grad_out is None:
        grad_out = seeded_randn(*z.shape, device=z.device, dtype=z.dtype, seed=41)

    assert_module_agrees_with_oracle(
        model,
        reference_copy(model),
        z,
        call=lambda module, z: module(z, mask),
        infer_call=lambda module, z: module(z, mask, chunk_size),
        training=training,
        inference=inference,
        grad_out=grad_out,
        tol=tol or TRIANGLE_ATTENTION_TOL[z.dtype, precision],
        input_name="z",
        label=label,
    )


def assert_layer_rows_agree_with_oracle(
    model, z, mask, rows, *, training, chunk_size=None, precision="ieee", label=""
):
    """Check a one-layer stack on long inputs, on ``rows`` of ``I`` in the
    layer's orientation.

    Compares the output on those rows against an fp64 copy of the layer running
    :func:`chunked_reference`. The output comes from the training forward if
    ``training``, otherwise from the inference forward with ``chunk_size``. In
    training, this also checks the attention's output and input gradients on
    the tensors the layer passed to it, with :class:`Oracle` on the same rows.
    The gradients of the layer's LayerNorm and projections are not checked.
    """
    (layer,) = model.layers
    (ending,) = model.ending
    tol = TRIANGLE_ATTENTION_TOL[z.dtype, precision]
    captured = {}

    def capturing(q, k, v, bias=None, mask=None, scale=None):
        # Detached, so that holding them doesn't keep the layer's graph alive.
        captured.update(
            q=q.detach(),
            k=k.detach(),
            v=v.detach(),
            bias=bias.detach(),
            mask=mask,
            scale=scale,
        )
        return attention(q, k, v, bias=bias, mask=mask, scale=scale)

    attention = layer.attention
    if training:
        layer.attention = capturing
        try:
            out = model(z, mask)
        finally:
            layer.attention = attention
    else:
        with torch.inference_mode():
            out = model(z, mask, chunk_size)
    if ending:
        out = out.transpose(-2, -3)
        z, mask = transpose_pair(z, mask)
    index = torch.tensor(rows, device=z.device)
    # Keep only the checked rows, so that the training graph is freed before
    # the reference runs.
    out = out.index_select(1, index).detach()
    with torch.no_grad():
        expected = z.index_select(1, index).double() + reference_copy(
            layer
        ).update_rows(z, mask, rows)
    assert_matches_oracle([("out", out, expected)], tol, label=label)
    if training:
        tensors = {k: captured[k].requires_grad_() for k in ("q", "k", "v", "bias")}
        Oracle(
            tensors,
            captured["mask"],
            captured["scale"],
            rows=rows,
            oracle=chunked_reference,
        ).check(attention, tol, label=f"{label} attention")
