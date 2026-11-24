# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""In-place triangle multiplicative update, a chunk of output columns at a time.

The chunking algorithm is from OpenFold3's in-place inference path. Uses the
Triton stages in ``_kernels.trimul``.
"""

import torch

from geometric_nn_ops_torch.triangle._kernels.trimul import (
    gated_projection,
    matmul_precision,
    output,
    triton_layernorm,
)
from geometric_nn_ops_torch.triangle._masking import check_bool_mask
from geometric_nn_ops_torch.triangle._reference.trimul import is_outgoing

# Pair dims of ``x``. The mask shares them after a trailing singleton dim is added.
_ROW, _COL = -3, -2


def _narrow(
    t: torch.Tensor | None, dim: int, start: int, length: int
) -> torch.Tensor | None:
    return None if t is None else t.narrow(dim, start, length)


def _chunks(low: int, high: int, chunk_size: int):
    """``(start, length)`` pairs covering ``range(low, high)``."""
    for start in range(low, high, chunk_size):
        yield start, min(chunk_size, high - start)


# _IncomingChunks and _OutgoingChunks iterate over chunks of the output columns
# of ``x``, left to right. For each they yield ``(columns, b_input, b_mask)``:
# ``columns`` is the view of ``x`` the update for those columns should be added
# into, and ``b_input`` holds the entries of the original ``x`` that ``b`` is
# projected from, laid out like ``columns`` (``[*, N, length, C_z]``), with
# ``b_mask`` matching it. The caller may write into columns already yielded but
# must not change ``x`` anywhere else.


class _IncomingChunks:
    """Incoming projects ``b`` for output columns ``j`` from columns ``j`` of
    ``x``, which are not overwritten until then."""

    def __init__(self, x, mask, chunk_size):
        self.x, self.mask, self.chunk_size = x, mask, chunk_size

    def __iter__(self):
        for start, length in _chunks(0, self.x.shape[_COL], self.chunk_size):
            columns = self.x.narrow(_COL, start, length)
            yield columns, columns, _narrow(self.mask, _COL, start, length)


class _OutgoingChunks:
    """Outgoing projects ``b`` for output columns ``j`` from rows ``j`` of
    ``x``, which immediately start getting overwritten.

    To reduce memory usage we avoid keeping a full copy of ``x``. Instead, we
    use a partial copy bounded at half the size of ``x``. We divide ``x`` into 4
    quadrants. Each half of ``x`` is chunked separately so that no chunk
    overlaps both halves. Then when computing the projection for the first half
    of ``x`` we need the first half of the rows: the top-left and top-right
    quadrants of ``x``. When computing the projection for the second half of
    ``x``, we need the second half of the rows: the bottom-left and bottom-right
    quadrants of ``x``. As soon as the first chunk is processed, the left half
    of ``x`` is partially overwritten, so at the start we make a copy of the
    top-left and bottom-left quadrants. As we process the first half of ``x``,
    we return row chunks formed from the copied top-left quadrant and a view on
    the top-right quadrant of ``x``. Once the first half is finished
    processing, we no longer need the top-left quadrant, so we free it. The
    bottom-right quadrant is still untouched, so we can make a copy of it at
    that point. Then we proceed, returning row chunks formed from the copied
    bottom-left and bottom-right quadrants. We never hold more than two
    quadrants of ``x``, so the total storage cost is only half the size of
    ``x`` (plus chunk-size intermediates).
    """

    def __init__(self, x, mask, chunk_size):
        self.x, self.mask, self.chunk_size = x, mask, chunk_size

    def _yield_rows(self, low, high, left, right):
        """Chunks for rows ``low:high``, rebuilt from ``left`` and ``right``,
        which hold columns ``:half_n`` and ``half_n:`` of those rows."""
        for start, length in _chunks(low, high, self.chunk_size):
            b_mask = _narrow(self.mask, _ROW, start, length)
            if b_mask is not None:
                b_mask = b_mask.transpose(_ROW, _COL)
            # The concatenated rows go straight to the caller, so they are freed
            # as soon as the caller is done with them.
            yield (
                self.x.narrow(_COL, start, length),
                torch.cat(
                    [
                        left.narrow(_ROW, start - low, length),
                        right.narrow(_ROW, start - low, length),
                    ],
                    dim=_COL,
                ).transpose(_ROW, _COL),
                b_mask,
            )

    def __iter__(self):
        x = self.x
        n = x.shape[_ROW]
        half_n = (n + 1) // 2
        top, bottom = x.narrow(_ROW, 0, half_n), x.narrow(_ROW, half_n, n - half_n)
        top_left = top.narrow(_COL, 0, half_n).clone()
        bottom_left = bottom.narrow(_COL, 0, half_n).clone()
        top_right = top.narrow(_COL, half_n, n - half_n)
        yield from self._yield_rows(0, half_n, top_left, top_right)
        del top_left
        bottom_right = bottom.narrow(_COL, half_n, n - half_n).clone()
        yield from self._yield_rows(half_n, n, bottom_left, bottom_right)


def trimul_triton_chunked_add_(
    x,
    direction="outgoing",
    mask=None,
    *,
    chunk_size,
    eps=1e-5,
    norm_in_weight=None,
    norm_in_bias=None,
    p_in_weight=None,
    p_in_bias=None,
    g_in_weight=None,
    g_in_bias=None,
    norm_out_weight=None,
    norm_out_bias=None,
    p_out_weight=None,
    p_out_bias=None,
    g_out_weight=None,
    g_out_bias=None,
) -> None:
    """``x += trimul_triton(x, ...)``, computed ``chunk_size`` columns of the
    output at a time to reduce memory usage. Forward only. Takes the same
    arguments as ``trimul_triton`` plus ``chunk_size``

    ``a`` is computed in full. Each step then projects ``chunk_size`` rows
    (outgoing) or columns (incoming) of ``x`` into a chunk of ``b``, combines it
    with ``a`` into ``chunk_size`` columns of the update, and adds those into
    ``x``. Outgoing keeps a half-size copy of the necessary rows of ``x`` (see
    :class:`_OutgoingChunks`) because later steps read rows whose left columns
    have already been overwritten. Incoming only reads columns not yet written.

    Beyond ``x``, this holds ``a`` (``C_hidden / C_z`` the size of ``x``), the
    row copies (half the size of ``x``, outgoing only), and chunk-sized
    intermediates.
    """
    outgoing = is_outgoing(direction)
    check_bool_mask(mask)
    precision = matmul_precision(x.dtype)
    c_hidden = p_in_weight.shape[0] // 2

    def normalize(t):
        return triton_layernorm(t, norm_in_weight, norm_in_bias, eps)

    def project(z_n, mask, channels):
        """Channel-first: ``[*, R, C, C_z]`` becomes ``[*, C_hidden, R, C]``."""
        p = gated_projection(
            z_n,
            mask,
            precision,
            channels,
            p_in_weight,
            p_in_bias,
            g_in_weight,
            g_in_bias,
        )
        return p.movedim(-1, -3)

    n = x.shape[_ROW]
    if mask is not None:
        # explicit broadcast so it can be sliced.
        mask = mask.expand(*mask.shape[:-2], n, n).unsqueeze(-1)
    # `a`, laid out as the left operand of the combine, [*, C_hidden, N, N].
    a = x.new_empty(*x.shape[:-3], c_hidden, n, n)
    for start, length in _chunks(0, n, chunk_size):
        p = project(
            normalize(x.narrow(_ROW, start, length)),
            _narrow(mask, _ROW, start, length),
            slice(None, c_hidden),
        )
        if outgoing:
            a.narrow(-2, start, length).copy_(p)
        else:
            a.narrow(-1, start, length).copy_(p.transpose(-1, -2))
    # The last chunk's projection would otherwise stay alive through the loop
    # below.
    p = None
    chunks = (_OutgoingChunks if outgoing else _IncomingChunks)(x, mask, chunk_size)
    for columns, b_input, b_mask in chunks:
        z_n = normalize(columns)
        # Incoming projects b from the same columns the output gate reads.
        z_n_b = z_n if b_input is columns else normalize(b_input)
        # For outgoing this is a chunk-sized copy we can free now
        del b_input
        b = project(z_n_b, b_mask, slice(c_hidden, None))
        o = torch.matmul(a, b).movedim(-3, -1)
        update = output(
            o,
            z_n,
            precision,
            eps,
            norm_out_weight,
            norm_out_bias,
            p_out_weight,
            p_out_bias,
            g_out_weight,
            g_out_bias,
            add=columns,
        )
        columns.copy_(update)
