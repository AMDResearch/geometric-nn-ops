# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for triangle_multiplicative_update_, the in-place inference variant.

Tests that only make sense for the inplace path. The main tests are in
test_trimul.py
"""

import pytest
import torch

from geometric_nn_ops_torch.triangle import triangle_multiplicative_update_
from geometric_nn_ops_torch.triangle._kernels.trimul_chunked import (
    _COL,
    _ROW,
    _IncomingChunks,
    _OutgoingChunks,
)
from geometric_nn_ops_torch.triangle._reference.trimul import trimul_torch
from testutil import cast_tree
from testutil.metrics import TRIMUL_TOL, assert_matches_oracle
from testutil.precision import torch_matmul_precision
from testutil.trimul import make_pair_input, make_projection_biases, make_weights
from utils import DIRECTIONS_PARAM, requires_gpu

# Narrower hidden width than C_z, so a channel slice in the wrong place cannot
# line up by accident.
C_Z, C_HIDDEN = 32, 16


def make_case(batch, n, device, dtype, *, seed=0, requires_grad=False):
    weights = make_weights(
        C_Z, C_HIDDEN, device, dtype, seed=seed, requires_grad=requires_grad
    ) | make_projection_biases(
        C_Z, C_HIDDEN, device, dtype, seed=seed + 1, requires_grad=requires_grad
    )
    x, mask = make_pair_input(
        batch, n, C_Z, device, dtype, seed=seed + 2, requires_grad=requires_grad
    )
    return x, mask, weights


def run_inplace(x, mask, weights, direction, chunk_size):
    """The in-place op on a copy of ``x``, checking it returns that copy."""
    out = x.detach().clone()
    with torch.no_grad():
        result = triangle_multiplicative_update_(
            out, direction, mask, **weights, chunk_size=chunk_size
        )
    assert result is out
    return out


def assert_agrees_with_the_oracle(x, mask, weights, direction, chunk_size, label):
    """Compare the in-place result against ``x`` plus the reference's update,
    computed in fp64."""
    with torch_matmul_precision("ieee"):
        out = run_inplace(x, mask, weights, direction, chunk_size)
    expected = x.double() + trimul_torch(
        x.double(),
        direction,
        mask,
        **cast_tree(weights, torch.float64, requires_grad=False),
    )
    assert_matches_oracle(
        [("x + update", out.double(), expected)],
        TRIMUL_TOL[x.dtype, "ieee"],
        label=label,
    )


@requires_gpu
@DIRECTIONS_PARAM
@pytest.mark.parametrize("n", [31, 32], ids=["odd_n", "even_n"])
@pytest.mark.parametrize("chunk_size", [None, 1, 5, 6, 8, 64])
def test_matches_oracle(direction, n, chunk_size):
    """Both N have midpoint 16. Chunk sizes 5 and 6 leave a short chunk before
    it and 8 ends exactly on it; 64 is a single chunk per half."""
    x, mask, weights = make_case(1, n, "cuda", torch.float32, seed=3)
    assert_agrees_with_the_oracle(
        x, mask, weights, direction, chunk_size, f"{direction}/N{n}/{chunk_size}"
    )


@requires_gpu
@DIRECTIONS_PARAM
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("chunk_size", [1, 5, 16, 31])
def test_chunk_size_does_not_change_the_result(direction, dtype, chunk_size):
    """Every output element goes through the same kernels with the same inputs
    whatever the chunking, so the results are identical, not just close. This
    also relies on the combine matmul giving the same values for a slice of its
    columns as for all of them."""
    x, mask, weights = make_case(1, 31, "cuda", dtype, seed=15)
    unchunked = run_inplace(x, mask, weights, direction, None)
    chunked = run_inplace(x, mask, weights, direction, chunk_size)
    assert torch.equal(chunked, unchunked)


@requires_gpu
@DIRECTIONS_PARAM
@pytest.mark.parametrize("batch", [2, (2, 3)], ids=["batched", "5D"])
@pytest.mark.parametrize("mask_kind", ["none", "full", "broadcast"])
def test_batch_and_mask(direction, batch, mask_kind):
    """Leading dims (templates are 5D), and a mask that is absent or has fewer
    leading dims than ``x``."""
    x, mask, weights = make_case(batch, 32, "cuda", torch.float32, seed=5)
    if mask_kind == "none":
        mask = None
    elif mask_kind == "broadcast":
        mask = mask.reshape(-1, 32, 32)[0]
    assert_agrees_with_the_oracle(
        x,
        mask,
        weights,
        direction,
        4,
        f"{direction}/{batch}/{mask_kind}",
    )


@requires_gpu
@DIRECTIONS_PARAM
def test_bf16(direction):
    x, mask, weights = make_case(1, 33, "cuda", torch.bfloat16, seed=7)
    assert_agrees_with_the_oracle(x, mask, weights, direction, 5, direction)


@pytest.mark.parametrize("chunk_size", [None, 2])
@pytest.mark.parametrize(
    "which", ["input", "weights"], ids=["input_requires_grad", "weights_require_grad"]
)
def test_rejects_autograd(which, chunk_size):
    """It overwrites ``x`` and has no backward, so it must not run where
    autograd would record it. ``x`` is left untouched."""
    x, mask, weights = make_case(1, 6, "cpu", torch.float32, seed=9)
    if which == "input":
        x.requires_grad_()
    else:
        weights["p_in_weight"].requires_grad_()
    before = x.detach().clone()

    with pytest.raises(RuntimeError, match="no backward"):
        triangle_multiplicative_update_(
            x, "outgoing", mask, **weights, chunk_size=chunk_size
        )

    torch.testing.assert_close(x.detach(), before, rtol=0, atol=0)


@requires_gpu
@DIRECTIONS_PARAM
@pytest.mark.parametrize("pair_shape", [(1, 32), (32, 1), (1, 1)])
def test_mask_broadcast_over_pair_dims(direction, pair_shape):
    """A mask broadcast along a pair dim, which the chunked path has to expand
    before slicing it."""
    x, _, weights = make_case((), 32, "cuda", torch.float32, seed=13)
    generator = torch.Generator(device="cuda").manual_seed(14)
    mask = torch.rand(pair_shape, device="cuda", generator=generator) < 0.7
    assert_agrees_with_the_oracle(
        x, mask, weights, direction, 5, f"{direction}/{pair_shape}"
    )


@DIRECTIONS_PARAM
@pytest.mark.parametrize("chunk_size", [None, 1, 2])
def test_rejects_overlapping_input(direction, chunk_size):
    """An ``x`` expanded along a pair dim has columns sharing memory. A
    one-column chunk is too narrow for the write itself to notice, and every
    write would land in all of them. It must be rejected without performing any
    updates."""
    x, mask, weights = make_case((), 6, "cpu", torch.float32, seed=9)
    expanded = x[:, :1].expand_as(x)
    before = x.clone()

    with torch.no_grad(), pytest.raises(ValueError, match="stride-0"):
        triangle_multiplicative_update_(
            expanded, direction, mask, **weights, chunk_size=chunk_size
        )

    torch.testing.assert_close(x, before, rtol=0, atol=0)


@requires_gpu
@pytest.mark.parametrize(
    "no_backward", [torch.no_grad, torch.inference_mode], ids=["no_grad", "inference"]
)
def test_no_grad_modes_accept_parameters(no_backward):
    """Parameters keep ``requires_grad`` in a model, which is fine wherever no
    graph is recorded, and changes nothing about the result."""
    x, mask, weights = make_case(1, 32, "cuda", torch.float32, seed=9)
    expected = run_inplace(x, mask, weights, "outgoing", 2)
    weights = cast_tree(weights, torch.float32, requires_grad=True)
    with no_backward():
        triangle_multiplicative_update_(x, "outgoing", mask, **weights, chunk_size=2)
    assert torch.equal(x, expected)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_rejects_nonpositive_chunk_size(chunk_size):
    x, mask, weights = make_case(1, 6, "cpu", torch.float32, seed=9)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        triangle_multiplicative_update_(
            x, "outgoing", mask, **weights, chunk_size=chunk_size
        )


def test_without_triton_adds_the_reference():
    """Off the Triton path, it adds the reference's update."""
    x, mask, weights = make_case(1, 6, "cpu", torch.float32, seed=9)
    expected = x + trimul_torch(x, "outgoing", mask, **weights)
    out = run_inplace(x, mask, weights, "outgoing", None)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_without_triton_rejects_chunking():
    """Chunking is only supported with Triton."""
    x, mask, weights = make_case(1, 6, "cpu", torch.float32, seed=9)
    before = x.clone()
    with torch.no_grad(), pytest.raises(ValueError, match="needs the Triton path"):
        triangle_multiplicative_update_(x, "outgoing", mask, **weights, chunk_size=2)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


# How much the chunk iterator tests add into each yielded columns chunk.
_WRITTEN = 1000.0


@DIRECTIONS_PARAM
@pytest.mark.parametrize("n", [1, 6, 7])
@pytest.mark.parametrize("chunk_size", [1, 2, 4, 16])
@pytest.mark.parametrize("contiguous", [True, False], ids=["contiguous", "strided"])
def test_chunks_yield_the_original_input(direction, n, chunk_size, contiguous):
    """The chunk iterators alone, on CPU: each ``b_input`` is the original
    input even though every earlier ``columns`` chunk has been written, and the
    ``columns`` chunks cover ``x`` exactly once, left to right."""
    generator = torch.Generator().manual_seed(n)
    base = torch.randn(2, n, n, 3, generator=generator, dtype=torch.float64)
    x = base if contiguous else base.transpose(_ROW, _COL)
    original = x.clone()
    outgoing = direction == "outgoing"
    chunks = (_OutgoingChunks if outgoing else _IncomingChunks)(x, None, chunk_size)

    start = 0
    for columns, b_input, _ in chunks:
        length = columns.shape[_COL]
        if outgoing:
            expected = original.narrow(_ROW, start, length).transpose(_ROW, _COL)
        else:
            expected = original.narrow(_COL, start, length)
        assert torch.equal(b_input, expected), f"b_input at column {start}"
        columns.add_(_WRITTEN)
        start += length

    assert start == n
    torch.testing.assert_close(x, original + _WRITTEN, rtol=0, atol=0)


def peak_bytes_above_baseline(fn) -> int:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - baseline


@requires_gpu
@DIRECTIONS_PARAM
def test_chunking_bounds_peak_memory(direction):
    """With small chunks, the memory beyond ``x`` approaches ``a`` (the size of
    ``x`` here, as ``C_hidden == C_z``) plus, for outgoing, the copies of half
    of ``x``. Unchunked, it is several times ``x``."""
    n, c = 128, 32
    weights = make_weights(c, c, "cuda", torch.float32, seed=11)
    x, mask = make_pair_input((), n, c, "cuda", torch.float32, seed=12)
    x_bytes = x.numel() * x.element_size()

    def run(chunk_size):
        triangle_multiplicative_update_(
            x, direction, mask, **weights, chunk_size=chunk_size
        )

    with torch.no_grad():
        # The first call also allocates one-time state such as the BLAS
        # workspace.
        run(None)
        unchunked = peak_bytes_above_baseline(lambda: run(None)) / x_bytes
        chunked = peak_bytes_above_baseline(lambda: run(n // 16)) / x_bytes

    bound = 2.25 if direction == "outgoing" else 1.75
    assert chunked < bound, f"chunked peak {chunked:.2f}x x, unchunked {unchunked:.2f}x"
    assert unchunked > 2 * chunked, (
        f"chunking saved little: {chunked:.2f}x x vs {unchunked:.2f}x unchunked"
    )
