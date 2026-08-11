# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared metrics."""

import pytest
import torch

from testutil.metrics import Tolerance, assert_matches_oracle, cos

EXACT = Tolerance(1.0, 0.0)


def test_zero_matches_zero():
    """A structurally zero gradient equals a zero oracle.

    Cosine similarity is undefined for a zero vector and torch reports 0, which
    would reject the exact match.
    """
    zero = torch.zeros(4, 8)

    assert cos(zero, zero) == 1.0
    assert_matches_oracle([("zero", zero, zero)], EXACT, label="zeros")


@pytest.mark.parametrize("zero_side", ["candidate", "oracle"])
def test_zero_does_not_match_nonzero(zero_side):
    """Only one side being zero is still a mismatch."""
    zero = torch.zeros(4, 8)
    nonzero = torch.ones(4, 8)
    candidate, oracle = (zero, nonzero) if zero_side == "candidate" else (nonzero, zero)

    assert cos(candidate, oracle) == 0.0
    with pytest.raises(AssertionError):
        assert_matches_oracle([("g", candidate, oracle)], EXACT, label=zero_side)
