# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The shim's precision checking -- no GPU required.

We read torch's fp32 matmul precision and have no per-call override, so cueq's
``precision`` argument can only be checked against that setting.
"""

import pytest
from cuequivariance_ops_torch import TriMulPrecision
from cuequivariance_ops_torch.triangle_multiplicative_update import _check_precision

from testutil.precision import torch_matmul_precision


def spellings(*members):
    """Every way a caller can name the IntEnum ``members``."""
    return [
        spelling
        for member in members
        for spelling in (member, int(member), member.name, member.name.lower())
    ]


NO_PREFERENCE = spellings(TriMulPrecision.NONE, TriMulPrecision.DEFAULT)
NEEDS_FULL_PRECISION = spellings(TriMulPrecision.IEEE, TriMulPrecision.TF32x3)
PERMITS_REDUCED = spellings(TriMulPrecision.TF32)


@pytest.mark.parametrize("precision", [None, *spellings(*TriMulPrecision)], ids=str)
def test_accepts_every_request_when_torch_is_ieee(precision):
    """Full precision satisfies every request, including those that would have
    permitted less."""
    with torch_matmul_precision("ieee"):
        _check_precision(precision)


@pytest.mark.parametrize("precision", [None, *NO_PREFERENCE], ids=str)
def test_no_preference_passes_whatever_torch_is_set_to(precision):
    """NONE and DEFAULT express no request."""
    with torch_matmul_precision("tf32"):
        _check_precision(precision)


@pytest.mark.parametrize("precision", NEEDS_FULL_PRECISION, ids=str)
def test_rejects_more_precision_than_torch_is_set_to(precision):
    """We can't run with reduced precision when the caller asked for full."""
    with (
        torch_matmul_precision("tf32"),
        pytest.raises(ValueError, match="may not be able to satisfy"),
    ):
        _check_precision(precision)


@pytest.mark.parametrize("precision", PERMITS_REDUCED, ids=str)
def test_warns_when_torch_gives_more_precision_than_requested(precision):
    """Accuracy is satisfied, so this is not an error, but the caller asked for
    a speedup they are not getting."""
    with (
        torch_matmul_precision("ieee"),
        pytest.warns(UserWarning, match="without the speedup"),
    ):
        _check_precision(precision)


@pytest.mark.parametrize("precision", [7, -2], ids=str)
def test_rejects_an_unknown_int(precision):
    """Ints go through the enum, which has a fixed set of values."""
    with pytest.raises(ValueError):
        _check_precision(precision)


@pytest.mark.parametrize("precision", ["bogus", "mxfp8", "bf16"], ids=str)
def test_tolerates_a_name_it_does_not_know(precision):
    """Deliberate: the check compares the requested name to torch's setting
    rather than against a fixed list, so names added by later cuequivariance
    releases warn and run in full fp32 instead of hard-failing. A typo gets the
    same treatment, which is the cost of not pinning the set."""
    with torch_matmul_precision("ieee"), pytest.warns(UserWarning):
        _check_precision(precision)
