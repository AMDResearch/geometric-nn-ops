# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from geometric_nn_ops_torch.triangle.experimental.policy import (
    benchmark_order,
    recommend_variants,
)


@pytest.mark.parametrize(
    "n,c,preferred",
    [
        (65, 64, "reference"),
        (128, 256, "dual_projection_fp32"),
        (1024, 384, "layout_aware"),
    ],
)
def test_provisional_preferences(n, c, preferred):
    recommendation = recommend_variants(
        n,
        c,
        c,
        architecture="gfx950:sramecc+",
        ieee_fp32=True,
        contiguous=True,
    )
    assert recommendation.preferred[0] == preferred
    assert recommendation.requires_benchmark
    candidates = ["reference", "layout_aware", "dual_projection_fp32", "pointwise"]
    ordered = benchmark_order(candidates, recommendation)
    assert set(ordered) == set(candidates)
    assert len(ordered) == len(candidates)
    assert candidates[0] == "reference"


@pytest.mark.parametrize(
    "overrides",
    [
        {"architecture": "gfx942"},
        {"ieee_fp32": False},
        {"contiguous": False},
        {"batch": 2},
        {"hidden": 64},
    ],
)
def test_unknown_envelope_does_not_recommend(overrides):
    args = dict(
        n=512, c=128, hidden=128, architecture="gfx950", ieee_fp32=True, contiguous=True
    )
    recommendation = recommend_variants(**(args | overrides))
    assert recommendation.preferred == ()
    assert recommendation.requires_benchmark


def test_invalid_dimensions_and_candidate_are_explicit_errors():
    with pytest.raises(ValueError, match="positive"):
        recommend_variants(
            0, 64, 64, architecture="gfx950", ieee_fp32=True, contiguous=True
        )
    recommendation = recommend_variants(
        128, 128, 128, architecture="gfx950", ieee_fp32=True, contiguous=True
    )
    with pytest.raises(ValueError, match="Unknown"):
        benchmark_order(["typo"], recommendation)
