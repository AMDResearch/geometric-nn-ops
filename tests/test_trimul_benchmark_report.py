# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

from benchmarks.benchmark_trimul_variants import qualified_report


def test_comparison_contains_only_qualified_measured_cases():
    base = {"n": 65, "c": 64, "hidden": 64, "direction": "outgoing"}
    passing = base | {
        "variant": "passing",
        "status": "measured",
        "qualified": True,
        "finite": True,
    }
    failed = base | {
        "variant": "failed",
        "status": "numerical_failure",
        "qualified": False,
        "finite": True,
    }
    unmeasured = base | {
        "variant": "unmeasured",
        "status": "rejected_or_execution_error",
        "qualified": True,
        "finite": True,
    }
    nonfinite = base | {
        "variant": "nonfinite",
        "status": "measured",
        "qualified": True,
        "finite": False,
    }
    report = {
        "results": [passing, failed, unmeasured, nonfinite],
        "winners": [base | {"variant": "passing"}, base | {"variant": "failed"}],
        "variants": {
            name: {} for name in ("passing", "failed", "unmeasured", "nonfinite")
        },
        "oracle": "reference_bounded",
    }
    comparison = qualified_report(report)
    assert comparison["results"] == [passing]
    assert comparison["winners"] == [base | {"variant": "passing"}]
    assert set(comparison["variants"]) == {"passing"}
    assert comparison["oracle"] == "reference_bounded"
    assert len(report["results"]) == 4


def test_empty_qualification_has_no_winners():
    report = {"results": [], "winners": [], "variants": {"unused": {}}}
    comparison = qualified_report(report)
    assert comparison["results"] == []
    assert comparison["winners"] == []
    assert comparison["variants"] == {}
