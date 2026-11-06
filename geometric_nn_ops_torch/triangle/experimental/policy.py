# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Provisional benchmark recommender, not automatic execution dispatch.

Fusion-library/shape-aware mux concept and assembly
TODO: Train or derive an effective selection policy from broader, uncontended
measurements, accounting for hardware/runtime, layout, precision, memory and
qualification envelopes. Current observations do not establish universal rules.
"""

from dataclasses import dataclass

from .registry import VARIANTS

POLICY_VERSION = "provisional-common-sizes-v1"


@dataclass(frozen=True)
class Recommendation:
    preferred: tuple[str, ...]
    reason: str
    policy_version: str = POLICY_VERSION
    requires_benchmark: bool = True


def recommend_variants(
    n: int,
    c: int,
    hidden: int,
    direction: str = "outgoing",
    *,
    architecture: str,
    ieee_fp32: bool,
    contiguous: bool,
    batch: int = 1,
) -> Recommendation:
    if min(n, c, hidden, batch) <= 0:
        raise ValueError("Policy dimensions must be positive")
    if direction not in ("incoming", "outgoing"):
        raise ValueError("Policy direction must be incoming or outgoing")
    if (
        not architecture.startswith("gfx950")
        or not ieee_fp32
        or not contiguous
        or batch != 1
        or c != hidden
        or c not in (64, 128, 256, 384)
    ):
        return Recommendation((), "Outside observed envelope; benchmark all candidates")
    if n <= 128 and (c == 64 or (n < 128 and c < 384)):
        return Recommendation(
            ("reference",), "Tiny/narrow cases favored library reference"
        )
    if n <= 256:
        preferred = ("dual_projection_fp32", "layout_aware")
        if n == 256 and c == 128 and direction == "incoming":
            preferred = ("channel_major", *preferred)
        return Recommendation(preferred, "Small/medium cases favored projection fusion")
    return Recommendation(
        ("layout_aware", "dual_projection_fp32"),
        "Larger cases favored consumer-layout execution; extrapolation unqualified",
    )


def benchmark_order(
    candidates: list[str],
    recommendation: Recommendation,
) -> list[str]:
    unknown = set(candidates) - VARIANTS.keys()
    if unknown:
        raise ValueError(f"Unknown benchmark candidates: {sorted(unknown)}")
    ordered = [name for name in recommendation.preferred if name in candidates]
    return ordered + [name for name in candidates if name not in ordered]
