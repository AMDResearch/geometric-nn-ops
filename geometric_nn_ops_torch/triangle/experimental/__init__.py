# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in TriMul fusion experiments; not part of automatic production dispatch."""

from .policy import recommend_variants
from .registry import VARIANTS, run_variant

__all__ = ["VARIANTS", "run_variant", "recommend_variants"]
