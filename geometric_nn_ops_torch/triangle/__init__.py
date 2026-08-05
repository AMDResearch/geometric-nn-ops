# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Triangle (protein-folding) ops."""

from ._triattn import triangle_attention
from ._trimul import triangle_multiplicative_update

__all__ = [
    "triangle_attention",
    "triangle_multiplicative_update",
]
