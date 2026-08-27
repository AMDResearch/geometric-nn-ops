# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Triangle (protein-folding) ops."""

from ._attention_pair_bias import attention_pair_bias
from ._triattn import triangle_attention
from ._trimul import triangle_multiplicative_update

__all__ = [
    "attention_pair_bias",
    "triangle_attention",
    "triangle_multiplicative_update",
]
