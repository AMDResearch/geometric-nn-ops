# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Triangle (protein-folding) ops.

The public ``triangle_attention`` mirrors the call signature of
``cuequivariance_torch.primitives.triangle.triangle_attention`` so call sites
can switch backends with a single import change.
"""

from ._kernels.triangle_attention import TritonEvoformer
from ._trimul import triangle_multiplicative_update
from ._triattn import triangle_attention

__all__ = [
    "triangle_attention",
    "triangle_multiplicative_update",
    "TritonEvoformer",
]
