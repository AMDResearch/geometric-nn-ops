# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shadow of NVIDIA's closed ``cuequivariance_ops_torch``, backed by Triton.

Installing this package makes ``from cuequivariance_ops_torch import ...``
resolve to the geometric-nn-ops kernels, so unmodified code using
``cuequivariance_torch.primitives.triangle`` runs on AMD ROCm with zero source
changes.

This is the optional, zero-code-change convenience surface. Where possible, we
recommend using ``geometric-nn-ops`` directly instead.
"""

# The imports below intentionally run after `_guard.check()`, so E402 (import
# not at top of file) is disabled for this module. The guard must reject a
# conflicting closed-source wheel before we import the shadowing submodules.
# ruff: noqa: E402

from . import _guard

_guard.check()

from ._trimul_precision import TriMulPrecision

# Rebinds the name to the function while the submodule stays importable, so both
# `from cuequivariance_ops_torch import triangle_attention` and the deep
# `from cuequivariance_ops_torch.triangle_attention import CUEQ_TRIATTN_FALLBACK_THRESHOLD` # noqa: E501
# resolve.
from .triangle_attention import triangle_attention
from .triangle_multiplicative_update import triangle_multiplicative_update

__version__ = "0.1.0"

__all__ = [
    "triangle_attention",
    "triangle_multiplicative_update",
    "TriMulPrecision",
]
