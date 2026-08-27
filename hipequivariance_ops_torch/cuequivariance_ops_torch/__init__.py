# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shadow of NVIDIA's closed ``cuequivariance_ops_torch``, backed by Triton.

Installing this package makes ``from cuequivariance_ops_torch import ...``
resolve to the geometric-nn-ops kernels, so unmodified code using
``cuequivariance_torch.primitives.triangle`` runs on AMD ROCm with zero source
changes.

This is the optional, zero-code-change convenience surface. Where possible, we
recommend using ``geometric-nn-ops-torch`` directly instead.
"""

from . import _guard

_guard.check()

# The imports below must run after `_guard.check()`, so disable E402 (import not
# at top of file) for this file. `isort: split` starts a fresh sorting block so
# isort doesn't hoist a new import.

# ruff: noqa: E402
# isort: split
from ._trimul_precision import TriMulPrecision

# Rebinds the name to the function while the submodule stays importable, so both
# `from cuequivariance_ops_torch import triangle_attention` and the deep
# `from cuequivariance_ops_torch.triangle_attention import CUEQ_TRIATTN_FALLBACK_THRESHOLD` # noqa: E501
# resolve.
from .attention_pair_bias_torch import attention_pair_bias
from .triangle_attention import triangle_attention
from .triangle_multiplicative_update import triangle_multiplicative_update

__version__ = "0.1.0"

__all__ = [
    "attention_pair_bias",
    "triangle_attention",
    "triangle_multiplicative_update",
    "TriMulPrecision",
]
