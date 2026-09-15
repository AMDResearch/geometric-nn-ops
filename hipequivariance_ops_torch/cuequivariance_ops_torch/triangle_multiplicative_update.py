# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Implements the API of cuequivariance_ops_torch.triangle_multiplicative_update, backed by geometric-nn-ops."""  # noqa: E501

import enum
import warnings

from geometric_nn_ops_torch import _precision
from geometric_nn_ops_torch.triangle import triangle_multiplicative_update as _tmu

from ._trimul_precision import TriMulPrecision

# Provided to match cuEquivariance's documented API. We don't actually use this
# to determine when to fall back. The value here is chosen somewhat arbitrarily.
CUEQ_TRIMUL_FALLBACK_THRESHOLD = 16

# Spellings that express no preference, so there is nothing to check against.
_NO_REQUEST = frozenset(
    {TriMulPrecision.NONE.name.lower(), TriMulPrecision.DEFAULT.name.lower()}
)


def _check_precision(precision) -> None:
    """Warn or raise if torch's setting does not match ``precision``.

    We read torch's fp32 matmul precision and have no per-call override, so a
    request can only be checked. Asking for more precision than torch is set to
    is an error as we cannot deliver that precision. Otherwise, asking for
    something other than the torch setting is an error, as we can't guarantee
    satisfying the required precision. Note that we could make this tighter by
    producing an ordering over possible precision modes, but for now the
    cuequivariance documentation says only "tf32" and "none" are allowed.

    cuequivariance documents the argument as an enum, but appears to actually
    only accept strings. We accept both as well as ints, which are cast to the
    enum.
    """
    if precision is None:
        return
    precision_orig = precision
    if isinstance(precision, int):
        precision = TriMulPrecision(precision)
    if isinstance(precision, enum.IntEnum):
        precision = precision.name
    if isinstance(precision, str):
        precision = precision.lower()
    if precision in _NO_REQUEST:
        return

    torch_precision = _precision.torch_precision()

    if precision != torch_precision:
        if torch_precision == _precision.IEEE:
            warnings.warn(
                f"requested triangle_multiplicative_update "
                f"precision={precision_orig!r}, but "
                f"torch.backends.cuda.matmul.fp32_precision "
                f"is {torch_precision!r}. Matmuls will run in full fp32 without the "
                f"speedup. Set the torch global to control precision.",
                stacklevel=3,
            )
        else:
            raise ValueError(
                f"requested triangle_multiplicative_update "
                f"precision={precision_orig!r}, but "
                f"torch.backends.cuda.matmul.fp32_precision "
                f"is {torch_precision!r}, which may not be able to satisfy the "
                f"requirement. Set the torch global to control precision.",
            )


def triangle_multiplicative_update(
    x,
    direction="outgoing",
    mask=None,
    norm_in_weight=None,
    norm_in_bias=None,
    p_in_weight=None,
    p_in_bias=None,
    g_in_weight=None,
    g_in_bias=None,
    norm_out_weight=None,
    norm_out_bias=None,
    p_out_weight=None,
    p_out_bias=None,
    g_out_weight=None,
    g_out_bias=None,
    eps=1e-5,
    precision=None,
):
    _check_precision(precision)
    return _tmu(
        x,
        direction=direction,
        mask=mask,
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        p_in_weight=p_in_weight,
        p_in_bias=p_in_bias,
        g_in_weight=g_in_weight,
        g_in_bias=g_in_bias,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        p_out_bias=p_out_bias,
        g_out_weight=g_out_weight,
        g_out_bias=g_out_bias,
        eps=eps,
    )
