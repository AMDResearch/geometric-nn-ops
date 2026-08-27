# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shadow of cuequivariance_ops_torch.triangle_multiplicative_update backed by geometric-nn-ops."""  # noqa: E501

from geometric_nn_ops_torch.triangle import triangle_multiplicative_update as _tmu

# Provided to match cuEquivariance's documented API. We don't actually use this
# to determine when to fall back. The value here is chosen somewhat arbitrarily.
CUEQ_TRIMUL_FALLBACK_THRESHOLD = 16


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
        precision=precision,
    )
