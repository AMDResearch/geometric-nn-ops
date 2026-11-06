# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""IEEE projection fusion with library contraction and full-row normalization."""

import torch

from geometric_nn_ops_torch.triangle._reference.trimul import is_outgoing

from .dual_projection import (
    triton_dual_linear_sigmoid_mul,
    triton_linear_sigmoid_mul,
)
from .layout_aware import _bounded_norm
from .reference_bounded import bounded_combine, bounded_linear


def update(x, direction, mask, eps, p, *, normalize=None):
    if torch.get_float32_matmul_precision() != "highest":
        raise ValueError(
            "FP32 dual projection requires highest/IEEE Torch matmul precision"
        )
    if normalize is None:
        normalize = _bounded_norm
    c, h = x.shape[-1], p.hidden
    z = normalize(
        x.reshape(-1, c),
        p.norm_in_weight,
        p.norm_in_bias,
        eps,
    ).reshape(x.shape)
    row_mask = None if mask is None else mask.expand(x.shape[:-1]).unsqueeze(-1)
    ab = triton_dual_linear_sigmoid_mul(
        z,
        p.g_in_weight,
        p.g_in_bias,
        p.p_in_weight,
        p.p_in_bias,
        row_mask,
        use_bf16=False,
    )
    a, b = ab.split(h, dim=-1)
    o = bounded_combine(a, b, is_outgoing(direction))
    del ab, a, b
    on = normalize(
        o.reshape(-1, h),
        p.norm_out_weight,
        p.norm_out_bias,
        eps,
    ).reshape(o.shape)
    projected = bounded_linear(on, p.p_out_weight, p.p_out_bias)
    return triton_linear_sigmoid_mul(
        projected,
        z,
        p.g_out_weight,
        use_bf16=False,
        bias_g=p.g_out_bias,
    )
