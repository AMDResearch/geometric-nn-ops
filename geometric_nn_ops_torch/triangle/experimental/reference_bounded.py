# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch oracle with bounded library calls for large pair tensors."""

import torch
import torch.nn.functional as F

from geometric_nn_ops_torch.triangle._reference.trimul import (
    _combine_projections,
    is_outgoing,
)

from .layout_aware import _bounded_norm

LIBRARY_MAX_ELEMS = 1 << 28


def bounded_linear(x, weight, bias):
    width = weight.shape[0]
    rows = LIBRARY_MAX_ELEMS // max(x.shape[-1], width)
    if rows < 1:
        raise ValueError("Projection channel width exceeds library element bound")
    flat = x.reshape(-1, x.shape[-1])
    if flat.shape[0] <= rows:
        return F.linear(x, weight, bias)
    result = x.new_empty(flat.shape[0], width)
    for start in range(0, flat.shape[0], rows):
        result[start : start + rows] = F.linear(
            flat[start : start + rows], weight, bias
        )
    return result.reshape(*x.shape[:-1], width)


def bounded_combine(a, b, outgoing):
    n, h = a.shape[-2], a.shape[-1]
    channels = LIBRARY_MAX_ELEMS // (n * n)
    if channels < 1:
        raise ValueError("Residue count exceeds the single-channel contraction bound")
    if a.numel() <= LIBRARY_MAX_ELEMS:
        return _combine_projections(a, b, outgoing)
    af = a.reshape(-1, n, n, h)
    bf = b.reshape(-1, n, n, h)
    result = torch.empty_like(af, memory_format=torch.contiguous_format)
    for batch in range(af.shape[0]):
        for start in range(0, h, channels):
            result[batch, ..., start : start + channels] = _combine_projections(
                af[batch, ..., start : start + channels],
                bf[batch, ..., start : start + channels],
                outgoing,
            )
    return result.reshape(a.shape)


def update(x, direction, mask, eps, p):
    if torch.get_float32_matmul_precision() != "highest":
        raise ValueError(
            "Bounded reference requires highest/IEEE Torch matmul precision"
        )
    c, h = x.shape[-1], p.hidden
    z = _bounded_norm(
        x.reshape(-1, c),
        p.norm_in_weight,
        p.norm_in_bias,
        eps,
    ).reshape(x.shape)
    ab = bounded_linear(z, p.p_in_weight, p.p_in_bias)
    ab.mul_(bounded_linear(z, p.g_in_weight, p.g_in_bias).sigmoid_())
    if mask is not None:
        ab.mul_(mask.unsqueeze(-1))
    a, b = ab.split(h, dim=-1)
    o = bounded_combine(a, b, is_outgoing(direction))
    del ab, a, b
    on = _bounded_norm(
        o.reshape(-1, h),
        p.norm_out_weight,
        p.norm_out_bias,
        eps,
    ).reshape(o.shape)
    projected = bounded_linear(on, p.p_out_weight, p.p_out_bias)
    projected.mul_(bounded_linear(z, p.g_out_weight, p.g_out_bias).sigmoid_())
    return projected
