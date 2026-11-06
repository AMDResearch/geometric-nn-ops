# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Functional adaptation of Conductor layout-aware v2, without model monkeypatches."""

import torch

from geometric_nn_ops_torch.triangle._reference.trimul import is_outgoing

from ._pipeline import norm

LN_MAX_ELEMS = 1 << 30
BUDGET_BYTES = 12 << 30


def _divisor_at_most(n, cap):
    for size in range(min(n, cap), 0, -1):
        if n % size == 0:
            return size
    raise ValueError("Budget cannot hold a single projection channel")


def _bounded_norm(x, weight, bias, eps):
    rows = LN_MAX_ELEMS // x.shape[-1]
    if rows < 1:
        raise ValueError("Channel width exceeds LayerNorm element bound")
    if x.shape[0] <= rows:
        return norm(x, weight, bias, eps)
    result = torch.empty_like(x)
    for start in range(0, x.shape[0], rows):
        result[start : start + rows] = norm(x[start : start + rows], weight, bias, eps)
    return result


def update(x, direction, mask, eps, p, *, channel_major_norm=None, input_norm=None):
    if x.dtype != torch.float32 or torch.is_autocast_enabled(x.device.type):
        raise ValueError("Layout-aware experiment requires FP32 without autocast")
    if torch.is_grad_enabled() and (
        x.requires_grad
        or any(t is not None and t.requires_grad for t in p.kwargs().values())
        or (mask is not None and mask.requires_grad)
    ):
        raise ValueError("Layout-aware experiment is forward-only")
    n, c, h = x.shape[-2], x.shape[-1], p.hidden
    if input_norm is None:
        input_norm = _bounded_norm
    m = n * n
    outgoing = is_outgoing(direction)
    hc = _divisor_at_most(h, BUDGET_BYTES // (4 * m * x.element_size()))
    batches = x.reshape(-1, n, n, c)
    masks = None if mask is None else mask.expand(x.shape[:-1]).reshape(-1, m)
    result = torch.empty_like(batches, memory_format=torch.contiguous_format)
    for batch_index, batch in enumerate(batches):
        zn = input_norm(
            batch.reshape(m, c),
            p.norm_in_weight,
            p.norm_in_bias,
            eps,
        )
        mask_row = None if masks is None else masks[batch_index].to(x.dtype)
        contraction = torch.empty(h, n, n, dtype=x.dtype, device=x.device)
        for start in range(0, h, hc):
            indices = (slice(start, start + hc), slice(h + start, h + start + hc))
            weights = torch.cat(
                [
                    p.p_in_weight[indices[0]],
                    p.p_in_weight[indices[1]],
                    p.g_in_weight[indices[0]],
                    p.g_in_weight[indices[1]],
                ]
            )
            pg = torch.mm(weights, zn.t())
            if p.p_in_bias is not None or p.g_in_bias is not None:
                zeros = x.new_zeros(2 * h)
                pb = zeros if p.p_in_bias is None else p.p_in_bias
                gb = zeros if p.g_in_bias is None else p.g_in_bias
                bias = torch.cat(
                    [pb[indices[0]], pb[indices[1]], gb[indices[0]], gb[indices[1]]]
                )
                pg.add_(bias[:, None])
            ab = pg[: 2 * hc].mul_(pg[2 * hc :].sigmoid_())
            if mask_row is not None:
                ab.mul_(mask_row)
            a, b = ab[:hc].reshape(hc, n, n), ab[hc:].reshape(hc, n, n)
            torch.bmm(
                a if outgoing else a.transpose(1, 2),
                b.transpose(1, 2) if outgoing else b,
                out=contraction[start : start + hc],
            )
        xf = contraction.reshape(h, m)
        rows = min(m, LN_MAX_ELEMS // max(c, h), BUDGET_BYTES // (4 * (2 * h + 3 * c)))
        if rows < 1:
            raise ValueError("Budget cannot hold one output row")
        output = result[batch_index].reshape(m, c)
        for start in range(0, m, rows):
            xc = xf[:, start : start + rows]
            if channel_major_norm is None:
                var, mean = torch.var_mean(xc, dim=0, correction=0, keepdim=True)
                xn = (xc - mean) * torch.rsqrt(var + eps)
                if p.norm_out_weight is not None:
                    xn.mul_(p.norm_out_weight[:, None])
                if p.norm_out_bias is not None:
                    xn.add_(p.norm_out_bias[:, None])
            else:
                xn = channel_major_norm(xc, p.norm_out_weight, p.norm_out_bias, eps)
            projected = torch.mm(xn.t(), p.p_out_weight.t())
            gate = torch.mm(zn[start : start + rows], p.g_out_weight.t())
            if p.p_out_bias is not None:
                projected.add_(p.p_out_bias)
            if p.g_out_bias is not None:
                gate.add_(p.g_out_bias)
            output[start : start + rows] = projected.mul_(gate.sigmoid_())
    return result.reshape(x.shape)
