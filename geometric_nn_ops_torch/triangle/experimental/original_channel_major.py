# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Functional adapter for the initial forward kernel collection."""

import torch
import triton

from geometric_nn_ops_torch.triangle._reference.trimul import is_outgoing

from . import initial_kernels as kernels
from .dual_projection_norm import row_norm


def original_norm(x, weight, bias, eps):
    x = x.contiguous()
    m, c = x.shape
    weight = x.new_ones(c) if weight is None else weight
    bias = x.new_zeros(c) if bias is None else bias
    output = torch.empty_like(x)
    kernels._layer_norm_2d_fwd_fused[
        lambda meta: (triton.cdiv(m, meta["BLOCK_ROWS"]),)
    ](x, output, weight, bias, x.stride(0), m, c, eps)
    return output


def update(
    x,
    direction,
    mask,
    eps,
    p,
    *,
    register_norm=False,
    normalize=None,
    combine=None,
    pack_weights=True,
    producer_kernel=None,
    output_kernel=None,
):
    if torch.get_float32_matmul_precision() != "highest":
        raise ValueError("Original channel-major comparison requires highest/IEEE")
    c, n = x.shape[-1], x.shape[-2]
    if p.hidden != c:
        raise ValueError("Original channel-major adapter requires C_hidden == C")
    if register_norm and c > 4096:
        raise ValueError("Register normalization supports at most 4096 channels")
    if normalize is None:
        normalize = row_norm if register_norm else original_norm
    if combine is None:
        combine = torch.bmm
    if producer_kernel is None:
        producer_kernel = kernels.fused_gemm_sigmoid_multiply_channel_major
    if output_kernel is None:
        output_kernel = kernels.fused_gemm_sigmoid_multiply

    def weight_view(weight):
        transposed = weight.t()
        return transposed.contiguous() if pack_weights else transposed

    z = normalize(x.reshape(-1, c), p.norm_in_weight, p.norm_in_bias, eps)
    m = z.shape[0]
    batches = m // (n * n)
    mf = None if mask is None else mask.expand(x.shape[:-1]).reshape(-1).contiguous()
    parts = []
    for index in range(2):
        rows = slice(index * c, (index + 1) * c)
        wg = weight_view(p.g_in_weight[rows])
        wp = weight_view(p.p_in_weight[rows])
        out = x.new_empty(batches * c, n, n)
        producer_kernel[
            lambda meta: (
                1,
                triton.cdiv(m, meta["BLOCK_M"]),
                triton.cdiv(c, meta["BLOCK_N"]),
            )
        ](
            z,
            wg,
            wp,
            out,
            None if p.g_in_bias is None else p.g_in_bias[rows],
            None if p.p_in_bias is None else p.p_in_bias[rows],
            mf,
            None,
            0,
            *z.stride(),
            *wg.stride(),
            *wp.stride(),
            *out.stride(),
            0,
            0,
            0,
            m,
            c,
            c,
            batches,
            n,
        )
        parts.append(out)
    a, b = parts
    combined = combine(
        a if is_outgoing(direction) else a.transpose(-1, -2),
        b.transpose(-1, -2) if is_outgoing(direction) else b,
    )
    channel_last = combined.reshape(batches, c, n * n).transpose(1, 2)
    on = normalize(channel_last.reshape(m, c), p.norm_out_weight, p.norm_out_bias, eps)
    wg, wp = weight_view(p.g_out_weight), weight_view(p.p_out_weight)
    output = x.new_empty(m, c)
    output_kernel[
        lambda meta: (
            1,
            triton.cdiv(m, meta["BLOCK_M"]),
            triton.cdiv(c, meta["BLOCK_N"]),
        )
    ](
        z,
        wg,
        wp,
        output,
        p.g_out_bias,
        p.p_out_bias,
        None,
        on,
        0,
        *z.stride(),
        *wg.stride(),
        *wp.stride(),
        0,
        *output.stride(),
        0,
        *on.stride(),
        m,
        c,
        c,
    )
    return output.reshape(x.shape)
