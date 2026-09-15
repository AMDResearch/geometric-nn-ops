# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Triton building-block kernels for triangle_multiplicative_update.

These are fused layernorm / linear (+ sigmoid / mul / mask / add) primitives;
depend only on ``torch`` + ``triton``. Forward-only (no autograd).
"""

import torch

from geometric_nn_ops_torch import _precision

# For now we reuse this from the pure torch implementation.
from geometric_nn_ops_torch.triangle._reference.trimul import (
    _combine_projections,
    is_outgoing,
)

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:

    @triton.jit
    def layernorm_kernel(
        x_ptr,
        output_ptr,
        weight_ptr,
        bias_ptr,
        M,
        N,
        eps,
        BLOCK_SIZE: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
    ):
        """
        Vectorized LayerNorm kernel with multi-row processing and safe bounds checking.
        Each program processes ROWS_PER_PROGRAM rows to amortize launch overhead.
        Uses vectorized tl.sum reductions instead of scalar Welford loop.
        Normalizes over the last dimension (N).
        """
        row_start = tl.program_id(0) * ROWS_PER_PROGRAM

        # Early exit if out of bounds
        if row_start >= M:
            return

        for row in range(ROWS_PER_PROGRAM):
            row_idx = row_start + row
            if row_idx < M:
                row_offset = row_idx.to(tl.int64) * N
                x_row_ptr = x_ptr + row_offset
                out_row_ptr = output_ptr + row_offset

                # Pass 1: vectorized mean and variance computation
                sum_val = 0.0
                sum_sq = 0.0
                for block_start in range(0, N, BLOCK_SIZE):
                    offs = block_start + tl.arange(0, BLOCK_SIZE)
                    mask = (offs >= 0) & (offs < N)
                    safe_offs = offs.to(tl.int64)
                    x_block = tl.load(x_row_ptr + safe_offs, mask=mask, other=0.0).to(
                        tl.float32
                    )
                    sum_val += tl.sum(x_block, axis=0)
                    sum_sq += tl.sum(x_block * x_block, axis=0)

                mean = sum_val / N
                var = sum_sq / N - mean * mean
                rstd = 1.0 / tl.sqrt(tl.maximum(var, eps))

                # Pass 2: normalize and apply affine transform
                for block_start in range(0, N, BLOCK_SIZE):
                    offs = block_start + tl.arange(0, BLOCK_SIZE)
                    mask = (offs >= 0) & (offs < N)

                    safe_offs = offs.to(tl.int64)
                    x_vals = tl.load(x_row_ptr + safe_offs, mask=mask, other=0.0).to(
                        tl.float32
                    )
                    w_vals = tl.load(weight_ptr + safe_offs, mask=mask, other=1.0).to(
                        tl.float32
                    )
                    b_vals = tl.load(bias_ptr + safe_offs, mask=mask, other=0.0).to(
                        tl.float32
                    )

                    out = (x_vals - mean) * rstd * w_vals + b_vals
                    tl.store(out_row_ptr + safe_offs, out, mask=mask)

    @triton.jit
    def linear_fused_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        other_ptr,
        mask_ptr,
        add_tensor_ptr,
        output_ptr,
        M,
        K,
        N,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        stride_other_m,
        stride_other_n,
        stride_mask_m,
        stride_mask_n,
        stride_add_m,
        stride_add_n,
        HAS_BIAS: tl.constexpr,
        APPLY_SIGMOID: tl.constexpr,
        APPLY_MUL: tl.constexpr,
        HAS_MASK: tl.constexpr,
        HAS_ADD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        MATMUL_PRECISION: tl.constexpr,
    ):
        """
        Fused linear layer with optional sigmoid, elementwise multiply, mask, and add
        Supports combinations: linear [+ sigmoid] [* other] [* mask] [+ add_tensor]
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        m_mask = rm < M
        n_mask = rn < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            k_mask = rk < K

            x_offsets = (
                rm[:, None].to(tl.int64) * stride_xm
                + rk[None, :].to(tl.int64) * stride_xk
            )

            w_offsets = (
                rn[None, :].to(tl.int64) * stride_wn
                + rk[:, None].to(tl.int64) * stride_wk
            )

            x = tl.load(
                x_ptr + x_offsets,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            w = tl.load(
                w_ptr + w_offsets,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            acc = tl.dot(x, w, acc, input_precision=MATMUL_PRECISION)

        if HAS_BIAS:
            bias_offsets = rn.to(tl.int64)
            bias = tl.load(bias_ptr + bias_offsets, mask=n_mask, other=0.0)
            acc += bias[None, :]

        if APPLY_SIGMOID:
            # Clamp to avoid exp overflow
            acc = tl.where(acc > 20.0, 20.0, acc)
            acc = tl.where(acc < -20.0, -20.0, acc)
            acc = 1.0 / (1.0 + tl.exp(-acc))

        if APPLY_MUL:
            other_offsets = (
                rm[:, None].to(tl.int64) * stride_other_m
                + rn[None, :].to(tl.int64) * stride_other_n
            )
            other_vals = tl.load(
                other_ptr + other_offsets,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc = acc * other_vals

        if HAS_MASK:
            mask_offsets = (
                rm[:, None].to(tl.int64) * stride_mask_m
                + rn[None, :].to(tl.int64) * stride_mask_n
            )
            mask_vals = tl.load(
                mask_ptr + mask_offsets,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc = acc * mask_vals

        if HAS_ADD:
            add_offsets = (
                rm[:, None].to(tl.int64) * stride_add_m
                + rn[None, :].to(tl.int64) * stride_add_n
            )
            add_vals = tl.load(
                add_tensor_ptr + add_offsets,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc = acc + add_vals

        out_offsets = (
            rm[:, None].to(tl.int64) * stride_om + rn[None, :].to(tl.int64) * stride_on
        )

        out_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(
            output_ptr + out_offsets,
            acc,
            mask=out_mask,
        )


def triton_layernorm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    Triton LayerNorm with vectorized reductions and multi-row processing.

    Args:
        x: Input tensor [..., N]
        weight: Scale parameters [N]
        bias: Shift parameters [N]
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor with same shape as x
    """
    input_shape = x.shape
    N = input_shape[-1]
    M = x.numel() // N

    x_2d = x.reshape(M, N).contiguous()
    output_2d = torch.empty_like(x_2d)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
    ROWS_PER_PROGRAM = 8
    grid = (triton.cdiv(M, ROWS_PER_PROGRAM),)

    layernorm_kernel[grid](
        x_2d,
        output_2d,
        weight,
        bias,
        M,
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
    )

    return output_2d.reshape(input_shape)


def triton_linear_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    matmul_precision: str,
    bias: torch.Tensor | None = None,
    other: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    add_tensor: torch.Tensor | None = None,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    """
    Fused linear layer with optional sigmoid, elementwise multiply, mask, and add.

    Computes: linear(x) [* sigmoid] [* other] [* mask] [+ add_tensor]

    Args:
        x: Input tensor [..., K]
        weight: Weight tensor [N, K] (same layout as nn.Linear.weight)
        matmul_precision: tl.dot mode to use.
        bias: Optional bias tensor [N]
        other: Optional tensor to multiply with (same shape as output)
        mask: Optional mask tensor (broadcastable with output)
        add_tensor: Optional tensor to add to result (same shape as output)
        apply_sigmoid: Whether to apply sigmoid to linear result

    Returns:
        Output tensor [..., N]
    """
    input_shape = x.shape
    K = input_shape[-1]
    M = x.numel() // K
    N = weight.shape[0]

    x_2d = x.reshape(M, K).contiguous()
    output_2d = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_M = min(128, triton.next_power_of_2(M))
    BLOCK_N = min(64, triton.next_power_of_2(N))
    BLOCK_K = min(64, triton.next_power_of_2(K))

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    if other is not None:
        # other should match the output shape [..., N], so reshape accordingly
        other_M = other.numel() // N
        other_2d = other.reshape(other_M, N).contiguous()
        if other_M != M:
            raise ValueError(
                f"Shape mismatch: tensor expects {other_M} rows but output has {M}"
            )
        other_ptr = other_2d
        other_stride_m, other_stride_n = other_2d.stride(0), other_2d.stride(1)
    else:
        other_ptr = x_2d  # guarded by APPLY_MUL constexpr
        other_stride_m, other_stride_n = 0, 0

    if mask is not None:
        # mask should broadcast to output shape [..., N]
        mask_M = mask.numel() // (mask.shape[-1] if mask.numel() > 0 else 1)
        mask_features = mask.shape[-1] if mask.ndim > 0 else 1
        if mask_features == 1:
            # Broadcast mask to match output
            mask_2d = mask.reshape(mask_M, 1).expand(M, N).contiguous()
        else:
            mask_2d = mask.reshape(mask_M, mask_features).contiguous()
            if mask_M != M or mask_features != N:
                raise ValueError(
                    f"Mask shape mismatch: got {mask_M}x{mask_features}, "
                    f"expected {M}x{N}"
                )
        mask_ptr = mask_2d
        mask_stride_m, mask_stride_n = mask_2d.stride(0), mask_2d.stride(1)
    else:
        mask_ptr = x_2d  # guarded by HAS_MASK constexpr
        mask_stride_m, mask_stride_n = 0, 0

    if add_tensor is not None:
        # add_tensor should match the output shape [..., N]
        add_M = add_tensor.numel() // N
        add_tensor_2d = add_tensor.reshape(add_M, N).contiguous()
        if add_M != M:
            raise ValueError(
                f"Shape mismatch: add_tensor expects {add_M} rows but output has {M}"
            )
        add_tensor_ptr = add_tensor_2d
        add_stride_m, add_stride_n = add_tensor_2d.stride(0), add_tensor_2d.stride(1)
    else:
        add_tensor_ptr = x_2d  # guarded by HAS_ADD constexpr
        add_stride_m, add_stride_n = 0, 0

    bias_ptr = bias if bias is not None else x_2d  # guarded by HAS_BIAS constexpr

    linear_fused_kernel[grid](
        x_2d,
        weight,
        bias_ptr,
        other_ptr,
        mask_ptr,
        add_tensor_ptr,
        output_2d,
        M,
        K,
        N,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        other_stride_m,
        other_stride_n,
        mask_stride_m,
        mask_stride_n,
        add_stride_m,
        add_stride_n,
        HAS_BIAS=(bias is not None),
        APPLY_SIGMOID=apply_sigmoid,
        APPLY_MUL=(other is not None),
        HAS_MASK=(mask is not None),
        HAS_ADD=(add_tensor is not None),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        MATMUL_PRECISION=matmul_precision,
    )

    output_shape = input_shape[:-1] + (N,)
    return output_2d.reshape(output_shape)


def trimul_triton(
    x,
    direction="outgoing",
    mask=None,
    *,
    eps=1e-5,
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
):
    """Triangle multiplicative update using fused Triton kernels. Forward only.

    Takes the same arguments as ``trimul_torch``; see it for the layouts.
    """
    outgoing = is_outgoing(direction)
    c_hidden = p_in_weight.shape[0] // 2
    m = None
    if mask is not None:
        # The kernels index one entry per pair row, so we need to make the
        # broadcast explicit.
        m = mask.unsqueeze(-1).expand(*x.shape[:-1], 1)
    precision = _precision.matmul_precision(fp32_operands=x.dtype == torch.float32)
    z = triton_layernorm(x, norm_in_weight, norm_in_bias, eps)
    pg = triton_linear_fused(
        z, g_in_weight, precision, bias=g_in_bias, apply_sigmoid=True
    )
    p = triton_linear_fused(z, p_in_weight, precision, bias=p_in_bias, other=pg, mask=m)
    a = p[..., :c_hidden]
    b = p[..., c_hidden:]
    o = _combine_projections(a, b, outgoing)
    o = triton_layernorm(o, norm_out_weight, norm_out_bias, eps)
    o = triton_linear_fused(o, p_out_weight, precision, bias=p_out_bias)
    g = triton_linear_fused(
        z, g_out_weight, precision, bias=g_out_bias, apply_sigmoid=True
    )
    return o * g
