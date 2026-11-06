# Copyright 2025 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Historical kernels; provenance and numerical limitations in README.md."""

import torch
import triton
import triton.language as tl


def _dual(gate_input, projection_input, wg, wp, bg, bp, mask, kernel):
    m, k = gate_input.shape
    if projection_input.shape != (m, k):
        raise ValueError("Historical dual-input kernel requires equal input widths")
    result = torch.empty(
        m, wp.shape[0], device=gate_input.device, dtype=gate_input.dtype
    )
    wg, wp = wg.t(), wp.t()
    kernel[
        lambda meta: (
            1,
            triton.cdiv(m, meta["BLOCK_M"]),
            triton.cdiv(result.shape[1], meta["BLOCK_N"]),
        )
    ](
        gate_input,
        wg,
        wp,
        result,
        bg,
        bp,
        mask,
        projection_input,
        0,
        *gate_input.stride(),
        *wg.stride(),
        *wp.stride(),
        0,
        *result.stride(),
        0,
        *projection_input.stride(),
        m,
        k,
        result.shape[1],
    )
    return result


def update(x, z, direction, mask, eps, p):
    from geometric_nn_ops_torch.triangle._reference.trimul import (
        _combine_projections,
        is_outgoing,
    )

    from ._pipeline import norm

    c = x.shape[-1]
    if p.hidden != c:
        raise ValueError("Historical unified kernel requires C_hidden == C")
    zf = z.reshape(-1, c).contiguous()
    mf = None if mask is None else mask.expand(x.shape[:-1]).reshape(-1).contiguous()
    parts = []
    for index in range(2):
        rows = slice(index * c, (index + 1) * c)
        parts.append(
            _dual(
                zf,
                zf,
                p.g_in_weight[rows],
                p.p_in_weight[rows],
                None if p.g_in_bias is None else p.g_in_bias[rows],
                None if p.p_in_bias is None else p.p_in_bias[rows],
                mf,
                fused_gemm_sigmoid_multiply,
            ).reshape(x.shape)
        )
    o = _combine_projections(*parts, is_outgoing(direction))
    on = norm(o, p.norm_out_weight, p.norm_out_bias, eps).reshape(-1, c).contiguous()
    return _dual(
        zf,
        on,
        p.g_out_weight,
        p.p_out_weight,
        p.g_out_bias,
        p.p_out_bias,
        None,
        fused_gemm_sigmoid_multiply,
    ).reshape(x.shape)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=2
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=2
        ),
    ],
    key=["M", "K", "N"],
)
@triton.jit
def fused_gemm_sigmoid_multiply(
    A_ptr,
    W_g_ptr,
    W_p_ptr,  # Input and weight matrices for gate and proj
    C_ptr,
    bias_g_ptr,
    bias_p_ptr,
    mask_ptr,
    # Optional second input for proj GEMM
    A_p_ptr,  # If provided, use this for proj GEMM instead of A_ptr
    stride_Ab,
    stride_Ai,
    stride_Ak,
    stride_Wgk,
    stride_Wgj,
    stride_Wpk,
    stride_Wpj,
    stride_Cb,
    stride_Ci,
    stride_Cj,
    # Strides for A_p (only used if A_p_ptr is not None)
    stride_Apb,
    stride_Api,
    stride_Apk,
    M,
    K,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel that computes:
    gate = A @ W_g.T + bias_g
    proj = (A_p if A_p_ptr else A) @ W_p.T + bias_p
    output = sigmoid(gate) * proj * mask
    """
    b = tl.program_id(0)  # noqa: F841 - retained historical, unused batch index

    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulators for both gate and proj
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_p = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Compute both matrix multiplications in the same loop
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A tile for gate computation
        A = tl.load(
            A_ptr + offs_m[:, None] * stride_Ai + offs_k[None, :] * stride_Ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        # Load A_p tile for proj computation if provided, else use A
        if A_p_ptr is not None:
            A_p = tl.load(
                A_p_ptr + offs_m[:, None] * stride_Api + offs_k[None, :] * stride_Apk,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
        else:
            A_p = A

        # Load W_g tile
        W_g = tl.load(
            W_g_ptr + offs_k[:, None] * stride_Wgk + offs_n[None, :] * stride_Wgj,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        # Load W_p tile
        W_p = tl.load(
            W_p_ptr + offs_k[:, None] * stride_Wpk + offs_n[None, :] * stride_Wpj,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        # Accumulate both matrix multiplications
        acc_g += tl.dot(A.to(tl.float32), W_g.to(tl.float32))
        acc_p += tl.dot(A_p.to(tl.float32), W_p.to(tl.float32))

    # Add biases if provided
    if bias_g_ptr is not None:
        bias_g = tl.load(bias_g_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc_g = acc_g + bias_g[None, :]

    if bias_p_ptr is not None:
        bias_p = tl.load(bias_p_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc_p = acc_p + bias_p[None, :]

    # Apply sigmoid to gate
    gate_sigmoid = 1.0 / (1.0 + tl.exp(-acc_g))

    # Multiply sigmoid(gate) * proj
    result = gate_sigmoid * acc_p

    # Apply mask if provided
    if mask_ptr is not None:
        mask = tl.load(
            mask_ptr + offs_m,  # Just use offs_m directly
            mask=offs_m < M,
            other=0.0,
        )
        result = result * mask[:, None]

    # Store result
    tl.store(
        C_ptr + offs_m[:, None] * stride_Ci + offs_n[None, :] * stride_Cj,
        result,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )
