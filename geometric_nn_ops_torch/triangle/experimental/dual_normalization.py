# Copyright 2025 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Historical kernels; provenance and numerical limitations in README.md."""

import torch
import triton
import triton.language as tl


def output(o, x, p, eps):
    c, h = x.shape[-1], p.hidden
    xf, of = x.reshape(-1, c).contiguous(), o.reshape(-1, h).contiguous()
    result = torch.empty_like(xf)
    wz, wg = p.p_out_weight.t(), p.g_out_weight.t()
    lw = x.new_ones(h) if p.norm_out_weight is None else p.norm_out_weight
    lb = x.new_zeros(h) if p.norm_out_bias is None else p.norm_out_bias
    iw = x.new_ones(c) if p.norm_in_weight is None else p.norm_in_weight
    ib = x.new_zeros(c) if p.norm_in_bias is None else p.norm_in_bias
    # Historical BLOCK_K=4 configuration cannot normalize complete channel rows.
    block_k = max(16, triton.next_power_of_2(max(c, h)))
    fused_dual_layernorm_gemm_sigmoid_multiply_kernel.fn[
        (triton.cdiv(xf.shape[0], 32), triton.cdiv(c, 32))
    ](
        of,
        xf,
        wz,
        wg,
        lw,
        lb,
        iw,
        ib,
        result,
        p.p_out_bias,
        p.g_out_bias,
        *of.stride(),
        *xf.stride(),
        *wz.stride(),
        *wg.stride(),
        *result.stride(),
        xf.shape[0],
        h,
        c,
        c,
        eps,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=1,
    )
    return result.reshape(x.shape)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 4}, num_stages=3, num_warps=8
        ),
    ],
    key=["M", "K_x", "K_z", "N"],
)
@triton.jit
def fused_dual_layernorm_gemm_sigmoid_multiply_kernel(
    # Input tensors
    X_ptr,  # [M, K_x], K_x = c_hidden: layer_norm_out + linear_z
    Z_ptr,  # [M, K_z], K_z = c_z: layer_norm_in + linear_g
    # Weight matrices
    W_z_ptr,  # linear_z weight [K_x, N] where K_x = c_hidden, N = c_z
    W_g_ptr,  # linear_g weight [K_z, N] where K_z = c_z, N = c_z
    # LayerNorm parameters for X (layer_norm_out)
    ln_out_weight_ptr,
    ln_out_bias_ptr,
    # LayerNorm parameters for Z (layer_norm_in)
    ln_in_weight_ptr,
    ln_in_bias_ptr,
    # Output
    C_ptr,  # output [M, N]
    # Bias (optional)
    bias_z_ptr,
    bias_g_ptr,
    # Strides
    stride_Xi,
    stride_Xk,
    stride_Zi,
    stride_Zk,
    stride_Wzk,
    stride_Wzn,
    stride_Wgk,
    stride_Wgn,
    stride_Ci,
    stride_Cn,
    # Dimensions
    M,
    K_x,
    K_z,
    N,  # K_x = c_hidden, K_z = c_z, N = c_z
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel with REGISTER CACHING that computes:
    1. x_normed = LayerNorm_out(X)  (layer_norm_out on x_chunk)
    2. proj = x_normed @ W_z.T + bias_z  (linear_z)
    3. z_normed = LayerNorm_in(Z)  (layer_norm_in on z_chunk_g)
    4. gate = z_normed @ W_g.T + bias_g  (linear_g)
    5. gate_sigmoid = sigmoid(gate)
    6. output = gate_sigmoid * proj

    OPTIMIZATION: Single-pass register caching
    - Load X and Z data ONCE into registers
    - Compute LayerNorm statistics from cached data
    - Apply normalization and GEMM using same cached data
    - Eliminates second global memory read (50% memory traffic reduction)

    Constraint: K_x and K_z must fit in BLOCK_K for single-pass operation.
    For c_hidden=128 and c_z=128, use BLOCK_K=128.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # ========== SINGLE LOAD: Cache X in registers ==========
    X_cached = tl.load(
        X_ptr + offs_m[:, None] * stride_Xi + offs_k[None, :] * stride_Xk,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K_x),
        other=0.0,
    )

    # ========== SINGLE LOAD: Cache Z in registers ==========
    Z_cached = tl.load(
        Z_ptr + offs_m[:, None] * stride_Zi + offs_k[None, :] * stride_Zk,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K_z),
        other=0.0,
    )

    # ========== LayerNorm for X using cached data ==========
    valid_mask_x = offs_k[None, :] < K_x
    n_valid_x = tl.sum(valid_mask_x.to(tl.float32), axis=1)

    X_fp32 = X_cached.to(tl.float32)
    row_sum_x = tl.sum(tl.where(valid_mask_x, X_fp32, 0.0), axis=1)
    mean_x = tl.where(n_valid_x > 0, row_sum_x / n_valid_x, 0.0)

    diff_x = tl.where(valid_mask_x, X_fp32 - mean_x[:, None], 0.0)
    var_x = tl.sum(diff_x * diff_x, axis=1) / tl.where(n_valid_x > 0, n_valid_x, 1.0)
    rstd_x = 1.0 / tl.sqrt(var_x + eps)

    # ========== LayerNorm for Z using cached data ==========
    valid_mask_z = offs_k[None, :] < K_z
    n_valid_z = tl.sum(valid_mask_z.to(tl.float32), axis=1)

    Z_fp32 = Z_cached.to(tl.float32)
    row_sum_z = tl.sum(tl.where(valid_mask_z, Z_fp32, 0.0), axis=1)
    mean_z = tl.where(n_valid_z > 0, row_sum_z / n_valid_z, 0.0)

    diff_z = tl.where(valid_mask_z, Z_fp32 - mean_z[:, None], 0.0)
    var_z = tl.sum(diff_z * diff_z, axis=1) / tl.where(n_valid_z > 0, n_valid_z, 1.0)
    rstd_z = 1.0 / tl.sqrt(var_z + eps)

    # ========== Load LayerNorm parameters ==========
    ln_out_w = tl.load(ln_out_weight_ptr + offs_k, mask=offs_k < K_x, other=1.0)
    ln_out_b = tl.load(ln_out_bias_ptr + offs_k, mask=offs_k < K_x, other=0.0)
    ln_in_w = tl.load(ln_in_weight_ptr + offs_k, mask=offs_k < K_z, other=1.0)
    ln_in_b = tl.load(ln_in_bias_ptr + offs_k, mask=offs_k < K_z, other=0.0)

    # ========== Apply LayerNorm to cached X (NO second global read!) ==========
    X_normed = (X_fp32 - mean_x[:, None]) * rstd_x[:, None] * ln_out_w.to(tl.float32)[
        None, :
    ] + ln_out_b.to(tl.float32)[None, :]

    # ========== Apply LayerNorm to cached Z (NO second global read!) ==========
    Z_normed = (Z_fp32 - mean_z[:, None]) * rstd_z[:, None] * ln_in_w.to(tl.float32)[
        None, :
    ] + ln_in_b.to(tl.float32)[None, :]

    # ========== Load weight matrices ==========
    W_z = tl.load(
        W_z_ptr + offs_k[:, None] * stride_Wzk + offs_n[None, :] * stride_Wzn,
        mask=(offs_k[:, None] < K_x) & (offs_n[None, :] < N),
        other=0.0,
    )

    W_g = tl.load(
        W_g_ptr + offs_k[:, None] * stride_Wgk + offs_n[None, :] * stride_Wgn,
        mask=(offs_k[:, None] < K_z) & (offs_n[None, :] < N),
        other=0.0,
    )

    # ========== GEMM: X_normed @ W_z ==========
    acc_proj = tl.dot(X_normed.to(W_z.dtype), W_z.to(X_normed.dtype))

    # ========== GEMM: Z_normed @ W_g ==========
    acc_gate = tl.dot(Z_normed.to(W_g.dtype), W_g.to(Z_normed.dtype))

    # Add biases if provided
    if bias_z_ptr is not None:
        bias_z = tl.load(bias_z_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc_proj = acc_proj.to(tl.float32) + bias_z.to(tl.float32)[None, :]

    if bias_g_ptr is not None:
        bias_g = tl.load(bias_g_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc_gate = acc_gate.to(tl.float32) + bias_g.to(tl.float32)[None, :]

    # Apply sigmoid to gate and multiply with proj
    gate_sigmoid = tl.sigmoid(acc_gate)
    result = gate_sigmoid * acc_proj

    # Store result
    tl.store(
        C_ptr + offs_m[:, None] * stride_Ci + offs_n[None, :] * stride_Cn,
        result.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )
