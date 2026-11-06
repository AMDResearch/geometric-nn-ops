# Copyright 2026 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Historical kernels; provenance and numerical limitations in README.md."""

import torch
import triton
import triton.language as tl


@triton.jit
def dual_linear_sigmoid_mul_kernel(
    x_ptr,
    wg_ptr,
    bg_ptr,
    wp_ptr,
    bp_ptr,
    mask_ptr,
    output_ptr,
    M,
    K,
    N,
    stride_xm,
    stride_xk,
    stride_wgn,
    stride_wgk,
    stride_wpn,
    stride_wpk,
    stride_om,
    stride_on,
    HAS_BIAS_G: tl.constexpr,
    HAS_BIAS_P: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_BF16: tl.constexpr = True,
):
    """
    Fully fused: output = sigmoid(x @ Wg.T [+ bg]) * (x @ Wp.T [+ bp]) [* mask]
    Both GEMMs share x — each x tile is loaded once from GMEM.
    Two accumulators stay in registers through the K-loop.
    Sigmoid * multiply (and optional mask) applied in epilogue;
    no intermediate GMEM writes.
    mask is [M, 1] — one scalar per row, broadcast across N columns.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if not USE_BF16:
        rm = rm.to(tl.int64)
        rn = rn.to(tl.int64)

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_p = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        if not USE_BF16:
            rk = rk.to(tl.int64)

        # Load the shared input once for both projections.
        x = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )

        # Load Wg tile [BLOCK_K, BLOCK_N] (transposed from [N, K])
        wg = tl.load(
            wg_ptr + rn[None, :] * stride_wgn + rk[:, None] * stride_wgk,
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )

        # Load Wp tile [BLOCK_K, BLOCK_N]
        wp = tl.load(
            wp_ptr + rn[None, :] * stride_wpn + rk[:, None] * stride_wpk,
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )

        if USE_BF16:
            x = x.to(tl.bfloat16)
            wg = wg.to(tl.bfloat16)
            wp = wp.to(tl.bfloat16)
            acc_g = tl.dot(x, wg, acc_g)
            acc_p = tl.dot(x, wp, acc_p)
        else:
            acc_g = tl.dot(x, wg, acc_g, input_precision="ieee")
            acc_p = tl.dot(x, wp, acc_p, input_precision="ieee")

    # Add biases (guarded by constexpr flags for layers without bias)
    n_mask = rn < N
    if HAS_BIAS_G:
        bg = tl.load(bg_ptr + rn, mask=n_mask, other=0.0)
        acc_g += bg[None, :]
    if HAS_BIAS_P:
        bp = tl.load(bp_ptr + rn, mask=n_mask, other=0.0)
        acc_p += bp[None, :]

    # Fused epilogue in registers: sigmoid(gate) * projection [* mask]
    result = (1.0 / (1.0 + tl.exp(-acc_g))) * acc_p

    if HAS_MASK:
        # mask is [M, 1] — one value per row, broadcast across columns
        m_vals = tl.load(mask_ptr + rm, mask=rm < M, other=0.0)
        result *= m_vals[:, None]

    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(
        output_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on,
        result,
        mask=out_mask,
    )


def triton_dual_linear_sigmoid_mul(
    x: torch.Tensor,
    weight_g: torch.Tensor,
    bias_g: torch.Tensor,
    weight_p: torch.Tensor,
    bias_p: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    use_bf16: bool = True,
) -> torch.Tensor:
    """
    Fully fused: sigmoid(x @ Wg.T + bg) * (x @ Wp.T + bp) [* mask]
    Single kernel — x loaded once, both GEMMs computed with dual accumulators,
    sigmoid * multiply (and optional row-wise mask) in epilogue.

    Args:
        x: Input tensor [..., K]
        weight_g: Gate weight [N, K]
        bias_g: Gate bias [N]
        weight_p: Projection weight [N, K]
        bias_p: Projection bias [N]
        mask: Optional mask [..., 1] — one value per row, broadcast across N

    Returns:
        Output tensor [..., N]
    """
    input_shape = x.shape
    K = input_shape[-1]
    M = x.numel() // K
    N = weight_g.shape[0]

    x_2d = x.reshape(M, K).contiguous()
    output_2d = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_M = 32 if use_bf16 else 64
    BLOCK_N = min(128, triton.next_power_of_2(N))
    BLOCK_K = min(128 if use_bf16 else 32, triton.next_power_of_2(K))
    if not use_bf16:
        BLOCK_N = max(16, BLOCK_N)
        BLOCK_K = max(16, BLOCK_K)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    has_bias_g = bias_g is not None
    has_bias_p = bias_p is not None
    has_mask = mask is not None

    # Dummy pointers for None args, guarded by constexpr flags
    bg_ptr = bias_g if has_bias_g else weight_g
    bp_ptr = bias_p if has_bias_p else weight_p
    if has_mask:
        mask_flat = mask.reshape(M).contiguous()
    else:
        mask_flat = weight_g

    dual_linear_sigmoid_mul_kernel[grid](
        x_2d,
        weight_g,
        bg_ptr,
        weight_p,
        bp_ptr,
        mask_flat,
        output_2d,
        M,
        K,
        N,
        x_2d.stride(0),
        x_2d.stride(1),
        weight_g.stride(0),
        weight_g.stride(1),
        weight_p.stride(0),
        weight_p.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        HAS_BIAS_G=has_bias_g,
        HAS_BIAS_P=has_bias_p,
        HAS_MASK=has_mask,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        USE_BF16=use_bf16,
        num_stages=2 if use_bf16 else 1,
    )

    output_shape = input_shape[:-1] + (N,)
    return output_2d.reshape(output_shape)


@triton.jit
def batched_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    B,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Batched matmul: C[b] = A[b] @ B[b]
    A: [B, M, K], B: [B, K, N], C: [B, M, N]
    Uses tl.dot for hardware-accelerated matmul (MFMA on AMD).
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)

        a = tl.load(
            a_ptr
            + pid_b * stride_ab
            + rm[:, None] * stride_am
            + rk[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        ).to(tl.bfloat16)

        b = tl.load(
            b_ptr
            + pid_b * stride_bb
            + rk[:, None] * stride_bk
            + rn[None, :] * stride_bn,
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        ).to(tl.bfloat16)

        acc = tl.dot(a, b, acc)

    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(
        c_ptr + pid_b * stride_cb + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc,
        mask=out_mask,
    )


def triton_batched_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Batched matmul: C = A @ B for tensors with arbitrary leading batch dims.
    Equivalent to torch.einsum("...ij,...jk->...ik", a, b).

    Args:
        a: [..., M, K]
        b: [..., K, N]

    Returns:
        [..., M, N]
    """
    batch_shape = a.shape[:-2]
    M, K = a.shape[-2], a.shape[-1]
    N = b.shape[-1]

    B = 1
    for s in batch_shape:
        B *= s

    a_3d = a.reshape(B, M, K).contiguous()
    b_3d = b.reshape(B, K, N).contiguous()
    c_3d = torch.empty((B, M, N), dtype=a.dtype, device=a.device)

    BLOCK_M = 32
    BLOCK_N = min(128, triton.next_power_of_2(N))
    BLOCK_K = min(128, triton.next_power_of_2(K))

    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    batched_matmul_kernel[grid](
        a_3d,
        b_3d,
        c_3d,
        B,
        M,
        N,
        K,
        a_3d.stride(0),
        a_3d.stride(1),
        a_3d.stride(2),
        b_3d.stride(0),
        b_3d.stride(1),
        b_3d.stride(2),
        c_3d.stride(0),
        c_3d.stride(1),
        c_3d.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        # Two-stage lowering corrupts masked BF16 reduction tails on gfx950.
        num_stages=1,
    )

    return c_3d.reshape(batch_shape + (M, N))


@triton.jit
def linear_sigmoid_mul_kernel(
    x_ptr,
    z_ptr,
    wg_ptr,
    output_ptr,
    M,
    K,
    N,
    stride_xm,
    stride_xn,
    stride_zm,
    stride_zk,
    stride_wgn,
    stride_wgk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_BF16: tl.constexpr = True,
    bias_g_ptr=None,
    HAS_BIAS_G: tl.constexpr = False,
):
    """
    Fused: output = x * sigmoid(z @ Wg.T)
    Gate GEMM computed tile-by-tile, x loaded in epilogue,
    sigmoid * multiply in registers. Gate never hits GMEM.
    x: [M, N], z: [M, K], Wg: [N, K], output: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if not USE_BF16:
        rm = rm.to(tl.int64)
        rn = rn.to(tl.int64)

    # Accumulate gate GEMM: z @ Wg.T
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        if not USE_BF16:
            rk = rk.to(tl.int64)

        z_tile = tl.load(
            z_ptr + rm[:, None] * stride_zm + rk[None, :] * stride_zk,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )

        wg_tile = tl.load(
            wg_ptr + rn[None, :] * stride_wgn + rk[:, None] * stride_wgk,
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )

        if USE_BF16:
            acc_g = tl.dot(z_tile.to(tl.bfloat16), wg_tile.to(tl.bfloat16), acc_g)
        else:
            acc_g = tl.dot(z_tile, wg_tile, acc_g, input_precision="ieee")

    if HAS_BIAS_G:
        acc_g += tl.load(bias_g_ptr + rn, mask=rn < N, other=0.0)[None, :]
    # Epilogue: load x, compute x * sigmoid(gate) in registers
    x_tile = tl.load(
        x_ptr + rm[:, None] * stride_xm + rn[None, :] * stride_xn,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
        other=0.0,
    )

    result = x_tile * (1.0 / (1.0 + tl.exp(-acc_g)))

    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(
        output_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on,
        result,
        mask=out_mask,
    )


def triton_linear_sigmoid_mul(
    x: torch.Tensor,
    z: torch.Tensor,
    weight_g: torch.Tensor,
    *,
    use_bf16: bool = True,
    bias_g: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused: output = x * sigmoid(z @ Wg.T)
    Gate GEMM tile-by-tile, x loaded in epilogue, sigmoid*mul in registers.

    Args:
        x: Input tensor [..., N]
        z: Gate input tensor [..., K]
        weight_g: Gate weight [N, K]

    Returns:
        Output tensor [..., N]
    """
    input_shape = x.shape
    N = input_shape[-1]
    M = x.numel() // N
    K = z.shape[-1]

    x_2d = x.reshape(M, N).contiguous()
    z_2d = z.reshape(M, K).contiguous()
    output_2d = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_M = 32 if use_bf16 else 64
    BLOCK_N = min(128, triton.next_power_of_2(N))
    BLOCK_K = min(128 if use_bf16 else 32, triton.next_power_of_2(K))
    if not use_bf16:
        BLOCK_N = max(16, BLOCK_N)
        BLOCK_K = max(16, BLOCK_K)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    linear_sigmoid_mul_kernel[grid](
        x_2d,
        z_2d,
        weight_g,
        output_2d,
        M,
        K,
        N,
        x_2d.stride(0),
        x_2d.stride(1),
        z_2d.stride(0),
        z_2d.stride(1),
        weight_g.stride(0),
        weight_g.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        USE_BF16=use_bf16,
        bias_g_ptr=bias_g,
        HAS_BIAS_G=bias_g is not None,
        num_stages=2 if use_bf16 else 1,
    )

    return output_2d.reshape(input_shape)
