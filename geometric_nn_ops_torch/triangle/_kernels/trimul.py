# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable Triton triangle_multiplicative_update (forward + backward).

The Triton backend for
:func:`geometric_nn_ops_torch.triangle.triangle_multiplicative_update`, exposed as a
single ``autograd.Function`` used on the CUDA/Triton path for both inference and
training.

  * Forward uses a channel-major fused GEMM (LayerNorm -> gate/proj GEMM ->
    sigmoid -> multiply -> mask, written channel-major ``[B*C, N, N]``) and a
    per-channel batched matmul that combines the two projections.
  * Backward is a hybrid: the heavy per-channel combine matmul reuses the
    batched-GEMM Triton kernel (transposed operands give the two combine
    gradients); the remaining LayerNorm / linear / sigmoid / elementwise steps
    are differentiated with eager PyTorch ops (cheap relative to the fused
    projections).

The autograd.Function saves only minimal intermediates and recomputes the input
projections in backward, trading compute for a large activation-memory saving.

Weight convention (fused layout, bias-free projections): ``p_in_weight`` /
``g_in_weight`` are ``[2D, D]`` (rows ``0:D`` -> ``a``, ``D:2D`` -> ``b``);
``p_out_weight`` (= ``linear_z``) and ``g_out_weight`` (= ``linear_g``) are
``[D, D]``. All nn.Linear layout ``(out, in)``. This path assumes ``c_hidden ==
c_z == D`` and no projection biases.
"""

import torch

from geometric_nn_ops_torch.triangle._reference.trimul import is_outgoing

# Note: The limit for the number of tiles along grid.x is `2^32` whereas for the
# other dimensions it's `2^16`. Dimensions that grows with sequence length
# squared (e.g. M = B*N*N) can exceed the smaller limit and need to go on
# grid.x.

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def _cfg_fused_gemm(*, N):
    block_n = min(128, max(16, triton.next_power_of_2(N)))
    if block_n <= 64:
        return {
            "BLOCK_M": 128,
            "BLOCK_N": block_n,
            "BLOCK_K": 32,
            "num_warps": 8,
            "num_stages": 1,
        }
    return {
        "BLOCK_M": 128,
        "BLOCK_N": block_n,
        "BLOCK_K": 32,
        "num_warps": 16,
        "num_stages": 2,
    }


def _cfg_batched_gemm(*, N, total_channels):
    if N <= 640:
        return {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "GROUP_SIZE_M": 4,
            "num_warps": 4,
            "num_stages": 2,
        }
    elif total_channels <= 64:
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32,
            "GROUP_SIZE_M": 4,
            "num_warps": 8,
            "num_stages": 3,
        }
    return {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 32,
        "GROUP_SIZE_M": 4,
        "num_warps": 4,
        "num_stages": 1,
    }


def _cfg_layer_norm():
    return {"BLOCK_ROWS": 16, "BLOCK_C": 64, "num_warps": 2, "num_stages": 3}


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.jit
    def _batched_gemm_multi_channel(
        A_ptr,
        B_ptr,
        C_ptr,
        stride_Ab,
        stride_Ai,
        stride_Ak,
        stride_Bb,
        stride_Bk,
        stride_Bj,
        stride_Cb,
        stride_Ci,
        stride_Cj,
        M,
        K,
        N,
        total_channels,
        channels_per_batch,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """Per-channel C[c] = A[c] @ B[c], with grouped tile swizzle.

        Channels are processed in groups to amortize launch overhead. All channel
        offsets are computed in int64 to avoid INT32 overflow when stride == N^2.
        """
        batch_group = tl.program_id(0)
        pid = tl.program_id(1)

        num_tiles_m = tl.cdiv(M, BLOCK_M)
        num_tiles_n = tl.cdiv(N, BLOCK_N)

        num_pid_in_group = GROUP_SIZE_M * num_tiles_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_tiles_m - first_pid_m, GROUP_SIZE_M)
        pid_in_group = pid % num_pid_in_group

        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        start_channel = batch_group * channels_per_batch
        end_channel = tl.minimum(start_channel + channels_per_batch, total_channels)

        for channel in range(start_channel, end_channel):
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            channel_i64 = channel.to(tl.int64)

            for k_start in range(0, K, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)

                A = tl.load(
                    A_ptr
                    + channel_i64 * stride_Ab
                    + offs_m[:, None] * stride_Ai
                    + offs_k[None, :] * stride_Ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0,
                )
                B = tl.load(
                    B_ptr
                    + channel_i64 * stride_Bb
                    + offs_k[:, None] * stride_Bk
                    + offs_n[None, :] * stride_Bj,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0,
                )
                acc = tl.dot(A, B, acc)

            tl.store(
                C_ptr
                + channel_i64 * stride_Cb
                + offs_m[:, None] * stride_Ci
                + offs_n[None, :] * stride_Cj,
                acc,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            )

    @triton.jit
    def _fused_gemm_sigmoid_multiply_channel_major(
        A_ptr,
        W_g_ptr,
        W_p_ptr,
        C_ptr,
        mask_ptr,
        stride_Ai,
        stride_Ak,
        stride_Wgk,
        stride_Wgj,
        stride_Wpk,
        stride_Wpj,
        stride_Cb,
        stride_Ci,
        stride_Cj,
        M,
        K,
        N,
        N_spatial,
        SAVE_GP: tl.constexpr,
        G_ptr,
        P_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """gate = A @ W_g.T; proj = A @ W_p.T; out = sigmoid(gate)*proj*mask.

        Output written channel-major: input row m = b*N^2 + i*N + j maps to output
        (b*C + n, i, j). If SAVE_GP, also writes pre-activation gate (row-major,
        [M, N]) and proj into G_ptr/P_ptr for the backward.
        """
        # Large M dimension tiling on grid.x. See note at top of file.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m_i64 = offs_m.to(tl.int64)

        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_p = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            A = tl.load(
                A_ptr + offs_m_i64[:, None] * stride_Ai + offs_k[None, :] * stride_Ak,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            W_g = tl.load(
                W_g_ptr + offs_k[:, None] * stride_Wgk + offs_n[None, :] * stride_Wgj,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            W_p = tl.load(
                W_p_ptr + offs_k[:, None] * stride_Wpk + offs_n[None, :] * stride_Wpj,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc_g = tl.dot(A.to(tl.float32), W_g.to(tl.float32), acc_g)
            acc_p = tl.dot(A.to(tl.float32), W_p.to(tl.float32), acc_p)

        gate_sigmoid = 1.0 / (1.0 + tl.exp(-acc_g))
        result = gate_sigmoid * acc_p

        if mask_ptr is not None:
            m = tl.load(mask_ptr + offs_m, mask=offs_m < M, other=0.0)
            result = result * m[:, None]

        if SAVE_GP:
            gp_off = offs_m_i64[:, None] * N + offs_n[None, :]
            store_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl.store(G_ptr + gp_off, acc_g, mask=store_mask)
            tl.store(P_ptr + gp_off, acc_p, mask=store_mask)

        batch_idx = offs_m // (N_spatial * N_spatial)
        remainder = offs_m % (N_spatial * N_spatial)
        spatial_i = remainder // N_spatial
        spatial_j = remainder % N_spatial

        out_offsets = (
            (batch_idx[:, None].to(tl.int64) * N + offs_n[None, :]) * stride_Cb
            + spatial_i[:, None].to(tl.int64) * stride_Ci
            + spatial_j[:, None].to(tl.int64) * stride_Cj
        )
        tl.store(
            C_ptr + out_offsets,
            result,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    @triton.jit
    def _fused_gemm_sigmoid_multiply(
        A_ptr,
        W_g_ptr,
        W_p_ptr,
        C_ptr,
        A_p_ptr,
        stride_Ai,
        stride_Ak,
        stride_Wgk,
        stride_Wgj,
        stride_Wpk,
        stride_Wpj,
        stride_Ci,
        stride_Cj,
        stride_Api,
        stride_Apk,
        M,
        K,
        N,
        SAVE_G: tl.constexpr,
        G_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """out = sigmoid(A @ W_g.T) * (A_p @ W_p.T), row-major [M, N].

        Used for the output projection: gate from z_normed, proj from x_normed.
        If SAVE_G, writes the pre-sigmoid gate into G_ptr for the backward.
        """

        # Large M dimension tiling on grid.x. See note at top of file.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m_i64 = offs_m.to(tl.int64)

        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_p = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            A = tl.load(
                A_ptr + offs_m_i64[:, None] * stride_Ai + offs_k[None, :] * stride_Ak,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            A_p = tl.load(
                A_p_ptr
                + offs_m_i64[:, None] * stride_Api
                + offs_k[None, :] * stride_Apk,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            W_g = tl.load(
                W_g_ptr + offs_k[:, None] * stride_Wgk + offs_n[None, :] * stride_Wgj,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            W_p = tl.load(
                W_p_ptr + offs_k[:, None] * stride_Wpk + offs_n[None, :] * stride_Wpj,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc_g = tl.dot(A.to(tl.float32), W_g.to(tl.float32), acc_g)
            acc_p = tl.dot(A_p.to(tl.float32), W_p.to(tl.float32), acc_p)

        gate_sigmoid = 1.0 / (1.0 + tl.exp(-acc_g))
        result = gate_sigmoid * acc_p

        store_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if SAVE_G:
            tl.store(
                G_ptr + offs_m_i64[:, None] * N + offs_n[None, :],
                acc_g,
                mask=store_mask,
            )

        tl.store(
            C_ptr + offs_m_i64[:, None] * stride_Ci + offs_n[None, :] * stride_Cj,
            result,
            mask=store_mask,
        )

    @triton.jit
    def _layer_norm_2d_fwd(
        X,
        Y,
        W,
        B,
        Mean,
        Rstd,
        stride,
        total_rows,
        C,
        eps,
        SAVE_STATS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Row-wise LayerNorm over the last dim (C). Optionally saves mean/rstd."""
        pid = tl.program_id(0)
        row_start = pid * BLOCK_ROWS

        for row_offset in range(BLOCK_ROWS):
            row = row_start + row_offset
            row_valid = row < total_rows
            row_i64 = row.to(tl.int64)
            X_row = X + row_i64 * stride
            Y_row = Y + row_i64 * stride

            if row_valid:
                mean = 0.0
                M2 = 0.0
                count = 0
                for c_start in range(0, C, BLOCK_C):
                    c_offsets = c_start + tl.arange(0, BLOCK_C)
                    cmask = c_offsets < C
                    x = tl.load(X_row + c_offsets, mask=cmask, other=0.0).to(tl.float32)
                    n_valid = tl.sum(cmask.to(tl.int32))
                    if n_valid > 0:
                        local_sum = tl.sum(tl.where(cmask, x, 0.0))
                        local_mean = local_sum / n_valid
                        new_count = count + n_valid
                        delta = local_mean - mean
                        mean = mean + delta * n_valid / new_count
                        local_m2 = tl.sum(
                            tl.where(cmask, (x - local_mean) * (x - local_mean), 0.0)
                        )
                        M2 = M2 + local_m2 + delta * delta * count * n_valid / new_count
                        count = new_count

                var = M2 / C
                rstd = 1.0 / tl.sqrt(var + eps)

                if SAVE_STATS:
                    tl.store(Mean + row_i64, mean)
                    tl.store(Rstd + row_i64, rstd)

                for c_start in range(0, C, BLOCK_C):
                    c_offsets = c_start + tl.arange(0, BLOCK_C)
                    cmask = c_offsets < C
                    w = tl.load(W + c_offsets, mask=cmask)
                    b = tl.load(B + c_offsets, mask=cmask)
                    x = tl.load(X_row + c_offsets, mask=cmask, other=0.0).to(tl.float32)
                    x_hat = (x - mean) * rstd
                    y = x_hat * w + b
                    tl.store(Y_row + c_offsets, y, mask=cmask & row_valid)


# ---------------------------------------------------------------------------
# Python-side launch helpers
# ---------------------------------------------------------------------------


def _triton_layernorm(x_flat, weight, bias, eps, save_stats=False):
    """LayerNorm over last dim of a [M, C] tensor. Returns (y, mean, rstd)."""
    M, C = x_flat.shape
    y = torch.empty_like(x_flat)
    if save_stats:
        mean = torch.empty(M, device=x_flat.device, dtype=torch.float32)
        rstd = torch.empty(M, device=x_flat.device, dtype=torch.float32)
    else:
        mean = x_flat  # unused
        rstd = x_flat
    cfg = _cfg_layer_norm()

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_ROWS"]),)

    _layer_norm_2d_fwd[grid](
        x_flat,
        y,
        weight,
        bias,
        mean,
        rstd,
        x_flat.stride(0),
        M,
        C,
        eps,
        SAVE_STATS=save_stats,
        **cfg,
    )
    if save_stats:
        return y, mean, rstd
    return y, None, None


def _combine_projection(a, b, batch_dims, c_hidden):
    """Per-channel a@b. a,b: [B*C, N, N] channel-major. Returns [..., N, N, C]."""
    BC, N, _ = a.shape
    out = torch.empty(BC, N, N, device=a.device, dtype=a.dtype)
    MAX_CH = 16
    num_batches = (BC + MAX_CH - 1) // MAX_CH
    channels_per_batch = (BC + num_batches - 1) // num_batches
    cfg = _cfg_batched_gemm(N=N, total_channels=BC)

    def grid(meta):
        nm = triton.cdiv(N, meta["BLOCK_M"])
        nn = triton.cdiv(N, meta["BLOCK_N"])
        return (num_batches, nm * nn)

    _batched_gemm_multi_channel[grid](
        a,
        b,
        out,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        N,
        N,
        N,
        BC,
        channels_per_batch,
        **cfg,
    )
    Bflat = 1
    for d in batch_dims:
        Bflat *= d
    out = (
        out.reshape(Bflat, c_hidden, N, N)
        .permute(0, 2, 3, 1)
        .reshape(Bflat, N, N, c_hidden)
    )
    return out


def _combine_projection_raw(a, b):
    """Per-channel a@b. a,b: [BC, M, K]/[BC, K, N]. Returns [BC, M, N]. No reshape."""
    BC, M, K = a.shape
    N = b.shape[2]
    out = torch.empty(BC, M, N, device=a.device, dtype=a.dtype)
    MAX_CH = 16
    num_batches = (BC + MAX_CH - 1) // MAX_CH
    channels_per_batch = (BC + num_batches - 1) // num_batches
    cfg = _cfg_batched_gemm(N=max(M, N), total_channels=BC)

    def grid(meta):
        nm = triton.cdiv(M, meta["BLOCK_M"])
        nn = triton.cdiv(N, meta["BLOCK_N"])
        return (num_batches, nm * nn)

    _batched_gemm_multi_channel[grid](
        a,
        b,
        out,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        M,
        K,
        N,
        BC,
        channels_per_batch,
        **cfg,
    )
    return out


# ---------------------------------------------------------------------------
# Backward helpers (eager PyTorch; correctness-first hybrid).
# ---------------------------------------------------------------------------


def _layernorm_backward(dy, x, weight, mean, rstd, eps):
    """Backward of LayerNorm over last dim. All [M, C]. mean/rstd [M] fp32.

    Returns (dx, dweight, dbias). dweight/dbias summed over rows.
    """
    M, C = x.shape
    x = x.float()
    dy = dy.float()
    w = weight.float()
    mean = mean.view(M, 1)
    rstd = rstd.view(M, 1)
    x_hat = (x - mean) * rstd
    # dx
    dy_w = dy * w
    dx = rstd * (
        dy_w
        - dy_w.mean(dim=1, keepdim=True)
        - x_hat * (dy_w * x_hat).mean(dim=1, keepdim=True)
    )
    dweight = (dy * x_hat).sum(dim=0)
    dbias = dy.sum(dim=0)
    return dx, dweight, dbias


class _TriangleMultiplicativeUpdateFn(torch.autograd.Function):
    """Differentiable TMU. Forward uses Triton fused kernels; backward is a
    correctness-first hybrid (Triton batched-GEMM for the combine step, eager
    PyTorch for LN/linear/sigmoid/elementwise).

    Weight convention (fused, bias-free): W_p_in / W_g_in are [2D, D] (rows 0:D
    -> a, D:2D -> b); W_z (= linear_z) and W_g_out (= linear_g) are [D, D]. All
    nn.Linear layout (out, in), bias-free.
    """

    @staticmethod
    def forward(
        ctx,
        z,
        mask,
        outgoing,
        norm_in_weight,
        norm_in_bias,
        p_in_weight,
        g_in_weight,
        norm_out_weight,
        norm_out_bias,
        p_out_weight,
        g_out_weight,
        eps,
    ):
        B = 1
        batch_dims = z.shape[:-3]
        for d in batch_dims:
            B *= d
        N = z.shape[-3]
        D = z.shape[-1]
        compute_dtype = z.dtype

        # Cast weights to compute dtype.
        p_in_weight = p_in_weight.to(compute_dtype)
        g_in_weight = g_in_weight.to(compute_dtype)
        p_out_weight = p_out_weight.to(compute_dtype)
        g_out_weight = g_out_weight.to(compute_dtype)
        mask = mask.to(compute_dtype)

        # Split fused weights and transpose to [K, N] = [D, D] for the kernels.
        W_ap = p_in_weight[:D].T.contiguous()
        W_bp = p_in_weight[D:].T.contiguous()
        W_ag = g_in_weight[:D].T.contiguous()
        W_bg = g_in_weight[D:].T.contiguous()

        z_flat = z.reshape(-1, D).contiguous()
        M = z_flat.shape[0]

        z_n_flat, mean_in, rstd_in = _triton_layernorm(
            z_flat, norm_in_weight, norm_in_bias, eps, save_stats=True
        )

        mask_flat = mask.reshape(-1)

        a_cm = torch.empty(B * D, N, N, device=z.device, dtype=compute_dtype)
        b_cm = torch.empty(B * D, N, N, device=z.device, dtype=compute_dtype)

        fg = _cfg_fused_gemm(N=D)

        def proj_grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]), 1, triton.cdiv(D, meta["BLOCK_N"]))

        # a projection (no need to save g/p preactivations; recomputed in bwd)
        _fused_gemm_sigmoid_multiply_channel_major[proj_grid](
            z_n_flat,
            W_ag,
            W_ap,
            a_cm,
            mask_flat,
            z_n_flat.stride(0),
            z_n_flat.stride(1),
            W_ag.stride(0),
            W_ag.stride(1),
            W_ap.stride(0),
            W_ap.stride(1),
            a_cm.stride(0),
            a_cm.stride(1),
            a_cm.stride(2),
            M,
            D,
            D,
            N,
            False,
            a_cm,
            a_cm,
            **fg,
        )
        _fused_gemm_sigmoid_multiply_channel_major[proj_grid](
            z_n_flat,
            W_bg,
            W_bp,
            b_cm,
            mask_flat,
            z_n_flat.stride(0),
            z_n_flat.stride(1),
            W_bg.stride(0),
            W_bg.stride(1),
            W_bp.stride(0),
            W_bp.stride(1),
            b_cm.stride(0),
            b_cm.stride(1),
            b_cm.stride(2),
            M,
            D,
            D,
            N,
            False,
            b_cm,
            b_cm,
            **fg,
        )

        # Direction-specific transposes for the combine matmul.
        if outgoing:
            b_comb = b_cm.transpose(-1, -2).contiguous()
            a_comb = a_cm
        else:
            a_comb = a_cm.transpose(-1, -2).contiguous()
            b_comb = b_cm

        x_out = _combine_projection(a_comb, b_comb, batch_dims, D)  # [Bflat,N,N,D]
        x_flat = x_out.reshape(-1, D).contiguous()

        x_n_flat, mean_out, rstd_out = _triton_layernorm(
            x_flat, norm_out_weight, norm_out_bias, eps, save_stats=True
        )

        W_g_out = g_out_weight.T.contiguous()
        W_p_out = p_out_weight.T.contiguous()

        out_flat = torch.empty(M, D, device=z.device, dtype=compute_dtype)
        g_out_pre = torch.empty(M, D, device=z.device, dtype=torch.float32)

        def out_grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]), 1, triton.cdiv(D, meta["BLOCK_N"]))

        _fused_gemm_sigmoid_multiply[out_grid](
            z_n_flat,
            W_g_out,
            W_p_out,
            out_flat,
            x_n_flat,
            z_n_flat.stride(0),
            z_n_flat.stride(1),
            W_g_out.stride(0),
            W_g_out.stride(1),
            W_p_out.stride(0),
            W_p_out.stride(1),
            out_flat.stride(0),
            out_flat.stride(1),
            x_n_flat.stride(0),
            x_n_flat.stride(1),
            M,
            D,
            D,
            True,
            g_out_pre,
            **fg,
        )

        ctx.save_for_backward(
            z_flat,
            mask_flat,
            z_n_flat,
            mean_in,
            rstd_in,
            a_cm,
            b_cm,
            x_flat,
            x_n_flat,
            mean_out,
            rstd_out,
            g_out_pre,
            p_in_weight,
            g_in_weight,
            p_out_weight,
            g_out_weight,
            norm_in_weight,
            norm_in_bias,
            norm_out_weight,
            norm_out_bias,
        )
        ctx.outgoing = outgoing
        ctx.eps = eps
        ctx.B = B
        ctx.N = N
        ctx.D = D
        ctx.batch_dims = tuple(batch_dims)

        return out_flat.reshape(*batch_dims, N, N, D)

    @staticmethod
    def backward(ctx, grad_out):
        (
            z_flat,
            mask_flat,
            z_n_flat,
            mean_in,
            rstd_in,
            a_cm,
            b_cm,
            x_flat,
            x_n_flat,
            mean_out,
            rstd_out,
            g_out_pre,
            p_in_weight,
            g_in_weight,
            p_out_weight,
            g_out_weight,
            norm_in_weight,
            norm_in_bias,
            norm_out_weight,
            norm_out_bias,
        ) = ctx.saved_tensors
        outgoing = ctx.outgoing
        eps = ctx.eps
        N, D = ctx.N, ctx.D
        batch_dims = ctx.batch_dims
        M = z_flat.shape[0]

        # Precision: GEMM operands are the compute dtype (e.g. bf16 under
        # autocast) elementwise ops and matmuls accumulation happen in fp32 in
        # registers, but are down cast to bf16 for storage.
        cdtype = z_n_flat.dtype
        grad_out_flat = grad_out.reshape(-1, D)
        z_n = z_n_flat
        x_n = x_n_flat
        Wz = p_out_weight.to(cdtype)
        Wg_out = g_out_weight.to(cdtype)

        # ---- Output projection backward ----
        # out = sigmoid(z_n @ Wg^T) * (x_n @ Wz^T)
        sig_g = torch.sigmoid(g_out_pre)  # fp32 (saved preact)
        xz = (x_n @ Wz.T).float()  # proj [M, D]
        go = grad_out_flat.float()
        d_g_pre = (go * xz * sig_g * (1.0 - sig_g)).to(cdtype)  # wrt gate preact
        d_proj = (go * sig_g).to(cdtype)  # wrt proj

        d_p_out_weight = d_proj.T @ x_n  # [D, D]
        d_x_n = d_proj @ Wz  # [M, D]
        d_g_out_weight = d_g_pre.T @ z_n  # [D, D]
        d_z_n = (d_g_pre @ Wg_out).float()  # accumulate (fp32)

        # ---- LN_out backward ----
        d_x_flat, d_norm_out_weight, d_norm_out_bias = _layernorm_backward(
            d_x_n, x_flat, norm_out_weight, mean_out, rstd_out, eps
        )

        # ---- Combine backward ----
        # x[bflat,N,N,D] = combine(a_comb,b_comb) per channel.
        Bflat = 1
        for d in batch_dims:
            Bflat *= d
        d_x_cm = (
            d_x_flat.to(cdtype)
            .reshape(Bflat, N, N, D)
            .permute(0, 3, 1, 2)
            .reshape(Bflat * D, N, N)
            .contiguous()
        )

        # The batched GEMM kernel reads operands via arbitrary strides, so
        # transposed views can be passed directly (no .contiguous() copies).
        if outgoing:
            b_comb = b_cm.transpose(-1, -2)
            a_comb = a_cm
        else:
            a_comb = a_cm.transpose(-1, -2)
            b_comb = b_cm

        # p = a_comb @ b_comb  -> da = dp @ b_comb^T ; db = a_comb^T @ dp
        d_a_comb = _combine_projection_raw(d_x_cm, b_comb.transpose(-1, -2))
        d_b_comb = _combine_projection_raw(a_comb.transpose(-1, -2), d_x_cm)

        if outgoing:
            d_a_cm = d_a_comb
            d_b_cm = d_b_comb.transpose(-1, -2)
        else:
            d_a_cm = d_a_comb.transpose(-1, -2)
            d_b_cm = d_b_comb

        # back to row-major [M, D]
        d_a = d_a_cm.reshape(Bflat, D, N, N).permute(0, 2, 3, 1).reshape(M, D)
        d_b = d_b_cm.reshape(Bflat, D, N, N).permute(0, 2, 3, 1).reshape(M, D)

        # ---- Input projections backward ----
        # a = mask * sigmoid(z_n @ Wag^T) * (z_n @ Wap^T)
        mask_col = mask_flat.float().view(M, 1)
        Wap = p_in_weight[:D].to(cdtype)
        Wbp = p_in_weight[D:].to(cdtype)
        Wag = g_in_weight[:D].to(cdtype)
        Wbg = g_in_weight[D:].to(cdtype)

        def proj_in_bwd(d_out, Wp, Wg):
            ap = (z_n @ Wp.T).float()
            sa = torch.sigmoid((z_n @ Wg.T).float())
            d_after_mask = d_out.float() * mask_col
            d_ap = (d_after_mask * sa).to(cdtype)
            d_ag = (d_after_mask * ap * sa * (1.0 - sa)).to(cdtype)
            d_Wp = d_ap.T @ z_n
            d_Wg = d_ag.T @ z_n
            d_zn = (d_ap @ Wp + d_ag @ Wg).float()
            return d_zn, d_Wp, d_Wg

        d_zn_a, d_Wap, d_Wag = proj_in_bwd(d_a, Wap, Wag)
        d_zn_b, d_Wbp, d_Wbg = proj_in_bwd(d_b, Wbp, Wbg)
        d_z_n = d_z_n + d_zn_a + d_zn_b

        # ---- LN_in backward ----
        d_z_flat, d_norm_in_weight, d_norm_in_bias = _layernorm_backward(
            d_z_n, z_flat, norm_in_weight, mean_in, rstd_in, eps
        )

        # Assemble fused weight grads.
        d_p_in_weight = torch.cat([d_Wap, d_Wbp], dim=0)
        d_g_in_weight = torch.cat([d_Wag, d_Wbg], dim=0)

        d_z = d_z_flat.reshape(*batch_dims, N, N, D).to(grad_out.dtype)

        return (
            d_z,
            None,  # mask
            None,  # outgoing
            d_norm_in_weight.to(norm_in_weight.dtype),
            d_norm_in_bias.to(norm_in_bias.dtype),
            d_p_in_weight.to(p_in_weight.dtype),
            d_g_in_weight.to(g_in_weight.dtype),
            d_norm_out_weight.to(norm_out_weight.dtype),
            d_norm_out_bias.to(norm_out_bias.dtype),
            d_p_out_weight.to(p_out_weight.dtype),
            d_g_out_weight.to(g_out_weight.dtype),
            None,  # eps
        )


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
    """Triangle multiplicative update on the fused Triton kernels.

    Takes the same arguments as ``trimul_torch``; see it for the layouts.

    The kernels implement the standard config, and the configs outside it are
    rejected here rather than ignored so that calling this directly cannot
    silently produce something else. The public op routes them to the torch
    path instead.
    """
    named = [
        name
        for name, value in (
            ("p_in_bias", p_in_bias),
            ("g_in_bias", g_in_bias),
            ("p_out_bias", p_out_bias),
            ("g_out_bias", g_out_bias),
        )
        if value is not None
    ]
    if named:
        raise ValueError(
            f"the Triton kernels have bias-free projections, got {', '.join(named)}"
        )
    c_hidden, c_z = p_in_weight.shape[0] // 2, x.shape[-1]
    if c_hidden != c_z:
        raise ValueError(
            f"the Triton kernels need a square hidden dim, got "
            f"c_hidden={c_hidden} against c_z={c_z}"
        )

    *batch_dims, n, _, d = x.shape
    x_flat = x.reshape(-1, n, n, d)
    if mask is None:
        mask_flat = x_flat.new_ones(x_flat.shape[:-1])
    else:
        mask_flat = mask.expand(x.shape[:-1]).reshape(-1, n, n).contiguous()

    out = _TriangleMultiplicativeUpdateFn.apply(
        x_flat,
        mask_flat,
        is_outgoing(direction),
        norm_in_weight,
        norm_in_bias,
        p_in_weight,
        g_in_weight,
        norm_out_weight,
        norm_out_bias,
        p_out_weight,
        g_out_weight,
        eps,
    )
    return out.reshape(*batch_dims, n, n, d)
