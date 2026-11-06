# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Triton kernels for the triangular multiplicative update (forward).

Triton kernels for the triangle multiplicative update: fused
GEMM+sigmoid+multiply projections, a multi-channel batched GEMM for the
per-channel combine, and a fused 2D LayerNorm.
"""

import torch
import triton
import triton.language as tl


# Multi-channel batched GEMM kernel that processes multiple channels per launch
@triton.autotune(
    configs=[
        # Larger block sizes for better GPU utilization
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=8,
        ),
        # Original configs as fallback
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_stages=3,
            num_warps=2,
        ),
    ],
    key=["M", "K", "N"],
)
@triton.jit
def batched_gemm_multi_channel(
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
    # batch index
    batch_group = tl.program_id(0)

    # tile indices
    pid = tl.program_id(1)  # Linear tile index

    # Number of tiles
    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)

    # Swizzling calculation
    num_pid_in_group = GROUP_SIZE_M * num_tiles_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_tiles_m - first_pid_m, GROUP_SIZE_M)
    pid_in_group = pid % num_pid_in_group

    # Final tile indices
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    # Rest of kernel uses pid_m and pid_n...
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Process multiple channels in this kernel
    start_channel = batch_group * channels_per_batch
    end_channel = tl.minimum(start_channel + channels_per_batch, total_channels)

    # Process each channel in the group
    for channel in range(start_channel, end_channel):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)

            A = tl.load(
                A_ptr
                + channel * stride_Ab
                + offs_m[:, None] * stride_Ai
                + offs_k[None, :] * stride_Ak,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )

            B = tl.load(
                B_ptr
                + channel * stride_Bb
                + offs_k[:, None] * stride_Bk
                + offs_n[None, :] * stride_Bj,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )

            acc = tl.dot(A, B, acc)

        tl.store(
            C_ptr
            + channel * stride_Cb
            + offs_m[:, None] * stride_Ci
            + offs_n[None, :] * stride_Cj,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


# Fused GEMM + sigmoid + multiply kernel with channel-major output
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
def fused_gemm_sigmoid_multiply_channel_major(
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
    B,
    N_spatial,  # B is number of batches, N_spatial is spatial dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel that computes:
    gate = A @ W_g.T + bias_g
    proj = (A_p if A_p_ptr else A) @ W_p.T + bias_p
    output = sigmoid(gate) * proj * mask

    Key difference: Output is written in channel-major format [B*C, N, N]
    where each input row (b*N*N + i*N + j) writes to output (b*C + c, i, j)
    """
    b = tl.program_id(0)

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
        acc_g = tl.dot(A.to(tl.float32), W_g.to(tl.float32), acc_g)
        acc_p = tl.dot(A_p.to(tl.float32), W_p.to(tl.float32), acc_p)

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

    # Store result in channel-major format
    # For each input row m and output channel n:
    # Input row m = batch_idx * N_spatial^2 + spatial_i * N_spatial + spatial_j
    # We need to extract batch_idx, spatial_i, spatial_j

    # Vectorized approach - compute indices for all elements in the tile
    # Compute batch_idx, spatial_i, spatial_j for all elements in offs_m
    batch_idx = offs_m // (N_spatial * N_spatial)  # [BLOCK_M]
    remainder = offs_m % (N_spatial * N_spatial)  # [BLOCK_M]
    spatial_i = remainder // N_spatial  # [BLOCK_M]
    spatial_j = remainder % N_spatial  # [BLOCK_M]

    # Compute output offsets for all (m, n) pairs
    # out_idx[m, n] = (batch_idx[m] * N + n) * stride_Cb + spatial_i[m] * stride_Ci + spatial_j[m] * stride_Cj
    out_offsets = (
        (batch_idx[:, None] * N + offs_n[None, :]) * stride_Cb
        + spatial_i[:, None] * stride_Ci
        + spatial_j[:, None] * stride_Cj
    )

    # Single vectorized store
    tl.store(
        C_ptr + out_offsets, result, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


# Original fused GEMM + sigmoid + multiply kernel (for other uses)
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
    b = tl.program_id(0)

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
        acc_g = tl.dot(A.to(tl.float32), W_g.to(tl.float32), acc_g)
        acc_p = tl.dot(A_p.to(tl.float32), W_p.to(tl.float32), acc_p)

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


def _is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def triton_combine_projection(a, b, outgoing=True, batch_dims=None, c_hidden=None):
    """
    Optimized version that accepts inputs already in [B*C, N, N] format.
    No permutations needed!

    Args:
        a: [B*C, N, N] tensor (already in channel-major format)
        b: [B*C, N, N] tensor (already in channel-major format)
        outgoing: whether this is outgoing or incoming update
        batch_dims: original batch dimensions (for reshaping output)
        c_hidden: hidden channel dimension (for reshaping output)

    Returns:
        [..., N, N, C] tensor
    """
    # Get dimensions - tensors should already be on correct device
    BC, N, _ = a.shape

    # Preserve input dtype for output
    ab_matmul = torch.empty(BC, N, N, device=a.device, dtype=a.dtype)

    # Batch multiple channels together to reduce kernel launches
    MAX_CHANNELS_PER_BATCH = 16  # Process up to 16 channels in one kernel launch

    # Calculate how many channels we can batch together
    num_batches = (BC + MAX_CHANNELS_PER_BATCH - 1) // MAX_CHANNELS_PER_BATCH
    channels_per_batch = (BC + num_batches - 1) // num_batches

    def grid(meta):
        num_m_tiles = triton.cdiv(N, meta["BLOCK_M"])
        num_n_tiles = triton.cdiv(N, meta["BLOCK_N"])
        total_tiles = num_m_tiles * num_n_tiles
        return (
            num_batches,  # Reduced number of batches
            total_tiles,
        )

    # Run the batched GEMM kernel directly - no reshaping needed!
    batched_gemm_multi_channel[grid](
        a,
        b,
        ab_matmul,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        ab_matmul.stride(0),
        ab_matmul.stride(1),
        ab_matmul.stride(2),
        N,
        N,
        N,  # M, K, N for square matrices
        BC,  # Total number of channels
        channels_per_batch,  # Channels to process per kernel
    )

    # Convert output to [..., N, N, C] format if batch_dims provided
    if batch_dims is not None and c_hidden is not None:
        Bflat = 1
        for d in batch_dims:
            Bflat *= d
        # Reshape from [B*C, N, N] to [..., N, N, C]
        ab_matmul = (
            ab_matmul.reshape(Bflat, c_hidden, N, N)
            .permute(0, 2, 3, 1)
            .reshape(Bflat, N, N, c_hidden)
        )

    return ab_matmul


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROWS": 16, "BLOCK_C": 128}, num_warps=4),
        triton.Config({"BLOCK_ROWS": 32, "BLOCK_C": 128}, num_warps=8),
        triton.Config({"BLOCK_ROWS": 64, "BLOCK_C": 128}, num_warps=8),
        triton.Config({"BLOCK_ROWS": 16, "BLOCK_C": 64}, num_warps=2),
        triton.Config({"BLOCK_ROWS": 32, "BLOCK_C": 64}, num_warps=4),
    ],
    key=["total_rows", "C"],
)
@triton.jit
def _layer_norm_2d_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    stride,  # stride between rows
    total_rows,  # total number of rows
    C,  # number of columns (channels)
    eps,  # epsilon to avoid division by zero
    BLOCK_ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Each program handles BLOCK_ROWS rows
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROWS

    # Process BLOCK_ROWS rows in this kernel
    for row_offset in range(BLOCK_ROWS):
        row = row_start + row_offset

        # Use masking instead of break/continue
        row_valid = row < total_rows

        # Pointers for this row
        X_row = X + row * stride
        Y_row = Y + row * stride

        # Only process if row is valid
        if row_valid:
            # Use Welford's algorithm for numerical stability
            mean = 0.0
            M2 = 0.0  # Sum of squared differences from mean
            count = 0

            # Single pass to compute mean and variance
            for c_start in range(0, C, BLOCK_C):
                c_offsets = c_start + tl.arange(0, BLOCK_C)
                mask = c_offsets < C
                x = tl.load(X_row + c_offsets, mask=mask, other=0.0).to(tl.float32)

                # Welford's - vectorized version
                valid_mask = (c_offsets < C) & mask
                n_valid = tl.sum(valid_mask.to(tl.int32))

                if n_valid > 0:
                    # Compute local mean for this block
                    local_sum = tl.sum(tl.where(valid_mask, x, 0.0))
                    local_mean = local_sum / n_valid

                    # Update global mean and M2
                    new_count = count + n_valid
                    delta = local_mean - mean
                    mean = mean + delta * n_valid / new_count

                    # Update M2 (sum of squared differences)
                    local_m2 = tl.sum(
                        tl.where(valid_mask, (x - local_mean) * (x - local_mean), 0.0)
                    )
                    M2 = M2 + local_m2 + delta * delta * count * n_valid / new_count
                    count = new_count

            # Final variance and rstd
            var = M2 / C
            rstd = 1.0 / tl.sqrt(var + eps)

            # Normalize and apply linear transformation
            for c_start in range(0, C, BLOCK_C):
                c_offsets = c_start + tl.arange(0, BLOCK_C)
                mask = c_offsets < C

                # Load weights and biases
                w = tl.load(W + c_offsets, mask=mask)
                b = tl.load(B + c_offsets, mask=mask)

                # Load input, normalize, and apply affine transform
                x = tl.load(X_row + c_offsets, mask=mask, other=0.0).to(tl.float32)
                x_hat = (x - mean) * rstd
                y = x_hat * w + b

                # Store output
                tl.store(Y_row + c_offsets, y, mask=mask & row_valid)
