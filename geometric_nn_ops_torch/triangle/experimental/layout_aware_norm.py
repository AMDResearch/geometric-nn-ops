# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Transpose-free channel-major normalization with library GEMMs."""

import torch
import triton
import triton.language as tl

from . import layout_aware


@triton.jit
def _channel_major_norm(
    X,
    W,
    B,
    Y,
    H: tl.constexpr,
    M: tl.constexpr,
    SXH: tl.constexpr,
    SXM: tl.constexpr,
    EPS: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
    BH: tl.constexpr,
    BM: tl.constexpr,
):
    channels = tl.arange(0, BH)
    rows = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    offsets = channels[:, None].to(tl.int64) * SXH + rows[None, :] * SXM
    valid = (channels[:, None] < H) & (rows[None, :] < M)
    x = tl.load(X + offsets, valid, other=0.0)
    mean = tl.sum(x, axis=0) / H
    centered = tl.where(channels[:, None] < H, x - mean[None, :], 0.0)
    variance = tl.sum(centered * centered, axis=0) / H
    y = centered * tl.rsqrt(variance[None, :] + EPS)
    if HAS_W:
        weight = tl.load(W + channels, channels < H, other=0.0)
        y *= weight[:, None]
    if HAS_B:
        bias = tl.load(B + channels, channels < H, other=0.0)
        y += bias[:, None]
    tl.store(Y + channels[:, None].to(tl.int64) * M + rows[None, :], y, valid)


def channel_major_norm(x, weight, bias, eps):
    h, m = x.shape
    if not x.is_cuda or x.dtype != torch.float32:
        raise ValueError("Channel-major normalization requires CUDA/HIP FP32")
    if h > 4096:
        raise ValueError("Channel-major normalization supports at most 4096 channels")
    output = torch.empty((h, m), dtype=x.dtype, device=x.device)
    bh = triton.next_power_of_2(h)
    bm = min(64, max(1, 32768 // bh))
    _channel_major_norm[(triton.cdiv(m, bm),)](
        x,
        weight if weight is not None else x,
        bias if bias is not None else x,
        output,
        h,
        m,
        x.stride(0),
        x.stride(1),
        eps,
        weight is not None,
        bias is not None,
        bh,
        bm,
        num_warps=4,
    )
    return output


def update(x, direction, mask, eps, p):
    if torch.get_float32_matmul_precision() != "highest":
        raise ValueError("Layout-aware normalization requires highest/IEEE matmul")
    if p.hidden > 4096:
        raise ValueError("Channel-major normalization supports at most 4096 channels")
    return layout_aware.update(
        x, direction, mask, eps, p, channel_major_norm=channel_major_norm
    )
