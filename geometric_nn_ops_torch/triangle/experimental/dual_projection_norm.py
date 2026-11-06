# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""IEEE dual projection with register-resident, full-row normalization."""

import torch
import triton
import triton.language as tl

from . import dual_projection_fp32


@triton.jit
def _row_norm(
    X,
    W,
    B,
    Y,
    M: tl.constexpr,
    C: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    EPS: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
    BC: tl.constexpr,
    ROWS: tl.constexpr,
):
    channels = tl.arange(0, BC)
    if HAS_W:
        weight = tl.load(W + channels, channels < C, other=0.0)
    if HAS_B:
        bias = tl.load(B + channels, channels < C, other=0.0)
    for offset in range(ROWS):
        row = (tl.program_id(0) * ROWS + offset).to(tl.int64)
        if row < M:
            x = tl.load(
                X + row * S0 + channels.to(tl.int64) * S1,
                channels < C,
                other=0.0,
            )
            mean = tl.sum(x, 0) / C
            centered = tl.where(channels < C, x - mean, 0.0)
            variance = tl.sum(centered * centered, 0) / C
            y = centered * tl.rsqrt(variance + EPS)
            if HAS_W:
                y *= weight
            if HAS_B:
                y += bias
            tl.store(Y + row * C + channels, y, channels < C)


def row_norm(x, weight, bias, eps):
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.float32:
        raise ValueError("Row normalization requires a 2D CUDA/HIP FP32 tensor")
    m, c = x.shape
    if m == 0 or not 1 <= c <= 4096:
        raise ValueError("Row normalization requires nonempty rows and C<=4096")
    x = x.contiguous()
    output = torch.empty((m, c), device=x.device, dtype=x.dtype)
    _row_norm[(triton.cdiv(m, 4),)](
        x,
        weight if weight is not None else x,
        bias if bias is not None else x,
        output,
        m,
        c,
        x.stride(0),
        x.stride(1),
        eps,
        weight is not None,
        bias is not None,
        triton.next_power_of_2(c),
        4,
        num_warps=1,
        num_stages=2,
    )
    return output


def update(x, direction, mask, eps, p):
    if max(x.shape[-1], p.hidden) > 4096:
        raise ValueError("Row normalization supports at most 4096 channels")
    return dual_projection_fp32.update(x, direction, mask, eps, p, normalize=row_norm)
