# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""IEEE FP32 Triton compute path; Torch is used for allocation and views."""

import torch
import triton
import triton.language as tl

from geometric_nn_ops_torch.triangle._reference.trimul import is_outgoing

from . import initial_kernels as kernels
from . import original_channel_major
from .dual_projection_norm import row_norm
from .layout_aware_norm import channel_major_norm


@triton.jit
def _norm_producer(
    X,
    NW,
    NB,
    WG,
    WP,
    BG,
    BP,
    MASK,
    Z,
    AB,
    M: tl.constexpr,
    C: tl.constexpr,
    SPATIAL: tl.constexpr,
    BATCH: tl.constexpr,
    EPS: tl.constexpr,
    HAS_NW: tl.constexpr,
    HAS_NB: tl.constexpr,
    HAS_BG: tl.constexpr,
    HAS_BP: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BC: tl.constexpr,
):
    rows = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    channels = tl.arange(0, BC)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    x = tl.load(
        X + rows[:, None] * C + channels[None, :],
        (rows[:, None] < M) & (channels[None, :] < C),
        other=0.0,
    )
    mean = tl.sum(x, 1) / C
    centered = tl.where(channels[None, :] < C, x - mean[:, None], 0.0)
    variance = tl.sum(centered * centered, 1) / C
    z = centered * tl.rsqrt(variance[:, None] + EPS)
    if HAS_NW:
        z *= tl.load(NW + channels, channels < C, other=0.0)[None, :]
    if HAS_NB:
        z += tl.load(NB + channels, channels < C, other=0.0)[None, :]
    z = tl.where(channels[None, :] < C, z, 0.0)
    if tl.program_id(1) == 0:
        tl.store(
            Z + rows[:, None] * C + channels[None, :],
            z,
            (rows[:, None] < M) & (channels[None, :] < C),
        )
    wg = tl.load(
        WG + cols[None, :].to(tl.int64) * C + channels[:, None],
        (cols[None, :] < 2 * C) & (channels[:, None] < C),
        other=0.0,
    )
    wp = tl.load(
        WP + cols[None, :].to(tl.int64) * C + channels[:, None],
        (cols[None, :] < 2 * C) & (channels[:, None] < C),
        other=0.0,
    )
    gate = tl.dot(z, wg, input_precision="ieee")
    projection = tl.dot(z, wp, input_precision="ieee")
    if HAS_BG:
        gate += tl.load(BG + cols, cols < 2 * C, other=0.0)[None, :]
    if HAS_BP:
        projection += tl.load(BP + cols, cols < 2 * C, other=0.0)[None, :]
    out = projection / (1.0 + tl.exp(-gate))
    if HAS_MASK:
        out *= tl.load(MASK + rows, rows < M, other=0.0)[:, None]
    offsets = (
        (cols[None, :] // C).to(tl.int64) * BATCH * C * SPATIAL
        + (rows[:, None] // SPATIAL) * C * SPATIAL
        + (cols[None, :] % C) * SPATIAL
        + rows[:, None] % SPATIAL
    )
    tl.store(AB + offsets, out, (rows[:, None] < M) & (cols[None, :] < 2 * C))


@triton.jit
def _combine(
    A,
    B,
    O,
    N: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    A2: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    channel = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * BM + tl.arange(0, BM)
    j = tl.program_id(2) * BN + tl.arange(0, BN)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for start in range(triton.cdiv(N, BK)):
        k = start * BK + tl.arange(0, BK)
        a = tl.load(
            A + channel * A0 + i[:, None].to(tl.int64) * A1 + k[None, :] * A2,
            (i[:, None] < N) & (k[None, :] < N),
            other=0.0,
        )
        b = tl.load(
            B + channel * B0 + k[:, None].to(tl.int64) * B1 + j[None, :] * B2,
            (k[:, None] < N) & (j[None, :] < N),
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
    tl.store(
        O + channel * N * N + i[:, None].to(tl.int64) * N + j[None, :],
        accumulator,
        (i[:, None] < N) & (j[None, :] < N),
    )


def combine(a, b):
    channels, n, _ = a.shape
    output = torch.empty((channels, n, n), device=a.device, dtype=a.dtype)
    block = 32 if n < 128 else 64
    _combine[(channels, triton.cdiv(n, block), triton.cdiv(n, block))](
        a,
        b,
        output,
        n,
        *a.stride(),
        *b.stride(),
        block,
        block,
        32,
        num_warps=4,
        num_stages=1,
    )
    return output


def normalize(x, weight, bias, eps):
    if x.stride(0) == 1 and x.stride(1) != 1:
        return channel_major_norm(x.t(), weight, bias, eps).t()
    return row_norm(x, weight, bias, eps)


def update(x, direction, mask, eps, p):
    if max(x.shape[-1], p.hidden) > 4096:
        raise ValueError("All-Triton normalization supports at most 4096 channels")
    return original_channel_major.update(
        x,
        direction,
        mask,
        eps,
        p,
        normalize=normalize,
        combine=combine,
        pack_weights=False,
    )


def update_fused(x, direction, mask, eps, p):
    if p.hidden != x.shape[-1] or x.shape[-1] > 512:
        raise ValueError("Fused all-Triton producer requires C_hidden == C <= 512")
    if torch.get_float32_matmul_precision() != "highest":
        raise ValueError("Fused all-Triton requires highest/IEEE matmul")

    def vector(value):
        return None if value is None else value.contiguous()

    n, c = x.shape[-2:]
    flat = x.reshape(-1, c).contiguous()
    m = flat.shape[0]
    batches = m // (n * n)
    z = torch.empty_like(flat)
    ab = x.new_empty(2, batches * c, n, n)
    mf = None if mask is None else mask.expand(x.shape[:-1]).reshape(-1).contiguous()
    # Weight views are supported by the baseline; this fused kernel uses native
    # nn.Linear storage so normalize strided weights explicitly.
    wg, wp = p.g_in_weight.contiguous(), p.p_in_weight.contiguous()
    _norm_producer[(triton.cdiv(m, 16), triton.cdiv(2 * c, 64))](
        flat,
        vector(p.norm_in_weight),
        vector(p.norm_in_bias),
        wg,
        wp,
        vector(p.g_in_bias),
        vector(p.p_in_bias),
        mf,
        z,
        ab,
        m,
        c,
        n * n,
        batches,
        eps,
        p.norm_in_weight is not None,
        p.norm_in_bias is not None,
        p.g_in_bias is not None,
        p.p_in_bias is not None,
        mf is not None,
        16,
        64,
        max(16, triton.next_power_of_2(c)),
        num_warps=4,
        num_stages=1,
    )
    a, b = ab.unbind(0)
    outgoing = is_outgoing(direction)
    o = combine(
        a if outgoing else a.transpose(-1, -2), b.transpose(-1, -2) if outgoing else b
    )
    on = normalize(
        o.reshape(batches, c, n * n).transpose(1, 2).reshape(m, c),
        vector(p.norm_out_weight),
        vector(p.norm_out_bias),
        eps,
    )
    output = x.new_empty(m, c)
    wg, wp = p.g_out_weight.t(), p.p_out_weight.t()
    kernels.fused_gemm_sigmoid_multiply[
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
        vector(p.g_out_bias),
        vector(p.p_out_bias),
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
