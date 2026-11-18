# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prepared triangle attention forward kernel."""

import triton
import triton.language as tl

from .triangle_attention import _attn_fwd_inner


@triton.jit
def _attention(
    Q,
    K,
    V,
    MASK,
    BIAS,
    GATE,
    O,
    M,
    SQ0: tl.constexpr,
    SQ1: tl.constexpr,
    SQ2: tl.constexpr,
    SQ3: tl.constexpr,
    SQ4: tl.constexpr,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    SK2: tl.constexpr,
    SK3: tl.constexpr,
    SK4: tl.constexpr,
    SV0: tl.constexpr,
    SV1: tl.constexpr,
    SV2: tl.constexpr,
    SV3: tl.constexpr,
    SV4: tl.constexpr,
    SO0: tl.constexpr,
    SO1: tl.constexpr,
    SO2: tl.constexpr,
    SO3: tl.constexpr,
    SO4: tl.constexpr,
    SG0: tl.constexpr,
    SG1: tl.constexpr,
    SG2: tl.constexpr,
    SG3: tl.constexpr,
    SG4: tl.constexpr,
    SM1: tl.constexpr,
    SM4: tl.constexpr,
    SB2: tl.constexpr,
    SB3: tl.constexpr,
    SB4: tl.constexpr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    FUSE_GATE: tl.constexpr,
):
    # Adjacent programs share a head/query tile and vary the triangle row.
    pid = tl.program_id(0)
    row = (pid % ROWS).to(tl.int64)
    rest = pid // ROWS
    query = rest % tl.cdiv(N, 64)
    head = (rest // tl.cdiv(N, 64)).to(tl.int64)
    q = (query * 64 + tl.arange(0, 64)).to(tl.int64)
    k = tl.arange(0, 32).to(tl.int64)
    d = tl.arange(0, BLOCK_DIM).to(tl.int64)
    qp = Q + row * SQ1 + head * SQ2 + q[:, None] * SQ3 + d[None, :] * SQ4
    kp = K + row * SK1 + head * SK2 + d[:, None] * SK4 + k[None, :] * SK3
    vp = V + row * SV1 + head * SV2 + k[:, None] * SV3 + d[None, :] * SV4
    bp = BIAS + head * SB2 + q[:, None] * SB3 + k[None, :] * SB4
    mp = MASK + row * SM1 + k[None, :] * SM4
    qb = tl.load(qp, (q[:, None] < N) & (d[None, :] < DIM), 0.0)
    acc = tl.zeros((64, BLOCK_DIM), tl.float32)
    denominator = tl.full((64,), 1.0, tl.float32)
    maximum = tl.full((64,), float("-inf"), tl.float32)
    acc, denominator, maximum = _attn_fwd_inner(
        acc,
        denominator,
        maximum,
        qb,
        kp,
        vp,
        mp,
        bp,
        query,
        DIM,
        SK3,
        SV3,
        SM4,
        SB4,
        SCALE,
        N % 64 == 0,
        N % 32 == 0,
        DIM == BLOCK_DIM,
        True,
        64,
        32,
        BLOCK_DIM,
        q,
        k,
        d,
        N,
        "ieee",
        False,
    )
    maximum += tl.math.log(denominator)
    acc = acc / denominator[:, None]
    if FUSE_GATE:
        gp = GATE + row * SG1 + head * SG2 + q[:, None] * SG3 + d[None, :] * SG4
        acc *= tl.load(gp, (q[:, None] < N) & (d[None, :] < DIM), 0.0)
    op = O + row * SO1 + head * SO2 + q[:, None] * SO3 + d[None, :] * SO4
    tl.store(op, acc, (q[:, None] < N) & (d[None, :] < DIM))
    tl.store(M + (row * HEADS + head) * N + q, maximum, q < N)
