# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Implements the API of cuequivariance_ops_torch.triangle_attention, backed by geometric-nn-ops.

geometric-nn-ops expects q/k/v in (B, N, S, H, D), so this shim transposes the H
and S axes on the way in and transposes the output back to the cueq layout. bias
(B, 1, H, S, S) and mask (B, N, 1, 1, S) are already layout-compatible and pass
through unchanged.

The result is always rank 5, matching cueq. geometric-nn-ops preserves the input
rank, so the dropped axis is restored here.
"""  # noqa: E501

import torch

from geometric_nn_ops_torch.triangle import triangle_attention as _ta

# Provided to match cuEquivariance's documented API. Our kernel handles short
# sequences fine, so indicate a low threshold to prefer it. We do not actually
# use this value to trigger a fallback.
CUEQ_TRIATTN_FALLBACK_THRESHOLD = 16


def triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
    return_aux: bool = False,
):
    if return_aux:
        raise NotImplementedError(
            "hipequivariance_ops_torch: triangle_attention(return_aux=True) is "
            "not supported yet (the Triton kernel does not expose lse/max)."
        )
    # cueq (B, N, H, S, D) -> geometric-nn-ops (B, N, S, H, D); output back.
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)
    # cueq documents the mask as "cast to bool internally"; our op requires bool.
    if mask is not None:
        mask = mask.bool()
    o = _ta(q, k, v, bias, mask, scale).transpose(-2, -3)
    # cueq pads a sub-rank-5 input up to rank 5 and returns it that way; our op
    # returns the rank it was given.
    if o.dim() < 5:
        pad = 5 - o.dim()
        o = o.reshape((1,) * pad + tuple(o.shape))
    return o
