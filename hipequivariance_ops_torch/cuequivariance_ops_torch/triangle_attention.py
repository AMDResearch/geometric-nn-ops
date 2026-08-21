# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shadow of cuequivariance_ops_torch.triangle_attention, backed by geometric-nn-ops.

Matches the contract that cuequivariance_torch calls:
    triangle_attention(q, k, v, bias, mask, scale, return_aux)   # positional
where q/k/v are (B, N, H, S, D) (B=1 may be dropped to 4D).

geometric-nn-ops expects q/k/v in (B, N, S, H, D). This shim is the compatibility
boundary, so it transposes the H and S axes on the way in and transposes the
output back to the cueq layout. bias (B, 1, H, S, S) and mask (B, N, 1, 1, S) are
already layout-compatible and pass through unchanged.
"""

import torch

from geometric_nn_ops_torch.triangle import triangle_attention as _ta

# Our kernel handles short sequences fine, so expose a low threshold to prefer
# it.
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
    o = _ta(q, k, v, bias, mask, scale)
    return o.transpose(-2, -3)
