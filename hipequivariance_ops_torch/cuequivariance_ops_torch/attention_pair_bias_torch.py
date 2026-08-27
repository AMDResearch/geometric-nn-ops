# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shadow of cuequivariance_ops_torch.attention_pair_bias_torch.

cuequivariance_torch deep-imports ``attention_pair_bias`` from this exact module
path, so the module name must match upstream. This is a thin adapter: it maps
cuEquivariance's signature onto the geometric-nn-ops primary surface
(``geometric_nn_ops_torch.triangle.attention_pair_bias``), which owns the
semantics.
"""

import sys

import torch

from geometric_nn_ops_torch.triangle import attention_pair_bias as _apb

# Provided to match cuEquivariance's documented API. Currently we always fall
# back, so listed as a very large value for documentation, but we don't actually
# check its value anywhere.
CUEQ_ATTENTION_PAIR_BIAS_FALLBACK_THRESHOLD = sys.maxsize


def attention_pair_bias(
    s: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor,
    num_heads: int,
    w_proj_z: torch.Tensor | None,
    w_proj_g: torch.Tensor,
    w_proj_o: torch.Tensor,
    w_ln_z: torch.Tensor | None = None,
    b_ln_z: torch.Tensor | None = None,
    b_proj_z: torch.Tensor | None = None,
    b_proj_g: torch.Tensor | None = None,
    b_proj_o: torch.Tensor | None = None,
    inf: float = 1e6,
    eps: float = 1e-5,
    attn_scale: float | None = None,
    return_z_proj: bool = True,
    is_cached_z_proj: bool = False,
):
    if q.shape[1] != num_heads:
        raise ValueError(
            f"num_heads={num_heads} does not match q head dim {q.shape[1]}"
        )
    result = _apb(
        s,
        q,
        k,
        v,
        z,
        proj_g_weight=w_proj_g,
        proj_o_weight=w_proj_o,
        proj_z_weight=w_proj_z,
        proj_z_bias=b_proj_z,
        ln_z_weight=w_ln_z,
        ln_z_bias=b_ln_z,
        proj_g_bias=b_proj_g,
        proj_o_bias=b_proj_o,
        mask=mask,
        scale=attn_scale,
        inf=inf,
        eps=eps,
        z_is_projected=is_cached_z_proj,
        return_z_proj=return_z_proj,
    )
    return result
