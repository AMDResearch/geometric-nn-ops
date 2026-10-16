# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Implements the API of cuequivariance_ops_torch.attention_pair_bias_torch, backed by geometric-nn-ops.

cuequivariance_torch deep-imports ``attention_pair_bias`` from this exact module
path, so the file name must match.

The shim warns about and ignores ``b_ln_z`` and ``b_proj_z`` which
mathematically cannot affect the main output and just add noise. cuEquivariance
folds them into its returned ``z_proj``, so with either set ours differs from
theirs by a per-head constant, but is internally consistent for subsequent
function calls.
"""  # noqa: E501

import sys
import warnings

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
    for name, bias in (("b_ln_z", b_ln_z), ("b_proj_z", b_proj_z)):
        if bias is not None:
            warnings.warn(
                f"{name} is ignored: softmax cancels the per-head constant it "
                "adds, so it mathematically cannot affect the output and its gradient "
                "should be zero, but in low precision can diverge. Don't pass it in "
                "to this function and we recommend not training it at all.",
                stacklevel=2,
            )
    del b_ln_z, b_proj_z

    result = _apb(
        s,
        q,
        k,
        v,
        z,
        proj_g_weight=w_proj_g,
        proj_o_weight=w_proj_o,
        proj_z_weight=w_proj_z,
        ln_z_weight=w_ln_z,
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
