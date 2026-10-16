# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Functional attention_pair_bias."""

import torch

from ._reference.attention_pair_bias import attention_pair_bias_torch


def attention_pair_bias(
    s: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    *,
    proj_g_weight: torch.Tensor,
    proj_o_weight: torch.Tensor,
    proj_z_weight: torch.Tensor | None = None,
    ln_z_weight: torch.Tensor | None = None,
    proj_g_bias: torch.Tensor | None = None,
    proj_o_bias: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
    inf: float | None = None,
    eps: float = 1e-5,
    z_is_projected: bool = False,
    return_z_proj: bool = False,
):
    """Attention with pair bias.

    Takes **pre-projected** q, k, v; this op does z-LayerNorm + z-projection ->
    per-head bias, masked attention, gating from the single rep, and the output
    projection. It does *not* do the q/k/v projection or the input LayerNorm on
    the single rep. When return_z_proj is set, returns projected pair bias for
    reuse in subsequent calls with z_is_projected set.

    The z LayerNorm and projection take no bias: softmax cancels the per-head
    constant they would add, so they can't affect the output mathematically (in
    low-precision they can perturb it, usually destructively).

    Args:
        s: ``(BM, S, D)`` single/token representation; the gate is projected from
            it and the output has this shape.
        q: ``(BM, H, U, DH)`` pre-projected queries.
        k, v: ``(BM, H, V, DH)`` pre-projected keys / values.
        z: pair representation ``(B, U, V, Zc)``, or the already-projected
            per-head bias ``(B, H, U, V)`` when ``z_is_projected=True``.
        proj_g_weight: ``(D, D)`` gate projection weight (applied to ``s``).
        proj_o_weight: ``(D, D)`` output projection weight.
        proj_z_weight: ``(H, Zc)`` pair-bias projection weight. Required unless
            ``z_is_projected``.
        ln_z_weight: ``(Zc,)`` LayerNorm affine param for ``z``.
        proj_g_bias: ``(D,)`` optional gate projection bias.
        proj_o_bias: ``(D,)`` optional output projection bias.
        mask: ``(B, V)`` or ``(BM, V)`` key mask, ``1`` = keep, ``0`` = mask.
        scale: attention scale; ``None`` -> ``DH ** -0.5``.
        inf: magnitude used to push masked keys out of the softmax. This exists
            primarily for compatibility with cuequivariance. We recommend using
            the default ``None``, which selects a value suited to the logits
            dtype; a value given here is used as-is even when it would overflow
            the dtype.
        eps: LayerNorm epsilon.
        z_is_projected: when ``True``, ``z`` is the cached ``(B,H,U,V)`` bias and
            the LayerNorm + projection are skipped.
        return_z_proj: when ``True``, also return the per-head bias ``(B, H, U, V)``,
            so it can be cached and reused for future calls with ``z_is_projected``.

    Returns:
        ``(out, proj_z)``:
            - ``out``: ``(BM, S, D)``
            - ``proj_z``:  ``(B, H, U, V)`` projection of ``z`` when ``return_z_proj``,
                else ``None``. Can be reused for future calls with ``z_is_projected``.
                Note that this does not carry the mask, which must be passed in again on
                future calls.

    fp32 matmuls follow ``torch.backends.cuda.matmul.fp32_precision``.
    """
    return attention_pair_bias_torch(
        s,
        q,
        k,
        v,
        z,
        proj_g_weight=proj_g_weight,
        proj_o_weight=proj_o_weight,
        proj_z_weight=proj_z_weight,
        ln_z_weight=ln_z_weight,
        proj_g_bias=proj_g_bias,
        proj_o_bias=proj_o_bias,
        mask=mask,
        scale=scale,
        inf=inf,
        eps=eps,
        z_is_projected=z_is_projected,
        return_z_proj=return_z_proj,
    )
