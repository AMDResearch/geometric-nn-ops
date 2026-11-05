# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Functional attention_pair_bias.

Two backends with identical math: the Triton path in
``_kernels.attention_pair_bias`` and the pure-torch reference, which also serves
as the numerical oracle in the tests. Dispatch picks the torch path off CUDA,
without Triton, for fp64, for non-square attention, and below a size where the
Triton path's fixed launch overhead outweighs its savings.
"""

import torch

from ._kernels.attention_pair_bias import TRITON_AVAILABLE, attention_pair_bias_triton
from ._reference.attention_pair_bias import attention_pair_bias_torch

__all__ = ["attention_pair_bias"]

# Dtypes the Triton kernels compile for. Their accumulators are fp32 and would
# throw away fp64's precision, so fp64 takes the torch path.
_TRITON_DTYPES = (torch.float32, torch.bfloat16, torch.float16)

# Size gate. The Triton path has a roughly fixed launch-overhead floor, so at
# small shapes the torch path, whose cost scales with the work, is faster. The
# crossover is captured by an estimate of the N^2-scaling part of the work:
#
#     work = U * V * (Zc + M * H)
#
# i.e. the z LayerNorm (U*V*Zc) plus the attention (U*V*M*H). Zc drops out when
# z is already projected. Training has a higher threshold because the backward
# adds kernels. Calibrated on MI300X (H=16, DH=48, Zc=128) for U == V.
_TRAIN_TRITON_MIN_WORK = 55_000_000
_INFER_TRITON_MIN_WORK = 40_000_000


def _use_torch_path(q, k, z, z_is_projected: bool, training: bool) -> bool:
    """Whether to run the torch reference rather than the Triton path."""
    if not TRITON_AVAILABLE or not q.is_cuda or q.dtype not in _TRITON_DTYPES:
        return True
    BM, H, U, _ = q.shape
    V = k.shape[2]
    if U != V:
        return True
    M = BM // z.shape[0]
    Zc = 0 if z_is_projected else z.shape[-1]
    work = U * V * (Zc + M * H)
    threshold = _TRAIN_TRITON_MIN_WORK if training else _INFER_TRITON_MIN_WORK
    return work < threshold


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
    tensors = (
        s,
        q,
        k,
        v,
        z,
        proj_g_weight,
        proj_o_weight,
        proj_z_weight,
        ln_z_weight,
        proj_g_bias,
        proj_o_bias,
    )
    training = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in tensors
    )
    use_torch = _use_torch_path(q, k, z, z_is_projected, training)
    impl = attention_pair_bias_torch if use_torch else attention_pair_bias_triton
    return impl(
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
