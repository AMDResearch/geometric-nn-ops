# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Functional triangle_multiplicative_update.

Two backends with identical math:
  * Triton fast path (forward-only) using the fused kernels in ``_kernels.trimul``.
  * Pure-torch fallback that is fully autograd-differentiable.

Dispatch picks the torch path when a backward could run (grad is enabled and
some input requires it), the tensor is not on CUDA, its dtype has no kernel, or
Triton is unavailable; otherwise the Triton path runs. The torch path doubles as
the numerical correctness oracle.

``p_in_weight`` / ``g_in_weight`` are the concatenated ``[a; b]`` projections
(shape ``[2 * C_hidden, C_z]``), matching cuequivariance / Boltz fusion.
"""

import torch

from ._kernels.trimul import TRITON_AVAILABLE, trimul_triton
from ._reference.trimul import trimul_torch

__all__ = ["triangle_multiplicative_update"]


# Dtypes the Triton kernels compile for. Anything else takes the torch path.
_TRITON_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _needs_torch_path(x: torch.Tensor, weights) -> bool:
    if not TRITON_AVAILABLE or not x.is_cuda or x.dtype not in _TRITON_DTYPES:
        return True
    return torch.is_grad_enabled() and (
        x.requires_grad or any(w is not None and w.requires_grad for w in weights)
    )


def triangle_multiplicative_update(
    x: torch.Tensor,
    direction: str = "outgoing",
    mask: torch.Tensor | None = None,
    norm_in_weight: torch.Tensor | None = None,
    norm_in_bias: torch.Tensor | None = None,
    p_in_weight: torch.Tensor | None = None,
    p_in_bias: torch.Tensor | None = None,
    g_in_weight: torch.Tensor | None = None,
    g_in_bias: torch.Tensor | None = None,
    norm_out_weight: torch.Tensor | None = None,
    norm_out_bias: torch.Tensor | None = None,
    p_out_weight: torch.Tensor | None = None,
    p_out_bias: torch.Tensor | None = None,
    g_out_weight: torch.Tensor | None = None,
    g_out_bias: torch.Tensor | None = None,
    eps: float = 1e-5,
    precision=None,
) -> torch.Tensor:
    """Triangle multiplicative update over a pair representation.

    Args:
        x: ``[*, N, N, C_z]`` pair representation.
        direction: ``"outgoing"`` or ``"incoming"``.
        mask: optional ``[*, N, N]`` pair mask (1 = keep).
        *_weight / *_bias: layernorm/projection/gate parameters. ``p_in_weight`` and
            ``g_in_weight`` are concatenated ``[a; b]`` (``[2 * C_hidden, C_z]``).
        eps: layernorm epsilon.
        precision: accepted for cueq compatibility; currently ignored (kernels
            accumulate in fp32).

    Returns:
        ``[*, N, N, C_z]`` updated pair representation.
    """
    if direction not in ("outgoing", "incoming"):
        raise ValueError(
            f"direction must be 'outgoing' or 'incoming', got {direction!r}"
        )
    outgoing = direction == "outgoing"
    c_hidden = p_in_weight.shape[0] // 2

    m = mask.unsqueeze(-1) if mask is not None else None

    weights = (
        norm_in_weight,
        norm_in_bias,
        p_in_weight,
        p_in_bias,
        g_in_weight,
        g_in_bias,
        norm_out_weight,
        norm_out_bias,
        p_out_weight,
        p_out_bias,
        g_out_weight,
        g_out_bias,
    )

    if _needs_torch_path(x, weights):
        return trimul_torch(
            x,
            outgoing,
            m,
            c_hidden,
            eps,
            norm_in_weight,
            norm_in_bias,
            p_in_weight,
            p_in_bias,
            g_in_weight,
            g_in_bias,
            norm_out_weight,
            norm_out_bias,
            p_out_weight,
            p_out_bias,
            g_out_weight,
            g_out_bias,
        )
    return trimul_triton(
        x,
        outgoing,
        m,
        c_hidden,
        eps,
        norm_in_weight,
        norm_in_bias,
        p_in_weight,
        p_in_bias,
        g_in_weight,
        g_in_bias,
        norm_out_weight,
        norm_out_bias,
        p_out_weight,
        p_out_bias,
        g_out_weight,
        g_out_bias,
    )
