# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Functional triangle_multiplicative_update.

Two backends with identical math:
  * A Triton path using the fused kernels in ``_kernels.trimul``.
  * A pure-torch fallback that is fully general.

The fused kernels implement the standard config only, so dispatch picks the
torch path for anything outside it, when Triton isn't available, or for a dtype
that has no kernel. The torch path doubles as the numerical correctness oracle.

``p_in_weight`` / ``g_in_weight`` are the concatenated ``[a; b]`` projections
(shape ``[2 * C_hidden, C_z]``), matching cuequivariance / Boltz fusion.
"""

import torch

from ._kernels.trimul import TRITON_AVAILABLE, trimul_triton
from ._reference.trimul import trimul_torch

__all__ = ["triangle_multiplicative_update"]


# Dtypes the Triton kernels compile for. fp64 has no MFMA path and the fp32
# accumulators would throw its precision away, so it takes the torch path.
_TRITON_DTYPES = (torch.float32, torch.bfloat16, torch.float16)

# Widest hidden dim the forward LayerNorm covers, which reads one BLOCK_C block.
_MAX_TRITON_HIDDEN = 128

BIAS_KEYS = ("p_in_bias", "g_in_bias", "p_out_bias", "g_out_bias")


def _needs_torch_path(x: torch.Tensor, mask, weights: dict) -> bool:
    """Whether the fused kernels can serve this call.

    They implement the standard config, so everything else goes to the general
    torch path: a hidden dim that is not square with the channel dim or is
    wider than one LayerNorm block, a projection bias (the fused GEMMs carry
    none), and a mask needing a gradient (the kernels treat it as a constant
    and return none for it).
    """
    if not TRITON_AVAILABLE or not x.is_cuda or x.dtype not in _TRITON_DTYPES:
        return True
    if weights["p_in_weight"].shape[0] // 2 != x.shape[-1]:
        return True
    if x.shape[-1] > _MAX_TRITON_HIDDEN:
        return True
    if any(weights[key] is not None for key in BIAS_KEYS):
        return True
    return mask is not None and mask.requires_grad


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
) -> torch.Tensor:
    """Triangle multiplicative update over a pair representation.

    Args:
        x: ``[*, N, N, C_z]`` pair representation.
        direction: ``"outgoing"`` or ``"incoming"``.
        mask: optional ``[*, N, N]`` pair mask (1 = keep).
        *_weight / *_bias: layernorm/projection/gate parameters. ``p_in_weight`` and
            ``g_in_weight`` are concatenated ``[a; b]`` (``[2 * C_hidden, C_z]``).
        eps: layernorm epsilon.

    Returns:
        ``[*, N, N, C_z]`` updated pair representation.

    fp32 matmuls follow ``torch.backends.cuda.matmul.fp32_precision``.
    """
    weights = dict(
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        p_in_weight=p_in_weight,
        p_in_bias=p_in_bias,
        g_in_weight=g_in_weight,
        g_in_bias=g_in_bias,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        p_out_bias=p_out_bias,
        g_out_weight=g_out_weight,
        g_out_bias=g_out_bias,
    )

    impl = trimul_torch if _needs_torch_path(x, mask, weights) else trimul_triton
    return impl(x, direction, mask, eps=eps, **weights)
