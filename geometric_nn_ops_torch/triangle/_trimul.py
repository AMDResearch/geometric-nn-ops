# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Triangle Multiplicative Update operation.

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
from ._kernels.trimul_chunked import trimul_triton_chunked_add_
from ._reference.trimul import trimul_torch

# Dtypes the Triton kernels compile for. Anything else takes the torch path.
_TRITON_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _has_kernel(x: torch.Tensor) -> bool:
    return TRITON_AVAILABLE and x.is_cuda and x.dtype in _TRITON_DTYPES


def _requires_grad(x: torch.Tensor, weights) -> bool:
    return torch.is_grad_enabled() and (
        x.requires_grad or any(w is not None and w.requires_grad for w in weights)
    )


def _needs_torch_path(x: torch.Tensor, weights) -> bool:
    return not _has_kernel(x) or _requires_grad(x, weights)


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
        mask: optional boolean ``[*, N, N]`` pair mask (``True`` = keep).
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

    impl = trimul_torch if _needs_torch_path(x, weights.values()) else trimul_triton
    return impl(x, direction, mask, eps=eps, **weights)


def triangle_multiplicative_update_(
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
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """In-place residual triangle multiplicative update, for inference.

    Adds :func:`triangle_multiplicative_update` of ``x`` to ``x`` in place and
    returns ``x``. Takes the same arguments, plus ``chunk_size``.

    With ``chunk_size``, the update is computed ``chunk_size`` columns at a time
    and written into ``x`` as it goes. Besides ``x`` this holds the ``a``
    projection (``C_hidden / C_z`` the size of ``x``), and chunk-sized
    intermediates. For outgoing, it also keeps a copy of half of ``x``. Smaller
    chunks use less memory. Chunking needs the Triton path; without it, a
    ``chunk_size`` raises. Without ``chunk_size``, peak memory is the same as
    :func:`triangle_multiplicative_update`; adding in place only avoids
    allocating ``x + update`` afterwards.

    Forward only: raises if autograd would record the update, since it
    overwrites ``x``. Use it under ``torch.no_grad()`` or
    ``torch.inference_mode()``.

    ``x`` must not have elements that share memory; as with torch's own in-place
    ops, it is up to the caller to ensure that. A stride-0 dim, as after
    ``expand``, is rejected as a safety net for that common case, but other
    overlapping layouts are not detected and give wrong results.
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
    if _requires_grad(x, weights.values()):
        raise RuntimeError(
            "triangle_multiplicative_update_ overwrites its input and has no "
            "backward; call it under torch.no_grad() or torch.inference_mode()"
        )
    sizes_and_strides = zip(x.shape, x.stride(), strict=True)
    if any(size > 1 and stride == 0 for size, stride in sizes_and_strides):
        raise ValueError(
            "triangle_multiplicative_update_ writes into x, but x has a "
            f"stride-0 dim (shape {tuple(x.shape)}, strides {x.stride()})"
        )

    if chunk_size is None:
        # Add the residual inside the output kernel, as the chunked path does,
        # so we get identical output regardless of chunking.
        impl = trimul_triton if _has_kernel(x) else trimul_torch
        x.copy_(impl(x, direction, mask, eps=eps, residual=x, **weights))
        return x
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not _has_kernel(x):
        raise ValueError(
            "chunk_size needs the Triton path, which is unavailable for a "
            f"{x.dtype} tensor on {x.device} (Triton available: {TRITON_AVAILABLE})"
        )
    trimul_triton_chunked_add_(
        x, direction, mask, chunk_size=chunk_size, eps=eps, **weights
    )
    return x
