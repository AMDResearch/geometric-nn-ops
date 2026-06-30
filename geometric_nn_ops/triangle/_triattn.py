# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Triangle attention adapter on top of the Triton evoformer kernel.

Exposes a `triangle_attention(q, k, v, bias, mask, scale)` function with the
same calling convention as
`cuequivariance_torch.primitives.triangle.triangle_attention`.

Shape and dtype handling here mirror openfold-3's
`openfold3/core/model/primitives/attention.py:_triton_evo_attn` (commit
90162862 on the `amd-rocm-inference` branch) — see that function for the
canonical OF3-internal usage of the same kernel. Differences are the API
shape (cueq's `(q, k, v, bias, mask, scale)` vs OF3's `(q, k, v, biases)`
list convention) and a `scale` argument that pre-multiplies `q` so the
kernel's built-in `1/sqrt(D)` yields the user-requested scale.
"""

import math

import torch

from ._kernels.triangle_attention import TritonEvoformer

NEG_INF = -1e9


def triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Triangle attention with a cueq-compatible signature.

    Args:
        q, k, v: shape ``[..., N_seq, N_res, H, D]``. 4D, 5D, or 6D inputs are
            accepted; leading dims are collapsed to a single batch.
        bias: pair bias broadcastable to ``[B, 1, H, N_res, N_res]``.
        mask: boolean tensor where ``True`` keeps and ``False`` masks; shape
            broadcastable to ``[B, N_seq, 1, 1, N_res]``. Additive masks
            (0 = keep, large-negative = mask) are also accepted.
        scale: softmax scale. If ``None`` or equal to ``1/sqrt(D)``, the
            kernel's built-in scale runs unchanged. Otherwise ``q`` is
            pre-multiplied so the kernel's built-in scale yields the
            requested value.

    Returns:
        Tensor of the same shape and dtype as ``q``.
    """
    if TritonEvoformer is None:
        raise RuntimeError(
            "Triton kernel unavailable. Ensure `triton` is installed and "
            "ROCm / CUDA is visible."
        )

    orig_shape = q.shape
    orig_dtype = q.dtype

    # Normalize q/k/v to 5D: (Batch, N_seq, N_res, H, D)
    if q.dim() < 5:
        pad = 5 - q.dim()
        q = q.reshape((1,) * pad + tuple(q.shape))
        k = k.reshape((1,) * pad + tuple(k.shape))
        v = v.reshape((1,) * pad + tuple(v.shape))
    elif q.dim() > 5:
        q = q.reshape(-1, *q.shape[-4:])
        k = k.reshape(-1, *k.shape[-4:])
        v = v.reshape(-1, *v.shape[-4:])

    Batch, N_seq, N_res, Head, Dim = q.shape

    # bias -> [B, 1, H, N_res, N_res]
    if bias.dim() == 4:
        bias = bias.unsqueeze(-4)  # [B, H, N, N] -> [B, 1, H, N, N]
    elif bias.dim() > 5:
        bias = bias.reshape(-1, *bias.shape[-4:])
    pair_bias = bias.expand(Batch, 1, Head, N_res, N_res).contiguous()

    # mask: bool -> additive ([NEG_INF, 0]); reshape/broadcast to [B, N_seq, 1, 1, N_res]
    if mask.dtype == torch.bool:
        res_mask = torch.where(
            mask, torch.tensor(0.0, device=mask.device), torch.tensor(NEG_INF, device=mask.device)
        ).to(torch.float32)
    else:
        res_mask = mask.to(torch.float32)
    # Collapse leading dims of mask to match Batch
    while res_mask.dim() > 5:
        res_mask = res_mask.reshape(-1, *res_mask.shape[-4:])
    # Broadcast to [B, N_seq, 1, 1, N_res]
    res_mask = res_mask.reshape(*res_mask.shape, *([1] * (5 - res_mask.dim())))
    res_mask = res_mask.broadcast_to(Batch, N_seq, 1, 1, N_res).contiguous()

    # If user passed a non-default scale, pre-multiply q so kernel's
    # built-in scale = 1/sqrt(D) yields the requested scale.
    default_scale = Dim ** -0.5
    if scale is not None and not math.isclose(scale, default_scale, rel_tol=1e-6):
        q = q * (scale / default_scale)

    # Kernel requires fp16 / bf16
    if orig_dtype not in (torch.bfloat16, torch.float16):
        target = torch.bfloat16
        q_, k_, v_ = q.to(target), k.to(target), v.to(target)
        rm_, pb_ = res_mask.to(target), pair_bias.to(target)
    else:
        q_, k_, v_ = q, k, v
        rm_, pb_ = res_mask.to(orig_dtype), pair_bias.to(orig_dtype)

    o = TritonEvoformer(q_, k_, v_, rm_, pb_)
    o = o.to(orig_dtype)

    return o.reshape(orig_shape)
