# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Functional form of triangle attention on top of the Triton evoformer kernel.

Exposes the public `triangle_attention(q, k, v, bias, mask, scale)` API.

Shape and dtype handling here mirror openfold-3's
`openfold3/core/model/primitives/attention.py:_triton_evo_attn` — see that
function for the canonical OF3-internal usage of the same kernel. The kernel
runs bf16/fp16/fp32 natively (fp32 accumulators), so q/k/v are passed through
in their input dtype; only unsupported dtypes are cast down to fp32.
Differences from OF3 are the API shape (cueq's `(q, k, v, bias, mask, scale)`
vs OF3's `(q, k, v, biases)` list convention) and a `scale` argument passed
through to the kernel's `softmax_scale` runtime parameter.
"""

import torch

from ._kernels.triangle_attention import triangle_attention_triton


def triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
    dynamic: bool = False,
    use_exp2: bool = False,
) -> torch.Tensor:
    """Triangle attention with a cueq-compatible signature.

    Args:
        q, k, v: shape ``[..., N_seq, N_res, H, D]``. 4D, 5D, or 6D inputs are
            accepted; leading dims are collapsed to a single batch.
        bias: pair bias broadcastable to ``[B, 1, H, N_res, N_res]``, or ``None``
            for no bias (skips all bias work in the kernel).
        mask: boolean tensor where ``True`` keeps and ``False`` masks; shape
            broadcastable to ``[B, N_seq, 1, 1, N_res]``, or ``None`` for
            all-keep.
        scale: softmax scale, passed to the kernel as its ``softmax_scale``
            runtime argument. ``None`` uses the kernel default ``1/sqrt(D)``.
        dynamic: when ``True``, use the variable-shape kernel entry point (one
            Triton compile across all sequence lengths). Trades a small per-call
            cost for no per-size recompile; best for variable-size / cold-cache
            workloads. Output is identical to the default specialized path.
        use_exp2: when ``True``, evaluate the softmax in base 2, which is faster
            on some backends. Mathematically identical to the default base-e
            softmax, but not bit-identical.

    Returns:
        Tensor of the same shape and dtype as ``q``.
    """
    if mask is not None and mask.dtype != torch.bool:
        raise ValueError(f"mask must be bool (keep=True/mask=False), got {mask.dtype}")

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

    # bias -> [B, 1, H, N_res, N_res], mask -> [B, N_seq, 1, 1, N_res].
    if bias is not None:
        if bias.dim() == 4:
            bias = bias.unsqueeze(-4)  # [B, H, N, N] -> [B, 1, H, N, N]
        elif bias.dim() > 5:
            bias = bias.reshape(-1, *bias.shape[-4:])
        bias = bias.expand(Batch, 1, Head, N_res, N_res).contiguous()
    if mask is not None:
        while mask.dim() > 5:
            mask = mask.reshape(-1, *mask.shape[-4:])
        mask = mask.reshape(*mask.shape, *([1] * (5 - mask.dim())))
        mask = mask.broadcast_to(Batch, N_seq, 1, 1, N_res).contiguous()

    # Kernel runs bf16/fp16/fp32 natively (fp32 accumulators); only cast
    # unsupported dtypes (e.g. fp64) down to fp32.
    if orig_dtype in (torch.bfloat16, torch.float16, torch.float32):
        compute_dtype = orig_dtype
    else:
        compute_dtype = torch.float32

    o = triangle_attention_triton(
        q=q.to(compute_dtype),
        k=k.to(compute_dtype),
        v=v.to(compute_dtype),
        bias=None if bias is None else bias.to(compute_dtype),
        mask=mask,
        scale=scale,
        dynamic=dynamic,
        use_exp2=use_exp2,
    )
    return o.to(orig_dtype).reshape(orig_shape)
