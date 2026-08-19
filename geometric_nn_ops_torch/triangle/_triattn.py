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

from ._kernels.triangle_attention import TritonEvoformer

NEG_INF = -1e9


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
            broadcastable to ``[B, N_seq, 1, 1, N_res]``. Additive masks
            (0 = keep, large-negative = mask) are also accepted. ``None`` means
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

    # bias -> [B, 1, H, N_res, N_res]. When bias is None, expand a single scalar
    # zero (stride-0) instead of allocating; has_pair_bias=False then skips all
    # bias work in the kernel, so its values are never read.
    has_pair_bias = bias is not None
    if has_pair_bias:
        if bias.dim() == 4:
            bias = bias.unsqueeze(-4)  # [B, H, N, N] -> [B, 1, H, N, N]
        elif bias.dim() > 5:
            bias = bias.reshape(-1, *bias.shape[-4:])
        pair_bias = bias.expand(Batch, 1, Head, N_res, N_res).contiguous()
    else:
        pair_bias = q.new_zeros(()).expand(Batch, 1, Head, N_res, N_res)

    # mask -> additive [B, N_seq, 1, 1, N_res]. None means all-keep; expand a
    # single scalar zero (stride-0) rather than materializing a full mask.
    if mask is None:
        res_mask = q.new_zeros(()).expand(Batch, N_seq, 1, 1, N_res)
    else:
        # bool -> additive ([NEG_INF, 0]); else assume already additive.
        if mask.dtype == torch.bool:
            # True = keep (0), False = mask (NEG_INF)
            res_mask = (~mask).to(torch.float32) * NEG_INF
        else:
            res_mask = mask.to(torch.float32)
        # Collapse leading dims of mask to match Batch
        while res_mask.dim() > 5:
            res_mask = res_mask.reshape(-1, *res_mask.shape[-4:])
        # Broadcast to [B, N_seq, 1, 1, N_res]
        res_mask = res_mask.reshape(*res_mask.shape, *([1] * (5 - res_mask.dim())))
        res_mask = res_mask.broadcast_to(Batch, N_seq, 1, 1, N_res).contiguous()

    # Kernel runs bf16/fp16/fp32 natively (fp32 accumulators), so pass q/k/v
    # through as-is for those dtypes; only cast unsupported dtypes (e.g. fp64)
    # down to fp32. res_mask/pair_bias are cast to match the compute dtype.
    if orig_dtype in (torch.bfloat16, torch.float16, torch.float32):
        compute_dtype = orig_dtype
    else:
        compute_dtype = torch.float32
    q_, k_, v_ = q.to(compute_dtype), k.to(compute_dtype), v.to(compute_dtype)
    rm_, pb_ = res_mask.to(compute_dtype), pair_bias.to(compute_dtype)

    o = TritonEvoformer(q_, k_, v_, rm_, pb_, has_pair_bias, dynamic, scale, use_exp2)
    o = o.to(orig_dtype)

    return o.reshape(orig_shape)
