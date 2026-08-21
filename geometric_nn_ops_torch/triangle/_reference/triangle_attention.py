# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch triangle_attention for the canonical 5D layout."""

import torch


def mask_val(dtype: torch.dtype) -> float:
    """How far a masked key's logit is pushed below zero, for ``dtype``.

    Shared with the Triton kernel so both fold a boolean mask identically.

    ``1e9`` for fp32 and bf16. fp16 cannot represent it and would get ``inf`` --
    a fully masked key *tile* then takes ``-inf`` as its running maximum and the
    online softmax evaluates ``-inf - -inf``, giving NaNs even for rows that are
    not fully masked -- so it takes half the dtype maximum instead, which still
    squashes any reasonable logit. fp64 needs a larger value than fp32 to behave
    the same way: at ``1e9`` its ulp is far below ``log(SEQ_LEN)``, so a fully
    masked row would keep its logsumexp term where fp32 loses it, and the two
    would disagree.
    """
    if dtype == torch.float64:
        return 1e18
    elif dtype == torch.float32 or dtype == torch.bfloat16:
        return 1e9
    elif dtype == torch.float16:
        return torch.finfo(dtype).max / 2
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def masked_key_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Boolean mask -> additive bias, ``0`` where kept and ``-mask_val`` where not."""
    # Note that we deliberately spell this the same way openfold-3 has
    # historically, so that for now we can get bitwise identical results. A
    # masked fill or `(~mask) * -mask_val` would be more natural, but would yield
    # -0.0 on kept entries rather than +0.0, which is enough to shift bf16 and
    # fp16 results.
    return mask_val(dtype) * (mask.to(torch.float32) - 1)


def triangle_attention_torch(q, k, v, bias=None, mask=None, scale=None):
    """Triangle attention in stock PyTorch.

    Args:
        q, k, v: ``[B, N_seq, N_res, H, D]``.
        bias: broadcastable to ``[B, 1, H, N_res, N_res]``; a ``[B, H, N, N]``
            bias is unsqueezed to add the ``N_seq`` axis. ``None`` for no bias.
        mask: boolean, broadcastable to ``[B, N_seq, 1, 1, N_res]``; ``True``
            keeps. ``None`` means all-keep.
        scale: softmax scale; ``None`` -> ``D ** -0.5``.

    Returns:
        Tensor of the same shape and dtype as ``q``.
    """
    dim = q.shape[-1]
    if scale is None:
        scale = dim**-0.5
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32

    qs = q.transpose(-2, -3).to(acc)  # [B, N_seq, H, N_res, D]
    ks = k.transpose(-2, -3).to(acc)
    vs = v.transpose(-2, -3).to(acc)

    logits = torch.einsum("bshid,bshjd->bshij", qs * scale, ks)
    if bias is not None:
        if bias.dim() == 4:
            bias = bias.unsqueeze(-4)  # [B, H, N, N] -> [B, 1, H, N, N]
        logits = logits + bias.to(acc)
    if mask is not None:
        logits = logits + masked_key_bias(mask, acc).to(acc)

    probs = torch.softmax(logits, dim=-1)
    o = torch.einsum("bshij,bshjd->bshid", probs, vs)
    return o.transpose(-2, -3).to(q.dtype)
