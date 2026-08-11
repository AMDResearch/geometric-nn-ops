# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch triangle_attention for the canonical 5D layout."""

import torch


def triangle_attention_torch(q, k, v, bias=None, mask=None, scale=None):
    """Triangle attention in stock PyTorch.

    Args:
        q, k, v: ``[B, N_seq, N_res, H, D]``.
        bias: broadcastable to ``[B, 1, H, N_res, N_res]``; a ``[B, H, N, N]``
            bias is unsqueezed to add the ``N_seq`` axis. ``None`` for no bias.
        mask: broadcastable to ``[B, N_seq, 1, 1, N_res]``; boolean (``True``
            keeps) or already additive. ``None`` means all-keep.
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
        if mask.dtype == torch.bool:
            logits = logits.masked_fill(~mask, float("-inf"))
        else:
            logits = logits + mask.to(acc)

    probs = torch.softmax(logits, dim=-1)
    o = torch.einsum("bshij,bshjd->bshid", probs, vs)
    return o.transpose(-2, -3).to(q.dtype)
