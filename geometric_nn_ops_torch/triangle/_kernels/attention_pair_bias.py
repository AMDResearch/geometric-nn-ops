# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Triton-backed attention_pair_bias.

The attention core runs on the triangle-attention flash kernel. The pair bias
comes from a Triton LayerNorm and an ``F.linear``, or from one fused Triton
kernel when no gradient is needed. The mask fold, gating and output projection
are eager torch. Everything is autograd-differentiable.
"""

import torch
import torch.nn.functional as F

from geometric_nn_ops_torch.triangle._triattn import triangle_attention

from .fused_ln_proj import TRITON_AVAILABLE, fused_ln_proj_bias
from .layernorm import triton_layer_norm

__all__ = ["TRITON_AVAILABLE", "attention_pair_bias_triton"]


def _composed_pair_bias(z, ln_z_weight, proj_z_weight, eps):
    """LayerNorm + projection + permute, differentiable."""
    if ln_z_weight is not None:
        zc = triton_layer_norm(z, ln_z_weight, eps)
    else:
        zc = F.layer_norm(z, (z.shape[-1],), None, None, eps)
    pz = F.linear(zc, proj_z_weight)  # (B,U,V,H)
    return pz.permute(0, 3, 1, 2).contiguous()  # (B,H,U,V)


def _project_pair_bias(z, ln_z_weight, proj_z_weight, eps):
    """z ``(B,U,V,Zc)`` -> per-head bias ``(B,H,U,V)``.

    Without a gradient, one fused kernel does the LayerNorm, the projection and
    the ``(B,H,U,V)`` scatter. With one, the composed path is used: its autograd
    backward, which reuses the saved LayerNorm output and torch's linear
    backward, measured faster than a fused backward on MI300X.

    The fused kernel needs the LayerNorm weight; without it (cueq's no-affine
    case) the composed path is used.
    """
    grad_needed = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (z, ln_z_weight, proj_z_weight)
    )
    if ln_z_weight is not None and not grad_needed:
        return fused_ln_proj_bias(z, ln_z_weight, proj_z_weight, eps)
    return _composed_pair_bias(z, ln_z_weight, proj_z_weight, eps)


def _key_mask_additive(mask, inf, dtype):
    return ((mask.to(dtype) - 1.0) * inf)[:, None, None, :]  # (batch,1,1,V)


def attention_pair_bias_triton(
    s,
    q,
    k,
    v,
    z,
    *,
    proj_g_weight,
    proj_o_weight,
    proj_z_weight=None,
    ln_z_weight=None,
    proj_g_bias=None,
    proj_o_bias=None,
    mask=None,
    scale=None,
    inf=None,
    eps=1e-5,
    z_is_projected=False,
    return_z_proj=False,
):
    """Attention with pair bias on the Triton kernels.

    Takes the same arguments as ``attention_pair_bias_torch``; see the public
    op for their documentation. Square attention (``U == V``) only.
    """
    BM, H, U, DH = q.shape
    V = k.shape[2]
    if U != V:
        raise ValueError(f"the Triton path needs U == V, got U={U}, V={V}")
    D = H * DH
    B = z.shape[0]
    if BM % B != 0:
        raise ValueError(f"q batch {BM} is not a multiple of z batch {B}")
    M = BM // B

    pz = z
    if not z_is_projected:
        pz = _project_pair_bias(z, ln_z_weight, proj_z_weight, eps)

    # Broadcast over the multiplicity to BM.
    bias = pz
    if M != 1:
        bias = pz.reshape(B, 1, H, U, V).expand(B, M, H, U, V).reshape(BM, H, U, V)
    key_mask = None
    if mask is not None:
        mb = mask.shape[0]
        if mb == B and M != 1:
            m_bm = mask.reshape(B, 1, V).expand(B, M, V).reshape(BM, V)
        elif mb in (B, BM):
            m_bm = mask
        else:
            raise ValueError(f"mask batch {mb} must be B={B} or BM={BM}")
        if inf is None:
            # The kernel masks with the same default magnitude as the reference.
            key_mask = (m_bm != 0).reshape(BM, 1, 1, 1, V)
        else:
            bias = bias + _key_mask_additive(m_bm, inf, bias.dtype)

    # The triangle-attention layout is (Batch, N_seq, N_res, H, D); fold M into
    # Batch with N_seq=1.
    q5 = q.permute(0, 2, 1, 3).unsqueeze(1)  # (BM,1,U,H,DH)
    k5 = k.permute(0, 2, 1, 3).unsqueeze(1)  # (BM,1,V,H,DH)
    v5 = v.permute(0, 2, 1, 3).unsqueeze(1)
    o5 = triangle_attention(q5, k5, v5, bias=bias[:, None], mask=key_mask, scale=scale)
    o = o5.squeeze(1).reshape(BM, U, D).to(s.dtype)  # merge heads

    g = torch.sigmoid(F.linear(s, proj_g_weight, proj_g_bias))
    out = F.linear(o * g, proj_o_weight, proj_o_bias)

    return out, (pz if return_z_proj else None)
