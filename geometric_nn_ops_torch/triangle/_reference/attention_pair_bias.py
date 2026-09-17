# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch reference math for attention_pair_bias.

Defines the documented semantics of the op and doubles as the numerical oracle
in the tests. Fully autograd-differentiable; imports no Triton. See
``.._attention_pair_bias`` for the layout conventions and the argument
documentation.
"""

import torch
import torch.nn.functional as F

from geometric_nn_ops_torch.triangle._masking import mask_val


def _project_pair_bias(z, ln_z_weight, ln_z_bias, proj_z_weight, proj_z_bias, eps):
    """z ``(B,U,V,Zc)`` -> per-head bias ``(B,H,U,V)``."""
    Zc = z.shape[-1]
    zc = F.layer_norm(z, (Zc,), ln_z_weight, ln_z_bias, eps)  # (B,U,V,Zc)
    pz = F.linear(zc, proj_z_weight, proj_z_bias)  # (B,U,V,H)
    return pz.permute(0, 3, 1, 2).contiguous()  # (B,H,U,V)


def attention_pair_bias_torch(
    s,
    q,
    k,
    v,
    z,
    *,
    proj_g_weight,
    proj_o_weight,
    proj_z_weight=None,
    proj_z_bias=None,
    ln_z_weight=None,
    ln_z_bias=None,
    proj_g_bias=None,
    proj_o_bias=None,
    mask=None,
    scale=None,
    inf=None,
    eps=1e-5,
    z_is_projected=False,
    return_z_proj=False,
):
    """Attention with pair bias, in stock PyTorch."""
    BM, H, U, DH = q.shape
    V = k.shape[2]
    D = H * DH
    B = z.shape[0]
    if BM % B != 0:
        raise ValueError(f"q batch {BM} is not a multiple of z batch {B}")
    M = BM // B
    if scale is None:
        scale = DH**-0.5

    pz = z
    if not z_is_projected:
        pz = _project_pair_bias(
            z, ln_z_weight, ln_z_bias, proj_z_weight, proj_z_bias, eps
        )

    # q/k/v carry the multiplicity folded into their batch. Splitting them back
    # out explicitly to get natural broadcasting behavior.
    qm = q.reshape(B, M, H, U, DH)
    km = k.reshape(B, M, H, V, DH)
    vm = v.reshape(B, M, H, V, DH)
    logits = torch.einsum("bmhud,bmhvd->bmhuv", qm, km) * scale
    logits = logits + pz[:, None]
    if mask is not None:
        mb = mask.shape[0]
        if mb == B:
            keep = mask.reshape(B, 1, 1, 1, V)
        elif mb == BM:
            keep = mask.reshape(B, M, 1, 1, V)
        else:
            raise ValueError(f"mask batch {mb} must be B={B} or BM={BM}")
        if inf is None:
            inf = mask_val(logits.dtype)
        logits = logits + (keep.to(logits.dtype) - 1.0) * inf
    attn = torch.softmax(logits, dim=-1)
    o = torch.einsum("bmhuv,bmhvd->bmhud", attn, vm).reshape(BM, H, U, DH)

    # Merge heads -> (BM, S, D).
    o = o.transpose(1, 2).reshape(BM, U, D).to(s.dtype)

    # Gate from s + output projection.
    g = torch.sigmoid(F.linear(s, proj_g_weight, proj_g_bias))
    out = F.linear(o * g, proj_o_weight, proj_o_bias)

    return out, (pz if return_z_proj else None)
