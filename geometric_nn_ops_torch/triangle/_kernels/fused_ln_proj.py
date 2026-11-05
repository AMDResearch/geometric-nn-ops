# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fused z-LayerNorm + pair-bias projection for attention_pair_bias.

The pair bias is ``LN(z) @ proj_z_weight.T`` laid out as ``(B,H,U,V)``. Composed,
that is three ops -- a LayerNorm over ``(B,U,V,Zc)``, an ``F.linear`` to
``(B,U,V,H)`` and a ``permute(0,3,1,2)`` copy -- with both intermediates
round-tripping through memory, on an op bound by the ``z`` traffic.

This kernel does the three in one pass: each program LayerNorms a block of
``BLOCK_V`` contiguous ``(u,v)`` rows of ``z`` in registers, projects them
against the ``(H,Zc)`` weight with ``tl.dot``, and writes the ``[H,BLOCK_V]``
tile straight to the output. H is small (typically 4-16), so the projection
costs little next to the ``z`` reads. A v-block is a contiguous run of the
output's fast axis, so each head's stores coalesce.

Forward only: a fused backward measured slower than letting autograd
differentiate the composed path, which reuses the saved LayerNorm output and
torch's linear backward. So the LayerNorm statistics are never written out,
``fused_ln_proj_bias`` rejects grad-requiring inputs, and the caller uses the
composed path when gradients are needed.
"""

import torch

from geometric_nn_ops_torch import _precision

from .layernorm import active_autocast_dtype

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def _cfg_fused():
    return {"BLOCK_V": 32, "num_warps": 4, "num_stages": 2}


if TRITON_AVAILABLE:

    @triton.jit
    def _ln_proj_fwd(
        Z,  # (B*U*V, Zc) row-major, contiguous
        Wln,  # (Zc,)
        Wp,  # (H, Zc) projection weight (row-major, contiguous)
        Out,  # (B, H, U, V) output bias
        stride_z,
        stride_ob,  # Out stride for the B axis (== H*U*V)
        stride_oh,  # Out stride for the H axis (== U*V)
        stride_ouv,  # Out stride for the flattened (u,v) axis (== 1, contiguous)
        total_rows,
        UV,  # U*V (rows per batch)
        Zc,
        H,
        eps,
        BLOCK_V: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_H: tl.constexpr,
        MATMUL_PRECISION: tl.constexpr,
    ):
        """Fused LayerNorm(z) + projection -> per-head bias in (B,H,U,V).

        Grid is 1-D over blocks of ``BLOCK_V`` contiguous ``(u,v)`` rows; the
        ``(u,v)`` plane is row-major, so those rows are a contiguous run of the
        output's fast axis. Each program loads a ``[BLOCK_V, Zc]`` tile of ``z``,
        LayerNorms each row in registers, projects to ``[BLOCK_V, H]`` via
        ``tl.dot`` against ``Wp``, then transposes to ``[H, BLOCK_V]`` so the
        store's innermost axis is v (unit output stride) and each head lands as
        one coalesced run.
        """
        pid = tl.program_id(0)
        # i64 throughout: at N >= ~5k the row index and the H*U*V output offsets
        # exceed 2**31.
        row_start = pid.to(tl.int64) * BLOCK_V
        v_off = tl.arange(0, BLOCK_V)
        rows = row_start + v_off.to(tl.int64)
        rmask = rows < total_rows

        c_offsets = tl.arange(0, BLOCK_C)
        cmask = c_offsets < Zc
        h_offsets = tl.arange(0, BLOCK_H)
        hmask = h_offsets < H

        wln = tl.load(Wln + c_offsets, mask=cmask, other=0.0).to(tl.float32)
        # Masked lanes are 0 so they drop out of the dot product.
        wp = tl.load(
            Wp + h_offsets[:, None] * Zc + c_offsets[None, :],
            mask=hmask[:, None] & cmask[None, :],
            other=0.0,
        ).to(tl.float32)

        Zc_f = Zc.to(tl.float32)

        x = tl.load(
            Z + rows[:, None] * stride_z + c_offsets[None, :],
            mask=rmask[:, None] & cmask[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_V, C)
        mean = tl.sum(x, axis=1) / Zc_f
        x_centered = tl.where(cmask[None, :], x - mean[:, None], 0.0)
        var = tl.sum(x_centered * x_centered, axis=1) / Zc_f
        rstd = 1.0 / tl.sqrt(var + eps)
        zc = x_centered * rstd[:, None] * wln[None, :]  # (BLOCK_V, C)

        # A 3D broadcast reduction here trips an LLVM SimplifyCFG assertion in
        # the AMD backend, so this is a dot.
        hb_vh = tl.dot(
            zc, tl.trans(wp), input_precision=MATMUL_PRECISION
        )  # (BLOCK_V, H)
        head_bias = tl.trans(hb_vh)  # (H, BLOCK_V)

        # row = b*UV + uv. h * stride_oh overflows i32 at N > ~11.6k.
        b_idx = rows // UV
        uv_idx = rows % UV
        out_col = b_idx * stride_ob + uv_idx * stride_ouv
        tl.store(
            Out + h_offsets[:, None].to(tl.int64) * stride_oh + out_col[None, :],
            head_bias.to(Out.dtype.element_ty),
            mask=hmask[:, None] & rmask[None, :],
        )


def _fused_ln_proj_fwd(z, ln_w, proj_w, eps):
    """z (B,U,V,Zc) -> bias (B,H,U,V)."""
    B, U, V, Zc = z.shape
    H = proj_w.shape[0]
    UV = U * V
    total_rows = B * UV

    z_flat = z.reshape(total_rows, Zc).contiguous()
    ln_w = ln_w.contiguous()
    proj_w = proj_w.contiguous()

    out = torch.empty((B, H, U, V), device=z.device, dtype=z.dtype)

    block_c = max(16, triton.next_power_of_2(Zc))
    # tl.dot needs every operand dim >= 16, so the H tile is floored at 16 even
    # when H < 16 (masked lanes carry 0 weight and drop out of the dot).
    block_h = max(16, triton.next_power_of_2(H))
    cfg = _cfg_fused()

    def grid(meta):
        return (triton.cdiv(total_rows, meta["BLOCK_V"]),)

    # The LayerNorm output feeding the dot is fp32 whatever the input dtype.
    precision = _precision.matmul_precision(fp32_operands=True)
    _ln_proj_fwd[grid](
        z_flat,
        ln_w,
        proj_w,
        out,
        z_flat.stride(0),
        out.stride(0),
        out.stride(1),
        1,  # contiguous UV plane
        total_rows,
        UV,
        Zc,
        H,
        eps,
        BLOCK_H=block_h,
        BLOCK_C=block_c,
        MATMUL_PRECISION=precision,
        **cfg,
    )
    return out


def fused_ln_proj_bias(z, ln_w, proj_w, eps):
    """Fused bias-free LayerNorm(z) + pair-bias projection -> ``(B,H,U,V)``.

    Equivalent to ``F.linear(F.layer_norm(z, (Zc,), ln_w, None, eps), proj_w)``
    followed by ``.permute(0,3,1,2).contiguous()``. Forward-only: raises if any
    input requires a gradient, since autograd cannot see the Triton write.
    """
    if torch.is_grad_enabled() and any(t.requires_grad for t in (z, ln_w, proj_w)):
        raise RuntimeError(
            "fused_ln_proj_bias is forward-only; use the composed LN+linear path "
            "when gradients are needed."
        )

    dt = active_autocast_dtype()
    if dt is not None:
        z = z.to(dt)
        ln_w = ln_w.to(dt)
        proj_w = proj_w.to(dt)
    return _fused_ln_proj_fwd(z, ln_w, proj_w, eps)
