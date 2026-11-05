# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Register-resident Triton LayerNorm over the last dim, without a bias.

For the many short rows of the pair representation in ``attention_pair_bias``
(a channel dim of ~128), ROCm's ``F.layer_norm`` kernel runs far below the HBM
roofline. Here the whole row fits one block, so ``x`` is read once and reused
for the mean, the variance and the affine step.

``triton_layer_norm(x, weight, eps)`` is a ``torch.autograd.Function`` (forward
and backward). Leading dims are flattened to rows; reductions are in fp32.
"""

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def _cfg_layer_norm():
    # Tuned on MI300X at C=128: the whole row is one block, and a single warp
    # with BLOCK_ROWS=4 is fastest for both bf16 and fp32.
    return {"BLOCK_ROWS": 4, "num_warps": 1, "num_stages": 2}


def _cfg_layer_norm_bwd():
    # A single warp is fastest on MI300X: the row fits one warp's registers and
    # more warps only add cross-warp reduction.
    return {"num_warps": 1, "num_stages": 2}


if TRITON_AVAILABLE:

    @triton.jit
    def _layer_norm_2d_fwd(
        X,
        Y,
        W,
        Mean,
        Rstd,
        stride,
        total_rows,
        C,
        eps,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Row-wise LayerNorm over the last dim (C); saves mean/rstd.

        Requires ``BLOCK_C >= C``, so each row is one block and ``x`` is loaded
        into registers once for both reductions and the affine step.
        """
        pid = tl.program_id(0)
        # i64 since row * stride exceeds 2**31 at large N.
        row_start = pid.to(tl.int64) * BLOCK_ROWS

        c_offsets = tl.arange(0, BLOCK_C)
        cmask = c_offsets < C
        C_f = C.to(tl.float32)
        w = tl.load(W + c_offsets, mask=cmask)

        for row_offset in range(BLOCK_ROWS):
            row = row_start + row_offset
            if row < total_rows:
                X_row = X + row * stride
                Y_row = Y + row * stride
                x = tl.load(X_row + c_offsets, mask=cmask, other=0.0).to(tl.float32)
                mean = tl.sum(x) / C_f
                x_centered = tl.where(cmask, x - mean, 0.0)
                var = tl.sum(x_centered * x_centered) / C_f
                rstd = 1.0 / tl.sqrt(var + eps)

                tl.store(Mean + row, mean)
                tl.store(Rstd + row, rstd)

                y = x_centered * rstd * w
                tl.store(Y_row + c_offsets, y, mask=cmask)

    @triton.jit
    def _layer_norm_2d_bwd(
        DY,
        X,
        W,
        Mean,
        Rstd,
        DX,
        DW_partial,
        stride_dy,
        stride_x,
        stride_dx,
        total_rows,
        C,
        n_programs,
        BLOCK_C: tl.constexpr,
    ):
        """Backward of row-wise LayerNorm over the last dim (C).

        Each program grid-strides over rows, writing ``dx`` per row and
        accumulating its partial ``dweight`` into ``DW_partial`` (``[n_programs,
        C]``), which the caller sums over the program axis.
        """
        pid = tl.program_id(0)

        c_offsets = tl.arange(0, BLOCK_C)
        cmask = c_offsets < C
        w = tl.load(W + c_offsets, mask=cmask, other=0.0).to(tl.float32)

        dw_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

        C_f = C.to(tl.float32)

        for row in range(pid, total_rows, n_programs):
            row_i64 = row.to(tl.int64)
            dy = tl.load(
                DY + row_i64 * stride_dy + c_offsets, mask=cmask, other=0.0
            ).to(tl.float32)
            x = tl.load(X + row_i64 * stride_x + c_offsets, mask=cmask, other=0.0).to(
                tl.float32
            )
            mean = tl.load(Mean + row_i64).to(tl.float32)
            rstd = tl.load(Rstd + row_i64).to(tl.float32)

            x_hat = (x - mean) * rstd
            dy_w = dy * w
            # Masked-out lanes are 0, so they drop out of the reductions.
            mean_dy_w = tl.sum(dy_w) / C_f
            mean_dy_w_xhat = tl.sum(dy_w * x_hat) / C_f
            dx = rstd * (dy_w - mean_dy_w - x_hat * mean_dy_w_xhat)
            tl.store(DX + row_i64 * stride_dx + c_offsets, dx, mask=cmask)

            dw_acc += dy * x_hat

        out_off = pid.to(tl.int64) * C + c_offsets
        tl.store(DW_partial + out_off, dw_acc, mask=cmask)


def _triton_layernorm_fwd(x_flat, weight, eps):
    """LayerNorm over last dim of a [M, C] tensor. Returns (y, mean, rstd)."""
    M, C = x_flat.shape
    y = torch.empty_like(x_flat)
    mean = torch.empty(M, device=x_flat.device, dtype=torch.float32)
    rstd = torch.empty(M, device=x_flat.device, dtype=torch.float32)
    block_c = max(16, triton.next_power_of_2(C))
    cfg = _cfg_layer_norm()

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_ROWS"]),)

    _layer_norm_2d_fwd[grid](
        x_flat,
        y,
        weight,
        mean,
        rstd,
        x_flat.stride(0),
        M,
        C,
        eps,
        BLOCK_C=block_c,
        **cfg,
    )
    return y, mean, rstd


def _triton_layernorm_bwd(dy, x, weight, mean, rstd, out_dtype):
    """Backward of LayerNorm over the last dim of [M, C] tensors.

    Returns ``(dx, dweight)``: ``dx`` in ``out_dtype``, ``dweight`` in fp32.
    """
    M, C = x.shape
    dy = dy.contiguous()
    x = x.contiguous()
    dx = torch.empty(M, C, device=x.device, dtype=out_dtype)
    block_c = max(16, triton.next_power_of_2(C))
    cfg = _cfg_layer_norm_bwd()

    # Bound the number of programs so the partial buffer and its reduction stay
    # small; each program grid-strides over its share of rows.
    num_programs = min(M, 8192)
    dw_partial = torch.empty(num_programs, C, device=x.device, dtype=torch.float32)

    _layer_norm_2d_bwd[(num_programs,)](
        dy,
        x,
        weight,
        mean,
        rstd,
        dx,
        dw_partial,
        dy.stride(0),
        x.stride(0),
        dx.stride(0),
        M,
        C,
        num_programs,
        BLOCK_C=block_c,
        **cfg,
    )
    return dx, dw_partial.sum(dim=0)


def active_autocast_dtype():
    """The active CUDA autocast compute dtype, or ``None`` if autocast is off."""
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return None


class _TritonLayerNorm(torch.autograd.Function):
    """Bias-free LayerNorm over the last dim. Matches
    ``F.layer_norm(x, (C,), weight, None, eps)``.

    Under autocast, inputs are cast to the autocast dtype, as autocast would do
    for the op it replaces.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        dt = active_autocast_dtype()
        if dt is not None:
            x = x.to(dt)
            weight = weight.to(dt)
        C = x.shape[-1]
        x_flat = x.reshape(-1, C).contiguous()
        weight = weight.contiguous()
        y_flat, mean, rstd = _triton_layernorm_fwd(x_flat, weight, eps)
        ctx.save_for_backward(x_flat, weight, mean, rstd)
        ctx.x_shape = x.shape
        return y_flat.view_as(x)

    @staticmethod
    def backward(ctx, dy):
        x_flat, weight, mean, rstd = ctx.saved_tensors
        C = x_flat.shape[-1]
        dx_flat, dweight = _triton_layernorm_bwd(
            dy.reshape(-1, C), x_flat, weight, mean, rstd, out_dtype=x_flat.dtype
        )
        return dx_flat.view(ctx.x_shape), dweight.to(weight.dtype), None


def triton_layer_norm(x, weight, eps):
    """LayerNorm of ``x`` over its last dim, scaled by ``weight``, no bias."""
    return _TritonLayerNorm.apply(x, weight, eps)
