# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable Triton triangle_multiplicative_update (forward + backward).

The Triton backend for
:func:`geometric_nn_ops_torch.triangle.triangle_multiplicative_update`, exposed as a
single ``autograd.Function`` used on the CUDA/Triton path for both inference and
training.

  * Forward uses a channel-major fused GEMM (LayerNorm -> gate/proj GEMM ->
    sigmoid -> multiply -> mask, written channel-major ``[B*C, N, N]``) and a
    per-channel batched matmul (``torch.bmm``) that combines the two projections.
  * Backward is a hybrid: the heavy per-channel combine matmul is the same
    strided-batched ``torch.bmm`` (transposed operand views give the two combine
    gradients); the two LayerNorm backwards use a fused Triton kernel
    (``_layer_norm_2d_bwd``); the input-projection backward fuses its recompute
    + sigmoid-derivative pointwise into a Triton kernel
    (``_proj_in_bwd_grads_dual``); the remaining output-projection linear /
    sigmoid / elementwise steps are differentiated with eager PyTorch ops (cheap
    relative to the fused projections).

The autograd.Function saves only minimal intermediates and recomputes the input
projections (and the output gate pre-activation) in backward, trading compute
for a large activation-memory saving. Large [M, D] pointwise intermediates stay
in the compute dtype (bf16). Together these keep the training peak at/below
eager.

Weight convention (fused layout, bias-free projections): ``p_in_weight`` /
``g_in_weight`` are ``[2D, D]`` (rows ``0:D`` -> ``a``, ``D:2D`` -> ``b``);
``p_out_weight`` (= ``linear_z``) and ``g_out_weight`` (= ``linear_g``) are
``[D, D]``. All nn.Linear layout ``(out, in)``. This path assumes ``c_hidden ==
c_z == D`` and no projection biases.
"""

import torch

from geometric_nn_ops_torch.triangle._reference.trimul import is_outgoing

# Note: The limit for the number of tiles along grid.x is `2^32` whereas for the
# other dimensions it's `2^16`. Dimensions that grows with sequence length
# squared (e.g. M = B*N*N) can exceed the smaller limit and need to go on
# grid.x.

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def _cfg_fused_gemm(*, N, spatial=None):
    """Fused projection GEMM (M = spatial^2 rows, K = D, N_out = D).

    Tuned on MI300X.
    """
    block_n = min(128, max(16, triton.next_power_of_2(N)))
    num_warps = 8 if block_n <= 64 else 16
    return {
        "BLOCK_M": 128,
        "BLOCK_N": block_n,
        "BLOCK_K": 32,
        "num_warps": num_warps,
        "num_stages": 2,
    }


def _cfg_layer_norm():
    # MI300X: BLOCK_C=128 covers the full C=128 row in one pass (no inner loop);
    # BLOCK_ROWS=4 with a single warp is fastest for both bf16 and fp32.
    return {"BLOCK_ROWS": 4, "BLOCK_C": 128, "num_warps": 1, "num_stages": 2}


def _cfg_layer_norm_bwd(*, C):
    # dx kernel: programs grid-stride over rows; whole channel dim in one block
    # (C == D is small, <= a few hundred). Reductions across C are in-register.
    # Tuned on MI300X for the backward: a single warp is the consistent winner:
    # the whole C=128 row fits one warp's registers and extra warps only add
    # cross-warp reduction overhead here.
    block_c = max(16, triton.next_power_of_2(C))
    return {"BLOCK_C": block_c, "num_warps": 1, "num_stages": 2}


def _cfg_proj_bwd(*, dtype):
    """Fused projection backward GEMM+pointwise config (proj-in / proj-in-dual /
    proj-out). M = spatial^2 rows, K = D, N_out = D.

    These kernels recompute the proj+gate GEMMs (K=D) and do the sigmoid-
    derivative pointwise emitting d_ap/d_ag; a different access pattern than the
    forward projection GEMM, so they get their own config rather than inheriting
    ``_cfg_fused_gemm``. Tuned on MI300X.
    """
    if dtype == torch.float32:
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32,
            "num_warps": 16,
            "num_stages": 2,
        }
    return {
        "BLOCK_M": 64,
        "BLOCK_N": 128,
        "BLOCK_K": 32,
        "num_warps": 8,
        "num_stages": 2,
    }


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.jit
    def _fused_gemm_sigmoid_multiply_channel_major(
        A_ptr,
        W_g_ptr,
        W_p_ptr,
        C_ptr,
        mask_ptr,
        stride_Ai,
        stride_Ak,
        stride_Wgk,
        stride_Wgj,
        stride_Wpk,
        stride_Wpj,
        stride_Cb,
        stride_Ci,
        stride_Cj,
        M,
        K,
        N,
        N_spatial,
        SAVE_GP: tl.constexpr,
        G_ptr,
        P_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """gate = A @ W_g.T; proj = A @ W_p.T; out = sigmoid(gate)*proj*mask.

        Output written channel-major: input row m = b*N^2 + i*N + j maps to output
        (b*C + n, i, j). If SAVE_GP, also writes pre-activation gate (row-major,
        [M, N]) and proj into G_ptr/P_ptr for the backward.
        """
        # Large M dimension tiling on grid.x. See note at top of file.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m_i64 = offs_m.to(tl.int64)

        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_p = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            A = tl.load(
                A_ptr + offs_m_i64[:, None] * stride_Ai + offs_k[None, :] * stride_Ak,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            W_g = tl.load(
                W_g_ptr + offs_k[:, None] * stride_Wgk + offs_n[None, :] * stride_Wgj,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            W_p = tl.load(
                W_p_ptr + offs_k[:, None] * stride_Wpk + offs_n[None, :] * stride_Wpj,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc_g = tl.dot(A.to(tl.float32), W_g.to(tl.float32), acc_g)
            acc_p = tl.dot(A.to(tl.float32), W_p.to(tl.float32), acc_p)

        gate_sigmoid = 1.0 / (1.0 + tl.exp(-acc_g))
        result = gate_sigmoid * acc_p

        if mask_ptr is not None:
            m = tl.load(mask_ptr + offs_m, mask=offs_m < M, other=0.0)
            result = result * m[:, None]

        if SAVE_GP:
            gp_off = offs_m_i64[:, None] * N + offs_n[None, :]
            store_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl.store(G_ptr + gp_off, acc_g, mask=store_mask)
            tl.store(P_ptr + gp_off, acc_p, mask=store_mask)

        batch_idx = offs_m // (N_spatial * N_spatial)
        remainder = offs_m % (N_spatial * N_spatial)
        spatial_i = remainder // N_spatial
        spatial_j = remainder % N_spatial

        out_offsets = (
            (batch_idx[:, None].to(tl.int64) * N + offs_n[None, :]) * stride_Cb
            + spatial_i[:, None].to(tl.int64) * stride_Ci
            + spatial_j[:, None].to(tl.int64) * stride_Cj
        )
        tl.store(
            C_ptr + out_offsets,
            result,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    @triton.jit
    def _fused_gemm_sigmoid_multiply(
        A_ptr,
        W_g_ptr,
        W_p_ptr,
        C_ptr,
        A_p_ptr,
        stride_Ai,
        stride_Ak,
        stride_Wgk,
        stride_Wgj,
        stride_Wpk,
        stride_Wpj,
        stride_Ci,
        stride_Cj,
        stride_Api,
        stride_Apk,
        M,
        K,
        N,
        SAVE_G: tl.constexpr,
        G_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """out = sigmoid(A @ W_g.T) * (A_p @ W_p.T), row-major [M, N].

        Used for the output projection: gate from z_normed, proj from x_normed.
        If SAVE_G, writes the pre-sigmoid gate into G_ptr for the backward.
        """

        # Large M dimension tiling on grid.x. See note at top of file.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m_i64 = offs_m.to(tl.int64)

        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_p = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            A = tl.load(
                A_ptr + offs_m_i64[:, None] * stride_Ai + offs_k[None, :] * stride_Ak,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            A_p = tl.load(
                A_p_ptr
                + offs_m_i64[:, None] * stride_Api
                + offs_k[None, :] * stride_Apk,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            W_g = tl.load(
                W_g_ptr + offs_k[:, None] * stride_Wgk + offs_n[None, :] * stride_Wgj,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            W_p = tl.load(
                W_p_ptr + offs_k[:, None] * stride_Wpk + offs_n[None, :] * stride_Wpj,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc_g = tl.dot(A.to(tl.float32), W_g.to(tl.float32), acc_g)
            acc_p = tl.dot(A_p.to(tl.float32), W_p.to(tl.float32), acc_p)

        gate_sigmoid = 1.0 / (1.0 + tl.exp(-acc_g))
        result = gate_sigmoid * acc_p

        store_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if SAVE_G:
            tl.store(
                G_ptr + offs_m_i64[:, None] * N + offs_n[None, :],
                acc_g,
                mask=store_mask,
            )

        tl.store(
            C_ptr + offs_m_i64[:, None] * stride_Ci + offs_n[None, :] * stride_Cj,
            result,
            mask=store_mask,
        )

    @triton.jit
    def _proj_in_bwd_grads_dual(
        Zn_ptr,
        Wp_ptr,
        Wg_ptr,
        Douta_ptr,
        Doutb_ptr,
        mask_ptr,
        Dall_ptr,
        stride_Zi,
        stride_Zk,
        stride_Wpb,
        stride_Wpk,
        stride_Wpj,
        stride_Wgb,
        stride_Wgk,
        stride_Wgj,
        stride_Dai,
        stride_Daj,
        stride_Dbi,
        stride_Dbj,
        stride_Om,
        stride_Op,
        stride_On,
        M,
        K,
        N,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Two fused input-projection backwards (a and b) in one launch.

        Recomputes proj ``ap = Zn @ W_p^T`` and gate preact ``ag = Zn @ W_g^T``
        (over K=D), then produces, all in fp32 internally::

            m           = mask[row]
            sa          = sigmoid(ag)
            d_after     = Dout * m
            d_ap        = d_after * sa
            d_ag        = d_after * ap * sa * (1 - sa)

        Writing ``d_ap`` and ``d_ag`` (compute dtype) directly. This fuses the two
        recompute GEMMs with the sigmoid-derivative elementwise math so no large
        fp32 [M, D] temporaries (ap, sa, d_after, ...) hit HBM; the downstream
        weight/input GEMMs then run in the compute dtype via ``torch.bmm``.

        The grid carries a third axis ``pid_b in {0, 1}`` selecting the a- or
        b-projection. The two projections share ``z_n`` and ``mask`` but use
        distinct fused weights (``Wp_ptr``/``Wg_ptr`` are ``[2, D, D]`` stacked
        a/b) and distinct upstream grads (``Douta_ptr`` / ``Doutb_ptr``, each a
        possibly-strided ``[M, D]`` view). This halves the launch count vs
        running as separate kernels.

        The 4 outputs (d_ap/d_ag for a and b) are written into one ``[M, 4, N]``
        buffer ``Dall`` interleaved along the feature (proj) axis: proj-index
        ``2*pid_b`` = d_ap, ``2*pid_b+1`` = d_ag. Laid out this way,
        ``Dall.reshape(M, 4N)^T @ z_n`` computes all four weight gradients in a
        single GEMM (rows ``[j*D:(j+1)*D]`` = grad j), and the shared ``z_n``
        operand is read once instead of four times -- the weight-grad GEMM
        category is memory-bandwidth bound, so this is a win. Individual
        ``d_ap_a`` etc. remain available as ``Dall[:, j, :]`` views for the
        ``d_z_n`` accumulation.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m_i64 = offs_m.to(tl.int64)

        w_base = pid_b.to(tl.int64) * stride_Wpb
        wg_base = pid_b.to(tl.int64) * stride_Wgb
        # Select the a- or b-projection upstream grad by pid_b. Strides may differ
        # between the two views, so pick both pointer base and strides.
        if pid_b == 0:
            d_base = Douta_ptr
            d_si = stride_Dai
            d_sj = stride_Daj
        else:
            d_base = Doutb_ptr
            d_si = stride_Dbi
            d_sj = stride_Dbj

        acc_p = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            Z = tl.load(
                Zn_ptr + offs_m_i64[:, None] * stride_Zi + offs_k[None, :] * stride_Zk,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            W_p = tl.load(
                Wp_ptr
                + w_base
                + offs_k[:, None] * stride_Wpk
                + offs_n[None, :] * stride_Wpj,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            W_g = tl.load(
                Wg_ptr
                + wg_base
                + offs_k[:, None] * stride_Wgk
                + offs_n[None, :] * stride_Wgj,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            Zf = Z.to(tl.float32)
            acc_p = tl.dot(Zf, W_p.to(tl.float32), acc_p)
            acc_g = tl.dot(Zf, W_g.to(tl.float32), acc_g)

        m = tl.load(mask_ptr + offs_m, mask=offs_m < M, other=0.0)
        d_out = tl.load(
            d_base + offs_m_i64[:, None] * d_si + offs_n[None, :] * d_sj,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)

        sa = 1.0 / (1.0 + tl.exp(-acc_g))
        d_after = d_out * m[:, None]
        d_ap = d_after * sa
        d_ag = d_after * acc_p * sa * (1.0 - sa)

        store_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        # Interleave into the [M, 4, N] buffer: proj-index 2*pid_b = d_ap,
        # 2*pid_b+1 = d_ag. Order along the proj axis: [ap_a, ag_a, ap_b, ag_b].
        p_ap = (2 * pid_b).to(tl.int64)
        p_ag = (2 * pid_b + 1).to(tl.int64)
        m_off = offs_m_i64[:, None] * stride_Om + offs_n[None, :] * stride_On
        tl.store(Dall_ptr + p_ap * stride_Op + m_off, d_ap, mask=store_mask)
        tl.store(Dall_ptr + p_ag * stride_Op + m_off, d_ag, mask=store_mask)

    @triton.jit
    def _proj_out_bwd_grads(
        Zn_ptr,
        Xn_ptr,
        W_p_ptr,
        W_g_ptr,
        Dout_ptr,
        Dproj_ptr,
        Dgpre_ptr,
        stride_Zi,
        stride_Zk,
        stride_Xi,
        stride_Xk,
        stride_Wpk,
        stride_Wpj,
        stride_Wgk,
        stride_Wgj,
        stride_Di,
        stride_Dj,
        M,
        K,
        N,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused output-projection backward pointwise grads.

        The output projection is ``out = sigmoid(z_n @ W_g^T) * (x_n @ W_p^T)``
        (gate from ``z_n``, proj from ``x_n``; no mask). This recomputes the gate
        preact ``ag = z_n @ W_g^T`` and the proj ``xz = x_n @ W_p^T`` (over K=D),
        then produces, all in fp32 internally::

            sg          = sigmoid(ag)
            d_proj      = Dout * sg
            d_g_pre     = Dout * xz * sg * (1 - sg)

        Writing ``d_proj`` and ``d_g_pre`` (compute dtype) directly. Fuses the
        two recompute GEMMs with the sigmoid-derivative elementwise math so no
        large fp32 [M, D] temporaries (sg, xz, d_proj, d_g_pre) hit HBM; the
        downstream weight/input GEMMs then run in the compute dtype via torch
        matmul. Mirrors ``_proj_in_bwd_grads_dual`` but with distinct gate/proj
        inputs and no mask.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m_i64 = offs_m.to(tl.int64)

        acc_p = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            Z = tl.load(
                Zn_ptr + offs_m_i64[:, None] * stride_Zi + offs_k[None, :] * stride_Zk,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            X = tl.load(
                Xn_ptr + offs_m_i64[:, None] * stride_Xi + offs_k[None, :] * stride_Xk,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            W_p = tl.load(
                W_p_ptr + offs_k[:, None] * stride_Wpk + offs_n[None, :] * stride_Wpj,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            W_g = tl.load(
                W_g_ptr + offs_k[:, None] * stride_Wgk + offs_n[None, :] * stride_Wgj,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc_p = tl.dot(X.to(tl.float32), W_p.to(tl.float32), acc_p)
            acc_g = tl.dot(Z.to(tl.float32), W_g.to(tl.float32), acc_g)

        d_out = tl.load(
            Dout_ptr + offs_m_i64[:, None] * stride_Di + offs_n[None, :] * stride_Dj,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)

        sg = 1.0 / (1.0 + tl.exp(-acc_g))
        d_proj = d_out * sg
        d_g_pre = d_out * acc_p * sg * (1.0 - sg)

        store_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        out_off = offs_m_i64[:, None] * N + offs_n[None, :]
        tl.store(Dproj_ptr + out_off, d_proj, mask=store_mask)
        tl.store(Dgpre_ptr + out_off, d_g_pre, mask=store_mask)

    @triton.jit
    def _layer_norm_2d_fwd(
        X,
        Y,
        W,
        B,
        Mean,
        Rstd,
        stride,
        total_rows,
        C,
        eps,
        SAVE_STATS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Row-wise LayerNorm over the last dim (C). Optionally saves mean/rstd."""
        pid = tl.program_id(0)
        row_start = pid * BLOCK_ROWS

        for row_offset in range(BLOCK_ROWS):
            row = row_start + row_offset
            row_valid = row < total_rows
            row_i64 = row.to(tl.int64)
            X_row = X + row_i64 * stride
            Y_row = Y + row_i64 * stride

            if row_valid:
                mean = 0.0
                M2 = 0.0
                count = 0
                for c_start in range(0, C, BLOCK_C):
                    c_offsets = c_start + tl.arange(0, BLOCK_C)
                    cmask = c_offsets < C
                    x = tl.load(X_row + c_offsets, mask=cmask, other=0.0).to(tl.float32)
                    n_valid = tl.sum(cmask.to(tl.int32))
                    if n_valid > 0:
                        local_sum = tl.sum(tl.where(cmask, x, 0.0))
                        local_mean = local_sum / n_valid
                        new_count = count + n_valid
                        delta = local_mean - mean
                        mean = mean + delta * n_valid / new_count
                        local_m2 = tl.sum(
                            tl.where(cmask, (x - local_mean) * (x - local_mean), 0.0)
                        )
                        M2 = M2 + local_m2 + delta * delta * count * n_valid / new_count
                        count = new_count

                var = M2 / C
                rstd = 1.0 / tl.sqrt(var + eps)

                if SAVE_STATS:
                    tl.store(Mean + row_i64, mean)
                    tl.store(Rstd + row_i64, rstd)

                for c_start in range(0, C, BLOCK_C):
                    c_offsets = c_start + tl.arange(0, BLOCK_C)
                    cmask = c_offsets < C
                    w = tl.load(W + c_offsets, mask=cmask)
                    b = tl.load(B + c_offsets, mask=cmask)
                    x = tl.load(X_row + c_offsets, mask=cmask, other=0.0).to(tl.float32)
                    x_hat = (x - mean) * rstd
                    y = x_hat * w + b
                    tl.store(Y_row + c_offsets, y, mask=cmask & row_valid)

    @triton.jit
    def _layer_norm_2d_bwd(
        DY,
        X,
        W,
        Mean,
        Rstd,
        DX,
        DW_partial,
        DB_partial,
        stride_dy,
        stride_x,
        stride_dx,
        total_rows,
        C,
        n_programs,
        BLOCK_C: tl.constexpr,
    ):
        """Backward of row-wise LayerNorm over the last dim (C).

        Each program grid-strides over rows (stride = n_programs), processing the
        full channel dim in a single block (BLOCK_C >= C). It writes dx per row
        and accumulates its partial dweight/dbias across the rows it visits into
        DW_partial/DB_partial (shape [n_programs, C]); the caller sums those over
        the program axis (a native torch reduction). All math is in fp32.
        """
        pid = tl.program_id(0)

        c_offsets = tl.arange(0, BLOCK_C)
        cmask = c_offsets < C
        w = tl.load(W + c_offsets, mask=cmask, other=0.0).to(tl.float32)

        dw_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        db_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

        C_f = C.to(tl.float32)

        for row in range(pid, total_rows, n_programs):
            row_i64 = row.to(tl.int64)
            dy = tl.load(
                DY + row_i64 * stride_dy + c_offsets, mask=cmask, other=0.0
            ).to(tl.float32)
            x = tl.load(X + row_i64 * stride_x + c_offsets, mask=cmask, other=0.0).to(
                tl.float32
            )
            rstd = tl.load(Rstd + row_i64).to(tl.float32)
            mean = tl.load(Mean + row_i64).to(tl.float32)
            # Recompute x_hat from the saved pre-norm combine result. We tried
            # reconstructing it from the affine output, which would avoid the
            # need to store the pre-norm input, but this breaks at gamma=0 and
            # the memory savings didn't seem worth special casing for.
            x_hat = (x - mean) * rstd
            dy_w = dy * w
            # Reductions over the channel dim (masked-out lanes are 0).
            mean_dy_w = tl.sum(dy_w) / C_f
            mean_dy_w_xhat = tl.sum(dy_w * x_hat) / C_f
            dx = rstd * (dy_w - mean_dy_w - x_hat * mean_dy_w_xhat)
            tl.store(DX + row_i64 * stride_dx + c_offsets, dx, mask=cmask)

            dw_acc += dy * x_hat
            db_acc += dy

        out_off = pid.to(tl.int64) * C + c_offsets
        tl.store(DW_partial + out_off, dw_acc, mask=cmask)
        tl.store(DB_partial + out_off, db_acc, mask=cmask)


# ---------------------------------------------------------------------------
# Python-side launch helpers
# ---------------------------------------------------------------------------


def _triton_layernorm(x_flat, weight, bias, eps, save_stats=False):
    """LayerNorm over last dim of a [M, C] tensor. Returns (y, mean, rstd)."""
    M, C = x_flat.shape
    y = torch.empty_like(x_flat)
    if save_stats:
        mean = torch.empty(M, device=x_flat.device, dtype=torch.float32)
        rstd = torch.empty(M, device=x_flat.device, dtype=torch.float32)
    else:
        mean = x_flat  # unused
        rstd = x_flat
    cfg = _cfg_layer_norm()

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_ROWS"]),)

    _layer_norm_2d_fwd[grid](
        x_flat,
        y,
        weight,
        bias,
        mean,
        rstd,
        x_flat.stride(0),
        M,
        C,
        eps,
        SAVE_STATS=save_stats,
        **cfg,
    )
    if save_stats:
        return y, mean, rstd
    return y, None, None


def _triton_layernorm_backward(dy, x, weight, mean, rstd, out_dtype=None):
    """Fused Triton backward of LayerNorm over last dim of [M, C] tensors.

    Args:
        dy: [M, C]
        x: raw pre-norm input
        weight: [C]
        mean: [M]
        rstd: [M]
        out_dtype: return type for dx (default dy.dtype)

    dweight/dbias are always returned in fp32. All reductions run in fp32. Math (per
    row m, channel c, with x_hat = (x - mean) * rstd and dy_w = dy * weight):

        dx[m,c]  = rstd[m] * (dy_w[m,c] - mean_c(dy_w[m]) - x_hat[m,c] *
                              mean_c(dy_w[m] * x_hat[m]))
        dweight  = sum_m (dy[m] * x_hat[m])
        dbias    = sum_m dy[m]
    """
    M, C = x.shape
    if out_dtype is None:
        out_dtype = dy.dtype
    dy = dy.contiguous()
    x = x.contiguous()
    dx = torch.empty(M, C, device=x.device, dtype=out_dtype)
    cfg = _cfg_layer_norm_bwd(C=C)

    # Bound the number of programs so the partial buffer + its reduction stay
    # small; each program grid-strides over its share of rows.
    num_programs = min(M, 8192)

    dw_partial = torch.empty(num_programs, C, device=x.device, dtype=torch.float32)
    db_partial = torch.empty(num_programs, C, device=x.device, dtype=torch.float32)

    _layer_norm_2d_bwd[(num_programs,)](
        dy,
        x,
        weight,
        mean,
        rstd,
        dx,
        dw_partial,
        db_partial,
        dy.stride(0),
        x.stride(0),
        dx.stride(0),
        M,
        C,
        num_programs,
        **cfg,
    )

    # Sum partials over the program axis with a native torch reduction.
    dweight = dw_partial.sum(dim=0)
    dbias = db_partial.sum(dim=0)
    return dx, dweight, dbias


def _combine_projection(a, b, batch_dims, c_hidden):
    """Per-channel a@b. a,b: [B*C, N, N] channel-major. Returns [..., N, N, C].

    Uses strided-batched GEMM (``torch.bmm``), which is generally faster than
    any custom kernel.
    """
    BC, N, _ = a.shape
    out = torch.bmm(a, b)
    Bflat = 1
    for d in batch_dims:
        Bflat *= d
    out = (
        out.reshape(Bflat, c_hidden, N, N)
        .permute(0, 2, 3, 1)
        .reshape(Bflat, N, N, c_hidden)
    )
    return out


# ---------------------------------------------------------------------------
# Backward helpers.
# ---------------------------------------------------------------------------


def _proj_in_bwd_pointwise_dual(z_n, Wp_t2, Wg_t2, d_out_a, d_out_b, mask_col_flat):
    """Both input-projection backwards in one launch.

    z_n: [M, D]. Wp_t2/Wg_t2: [2, D, D] = stacked a/b W^T (K, N) contiguous.
    d_out_a/d_out_b: [M, D] (possibly strided views). mask_col_flat: [M], compute dtype.
    Returns d_all: [M, 4, D] in the compute dtype, proj axis ordered
    [d_ap_a, d_ag_a, d_ap_b, d_ag_b]. Reshaped to [M, 4D] the transpose-GEMM
    against z_n yields all four weight grads in one launch (see kernel docstring).
    """
    M, K = z_n.shape
    N = Wp_t2.shape[2]
    cdtype = z_n.dtype
    d_all = torch.empty(M, 4, N, device=z_n.device, dtype=cdtype)
    fg = _cfg_proj_bwd(dtype=cdtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), 2)

    _proj_in_bwd_grads_dual[grid](
        z_n,
        Wp_t2,
        Wg_t2,
        d_out_a,
        d_out_b,
        mask_col_flat,
        d_all,
        z_n.stride(0),
        z_n.stride(1),
        Wp_t2.stride(0),
        Wp_t2.stride(1),
        Wp_t2.stride(2),
        Wg_t2.stride(0),
        Wg_t2.stride(1),
        Wg_t2.stride(2),
        d_out_a.stride(0),
        d_out_a.stride(1),
        d_out_b.stride(0),
        d_out_b.stride(1),
        d_all.stride(0),
        d_all.stride(1),
        d_all.stride(2),
        M,
        K,
        N,
        BLOCK_M=fg["BLOCK_M"],
        BLOCK_N=fg["BLOCK_N"],
        BLOCK_K=fg["BLOCK_K"],
        num_warps=fg["num_warps"],
        num_stages=fg["num_stages"],
    )
    return d_all


def _proj_out_bwd_pointwise(z_n, x_n, Wp_t, Wg_t, d_out):
    """Fused recompute + sigmoid-derivative for the output-projection backward.

    z_n / x_n: [M, D] (compute dtype); gate input / proj input respectively.
    Wp_t/Wg_t: [D, D] = W^T (K, N) contiguous. d_out: [M, D] (compute dtype).
    Returns (d_proj, d_g_pre), each [M, D] in the compute dtype.
    """
    M, K = z_n.shape
    N = Wp_t.shape[1]
    cdtype = z_n.dtype
    d_proj = torch.empty(M, N, device=z_n.device, dtype=cdtype)
    d_g_pre = torch.empty(M, N, device=z_n.device, dtype=cdtype)
    fg = _cfg_proj_bwd(dtype=cdtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    _proj_out_bwd_grads[grid](
        z_n,
        x_n,
        Wp_t,
        Wg_t,
        d_out,
        d_proj,
        d_g_pre,
        z_n.stride(0),
        z_n.stride(1),
        x_n.stride(0),
        x_n.stride(1),
        Wp_t.stride(0),
        Wp_t.stride(1),
        Wg_t.stride(0),
        Wg_t.stride(1),
        d_out.stride(0),
        d_out.stride(1),
        M,
        K,
        N,
        BLOCK_M=fg["BLOCK_M"],
        BLOCK_N=fg["BLOCK_N"],
        BLOCK_K=fg["BLOCK_K"],
        num_warps=fg["num_warps"],
        num_stages=fg["num_stages"],
    )
    return d_proj, d_g_pre


class _TriangleMultiplicativeUpdateFn(torch.autograd.Function):
    """Differentiable TMU. Forward uses Triton fused kernels; backward is a
    hybrid (strided-batched ``torch.bmm`` for the combine step, fused Triton
    LayerNorm backward, eager PyTorch for linear/sigmoid/elementwise).

    Weight convention (fused, bias-free): W_p_in / W_g_in are [2D, D] (rows 0:D
    -> a, D:2D -> b); W_z (= linear_z) and W_g_out (= linear_g) are [D, D]. All
    nn.Linear layout (out, in), bias-free.
    """

    @staticmethod
    def forward(
        ctx,
        z,
        mask,
        outgoing,
        norm_in_weight,
        norm_in_bias,
        p_in_weight,
        g_in_weight,
        norm_out_weight,
        norm_out_bias,
        p_out_weight,
        g_out_weight,
        eps,
    ):
        B = 1
        batch_dims = z.shape[:-3]
        for d in batch_dims:
            B *= d
        N = z.shape[-3]
        D = z.shape[-1]
        compute_dtype = z.dtype

        # Cast weights to compute dtype.
        p_in_weight = p_in_weight.to(compute_dtype)
        g_in_weight = g_in_weight.to(compute_dtype)
        p_out_weight = p_out_weight.to(compute_dtype)
        g_out_weight = g_out_weight.to(compute_dtype)
        mask = mask.to(compute_dtype)

        # Split fused weights and transpose to [K, N] = [D, D] for the kernels.
        W_ap = p_in_weight[:D].T.contiguous()
        W_bp = p_in_weight[D:].T.contiguous()
        W_ag = g_in_weight[:D].T.contiguous()
        W_bg = g_in_weight[D:].T.contiguous()

        z_flat = z.reshape(-1, D).contiguous()
        M = z_flat.shape[0]

        z_n_flat, mean_in, rstd_in = _triton_layernorm(
            z_flat, norm_in_weight, norm_in_bias, eps, save_stats=True
        )

        mask_flat = mask.reshape(-1)

        a_cm = torch.empty(B * D, N, N, device=z.device, dtype=compute_dtype)
        b_cm = torch.empty(B * D, N, N, device=z.device, dtype=compute_dtype)

        fg = _cfg_fused_gemm(N=D, spatial=N)

        def proj_grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]), 1, triton.cdiv(D, meta["BLOCK_N"]))

        # a projection (no need to save g/p preactivations; recomputed in bwd)
        _fused_gemm_sigmoid_multiply_channel_major[proj_grid](
            z_n_flat,
            W_ag,
            W_ap,
            a_cm,
            mask_flat,
            z_n_flat.stride(0),
            z_n_flat.stride(1),
            W_ag.stride(0),
            W_ag.stride(1),
            W_ap.stride(0),
            W_ap.stride(1),
            a_cm.stride(0),
            a_cm.stride(1),
            a_cm.stride(2),
            M,
            D,
            D,
            N,
            False,
            a_cm,
            a_cm,
            **fg,
        )
        _fused_gemm_sigmoid_multiply_channel_major[proj_grid](
            z_n_flat,
            W_bg,
            W_bp,
            b_cm,
            mask_flat,
            z_n_flat.stride(0),
            z_n_flat.stride(1),
            W_bg.stride(0),
            W_bg.stride(1),
            W_bp.stride(0),
            W_bp.stride(1),
            b_cm.stride(0),
            b_cm.stride(1),
            b_cm.stride(2),
            M,
            D,
            D,
            N,
            False,
            b_cm,
            b_cm,
            **fg,
        )

        # Direction-specific transposes for the combine matmul.
        if outgoing:
            b_comb = b_cm.transpose(-1, -2)
            a_comb = a_cm
        else:
            a_comb = a_cm.transpose(-1, -2)
            b_comb = b_cm

        x_out = _combine_projection(a_comb, b_comb, batch_dims, D)  # [Bflat,N,N,D]
        x_flat = x_out.reshape(-1, D).contiguous()

        x_n_flat, mean_out, rstd_out = _triton_layernorm(
            x_flat, norm_out_weight, norm_out_bias, eps, save_stats=True
        )

        W_g_out = g_out_weight.T.contiguous()
        W_p_out = p_out_weight.T.contiguous()

        out_flat = torch.empty(M, D, device=z.device, dtype=compute_dtype)
        # Gate pre-activation is not saved: backward recomputes z_n @ Wg_out^T
        # (one cheap [M,D]@[D,D] GEMM) rather than carrying a [M,D] fp32 buffer.

        def out_grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]), 1, triton.cdiv(D, meta["BLOCK_N"]))

        _fused_gemm_sigmoid_multiply[out_grid](
            z_n_flat,
            W_g_out,
            W_p_out,
            out_flat,
            x_n_flat,
            z_n_flat.stride(0),
            z_n_flat.stride(1),
            W_g_out.stride(0),
            W_g_out.stride(1),
            W_p_out.stride(0),
            W_p_out.stride(1),
            out_flat.stride(0),
            out_flat.stride(1),
            x_n_flat.stride(0),
            x_n_flat.stride(1),
            M,
            D,
            D,
            False,  # SAVE_G
            None,  # G_ptr: unused when SAVE_G=False (constexpr branch compiled out)
            **fg,
        )

        ctx.save_for_backward(
            mask_flat,
            z_n_flat,
            z_flat,
            mean_in,
            rstd_in,
            a_cm,
            b_cm,
            x_n_flat,
            x_flat,
            mean_out,
            rstd_out,
            p_in_weight,
            g_in_weight,
            p_out_weight,
            g_out_weight,
            norm_in_weight,
            norm_in_bias,
            norm_out_weight,
            norm_out_bias,
        )
        ctx.outgoing = outgoing
        ctx.eps = eps
        ctx.B = B
        ctx.N = N
        ctx.D = D
        ctx.batch_dims = tuple(batch_dims)

        return out_flat.reshape(*batch_dims, N, N, D)

    @staticmethod
    def backward(ctx, grad_out):
        (
            mask_flat,
            z_n_flat,
            z_flat,
            mean_in,
            rstd_in,
            a_cm,
            b_cm,
            x_n_flat,
            x_flat,
            mean_out,
            rstd_out,
            p_in_weight,
            g_in_weight,
            p_out_weight,
            g_out_weight,
            norm_in_weight,
            norm_in_bias,
            norm_out_weight,
            norm_out_bias,
        ) = ctx.saved_tensors
        outgoing = ctx.outgoing
        N, D = ctx.N, ctx.D
        batch_dims = ctx.batch_dims
        M = z_n_flat.shape[0]

        # Precision: GEMM operands are the compute dtype (e.g. bf16 under
        # autocast) elementwise ops and matmuls accumulation happen in fp32 in
        # registers, but are down cast to bf16 for storage.
        cdtype = z_n_flat.dtype
        grad_out_flat = grad_out.reshape(-1, D)
        z_n = z_n_flat
        x_n = x_n_flat
        Wz = p_out_weight.to(cdtype)
        Wg_out = g_out_weight.to(cdtype)

        # ---- Output projection backward ----
        # out = sigmoid(z_n @ Wg^T) * (x_n @ Wz^T). A fused Triton kernel
        # recomputes the gate preact (z_n @ Wg^T) and proj (x_n @ Wz^T) and does
        # the sigmoid-derivative pointwise, emitting d_proj / d_g_pre directly
        # (no large fp32 [M, D] temporaries hitting HBM). The 4 downstream weight/
        # input GEMMs use ``torch.bmm``. Mirrors the input-projection backward.
        Wz_t = Wz.T.contiguous()  # [K, N] = [D, D]
        Wg_out_t = Wg_out.T.contiguous()
        d_proj, d_g_pre = _proj_out_bwd_pointwise(
            z_n, x_n, Wz_t, Wg_out_t, grad_out_flat.to(cdtype).contiguous()
        )

        d_p_out_weight = d_proj.T @ x_n  # [D, D]
        d_x_n = d_proj @ Wz  # [M, D], cdtype
        del d_proj
        d_g_out_weight = d_g_pre.T @ z_n  # [D, D]
        d_z_n = d_g_pre @ Wg_out  # accumulate (cdtype)
        del d_g_pre

        # ---- LN_out backward ----
        d_x_flat, d_norm_out_weight, d_norm_out_bias = _triton_layernorm_backward(
            d_x_n,
            x_flat,
            norm_out_weight,
            mean_out,
            rstd_out,
        )
        del d_x_n

        # ---- Combine backward ----
        # x[bflat,N,N,D] = combine(a_comb,b_comb) per channel.
        Bflat = 1
        for d in batch_dims:
            Bflat *= d
        # d_x_flat is already cdtype (LN backward emits the compute dtype). The
        # permute to channel-major needs a contiguous buffer: passing d_x as a
        # strided view directly to the combine GEMMs was measured ~3x slower on
        # MI300X (non-unit stride on the GEMM's inner dims destroys coalescing
        # and the combine GEMM is compute/layout-bound).
        d_x_cm = (
            d_x_flat.reshape(Bflat, N, N, D)
            .permute(0, 3, 1, 2)
            .reshape(Bflat * D, N, N)
            .contiguous()
        )
        del d_x_flat

        # The batched GEMM kernel reads operands via arbitrary strides, so
        # transposed views can be passed directly (no .contiguous() copies).
        if outgoing:
            b_comb = b_cm.transpose(-1, -2)
            a_comb = a_cm
        else:
            a_comb = a_cm.transpose(-1, -2)
            b_comb = b_cm

        # p = a_comb @ b_comb  -> da = dp @ b_comb^T ; db = a_comb^T @ dp
        d_a_comb = torch.bmm(d_x_cm, b_comb.transpose(-1, -2))
        d_b_comb = torch.bmm(a_comb.transpose(-1, -2), d_x_cm)
        del d_x_cm

        if outgoing:
            d_a_cm = d_a_comb
            d_b_cm = d_b_comb.transpose(-1, -2)
        else:
            d_a_cm = d_a_comb.transpose(-1, -2)
            d_b_cm = d_b_comb

        # Row-major [M, D] view of the channel-major [Bflat, D, N, N] combine
        # grads. The proj-in backward kernel reads its d_out via arbitrary
        # strides, so for the common Bflat==1 case we pass a strided view
        # (stride_m=1 over the N*N spatial dim, stride_d=N*N over channels) and
        # skip the permute->contiguous copy of two [M, D] tensors. For Bflat>1
        # the channel-major->row-major map is not a single 2D stride, so fall
        # back to a materialized permute.
        if Bflat == 1:
            # d_*_cm may be a transposed (non-contiguous) view; realize the
            # [D, N, N] channel-major layout without a row-major copy so the
            # [M, D] strided view is well-defined.
            d_a_cm_c = (
                d_a_cm.reshape(D, N, N).contiguous()
                if not d_a_cm.is_contiguous()
                else d_a_cm.reshape(D, N, N)
            )
            d_b_cm_c = (
                d_b_cm.reshape(D, N, N).contiguous()
                if not d_b_cm.is_contiguous()
                else d_b_cm.reshape(D, N, N)
            )
            NN = N * N
            d_a = d_a_cm_c.reshape(D, NN).as_strided((NN, D), (1, NN))
            d_b = d_b_cm_c.reshape(D, NN).as_strided((NN, D), (1, NN))
            del d_a_cm_c, d_b_cm_c
        else:
            # back to row-major [M, D]
            d_a = d_a_cm.reshape(Bflat, D, N, N).permute(0, 2, 3, 1).reshape(M, D)
            d_b = d_b_cm.reshape(Bflat, D, N, N).permute(0, 2, 3, 1).reshape(M, D)
        del d_a_comb, d_b_comb, d_a_cm, d_b_cm

        # ---- Input projections backward ----
        # a = mask * sigmoid(z_n @ Wag^T) * (z_n @ Wap^T)
        Wp_in = p_in_weight.to(cdtype)  # [2D, D], rows 0:D -> a, D:2D -> b
        Wg_in = g_in_weight.to(cdtype)
        # Stacked W^T = [2, K, N] = [2, D, D] contiguous for the dual kernel.
        # reshape [2D,D]->[2,D,D] (out,in) then transpose last two -> W^T per proj.
        Wp_t2 = Wp_in.reshape(2, D, D).transpose(-1, -2).contiguous()
        Wg_t2 = Wg_in.reshape(2, D, D).transpose(-1, -2).contiguous()

        # Both projections' sigmoid-derivative grads in a single launch (halves
        # the two small proj-in-bwd launches; a win in the launch-bound small-N
        # regime, neutral at large N). d_a/d_b may be strided [M, D] views over
        # the channel-major combine grad; the kernel reads them via their own
        # strides, so no copy here. Output is [M, 4, D] interleaved along the
        # proj axis: [d_ap_a, d_ag_a, d_ap_b, d_ag_b].
        d_all = _proj_in_bwd_pointwise_dual(z_n, Wp_t2, Wg_t2, d_a, d_b, mask_flat)
        del d_a, d_b
        d_ap_a, d_ag_a, d_ap_b, d_ag_b = (
            d_all[:, 0],
            d_all[:, 1],
            d_all[:, 2],
            d_all[:, 3],
        )

        Wap = Wp_in[:D]
        Wbp = Wp_in[D:]
        Wag = Wg_in[:D]
        Wbg = Wg_in[D:]
        # All four input-proj weight grads in one GEMM: the 4 upstream grads sit
        # contiguously along the feature axis of d_all, so [M, 4D]^T @ z_n reads
        # the shared z_n operand once (this GEMM category is HBM-bandwidth
        # bound; batching the shared read is the win over 4 separate ops).
        d_W_in = (d_all.reshape(M, 4 * D).T @ z_n).float()  # [4D, D]
        d_Wap = d_W_in[0 * D : 1 * D]
        d_Wag = d_W_in[1 * D : 2 * D]
        d_Wbp = d_W_in[2 * D : 3 * D]
        d_Wbg = d_W_in[3 * D : 4 * D]
        d_z_n = d_z_n + d_ap_a @ Wap + d_ag_a @ Wag
        d_z_n = d_z_n + d_ap_b @ Wbp + d_ag_b @ Wbg
        del d_all

        # ---- LN_in backward ----
        d_z_flat, d_norm_in_weight, d_norm_in_bias = _triton_layernorm_backward(
            d_z_n,
            z_flat,
            norm_in_weight,
            mean_in,
            rstd_in,
        )

        # Assemble fused weight grads.
        d_p_in_weight = torch.cat([d_Wap, d_Wbp], dim=0)
        d_g_in_weight = torch.cat([d_Wag, d_Wbg], dim=0)

        d_z = d_z_flat.reshape(*batch_dims, N, N, D).to(grad_out.dtype)

        return (
            d_z,
            None,  # mask
            None,  # outgoing
            d_norm_in_weight.to(norm_in_weight.dtype),
            d_norm_in_bias.to(norm_in_bias.dtype),
            d_p_in_weight.to(p_in_weight.dtype),
            d_g_in_weight.to(g_in_weight.dtype),
            d_norm_out_weight.to(norm_out_weight.dtype),
            d_norm_out_bias.to(norm_out_bias.dtype),
            d_p_out_weight.to(p_out_weight.dtype),
            d_g_out_weight.to(g_out_weight.dtype),
            None,  # eps
        )


def trimul_triton(
    x,
    direction="outgoing",
    mask=None,
    *,
    eps=1e-5,
    norm_in_weight=None,
    norm_in_bias=None,
    p_in_weight=None,
    p_in_bias=None,
    g_in_weight=None,
    g_in_bias=None,
    norm_out_weight=None,
    norm_out_bias=None,
    p_out_weight=None,
    p_out_bias=None,
    g_out_weight=None,
    g_out_bias=None,
):
    """Triangle multiplicative update on the fused Triton kernels.

    Takes the same arguments as ``trimul_torch``; see it for the layouts.

    The kernels implement the standard config, and the configs outside it are
    rejected here rather than ignored so that calling this directly cannot
    silently produce something else. The public op routes them to the torch
    path instead.
    """
    named = [
        name
        for name, value in (
            ("p_in_bias", p_in_bias),
            ("g_in_bias", g_in_bias),
            ("p_out_bias", p_out_bias),
            ("g_out_bias", g_out_bias),
        )
        if value is not None
    ]
    if named:
        raise ValueError(
            f"the Triton kernels have bias-free projections, got {', '.join(named)}"
        )
    c_hidden, c_z = p_in_weight.shape[0] // 2, x.shape[-1]
    if c_hidden != c_z:
        raise ValueError(
            f"the Triton kernels need a square hidden dim, got "
            f"c_hidden={c_hidden} against c_z={c_z}"
        )

    *batch_dims, n, _, d = x.shape
    x_flat = x.reshape(-1, n, n, d)
    if mask is None:
        mask_flat = x_flat.new_ones(x_flat.shape[:-1])
    else:
        mask_flat = mask.expand(x.shape[:-1]).reshape(-1, n, n).contiguous()

    out = _TriangleMultiplicativeUpdateFn.apply(
        x_flat,
        mask_flat,
        is_outgoing(direction),
        norm_in_weight,
        norm_in_bias,
        p_in_weight,
        g_in_weight,
        norm_out_weight,
        norm_out_bias,
        p_out_weight,
        g_out_weight,
        eps,
    )
    return out.reshape(*batch_dims, n, n, d)
