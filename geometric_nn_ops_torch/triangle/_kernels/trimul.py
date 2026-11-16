# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layout-aware Triton-assisted triangle multiplication, forward only."""

import math
import os

import torch

from geometric_nn_ops_torch.triangle._reference.trimul import is_outgoing

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _welford_merge(mean_a, m2_a, count_a, mean_b, m2_b, count_b):
        count = count_a + count_b
        ratio = count_b / tl.maximum(count, 1.0)
        delta = mean_b - mean_a
        return (
            mean_a + delta * ratio,
            m2_a + m2_b + delta * delta * count_a * ratio,
            count,
        )

    @triton.jit
    def _row_norm(
        X,
        W,
        B,
        Y,
        C: tl.constexpr,
        SX: tl.constexpr,
        EPS: tl.constexpr,
        BC: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        col = tl.arange(0, BC)
        x = tl.load(X + row * SX + col, col < C, 0).to(tl.float32)
        # Shifting preserves small variances when the input has a large offset.
        origin = tl.load(X + row * SX).to(tl.float32)
        x = tl.where(col < C, x - origin, 0.0)
        mean, m2, count = tl.reduce(
            (x, tl.full((BC,), 0.0, tl.float32), (col < C).to(tl.float32)),
            0,
            _welford_merge,
        )
        y = (x - mean) * tl.rsqrt(m2 / count + EPS)
        y = y * tl.load(W + col, col < C, 0) + tl.load(B + col, col < C, 0)
        tl.store(Y + row * C + col, y, col < C)

    @triton.jit
    def _channel_norm(
        X,
        W,
        B,
        Y,
        R: tl.constexpr,
        C: tl.constexpr,
        SX: tl.constexpr,
        EPS: tl.constexpr,
        BC: tl.constexpr,
        BR: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
        col = tl.arange(0, BC)
        valid = (col[:, None] < C) & (row[None, :] < R)
        x = tl.load(X + col[:, None].to(tl.int64) * SX + row[None, :], valid, 0).to(
            tl.float32
        )
        origin = tl.load(X + row, row < R, 0).to(tl.float32)
        x = tl.where(valid, x - origin[None, :], 0.0)
        counts = tl.broadcast_to((col[:, None] < C).to(tl.float32), (BC, BR))
        mean, m2, count = tl.reduce(
            (x, tl.full((BC, BR), 0.0, tl.float32), counts),
            0,
            _welford_merge,
        )
        y = (x - mean[None, :]) * tl.rsqrt(m2[None, :] / count[None, :] + EPS)
        y *= tl.load(W + col, col < C, 0)[:, None]
        y += tl.load(B + col, col < C, 0)[:, None]
        tl.store(Y + col[:, None].to(tl.int64) * R + row[None, :], y, valid)

    def welford_norm(x, weight, bias, eps, *, channel_major=False):
        """Normalize a validated matrix, retaining its consumer's layout."""
        rows, channels = x.shape
        if channel_major:
            y = torch.empty_strided(
                (rows, channels), (1, rows), device=x.device, dtype=x.dtype
            )
            _channel_norm[(triton.cdiv(rows, 16),)](
                x,
                weight.detach(),
                bias.detach(),
                y,
                rows,
                channels,
                x.stride(1),
                eps,
                triton.next_power_of_2(channels),
                16,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            y = torch.empty((rows, channels), device=x.device, dtype=x.dtype)
            _row_norm[(rows,)](
                x,
                weight.detach(),
                bias.detach(),
                y,
                channels,
                x.stride(0),
                eps,
                triton.next_power_of_2(channels),
                num_warps=4,
                enable_fp_fusion=False,
            )
        return y

    @triton.jit
    def _input_gate(PG, MASK, M: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = i < K * M
        p = tl.load(PG + i, valid, 0).to(tl.float32)
        g = tl.load(PG + i + K * M, valid, 0).to(tl.float32)
        mask = tl.load(MASK + i % M, valid, 0).to(tl.float32)
        tl.store(PG + i, p * (1.0 / (1.0 + tl.exp(-g))) * mask, valid)

    @triton.jit
    def _output_gate(P, G, SIZE: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = i < SIZE
        p = tl.load(P + i, valid, 0).to(tl.float32)
        g = tl.load(G + i, valid, 0).to(tl.float32)
        tl.store(P + i, p * (1.0 / (1.0 + tl.exp(-g))), valid)


_LN_MAX_ELEMENTS = 1 << 30
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _require_inference(x):
    if torch.is_grad_enabled() or torch.is_autocast_enabled():
        raise ValueError("Optimized inference requires no_grad and no autocast")
    if x.dtype not in _SUPPORTED_DTYPES or x.requires_grad:
        raise ValueError(
            "Optimized inference requires detached FP32, FP16 or BF16 activations"
        )
    if os.environ.get("TRITON_INTERPRET") == "1" and x.device.type == "cpu":
        return
    if x.device.type != "cuda":
        raise ValueError("Optimized inference requires a gfx950 GPU")
    properties = torch.cuda.get_device_properties(x.device)
    if getattr(properties, "gcnArchName", "").split(":")[0] != "gfx950":
        raise ValueError("Optimized inference is currently supported only on gfx950")


def _tensor(value, shape, device, dtype, name, *, contiguous=True):
    if (
        value.shape != shape
        or value.dtype != dtype
        or value.device != device
        or (contiguous and not value.is_contiguous())
    ):
        raise ValueError(
            f"{name} must be {dtype} {shape} on {device}"
            + (" and contiguous" if contiguous else "")
        )


class LayoutAwareTriangleMultiplicativeUpdate:
    """Bounded, bias-free projection TriMul with true Welford normalizations.

    Parameters use the existing concatenated [a; b] p_in/g_in convention.
    Torch GEMMs and bmm perform projections/contractions. No model import,
    global hooks or training fallback is installed. Weights remain caller-owned;
    ordinary in-place weight updates invalidate cached grouped weights.
    Rebuild this object after replacing parameter tensors (e.g. model.to()).
    """

    def __init__(
        self,
        *,
        norm_in_weight,
        norm_in_bias,
        p_in_weight,
        g_in_weight,
        norm_out_weight,
        norm_out_bias,
        p_out_weight,
        g_out_weight,
        p_in_bias=None,
        g_in_bias=None,
        p_out_bias=None,
        g_out_bias=None,
        eps=1e-5,
        norm_out_eps=None,
        memory_budget_bytes=12 << 30,
        triton_input_gate=False,
        triton_output_gate=False,
    ):
        self.weights = (
            norm_in_weight,
            norm_in_bias,
            p_in_weight,
            g_in_weight,
            norm_out_weight,
            norm_out_bias,
            p_out_weight,
            g_out_weight,
        )
        self.biases = (p_in_bias, g_in_bias, p_out_bias, g_out_bias)
        self.eps = float(eps)
        self.norm_out_eps = self.eps if norm_out_eps is None else float(norm_out_eps)
        if not all(math.isfinite(e) and e > 0 for e in (self.eps, self.norm_out_eps)):
            raise ValueError("LayerNorm epsilons must be positive and finite")
        if (
            not isinstance(memory_budget_bytes, int)
            or isinstance(memory_budget_bytes, bool)
            or memory_budget_bytes < 1
        ):
            raise ValueError("memory_budget_bytes must be a positive integer")
        if not isinstance(triton_input_gate, bool) or not isinstance(
            triton_output_gate, bool
        ):
            raise ValueError("Triton gate switches must be boolean")
        self.memory_budget_bytes = memory_budget_bytes
        self.triton_input_gate = triton_input_gate
        self.triton_output_gate = triton_output_gate
        self._cached_key = None
        self._cached_weights = None

    def _grouped_weights(self, hc, h, c):
        pw, gw = self.weights[2:4]
        # Inference tensors have no version counter; never cache them.
        key = (
            None
            if pw.is_inference() or gw.is_inference()
            else (
                hc,
                pw.data_ptr(),
                pw._version,
                gw.data_ptr(),
                gw._version,
            )
        )
        if key is not None and key == self._cached_key:
            return self._cached_weights
        parts = (*pw.detach().split(h), *gw.detach().split(h))
        grouped = (
            torch.stack(parts)
            .view(4, h // hc, hc, c)
            .transpose(0, 1)
            .reshape(h // hc, 4 * hc, c)
            .contiguous()
        )
        self._cached_key, self._cached_weights = key, grouped
        return grouped

    def __call__(self, x, direction="outgoing", mask=None, *, out=None, add=False):
        """Update [*, N, N, C]; optional contiguous ``out`` may alias ``x``.

        ``add=True`` requires ``out`` and adds the update to its existing values.
        Masks may be broadcast bool or floating point, including fractional weights.
        """
        _require_inference(x)
        outgoing = is_outgoing(direction)
        if (
            x.ndim < 3
            or x.shape[-3] != x.shape[-2]
            or not x.is_contiguous()
            or x.numel() == 0
        ):
            raise ValueError("x must be a nonempty contiguous square pair tensor")
        n, c = x.shape[-2:]
        pw = self.weights[2]
        if pw.ndim != 2 or pw.shape[0] % 2:
            raise ValueError("p_in_weight must have 2 * hidden rows")
        h = pw.shape[0] // 2
        if not min(h, c) >= 1 or max(h, c) > 1024:
            raise ValueError("Input and hidden channels must be in [1, 1024]")
        shapes = [(c,), (c,), (2 * h, c), (2 * h, c), (h,), (h,), (c, h), (c, c)]
        for i, (weight, shape) in enumerate(zip(self.weights, shapes, strict=True)):
            _tensor(weight, shape, x.device, x.dtype, f"weight[{i}]")
        for i, (bias, shape) in enumerate(
            zip(self.biases, ((2 * h,), (2 * h,), (c,), (c,)), strict=True)
        ):
            if bias is not None:
                _tensor(bias, shape, x.device, x.dtype, f"bias[{i}]")
        if not isinstance(add, bool) or (add and out is None):
            raise ValueError("add must be boolean and requires out")
        if out is not None:
            _tensor(out, x.shape, x.device, x.dtype, "out")
            if out.requires_grad:
                raise ValueError("out must not require gradients")
            storage = out.untyped_storage().data_ptr()
            if (
                storage == x.untyped_storage().data_ptr()
                and out.data_ptr() != x.data_ptr()
            ):
                raise ValueError("out may alias x exactly, not partially")
            if any(storage == w.untyped_storage().data_ptr() for w in self.weights):
                raise ValueError("out must not alias weights")
        if mask is not None:
            if mask.device != x.device or mask.dtype not in (
                torch.bool,
                *_SUPPORTED_DTYPES,
            ):
                raise ValueError(
                    "mask must be bool, FP32, FP16 or BF16 on the input device"
                )
            if mask.requires_grad:
                raise ValueError("mask must not require gradients")
            if (
                out is not None
                and mask.untyped_storage().data_ptr()
                == out.untyped_storage().data_ptr()
            ):
                raise ValueError("out must not alias mask")
            mask = mask.broadcast_to(x.shape[:-1]).reshape(-1, n, n)
        m = n * n
        if self.memory_budget_bytes < max(4 * m * 4, 4 * (2 * h + 3 * c)):
            raise ValueError("memory_budget_bytes cannot hold one head/output row")
        cap = min(h, self.memory_budget_bytes // (4 * m * 4))
        hc = next(d for d in range(cap, 0, -1) if h % d == 0)
        grouped = self._grouped_weights(hc, h, c)
        result = torch.empty_like(x) if out is None else out
        xb, rb = x.view(-1, n, n, c), result.view(-1, n, n, c)
        for batch in range(xb.shape[0]):
            pair_mask = (
                x.new_ones(1, m)
                if mask is None
                else mask[batch].reshape(1, m).to(x.dtype).contiguous()
            )
            self._update(xb[batch], rb[batch], pair_mask, grouped, h, hc, outgoing, add)
        return result

    def _update(self, x, result, mask, grouped, h, hc, outgoing, add):
        import triton

        n, _, c = x.shape
        m = n * n
        niw, nib, _, _, now, nob, pow_, gow = self.weights
        pib, gib, pob, gob = self.biases
        xf = x.view(m, c)
        ln_rows = max(1, _LN_MAX_ELEMENTS // c)
        if m <= ln_rows:
            zn = welford_norm(xf, niw, nib, self.eps)
        else:
            zn = torch.empty_like(xf)
            for r in range(0, m, ln_rows):
                zn[r : r + ln_rows] = welford_norm(
                    xf[r : r + ln_rows], niw, nib, self.eps
                )
        contracted = x.new_empty(h, n, n)
        for i, weight in enumerate(grouped):
            pg = torch.mm(weight, zn.t())
            if pib is not None or gib is not None:
                p_bias = pib if pib is not None else x.new_zeros(2 * h)
                g_bias = gib if gib is not None else x.new_zeros(2 * h)
                bias = torch.cat(
                    (
                        p_bias[i * hc : (i + 1) * hc],
                        p_bias[h + i * hc : h + (i + 1) * hc],
                        g_bias[i * hc : (i + 1) * hc],
                        g_bias[h + i * hc : h + (i + 1) * hc],
                    )
                )
                pg.add_(bias[:, None])
            if self.triton_input_gate:
                _input_gate[(triton.cdiv(2 * hc * m, 1024),)](
                    pg, mask, m, 2 * hc, 1024, num_warps=4, enable_fp_fusion=False
                )
                ab = pg[: 2 * hc]
            else:
                ab = pg[: 2 * hc].mul_(pg[2 * hc :].sigmoid_()).mul_(mask)
            a, b = ab[:hc].view(hc, n, n), ab[hc:].view(hc, n, n)
            target = contracted[i * hc : (i + 1) * hc]
            if outgoing:
                torch.bmm(a, b.transpose(1, 2), out=target)
            else:
                torch.bmm(a.transpose(1, 2), b, out=target)
            del pg, ab, a, b
        rows = max(
            1,
            min(
                m,
                _LN_MAX_ELEMENTS // max(h, c),
                self.memory_budget_bytes // (4 * (2 * h + 3 * c)),
            ),
        )
        chunks = triton.cdiv(m, rows)
        rows = triton.cdiv(m, chunks)
        cf = contracted.view(h, m)
        rf = result.view(m, c)
        for r in range(0, m, rows):
            normalized = welford_norm(
                cf[:, r : r + rows].t(),
                now,
                nob,
                self.norm_out_eps,
                channel_major=True,
            )
            value = torch.mm(normalized, pow_.t())
            gate = torch.mm(zn[r : r + rows], gow.t())
            if pob is not None:
                value.add_(pob)
            if gob is not None:
                gate.add_(gob)
            if self.triton_output_gate:
                _output_gate[(triton.cdiv(value.numel(), 1024),)](
                    value,
                    gate,
                    value.numel(),
                    1024,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
                value.mul_(gate.sigmoid_())
            if add:
                rf[r : r + rows].add_(value)
            else:
                rf[r : r + rows].copy_(value)
            del normalized, gate, value


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
    """Layout-aware forward-only triangle multiplication."""
    with torch.no_grad():
        op = LayoutAwareTriangleMultiplicativeUpdate(
            norm_in_weight=norm_in_weight,
            norm_in_bias=norm_in_bias,
            p_in_weight=p_in_weight,
            p_in_bias=p_in_bias,
            g_in_weight=g_in_weight,
            g_in_bias=g_in_bias,
            norm_out_weight=norm_out_weight,
            norm_out_bias=norm_out_bias,
            p_out_weight=p_out_weight,
            p_out_bias=p_out_bias,
            g_out_weight=g_out_weight,
            g_out_bias=g_out_bias,
            eps=eps,
            triton_input_gate=True,
            triton_output_gate=True,
        )
        return op(x.detach().contiguous(), direction, mask)
