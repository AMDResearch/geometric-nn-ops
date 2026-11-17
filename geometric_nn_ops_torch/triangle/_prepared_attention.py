# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prepared inference-only triangle attention."""

import math
import os

import torch

from geometric_nn_ops_torch import _precision

from ._masking import masked_key_bias


def _require_inference(x):
    if torch.is_grad_enabled() or torch.is_autocast_enabled():
        raise ValueError("Optimized FP32 inference requires no_grad and no autocast")
    if x.dtype != torch.float32 or x.requires_grad:
        raise ValueError("Optimized inference requires detached FP32 activations")
    if os.environ.get("TRITON_INTERPRET") == "1" and x.device.type == "cpu":
        return
    if x.device.type != "cuda":
        raise ValueError("Optimized inference requires a gfx950 GPU")
    properties = torch.cuda.get_device_properties(x.device)
    if getattr(properties, "gcnArchName", "").split(":")[0] != "gfx950":
        raise ValueError("Optimized inference is currently supported only on gfx950")
    if _precision.torch_precision() != "ieee":
        raise ValueError("Optimized inference requires FP32 matmul precision 'ieee'")


def _tensor(value, shape, device, name, *, contiguous=True):
    if (
        value.shape != shape
        or value.dtype != torch.float32
        or value.device != device
        or (contiguous and not value.is_contiguous())
    ):
        raise ValueError(
            f"{name} must be FP32 {shape} on {device}"
            + (" and contiguous" if contiguous else "")
        )


class PreparedTriangleAttention:
    """Prepare invocation-local pair bias and mask for primary pair attention.

    ``pair_bias``: [1, 1, 12, N, N] FP32; ``mask``: [1, N, 1, 1, N] bool,
    True means keep. Q/K/V and optional *already sigmoid-transformed* gate use
    the public [1, rows, N, 12, 32] layout. Only batch=1, rows<=32, N<=3488,
    FP32 gfx950 inference is supported. Recreate after bias/mask updates.

    Construct once per shared bias and mask, then call with each Q/K/V chunk
    and its starting row. The existing functional attention API is unchanged.
    """

    def __init__(self, pair_bias, mask):
        _require_inference(pair_bias)
        if pair_bias.ndim != 5:
            raise ValueError("pair_bias must have five axes")
        n = pair_bias.shape[-1]
        if not 1 <= n <= 3488:
            raise ValueError("Prepared attention requires 1 <= N <= 3488")
        _tensor(
            pair_bias, (1, 1, 12, n, n), pair_bias.device, "pair_bias", contiguous=False
        )
        if (
            mask.shape != (1, n, 1, 1, n)
            or mask.dtype != torch.bool
            or mask.device != pair_bias.device
        ):
            raise ValueError("mask must be bool [1, N, 1, 1, N] on the bias device")
        self.n = n
        # Own snapshots rather than silently retaining mutable caller storage.
        self.bias = pair_bias.detach().clone(memory_format=torch.contiguous_format)
        self.mask = masked_key_bias(mask, torch.float32).contiguous()

    def __call__(
        self, q, k, v, *, start, gate=None, scale=None, return_logsumexp=False
    ):
        """Return O, or (O, natural-log sumexp [1, rows, 12, N])."""
        _require_inference(q)
        if not isinstance(return_logsumexp, bool):
            raise ValueError("return_logsumexp must be boolean")
        if q.ndim != 5:
            raise ValueError("Q/K/V must have five axes")
        rows = q.shape[1]
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not 1 <= rows <= 32
            or not 0 <= start <= self.n - rows
        ):
            raise ValueError("Invalid attention row chunk/start")
        shape = (1, rows, self.n, 12, 32)
        for name, value in (("q", q), ("k", k), ("v", v)):
            _tensor(value, shape, self.bias.device, name, contiguous=False)
            if (
                value.requires_grad
                or value.stride(-1) != 1
                or any(s <= 0 for s in value.stride())
            ):
                raise ValueError(f"{name} requires detached, positive-stride channels")
        if gate is not None:
            _tensor(gate, shape, q.device, "gate", contiguous=False)
            if (
                gate.requires_grad
                or gate.stride(-1) != 1
                or any(s <= 0 for s in gate.stride())
            ):
                raise ValueError("gate requires detached, positive-stride channels")
        scale = 32**-0.5 if scale is None else float(scale)
        if not math.isfinite(scale):
            raise ValueError("scale must be finite")
        import triton

        from ._kernels.prepared_attention import _attention
        from ._kernels.triangle_attention import _hip_kernel_options

        canonical = [value.transpose(2, 3) for value in (q, k, v)]
        g = canonical[0] if gate is None else gate.transpose(2, 3)
        o = torch.empty_like(canonical[0])
        m = torch.empty((1, rows, 12, self.n), device=q.device, dtype=q.dtype)
        mask = self.mask[:, start : start + rows]
        options = dict(num_warps=4, num_stages=1)
        if q.device.type == "cuda":
            options.update(_hip_kernel_options(dim=32))
        _attention[(triton.cdiv(self.n, 64) * rows * 12,)](
            *canonical,
            mask,
            self.bias,
            g,
            o,
            m,
            *canonical[0].stride(),
            *canonical[1].stride(),
            *canonical[2].stride(),
            *o.stride(),
            *g.stride(),
            mask.stride(1),
            mask.stride(4),
            self.bias.stride(2),
            self.bias.stride(3),
            self.bias.stride(4),
            rows,
            self.n,
            scale,
            gate is not None,
            **options,
        )
        result = o.transpose(2, 3)
        return (result, m) if return_logsumexp else result
