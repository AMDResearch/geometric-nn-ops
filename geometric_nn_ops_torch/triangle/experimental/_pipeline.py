# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Common functional contract and stage boundaries for historical experiments."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from geometric_nn_ops_torch.triangle._reference.trimul import (
    _combine_projections,
    is_outgoing,
)


@dataclass
class Parameters:
    p_in_weight: torch.Tensor
    g_in_weight: torch.Tensor
    p_out_weight: torch.Tensor
    g_out_weight: torch.Tensor
    norm_in_weight: torch.Tensor | None = None
    norm_in_bias: torch.Tensor | None = None
    norm_out_weight: torch.Tensor | None = None
    norm_out_bias: torch.Tensor | None = None
    p_in_bias: torch.Tensor | None = None
    g_in_bias: torch.Tensor | None = None
    p_out_bias: torch.Tensor | None = None
    g_out_bias: torch.Tensor | None = None

    @property
    def hidden(self):
        return self.p_in_weight.shape[0] // 2

    def kwargs(self):
        return vars(self)


def validate(x, direction, mask, eps, p, *, gpu=False):
    is_outgoing(direction)
    if x.ndim < 3 or x.shape[-3] != x.shape[-2] or min(x.shape) <= 0:
        raise ValueError("Expected nonempty [*, N, N, C] input")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("LayerNorm epsilon must be finite and positive")
    if p.p_in_weight.ndim != 2:
        raise ValueError("p_in_weight must be a matrix")
    c, h = x.shape[-1], p.hidden
    shapes = {
        "p_in_weight": (2 * h, c),
        "g_in_weight": (2 * h, c),
        "p_out_weight": (c, h),
        "g_out_weight": (c, c),
        "p_in_bias": (2 * h,),
        "g_in_bias": (2 * h,),
        "p_out_bias": (c,),
        "g_out_bias": (c,),
        "norm_in_weight": (c,),
        "norm_in_bias": (c,),
        "norm_out_weight": (h,),
        "norm_out_bias": (h,),
    }
    if h <= 0:
        raise ValueError("Hidden channels must be positive")
    for name, shape in shapes.items():
        tensor = getattr(p, name)
        if tensor is not None and (
            tuple(tensor.shape) != shape
            or tensor.device != x.device
            or tensor.dtype != x.dtype
        ):
            raise ValueError(f"{name} must have shape {shape}, input dtype/device")
    if mask is not None:
        if mask.device != x.device:
            raise ValueError("Mask must share the input device")
        if torch.broadcast_shapes(mask.shape, x.shape[:-1]) != x.shape[:-1]:
            raise ValueError(
                "Mask must broadcast to input pair dimensions, not expand them"
            )
    if gpu and (not x.is_cuda or x.dtype != torch.float32):
        raise ValueError("Historical kernel experiments require CUDA/HIP FP32 inputs")
    if (
        gpu
        and torch.is_grad_enabled()
        and (
            x.requires_grad
            or any(t is not None and t.requires_grad for t in p.kwargs().values())
            or (mask is not None and mask.requires_grad)
        )
    ):
        raise ValueError("Historical kernels are forward-only; disable gradients")
    if gpu and torch.is_autocast_enabled(x.device.type):
        raise ValueError("Historical kernels require autocast disabled")


def norm(x, weight, bias, eps):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def finish(o, z, p, eps):
    projected = F.linear(
        norm(o, p.norm_out_weight, p.norm_out_bias, eps),
        p.p_out_weight,
        p.p_out_bias,
    )
    return projected * torch.sigmoid(F.linear(z, p.g_out_weight, p.g_out_bias))


def projections(z, mask, p):
    ab = F.linear(z, p.p_in_weight, p.p_in_bias) * torch.sigmoid(
        F.linear(z, p.g_in_weight, p.g_in_bias)
    )
    if mask is not None:
        ab = ab * mask.unsqueeze(-1)
    return ab.split(p.hidden, dim=-1)


def compose(kind, x, direction, mask, eps, p):
    """Same full update, varying only the selected historical kernel boundaries."""
    from importlib import import_module

    if kind == "dual_projection" and p.g_out_bias is not None:
        raise ValueError("Historical dual-projection output gate has no bias support")
    kernels = import_module(f"{__package__}.{kind}")
    z = norm(x, p.norm_in_weight, p.norm_in_bias, eps)
    row_mask = None if mask is None else mask.expand(x.shape[:-1]).unsqueeze(-1)
    if kind == "pointwise":
        gate = torch.sigmoid(F.linear(z, p.g_in_weight, p.g_in_bias))
        ab = kernels.triton_dual_elementwise(
            F.linear(z, p.p_in_weight, p.p_in_bias),
            gate,
            row_mask,
        )
        a, b = ab.split(p.hidden, dim=-1)
    elif kind in ("projection_epilogue", "projection_epilogue_300x"):
        gate = kernels.triton_linear_fused(
            z,
            p.g_in_weight,
            p.g_in_bias,
            apply_sigmoid=True,
        )
        ab = kernels.triton_linear_fused(
            z,
            p.p_in_weight,
            p.p_in_bias,
            other=gate,
            mask=row_mask,
        )
        a, b = ab.split(p.hidden, dim=-1)
    elif kind == "dual_projection":
        ab = kernels.triton_dual_linear_sigmoid_mul(
            z,
            p.g_in_weight,
            p.g_in_bias,
            p.p_in_weight,
            p.p_in_bias,
            row_mask,
        )
        a, b = ab.split(p.hidden, dim=-1)
    elif kind == "dual_normalization":
        a, b = projections(z, mask, p)
    else:
        return kernels.update(x, z, direction, mask, eps, p)
    outgoing = is_outgoing(direction)
    if kind == "dual_projection":
        aa = a.movedim(-1, -3)
        bb = b.movedim(-1, -3)
        o = kernels.triton_batched_matmul(
            aa if outgoing else aa.transpose(-1, -2),
            bb.transpose(-1, -2) if outgoing else bb,
        ).movedim(-3, -1)
    else:
        o = _combine_projections(a, b, outgoing)
    if kind == "pointwise":
        projected = F.linear(
            norm(o, p.norm_out_weight, p.norm_out_bias, eps),
            p.p_out_weight,
            p.p_out_bias,
        )
        return kernels.triton_sigmoid_mul(
            projected,
            F.linear(z, p.g_out_weight, p.g_out_bias),
        )
    if kind in ("projection_epilogue", "projection_epilogue_300x"):
        gate = kernels.triton_linear_fused(
            z,
            p.g_out_weight,
            p.g_out_bias,
            apply_sigmoid=True,
        )
        return kernels.triton_linear_fused(
            norm(o, p.norm_out_weight, p.norm_out_bias, eps),
            p.p_out_weight,
            p.p_out_bias,
            other=gate,
        )
    if kind == "dual_projection":
        projected = F.linear(
            norm(o, p.norm_out_weight, p.norm_out_bias, eps),
            p.p_out_weight,
            p.p_out_bias,
        )
        return kernels.triton_linear_sigmoid_mul(projected, z, p.g_out_weight)
    return kernels.output(o, x, p, eps)
