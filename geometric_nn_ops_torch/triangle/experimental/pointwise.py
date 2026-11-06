# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Advanced Micro Devices, Inc.
# Copyright 2021 DeepMind Technologies Limited
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Historical kernels; provenance and numerical limitations in README.md."""

import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_mul_kernel(
    x_ptr,
    gate_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: output = x * sigmoid(gate)
    """
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    if block_start >= N:
        return

    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offs >= 0) & (offs < N)

    safe_offs = offs.to(tl.int64)
    x_vals = tl.load(x_ptr + safe_offs, mask=mask, other=0.0)
    gate_vals = tl.load(gate_ptr + safe_offs, mask=mask, other=0.0)

    sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate_vals))
    result = x_vals * sigmoid_gate

    tl.store(output_ptr + safe_offs, result, mask=mask)


@triton.jit
def dual_elementwise_ops_kernel(
    x1_ptr,
    x2_ptr,
    x3_ptr,
    output_ptr,
    N,
    HAS_X3: tl.constexpr,  # Whether x3 is provided
    MUL_OP: tl.constexpr,  # True for multiply, False for add (only used if HAS_X3)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for elementwise operations:
    - If HAS_X3=False: output = x1 * x2
    - If HAS_X3=True and MUL_OP=True:  output = (x1 * x2) * x3
    - If HAS_X3=True and MUL_OP=False: output = (x1 * x2) + x3
    """
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    if block_start >= N:
        return

    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offs >= 0) & (offs < N)

    safe_offs = offs.to(tl.int64)
    x1_vals = tl.load(x1_ptr + safe_offs, mask=mask, other=0.0)
    x2_vals = tl.load(x2_ptr + safe_offs, mask=mask, other=0.0)

    result = x1_vals * x2_vals

    if HAS_X3:
        x3_vals = tl.load(x3_ptr + safe_offs, mask=mask, other=0.0)
        if MUL_OP:
            result = result * x3_vals
        else:
            result = result + x3_vals

    tl.store(output_ptr + safe_offs, result, mask=mask)


def triton_sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Fused sigmoid + elementwise multiply: x * sigmoid(gate)
    Supports broadcasting like PyTorch's native operations.

    Args:
        x: Input tensor
        gate: Gate tensor (broadcastable with x)

    Returns:
        x * sigmoid(gate)
    """
    broadcasted_shape = torch.broadcast_shapes(x.shape, gate.shape)
    x_expanded = x.expand(broadcasted_shape)
    gate_expanded = gate.expand(broadcasted_shape)

    x_flat = x_expanded.contiguous().reshape(-1)
    gate_flat = gate_expanded.contiguous().reshape(-1)
    N = x_flat.numel()

    output_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    sigmoid_mul_kernel[grid](x_flat, gate_flat, output_flat, N, BLOCK_SIZE=BLOCK_SIZE)

    return output_flat.reshape(broadcasted_shape)


def triton_dual_elementwise(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor | None = None,
    mul_op: bool = True,
) -> torch.Tensor:
    """
    Fused elementwise operation:
    - If x3 is None: output = x1 * x2
    - If x3 is provided and mul_op=True:  output = (x1 * x2) * x3
    - If x3 is provided and mul_op=False: output = (x1 * x2) + x3

    Args:
        x1, x2: Input tensors (must be same shape or broadcastable)
        x3: Optional third tensor for additional operation
        mul_op: If True, multiplies by x3; if False, adds x3 (only used if x3 provided)

    Returns:
        Result tensor
    """
    # Handle broadcasting
    if x3 is not None:
        broadcasted_shape = torch.broadcast_shapes(x1.shape, x2.shape, x3.shape)
        x3_flat = x3.expand(broadcasted_shape).contiguous().reshape(-1)
        x3_ptr = x3_flat
    else:
        broadcasted_shape = torch.broadcast_shapes(x1.shape, x2.shape)
        # Create a dummy tensor for x3_ptr (won't be used due to HAS_X3=False)
        x3_ptr = x1  # placeholder, guarded by HAS_X3 constexpr

    x1_flat = x1.expand(broadcasted_shape).contiguous().reshape(-1)
    x2_flat = x2.expand(broadcasted_shape).contiguous().reshape(-1)

    N = x1_flat.numel()
    output_flat = torch.empty_like(x1_flat)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    dual_elementwise_ops_kernel[grid](
        x1_flat,
        x2_flat,
        x3_ptr if x3 is not None else x1_flat,
        output_flat,
        N,
        HAS_X3=(x3 is not None),
        MUL_OP=mul_op,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output_flat.reshape(broadcasted_shape)
