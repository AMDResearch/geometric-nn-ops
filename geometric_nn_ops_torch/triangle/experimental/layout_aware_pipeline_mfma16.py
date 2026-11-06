# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Wider MFMA16 producer tiles; output schedule and library contraction unchanged."""

from dataclasses import replace

from . import initial_kernels, layout_aware_pipeline

_producer = layout_aware_pipeline._ProjectionSchedule(
    initial_kernels.fused_gemm_sigmoid_multiply_channel_major,
    block_n=128,
    matrix_instr_nonkdim=16,
)


def update(x, direction, mask, eps, p):
    p = replace(
        p,
        **{
            name: value.contiguous()
            for name, value in p.kwargs().items()
            if value is not None and value.ndim == 1
        },
    )
    return layout_aware_pipeline.update(
        x, direction, mask, eps, p, producer_kernel=_producer
    )
