# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Channel-major Triton projections with a measured single-stage schedule."""

from . import initial_kernels, original_channel_major
from .all_triton import normalize


class _ProjectionSchedule:
    def __init__(self, kernel, *, block_n=64, matrix_instr_nonkdim=None):
        self.kernel = kernel.fn
        self.block_n = block_n
        self.options = (
            {}
            if matrix_instr_nonkdim is None
            else {"matrix_instr_nonkdim": matrix_instr_nonkdim}
        )

    def __getitem__(self, grid):
        def launch(*args):
            meta = {"BLOCK_M": 128, "BLOCK_N": self.block_n, "BLOCK_K": 32}
            return self.kernel[grid(meta)](
                *args, **meta, num_stages=1, num_warps=4, **self.options
            )

        return launch


_producer = _ProjectionSchedule(
    initial_kernels.fused_gemm_sigmoid_multiply_channel_major
)
_output = _ProjectionSchedule(initial_kernels.fused_gemm_sigmoid_multiply)


def update(x, direction, mask, eps, p, *, producer_kernel=None):
    if max(x.shape[-1], p.hidden) > 4096:
        raise ValueError("Layout-aware pipeline supports at most 4096 channels")
    return original_channel_major.update(
        x,
        direction,
        mask,
        eps,
        p,
        normalize=normalize,
        pack_weights=False,
        producer_kernel=_producer if producer_kernel is None else producer_kernel,
        output_kernel=_output,
    )
