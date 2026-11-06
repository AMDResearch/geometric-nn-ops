# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layout-aware library GEMMs with Triton input and output normalization."""

import torch

from . import layout_aware
from .dual_projection_norm import row_norm
from .layout_aware_norm import channel_major_norm


def update(x, direction, mask, eps, p):
    if torch.get_float32_matmul_precision() != "highest":
        raise ValueError("Layout-aware normalization requires highest/IEEE matmul")
    if max(x.shape[-1], p.hidden) > 4096:
        raise ValueError("Layout-aware normalization supports at most 4096 channels")
    return layout_aware.update(
        x,
        direction,
        mask,
        eps,
        p,
        channel_major_norm=channel_major_norm,
        input_norm=row_norm,
    )
