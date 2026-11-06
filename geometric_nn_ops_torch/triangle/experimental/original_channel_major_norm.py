# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Original forward projections plus bmm and register-resident LayerNorm."""

from . import original_channel_major


def update(x, direction, mask, eps, p):
    return original_channel_major.update(x, direction, mask, eps, p, register_norm=True)
