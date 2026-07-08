# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TriMulPrecision enum, mirroring the fallback in cuequivariance_torch."""

import enum


class TriMulPrecision(enum.IntEnum):
    NONE = -1
    DEFAULT = 0
    TF32 = 1
    TF32x3 = 2
    IEEE = 3
