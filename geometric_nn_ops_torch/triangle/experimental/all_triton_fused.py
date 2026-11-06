# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""All-Triton path with fused input normalization and channel-major producers."""

from .all_triton import update_fused as update

__all__ = ["update"]
