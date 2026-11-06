# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mask conversion shared by the shim ops."""

import torch


def to_bool_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    """Cast a cuequivariance 0/1 mask to bool.

    Raises if the mask requires grad, which the cast would silently drop.
    """
    if mask is None:
        return None
    if mask.requires_grad:
        raise ValueError("mask is a keep/drop flag and cannot require grad")
    return mask.bool()
