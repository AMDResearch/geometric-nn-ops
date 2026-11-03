# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test utils for torch precision setting."""

import contextlib
from unittest import mock

import torch

from geometric_nn_ops_torch.triangle._kernels import dot_precision


def fast_reduced_precision_dot(available: bool):
    """Pretend the device does or doesn't have a fast reduced-precision dot, so
    tests of how the setting is read and plumbed don't depend on the hardware."""
    return mock.patch.object(
        dot_precision, "has_fast_reduced_precision_fp32_dot", return_value=available
    )


@contextlib.contextmanager
def torch_matmul_precision(precision: str):
    """Set torch's fp32 matmul precision for the duration.

    Falls back to ``allow_tf32`` on torch versions predating
    ``matmul.fp32_precision``, matching ``dot_precision.torch_fp32_precision``.

    Swaps a process global, torch having no scoped API, so this is not thread
    safe (hence test only). Both the forward and the backward must be called
    inside the context.
    """
    matmul = torch.backends.cuda.matmul
    modern = hasattr(matmul, "fp32_precision")
    attribute = "fp32_precision" if modern else "allow_tf32"
    previous = getattr(matmul, attribute)
    setattr(matmul, attribute, precision if modern else precision == "tf32")
    try:
        yield
    finally:
        setattr(matmul, attribute, previous)
