# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""pytest-only helpers.

The metrics, tolerances and the oracle assertion live in ``testutil.metrics``,
which benchmarks can import too. Only things that depend on pytest belong here.
"""

import pytest
import torch


def skip_without_device_memory(gib: float) -> None:
    """Skip unless the device reports ``need_gib`` free.

    Pairs with the ``large`` marker: the marker keeps big shapes out of a
    default run, and this keeps an opted-in run on a smaller device from OOMing.
    """
    free, _ = torch.cuda.mem_get_info()
    free_gib = free / 1024**3
    if free_gib < gib:
        pytest.skip(f"needs ~{gib:.0f} GiB free, device reports {free_gib:.0f}")


DTYPES_PARAM = pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16],
    ids=["fp32", "bf16", "fp16"],
)
