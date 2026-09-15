# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""pytest-only helpers.

The metrics, tolerances and the oracle assertion live in ``testutil.metrics``,
which benchmarks can import too. Only things that depend on pytest belong here.
"""

import math

import pytest
import torch

from geometric_nn_ops_torch import _precision
from testutil.precision import torch_matmul_precision

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs GPU + Triton"
)


def skip_without_reduced_precision_dot() -> None:
    """Skip unless the target has a reduced-precision fp32 matmul.

    Where it does not, tf32 correctly resolves to ieee and there is nothing to
    tell apart.
    """
    with torch_matmul_precision("tf32"):
        if _precision.matmul_precision(fp32_operands=True) == _precision.IEEE:
            pytest.skip("no reduced-precision fp32 dot on this target")


def require_device_memory(gib: float) -> None:
    """Fail unless ``gib`` of device memory is available.

    This is a failure rather than skip to avoid silent skips based on dynamic
    behavior. Tests using this should generally be gated behind the 'large'
    marker, so if they're getting run it's because they've been specifically
    requested. We could just let the OOM happen, but the tests are slow to run
    and OOMs can take down unified memory systems.

    Counts blocks the caching allocator holds but is not using: torch reuses or
    releases those as it needs them, so they are available even though
    `mem_get_info` reports them as used. Without that a mid-suite call
    under-reports by however much the suite has churned through. Free rather
    than total, because torch cannot reclaim what another process holds.
    """
    # Report the calling test's location rather than this helper's.
    __tracebackhide__ = True
    free, _ = torch.cuda.mem_get_info()
    cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    available_gib = (free + cached) / 1024**3
    if available_gib < gib:
        diff = gib - available_gib
        first_diff_digit = math.floor(math.log10(diff))
        precision = 0 if first_diff_digit >= 0 else abs(first_diff_digit)
        pytest.fail(
            f"requires {gib} GiB of device memory, only "
            f"{available_gib:.{precision}f} GiB available"
        )


def gpu_arch() -> str:
    """The architecture name of the current device.

    e.g. ``gfx1151`` on ROCm, ``sm_90`` on CUDA.
    """
    props = torch.cuda.get_device_properties(0)
    if torch.version.hip:
        # The name carries feature flags on some parts: ``gfx90a:sramecc+:xnack-``.
        return props.gcnArchName.split(":")[0]
    return f"sm_{props.major}{props.minor}"


DTYPES_PARAM = pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16],
    ids=["fp32", "bf16", "fp16"],
)

DTYPE_PRECISION_PARAM = pytest.mark.parametrize(
    ("dtype", "precision"),
    [
        (torch.float32, "ieee"),
        (torch.float32, "tf32"),
        (torch.bfloat16, "ieee"),
        (torch.float16, "ieee"),
    ],
    ids=["fp32_ieee", "fp32_tf32", "bf16", "fp16"],
)
