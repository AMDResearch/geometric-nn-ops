# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Rank the kernels inside triangle_multiplicative_update by device time.

Profiles the steps ``bench_trimul_micro.py`` times. The training steps don't
apply to the forward-only Triton path.

Usage:
    python benchmarks/profile_trimul.py --sizes 768 --rows 15
"""

import argparse

import torch
from _bench_common import (
    OP_MEASURES,
    add_backend_arguments,
    add_dtype_arguments,
    print_provenance,
    require_gpu,
)
from _trimul_common import TRIMUL_IMPLEMENTATIONS, add_shape_arguments
from bench_trimul_micro import Case
from torch.profiler import ProfilerActivity, profile

from testutil.precision import torch_matmul_precision


def profile_case(args, backend, n, direction, measure):
    # The inputs are built before the trace, so that constructing and
    # randomizing them doesn't land in it and outrank the kernels.
    case = Case(args, (n, direction), [measure], oracle=False)
    step = case.steps(backend, [measure]).get(measure)
    if step is None:
        print(f"\n===== backend={backend}: {measure} doesn't apply, skipped =====")
        return
    for _ in range(args.warmup):  # compile and autotune outside the trace
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.active):
            step()
        torch.cuda.synchronize()

    print(
        f"\n===== backend={backend} N={n} dir={direction} dtype={args.dtype} "
        f"measure={measure} (active={args.active}) ====="
    )
    table = prof.key_averages().table(
        sort_by="self_device_time_total", row_limit=args.rows
    )
    print(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shape_arguments(parser, sizes=[768], directions=["outgoing"])
    add_dtype_arguments(parser)
    add_backend_arguments(parser, TRIMUL_IMPLEMENTATIONS, default=["triton"])
    parser.add_argument(
        "--measure", nargs="+", default=["fwd_infer"], choices=OP_MEASURES
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="untraced iterations before the trace"
    )
    parser.add_argument("--active", type=int, default=10, help="profiled iterations")
    parser.add_argument("--rows", type=int, default=15, help="top-N ops to print")
    args = parser.parse_args()
    args.warmup = max(1, args.warmup)
    args.active = max(1, args.active)

    require_gpu()
    print_provenance(
        args.dtype,
        channels=args.channels,
        biases=args.biases,
        fp32_precision=args.fp32_precision,
    )
    with torch_matmul_precision(args.fp32_precision):
        for n in args.sizes:
            for direction in args.directions:
                for measure in args.measure:
                    for backend in args.backends:
                        profile_case(args, backend, n, direction, measure)


if __name__ == "__main__":
    main()
