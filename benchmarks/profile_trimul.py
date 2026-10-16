# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Rank the kernels inside triangle_multiplicative_update by device time.

``--mode fwdbwd`` needs a differentiable implementation, so it does not apply
to the forward-only Triton path.

Usage:
    python benchmarks/profile_trimul.py --sizes 768 --rows 15
"""

import argparse

import torch
from _bench_common import (
    DTYPES,
    print_provenance,
    release_blas_workspaces,
    require_gpu,
)
from _trimul_common import (
    TRIMUL_IMPLEMENTATIONS,
    add_backend_argument,
    add_shape_arguments,
)
from torch.profiler import ProfilerActivity, profile

from testutil import grad_targets
from testutil.trimul import (
    make_pair_input,
    make_weights,
)


def make_step(backend, n, channels, direction, dtype, mode):
    """A callable running the op alone.

    Inputs are built here rather than inside it, so that constructing and
    randomizing them does not land in the trace and outrank the kernels the
    profile is meant to show.
    """
    impl = TRIMUL_IMPLEMENTATIONS[backend]
    train = mode == "fwdbwd"
    weights = make_weights(
        channels, channels, "cuda", dtype, seed=3, requires_grad=train
    )
    x, mask = make_pair_input(
        1, n, channels, "cuda", dtype, seed=4, requires_grad=train
    )

    if not train:

        def forward():
            with torch.inference_mode():
                impl(x, direction, mask, **weights)

        return forward

    probe = impl(x, direction, mask, **weights)
    if not probe.requires_grad:
        raise RuntimeError(f"backend '{backend}' has no backward to profile")
    grad_out = torch.randn_like(probe)
    targets = [x, *grad_targets(weights).values()]

    def forward_backward():
        out = impl(x, direction, mask, **weights)
        torch.autograd.grad(out, targets, grad_outputs=[grad_out])

    return forward_backward


def profile_case(backend, n, channels, direction, dtype, mode, warmup, active, rows):
    step = make_step(backend, n, channels, direction, dtype, mode)
    for _ in range(warmup):  # compile and autotune outside the trace
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(active):
            step()
        torch.cuda.synchronize()

    print(
        f"\n===== backend={backend} N={n} dir={direction} dtype={dtype} "
        f"mode={mode} (active={active}) ====="
    )
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shape_arguments(parser, sizes=[768], directions=["outgoing"])
    add_backend_argument(parser, default=["triton"])
    parser.add_argument("--mode", nargs="+", default=["fwd"], choices=["fwd", "fwdbwd"])
    parser.add_argument(
        "--warmup", type=int, default=10, help="untraced iterations before the trace"
    )
    parser.add_argument("--active", type=int, default=10, help="profiled iterations")
    parser.add_argument("--rows", type=int, default=15, help="top-N ops to print")
    args = parser.parse_args()

    require_gpu()
    dtype = DTYPES[args.dtype]
    print_provenance(args.dtype, channels=args.channels)

    for n in args.sizes:
        for direction in args.directions:
            for mode in args.mode:
                for backend in args.backends:
                    profile_case(
                        backend,
                        n,
                        args.channels,
                        direction,
                        dtype,
                        mode,
                        max(1, args.warmup),
                        max(1, args.active),
                        args.rows,
                    )
                    release_blas_workspaces()


if __name__ == "__main__":
    main()
