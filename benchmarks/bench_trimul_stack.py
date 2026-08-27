# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark triangle_multiplicative_update as a layer and as a trunk.

The op wrapped in a module that owns its parameters, timed for an inference
step and a training step, with peak memory for each. This is where holding
activations costs something, which the isolated op benchmark cannot see, and
``--depth`` compounds it the way a pairformer trunk does. ``--directions
alternate`` runs the outgoing and incoming updates in turn, as the trunk does.

``--backends`` names an implementation, as in ``bench_trimul_micro.py``. The
Triton path is forward-only, so its training columns are empty.

Usage:
    python benchmarks/bench_trimul_stack.py \
        --depth 48 --directions alternate --sizes 384
"""

import argparse
from pathlib import Path

import torch
from _bench_common import (
    DTYPES,
    add_timing_arguments,
    check_clean_start,
    describe_case,
    finish,
    fmt,
    git_commit,
    pct,
    peak_memory_gib,
    print_provenance,
    release_blas_workspaces,
    require_gpu,
)
from _trimul_common import (
    BASELINE,
    TRIMUL_IMPLEMENTATIONS,
    add_backend_argument,
    add_shape_arguments,
)
from triton.testing import do_bench

from testutil import seeded_randn
from testutil.trimul import TrimulStack, make_pair_input

CASE_KEYS = ("depth", "size", "direction", "backend")

MEASURES = ["infer_step", "train_step"]


def bench_case(
    backend, n, channels, direction, dtype, depth, warmup, rep, measures, biases
):
    model = TrimulStack(
        TRIMUL_IMPLEMENTATIONS[backend],
        depth,
        channels,
        channels,
        direction,
        dtype,
        biases=biases,
    )
    x, mask = make_pair_input(1, n, channels, "cuda", dtype, seed=4, requires_grad=True)
    grad_out = seeded_randn(*x.shape, device="cuda", dtype=dtype, seed=41)

    def infer_step():
        with torch.inference_mode():
            return model(x, mask)

    def train_step():
        model.zero_grad(set_to_none=True)
        x.grad = None
        model(x, mask).backward(grad_out)

    if not torch.isfinite(infer_step()).all():
        raise RuntimeError("non-finite output")

    # A layer's own output, not the stack's: the residual add re-attaches a
    # graph around a detached layer, so the stack stays differentiable through
    # the skip connection while the op contributes no gradient at all. Timing
    # that would report the residual's backward as the op's. A forward-only
    # implementation has no training step.
    differentiable = model.layers[0](x, mask).requires_grad

    row = {}
    for name, step in (("infer_step", infer_step), ("train_step", train_step)):
        wanted = name in measures and (differentiable or name == "infer_step")
        if not wanted:
            row[f"{name}_ms"] = None
            row[f"{name}_alloc_gib"] = row[f"{name}_reserved_gib"] = None
            continue
        row[f"{name}_ms"] = do_bench(step, warmup=warmup, rep=rep, return_mode="median")
        allocated, reserved = peak_memory_gib(step)
        row[f"{name}_alloc_gib"] = allocated
        row[f"{name}_reserved_gib"] = reserved
    return row


def report(rows, backends):
    header = (
        f"{'size':>5} {'dir':<9} {'backend':<10} "
        f"{'infer_ms':>10} {'inf_alloc':>10} {'inf_resv':>10} "
        f"{'train_ms':>10} {'trn_alloc':>10} {'trn_resv':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["error"]:
            print(f"{describe_case(row, CASE_KEYS)}  FAILED {row['error']}")
            continue
        print(
            f"{row['size']:>5} {row['direction']:<9} {row['backend']:<10} "
            f"{fmt(row['infer_step_ms'])} {fmt(row['infer_step_alloc_gib'])} "
            f"{fmt(row['infer_step_reserved_gib'])} "
            f"{fmt(row['train_step_ms'])} {fmt(row['train_step_alloc_gib'])} "
            f"{fmt(row['train_step_reserved_gib'])}"
        )

    if BASELINE not in backends or len(backends) == 1:
        return
    print(f"\n# As % of {BASELINE} (lower is better, for both time and memory)")
    comparison = (
        f"{'size':>5} {'dir':<9} {'backend':<10} "
        f"{'infer_%':>10} {'inf_resv_%':>12} {'train_%':>10} {'trn_resv_%':>12}"
    )
    print(comparison)
    print("-" * len(comparison))
    by_case = {}
    for row in rows:
        by_case.setdefault((row["size"], row["direction"]), {})[row["backend"]] = row
    for (n, direction), per_backend in sorted(by_case.items()):
        base = per_backend.get(BASELINE)
        if base is None:
            continue
        for backend, row in per_backend.items():
            if backend == BASELINE:
                continue
            # Reserved is what the comparison uses: it is the pool the device
            # actually had to give up for each implementation.
            infer_memory = pct(
                row["infer_step_reserved_gib"], base["infer_step_reserved_gib"], 12
            )
            train_memory = pct(
                row["train_step_reserved_gib"], base["train_step_reserved_gib"], 12
            )
            print(
                f"{n:>5} {direction:<9} {backend:<10} "
                f"{pct(row['infer_step_ms'], base['infer_step_ms'])} {infer_memory} "
                f"{pct(row['train_step_ms'], base['train_step_ms'])} {train_memory}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shape_arguments(
        parser,
        sizes=[256, 512, 768],
        # `alternate` is how a trunk actually runs, and only means anything
        # once there is more than one layer to alternate between.
        directions=["outgoing", "incoming", "alternate"],
    )
    add_backend_argument(parser)
    add_timing_arguments(parser)
    parser.add_argument(
        "--depth", type=int, default=1, help="1 is a single layer, >1 a residual stack"
    )
    parser.add_argument(
        "--measure", nargs="+", default=list(MEASURES), choices=MEASURES
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    require_gpu()
    dtype = DTYPES[args.dtype]
    measures = set(args.measure)
    print_provenance(
        args.dtype,
        backends=",".join(args.backends),
        channels=args.channels,
        biases=args.biases,
        depth=args.depth,
        measure=",".join(m for m in MEASURES if m in measures),
    )

    rows = []
    for n in args.sizes:
        for direction in args.directions:
            for backend in args.backends:
                row = {
                    "depth": args.depth,
                    "size": n,
                    "direction": direction,
                    "backend": backend,
                    "channels": args.channels,
                    "biases": args.biases,
                    "dtype": args.dtype,
                    "device": torch.cuda.get_device_name(0),
                    "commit": git_commit(),
                    "infer_step_ms": None,
                    "infer_step_alloc_gib": None,
                    "infer_step_reserved_gib": None,
                    "train_step_ms": None,
                    "train_step_alloc_gib": None,
                    "train_step_reserved_gib": None,
                    "error": None,
                }
                try:
                    check_clean_start(describe_case(row, CASE_KEYS))
                    row |= bench_case(
                        backend,
                        n,
                        args.channels,
                        direction,
                        dtype,
                        args.depth,
                        args.warmup,
                        args.rep,
                        measures,
                        args.biases,
                    )
                except Exception as error:  # noqa: BLE001
                    row["error"] = f"{type(error).__name__}: {error}"
                rows.append(row)
                release_blas_workspaces()

    report(rows, args.backends)
    finish(rows, args.out, case_keys=CASE_KEYS)


if __name__ == "__main__":
    main()
