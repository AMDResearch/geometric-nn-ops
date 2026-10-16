# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark the isolated triangle_multiplicative_update op.

``--backends`` names an implementation to time. ``reference`` is the baseline
the others are reported against; ``public`` measures the op a caller gets,
whichever implementation dispatch picks for it. The Triton path is
forward-only, so its backward column is empty.

Each backend is checked against the reference in fp64 before being timed, to
the bounds the tests use. Full correctness lives in ``tests/test_trimul.py``.

Usage:
    python benchmarks/bench_trimul_micro.py \
        --sizes 256 384 512 768 --dtype bf16 --out results/trimul_micro.csv
"""

import argparse
from pathlib import Path

import torch
from _bench_common import (
    DTYPES,
    add_timing_arguments,
    describe_case,
    finish,
    fmt,
    git_commit,
    pct,
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

from testutil import grad_targets
from testutil.trimul import (
    assert_forward_agrees_with_the_oracle,
    make_pair_input,
    make_projection_biases,
    make_weights,
)

CASE_KEYS = ("size", "direction", "backend")

MEASURES = ["fwd_train", "fwd_infer", "bwd"]


def bench_case(backend, n, channels, direction, dtype, warmup, rep, measures, biases):
    impl = TRIMUL_IMPLEMENTATIONS[backend]
    weights = make_weights(
        channels, channels, "cuda", dtype, seed=3, requires_grad=True
    )
    if biases:
        weights |= make_projection_biases(
            channels, channels, "cuda", dtype, seed=5, requires_grad=True
        )
    x, mask = make_pair_input(1, n, channels, "cuda", dtype, seed=4, requires_grad=True)

    assert_forward_agrees_with_the_oracle(
        x, mask, weights, direction, impl=impl, label=f"{backend}/{direction}"
    )

    def timed(fn):
        return do_bench(fn, warmup=warmup, rep=rep, return_mode="median")

    def fwd_train():
        return impl(x, direction, mask, **weights)

    def fwd_infer():
        with torch.inference_mode():
            return impl(x, direction, mask, **weights)

    row = {
        "fwd_train_ms": timed(fwd_train) if "fwd_train" in measures else None,
        "fwd_infer_ms": timed(fwd_infer) if "fwd_infer" in measures else None,
        "bwd_ms": None,
    }

    if "bwd" in measures:
        out = fwd_train()
        # A forward-only implementation returns a tensor with no graph behind
        # it, so there is no backward to time. Absent, not failed.
        if out.requires_grad:
            grad_out = torch.randn_like(out)
            targets = [x, *grad_targets(weights).values()]
            row["bwd_ms"] = timed(
                lambda: torch.autograd.grad(
                    [out], targets, grad_outputs=[grad_out], retain_graph=True
                )
            )
    return row


def report(rows, backends):
    header = (
        f"{'size':>5} {'dir':<9} {'backend':<10} "
        f"{'ftrain_ms':>10} {'finfer_ms':>10} {'bwd_ms':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["error"]:
            print(f"{describe_case(row, CASE_KEYS)}  FAILED {row['error']}")
            continue
        print(
            f"{row['size']:>5} {row['direction']:<9} {row['backend']:<10} "
            f"{fmt(row['fwd_train_ms'])} {fmt(row['fwd_infer_ms'])} "
            f"{fmt(row['bwd_ms'])}"
        )

    if BASELINE not in backends or len(backends) == 1:
        return
    print(f"\n# Time as % of {BASELINE} (lower is faster)")
    comparison = (
        f"{'size':>5} {'dir':<9} {'backend':<10} "
        f"{'ftrain_%':>10} {'finfer_%':>10} {'bwd_%':>10}"
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
            print(
                f"{n:>5} {direction:<9} {backend:<10} "
                f"{pct(row['fwd_train_ms'], base['fwd_train_ms'])} "
                f"{pct(row['fwd_infer_ms'], base['fwd_infer_ms'])} "
                f"{pct(row['bwd_ms'], base['bwd_ms'])}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shape_arguments(parser, sizes=[256, 384, 512, 768])
    add_backend_argument(parser)
    add_timing_arguments(parser)
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
        measure=",".join(m for m in MEASURES if m in measures),
    )

    rows = []
    for n in args.sizes:
        for direction in args.directions:
            for backend in args.backends:
                row = {
                    "size": n,
                    "direction": direction,
                    "backend": backend,
                    "channels": args.channels,
                    "biases": args.biases,
                    "dtype": args.dtype,
                    "device": torch.cuda.get_device_name(0),
                    "commit": git_commit(),
                    "fwd_train_ms": None,
                    "fwd_infer_ms": None,
                    "bwd_ms": None,
                    "error": None,
                }
                try:
                    row |= bench_case(
                        backend,
                        n,
                        args.channels,
                        direction,
                        dtype,
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
