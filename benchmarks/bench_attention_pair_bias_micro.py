# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark the isolated attention_pair_bias op.

``--backends`` names an implementation to time. ``reference`` is the baseline
the others are reported against; ``public`` measures the op a caller gets,
whichever implementation dispatch picks for it.

Each backend is checked against the reference in fp64 before being timed, to
the bounds the tests use. Full correctness lives in
``tests/test_attention_pair_bias.py``.

Usage:
    python benchmarks/bench_attention_pair_bias_micro.py \
        --sizes 256 384 512 768 --dtype bf16 --out results/apb_micro.csv
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
from triton.testing import do_bench

from geometric_nn_ops_torch.triangle import attention_pair_bias
from geometric_nn_ops_torch.triangle._kernels.attention_pair_bias import (
    attention_pair_bias_triton,
)
from geometric_nn_ops_torch.triangle._reference.attention_pair_bias import (
    attention_pair_bias_torch,
)
from testutil import cast_tree, grad_targets
from testutil.attention_pair_bias import (
    assert_agrees_with_the_oracle,
    make_key_mask,
    make_op_tensors,
)

# `public` decides which implementation to delegate to, so choosing it measures
# whatever a real caller would get rather than a specific implementation.
IMPLEMENTATIONS = {
    "reference": attention_pair_bias_torch,
    "triton": attention_pair_bias_triton,
    "public": attention_pair_bias,
}

BASELINE = "reference"

CASE_KEYS = ("size", "multiplicity", "backend")

MEASURES = ["fwd_train", "fwd_infer", "bwd"]


def bench_case(backend, n, m, shape, dtype, warmup, rep, measures):
    impl = IMPLEMENTATIONS[backend]
    master = make_op_tensors(
        1, m, shape.heads, n, n, shape.head_dim, shape.zc, "cuda", torch.float32
    )
    mask = make_key_mask(1, n, "cuda", dtype, seed=5)

    # Smoke test of what we're about to time. Full testing left to the test suite.
    assert_agrees_with_the_oracle(
        master, dtype, impl=impl, mask=mask, label=f"{backend}/N{n}/M{m}"
    )

    tensors = cast_tree(master, dtype)
    del master

    def timed(fn):
        return do_bench(fn, warmup=warmup, rep=rep, return_mode="median")

    def fwd_train():
        out, _ = impl(**tensors, mask=mask)
        return out

    def fwd_infer():
        with torch.inference_mode():
            return impl(**tensors, mask=mask)

    row = {
        "fwd_train_ms": timed(fwd_train) if "fwd_train" in measures else None,
        "fwd_infer_ms": timed(fwd_infer) if "fwd_infer" in measures else None,
        "bwd_ms": None,
    }

    if "bwd" in measures:
        out = fwd_train()
        grad_out = torch.randn_like(out)
        targets = list(grad_targets(tensors).values())
        row["bwd_ms"] = timed(
            lambda: torch.autograd.grad(
                [out], targets, grad_outputs=[grad_out], retain_graph=True
            )
        )
    return row


def report(rows, backends):
    header = (
        f"{'size':>5} {'M':>3} {'backend':<10} "
        f"{'ftrain_ms':>10} {'finfer_ms':>10} {'bwd_ms':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["error"]:
            print(f"{describe_case(row, CASE_KEYS)}  FAILED {row['error']}")
            continue
        print(
            f"{row['size']:>5} {row['multiplicity']:>3} {row['backend']:<10} "
            f"{fmt(row['fwd_train_ms'])} {fmt(row['fwd_infer_ms'])} "
            f"{fmt(row['bwd_ms'])}"
        )

    if BASELINE not in backends or len(backends) == 1:
        return
    print(f"\n# Time as % of {BASELINE} (lower is faster)")
    comparison = (
        f"{'size':>5} {'M':>3} {'backend':<10} "
        f"{'ftrain_%':>10} {'finfer_%':>10} {'bwd_%':>10}"
    )
    print(comparison)
    print("-" * len(comparison))
    by_case = {}
    for row in rows:
        by_case.setdefault((row["size"], row["multiplicity"]), {})[row["backend"]] = row
    for (n, m), per_backend in sorted(by_case.items()):
        base = per_backend.get(BASELINE)
        if base is None:
            continue
        for backend, row in per_backend.items():
            if backend == BASELINE:
                continue
            print(
                f"{n:>5} {m:>3} {backend:<10} "
                f"{pct(row['fwd_train_ms'], base['fwd_train_ms'])} "
                f"{pct(row['fwd_infer_ms'], base['fwd_infer_ms'])} "
                f"{pct(row['bwd_ms'], base['bwd_ms'])}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[256, 384, 512, 768], help="N"
    )
    parser.add_argument(
        "--multiplicities", type=int, nargs="+", default=[1], help="diffusion M"
    )
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=48)
    parser.add_argument("--zc", type=int, default=128, help="pair channel dim")
    parser.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["reference", "triton"],
        choices=list(IMPLEMENTATIONS),
    )
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
        heads=args.heads,
        head_dim=args.head_dim,
        zc=args.zc,
        measure=",".join(m for m in MEASURES if m in measures),
    )

    rows = []
    for n in args.sizes:
        for m in args.multiplicities:
            for backend in args.backends:
                row = {
                    "size": n,
                    "multiplicity": m,
                    "backend": backend,
                    "heads": args.heads,
                    "head_dim": args.head_dim,
                    "zc": args.zc,
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
                        backend, n, m, args, dtype, args.warmup, args.rep, measures
                    )
                except Exception as error:  # noqa: BLE001
                    row["error"] = f"{type(error).__name__}: {error}"
                rows.append(row)
                release_blas_workspaces()

    report(rows, args.backends)
    finish(rows, args.out, case_keys=CASE_KEYS)


if __name__ == "__main__":
    main()
