# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark the isolated triangle_multiplicative_update op.

``--backends`` names the implementations to time. ``reference`` is the
pure-torch reference, ``triton`` our Triton kernel, which has no backward, and
``public`` the public API, which could dispatch to either.

Each backend is checked against the fp64 reference before being timed: the
forward, and the gradients if training is measured. Full correctness lives in
``tests/test_trimul.py``. See ``run_benchmark`` for how steps are timed and
their memory measured.

Usage:
    python benchmarks/bench_trimul_micro.py \
        --sizes 256 384 512 768 --dtype bf16 --out results/trimul_micro.csv
"""

import argparse
import functools
import itertools

from _bench_common import (
    DTYPES,
    OP_BACKWARD_MEASURES,
    OP_MEASURES,
    OP_TRAINING_MEASURES,
    BenchCase,
    Column,
    add_benchmark_arguments,
    op_steps,
    run_benchmark,
    run_fields,
    tolerance_precision,
)
from _trimul_common import (
    DEFAULT_BACKENDS,
    DEFAULT_BASELINES,
    FORWARD_ONLY,
    TRIMUL_IMPLEMENTATIONS,
    add_shape_arguments,
)

from testutil import grad_targets, seeded_randn
from testutil.trimul import (
    assert_agrees_with_oracle,
    assert_forward_agrees_with_oracle,
    make_pair_input,
    make_projection_biases,
    make_weights,
)

CASE = (
    Column("size", "size", 5),
    Column("direction", "dir", 9, align="<"),
    Column("backend", "backend", 10, align="<"),
)


class Case(BenchCase):
    """The pair rep, its mask and the weights for one size and direction."""

    def __init__(self, args, key, measures, *, oracle):
        n, self.direction = key
        dtype = DTYPES[args.dtype]
        self.training = bool(set(measures) & OP_TRAINING_MEASURES)
        self.precision = tolerance_precision(dtype)
        self.weights = make_weights(
            args.channels,
            args.channels,
            "cuda",
            dtype,
            seed=3,
            requires_grad=self.training,
        )
        if args.biases:
            self.weights |= make_projection_biases(
                args.channels,
                args.channels,
                "cuda",
                dtype,
                seed=5,
                requires_grad=self.training,
            )
        self.x, self.mask = make_pair_input(
            1, n, args.channels, "cuda", dtype, seed=4, requires_grad=self.training
        )
        self.grad_out = None
        if set(measures) & OP_BACKWARD_MEASURES:
            self.grad_out = seeded_randn(
                *self.x.shape, device="cuda", dtype=dtype, seed=11
            )
        self.targets = [self.x, *grad_targets(self.weights).values()]

    def steps(self, backend, measures) -> dict:
        impl = TRIMUL_IMPLEMENTATIONS[backend]

        def forward():
            return impl(self.x, self.direction, self.mask, **self.weights)

        return op_steps(
            measures,
            infer=forward,
            train=None if backend in FORWARD_ONLY else forward,
            targets=self.targets,
            grad_out=self.grad_out,
        )

    def check(self, backend, measures, label):
        impl = TRIMUL_IMPLEMENTATIONS[backend]
        inputs = (self.x, self.mask, self.weights, self.direction)
        options = {"impl": impl, "precision": self.precision, "label": label}
        # Without gradients enabled, which may dispatch differently.
        if "fwd_infer" in measures:
            assert_forward_agrees_with_oracle(*inputs, **options)
        if set(measures) & OP_TRAINING_MEASURES:
            assert_agrees_with_oracle(*inputs, grad_out=self.grad_out, **options)


def make_row(args, key) -> dict:
    """The case's own result row fields."""
    n, direction = key
    return {
        "size": n,
        "direction": direction,
        "backend": None,
        "channels": args.channels,
        "biases": args.biases,
        **run_fields(args),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shape_arguments(parser, sizes=[256, 384, 512, 768])
    add_benchmark_arguments(
        parser,
        measures=OP_MEASURES,
        backends=TRIMUL_IMPLEMENTATIONS,
        default_backends=DEFAULT_BACKENDS,
        baselines=DEFAULT_BASELINES,
    )
    args = parser.parse_args()
    run_benchmark(
        args,
        measures=OP_MEASURES,
        cases=itertools.product(args.sizes, args.directions),
        make_case=functools.partial(Case, args),
        make_row=functools.partial(make_row, args),
        case_columns=CASE,
        provenance={"channels": args.channels, "biases": args.biases},
    )


if __name__ == "__main__":
    main()
