# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark triangle_multiplicative_update as a layer and as a trunk.

The op wrapped in a module that owns its parameters, timed for an inference
step and a training step, with peak memory for each. This is where holding
activations costs something, which the isolated op benchmark cannot see, and
``--depth`` compounds it the way a pairformer trunk does. ``--directions
alternate`` runs the outgoing and incoming updates in turn, as the trunk does.

``--backends`` names an implementation, as in ``bench_trimul_micro.py``. The
Triton path is forward-only, so its training columns are empty. Before a stack is
timed, its first layer is checked against an fp64 copy of itself, comparing the
output and, for training, the gradients, and the whole stack's output is checked
to be finite. See ``run_benchmark`` for how steps are timed and their memory
measured.

Usage:
    python benchmarks/bench_trimul_stack.py \
        --depth 48 --directions alternate --sizes 384
"""

import argparse
import functools
import itertools

import torch
from _bench_common import (
    DTYPES,
    BenchCase,
    Column,
    add_benchmark_arguments,
    assert_finite,
    positive_int,
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

from testutil import seeded_randn
from testutil.trimul import (
    TrimulStack,
    assert_stack_agrees_with_oracle,
    make_pair_input,
)

CASE = (
    Column("depth", "depth", 5),
    Column("size", "size", 5),
    Column("direction", "dir", 9, align="<"),
    Column("backend", "backend", 10, align="<"),
)

MEASURES = ["infer_step", "train_step"]


class Case(BenchCase):
    """The pair rep and its mask for one size and direction."""

    def __init__(self, args, key, measures, *, oracle):
        n, self.direction = key
        self.args = args
        self.dtype = DTYPES[args.dtype]
        self.inference = "infer_step" in measures
        self.training = "train_step" in measures
        self.x, self.mask = make_pair_input(
            1, n, args.channels, "cuda", self.dtype, seed=4, requires_grad=self.training
        )
        self.grad_out = None
        if self.training:
            self.grad_out = seeded_randn(
                *self.x.shape, device="cuda", dtype=self.dtype, seed=41
            )

    def stack(self, backend, depth=None):
        # Fixed seeds, so the stacks of all backends have the same weights, and
        # the first layer is the same at any depth.
        return TrimulStack(
            TRIMUL_IMPLEMENTATIONS[backend],
            depth or self.args.depth,
            self.args.channels,
            self.args.channels,
            self.direction,
            self.dtype,
            biases=self.args.biases,
        )

    def steps(self, backend, measures) -> dict:
        model = self.stack(backend)

        def infer_step():
            with torch.inference_mode():
                return model(self.x, self.mask)

        def train_step():
            model.zero_grad(set_to_none=True)
            self.x.grad = None
            model(self.x, self.mask).backward(self.grad_out)

        steps = {"infer_step": infer_step}
        if backend not in FORWARD_ONLY:
            steps["train_step"] = train_step
        return {m: steps[m] for m in measures if m in steps}

    def check(self, backend, measures, label):
        """Check the first layer against the fp64 oracle, and that the whole
        stack's output is finite. Error grows with depth, so a deep stack
        would exceed tolerances set for a single op."""
        self.x.grad = None
        inference, training = "infer_step" in measures, "train_step" in measures
        assert_stack_agrees_with_oracle(
            self.stack(backend, depth=1),
            self.x,
            self.mask,
            training=training,
            inference=inference,
            grad_out=self.grad_out,
            precision=tolerance_precision(self.dtype),
            label=label,
        )
        model = self.stack(backend)
        if inference:
            with torch.inference_mode():
                assert_finite(model(self.x, self.mask), f"{label} stack inference")
        if training:
            assert_finite(model(self.x, self.mask), f"{label} stack training")


def make_row(args, key) -> dict:
    """The case's own result row fields."""
    n, direction = key
    return {
        "depth": args.depth,
        "size": n,
        "direction": direction,
        "backend": None,
        "channels": args.channels,
        "biases": args.biases,
        **run_fields(args),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shape_arguments(
        parser,
        sizes=[256, 512, 768],
        # `alternate` is how a trunk actually runs, and only means anything
        # once there is more than one layer to alternate between.
        directions=["outgoing", "incoming", "alternate"],
    )
    add_benchmark_arguments(
        parser,
        measures=MEASURES,
        backends=TRIMUL_IMPLEMENTATIONS,
        default_backends=DEFAULT_BACKENDS,
        baselines=DEFAULT_BASELINES,
    )
    parser.add_argument(
        "--depth",
        type=positive_int,
        default=1,
        help="1 is a single layer, >1 a residual stack",
    )
    args = parser.parse_args()
    run_benchmark(
        args,
        measures=MEASURES,
        cases=itertools.product(args.sizes, args.directions),
        make_case=functools.partial(Case, args),
        make_row=functools.partial(make_row, args),
        case_columns=CASE,
        provenance={
            "channels": args.channels,
            "biases": args.biases,
            "depth": args.depth,
        },
    )


if __name__ == "__main__":
    main()
