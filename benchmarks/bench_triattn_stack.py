# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark triangle_attention inside a layer, and as a trunk.

The layer is ``testutil``'s ``TriangleAttentionLayer``, modeled after
openfold-3's. Each backend is timed for an inference step and a training step,
with peak memory for each. This includes costs the isolated op benchmark doesn't
see: the projections, the layout copies between them and the attention, and the
activations a training step holds.

``--depth`` stacks residual layers as in an actual model trunk. ``--node`` picks
the starting or ending node, or alternates them as in a pairformer.
``--chunk-size`` runs inference in chunks of N_seq, mimicking openfold-3's
chunking.

``--backends`` names the attention implementation, as in
``bench_triattn_micro.py``. Before a stack is timed, its first layer is checked
against an fp64 copy of itself, comparing the output and the gradients, and the
whole stack's output is checked to be finite. With ``--oracle-rows``, the first
layer is checked on some rows instead (see
``assert_layer_rows_agree_with_oracle``). See ``run_benchmark`` for how
steps are timed and their memory measured.

Usage:
    python benchmarks/bench_triattn_stack.py --sizes 384 \
        --head-dims 32 --node alternate --depth 4
    python benchmarks/bench_triattn_stack.py --sizes 2048 --chunk-size 1024 \
        --oracle-rows 4 --measure infer_step
"""

import argparse
import functools
import itertools

import torch
from _bench_common import (
    DTYPES,
    BenchCase,
    Column,
    assert_finite,
    positive_int,
    run_benchmark,
    run_fields,
    tolerance_precision,
)
from _triattn_common import (
    add_benchmark_arguments,
    add_oracle_argument,
    add_shape_arguments,
    checked_row_count,
    make_implementations,
    sdpa_backend,
    sdpa_label,
    sdpa_triangle_attention,
)

from testutil import seeded_randn
from testutil.triangle_attention import (
    TriangleAttentionStack,
    assert_layer_rows_agree_with_oracle,
    assert_stack_agrees_with_oracle,
    checked_rows,
)
from testutil.trimul import make_pair_input

CASE = (
    Column("depth", "depth", 5),
    Column("size", "size", 5),
    Column("head_dim", "D", 4),
    Column("backend", "backend", 10, align="<"),
)

MEASURES = ["infer_step", "train_step"]


class Case(BenchCase):
    """The pair representation for one shape."""

    def __init__(self, args, implementations, key, measures, *, oracle):
        self.head_dim, n = key
        self.args = args
        self.implementations = implementations
        self.dtype = DTYPES[args.dtype]
        self.inference = "infer_step" in measures
        self.training = "train_step" in measures
        self.z, mask = make_pair_input(
            args.batch,
            n,
            args.c_z,
            "cuda",
            self.dtype,
            seed=4,
            requires_grad=self.training,
        )
        self.mask = mask.bool() if args.mask else None
        self.grad_out = None
        if self.training:
            self.grad_out = seeded_randn(
                *self.z.shape, device="cuda", dtype=self.dtype, seed=41
            )
        self.rows = None
        if oracle and args.oracle_rows:
            self.rows = checked_rows(n, args.oracle_rows)

    def stack(self, attention, depth=None):
        # Fixed seeds, so the stacks of all backends have the same weights, and
        # the first layer is the same at any depth.
        return TriangleAttentionStack(
            attention,
            depth or self.args.depth,
            self.args.c_z,
            self.args.heads,
            self.head_dim,
            self.dtype,
            node=self.args.node,
        )

    def model_steps(self, model, measures) -> dict:
        """A callable per step of ``model``. Inference runs in
        ``--chunk-size`` chunks of N_seq, and training is unchunked."""

        def infer_step():
            with torch.inference_mode():
                return model(self.z, self.mask, self.args.chunk_size)

        def train_step():
            model.zero_grad(set_to_none=True)
            self.z.grad = None
            model(self.z, self.mask).backward(self.grad_out)

        steps = {"infer_step": infer_step, "train_step": train_step}
        return {m: steps[m] for m in measures}

    def steps(self, backend, measures) -> dict:
        model = self.stack(self.implementations[backend])
        return self.model_steps(model, measures)

    def check(self, backend, measures, label):
        """Check the first layer against the fp64 oracle, and that the whole
        stack's output is finite. Error grows with depth, so a deep stack
        would exceed tolerances set for a single op. With ``--oracle-rows``,
        only some rows of the first layer are checked."""
        impl = self.implementations[backend]
        inference, training = "infer_step" in measures, "train_step" in measures
        first = self.stack(impl, depth=1)
        self.z.grad = None
        precision = tolerance_precision(self.dtype)
        if self.rows is None:
            assert_stack_agrees_with_oracle(
                first,
                self.z,
                self.mask,
                training=training,
                inference=inference,
                chunk_size=self.args.chunk_size,
                grad_out=self.grad_out,
                precision=precision,
                label=label,
            )
        else:
            for mode_training in [False] * inference + [True] * training:
                assert_layer_rows_agree_with_oracle(
                    first,
                    self.z,
                    self.mask,
                    self.rows,
                    training=mode_training,
                    chunk_size=self.args.chunk_size,
                    precision=precision,
                    label=label,
                )
        del first
        model = self.stack(impl)
        if inference:
            with torch.inference_mode():
                out = model(self.z, self.mask, self.args.chunk_size)
            assert_finite(out, f"{label} stack inference")
            del out
        if training:
            assert_finite(model(self.z, self.mask), f"{label} stack training")

    def fields(self, backend) -> dict:
        """The SDPA backends torch would pick for the stack's attention calls in
        each measured step, if they weren't pinned."""
        if backend != "sdpa":
            return {}
        seen = {}

        def recording(q, k, v, bias=None, mask=None, scale=None):
            mode = "infer" if torch.is_inference_mode_enabled() else "train"
            seen.setdefault(mode, set()).add(sdpa_backend(q, k, v, bias, mask, scale))
            return sdpa_triangle_attention(q, k, v, bias, mask, scale)

        model = self.stack(recording)
        if self.inference:
            with torch.inference_mode():
                model(self.z, self.mask, self.args.chunk_size)
        if self.training:
            model(self.z, self.mask)
        found = {mode: ",".join(sorted(s)) for mode, s in seen.items()}
        return {"sdpa_backend": sdpa_label(found)}


def make_row(args, key) -> dict:
    """The case's own result row fields."""
    head_dim, n = key
    row = {
        "depth": args.depth,
        "size": n,
        "head_dim": head_dim,
        "backend": None,
        "node": args.node,
        "batch": args.batch,
        "heads": args.heads,
        "c_z": args.c_z,
        "mask": args.mask,
        "chunk_size": args.chunk_size,
        "oracle_rows": checked_row_count(args.oracle_rows, n),
        "dynamic": args.dynamic,
        "use_exp2": args.use_exp2,
        "flex_dynamic": args.flex_dynamic,
        **run_fields(args),
    }
    if "sdpa" in args.backends:
        row["sdpa_backend"] = None
    return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shape_arguments(parser, sizes=[256, 384, 512], head_dims=[32], n_seq=False)
    add_benchmark_arguments(parser, measures=MEASURES)
    parser.add_argument("--c-z", type=positive_int, default=128, help="pair channels")
    parser.add_argument(
        "--node", default="starting", choices=["starting", "ending", "alternate"]
    )
    parser.add_argument(
        "--depth",
        type=positive_int,
        default=1,
        help="1 is a single layer, >1 a residual stack",
    )
    parser.add_argument(
        "--chunk-size",
        type=positive_int,
        default=None,
        help="run inference in chunks of this many rows of N_seq, as "
        "openfold-3 does. Training is never chunked",
    )
    add_oracle_argument(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    implementations = make_implementations(args)
    extra_columns = []
    if "sdpa" in args.backends:
        extra_columns.append(Column("sdpa_backend", "sdpa", 0, align="<"))
    run_benchmark(
        args,
        measures=MEASURES,
        cases=itertools.product(args.head_dims, args.sizes),
        make_case=functools.partial(Case, args, implementations),
        make_row=functools.partial(make_row, args),
        case_columns=CASE,
        extra_columns=extra_columns,
        provenance={
            "node": args.node,
            "depth": args.depth,
            "batch": args.batch,
            "heads": args.heads,
            "c_z": args.c_z,
            "mask": args.mask,
            "chunk_size": args.chunk_size,
            "oracle_rows": args.oracle_rows or "all",
            "dynamic": args.dynamic,
            "use_exp2": args.use_exp2,
            "flex_dynamic": args.flex_dynamic,
        },
    )


if __name__ == "__main__":
    main()
