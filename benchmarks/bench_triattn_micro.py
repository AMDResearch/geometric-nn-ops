# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark the isolated triangle_attention op against the alternatives.

``--backends`` names the implementations to time:

- ``triton``: our Triton kernel's entry point
- ``public``: the public API, which could dispatch to triton, native torch, or
  something else. Includes some input transformation.
- ``reference``: the pure-torch reference.
- ``sdpa``: ``F.scaled_dot_product_attention`` with the bias as ``attn_mask``,
  pinned to the "memory-efficient" SDPA backend because that's the only one that
  supports an additive mask. The table shows which SDPA backend Torch would have
  picked if it weren't pinned.
- ``flex``: ``flex_attention`` under ``torch.compile``.

Every backend is reported as a percentage of each of ``--baselines``.

Inference runs on inputs with requires_grad==False: the triton op changes
behavior based on this because inside an autograd function it can't tell we're
in ``torch.inference_mode``. ``bwd`` is a backward over a retained graph, which
compiled flex cannot run, so use ``fwd_bwd`` to measure flex's backward. The
retained graph of
every backend is held while all of them are timed; leave ``bwd`` out at the
longest sequences to avoid OOMs.

Each backend is checked against the fp64 reference before being timed, inference
forward, training forward, and the gradients. See ``run_benchmark`` for how
steps are timed and their memory measured.

Usage:
    python benchmarks/bench_triattn_micro.py \
        --sizes 256 384 512 768 --head-dims 32 --dtype bf16 \
        --out results/triattn_micro.csv
    python benchmarks/bench_triattn_micro.py --sizes 2048 4096 \
        --n-seq-max 1024 --oracle-rows 4 --measure fwd_infer
"""

import argparse
import functools
import itertools

import torch
from _bench_common import (
    DTYPES,
    OP_BACKWARD_MEASURES,
    OP_MEASURES,
    OP_TRAINING_MEASURES,
    BenchCase,
    Column,
    op_steps,
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
)

from testutil import seeded_randn
from testutil.metrics import TRIANGLE_ATTENTION_TOL
from testutil.triangle_attention import (
    Oracle,
    checked_rows,
    chunked_reference,
    make_inputs,
    make_key_mask,
)

CASE = (
    Column("size", "size", 5),
    Column("head_dim", "D", 4),
    Column("backend", "backend", 10, align="<"),
)

# Compiled flex frees the buffers its backward was donated, so it cannot run
# a backward on a retained graph.
NO_RETAINED_BACKWARD = {"flex"}

# Seed of the output gradient, the oracle's (see testutil's Oracle).
GRAD_OUT_SEED = 31


def n_seq_for(args, n) -> int:
    """N_seq for N_res = ``n``: ``--n-seq`` or ``n``, capped at ``--n-seq-max``."""
    n_seq = args.n_seq or n
    return min(n_seq, args.n_seq_max) if args.n_seq_max else n_seq


class Case(BenchCase):
    """q, k, v, the bias and the key mask for one shape, and with ``oracle``
    their fp64 reference results."""

    def __init__(self, args, implementations, key, measures, *, oracle):
        head_dim, n = key
        n_seq = n_seq_for(args, n)
        self.implementations = implementations
        self.training = bool(set(measures) & OP_TRAINING_MEASURES)
        self.inference = "fwd_infer" in measures
        self.backward = bool(set(measures) & OP_BACKWARD_MEASURES)
        self.tensors = make_inputs(
            args.batch,
            n_seq,
            n,
            args.heads,
            head_dim,
            "cuda",
            DTYPES[args.dtype],
            seed=3,
            requires_grad=self.training,
        )
        self.infer_tensors = {k: t.detach() for k, t in self.tensors.items()}
        self.mask = None
        if args.mask:
            self.mask = make_key_mask(args.batch, n_seq, n, "cuda", seed=1)
        self.scale = head_dim**-0.5
        self.targets = list(self.tensors.values())
        self.sampled = False
        self.grad_out = None
        if oracle:
            # The backends use the oracle's output gradient until it is
            # dropped, so that only one tensor that size is held.
            rows = checked_rows(n_seq, args.oracle_rows) if args.oracle_rows else None
            self.sampled = rows is not None
            self.oracle = Oracle(
                self.tensors,
                self.mask,
                self.scale,
                rows=rows,
                backward=self.training,
                oracle=chunked_reference,
            )
            self.grad_out = self.oracle.grad_out
        elif self.backward:
            self.grad_out = self.dense_grad_out()

    def dense_grad_out(self):
        q = self.tensors["q"]
        return seeded_randn(
            *q.shape, device=q.device, dtype=q.dtype, seed=GRAD_OUT_SEED
        )

    def drop_oracle(self):
        """Free the fp64 reference results, and the output gradient unless a
        backward step is timed. With ``--oracle-rows`` the oracle's output
        gradient is zero outside the checked rows, so it is replaced with a
        dense one for timing."""
        del self.oracle
        if not self.backward:
            self.grad_out = None
        elif self.sampled:
            self.grad_out = None
            self.grad_out = self.dense_grad_out()

    def steps(self, backend, measures) -> dict:
        impl = self.implementations[backend]

        def forward(tensors):
            return lambda: impl(**tensors, mask=self.mask, scale=self.scale)

        return op_steps(
            measures,
            infer=forward(self.infer_tensors),
            train=forward(self.tensors),
            targets=self.targets,
            grad_out=self.grad_out,
            retained_backward=backend not in NO_RETAINED_BACKWARD,
        )

    def check(self, backend, measures, label):
        impl = self.implementations[backend]
        dtype = self.tensors["q"].dtype
        tol = TRIANGLE_ATTENTION_TOL[dtype, tolerance_precision(dtype)]
        if set(measures) & OP_TRAINING_MEASURES:
            self.oracle.check(impl, tol, label=label)
        if "fwd_infer" in measures:
            self.oracle.check_forward(impl, self.infer_tensors, tol, label=label)

    def fields(self, backend) -> dict:
        """The SDPA backend torch would pick for each mode run if it weren't
        pinned."""
        if backend != "sdpa":
            return {}
        found = {}
        if self.inference:
            with torch.inference_mode():
                found["infer"] = sdpa_backend(
                    **self.infer_tensors, mask=self.mask, scale=self.scale
                )
        if self.training:
            found["train"] = sdpa_backend(
                **self.tensors, mask=self.mask, scale=self.scale
            )
        return {"sdpa_backend": sdpa_label(found)}


def make_row(args, key) -> dict:
    """The case's own result row fields."""
    head_dim, n = key
    n_seq = n_seq_for(args, n)
    row = {
        "size": n,
        "n_seq": n_seq,
        "oracle_rows": checked_row_count(args.oracle_rows, n_seq),
        "batch": args.batch,
        "heads": args.heads,
        "head_dim": head_dim,
        "backend": None,
        "mask": args.mask,
        "dynamic": args.dynamic,
        "use_exp2": args.use_exp2,
        "flex_dynamic": args.flex_dynamic,
        **run_fields(args),
    }
    if "sdpa" in args.backends:
        row["sdpa_backend"] = None
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shape_arguments(parser, sizes=[256, 384, 512, 768], head_dims=[32])
    add_benchmark_arguments(parser, measures=OP_MEASURES)
    add_oracle_argument(parser)
    args = parser.parse_args()
    implementations = make_implementations(args)
    extra_columns = []
    if "sdpa" in args.backends:
        extra_columns.append(Column("sdpa_backend", "sdpa", 0, align="<"))
    run_benchmark(
        args,
        measures=OP_MEASURES,
        cases=itertools.product(args.head_dims, args.sizes),
        make_case=functools.partial(Case, args, implementations),
        make_row=functools.partial(make_row, args),
        case_columns=CASE,
        extra_columns=extra_columns,
        provenance={
            "batch": args.batch,
            "n_seq": args.n_seq or "N_res",
            "n_seq_max": args.n_seq_max,
            "oracle_rows": args.oracle_rows or "all",
            "heads": args.heads,
            "mask": args.mask,
            "dynamic": args.dynamic,
            "use_exp2": args.use_exp2,
            "flex_dynamic": args.flex_dynamic,
        },
    )


if __name__ == "__main__":
    main()
