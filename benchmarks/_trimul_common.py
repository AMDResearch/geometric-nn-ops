# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the trimul benchmark scripts."""

import argparse

from _bench_common import UniqueValues, positive_int

from geometric_nn_ops_torch.triangle import triangle_multiplicative_update
from geometric_nn_ops_torch.triangle._kernels.trimul import trimul_triton
from geometric_nn_ops_torch.triangle._reference.trimul import trimul_torch

# `public` measures the op a caller gets, whichever implementation dispatch
# picks for it.
TRIMUL_IMPLEMENTATIONS = {
    "reference": trimul_torch,
    "triton": trimul_triton,
    "public": triangle_multiplicative_update,
}

DEFAULT_BACKENDS = ("reference", "triton")

DEFAULT_BASELINES = ("reference",)

# Implementations without a backward, so there are no training steps to time.
FORWARD_ONLY = {"triton"}


def add_shape_arguments(parser, *, sizes, directions=("outgoing", "incoming")):
    """Add the arguments for the pair shape."""
    parser.add_argument(
        "--sizes",
        type=positive_int,
        nargs="+",
        default=list(sizes),
        action=UniqueValues,
        help="pair dim N",
    )
    parser.add_argument(
        "--channels",
        type=positive_int,
        default=128,
        help="channel dim (c_z == c_hidden)",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=list(directions),
        choices=list(directions),
        action=UniqueValues,
    )
    parser.add_argument(
        "--biases",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="give the projections biases. Off by default because the callers "
        "we serve run without them, but the Triton path compiles a separate "
        "kernel for each case, so one says nothing about the other",
    )
    return parser
