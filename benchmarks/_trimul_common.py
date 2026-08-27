# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the trimul benchmark scripts."""

from _bench_common import DTYPES

from geometric_nn_ops_torch.triangle import triangle_multiplicative_update
from geometric_nn_ops_torch.triangle._kernels.trimul import trimul_triton
from geometric_nn_ops_torch.triangle._reference.trimul import trimul_torch

# `public` decided which implementation to delegate to, so choosing it measures
# whatever a real caller would get rather than a specific implementation.
TRIMUL_IMPLEMENTATIONS = {
    "reference": trimul_torch,
    "triton": trimul_triton,
    "public": triangle_multiplicative_update,
}

BASELINE = "reference"


def add_shape_arguments(parser, *, sizes, directions=("outgoing", "incoming")):
    """The pair shape and dtype to run on."""
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=list(sizes), help="pair dim N"
    )
    parser.add_argument(
        "--channels", type=int, default=128, help="channel dim (c_z == c_hidden)"
    )
    parser.add_argument(
        "--directions", nargs="+", default=list(directions), choices=list(directions)
    )
    parser.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    parser.add_argument(
        "--biases",
        action="store_true",
        help="give the projections biases. Off by default because the callers "
        "we serve run without them, but the Triton path compiles a separate "
        "kernel for each case, so one says nothing about the other",
    )
    return parser


def add_backend_argument(parser, *, default=("reference", "triton")):
    """Which implementations to run."""
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(default),
        choices=list(TRIMUL_IMPLEMENTATIONS),
    )
    return parser
