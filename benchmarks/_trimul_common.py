# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the trimul benchmark scripts."""

import hashlib
from functools import partial
from pathlib import Path

import torch
from _bench_common import DTYPES

from geometric_nn_ops_torch.triangle import triangle_multiplicative_update
from geometric_nn_ops_torch.triangle._kernels.trimul import trimul_triton
from geometric_nn_ops_torch.triangle._reference.trimul import trimul_torch
from geometric_nn_ops_torch.triangle.experimental import VARIANTS, run_variant

# `public` decided which implementation to delegate to, so choosing it measures
# whatever a real caller would get rather than a specific implementation.
TRIMUL_IMPLEMENTATIONS = {
    "reference": trimul_torch,
    "triton": trimul_triton,
    "public": triangle_multiplicative_update,
}
EXPERIMENTAL_BACKENDS = {
    f"experimental:{name}": partial(run_variant, name)
    for name in VARIANTS
    if name not in ("reference", "geometric")
}
TRIMUL_IMPLEMENTATIONS.update(EXPERIMENTAL_BACKENDS)

BASELINE = "reference"


def benchmark_source_sha256():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ("benchmarks", "geometric_nn_ops_torch", "testutil"):
        for path in sorted((root / folder).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def configure_experiments(parser, args):
    if any(name in EXPERIMENTAL_BACKENDS for name in args.backends):
        if args.dtype != "fp32":
            parser.error("Experimental backends require --dtype fp32")
        torch.set_float32_matmul_precision("highest")


def assert_qualified(actual, expected):
    if not torch.isfinite(expected).all():
        raise AssertionError("Bounded oracle returned non-finite output")
    if not torch.isfinite(actual).all():
        raise AssertionError("Experimental backend returned non-finite output")
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def qualify_experimental(impl, x, direction, mask, weights):
    """Strict FP32 qualification, with an independently repeated bounded oracle."""
    with torch.no_grad():
        expected = run_variant("reference_bounded", x, direction, mask, **weights)
        repeated = run_variant("reference_bounded", x, direction, mask, **weights)
        assert_qualified(repeated, expected)
        actual = impl(x, direction, mask, **weights)
        assert_qualified(actual, expected)
    return actual


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
