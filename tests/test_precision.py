# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for fp32 matmul precision: resolution, reading it from torch, and the
setting reaching the kernels.

The per-op tolerances are sized for the loosest hardware, so they would pass
even if the setting were ignored. These close that gap.
"""

# The kernel imports run after `importorskip` so the module skips cleanly when
# Triton is missing; E402 (import not at top of file) is disabled accordingly.
# ruff: noqa: E402

import ast
import importlib.util
import pathlib
import pkgutil
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

triton = pytest.importorskip("triton")

import geometric_nn_ops_torch
from geometric_nn_ops_torch import _precision
from testutil.precision import torch_matmul_precision

# Allowed-precision sets as the Triton backends report them, so resolution can
# be tested without the hardware.
NVIDIA_ALLOWED = ("tf32", "tf32x3", "ieee", "bf16x3", "bf16x6")
CDNA3_ALLOWED = ("bf16x3", "bf16x6", "ieee", "tf32")
CDNA4_ALLOWED = ("ieee", "bf16x3", "bf16x6")
IEEE_ONLY_ALLOWED = ("ieee",)


# The resolver, as a pure function. No device needed.


ALL_ALLOWED = (NVIDIA_ALLOWED, CDNA3_ALLOWED, CDNA4_ALLOWED, IEEE_ONLY_ALLOWED)


@pytest.mark.parametrize(
    ("allowed", "expected"),
    [
        (NVIDIA_ALLOWED, "tf32"),
        (CDNA3_ALLOWED, "tf32"),
        (CDNA4_ALLOWED, "bf16x3"),
        (IEEE_ONLY_ALLOWED, "ieee"),
    ],
)
def test_tf32_resolves_to_the_first_supported_mode(allowed, expected):
    """Picks the fastest mode the target has, falling back towards ieee."""
    assert _precision.resolve("tf32", allowed) == expected


@pytest.mark.parametrize("allowed", ALL_ALLOWED)
def test_ieee_never_resolves_to_a_reduced_mode(allowed):
    """ieee promises full fp32, whatever else is on offer."""
    assert _precision.resolve("ieee", allowed) == "ieee"


def test_resolve_rejects_an_unknown_precision():
    with pytest.raises(ValueError, match="unknown precision"):
        _precision.resolve("bf16", NVIDIA_ALLOWED)


def test_resolve_fails_when_no_mode_is_available():
    """A backend offering nothing usable is an error, not a silent pick."""
    with pytest.raises(RuntimeError, match="supports none of"):
        _precision.resolve("tf32", ("bf16x6",))


# Reading the precision from torch.


@pytest.mark.parametrize("precision", ["ieee", "tf32"])
def test_follows_the_torch_setting(precision):
    """torch's setting is the only input."""
    with torch_matmul_precision(precision):
        assert _precision.torch_precision() == precision


def test_reads_ieee_when_torch_has_no_setting():
    """torch reports "none" until something is set, which is full precision.
    Nothing here sets it, so this also catches a test leaving it behind."""
    assert _precision.torch_precision() == "ieee"


def test_reads_allow_tf32_on_torch_without_fp32_precision():
    """The package supports torch back to 2.4, before ``fp32_precision`` existed.
    Stubbing the module is the only way to reach that branch on a modern torch."""
    legacy = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))
        )
    )
    with mock.patch.object(_precision, "torch", legacy):
        assert _precision.torch_precision() == "tf32"
        legacy.backends.cuda.matmul.allow_tf32 = False
        assert _precision.torch_precision() == "ieee"


def test_reading_the_torch_setting_leaves_the_legacy_getters_working():
    """Reading must not break a caller still using the legacy getters. Only
    writing trips torch's mixed-API check."""
    with torch_matmul_precision("ieee"):
        assert _precision.torch_precision() == "ieee"
        assert torch.get_float32_matmul_precision() == "highest"
        assert torch.backends.cuda.matmul.allow_tf32 is False


@pytest.mark.parametrize("precision", ["ieee", "tf32"])
def test_reduced_dtypes_pin_to_ieee(precision):
    """Triton ignores input_precision unless the operands are fp32, so the ops
    pin those dtypes to ieee and compile one kernel rather than two."""
    with torch_matmul_precision(precision):
        assert _precision.matmul_precision(fp32_operands=False) == "ieee"


# Make sure every tl.dot passes an explicit precision argument. Checking the
# source is blunt, but it is the thing most likely to catch an oversight.


def package_sources(package):
    """``(module name, source path)`` for every module in ``package``.

    Enumerated through the import system, so this covers exactly what the
    package imports and avoids e.g. gitignored .py files.
    """
    names = [package.__name__] + [
        info.name
        for info in pkgutil.walk_packages(package.__path__, f"{package.__name__}.")
    ]
    for name in names:
        origin = importlib.util.find_spec(name).origin
        if origin and origin.endswith(".py"):
            yield name, pathlib.Path(origin)


def tl_dot_calls(path):
    """Every ``tl.dot(...)`` call in a source file."""
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "dot"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tl"
        ):
            yield node


def test_every_tl_dot_is_given_a_precision():
    found, missing = 0, []
    for name, path in package_sources(geometric_nn_ops_torch):
        for call in tl_dot_calls(path):
            found += 1
            if not any(kw.arg == "input_precision" for kw in call.keywords):
                missing.append(f"{name}:{call.lineno}")

    assert found, "found no tl.dot calls at all; the scan is broken"
    assert not missing, "tl.dot without input_precision at " + ", ".join(missing)
