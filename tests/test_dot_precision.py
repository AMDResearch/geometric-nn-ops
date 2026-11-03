# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for fp32 matmul precision: resolution, reading it from torch, and the
setting reaching the kernels.

The per-op tolerances are sized for the loosest hardware, so they would pass
even if the setting were ignored. These close that gap.
"""

import ast
import importlib.util
import pathlib
import pkgutil
from types import SimpleNamespace
from unittest import mock

import pytest

triton = pytest.importorskip("triton")

# Imports after the Triton skip
# ruff: noqa: E402

import torch

import geometric_nn_ops_torch
from geometric_nn_ops_torch.triangle._kernels import dot_precision
from testutil.precision import torch_matmul_precision

# Allowed-precision sets as the Triton backends report them, so resolution can
# be tested without the hardware.
NVIDIA_ALLOWED = ("tf32", "tf32x3", "ieee", "bf16x3", "bf16x6")
CDNA3_ALLOWED = ("bf16x3", "bf16x6", "ieee", "tf32")
CDNA4_ALLOWED = ("ieee", "bf16x3", "bf16x6")
IEEE_ONLY_ALLOWED = ("ieee",)
ALL_ALLOWED = (NVIDIA_ALLOWED, CDNA3_ALLOWED, CDNA4_ALLOWED, IEEE_ONLY_ALLOWED)


# The resolver, as a pure function. No device needed.


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
    assert dot_precision.resolve("tf32", allowed) == expected


@pytest.mark.parametrize("allowed", ALL_ALLOWED)
def test_ieee_never_resolves_to_a_reduced_mode(allowed):
    """ieee promises full fp32, whatever else is on offer."""
    assert dot_precision.resolve("ieee", allowed) == "ieee"


def test_resolve_rejects_an_unknown_precision():
    with pytest.raises(ValueError, match="unknown precision"):
        dot_precision.resolve("bf16", NVIDIA_ALLOWED)


def test_resolve_fails_when_no_mode_is_available():
    """A backend offering nothing usable is an error, not a silent pick."""
    with pytest.raises(RuntimeError, match="supports none of"):
        dot_precision.resolve("tf32", ("bf16x6",))


# Reading the precision from torch.


@pytest.mark.parametrize("precision", ["ieee", "tf32"])
def test_follows_the_torch_setting(precision):
    """torch's setting is the only input."""
    with torch_matmul_precision(precision):
        assert dot_precision.torch_fp32_precision() == precision


def test_reads_ieee_when_torch_has_no_setting():
    """torch reports "none" until something is set, which is full precision."""
    assert dot_precision.torch_fp32_precision() == "ieee"


@pytest.mark.parametrize(("allow_tf32", "expected"), [(True, "tf32"), (False, "ieee")])
def test_reads_allow_tf32_on_torch_without_fp32_precision(allow_tf32, expected):
    """torch before 2.9 has only ``matmul.allow_tf32``. Stubbing the module is
    the only way to reach that branch on a newer torch."""
    legacy = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=allow_tf32))
        )
    )
    with mock.patch.object(dot_precision, "torch", legacy):
        assert dot_precision.torch_fp32_precision() == expected


def test_reading_the_torch_setting_leaves_the_legacy_getters_working():
    """Reading must not break a caller still using the legacy getters. Only
    writing trips torch's mixed-API check."""
    with torch_matmul_precision("ieee"):
        assert dot_precision.torch_fp32_precision() == "ieee"
        assert torch.get_float32_matmul_precision() == "highest"
        assert torch.backends.cuda.matmul.allow_tf32 is False


@pytest.mark.parametrize("precision", ["ieee", "tf32"])
def test_reduced_dtypes_pin_to_ieee(precision):
    with torch_matmul_precision(precision):
        assert dot_precision.current_dot_precision(fp32_operands=False) == "ieee"


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
