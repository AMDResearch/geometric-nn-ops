# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Honors torch's fp32 matmul precision setting in the Triton kernels.

``torch.backends.cuda.matmul.fp32_precision`` selects ``"ieee"`` or ``"tf32"``.
Which reduced-precision dot a target actually has varies so this resolves to the
fastest supported mode with at least the required precision.
"""

import torch

try:
    import triton

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

IEEE = "ieee"
TF32 = "tf32"

# Triton dot modes for each setting, fastest first. Every chain ends at "ieee",
# which all backends support, so resolution always succeeds and never lands on
# something less accurate than asked for.
_MODES = {
    TF32: ("tf32", "bf16x3", "ieee"),
    IEEE: ("ieee",),
}

_ALLOWED_BY_TARGET: dict[object, tuple[str, ...]] = {}


def torch_precision() -> str:
    """torch's current fp32 matmul precision.

    Uses ``matmul.fp32_precision`` rather than the older
    ``get_float32_matmul_precision()`` or ``matmul.allow_tf32``: those raise if
    a process has mixed torch's precision APIs whereas this works no matter how
    the precision was set.
    """
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        return TF32 if torch.backends.cuda.matmul.fp32_precision == TF32 else IEEE
    return TF32 if torch.backends.cuda.matmul.allow_tf32 else IEEE


def resolve(precision: str, allowed: tuple[str, ...]) -> str:
    """Triton dot mode for ``precision`` on a target supporting ``allowed``."""
    if precision not in _MODES:
        raise ValueError(
            f"unknown precision {precision!r}, expected one of {tuple(_MODES)}"
        )
    for mode in _MODES[precision]:
        if mode in allowed:
            return mode
    # This should never happen because all targets should support ieee and we
    # should always bottom out at that.
    raise RuntimeError(
        f"target supports none of {_MODES[precision]}; it allows only {allowed}"
    )


def _allowed_dot_precisions() -> tuple[str, ...]:
    """Allowed tl.dot input precisions the active device supports, cached per target."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required to query dot input precisions")
    target = triton.runtime.driver.active.get_current_target()
    if target not in _ALLOWED_BY_TARGET:
        options = triton.compiler.make_backend(target).parse_options({})
        _ALLOWED_BY_TARGET[target] = tuple(options.allowed_dot_input_precisions)
    return _ALLOWED_BY_TARGET[target]


def matmul_precision(*, fp32_operands: bool) -> str:
    """Mode to use for ``tl.dot`` based on torch global.

    Triton ignores the setting unless the operands are fp32, so
    ``fp32_operands=False`` pins to ieee rather than compiling a second,
    identical kernel.
    """
    if not fp32_operands:
        return IEEE
    return resolve(torch_precision(), _allowed_dot_precisions())
