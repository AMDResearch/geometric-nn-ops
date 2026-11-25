# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Numerical metrics and per-op oracle tolerances

Shared by tests and benchmarks."""

from typing import NamedTuple

import torch


class Tolerance(NamedTuple):
    """Acceptance bounds against the fp64 oracle."""

    cos_min: float
    rel_fro_max: float


# Accumulation dtype for the metrics. Callers comparing tensors too large to
# copy at fp64 -- the large-N benchmarks, where the operands are already
# multiple gigabytes -- pass fp32.
ACCUMULATE = torch.float64


def cos(a: torch.Tensor, b: torch.Tensor, *, accumulate=ACCUMULATE) -> float:
    """Cosine similarity of two tensors, flattened.

    Two all-zero tensors count as a perfect match. The angle between them is
    undefined and torch reports 0, which would reject an exact match; a
    structurally zero gradient is a legitimate oracle result. One zero side
    against a non-zero one still scores 0.

    Computed on ``b``'s device, see ``testutil.ORACLE_DEVICE``.
    """
    a = a.detach().flatten().to(device=b.device, dtype=accumulate)
    b = b.detach().flatten().to(accumulate)
    if torch.linalg.vector_norm(a) == 0 and torch.linalg.vector_norm(b) == 0:
        return 1.0
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def rel_fro(a: torch.Tensor, b: torch.Tensor, *, accumulate=ACCUMULATE) -> float:
    """``||a - b|| / ||b||``. ``b`` is the reference, and sets the device."""
    a = a.detach().to(device=b.device, dtype=accumulate)
    b = b.detach().to(accumulate)
    return (
        torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-30)
    ).item()


def assert_matches_oracle(comparisons, tol: Tolerance, *, label: str = "") -> None:
    """Assert every ``(name, candidate, oracle)`` triple is within ``tol``.

    Uses cosine similarity and relative Frobenius norm.
    """
    comparisons = list(comparisons)
    for name, a, b in comparisons:
        assert a.shape == b.shape, (
            f"[{label}] {name}: candidate has shape {tuple(a.shape)}, "
            f"oracle {tuple(b.shape)}"
        )
        assert torch.isfinite(b).all(), f"[{label}] {name}: oracle has non-finite"
        assert torch.isfinite(a).all(), f"[{label}] {name}: candidate has non-finite"

    worst_cos, worst_fro, msgs = 1.0, 0.0, []
    for name, a, b in comparisons:
        c, f = cos(a, b), rel_fro(a, b)
        worst_cos = min(worst_cos, c)
        worst_fro = max(worst_fro, f)
        msgs.append(f"{name}: cos={c:.6f} fro={f:.3e}")

    detail = "\n".join("  " + m for m in msgs)
    assert worst_cos >= tol.cos_min, (
        f"[{label}] cosine {worst_cos:.6f} < {tol.cos_min:.6f}\n{detail}"
    )
    assert worst_fro <= tol.rel_fro_max, (
        f"[{label}] rel-fro {worst_fro:.3e} > {tol.rel_fro_max:.1e}\n{detail}"
    )


# Per-op, per-dtype relative Frobenius norm and cosine similarity tolerances
# against the fp64 oracle, over the forward output and every gradient. Kernels
# legitimately differ: a flash attention reduces over the key axis in a
# different order than a LayerNorm does over channels, so it drifts further for
# the same input.
#
# Keyed by (dtype, fp32 matmul precision). Only fp32 has both precisions: for
# fp16/bf16 the dots are already reduced precision and the setting cannot change
# them, so those combinations are absent and a lookup for one is a KeyError.
#
# The fp32/tf32 bound covers 10-mantissa-bits, since this is what the mode
# guarantees, so the tolerance is slack for more accurate emulations such as
# bf16x3.
#
# TODO(gmngeoffrey): re-calibrate the bounds here.

TRIMUL_TOL = {
    (torch.float32, "ieee"): Tolerance(1.0 - 1e-6, 1.5e-3),
    (torch.float32, "tf32"): Tolerance(1.0 - 1e-6, 2e-3),
    (torch.float16, "ieee"): Tolerance(1.0 - 1e-6, 2e-3),
    (torch.bfloat16, "ieee"): Tolerance(1.0 - 1e-4, 1.5e-2),
}

TRIANGLE_ATTENTION_TOL = {
    (torch.float32, "ieee"): Tolerance(1.0 - 1e-5, 1.5e-3),
    (torch.float32, "tf32"): Tolerance(1.0 - 1e-5, 2e-3),
    (torch.float16, "ieee"): Tolerance(1.0 - 1e-4, 2e-2),
    (torch.bfloat16, "ieee"): Tolerance(1.0 - 1e-4, 5e-2),
}

ATTENTION_PAIR_BIAS_TOL = {
    (torch.float32, "ieee"): Tolerance(1.0 - 1e-5, 2e-3),
    (torch.float32, "tf32"): Tolerance(1.0 - 1e-5, 3e-3),
    (torch.float16, "ieee"): Tolerance(1.0 - 1e-4, 5e-3),
    (torch.bfloat16, "ieee"): Tolerance(1.0 - 1e-4, 2e-2),
}
