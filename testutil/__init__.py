# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Input builders shared by tests and the benchmarks.

Not part of the distributed package. Import as ``testutil.<op>``; the repo root
must be on ``sys.path``.
"""

import os

import torch

# rocjitsu emulates the GPU through an LD_PRELOAD interposer, under both its own
# CLI and `mirage`. Emulation is orders of magnitude slower than hardware, and
# fp64 especially so, so the fp64 oracles run on the CPU there instead.
EMULATED_GPU = "librocjitsu" in os.environ.get("LD_PRELOAD", "")
ORACLE_DEVICE = "cpu" if EMULATED_GPU else None


def generator(device, seed: int) -> torch.Generator:
    """Seeded per-device generator, so builders never touch the global RNG."""
    return torch.Generator(device=device).manual_seed(seed)


def seeded_randn(*shape, device, dtype=torch.float32, seed: int = 0):
    """Standard normal of the requested shape, from a seeded generator."""
    g = generator(device, seed)
    return torch.randn(*shape, device=device, dtype=dtype, generator=g)


def masked_noise(t: torch.Tensor, mask: torch.Tensor, seed: int, size: float = 1e3):
    """``t`` with noise far larger than its own values wherever ``mask`` is set."""
    noise = size * seeded_randn(t.shape, device=t.device, dtype=t.dtype, seed=seed)
    return torch.where(mask, t + noise, t)


def grad_targets(tensors: dict) -> dict:
    """The entries of ``tensors`` that require a gradient."""
    return {
        key: val
        for key, val in tensors.items()
        if isinstance(val, torch.Tensor) and val.requires_grad
    }


def cast_tree(tensors: dict, dtype, *, requires_grad: bool = True, device=None) -> dict:
    """Cast a dict of tensors to fresh leaves, passing ``None`` through.

    ``device=None`` keeps each tensor on its own device.
    """
    return {
        key: (
            None
            if val is None
            else val.to(device=device, dtype=dtype)
            .detach()
            .requires_grad_(requires_grad)
        )
        for key, val in tensors.items()
    }


def to_oracle(t: torch.Tensor) -> torch.Tensor:
    """``t`` in fp64 on the device the oracles run on."""
    return t.to(device=ORACLE_DEVICE, dtype=torch.float64)


def oracle_tree(tensors: dict, *, requires_grad: bool = True) -> dict:
    """``cast_tree`` to fp64 on the device the oracles run on."""
    return cast_tree(
        tensors, torch.float64, requires_grad=requires_grad, device=ORACLE_DEVICE
    )
