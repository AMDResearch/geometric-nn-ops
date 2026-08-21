# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Input builders shared by tests and the benchmarks.

Not part of the distributed package. Import as ``testutil.<op>``; the repo root
must be on ``sys.path``.
"""

import torch


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


def cast_tree(tensors: dict, dtype, *, requires_grad: bool = True) -> dict:
    """Cast a dict of tensors to fresh leaves, passing ``None`` through."""
    return {
        key: (
            None
            if val is None
            else val.to(dtype).detach().requires_grad_(requires_grad)
        )
        for key, val in tensors.items()
    }
