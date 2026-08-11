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


def make_grad_output(*shape, device, dtype=torch.float32, seed: int = 0):
    """Seeded upstream gradient with the requested shape."""
    g = generator(device, seed)
    return torch.randn(*shape, device=device, dtype=dtype, generator=g)


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
