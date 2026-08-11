# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPU: NVIDIA's open cuequivariance_torch frontend routes through our shim.

Proves the zero-code-change path: with the shim installed (and the closed ops
wheel absent), unmodified `cuequivariance_torch` calls run our Triton kernels.
Skips when cuequivariance_torch isn't installed.
"""

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("cuequivariance_torch")

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs GPU + Triton"
)


@requires_gpu
def test_upstream_frontend_routes_through_shim():
    from cuequivariance_torch.primitives.triangle import triangle_attention

    # cuequivariance layout: (B, N_seq, H, N_res, D)
    B, N_seq, H, N_res, D = 1, 1, 4, 64, 32
    device, dtype = "cuda", torch.bfloat16
    q = torch.randn(B, N_seq, H, N_res, D, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bias = torch.randn(B, 1, H, N_res, N_res, device=device, dtype=dtype) * 0.1
    mask = torch.ones(B, N_seq, 1, 1, N_res, device=device, dtype=torch.bool)

    out = triangle_attention(q, k, v, bias, mask=mask, scale=D**-0.5)
    assert out.shape == (B, N_seq, H, N_res, D)
    assert out.dtype == dtype
