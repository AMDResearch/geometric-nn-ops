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
def test_tri_attn_routes_through_shim():
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

@requires_gpu
def test_tri_attn_batchless_call_matches_explicit_batch():
    """The batch axis being dropped must not change the result.

    cuequivariance documents that ``B == 1`` may be left out of q/k/v, the bias
    and the mask. Adding the axis back by hand should be a no-op.
    """
    from cuequivariance_torch.primitives.triangle import triangle_attention

    N_seq, H, N_res, D = 2, 4, 64, 32
    device, dtype = "cuda", torch.bfloat16
    q = torch.randn(N_seq, H, N_res, D, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bias = torch.randn(1, H, N_res, N_res, device=device, dtype=dtype) * 0.1
    mask = torch.rand(N_seq, 1, 1, N_res, device=device) < 0.75
    mask[..., 0] = True  # a fully masked row leaves softmax undefined

    batchless = triangle_attention(q, k, v, bias, mask=mask, scale=D**-0.5)
    batched = triangle_attention(
        q[None], k[None], v[None], bias[None], mask=mask[None], scale=D**-0.5
    )

    assert batchless.shape == q.shape
    torch.testing.assert_close(batchless[None], batched, rtol=0, atol=0)


@requires_gpu
def test_attention_pair_bias_routes_through_shim():
    from cuequivariance_torch.primitives.triangle import attention_pair_bias

    # cuequivariance layout: s (BM, S, D), q/k/v (BM, H, U/V, DH), z (B, U, V, Zc).
    B, M, H, U, V, DH, Zc = 2, 3, 4, 32, 32, 16, 16
    D, BM = H * DH, B * M
    device, dtype = "cuda", torch.bfloat16

    def rn(*shape, scale=1.0):
        return torch.randn(*shape, device=device, dtype=dtype) * scale

    out, proj_z = attention_pair_bias(
        rn(BM, U, D),
        rn(BM, H, U, DH),
        rn(BM, H, V, DH),
        rn(BM, H, V, DH),
        rn(B, U, V, Zc),
        (torch.rand(B, V, device=device) > 0.1).to(dtype),
        H,
        w_proj_z=rn(H, Zc, scale=Zc**-0.5),
        w_proj_g=rn(D, D, scale=D**-0.5),
        w_proj_o=rn(D, D, scale=D**-0.5),
        w_ln_z=torch.ones(Zc, device=device, dtype=dtype),
        b_ln_z=torch.zeros(Zc, device=device, dtype=dtype),
        attn_scale=DH**-0.5,
    )

    assert out.shape == (BM, U, D)
    assert out.dtype == dtype
    assert proj_z is not None and proj_z.shape == (B, H, U, V)
