# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPU: cuequivariance_torch frontend smoke tests through our shim.

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

    The result carries the batch axis either way: cueq pads the input up to rank
    5 and returns it at that rank.
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

    assert batchless.shape == (1, *q.shape)
    torch.testing.assert_close(batchless, batched, rtol=0, atol=0)


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
        attn_scale=DH**-0.5,
    )

    assert out.shape == (BM, U, D)
    assert out.dtype == dtype
    assert proj_z is not None and proj_z.shape == (B, H, U, V)


def _apb_kwargs(B, M, H, U, V, DH, Zc, device, dtype, seed=0):
    """Build a complete keyword argument set for an APB shim call."""
    D, BM = H * DH, B * M
    g = torch.Generator(device=device).manual_seed(seed)

    def rn(*shape, scale=1.0):
        return torch.randn(*shape, device=device, dtype=dtype, generator=g) * scale

    return dict(
        s=rn(BM, U, D),
        q=rn(BM, H, U, DH),
        k=rn(BM, H, V, DH),
        v=rn(BM, H, V, DH),
        z=rn(B, U, V, Zc),
        mask=(torch.rand(B, V, device=device, generator=g) > 0.1).to(dtype),
        num_heads=H,
        w_proj_z=rn(H, Zc, scale=Zc**-0.5),
        w_proj_g=rn(D, D, scale=D**-0.5),
        w_proj_o=rn(D, D, scale=D**-0.5),
        w_ln_z=torch.ones(Zc, device=device, dtype=dtype),
        attn_scale=DH**-0.5,
    )


@requires_gpu
@pytest.mark.parametrize("bias_arg", ["b_ln_z", "b_proj_z"])
@pytest.mark.parametrize("fill", [0.0, 1.0], ids=["zeros", "nonzero"])
def test_inert_pair_bias_bias_warns(bias_arg, fill):
    """Warns on any non-None value, including the zeros a stock nn.LayerNorm
    starts with -- the point is that the parameter should not be carried."""
    from cuequivariance_torch.primitives.triangle import attention_pair_bias

    B, M, H, U, V, DH, Zc = 1, 1, 4, 16, 16, 8, 16
    kwargs = _apb_kwargs(B, M, H, U, V, DH, Zc, "cuda", torch.float32)
    size = Zc if bias_arg == "b_ln_z" else H
    kwargs[bias_arg] = torch.full((size,), fill, device="cuda")

    with pytest.warns(UserWarning, match=f"{bias_arg} is ignored"):
        attention_pair_bias(**kwargs)


@requires_gpu
@pytest.mark.parametrize("bias_arg", ["b_ln_z", "b_proj_z"])
def test_inert_pair_bias_bias_does_not_change_the_output(bias_arg):
    """An ignored bias leaves the output and projected bias unchanged."""
    from cuequivariance_torch.primitives.triangle import attention_pair_bias

    B, M, H, U, V, DH, Zc = 1, 1, 4, 16, 16, 8, 16
    kwargs = _apb_kwargs(B, M, H, U, V, DH, Zc, "cuda", torch.float64)
    size = Zc if bias_arg == "b_ln_z" else H
    baseline, baseline_z = attention_pair_bias(**kwargs)

    with pytest.warns(UserWarning):
        biased, biased_z = attention_pair_bias(
            **kwargs, **{bias_arg: torch.full((size,), 1e3, device="cuda")}
        )

    torch.testing.assert_close(biased, baseline, rtol=0, atol=0)
    torch.testing.assert_close(biased_z, baseline_z, rtol=0, atol=0)


@requires_gpu
def test_returned_pair_bias_round_trips():
    """A projected bias returned by the shim can be reused as cached input."""
    from cuequivariance_torch.primitives.triangle import attention_pair_bias

    B, M, H, U, V, DH, Zc = 1, 1, 4, 16, 16, 8, 16
    kwargs = _apb_kwargs(B, M, H, U, V, DH, Zc, "cuda", torch.float64)
    direct, proj_z = attention_pair_bias(**kwargs)

    cached, _ = attention_pair_bias(
        **{**kwargs, "z": proj_z, "w_proj_z": None, "w_ln_z": None},
        is_cached_z_proj=True,
    )

    torch.testing.assert_close(cached, direct, rtol=0, atol=0)


@requires_gpu
def test_upstream_trimul_routes_through_shim():
    from cuequivariance_torch.primitives.triangle import (
        triangle_multiplicative_update,
    )

    # cuequivariance layout: x (B, N, N, C_z), mask (B, N, N). The input
    # projections are the concatenated [a; b] form, so they carry 2 * C_hidden
    # rows.
    B, N, C_z, C_hidden = 1, 32, 32, 32
    device, dtype = "cuda", torch.bfloat16

    def rn(*shape, scale=1.0):
        return torch.randn(*shape, device=device, dtype=dtype) * scale

    out = triangle_multiplicative_update(
        rn(B, N, N, C_z, scale=0.5),
        "outgoing",
        mask=(torch.rand(B, N, N, device=device) > 0.1).to(dtype),
        norm_in_weight=torch.ones(C_z, device=device, dtype=dtype),
        norm_in_bias=torch.zeros(C_z, device=device, dtype=dtype),
        p_in_weight=rn(2 * C_hidden, C_z, scale=0.05),
        g_in_weight=rn(2 * C_hidden, C_z, scale=0.05),
        norm_out_weight=torch.ones(C_hidden, device=device, dtype=dtype),
        norm_out_bias=torch.zeros(C_hidden, device=device, dtype=dtype),
        p_out_weight=rn(C_z, C_hidden, scale=0.05),
        g_out_weight=rn(C_z, C_z, scale=0.05),
    )

    assert out.shape == (B, N, N, C_z)
    assert out.dtype == dtype
