# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for triangle_attention (Triton flash kernel + adapter).

Correctness is measured against an fp64 eager PyTorch oracle (cos + rel-fro) on
the forward output and on leaf gradients (q, k, v, and bias when present).
"""

import pytest
import torch

triton = pytest.importorskip("triton")

from geometric_nn_ops_torch.triangle import triangle_attention  # noqa: E402
from geometric_nn_ops_torch.triangle._triattn import NEG_INF  # noqa: E402

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs GPU + Triton"
)

_INPUT_KEYS = ["q", "k", "v"]
_TOL = {
    torch.float32: (1.0 - 1e-5, 3e-3),
    torch.bfloat16: (1.0 - 1e-4, 5e-2),
    torch.float16: (1.0 - 1e-4, 2e-2),
}


def _cos(a, b):
    a = a.detach().flatten().double()
    b = b.detach().flatten().double()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _rel_fro(a, b):
    a = a.detach().double()
    b = b.detach().double()
    return (
        torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-30)
    ).item()


def _eager_triangle_attention(q, k, v, bias=None, mask=None, scale=None):
    """Autograd-friendly reference (mirrors test_smoke's eager helper)."""
    dim = q.shape[-1]
    if scale is None:
        scale = dim**-0.5
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    qs = q.transpose(-2, -3).to(acc)
    ks = k.transpose(-2, -3).to(acc)
    vs = v.transpose(-2, -3).to(acc)
    logits = torch.einsum("bshid,bshjd->bshij", qs * scale, ks)
    if bias is not None:
        b = bias
        if b.dim() == 4:
            b = b.unsqueeze(-4)
        logits = logits + b.to(acc)
    if mask is not None:
        if mask.dtype == torch.bool:
            logits = logits.masked_fill(~mask, float("-inf"))
        else:
            logits = logits + mask.to(acc)
    probs = torch.softmax(logits, dim=-1)
    o = torch.einsum("bshij,bshjd->bshid", probs, vs)
    return o.transpose(-2, -3).to(q.dtype)


def _make_inputs(B, N_seq, N_res, H, D, device, dtype, seed, *, with_bias=True):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, N_seq, N_res, H, D, device=device, dtype=dtype, generator=g)
    k = torch.randn(B, N_seq, N_res, H, D, device=device, dtype=dtype, generator=g)
    v = torch.randn(B, N_seq, N_res, H, D, device=device, dtype=dtype, generator=g)
    bias = None
    if with_bias:
        bias = (
            torch.randn(B, 1, H, N_res, N_res, device=device, dtype=dtype, generator=g)
            * 0.1
        )
    return q, k, v, bias


def _bool_mask(B, N_seq, N_res, device, seed, *, drop_last=1):
    del seed
    mask = torch.ones(B, N_seq, 1, 1, N_res, device=device, dtype=torch.bool)
    if drop_last:
        mask[..., -drop_last:] = False
    return mask


def _run_triton(q, k, v, bias, mask, scale):
    return triangle_attention(q, k, v, bias=bias, mask=mask, scale=scale)


def _run_oracle(q, k, v, bias, mask, scale):
    return _eager_triangle_attention(q, k, v, bias=bias, mask=mask, scale=scale)


def _fwd_bwd(fn, q, k, v, bias, mask, scale, go):
    out = fn(q, k, v, bias, mask, scale)
    targets = [q, k, v]
    if bias is not None:
        targets.append(bias)
    grads = torch.autograd.grad(out, targets, grad_outputs=go, retain_graph=False)
    return out, grads


def _assert_accuracy(
    dtype, out_t, g_t, out_o, g_o, *, tol=None, label="", grad_keys=None
):
    cos_min, fro_max = tol if tol is not None else _TOL[dtype]
    if grad_keys is None:
        grad_keys = _INPUT_KEYS + (["bias"] if len(g_t) > 3 else [])
    names = ["out"] + [f"grad_{k}" for k in grad_keys]
    tensors_t = [out_t, *g_t]
    tensors_o = [out_o, *g_o]
    worst_cos, worst_fro = 1.0, 0.0
    msgs = []
    for name, a, b in zip(names, tensors_t, tensors_o, strict=True):
        assert torch.isfinite(a).all(), f"[{label}] {name}: candidate non-finite"
        assert torch.isfinite(b).all(), f"[{label}] {name}: oracle non-finite"
        c, f = _cos(a, b), _rel_fro(a, b)
        worst_cos = min(worst_cos, c)
        worst_fro = max(worst_fro, f)
        msgs.append(f"{name}: cos={c:.6f} fro={f:.3e}")
    print(f"\n[{label}] worst cos={worst_cos:.6f} worst fro={worst_fro:.3e}")
    for m in msgs:
        print("  " + m)
    assert worst_cos >= cos_min, (
        f"cosine {worst_cos:.6f} < {cos_min:.6f}\n" + "\n".join(msgs)
    )
    assert worst_fro <= fro_max, (
        f"rel-fro {worst_fro:.3e} > {fro_max:.1e}\n" + "\n".join(msgs)
    )


_SHAPES = [
    pytest.param((1, 1, 32, 4, 32), id="pairformer_32"),
    pytest.param((1, 4, 32, 8, 8), id="msa_d8"),
    pytest.param((1, 1, 64, 16, 48), id="atom_d48"),
    pytest.param((2, 2, 48, 4, 32), id="batched"),
]


@requires_gpu
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16],
    ids=["fp32", "bf16", "fp16"],
)
def test_triton_vs_fp64_oracle(shape, dtype):
    """Triton adapter fwd+bwd vs fp64 eager oracle."""
    B, N_seq, N_res, H, D = shape
    scale = D**-0.5
    q0, k0, v0, b0 = _make_inputs(B, N_seq, N_res, H, D, "cuda", torch.float32, seed=3)
    mask = _bool_mask(B, N_seq, N_res, "cuda", seed=1)

    q = q0.to(dtype).detach().requires_grad_(True)
    k = k0.to(dtype).detach().requires_grad_(True)
    v = v0.to(dtype).detach().requires_grad_(True)
    bias = b0.to(dtype).detach().requires_grad_(True)
    probe = _run_triton(q, k, v, bias, mask, scale)
    go = torch.randn_like(probe)
    out_t, g_t = _fwd_bwd(_run_triton, q, k, v, bias, mask, scale, go)

    qo = q0.double().detach().requires_grad_(True)
    ko = k0.double().detach().requires_grad_(True)
    vo = v0.double().detach().requires_grad_(True)
    bo = b0.double().detach().requires_grad_(True)
    out_o, g_o = _fwd_bwd(_run_oracle, qo, ko, vo, bo, mask, scale, go.double())

    _assert_accuracy(dtype, out_t, g_t, out_o, g_o, label=f"tri-attn/{dtype}/{shape}")


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_no_bias(dtype):
    """bias=None skips pair-bias work in the kernel."""
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 32
    scale = D**-0.5
    q0, k0, v0, _ = _make_inputs(
        B, N_seq, N_res, H, D, "cuda", torch.float32, seed=5, with_bias=False
    )
    mask = _bool_mask(B, N_seq, N_res, "cuda", seed=2, drop_last=0)

    q = q0.to(dtype).detach().requires_grad_(True)
    k = k0.to(dtype).detach().requires_grad_(True)
    v = v0.to(dtype).detach().requires_grad_(True)
    go = torch.randn(B, N_seq, N_res, H, D, device="cuda", dtype=dtype)
    out_t, g_t = _fwd_bwd(_run_triton, q, k, v, None, mask, scale, go)

    qo = q0.double().detach().requires_grad_(True)
    ko = k0.double().detach().requires_grad_(True)
    vo = v0.double().detach().requires_grad_(True)
    out_o, g_o = _fwd_bwd(_run_oracle, qo, ko, vo, None, mask, scale, go.double())

    _assert_accuracy(
        dtype, out_t, g_t, out_o, g_o, grad_keys=_INPUT_KEYS, label=f"no_bias/{dtype}"
    )


@requires_gpu
def test_no_mask():
    """mask=None (all keys kept)."""
    dtype = torch.bfloat16
    B, N_seq, N_res, H, D = 1, 2, 32, 4, 32
    scale = D**-0.5
    q0, k0, v0, b0 = _make_inputs(B, N_seq, N_res, H, D, "cuda", torch.float32, seed=7)

    q = q0.to(dtype).detach().requires_grad_(True)
    k = k0.to(dtype).detach().requires_grad_(True)
    v = v0.to(dtype).detach().requires_grad_(True)
    bias = b0.to(dtype).detach().requires_grad_(True)
    go = torch.randn_like(q)
    out_t, g_t = _fwd_bwd(_run_triton, q, k, v, bias, None, scale, go)

    qo = q0.double().detach().requires_grad_(True)
    ko = k0.double().detach().requires_grad_(True)
    vo = v0.double().detach().requires_grad_(True)
    bo = b0.double().detach().requires_grad_(True)
    out_o, g_o = _fwd_bwd(_run_oracle, qo, ko, vo, bo, None, scale, go.double())

    _assert_accuracy(dtype, out_t, g_t, out_o, g_o, label="no_mask")


@requires_gpu
def test_additive_mask():
    """Float additive mask (0=keep, NEG_INF=mask) vs bool mask equivalence."""
    dtype = torch.bfloat16
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 32
    scale = D**-0.5
    q0, k0, v0, b0 = _make_inputs(B, N_seq, N_res, H, D, "cuda", torch.float32, seed=11)
    bool_m = _bool_mask(B, N_seq, N_res, "cuda", seed=4)
    add_m = (~bool_m).to(torch.float32) * NEG_INF

    q = q0.to(dtype).detach().requires_grad_(True)
    k = k0.to(dtype).detach().requires_grad_(True)
    v = v0.to(dtype).detach().requires_grad_(True)
    bias = b0.to(dtype).detach().requires_grad_(True)
    go = torch.randn_like(q)
    out_bool, _ = _fwd_bwd(_run_triton, q, k, v, bias, bool_m, scale, go)

    q2 = q0.to(dtype).detach().requires_grad_(True)
    k2 = k0.to(dtype).detach().requires_grad_(True)
    v2 = v0.to(dtype).detach().requires_grad_(True)
    bias2 = b0.to(dtype).detach().requires_grad_(True)
    out_add, _ = _fwd_bwd(_run_triton, q2, k2, v2, bias2, add_m, scale, go)

    assert _cos(out_bool, out_add) >= 1.0 - 1e-4
    assert _rel_fro(out_bool, out_add) <= 2e-2


@requires_gpu
def test_4d_bias_layout():
    """4D bias [B,H,N,N] (no N_seq dim) as used by attention_pair_bias."""
    dtype = torch.bfloat16
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 32
    scale = D**-0.5
    q0, k0, v0, _ = _make_inputs(
        B, N_seq, N_res, H, D, "cuda", torch.float32, seed=13, with_bias=False
    )
    bias4 = torch.randn(B, H, N_res, N_res, device="cuda", dtype=dtype) * 0.05
    mask = _bool_mask(B, N_seq, N_res, "cuda", seed=3)

    q = q0.to(dtype).detach().requires_grad_(True)
    k = k0.to(dtype).detach().requires_grad_(True)
    v = v0.to(dtype).detach().requires_grad_(True)
    b = bias4.detach().requires_grad_(True)
    go = torch.randn_like(q)
    out_t, g_t = _fwd_bwd(_run_triton, q, k, v, b, mask, scale, go)

    qo = q0.double().detach().requires_grad_(True)
    ko = k0.double().detach().requires_grad_(True)
    vo = v0.double().detach().requires_grad_(True)
    bo = bias4.double().detach().requires_grad_(True)
    out_o, g_o = _fwd_bwd(_run_oracle, qo, ko, vo, bo, mask, scale, go.double())

    _assert_accuracy(dtype, out_t, g_t, out_o, g_o, label="bias4d")


@requires_gpu
def test_custom_scale():
    """Non-default softmax scale."""
    dtype = torch.bfloat16
    B, N_seq, N_res, H, D = 1, 2, 24, 4, 32
    scale = (D**-0.5) * 1.75
    q0, k0, v0, b0 = _make_inputs(B, N_seq, N_res, H, D, "cuda", torch.float32, seed=17)
    mask = _bool_mask(B, N_seq, N_res, "cuda", seed=5, drop_last=0)

    q = q0.to(dtype).detach().requires_grad_(True)
    k = k0.to(dtype).detach().requires_grad_(True)
    v = v0.to(dtype).detach().requires_grad_(True)
    bias = b0.to(dtype).detach().requires_grad_(True)
    go = torch.randn_like(q)
    out_t, g_t = _fwd_bwd(_run_triton, q, k, v, bias, mask, scale, go)

    qo = q0.double().detach().requires_grad_(True)
    ko = k0.double().detach().requires_grad_(True)
    vo = v0.double().detach().requires_grad_(True)
    bo = b0.double().detach().requires_grad_(True)
    out_o, g_o = _fwd_bwd(_run_oracle, qo, ko, vo, bo, mask, scale, go.double())

    _assert_accuracy(dtype, out_t, g_t, out_o, g_o, label="custom_scale")


@requires_gpu
def test_dynamic_entry_matches_specialized():
    """dynamic=True variable-shape entry vs default specialized kernel."""
    dtype = torch.bfloat16
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 32
    scale = D**-0.5
    q, k, v, bias = _make_inputs(B, N_seq, N_res, H, D, "cuda", dtype, seed=19)
    mask = _bool_mask(B, N_seq, N_res, "cuda", seed=6, drop_last=0)

    o_spec = triangle_attention(
        q, k, v, bias=bias, mask=mask, scale=scale, dynamic=False
    )
    o_dyn = triangle_attention(q, k, v, bias=bias, mask=mask, scale=scale, dynamic=True)
    assert _cos(o_spec, o_dyn) >= 1.0 - 1e-4
    assert _rel_fro(o_spec, o_dyn) <= 2e-2


@requires_gpu
def test_large_n_forward():
    """N=384 forward vs fp64 oracle (OF3 pairformer scale)."""
    dtype = torch.bfloat16
    B, N_seq, N_res, H, D = 1, 1, 384, 16, 48
    scale = D**-0.5
    q0, k0, v0, b0 = _make_inputs(B, N_seq, N_res, H, D, "cuda", torch.float32, seed=21)
    mask = _bool_mask(B, N_seq, N_res, "cuda", seed=7, drop_last=0)

    with torch.no_grad():
        out_t = _run_triton(
            q0.to(dtype),
            k0.to(dtype),
            v0.to(dtype),
            b0.to(dtype),
            mask,
            scale,
        )
        out_o = _run_oracle(
            q0.double(),
            k0.double(),
            v0.double(),
            b0.double(),
            mask,
            scale,
        )
    assert _cos(out_t, out_o) >= 1.0 - 1e-4
    assert _rel_fro(out_t, out_o) <= 5e-2


@requires_gpu
def test_eager_reference_vs_fp64_oracle():
    """fp32 eager reference vs fp64 oracle (validates the reference itself)."""
    dtype = torch.float32
    B, N_seq, N_res, H, D = 1, 1, 32, 4, 32
    scale = D**-0.5
    q0, k0, v0, b0 = _make_inputs(B, N_seq, N_res, H, D, "cuda", torch.float32, seed=23)
    mask = _bool_mask(B, N_seq, N_res, "cuda", seed=8)

    q = q0.detach().requires_grad_(True)
    k = k0.detach().requires_grad_(True)
    v = v0.detach().requires_grad_(True)
    bias = b0.detach().requires_grad_(True)
    go = torch.randn_like(q)
    out_t, g_t = _fwd_bwd(_run_oracle, q, k, v, bias, mask, scale, go)

    qo = q0.double().detach().requires_grad_(True)
    ko = k0.double().detach().requires_grad_(True)
    vo = v0.double().detach().requires_grad_(True)
    bo = b0.double().detach().requires_grad_(True)
    out_o, g_o = _fwd_bwd(_run_oracle, qo, ko, vo, bo, mask, scale, go.double())

    _assert_accuracy(
        dtype, out_t, g_t, out_o, g_o, tol=(1.0 - 1e-6, 1e-3), label="eager_ref"
    )
