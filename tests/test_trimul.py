# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for triangle_multiplicative_update.

Correctness is measured against an fp64 torch oracle using two scale-robust
metrics, on the forward output and on every gradient:

  * cosine similarity of the flattened tensors (catches directional / structural
    error; insensitive to overall scale), and
  * relative Frobenius norm ``||a - b|| / ||b||`` (catches magnitude error;
    unlike max-elementwise-rel it is not dominated by near-zero denominators, so
    it stays meaningful for large inputs).

Tolerances are per-dtype and calibrated on MI300X. fp32 is looser than true IEEE
fp32 (~1e-6) because the Triton projection GEMMs (tl.dot) uses reduced precision
(equivalent to torch "high" fp32 matmul precision). torch op matmuls stay full
fp32. The bounds carry ~1.5-2x headroom over a correct kernel but still fail a
structurally wrong one.
"""

import pytest
import torch

triton = pytest.importorskip("triton")
# ruff: noqa E402

from geometric_nn_ops_torch.triangle import _trimul as T
from geometric_nn_ops_torch.triangle import triangle_multiplicative_update

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs GPU + Triton"
)

# Weight order for gradient comparison (bias-free, square config).
_WEIGHT_KEYS = [
    "norm_in_weight",
    "norm_in_bias",
    "p_in_weight",
    "g_in_weight",
    "norm_out_weight",
    "norm_out_bias",
    "p_out_weight",
    "g_out_weight",
]

# Per-dtype acceptance bounds vs the fp64 oracle: (min cosine, max rel-fro).
# Calibrated on MI300X against the measured worst case over the output + all
# gradients (fp32 fro 8.9e-4, fp16 fro 1.0e-3, bf16 cos 1-4.9e-5 / fro 9.9e-3),
# leaving ~1.5-2x headroom; a correct kernel sits comfortably inside, a
# structurally wrong one does not.
_TOL = {
    torch.float32: (1.0 - 1e-6, 1.5e-3),
    torch.float16: (1.0 - 1e-6, 2e-3),
    torch.bfloat16: (1.0 - 1e-4, 1.5e-2),
}


def _weights(C_z, C_h, device, dtype, *, requires_grad=False, bias=True):
    g = requires_grad

    def t(*shape, scale=1.0):
        return (torch.randn(*shape, device=device, dtype=dtype) * scale).requires_grad_(
            g
        )

    def b(*shape, scale=1.0):
        return t(*shape, scale=scale) if bias else None

    return dict(
        norm_in_weight=torch.ones(C_z, device=device, dtype=dtype).requires_grad_(g),
        norm_in_bias=torch.zeros(C_z, device=device, dtype=dtype).requires_grad_(g),
        p_in_weight=t(2 * C_h, C_z, scale=0.05),
        p_in_bias=b(2 * C_h, scale=0.05),
        g_in_weight=t(2 * C_h, C_z, scale=0.05),
        g_in_bias=b(2 * C_h, scale=0.05),
        norm_out_weight=torch.ones(C_h, device=device, dtype=dtype).requires_grad_(g),
        norm_out_bias=torch.zeros(C_h, device=device, dtype=dtype).requires_grad_(g),
        p_out_weight=t(C_z, C_h, scale=0.05),
        p_out_bias=b(C_z, scale=0.05),
        g_out_weight=t(C_z, C_z, scale=0.05),
        g_out_bias=b(C_z, scale=0.05),
    )


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


def _bias_free_weights(C, device, seed):
    """Fixed fp32 bias-free, square (C_hidden == C_z == C) weights (values only)."""
    torch.manual_seed(seed)
    w = _weights(C, C, device, torch.float32, bias=False)
    return {k: w[k] for k in _WEIGHT_KEYS}


def _clone(weights, dtype, requires_grad):
    return {
        k: v.to(dtype).detach().requires_grad_(requires_grad)
        for k, v in weights.items()
    }


def _run(x, mask, direction, w, go):
    """Forward + grads wrt (x, *weights).

    Uses whatever path the current TRITON_AVAILABLE / dtype select."""
    out = triangle_multiplicative_update(x, direction=direction, mask=mask, **w)
    targets = [x] + [w[k] for k in _WEIGHT_KEYS]
    grads = torch.autograd.grad(out, targets, grad_outputs=go, retain_graph=False)
    return out, grads


def _assert_accuracy(dtype, out_t, g_t, out_o, g_o, *, tol=None, capsys_label=""):
    cos_min, fro_max = tol if tol is not None else _TOL[dtype]
    checks = [("out", out_t, out_o)] + [
        (f"grad_{k}", g_t[i], g_o[i]) for i, k in enumerate(["x"] + _WEIGHT_KEYS)
    ]
    worst_cos, worst_fro, msgs, nonfinite = 1.0, 0.0, [], []
    for name, a, b in checks:
        # Guard against NaN/Inf first: cos/fro fold with min()/max(), which
        # silently skip NaN, so a non-finite gradient would otherwise pass.
        if not torch.isfinite(a).all():
            nonfinite.append(name)
        c = _cos(a, b)
        f = _rel_fro(a, b)
        worst_cos = min(worst_cos, c)
        worst_fro = max(worst_fro, f)
        msgs.append(f"{name}: cos={c:.6f} fro={f:.3e}")
    print(f"\n[{capsys_label}] worst cos={worst_cos:.6f} worst fro={worst_fro:.3e}")
    for m in msgs:
        print("  " + m)
    assert not nonfinite, (
        f"non-finite (NaN/Inf) values in: {', '.join(nonfinite)}\n" + "\n".join(msgs)
    )
    assert worst_cos >= cos_min, (
        f"cosine {worst_cos:.6f} < {cos_min:.6f}\n" + "\n".join(msgs)
    )
    assert worst_fro <= fro_max, (
        f"rel-fro {worst_fro:.3e} > {fro_max:.1e}\n" + "\n".join(msgs)
    )


@requires_gpu
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16],
    ids=["fp32", "bf16", "fp16"],
)
def test_accuracy(direction, dtype, monkeypatch):
    """Triton forward + backward vs an fp64 torch oracle (cos + rel-fro)."""
    N, C = 32, 32
    weights_orig = _bias_free_weights(C, "cuda", seed=3)
    x0 = torch.randn(N, N, C, device="cuda") * 0.5
    mask = (torch.rand(N, N, device="cuda") > 0.1).float()

    # Triton path at the tested dtype.
    assert T.TRITON_AVAILABLE
    x_t = x0.to(dtype).detach().requires_grad_(True)
    w_t = _clone(weights_orig, dtype, requires_grad=True)
    out_ref_shape = triangle_multiplicative_update(
        x_t, direction=direction, mask=mask.to(dtype), **w_t
    )
    go = torch.randn_like(out_ref_shape)
    out_t, g_t = _run(x_t, mask.to(dtype), direction, w_t, go)

    # fp64 oracle (force the torch reference path).
    monkeypatch.setattr(T, "TRITON_AVAILABLE", False)
    x_o = x0.double().detach().requires_grad_(True)
    w_o = _clone(weights_orig, torch.float64, requires_grad=True)
    out_o, g_o = _run(x_o, mask.double(), direction, w_o, go.double())

    _assert_accuracy(dtype, out_t, g_t, out_o, g_o, capsys_label=f"{dtype}/{direction}")


@requires_gpu
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_autocast_mixed_dtype(direction, monkeypatch):
    """Regression: torch.autocast(bf16) with an fp32 pair rep + fp32 weights.

    OF3 under autocast passes things in this mix of dtypes and the kernel needs
    to handle it.
    """
    N, C = 32, 32
    weights_orig = _bias_free_weights(C, "cuda", seed=5)
    x0 = torch.randn(N, N, C, device="cuda") * 0.5
    mask = (torch.rand(N, N, device="cuda") > 0.1).float()
    assert T.TRITON_AVAILABLE

    # fp32 activation + fp32 weights, deliberately NOT pre-cast to a matched dtype.
    x_t = x0.float().detach().requires_grad_(True)
    w_t = _clone(weights_orig, torch.float32, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_t = triangle_multiplicative_update(
            x_t, direction=direction, mask=mask.float(), **w_t
        )
    assert out_t.dtype == torch.bfloat16, (
        f"expected bf16 under autocast, got {out_t.dtype}"
    )
    go = torch.randn_like(out_t)
    targets = [x_t] + [w_t[k] for k in _WEIGHT_KEYS]
    g_t = torch.autograd.grad(out_t, targets, grad_outputs=go)

    # fp64 oracle: torch reference path, no autocast, true-precision math.
    monkeypatch.setattr(T, "TRITON_AVAILABLE", False)
    x_o = x0.double().detach().requires_grad_(True)
    w_o = _clone(weights_orig, torch.float64, requires_grad=True)
    out_o, g_o = _run(x_o, mask.double(), direction, w_o, go.double())

    _assert_accuracy(
        torch.bfloat16,
        out_t,
        g_t,
        out_o,
        g_o,
        capsys_label=f"autocast-mixed/{direction}",
    )


@requires_gpu
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_large(direction, monkeypatch):
    """Large-input bf16 accuracy: norm/cosine stay meaningful where max-rel would
    be dominated by near-zero denominators."""
    N, C = 1024, 128
    dtype = torch.bfloat16
    weights_orig = _bias_free_weights(C, "cuda", seed=5)
    x0 = torch.randn(N, N, C, device="cuda") * 0.5
    mask = (torch.rand(N, N, device="cuda") > 0.1).float()

    assert T.TRITON_AVAILABLE
    x_t = x0.to(dtype).detach().requires_grad_(True)
    w_t = _clone(weights_orig, dtype, requires_grad=True)
    shape_probe = triangle_multiplicative_update(
        x_t, direction=direction, mask=mask.to(dtype), **w_t
    )
    go = torch.randn_like(shape_probe)
    out_t, g_t = _run(x_t, mask.to(dtype), direction, w_t, go)

    monkeypatch.setattr(T, "TRITON_AVAILABLE", False)
    x_o = x0.double().detach().requires_grad_(True)
    w_o = _clone(weights_orig, torch.float64, requires_grad=True)
    out_o, g_o = _run(x_o, mask.double(), direction, w_o, go.double())

    # Slightly looser rel-fro headroom for the deeper N=1024 reductions.
    _assert_accuracy(
        dtype,
        out_t,
        g_t,
        out_o,
        g_o,
        tol=(1.0 - 1e-4, 1.5e-2),
        capsys_label=f"large/{direction}",
    )


def _gpu_total_gib():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


@requires_gpu
@pytest.mark.skipif(
    _gpu_total_gib() < 48,
    reason="large-N int32-overflow check needs a >=48 GiB GPU",
)
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_large_n_no_int32_overflow(direction, monkeypatch):
    """Guards the kernel's int64 offset math at scale.

    At N=4096 the flattened row count M = N*N gives M*C = 2**31, so any offset that
    slipped to int32 overflows and produces finite-but-wrong output that an isfinite
    check misses. Compare the Triton forward against the torch reference at bf16 (an
    indexing/overflow bug diverges hugely; bf16-vs-bf16 rounding is tiny). Forward-
    only + inference_mode to stay within memory.
    """
    dtype = torch.bfloat16
    N, C = 4096, 128  # 4096**2 * 128 == 2**31 -> exercises the int32 boundary
    weights_orig = _bias_free_weights(C, "cuda", seed=13)
    x = (torch.randn(N, N, C, device="cuda") * 0.5).to(dtype)
    mask = torch.ones(N, N, device="cuda", dtype=dtype)

    assert T.TRITON_AVAILABLE
    with torch.inference_mode():
        out_t = triangle_multiplicative_update(
            x, direction=direction, mask=mask, **_clone(weights_orig, dtype, False)
        ).float()
    torch.cuda.empty_cache()

    monkeypatch.setattr(T, "TRITON_AVAILABLE", False)
    with torch.inference_mode():
        out_ref = triangle_multiplicative_update(
            x, direction=direction, mask=mask, **_clone(weights_orig, dtype, False)
        ).float()

    # Memory-frugal fp32 metrics (avoid the double-precision full-tensor copies in
    # the _cos/_rel_fro helpers, which are too large at this N).
    fro = (
        torch.linalg.vector_norm(out_t - out_ref)
        / torch.linalg.vector_norm(out_ref).clamp_min(1e-30)
    ).item()
    cos = torch.nn.functional.cosine_similarity(
        out_t.flatten(), out_ref.flatten(), dim=0
    ).item()
    print(f"\n[scale/{direction}] N={N} M*C={N * N * C} cos={cos:.6f} fro={fro:.3e}")
    assert cos >= 1.0 - 1e-3 and fro <= 3e-2, (
        f"large-N Triton diverges from the torch reference "
        f"(cos={cos:.6f}, fro={fro:.3e}) -- likely an int32 offset overflow at "
        f"M*C={N * N * C} >= 2**31"
    )


@requires_gpu
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
@pytest.mark.parametrize("C", [96, 129, 256], ids=["c96", "c129", "c256"])
def test_channel_widths(direction, C, monkeypatch):
    """Non-standar channel widths."""
    dtype = torch.bfloat16
    N = 24
    weights_orig = _bias_free_weights(C, "cuda", seed=17)
    x0 = torch.randn(N, N, C, device="cuda") * 0.5
    mask = (torch.rand(N, N, device="cuda") > 0.1).float()

    assert T.TRITON_AVAILABLE
    x_t = x0.to(dtype).detach().requires_grad_(True)
    w_t = _clone(weights_orig, dtype, requires_grad=True)
    probe = triangle_multiplicative_update(
        x_t, direction=direction, mask=mask.to(dtype), **w_t
    )
    go = torch.randn_like(probe)
    out_t, g_t = _run(x_t, mask.to(dtype), direction, w_t, go)

    monkeypatch.setattr(T, "TRITON_AVAILABLE", False)
    x_o = x0.double().detach().requires_grad_(True)
    w_o = _clone(weights_orig, torch.float64, requires_grad=True)
    out_o, g_o = _run(x_o, mask.double(), direction, w_o, go.double())

    _assert_accuracy(
        dtype,
        out_t,
        g_t,
        out_o,
        g_o,
        tol=(1.0 - 1e-4, 2e-2),
        capsys_label=f"C{C}/{direction}",
    )


@requires_gpu
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
@pytest.mark.parametrize("affine", ["zero_gamma", "small_gamma", "learned"])
def test_layernorm_affine(direction, affine, monkeypatch):
    """Nontrivial LayerNorm affine, including zero/near-zero gamma."""
    dtype = torch.bfloat16
    N, C = 24, 128
    weights_orig = _bias_free_weights(C, "cuda", seed=19)
    gin = torch.ones(C, device="cuda")
    gout = torch.ones(C, device="cuda")
    if affine == "zero_gamma":
        gin[0] = 0.0
        gin[5] = 0.0
        gout[3] = 0.0
    elif affine == "small_gamma":
        gin[0] = 1e-2
        gin[1] = 1e-3
        gin[2] = 1e-4
        gout[0] = 1e-3
    else:  # learned: all channels nonzero but nontrivial
        torch.manual_seed(23)
        gin = torch.randn(C, device="cuda") * 0.5 + 1.0
        gout = torch.randn(C, device="cuda") * 0.5 + 1.0
    weights_orig["norm_in_weight"] = gin
    weights_orig["norm_out_weight"] = gout
    weights_orig["norm_in_bias"] = torch.randn(C, device="cuda") * 0.1
    weights_orig["norm_out_bias"] = torch.randn(C, device="cuda") * 0.1

    x0 = torch.randn(N, N, C, device="cuda") * 0.5
    mask = (torch.rand(N, N, device="cuda") > 0.1).float()

    assert T.TRITON_AVAILABLE
    x_t = x0.to(dtype).detach().requires_grad_(True)
    w_t = _clone(weights_orig, dtype, requires_grad=True)
    probe = triangle_multiplicative_update(
        x_t, direction=direction, mask=mask.to(dtype), **w_t
    )
    go = torch.randn_like(probe)
    out_t, g_t = _run(x_t, mask.to(dtype), direction, w_t, go)

    monkeypatch.setattr(T, "TRITON_AVAILABLE", False)
    x_o = x0.double().detach().requires_grad_(True)
    w_o = _clone(weights_orig, torch.float64, requires_grad=True)
    out_o, g_o = _run(x_o, mask.double(), direction, w_o, go.double())

    _assert_accuracy(
        dtype,
        out_t,
        g_t,
        out_o,
        g_o,
        tol=(1.0 - 1e-4, 2e-2),
        capsys_label=f"{affine}/{direction}",
    )


def test_torch_fallback_is_differentiable():
    """On CPU (no Triton path), grads must flow through the op."""
    torch.manual_seed(2)
    N, C_z, C_h = 8, 16, 12
    x = torch.randn(N, N, C_z, dtype=torch.float32, requires_grad=True)
    mask = torch.ones(N, N, dtype=torch.float32)
    w = _weights(C_z, C_h, "cpu", torch.float32, requires_grad=True)

    out = triangle_multiplicative_update(x, direction="outgoing", mask=mask, **w)
    out.sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert w["p_in_weight"].grad is not None
    assert w["g_out_weight"].grad is not None


@requires_gpu
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
@pytest.mark.parametrize("frac", [0.5, 0.75], ids=["L050", "L075"])
@pytest.mark.parametrize("batch", [1, 2], ids=["B1", "B2"])
def test_block_padded(direction, frac, batch, monkeypatch):
    """Block-padded mask (true length L < N) vs the fp64 oracle."""
    dtype = torch.bfloat16
    N, C = 64, 128
    L = round(frac * N)
    weights_orig = _bias_free_weights(C, "cuda", seed=11)
    x0 = torch.randn(batch, N, N, C, device="cuda") * 0.5

    # Block-padded pair mask: keep the first L residues, mask the rest. For
    # batch>1 give each element a distinct L (<= the global L).
    mask = torch.zeros(batch, N, N, device="cuda")
    for bi in range(batch):
        Lb = L if bi == 0 else max(1, L - 8 * bi)
        # random keep inside the live block, so it is not trivially all-ones
        inner = (torch.rand(Lb, Lb, device="cuda") > 0.1).float()
        mask[bi, :Lb, :Lb] = inner

    assert T.TRITON_AVAILABLE
    x_t = x0.to(dtype).detach().requires_grad_(True)
    w_t = _clone(weights_orig, dtype, requires_grad=True)
    probe = triangle_multiplicative_update(
        x_t, direction=direction, mask=mask.to(dtype), **w_t
    )
    go = torch.randn_like(probe)
    out_t, g_t = _run(x_t, mask.to(dtype), direction, w_t, go)

    monkeypatch.setattr(T, "TRITON_AVAILABLE", False)
    x_o = x0.double().detach().requires_grad_(True)
    w_o = _clone(weights_orig, torch.float64, requires_grad=True)
    out_o, g_o = _run(x_o, mask.double(), direction, w_o, go.double())

    _assert_accuracy(
        dtype,
        out_t,
        g_t,
        out_o,
        g_o,
        tol=(1.0 - 1e-4, 1.5e-2),
        capsys_label=f"padded/{direction}/L{L}/B{batch}",
    )


@requires_gpu
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
@pytest.mark.parametrize(
    "shape", [(2, 48, 48, 128), (2, 3, 40, 40, 128)], ids=["B2", "5D"]
)
def test_batched(direction, shape, monkeypatch):
    """Batched (Bflat>1) and multi-leading-dim inputs vs fp64 oracle."""
    dtype = torch.bfloat16
    C = shape[-1]
    weights_orig = _bias_free_weights(C, "cuda", seed=7)
    x0 = torch.randn(*shape, device="cuda") * 0.5
    mask = (torch.rand(*shape[:-1], device="cuda") > 0.1).float()

    assert T.TRITON_AVAILABLE
    x_t = x0.to(dtype).detach().requires_grad_(True)
    w_t = _clone(weights_orig, dtype, requires_grad=True)
    probe = triangle_multiplicative_update(
        x_t, direction=direction, mask=mask.to(dtype), **w_t
    )
    go = torch.randn_like(probe)
    out_t, g_t = _run(x_t, mask.to(dtype), direction, w_t, go)

    monkeypatch.setattr(T, "TRITON_AVAILABLE", False)
    x_o = x0.double().detach().requires_grad_(True)
    w_o = _clone(weights_orig, torch.float64, requires_grad=True)
    out_o, g_o = _run(x_o, mask.double(), direction, w_o, go.double())

    _assert_accuracy(
        dtype,
        out_t,
        g_t,
        out_o,
        g_o,
        tol=(1.0 - 1e-4, 1.5e-2),
        capsys_label=f"batched{tuple(shape)}/{direction}",
    )


# ---------------------------------------------------------------------------
# Dispatch contract: inputs the Triton kernels cannot serve must fall back to
# the torch reference rather than silently changing results.
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_differentiable_mask(direction):
    """A mask that requires grad must still produce a mask gradient.

    The Triton autograd.Function treats the mask as a constant and returns no
    gradient for it, so dispatch has to route this to the torch reference.
    """
    C, N = 32, 16
    weights = _bias_free_weights(C, "cuda", seed=23)
    x0 = torch.randn(N, N, C, device="cuda") * 0.5
    mask0 = torch.rand(N, N, device="cuda")

    def run(triton_enabled, monkeypatch=None):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(T, "TRITON_AVAILABLE", triton_enabled)
            x = x0.clone().requires_grad_(True)
            m = mask0.clone().requires_grad_(True)
            w = _clone(weights, torch.float32, requires_grad=True)
            out = triangle_multiplicative_update(x, direction=direction, mask=m, **w)
            gx, gm = torch.autograd.grad(
                out, [x, m], torch.ones_like(out), allow_unused=True
            )
            return out, gx, gm

    out_t, gx_t, gm_t = run(True)
    out_r, gx_r, gm_r = run(False)

    assert gm_r is not None, "torch reference should differentiate the mask"
    assert gm_t is not None, "Triton path silently dropped the mask gradient"
    torch.testing.assert_close(out_t, out_r, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(gx_t, gx_r, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(gm_t, gm_r, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_float64():
    """fp64 has no Triton path; it must fall back rather than fail to compile.

    The fallback keeps full fp64 precision, so the result must match a pure
    torch fp64 evaluation exactly.
    """
    C, N = 32, 12
    weights = _bias_free_weights(C, "cuda", seed=29)
    x0 = torch.randn(N, N, C, device="cuda", dtype=torch.float64)
    mask = (torch.rand(N, N, device="cuda") > 0.1).double()

    assert T.TRITON_AVAILABLE
    w = _clone(weights, torch.float64, requires_grad=False)
    out = triangle_multiplicative_update(x0, direction="outgoing", mask=mask, **w)
    assert out.dtype == torch.float64
    assert torch.isfinite(out).all()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(T, "TRITON_AVAILABLE", False)
        ref = triangle_multiplicative_update(x0, direction="outgoing", mask=mask, **w)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@requires_gpu
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_broadcast_mask_matches_torch(direction):
    """A mask with fewer leading dims than x."""
    C, N, B = 32, 12, 2
    weights = _bias_free_weights(C, "cuda", seed=31)
    x0 = torch.randn(B, N, N, C, device="cuda") * 0.5
    mask = (torch.rand(N, N, device="cuda") > 0.1).float()  # shared across batch

    w = _clone(weights, torch.float32, requires_grad=False)
    assert T.TRITON_AVAILABLE
    out = triangle_multiplicative_update(x0, direction=direction, mask=mask, **w)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(T, "TRITON_AVAILABLE", False)
        ref = triangle_multiplicative_update(x0, direction=direction, mask=mask, **w)

    assert out.shape == x0.shape
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)


@requires_gpu
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_mask_none(direction):
    C, N = 32, 12
    weights = _bias_free_weights(C, "cuda", seed=37)
    x0 = torch.randn(N, N, C, device="cuda") * 0.5
    w = _clone(weights, torch.float32, requires_grad=False)

    assert T.TRITON_AVAILABLE
    out = triangle_multiplicative_update(x0, direction=direction, mask=None, **w)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(T, "TRITON_AVAILABLE", False)
        ref = triangle_multiplicative_update(x0, direction=direction, mask=None, **w)

    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
