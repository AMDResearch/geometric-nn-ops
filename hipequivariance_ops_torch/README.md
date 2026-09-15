# hipequivariance-ops-torch

Zero-code-change drop-in for NVIDIA's `cuequivariance_ops_torch` kernels, backed
by Triton, with first-class support for AMD ROCm GPUs.

This is an *optional convenience* package. The real compute — and the
recommended integration path — lives in the `geometric-nn-ops-torch` package. We
recommend a direct dependency on that API instead where possible.

## What it does

`hipequivariance-ops-torch` installs a Python package that implements the
`cuequivariance_ops_torch` API under the same name module name and provides the
symbols `cuequivariance_torch` imports:

- `triangle_attention` — routed to `geometric-nn-ops`
- `triangle_multiplicative_update` — routed to `geometric-nn-ops`
- `attention_pair_bias` — routed to `geometric-nn-ops`
- `TriMulPrecision`
- `CUEQ_TRIATTN_FALLBACK_THRESHOLD` (deep import from
  `cuequivariance_ops_torch.triangle_attention`)
- `CUEQ_ATTENTION_PAIR_BIAS_FALLBACK_THRESHOLD` (deep import from
  `cuequivariance_ops_torch.attention_pair_bias_torch`)

Unmodified code calling e.g.
`from cuequivariance_torch.primitives.triangle import triangle_attention` then
runs on AMD GPUs with no source changes. The shim attempts to be zero-cost but
may incur a performance penalty.

### Precision

The underlying ops take their fp32 matmul precision from
`torch.backends.cuda.matmul.fp32_precision` and have no per-call override. The
ops sometimes make use of torch operations or fallbacks where such a setting
cannot be honored because it relies on a Torch global with no scoped option (a
custom context manager toggling the global would not be thread safe). This
conflicts with the cuequivariance `triangle_multiplicative_update`'s `precision`
argument, which presents an API indicating that the user can select their
desired precision. The actual behavior there is inconsistent however, with the
operation sometimes not honoring the setting: likely for the same reason that
some paths use Torch directly. Rather than providing an inconsistent API or
making guarantees we cannot honor, the shim checks the precision setting for
compatibility with the Torch global. A request for more precision than the torch
setting (i.e. `precision="ieee"` when the torch global is "tf32") raises, as the
required precision cannot be satisifed, creating a correctness issue. A request
for less precision (i.e. `precision="tf32"` when the Torch global is "ieee")
warns, as falling back to full precision is a performance issue but not a
correctness one (and matches the behavior on hardware with no reduced-precision
fast path).

## Install

Packages are not yet published on PyPI. Installing the shim is deliberately
separate from the main package and it is never pulled in implicitly. It raises
at import if a `cuequivariance-ops-torch-*` wheel is detected as installed, and
warns if the installed `cuequivariance-torch` differs from the version it was
designed for.

### Today

You can check out the source and use editable installs or install directly from
GitHub.

```bash
# Install PyTorch and Triton for your environment, e.g. following https://pytorch.org/get-started/locally/
git clone https://github.com/AMDResearch/geometric-nn-ops.git
cd geometric-nn-ops/
pip install --no-deps -e .  # the kernels (required)
pip install --no-deps -e hipequivariance_ops_torch  # this shim (provides cuequivariance_ops_torch module)
pip install "cuequivariance==0.9.1" "cuequivariance-torch==0.9.1"   # NVIDIA's frontend python
# do NOT install cuequivariance-ops-torch-cu12 / -cu13
```

### Once Packages Are Published on PyPI

```bash
pip install geometric-nn-ops-torch  # the kernels (required)
pip install hipequivariance-ops-torch  # this shim (provides cuequivariance_ops_torch module)
pip install "cuequivariance==0.9.1" "cuequivariance-torch==0.9.1"   # NVIDIA's frontend python
# do NOT install cuequivariance-ops-torch-cu12 / -cu13
```

## Status

| Op                               | Forward                                | Backward                                        |
| -------------------------------- | -------------------------------------- | ----------------------------------------------- |
| `triangle_attention`             | yes                                    | yes                                             |
| `triangle_multiplicative_update` | yes                                    | yes (torch fallback; native Triton bwd pending) |
| `attention_pair_bias`            | yes (pure torch; no Triton kernel yet) | yes (pure torch; no Triton kernel yet)          |

## License

Apache-2.0. Copyright Advanced Micro Devices, Inc. See [LICENSE](../LICENSE).
