# hipequivariance-ops-torch

Zero-code-change drop-in for NVIDIA's `cuequivariance_ops_torch` kernels, backed
by Triton, with first-class support for AMD ROCm GPUs.

This is an *optional convenience* package. The real compute — and the
recommended integration path — lives in the
[`geometric-nn-ops-torch`](https://pypi.org/project/geometric-nn-ops-torch/)
package. We recommend a direct dependency on that API instead where possible.

## What it does

`hipequivariance-ops-torch` installs a Python package that implements the
`cuequivariance_ops_torch` API under the same module name and provides the
symbols `cuequivariance_torch` imports:

- `triangle_attention` — routed to `geometric-nn-ops`
- `triangle_multiplicative_update` — routed to `geometric-nn-ops`
- `attention_pair_bias` — routed to `geometric-nn-ops`
- `TriMulPrecision`
- `CUEQ_TRIATTN_FALLBACK_THRESHOLD` (deep import from
  `cuequivariance_ops_torch.triangle_attention`)
- `CUEQ_TRIMUL_FALLBACK_THRESHOLD` (deep import from
  `cuequivariance_ops_torch.triangle_multiplicative_update`)
- `CUEQ_ATTENTION_PAIR_BIAS_FALLBACK_THRESHOLD` (deep import from
  `cuequivariance_ops_torch.attention_pair_bias_torch`)

Unmodified code calling e.g.
`from cuequivariance_torch.primitives.triangle import triangle_attention` then
runs on AMD GPUs with no source changes. The shim attempts to be zero-cost but
may incur a performance penalty.

### Precision

The underlying ops take their fp32 matmul precision from
`torch.backends.cuda.matmul.fp32_precision` and have no per-call override. The
ops sometimes make use of torch operations which can only use the unscoped torch
global. This conflicts with the cuequivariance
`triangle_multiplicative_update`'s `precision` argument, which presents an API
indicating that the user can select their desired precision. cuequivariance does
not consistently honor this API, however. Rather than inconsistent behavior, the
shim checks the precision setting for compatibility with the torch global:

- requesting more precision than the torch setting (e.g. `precision="ieee"` when
  the torch global is "tf32") raises, as the required precision cannot be
  satisfied, creating a correctness issue.
- requesting less (e.g. `precision="tf32"` when the torch global is "ieee")
  warns, as falling back to full precision is a performance issue but not a
  correctness one.

## Install

Installing the shim is an optional step separate from the main package, which
does not depend on it. The shim raises at import if a
`cuequivariance-ops-torch-*` wheel is detected as installed, and warns if the
installed `cuequivariance-torch` differs from the version it was designed for.

Install PyTorch for your platform first, following
[the PyTorch instructions](https://pytorch.org/get-started/locally/). Otherwise
pip installs the default PyTorch build from PyPI, which on Linux targets NVIDIA
CUDA rather than AMD ROCm.

```bash
pip install hipequivariance-ops-torch  # this shim, pulls in the kernels package as a dep
pip install "cuequivariance==0.9.1" "cuequivariance-torch==0.9.1"  # NVIDIA's frontend Python
# do NOT install cuequivariance-ops-torch-cu12 / -cu13
```

For an editable install from source instead:

```bash
git clone https://github.com/AMDResearch/geometric-nn-ops.git
cd geometric-nn-ops/
pip install -e .  # the kernels
pip install -e hipequivariance_ops_torch  # this shim
```

## Status

| Op                               | Forward                                | Backward                                        |
| -------------------------------- | -------------------------------------- | ----------------------------------------------- |
| `triangle_attention`             | yes                                    | yes                                             |
| `triangle_multiplicative_update` | yes                                    | yes (torch fallback; native Triton bwd pending) |
| `attention_pair_bias`            | yes                                    | yes                                             |

## License

Apache-2.0. Copyright Advanced Micro Devices, Inc. See
[LICENSE](https://github.com/AMDResearch/geometric-nn-ops/blob/main/hipequivariance_ops_torch/LICENSE).
