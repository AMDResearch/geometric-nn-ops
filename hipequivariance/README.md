# hipequivariance

Zero-code-change drop-in for NVIDIA's proprietary `cuequivariance_ops_torch`
kernels, backed by Triton, with first-class support for AMD ROCm GPUs.

This is the *optional convenience* half of this repo. The real compute — and the
recommended integration path — lives in the `geometric-nn-ops` package (portable
Triton kernels, fwd+bwd, AMD + NVIDIA). We recommend a direct dependency on that
API instead where possible.

## What it does

`hipequivariance` installs a Python package that shadows the
`cuequivariance_ops_torch` kernels package and exposes the symbols the open
`cuequivariance_torch` frontend imports:

- `triangle_attention` — routed to `geometric_nn_ops` (Triton fwd+bwd)
- `triangle_multiplicative_update` — routed to `geometric_nn_ops`
- `attention_pair_bias` — stub (`NotImplementedError`)
- `TriMulPrecision`
- `CUEQ_TRIATTN_FALLBACK_THRESHOLD`

Unmodified code calling
`from cuequivariance_torch.primitives.triangle import triangle_attention` then
runs on AMD GPUs with no source changes. The shim attempts to be zero-cost but
may incur a performance penalty.

## Install

```bash
pip install geometric-nn-ops        # the kernels (required)
pip install hipequivariance         # this shim (shadows cuequivariance_ops_torch)
pip install "cuequivariance==0.9.1" "cuequivariance-torch==0.9.1"   # NVIDIA's OPEN python
# do NOT install cuequivariance-ops-torch-cu12 / -cu13 (the closed binary)
```

Installing the shim is deliberately separate from the main package and it is
never pulled in implicitly. It raises at import if the closed ops wheel is
detected alongside it, and warns if the installed `cuequivariance-torch` differs
from the version it was modeled after.

## Status

| Op                               | Forward       | Backward                                        |
| -------------------------------- | ------------- | ----------------------------------------------- |
| `triangle_attention`             | yes           | yes                                             |
| `triangle_multiplicative_update` | yes           | yes (torch fallback; native Triton bwd pending) |
| `attention_pair_bias`            | stub (raises) | stub                                            |

## License

Apache-2.0. Copyright Advanced Micro Devices, Inc. See [LICENSE](../LICENSE).
