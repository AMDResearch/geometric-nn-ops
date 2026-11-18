# geometric-nn-ops

Vendor-neutral GPU kernels (forward + backward) for geometry-aware neural nets,
written in Triton that run on both AMD/ROCm and NVIDIA.

Kernels are namespaced by family:

- `geometric_nn_ops_torch.triangle` — protein-folding (triangle) ops.
  - `triangle_attention` (forward + backward)
  - `triangle_multiplicative_update` (Triton forward; pure-torch differentiable
    fallback)
  - `attention_pair_bias` (pure torch; no Triton kernel yet)

## Install

Install PyTorch for your platform first, following
[the PyTorch instructions](https://pytorch.org/get-started/locally/). Otherwise
pip installs the default PyTorch build from PyPI, which on Linux targets NVIDIA
CUDA rather than AMD ROCm.

```bash
pip install geometric-nn-ops-torch
```

For an editable install from source instead:

```bash
git clone https://github.com/AMDResearch/geometric-nn-ops.git
cd geometric-nn-ops/
pip install -e .  # the kernels
```

## Usage

```python
from geometric_nn_ops_torch.triangle import triangle_attention

o = triangle_attention(q, k, v, bias=pair_bias, mask=mask, scale=scale)
```

See the docstring for the appropriate tensor layouts and other options.

For inference with shared pair bias and mask across row chunks,
`PreparedTriangleAttention` snapshots those inputs once and optionally fuses
multiplication by an already sigmoid-transformed gate. Head count is inferred
from the bias; head dimension is inferred from Q/K/V. The opt-in API accepts
FP32 IEEE inference on gfx950, batch size one, head dimensions from 1 to 128,
up to 32 rows per call, and sequence lengths up to 3488. Recreate the prepared
object after changing the shared bias or mask. The functional API is unchanged.

Prepared-attention tests sweep head counts and dimensions, including the
template and pair-attention configurations from the public
[OpenFold3 model configuration](https://github.com/aqlaboratory/openfold-3/blob/b600fbf74720b3259d59b93ffa62a07516099d06/openfold3/projects/of3_all_atom/config/model_config.py),
alongside synthetic shape and boundary cases. These are operation-level tests,
not end-to-end OpenFold3 validation.

```python
from geometric_nn_ops_torch.triangle import triangle_multiplicative_update

z = triangle_multiplicative_update(x, direction="outgoing", mask=mask, **weights)
```

`triangle_multiplicative_update` matches the cuequivariance signature
(concatenated `[a; b]` `p_in`/`g_in` projections). It runs the Triton fast path
for inference and auto-falls back to a pure-torch differentiable implementation
when gradients are needed or no GPU/Triton is available.

## fp32 matmul precision

For fp32 inputs, these ops read `torch.backends.cuda.matmul.fp32_precision`. In
the Triton kernels, `"tf32"` resolves in the same way as
`torch.set_float32_matmul_precision("high")` to whatever reduced-precision fp32
matmul the target has with at least TF32 precision — TF32 tensor cores on NVIDIA
Ampere and above, XF32 MFMA on AMD CDNA3, bf16 emulation elsewhere — and to full
fp32 where there is none.

## Packages

The [geometric-nn-ops](https://github.com/AMDResearch/geometric-nn-ops)
repository publishes two separately installable packages:

- [`geometric-nn-ops-torch`](https://pypi.org/project/geometric-nn-ops-torch/):
  the portable Triton kernels and their direct, vendor-neutral API. This is the
  recommended dependency.
- [`hipequivariance-ops-torch`](https://pypi.org/project/hipequivariance-ops-torch/):
  an optional zero-code-change shim that implements the API of NVIDIA's
  `cuequivariance_ops_torch` under the same module name and routes into these
  kernels. It translates any API differences. This is provided as a convenience
  for codebases already integrated with `cuequivariance`. For long term
  stability, we suggest using the direct dependency where possible.

## License

Apache-2.0. Copyright Advanced Micro Devices, Inc. See
[LICENSE](https://github.com/AMDResearch/geometric-nn-ops/blob/main/LICENSE).
