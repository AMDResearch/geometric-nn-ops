# geometric-nn-ops

Vendor-neutral GPU kernels (forward + backward) for geometry-aware neural nets,
written in Triton that run on both AMD/ROCm and NVIDIA.

Kernels are namespaced by family:

- `geometric_nn_ops_torch.triangle` — protein-folding (triangle) ops. **In
  scope.**
  - `triangle_attention` (forward + backward)
  - `triangle_multiplicative_update` (Triton forward; pure-torch differentiable
    fallback)
  - `triangle_multiplicative_update_` (in-place, inference only)
  - `attention_pair_bias` (pure torch; no Triton kernel yet)

## Usage

```python
from geometric_nn_ops_torch.triangle import triangle_attention

o = triangle_attention(q, k, v, bias=pair_bias, mask=mask, scale=scale)
```

See the docstring for the appropriate tensor layouts and other options.

```python
from geometric_nn_ops_torch.triangle import triangle_multiplicative_update

z = triangle_multiplicative_update(x, direction="outgoing", mask=mask, **weights)
```

`triangle_multiplicative_update` matches the cuequivariance signature
(concatenated `[a; b]` `p_in`/`g_in` projections). It runs the Triton fast path
for inference and auto-falls back to a pure-torch differentiable implementation
when gradients are needed or no GPU/Triton is available.

For inference on long sequences, `triangle_multiplicative_update_` adds the
update to `x` in place and can compute it in chunks to limit memory usage:

```python
with torch.inference_mode():
    triangle_multiplicative_update_(x, "outgoing", mask, **weights, chunk_size=256)
```

All ops take a boolean `mask` (`True` = keep) or `None`, and raise on any other
dtype. The cuequivariance shim accepts cuequivariance's 0/1 masks and casts them
to bool, erroring if the mask has `requires_grad` set.

## fp32 matmul precision

For fp32 inputs, these ops read `torch.backends.cuda.matmul.fp32_precision`. In
the Triton kernels, `"tf32"` resolves in the same way as
`torch.set_float32_matmul_precision("high")` to whatever reduced-precision fp32
matmul the target has with at least TF32 precision — TF32 tensor cores on NVIDIA
Ampere and above, XF32 MFMA on AMD CDNA3, bf16 emulation elsewhere — and to full
fp32 where there is none.

## Packages in this repo

This repo ships two separately-installable packages:

- **`geometric-nn-ops-torch`**: the portable Triton kernels and their direct,
  vendor-neutral API. This is the recommended dependency.
- **`hipequivariance-ops-torch`**: an optional zero-code-change shim that
  implements the API of NVIDIA's `cuequivariance_ops_torch` under the same
  module name routes into these kernels. It translates any API differences. See
  [hipequivariance_ops_torch/README.md](./hipequivariance_ops_torch/README.md).
  This is provided as a convenience for codebases already integrated with
  `cuequivariance`. For long term stability, we suggest using the direct
  dependency where possible.

## License

Apache-2.0. Copyright Advanced Micro Devices, Inc. See [LICENSE](./LICENSE).
