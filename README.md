# geometric-nn-ops

Vendor-neutral GPU kernels (forward + backward) for geometry-aware neural nets,
written in Triton that run on both AMD/ROCm and NVIDIA.

Kernels are namespaced by family:

- `geometric_nn_ops_torch.triangle` — protein-folding (triangle) ops. **In scope.**
  - `triangle_attention` (forward + backward)
  - `triangle_multiplicative_update` (Triton forward; pure-torch differentiable
    fallback)

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

## Packages in this repo

This repo ships two separately-installable packages:

- **`geometric-nn-ops-torch`**: the portable Triton kernels and their direct,
  vendor-neutral API. This is the recommended dependency.
- **`hipequivariance`**: an optional zero-code-change shim that shadows NVIDIA's
  `cuequivariance_ops_torch` and routes into these kernels. It translates any
  API differences. See [hipequivariance/README.md](./hipequivariance/README.md).
  This is provided as a convenience for codebases already integrated with
  `cuequivariance`. For long term stability, we suggest using the direct
  dependency where possible.

## License

Apache-2.0. Copyright Advanced Micro Devices, Inc. See [LICENSE](./LICENSE).
