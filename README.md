# geometric-nn-ops

Vendor-neutral GPU kernels (forward + backward) for geometry-aware neural nets,
written in Triton with an `is_hip()` branch so they run on both AMD/ROCm and NVIDIA.

Kernels are namespaced by family:

- `geometric_nn_ops.triangle` — protein-folding (triangle) ops. **In scope.**
  - `triangle_attention` (forward + backward)
  - `triangle_multiplicative_update` (Triton forward; pure-torch differentiable fallback)
- `geometric_nn_ops.tensor_product` — equivariant tensor products. *Reserved / deferred.*

## Usage

```python
from geometric_nn_ops.triangle import triangle_attention

o = triangle_attention(q, k, v, bias=pair_bias, mask=mask, scale=scale)
```

`triangle_attention(q, k, v, bias, mask, scale=None)` matches the call signature of
`cuequivariance_torch.primitives.triangle.triangle_attention`.

```python
from geometric_nn_ops.triangle import triangle_multiplicative_update

z = triangle_multiplicative_update(x, direction="outgoing", mask=mask, **weights)
```

`triangle_multiplicative_update` matches the cuequivariance signature (concatenated
`[a; b]` `p_in`/`g_in` projections). It runs the Triton fast path for inference and
auto-falls back to a pure-torch differentiable implementation when gradients are
needed or no GPU/Triton is available.

## License

Apache-2.0. Copyright Advanced Micro Devices, Inc.
