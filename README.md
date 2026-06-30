# geometric-nn-ops

Vendor-neutral GPU kernels (forward + backward) for geometry-aware neural nets,
written in Triton with an `is_hip()` branch so they run on both AMD/ROCm and NVIDIA.

Kernels are namespaced by family:

- `geometric_nn_ops.triangle` — protein-folding (triangle) ops. **In scope.**
  - `triangle_attention` (forward + backward)
- `geometric_nn_ops.tensor_product` — equivariant tensor products. *Reserved / deferred.*

## Usage

```python
from geometric_nn_ops.triangle import triangle_attention

o = triangle_attention(q, k, v, bias=pair_bias, mask=mask, scale=scale)
```

`triangle_attention(q, k, v, bias, mask, scale=None)` matches the call signature of
`cuequivariance_torch.primitives.triangle.triangle_attention`.

## License

Apache-2.0. Copyright Advanced Micro Devices, Inc.
