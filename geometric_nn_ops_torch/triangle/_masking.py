# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Turning a key mask into an additive bias on the softmax logits.

Shared by the ops that mask keys out of a softmax, so they all push a masked
key down by the same amount and fold a boolean mask the same way.
"""

import torch


def mask_val(dtype: torch.dtype) -> float:
    """How far a masked key's logit is pushed below zero, for ``dtype``.

    ``1e9`` for fp32 and bf16. fp16 cannot represent it and would get ``inf`` --
    a fully masked key *tile* then takes ``-inf`` as its running maximum and the
    online softmax evaluates ``-inf - -inf``, giving NaNs even for rows that are
    not fully masked -- so it takes half the dtype maximum instead, which still
    squashes any reasonable logit. fp64 needs a larger value than fp32 to behave
    the same way: at ``1e9`` its ulp is far below ``log(SEQ_LEN)``, so a fully
    masked row would keep its logsumexp term where fp32 loses it, and the two
    would disagree.
    """
    if dtype == torch.float64:
        return 1e18
    elif dtype == torch.float32 or dtype == torch.bfloat16:
        return 1e9
    elif dtype == torch.float16:
        return torch.finfo(dtype).max / 2
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def masked_key_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Boolean mask -> additive bias, ``0`` where kept and ``-mask_val`` where not."""
    # Note that we deliberately spell this the same way openfold-3 has
    # historically, so that for now we can get bitwise identical results. A
    # masked fill or `(~mask) * -mask_val` would be more natural, but would yield
    # -0.0 on kept entries rather than +0.0, which is enough to shift bf16 and
    # fp16 results.
    return mask_val(dtype) * (mask.to(torch.float32) - 1)
