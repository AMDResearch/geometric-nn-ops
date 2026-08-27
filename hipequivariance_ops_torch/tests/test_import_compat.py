# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Import-compatibility checks for the shim -- no GPU required.

Asserts that every dotted path cuequivariance_torch deep-import from the ops
layer actually resolves against our shim.
"""

# The whole point of this test is to test imports, so disable I001 (unsorted-imports).
# ruff: noqa: I001


def test_shim_paths_resolve():
    # Exactly the imports cuequivariance_torch (v0.9.1) performs against the ops layer.
    from cuequivariance_ops_torch import (
        TriMulPrecision,
        triangle_attention,
        triangle_multiplicative_update,
    )
    from cuequivariance_ops_torch.attention_pair_bias_torch import (
        attention_pair_bias,
    )

    # OpenFold3 deep-imports this constant.
    from cuequivariance_ops_torch.triangle_attention import (
        CUEQ_TRIATTN_FALLBACK_THRESHOLD,
    )

    # Other constants are also presumably deep-imported although no one seems to
    # ever actually import them
    from cuequivariance_ops_torch.attention_pair_bias_torch import (
        CUEQ_ATTENTION_PAIR_BIAS_FALLBACK_THRESHOLD,
    )

    assert callable(triangle_attention)
    assert callable(triangle_multiplicative_update)
    assert callable(attention_pair_bias)
    assert isinstance(CUEQ_TRIATTN_FALLBACK_THRESHOLD, int)
    assert isinstance(CUEQ_ATTENTION_PAIR_BIAS_FALLBACK_THRESHOLD, int)
    assert int(TriMulPrecision.IEEE) == 3
