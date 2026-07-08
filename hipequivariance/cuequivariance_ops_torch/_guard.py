# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Import-time safety checks for the cuequivariance_ops_torch shim.

This package shadows NVIDIA's closed binary ``cuequivariance_ops_torch``. Two
hazards we guard against:

1. The real closed ops wheel is also installed -> two packages claim the same
   import name; whichever wins is nondeterministic. Fail loudly instead.
2. The installed ``cuequivariance-torch`` frontend is a version we are not pinned
   to -> its private import contract may have drifted. Warn (don't fail).
"""

import warnings
from importlib import metadata

# Closed binary ops wheels we must NOT coexist with.
_REAL_OPS_DISTS = (
    "cuequivariance-ops-torch",
    "cuequivariance-ops-torch-cu12",
    "cuequivariance-ops-torch-cu13",
)

# The cuequivariance-torch frontend version this shim's layout is pinned to.
_PINNED_CUEQ_TORCH = "0.9.1"


def _installed_distributions() -> set[str]:
    names = set()
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            names.add(name.lower())
    return names


def check() -> None:
    installed = _installed_distributions()

    clashes = [d for d in _REAL_OPS_DISTS if d in installed]
    if clashes:
        raise RuntimeError(
            "hipequivariance: both the real closed cuequivariance ops wheel "
            f"({', '.join(clashes)}) and the hipequivariance shim are installed. "
            "They both provide the import package 'cuequivariance_ops_torch', so "
            "which one loads is nondeterministic. Uninstall the closed wheel "
            "(`pip uninstall " + " ".join(clashes) + "`) to use the AMD shim."
        )

    try:
        cueq_torch_version = metadata.version("cuequivariance-torch")
    except metadata.PackageNotFoundError:
        cueq_torch_version = None
    if cueq_torch_version is not None and cueq_torch_version != _PINNED_CUEQ_TORCH:
        warnings.warn(
            "hipequivariance: the cuequivariance_ops_torch shim is pinned to "
            f"cuequivariance-torch=={_PINNED_CUEQ_TORCH}, but "
            f"{cueq_torch_version} is installed. The frontend's private import "
            "contract may have drifted; routing may break.",
            RuntimeWarning,
            stacklevel=2,
        )
