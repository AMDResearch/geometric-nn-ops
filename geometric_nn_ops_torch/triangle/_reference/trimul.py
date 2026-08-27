# Copyright 2026 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch reference math for triangle_multiplicative_update.

Adapted from OpenFold-3.
"""

import torch
import torch.nn.functional as F

DIRECTIONS = ("outgoing", "incoming")


def is_outgoing(direction: str) -> bool:
    """Validate ``direction`` and reduce it to the combine's orientation."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    return direction == "outgoing"


def _permute_final_dims(t: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    """Permute the last ``len(inds)`` dims of ``t`` (mirrors OF3's helper)."""
    zero_index = -1 * len(inds)
    first_inds = list(range(len(t.shape[:zero_index])))
    return t.permute(first_inds + [zero_index + i for i in inds])


def _combine_projections(
    a: torch.Tensor, b: torch.Tensor, outgoing: bool
) -> torch.Tensor:
    if outgoing:
        a = _permute_final_dims(a, (2, 0, 1))
        b = _permute_final_dims(b, (2, 1, 0))
    else:
        a = _permute_final_dims(a, (2, 1, 0))
        b = _permute_final_dims(b, (2, 0, 1))
    p = torch.einsum("...ij,...jk->...ik", a, b)
    return _permute_final_dims(p, (1, 2, 0))


def trimul_torch(
    x,
    direction="outgoing",
    mask=None,
    *,
    eps=1e-5,
    norm_in_weight=None,
    norm_in_bias=None,
    p_in_weight=None,
    p_in_bias=None,
    g_in_weight=None,
    g_in_bias=None,
    norm_out_weight=None,
    norm_out_bias=None,
    p_out_weight=None,
    p_out_bias=None,
    g_out_weight=None,
    g_out_bias=None,
):
    """Triangle multiplicative update in stock PyTorch.

    ``x`` is ``[*, N, N, C_z]`` and ``mask`` ``[*, N, N]``. ``p_in_weight`` and
    ``g_in_weight`` are the concatenated ``[a; b]`` projections, so the hidden
    width follows from their row count.
    """
    outgoing = is_outgoing(direction)
    c_z = x.shape[-1]
    c_hidden = p_in_weight.shape[0] // 2
    z = F.layer_norm(x, (c_z,), norm_in_weight, norm_in_bias, eps)
    p = F.linear(z, p_in_weight, p_in_bias) * torch.sigmoid(
        F.linear(z, g_in_weight, g_in_bias)
    )
    if mask is not None:
        p = p * mask.unsqueeze(-1)
    a = p[..., :c_hidden]
    b = p[..., c_hidden:]
    o = _combine_projections(a, b, outgoing)
    o = F.layer_norm(o, (c_hidden,), norm_out_weight, norm_out_bias, eps)
    o = F.linear(o, p_out_weight, p_out_bias)
    g = torch.sigmoid(F.linear(z, g_out_weight, g_out_bias))
    return o * g
