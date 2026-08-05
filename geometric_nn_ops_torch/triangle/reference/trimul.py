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
    outgoing,
    m,
    c_hidden,
    eps,
    norm_in_weight,
    norm_in_bias,
    p_in_weight,
    p_in_bias,
    g_in_weight,
    g_in_bias,
    norm_out_weight,
    norm_out_bias,
    p_out_weight,
    p_out_bias,
    g_out_weight,
    g_out_bias,
):
    c_z = x.shape[-1]
    z = F.layer_norm(x, (c_z,), norm_in_weight, norm_in_bias, eps)
    p = F.linear(z, p_in_weight, p_in_bias) * torch.sigmoid(
        F.linear(z, g_in_weight, g_in_bias)
    )
    if m is not None:
        p = p * m
    a = p[..., :c_hidden]
    b = p[..., c_hidden:]
    o = _combine_projections(a, b, outgoing)
    o = F.layer_norm(o, (c_hidden,), norm_out_weight, norm_out_bias, eps)
    o = F.linear(o, p_out_weight, p_out_bias)
    g = torch.sigmoid(F.linear(z, g_out_weight, g_out_bias))
    return o * g
