# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the triangle attention benchmark scripts.

All implementations take the op's signature: q/k/v ``[B, N_seq, N_res, H, D]``,
a bias ``[B, 1, H, N_res, N_res]`` shared across N_seq, a boolean key mask
``[B, N_seq, 1, 1, N_res]``, and a scale. The timings of ``sdpa`` and ``flex``
include converting to and from their layouts.
"""

import argparse
import functools

import _bench_common as bench_common
import torch
import torch.nn.functional as F
from _bench_common import UniqueValues, positive_int
from torch.nn.attention import SDPBackend, sdpa_kernel

from geometric_nn_ops_torch.triangle import triangle_attention
from geometric_nn_ops_torch.triangle._kernels.triangle_attention import (
    triangle_attention_triton,
)
from geometric_nn_ops_torch.triangle._masking import masked_key_bias
from geometric_nn_ops_torch.triangle._reference.triangle_attention import (
    triangle_attention_torch,
)

# `public` is the public API, including its input handling. `sdpa` and `flex`
# are alternatives to our op that a caller could use.
BACKENDS = ("reference", "triton", "public", "sdpa", "flex")

DEFAULT_BACKENDS = ("reference", "triton", "sdpa")

DEFAULT_BASELINES = ("reference", "sdpa")

# A large recompile limit ensures flex doesn't fall back to eager. See `compile_flex`.
FLEX_RECOMPILE_LIMIT = 1024


def sdpa_operands(q, k, v, bias, mask):
    """Lay out q, k, v and ``attn_mask`` for ``scaled_dot_product_attention``.

    Heads fold into SDPA's batch axis and N_seq becomes its head axis. The bias
    is then ``[B*H, 1, N_res, N_res]`` and broadcasts over N_seq without being
    expanded. SDPA takes a single additive mask, and a key mask varies along
    N_seq, so adding it to the bias materializes ``[B*H, N_seq, N_res, N_res]``.
    q, k and v are folded as views where their strides allow, as they would be
    for a caller rearranging them.
    """
    batch, n_seq, n_res, heads, dim = q.shape

    def fold(t):
        return t.permute(0, 3, 1, 2, 4).reshape(batch * heads, n_seq, n_res, dim)

    # Make both terms contiguous while they're still small. The fused backends
    # need the mask's last dim unit-strided, and otherwise fall back to the
    # math backend. The sum takes its layout from its operands.
    attn_mask = None
    if bias is not None:
        attn_mask = bias.reshape(batch * heads, 1, n_res, n_res).contiguous()
    if mask is not None:
        key_bias = masked_key_bias(mask, q.dtype).to(q.dtype).contiguous()
        key_bias = (
            key_bias.reshape(batch, 1, n_seq, 1, n_res)
            .expand(batch, heads, n_seq, 1, n_res)
            .reshape(batch * heads, n_seq, 1, n_res)
        )
        attn_mask = key_bias if attn_mask is None else attn_mask + key_bias
    return fold(q), fold(k), fold(v), attn_mask


def sdpa_triangle_attention(q, k, v, bias=None, mask=None, scale=None):
    """Triangle attention through ``F.scaled_dot_product_attention``.

    Pinned to the memory-efficient backend, the only fused backend that takes
    an additive mask. Inputs it cannot run then raise "No available kernel"
    rather than silently falling back to the math backend, which materializes
    the scores.
    """
    batch, n_seq, n_res, heads, dim = q.shape
    q4, k4, v4, attn_mask = sdpa_operands(q, k, v, bias, mask)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        o = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=attn_mask, scale=scale)
    return o.reshape(batch, heads, n_seq, n_res, dim).permute(0, 2, 3, 1, 4)


def sdpa_backend(q, k, v, bias, mask, scale) -> str:
    """The SDPA backend torch would pick for these inputs if it weren't pinned.

    Calls the same selection function as ``scaled_dot_product_attention``, so
    requires_grad, dtype and head dim are taken into account, and nothing runs.
    Since :func:`sdpa_triangle_attention` pins the backend, this shows whether
    pinning changes anything.
    """
    q4, k4, v4, attn_mask = sdpa_operands(q, k, v, bias, mask)
    choice = torch._fused_sdp_choice(q4, k4, v4, attn_mask, 0.0, False, scale=scale)
    return SDPBackend(choice).name


def sdpa_label(found: dict[str, str]) -> str | None:
    """Format the SDPA backend chosen in each mode, as one name if they agree."""
    if len(set(found.values())) <= 1:
        return next(iter(found.values()), None)
    return " ".join(f"{mode}={name}" for mode, name in found.items())


def flex_triangle_attention(
    q, k, v, bias=None, mask=None, scale=None, *, compiled_flex
):
    """Triangle attention through ``flex_attention``.

    (B, N_seq) fold into flex's batch axis. The score mod indexes the bias with
    ``b // N_seq``, so it stays ``[B, H, N_res, N_res]`` and isn't expanded over
    N_seq. Its gradient flows through the captured tensor.
    """
    batch, n_seq, n_res, heads, dim = q.shape

    def fold(t):
        return t.reshape(batch * n_seq, n_res, heads, dim).transpose(1, 2)

    terms = []
    if bias is not None:
        # Copied: flex has returned wrong output for a non-contiguous bias.
        bias4 = bias.reshape(batch, heads, n_res, n_res).contiguous()
        terms.append(lambda b, h, q_idx, kv_idx: bias4[b // n_seq, h, q_idx, kv_idx])
    if mask is not None:
        key_bias = masked_key_bias(mask, q.dtype).to(q.dtype)
        key_bias = key_bias.reshape(batch * n_seq, n_res)
        terms.append(lambda b, h, q_idx, kv_idx: key_bias[b, kv_idx])

    def score_mod(score, b, h, q_idx, kv_idx):
        for term in terms:
            score = score + term(b, h, q_idx, kv_idx)
        return score

    o = compiled_flex(fold(q), fold(k), fold(v), score_mod=score_mod, scale=scale)
    return o.transpose(1, 2).reshape(batch, n_seq, n_res, heads, dim)


def compile_flex(dynamic: bool):
    """Compile ``flex_attention`` ensuring no eager fallback.

    Each new shape compiles a new flex graph. Past Dynamo's recompile limit,
    flex runs eagerly and materializes the scores, with only a log message.
    Raise the limit above any expected sweep size and error if we hit it.

    torch 2.7 renamed these settings to ``*recompile_limit*`` but still accepts
    the old names. Switch once the flex backend no longer supports 2.6.
    """
    config = torch._dynamo.config
    if not hasattr(config, "fail_on_cache_limit_hit"):
        raise RuntimeError(
            f"the flex backend needs torch 2.6 or later, found {torch.__version__}"
        )
    # Imported here because older torch versions don't have it.
    from torch._inductor import config as inductor_config
    from torch.nn.attention.flex_attention import flex_attention

    config.cache_size_limit = FLEX_RECOMPILE_LIMIT
    config.accumulated_cache_size_limit = FLEX_RECOMPILE_LIMIT
    config.fail_on_cache_limit_hit = True
    # A graph loaded from Inductor's FX graph cache keeps a small allocation
    # alive, which would fail the allocator check after each case.
    inductor_config.fx_graph_cache = False
    inductor_config.fx_graph_remote_cache = False
    return torch.compile(flex_attention, dynamic=dynamic)


def make_implementations(args) -> dict:
    """Make the implementations named in ``args.backends``. This compiles flex,
    so call it once.

    ``--dynamic`` and ``--use-exp2`` are options of our op, so they apply only
    to ``triton`` and ``public``.
    """
    options = {"dynamic": args.dynamic, "use_exp2": args.use_exp2}
    implementations = {
        "reference": triangle_attention_torch,
        "triton": functools.partial(triangle_attention_triton, **options),
        "public": functools.partial(triangle_attention, **options),
        "sdpa": sdpa_triangle_attention,
    }
    if "flex" in args.backends:
        implementations["flex"] = functools.partial(
            flex_triangle_attention, compiled_flex=compile_flex(args.flex_dynamic)
        )
    return {name: implementations[name] for name in args.backends}


def add_shape_arguments(parser, *, sizes, head_dims, n_seq=True):
    """Add the arguments for the attention shape.

    ``n_seq`` adds ``--n-seq`` and ``--n-seq-max``, for inputs that aren't
    square.
    """
    parser.add_argument(
        "--sizes",
        type=positive_int,
        nargs="+",
        default=list(sizes),
        action=UniqueValues,
        help="N_res",
    )
    if n_seq:
        parser.add_argument(
            "--n-seq",
            type=positive_int,
            default=None,
            help="N_seq. Default is N_seq = N_res",
        )
        parser.add_argument(
            "--n-seq-max",
            type=positive_int,
            default=None,
            help="cap N_seq here, as openfold-3 does by chunking N_seq in inference",
        )
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--heads", type=positive_int, default=4)
    parser.add_argument(
        "--head-dims",
        type=positive_int,
        nargs="+",
        default=list(head_dims),
        action=UniqueValues,
        help="D",
    )
    parser.add_argument(
        "--mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply a random boolean key mask",
    )
    return parser


def add_benchmark_arguments(parser, *, measures):
    """Add the arguments every benchmark takes (see ``run_benchmark``), and the
    options of the implementations."""
    bench_common.add_benchmark_arguments(
        parser,
        measures=measures,
        backends=BACKENDS,
        default_backends=DEFAULT_BACKENDS,
        baselines=DEFAULT_BASELINES,
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="our op's variable-shape entry point, one compile for all sizes",
    )
    parser.add_argument(
        "--use-exp2", action="store_true", help="our op's base-2 softmax"
    )
    parser.add_argument(
        "--flex-dynamic",
        action="store_true",
        help="compile flex with dynamic=True (one graph for all sizes) rather "
        "than one static graph per shape",
    )
    return parser


def checked_row_count(requested: int, n_rows: int) -> int | str:
    """The number of rows ``checked_rows(n_rows, requested)`` returns, or
    ``"all"``, without sampling them. ``requested`` 0 means all."""
    return requested if requested and requested < n_rows else "all"


def oracle_rows(text) -> int:
    """argparse type for ``--oracle-rows``. 0 means all rows. Otherwise it must
    be at least 2, since the first and last rows are always checked."""
    value = int(text)
    if value != 0 and value < 2:
        raise argparse.ArgumentTypeError(f"must be 0 or at least 2, got {value}")
    return value


def add_oracle_argument(parser):
    """Add ``--oracle-rows``, how much of the fp64 oracle check to run."""
    parser.add_argument(
        "--oracle-rows",
        type=oracle_rows,
        default=0,
        help="check only this many rows of N_seq against the fp64 oracle: the "
        "first, the last and random ones. For inputs too large to check all "
        "rows. 0 checks all rows. See testutil's Oracle",
    )
    return parser
