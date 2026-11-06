# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Explicit implementation registry; performance qualification precedes dispatch."""

from dataclasses import dataclass
from importlib import import_module

from geometric_nn_ops_torch.triangle._reference.trimul import trimul_torch

from ._pipeline import Parameters, compose, validate


@dataclass(frozen=True)
class Variant:
    description: str
    arithmetic: str
    evidence: str = "ported experiment; qualification required per shape/runtime"


VARIANTS = {
    "layout_aware_pipeline_mfma16": Variant(
        "128x128x32 MFMA16 producers; original pipeline output and library contraction",
        "FP32; original projection dot policy, highest/IEEE library contraction",
        "C=H<=4096; producer-only schedule experiment",
    ),
    "layout_aware_pipeline": Variant(
        "Single-stage channel-major Triton projections/norms and library contraction",
        "FP32; original projection dot policy, highest/IEEE library contraction",
        "C=H<=4096; measured schedule experiment, not all-Triton",
    ),
    "all_triton_fused": Variant(
        "Fused input norm/dual producer, IEEE contraction and Triton output stages",
        "FP32; IEEE producer and contraction, original output dot policy",
        "C=H<=512; qualification required per shape/runtime",
    ),
    "all_triton": Variant(
        "Original fused projections, IEEE Triton contraction and layout-aware norms",
        "FP32; IEEE contraction, original projection dot policy",
        "new all-Triton compute experiment; qualify per shape/runtime",
    ),
    "original_channel_major": Variant(
        "Initial forward kernels with library bmm contraction",
        "FP32 input; original Triton dot policy",
        "original contribution + documented functional adapter; qualify per case",
    ),
    "original_channel_major_norm": Variant(
        "Original contribution plus bmm and register-resident centered LayerNorm",
        "FP32 input; original Triton dot policy",
        "controlled LayerNorm comparison; qualify per case",
    ),
    "reference": Variant("Stock PyTorch oracle", "input dtype", "reference"),
    "reference_bounded": Variant(
        "PyTorch oracle with bounded projection/contraction/normalization calls",
        "IEEE FP32; forward-only",
        "bounded oracle; verify repeatability and small-shape agreement",
    ),
    "geometric": Variant("Existing geometric-nn-ops Triton", "library policy"),
    "pointwise": Variant(
        "Mask/projection multiply and output sigmoid multiply", "FP32"
    ),
    "projection_epilogue": Variant("Single GEMMs with pointwise epilogues", "FP32"),
    "projection_epilogue_300x": Variant("Alternate projection GEMM schedule", "FP32"),
    "dual_projection": Variant(
        "Dual input projection/gate + Triton contraction + output gate", "BF16 dots"
    ),
    "dual_projection_fp32": Variant(
        "IEEE dual producer + library contraction + IEEE fused output gate",
        "IEEE FP32 dots; highest/IEEE Torch library GEMMs required",
        "new precision-safe adaptation; qualification required per shape/runtime",
    ),
    "dual_projection_norm": Variant(
        "IEEE dual producer with one-warp register-resident row normalization",
        "IEEE FP32; highest/IEEE Torch library GEMMs required",
        "new normalization-only experiment; qualification required per shape/runtime",
    ),
    "dual_normalization": Variant(
        "Output dual LayerNorm/projection/gate fusion", "BF16 output rounding"
    ),
    "unified_projection": Variant(
        "Shared/different-input dual GEMM epilogues", "historical dot policy"
    ),
    "channel_major": Variant(
        "Consumer-layout projections + multi-channel swizzled contraction",
        "historical dot policy",
    ),
    "layout_aware": Variant(
        "Bounded head/row chunks, channel-major Torch GEMMs",
        "FP32",
        "adapted from Conductor v2; functional port needs qualification",
    ),
    "layout_aware_norm": Variant(
        "Bounded channel-major library GEMMs with transpose-free fused output norm",
        "IEEE FP32; highest/IEEE Torch library GEMMs required",
        "new experiment; qualification required per shape/runtime",
    ),
    "layout_aware_norm_input": Variant(
        "Layout-aware library GEMMs with Triton input and channel-major output norm",
        "IEEE FP32; highest/IEEE Torch library GEMMs required",
        "input-normalization step toward all-Triton; qualify per shape/runtime",
    ),
}


def run_variant(name, x, direction="outgoing", mask=None, *, eps=1e-5, **weights):
    """Full update with the same weights as the public functional API; no fallback."""
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant {name!r}; choose from {tuple(VARIANTS)}")
    p = Parameters(**weights)
    validate(x, direction, mask, eps, p, gpu=name not in ("reference", "layout_aware"))
    if name == "reference":
        return trimul_torch(x, direction, mask, eps=eps, **p.kwargs())
    if name == "geometric":
        from geometric_nn_ops_torch.triangle._kernels.trimul import trimul_triton

        parameters = p.kwargs().copy()
        for name, size in (
            ("norm_in_weight", x.shape[-1]),
            ("norm_in_bias", x.shape[-1]),
            ("norm_out_weight", p.hidden),
            ("norm_out_bias", p.hidden),
        ):
            if parameters[name] is None:
                parameters[name] = (
                    x.new_ones(size) if name.endswith("weight") else x.new_zeros(size)
                )
        return trimul_triton(x, direction, mask, eps=eps, **parameters)
    if name in (
        "layout_aware_pipeline_mfma16",
        "layout_aware_pipeline",
        "all_triton_fused",
        "all_triton",
        "original_channel_major",
        "original_channel_major_norm",
        "layout_aware",
        "layout_aware_norm",
        "layout_aware_norm_input",
        "dual_projection_fp32",
        "dual_projection_norm",
        "reference_bounded",
    ):
        return import_module(f"{__package__}.{name}").update(
            x,
            direction,
            mask,
            eps,
            p,
        )
    return compose(name, x, direction, mask, eps, p)
