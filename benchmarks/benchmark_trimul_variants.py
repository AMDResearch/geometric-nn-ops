# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qualify full-update variants, then benchmark only numerically passing cases."""

import argparse
import hashlib
import itertools
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import triton
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from geometric_nn_ops_torch.triangle.experimental import VARIANTS, registry, run_variant
from geometric_nn_ops_torch.triangle.experimental.policy import (
    POLICY_VERSION,
    benchmark_order,
    recommend_variants,
)

COMMON_SIZES_USED = ((64, 64), (128, 128), (256, 256), (384, 384))


def qualified_report(report):
    rows = [
        row
        for row in report["results"]
        if row["status"] == "measured"
        and row.get("qualified") is True
        and row.get("finite") is True
    ]
    names = {row["variant"] for row in rows}
    keys = ("n", "c", "hidden", "direction", "variant")
    measured_keys = {tuple(row[key] for key in keys) for row in rows}
    return report | {
        "results": rows,
        "winners": [
            row
            for row in report["winners"]
            if tuple(row[key] for key in keys) in measured_keys
        ],
        "variants": {
            name: info for name, info in report["variants"].items() if name in names
        },
        "report_scope": (
            "Only finite, numerically qualified, measured variant/case pairs. "
            "Qualification applies to these inputs/shapes/runtime, not all inputs."
        ),
    }


def estimated_case_bytes(n, c, h, batch):
    """Conservative screening estimate, not an allocator/ compiler guarantee."""
    return batch * n * n * (4 * (8 * c + 8 * h) + 8)


def make_case(n, c, h, batch, device, biases=False):
    x = torch.randn(batch, n, n, c, device=device)
    mask = torch.rand(batch, n, n, device=device) > 0.2
    weights = {
        "p_in_weight": torch.randn(2 * h, c, device=device) / c**0.5,
        "g_in_weight": torch.randn(2 * h, c, device=device) / c**0.5,
        "p_out_weight": torch.randn(c, h, device=device) / h**0.5,
        "g_out_weight": torch.randn(c, c, device=device) / c**0.5,
        "norm_in_weight": torch.rand(c, device=device) + 0.5,
        "norm_in_bias": torch.randn(c, device=device) * 0.1,
        "norm_out_weight": torch.rand(h, device=device) + 0.5,
        "norm_out_bias": torch.randn(h, device=device) * 0.1,
    }
    if biases:
        for key, width in (
            ("p_in_bias", 2 * h),
            ("g_in_bias", 2 * h),
            ("p_out_bias", c),
            ("g_out_bias", c),
        ):
            weights[key] = torch.randn(width, device=device) * 0.1
    return x, mask, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--c", nargs="+", type=int)
    parser.add_argument("--hidden", nargs="+", type=int)
    parser.add_argument(
        "--common-sizes",
        action="store_true",
        help="Use common sizes used: paired C/H=64/64,128/128,256/256,384/384",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=["incoming", "outgoing"],
        default=["outgoing", "incoming"],
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--recommend-order",
        action="store_true",
        help="Use provisional policy to order, never prune, candidates",
    )
    parser.add_argument("--biases", action="store_true")
    parser.add_argument(
        "--tune-producer",
        action="store_true",
        help="Also qualify/time 432 M/N/K, warp, stage and MFMA producer schedules "
        "as full updates; output schedule and library contraction stay fixed",
    )
    parser.add_argument(
        "--layout", choices=["contiguous", "transposed"], default="contiguous"
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--qualified-report",
        type=Path,
        help="Separate comparison report containing passing cases only",
    )
    parser.add_argument(
        "--oracle", choices=["reference", "reference_bounded"], default="reference"
    )
    args = parser.parse_args()
    schedule_impls, schedule_metadata, schedule_configs = {}, {}, {}
    if args.tune_producer:
        from _trimul_schedule_search import schedule_candidates

        schedule_impls, schedule_metadata, schedule_configs = schedule_candidates()

    def execute(name, *pos, **kw):
        if name in schedule_impls:
            return schedule_impls[name](*pos, **kw)
        return run_variant(name, *pos, **kw)

    candidate_names = list(args.variants) + list(schedule_impls)
    if args.common_sizes and (args.c is not None or args.hidden is not None):
        parser.error("--common-sizes cannot be combined with --c or --hidden")
    sizes = (
        COMMON_SIZES_USED
        if args.common_sizes
        else tuple(itertools.product(args.c or [64, 128], args.hidden or [64, 128]))
    )
    if (
        min(args.n + [v for pair in sizes for v in pair] + [args.batch, args.repeats])
        <= 0
        or args.warmup < 0
    ):
        parser.error("Dimensions/repeats must be positive and warmup nonnegative")
    if any(not math.isfinite(t) or t < 0 for t in (args.atol, args.rtol)):
        parser.error("Tolerances must be finite and nonnegative")
    if args.report.exists():
        raise FileExistsError(args.report)
    if args.qualified_report is not None:
        if args.qualified_report.resolve() == args.report.resolve():
            parser.error("Diagnostic and qualified reports must have different paths")
        if args.qualified_report.exists():
            raise FileExistsError(args.qualified_report)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    report = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "hip": torch.version.hip,
        "device": props.name,
        "architecture": getattr(props, "gcnArchName", None),
        "device_uuid": str(getattr(props, "uuid", "")),
        "source_sha256": {
            str(
                path.relative_to(Path(registry.__file__).parent.parent)
            ): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(registry.__file__).parent.parent.rglob("*.py"))
        },
        "arguments": vars(args)
        | {
            "report": str(args.report),
            "qualified_report": (
                None if args.qualified_report is None else str(args.qualified_report)
            ),
        },
        "channel_pairs": sizes,
        "memory_screen": (
            "Estimated batch*N*N*(4*(8*C+8*H)+8) bytes; skip above 80% "
            "of current free GPU memory. Conservative estimate, not a guarantee."
        ),
        "variants": {
            name: asdict(info) for name, info in (VARIANTS | schedule_metadata).items()
        },
        "producer_schedule_configs": schedule_configs,
        "method": (
            "Full update including layout conversions/functional weight packing; "
            "same tensors for every variant. First-invocation wall separate "
            "(may include compile/autotune, caches may already be warm), "
            "CUDA-event warmed median; peak incremental allocated memory, not "
            "process RSS. Failed qualification is never a winner."
        ),
        "results": [],
        "winners": [],
        "recommendations": [],
        "recommendation_policy": POLICY_VERSION if args.recommend_order else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.qualified_report is not None:
        args.qualified_report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if args.qualified_report is not None:
            args.qualified_report.write_text(
                json.dumps(qualified_report(report), indent=2, allow_nan=False) + "\n"
            )

    for n, (c, h), direction in itertools.product(args.n, sizes, args.directions):
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info()
        estimate = estimated_case_bytes(n, c, h, args.batch)
        if estimate > 0.8 * free_bytes:
            for name in candidate_names:
                report["results"].append(
                    {
                        "variant": name,
                        "n": n,
                        "c": c,
                        "hidden": h,
                        "direction": direction,
                        "status": "memory_budget_skipped",
                        "estimated_case_bytes": estimate,
                        "free_device_bytes": free_bytes,
                    }
                )
            print(
                "MEMORY_BUDGET_SKIPPED",
                n,
                c,
                h,
                direction,
                estimate,
                file=sys.stderr,
                flush=True,
            )
            save()
            continue
        x, mask, weights = make_case(n, c, h, args.batch, device, args.biases)
        if args.layout == "transposed":
            x = x.transpose(-2, -3)
        expected = run_variant(args.oracle, x, direction, mask, **weights)
        repeated = run_variant(args.oracle, x, direction, mask, **weights)
        torch.cuda.synchronize()
        valid_oracle = (
            torch.isfinite(expected).all().item()
            and torch.isfinite(repeated).all().item()
            and torch.allclose(expected, repeated, atol=args.atol, rtol=args.rtol)
        )
        del repeated
        if not valid_oracle:
            report["results"].append(
                {
                    "variant": args.oracle,
                    "n": n,
                    "c": c,
                    "hidden": h,
                    "direction": direction,
                    "status": "oracle_failure",
                    "error": (
                        "Oracle is nonfinite or not repeatable at requested tolerance"
                    ),
                }
            )
            print("ORACLE_FAILURE", n, c, h, direction, file=sys.stderr, flush=True)
            save()
            del x, mask, weights, expected
            continue
        passing = []
        candidates = candidate_names
        if args.recommend_order:
            recommendation = recommend_variants(
                n,
                c,
                h,
                direction,
                architecture=getattr(props, "gcnArchName", ""),
                ieee_fp32=torch.get_float32_matmul_precision() == "highest",
                contiguous=x.is_contiguous(),
                batch=args.batch,
            )
            candidates = benchmark_order(candidates, recommendation)
            report["recommendations"].append(
                {
                    "n": n,
                    "c": c,
                    "hidden": h,
                    "direction": direction,
                    **asdict(recommendation),
                    "benchmark_order": candidates,
                }
            )
        for name in candidates:
            row = {
                "variant": name,
                "n": n,
                "c": c,
                "hidden": h,
                "direction": direction,
                "shape": list(x.shape),
                "strides": list(x.stride()),
                "dtype": str(x.dtype),
            }
            if name in schedule_configs:
                row["producer_schedule"] = schedule_configs[name]
            try:
                start = time.perf_counter()
                output = execute(name, x, direction, mask, **weights)
                torch.cuda.synchronize()
                row["first_call_wall_s"] = time.perf_counter() - start
                if output.shape != expected.shape or output.dtype != expected.dtype:
                    raise ValueError("Output shape/dtype differs from reference")
                finite = torch.isfinite(output).all().item()
                close = finite and torch.allclose(
                    output, expected, atol=args.atol, rtol=args.rtol
                )
                row.update(finite=finite, qualified=bool(close))
                if finite:
                    diff = output - expected
                    row.update(
                        max_abs_error=diff.abs().max().item(),
                        rms_error=diff.square().mean().sqrt().item(),
                        relative_rms_error=(
                            diff.norm() / expected.norm().clamp_min(1e-30)
                        ).item(),
                    )
                    del diff
                if not close:
                    row["status"] = "numerical_failure"
                    print("QUALIFICATION_FAILED", row, file=sys.stderr, flush=True)
                    del output
                else:
                    del output
                    for _ in range(args.warmup):
                        output = execute(name, x, direction, mask, **weights)
                        del output
                    torch.cuda.synchronize()
                    baseline_bytes = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    samples = []
                    for _ in range(args.repeats):
                        begin, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        begin.record()
                        output = execute(name, x, direction, mask, **weights)
                        end.record()
                        end.synchronize()
                        samples.append(begin.elapsed_time(end))
                        del output
                    row.update(
                        status="measured",
                        samples_ms=samples,
                        median_ms=statistics.median(samples),
                        peak_incremental_allocated_bytes=torch.cuda.max_memory_allocated()
                        - baseline_bytes,
                    )
                    passing.append(row)
            except (ValueError, CompilationError, OutOfResources, RuntimeError) as exc:
                row.update(
                    status="rejected_or_execution_error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                print("VARIANT_ERROR", row, file=sys.stderr, flush=True)
                if isinstance(exc, RuntimeError):
                    report["results"].append(row)
                    save()
                    raise
            report["results"].append(row)
            save()
        if passing:
            best = min(passing, key=lambda row: row["median_ms"])
            report["winners"].append(
                {
                    key: best[key]
                    for key in ("n", "c", "hidden", "direction", "variant", "median_ms")
                }
            )
        else:
            print(
                "NO_QUALIFIED_VARIANT", n, c, h, direction, file=sys.stderr, flush=True
            )
        save()
        del x, mask, weights, expected
    print("COMPLETE", args.report, flush=True)


if __name__ == "__main__":
    main()
