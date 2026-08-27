# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Timing, reporting and argument plumbing shared by the benchmarks.

Inputs, tolerances and metrics come from ``testutil``, the same place the tests
get them, so a benchmark and a test of the same op feed it the same tensors and
accept the same error.
"""

import csv
import os
import pathlib
import platform
import subprocess
import sys

import torch

import geometric_nn_ops_torch

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

GIB = 1024**3

COMMIT_ENV_VAR = "GEOMETRIC_NN_OPS_COMMIT"


def package_root() -> pathlib.Path:
    """Directory of the package actually imported.

    The benchmark scripts and the package they measure can come from different
    checkouts, so provenance is read from the import rather than from the
    script's own location.
    """
    return pathlib.Path(geometric_nn_ops_torch.__file__).resolve().parent


def git_commit() -> str:
    """Short commit hash of the package under test, for stamping every result.

    Read from ``$GEOMETRIC_NN_OPS_COMMIT`` when set, for running against a copy
    of the source with no ``.git`` directory.
    """
    env = os.environ.get(COMMIT_ENV_VAR)
    if env:
        return env
    command = ["git", "-C", str(package_root()), "rev-parse", "--short", "HEAD"]
    hint = f"`{' '.join(command)}` failed; set ${COMMIT_ENV_VAR} to supply the commit."
    try:
        proc = subprocess.run(command, capture_output=True, text=True)
    except OSError as error:
        raise RuntimeError(f"{hint}\n{error}") from None
    if proc.returncode != 0:
        raise RuntimeError(f"{hint} Exited {proc.returncode}:\n{proc.stderr}")
    commit = proc.stdout.strip()
    if not commit:
        raise RuntimeError(f"{hint} It exited cleanly but printed nothing")
    return commit


def release_blas_workspaces():
    """Return the BLAS workspaces to the allocator.

    They are allocated on first use and held for the process, so without this a
    case inherits ~150 MiB from whatever ran before it, which shifts allocator
    behaviour and makes an absolute memory figure depend on run order. The
    entry point is private, so treat its absence as nothing to release.
    """
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear is not None:
        clear()
    torch.cuda.empty_cache()


def peak_memory_gib(fn) -> tuple[float, float]:
    """Peak allocated and reserved memory (GiB) over one call to ``fn``.

    Allocated is what the live tensors ask for. Reserved is what the caching
    allocator took from the driver, so it carries the rounding and
    fragmentation too and is the closer of the two to what the device reports.
    Both are recorded because they answer different questions.
    """
    torch.cuda.synchronize()
    release_blas_workspaces()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (
        torch.cuda.max_memory_allocated() / GIB,
        torch.cuda.max_memory_reserved() / GIB,
    )


def check_clean_start(label, *, warn_gib=0.05, max_gib=1.0):
    """Fail, or warn, if memory is still held before a case builds anything.

    Nothing of the case exists at this point, so anything resident is left over
    from the one before and will be charged to this one's absolute peak. The
    bounds are deliberately loose: the expected value is zero, so they are
    there to catch a leak rather than to police a budget.
    """
    resident = torch.cuda.memory_allocated() / GIB
    if resident > max_gib:
        raise RuntimeError(
            f"{resident:.3f} GiB still resident before {label}; a previous case "
            f"leaked and the peak reported here would include it"
        )
    if resident > warn_gib:
        print(
            f"# warning: {resident:.3f} GiB resident before {label}",
            file=sys.stderr,
        )


def fmt(value, width=10, prec=4):
    """A measurement, or ``-`` where it does not apply."""
    return f"{value:>{width}.{prec}f}" if value is not None else f"{'-':>{width}}"


def pct(value, base, width=10):
    """``value`` as a percentage of ``base``."""
    if value is not None and base:
        return f"{100.0 * value / base:>{width}.1f}"
    return f"{'-':>{width}}"


def add_timing_arguments(parser):
    """The two ``do_bench`` windows. Wall-clock milliseconds, not counts."""
    parser.add_argument("--warmup", type=int, default=25, help="warmup window (ms)")
    parser.add_argument("--rep", type=int, default=100, help="measurement window (ms)")
    return parser


def require_gpu():
    if not torch.cuda.is_available():
        print("No GPU available", file=sys.stderr)
        sys.exit(1)


def print_provenance(dtype_name, **fields):
    """The run's identity, so a pasted result says what produced it.

    The package path is printed because it is not necessarily the checkout the
    script came from.
    """
    print(
        f"# device={torch.cuda.get_device_name(0)} dtype={dtype_name} "
        f"torch={torch.__version__} commit={git_commit()}"
    )
    print(f"# package={package_root()}")
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"# host={platform.node()} {extra}")


def finish(rows, out, *, case_keys):
    """Write the CSV, then exit non-zero if any case failed.

    Failed cases are written too, with their measurements empty and the reason
    in ``error``, so the file covers every case that was asked for. Dropping
    them would leave a file that reads as a complete sweep of a smaller one.
    """
    if out and rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n# wrote {len(rows)} rows -> {out}")

    failed = [row for row in rows if row.get("error")]
    if failed:
        print(f"\n# {len(failed)} case(s) FAILED:", file=sys.stderr)
        for row in failed:
            print(
                f"#   {describe_case(row, case_keys)}: {row['error']}", file=sys.stderr
            )
        sys.exit(1)


def describe_case(row, case_keys):
    """The fields that identify a case, for a table row or an error message."""
    return " ".join(f"{key}={row[key]}" for key in case_keys)
