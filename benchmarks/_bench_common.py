# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Timing, reporting and argument plumbing shared by the benchmarks.

Inputs, tolerances and metrics come from ``testutil``, the same place the tests
get them, so a benchmark and a test of the same op feed it the same tensors and
accept the same error.
"""

import argparse
import csv
import functools
import gc
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Hashable, Sequence
from typing import NamedTuple

import torch
from triton.testing import do_bench

import geometric_nn_ops_torch
from geometric_nn_ops_torch import _precision
from testutil.precision import torch_matmul_precision

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


@functools.cache
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


class Column(NamedTuple):
    """A report table column: which row field it shows and how it's formatted."""

    key: str
    heading: str
    width: int = 10
    # Decimal places, for float fields.
    precision: int = 4
    align: str = ">"


class AllocatorState(NamedTuple):
    """Live allocations in the CUDA caching allocator."""

    allocations: int
    bytes: int


def allocator_state() -> AllocatorState:
    """The live allocations now, after collecting garbage in reference cycles."""
    gc.collect()
    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()
    return AllocatorState(
        stats["allocation.all.current"], stats["allocated_bytes.all.current"]
    )


def warm_up_blas() -> AllocatorState:
    """Create the BLAS workspaces, which stay allocated for the rest of the
    process, and return the allocator state with only them allocated.

    There is a workspace per BLAS library and per thread. On ROCm, a matmul in
    bf16 and one in fp64 use both rocBLAS and hipBLASLt(hipBLASLt has no fp64),
    and the backward runs on autograd's thread.
    """
    for dtype in (torch.bfloat16, torch.float64):
        a = torch.ones(256, 256, device="cuda", dtype=dtype, requires_grad=True)
        (a @ a).sum().backward()
        del a
    return allocator_state()


class PeakMemory(NamedTuple):
    """Memory over one call, in GiB. See :func:`peak_memory_gib`."""

    # Memory held on entry
    entry_alloc_gib: float
    entry_reserved_gib: float
    # Peaks above what was allocated and reserved on entry.
    alloc_gib: float
    reserved_gib: float


def peak_memory_gib(fn) -> PeakMemory:
    """Peak allocated and reserved memory during a call to ``fn``.

    Returned split out as the memory held on entry to the function and the max
    delta above that reached while the function ran.

    Recording the entry memory separately distinguishes between allocations for
    the function inputs and other state used by the libraries (e.g. BLAS
    workspaces) from the memory directly used by the function. Because of state
    that is allocated on first use, this is only consistent for warm runs.

    Allocated is what the live tensors ask for. Reserved is what the caching
    allocator took from the driver, so it carries the rounding and
    fragmentation too and is the closer of the two to what the device reports.
    Both are recorded because they answer different questions. Garbage
    collection runs first so that unreachable reference cycles don't count.
    """
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    fn()
    torch.cuda.synchronize()
    return PeakMemory(
        entry_alloc_gib=allocated / GIB,
        entry_reserved_gib=reserved / GIB,
        alloc_gib=(torch.cuda.max_memory_allocated() - allocated) / GIB,
        reserved_gib=(torch.cuda.max_memory_reserved() - reserved) / GIB,
    )


def memory_fields(measure, peak: PeakMemory | None = None) -> dict:
    """Result row fields for ``measure``'s peak memory, empty without
    ``peak``."""
    return {
        f"{measure}_{name}": None if peak is None else getattr(peak, name)
        for name in PeakMemory._fields
    }


class LeftoverAllocations(Exception):
    """The allocator holds more than the warm-up's allocations."""

    def __init__(self, when: str, state: AllocatorState, expected: AllocatorState):
        super().__init__(f"allocations {when}: {state}, expected {expected}")
        self.state = state


def check_allocator(expected: AllocatorState, when: str):
    """Raise :class:`LeftoverAllocations` unless the allocator is in the
    ``expected`` state."""
    state = allocator_state()
    if state != expected:
        raise LeftoverAllocations(when, state, expected)


def measure_step_memory(build_step, expected: AllocatorState) -> PeakMemory:
    """Peak memory of a step built by ``build_step()``.

    The allocator state is verified to match ``expected`` before and after. The
    cache is emptied before the inputs are built and again before the step runs,
    so the allocator starts from the same state for every measurement, and
    reserved memory doesn't depend on what ran before.
    """
    check_allocator(expected, "before measuring memory")
    torch.cuda.empty_cache()
    step = build_step()
    torch.cuda.empty_cache()
    peak = peak_memory_gib(step)
    del step
    check_allocator(expected, "after measuring memory")
    return peak


def add_error(row, message):
    """Append ``message`` to ``row``'s error."""
    row["error"] = f"{row['error']}. {message}" if row["error"] else message


def measure_memory(row, builds: dict, expected: AllocatorState) -> AllocatorState:
    """Measure each step in ``builds``, ``{measure: build_step}``, into ``row``.

    Returns the allocator state the next measurement should expect: if one step
    leaks, then the new expectation includes those leaks so subsequent steps
    aren't marked as failed.
    """
    for measure, build_step in builds.items():
        failed = False
        try:
            row |= memory_fields(measure, measure_step_memory(build_step, expected))
        except LeftoverAllocations as error:
            add_error(row, f"{measure} memory: {error}")
            expected = error.state
            continue
        except Exception as error:  # noqa: BLE001
            add_error(row, f"{measure} memory: {error_text(error)}")
            failed = True
        if failed:
            # Checked here, once the traceback no longer holds the step's frames.
            state = allocator_state()
            if state != expected:
                add_error(
                    row,
                    f"{measure} memory: allocations after the failure: {state}, "
                    f"expected {expected}",
                )
                expected = state
    return expected


class BenchCase:
    """The inputs of one case, shared by all backends.

    A benchmark subclasses this with its inputs and steps. :func:`run_benchmark`
    builds one with ``oracle=True`` for the checks and the timing, and one per
    memory measurement with ``oracle=False`` and only that measure, so that the
    step measured is the step timed.
    """

    def steps(self, backend: str, measures) -> dict[str, Callable]:
        """A callable running one step for each of ``measures`` that applies
        to ``backend``."""
        raise NotImplementedError

    def check(self, backend: str, measures, label: str):
        """Check ``backend`` against the oracle in the modes of ``measures``,
        the measures that apply to it. Raises on a mismatch."""
        raise NotImplementedError

    def drop_oracle(self):
        """Free the oracle's results once all backends are checked."""

    def fields(self, backend: str) -> dict:
        """Extra result row fields for ``backend``."""
        return {}


def prepare_backend(case: BenchCase, backend, measures, label) -> dict:
    """Time the first call of each of ``backend``'s steps, then check it.

    Each step is built just before its first call, in the order of
    ``measures``, so the first call of an earlier measure includes compiling
    anything it shares with later ones.
    """
    fields = {}
    applicable = []
    for measure in measures:
        step = case.steps(backend, [measure]).get(measure)
        if step is not None:
            fields[f"{measure}_first_s"] = wall_seconds(step)
            applicable.append(measure)
        del step
    if not applicable:
        return {}
    case.check(backend, applicable, label)
    return fields | case.fields(backend)


OP_MEASURES = ["fwd_infer", "fwd_train", "bwd", "fwd_bwd"]

OP_TRAINING_MEASURES = {"fwd_train", "bwd", "fwd_bwd"}

# The steps that need an output gradient.
OP_BACKWARD_MEASURES = {"bwd", "fwd_bwd"}


def op_steps(
    measures, *, infer, train=None, targets=(), grad_out=None, retained_backward=True
) -> dict:
    """The steps of an op's microbenchmark: ``fwd_infer``, ``fwd_train``,
    ``bwd`` and ``fwd_bwd``, for those of ``measures`` that apply.

    ``infer()`` and ``train()`` run the forward for inference and for training.
    Without ``train`` the op has no training steps. ``bwd`` is a backward over a
    retained graph, from a forward run now. It is left out without
    ``retained_backward``.
    """
    steps = {}

    def fwd_infer():
        with torch.inference_mode():
            return infer()

    steps["fwd_infer"] = fwd_infer
    if train is not None:

        def fwd_bwd():
            return torch.autograd.grad(train(), targets, grad_outputs=grad_out)

        steps["fwd_train"] = train
        steps["fwd_bwd"] = fwd_bwd
        if "bwd" in measures and retained_backward:
            out = train()
            steps["bwd"] = lambda: torch.autograd.grad(
                out, targets, grad_outputs=grad_out, retain_graph=True
            )
    return {m: steps[m] for m in measures if m in steps}


def add_steps(steps: dict, case: BenchCase, backend, measures) -> dict:
    """Add ``backend``'s steps to ``steps`` as ``{(backend, measure): step}``."""
    for measure, step in case.steps(backend, measures).items():
        steps[backend, measure] = step
    return {}


def bench_case(make_case, key, base_row, args, measures, case_keys) -> list:
    """Check, then time all backends of one case. Returns a row per backend."""
    case = make_case(key, measures, oracle=True)
    rows = {backend: base_row | {"backend": backend} for backend in args.backends}
    for backend, row in rows.items():
        label = describe_case(row, case_keys)
        run_case(
            row, functools.partial(prepare_backend, case, backend, measures, label)
        )
    case.drop_oracle()
    steps = {}
    for backend, row in rows.items():
        if not row["error"]:
            run_case(row, functools.partial(add_steps, steps, case, backend, measures))
    time_backends(steps, rows, args)
    return list(rows.values())


def memory_builds(make_case, key, row, measures) -> dict:
    """``{measure: build_step}`` for the measures timed in ``row``. Each build
    makes a case of its own, with only that measure."""

    def build(measure):
        def build_step():
            case = make_case(key, [measure], oracle=False)
            return case.steps(row["backend"], [measure])[measure]

        return build_step

    return {m: build(m) for m in measures if row[f"{m}_ms"] is not None}


def measure_fields(measures, *, memory) -> dict:
    """Empty result row fields for ``measures``."""
    fields = {}
    for m in measures:
        fields |= {f"{m}_first_s": None, f"{m}_ms": None, f"{m}_spread_pct": None}
        if memory:
            fields |= memory_fields(m)
    return fields


def sweep(args, measures, cases, make_case, make_row, case_keys) -> list:
    """Run each of ``cases`` (see :func:`run_benchmark`), returning the rows.

    A failure outside the per-backend recording, such as running out of memory
    building the shared inputs, is recorded for all backends of the case.

    After each case the allocator must be back to its state after the warm-up.
    Then with ``--memory`` each backend's timed steps are measured one at a
    time on fresh inputs (see :func:`measure_step_memory`). Leftover
    allocations are an error on the case's or the backend's row, and the state
    found becomes the expected one, so a leak is only recorded as a failure
    where it happened.
    """
    expected = warm_up_blas()
    rows = []
    for key in cases:
        row = make_row(key) | measure_fields(measures, memory=args.memory)
        row["error"] = None
        try:
            case_rows = bench_case(make_case, key, row, args, measures, case_keys)
        except Exception as error:  # noqa: BLE001
            case_rows = [
                row | {"backend": backend, "error": error_text(error)}
                for backend in args.backends
            ]
        try:
            check_allocator(expected, "left after the case")
        except LeftoverAllocations as error:
            for case_row in case_rows:
                add_error(case_row, str(error))
            expected = error.state
        if args.memory:
            for case_row in case_rows:
                if not case_row["error"]:
                    builds = memory_builds(make_case, key, case_row, measures)
                    expected = measure_memory(case_row, builds, expected)
        rows += case_rows
    return rows


def report_columns(measures, *, memory) -> tuple[list[Column], list[Column]]:
    """The measured and compared report columns for ``measures``.

    Reserved growth is the memory shown and compared: it is the pool the device
    actually had to give up for each implementation.
    """

    def column(key, heading, precision=4):
        return Column(key, heading, max(len(heading) + 1, 9), precision)

    measured, compared = [], []
    for m in measures:
        measured += [
            column(f"{m}_ms", f"{m}_ms"),
            Column(f"{m}_spread_pct", "sprd%", 6, 1),
        ]
        compared.append(column(f"{m}_ms", f"{m}_%"))
        if memory:
            measured.append(column(f"{m}_reserved_gib", f"{m}_GiB", 3))
            compared.append(column(f"{m}_reserved_gib", f"{m}_GiB_%"))
    measured += [column(f"{m}_first_s", f"{m}_1st_s", 2) for m in measures]
    return measured, compared


def run_benchmark(
    args,
    *,
    measures,
    cases,
    make_case,
    make_row,
    case_columns: Sequence[Column],
    extra_columns: Sequence[Column] = (),
    provenance: dict | None = None,
):
    """Run, report and write a benchmark, then exit non-zero if any case
    failed.

    For each key in ``cases``, ``make_case(key, measures, *, oracle)`` builds a
    :class:`BenchCase`. Each backend's first call of each step is timed on its
    own, since it includes compilation, and then the backend is checked
    against the oracle. A backend that fails the check is not timed. Then all
    backends' steps are timed in interleaved rounds (see
    :func:`time_interleaved`) and reported as the median, with the spread over
    rounds. With ``--memory``, each step's peak memory is then measured on its
    own inputs (see :func:`measure_step_memory`).

    ``make_row(key)`` returns the case's own result row fields, with
    ``backend`` among them. The CSV has columns only for the requested
    measures. A blank cell in one means the step doesn't apply to that backend.
    Failures are in the ``error`` column. ``measures`` is the script's
    measures, in the order to run them.
    """
    require_gpu()
    measures = [m for m in measures if m in args.measure]
    print_provenance(
        args.dtype,
        backends=",".join(args.backends),
        fp32_precision=args.fp32_precision,
        rounds=args.rounds,
        memory=args.memory,
        measure=",".join(measures),
        **(provenance or {}),
    )
    case_keys = [column.key for column in case_columns]
    with torch_matmul_precision(args.fp32_precision):
        rows = sweep(args, measures, cases, make_case, make_row, case_keys)
    measured, compared = report_columns(measures, memory=args.memory)
    report(
        rows,
        case=case_columns,
        measured=[*measured, *extra_columns],
        compared=compared,
        baselines=args.baselines,
    )
    finish(rows, args.out, case_keys=case_keys)


def add_memory_argument(parser):
    """Add ``--memory/--no-memory``."""
    parser.add_argument(
        "--memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="measure the peak memory of each step on its own inputs (see "
        "measure_step_memory)",
    )
    return parser


def add_dtype_arguments(parser):
    """Add ``--dtype`` and ``--fp32-precision``."""
    parser.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    parser.add_argument(
        "--fp32-precision",
        default="ieee",
        choices=["ieee", "tf32"],
        help="torch.backends.cuda.matmul.fp32_precision, for fp32 only",
    )
    return parser


def add_benchmark_arguments(parser, *, measures, backends, default_backends, baselines):
    """Add the arguments :func:`run_benchmark` reads."""
    add_dtype_arguments(parser)
    add_backend_arguments(
        parser, backends, default=default_backends, baselines=baselines
    )
    parser.add_argument(
        "--measure",
        nargs="+",
        default=list(measures),
        choices=list(measures),
        action=UniqueValues,
    )
    add_timing_arguments(parser)
    add_memory_argument(parser)
    parser.add_argument("--out", type=pathlib.Path, default=None, help="CSV to write")
    return parser


def tolerance_precision(dtype) -> str:
    """The precision key for looking up the oracle tolerance for ``dtype``."""
    return _precision.torch_precision() if dtype == torch.float32 else "ieee"


def pct(value, base, width=10):
    """``value`` as a percentage of ``base``."""
    if value is not None and base:
        return f"{100.0 * value / base:>{width}.1f}"
    return f"{'-':>{width}}"


def cell(row, column: Column) -> str:
    """Format ``row``'s value for ``column``."""
    value = row[column.key]
    if value is None:
        return f"{'-':>{column.width}}"
    if isinstance(value, float):
        return f"{value:{column.align}{column.width}.{column.precision}f}"
    return f"{value!s:{column.align}{column.width}}"


def print_header(columns: Sequence[Column]):
    header = " ".join(f"{c.heading:{c.align}{c.width}}" for c in columns)
    print(header)
    print("-" * len(header))


def report(
    rows,
    *,
    case: Sequence[Column],
    measured: Sequence[Column],
    compared: Sequence[Column],
    baselines: Sequence[str],
):
    """Print the ``measured`` fields of each row, then a comparison table for
    each of ``baselines``.

    A comparison table shows the ``compared`` fields of the other backends as
    a percentage of the baseline's. ``case`` are the fields that identify a
    row, including ``backend``. Rows that differ only in backend are compared
    with each other.
    """
    case_keys = [column.key for column in case]
    print_header([*case, *measured])
    for row in rows:
        if row["error"]:
            print(f"{describe_case(row, case_keys)}  FAILED {row['error']}")
            continue
        print(" ".join(cell(row, column) for column in [*case, *measured]))
    for baseline in baselines:
        report_relative(rows, case, compared, baseline)


def report_relative(rows, case: Sequence[Column], compared, baseline: str):
    """Print the ``compared`` fields as a percentage of ``baseline``'s, for
    each case."""
    if baseline not in {row["backend"] for row in rows}:
        return
    by_case = {}
    for row in rows:
        key = tuple(row[column.key] for column in case if column.key != "backend")
        by_case.setdefault(key, {})[row["backend"]] = row
    lines = []
    for _, per_backend in sorted(by_case.items()):
        base = per_backend.get(baseline)
        if base is None or base["error"]:
            continue
        for backend, row in per_backend.items():
            if backend == baseline or row["error"]:
                continue
            cells = [cell(row, column) for column in case]
            cells += [pct(row[c.key], base[c.key], c.width) for c in compared]
            lines.append(" ".join(cells))
    if not lines:
        return
    print(f"\n# As % of {baseline} (lower is better)")
    print_header([*case, *compared])
    print("\n".join(lines))


class UniqueValues(argparse.Action):
    """argparse action for a list option whose values must not repeat."""

    def __call__(self, parser, namespace, values, option_string=None):
        if len(set(values)) != len(values):
            parser.error(f"{option_string} lists a value more than once: {values}")
        setattr(namespace, self.dest, values)


def add_backend_arguments(parser, choices, *, default, baselines=None):
    """Add ``--backends``, chosen from the backend names in ``choices``.

    If ``baselines`` is given, also add ``--baselines`` with it as the default.
    The report prints a comparison table for each baseline.
    """
    choices = list(choices)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(default),
        choices=choices,
        action=UniqueValues,
    )
    if baselines is not None:
        parser.add_argument(
            "--baselines",
            nargs="+",
            default=list(baselines),
            choices=choices,
            help="report the other backends as a percentage of each of these",
        )
    return parser


def run_fields(args) -> dict:
    """Provenance fields for a result row."""
    return {
        "dtype": args.dtype,
        "fp32_precision": args.fp32_precision,
        "device": torch.cuda.get_device_name(0),
        "commit": git_commit(),
    }


def run_case(row: dict, measure: Callable[[], dict]) -> dict:
    """Update ``row`` with ``measure()``, or record the failure in
    ``row["error"]``.

    Recording the failure rather than raising lets the sweep continue.
    :func:`finish` reports failures and exits non-zero.
    """
    try:
        row |= measure()
    except Exception as error:  # noqa: BLE001
        row["error"] = error_text(error)
    return row


def error_text(error: BaseException) -> str:
    """Format an exception for a result row's ``error`` field."""
    return f"{type(error).__name__}: {error}"


def add_timing_arguments(parser):
    """The two ``do_bench`` windows, wall-clock milliseconds, and the number of
    interleaved rounds (see :func:`time_interleaved`)."""
    parser.add_argument("--warmup", type=int, default=25, help="warmup window (ms)")
    parser.add_argument("--rep", type=int, default=100, help="measurement window (ms)")
    parser.add_argument(
        "--rounds",
        type=positive_int,
        default=5,
        help="interleaved rounds. Each result is the median over them",
    )
    return parser


def positive_int(text) -> int:
    """argparse type for a count that must be at least one."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


class Timing(NamedTuple):
    """The median time of one step over several rounds."""

    median_ms: float
    # (max - min) / median over the rounds, as a percentage.
    spread_pct: float


class StepFailed(Exception):
    """Raised by :func:`time_interleaved` when a step fails. ``name`` is the
    step's key and ``error`` the exception it raised."""

    def __init__(self, name: Hashable, error: Exception):
        super().__init__(f"{name} failed while being timed")
        self.name = name
        self.error = error


def time_interleaved(
    steps: dict[Hashable, Callable], *, rounds, warmup, rep
) -> dict[Hashable, Timing]:
    """Time each step in ``steps``, alternating between them.

    Each round times each step once with ``do_bench``, rotating which step
    goes first. The result for a step is the median over rounds, with the
    spread. If the GPU's clocks drift (thermal throttling, other load), timing
    one step's rounds back to back would inflate the times for the steps that
    ran later. Interleaving spreads it across the steps, and the spread shows
    how large it was.

    Raises :class:`StepFailed` if a step raises.
    """
    names = list(steps)
    samples = {name: [] for name in names}
    for index in range(rounds):
        shift = index % len(names)
        for name in names[shift:] + names[:shift]:
            try:
                time_ms = do_bench(
                    steps[name], warmup=warmup, rep=rep, return_mode="median"
                )
            except Exception as error:
                raise StepFailed(name, error) from error
            samples[name].append(time_ms)
    timings = {}
    for name, times in samples.items():
        median = statistics.median(times)
        timings[name] = Timing(median, 100.0 * (max(times) - min(times)) / median)
    return timings


def compiled_graph_count() -> int:
    """The number of graphs Dynamo has compiled in this process, to detect a
    recompile."""
    return torch._dynamo.utils.counters["stats"]["unique_graphs"]


def time_backends(steps, rows, args):
    """Time ``{(backend, measure): step}`` interleaved, and record the results
    in ``rows[backend]``.

    If a backend fails while being timed, its row gets the error. The other
    backends' rounds are then incomplete, so they are marked as not timed. If
    Dynamo compiles a graph while they are timed, all of them fail, since the
    timings would include compilation.
    """
    if not steps:
        return
    graphs = compiled_graph_count()
    try:
        timings = time_interleaved(
            steps, rounds=args.rounds, warmup=args.warmup, rep=args.rep
        )
    except StepFailed as failure:
        failed_backend, _ = failure.name
        mark_timing_failure(rows, failed_backend, error_text(failure.error))
        return
    if compiled_graph_count() != graphs:
        for row in rows.values():
            if not row["error"]:
                row["error"] = "Dynamo compiled a graph while being timed"
        return
    for (backend, measure), timing in timings.items():
        rows[backend][f"{measure}_ms"] = timing.median_ms
        rows[backend][f"{measure}_spread_pct"] = timing.spread_pct


def mark_timing_failure(rows, failed_backend: str, error: str):
    """Record ``error`` for ``failed_backend``, and mark the other backends as
    not timed."""
    for backend, row in rows.items():
        if row["error"]:
            continue
        if backend == failed_backend:
            row["error"] = error
        else:
            row["error"] = f"not timed: {failed_backend} failed while being timed"


def assert_finite(out, label):
    """Raise if ``out`` has a non-finite element."""
    if not torch.isfinite(out).all():
        raise RuntimeError(f"[{label}] non-finite output")


def wall_seconds(fn) -> float:
    """Host-clock seconds of one call to ``fn``, e.g. a first call that compiles."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


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

    Failed cases are written too, with the reason in ``error``, so the file
    covers every case that was asked for. Dropping them would leave a file that
    reads as a complete sweep of a smaller one. A failed row may still hold
    measurements made before the failure, which shouldn't be trusted.
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
