# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton")


@pytest.fixture
def common(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("_trimul_common")


def test_experimental_backends_are_explicit_and_defaults_unchanged(common):
    parser = argparse.ArgumentParser()
    common.add_backend_argument(parser)
    assert parser.parse_args([]).backends == ["reference", "triton"]
    assert "experimental:dual_projection_fp32" in common.TRIMUL_IMPLEMENTATIONS
    assert "experimental:layout_aware" in common.TRIMUL_IMPLEMENTATIONS


def test_experimental_dtype_rejection(common):
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(backends=["experimental:layout_aware"], dtype="bf16")
    with pytest.raises(SystemExit):
        common.configure_experiments(parser, args)


@pytest.mark.parametrize("nonfinite", [False, True])
def test_strict_gate_rejects_bad_outputs(common, nonfinite):
    actual = torch.tensor([float("nan") if nonfinite else 1.01])
    with pytest.raises(AssertionError):
        common.assert_qualified(actual, torch.ones(1))


def test_repeated_oracle_checked_before_candidate(common, monkeypatch):
    calls = []

    def oracle(*args, **kwargs):
        calls.append("oracle")
        return torch.tensor([float(len(calls))])

    monkeypatch.setattr(common, "run_variant", oracle)
    with pytest.raises(AssertionError):
        common.qualify_experimental(
            lambda *args, **kwargs: calls.append("candidate"),
            torch.ones(1),
            "outgoing",
            None,
            {},
        )
    assert calls == ["oracle", "oracle"]


def test_micro_experimental_never_times_training(common, monkeypatch):
    micro = importlib.import_module("bench_trimul_micro")
    backend = "experimental:layout_aware"
    monkeypatch.setitem(micro.TRIMUL_IMPLEMENTATIONS, backend, lambda x, *a, **k: x * 2)
    monkeypatch.setattr(micro, "make_weights", lambda *a, **k: {})
    monkeypatch.setattr(micro, "make_pair_input", lambda *a, **k: (torch.ones(1), None))
    monkeypatch.setattr(micro, "qualify_experimental", lambda *a, **k: None)
    monkeypatch.setattr(
        micro, "assert_forward_agrees_with_the_oracle", lambda *a, **k: None
    )
    timed = []

    def timing(fn, **kwargs):
        timed.append(fn.__name__)
        fn()
        return 1.0

    monkeypatch.setattr(micro, "do_bench", timing)
    row = micro.bench_case(
        backend, 1, 1, "outgoing", torch.float32, 1, 1, set(micro.MEASURES), False
    )
    assert timed == ["fwd_infer"]
    assert row == {"fwd_train_ms": None, "fwd_infer_ms": 1.0, "bwd_ms": None}


def test_stack_experimental_never_times_residual_backward(common, monkeypatch):
    stack = importlib.import_module("bench_trimul_stack")

    class Layer(torch.nn.Module):
        parameter_names = ()
        direction = "outgoing"

        @staticmethod
        def impl(x, *args, **kwargs):
            return x.detach() * 2

        def forward(self, x, mask):
            return self.impl(x)

    class Model(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(), Layer()])

        def forward(self, x, mask):
            for layer in self.layers:
                x = x + layer(x, mask)
            return x

    monkeypatch.setattr(stack, "TrimulStack", Model)
    monkeypatch.setattr(
        stack,
        "make_pair_input",
        lambda *a, **k: (torch.ones(1, requires_grad=True), None),
    )
    monkeypatch.setattr(stack, "seeded_randn", lambda *a, **k: torch.ones(1))

    def oracle(name, x, *args, **kwargs):
        return x * 2

    monkeypatch.setattr(stack, "run_variant", oracle)
    monkeypatch.setattr(common, "run_variant", oracle)
    timed = []

    def timing(fn, **kwargs):
        timed.append(fn.__name__)
        fn()
        return 1.0

    monkeypatch.setattr(stack, "do_bench", timing)
    monkeypatch.setattr(stack, "peak_memory_gib", lambda fn: (0.1, 0.2))
    row = stack.bench_case(
        "experimental:layout_aware",
        1,
        1,
        "outgoing",
        torch.float32,
        2,
        1,
        1,
        set(stack.MEASURES),
        False,
    )
    assert timed == ["infer_step"]
    assert row["train_step_ms"] is None
