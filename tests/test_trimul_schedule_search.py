# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest
import torch
from test_trimul_variants import weights

from geometric_nn_ops_torch.triangle.experimental import run_variant

spec = importlib.util.spec_from_file_location(
    "schedule_search",
    Path(__file__).parents[1] / "benchmarks" / "_trimul_schedule_search.py",
)
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)


def test_complete_grid():
    impls, metadata, configs = search.schedule_candidates()
    assert len(impls) == len(metadata) == len(configs) == 432
    assert len({tuple(c.values()) for c in configs.values()}) == 432
    assert {c["block_m"] for c in configs.values()} == {32, 64, 128, 256}
    assert {c["block_k"] for c in configs.values()} == {16, 32, 64}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/HIP")
@pytest.mark.parametrize("m,k", [(32, 16), (64, 32), (128, 64), (256, 16)])
@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
def test_schedule_full_update(m, k, direction):
    torch.manual_seed(42)
    x = torch.randn(1, 17, 17, 32, device="cuda")
    p = {name: value.cuda() for name, value in weights(32, 32, True).items()}
    mask = torch.rand(1, 17, 17, device="cuda") > 0.2
    producer = search.ProducerSchedule(
        dict(
            block_m=m,
            block_n=64,
            block_k=k,
            warps=4,
            stages=1,
            mapping=16,
        )
    )
    with torch.no_grad():
        expected = run_variant("reference", x, direction, mask, **p)
        actual = search._run(producer, x, direction, mask, **p)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
