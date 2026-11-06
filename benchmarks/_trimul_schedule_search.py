# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Explicit producer schedules for opt-in full-update benchmark qualification."""

import itertools
from functools import partial

from geometric_nn_ops_torch.triangle.experimental import (
    initial_kernels,
    layout_aware_pipeline,
)
from geometric_nn_ops_torch.triangle.experimental._pipeline import Parameters, validate
from geometric_nn_ops_torch.triangle.experimental.registry import Variant


def producer_configs():
    for m, n, k, warps, stages, mapping in itertools.product(
        (32, 64, 128, 256),
        (64, 128),
        (16, 32, 64),
        (4, 8, 16),
        (1, 2, 3),
        (16, 32),
    ):
        yield dict(
            block_m=m, block_n=n, block_k=k, warps=warps, stages=stages, mapping=mapping
        )


class ProducerSchedule:
    def __init__(self, config):
        self.config = config

    def __getitem__(self, grid):
        def launch(*args):
            c = self.config
            meta = dict(
                BLOCK_M=c["block_m"], BLOCK_N=c["block_n"], BLOCK_K=c["block_k"]
            )
            kernel = initial_kernels.fused_gemm_sigmoid_multiply_channel_major.fn
            return kernel[grid(meta)](
                *args,
                **meta,
                num_warps=c["warps"],
                num_stages=c["stages"],
                matrix_instr_nonkdim=c["mapping"],
            )

        return launch


def _run(producer, x, direction, mask, *, eps=1e-5, **weights):
    p = Parameters(**weights)
    validate(x, direction, mask, eps, p, gpu=True)
    return layout_aware_pipeline.update(
        x,
        direction,
        mask,
        eps,
        p,
        producer_kernel=producer,
    )


def schedule_candidates():
    implementations, metadata, configs = {}, {}, {}
    for config in producer_configs():
        name = (
            f"producer_m{config['block_m']}_n{config['block_n']}"
            f"_k{config['block_k']}_w{config['warps']}"
            f"_s{config['stages']}_mfma{config['mapping']}"
        )
        implementations[name] = partial(_run, ProducerSchedule(config))
        metadata[name] = Variant(
            "Producer schedule; fixed pipeline output and library contraction",
            "FP32 runtime-default projection dots; highest/IEEE library contraction",
            "full-update qualification and timing required per case",
        )
        configs[name] = config
    return implementations, metadata, configs
