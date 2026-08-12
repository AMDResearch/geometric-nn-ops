# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reference (non-Triton) implementations.

One pure-torch function per op, defining its documented semantics. These are the
oracles the Triton paths are tested against, and where the op supports it, what
the public entry point falls back to when the kernels cannot run.
"""
