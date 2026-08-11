# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kernel implementations (Triton today; Gluon for hot paths later).

This boundary is an implementation detail: a per-kernel Triton->Gluon swap is
an internal change here, never a public-API change.
"""
