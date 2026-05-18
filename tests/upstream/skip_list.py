# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Skip / xfail list for upstream PyTorch tests run against the TPU backend.

Entries are keyed by (test_module, class_name, method_basename), where
method_basename is the method name *without* the device-type suffix that
`instantiate_device_type_tests` appends (e.g. match "test_add" not
"test_add_privateuse1" or "test_add_tpu").

Use SKIP for tests that cannot run today and XFAIL for tests we expect to fix.
"""

from __future__ import annotations

from typing import Literal

Action = Literal["skip", "xfail"]


# (module, class, method_basename) -> (action, reason)
#
# IMPORTANT: TPU devices are exclusive per-process — when a test SIGABRTs the
# TPU compiler, no other process can acquire the TPU until the dead worker is
# fully reaped (~minutes). xdist-style worker restart does not help. The only
# reliable fix is to skip known crashers.
ENTRIES: dict[tuple[str, str, str], tuple[Action, str]] = {
    # Crashes the worker process with SIGABRT inside torch/_refs/_conversions.py.
    # Class suffix is "PRIVATEUSE1" (instantiate_device_type_tests uses
    # base.device_type.upper() before the rename takes effect).
    ("test_unary_ufuncs", "TestUnaryUfuncsPRIVATEUSE1", "test_batch_vs_slicing"):
        ("skip", "SIGABRT in torch._refs._conversions — triage TBD"),

    # Crashes TPU compiler: xprecision_rewriter.cc:6184
    # Check failed: 2 == hlo->shape().tuple_shapes().size()
    # Test name from run 26027364339 (-v output).
    ("test_ops", "TestCommonPRIVATEUSE1", "test_dtypes___getitem__"):
        ("skip", "xprecision_rewriter SIGABRT — triage TBD"),
}


def lookup(module: str, class_name: str, method: str) -> tuple[Action, str] | None:
    """Return (action, reason) for a test, stripping the device suffix on method."""
    # Strip any trailing "_<device>" suffix added by instantiate_device_type_tests.
    # We try the full name first, then a few common strippings.
    for candidate in (method, _strip_device_suffix(method)):
        hit = ENTRIES.get((module, class_name, candidate))
        if hit is not None:
            return hit
    return None


_DEVICE_SUFFIXES = ("_privateuse1", "_privateuseone", "_tpu", "_xla_cuda", "_xla_cpu")


def _strip_device_suffix(method: str) -> str:
    for suffix in _DEVICE_SUFFIXES:
        if method.endswith(suffix):
            return method[: -len(suffix)]
    return method
