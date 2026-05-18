# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pytest plugin for running upstream PyTorch tests against torch_tpu.

This plugin:
  1. Registers the torch_tpu PrivateUse1 backend before any tests are collected.
  2. Filters collected items to only the PrivateUse1/TPU device variants.
  3. Applies the skip/xfail list at collection time.

It is loaded as a named plugin (NOT as a conftest) because upstream test files
live in the pytorch source tree, where pytest's conftest auto-discovery would
not find ours. The run script adds tests/upstream/ to PYTHONPATH and passes
`-p upstream_plugin` to pytest.

The upstream tests themselves are not vendored — they are imported from the
pytorch checkout pointed to by PYTORCH_SRC_DIR.
"""

from __future__ import annotations

import json
import os

import pytest

# Load torch_tpu *before* upstream test modules import `torch` and snapshot the
# set of registered device types via instantiate_device_type_tests. We do NOT
# put PYTORCH_SRC_DIR on sys.path — `import torch` must resolve to the installed
# wheel (the source tree is unbuilt and missing the generated torch.version
# module). Pytest collects the upstream test files by absolute path.
import torch_tpu  # noqa: F401
from torch_tpu import api as _tpu_api

_tpu_api.tpu_device()  # triggers _loader.load() and rename_privateuse1_backend("tpu")

import skip_list  # sibling module — tests/upstream/ is added to PYTHONPATH by the runner


# Class-name suffix that `instantiate_device_type_tests` appends for our backend.
# After rename_privateuse1_backend("tpu") this is "TPU"; the literal "PRIVATEUSE1"
# is kept as a fallback in case the rename doesn't take effect.
_DEVICE_CLASS_SUFFIXES = ("TPU", "PRIVATEUSE1")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Filter to TPU-device tests and apply the skip/xfail list."""
    kept: list[pytest.Item] = []
    for item in items:
        parts = item.nodeid.split("::")
        if len(parts) < 3:
            continue
        module = parts[0].rsplit("/", 1)[-1].removesuffix(".py")
        class_name = parts[1]
        method = parts[2]

        if not any(class_name.endswith(s) for s in _DEVICE_CLASS_SUFFIXES):
            continue

        hit = skip_list.lookup(module, class_name, method)
        if hit is not None:
            action, reason = hit
            if action == "skip":
                item.add_marker(pytest.mark.skip(reason=reason))
            elif action == "xfail":
                item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
        kept.append(item)

    items[:] = kept


# Incremental per-test result log. When UPSTREAM_RESULTS_LOG is set, every
# completed test appends one JSON line + flush. Robust against pytest crashing
# mid-session (junit XML is only flushed at session end and is therefore lost
# on SIGABRT). The chunked runner aggregates these JSONL files.
_RESULTS_LOG_PATH = os.environ.get("UPSTREAM_RESULTS_LOG")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if _RESULTS_LOG_PATH is None or report.when != "call":
        # report.when is "setup", "call", or "teardown". The "call" phase carries
        # the test's real outcome; skipped tests come through as when="setup".
        if not (report.when == "setup" and report.outcome == "skipped"):
            return
    record = {
        "nodeid": report.nodeid,
        "outcome": report.outcome,
        "duration": getattr(report, "duration", 0.0),
        "when": report.when,
    }
    with open(_RESULTS_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
