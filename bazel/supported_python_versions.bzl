# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for single-sourcing the supported Python versions.

This file is the single source of truth for every Python version the project
builds, tests or ships against.

It is deliberately written so that it is simultaneously valid Starlark *and*
valid Python, and contains nothing but `UPPER_CASE` assignments of literals
and slices of them (in particular: no `load()` statements). That lets the
non-Starlark consumers -- GitHub Actions workflows, shell scripts and
noxfile.py -- read these constants through ci/tools/python_versions.py rather
than duplicating them. ci/tools/python_versions_test.py enforces those
properties; do not add `load()` statements or function calls here.

Consumers that cannot read this file at all (MODULE.bazel, which cannot
load() Starlark; the workflow_dispatch `choice` options in
.github/workflows/run_benchmarks.yml, which GitHub requires to be literals;
the required-check names in copy.bara.sky; and .bazelrc) keep a hand-written
copy that is verified against this file by
ci/tools/check_python_version_sync.py.
"""

SUPPORTED_PYTHON_VERSIONS = [
    "3.11",
    "3.12",
    "3.13",
    "3.14",
]

# The Python version the standard build defaults to (python.defaults in
# MODULE.bazel). The extra PyTorch versions' torch wheels currently resolve
# only for this version (see //shims/torch:torch_wheel_select.bzl).
# MODULE.bazel cannot load() this file, so its copy of this constant is kept
# in sync by hand, checked by ci/tools/check_python_version_sync.py.
DEFAULT_PYTHON_VERSION = "3.14"

# The ends of the supported range. Derived rather than written out so that
# adding or dropping a version is a one-line change to the list above.
OLDEST_PYTHON_VERSION = SUPPORTED_PYTHON_VERSIONS[0]
LATEST_PYTHON_VERSION = SUPPORTED_PYTHON_VERSIONS[-1]

# CPU presubmit builds and tests the two endpoints of the supported range
# only. That is not an arbitrary subset: the oldest version builds the
# release wheel flavor (--config=wheel_release, no PyTorch-nightly glue) and
# the default version builds the nightly-glue flavor, so every PR exercises
# both flavors. The intermediate versions add no flavor coverage and differ
# only in interpreter minor, so they are deferred to postsubmit to keep PR
# latency and runner cost down.
PRESUBMIT_CPU_PYTHON_VERSIONS = [
    OLDEST_PYTHON_VERSION,
    DEFAULT_PYTHON_VERSION,
]

# TPU runner capacity is scarce, so presubmit covers the default version
# only; the full sweep runs in postsubmit and in the nightly TPU workflow.
PRESUBMIT_TPU_PYTHON_VERSIONS = [DEFAULT_PYTHON_VERSION]

# Postsubmit (and any PR carrying the ci:run_all_versions / ci:postsubmit /
# tpu-nightly labels) sweeps the whole supported range.
POSTSUBMIT_PYTHON_VERSIONS = SUPPORTED_PYTHON_VERSIONS
