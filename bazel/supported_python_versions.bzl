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

"""Module for single-sourcing the supported Python versions."""

# When updating SUPPORTED_PYTHON_VERSIONS (or the latest supported version), also update:
# - .github/workflows/build_wheel.yml
# - .github/workflows/nightly_unpinned_deps.yml
# - noxfile.py
# - requirements/lock_environments.sh
# LINT.IfChange(supported_python_versions)
SUPPORTED_PYTHON_VERSIONS = [
    "3.11",
    "3.12",
    "3.13",
    "3.14",
]
# LINT.ThenChange(
#     ../pyproject.toml:requires_python,
#     ../README.md:python_version_support,
#     ../../docs/developers/python_support_policy.md:python_version_support,
# )

# The Python version the standard build defaults to (python.defaults in
# MODULE.bazel). The extra PyTorch versions' torch wheels currently resolve
# only for this version (see //shims/torch:torch_wheel_select.bzl).
# MODULE.bazel cannot load() this file, so its copy of this constant is kept
# in sync by hand, checked by ci/tools/sync_module_bazel_versions.py.
# Pinned back to 3.12 from upstream's 3.14 (f594ebf78, 2026-09-15, "Shift
# TorchTPU default Python baseline from 3.12 to 3.14") so this wheel is usable
# on a coreflow TPU host, whose interpreter is 3.12.
#
# This constant is not just a default: per the note above, the extra PyTorch
# versions' torch wheels resolve ONLY for it. --repo_env=HERMETIC_PYTHON_VERSION
# retargets the default hub (MODULE.bazel gives that one a python_version_env)
# but NOT the extra-version hubs, so with the baseline at 3.14 a cp312 wheel
# still links python-3.14 torch into every non-default glue -- verified by the
# `torch_tpu_pypi_2_12_0_314_torch` strings baked into glue_2_12_0, against
# `..._312_torch` in the August image build. Running torch 2.12.1 selects the
# 2.12 glue, so that mismatch is on the hot path and measured 4% slower.
DEFAULT_PYTHON_VERSION = "3.12"
