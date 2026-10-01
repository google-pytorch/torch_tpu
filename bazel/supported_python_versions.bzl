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
    "3.14t",
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
DEFAULT_PYTHON_VERSION = "3.14"
