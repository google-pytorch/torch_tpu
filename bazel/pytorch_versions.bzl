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

"""Single source of truth for the PyTorch versions the multi-ABI wheel targets.

The wheel ships one thin extension glue per version in `WHEEL_TORCH_VERSIONS`,
all sharing a single XLA/MLIR base library. One entry per supported PyTorch
minor, at its latest patch (see the runtime dispatch in
`torch_tpu/_versioned_so_loader.py`).

This file is deliberately data-only (plain assignments, no `load()`): it is
loaded by Starlark (`build_files/torch_version.bzl`, and thence
`shims/pybind11/pybind.bzl`) and also parsed as Python by
`requirements/lock_environments.sh`, which regenerates the per-version torch
locks. `MODULE.bazel` cannot `load()`, so its `EXTRA_PYTORCH_VERSIONS` is kept
in sync by hand -- exactly as `SUPPORTED_PYTHON_VERSIONS` already is.
"""

# The default PyTorch version: the one pinned in pyproject.toml ("torch>=...")
# and locked together with the whole runtime environment in
# requirements_<py>.txt. Its glue is built from the default pip hub, so unlike
# the extras it needs no standalone torch-only lock.
DEFAULT_TORCH_VERSION = "2.11.0"

# Extra PyTorch versions the wheel also ships a glue for. Everything but torch
# is ABI-independent and shared, so each extra version needs only a torch-only
# lock (requirements_torch_<version>.txt) and its own pip hub in MODULE.bazel.
EXTRA_PYTORCH_VERSIONS = [
    "2.12.1",
]

# Every PyTorch version the wheel is built against -- the default plus the
# extras -- one thin glue `.so` each.
WHEEL_TORCH_VERSIONS = [DEFAULT_TORCH_VERSION] + EXTRA_PYTORCH_VERSIONS

# Intermediate patch releases the wheel ships no exact glue for: earlier
# patches of a minor whose latest patch is in WHEEL_TORCH_VERSIONS. The
# runtime dispatch loader serves these from the same-minor glue, relying on
# PyTorch's ABI being stable across patch releases; the
# `intermediate_patch_abi` nox session (noxfile.py) proves that stability in
# practice for each entry here.
INTERMEDIATE_PATCH_VERSIONS = [
    "2.12.0",
]
