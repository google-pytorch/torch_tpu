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
all sharing a single XLA/MLIR base library. A version is added here only when
that PyTorch release breaks ABI compatibility with the previous glue; the
runtime dispatch (`torch_tpu/_versioned_so_loader.py`) then serves each install
from the newest glue at or below its version, so every release between two
entries -- and everything after the last one -- loads the preceding glue. The
`intermediate_patch_abi` nox sessions keep the list honest: when a release in
INTERMEDIATE_PATCH_VERSIONS starts failing them, it broke ABI and must move
into this list.

This file is deliberately data-only (plain assignments, no `load()`): it is
loaded by Starlark (`shims/build_files/torch_version.bzl`, and thence
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
    "2.12.0",
    "2.13.0",
]

# Every released PyTorch version the wheel is built against -- the default plus
# the extras -- one thin glue `.so` each.
WHEEL_TORCH_VERSIONS = [DEFAULT_TORCH_VERSION] + EXTRA_PYTORCH_VERSIONS

# The release version the PyTorch nightly channel currently leads up to. The
# nightly glue is one more glue like any other -- named by this version (glue
# package glue_2_14_0, config setting //:torch_2_14_0, artifacts like
# env_2_14_0.so), and served by the runtime dispatch through the ordinary
# newest-at-or-below rule (a dev build's version triple is the release it
# precedes). Only its torch differs: it is built against the dev snapshot
# pinned in requirements/requirements_torch_nightly.txt, so this constant must
# stay the release triple of that pin -- lock_environments.sh refreshes the
# pin and errors if this no longer matches. Its one packaging difference: the
# //:include_nightly_glue flag (on by default, off under --config
# wheel_release) selects it out of artifacts published to an index, since a
# transient dev snapshot must not ship to PyPI.
NIGHTLY_TORCH_VERSION = "2.14.0"

# Every glue the build can produce: the released versions plus the nightly
# channel's. This is what the per-glue build machinery (target fan-out, glue
# packages, torch shim selects) iterates over; WHEEL_TORCH_VERSIONS remains
# the released subset actually published in every wheel.
GLUE_TORCH_VERSIONS = WHEEL_TORCH_VERSIONS + [NIGHTLY_TORCH_VERSION]

# Released PyTorch versions the wheel ships no exact glue for; the runtime
# dispatch serves each from the newest older glue. Every released version in
# the supported range that is not in WHEEL_TORCH_VERSIONS belongs here, so the
# `intermediate_patch_abi` nox sessions (noxfile.py) prove the served glue
# really is ABI-compatible -- a failure here means the version broke ABI and
# needs a glue of its own.
INTERMEDIATE_PATCH_VERSIONS = [
    "2.12.1",
]

# The //:_torch_version value meaning "no specific PyTorch version" -- the
# flag's default, and what the pinning transitions reset to. Everything that
# does not need a particular torch ABI builds under it: the plain build, the
# shared XLA/MLIR backend, and the default version's glue (whose torch comes
# from the default pip hub, needing no reconfiguration). It must never appear
# in WHEEL_TORCH_VERSIONS.
SENTINEL_TORCH_VERSION = "0.0.0"
