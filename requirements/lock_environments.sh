#!/bin/bash
# Copyright 2025 Google LLC
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

# This script is used to lock the environments in the torch_tpu repository.
#
# Dependencies are defined in the standard pyproject.toml file. uv allows
# dependencies to be locked in version-specific requirements_3_x.txt formats
# (e.g., requirements_3_12.txt).
#
# Usage:
#   Install uv following the instructions if you haven't already:
#     https://docs.astral.sh/uv/getting-started/installation/
#   Navigate to the directory containing the pyproject.toml file.
#   Run this script.
#     $ ./requirements/lock_environments.sh

set -e

script_dir=$(dirname "$(readlink -f "$0")")
working_dir="$script_dir/.."
cd "$working_dir"

PYTHON_PLATFORM="x86_64-manylinux_2_31"
# Ignore host-level uv configuration (e.g. /etc/uv/uv.toml on Cloudtop) so
# lockfiles always resolve directly against PyPI and the indexes in
# pyproject.toml.
export UV_NO_CONFIG=1

# Loop over supported Python versions and generate the full-environment locks.
for version in "3.11" "3.12" "3.13" "3.14"; do
  version_und=$(echo "$version" | tr '.' '_')
  REQUIREMENTS_FILE="requirements/requirements_${version_und}.txt"

  echo "Generating lock file for Python $version -> $REQUIREMENTS_FILE"
  if [ -f "$REQUIREMENTS_FILE" ]; then
    rm "$REQUIREMENTS_FILE"
  fi

  # See https://docs.astral.sh/uv/reference/cli/#uv-pip-compile for more details.
  uv pip compile pyproject.toml \
    --all-extras \
    --python-version "$version" \
    --python-platform "$PYTHON_PLATFORM" \
    --resolution lowest-direct \
    --generate-hashes \
    --output-file "$REQUIREMENTS_FILE"

  REQUIREMENTS_CUDA_FILE="requirements/requirements_cuda_${version_und}.txt"
  echo "Generating CUDA lock file for Python $version -> $REQUIREMENTS_CUDA_FILE"
  if [ -f "$REQUIREMENTS_CUDA_FILE" ]; then
    rm "$REQUIREMENTS_CUDA_FILE"
  fi

  # --no-sources ignores the [tool.uv.sources] pytorch-cpu index pins, so
  # torch resolves as the CUDA build from PyPI instead of the +cpu wheel.
  uv pip compile pyproject.toml \
    --no-sources \
    --all-extras \
    --python-version "$version" \
    --python-platform "$PYTHON_PLATFORM" \
    --resolution lowest-direct \
    --generate-hashes \
    --index-url https://pypi.org/simple \
    --output-file "$REQUIREMENTS_CUDA_FILE"
done

# Full-environment lock for the latest supported Python version pinning the upper
# bound of all dependencies.
# Keep in sync with the latest version in bazel/supported_python_versions.bzl.
LATEST_PYTHON_VERSION="3.14"
latest_version_und=$(echo "$LATEST_PYTHON_VERSION" | tr '.' '_')
REQUIREMENTS_FILE="requirements/requirements_${latest_version_und}_latest.txt"

echo "Generating latest lock file for Python $LATEST_PYTHON_VERSION -> $REQUIREMENTS_FILE"
if [ -f "$REQUIREMENTS_FILE" ]; then
  rm "$REQUIREMENTS_FILE"
fi

uv pip compile pyproject.toml \
  --all-extras \
  --python-version "$LATEST_PYTHON_VERSION" \
  --python-platform "$PYTHON_PLATFORM" \
  --resolution highest \
  --generate-hashes \
  --output-file "$REQUIREMENTS_FILE"

# Torch-only locks for the extra PyTorch versions the multi-ABI wheel ships a
# glue for. The default version's torch is already pinned by the full locks
# above (it comes from pyproject.toml), so only the extras need a standalone
# lock. The version list is read from the single source of truth in
# //bazel:pytorch_versions.bzl so it stays in step with the pip hubs in
# MODULE.bazel.
for version in $(uv run --no-project bazel/pytorch_versions.py EXTRA_PYTORCH_VERSIONS); do
  version_und=$(echo "$version" | tr '.' '_')
  REQUIREMENTS_FILE="requirements/requirements_torch_${version_und}.txt"

  echo "Generating torch-only lock for PyTorch $version -> $REQUIREMENTS_FILE"
  if [ -f "$REQUIREMENTS_FILE" ]; then
    rm "$REQUIREMENTS_FILE"
  fi

  # Lock torch itself and nothing else: torch's runtime dependencies are
  # ABI-independent and already covered by the full environment locks above, so
  # --no-deps keeps this file to just "torch==<version>+cpu". --universal lists
  # every wheel for the version (all supported Python versions and platforms);
  # pip/rules_python selects the matching wheel at fetch time.
  # --index-strategy unsafe-best-match lets the "+cpu" local build resolve from
  # the PyTorch CPU index while PyPI stays the primary index.
  uv pip compile - \
    --no-deps \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --python-platform "$PYTHON_PLATFORM" \
    --index-strategy unsafe-best-match \
    --generate-hashes \
    --output-file "$REQUIREMENTS_FILE" <<EOF
torch==${version}+cpu
EOF
done

# Torch-only lock for the nightly glue. Unlike the release locks above this is
# not pinned by us: each run re-resolves to the newest dev build on the nightly
# index and pins that, so refreshing the nightly snapshot the glue is built
# against is just re-running this script. The nightly index only carries
# recent snapshots, so this needs re-running periodically anyway.
REQUIREMENTS_FILE="requirements/requirements_torch_nightly.txt"
echo "Generating torch-only lock for PyTorch nightly -> $REQUIREMENTS_FILE"
if [ -f "$REQUIREMENTS_FILE" ]; then
  rm "$REQUIREMENTS_FILE"
fi

uv pip compile - \
  --no-deps \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  --python-platform "$PYTHON_PLATFORM" \
  --index-strategy unsafe-best-match \
  --prerelease allow \
  --generate-hashes \
  --output-file "$REQUIREMENTS_FILE" <<EOF
torch
EOF

# The nightly glue is named by the release its snapshot leads up to
# (NIGHTLY_TORCH_VERSION in bazel/pytorch_versions.bzl, mirrored by hand in
# MODULE.bazel like the version lists). When the fresh pin rolls into a new
# release cycle those constants must roll with it, so fail loudly rather than
# let a stale name ship a glue built against a different ABI.
pinned_snapshot=$(grep -m1 '^torch==' "$REQUIREMENTS_FILE" | cut -d'=' -f3)
pinned_release=$(echo "$pinned_snapshot" | cut -d'+' -f1 | cut -d'.' -f1-3)
declared=$(uv run --no-project bazel/pytorch_versions.py NIGHTLY_TORCH_VERSION)
if [ "$pinned_release" != "$declared" ]; then
  echo "ERROR: the nightly lock now pins torch $pinned_snapshot (release $pinned_release)," >&2
  echo "but NIGHTLY_TORCH_VERSION is $declared. Update it in bazel/pytorch_versions.bzl" >&2
  echo "and MODULE.bazel to $pinned_release." >&2
  exit 1
fi

# Python 3.15 caveats:
# - grain and array-record (a required Linux dependency of tensorflow-datasets)
#   do not publish source distributions or Python 3.15 wheels on PyPI, and
#   flax (via tensorstore/uvloop) and dm-tree do not yet publish Python 3.15
#   binary wheels for their C/C++ extensions, so we exclude them when locking
#   Python 3.15.
# - Under --resolution lowest-direct, uv otherwise resolves older minimum
#   versions that predate Python 3.15 and only ship source distributions;
#   --only-binary forces resolution to versions with prebuilt cp315 wheels.
# - PyTorch < 2.15 blocks torch.compile on Python 3.15 and does not publish
#   cp315 CUDA wheels on PyPI, so constrain torch>=2.15.0 and
#   torchvision>=0.29.2.
# - datasets<5.0.0 constrains pyarrow<24.0.0 (which has no cp315 wheel);
#   huggingface-hub<1.19.0 imports dataclasses._MISSING_TYPE (removed in Python
#   3.15); torchax<0.0.14.dev20261002 accesses torch.ops.aten.prod.dim_Dimname
#   and torch.ops.aten.qr (removed in PyTorch 2.15).
for variant in "cpu" "cuda"; do
  if [ "$variant" = "cpu" ]; then
    REQUIREMENTS_FILE="requirements/requirements_3_15.txt"
    VARIANT_ARGS=()
  else
    REQUIREMENTS_FILE="requirements/requirements_cuda_3_15.txt"
    VARIANT_ARGS=(--no-sources --index-url https://pypi.org/simple)
  fi
  echo "Generating ${variant^^} lock file for Python 3.15 -> $REQUIREMENTS_FILE"
  rm -f "$REQUIREMENTS_FILE"
  uv pip compile pyproject.toml \
    "${VARIANT_ARGS[@]}" \
    --all-extras \
    --python-version 3.15 \
    --python-platform "$PYTHON_PLATFORM" \
    --resolution lowest-direct \
    --generate-hashes \
    --only-binary numpy,scipy,pillow,matplotlib,grpcio-tools,pydantic-core,pyarrow,aiohttp \
    --constraint <(printf "datasets>=5.0.0\nhuggingface-hub>=1.19.0\ntorch>=2.15.0\ntorchvision>=0.29.2\n") \
    --overrides <(printf "grain ; python_version < '3.15'\ntensorflow-datasets ; python_version < '3.15'\nflax ; python_version < '3.15'\ndm-tree ; python_version < '3.15'\ntorchax>=0.0.14.dev20261002\n") \
    --output-file "$REQUIREMENTS_FILE"
done
