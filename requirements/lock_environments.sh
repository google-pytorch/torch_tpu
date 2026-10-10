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

# Generate the full-environment CPU and (optionally) CUDA locks for a single
# Python version.
# Usage: lock_environment <python_version> [--skip-cuda] [<extra_uv_args>...]
lock_environment() {
  local version="$1"
  shift
  local skip_cuda=false
  local extra_uv_args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --skip-cuda)
        skip_cuda=true
        shift
        ;;
      *)
        extra_uv_args+=("$1")
        shift
        ;;
    esac
  done

  local version_und
  version_und=$(echo "$version" | tr '.' '_')
  local requirements_file="requirements/requirements_${version_und}.txt"

  echo "Generating lock file for Python $version -> $requirements_file"
  if [ -f "$requirements_file" ]; then
    rm "$requirements_file"
  fi

  # See https://docs.astral.sh/uv/reference/cli/#uv-pip-compile for more details.
  uv pip compile pyproject.toml \
    --all-extras \
    --python-version "$version" \
    --python-platform "$PYTHON_PLATFORM" \
    --resolution lowest-direct \
    --generate-hashes \
    "${extra_uv_args[@]}" \
    --output-file "$requirements_file"

  if [ "$skip_cuda" = false ]; then
    local requirements_cuda_file="requirements/requirements_cuda_${version_und}.txt"
    echo "Generating CUDA lock file for Python $version -> $requirements_cuda_file"
    if [ -f "$requirements_cuda_file" ]; then
      rm "$requirements_cuda_file"
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
      "${extra_uv_args[@]}" \
      --output-file "$requirements_cuda_file"
  fi
}

# Loop over supported Python versions and generate the full-environment locks.
for version in "3.11" "3.12" "3.13" "3.14"; do
  lock_environment "$version"
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
  # the PyTorch CPU (or RC test) index while PyPI stays the primary index.
  # TODO(b/571958437): `--extra-index-url https://download.pytorch.org/whl/test/cpu`
  # is only needed while PyTorch 2.15.0 is a release candidate. Remove it and
  # re-run this script once PyTorch 2.15.0 final is published to /whl/cpu.
  uv pip compile - \
    --no-deps \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://download.pytorch.org/whl/test/cpu \
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
# - PyTorch does not yet publish Python 3.15 CUDA wheels on PyPI (--skip-cuda).
# - PyTorch < 2.15 blocks torch.compile on Python 3.15, so pin torch==2.15.0+cpu
#   (matching the 2.15.0 glue in requirements_torch_2_15_0.txt) and
#   torchvision>=0.29.2 from the PyTorch CPU and RC test indexes.
# - grain and array-record (a required Linux dependency of tensorflow-datasets)
#   do not publish source distributions or Python 3.15 wheels on PyPI, and
#   flax (via tensorstore/uvloop), dm-tree, and datasets (via pyarrow/aiohttp)
#   do not yet publish Python 3.15 binary wheels for their C/C++ extensions, so
#   we exclude them when locking Python 3.15.
# - Under --resolution lowest-direct, uv otherwise resolves older minimum
#   versions that predate Python 3.15 and only ship source distributions;
#   --only-binary forces resolution to versions with prebuilt cp315 wheels.
# - libtpu==0.0.46 (the earliest cp315 wheel) has a dynamic-shape host DMA
#   alignment regression and multi-process TPU initialization hang fixed in
#   libtpu>=0.0.48; huggingface-hub<1.19.0 imports dataclasses._MISSING_TYPE
#   (removed in Python 3.15); torchax<0.0.14.dev20261002 accesses
#   torch.ops.aten.prod.dim_Dimname and torch.ops.aten.qr (removed in PyTorch
#   2.15).
lock_environment "3.15" --skip-cuda \
  --no-sources \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://download.pytorch.org/whl/test/cpu \
  --index-strategy unsafe-best-match \
  --only-binary numpy,scipy,pillow,matplotlib,grpcio-tools,pydantic-core \
  --overrides <(printf "grain ; python_version < '3.15'\ntensorflow-datasets ; python_version < '3.15'\nflax ; python_version < '3.15'\ndm-tree ; python_version < '3.15'\ndatasets ; python_version < '3.15'\nlibtpu>=0.0.49\nhuggingface-hub>=1.19.0\ntorch==2.15.0+cpu\ntorchvision>=0.29.2\ntorchax>=0.0.14.dev20261002\n")
