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
done

# Torch-only locks for the extra PyTorch versions the multi-ABI wheel ships a
# glue for. The default version's torch is already pinned by the full locks
# above (it comes from pyproject.toml), so only the extras need a standalone
# lock. The version list is read from the single source of truth in
# //bazel:pytorch_versions.bzl so it stays in step with the pip hubs in
# MODULE.bazel.
for version in $(uv run bazel/pytorch_versions.py EXTRA_PYTORCH_VERSIONS); do
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
