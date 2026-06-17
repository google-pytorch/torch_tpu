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
# dependencies to be locked in the requirements.txt format.
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

# Ensure versions match those in MODULE.bazel
PYTORCH_VERSIONS=("2.10.0" "2.11.0" "2.12.0")

# Loop over supported Python versions and PyTorch versions to generate lock files
for py_version in "3.11" "3.12" "3.13" "3.14"; do
  py_version_und=$(echo "$py_version" | tr '.' '_')

  for torch_version in "${PYTORCH_VERSIONS[@]}"; do
    torch_version_und=$(echo "$torch_version" | tr '.' '_')
    REQUIREMENTS_FILE="requirements/requirements_${py_version_und}_torch_${torch_version_und}.txt"

    echo "Generating lock file for Python $py_version and PyTorch $torch_version -> $REQUIREMENTS_FILE"
    if [ -f "$REQUIREMENTS_FILE" ]; then
      rm "$REQUIREMENTS_FILE"
    fi

    # Create a temporary constraints file for the specific PyTorch version
    CONSTRAINT_FILE=$(mktemp)
    echo "torch==${torch_version}" >"$CONSTRAINT_FILE"

    # See https://docs.astral.sh/uv/reference/cli/#uv-pip-compile for more details.
    uv pip compile pyproject.toml \
      --all-extras \
      --python-version "$py_version" \
      --python-platform x86_64-manylinux_2_31 \
      --resolution lowest-direct \
      --generate-hashes \
      --constraint "$CONSTRAINT_FILE" \
      --output-file "$REQUIREMENTS_FILE"

    rm "$CONSTRAINT_FILE"
  done
done
