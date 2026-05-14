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

# Loop over supported Python versions and generate lock files
for version in "3.11" "3.12" "3.14"; do
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
    --python-platform x86_64-manylinux_2_31 \
    --generate-hashes \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    --output-file "$REQUIREMENTS_FILE"
done
