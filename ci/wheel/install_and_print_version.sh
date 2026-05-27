#!/bin/bash
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

set -e
set -o pipefail

PYTHON_VERSION=$1
if [ -z "$PYTHON_VERSION" ]; then
  echo "Error: PYTHON_VERSION argument is required." >&2
  exit 1
fi

uv venv --python "$PYTHON_VERSION"
. .venv/bin/activate

WHEEL_PATH=$(ls dist/torch_tpu-*.whl | head -n 1)
if [ -z "$WHEEL_PATH" ]; then
  echo "Error: TorchTPU wheel file not found in dist/" >&2
  exit 1
fi

if [ -n "$TORCH_SOURCE" ]; then
  uv pip install "$WHEEL_PATH" "$TORCH_SOURCE"
else
  uv pip install "$WHEEL_PATH"
fi

VERSION=$(python ci/wheel/print_torch_version.py | grep "TorchTPU built against PyTorch version:" | awk '{print $NF}')

if [ -n "$GITHUB_ENV" ]; then
  echo "TORCH_TPU_PYTORCH_BUILD_VERSION=$VERSION" >> "$GITHUB_ENV"
fi
echo "Captured and exported TORCH_TPU_PYTORCH_BUILD_VERSION=$VERSION"
