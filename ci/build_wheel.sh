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
#
# A script to build Python wheels for TorchTPU in Kokoro, located in the OSS repo.

set -exu -o history -o allexport

echo "===> Starting Python wheel build in Kokoro..."

# Prepare wheel version string with metadata date suffix (similar to GitHub)
WHEEL_VERSION_EXTRAS=".dev$(date +%Y%m%d%H%M%S)"
export WHEEL_VERSION_EXTRAS
echo "WHEEL_VERSION_EXTRAS: ${WHEEL_VERSION_EXTRAS}"

# Define target wheel dir inside Kokoro artifacts folder
KOKORO_ARTIFACTS_DIR="${KOKORO_ARTIFACTS_DIR:-$(pwd)/../../artifacts}"
WHEEL_DIR="${KOKORO_ARTIFACTS_DIR}/dist"
mkdir -p "${WHEEL_DIR}"

# We use the standard ml-build container toolchain to build the wheel hermetically
CONTAINER_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest"

echo "===> Pulling ml-build container: ${CONTAINER_IMAGE}..."
docker pull "${CONTAINER_IMAGE}"

echo "===> Compiling wheel via Bazel inside ml-build container..."
# We map the checked out directory into /workspace and run standard bazel build
# rm -rf dist
mkdir -p dist
docker run --rm \
  -v "$(pwd):/workspace" \
  -w "/workspace" \
  -e WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS}" \
  "${CONTAINER_IMAGE}" \
  bash -c "bazel build -c opt --config=no_rbe //ci/wheel:torch_tpu_wheel --repo_env=WHEEL_VERSION_EXTRAS=\$WHEEL_VERSION_EXTRAS && cp bazel-bin/ci/wheel/torch_tpu-0.1.1*.whl /workspace/dist/"

# Move the built wheels from local dist back to Kokoro artifacts directory
if [[ -d dist && -n "$(ls -A dist/*.whl 2>/dev/null)" ]]; then
  cp dist/*.whl "${WHEEL_DIR}/"
  echo "===> Wheels created and copied to Kokoro artifacts successfully:"
  ls -lh "${WHEEL_DIR}"
else
  echo "ERROR: Python wheel build failed - No .whl files found in dist/"
  exit 1
fi

# Perform inline Twine checks to ensure metadata meets general quality rule
echo "===> Running Twine check over built wheels..."
docker run --rm \
  -v "${WHEEL_DIR}:/dist" \
  "${CONTAINER_IMAGE}" \
  bash -c "uv run --isolated --with twine twine check /dist/*.whl"

echo "===> Kokoro wheel build successful!"
