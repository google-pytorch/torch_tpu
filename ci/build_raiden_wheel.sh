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
# Kokoro orchestration for the tpu_sync torch wheel build, located in the OSS repo.

set -exu -o history -o allexport

echo "===> Starting tpu_sync torch wheel build in Kokoro..."

# Prepare wheel version string with metadata date suffix (similar to GitHub).
# Unlike the date-only .devYYYYMMDD default of ci/build_wheels.sh, it includes
# the time of day, so that wheels built on the same day have different file
# names: Artifact Registry rejects a second upload of the same file name.
WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS:-.dev$(date +%Y%m%d%H%M%S)}"
export WHEEL_VERSION_EXTRAS
echo "WHEEL_VERSION_EXTRAS: ${WHEEL_VERSION_EXTRAS}"
export TORCH_TPU_SRC="$(pwd)"

# Define target wheel dir inside Kokoro artifacts folder
KOKORO_ARTIFACTS_DIR="${KOKORO_ARTIFACTS_DIR:-$(pwd)/../../artifacts}"
export KOKORO_ARTIFACTS_DIR
WHEEL_DIR="${KOKORO_ARTIFACTS_DIR}/dist"
export WHEEL_DIR
mkdir -p "${WHEEL_DIR}"

CONTAINER_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIDEN_DIR="${RAIDEN_DIR:-${SCRIPT_DIR}/../../tpu_raiden}"

if [[ ! -d "${RAIDEN_DIR}" ]]; then
  echo "===> tpu_sync directory not found at '${RAIDEN_DIR}'. Cloning from GitHub..."
  git clone https://github.com/google/tpu-sync.git "${RAIDEN_DIR}"
fi

if [[ ! -f "${RAIDEN_DIR}/ci/build_wheel.sh" ]]; then
  echo "ERROR: RAIDEN_DIR/ci/build_wheel.sh not found at '${RAIDEN_DIR}'." >&2
  exit 1
fi

echo "===> Invoking tpu_sync wheel build: ${RAIDEN_DIR}/ci/build_wheel.sh..."
bash "${RAIDEN_DIR}/ci/build_wheel.sh" torch

# Check that the wheel was created
if ! ls "${WHEEL_DIR}"/tpu_sync_torch-*.whl >/dev/null 2>&1; then
  echo "ERROR: tpu_sync wheel build failed - No tpu_sync_torch-*.whl files found in ${WHEEL_DIR}" >&2
  exit 1
fi

# Perform inline Twine checks to ensure metadata meets general quality rules
echo "===> Running Twine check over tpu_sync wheels..."
docker run --rm \
  -v "${WHEEL_DIR}:/dist" \
  "${CONTAINER_IMAGE}" \
  bash -c "uv run --isolated --with twine twine check /dist/tpu_sync_torch-*.whl"

echo "===> Kokoro tpu_sync wheel build successful!"

echo "===> Uploading tpu_sync wheels via upload_wheel.sh..."
bash "${SCRIPT_DIR}/upload_wheel.sh" "tpu_sync_torch-*.whl"
