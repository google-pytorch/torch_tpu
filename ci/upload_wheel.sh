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
# A script to upload Python wheels to Artifact Registry.
#
# Usage:
#   ./upload_wheel.sh <WHEEL_PATTERN>
#
# Arguments:
#   WHEEL_PATTERN: Required. Filename glob/pattern of wheels in dist/ to upload
#                  (e.g., "torch_tpu-*.whl", "tpu_raiden_torch-*.whl").
#
# Environment Variables:
#   UPLOAD_WHEEL_TO_AR: Optional. Whether to upload wheels to Artifact Registry.
#                       Defaults to "true". Set to "false" to skip upload.
#   KOKORO_ARTIFACTS_DIR: Optional. Base directory containing the dist/ folder.
#                         Defaults to "$(pwd)/../../artifacts".
#
# Examples:
#   ./upload_wheel.sh "torch_tpu-*.whl"
#   ./upload_wheel.sh "tpu_raiden_torch-*.whl"

set -exu -o history -o allexport

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  echo "ERROR: WHEEL_PATTERN argument is required." >&2
  echo "Usage: $0 <WHEEL_PATTERN>" >&2
  echo "Example: $0 \"torch_tpu-*.whl\"" >&2
  exit 1
fi

# Wheel filename pattern to upload
WHEEL_PATTERN="$1"

# Define target wheel dir inside Kokoro artifacts folder
KOKORO_ARTIFACTS_DIR="${KOKORO_ARTIFACTS_DIR:-$(pwd)/../../artifacts}"
WHEEL_DIR="${KOKORO_ARTIFACTS_DIR}/dist"

CONTAINER_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest"

# Upload to Google Artifact Registry via Twine and keyring auth, mirroring the GitHub action process.
export UPLOAD_WHEEL_TO_AR=${UPLOAD_WHEEL_TO_AR:-true}
if [[ "${UPLOAD_WHEEL_TO_AR}" == "true" ]]; then
  echo "===> Uploading wheels matching '${WHEEL_PATTERN}' to internal Artifact Registry..."
  docker run --rm \
    -v "${WHEEL_DIR}:/dist" \
    "${CONTAINER_IMAGE}" \
    bash -c "
      uv run --isolated \
        --with twine \
        --with keyrings.google-artifactregistry-auth \
        twine upload --repository-url https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-testing-registry/ /dist/${WHEEL_PATTERN}
    "
  echo "===> Wheels uploaded successfully to Artifact Registry!"
fi
