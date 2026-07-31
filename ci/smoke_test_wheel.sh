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
# A script to run nox smoke tests against built Python wheels in Kokoro.

set -exu -o history -o allexport

echo "===> Running noxfile smoke test over built wheel..."

KOKORO_ARTIFACTS_DIR="${KOKORO_ARTIFACTS_DIR:-$(pwd)/../../artifacts}"
WHEEL_DIR="${KOKORO_ARTIFACTS_DIR}/dist"
CONTAINER_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest"

PYTHON_VERSION="${PYTHON_VERSION:?PYTHON_VERSION must be set}"

docker run --rm \
  -v "${WHEEL_DIR}:/dist" \
  -v "$(pwd):/workspace" \
  -w "/workspace" \
  "${CONTAINER_IMAGE}" \
  bash -c "set -ex
    mkdir -p dist
    cp /dist/*.whl dist/ || true
    echo '===> Running CPU smoke test for Python ${PYTHON_VERSION}...'
    uvx nox --force-python \"${PYTHON_VERSION}\" -s \"smoke_test(cpu-nightly)\"
  "

echo "===> Nox smoke test passed successfully!"
