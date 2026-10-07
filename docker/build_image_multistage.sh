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

# Script to build torch-tpu docker images using multi-stage Dockerfile.
# It builds both the base image and the final image.

set -e

# Usage: ./build_image_multistage.sh [--wheel <path_to_wheel>] [--examples <path_to_examples>] [LOCAL_REPO_ROOT] [IMAGE_TAG_PREFIX]
# LOCAL_REPO_ROOT: Path to the root of the torch_tpu repository (optional,
#                  defaults to parent dir).
# IMAGE_TAG_PREFIX: Prefix for the built images (optional, defaults to
#                   torch-tpu).
#                   Will produce ${IMAGE_TAG_PREFIX}-base and ${IMAGE_TAG_PREFIX}.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../" && pwd)"

LOCAL_REPO_ROOT="${DEFAULT_REPO_ROOT}"
IMAGE_TAG_PREFIX="torch-tpu"
WHEEL_PATH=""
EXAMPLES_PATH=""

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    -w|--wheel)
      if [ -z "${2:-}" ]; then
        echo "Error: --wheel requires a file path argument"
        exit 1
      fi
      WHEEL_PATH="$2"
      shift 2
      ;;
    --wheel=*)
      WHEEL_PATH="${1#*=}"
      shift
      ;;
    -e|--examples)
      if [ -z "${2:-}" ]; then
        echo "Error: --examples requires a directory path argument"
        exit 1
      fi
      EXAMPLES_PATH="$2"
      shift 2
      ;;
    --examples=*)
      EXAMPLES_PATH="${1#*=}"
      shift
      ;;
    *)
      if [ -z "${LOCAL_REPO_ROOT_SET}" ]; then
        LOCAL_REPO_ROOT="$1"
        LOCAL_REPO_ROOT_SET=1
      elif [ -z "${IMAGE_TAG_PREFIX_SET}" ]; then
        IMAGE_TAG_PREFIX="$1"
        IMAGE_TAG_PREFIX_SET=1
      else
        echo "Unknown argument: $1"
        exit 1
      fi
      shift
      ;;
  esac
done

ABS_REPO_ROOT=$(cd "${LOCAL_REPO_ROOT}" && pwd)
DOCKERFILE="docker/Dockerfile.multistage"

if [ ! -f "${ABS_REPO_ROOT}/${DOCKERFILE}" ]; then
  echo "Error: Dockerfile not found at ${ABS_REPO_ROOT}/${DOCKERFILE}"
  exit 1
fi

mkdir -p "${ABS_REPO_ROOT}/dist"
if [ -n "${WHEEL_PATH}" ]; then
  if [ ! -f "${WHEEL_PATH}" ]; then
    echo "Error: Wheel file not found at ${WHEEL_PATH}"
    exit 1
  fi
  ABS_WHEEL_PATH=$(readlink -f "${WHEEL_PATH}")
  echo "===> Using provided prebuilt wheel: ${ABS_WHEEL_PATH}"
  rm -f "${ABS_REPO_ROOT}/dist"/*.whl
  cp -f "${ABS_WHEEL_PATH}" "${ABS_REPO_ROOT}/dist/"
else
  # No wheel explicitly provided; clear dist so Dockerfile fetches latest nightly from Artifact Registry
  rm -f "${ABS_REPO_ROOT}/dist"/*.whl
fi

if [ -n "${EXAMPLES_PATH}" ]; then
  if [ ! -d "${EXAMPLES_PATH}" ]; then
    echo "Error: Examples directory not found at ${EXAMPLES_PATH}"
    exit 1
  fi
  ABS_EXAMPLES_PATH=$(cd "${EXAMPLES_PATH}" && pwd)
  echo "===> Using provided examples directory: ${ABS_EXAMPLES_PATH}"
  if [ "${ABS_EXAMPLES_PATH}" != "${ABS_REPO_ROOT}/examples" ]; then
    mkdir -p "${ABS_REPO_ROOT}/examples"
    cp -r "${ABS_EXAMPLES_PATH}"/* "${ABS_REPO_ROOT}/examples/"
  fi
fi

echo "===> Building Docker images with prefix '${IMAGE_TAG_PREFIX}' using local source at ${ABS_REPO_ROOT}..."

# Build base image
echo "===> Building Base Image..."
docker build \
  --progress=plain \
  --target base \
  -f "${ABS_REPO_ROOT}/${DOCKERFILE}" \
  -t "${IMAGE_TAG_PREFIX}-base:latest" \
  "${ABS_REPO_ROOT}"

# Build final image
echo "===> Building Final Image..."
SECRET_BUILD_ARGS=()
if [ -z "${WHEEL_PATH}" ]; then
  if command -v gcloud &>/dev/null; then
    TOKEN="$(gcloud auth print-access-token 2>/dev/null || true)"
    if [ -n "${TOKEN}" ]; then
      SECRET_BUILD_ARGS+=(--build-arg "GCLOUD_TOKEN=${TOKEN}")
    fi
  fi
fi

docker build \
  --progress=plain \
  "${SECRET_BUILD_ARGS[@]}" \
  -f "${ABS_REPO_ROOT}/${DOCKERFILE}" \
  -t "${IMAGE_TAG_PREFIX}:latest" \
  "${ABS_REPO_ROOT}"

echo "===> Done! Images built successfully."
echo "Base image: ${IMAGE_TAG_PREFIX}-base:latest"
echo "Final image: ${IMAGE_TAG_PREFIX}:latest"
