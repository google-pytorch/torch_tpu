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
# Builds the torch_tpu wheel for one or more Python versions.
#
# This is the single source of truth for how a torch_tpu wheel is built:
# every meaningful build flag lives here, so the wheels the CI smoke tests
# exercise and the wheels the Kokoro publisher uploads come from the same
# invocation. Callers (.github/actions/build-wheel, ci/build_wheel.sh) only
# orchestrate the environment: where the container runs, where the wheels
# go, and environment-specific bazel flags (e.g. presubmit vs nightly configs).
#
# Usage:
#   ci/build_wheels.sh [--release] [--output-dir DIR] PY_VERSION... [-- BAZEL_ARG...]
#
#   --release       Build a release wheel: no .devYYYYMMDD version suffix.
#                   The default is a dev build, whose suffix is computed once
#                   here so every Python version in one invocation shares the
#                   same version string.
#   --output-dir    Directory the built wheels are copied into (default: dist).
#   PY_VERSION...   Python versions to build for, e.g. "3.11 3.12".
#   BAZEL_ARG...    Extra bazel flags appended to the build (after `--`).

set -euo pipefail

RELEASE=false
OUTPUT_DIR="dist"
PY_VERSIONS=()
EXTRA_BAZEL_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release)
      RELEASE=true
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_BAZEL_ARGS=("$@")
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      PY_VERSIONS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#PY_VERSIONS[@]} -eq 0 ]]; then
  echo "No Python versions given." >&2
  echo "Usage: $0 [--release] [--output-dir DIR] PY_VERSION... [-- BAZEL_ARG...]" >&2
  exit 2
fi

# One version suffix per invocation, so all wheels built together carry the
# same version. Release wheels carry no suffix (the base version comes from
# pyproject.toml).
if [[ "${RELEASE}" == "true" ]]; then
  WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS:-}"
else
  WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS:-.dev$(date +%Y%m%d)}"
fi
echo "===> WHEEL_VERSION_EXTRAS: '${WHEEL_VERSION_EXTRAS}'"

mkdir -p "${OUTPUT_DIR}"

for py_ver in "${PY_VERSIONS[@]}"; do
  echo "===> Building wheel for Python ${py_ver}"
  hermetic_py_ver="${py_ver%t}"
  if [[ "${py_ver}" == *t ]]; then
    freethreaded="yes"
  else
    freethreaded="no"
  fi
  # --config=wheel_common is REQUIRED for a working wheel: it sets
  # --//:wheel_build=True, which routes the shared XLA/MLIR backend through
  # the torch_version-reset transition so pywrap factors it into the single
  # libxla_base.so. Without it every per-version common carries a full
  # backend copy and the wheel aborts on import with duplicate static
  # registrations. Caller-supplied flags come after it, so callers can select
  # specific wheel configs (e.g. --config=wheel_nightly) or append
  # --config=no_rbe to strip RBE flags in environments without RBE credentials.
  bazel build -c opt --config=wheel_common //ci/wheel:torch_tpu_wheel \
    --repo_env=WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS}" \
    --repo_env=HERMETIC_PYTHON_VERSION="${hermetic_py_ver}" \
    --define PYTHON_VERSION="${py_ver}" \
    --@rules_python//python/config_settings:py_freethreaded="${freethreaded}" \
    ${EXTRA_BAZEL_ARGS[@]+"${EXTRA_BAZEL_ARGS[@]}"}
  cp bazel-bin/ci/wheel/*.whl "${OUTPUT_DIR}/"
done

echo "===> Built wheels:"
ls -lh "${OUTPUT_DIR}"
