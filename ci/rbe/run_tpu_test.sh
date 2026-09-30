#!/usr/bin/env bash
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

set -euo pipefail

# Default configuration values
DEFAULT_TARGET="//tests:autoload_test_tpu"
DEFAULT_INSTANCE="projects/rbe-tpu-oss/instances/default_instance_west1"
DEFAULT_ENDPOINT="grpcs://us-west1-remotebuildexecution.googleapis.com"
DEFAULT_TPU_POOL="ci_tpu_v5_full_rbe"
DEFAULT_OUTPUT="all"

usage() {
  local exit_code="${1:-0}"
  cat <<EOF
Usage: $(basename "$0") [OPTIONS] [BAZEL_TARGET] [-- BAZEL_FLAGS...]

Runs a TorchTPU test on Google Cloud Remote Build Execution (RBE) in us-west1.
Routes compile and link actions to the CPU "default" worker pool (n2-standard-32),
and routes hardware test execution to the dedicated TPU v5e pool.

Target:
  BAZEL_TARGET            Target to test (default: ${DEFAULT_TARGET})
                          Common presubmit targets:
                            //tests:autoload_test_tpu
                            //tests:empty_test
                            //tests:env_test
                            //tests:synchronize_test
                            //tests:tensor_test
                            //tests:tpu_errors_test

Options:
  --pool=NAME             TPU pool name (default: ${DEFAULT_TPU_POOL})
                          Options: ci_tpu_v5_full_rbe, ci_tpu_v5e_reserved
  --instance=NAME         RBE instance name (default: ${DEFAULT_INSTANCE})
  --endpoint=URI          RBE gRPC endpoint (default: ${DEFAULT_ENDPOINT})
  --output=MODE           Test output mode (default: ${DEFAULT_OUTPUT})
                          Options: all, errors, summary, streamed
  -h, --help              Show this help message and exit

Examples:
  # Run the default presubmit test
  $(basename "$0")

  # Run a specific presubmit test
  $(basename "$0") //tests:empty_test

  # Run on the reserved single-worker pool
  $(basename "$0") --pool=ci_tpu_v5e_reserved //tests:autoload_test_tpu

  # Pass additional flags to Bazel
  $(basename "$0") //tests:autoload_test_tpu -- --test_arg=--verbose
EOF
  exit "${exit_code}"
}

# Parse options
TARGET=""
TPU_POOL="${DEFAULT_TPU_POOL}"
INSTANCE="${DEFAULT_INSTANCE}"
ENDPOINT="${DEFAULT_ENDPOINT}"
OUTPUT="${DEFAULT_OUTPUT}"
EXTRA_BAZEL_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pool=*)
      TPU_POOL="${1#*=}"
      shift
      ;;
    --pool)
      TPU_POOL="$2"
      shift 2
      ;;
    --instance=*)
      INSTANCE="${1#*=}"
      shift
      ;;
    --instance)
      INSTANCE="$2"
      shift 2
      ;;
    --endpoint=*)
      ENDPOINT="${1#*=}"
      shift
      ;;
    --endpoint)
      ENDPOINT="$2"
      shift 2
      ;;
    --output=*)
      OUTPUT="${1#*=}"
      shift
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    --)
      shift
      EXTRA_BAZEL_ARGS+=("$@")
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage 1
      ;;
    *)
      if [[ -z "${TARGET}" ]]; then
        TARGET="$1"
      else
        EXTRA_BAZEL_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

TARGET="${TARGET:-${DEFAULT_TARGET}}"

# Verify Application Default Credentials
if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
  echo "Error: Application Default Credentials not found or expired." >&2
  echo "Run: gcloud auth application-default login" >&2
  exit 1
fi

# Select execution platform matching requested pool
case "${TPU_POOL}" in
  ci_tpu_v5_full_rbe)
    PLATFORM="//bazel/platforms:rbe_tpu_v5e_platform"
    ;;
  ci_tpu_v5e_reserved)
    PLATFORM="//bazel/platforms:rbe_tpu_v5e_reserved_platform"
    ;;
  *)
    echo "Warning: Unrecognized TPU pool '${TPU_POOL}'. Defaulting to //bazel/platforms:rbe_tpu_v5e_platform." >&2
    PLATFORM="//bazel/platforms:rbe_tpu_v5e_platform"
    ;;
esac

echo "============================================================"
echo " TorchTPU RBE Hardware Test Runner"
echo "============================================================"
echo " Target:                  ${TARGET}"
echo " RBE Instance:            ${INSTANCE}"
echo " Regional Endpoint:       ${ENDPOINT}"
echo " CPU Compilation Pool:    default (n2-standard-32)"
echo " TPU Hardware Test Pool:  ${TPU_POOL} (ct5lp-hightpu-1t)"
echo " Execution Platform:      ${PLATFORM}"
echo " Test Output Mode:        ${OUTPUT}"
echo "============================================================"
echo

# Run Bazel test with dual-pool routing
exec bazel test "${TARGET}" \
  --remote_executor="${ENDPOINT}" \
  --remote_instance_name="${INSTANCE}" \
  --extra_execution_platforms="${PLATFORM}" \
  --google_default_credentials \
  --jobs=100 \
  --remote_timeout=3600 \
  --test_timeout=3600 \
  --test_output="${OUTPUT}" \
  "${EXTRA_BAZEL_ARGS[@]+"${EXTRA_BAZEL_ARGS[@]}"}"
