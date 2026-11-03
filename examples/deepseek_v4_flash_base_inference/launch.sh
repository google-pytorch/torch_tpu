#!/bin/bash
# Copyright 2026 The TorchTPU Authors
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

# A launch script to run DeepSeek-V4-Flash LLM inference fallbacks and generation on Cloud TPU.
#
# Usage:
#   ./launch.sh [test|generate|all]
#
# Examples:
#   # Run fallback verification unit tests:
#   ./launch.sh test
#
#   # Run standalone autoregressive token generation:
#   ./launch.sh generate
#
#   # Run both test and generate:
#   ./launch.sh all

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Add repo root to PYTHONPATH if running from a local checkout or container.
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${SCRIPT_DIR}:${PYTHONPATH}"

# Configure TPU environment for single-device execution.
export PYTHONUNBUFFERED=1
export TPU_LAUNCH_BLOCKING=1
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_dvfs_p_state=7"

# Select python binary (falls back to local venv if system python lacks torch)
PYTHON_BIN="${PYTHON:-python3}"
if ! "${PYTHON_BIN}" -c "import torch" &>/dev/null; then
  if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  elif [ -x "${HOME}/.venv/bin/python" ]; then
    PYTHON_BIN="${HOME}/.venv/bin/python"
  fi
fi

MODE="${1:-all}"

case "${MODE}" in
  test)
    echo "Running fallback verification tests..."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/fallbacks_test.py"
    ;;
  generate)
    echo "Running functional generation harness..."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/generate.py"
    ;;
  all)
    echo "Running fallback tests and generation harness..."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/fallbacks_test.py"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/generate.py"
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Usage: ./launch.sh [test|generate|all]"
    exit 1
    ;;
esac
