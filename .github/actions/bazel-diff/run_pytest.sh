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

# --- Arguments ---
WORKSPACE_DIR="${1:-$GITHUB_WORKSPACE}"
MODE="${2}" # --full or --diff
TARGETS_FILE="${3:-}"
BAZEL_CONFIG="${4:-}"
EXTRA_FLAGS="${5:-}"

# --- Configuration ---
# Exclude test not for wheel builds or are manually run.
DEFAULT_TAG_EXCLUDES="manual|notest|wheel_test_excluded"

# Define what to KEEP (Python files under tests/ or examples/)
DEFAULT_TESTS_INCLUSIONS="^(tests|examples)/.*\.py$"

# Define what to EXCLUDE (e.g., benchmarks/)
# TODO(ecalubaquib): Remove this exclusions once all tests are runnable.
DEFAULT_TESTS_EXCLUSIONS="^examples/benchmarks/"
DEFAULT_TESTS_EXCLUSIONS+="|^tests/fused_adagrad_test\.py$"
DEFAULT_TESTS_EXCLUSIONS+="|^tests/index_put_op_unit_test\.py$"
DEFAULT_TESTS_EXCLUSIONS+="|^tests/ops_test\.py$"
DEFAULT_TESTS_EXCLUSIONS+="|^tests/tracing_test\.py$"


# Apply environment-specific exclusions
if [[ "$BAZEL_CONFIG" == *"cpu"* ]]; then
  # Exclude tpu and accelerator related tags.
  DEFAULT_TAG_EXCLUDES+="|requires-tpu|requires-gpu|requires-viper|requires-ghost"
  # Automatically skip files ending in _tpu_test.py or _gpu_test.py
  DEFAULT_TESTS_EXCLUSIONS+="|_tpu_test\.py$|_gpu_test\.py$"
  # Exclude other known CPU failures
elif [[ "$BAZEL_CONFIG" == *"tpu"* ]]; then
  # Automatically skip files ending in _cpu_test.py or _gpu_test.py
  DEFAULT_TESTS_EXCLUSIONS+="|_cpu_test\.py$|_gpu_test\.py$"
  # Exclude tests that are not yet ready/stable on TPU for pytest
  DEFAULT_TESTS_EXCLUSIONS+="|^tests/distributed/"
  DEFAULT_TESTS_EXCLUSIONS+="|^tests/pallas/"
  DEFAULT_TESTS_EXCLUSIONS+="|^tests/compile/model_coverage_hf_test\.py$"
  DEFAULT_TESTS_EXCLUSIONS+="|^tests/compile/model_coverage_timm_test\.py$"
  DEFAULT_TESTS_EXCLUSIONS+="|^tests/compiler_option_test\.py$"
  DEFAULT_TESTS_EXCLUSIONS+="|^tests/errors_test\.py$"
  DEFAULT_TESTS_EXCLUSIONS+="|^tests/tpu_backend_config_test\.py$"
  DEFAULT_TESTS_EXCLUSIONS+="|^tests/tpu_errors_test\.py$"
  DEFAULT_TESTS_EXCLUSIONS+="|^tests/fused_adam_test\.py$"
  DEFAULT_TESTS_EXCLUSIONS+="|^tests/fused_sgd_test\.py$"
  DEFAULT_TESTS_EXCLUSIONS+="|^tests/fused_adamw_test\.py$"
  DEFAULT_TESTS_EXCLUSIONS+="|^tests/fused_moving_avg_obs_fq_helper_test\.py$"
  DEFAULT_TESTS_EXCLUSIONS+="|^tests/compile/activation_checkpointing_mem_test\.py$"

fi

# Helper function to find impacted Python test files from bazel-diff targets.
translate_targets_to_pytest_files() {
  local query_expr="$1"
  local output_file="$2"
  local raw_file="${output_file%.txt}_raw.txt"

  local tag_excludes="$DEFAULT_TAG_EXCLUDES"
  local tests_exclusions="$DEFAULT_TESTS_EXCLUSIONS"
  local tests_inclusions="$DEFAULT_TESTS_INCLUSIONS"

  # Query all test targets, excluding those with specific tags.
  bazel query "labels(srcs, \
    kind('.*test rule', ${query_expr}) \
    except attr('tags', '${tag_excludes}', ${query_expr}) \
  )" --output=label > "$raw_file"


  # Convert Bazel labels to file paths (e.g., //tests:foo.py -> tests/foo.py)
  # 1. Strip leading '//'
  # 2. Replace ':' with '/'
  sed 's|^//||; s|:|/|' "$raw_file" | \
      grep -E "$tests_inclusions" | \
      grep -Ev "$tests_exclusions" > "$output_file" || true
}

# Helper function to run pytest on a list of files (CWD isolation is handled by conftest.py)
run_pytest_on_files() {
  local files_file="$1"
  if [ -s "$files_file" ]; then
    FILE_COUNT=$(wc -l < "$files_file")
    echo "Found $FILE_COUNT impacted Python test file(s)."
    echo "--- Impacted Test Files List ---"
    cat "$files_file"
    echo "--------------------------------"

    # Prefer local .venv pytest binary if available
    # (e.g., when running locally on cloudtop/laptop)
    if [ -x "$WORKSPACE_DIR/.venv/bin/pytest" ]; then
      local pytest_cmdir="$WORKSPACE_DIR/.venv/bin/pytest"
    else
      local pytest_cmdir="pytest"
    fi

    # Run pytest on each test file in its own isolated process
    # to prevent C++ extension/Triton collisions
    OVERALL_EXIT_CODE=0
    while IFS= read -r test_file; do
      if [ -n "$test_file" ]; then
        echo "Running pytest on: $test_file"
        set +e
        # TODO(ecalubaquib): Remove --ignore=torch_tpu once all directories are
        # under proper src/ and tests/ directories.
        "$pytest_cmdir" -v --ignore=torch_tpu --import-mode=importlib $EXTRA_FLAGS "$test_file"
        PYTEST_EXIT_CODE=$?
        set -e

        # Pytest exit code 5 means no tests were collected
        # (e.g., if files only contained helper classes or bazel-only tests)
        if [ $PYTEST_EXIT_CODE -ne 0 ] && [ $PYTEST_EXIT_CODE -ne 5 ]; then
          echo "Pytest failed on $test_file with exit code $PYTEST_EXIT_CODE."
          OVERALL_EXIT_CODE=$PYTEST_EXIT_CODE
        fi
      fi
    done < "$files_file"

    if [ $OVERALL_EXIT_CODE -ne 0 ]; then
      exit $OVERALL_EXIT_CODE
    fi
    echo "All pytest runs completed successfully."
  else
    echo "No Python test files impacted by these changes. Skipping pytest entirely!"
  fi
}

if [ "$MODE" == "--full" ]; then
  echo "Running all pytest tests..."
  translate_targets_to_pytest_files "//tests/... + //examples/..." "$WORKSPACE_DIR/all_py_files.txt"
  run_pytest_on_files "$WORKSPACE_DIR/all_py_files.txt"
elif [ "$MODE" == "--diff" ]; then
  if [ -s "$TARGETS_FILE" ]; then
    echo "Running impacted pytest tests..."
    TARGETS=$(tr '\n' ' ' < "$TARGETS_FILE")
    translate_targets_to_pytest_files "set(${TARGETS})" "$WORKSPACE_DIR/impacted_py_files.txt"
    run_pytest_on_files "$WORKSPACE_DIR/impacted_py_files.txt"
  else
    echo "No impacted targets file provided or file is empty."
  fi
else
  echo "Unknown mode: $MODE"
  exit 1
fi
