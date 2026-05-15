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

# This script is the core execution engine for checking and applying C++
# static analysis rules using `clang-tidy`.
#
# Design Intent (Incremental Linting):
#   Instead of blindly analyzing the entire repository, this script uses
#   `git diff` against a base tracking branch ($BASE_SHA). It strictly evaluates
#   and reports on only the specific files modified in the current Pull Request
#   or local working tree. It separates rulesets for production vs test files.
#
# Modes:
#   - 'lint' : Used by CI. Detects static analysis violations, outputs
#              self-serve fix instructions, and exits with a failure code.
#   - 'fix'  : Used locally by developers. Blindly applies safe auto-fixes to
#              modified files in-place without failing.

set -e
set -o pipefail

BASE_SHA="${BASE_SHA:-origin/main}"
MODE="${1}"

# 1. Identify changed C++ files
FILES=$(git diff --diff-filter=d --name-only "$BASE_SHA" HEAD -- "*.cc" "*.cpp" "*.h")

if [ -z "$FILES" ]; then
    echo "INFO: No C++ files modified. Skipping clang-tidy."
    exit 0
fi

# 2. Safely split files into prod and test buckets.
TEST_FILES=$(echo "$FILES" | grep -E "_test\.cc$" || true)
PROD_FILES=$(echo "$FILES" | grep -v -E "_test\.cc$" || true)

# 3. Determine compilation context
# clang-tidy needs to know how to compile the code. It prefers a compilation database.
if [ -n "$BUILD_DIR" ] && [ -f "$BUILD_DIR/compile_commands.json" ]; then
    echo "INFO: Using compilation database from $BUILD_DIR"
    TIDY_COMPILE_ARGS="-p $BUILD_DIR"
else
    echo "WARNING: No BUILD_DIR or compile_commands.json provided. Using fallback flags."
    TIDY_COMPILE_ARGS="-- -std=c++20 -I."
fi

case "$MODE" in
  lint)
    echo "INFO: Running clang-tidy check on modified files..."
    set +e # Temporarily disable exit-on-error to capture exit codes

    PROD_EXIT=0
    TEST_EXIT=0

    # Run strict checks on production code
    if [ -n "$PROD_FILES" ]; then
      echo "Linting Production Files..."
      clang-tidy $PROD_FILES --warnings-as-errors='*' $TIDY_COMPILE_ARGS
      PROD_EXIT=$?
    fi

    # Run relaxed checks on test code
    if [ -n "$TEST_FILES" ]; then
      echo "Linting Test Files..."
      clang-tidy $TEST_FILES --warnings-as-errors='*' \
        -checks="-bugprone-unchecked-optional-access,-performance-for-range-copy,-performance-inefficient-vector-operation,-performance-unnecessary-copy-initialization" \
        $TIDY_COMPILE_ARGS
      TEST_EXIT=$?
    fi
    
    set -e # Re-enable exit-on-error

    # Fail the script if either run failed
    if [ $PROD_EXIT -ne 0 ] || [ $TEST_EXIT -ne 0 ]; then
        echo "================================================================="
        echo "ERROR: Static analysis issues found in your changes."
        echo "To fix automatically (where possible):"
        echo "1. Install tox locally: pip install tox tox-uv"
        echo "2. Run the formatter: tox -e tidy-format"
        echo "================================================================="
        exit 1
    fi
    echo "INFO: clang-tidy check passed successfully."
    ;;

  fix)
    echo "INFO: Applying clang-tidy fixes to modified files..."
    
    if [ -n "$PROD_FILES" ]; then
      clang-tidy $PROD_FILES -fix -fix-errors $TIDY_COMPILE_ARGS
    fi

    if [ -n "$TEST_FILES" ]; then
      clang-tidy $TEST_FILES -fix -fix-errors \
        -checks="-bugprone-unchecked-optional-access,-performance-for-range-copy,-performance-inefficient-vector-operation,-performance-unnecessary-copy-initialization" \
        $TIDY_COMPILE_ARGS
    fi
    echo "INFO: clang-tidy formatting applied."
    ;;

  *)
    echo "ERROR: Unknown mode '$MODE'. Explicit arguments required: 'lint' or 'fix'." >&2
    exit 2
    ;;
esac
