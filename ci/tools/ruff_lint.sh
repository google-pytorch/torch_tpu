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

# This script checks and applies Python linting rules using `ruff`.
#
# Design Intent (Incremental Linting):
#   Instead of running ruff across the entire repository (which could cause
#   massive diffs on existing code), this script uses `git diff` against a base
#   tracking branch ($BASE_SHA). It strictly evaluates and fixes only the
#   specific Python files modified in the current Pull Request or local working tree.
#
# Modes:
#   - 'lint'   : Used by CI. Detects linting violations, outputs self-serve fix
#                instructions, and exits with a failure code.
#   - 'fix'    : Used locally by developers. Automatically applies ruff lint fixes
#                to modified Python files in-place without failing.

set -e

# Fall back to origin/main if BASE_SHA isn't passed by CI.
BASE_SHA="${BASE_SHA:-origin/main}"
MODE="${1}"

# Get the list of added, copied, modified, or renamed Python files
PYTHON_FILES=$(git diff --name-only --diff-filter=ACMR "$BASE_SHA" HEAD -- "*.py" 2>/dev/null || true)

# Ensure a valid mode is explicitly passed; fail on unexpected args.
case "$MODE" in
  lint)
    echo "INFO: Running ruff check on modified Python files..."
    if [ -n "$PYTHON_FILES" ]; then
        # Check rule violations
        if ! echo "$PYTHON_FILES" | xargs -r ruff check; then
            echo "================================================================="
            echo "ERROR: Python linting issues found in modified files."
            echo "================================================================="
            echo "To fix automatically:"
            echo "1. Install nox locally: pip install nox"
            echo "2. Run the autofixer in the root of the repository: nox -s format"
            echo ""
            echo "To suppress specific ruff warnings when legitimate:"
            echo "Add a trailing comment on the affected line, e.g. # noqa: E501"
            echo "================================================================="
            exit 1
        fi
    else
        echo "INFO: No modified Python files found in git diff against $BASE_SHA."
    fi
    echo "INFO: ruff check passed successfully."
    ;;

  fix)
    echo "INFO: Applying ruff fixes to modified Python files..."
    if [ -n "$PYTHON_FILES" ]; then
        echo "$PYTHON_FILES" | xargs -r ruff check --fix || true
    else
        echo "INFO: No modified Python files found in git diff against $BASE_SHA."
    fi
    echo "INFO: Python lint fixes complete."
    ;;

  *)
    echo "ERROR: Unknown mode '$MODE'. Explicit arguments required: 'lint' or 'fix'." >&2
    echo "Usage: $0 {lint|fix}" >&2
    exit 2
    ;;
esac
