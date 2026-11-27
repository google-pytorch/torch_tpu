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
# formatting rules using `clang-format-diff.py`.
#
# Design Intent (Incremental Formatting):
#   Instead of blindly formatting the entire repository, this script uses
#   `git diff` against a base tracking branch ($BASE_SHA). It strictly evaluates
#   and formats only the specific lines modified in the current Pull Request or
#   local working tree.
#
# Modes:
#   - 'lint'   : Used by CI. Detects formatting violations, prints the diff,
#                outputs self-serve fix instructions, and exits with a failure
#                code.
#   - 'format' : Used locally by developers. Blindly applies fixes to modified
#                lines in-place without failing.
set -euo pipefail

# Fall back to origin/main if BASE_SHA isn't passed by CI.
BASE_SHA="${BASE_SHA:-origin/main}"
# Default to empty so an omitted mode reaches the usage help message
# instead of aborting with an "unbound variable" error.
MODE="${1:-}"

# Tell the user how to run this script correctly. Users may be tempted to run the script
# directly instead of through tox.
print_tool_use_hint() {
  echo "ERROR: clang-format check could not run to completion." >&2
  echo "It usually means the tool is not invoked properly. Run it via:" >&2
  echo "  pip install nox   # one-time setup" >&2
  echo "  nox -s format     # apply fixes" >&2
}

# Ensure a valid mode is explicitly passed; fail on unexpected args.
case "$MODE" in
  lint)
    echo "INFO: Running clang-format check..."
    set +e
    # Diff BASE_SHA against the working tree (not HEAD) so uncommitted edits
    # on disk are inspected too.
    DIFF=$(git diff -U0 --no-color "$BASE_SHA" -- "*.cc" "*.h" | clang-format-diff.py -binary clang-format -p1 -style=file)
    STATUS=$?
    set -e

    # clang-format-diff.py exits 0 when the changes are already formatted,
    # 1 when it finds formatting differences, and higher codes on tool errors.
    # Only exit codes larger than 1 mean the tool failed to run.
    if [ "$STATUS" -gt 1 ]; then
        print_tool_use_hint
        exit 3
    fi

    if [ -n "$DIFF" ]; then
        echo "================================================================="
        echo "ERROR: Formatting issues found in your changes."
        echo "$DIFF"
        echo "================================================================="
        echo "To fix automatically:"
        echo "1. Install nox locally: pip install nox"
        echo "2. Run the formatter in the root of the repository: nox -s format"
        echo ""
        echo "To bypass / suppress when needed:"
        echo "Wrap the code in // clang-format off and // clang-format on comments:"
        echo "  // clang-format off"
        echo "  ... code ..."
        echo "  // clang-format on"
        echo "================================================================="
        exit 1
    fi
    echo "INFO: clang-format check passed successfully."
    ;;

  format)
    echo "INFO: Applying clang-format fixes to modified files in-place."
    if ! git diff -U0 --no-color "$BASE_SHA" -- "*.cc" "*.h" | clang-format-diff.py -binary clang-format -p1 -style=file -i; then
        print_tool_use_hint
        exit 3
    fi
    echo "INFO: Successfully formatted changes."
    ;;

  *)
    echo "ERROR: Unknown mode '$MODE'. Explicit arguments required: 'lint' or 'format'." >&2
    echo "Usage: $0 {lint|format}" >&2
    exit 2
    ;;
esac
