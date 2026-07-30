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

# This script is the core execution engine for checking and repairing
# BUILD.bazel files and Starlark formatting using Gazelle and Buildifier,
# providing parity with internal build_cleaner for Open Source workflows.
#
# Modes:
#   - 'check'  : Used by CI. Detects out-of-sync BUILD.bazel files or formatting
#                issues, outputs self-serve fix instructions, and exits with a
#                failure code.
#   - 'fix'    : Used locally by developers. Automatically generates/updates
#                BUILD.bazel dependencies and applies formatting in-place.
set -euo pipefail

MODE="${1:-}"
BAZEL="${BAZEL:-bazel}"
BAZEL_FLAGS_ARRAY=(${BAZEL_FLAGS:-})

print_tool_use_hint() {
  echo "ERROR: build_cleaner check could not run to completion." >&2
  echo "It usually means the tool is not invoked properly. Run it via the following commands:" >&2
  echo "  pip install nox               # one-time setup" >&2
  echo "  nox -s build_cleaner_check    # check BUILD.bazel issues" >&2
  echo "  nox -s build_cleaner          # apply fixes automatically" >&2
}

case "$MODE" in
  check)
    echo "INFO: Running build_cleaner (Gazelle + Buildifier) check..."
    set +e
    GAZELLE_OUT=$("${BAZEL}" run "${BAZEL_FLAGS_ARRAY[@]}" --config=go_tools //:gazelle -- -mode=diff 2>&1)
    GAZELLE_STATUS=$?
    BUILDIFIER_OUT=$("${BAZEL}" run "${BAZEL_FLAGS_ARRAY[@]}" --config=go_tools //:buildifier -- -mode=check -r . 2>&1)
    BUILDIFIER_STATUS=$?
    set -e

    if [ "$GAZELLE_STATUS" -ne 0 ] || [ "$BUILDIFIER_STATUS" -ne 0 ]; then
        echo "================================================================="
        echo "ERROR: BUILD.bazel files are out of sync or unformatted."
        if [ "$GAZELLE_STATUS" -ne 0 ]; then
            echo "--- Gazelle dependency diff ---"
            echo "$GAZELLE_OUT"
        fi
        if [ "$BUILDIFIER_STATUS" -ne 0 ]; then
            echo "--- Buildifier formatting check ---"
            echo "$BUILDIFIER_OUT"
        fi
        echo "================================================================="
        echo "To fix automatically:"
        echo "1. Install nox locally: pip install nox"
        echo "2. Run build_cleaner in the root of the repository: nox -s build_cleaner"
        echo "   (or run: bazel run //:gazelle && bazel run //:buildifier -- -mode=fix -r .)"
        echo "================================================================="
        exit 1
    fi
    echo "INFO: build_cleaner check passed successfully."
    ;;

  fix)
    echo "INFO: Generating and updating BUILD.bazel files with Gazelle..."
    "${BAZEL}" run "${BAZEL_FLAGS_ARRAY[@]}" --config=go_tools //:gazelle
    echo "INFO: Formatting Starlark files with Buildifier..."
    "${BAZEL}" run "${BAZEL_FLAGS_ARRAY[@]}" --config=go_tools //:buildifier -- -mode=fix -r .
    echo "INFO: Successfully updated and formatted BUILD.bazel files."
    ;;

  *)
    echo "ERROR: Unknown mode '$MODE'. Explicit arguments required: 'check' or 'fix'." >&2
    echo "Usage: $0 {check|fix}" >&2
    exit 2
    ;;
esac
