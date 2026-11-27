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

# This script checks and applies Bazel BUILD/bzl formatting using `buildifier`.
#
# Design Intent (Incremental Linting):
#   Instead of running buildifier across the entire repository, this script uses
#   `git diff` against a base tracking branch ($BASE_SHA). It strictly evaluates
#   only the specific Bazel syntax files modified in the current Pull Request or
#   local working tree.
#
# Modes:
#   - 'lint'   : Used by CI. Runs Copybara-equivalent auto-fixes in a scratch
#                directory first, then checks for syntax errors or remaining
#                non-auto-fixable lint warnings so CI does not reject PRs for
#                formatting/lints that Copybara fixes automatically on import.
#   - 'format' : Used locally by developers. Automatically applies buildifier fixes
#                and formatting in-place without failing.

set -e

# Pin specific release version of buildifier for reproducible CI behavior
BUILDIFIER_VERSION="8.5.1"

# Fall back to origin/main if BASE_SHA isn't passed by CI.
BASE_SHA="${BASE_SHA:-origin/main}"
MODE="${1}"

# Ensure buildifier binary is available or install/download dynamically
if ! command -v buildifier &> /dev/null; then
    echo "INFO: buildifier not found in PATH. Attempting to install or download v${BUILDIFIER_VERSION}..."
    if command -v go &> /dev/null; then
        echo "INFO: Installing buildifier v${BUILDIFIER_VERSION} via go install..."
        go install github.com/bazelbuild/buildtools/buildifier@v${BUILDIFIER_VERSION} 2>/dev/null || true
        export PATH="$PATH:$(go env GOPATH)/bin:$HOME/go/bin:/usr/local/bin"
    fi
    if ! command -v buildifier &> /dev/null; then
        echo "INFO: Downloading pre-compiled buildifier v${BUILDIFIER_VERSION} binary directly from GitHub Releases..."
        mkdir -p "$HOME/.local/bin"
        curl -sSL "https://github.com/bazelbuild/buildtools/releases/download/v${BUILDIFIER_VERSION}/buildifier-linux-amd64" -o "$HOME/.local/bin/buildifier"
        chmod +x "$HOME/.local/bin/buildifier"
        export PATH="$PATH:$HOME/.local/bin"
    fi
    if ! command -v buildifier &> /dev/null; then
        echo "ERROR: Could not find or install buildifier." >&2
        exit 3
    fi
fi

# Get the list of added, copied, modified, renamed, or type-changed Bazel/Starlark files against $BASE_SHA
BUILDIFIER_PATTERNS=(
    "*BUILD*"
    "*.bzl"
    "*.star"
    "BUILD"
    "BUILD.bazel"
)

GREP_PATTERN='(BUILD|BUILD\.bazel|\.bzl|\.star)$'

BAZEL_FILES=$(git diff --name-only --diff-filter=ACMRT "$BASE_SHA" -- "${BUILDIFIER_PATTERNS[@]}" 2>/dev/null | grep -E "$GREP_PATTERN" || true)

run_non_autofixable_lint() {
  (
    set -euo pipefail
    local tmp_dir
    tmp_dir=$(mktemp -d)
    trap 'rm -rf "${tmp_dir}"' EXIT

    local -a tmp_files=()
    local file
    local target_dir
    while IFS= read -r file; do
      [[ -z "${file}" ]] && continue
      [[ -f "${file}" ]] || continue
      target_dir="${tmp_dir}/$(dirname "${file}")"
      mkdir -p "${target_dir}"
      cp "${file}" "${tmp_dir}/${file}"
      tmp_files+=("${tmp_dir}/${file}")
    done <<< "${BAZEL_FILES}"

    if [[ "${#tmp_files[@]}" -eq 0 ]]; then
      return 0
    fi

    # Apply the same formatting and auto-fixable lints that Copybara runs on
    # PR import (`format.buildifier(lint = "FIX")`) to scratch copies so CI only
    # fails on syntax errors or lint warnings Copybara cannot fix automatically.
    # LINT.IfChange(buildifier_lint_fix)
    local output
    if ! output=$(buildifier -mode=fix -lint=fix -warnings=-out-of-order-load "${tmp_files[@]}" 2>&1); then
      printf '%s\n' "${output//"${tmp_dir}/"/}" >&2
      return 1
    fi

    if ! output=$(buildifier -mode=check -lint=warn -warnings=-out-of-order-load "${tmp_files[@]}" 2>&1); then
      printf '%s\n' "${output//"${tmp_dir}/"/}" >&2
      return 1
    fi
    # LINT.ThenChange(../../../copy.bara.sky:buildifier_lint_fix)
  )
}

# Ensure a valid mode is explicitly passed; fail on unexpected args.
case "$MODE" in
  lint)
    echo "INFO: Running buildifier non-auto-fixable lint check on modified Bazel files against $BASE_SHA..."
    if [ -n "$BAZEL_FILES" ]; then
        if ! run_non_autofixable_lint; then
            echo "================================================================="
            echo "ERROR: Non-auto-fixable Bazel BUILD/bzl lint issues found."
            echo "================================================================="
            echo "Formatting and auto-fixable lints are fixed automatically on"
            echo "import, so the issues above require a manual fix or a targeted"
            echo "suppression comment (e.g. # buildifier: disable=<warning>)."
            echo "================================================================="
            exit 1
        fi
    else
        echo "INFO: No modified Bazel/Starlark files found in git diff against $BASE_SHA."
    fi
    echo "INFO: buildifier check passed successfully."
    ;;

  format)
    echo "INFO: Applying buildifier fixes and formatting to modified Bazel files..."
    if [ -n "$BAZEL_FILES" ]; then
        echo "$BAZEL_FILES" | xargs buildifier -mode=fix -lint=fix -warnings=-out-of-order-load || true
    else
        echo "INFO: No modified Bazel BUILD/bzl files found in git diff against $BASE_SHA."
    fi
    echo "INFO: Bazel buildifier formatting complete."
    ;;

  *)
    echo "ERROR: Unknown mode '$MODE'. Explicit arguments required: 'lint' or 'format'." >&2
    echo "Usage: $0 {lint|format}" >&2
    exit 2
    ;;
esac
