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
# Shared PR diff triage helper for GitHub Actions workflows.
# Inspects changed files between BASE_SHA and HEAD_SHA to determine whether
# workflows (presubmit tests, wheel builds) can be skipped for doc-only PRs.
#
# Usage:
#   check_diff_triage.sh [FORCE_OVERRIDE] [OUTPUT_VAR] [ACTION_NAME]
#
# Inputs (Environment Variables):
#   EVENT_NAME     The GitHub event name (e.g. "pull_request", "push", "schedule").
#   BASE_SHA       The base commit SHA for the PR diff.
#   HEAD_SHA       The target commit SHA being evaluated.
#   GITHUB_OUTPUT  Path to the GitHub Actions step output file.

set -euo pipefail

FORCE_OVERRIDE="${1:-}"
OUTPUT_VAR="${2:-should_run_tests}"
ACTION_NAME="${3:-execution}"

EVENT_NAME="${EVENT_NAME:-}"
BASE_SHA="${BASE_SHA:-}"
HEAD_SHA="${HEAD_SHA:-HEAD}"
GITHUB_OUTPUT="${GITHUB_OUTPUT:-/dev/null}"

# Push to main, scheduled nightly, workflow_dispatch, or override labels always trigger execution.
if [[ "${EVENT_NAME}" != "pull_request" || "${FORCE_OVERRIDE}" == "true" ]]; then
  echo "Non-PR event or force override label detected. Triggering ${ACTION_NAME}."
  echo "${OUTPUT_VAR}=true" >> "$GITHUB_OUTPUT"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFF_BASE="$("${SCRIPT_DIR}/resolve_base_sha.sh" "${HEAD_SHA}" "${BASE_SHA}")"

if [[ -z "${DIFF_BASE}" ]]; then
  echo "No base SHA available. Defaulting to ${ACTION_NAME}."
  echo "${OUTPUT_VAR}=true" >> "$GITHUB_OUTPUT"
  exit 0
fi

CHANGED_FILES=$(git diff --name-only "${DIFF_BASE}" "${HEAD_SHA}")

if [[ -z "${CHANGED_FILES}" ]]; then
  echo "No changed files detected. Defaulting to ${ACTION_NAME}."
  echo "${OUTPUT_VAR}=true" >> "$GITHUB_OUTPUT"
  exit 0
fi

echo "Changed files in PR:"
echo "${CHANGED_FILES}"

# Filter out files that do not affect the native build or test suites
NON_DOC_FILES=$(echo "${CHANGED_FILES}" | grep -v -E '(\.md$|^docs/|(^|/)OWNERS|^LICENSE|^\.clang|^\.gitignore|^\.vscode/)' || true)

if [[ -z "${NON_DOC_FILES}" ]]; then
  echo "All modified files are documentation or repo metadata. Skipping ${ACTION_NAME}."
  echo "${OUTPUT_VAR}=false" >> "$GITHUB_OUTPUT"
else
  echo "Found changes requiring ${ACTION_NAME}:"
  echo "${NON_DOC_FILES}"
  echo "${OUTPUT_VAR}=true" >> "$GITHUB_OUTPUT"
fi
