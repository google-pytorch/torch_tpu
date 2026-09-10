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
# Determines whether wheel builds and tests can be skipped for a GitHub Actions
# workflow run by inspecting changed files between BASE_SHA and HEAD_SHA.
#
# Non-PR events (push to main, schedule, workflow_dispatch) and PRs with force-build
# labels always trigger full builds. For standard PRs, if all modified files are
# documentation or repository metadata (.md, docs/, OWNERS, LICENSE, .clang*,
# .gitignore, .vscode/), the script emits should_build=false so downstream jobs
# can skip VM provisioning.
#
# Inputs (Environment Variables):
#   EVENT_NAME     The GitHub event name (e.g. "pull_request", "push", "schedule").
#   BASE_SHA       The base commit SHA for the PR diff.
#   HEAD_SHA       The target commit SHA being evaluated.
#   FORCE_BUILD    "true" if an override label is present (e.g. ci:force-wheel-build).
#   GITHUB_OUTPUT  Path to the GitHub Actions step output file.
#
# Output:
#   Appends `should_build=true` or `should_build=false` to $GITHUB_OUTPUT.
#
# Result Code:
#   0 on success (evaluation completed and should_build set).
#   Non-zero on error (e.g. git command failure under set -euo pipefail).

set -euo pipefail

# Push to main, scheduled nightly, workflow_dispatch, or override labels always build.
if [[ "${EVENT_NAME}" != "pull_request" || "${FORCE_BUILD}" == "true" ]]; then
  echo "Non-PR event or force-build label detected. Triggering full build."
  echo "should_build=true" >> "$GITHUB_OUTPUT"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFF_BASE="$("${SCRIPT_DIR}/tools/resolve_base_sha.sh" "${HEAD_SHA}" "${BASE_SHA:-}")"

if [[ -z "${DIFF_BASE}" ]]; then
  echo "No base SHA available. Defaulting to build."
  echo "should_build=true" >> "$GITHUB_OUTPUT"
  exit 0
fi

CHANGED_FILES=$(git diff --name-only "${DIFF_BASE}" "${HEAD_SHA}")

if [[ -z "${CHANGED_FILES}" ]]; then
  echo "No changed files detected. Defaulting to build."
  echo "should_build=true" >> "$GITHUB_OUTPUT"
  exit 0
fi

echo "Changed files in PR:"
echo "${CHANGED_FILES}"

# Filter out files that do not affect the wheel build or testing
NON_DOC_FILES=$(echo "${CHANGED_FILES}" | grep -v -E '(\.md$|^docs/|(^|/)OWNERS|^LICENSE|^\.clang|^\.gitignore|^\.vscode/)' || true)

if [[ -z "${NON_DOC_FILES}" ]]; then
  echo "All modified files are documentation or repo metadata. Skipping wheel build."
  echo "should_build=false" >> "$GITHUB_OUTPUT"
else
  echo "Found changes requiring wheel build:"
  echo "${NON_DOC_FILES}"
  echo "should_build=true" >> "$GITHUB_OUTPUT"
fi
