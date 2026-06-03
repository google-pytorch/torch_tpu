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

# Prerequisites: bazel-diff requires Git, Bazel >= 3.3.0 and Java >= 8.
# https://github.com/Tinder/bazel-diff/tree/16.0.0?tab=readme-ov-file#prerequisites
# As of 2026-04, our Docker container has Git 2.34.1, Bazel 7.4.1 and Java 21.0.10.

set -e

WORKSPACE_DIR="${1:-$GITHUB_WORKSPACE}"
BASE_SHA="${2}"
CURRENT_SHA="${3}"
BAZEL_CONFIG="${4}"
EXTRA_FLAGS="${5}"
DISABLE_BAZEL_DIFF="${6:-false}"

# If BASE_SHA is empty (e.g., workflow_dispatch, postsubmit, nightly on main) or
# label "ci:disable-bazel-diff" is applied, diff-based testing is invalid. Fall
# back to running all tests.
if [ -z "$BASE_SHA" ] || [ "$DISABLE_BAZEL_DIFF" == "true" ]; then
  echo "No BASE_SHA provided (not a PR), bazel-diff disabled via PR label or global configuration files modified. Skipping bazel-diff and running all tests."

  set +e
  eval bazel test --config="$BAZEL_CONFIG" $EXTRA_FLAGS //...
  BAZEL_EXIT_CODE=$?
  set -e

  exit $BAZEL_EXIT_CODE
fi

echo "Downloading bazel-diff..."
# Pin to v16.0.0 because latest releases are missing bazel-diff_deploy.jar:
# https://github.com/Tinder/bazel-diff/issues/320
curl -fLo /tmp/bazel-diff.jar https://github.com/Tinder/bazel-diff/releases/download/16.0.0/bazel-diff_deploy.jar

# Generates global configuration seed files dynamically.
generate_seed_filepaths() {
  local seed_file="$WORKSPACE_DIR/seed_filepaths.txt"
  rm -f "$seed_file"
  for f in .bazelrc MODULE.bazel WORKSPACE; do
    if [ -f "$WORKSPACE_DIR/$f" ]; then
      echo "$WORKSPACE_DIR/$f" >> "$seed_file"
    fi
  done
}

echo "--- Generating Base Hashes ---"
echo "PR detected. Fetching exact base SHA: $BASE_SHA"
git fetch --depth=1 origin "$BASE_SHA"
git checkout "$BASE_SHA"
generate_seed_filepaths
java -jar /tmp/bazel-diff.jar generate-hashes -w "$WORKSPACE_DIR" --seed-filepaths "$WORKSPACE_DIR/seed_filepaths.txt" "$WORKSPACE_DIR/base_hashes.json"

echo "--- Generating PR Hashes ---"
echo "Checking out current SHA: $CURRENT_SHA"
git checkout "$CURRENT_SHA"
generate_seed_filepaths
java -jar /tmp/bazel-diff.jar generate-hashes -w "$WORKSPACE_DIR" --seed-filepaths "$WORKSPACE_DIR/seed_filepaths.txt" "$WORKSPACE_DIR/pr_hashes.json"

echo "--- Determining Impacted Targets ---"
java -jar /tmp/bazel-diff.jar get-impacted-targets \
  -sh "$WORKSPACE_DIR/base_hashes.json" \
  -fh "$WORKSPACE_DIR/pr_hashes.json" \
  -o "$WORKSPACE_DIR/impacted_targets.txt"

echo "--- Running Impacted Bazel Tests ---"
if [ -s "$WORKSPACE_DIR/impacted_targets.txt" ]; then
  TARGET_COUNT=$(wc -l < "$WORKSPACE_DIR/impacted_targets.txt")
  echo "Found $TARGET_COUNT impacted target(s)."
  echo "--- Impacted Targets List ---"
  cat "$WORKSPACE_DIR/impacted_targets.txt"
  echo "-----------------------------"

  # Disable Bash's "exit on error" to manually evaluate Bazel's exit code
  set +e

  # Use eval to properly expand any nested quotes in EXTRA_FLAGS (like --run_under="$(pwd)/...")
  eval bazel test --config="$BAZEL_CONFIG" $EXTRA_FLAGS --target_pattern_file="$WORKSPACE_DIR/impacted_targets.txt"
  BAZEL_EXIT_CODE=$?

  # Re-enable "exit on error"
  set -e

  # Evaluate the Bazel Exit Code
  if [ $BAZEL_EXIT_CODE -eq 4 ]; then
    echo "Bazel returned Exit Code 4 (No tests found). Treating as success!"
    exit 0
  elif [ $BAZEL_EXIT_CODE -ne 0 ]; then
    echo "Bazel test failed with exit code $BAZEL_EXIT_CODE."
    exit $BAZEL_EXIT_CODE
  fi
else
  echo "No tests impacted by these changes. Skipping Bazel entirely!"
fi
