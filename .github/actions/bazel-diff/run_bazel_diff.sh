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
# As of 2026-06, our Docker container has Git 2.34.1, Bazel 8.6.0 and Java 21.0.10.

set -e

WORKSPACE_DIR="${1:-$GITHUB_WORKSPACE}"
BASE_SHA="${2}"
CURRENT_SHA="${3}"
BAZEL_CONFIG="${4}"
EXTRA_FLAGS="${5}"
DISABLE_BAZEL_DIFF="${6:-false}"

# Check if global Bazel configuration were modified. If so, bypass bazel-diff
# to guarantee a full build validation.
#
# We intentionally do not use bazel-diff's native `--seed-filepaths` flag for
# handling changes to global configuration files. It forces bazel-diff to
# calculate the entire repository graph, serialize all targets (5000+) to
# impacted_targets.txt, and feed them back to Bazel. This introduces a slight
# performance regression.
if [ -n "$BASE_SHA" ] && [ "$DISABLE_BAZEL_DIFF" != "true" ]; then
  git fetch --depth=1 origin "$BASE_SHA"
  CHANGED_FILES=$(git diff --name-only "$BASE_SHA" "$CURRENT_SHA")

  GLOBAL_BAZEL_CONFIGS=(  # One regex per line.
    # go/keep-sorted start
    'MODULE\.bazel'
    'REPO\.bazel'
    'WORKSPACE'
    'WORKSPACE\.bzlmod'
    '\.bazelrc'
    '\.bazelversion'
    # go/keep-sorted end
  )
  # Concatenate array elements into a string where each element is prefixed by
  # a pipe (`|`) symbol. Then, strip the leading pipe.
  GLOBAL_BAZEL_CONFIGS_PATTERN=$(printf "|%s" "${GLOBAL_BAZEL_CONFIGS[@]}")
  GLOBAL_BAZEL_CONFIGS_PATTERN=${GLOBAL_BAZEL_CONFIGS_PATTERN#|}

  if echo "$CHANGED_FILES" | grep -E -q "^(${GLOBAL_BAZEL_CONFIGS_PATTERN})$"; then
    echo "Global Bazel configurations modified."
    echo "Forcing a full test run to ensure global configuration validity."
    DISABLE_BAZEL_DIFF="true"
  fi
fi

# If BASE_SHA is empty (e.g., workflow_dispatch, postsubmit, nightly on main) or
# label "ci:disable-bazel-diff" is applied, diff-based testing is invalid. Fall
# back to running all tests.
if [ -z "$BASE_SHA" ] || [ "$DISABLE_BAZEL_DIFF" == "true" ]; then
  echo "No BASE_SHA provided (not a PR), bazel-diff disabled via PR label or global configuration files modified. Skipping bazel-diff and running all tests."

  set +e
  eval bazel test --config="$BAZEL_CONFIG" $EXTRA_FLAGS //...
  BAZEL_EXIT_CODE=$?
  set -e

  if [ $BAZEL_EXIT_CODE -eq 4 ]; then
    echo "Bazel returned Exit Code 4 (No tests found). Treating as success!"
    exit 0
  fi

  exit $BAZEL_EXIT_CODE
fi

echo "Downloading bazel-diff..."
# Pin to v16.0.0 because latest releases are missing bazel-diff_deploy.jar:
# https://github.com/Tinder/bazel-diff/issues/320
curl -fLo /tmp/bazel-diff.jar https://github.com/Tinder/bazel-diff/releases/download/16.0.0/bazel-diff_deploy.jar

echo "--- Generating Base Hashes ---"
echo "PR detected. Fetching exact base SHA: $BASE_SHA"
git fetch --depth=1 origin "$BASE_SHA"
git checkout "$BASE_SHA"
java -jar /tmp/bazel-diff.jar generate-hashes -w "$WORKSPACE_DIR" "$WORKSPACE_DIR/base_hashes.json" \
  --useCquery \
  --cqueryCommandOptions="--config=$BAZEL_CONFIG"

echo "--- Generating PR Hashes ---"
echo "Checking out current SHA: $CURRENT_SHA"
git checkout "$CURRENT_SHA"
java -jar /tmp/bazel-diff.jar generate-hashes -w "$WORKSPACE_DIR" "$WORKSPACE_DIR/pr_hashes.json" \
  --useCquery \
  --cqueryCommandOptions="--config=$BAZEL_CONFIG"

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
  # bazel-diff feeds the impacted targets as explicit labels, and Bazel treats
  # an explicitly requested target whose target_compatible_with is unsatisfied
  # as a hard error -- unlike the wildcard //... runs everywhere else, which
  # skip it. --skip_incompatible_explicit_targets restores the wildcard
  # behavior, so config-gated tests are skipped rather than failing presubmit.
  eval bazel test --config="$BAZEL_CONFIG" $EXTRA_FLAGS --skip_incompatible_explicit_targets --target_pattern_file="$WORKSPACE_DIR/impacted_targets.txt"
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
