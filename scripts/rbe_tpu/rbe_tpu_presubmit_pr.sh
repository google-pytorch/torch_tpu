#!/usr/bin/env bash
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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/../../ci/tools/rbe_tpu" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

DEFAULT_BUCKET="torch-tpu-mailbox-eval"
DEFAULT_REPO="google-pytorch/torch_tpu"
DEFAULT_FILTER="presubmit-v5,-fails-on-tpu-v5,-nopresubmit,-notest,-nobuild,-requires-tpu-v5lite:8"
DEFAULT_SCOPE="//tests/..."

readonly REPLACE_LABEL="ci:replace-tpu-v5"
readonly SHADOW_LABEL="ci:relay-tpu-v5"
readonly COMMENT_MARKER="<!-- torch-tpu-v5e-mailbox -->"
readonly GATING_CONTEXT="Presubmit on linux-x86-ct5lp-224-8tpu"
readonly SHADOW_CONTEXT="TPU v5e relay (shadow)"

CLI_PR=""
CLI_REPO="$DEFAULT_REPO"
CLI_MODE="shadow"
CLI_SHA=""
CLI_BUCKET="$DEFAULT_BUCKET"
CLI_TARGETS=()
CLI_JOBS=28
CLI_BAZEL_CONFIG="rbe_tpu"
CLI_BAZEL_FLAGS=()
CLI_OUTPUT_DIR=""
CLI_NO_REPORT=false
CLI_DRY_RUN=false
CLI_ADD_LABEL=false
CLI_SKIP_HEAD_CHECK=false
CLI_TEST_TIMEOUT="900"
CLI_SCOPE="$DEFAULT_SCOPE"
CLI_FILTER="$DEFAULT_FILTER"

_STATUS_POSTED=false

show_help() {
  cat <<'HELP'
Usage: scripts/mailbox_presubmit_pr.sh [options]
       (or: scripts/rbe_tpu_presubmit_pr.sh)

Executes the torch_tpu single-chip presubmit suite on Cloud TPU v5e hardware
via the GCS mailbox runner.

Modes:
  shadow       Advisory run. Reports under "TPU v5e relay (shadow)" and
               does not block PR merge (default).
  replacement  Gating run. Reports under "Presubmit on linux-x86-ct5lp-224-8tpu",
               resolving the required status check.

Options:
  --pr N                Pull request number to test.
  --repo OWNER/NAME     GitHub repository (default: google-pytorch/torch_tpu).
  --mode MODE           shadow (default) or replacement.
  --sha SHA             Commit SHA to report against (default: PR head SHA).
  --bucket NAME         GCS mailbox bucket (default: torch-tpu-mailbox-eval).
  --target TARGET       Specific test target to run. Can be repeated.
  --jobs N              Concurrent test jobs (default: 28).
  --bazel-config NAME   Bazel config to inherit (default: ci_tpu_v5_mailbox).
  --bazel-flag FLAG     Extra flag passed to bazel test. Can be repeated.
  --output-dir DIR      Directory to write execution reports and logs.
  --add-label           Add the mode's label to the PR (ci:replace-tpu-v5 or ci:relay-tpu-v5).
  --skip-head-check     Run even if the local checkout is not at the PR head commit.
  --no-report           Do not post commit statuses, labels, or comments to GitHub.
  --dry-run             Resolve targets and show execution plan without running.
  -h, --help            Show this help.
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pr)
      [[ $# -ge 2 ]] || { echo "ERROR: --pr requires an argument." >&2; exit 2; }
      CLI_PR="$2"
      shift 2
      ;;
    --repo)
      [[ $# -ge 2 ]] || { echo "ERROR: --repo requires an argument." >&2; exit 2; }
      CLI_REPO="$2"
      shift 2
      ;;
    --mode)
      [[ $# -ge 2 ]] || { echo "ERROR: --mode requires an argument." >&2; exit 2; }
      CLI_MODE="$2"
      shift 2
      ;;
    --sha)
      [[ $# -ge 2 ]] || { echo "ERROR: --sha requires an argument." >&2; exit 2; }
      CLI_SHA="$2"
      shift 2
      ;;
    --bucket)
      [[ $# -ge 2 ]] || { echo "ERROR: --bucket requires an argument." >&2; exit 2; }
      CLI_BUCKET="$2"
      shift 2
      ;;
    --target)
      [[ $# -ge 2 ]] || { echo "ERROR: --target requires an argument." >&2; exit 2; }
      CLI_TARGETS+=("$2")
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || { echo "ERROR: --jobs requires an argument." >&2; exit 2; }
      [[ "$2" =~ ^[0-9]+$ ]] || { echo "ERROR: --jobs must be a number." >&2; exit 2; }
      CLI_JOBS="$2"
      shift 2
      ;;
    --bazel-config)
      [[ $# -ge 2 ]] || { echo "ERROR: --bazel-config requires an argument." >&2; exit 2; }
      CLI_BAZEL_CONFIG="$2"
      shift 2
      ;;
    --bazel-flag)
      [[ $# -ge 2 ]] || { echo "ERROR: --bazel-flag requires an argument." >&2; exit 2; }
      CLI_BAZEL_FLAGS+=("$2")
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --output-dir requires an argument." >&2; exit 2; }
      CLI_OUTPUT_DIR="$2"
      shift 2
      ;;
    --add-label)
      CLI_ADD_LABEL=true
      shift
      ;;
    --skip-head-check)
      CLI_SKIP_HEAD_CHECK=true
      shift
      ;;
    --no-report)
      CLI_NO_REPORT=true
      shift
      ;;
    --dry-run)
      CLI_DRY_RUN=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      show_help >&2
      exit 2
      ;;
  esac
done

if [[ "$CLI_MODE" != "shadow" && "$CLI_MODE" != "replacement" ]]; then
  echo "ERROR: --mode must be 'shadow' or 'replacement', got '${CLI_MODE}'" >&2
  exit 2
fi

TARGET_CONTEXT="$SHADOW_CONTEXT"
TARGET_LABEL="$SHADOW_LABEL"
if [[ "$CLI_MODE" == "replacement" ]]; then
  TARGET_CONTEXT="$GATING_CONTEXT"
  TARGET_LABEL="$REPLACE_LABEL"
fi

post_status() {
  local sha="$1" state="$2" description="$3" context="$4"
  [[ "$CLI_NO_REPORT" == "true" ]] && return 0
  command -v gh >/dev/null 2>&1 || return 0
  description="${description:0:140}"
  gh api -X POST "repos/${CLI_REPO}/statuses/${sha}" \
    -f "state=${state}" \
    -f "context=${context}" \
    -f "description=${description}" >/dev/null 2>&1 || {
      echo "WARNING: Failed to post commit status to GitHub." >&2
    }
}

NOW_TAG="$(date -u +%Y%m%d_%H%M%SZ)"
if [[ -z "$CLI_OUTPUT_DIR" ]]; then
  if [[ -n "$CLI_PR" ]]; then
    CLI_OUTPUT_DIR="${REPO_ROOT}/mailbox_reports/pr${CLI_PR}_${NOW_TAG}"
  else
    CLI_OUTPUT_DIR="${REPO_ROOT}/mailbox_reports/run_${NOW_TAG}"
  fi
fi
mkdir -p "$CLI_OUTPUT_DIR"

if [[ -n "$CLI_PR" && -z "$CLI_SHA" ]]; then
  if command -v gh >/dev/null 2>&1; then
    CLI_SHA="$(gh api "repos/${CLI_REPO}/pulls/${CLI_PR}" --jq '.head.sha' 2>/dev/null || true)"
  fi
fi

if [[ -n "$CLI_SHA" && "$CLI_SKIP_HEAD_CHECK" == "false" ]]; then
  LOCAL_SHA="$(git rev-parse HEAD 2>/dev/null || true)"
  if [[ -n "$LOCAL_SHA" && "$LOCAL_SHA" != "$CLI_SHA" ]]; then
    echo "WARNING: Local checkout (${LOCAL_SHA:0:10}) does not match PR #${CLI_PR} head (${CLI_SHA:0:10})." >&2
    echo "Tests will run against local files rather than the PR commit." >&2
    echo "Pass --skip-head-check to proceed anyway." >&2
  fi
fi

echo "========================================================================"
echo "TPU Mailbox Presubmit Pipeline"
echo "========================================================================"
echo "Bucket:          gs://${CLI_BUCKET}"
echo "Output Directory:${CLI_OUTPUT_DIR}"
echo "Mode:            ${CLI_MODE}"
[[ -n "$CLI_PR" ]]  && echo "Pull Request:    #${CLI_PR} (${CLI_REPO})"
[[ -n "$CLI_SHA" ]] && echo "Commit SHA:      ${CLI_SHA}"
echo "Context:         ${TARGET_CONTEXT}"
echo "Target Label:    ${TARGET_LABEL}"
echo "========================================================================"

echo "Checking fleet status in gs://${CLI_BUCKET}/fleet/..."
FLEET_COUNT=$(gcloud storage ls "gs://${CLI_BUCKET}/fleet/" 2>/dev/null | grep -c '\.json$' || true)
if [[ "$FLEET_COUNT" -eq 0 ]]; then
  echo "WARNING: No healthy TPU VMs currently registered in gs://${CLI_BUCKET}/fleet/." >&2
else
  echo "Found ${FLEET_COUNT} registered TPU VM(s) in fleet."
fi

RESOLVED_TARGETS=()
if [[ ${#CLI_TARGETS[@]} -gt 0 ]]; then
  RESOLVED_TARGETS=("${CLI_TARGETS[@]}")
else
  echo "Querying test targets in ${CLI_SCOPE} matching ${CLI_FILTER}..."
  pos_tags=()
  neg_tags=()
  IFS=',' read -r -a filter_items <<< "$CLI_FILTER"
  for item in "${filter_items[@]}"; do
    item=$(echo "$item" | xargs)
    [[ -z "$item" ]] && continue
    if [[ "$item" == -* ]]; then
      neg_tags+=("${item#-}")
    else
      pos_tags+=("$item")
    fi
  done

  query_expr="tests(${CLI_SCOPE})"
  if [[ ${#pos_tags[@]} -gt 0 ]]; then
    pos_pattern=$(IFS='|'; echo "${pos_tags[*]}")
    query_expr="${query_expr} intersect attr(tags, \"${pos_pattern}\", ${CLI_SCOPE})"
  fi
  if [[ ${#neg_tags[@]} -gt 0 ]]; then
    neg_pattern=$(IFS='|'; echo "${neg_tags[*]}")
    query_expr="${query_expr} except attr(tags, \"${neg_pattern}\", ${CLI_SCOPE})"
  fi

  while IFS= read -r line; do
    [[ -n "$line" && "$line" == //* ]] && RESOLVED_TARGETS+=("$line")
  done < <(bazel query "$query_expr" --output=label 2>/dev/null || true)
fi

TARGET_COUNT="${#RESOLVED_TARGETS[@]}"
echo "Resolved ${TARGET_COUNT} target(s) to execute."

if [[ "$TARGET_COUNT" -eq 0 ]]; then
  echo "ERROR: No test targets resolved." >&2
  exit 1
fi

if [[ "$CLI_DRY_RUN" == "true" ]]; then
  echo "[DRY RUN] Execution plan confirmed. Resolved targets ($TARGET_COUNT):"
  printf "  %s\n" "${RESOLVED_TARGETS[@]}"
  if [[ "$CLI_ADD_LABEL" == "true" && -n "$CLI_PR" ]]; then
    echo "[DRY RUN] Would add label '${TARGET_LABEL}' to PR #${CLI_PR}"
  fi
  echo "[DRY RUN] Would report under context: '${TARGET_CONTEXT}'"
  exit 0
fi

if [[ -n "$CLI_PR" && "$CLI_ADD_LABEL" == "true" && "$CLI_NO_REPORT" == "false" ]]; then
  if command -v gh >/dev/null 2>&1; then
    echo "Adding label '${TARGET_LABEL}' to PR #${CLI_PR}..."
    gh pr edit "$CLI_PR" -R "$CLI_REPO" --add-label "$TARGET_LABEL" || {
      echo "WARNING: Failed to add label '${TARGET_LABEL}' to PR #${CLI_PR}." >&2
    }
  fi
fi

if [[ -n "$CLI_SHA" && "$CLI_NO_REPORT" == "false" ]]; then
  post_status "$CLI_SHA" "pending" "TPU mailbox presubmit running across ${FLEET_COUNT} chips" "$TARGET_CONTEXT"
  _STATUS_POSTED=true
fi

BAZEL_ARGS=("test")

if [[ -n "$CLI_BAZEL_CONFIG" ]]; then
  if [[ ${#CLI_TARGETS[@]} -eq 0 && "$CLI_BAZEL_CONFIG" == "ci_tpu_v5_mailbox" ]]; then
    BAZEL_ARGS+=("--config=ci_tpu_v5_presubmit")
  fi
  BAZEL_ARGS+=("--config=${CLI_BAZEL_CONFIG}")
fi

BAZEL_ARGS+=(
  "--test_env=TORCH_TPU_RELAY_BUCKET=${CLI_BUCKET}"
  "--run_under=//ci/tools/rbe_tpu:worker_main"
  "--test_timeout=${CLI_TEST_TIMEOUT}"
  "--jobs=${CLI_JOBS}"
  "--local_test_jobs=${CLI_JOBS}"
  "--test_output=errors"
  "--nocache_test_results"
)

if [[ ${#CLI_BAZEL_FLAGS[@]} -gt 0 ]]; then
  BAZEL_ARGS+=("${CLI_BAZEL_FLAGS[@]}")
fi

BAZEL_ARGS+=("${RESOLVED_TARGETS[@]}")

LOG_FILE="${CLI_OUTPUT_DIR}/bazel_test.log"
REPORT_FILE="${CLI_OUTPUT_DIR}/report.md"

START_TIME=$(date +%s)
echo "Starting Bazel test execution (logs -> ${LOG_FILE})..."

set +e
bazel "${BAZEL_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
BAZEL_RC=$?
set -e

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

VERDICT="PASSED"
STATUS_STATE="success"
[[ "$BAZEL_RC" -ne 0 ]] && VERDICT="FAILED" && STATUS_STATE="failure"

echo "Bazel execution completed in ${ELAPSED}s with verdict: ${VERDICT} (rc=${BAZEL_RC})"

cat > "$REPORT_FILE" <<RPT
${COMMENT_MARKER}
# TPU Mailbox Presubmit Report

**Verdict:** ${VERDICT}
**Mode:** ${CLI_MODE}
**Context:** \`${TARGET_CONTEXT}\`
**Wall Clock Time:** ${ELAPSED}s
**Bucket:** \`gs://${CLI_BUCKET}\`
**Targets Evaluated:** ${TARGET_COUNT}
**Fleet Capacity:** ${FLEET_COUNT} chips
**Timestamp:** ${NOW_TAG}

## Summary

| Metric | Value |
| :--- | :--- |
| **Status** | **${VERDICT}** |
| **Duration** | ${ELAPSED}s |
| **Exit Code** | ${BAZEL_RC} |
| **Targets** | ${TARGET_COUNT} |
| **Fleet Capacity** | ${FLEET_COUNT} chips registered |

RPT

if [[ "$BAZEL_RC" -ne 0 ]]; then
  echo "### Failures" >> "$REPORT_FILE"
  echo '```' >> "$REPORT_FILE"
  grep -E 'FAIL:|TIMEOUT:|ERROR:' "$LOG_FILE" | head -n 30 >> "$REPORT_FILE" || true
  echo '```' >> "$REPORT_FILE"
fi

echo "Report generated at: ${REPORT_FILE}"

if [[ -n "$CLI_SHA" && "$CLI_NO_REPORT" == "false" ]]; then
  post_status "$CLI_SHA" "$STATUS_STATE" "TPU mailbox presubmit ${VERDICT} in ${ELAPSED}s (${TARGET_COUNT} targets)" "$TARGET_CONTEXT"
fi

if [[ -n "$CLI_PR" && "$CLI_NO_REPORT" == "false" ]]; then
  if command -v gh >/dev/null 2>&1; then
    EXISTING_COMMENT_ID="$(gh api "repos/${CLI_REPO}/issues/${CLI_PR}/comments" --jq ".[] | select(.body | contains(\"${COMMENT_MARKER}\")) | .id" 2>/dev/null | tail -n 1 || true)"
    if [[ -n "$EXISTING_COMMENT_ID" ]]; then
      echo "Updating existing report comment ${EXISTING_COMMENT_ID} on PR #${CLI_PR}..."
      gh api -X PATCH "repos/${CLI_REPO}/issues/comments/${EXISTING_COMMENT_ID}" -f body="$(cat "$REPORT_FILE")" >/dev/null 2>&1 || {
        echo "WARNING: Failed to update comment on PR #${CLI_PR}." >&2
      }
    else
      echo "Posting report to PR #${CLI_PR} on ${CLI_REPO}..."
      gh pr comment "$CLI_PR" -R "$CLI_REPO" --body-file "$REPORT_FILE" || {
        echo "WARNING: Failed to post comment to PR #${CLI_PR}." >&2
      }
    fi
  fi
fi

exit "$BAZEL_RC"
