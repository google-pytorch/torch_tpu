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

# rbe_tpu/deploy/rbe_tpu_bazel_test.sh
# Runs arbitrary `bazel test` commands across the Cloud TPU v5e fleet via GCS mailbox.
#
# Examples:
#   rbe_tpu/deploy/rbe_tpu_bazel_test.sh //tests/pallas:pallas_test
#   rbe_tpu/deploy/rbe_tpu_bazel_test.sh //tests/pallas:pallas_test -- --runs_per_test=20
#   rbe_tpu/deploy/rbe_tpu_bazel_test.sh //tests/pallas/...
#   rbe_tpu/deploy/rbe_tpu_bazel_test.sh --scope //tests/pallas/... --filter presubmit-v5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
if [[ -d "${SCRIPT_DIR}/../../ci/tools/rbe_tpu" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
readonly REPO_ROOT

DEFAULT_BUCKET="torch-tpu-mailbox-eval"
DEFAULT_SCOPE="//tests/..."

CLI_TARGETS=()
CLI_BAZEL_FLAGS=()
CLI_BUCKET="$DEFAULT_BUCKET"
CLI_JOBS=28
CLI_BAZEL_CONFIG="rbe_tpu"
CLI_SCOPE=""
CLI_FILTER=""
CLI_OUTPUT_DIR=""
CLI_DRY_RUN=false
CLI_TEST_TIMEOUT="900"
CLI_KEEP=false

log() { echo "[rbe_tpu_test] $*"; }
die() { echo "ERROR [rbe_tpu_test]: $*" >&2; exit 1; }

show_help() {
  cat <<'EOF'
Usage: rbe_tpu/deploy/rbe_tpu_bazel_test.sh [options] [TARGET...] [-- BAZEL_FLAG...]

Runs bazel test on the standing Cloud TPU v5e fleet in rbe-tpu-oss via the GCS
mailbox runner. Supports arbitrary target labels, package patterns, and bazel flags.

Examples:
  rbe_tpu/deploy/rbe_tpu_bazel_test.sh //tests/pallas:pallas_test
  rbe_tpu/deploy/rbe_tpu_bazel_test.sh //tests/pallas:pallas_test -- --runs_per_test=20
  rbe_tpu/deploy/rbe_tpu_bazel_test.sh //tests/pallas/...
  rbe_tpu/deploy/rbe_tpu_bazel_test.sh --scope //tests/pallas/... --filter presubmit-v5

Targets can be exact labels or wildcard patterns (e.g. //tests/pallas/...).
Wildcard patterns are expanded with `bazel query tests(...)`.
With no targets specified, --scope and --filter determine what runs.

Options:
  --bucket NAME         GCS mailbox bucket (default: torch-tpu-mailbox-eval).
  --jobs N              Concurrent test jobs (default: 28).
  --bazel-config NAME   Bazel config to inherit (default: ci_tpu_v5_mailbox).
                        Pass "" for standalone execution.
  --bazel-flag FLAG     Extra flag passed to bazel test. Can be repeated.
  --scope PATTERN       Where to look when no targets are named (default: //tests/...).
  --filter TAG_EXPR     Tag filter when scope is used. Default: none.
  --timeout SECONDS     Per-test timeout in seconds (default: 900).
  --output-dir DIR      Destination directory for logs and reports.
  --dry-run             Expand targets, show execution plan, and exit.
  --keep                Retained for CLI parity (mailbox fleet is always available).
  -h, --help            Show this help message.

Flags after `--` are passed directly to `bazel test` (e.g. --runs_per_test=20).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      while [[ $# -gt 0 ]]; do
        CLI_BAZEL_FLAGS+=("$1")
        shift
      done
      break
      ;;
    --bucket)
      [[ $# -ge 2 ]] || die "--bucket requires an argument."
      CLI_BUCKET="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || die "--jobs requires an argument."
      [[ "$2" =~ ^[0-9]+$ ]] || die "--jobs must be a number."
      CLI_JOBS="$2"
      shift 2
      ;;
    --bazel-config)
      [[ $# -ge 2 ]] || die "--bazel-config requires an argument."
      CLI_BAZEL_CONFIG="$2"
      shift 2
      ;;
    --bazel-flag)
      [[ $# -ge 2 ]] || die "--bazel-flag requires an argument."
      CLI_BAZEL_FLAGS+=("$2")
      shift 2
      ;;
    --scope)
      [[ $# -ge 2 ]] || die "--scope requires an argument."
      CLI_SCOPE="$2"
      shift 2
      ;;
    --filter)
      [[ $# -ge 2 ]] || die "--filter requires an argument."
      CLI_FILTER="$2"
      shift 2
      ;;
    --timeout)
      [[ $# -ge 2 ]] || die "--timeout requires an argument."
      [[ "$2" =~ ^[0-9]+$ ]] || die "--timeout must be a number."
      CLI_TEST_TIMEOUT="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires an argument."
      CLI_OUTPUT_DIR="$2"
      shift 2
      ;;
    --dry-run)
      CLI_DRY_RUN=true
      shift
      ;;
    --keep)
      CLI_KEEP=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    -*)
      die "Unknown option: $1"
      ;;
    *)
      CLI_TARGETS+=("$1")
      shift
      ;;
  esac
done

SCRIPT_START_TIME=$(date +%s)
NOW_TAG="$(date -u +%Y%m%d_%H%M%SZ)"
if [[ -z "$CLI_OUTPUT_DIR" ]]; then
  CLI_OUTPUT_DIR="${REPO_ROOT}/mailbox_reports/test_${NOW_TAG}"
fi
mkdir -p "$CLI_OUTPUT_DIR"

FLEET_CHECK_START=$(date +%s)
log "Checking fleet status in gs://${CLI_BUCKET}/fleet/..."
FLEET_COUNT=$(gcloud storage ls "gs://${CLI_BUCKET}/fleet/" 2>/dev/null | grep -c '\.json$' || true)
if [[ "$FLEET_COUNT" -eq 0 ]]; then
  echo "WARNING: No healthy TPU VMs currently registered in gs://${CLI_BUCKET}/fleet/." >&2
else
  log "Found ${FLEET_COUNT} registered TPU VM(s) in fleet."
fi
FLEET_CHECK_END=$(date +%s)
FLEET_CHECK_ELAPSED=$((FLEET_CHECK_END - FLEET_CHECK_START))

QUERY_START=$(date +%s)
RESOLVED_TARGETS=()

if [[ ${#CLI_TARGETS[@]} -gt 0 ]]; then
  for tgt in "${CLI_TARGETS[@]}"; do
    if [[ "$tgt" == *:* && "$tgt" != *...* && "$tgt" != *:all* ]]; then
      RESOLVED_TARGETS+=("$tgt")
    else
      log "Expanding pattern: ${tgt}"
      while IFS= read -r line; do
        [[ -n "$line" && "$line" == //* ]] && RESOLVED_TARGETS+=("$line")
      done < <(bazel query "tests(${tgt})" --output=label 2>/dev/null || true)
    fi
  done
else
  target_scope="${CLI_SCOPE:-$DEFAULT_SCOPE}"
  log "Querying test targets in ${target_scope}..."
  if [[ -n "$CLI_FILTER" ]]; then
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

    query_expr="tests(${target_scope})"
    if [[ ${#pos_tags[@]} -gt 0 ]]; then
      pos_pattern=$(IFS='|'; echo "${pos_tags[*]}")
      query_expr="${query_expr} intersect attr(tags, \"${pos_pattern}\", ${target_scope})"
    fi
    if [[ ${#neg_tags[@]} -gt 0 ]]; then
      neg_pattern=$(IFS='|'; echo "${neg_tags[*]}")
      query_expr="${query_expr} except attr(tags, \"${neg_pattern}\", ${target_scope})"
    fi
  else
    query_expr="tests(${target_scope})"
  fi

  while IFS= read -r line; do
    [[ -n "$line" && "$line" == //* ]] && RESOLVED_TARGETS+=("$line")
  done < <(bazel query "$query_expr" --output=label 2>/dev/null || true)
fi

FINAL_TARGETS=()
for tgt in "${RESOLVED_TARGETS[@]}"; do
  if [[ "$tgt" == *":8"* ]] || [[ "$tgt" == *"8tpu"* ]]; then
    log "Skipping multi-chip target: ${tgt} (only single-chip v5litepod-1 supported)"
    continue
  fi
  FINAL_TARGETS+=("$tgt")
done

QUERY_END=$(date +%s)
QUERY_ELAPSED=$((QUERY_END - QUERY_START))

TARGET_COUNT="${#FINAL_TARGETS[@]}"
log "Resolved ${TARGET_COUNT} target(s) to execute."

if [[ "$TARGET_COUNT" -eq 0 ]]; then
  die "No test targets resolved."
fi

if [[ "$CLI_DRY_RUN" == "true" ]]; then
  echo "[DRY RUN] Execution plan confirmed. Resolved targets ($TARGET_COUNT):"
  printf "  %s\n" "${FINAL_TARGETS[@]}"
  if [[ ${#CLI_BAZEL_FLAGS[@]} -gt 0 ]]; then
    echo "[DRY RUN] Extra bazel flags:"
    printf "  %s\n" "${CLI_BAZEL_FLAGS[@]}"
  fi
  exit 0
fi

BAZEL_ARGS=("test")

if [[ -n "$CLI_BAZEL_CONFIG" ]]; then
  BAZEL_ARGS+=("--config=${CLI_BAZEL_CONFIG}")
fi

BAZEL_ARGS+=(
  "--test_env=TORCH_TPU_RELAY_BUCKET=${CLI_BUCKET}"
  "--run_under=//rbe_tpu:worker_main"
  "--test_timeout=${CLI_TEST_TIMEOUT}"
  "--jobs=${CLI_JOBS}"
  "--local_test_jobs=${CLI_JOBS}"
  "--test_output=errors"
  "--nocache_test_results"
)

if [[ ${#CLI_BAZEL_FLAGS[@]} -gt 0 ]]; then
  BAZEL_ARGS+=("${CLI_BAZEL_FLAGS[@]}")
fi

BAZEL_ARGS+=("${FINAL_TARGETS[@]}")

LOG_FILE="${CLI_OUTPUT_DIR}/bazel_test.log"
REPORT_FILE="${CLI_OUTPUT_DIR}/report.md"

START_TIME=$(date +%s)
log "Starting Bazel test execution (logs -> ${LOG_FILE})..."

set +e
bazel "${BAZEL_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
BAZEL_RC=$?
set -e

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
TOTAL_ELAPSED=$((END_TIME - SCRIPT_START_TIME))

BAZEL_CRITICAL_PATH="N/A"
if [[ -f "$LOG_FILE" ]]; then
  CRIT_LINE=$(grep -E "INFO: Elapsed time: .* Critical Path:" "$LOG_FILE" | tail -n 1 || true)
  if [[ "$CRIT_LINE" =~ Critical\ Path:\ ([0-9.]+)s ]]; then
    BAZEL_CRITICAL_PATH="${BASH_REMATCH[1]}s"
  fi
fi

VERDICT="PASSED"
[[ "$BAZEL_RC" -ne 0 ]] && VERDICT="FAILED"

log "Bazel execution completed in ${ELAPSED}s with verdict: ${VERDICT} (rc=${BAZEL_RC})"
log "=== Stage Timing Breakdown ==="
log "  1. Fleet discovery (GCS):  ${FLEET_CHECK_ELAPSED}s"
log "  2. Target resolution:      ${QUERY_ELAPSED}s"
log "  3. Bazel execution total:  ${ELAPSED}s (Critical path: ${BAZEL_CRITICAL_PATH})"
log "  Total wall clock:          ${TOTAL_ELAPSED}s"

cat > "$REPORT_FILE" <<RPT
# TPU Mailbox Test Run Report

**Verdict:** ${VERDICT}
**Wall Clock Time:** ${TOTAL_ELAPSED}s
**Bucket:** \`gs://${CLI_BUCKET}\`
**Targets Evaluated:** ${TARGET_COUNT}
**Fleet Capacity:** ${FLEET_COUNT} chips
**Timestamp:** ${NOW_TAG}

## Summary

| Metric | Value |
| :--- | :--- |
| **Status** | **${VERDICT}** |
| **Total Duration** | ${TOTAL_ELAPSED}s |
| **Exit Code** | ${BAZEL_RC} |
| **Targets** | ${TARGET_COUNT} |
| **Fleet Capacity** | ${FLEET_COUNT} chips registered |

## Stage Timing Breakdown

| Stage | Duration |
| :--- | :--- |
| **1. Fleet Discovery (GCS)** | ${FLEET_CHECK_ELAPSED}s |
| **2. Target Resolution** | ${QUERY_ELAPSED}s |
| **3. Bazel Execution Total** | ${ELAPSED}s |
| **3a. Test Critical Path** | ${BAZEL_CRITICAL_PATH} |
| **Total Wall Clock** | **${TOTAL_ELAPSED}s** |

RPT

if [[ "$BAZEL_RC" -ne 0 ]]; then
  cat >> "$REPORT_FILE" <<'FAILURES'
### Failures
```
FAILURES
  grep -E 'FAIL:|TIMEOUT:|ERROR:' "$LOG_FILE" | head -n 30 >> "$REPORT_FILE" || true
  echo '```' >> "$REPORT_FILE"
fi

log "Report generated at: ${REPORT_FILE}"
exit "$BAZEL_RC"
