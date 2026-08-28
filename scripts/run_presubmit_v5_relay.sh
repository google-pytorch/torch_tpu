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
# scripts/run_presubmit_v5_relay.sh
# Presubmit execution pipeline on physical Cloud TPU v5e via SSH relay.
set -euo pipefail
set -m

readonly ALLOWED_PROJECT="rbe-tpu-oss"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Honors TPU_SESSION_ENV so several relays can share a host, each pinned to its
# own TPU VM. relay_test_runner.sh reads the same variable.
readonly SESSION_ENV_FILE="${TPU_SESSION_ENV:-/tmp/tpu_active_session.env}"
if [[ -n "${SPOT_TPU_MANAGER_BIN:-}" ]]; then
  SPOT_MANAGER="${SPOT_TPU_MANAGER_BIN}"
elif command -v spot_tpu_manager.sh >/dev/null 2>&1 && [[ "$(command -v spot_tpu_manager.sh)" != "${SCRIPT_DIR}/spot_tpu_manager.sh" ]]; then
  SPOT_MANAGER="$(command -v spot_tpu_manager.sh)"
else
  SPOT_MANAGER="${SCRIPT_DIR}/spot_tpu_manager.sh"
fi
readonly SPOT_MANAGER

readonly RELAY_RUNNER="${REPO_ROOT}/ci/tools/relay_test_runner.sh"
readonly REPORTER="${GENERATE_PRESUBMIT_REPORT_BIN:-${SCRIPT_DIR}/generate_presubmit_report.py}"
readonly RELAY_STAGER="${STAGE_RELAY_BASE_BIN:-${REPO_ROOT}/ci/tools/stage_relay_base.sh}"

# Default configuration values
CLI_ZONE="europe-west4-b"
# The relay leases one v5litepod-1 VM per test action, so a test that asks for
# more than one chip can never pass here. Those targets carry a chip count in
# their tag (requires-tpu-v5lite:8); the plain tag means a single chip. Left in,
# they burn a full timeout each and then report as ordinary failures.
CLI_FILTER="presubmit-v5,-fails-on-tpu-v5,-nopresubmit,-notest,-nobuild,-requires-tpu-v5lite:8"
CLI_TARGETS=""
CLI_DRY_RUN=false
CLI_OUTPUT_DIR=""
CLI_TEST_TIMEOUT="900s"
CLI_KEEP_VM_ON_FAILURE=false
CLI_SESSION_POOL=""
CLI_JOBS=""
CLI_PROJECT="$ALLOWED_PROJECT"
# Empty means a plain local build, which is what a developer wants on a warm
# output base. CI has a cold one and needs --bazel-config=ci_tpu_v5_relay to
# send the compile actions to RBE.
CLI_BAZEL_CONFIG=""
# Extra flags appended after --config on both bazel invocations, so a caller can
# override anything a config pulled in. Repeated flags: the last one wins.
CLI_BAZEL_FLAGS=()

# Process supervision state
_BAZEL_PID=""
_TEARDOWN_DONE=0
_PRESUBMIT_FAILED=0
_FINAL_EXIT_CODE=0

show_help() {
  cat <<'EOF'
Usage: scripts/run_presubmit_v5_relay.sh [options]

Executes the torch_tpu presubmit-v5 test suite on physical Cloud TPU silicon
via the SSH relay test runner and Spot TPU instances in rbe-tpu-oss.

Options:
  --zone=ZONE              GCP zone for Spot TPU VM (default: europe-west4-b)
  --filter=TAG_EXPR        Test tag filter expression
                           (default: presubmit-v5,-fails-on-tpu-v5,-nopresubmit,-notest,-nobuild)
  --targets=LABELS         Comma- or space-separated target labels or package pattern
                           (default: all targets in //tests/... matching filter tags)
  --dry-run                Resolve target list, print plan, and exit without provisioning VM
  --output-dir=DIR         Directory for test logs and reports
                           (default: presubmit_reports/run_<timestamp>)
  --test-timeout=SEC       Per-test timeout duration (default: 900s)
  --keep-vm-on-failure     Do not delete TPU VM on test failure or error (for debugging)
  --session-pool=DIR       Run against a fleet built by scripts/spot_tpu_fleet.sh instead
                           of provisioning a single VM. Each test leases one VM from DIR.
  --jobs=N                 Tests to run at once (default: 1, or the pool size with
                           --session-pool). Never set this above the number of VMs.
  --project=PROJECT        GCP project (strictly restricted to rbe-tpu-oss)
  --bazel-config=NAME      Extra --config=NAME for the build and test invocations.
                           Use ci_tpu_v5_relay on a cold machine: it sends the
                           compile actions to RBE and still downloads the
                           runfiles trees the base cache is built from.
  --bazel-flag=FLAG        Extra flag for the build and test invocations, after
                           --config so it wins. Repeatable. The driver uses it
                           to pass --bes_backend= and skip the ResultStore
                           upload, which a personal account cannot do.
  -h, --help               Show this help message and exit
EOF
}

# 1. Project boundary enforcement
enforce_project_boundary() {
  local target_project="${1:-$ALLOWED_PROJECT}"
  if [[ "$target_project" != "$ALLOWED_PROJECT" ]]; then
    echo "ERROR [run_presubmit]: Project boundary violation: '$target_project' is not allowed." >&2
    echo "This presubmit runner is strictly restricted to project '$ALLOWED_PROJECT'." >&2
    exit 1
  fi

  local env_vars=("CLOUDSDK_CORE_PROJECT" "GOOGLE_CLOUD_PROJECT" "GCP_PROJECT" "GCLOUD_PROJECT" "TPU_PROJECT")
  for var in "${env_vars[@]}"; do
    if [[ -n "${!var:-}" && "${!var}" != "$ALLOWED_PROJECT" ]]; then
      echo "ERROR [run_presubmit]: Project boundary violation: Environment variable $var is set to '${!var}' (violates allowed project '$ALLOWED_PROJECT')." >&2
      exit 1
    fi
  done
}

# 2. CLI option parsing
parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --zone)
        [[ $# -lt 2 ]] && { echo "ERROR [run_presubmit]: --zone requires an argument." >&2; exit 1; }
        CLI_ZONE="$2"; shift 2 ;;
      --zone=*)
        CLI_ZONE="${1#*=}"; shift ;;
      --filter)
        [[ $# -lt 2 ]] && { echo "ERROR [run_presubmit]: --filter requires an argument." >&2; exit 1; }
        CLI_FILTER="$2"; shift 2 ;;
      --filter=*)
        CLI_FILTER="${1#*=}"; shift ;;
      --targets)
        [[ $# -lt 2 ]] && { echo "ERROR [run_presubmit]: --targets requires an argument." >&2; exit 1; }
        CLI_TARGETS="$2"; shift 2 ;;
      --targets=*)
        CLI_TARGETS="${1#*=}"; shift ;;
      --dry-run)
        CLI_DRY_RUN=true; shift ;;
      --output-dir)
        [[ $# -lt 2 ]] && { echo "ERROR [run_presubmit]: --output-dir requires an argument." >&2; exit 1; }
        CLI_OUTPUT_DIR="$2"; shift 2 ;;
      --output-dir=*)
        CLI_OUTPUT_DIR="${1#*=}"; shift ;;
      --test-timeout)
        [[ $# -lt 2 ]] && { echo "ERROR [run_presubmit]: --test-timeout requires an argument." >&2; exit 1; }
        CLI_TEST_TIMEOUT="$2"; shift 2 ;;
      --test-timeout=*)
        CLI_TEST_TIMEOUT="${1#*=}"; shift ;;
      --keep-vm-on-failure)
        CLI_KEEP_VM_ON_FAILURE=true; shift ;;
      --session-pool)
        [[ $# -lt 2 ]] && { echo "ERROR [run_presubmit]: --session-pool requires an argument." >&2; exit 1; }
        CLI_SESSION_POOL="$2"; shift 2 ;;
      --session-pool=*)
        CLI_SESSION_POOL="${1#*=}"; shift ;;
      --jobs)
        [[ $# -lt 2 ]] && { echo "ERROR [run_presubmit]: --jobs requires an argument." >&2; exit 1; }
        CLI_JOBS="$2"; shift 2 ;;
      --jobs=*)
        CLI_JOBS="${1#*=}"; shift ;;
      --project)
        [[ $# -lt 2 ]] && { echo "ERROR [run_presubmit]: --project requires an argument." >&2; exit 1; }
        CLI_PROJECT="$2"; shift 2 ;;
      --project=*)
        CLI_PROJECT="${1#*=}"; shift ;;
      --bazel-config)
        [[ $# -lt 2 ]] && { echo "ERROR [run_presubmit]: --bazel-config requires an argument." >&2; exit 1; }
        CLI_BAZEL_CONFIG="$2"; shift 2 ;;
      --bazel-config=*)
        CLI_BAZEL_CONFIG="${1#*=}"; shift ;;
      --bazel-flag)
        [[ $# -lt 2 ]] && { echo "ERROR [run_presubmit]: --bazel-flag requires an argument." >&2; exit 1; }
        CLI_BAZEL_FLAGS+=("$2"); shift 2 ;;
      --bazel-flag=*)
        CLI_BAZEL_FLAGS+=("${1#*=}"); shift ;;
      -h|--help)
        show_help; exit 0 ;;
      *)
        echo "ERROR [run_presubmit]: Unknown option: $1" >&2
        show_help >&2
        exit 1 ;;
    esac
  done

  enforce_project_boundary "$CLI_PROJECT"

  # Bazel's --test_timeout only takes a bare integer, but "900s" reads more
  # clearly on the command line and TEST_TIMEOUT accepts both.
  CLI_TEST_TIMEOUT="${CLI_TEST_TIMEOUT%[sS]}"
  [[ "$CLI_TEST_TIMEOUT" =~ ^[0-9]+$ ]] \
    || { echo "ERROR [run_presubmit]: --test-timeout must be seconds, got '$CLI_TEST_TIMEOUT'." >&2; exit 1; }
}

# 3. Session health inspection
is_tpu_session_healthy() {
  if [[ ! -f "$SESSION_ENV_FILE" ]]; then
    return 1
  fi

  local status_output=""
  if ! status_output=$("$SPOT_MANAGER" status --project="$ALLOWED_PROJECT" 2>&1); then
    return 1
  fi

  if [[ "$status_output" == *"NO_ACTIVE_SESSION"* ]]; then
    return 1
  fi

  if [[ "$status_output" == *"INACTIVE"* || "$status_output" == *"CLOSED"* ]]; then
    return 1
  fi

  return 0
}

# 4. Signal traps and teardown supervisor
cleanup_orchestrator() {
  local origin_rc=$?
  local sig="${1:-EXIT}"

  if [[ "$_TEARDOWN_DONE" -eq 1 ]]; then
    return
  fi
  _TEARDOWN_DONE=1
  trap '' INT TERM HUP

  local exit_code=0
  if [[ "$sig" != "EXIT" ]]; then
    case "$sig" in
      INT)  exit_code=130 ;;
      TERM) exit_code=143 ;;
      HUP)  exit_code=129 ;;
      *)    exit_code=1 ;;
    esac
    _PRESUBMIT_FAILED=1
    echo -e "\nCaught signal $sig. Initiating clean termination..." >&2
  else
    exit_code="$origin_rc"
    if [[ "$exit_code" -ne 0 ]]; then
      _PRESUBMIT_FAILED=1
    fi
  fi

  # Terminate active Bazel child process tree
  local b_pid="${_BAZEL_PID:-}"
  if [[ -n "$b_pid" ]] && (kill -0 "$b_pid" 2>/dev/null || kill -0 -"$b_pid" 2>/dev/null); then
    echo "Terminating Bazel process group (PID: $b_pid)..." >&2
    kill -TERM -"$b_pid" 2>/dev/null || kill -TERM "$b_pid" 2>/dev/null || true
    local elapsed=0
    while (kill -0 "$b_pid" 2>/dev/null || kill -0 -"$b_pid" 2>/dev/null) && [[ $elapsed -lt 20 ]]; do
      sleep 0.5
      elapsed=$(( elapsed + 1 ))
    done
    if kill -0 "$b_pid" 2>/dev/null || kill -0 -"$b_pid" 2>/dev/null; then
      echo "Bazel process did not shut down gracefully; sending SIGKILL..." >&2
      kill -KILL -"$b_pid" 2>/dev/null || kill -KILL "$b_pid" 2>/dev/null || true
    fi
    wait "$b_pid" 2>/dev/null || true
    _BAZEL_PID=""
  fi

  # Only tear down a VM this script provisioned. A fleet belongs to whoever
  # built it, so leave it for `spot_tpu_fleet.sh down`.
  if [[ "$CLI_DRY_RUN" == "false" && -z "$CLI_SESSION_POOL" ]]; then
    if [[ "$sig" == "EXIT" && "$CLI_KEEP_VM_ON_FAILURE" == "true" && "$_PRESUBMIT_FAILED" -eq 1 ]]; then
      local tpu_vm="unknown" tpu_z="unknown" tpu_ip="unknown"
      if [[ -f "$SESSION_ENV_FILE" ]]; then
        eval "$(grep -E '^(export )?(TPU_NAME|TPU_ZONE|TPU_IP)=' "$SESSION_ENV_FILE" 2>/dev/null || true)"
        tpu_vm="${TPU_NAME:-unknown}"
        tpu_z="${TPU_ZONE:-unknown}"
        tpu_ip="${TPU_IP:-unknown}"
      fi
      echo "========================================================================"
      echo "[PRESERVED] Spot TPU VM preserved for debugging (--keep-vm-on-failure):"
      echo "  VM Name:    $tpu_vm"
      echo "  Zone:       $tpu_z"
      echo "  Project:    $ALLOWED_PROJECT"
      echo "  IP:         $tpu_ip"
      echo "To connect:   gcloud compute tpus tpu-vm ssh '$tpu_vm' --zone='$tpu_z' --project='$ALLOWED_PROJECT'"
      echo "To teardown:  $SPOT_MANAGER down --project='$ALLOWED_PROJECT'"
      echo "========================================================================"
    else
      echo "Executing Spot TPU teardown..."
      "$SPOT_MANAGER" down --project="$ALLOWED_PROJECT" || true
    fi
  fi

  trap - EXIT INT TERM HUP
  exit "$exit_code"
}

# 5. Target enumeration and query
resolve_targets() {
  local target_arg="${1:-}"
  local filter_arg="${2:-}"
  local targets=()

  if [[ -n "$target_arg" ]]; then
    # Parse explicit targets or scope wildcard
    local clean_arg
    clean_arg=$(echo "$target_arg" | tr ',' ' ')
    read -r -a raw_targets <<< "$clean_arg"

    local has_wildcard=false
    for item in "${raw_targets[@]}"; do
      if [[ "$item" == *"..."* || "$item" == *":all"* ]]; then
        has_wildcard=true
        break
      fi
    done

    if [[ "$has_wildcard" == "true" ]]; then
      for pattern in "${raw_targets[@]}"; do
        if [[ "$pattern" == *"..."* || "$pattern" == *":all"* ]]; then
          local q_out
          q_out=$(bazel query "tests(${pattern})" --output=label 2>/dev/null || true)
          while IFS= read -r line; do
            [[ -n "$line" && "$line" == //* ]] && targets+=("$line")
          done <<< "$q_out"
        else
          targets+=("$pattern")
        fi
      done
    else
      targets=("${raw_targets[@]}")
    fi
  else
    # Build query expression from filter tags
    local pos_tags=()
    local neg_tags=()

    IFS=',' read -r -a filter_items <<< "$filter_arg"
    for item in "${filter_items[@]}"; do
      item=$(echo "$item" | xargs)
      [[ -z "$item" ]] && continue
      if [[ "$item" == -* ]]; then
        neg_tags+=("${item#-}")
      else
        pos_tags+=("$item")
      fi
    done

    local query_expr="tests(//tests/...)"
    if [[ ${#pos_tags[@]} -gt 0 ]]; then
      local pos_pattern
      pos_pattern=$(IFS='|'; echo "${pos_tags[*]}")
      query_expr="${query_expr} intersect attr(tags, \"${pos_pattern}\", //tests/...)"
    fi
    if [[ ${#neg_tags[@]} -gt 0 ]]; then
      local neg_pattern
      neg_pattern=$(IFS='|'; echo "${neg_tags[*]}")
      query_expr="${query_expr} except attr(tags, \"${neg_pattern}\", //tests/...)"
    fi

    local query_out
    query_out=$(bazel query "$query_expr" --output=label 2>/dev/null || true)
    while IFS= read -r line; do
      [[ -n "$line" && "$line" == //* ]] && targets+=("$line")
    done <<< "$query_out"
  fi

  if [[ ${#targets[@]} -eq 0 ]]; then
    echo "ERROR [run_presubmit]: No test targets found matching criteria." >&2
    exit 1
  fi

  printf "%s\n" "${targets[@]}"
}

# The flag list for the test invocation, in one place. --dry-run prints what
# main() is about to run rather than a hand-copied echo of it, which had already
# drifted once.
bazel_test_flags() {
  local jobs="$1"
  # --config goes first: a flag repeated on the command line takes the last
  # value, and everything below is what this script actually needs.
  [[ -z "$CLI_BAZEL_CONFIG" ]] || printf '%s\n' "--config=${CLI_BAZEL_CONFIG}"
  printf '%s\n' \
    "--run_under=${RELAY_RUNNER}" \
    "--modify_execution_info=TestRunner=+no-remote-exec" \
    "--strategy=TestRunner=local" \
    "--local_test_jobs=${jobs}" \
    "--keep_going" \
    "--nocache_test_results" \
    "--test_output=errors" \
    "--test_summary=detailed" \
    "--test_tag_filters=${CLI_FILTER}" \
    "--test_timeout=${CLI_TEST_TIMEOUT}"
  # Last, so a caller can override any of the above.
  (( ${#CLI_BAZEL_FLAGS[@]} == 0 )) || printf '%s\n' "${CLI_BAZEL_FLAGS[@]}"
}

# 6. Main execution flow
main() {
  parse_args "$@"

  # Set up output directory
  if [[ -z "$CLI_OUTPUT_DIR" ]]; then
    CLI_OUTPUT_DIR="${REPO_ROOT}/presubmit_reports/run_$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "$CLI_OUTPUT_DIR"

  # Register lifecycle supervisor traps
  trap 'cleanup_orchestrator EXIT' EXIT
  trap 'cleanup_orchestrator INT' INT
  trap 'cleanup_orchestrator TERM' TERM
  trap 'cleanup_orchestrator HUP' HUP

  # Resolve target list
  local resolved_targets_str
  resolved_targets_str=$(resolve_targets "$CLI_TARGETS" "$CLI_FILTER")
  local target_list=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && target_list+=("$line")
  done <<< "$resolved_targets_str"
  local total_count="${#target_list[@]}"

  local targets_file="${CLI_OUTPUT_DIR}/targets.txt"
  printf "%s\n" "${target_list[@]}" > "$targets_file"

  echo "========================================================================"
  echo "torch_tpu Presubmit-v5 Relay Execution Pipeline"
  echo "========================================================================"
  echo "Target Count:        $total_count"
  echo "GCP Zone:            $CLI_ZONE"
  echo "GCP Project:         $ALLOWED_PROJECT"
  echo "Tag Filter:          $CLI_FILTER"
  echo "Output Directory:    $CLI_OUTPUT_DIR"
  echo "Test Timeout:        $CLI_TEST_TIMEOUT"
  echo "Keep VM on Failure:  $CLI_KEEP_VM_ON_FAILURE"
  echo "Concurrency:         ${CLI_JOBS:-1 test at a time (one chip per VM)}"
  echo "========================================================================"

  # Handle dry-run execution
  if [[ "$CLI_DRY_RUN" == "true" ]]; then
    echo -e "\n[DRY RUN] Execution plan confirmed. Resolved test targets ($total_count):"
    for t in "${target_list[@]}"; do
      echo "  - $t"
    done
    echo -e "\n[DRY RUN] Planned Bazel invocation:"
    echo "bazel test \\"
    local flag
    while IFS= read -r flag; do
      echo "  ${flag} \\"
    done < <(bazel_test_flags "${CLI_JOBS:-1}")
    for t in "${target_list[@]}"; do
      echo "  $t \\"
    done

    # Generate dry-run reporting artifacts
    python3 "$REPORTER" \
      --output-dir="$CLI_OUTPUT_DIR" \
      --workspace-root="$REPO_ROOT" \
      --targets-file="$targets_file" \
      --dry-run >/dev/null 2>&1 || true

    echo -e "\nDry run complete. Zero resources provisioned."
    return 0
  fi

  # A pool is provisioned ahead of time by scripts/spot_tpu_fleet.sh, and each
  # test action leases one VM out of it. Otherwise fall back to a single VM,
  # which means one test at a time because it only has one chip.
  # Scopes the VM-side payload cache to this run. Every target is built before
  # any test starts, so within one run a target's runfiles tree is fixed and a
  # shard can safely reuse the tree an earlier shard unpacked.
  local run_started_at
  run_started_at="$(date +%s)"
  local run_id="r${run_started_at}p$$"
  local relay_env=(
    --test_env=TPU_SESSION_ENV="$SESSION_ENV_FILE"
    --test_env=TORCH_TPU_RELAY_RUN_ID="$run_id"
  )
  local jobs=1

  # Opt-in phase timing. Bazel scrubs the environment, so it only reaches the
  # relay if it is forwarded explicitly.
  [[ -z "${TORCH_TPU_RELAY_TIMING:-}" ]] \
    || relay_env+=(--test_env=TORCH_TPU_RELAY_TIMING=1)

  if [[ -n "$CLI_SESSION_POOL" ]]; then
    local pool_size
    pool_size=$(find "$CLI_SESSION_POOL" -maxdepth 1 -name '*.env' 2>/dev/null | wc -l)
    [[ "$pool_size" -gt 0 ]] || {
      echo "ERROR [run_presubmit]: No session files in $CLI_SESSION_POOL." >&2
      echo "Build a fleet first: scripts/spot_tpu_fleet.sh up --size N" >&2
      return 1
    }
    relay_env=(
      --test_env=TPU_SESSION_POOL="$CLI_SESSION_POOL"
      --test_env=TORCH_TPU_RELAY_RUN_ID="$run_id"
    )
    [[ -z "${TORCH_TPU_RELAY_TIMING:-}" ]] \
      || relay_env+=(--test_env=TORCH_TPU_RELAY_TIMING=1)
    jobs="${CLI_JOBS:-$pool_size}"
    echo "Running against a fleet of ${pool_size} TPU VMs, ${jobs} tests at a time."
  else
    jobs="${CLI_JOBS:-1}"
    if is_tpu_session_healthy; then
      echo "Found active and healthy Spot TPU session. Reusing existing instance."
      "$SPOT_MANAGER" status --project="$ALLOWED_PROJECT"
    else
      echo "No healthy active TPU session found. Initializing new Spot TPU in '$CLI_ZONE'..."
      if [[ -f "$SESSION_ENV_FILE" ]]; then
        "$SPOT_MANAGER" down --project="$ALLOWED_PROJECT" 2>/dev/null || true
      fi
      "$SPOT_MANAGER" up --zone="$CLI_ZONE" --project="$ALLOWED_PROJECT"
    fi
  fi

  # stage_relay_base.sh reads the runfiles trees bazel leaves in bazel-bin, so
  # anything not built yet is invisible to it and its dependencies never reach
  # the VMs. Build first, then stage, then test. Skipping this shows up as a
  # bare ModuleNotFoundError from a test that is otherwise fine.
  echo -e "\nBuilding test targets so the base cache can see every dependency..."
  local build_flags=()
  [[ -z "$CLI_BAZEL_CONFIG" ]] || build_flags+=("--config=${CLI_BAZEL_CONFIG}")
  build_flags+=(${CLI_BAZEL_FLAGS[@]+"${CLI_BAZEL_FLAGS[@]}"})
  if ! bazel build \
    ${build_flags[@]+"${build_flags[@]}"} \
    --test_tag_filters="$CLI_FILTER" \
    "${target_list[@]}" > >(tee "${CLI_OUTPUT_DIR}/bazel_build.log") 2>&1; then
    echo "ERROR [run_presubmit]: build failed; see ${CLI_OUTPUT_DIR}/bazel_build.log" >&2
    return 1
  fi

  local stage_args=(--session "$SESSION_ENV_FILE")
  [[ -z "$CLI_SESSION_POOL" ]] || stage_args=(--pool "$CLI_SESSION_POOL")
  local base_dir_file="${CLI_OUTPUT_DIR}/remote_base_dir.txt"
  echo -e "\nStaging the shared base cache..."
  "$RELAY_STAGER" "${stage_args[@]}" --emit-base-dir "$base_dir_file" || {
    echo "ERROR [run_presubmit]: base cache staging failed." >&2
    return 1
  }

  # The stager names the base directory after what it staged, so the tests have
  # to be told where it landed. Without this they would read whatever the last
  # run left at the unversioned path.
  local remote_base_dir=""
  [[ ! -s "$base_dir_file" ]] || remote_base_dir="$(cat "$base_dir_file")"
  if [[ -n "$remote_base_dir" ]]; then
    relay_env+=(--test_env=TORCH_TPU_RELAY_BASE_DIR="$remote_base_dir")
    echo "Tests will read the base cache from ${remote_base_dir}."
  fi

  echo -e "\nStarting test execution through relay runner..."
  local bazel_log="${CLI_OUTPUT_DIR}/bazel_presubmit.log"
  local start_time
  start_time=$(date +%s)

  local test_flags=()
  mapfile -t test_flags < <(bazel_test_flags "$jobs")

  set +e
  set -m
  bazel test \
    "${test_flags[@]}" \
    "${relay_env[@]}" \
    "${target_list[@]}" > >(tee "$bazel_log") 2>&1 &
  _BAZEL_PID=$!

  wait "$_BAZEL_PID"
  local bazel_rc=$?
  _BAZEL_PID=""
  wait
  set -e

  local end_time
  end_time=$(date +%s)
  local duration=$(( end_time - start_time ))

  echo "Bazel execution completed in ${duration}s with exit code $bazel_rc."

  if [[ $bazel_rc -ne 0 ]]; then
    _PRESUBMIT_FAILED=1
    _FINAL_EXIT_CODE="$bazel_rc"
  fi

  # Result aggregation and report synthesis
  echo "Synthesizing presubmit test reports in '$CLI_OUTPUT_DIR'..."
  set +e
  python3 "$REPORTER" \
    --output-dir="$CLI_OUTPUT_DIR" \
    --bazel-log="$bazel_log" \
    --workspace-root="$REPO_ROOT" \
    --targets-file="$targets_file" \
    --duration="$duration" \
    --bazel-exit-code="$bazel_rc" \
    --run-started-at="$run_started_at" \
    --session-env="$SESSION_ENV_FILE" \
    --session-pool="$CLI_SESSION_POOL"
  local report_rc=$?
  set -e

  if [[ $_FINAL_EXIT_CODE -eq 0 && $report_rc -ne 0 ]]; then
    _FINAL_EXIT_CODE="$report_rc"
  fi

  return "$_FINAL_EXIT_CODE"
}

main "$@"
