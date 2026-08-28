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
# ci/tools/relay_test_runner.sh
# Bazel --run_under test runner wrapper for physical Cloud TPU execution.
set -uo pipefail

readonly ALLOWED_PROJECT="rbe-tpu-oss"

readonly TEST_BIN="${1:-}"
if [[ -z "$TEST_BIN" ]]; then
  echo "ERROR [relay_test_runner]: No test binary specified." >&2
  echo "Usage: $0 <test_binary> [test_args...]" >&2
  exit 1
fi
shift || true
readonly TEST_ARGS=("$@")

SESSION_ENV="${TPU_SESSION_ENV:-/tmp/tpu_active_session.env}"
readonly TARGET_LABEL="${TEST_TARGET:-$(basename "$TEST_BIN")}"
readonly CLEAN_TARGET="$(echo "$TARGET_LABEL" | tr '/:' '__')"
readonly SANDBOX_ID="${CLEAN_TARGET}_$$_${RANDOM}"
readonly REMOTE_RELAY_DIR="/tmp/torch_tpu_relay"
readonly REMOTE_EXECUTOR="${REMOTE_RELAY_DIR}/remote_tpu_executor.sh"
readonly REMOTE_PAYLOAD_CACHE="${REMOTE_RELAY_DIR}/payloads"
# ci/tools/stage_relay_base.sh names this after the content it staged, so two
# runs sharing a VM pool cannot overwrite each other. A bare bazel run with no
# driver falls back to the unversioned path.
readonly REMOTE_BASE_CACHE="${TORCH_TPU_RELAY_BASE_DIR:-${REMOTE_RELAY_DIR}/base}"
readonly REMOTE_SANDBOX="${REMOTE_RELAY_DIR}/sandboxes/${SANDBOX_ID}"
readonly TIMEOUT_RAW="${TEST_TIMEOUT:-900}"
_clean_timeout="${TIMEOUT_RAW%[sS]}"
if [[ "$_clean_timeout" =~ ^[0-9]+$ && "$_clean_timeout" -gt 0 ]]; then
  readonly TIMEOUT_SEC="$_clean_timeout"
else
  readonly TIMEOUT_SEC="900"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/tools/relay_ssh.sh
source "${SCRIPT_DIR}/relay_ssh.sh"
readonly SSH_OPTS=("${RELAY_SSH_OPTS[@]}")

# Runs a command on the TPU VM over the shared ControlMaster socket.
rsh() {
  ssh -S "$SSH_CONTROL_PATH" "${SSH_OPTS[@]}" "${SSH_USER}@${TPU_IP}" "$@"
}

# Phase timing, off unless TORCH_TPU_RELAY_TIMING is set. Each mark reports
# seconds since this action started and seconds since the previous mark, so a
# run tells you which phase to go after rather than just how long it all took.
RELAY_T0="${EPOCHREALTIME/,/.}"
RELAY_TPREV="$RELAY_T0"
mark() {
  [[ -n "${TORCH_TPU_RELAY_TIMING:-}" ]] || return 0
  local now="${EPOCHREALTIME/,/.}"
  printf '[relay-timing] %-18s +%6.2fs  total %6.2fs\n' \
    "$1" \
    "$(awk -v a="$now" -v b="$RELAY_TPREV" 'BEGIN{printf "%.2f", a-b}')" \
    "$(awk -v a="$now" -v b="$RELAY_T0" 'BEGIN{printf "%.2f", a-b}')" >&2
  RELAY_TPREV="$now"
}

# Writes a single-case JUnit report when the remote run left none behind. With
# no arguments the case passes; with a message it fails.
write_stub_xml() {
  local failure_msg="${1:-}"
  local failure_detail="${2:-}"
  local xml_path="${XML_OUTPUT_FILE:-}"
  [[ -n "$xml_path" && ! -s "$xml_path" ]] || return 0

  local failures=0
  local testcase_tail="/>"
  if [[ -n "$failure_msg" ]]; then
    failures=1
    testcase_tail=">
      <failure message=\"${failure_msg}\"><![CDATA[${failure_detail}]]></failure>
    </testcase>"
  fi

  mkdir -p "$(dirname "$xml_path")"
  cat <<XML_EOF > "$xml_path"
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="${CLEAN_TARGET}" tests="1" failures="${failures}" errors="0" time="0.0">
    <testcase classname="${CLEAN_TARGET}" name="execution" time="0.0"${testcase_tail}
  </testsuite>
</testsuites>
XML_EOF
}

# Host-level trap for cleanup on premature termination
cleanup_host() {
  local origin_rc=$?
  local trap_sig="${1:-EXIT}"
  trap - EXIT INT TERM HUP
  local exit_code=1
  case "$trap_sig" in
    INT)  exit_code=130 ;;
    TERM) exit_code=143 ;;
    HUP)  exit_code=129 ;;
    *)    exit_code="$origin_rc" ;;
  esac

  if [[ -n "${SSH_CONTROL_PATH:-}" && -S "${SSH_CONTROL_PATH}" && -n "${TPU_IP:-}" && -n "${SSH_USER:-}" ]]; then
    rsh "rm -rf '${REMOTE_SANDBOX}'" 2>/dev/null || true
  fi

  if [[ "$exit_code" -ne 0 ]]; then
    write_stub_xml "Test interrupted by signal ${trap_sig}" "Process received signal ${trap_sig} (exit code ${exit_code})."
  fi

  exit "$exit_code"
}
trap 'cleanup_host EXIT' EXIT
trap 'cleanup_host INT' INT
trap 'cleanup_host TERM' TERM
trap 'cleanup_host HUP' HUP

# Enforce GCP project boundary strictly to ALLOWED_PROJECT
enforce_project_boundary() {
  local target_project="${1:-$ALLOWED_PROJECT}"
  if [[ -n "$target_project" && "$target_project" != "$ALLOWED_PROJECT" ]]; then
    echo "ERROR [relay_test_runner]: Project boundary violation: '$target_project' is not allowed." >&2
    echo "This runner is strictly restricted to project '$ALLOWED_PROJECT'." >&2
    write_stub_xml "Project Boundary Violation" "Project '$target_project' violates allowed project '$ALLOWED_PROJECT'."
    exit 1
  fi

  local env_vars=("CLOUDSDK_CORE_PROJECT" "GOOGLE_CLOUD_PROJECT" "GCP_PROJECT" "GCLOUD_PROJECT")
  for var in "${env_vars[@]}"; do
    if [[ -n "${!var:-}" && "${!var}" != "$ALLOWED_PROJECT" ]]; then
      echo "ERROR [relay_test_runner]: Project boundary violation: Environment variable $var is set to '${!var}' (violates allowed project '$ALLOWED_PROJECT')." >&2
      write_stub_xml "Project Boundary Violation" "Environment variable $var violates allowed project '$ALLOWED_PROJECT'."
      exit 1
    fi
  done
}

# 1. Environment discovery and session validation
enforce_project_boundary "${TPU_PROJECT:-$ALLOWED_PROJECT}"

# Point TPU_SESSION_POOL at a directory of session files to spread tests over a
# fleet. Each action holds an exclusive lock on one VM for as long as it runs,
# the same way parallel_accelerator_execute.sh leases a single chip. The lock fd
# stays open until this process exits, so a crashed test can't strand a VM.
#
# A VM that turns out to be unusable gets a .quarantine marker and is passed
# over from then on. Without that, a broken VM is the *fastest* member of the
# pool -- it fails in milliseconds and frees its lock immediately, so it wins
# the next race and the one after that. One rebooted VM failed 64 shards this
# way while seven healthy VMs sat idle.
quarantine_session() {
  local session="$1"
  local reason="$2"
  printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$reason" > "${session}.quarantine" 2>/dev/null || true
  echo "WARNING [relay_test_runner]: took $(basename "$session") out of the pool: ${reason}" >&2
}

lease_session_from_pool() {
  local deadline=$((SECONDS + TIMEOUT_SEC))
  while true; do
    local candidate
    for candidate in "$TPU_SESSION_POOL"/*.env; do
      [[ -f "$candidate" ]] || continue
      [[ -f "${candidate}.quarantine" ]] && continue
      local fd
      exec {fd}>"${candidate}.lock" || continue
      if flock -n "$fd"; then
        SESSION_ENV="$candidate"
        return 0
      fi
      exec {fd}>&-
    done
    if (( SECONDS >= deadline )); then
      return 1
    fi
    sleep 1
  done
}

if [[ -n "${TPU_SESSION_POOL:-}" ]]; then
  if ! lease_session_from_pool; then
    echo "ERROR [relay_test_runner]: No free TPU VM in $TPU_SESSION_POOL after ${TIMEOUT_SEC}s." >&2
    write_stub_xml "No free TPU in pool" "Every VM in $TPU_SESSION_POOL stayed busy for ${TIMEOUT_SEC}s."
    exit 1
  fi
fi
mark leased
readonly SESSION_ENV

if [[ ! -f "$SESSION_ENV" ]]; then
  echo "ERROR [relay_test_runner]: Active TPU session file $SESSION_ENV not found." >&2
  echo "Provision a Spot TPU first using: scripts/spot_tpu_manager.sh up" >&2
  write_stub_xml "No active TPU session" "Session file $SESSION_ENV not found on host."
  exit 1
fi

# shellcheck disable=SC1090
source "$SESSION_ENV"

TPU_PROJECT="$ALLOWED_PROJECT"

if [[ -z "${TPU_IP:-}" || -z "${SSH_USER:-}" || -z "${SSH_CONTROL_PATH:-}" ]]; then
  echo "ERROR [relay_test_runner]: Incomplete TPU session environment in $SESSION_ENV." >&2
  write_stub_xml "Incomplete TPU session" "Required variables missing in $SESSION_ENV (TPU_IP, SSH_USER, or SSH_CONTROL_PATH)."
  exit 1
fi

# 2. ControlMaster socket health check, auto-reconnection, and preemption detection
check_ssh_socket() {
  relay_ssh_master_alive "${SSH_CONTROL_PATH:-}" "${SSH_USER}@${TPU_IP}"
}

reconnect_ssh() {
  echo "WARNING [relay_test_runner]: SSH ControlMaster socket is not active. Attempting reconnection..." >&2
  relay_ssh_ensure_master "$SSH_CONTROL_PATH" "${SSH_USER}@${TPU_IP}"
}

check_preemption() {
  local vm_state query_rc

  # Keep "the VM is gone" separate from "I could not ask". Collapsing the two
  # reported 64 shards as preempted against a VM that was READY the whole time;
  # the real fault was local, and the report blamed Google Cloud for it.
  vm_state=$(gcloud compute tpus tpu-vm describe "${TPU_NAME:-}" \
    --project="$ALLOWED_PROJECT" \
    --zone="${TPU_ZONE:-europe-west4-b}" \
    --format="value(state)" 2>/dev/null)
  query_rc=$?

  if [[ "$query_rc" -ne 0 || -z "$vm_state" ]]; then
    echo "ERROR [relay_test_runner]: SSH to ${TPU_IP} failed and the state of '${TPU_NAME:-}' could not be read (gcloud rc=${query_rc})." >&2
    write_stub_xml "TPU State Unknown" \
      "SSH to ${TPU_IP} failed. Could not read the state of ${TPU_NAME:-} to tell preemption from a local fault."
    return 1
  fi

  if [[ "$vm_state" == "PREEMPTED" || "$vm_state" == "TERMINATED" ]]; then
    echo "FATAL [relay_test_runner]: TPU VM '${TPU_NAME:-}' in zone '${TPU_ZONE:-europe-west4-b}' is gone (state: $vm_state)." >&2
    write_stub_xml "Spot TPU Preempted" "VM ${TPU_NAME:-} entered state $vm_state during run."
    return 0
  fi

  # The node is alive but unreachable, which is what a guest reboot looks like:
  # the SSH control master dies with it and every later action fails instantly.
  echo "ERROR [relay_test_runner]: Cannot reach TPU VM at ${TPU_IP} over SSH though the node reports $vm_state." >&2
  write_stub_xml "SSH Connection Failed" "Host unreachable at ${TPU_IP} (node state: $vm_state)."
  return 1
}

if ! check_ssh_socket; then
  if ! reconnect_ssh; then
    [[ -z "${TPU_SESSION_POOL:-}" ]] || quarantine_session "$SESSION_ENV" "unreachable over SSH"
    check_preemption || true
    exit 255
  fi
fi
mark ssh_ready

# A TPU VM that reboots comes back with /tmp emptied, so the base cache that
# ci/tools/stage_relay_base.sh put there is gone. SSH still works and the node
# still reports READY, so nothing upstream notices: the test just runs without
# an interpreter and dies with a bare exit 1. Check before spending a payload
# on it.
if ! rsh "test -d '${REMOTE_BASE_CACHE}'" >/dev/null 2>&1; then
  echo "ERROR [relay_test_runner]: ${TPU_NAME:-the VM} has no base cache at ${REMOTE_BASE_CACHE}; it needs re-staging." >&2
  [[ -z "${TPU_SESSION_POOL:-}" ]] || quarantine_session "$SESSION_ENV" "base cache missing (rebooted?)"
  write_stub_xml "TPU Base Cache Missing" \
    "${REMOTE_BASE_CACHE} is gone on ${TPU_NAME:-the VM}. Re-run ci/tools/stage_relay_base.sh for this session."
  exit 1
fi
mark base_probe

# 3. Synchronize remote_tpu_executor.sh to remote TPU VM
ensure_remote_executor() {
  local local_executor="${SCRIPT_DIR}/remote_tpu_executor.sh"

  if [[ ! -f "$local_executor" ]]; then
    echo "ERROR [relay_test_runner]: $local_executor is missing." >&2
    write_stub_xml "Relay executor missing" "Expected $local_executor next to relay_test_runner.sh."
    exit 1
  fi

  local local_hash
  local_hash=$(sha256sum "$local_executor" | awk '{print $1}')

  # Ship the file only when the copy on the VM is stale or absent. Sandboxes
  # come and go, but the executor is shared by every test on this VM.
  rsh "if [ \"\$(sha256sum ${REMOTE_EXECUTOR} 2>/dev/null | awk '{print \$1}')\" != \"$local_hash\" ]; then mkdir -p ${REMOTE_RELAY_DIR} && cat > ${REMOTE_EXECUTOR} && chmod +x ${REMOTE_EXECUTOR}; fi" \
    < "$local_executor" 2>/dev/null || true
}
ensure_remote_executor
mark executor_sync

# 4. Resolve runfiles root directory
RUNFILES_ROOT="${TEST_SRCDIR:-}"
if [[ -z "$RUNFILES_ROOT" || ! -d "$RUNFILES_ROOT" ]]; then
  if [[ -d "${TEST_BIN}.runfiles" ]]; then
    RUNFILES_ROOT="${TEST_BIN}.runfiles"
  elif [[ -d "../_main" ]]; then
    RUNFILES_ROOT="$(cd .. && pwd)"
  elif [[ -d "./_main" ]]; then
    RUNFILES_ROOT="$(pwd)"
  elif [[ -d "${WORKSPACE_ROOT:-}" ]]; then
    RUNFILES_ROOT="${WORKSPACE_ROOT}"
  else
    RUNFILES_ROOT="$(dirname "$TEST_BIN")"
  fi
fi

# Safely escape test arguments for remote shell execution
remote_args=""
if [[ ${#TEST_ARGS[@]} -gt 0 ]]; then
  printf -v remote_args "%q " "${TEST_ARGS[@]}"
fi

# Bazel hands a sharded test its slice through TEST_SHARD_INDEX, and rejects the
# result unless the runner creates TEST_SHARD_STATUS_FILE to prove it understood
# the request. The test binary would normally create that file itself, but it
# runs on the VM where bazel can't see it, so forward the slice and stamp the
# file here.
remote_env=""
if [[ -n "${TEST_TOTAL_SHARDS:-}" ]]; then
  remote_env="TEST_SHARD_INDEX=${TEST_SHARD_INDEX:-0} TEST_TOTAL_SHARDS=${TEST_TOTAL_SHARDS} "
  [[ -z "${TEST_SHARD_STATUS_FILE:-}" ]] || : > "$TEST_SHARD_STATUS_FILE"
fi

# A test rule's `env = {...}` attribute lands in the test action's environment,
# which this wrapper inherits. The test itself runs on the VM, so whatever is
# not forwarded here is simply absent there. That cost real coverage:
# errors_test_tpu lost TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS=1, so XLA deferred
# the errors the test asserts on and three cases stopped raising; pallas_test
# lost TPU_PREMAPPED_BUFFER_SIZE=0 and stopped tracking buffer donation. All of
# it passes on the upstream ct5lp runner, where the test and the environment
# stay on one machine.
#
# Bazel does not say which variables came from the attribute, so forward by
# prefix. These cover every `env` key under tests/**/BUILD.
readonly FORWARDED_ENV_PREFIXES=(
  TORCH_ TPU_ XLA_ JAX_ PJRT_ LIBTPU_ CUDA_ IS_OSS
)

# Names that describe the host side of the hop or belong to the relay itself.
# TPU_NAME matters to libtpu on the VM and the session file's value is the wrong
# thing to hand it; the rest are set explicitly below.
relay_owns_var() {
  case "$1" in
    TPU_NAME|TPU_ZONE|TPU_IP|TPU_PROJECT) return 0 ;;
    TORCH_TPU_RELAY_*|TORCH_TPU_BASE_CACHE|TORCH_TPU_PAYLOAD_KEY) return 0 ;;
    *) return 1 ;;
  esac
}

while IFS='=' read -r -d '' _env_name _env_value; do
  relay_owns_var "$_env_name" && continue
  for _prefix in "${FORWARDED_ENV_PREFIXES[@]}"; do
    if [[ "$_env_name" == "$_prefix"* ]]; then
      remote_env+="${_env_name}=$(printf '%q' "$_env_value") "
      break
    fi
  done
done < <(env -0)
unset _env_name _env_value _prefix

# Set after the loop so the relay's own values win if a name collides.
[[ -z "${TORCH_TPU_RELAY_TIMING:-}" ]] || remote_env+="TORCH_TPU_RELAY_TIMING=1 "

remote_env+="TORCH_TPU_BASE_CACHE=${REMOTE_BASE_CACHE} "

# Every shard of a target ships the same runfiles tree, and the big ops targets
# have 30-50 shards each. Name the tree so the VM can keep a prepared copy and
# hardlink it for the later shards instead of unpacking it again.
#
# The run id comes from run_presubmit_v5_relay.sh, which builds every target
# before any test starts. Within one run a tree cannot change underneath us, and
# a key from a previous run is never reused, so a stale tree cannot be served.
# No run id means no caching, which is the safe default for a bare bazel run.
payload_key=""
if [[ -n "${TORCH_TPU_RELAY_RUN_ID:-}" ]]; then
  payload_key="${TORCH_TPU_RELAY_RUN_ID}_$(printf '%s' "$RUNFILES_ROOT" | sha256sum | cut -c1-16)"
  remote_env+="TORCH_TPU_PAYLOAD_KEY=${payload_key} "
fi

# 5. Stream this test's own files and run it
#
# The payload goes over as three concatenated archives, because the runfiles
# tree mixes several kinds of link that need different treatment.
#
#   1. Everything outside the venv, dereferenced. Bazel points these at
#      absolute paths in the local output base, which mean nothing on the VM.
#   2. The venv itself, links intact. Its site-packages is a farm of relative
#      links back up at the dependency repositories, which resolve on the VM
#      once the base cache is linked in at the sandbox root.
#   3. The few absolute links inside the venv that point somewhere other than a
#      staged repository -- pyvenv.cfg and the bazel .pth files. Those are real
#      content and have to be dereferenced.
#
# The three sets are kept strictly disjoint. GNU tar 1.34, which is what the TPU
# image ships, will not replace an existing symlink with a regular file: it
# keeps the link and still reports success. So whichever archive writes a name
# first wins, and (2) has to skip everything (3) is about to send.
#
# Deliberately not in (3): the ~41 venv links into external/, which name torch,
# libtpu and friends. Dereferencing those drags 1.7 GB through every single
# test action. remote_tpu_executor.sh repoints them at the base cache instead.
#
# Left out entirely: the dependency repositories, the shared solibs and this
# repo's own extension modules under _main/csrc, which
# ci/tools/stage_relay_base.sh already put on the VM. csrc matters most:
# libpywrap_torch_tpu_common.so is 493 MB of the 499 MB payload, and it is the
# same build for every test.
readonly PAYLOAD_EXCLUDES=(
  --exclude='*rules_python*'
  --exclude='*/_solib_x86_64*'
  --exclude='_solib_x86_64'
  --exclude='_main/csrc'
  --exclude='*.a'
  --exclude='*.o'
  --exclude='*.params'
  --exclude='*.cppmap'
  --exclude='*__pycache__*'
  --exclude='*.pyc'
  --exclude='MANIFEST'
  --exclude='_repo_mapping'
)

stream_payload() {
  cd "$RUNFILES_ROOT" || return 1

  tar -czhf - --ignore-failed-read "${PAYLOAD_EXCLUDES[@]}" \
    --exclude='*.venv' . 2>/dev/null

  local venvs=()
  mapfile -t venvs < <(find . -maxdepth 4 -name '*.venv' -type d 2>/dev/null)
  [[ ${#venvs[@]} -gt 0 ]] || return 0

  local host_only
  host_only="$(mktemp)"
  find "${venvs[@]}" -type l -lname '/*' ! -lname '*/external/*' \
    > "$host_only" 2>/dev/null || true

  tar -czf - --ignore-failed-read "${PAYLOAD_EXCLUDES[@]}" \
    --exclude-from="$host_only" "${venvs[@]}" 2>/dev/null

  [[ ! -s "$host_only" ]] || tar -czhf - --ignore-failed-read -T "$host_only" 2>/dev/null
  rm -f "$host_only"
}

run_remote() {
  timeout --foreground --signal=TERM --kill-after=10s "${TIMEOUT_SEC}s" \
    ssh -S "$SSH_CONTROL_PATH" "${SSH_OPTS[@]}" "${SSH_USER}@${TPU_IP}" \
      "${remote_env}${REMOTE_EXECUTOR} '${REMOTE_SANDBOX}' '${TEST_BIN}' -- ${remote_args}"
}

# Ask before sending. The probe rides the existing control master, so it costs
# a few milliseconds against the ~20s of tar and transfer it avoids.
payload_cached=0
if [[ -n "$payload_key" ]] &&
   rsh "test -d '${REMOTE_PAYLOAD_CACHE}/${payload_key}'" >/dev/null 2>&1; then
  payload_cached=1
fi
mark payload_probe

if [[ "$payload_cached" -eq 1 ]]; then
  run_remote < /dev/null
  ssh_rc=$?
  mark "run_cache_hit"
else
  stream_payload | run_remote
  pipestatus=("${PIPESTATUS[@]}")
  ssh_rc="${pipestatus[1]:-0}"
  mark "run_cache_miss"
fi

# 6. Fetch remote test metadata (exitcode and test.xml) and clean remote sandbox
remote_meta=$(rsh "cat '${REMOTE_SANDBOX}/test.exitcode' 2>/dev/null || true; echo '---TORCH_TPU_SPLIT---'; cat '${REMOTE_SANDBOX}/test.xml' 2>/dev/null || true; mv '${REMOTE_SANDBOX}' '${REMOTE_SANDBOX}.trash' 2>/dev/null && { setsid rm -rf '${REMOTE_SANDBOX}.trash' </dev/null >/dev/null 2>&1 & }" 2>/dev/null)
meta_rc=$?
mark fetch_and_release

remote_exitcode=$(echo "$remote_meta" | awk '/---TORCH_TPU_SPLIT---/{exit} {print}' | tr -d ' \r\n')
remote_xml=$(echo "$remote_meta" | awk 'f{print} /---TORCH_TPU_SPLIT---/{f=1}')

# Disarm cleanup trap since remote sandbox was deleted above
trap - EXIT INT TERM HUP

# 7. Turn what came back from the VM into a report and an exit code.
#
# Reads ssh_rc (how the test invocation ended), meta_rc (whether we managed to
# read the result back), remote_exitcode and remote_xml. Every path has to leave
# a JUnit report behind: bazel shows a bare "FAILED" with no explanation if it
# finds none.
resolve_outcome() {
  local xml_written=0
  if [[ -n "$remote_xml" && "$remote_xml" =~ \<testsuite && -n "${XML_OUTPUT_FILE:-}" ]]; then
    mkdir -p "$(dirname "$XML_OUTPUT_FILE")"
    echo "$remote_xml" > "$XML_OUTPUT_FILE"
    xml_written=1
  fi

  if [[ "$ssh_rc" -eq 124 ]]; then
    echo "ERROR [relay_test_runner]: Test action timed out after ${TIMEOUT_SEC}s." >&2
    write_stub_xml "Test Timeout" "Test execution timed out after ${TIMEOUT_SEC} seconds."
    return 124
  fi

  # 255 is ambiguous: ssh uses it for transport failures, but a test is also
  # free to exit with it. The exit code the VM recorded settles which happened.
  if [[ "$ssh_rc" -eq 255 ]]; then
    if [[ "$remote_exitcode" == "255" ]]; then
      [[ "$xml_written" -eq 1 ]] || write_stub_xml \
        "Test failed with exit code 255" "Process explicitly exited with code 255."
      return 255
    fi
    check_preemption || true
    return 255
  fi

  if [[ "$ssh_rc" -ne 0 ]]; then
    [[ "$xml_written" -eq 1 ]] || write_stub_xml \
      "Test failed with exit code $ssh_rc" \
      "Process exited with code $ssh_rc without generating valid test.xml."
    return "$ssh_rc"
  fi

  [[ "$xml_written" -eq 0 ]] || return 0

  # A clean exit with no report is only a pass if we know the report is
  # genuinely absent. If the fetch failed we never saw the VM's answer, and
  # calling that a pass invents a result nobody observed.
  if [[ "$meta_rc" -ne 0 || -z "$remote_exitcode" ]]; then
    write_stub_xml "Test Report Not Retrieved" \
      "The test exited 0 but its report could not be read back from ${TPU_NAME:-the VM} (fetch rc ${meta_rc})."
    return 1
  fi

  write_stub_xml
  return 0
}

resolve_outcome
exit $?
