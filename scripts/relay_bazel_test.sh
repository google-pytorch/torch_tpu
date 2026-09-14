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

# scripts/relay_bazel_test.sh
# Runs any `bazel test` on the Cloud TPU v5e fleet. No pull request, no tag
# filter, no report: just the targets you name, spread across the chips.
#
#   scripts/relay_bazel_test.sh //tests/pallas:pallas_test
#   scripts/relay_bazel_test.sh //tests/... -- --runs_per_test=20
#
# The pieces underneath already worked without a PR; what was missing was
# someone to lease the fleet, hand the pool to the runner, and give the VMs
# back afterwards. That is all this does.
#
# Leasing is the part worth getting right. This script gives the fleet back on
# the way out however it exits, including Ctrl-C, so an abandoned run does not
# leave 28 chips claimed until the four-hour TTL expires. If --pool already
# holds sessions the fleet is assumed to be somebody else's and is left alone.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT

FLEET="${SPOT_TPU_FLEET_BIN:-${SCRIPT_DIR}/spot_tpu_fleet.sh}"
RELAY="${RUN_PRESUBMIT_V5_RELAY_BIN:-${SCRIPT_DIR}/run_presubmit_v5_relay.sh}"

CLI_TARGETS=()
CLI_BAZEL_FLAGS=()
CLI_ZONES=()
CLI_POOL=""
CLI_JOBS=""
# RBE for the compile actions, which is the reason to run this rather than a
# plain `bazel test`. Pass "" for a local build on a warm output base.
CLI_BAZEL_CONFIG="ci_tpu_v5_relay"
CLI_SCOPE=""
# No tag filter. The presubmit needs one because it is picking a suite out of
# the tree; naming your own targets is already the filter.
CLI_FILTER=""
CLI_IP_MODE="auto"
CLI_OUTPUT_DIR=""
CLI_REPORT=false
CLI_UPLOAD_RESULTS=false
CLI_WAIT=true
CLI_WAIT_TIMEOUT=3600
CLI_KEEP=false
CLI_DRY_RUN=false

# Set once this run is the one holding the fleet, so the exit trap knows whether
# the VMs are ours to give back.
_ATTACHED=false

log() { echo "[relay_test] $*"; }
die() { echo "ERROR [relay_test]: $*" >&2; exit 1; }

show_help() {
  cat <<'EOF'
Usage: scripts/relay_bazel_test.sh [options] TARGET... [-- BAZEL_FLAG...]

Runs bazel test on the standing Cloud TPU v5e fleet in rbe-tpu-oss. Borrows the
VMs, runs, and gives them back.

  scripts/relay_bazel_test.sh //tests/pallas:pallas_test
  scripts/relay_bazel_test.sh //tests/... -- --runs_per_test=20
  scripts/relay_bazel_test.sh --scope //tests/pallas/... --filter presubmit-v5

Targets are bazel labels or patterns. A pattern containing ... or :all is
expanded with `bazel query tests(...)`, so only test rules make it through.
With no targets at all, --scope and --filter decide what runs.

Options:
  --pool DIR            Session pool. Default: /tmp/tpu_relay_test_<user>
                        If it already holds sessions, those VMs are reused and
                        left attached on exit, because they are not ours.
  --zone ZONE           Zone to look for fleet VMs in. Repeatable.
                        Default: whatever spot_tpu_fleet.sh uses.
  --jobs N              Tests at once. Default: however many VMs attached.
                        Never set this above the VM count.
  --bazel-config NAME   Default: ci_tpu_v5_relay, which sends compile actions
                        to RBE. Pass "" for a plain local build.
  --scope PATTERN       Where to look when no targets are named.
                        Default: //tests/...
  --filter TAG_EXPR     Tag filter. Default: none, so everything named runs.
  --ip-mode MODE        auto (default), internal or external. auto picks
                        external on a corp workstation and internal on a GCE VM
                        in the fleet's VPC.
  --output-dir DIR      Where logs land.
                        Default: relay_reports/test_<timestamp>
  --report              Also synthesize the presubmit-style report. Off by
                        default; bazel's own output is usually what you want.
  --upload-results      Send build events to ResultStore. Off by default: only
                        the CI service account can write there, and the upload
                        fails the run after the tests have already passed.
  --no-wait             Fail instead of queueing when somebody has the fleet.
  --wait-timeout SECS   How long to stand in line. Default: 3600.
  --keep                Leave the fleet attached on exit, so a follow-up run
                        starts instantly. You are then holding it: run
                        `spot_tpu_fleet.sh detach --pool DIR` when done.
  --dry-run             Print the plan and exit without touching hardware.
  -h, --help            This message.

Every test action leases one v5litepod-1 chip, so a target tagged
requires-tpu-v5lite:8 cannot pass here however it is invoked. That is the shape
of the fleet, not a limit of this script.

This script only borrows hardware. It calls attach and detach, never up or
down, so it cannot create or delete anyone's VMs.
EOF
}

parse_args() {
  local seen_ddash=false
  while [[ $# -gt 0 ]]; do
    if [[ "$seen_ddash" == "true" ]]; then
      CLI_BAZEL_FLAGS+=("$1"); shift; continue
    fi
    case "$1" in
      --) seen_ddash=true; shift ;;
      --pool) [[ $# -ge 2 ]] || die "--pool requires an argument."; CLI_POOL="$2"; shift 2 ;;
      --pool=*) CLI_POOL="${1#*=}"; shift ;;
      --zone) [[ $# -ge 2 ]] || die "--zone requires an argument."; CLI_ZONES+=("$2"); shift 2 ;;
      --zone=*) CLI_ZONES+=("${1#*=}"); shift ;;
      --jobs) [[ $# -ge 2 ]] || die "--jobs requires an argument."; CLI_JOBS="$2"; shift 2 ;;
      --jobs=*) CLI_JOBS="${1#*=}"; shift ;;
      --bazel-config) [[ $# -ge 2 ]] || die "--bazel-config requires an argument."; CLI_BAZEL_CONFIG="$2"; shift 2 ;;
      --bazel-config=*) CLI_BAZEL_CONFIG="${1#*=}"; shift ;;
      --scope) [[ $# -ge 2 ]] || die "--scope requires an argument."; CLI_SCOPE="$2"; shift 2 ;;
      --scope=*) CLI_SCOPE="${1#*=}"; shift ;;
      --filter) [[ $# -ge 2 ]] || die "--filter requires an argument."; CLI_FILTER="$2"; shift 2 ;;
      --filter=*) CLI_FILTER="${1#*=}"; shift ;;
      --ip-mode) [[ $# -ge 2 ]] || die "--ip-mode requires an argument."; CLI_IP_MODE="$2"; shift 2 ;;
      --ip-mode=*) CLI_IP_MODE="${1#*=}"; shift ;;
      --output-dir) [[ $# -ge 2 ]] || die "--output-dir requires an argument."; CLI_OUTPUT_DIR="$2"; shift 2 ;;
      --output-dir=*) CLI_OUTPUT_DIR="${1#*=}"; shift ;;
      --report) CLI_REPORT=true; shift ;;
      --upload-results) CLI_UPLOAD_RESULTS=true; shift ;;
      --wait) CLI_WAIT=true; shift ;;
      --no-wait) CLI_WAIT=false; shift ;;
      --wait-timeout) [[ $# -ge 2 ]] || die "--wait-timeout requires an argument."; CLI_WAIT_TIMEOUT="$2"; shift 2 ;;
      --wait-timeout=*) CLI_WAIT_TIMEOUT="${1#*=}"; shift ;;
      --keep) CLI_KEEP=true; shift ;;
      --dry-run) CLI_DRY_RUN=true; shift ;;
      -h|--help) show_help; exit 0 ;;
      -*) die "unknown option '$1'. Try --help." ;;
      *) CLI_TARGETS+=("$1"); shift ;;
    esac
  done

  if [[ -n "$CLI_JOBS" && ! "$CLI_JOBS" =~ ^[0-9]+$ ]]; then
    die "--jobs must be a number, got '${CLI_JOBS}'."
  fi
  [[ "$CLI_WAIT_TIMEOUT" =~ ^[0-9]+$ ]] \
    || die "--wait-timeout must be a number, got '${CLI_WAIT_TIMEOUT}'."
  case "$CLI_IP_MODE" in
    auto|internal|external) ;;
    *) die "--ip-mode must be auto, internal or external, got '${CLI_IP_MODE}'." ;;
  esac
  # Naming no targets and no scope would quietly fall back to the whole test
  # tree, which is a 57-target presubmit nobody asked for.
  if [[ ${#CLI_TARGETS[@]} -eq 0 && -z "$CLI_SCOPE" ]]; then
    die "name at least one target, or pass --scope. Try --help."
  fi

  [[ -n "$CLI_POOL" ]] || CLI_POOL="/tmp/tpu_relay_test_${USER:-relay}"
  [[ -n "$CLI_OUTPUT_DIR" ]] \
    || CLI_OUTPUT_DIR="${REPO_ROOT}/relay_reports/test_$(date +%Y%m%d_%H%M%S)"
}

preflight() {
  command -v gcloud >/dev/null 2>&1 || die "gcloud is required."
  [[ -x "$FLEET" ]] || die "cannot execute ${FLEET}."
  [[ -x "$RELAY" ]] || die "cannot execute ${RELAY}."
}

pool_session_count() {
  find "$CLI_POOL" -maxdepth 1 -name '*.env' 2>/dev/null | wc -l | tr -d '[:space:]'
}

# Only gives back what this run took. A pool that was already populated belongs
# to whoever filled it, and --keep means the caller wants to hold it.
release_fleet() {
  if [[ "$_ATTACHED" == "true" && "$CLI_KEEP" != "true" ]]; then
    _ATTACHED=false
    log "giving the fleet back"
    "$FLEET" detach --pool "$CLI_POOL" >/dev/null 2>&1 || true
  fi
}

# A trap that only releases would let the script carry on from where the signal
# hit and exit 0, reporting a run the operator cancelled as a pass. Release
# once, then leave with the conventional 128+signal code.
on_signal() {
  local sig="$1" code="$2"
  trap '' INT TERM HUP
  trap - EXIT
  echo >&2
  log "caught SIG${sig}, cleaning up"
  release_fleet
  exit "$code"
}

main() {
  parse_args "$@"
  preflight

  local zone_args=()
  local zone
  for zone in ${CLI_ZONES[@]+"${CLI_ZONES[@]}"}; do
    zone_args+=(--zone "$zone")
  done

  local relay_args=(--session-pool "$CLI_POOL" --output-dir "$CLI_OUTPUT_DIR")
  if [[ ${#CLI_TARGETS[@]} -gt 0 ]]; then
    relay_args+=(--targets "$(IFS=' '; echo "${CLI_TARGETS[*]}")")
  fi
  [[ -z "$CLI_SCOPE" ]] || relay_args+=(--scope "$CLI_SCOPE")
  relay_args+=(--filter "$CLI_FILTER")
  [[ -z "$CLI_JOBS" ]] || relay_args+=(--jobs "$CLI_JOBS")
  [[ -z "$CLI_BAZEL_CONFIG" ]] || relay_args+=(--bazel-config "$CLI_BAZEL_CONFIG")
  [[ "$CLI_REPORT" == "true" ]] || relay_args+=(--no-report)
  # The ci configs chain down to --config=resultstore_base, which uploads build
  # events to ml-oss-rbe-testing. Only the CI service account can write there,
  # so for anyone else the upload fails after the tests have already run and
  # bazel still exits non-zero. Clearing the backend drops the upload only.
  if [[ "$CLI_UPLOAD_RESULTS" != "true" ]]; then
    relay_args+=(--bazel-flag "--bes_backend=")
  fi
  local flag
  for flag in ${CLI_BAZEL_FLAGS[@]+"${CLI_BAZEL_FLAGS[@]}"}; do
    relay_args+=(--bazel-flag "$flag")
  done

  if [[ "$CLI_DRY_RUN" == "true" ]]; then
    echo "Would attach:  ${FLEET} attach --pool ${CLI_POOL} --ip-mode ${CLI_IP_MODE} ${zone_args[*]}"
    echo "Would run:     ${RELAY} ${relay_args[*]}"
    echo "Would detach:  ${FLEET} detach --pool ${CLI_POOL}"
    return 0
  fi

  mkdir -p "$CLI_POOL" "$CLI_OUTPUT_DIR"

  local existing
  existing=$(pool_session_count)
  if [[ "$existing" -gt 0 ]]; then
    log "reusing ${existing} session(s) already in ${CLI_POOL}, leaving them attached on exit"
  else
    local wait_args=()
    if [[ "$CLI_WAIT" == "true" ]]; then
      wait_args+=(--wait --wait-timeout "$CLI_WAIT_TIMEOUT")
      log "waiting for the fleet if somebody else has it (up to ${CLI_WAIT_TIMEOUT}s)"
    fi

    # Armed before attach, not after: a Ctrl-C partway through attach has
    # already claimed VMs on the hosts it reached.
    trap release_fleet EXIT
    trap 'on_signal INT 130' INT
    trap 'on_signal TERM 143' TERM
    trap 'on_signal HUP 129' HUP
    _ATTACHED=true

    log "borrowing VMs from the standing fleet"
    "$FLEET" attach --pool "$CLI_POOL" --ip-mode "$CLI_IP_MODE" \
      ${zone_args[@]+"${zone_args[@]}"} ${wait_args[@]+"${wait_args[@]}"}

    local attached
    attached=$(pool_session_count)
    [[ "$attached" -gt 0 ]] || die "no v5e VMs available to borrow.
This script borrows a standing fleet; it does not create hardware. Check
'spot_tpu_fleet.sh status', and name every zone the fleet spans with --zone."
    log "attached ${attached} VM(s)"
  fi

  log "running on the fleet"
  "$RELAY" "${relay_args[@]}"
}

main "$@"
