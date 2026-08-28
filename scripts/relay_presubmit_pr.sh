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
#
# scripts/relay_presubmit_pr.sh
# Runs the TPU v5 single-chip presubmit suite on Cloud TPU v5e hardware from
# this machine and reports the verdict back to a pull request.
#
# Why this runs here and not in GitHub Actions: a hierarchical firewall policy
# above rbe-tpu-oss denies port 22 from 0.0.0.0/0, so no GitHub-hosted runner
# can reach the fleet. Corp workstations get in through corp-ssh-helper, which
# /etc/ssh/ssh_config wires up automatically. See
# docs/CI_PRESUBMIT_AND_RBE_GUIDE.md section 5.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT
readonly DEFAULT_REPO="google-pytorch/torch_tpu"

# The check name branch protection already requires. A replacement run reports
# under it so the required check carries the relay's real verdict; presubmit.yml
# leaves it unclaimed whenever ci:replace-tpu-v5 is on.
readonly GATING_CONTEXT="Presubmit on linux-x86-ct5lp-224-8tpu"
readonly SHADOW_CONTEXT="TPU v5e relay (shadow)"
readonly REPLACE_LABEL="ci:replace-tpu-v5"
readonly SHADOW_LABEL="ci:relay-tpu-v5"
readonly COMMENT_MARKER="<!-- torch-tpu-v5e-relay -->"

FLEET="${SPOT_TPU_FLEET_BIN:-${SCRIPT_DIR}/spot_tpu_fleet.sh}"
RELAY="${RUN_PRESUBMIT_V5_RELAY_BIN:-${SCRIPT_DIR}/run_presubmit_v5_relay.sh}"
GH="${GH_BIN:-gh}"
SSH_PROXY="${CORP_SSH_HELPER_BIN:-corp-ssh-helper}"

CLI_PR=""
CLI_REPO="$DEFAULT_REPO"
CLI_SHA=""
CLI_MODE="shadow"
CLI_POOL=""
CLI_ZONES=()
CLI_JOBS=""
CLI_BAZEL_CONFIG="ci_tpu_v5_relay"
CLI_OUTPUT_DIR=""
CLI_DRY_RUN=false
CLI_SKIP_HEAD_CHECK=false
CLI_NO_REPORT=false
CLI_ADD_LABEL=false
CLI_UPLOAD_RESULTS=false
CLI_BAZEL_FLAGS=()
# Somebody else's run holds the fleet for about seven minutes. Waiting is
# almost always what you want; --no-wait is for a scripted run that would
# rather fail than block.
CLI_WAIT=true
CLI_WAIT_TIMEOUT=3600

# Set once the pending status is published, so the exit trap knows it owes the
# PR a terminal state.
_STATUS_POSTED=false
_ATTACHED=false

show_help() {
  cat <<'EOF'
Usage: scripts/relay_presubmit_pr.sh --pr N [options]

Runs the torch_tpu presubmit-v5 single-chip suite on Cloud TPU v5e VMs borrowed
from the standing fleet in rbe-tpu-oss, then publishes the verdict to the pull
request as a commit status.

Modes:
  shadow       Advisory. Reports under "TPU v5e relay (shadow)" and never
               affects whether the PR can merge. This is the default.
  replacement  Gating. Reports under "Presubmit on linux-x86-ct5lp-224-8tpu",
               the check branch protection already requires. The PR must carry
               the ci:replace-tpu-v5 label so presubmit.yml drops the real v5
               runner from its matrix and leaves that check for this run.

Options:
  --pr N                Pull request number. Required.
  --repo OWNER/NAME     Default: google-pytorch/torch_tpu
  --mode MODE           shadow (default) or replacement
  --sha SHA             Commit to report against. Default: the PR's head.
  --pool DIR            Fleet session pool. Default: /tmp/tpu_pool_<user>
  --zone ZONE           Zone to look for fleet VMs in. Repeatable.
                        Default: europe-west4-b
  --jobs N              Test actions at once. Default: however many VMs
                        attached. Never set this above the VM count.
  --bazel-config NAME   Default: ci_tpu_v5_relay, which sends compile actions
                        to RBE. Pass "" for a plain local build.
  --bazel-flag FLAG     Extra bazel flag for the relay's build and test runs.
                        Repeatable.
  --upload-results      Send build events to ResultStore. Off by default: the
                        ci configs point at the ml-oss-rbe-testing instance,
                        which a personal Google account cannot write to, and
                        the failed upload sinks an otherwise green run.
  --output-dir DIR      Report destination.
                        Default: relay_reports/pr<N>_<timestamp>
  --add-label           Add the mode's label to the PR before running.
  --no-wait             Fail instead of queueing when somebody else has the
                        fleet. By default a run waits its turn and then takes
                        every VM, because one run on 28 chips finishes faster
                        than two runs on 14 each.
  --wait-timeout SECS   How long to stand in line. Default: 3600.
  --skip-head-check     Run even if the working tree is not at the PR head.
  --no-report           Run the suite but leave the PR untouched.
  --dry-run             Print the plan and exit without touching hardware.
  -h, --help            This message.

This script only ever borrows hardware. It calls `attach` and `detach`, never
`up` or `down`, so it cannot create or delete anyone's VMs.
EOF
}

log() { echo "[relay_pr] $*"; }
die() { echo "ERROR [relay_pr]: $*" >&2; exit 1; }

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --pr) [[ $# -ge 2 ]] || die "--pr requires an argument."; CLI_PR="$2"; shift 2 ;;
      --pr=*) CLI_PR="${1#*=}"; shift ;;
      --repo) [[ $# -ge 2 ]] || die "--repo requires an argument."; CLI_REPO="$2"; shift 2 ;;
      --repo=*) CLI_REPO="${1#*=}"; shift ;;
      --mode) [[ $# -ge 2 ]] || die "--mode requires an argument."; CLI_MODE="$2"; shift 2 ;;
      --mode=*) CLI_MODE="${1#*=}"; shift ;;
      --sha) [[ $# -ge 2 ]] || die "--sha requires an argument."; CLI_SHA="$2"; shift 2 ;;
      --sha=*) CLI_SHA="${1#*=}"; shift ;;
      --pool) [[ $# -ge 2 ]] || die "--pool requires an argument."; CLI_POOL="$2"; shift 2 ;;
      --pool=*) CLI_POOL="${1#*=}"; shift ;;
      --zone) [[ $# -ge 2 ]] || die "--zone requires an argument."; CLI_ZONES+=("$2"); shift 2 ;;
      --zone=*) CLI_ZONES+=("${1#*=}"); shift ;;
      --jobs) [[ $# -ge 2 ]] || die "--jobs requires an argument."; CLI_JOBS="$2"; shift 2 ;;
      --jobs=*) CLI_JOBS="${1#*=}"; shift ;;
      --bazel-config) [[ $# -ge 2 ]] || die "--bazel-config requires an argument."; CLI_BAZEL_CONFIG="$2"; shift 2 ;;
      --bazel-config=*) CLI_BAZEL_CONFIG="${1#*=}"; shift ;;
      --bazel-flag) [[ $# -ge 2 ]] || die "--bazel-flag requires an argument."; CLI_BAZEL_FLAGS+=("$2"); shift 2 ;;
      --bazel-flag=*) CLI_BAZEL_FLAGS+=("${1#*=}"); shift ;;
      --upload-results) CLI_UPLOAD_RESULTS=true; shift ;;
      --output-dir) [[ $# -ge 2 ]] || die "--output-dir requires an argument."; CLI_OUTPUT_DIR="$2"; shift 2 ;;
      --output-dir=*) CLI_OUTPUT_DIR="${1#*=}"; shift ;;
      --add-label) CLI_ADD_LABEL=true; shift ;;
      --wait) CLI_WAIT=true; shift ;;
      --no-wait) CLI_WAIT=false; shift ;;
      --wait-timeout) [[ $# -ge 2 ]] || die "--wait-timeout requires an argument."; CLI_WAIT_TIMEOUT="$2"; shift 2 ;;
      --wait-timeout=*) CLI_WAIT_TIMEOUT="${1#*=}"; shift ;;
      --skip-head-check) CLI_SKIP_HEAD_CHECK=true; shift ;;
      --no-report) CLI_NO_REPORT=true; shift ;;
      --dry-run) CLI_DRY_RUN=true; shift ;;
      -h|--help) show_help; exit 0 ;;
      *) die "unknown argument '$1'. Try --help." ;;
    esac
  done

  [[ -n "$CLI_PR" ]] || die "--pr is required."
  [[ "$CLI_PR" =~ ^[0-9]+$ ]] || die "--pr must be a number, got '${CLI_PR}'."
  case "$CLI_MODE" in
    shadow|replacement) ;;
    *) die "--mode must be 'shadow' or 'replacement', got '${CLI_MODE}'." ;;
  esac
  [[ "$CLI_REPO" == */* ]] || die "--repo must look like OWNER/NAME, got '${CLI_REPO}'."
  if [[ -n "$CLI_JOBS" && ! "$CLI_JOBS" =~ ^[0-9]+$ ]]; then
    die "--jobs must be a number, got '${CLI_JOBS}'."
  fi

  [[ ${#CLI_ZONES[@]} -gt 0 ]] || CLI_ZONES=("europe-west4-b")
  [[ -n "$CLI_POOL" ]] || CLI_POOL="/tmp/tpu_pool_${USER:-relay}"
}

# shadow runs stay out of the required-check namespace on purpose; only a
# replacement run is allowed to answer for the check that gates the PR.
status_context() {
  if [[ "$1" == "replacement" ]]; then
    printf '%s' "$GATING_CONTEXT"
  else
    printf '%s' "$SHADOW_CONTEXT"
  fi
}

mode_label() {
  if [[ "$1" == "replacement" ]]; then
    printf '%s' "$REPLACE_LABEL"
  else
    printf '%s' "$SHADOW_LABEL"
  fi
}

# A workstation reaches the fleet through corp-ssh-helper, which the corp
# ssh_config installs as a ProxyCommand. Without it every SSH in the relay hangs
# until it times out, which is a slow and confusing way to fail.
check_ssh_path() {
  command -v "$SSH_PROXY" >/dev/null 2>&1 && return 0
  cat >&2 <<'EOF'
ERROR [relay_pr]: corp-ssh-helper is not on PATH.

The TPU VMs sit behind a hierarchical firewall that denies port 22 from the
open internet. Corp workstations get through because /etc/ssh/ssh_config
proxies GCP addresses via corp-ssh-helper. Without it the relay cannot open a
session and every test will time out.

Run this from a corp workstation or cloudtop. If you are on one and this still
fires, check that /usr/local/bin is on your PATH.
EOF
  return 1
}

preflight() {
  command -v "$GH" >/dev/null 2>&1 || die "the '${GH}' CLI is required. See go/gh-cli."
  command -v gcloud >/dev/null 2>&1 || die "gcloud is required."
  [[ -x "$FLEET" ]] || die "cannot execute ${FLEET}."
  [[ -x "$RELAY" ]] || die "cannot execute ${RELAY}."
  check_ssh_path || exit 1
}

resolve_head_sha() {
  "$GH" api "repos/${CLI_REPO}/pulls/${CLI_PR}" --jq '.head.sha' 2>/dev/null
}

pr_labels() {
  "$GH" api "repos/${CLI_REPO}/pulls/${CLI_PR}" --jq '.labels[].name' 2>/dev/null
}

add_label() {
  # `gh pr edit --add-label` silently no-ops on this repo, so go at the REST
  # endpoint directly.
  "$GH" api "repos/${CLI_REPO}/issues/${CLI_PR}/labels" \
    -f "labels[]=$1" >/dev/null
}

post_status() {
  local sha="$1" state="$2" description="$3" context="$4"
  # GitHub truncates descriptions past 140 characters.
  description="${description:0:140}"
  "$GH" api -X POST "repos/${CLI_REPO}/statuses/${sha}" \
    -f "state=${state}" \
    -f "context=${context}" \
    -f "description=${description}" >/dev/null
}

# Reads presubmit_summary.json rather than trusting the relay's exit code alone,
# so a partial run cannot be reported as a pass.
verdict_from_summary() {
  local summary="$1" relay_rc="$2"
  if [[ ! -f "$summary" ]]; then
    printf 'error\tthe relay left no summary behind; it died before reporting'
    return
  fi
  python3 - "$summary" "$relay_rc" <<'PY'
import json, sys

with open(sys.argv[1]) as handle:
  summary = json.load(handle)
relay_rc = int(sys.argv[2])

total = summary.get("total_targets", 0)
passed = summary.get("passed_targets", 0)
failed = summary.get("failed_targets", 0)
duration = summary.get("duration_seconds", 0)
status = summary.get("status", "UNKNOWN")

detail = f"{passed}/{total} targets passed in {duration / 60:.1f} min"
if status == "PASSED" and relay_rc == 0:
  print(f"success\t{detail}")
elif failed:
  print(f"failure\t{failed} of {total} targets failed ({detail})")
else:
  # Exit code says something went wrong but no target owns it: a build break, a
  # fleet problem, an interrupted run. Not a test failure, so don't call it one.
  print(f"error\tthe relay exited {relay_rc} with no failing target ({detail})")
PY
}

upsert_comment() {
  local body="$1"
  local existing
  existing=$("$GH" api "repos/${CLI_REPO}/issues/${CLI_PR}/comments?per_page=100" \
    --jq "[.[] | select(.body | contains(\"${COMMENT_MARKER}\"))] | last | .id // empty" \
    2>/dev/null || true)
  if [[ -n "$existing" ]]; then
    "$GH" api -X PATCH "repos/${CLI_REPO}/issues/comments/${existing}" \
      -f "body=${body}" >/dev/null
  else
    "$GH" api -X POST "repos/${CLI_REPO}/issues/${CLI_PR}/comments" \
      -f "body=${body}" >/dev/null
  fi
}

build_comment() {
  local state="$1" detail="$2" report="$3"
  printf '%s\n' "$COMMENT_MARKER"
  printf '### TPU v5e relay — %s run\n\n' "$CLI_MODE"
  case "$state" in
    success) printf '🟢 **Passed.** %s\n\n' "$detail" ;;
    failure) printf '🔴 **Failed.** %s\n\n' "$detail" ;;
    *) printf '⚠️ **Did not complete.** %s\n\n' "$detail" ;;
  esac
  # The backticks below are markdown for the PR comment, not command
  # substitution, so these format strings stay single-quoted.
  # shellcheck disable=SC2016
  printf 'Ran the `presubmit-v5` single-chip targets on Cloud TPU v5e VMs in `rbe-tpu-oss`, reported under `%s`.\n\n' \
    "$(status_context "$CLI_MODE")"
  if [[ "$CLI_MODE" == "replacement" ]]; then
    # shellcheck disable=SC2016
    printf 'This run stands in for `%s`. Targets tagged `requires-tpu-v5lite:8` are **not covered** — the relay leases one chip per test.\n\n' \
      "$GATING_CONTEXT"
  fi
  if [[ -f "$report" ]]; then
    printf '<details><summary>Full report</summary>\n\n'
    cat "$report"
    printf '\n</details>\n'
  fi
}

# Runs on every exit path. A replacement run that dies without publishing
# something would leave the required check pending forever.
on_exit() {
  local rc=$?
  if [[ "$_ATTACHED" == "true" ]]; then
    "$FLEET" detach --pool "$CLI_POOL" >/dev/null 2>&1 || true
    _ATTACHED=false
  fi
  if [[ $rc -ne 0 && "$_STATUS_POSTED" == "true" ]]; then
    post_status "$CLI_SHA" "error" \
      "the relay driver exited ${rc} before reporting a verdict" \
      "$(status_context "$CLI_MODE")" || true
  fi
  return $rc
}

main() {
  parse_args "$@"

  if [[ "$CLI_DRY_RUN" != "true" ]]; then
    preflight
  fi

  if [[ -z "$CLI_SHA" ]]; then
    CLI_SHA=$(resolve_head_sha) || true
    [[ -n "$CLI_SHA" ]] || die "could not read the head SHA of ${CLI_REPO}#${CLI_PR}. Is 'gh' authenticated?"
  fi

  local context
  context=$(status_context "$CLI_MODE")
  [[ -n "$CLI_OUTPUT_DIR" ]] || \
    CLI_OUTPUT_DIR="${REPO_ROOT}/relay_reports/pr${CLI_PR}_$(date +%Y%m%d_%H%M%S)"

  log "repo         ${CLI_REPO}"
  log "pull request #${CLI_PR} at ${CLI_SHA}"
  log "mode         ${CLI_MODE} (reports as '${context}')"
  log "pool         ${CLI_POOL}"
  log "zones        ${CLI_ZONES[*]}"
  if [[ "$CLI_WAIT" == "true" ]]; then
    log "fleet        wait for all of it, up to ${CLI_WAIT_TIMEOUT}s"
  else
    log "fleet        take what is free, fail if none is"
  fi
  log "output       ${CLI_OUTPUT_DIR}"

  if [[ "$CLI_DRY_RUN" == "true" ]]; then
    log "dry run, stopping here."
    return 0
  fi

  local head
  head=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo "")
  if [[ "$CLI_SKIP_HEAD_CHECK" != "true" && "$head" != "$CLI_SHA" ]]; then
    die "this tree is at ${head:-unknown}, but ${CLI_REPO}#${CLI_PR} is at ${CLI_SHA}.
Check out the PR head first, or pass --skip-head-check if you know the
difference does not matter. Reporting a verdict for code you did not run is
worse than not reporting one."
  fi

  local label
  label=$(mode_label "$CLI_MODE")
  if [[ "$CLI_ADD_LABEL" == "true" ]]; then
    log "adding label ${label}"
    add_label "$label"
  elif [[ "$CLI_MODE" == "replacement" ]] && ! pr_labels | grep -qx "$REPLACE_LABEL"; then
    die "a replacement run needs the ${REPLACE_LABEL} label on the PR.
Without it presubmit.yml still schedules the real TPU v5 runner, and both it
and this run would report under the same check name. Add it with --add-label."
  fi

  trap on_exit EXIT

  if [[ "$CLI_NO_REPORT" != "true" ]]; then
    post_status "$CLI_SHA" "pending" "relay running on Cloud TPU v5e" "$context"
    _STATUS_POSTED=true
  fi

  local zone_args=()
  local zone
  for zone in "${CLI_ZONES[@]}"; do
    zone_args+=(--zone "$zone")
  done

  # Wait for the whole fleet rather than racing for a slice of it. A run is
  # seven minutes on 28 chips, so queueing costs the second person one run's
  # worth of waiting and saves both of them a half-speed run.
  local wait_args=()
  if [[ "$CLI_WAIT" == "true" ]]; then
    wait_args+=(--wait --wait-timeout "$CLI_WAIT_TIMEOUT")
    log "waiting for the fleet if somebody else has it (up to ${CLI_WAIT_TIMEOUT}s)"
  fi

  log "borrowing VMs from the standing fleet"
  "$FLEET" attach --pool "$CLI_POOL" "${zone_args[@]}" ${wait_args[@]+"${wait_args[@]}"}
  _ATTACHED=true

  local attached
  attached=$(find "$CLI_POOL" -maxdepth 1 -name '*.env' 2>/dev/null | wc -l | tr -d '[:space:]')
  [[ "$attached" -gt 0 ]] || die "no READY TPU v5e VMs in ${CLI_ZONES[*]}.
The relay borrows a standing fleet; it does not create hardware. Ask whoever
owns the fleet to bring it up, or run 'spot_tpu_fleet.sh up' yourself and
remember that those VMs bill until deleted."
  log "attached to ${attached} VM(s)"

  local jobs="${CLI_JOBS:-$attached}"
  local relay_args=(
    --session-pool "$CLI_POOL"
    --output-dir "$CLI_OUTPUT_DIR"
    --jobs "$jobs"
  )
  [[ -z "$CLI_BAZEL_CONFIG" ]] || relay_args+=(--bazel-config "$CLI_BAZEL_CONFIG")
  # The ci configs chain down to --config=resultstore_base, which uploads build
  # events to the ml-oss-rbe-testing instance. Only the CI service account can
  # write there; for anyone else the upload fails after the tests have already
  # passed and bazel still exits non-zero. Clearing the backend drops the upload
  # and leaves the run itself alone.
  if [[ "$CLI_UPLOAD_RESULTS" != "true" ]]; then
    relay_args+=(--bazel-flag "--bes_backend=")
  fi
  local extra_flag
  for extra_flag in ${CLI_BAZEL_FLAGS[@]+"${CLI_BAZEL_FLAGS[@]}"}; do
    relay_args+=(--bazel-flag "$extra_flag")
  done

  local relay_rc=0
  "$RELAY" "${relay_args[@]}" || relay_rc=$?
  log "relay exited ${relay_rc}"

  local verdict state detail
  verdict=$(verdict_from_summary "${CLI_OUTPUT_DIR}/presubmit_summary.json" "$relay_rc")
  state="${verdict%%$'\t'*}"
  detail="${verdict#*$'\t'}"
  log "verdict ${state}: ${detail}"

  if [[ "$CLI_NO_REPORT" != "true" ]]; then
    post_status "$CLI_SHA" "$state" "$detail" "$context"
    upsert_comment "$(build_comment "$state" "$detail" "${CLI_OUTPUT_DIR}/presubmit_report.md")"
    _STATUS_POSTED=false
    log "published '${state}' to ${CLI_REPO}#${CLI_PR} under '${context}'"
  fi

  [[ "$state" == "success" ]] || return 1
  return 0
}

main "$@"
