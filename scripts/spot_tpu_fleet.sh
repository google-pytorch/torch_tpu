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

# Brings up a fleet of Spot TPU v5e VMs and writes one session file per VM into
# a pool directory. Point TPU_SESSION_POOL at that directory and
# relay_test_runner.sh will lease a VM per test, so a single Bazel invocation
# spreads its test actions across the whole fleet.
#
# Usage:
#   scripts/spot_tpu_fleet.sh up --size 10 [--pool DIR] [--zone ZONE] [--on-demand]
#                                [--deadline-minutes N]
#   scripts/spot_tpu_fleet.sh down [--pool DIR]
#   scripts/spot_tpu_fleet.sh attach [--pool DIR] [--zone ZONE] [--ssh-user USER]
#                                    [--ssh-identity PATH] [--no-key-push]
#                                    [--name-prefix PREFIX] [--accelerator-type TYPE]
#   scripts/spot_tpu_fleet.sh detach [--pool DIR]
#   scripts/spot_tpu_fleet.sh status [--pool DIR]
#
# `up` arms a detached deadline that runs `down` after --deadline-minutes, so a
# fleet cannot outlive a crashed orchestrator. Pass 0 to turn it off. `down`
# cancels it.
#
# `attach` joins a fleet somebody else already brought up: it lists the VMs,
# pushes the caller's SSH key to each one and writes the session files. It never
# creates or deletes a VM and never arms a deadline. `detach` drops the session
# files again and leaves the VMs running. Use that pair from CI against a
# long-lived pool, so a finished job cannot delete hardware another job is using.
#
# By default `attach` only picks up VMs this script created, whose names start
# with spot-tpu-v5e-. Reserved capacity is usually named something else, so
# --name-prefix widens the search; pass an empty prefix to consider every VM in
# the zone. Widening the name never widens what `down` deletes: the orphan sweep
# is pinned to the creation prefix on purpose.
#
#   # Borrow a reserved fleet named reserved-v5e-*.
#   scripts/spot_tpu_fleet.sh attach --name-prefix reserved-v5e- --zone us-east5-b
#
#   # Borrow every single-chip v5e in the zone, whatever it is called.
#   scripts/spot_tpu_fleet.sh attach --name-prefix '' --zone us-east5-b
#
# --accelerator-type guards the widened search. The relay leases one chip per
# test action, so attaching a v5p or a multi-chip host would burn a full timeout
# per test and report as an ordinary failure. It defaults to v5litepod-1.
#
# `attach` claims each VM it takes by leaving a marker on it, and walks past VMs
# somebody else is holding. Without that, two people attaching at the same time
# both take the whole fleet and run two tests on every chip. `detach` drops the
# marker; a claim nobody released ages out after --claim-ttl seconds (4h).
# --force-claim takes a VM anyway, --no-claim turns the whole thing off.
#
# With --wait, `attach` takes the whole fleet or none of it, and stands in line
# when somebody else has it. A run is seven minutes, so waiting your turn on all
# 28 chips beats splitting them and both going slowly. The line lives on one VM
# and is ordered by arrival:
#
#   scripts/spot_tpu_fleet.sh attach --wait --zone europe-west4-b
#   [spot_tpu_fleet] Waiting for the fleet: 1 run ahead of you
#
# --wait-timeout caps the wait (default 1h). A waiter that walks away loses its
# place after two minutes of silence, so nobody holds the line by crashing.


set -uo pipefail

readonly ALLOWED_PROJECT="rbe-tpu-oss"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SPOT_MANAGER="${SCRIPT_DIR}/spot_tpu_manager.sh"
readonly DEFAULT_POOL="/tmp/torch_tpu_relay/pool"
# Matches the names spot_tpu_manager.sh generates. `down`'s orphan sweep filters
# on this and nothing else, so a widened `attach` search can never turn into a
# widened delete.
readonly VM_NAME_PREFIX="spot-tpu-v5e-"

# Spot capacity moves around between zones, so spread the fleet rather than
# betting the whole run on one zone having room. us-east5-a is deliberately
# absent: rbe-tpu-oss has no v5e reservation there and every create returns
# "Reservation not found".
readonly DEFAULT_ZONES=("europe-west4-b" "us-central1-a")

POOL_DIR="$DEFAULT_POOL"
FLEET_SIZE=10
ZONES=()
ON_DEMAND=false
# Long enough that no presubmit suite gets cut off (a 57-target run takes about
# 50 minutes), short enough that a forgotten fleet is hours of billing, not days.
DEADLINE_MINUTES=180
REAPER_PID_FILE=""
# `attach` runs as whatever identity CI authenticated as, which is not the
# identity that created the VMs, so the key has to be pushed before SSH works.
SSH_IDENTITY_PATH="${SSH_IDENTITY:-${HOME}/.ssh/google_compute_engine}"
ATTACH_SSH_USER=""
ATTACH_PUSH_KEY=true
# Which VMs `attach` will consider. Defaults to the ones this script creates;
# --name-prefix widens it to reserved capacity, which is named by whoever
# reserved it. Empty means every VM in the zone.
ATTACH_NAME_PREFIX="$VM_NAME_PREFIX"
ATTACH_NAME_PREFIX_SET=false
# The relay hands one chip to one test action, so a multi-chip or non-v5e host
# cannot pass. This is the guard that makes a widened --name-prefix safe.
ATTACH_ACCELERATOR_TYPE="v5litepod-1"
# Two people attaching at the same time would otherwise both take all 28 VMs
# and run two tests on every chip. `attach` leaves a marker on each VM it takes
# and walks past VMs somebody else is holding; `detach` removes it.
ATTACH_CLAIM=true
ATTACH_CLAIM_FORCE=false
# A run is ten minutes. Four hours means a crashed run frees its VMs the same
# day without ever cutting a live one loose.
ATTACH_CLAIM_TTL=14400
readonly REMOTE_CLAIM_PATH="${TORCH_TPU_REMOTE_CLAIM_PATH:-/tmp/torch_tpu_relay/claim}"
# --wait turns "take what is free" into "take all of it or stand in line". The
# line lives on one VM of the fleet, so it needs no storage the relay does not
# already reach, and it survives nothing: if every VM goes away there is nothing
# left to queue for.
ATTACH_WAIT=false
ATTACH_WAIT_TIMEOUT=3600
ATTACH_WAIT_POLL=20
# Which VM holds the line. Empty means the first candidate by name, which every
# caller computes the same way.
QUEUE_HOST=""
QUEUE_HOST_ZONE=""
# A waiter touches its ticket every poll. Six polls of silence and the others
# assume it walked away, so a killed run cannot hold the line.
readonly QUEUE_STALE=120
readonly REMOTE_QUEUE_PATH="${TORCH_TPU_REMOTE_QUEUE_PATH:-/tmp/torch_tpu_relay/queue}"
# Set once a ticket exists, so the exit trap knows what to take back out.
QUEUE_TICKET=""
# What the last attach attempt managed: taken, seen, and held by someone else.
ATTACH_READY=0
ATTACH_FOUND=0
ATTACH_HELD=0


die() {
  echo "ERROR [spot_tpu_fleet]: $*" >&2
  exit 1
}

log() {
  echo "[spot_tpu_fleet] $*"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --size|--size=*)
        [[ "$1" == *=* ]] && FLEET_SIZE="${1#*=}" || { FLEET_SIZE="${2:-}"; shift; }
        shift ;;
      --on-demand)
        ON_DEMAND=true
        shift ;;
      --pool|--pool=*)
        [[ "$1" == *=* ]] && POOL_DIR="${1#*=}" || { POOL_DIR="${2:-}"; shift; }
        shift ;;
      --zone|--zone=*)
        local zone
        [[ "$1" == *=* ]] && zone="${1#*=}" || { zone="${2:-}"; shift; }
        ZONES+=("$zone")
        shift ;;
      --project|--project=*)
        local project
        [[ "$1" == *=* ]] && project="${1#*=}" || { project="${2:-}"; shift; }
        [[ "$project" == "$ALLOWED_PROJECT" ]] \
          || die "This fleet is restricted to project '$ALLOWED_PROJECT'."
        shift ;;
      --deadline-minutes|--deadline-minutes=*)
        [[ "$1" == *=* ]] && DEADLINE_MINUTES="${1#*=}" || { DEADLINE_MINUTES="${2:-}"; shift; }
        shift ;;
      --ssh-identity|--ssh-identity=*)
        [[ "$1" == *=* ]] && SSH_IDENTITY_PATH="${1#*=}" || { SSH_IDENTITY_PATH="${2:-}"; shift; }
        shift ;;
      --ssh-user|--ssh-user=*)
        [[ "$1" == *=* ]] && ATTACH_SSH_USER="${1#*=}" || { ATTACH_SSH_USER="${2:-}"; shift; }
        shift ;;
      --no-key-push)
        ATTACH_PUSH_KEY=false
        shift ;;
      --name-prefix|--name-prefix=*)
        [[ "$1" == *=* ]] && ATTACH_NAME_PREFIX="${1#*=}" || { ATTACH_NAME_PREFIX="${2:-}"; shift; }
        ATTACH_NAME_PREFIX_SET=true
        shift ;;
      --accelerator-type|--accelerator-type=*)
        [[ "$1" == *=* ]] && ATTACH_ACCELERATOR_TYPE="${1#*=}" || { ATTACH_ACCELERATOR_TYPE="${2:-}"; shift; }
        shift ;;
      --no-claim)
        ATTACH_CLAIM=false
        shift ;;
      --force-claim)
        ATTACH_CLAIM_FORCE=true
        shift ;;
      --claim-ttl|--claim-ttl=*)
        [[ "$1" == *=* ]] && ATTACH_CLAIM_TTL="${1#*=}" || { ATTACH_CLAIM_TTL="${2:-}"; shift; }
        shift ;;
      --wait)
        ATTACH_WAIT=true
        shift ;;
      --no-wait)
        ATTACH_WAIT=false
        shift ;;
      --wait-timeout|--wait-timeout=*)
        [[ "$1" == *=* ]] && ATTACH_WAIT_TIMEOUT="${1#*=}" || { ATTACH_WAIT_TIMEOUT="${2:-}"; shift; }
        shift ;;
      --wait-poll|--wait-poll=*)
        [[ "$1" == *=* ]] && ATTACH_WAIT_POLL="${1#*=}" || { ATTACH_WAIT_POLL="${2:-}"; shift; }
        shift ;;
      --queue-host|--queue-host=*)
        [[ "$1" == *=* ]] && QUEUE_HOST="${1#*=}" || { QUEUE_HOST="${2:-}"; shift; }
        shift ;;
      -h|--help)
        sed -n '16,72p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
        exit 0 ;;
      *)
        die "Unknown option '$1'." ;;
    esac
  done

  [[ "$FLEET_SIZE" =~ ^[0-9]+$ && "$FLEET_SIZE" -gt 0 ]] \
    || die "--size must be a positive integer, got '$FLEET_SIZE'."
  [[ "$DEADLINE_MINUTES" =~ ^[0-9]+$ ]] \
    || die "--deadline-minutes must be a non-negative integer, got '$DEADLINE_MINUTES'."
  [[ ${#ZONES[@]} -gt 0 ]] || ZONES=("${DEFAULT_ZONES[@]}")
  REAPER_PID_FILE="${POOL_DIR}/reaper.pid"
}

# Brings up one VM and drops its session file in the pool. Runs as a background
# job, so it reports failure through its exit status and its own log file.
# Starts in the zone it was assigned, then works through the rest, because Spot
# capacity runs out one zone at a time.
provision_one() {
  local index="$1"
  local first_zone="$2"
  local session="${POOL_DIR}/vm_${index}.env"
  local log_file="${POOL_DIR}/vm_${index}.log"

  local ordered_zones=("$first_zone")
  local zone
  for zone in "${ZONES[@]}"; do
    [[ "$zone" == "$first_zone" ]] || ordered_zones+=("$zone")
  done

  for zone in "${ordered_zones[@]}"; do
    local up_args=(--zone="$zone" --project="$ALLOWED_PROJECT")
    [[ "$ON_DEMAND" == "true" ]] && up_args+=(--on-demand)
    if TPU_SESSION_ENV="$session" "$SPOT_MANAGER" up \
        "${up_args[@]}" >>"$log_file" 2>&1; then
      log "vm_${index} ready in ${zone}"
      return 0
    fi
    log "vm_${index} could not start in ${zone}"
    rm -f "$session"
  done

  log "vm_${index} failed in every zone (see ${log_file})"
  return 1
}

cmd_up() {
  [[ -x "$SPOT_MANAGER" ]] || die "$SPOT_MANAGER is missing."
  mkdir -p "$POOL_DIR"

  local pids=() existing=0
  local i
  for (( i = 0; i < FLEET_SIZE; i++ )); do
    # Re-running `up` after a partial capacity failure should fill the gaps,
    # not stand up a second VM for a slot that already has one.
    if [[ -s "${POOL_DIR}/vm_${i}.env" ]]; then
      existing=$(( existing + 1 ))
      continue
    fi
    local zone="${ZONES[$(( i % ${#ZONES[@]} ))]}"
    provision_one "$i" "$zone" &
    pids+=("$!")
  done

  local kind="Spot"
  [[ "$ON_DEMAND" == "true" ]] && kind="on-demand"
  log "Bringing up ${#pids[@]} ${kind} TPU v5e VMs across ${ZONES[*]} (${existing} already up)"

  local ready="$existing" pid
  for pid in "${pids[@]}"; do
    wait "$pid" && ready=$(( ready + 1 ))
  done

  log "Fleet ready: ${ready}/${FLEET_SIZE} VMs in ${POOL_DIR}"
  [[ "$ready" -gt 0 ]] || die "No VMs came up. Check ${POOL_DIR}/vm_*.log."
  arm_deadline
  return 0
}

# The orchestrator tears the fleet down on exit, but a `kill -9` or a dropped
# connection skips that and the VMs bill until someone notices. This arms a
# detached process that runs the same teardown at a fixed deadline.
arm_deadline() {
  [[ "$DEADLINE_MINUTES" -gt 0 ]] || return 0

  local args=(deadline --pool "$POOL_DIR" --deadline-minutes "$DEADLINE_MINUTES")
  local zone
  for zone in "${ZONES[@]}"; do
    args+=(--zone "$zone")
  done

  # setsid, not just nohup. nohup only blocks SIGHUP, so a process-group kill
  # of the orchestrator still takes the reaper with it. That happened: the
  # reaper died with its parent and eight VMs billed for 21 hours. A new
  # session leader has no controlling terminal and no shared process group.
  local launcher=()
  command -v setsid >/dev/null 2>&1 && launcher=(setsid)

  # Kill any reaper already armed rather than just forgetting its pid. Dropping
  # the pid file leaves the old process sleeping with nothing pointing at it,
  # and it tears down whatever fleet is up when its own deadline lands.
  cancel_deadline
  nohup "${launcher[@]}" "${BASH_SOURCE[0]}" "${args[@]}" \
    >>"${POOL_DIR}/reaper.log" 2>&1 &
  disown %% 2>/dev/null || true

  # cmd_deadline writes REAPER_PID_FILE itself, because with setsid the pid
  # here belongs to the launcher rather than to the process that sleeps.
  local waited=0
  while [[ ! -s "$REAPER_PID_FILE" && "$waited" -lt 50 ]]; do
    sleep 0.1
    waited=$(( waited + 1 ))
  done

  if [[ -s "$REAPER_PID_FILE" ]]; then
    log "Deadline armed: fleet self-deletes in ${DEADLINE_MINUTES}m (pid $(cat "$REAPER_PID_FILE"))"
  else
    log "WARNING: could not arm the deadline; tear the fleet down by hand"
  fi
}

cmd_deadline() {
  # Claim the pid file only after retiring whoever held it. `arm_deadline`
  # already does this, but `deadline` also gets run by hand, and two live
  # reapers means the fleet disappears at the earlier of the two deadlines.
  cancel_deadline
  echo "$$" > "$REAPER_PID_FILE"
  log "Deadline armed for ${DEADLINE_MINUTES}m (pid $$)"
  sleep $(( DEADLINE_MINUTES * 60 ))
  log "Deadline reached, tearing the fleet down"
  cmd_down
}

# A reaper left running after a manual teardown would delete whatever fleet
# happens to be up when its deadline lands.
cancel_deadline() {
  [[ -f "$REAPER_PID_FILE" ]] || return 0
  local reaper_pid
  reaper_pid="$(cat "$REAPER_PID_FILE")"
  rm -f "$REAPER_PID_FILE"
  [[ "$reaper_pid" =~ ^[0-9]+$ && "$reaper_pid" != "$$" ]] || return 0
  kill "$reaper_pid" 2>/dev/null && log "Cancelled deadline (pid ${reaper_pid})"
  return 0
}

cmd_down() {
  cancel_deadline
  local pids=() session
  if [[ -d "$POOL_DIR" ]]; then
    for session in "$POOL_DIR"/*.env; do
      [[ -f "$session" ]] || continue
      TPU_SESSION_ENV="$session" "$SPOT_MANAGER" down --project="$ALLOWED_PROJECT" \
        >>"${session%.env}.log" 2>&1 &
      pids+=("$!")
    done

    local pid
    for pid in "${pids[@]}"; do
      wait "$pid" || true
    done

    rm -f "$POOL_DIR"/*.env "$POOL_DIR"/*.env.lock "$POOL_DIR"/*.quarantine
  fi

  sweep_orphans
  log "Fleet torn down. Verify with: $0 status"
}

# Who this pool is, as seen from a VM. Two pools belonging to the same person
# are two different runs and must not share chips, so the pool name is part of
# the identity and the login name alone is not.
claim_owner() {
  printf '%s:%s' "${USER:-$(whoami)}" "$(basename "$POOL_DIR")"
}

# What attach runs on each VM. Takes the claim and prints the login name, or
# prints CLAIMED_BY:<owner> and leaves the VM alone. mkdir is the lock: it
# either creates the directory or it does not, with nothing in between.
claim_script() {
  if [[ "$ATTACH_CLAIM" != "true" ]]; then
    printf 'whoami\n'
    return 0
  fi
  cat <<EOF
c="${REMOTE_CLAIM_PATH}"
mkdir -p "\$(dirname "\$c")"
now=\$(date +%s)
if [ -d "\$c" ]; then
  owner=\$(cat "\$c/owner" 2>/dev/null || echo unknown)
  since=\$(cat "\$c/at" 2>/dev/null || echo 0)
  if [ "\$owner" != "$(claim_owner)" ] \
     && [ \$(( now - since )) -lt ${ATTACH_CLAIM_TTL} ] \
     && [ "${ATTACH_CLAIM_FORCE}" != "true" ]; then
    echo "CLAIMED_BY:\$owner"
    exit 0
  fi
  rm -rf "\$c"
fi
mkdir "\$c" 2>/dev/null || { echo "CLAIMED_BY:someone who got there first"; exit 0; }
echo "$(claim_owner)" > "\$c/owner"
echo "\$now" > "\$c/at"
whoami
EOF
}

# Writes one session file for a VM that already exists. Returns non-zero and
# writes nothing if the VM has no reachable address, so a half-provisioned node
# cannot end up in the pool looking healthy.
attach_one() {
  local index="$1" name="$2" zone="$3"
  local session="${POOL_DIR}/vm_${index}.env"
  local log_file="${POOL_DIR}/vm_${index}.log"

  local ip
  ip=$(gcloud compute tpus tpu-vm describe "$name" \
    --zone="$zone" --project="$ALLOWED_PROJECT" \
    --format="value(networkEndpoints[0].accessConfig.externalIp)" 2>>"$log_file")
  if [[ -z "$ip" ]]; then
    ip=$(gcloud compute tpus tpu-vm describe "$name" \
      --zone="$zone" --project="$ALLOWED_PROJECT" \
      --format="value(networkEndpoints[0].ipAddress)" 2>>"$log_file")
  fi
  [[ -n "$ip" ]] || { log "${name}: no reachable address, skipping"; return 1; }

  # `gcloud ... ssh` uploads the caller's public key as a side effect, which is
  # the whole reason to call it: the identity attaching is not the identity that
  # created the VM, so nothing has authorised it yet. Its stdout also names the
  # account the VM actually logs us in as, which OS Login can rewrite, and
  # carries the verdict on the claim.
  local ssh_user="$ATTACH_SSH_USER"
  if [[ "$ATTACH_PUSH_KEY" == "true" || "$ATTACH_CLAIM" == "true" ]]; then
    local remote_out holder
    remote_out=$(gcloud compute tpus tpu-vm ssh "$name" \
      --zone="$zone" --project="$ALLOWED_PROJECT" \
      --command="$(claim_script)" 2>>"$log_file" | tr -d '\r')
    holder=$(printf '%s\n' "$remote_out" | sed -n 's/^CLAIMED_BY://p' | head -n 1)
    if [[ -n "$holder" ]]; then
      log "${name} is held by ${holder}, skipping"
      # 2, not 1: a held VM frees up, a broken one does not, and --wait has to
      # tell those apart to know whether standing in line will help.
      return 2
    fi
    if [[ "$ATTACH_PUSH_KEY" == "true" ]]; then
      local detected
      detected=$(printf '%s\n' "$remote_out" | tail -n 1)
      [[ -z "$detected" ]] || ssh_user="$detected"
    fi
  fi
  [[ -n "$ssh_user" ]] || ssh_user="${USER:-$(whoami)}"

  local control_path="/tmp/tpu_cm_${ip}_22_${ssh_user}"
  if [[ ${#control_path} -gt 107 ]]; then
    local sock_hash
    sock_hash=$(printf '%s_22_%s' "$ip" "$ssh_user" | sha256sum | cut -c1-16)
    control_path="/tmp/tpu_cm_${sock_hash}.sock"
  fi

  local tmp_env="${session}.tmp.$$"
  cat <<EOF > "$tmp_env"
export TPU_NAME="${name}"
export TPU_ZONE="${zone}"
export TPU_PROJECT="${ALLOWED_PROJECT}"
export TPU_IP="${ip}"
export SSH_CONTROL_PATH="${control_path}"
export SSH_USER="${ssh_user}"
export SSH_IDENTITY="${SSH_IDENTITY_PATH}"
EOF
  chmod 600 "$tmp_env"
  mv -f "$tmp_env" "$session"
  log "vm_${index} attached to ${name} in ${zone}"
}

# The gcloud filter `attach` uses to decide which VMs it may borrow.
#
# The accelerator clause is not decoration. Once --name-prefix is widened past
# the names this script creates, it is the only thing standing between the relay
# and a v5p or an 8-chip host, either of which would accept the lease and then
# fail every test on it. The match is anchored so v5litepod-1 does not also
# match v5litepod-16.
attach_filter() {
  local clauses=()
  [[ -n "$ATTACH_NAME_PREFIX" ]] && clauses+=("name~${ATTACH_NAME_PREFIX}")
  [[ -n "$ATTACH_ACCELERATOR_TYPE" ]] \
    && clauses+=("acceleratorType~${ATTACH_ACCELERATOR_TYPE}\$")
  clauses+=("state:READY")

  local filter="${clauses[0]}" i
  for (( i = 1; i < ${#clauses[@]}; i++ )); do
    filter+=" AND ${clauses[i]}"
  done
  echo "$filter"
}

# Every VM `attach` may borrow, as "name<TAB>zone" lines, sorted by name so that
# two callers looking at the same fleet agree on the order and therefore on who
# holds the queue.
discover_candidates() {
  local zone name
  for zone in "${ZONES[@]}"; do
    while read -r name; do
      [[ -n "$name" ]] || continue
      printf '%s\t%s\n' "$name" "$zone"
    done < <(gcloud compute tpus tpu-vm list \
      --zone="$zone" --project="$ALLOWED_PROJECT" \
      --filter="$(attach_filter)" \
      --format="value(name.basename())" 2>/dev/null)
  done | sort
}

# One pass over the candidates. Sets ATTACH_READY, ATTACH_FOUND and ATTACH_HELD;
# the caller decides whether that is good enough or worth retrying.
attach_attempt() {
  local candidates="$1"

  local attached=()
  local session
  for session in "$POOL_DIR"/*.env; do
    [[ -s "$session" ]] || continue
    attached+=("$(sed -n 's/^export TPU_NAME="\(.*\)"$/\1/p' "$session")")
  done

  local index=0 pids=() found=0
  local zone name
  while IFS=$'\t' read -r name zone; do
    [[ -n "$name" ]] || continue
    # Re-attaching must not hand the same VM out twice under two slots.
    local already=false entry
    for entry in ${attached[@]+"${attached[@]}"}; do
      [[ "$entry" == "$name" ]] && { already=true; break; }
    done
    if [[ "$already" == "true" ]]; then
      log "${name} is already in the pool"
      continue
    fi
    while [[ -s "${POOL_DIR}/vm_${index}.env" ]]; do
      index=$(( index + 1 ))
    done
    found=$(( found + 1 ))
    attach_one "$index" "$name" "$zone" &
    pids+=("$!")
    index=$(( index + 1 ))
  done <<< "$candidates"

  local ready=0 held=0 pid rc
  for pid in ${pids[@]+"${pids[@]}"}; do
    wait "$pid"
    rc=$?
    case "$rc" in
      0) ready=$(( ready + 1 )) ;;
      2) held=$(( held + 1 )) ;;
    esac
  done

  ATTACH_READY="$ready"
  ATTACH_FOUND="$found"
  ATTACH_HELD="$held"
}

# Drops this pool's claim on one VM. Best effort: a VM that has already gone
# away, or that never let us in, must not stop the rest of the pool from being
# handed back. The claim ages out on its own anyway.
release_one() {
  local session="$1"
  local TPU_IP="" SSH_USER="" SSH_CONTROL_PATH=""
  # shellcheck disable=SC1090
  source "$session"
  [[ -n "$TPU_IP" && -n "$SSH_USER" ]] || return 0
  ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 \
    ${SSH_CONTROL_PATH:+-o "ControlPath=${SSH_CONTROL_PATH}"} \
    "${SSH_USER}@${TPU_IP}" \
    "[ \"\$(cat ${REMOTE_CLAIM_PATH}/owner 2>/dev/null)\" = \"$(claim_owner)\" ] \
       && rm -rf ${REMOTE_CLAIM_PATH}" >/dev/null 2>&1 || true
}

release_claims() {
  [[ "$ATTACH_CLAIM" == "true" ]] || return 0
  local session pids=()
  for session in "$POOL_DIR"/*.env; do
    [[ -s "$session" ]] || continue
    release_one "$session" &
    pids+=("$!")
  done
  local pid
  for pid in ${pids[@]+"${pids[@]}"}; do
    wait "$pid" || true
  done
}

# Wipes the pool's session files without touching the hardware. Used both by
# `detach` and by a --wait attempt that has to give back a partial fleet.
forget_sessions() {
  rm -f "$POOL_DIR"/*.env "$POOL_DIR"/*.env.lock "$POOL_DIR"/*.quarantine
}

# The VM that holds the waiting line. Candidates are sorted, so every caller
# looking at the same fleet picks the same one without talking to the others.
# Callers who name different zones can pick differently; that costs ordering,
# not safety, because taking the fleet is still all or nothing.
queue_pick_host() {
  local candidates="$1"
  [[ -n "$QUEUE_HOST" ]] && return 0
  IFS=$'\t' read -r QUEUE_HOST QUEUE_HOST_ZONE <<< "$(head -n 1 <<< "$candidates")"
}

queue_ssh() {
  gcloud compute tpus tpu-vm ssh "$QUEUE_HOST" \
    --zone="$QUEUE_HOST_ZONE" --project="$ALLOWED_PROJECT" \
    --command="$1" 2>/dev/null | tr -d '\r'
}

# Takes a ticket. The name starts with a zero-padded timestamp taken on the
# queue host, so sorting the directory sorts by arrival and no workstation's
# clock can jump the line.
queue_join() {
  local out
  out=$(queue_ssh "
mkdir -p '${REMOTE_QUEUE_PATH}'
t='${REMOTE_QUEUE_PATH}/'\$(date +%s | awk '{printf \"%012d\", \$1}').\$\$.\$RANDOM
mkdir \"\$t\" 2>/dev/null || exit 1
printf '%s\n' '$(claim_owner)' > \"\$t/owner\"
echo \"TICKET:\$t\"
")
  QUEUE_TICKET=$(sed -n 's/^TICKET://p' <<< "$out" | head -n 1)
  [[ -n "$QUEUE_TICKET" ]]
}

# Refreshes this ticket, drops tickets nobody has touched in QUEUE_STALE
# seconds, and prints how many runs are ahead. Prints GONE if this ticket was
# swept, which means the caller stalled and has to take a new one.
queue_poll() {
  local out
  out=$(queue_ssh "
t='${QUEUE_TICKET}'
[ -d \"\$t\" ] || { echo GONE; exit 0; }
touch \"\$t\"
now=\$(date +%s)
for d in ${REMOTE_QUEUE_PATH}/*/; do
  [ -d \"\$d\" ] || continue
  n=\${d%/}
  [ \"\$n\" = \"\$t\" ] && continue
  age=\$(( now - \$(stat -c %Y \"\$n\" 2>/dev/null || echo \"\$now\") ))
  [ \"\$age\" -gt ${QUEUE_STALE} ] && rm -rf \"\$n\"
done
ls -1 '${REMOTE_QUEUE_PATH}' 2>/dev/null | sort \\
  | awk -v me=\"\$(basename \"\$t\")\" '\$0==me{exit} {c++} END{print \"AHEAD:\" c+0}'
")
  if [[ "$out" == *GONE* ]]; then
    echo "GONE"
    return 0
  fi
  sed -n 's/^AHEAD://p' <<< "$out" | head -n 1
}

queue_leave() {
  [[ -n "$QUEUE_TICKET" ]] || return 0
  queue_ssh "rm -rf '${QUEUE_TICKET}'" >/dev/null
  QUEUE_TICKET=""
}

cmd_attach() {
  mkdir -p "$POOL_DIR"

  local candidates
  candidates=$(discover_candidates)
  [[ -n "$candidates" ]] || die "No VMs to attach to. Bring a fleet up first."

  if [[ "$ATTACH_WAIT" != "true" ]]; then
    attach_attempt "$candidates"
    log "Attached ${ATTACH_READY}/${ATTACH_FOUND} VM(s) in ${POOL_DIR}"
    [[ "$ATTACH_READY" -gt 0 ]] || die "No VMs to attach to. Bring a fleet up first."
    return 0
  fi

  queue_pick_host "$candidates"
  if ! queue_join; then
    # The line is a convenience, not the lock. Losing it means people go in a
    # worse order, not that two runs land on one chip.
    log "WARNING: could not take a ticket on ${QUEUE_HOST}, waiting without one"
  fi
  trap 'queue_leave' EXIT

  local deadline=$(( SECONDS + ATTACH_WAIT_TIMEOUT )) announced=""
  while :; do
    local ahead="0" rejoined=false
    if [[ -n "$QUEUE_TICKET" ]]; then
      ahead=$(queue_poll)
      if [[ "$ahead" == "GONE" ]]; then
        # Swept for being quiet too long. Take a new ticket at the back and
        # read the line again next time round rather than barging in now.
        queue_join || log "WARNING: lost the queue on ${QUEUE_HOST}, waiting without a ticket"
        rejoined=true
      fi
      [[ -n "$ahead" ]] || ahead="0"
    fi

    if [[ "$rejoined" == "true" ]]; then
      : # fall through to the sleep
    elif [[ "$ahead" == "0" ]]; then
      attach_attempt "$candidates"
      if [[ "$ATTACH_HELD" -eq 0 && "$ATTACH_READY" -gt 0 ]]; then
        queue_leave
        trap - EXIT
        log "Attached ${ATTACH_READY}/${ATTACH_FOUND} VM(s) in ${POOL_DIR}"
        return 0
      fi
      # Giving back a half-taken fleet is the whole point: two waiters that
      # each keep their half would both sit there holding what the other needs.
      release_claims
      forget_sessions
      if [[ "$announced" != "busy:${ATTACH_HELD}" ]]; then
        log "The fleet is busy: ${ATTACH_HELD} of ${ATTACH_FOUND} VM(s) held, waiting"
        announced="busy:${ATTACH_HELD}"
      fi
    elif [[ "$ahead" != "$announced" ]]; then
      log "Waiting for the fleet: ${ahead} run(s) ahead of you"
      announced="$ahead"
    fi

    (( SECONDS < deadline )) || {
      queue_leave
      die "Gave up waiting for the fleet after ${ATTACH_WAIT_TIMEOUT}s.
If the run holding it is dead, take the VMs with --force-claim; otherwise its
claim ages out on its own after ${ATTACH_CLAIM_TTL}s."
    }
    sleep "$ATTACH_WAIT_POLL"
  done
}

# The counterpart to `attach`. Forgets the pool without touching the hardware,
# which is what a CI job has to do at the end of a run: `down` would delete VMs
# that belong to whoever brought the fleet up.
cmd_detach() {
  [[ -d "$POOL_DIR" ]] || return 0
  release_claims
  forget_sessions
  log "Detached from the fleet. The VMs are still running."
}

# Deletes fleet VMs that no session file knows about. Provisioning can create
# the node and then fail before writing its session, and the pool has no record
# of it, so the loop above walks straight past it while it keeps billing.
sweep_orphans() {
  local pids=() zone leftover
  for zone in "${ZONES[@]}"; do
    while read -r leftover; do
      [[ -n "$leftover" ]] || continue
      log "deleting unrecorded ${leftover} in ${zone}"
      gcloud compute tpus tpu-vm delete "$leftover" \
        --zone="$zone" --project="$ALLOWED_PROJECT" --quiet >/dev/null 2>&1 &
      pids+=("$!")
    done < <(gcloud compute tpus tpu-vm list \
      --zone="$zone" --project="$ALLOWED_PROJECT" \
      --filter="name~${VM_NAME_PREFIX}" \
      --format="value(name.basename())" 2>/dev/null)
  done

  local pid
  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done
}

cmd_status() {
  local zone
  for zone in "${DEFAULT_ZONES[@]}"; do
    echo "-- ${zone}"
    gcloud compute tpus tpu-vm list \
      --project="$ALLOWED_PROJECT" --zone="$zone" 2>&1 | sed 's/^/   /'
  done

  echo "-- pool ${POOL_DIR}"
  ls -1 "$POOL_DIR"/*.env 2>/dev/null | sed 's/^/   /' || echo "   (empty)"

  echo "-- deadline"
  local reaper_pid=""
  [[ -f "$REAPER_PID_FILE" ]] && reaper_pid="$(cat "$REAPER_PID_FILE")"
  if [[ -n "$reaper_pid" ]] && kill -0 "$reaper_pid" 2>/dev/null; then
    echo "   armed (pid ${reaper_pid})"
  else
    echo "   not armed"
  fi
}

main() {
  local subcommand="${1:-}"
  [[ $# -gt 0 ]] && shift
  parse_args "$@"

  # Only `attach` reads the widened search. Refusing it elsewhere means nobody
  # can pass it to `down` and believe they widened what gets deleted.
  if [[ "$ATTACH_NAME_PREFIX_SET" == "true" && "$subcommand" != "attach" ]]; then
    die "--name-prefix only applies to 'attach', not '${subcommand}'."
  fi

  case "$subcommand" in
    up)     cmd_up ;;
    down)   cmd_down ;;
    attach) cmd_attach ;;
    detach) cmd_detach ;;
    deadline) cmd_deadline ;;
    status) cmd_status ;;
    *)      die "Usage: $0 <up|down|attach|detach|status> [options]" ;;
  esac
}

main "$@"
