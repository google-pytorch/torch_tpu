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
#                                    [--ip-mode auto|internal|external]
#   scripts/spot_tpu_fleet.sh detach [--pool DIR]
#   scripts/spot_tpu_fleet.sh status [--pool DIR] [--zone ZONE]
#   scripts/spot_tpu_fleet.sh prepare [--zone ZONE] [--name-prefix PREFIX]
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
# Everyone who uses the fleet shares /tmp/torch_tpu_relay on each VM, so it has
# to be writable by all of them. Left to a plain mkdir it belongs to whoever ran
# first, and the next person's claim fails with a permission error that reads,
# unhelpfully, as a held VM. `prepare` installs a systemd-tmpfiles rule that
# recreates the tree root-owned and world-writable on every boot, and repairs
# VMs that are already wrong. Run it once per fleet, and again whenever someone
# reports CLAIM_ERROR:
#
#   scripts/spot_tpu_fleet.sh prepare --zone europe-west4-b
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
#
# Everything above assumes the driver is a corp workstation, which reaches the
# chips on their external addresses through corp-ssh-helper. A GCE VM in the
# same VPC is already inside the firewall and has to use the internal addresses
# instead. --ip-mode picks; the default `auto` looks for corp-ssh-helper first,
# then asks the metadata server, so a workstation behaves exactly as before.
#
#   scripts/spot_tpu_fleet.sh attach --ip-mode internal --zone europe-west4-b


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
# The tree every user of the fleet shares, and the rule that keeps it that way.
# See cmd_prepare for why it is 0777 and why `claim` is not in the rule.
readonly REMOTE_RELAY_ROOT="${TORCH_TPU_REMOTE_RELAY_ROOT:-/tmp/torch_tpu_relay}"
readonly REMOTE_TMPFILES_CONF="/etc/tmpfiles.d/torch-tpu-relay.conf"
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
# What the last attach attempt managed: taken, seen, held by someone else, and
# unusable because the relay directory on the VM will not let this user write.
ATTACH_READY=0
ATTACH_FOUND=0
ATTACH_HELD=0
ATTACH_BROKEN=0

# Which address the driver uses to reach the chips. A corp workstation has to go
# through the external IP, because corp-ssh-helper is what gets it past the
# firewall. A GCE VM in the same VPC is already inside and reaches the internal
# IP directly, where the external one is denied. `auto` works that out; see
# resolve_ip_mode.
IP_MODE="auto"
IP_MODE_RESOLVED=""
# Filled in by resolve_ip_mode. Internal mode needs `gcloud ... ssh` to target
# the internal address too, or it dials the external one and hangs.
GCLOUD_SSH_IP_ARGS=()
SSH_PROXY_BIN="${CORP_SSH_HELPER_BIN:-corp-ssh-helper}"
readonly METADATA_URL="${TORCH_TPU_METADATA_URL:-http://metadata.google.internal/computeMetadata/v1/instance/id}"


die() {
  echo "ERROR [spot_tpu_fleet]: $*" >&2
  exit 1
}

log() {
  echo "[spot_tpu_fleet] $*"
}

# Decides once, in the parent shell, which address family reaches the chips.
# Attach forks a job per VM, so resolving lazily would make every one of them
# repeat the probe.
#
#   corp-ssh-helper on PATH          -> external  (a corp workstation; this is
#                                       the existing path, and it short-circuits
#                                       before the probe so it costs nothing)
#   else the metadata server answers -> internal  (a GCE VM inside the VPC)
#   else                             -> external  (and the SSH error explains it)
resolve_ip_mode() {
  if [[ "$IP_MODE" != "auto" ]]; then
    IP_MODE_RESOLVED="$IP_MODE"
  elif command -v "$SSH_PROXY_BIN" >/dev/null 2>&1; then
    IP_MODE_RESOLVED="external"
  elif curl -fsS --max-time 1 -H 'Metadata-Flavor: Google' \
      "$METADATA_URL" >/dev/null 2>&1; then
    IP_MODE_RESOLVED="internal"
  else
    IP_MODE_RESOLVED="external"
  fi

  if [[ "$IP_MODE_RESOLVED" == "internal" ]]; then
    GCLOUD_SSH_IP_ARGS=(--internal-ip)
  else
    GCLOUD_SSH_IP_ARGS=()
  fi
}

# The usage text is the comment block at the top of this file, so there is only
# one copy of it to keep current. The block is found rather than hardcoded by
# line number: skip the licence, which ends at the first blank line, then print
# comments until the code starts. A hardcoded range silently truncates the help
# the moment anyone adds a paragraph up there.
show_help() {
  awk '
    !started { if ($0 == "") started = 1; next }
    /^#/     { sub(/^# ?/, ""); print; next }
    { exit }
  ' "${BASH_SOURCE[0]}"
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
      --ip-mode|--ip-mode=*)
        [[ "$1" == *=* ]] && IP_MODE="${1#*=}" || { IP_MODE="${2:-}"; shift; }
        shift ;;
      -h|--help)
        show_help
        exit 0 ;;
      *)
        die "Unknown option '$1'." ;;
    esac
  done

  [[ "$FLEET_SIZE" =~ ^[0-9]+$ && "$FLEET_SIZE" -gt 0 ]] \
    || die "--size must be a positive integer, got '$FLEET_SIZE'."
  [[ "$DEADLINE_MINUTES" =~ ^[0-9]+$ ]] \
    || die "--deadline-minutes must be a non-negative integer, got '$DEADLINE_MINUTES'."
  case "$IP_MODE" in
    auto|internal|external) ;;
    *) die "--ip-mode must be auto, internal or external, got '$IP_MODE'." ;;
  esac
  [[ ${#ZONES[@]} -gt 0 ]] || ZONES=("${DEFAULT_ZONES[@]}")
  REAPER_PID_FILE="${POOL_DIR}/reaper.pid"
  resolve_ip_mode
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
#
# A failed mkdir has two very different causes and they must not be reported the
# same way. Losing the race means somebody holds the VM, which clears on its own
# and is what --wait is for. Being unable to create the directory at all means
# the relay root belongs to another user and is not writable by this one, which
# no amount of waiting fixes. That confusion cost somebody twenty minutes in a
# --wait loop against a completely idle fleet.
claim_script() {
  if [[ "$ATTACH_CLAIM" != "true" ]]; then
    printf 'whoami\n'
    return 0
  fi
  cat <<EOF
c="${REMOTE_CLAIM_PATH}"
d=\$(dirname "\$c")
# 0777 because the fleet is shared and the next person to attach is not in this
# person's group. No sticky bit: expired claims have to be reapable by anyone.
mkdir -m 0777 -p "\$d" 2>/dev/null || mkdir -p "\$d" 2>/dev/null
now=\$(date +%s)

claim_error() {
  perms=\$(ls -ld "\$d" 2>/dev/null | awk '{print \$1, \$3}')
  echo "CLAIM_ERROR:\$1 (\$perms)"
  exit 0
}

[ -d "\$d" ] || claim_error "cannot create \$d"
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
  if [ -d "\$c" ]; then
    claim_error "cannot clear the expired claim at \$c"
  fi
fi
# 0777 on the claim itself, not just on its parent. Reaping an expired claim
# means rm -rf'ing the owner and at files inside it, and that needs write
# permission on the claim directory. Left at the creator's umask, --claim-ttl
# and --force-claim only ever work for the person who took the claim.
if ! mkdir -m 0777 "\$c" 2>/dev/null; then
  # A directory that appeared between the check above and this mkdir is a lost
  # race. Anything else is a permission wall.
  [ -d "\$c" ] || claim_error "cannot create \$c"
  echo "CLAIMED_BY:\$(cat "\$c/owner" 2>/dev/null || echo 'someone who got there first')"
  exit 0
fi
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

  # Internal mode does not fall back to the external address. From inside the
  # VPC the external one is denied by the hierarchical firewall, so offering it
  # would trade a clear "no address" for a ten-minute hang.
  local ip
  if [[ "$IP_MODE_RESOLVED" == "internal" ]]; then
    ip=$(gcloud compute tpus tpu-vm describe "$name" \
      --zone="$zone" --project="$ALLOWED_PROJECT" \
      --format="value(networkEndpoints[0].ipAddress)" 2>>"$log_file")
  else
    ip=$(gcloud compute tpus tpu-vm describe "$name" \
      --zone="$zone" --project="$ALLOWED_PROJECT" \
      --format="value(networkEndpoints[0].accessConfig.externalIp)" 2>>"$log_file")
    if [[ -z "$ip" ]]; then
      ip=$(gcloud compute tpus tpu-vm describe "$name" \
        --zone="$zone" --project="$ALLOWED_PROJECT" \
        --format="value(networkEndpoints[0].ipAddress)" 2>>"$log_file")
    fi
  fi
  [[ -n "$ip" ]] || { log "${name}: no reachable address, skipping"; return 1; }

  # `gcloud ... ssh` uploads the caller's public key as a side effect, which is
  # the whole reason to call it: the identity attaching is not the identity that
  # created the VM, so nothing has authorised it yet. Its stdout also names the
  # account the VM actually logs us in as, which OS Login can rewrite, and
  # carries the verdict on the claim.
  local ssh_user="$ATTACH_SSH_USER"
  if [[ "$ATTACH_PUSH_KEY" == "true" || "$ATTACH_CLAIM" == "true" ]]; then
    local remote_out holder broken
    remote_out=$(gcloud compute tpus tpu-vm ssh "$name" \
      --zone="$zone" --project="$ALLOWED_PROJECT" \
      "${GCLOUD_SSH_IP_ARGS[@]}" \
      --command="$(claim_script)" 2>>"$log_file" | tr -d '\r')
    broken=$(printf '%s\n' "$remote_out" | sed -n 's/^CLAIM_ERROR://p' | head -n 1)
    if [[ -n "$broken" ]]; then
      log "${name}: ${broken}"
      # 3, not 2: waiting cannot change a permission bit, so the caller has to
      # stop and say so rather than poll.
      return 3
    fi
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

  local ready=0 held=0 broken=0 pid rc
  for pid in ${pids[@]+"${pids[@]}"}; do
    wait "$pid"
    rc=$?
    case "$rc" in
      0) ready=$(( ready + 1 )) ;;
      2) held=$(( held + 1 )) ;;
      3) broken=$(( broken + 1 )) ;;
    esac
  done

  ATTACH_READY="$ready"
  ATTACH_FOUND="$found"
  ATTACH_HELD="$held"
  ATTACH_BROKEN="$broken"
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
    "${GCLOUD_SSH_IP_ARGS[@]}" \
    --command="$1" 2>/dev/null | tr -d '\r'
}

# Takes a ticket. The name starts with a zero-padded timestamp taken on the
# queue host, so sorting the directory sorts by arrival and no workstation's
# clock can jump the line.
queue_join() {
  local out
  out=$(queue_ssh "
mkdir -m 0777 -p '$(dirname "${REMOTE_QUEUE_PATH}")' 2>/dev/null
mkdir -m 0777 -p '${REMOTE_QUEUE_PATH}' 2>/dev/null || mkdir -p '${REMOTE_QUEUE_PATH}'
t='${REMOTE_QUEUE_PATH}/'\$(date +%s | awk '{printf \"%012d\", \$1}').\$\$.\$RANDOM
# 0777 so another waiter can sweep this ticket if this run walks away;
# see the claim directory above for the same reasoning.
mkdir -m 0777 \"\$t\" 2>/dev/null || exit 1
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

# A VM whose relay directory this user cannot write to is not busy and never
# will be, so say what is wrong and name the command that fixes it rather than
# letting the caller queue behind it.
die_if_claims_are_broken() {
  [[ "$ATTACH_BROKEN" -gt 0 ]] || return 0
  release_claims
  forget_sessions
  die "${ATTACH_BROKEN} of ${ATTACH_FOUND} VM(s) would not let this user take a claim.
The relay directory on them belongs to somebody else. Waiting will not help. Fix it with:
  $0 prepare$(printf ' --zone %s' "${ZONES[@]}")"
}

cmd_attach() {
  mkdir -p "$POOL_DIR"

  local candidates
  candidates=$(discover_candidates)
  [[ -n "$candidates" ]] || die "No VMs to attach to. Bring a fleet up first."

  if [[ "$ATTACH_WAIT" != "true" ]]; then
    attach_attempt "$candidates"
    die_if_claims_are_broken
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
      die_if_claims_are_broken
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

# What `prepare` runs on each VM.
#
# The mode is 0777 with no sticky bit, and that is deliberate. Sticky is the
# reflexive answer for a shared directory under /tmp, and it would break three
# things the relay depends on, all of which delete an entry another user
# created: reaping a claim past its TTL, --force-claim, and the base store
# reaper in ci/tools/stage_relay_base.sh. Sticky would turn a stale four-hour
# claim into a permanent one, which is a worse version of the bug this fixes.
# The tradeoff is that any user can delete any other user's cache or sandbox.
# On an ephemeral /tmp shared by a handful of trusted people, that is the
# intended behaviour.
prepare_script() {
  cat <<EOF
set -e
sudo tee ${REMOTE_TMPFILES_CONF} >/dev/null <<'CONF'
# Managed by scripts/spot_tpu_fleet.sh prepare. Do not edit by hand.
#
# Root-owned so no user owns the tree and "whoever attached first wins" cannot
# happen. World-writable so the second person to attach can take a claim. Not
# sticky: expired claims have to be reapable by whoever finds them.
#
# claim/ is deliberately absent. That directory IS the mutex, so pre-creating
# it would make every VM read as permanently held.
d ${REMOTE_RELAY_ROOT}           0777 root root -
d ${REMOTE_RELAY_ROOT}/queue     0777 root root -
d ${REMOTE_RELAY_ROOT}/base      0777 root root -
d ${REMOTE_RELAY_ROOT}/store     0777 root root -
d ${REMOTE_RELAY_ROOT}/payloads  0777 root root -
d ${REMOTE_RELAY_ROOT}/sandboxes 0777 root root -
CONF
sudo systemd-tmpfiles --create ${REMOTE_TMPFILES_CONF}
ls -ld ${REMOTE_RELAY_ROOT} | awk '{print \$1, \$3, \$4, \$NF}'
EOF
}

prepare_one() {
  local name="$1" zone="$2" out
  if ! out=$(gcloud compute tpus tpu-vm ssh "$name" \
      --zone="$zone" --project="$ALLOWED_PROJECT" \
      "${GCLOUD_SSH_IP_ARGS[@]}" \
      --command="$(prepare_script)" 2>&1 | tr -d '\r'); then
    log "${name}: FAILED -- $(printf '%s' "$out" | tail -n 1)"
    return 1
  fi
  log "${name}: $(printf '%s\n' "$out" | tail -n 1)"
}

# Installs the tmpfiles rule on every VM `attach` would consider and applies it
# now, which also repairs a VM whose relay directory already belongs to one
# user. Safe to re-run. This is the answer when somebody reports CLAIM_ERROR.
cmd_prepare() {
  local candidates
  candidates=$(discover_candidates)
  [[ -n "$candidates" ]] || die "No VMs found to prepare in: ${ZONES[*]}"

  local name zone pids=() total=0
  while IFS=$'\t' read -r name zone; do
    [[ -n "$name" ]] || continue
    total=$(( total + 1 ))
    prepare_one "$name" "$zone" &
    pids+=("$!")
  done <<< "$candidates"

  local pid ok=0
  for pid in ${pids[@]+"${pids[@]}"}; do
    wait "$pid" && ok=$(( ok + 1 ))
  done

  log "Prepared ${ok}/${total} VM(s)."
  [[ "$ok" -eq "$total" ]] || die "$(( total - ok )) VM(s) could not be prepared."
}

cmd_status() {
  local zone
  # ZONES, not DEFAULT_ZONES: parse_args already falls back to the defaults, and
  # reading them directly here threw away --zone without saying so.
  for zone in "${ZONES[@]}"; do
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
  # parse_args only ever sees what is left after the subcommand is shifted off,
  # so a bare -h would be read as the subcommand and fall through to the usage
  # error instead of printing anything.
  case "${1:-}" in
    -h|--help)
      show_help
      exit 0 ;;
    "")
      show_help >&2
      die "no subcommand given." ;;
  esac

  local subcommand="${1:-}"
  [[ $# -gt 0 ]] && shift
  parse_args "$@"

  # Only the subcommands that borrow VMs read the widened search. Refusing it
  # elsewhere means nobody can pass it to `down` and believe they widened what
  # gets deleted.
  case "$subcommand" in
    attach|prepare) ;;
    *)
      [[ "$ATTACH_NAME_PREFIX_SET" != "true" ]] \
        || die "--name-prefix only applies to 'attach' and 'prepare', not '${subcommand}'." ;;
  esac

  case "$subcommand" in
    up)     cmd_up ;;
    down)   cmd_down ;;
    attach) cmd_attach ;;
    detach) cmd_detach ;;
    deadline) cmd_deadline ;;
    status) cmd_status ;;
    prepare) cmd_prepare ;;
    # Exists so callers that wrap this script can ask which address family it
    # settled on rather than working it out a second time and disagreeing.
    ip-mode) printf '%s\n' "$IP_MODE_RESOLVED" ;;
    *)      die "Usage: $0 <up|down|attach|detach|status|prepare|ip-mode> [options]" ;;
  esac
}

main "$@"
