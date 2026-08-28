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

readonly ALLOWED_PROJECT="rbe-tpu-oss"

# Global state for with-tpu subcommand, in-flight provisioning, and trap handling
_WITH_TPU_CLEANUP_DONE=0
_WITH_TPU_CMD_EXIT=""
_WITH_TPU_CHILD_PID=""
_WITH_TPU_ACTIVE=0
_WITH_TPU_ACTIVE_NAME=""
_WITH_TPU_ACTIVE_ZONE=""
_CURRENT_PROVISIONING_NAME=""
_CURRENT_PROVISIONING_ZONE=""
# Honors TPU_SESSION_ENV so several relays can share a host, each pinned to its
# own TPU VM. relay_test_runner.sh reads the same variable.
readonly SESSION_ENV_FILE="${TPU_SESSION_ENV:-/tmp/tpu_active_session.env}"
# Where the TPU runtime gets pip-installed on the VM. The remote executor puts
# this on PYTHONPATH; nothing here depends on the image's Python version.
readonly REMOTE_BASE_DIR="/tmp/torch_tpu_relay/base"
# The key gcloud hands the VM and that the control master then authenticates
# with. CI has no developer home directory, so it points this at an ephemeral
# key it generated for the run.
readonly SSH_IDENTITY_PATH="${SSH_IDENTITY:-${HOME}/.ssh/google_compute_engine}"

cleanup_standalone_up() {
  local sig="${1:-TERM}"
  trap '' INT TERM HUP
  echo "Signal $sig received during provisioning. Cleaning up in-flight TPU VM..." >&2
  cmd_down --name "${_CURRENT_PROVISIONING_NAME:-}" --zone "${_CURRENT_PROVISIONING_ZONE:-}"
  _CURRENT_PROVISIONING_NAME=""
  _CURRENT_PROVISIONING_ZONE=""
  trap - INT TERM HUP
  case "$sig" in
    INT)  exit 130 ;;
    TERM) exit 143 ;;
    HUP)  exit 129 ;;
    *)    exit 1 ;;
  esac
}


if [[ -z "${TEST_TMP_DIR:-}" && -n "${GCLOUD_CALLS_LOG:-}" ]]; then
  export TEST_TMP_DIR="$(dirname "$GCLOUD_CALLS_LOG")"
fi

show_help() {
  cat <<'EOF'
Usage: scripts/spot_tpu_manager.sh <subcommand> [options]

Subcommands:
  up        Provision a Spot TPU v5e instance, preheat runtime, and start OpenSSH ControlMaster.
            Options:
              --zone ZONE       Target GCP zone (default: europe-west4-b, fallback: us-central1-a, us-east5-a)
              --name NAME       Custom TPU VM name (default: spot-tpu-v5e-<timestamp>-<rand>)
              --project PROJ    GCP project (must be rbe-tpu-oss)

  down      Terminate OpenSSH ControlMaster and delete the active TPU VM in rbe-tpu-oss.
            Options:
              --name NAME       Name of TPU VM to delete (default: from session file)
              --zone ZONE       Zone of TPU VM to delete (default: from session file)
              --project PROJ    GCP project (must be rbe-tpu-oss)

  status    Display status of the active TPU session, GCP VM state, and hardware nodes.
            Options:
              --project PROJ    GCP project (must be rbe-tpu-oss)

  with-tpu  Execute a command wrapped with automatic TPU provisioning and teardown on exit.
            Usage: scripts/spot_tpu_manager.sh with-tpu [--zone ZONE] <command> [args...]

  reap      Scan rbe-tpu-oss for orphaned Spot TPU VMs older than max-age-hours and purge them.
            Options:
              --max-age-hours N Age threshold in hours (default: 2)
              --dry-run         Print candidate VMs without deleting them
              --project PROJ    GCP project (must be rbe-tpu-oss)
EOF
}

enforce_project_boundary() {
  local target_project="${1:-$ALLOWED_PROJECT}"
  if [[ "$target_project" != "$ALLOWED_PROJECT" ]]; then
    echo "ERROR: Project boundary violation: '$target_project' is not allowed." >&2
    echo "This utility is strictly restricted to project '$ALLOWED_PROJECT'." >&2
    exit 1
  fi

  local env_vars=("CLOUDSDK_CORE_PROJECT" "GOOGLE_CLOUD_PROJECT" "GCP_PROJECT" "GCLOUD_PROJECT")
  for var in "${env_vars[@]}"; do
    if [[ -n "${!var:-}" && "${!var}" != "$ALLOWED_PROJECT" ]]; then
      echo "ERROR: Environment variable $var is set to '${!var}' (violates allowed project '$ALLOWED_PROJECT')." >&2
      exit 1
    fi
  done
}

# Reads one field off a TPU VM. Returns empty rather than failing, so callers
# can tell "no answer" apart from a specific state.
tpu_describe() {
  local tpu_name="$1"
  local zone="$2"
  local field="$3"
  gcloud compute tpus tpu-vm describe "$tpu_name" \
    --project="$ALLOWED_PROJECT" \
    --zone="$zone" \
    --format="value(${field})" 2>/dev/null || true
}

# Gives up on a half-built VM. Deleting it here is what keeps a failed
# provision from leaving Spot capacity billing in the background.
fail_and_teardown() {
  local tpu_name="$1"
  local zone="$2"
  shift 2
  echo "ERROR: $*" >&2
  cmd_down --name "$tpu_name" --zone "$zone"
  exit 1
}

cmd_up() {
  enforce_project_boundary "${CLI_PROJECT:-$ALLOWED_PROJECT}"

  if [[ "${_WITH_TPU_ACTIVE:-0}" -eq 0 ]]; then
    trap 'cleanup_standalone_up INT' INT
    trap 'cleanup_standalone_up TERM' TERM
    trap 'cleanup_standalone_up HUP' HUP
  fi

  _CURRENT_PROVISIONING_NAME="${CLI_NAME:-spot-tpu-v5e-$(date +%s)-$RANDOM}"
  local tpu_name="${_CURRENT_PROVISIONING_NAME}"
  local candidate_zones=()
  if [[ -n "${CLI_ZONE:-}" ]]; then
    candidate_zones=("${CLI_ZONE}")
  else
    candidate_zones=("europe-west4-b" "us-central1-a" "us-east5-a")
  fi

  local chosen_zone=""
  for z in "${candidate_zones[@]}"; do
    _CURRENT_PROVISIONING_ZONE="$z"
    echo "Attempting to create Spot TPU '$tpu_name' in zone '$z'..."
    local create_args=(
      --project="$ALLOWED_PROJECT"
      --zone="$z"
      --accelerator-type="v5litepod-1"
      --version="v2-alpha-tpuv5-lite"
    )
    # Spot v5e in this project has been getting preempted within minutes, which
    # a 78-minute presubmit cannot survive. --on-demand trades cost for a VM
    # that stays put.
    [[ "$CLI_ON_DEMAND" == "true" ]] || create_args+=(--spot)

    if gcloud compute tpus tpu-vm create "$tpu_name" "${create_args[@]}"; then
      chosen_zone="$z"
      _CURRENT_PROVISIONING_ZONE="$chosen_zone"
      if [[ "${_WITH_TPU_ACTIVE:-0}" -eq 1 ]]; then
        _WITH_TPU_ACTIVE_NAME="${tpu_name}"
        _WITH_TPU_ACTIVE_ZONE="${chosen_zone}"
      fi
      echo "Creation request accepted for zone '$z'."
      # The session file is only written once the VM is fully usable (see the
      # end of this function). Until then _CURRENT_PROVISIONING_* and
      # `spot_tpu_manager.sh reap` are what find a half-built VM.
      break
    else
      echo "WARNING: Creation failed in zone '$z'. Cleaning up any partial state..." >&2
      gcloud compute tpus tpu-vm delete "$tpu_name" \
        --project="$ALLOWED_PROJECT" \
        --zone="$z" \
        --quiet 2>/dev/null || true
      _CURRENT_PROVISIONING_ZONE=""
    fi
  done

  if [[ -z "$chosen_zone" ]]; then
    echo "ERROR: Failed to allocate Spot TPU in all candidate zones (${candidate_zones[*]})." >&2
    _CURRENT_PROVISIONING_NAME=""
    _CURRENT_PROVISIONING_ZONE=""
    exit 1
  fi

  echo "Waiting for TPU VM '$tpu_name' in '$chosen_zone' to reach READY state..."
  local state=""
  local max_attempts=60
  for ((i=1; i<=max_attempts; i++)); do
    state=$(tpu_describe "$tpu_name" "$chosen_zone" "state")
    if [[ "$state" == "READY" ]]; then
      echo "TPU VM entered READY state."
      break
    elif [[ "$state" == "FAILED" || "$state" == "PREEMPTED" ]]; then
      fail_and_teardown "$tpu_name" "$chosen_zone" \
        "TPU VM entered unexpected state '$state'."
    fi
    echo "Current state: '${state:-PROVISIONING}', waiting 5s (attempt $i/$max_attempts)..."
    sleep 5
  done

  if [[ "$state" != "READY" ]]; then
    fail_and_teardown "$tpu_name" "$chosen_zone" \
      "Timed out waiting for TPU VM '$tpu_name' to become READY (state: $state)."
  fi

  # Prefer the external IP; VMs created without one still answer on the
  # internal address from inside the VPC.
  local tpu_ip=""
  tpu_ip=$(tpu_describe "$tpu_name" "$chosen_zone" \
    "networkEndpoints[0].accessConfig.externalIp")

  if [[ -z "$tpu_ip" ]]; then
    echo "External IP not in accessConfig, falling back to networkEndpoints[0].ipAddress..." >&2
    tpu_ip=$(tpu_describe "$tpu_name" "$chosen_zone" "networkEndpoints[0].ipAddress")
  fi

  if [[ -z "$tpu_ip" ]]; then
    fail_and_teardown "$tpu_name" "$chosen_zone" \
      "Could not resolve IP for TPU VM '$tpu_name'."
  fi
  echo "Resolved TPU VM public IP: $tpu_ip"

  echo "Waiting for SSH connectivity and detecting authorized user..."
  local ssh_user=""
  local ssh_attempts=30
  for ((i=1; i<=ssh_attempts; i++)); do
    ssh_user=$(gcloud compute tpus tpu-vm ssh "$tpu_name" \
      --project="$ALLOWED_PROJECT" \
      --zone="$chosen_zone" \
      --command="whoami" 2>/dev/null | tr -d '\r\n')
    if [[ -n "$ssh_user" ]]; then
      break
    fi
    echo "Waiting for SSH to become responsive (attempt $i/$ssh_attempts)..."
    sleep 5
  done

  if [[ -z "$ssh_user" ]]; then
    ssh_user="${USER:-$(whoami)}"
    echo "WARNING: Could not detect remote user via gcloud ssh; defaulting to '$ssh_user'" >&2
  else
    echo "SSH connection ready. Remote user: $ssh_user"
  fi

  local control_path="/tmp/tpu_cm_${tpu_ip}_22_${ssh_user}"
  if [[ ${#control_path} -gt 107 ]]; then
    local sock_hash
    sock_hash=$(printf '%s_22_%s' "${tpu_ip}" "${ssh_user}" | sha256sum | cut -c1-16)
    control_path="/tmp/tpu_cm_${sock_hash}.sock"
  fi
  if [[ -S "$control_path" || -e "$control_path" ]]; then
    ssh -O exit -S "$control_path" "${ssh_user}@${tpu_ip}" 2>/dev/null || true
    rm -f "$control_path"
  fi

  # setsid so the master outlives this script. spot_tpu_fleet.sh runs `up` as a
  # background job and the whole process group goes away when it returns.
  echo "Starting OpenSSH ControlMaster daemon at $control_path..."
  setsid ssh -M -N -f \
    -S "$control_path" \
    -o ControlPersist=4h \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o IdentitiesOnly=yes \
    -i "${SSH_IDENTITY_PATH}" \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=4 \
    -o BatchMode=yes \
    "${ssh_user}@${tpu_ip}"

  if ! ssh -O check -S "$control_path" "${ssh_user}@${tpu_ip}" 2>/dev/null; then
    echo "ERROR: Failed to establish OpenSSH ControlMaster socket at $control_path." >&2
    cmd_down --name "$tpu_name" --zone "$chosen_zone"
    exit 1
  fi

  local start_ms end_ms latency
  start_ms=$(date +%s%3N)
  ssh -S "$control_path" -o BatchMode=yes "${ssh_user}@${tpu_ip}" "true"
  end_ms=$(date +%s%3N)
  latency=$(( end_ms - start_ms ))
  echo "OpenSSH multiplexing active. Round-trip ping latency: ${latency}ms"

  echo "Verifying VFIO permissions and clearing device locks..."
  ssh -S "$control_path" "${ssh_user}@${tpu_ip}" '
    sudo chmod a+rw /dev/vfio/* /dev/vfio 2>/dev/null || true
    sudo fuser -k -9 /dev/vfio/* 2>/dev/null || true
    fuser -k /dev/vfio/* 2>/dev/null || true
    sleep 0.5
  '

  # No Python runtime is installed here on purpose. The relay ships the
  # hermetic CPython and the exact wheels bazel resolved (see
  # ci/tools/stage_relay_base.sh), so anything pip put on the VM would be a
  # second, different copy of torch and libtpu with the wrong ABI.
  echo "Checking TPU device nodes..."
  if ! ssh -S "$control_path" "${ssh_user}@${tpu_ip}" '
    set -e
    [ -d /dev/vfio ] || { echo "no /dev/vfio directory" >&2; exit 1; }
    nodes=$(ls /dev/vfio | grep -v "^vfio$" || true)
    [ -n "$nodes" ] || { echo "no TPU device nodes under /dev/vfio" >&2; exit 1; }
    echo "TPU device nodes: $nodes"
  '; then
    fail_and_teardown "$tpu_name" "$chosen_zone" \
      "TPU silicon is not visible on VM '$tpu_name'."
  fi

  ssh -S "$control_path" "${ssh_user}@${tpu_ip}" "mkdir -p ${REMOTE_BASE_DIR}"

  # Nothing schedules a shutdown or a poweroff inside the guest. `shutdown -h`
  # does not release a TPU node: the service brings the guest straight back up,
  # so it keeps billing, but it comes back with /tmp wiped (that is where the
  # base cache lives) and with a dead SSH control master. A pool has no way to
  # tell that VM apart from a healthy one, so it keeps handing it work and it
  # keeps failing in milliseconds. The deadline belongs on the host, where
  # `spot_tpu_fleet.sh` can delete the node through the API.

  local session_env="$SESSION_ENV_FILE"
  local tmp_env="${session_env}.tmp.$$"
  cat <<EOF > "$tmp_env"
export TPU_NAME="${tpu_name}"
export TPU_ZONE="${chosen_zone}"
export TPU_PROJECT="${ALLOWED_PROJECT}"
export TPU_IP="${tpu_ip}"
export SSH_CONTROL_PATH="${control_path}"
export SSH_USER="${ssh_user}"
export SSH_IDENTITY="${SSH_IDENTITY_PATH}"
EOF
  chmod 600 "$tmp_env"
  mv -f "$tmp_env" "$session_env"
  if [[ "${_WITH_TPU_ACTIVE:-0}" -eq 0 ]]; then
    _CURRENT_PROVISIONING_NAME=""
    _CURRENT_PROVISIONING_ZONE=""
  else
    _WITH_TPU_ACTIVE_NAME="${tpu_name}"
    _WITH_TPU_ACTIVE_ZONE="${chosen_zone}"
  fi
  if [[ "${_WITH_TPU_ACTIVE:-0}" -eq 0 ]]; then
    trap - INT TERM HUP
  fi
  echo "Active TPU session initialized and written to $session_env"
}

cmd_down() {
  local arg_name=""
  local arg_zone=""
  local arg_project=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --name)
        if [[ $# -lt 2 ]]; then
          echo "ERROR: --name requires an argument." >&2
          exit 1
        fi
        arg_name="$2"
        shift 2
        ;;
      --name=*)
        arg_name="${1#*=}"
        shift
        ;;
      --zone)
        if [[ $# -lt 2 ]]; then
          echo "ERROR: --zone requires an argument." >&2
          exit 1
        fi
        arg_zone="$2"
        shift 2
        ;;
      --zone=*)
        arg_zone="${1#*=}"
        shift
        ;;
      --project)
        if [[ $# -lt 2 || -z "${2:-}" ]]; then
          echo "ERROR: --project requires a non-empty argument." >&2
          exit 1
        fi
        arg_project="$2"
        shift 2
        ;;
      --project=*)
        arg_project="${1#*=}"
        if [[ -z "$arg_project" ]]; then
          echo "ERROR: --project requires a non-empty argument." >&2
          exit 1
        fi
        shift
        ;;
      --)
        shift
        break
        ;;
      -h|--help)
        show_help
        return 0
        ;;
      -*)
        echo "ERROR: Unknown option for down: $1" >&2
        show_help >&2
        exit 1
        ;;
      *)
        if [[ -z "$arg_name" ]]; then
          arg_name="$1"
        elif [[ -z "$arg_zone" ]]; then
          arg_zone="$1"
        else
          echo "ERROR: Unexpected positional argument for down: $1" >&2
          exit 1
        fi
        shift
        ;;
    esac
  done

  local target_project="${arg_project:-${CLI_PROJECT:-$ALLOWED_PROJECT}}"
  enforce_project_boundary "$target_project"

  local session_env="${SESSION_ENV_FILE:-/tmp/tpu_active_session.env}"
  local tpu_name="${arg_name:-}"
  local tpu_zone="${arg_zone:-}"
  local control_path=""
  local tpu_ip=""
  local ssh_user=""

  if [[ -n "$tpu_name" && -n "$tpu_zone" ]]; then
    : # Explicit arguments present: bypass session env sourcing entirely
  else
    if [[ -f "$session_env" ]]; then
      # shellcheck disable=SC1090
      source "$session_env"
      local env_name="${TPU_NAME:-}"
      local env_zone="${TPU_ZONE:-}"
      control_path="${SSH_CONTROL_PATH:-}"
      tpu_ip="${TPU_IP:-}"
      ssh_user="${SSH_USER:-}"

      tpu_name="${arg_name:-${env_name:-}}"
      tpu_zone="${arg_zone:-${env_zone:-}}"
    fi
  fi

  if [[ -n "$tpu_name" && -z "$tpu_zone" ]]; then
    tpu_zone=$(gcloud compute tpus tpu-vm list --zone=- --project="$ALLOWED_PROJECT" \
      --filter="name.basename():$tpu_name" --format="value(name.segment(3))" 2>/dev/null | head -n 1 || true)
  fi

  if [[ -n "$control_path" && -S "$control_path" ]]; then
    echo "Terminating OpenSSH ControlMaster socket $control_path..."
    ssh -O exit -S "$control_path" "${ssh_user:-dummy}@${tpu_ip:-127.0.0.1}" 2>/dev/null || true
    rm -f "$control_path"
  fi

  if [[ -n "$tpu_name" && -n "$tpu_zone" ]]; then
    echo "Deleting TPU VM '$tpu_name' in zone '$tpu_zone'..."
    if ! timeout 120s gcloud compute tpus tpu-vm delete "$tpu_name" \
        --project="$ALLOWED_PROJECT" \
        --zone="$tpu_zone" \
        --quiet; then
      echo "Synchronous deletion timed out or failed; issuing async delete fallback..." >&2
      gcloud compute tpus tpu-vm delete "$tpu_name" \
        --project="$ALLOWED_PROJECT" \
        --zone="$tpu_zone" \
        --quiet \
        --async || true
    fi
  else
    echo "No active TPU VM specified or recorded to delete."
  fi

  if [[ -n "${session_env:-}" ]]; then
    rm -f "$session_env" "${session_env}.tmp."* 2>/dev/null || true
  fi
  unset TPU_NAME TPU_ZONE TPU_PROJECT TPU_IP SSH_CONTROL_PATH SSH_USER 2>/dev/null || true
  _CURRENT_PROVISIONING_NAME=""
  _CURRENT_PROVISIONING_ZONE=""
  _WITH_TPU_ACTIVE_NAME=""
  _WITH_TPU_ACTIVE_ZONE=""
  echo "Teardown complete."
}

cmd_status() {
  enforce_project_boundary "${CLI_PROJECT:-$ALLOWED_PROJECT}"

  local session_env="$SESSION_ENV_FILE"
  if [[ ! -f "$session_env" ]]; then
    echo "STATUS: NO_ACTIVE_SESSION (state file $session_env not found)"
    echo "Checking for any active TPU VMs in $ALLOWED_PROJECT across all zones..."
    gcloud compute tpus tpu-vm list --zone=- --project="$ALLOWED_PROJECT" \
      --format="table(name.basename(),name.segment(3),createTime,state,schedulingConfig.spot)"
    return 0
  fi

  # shellcheck disable=SC1090
  source "$session_env"
  echo "=== Active TPU Session ==="
  echo "VM Name:           ${TPU_NAME:-unknown}"
  echo "Zone:              ${TPU_ZONE:-unknown}"
  echo "Project:           ${TPU_PROJECT:-unknown}"
  echo "Public IP:         ${TPU_IP:-unknown}"
  echo "SSH User:          ${SSH_USER:-unknown}"
  echo "Control Socket:    ${SSH_CONTROL_PATH:-unknown}"

  local gcp_state
  gcp_state=$(gcloud compute tpus tpu-vm describe "${TPU_NAME:-}" \
    --project="$ALLOWED_PROJECT" \
    --zone="${TPU_ZONE:-}" \
    --format="value(state)" 2>&1 || echo "UNKNOWN")
  echo "GCP State:         $gcp_state"

  if [[ -n "${SSH_CONTROL_PATH:-}" && -S "${SSH_CONTROL_PATH}" ]] && \
     ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_USER}@${TPU_IP}" 2>/dev/null; then
    echo "SSH Socket:        ACTIVE"
    local vfio_nodes
    vfio_nodes=$(ssh -S "${SSH_CONTROL_PATH}" -o BatchMode=yes "${SSH_USER}@${TPU_IP}" \
      "ls -l /dev/vfio/* 2>/dev/null || echo 'No VFIO nodes'" 2>/dev/null || echo "UNREACHABLE")
    echo "Hardware Nodes:"
    echo "$vfio_nodes"
  else
    echo "SSH Socket:        INACTIVE / CLOSED"
  fi
}

cleanup_with_tpu() {
  local trap_rc=$?
  local sig="${1:-EXIT}"

  if [[ "${_WITH_TPU_CLEANUP_DONE:-0}" -eq 1 ]]; then
    return
  fi
  _WITH_TPU_CLEANUP_DONE=1
  trap '' INT TERM HUP

  local final_rc
  if [[ -n "${_WITH_TPU_CMD_EXIT:-}" ]]; then
    final_rc="${_WITH_TPU_CMD_EXIT}"
  else
    case "$sig" in
      INT)  final_rc=130 ;;
      TERM) final_rc=143 ;;
      HUP)  final_rc=129 ;;
      *)
        final_rc="$trap_rc"
        if [[ "$final_rc" -eq 0 ]]; then
          final_rc=1
        fi
        ;;
    esac
  fi

  if [[ -n "${_WITH_TPU_CHILD_PID:-}" ]] && kill -0 "${_WITH_TPU_CHILD_PID}" 2>/dev/null; then
    echo "Terminating wrapped child process ${_WITH_TPU_CHILD_PID}..." >&2
    kill -TERM "${_WITH_TPU_CHILD_PID}" 2>/dev/null || true
    wait "${_WITH_TPU_CHILD_PID}" 2>/dev/null || true
    _WITH_TPU_CHILD_PID=""
  fi

  echo "Tearing down TPU session from with-tpu trap (exit code: $final_rc)..." >&2
  cmd_down --name "${_WITH_TPU_ACTIVE_NAME:-${_CURRENT_PROVISIONING_NAME:-}}" --zone "${_WITH_TPU_ACTIVE_ZONE:-${_CURRENT_PROVISIONING_ZONE:-}}"
  _WITH_TPU_ACTIVE_NAME=""
  _WITH_TPU_ACTIVE_ZONE=""
  _CURRENT_PROVISIONING_NAME=""
  _CURRENT_PROVISIONING_ZONE=""
  trap - EXIT INT TERM HUP
  exit "$final_rc"
}

cmd_with_tpu() {
  enforce_project_boundary "${CLI_PROJECT:-$ALLOWED_PROJECT}"

  if [[ ${#REMAINING_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: No command specified for with-tpu." >&2
    echo "Usage: scripts/spot_tpu_manager.sh with-tpu [--zone ZONE] <command> [args...]" >&2
    exit 1
  fi

  _WITH_TPU_CLEANUP_DONE=0
  _WITH_TPU_CMD_EXIT=""
  _WITH_TPU_CHILD_PID=""
  _WITH_TPU_ACTIVE=1
  _WITH_TPU_ACTIVE_NAME=""
  _WITH_TPU_ACTIVE_ZONE=""

  trap 'cleanup_with_tpu EXIT' EXIT
  trap 'cleanup_with_tpu INT' INT
  trap 'cleanup_with_tpu TERM' TERM
  trap 'cleanup_with_tpu HUP' HUP

  cmd_up
  _WITH_TPU_ACTIVE_NAME="${_WITH_TPU_ACTIVE_NAME:-${_CURRENT_PROVISIONING_NAME:-}}"
  _WITH_TPU_ACTIVE_ZONE="${_WITH_TPU_ACTIVE_ZONE:-${_CURRENT_PROVISIONING_ZONE:-}}"
  echo "Running command with active TPU session: ${REMAINING_ARGS[*]}"
  ( trap - INT TERM HUP QUIT; exec "${REMAINING_ARGS[@]}" ) <&0 &
  _WITH_TPU_CHILD_PID=$!
  if wait "$_WITH_TPU_CHILD_PID"; then
    _WITH_TPU_CMD_EXIT=0
  else
    _WITH_TPU_CMD_EXIT=$?
  fi
  _WITH_TPU_CHILD_PID=""
}

cmd_reap() {
  enforce_project_boundary "${CLI_PROJECT:-$ALLOWED_PROJECT}"

  local max_age_hours="${CLI_MAX_AGE_HOURS:-2}"
  if ! [[ "$max_age_hours" =~ ^-?[0-9]+$ ]]; then
    echo "ERROR: --max-age-hours must be an integer, got '$max_age_hours'." >&2
    exit 1
  fi

  local dry_run="${CLI_DRY_RUN:-false}"
  local now_epoch
  now_epoch=$(date +%s)
  local max_age_seconds
  max_age_seconds=$(( max_age_hours * 3600 ))

  echo "Scanning $ALLOWED_PROJECT across all zones for Spot TPU VMs older than $max_age_hours hours..."
  local vms_raw
  vms_raw=$(gcloud compute tpus tpu-vm list --zone=- --project="$ALLOWED_PROJECT" \
    --format="value(name.basename(),name.segment(3),createTime,schedulingConfig.spot)" 2>/dev/null || true)

  local found_count=0
  local reaped_count=0

  while read -r name zone create_time is_spot; do
    [[ -z "${name:-}" ]] && continue

    if [[ "$is_spot" != "true" && "$name" != spot-tpu-* ]]; then
      continue
    fi

    if [[ -z "${create_time:-}" ]]; then
      continue
    fi

    local created_epoch=0
    created_epoch=$(date -d "$create_time" +%s 2>/dev/null || echo 0)
    if [[ "$created_epoch" -eq 0 ]]; then
      continue
    fi

    local age=$(( now_epoch - created_epoch ))
    if [[ $age -ge $max_age_seconds ]]; then
      found_count=$(( found_count + 1 ))
      local age_h
      age_h=$(awk "BEGIN {printf \"%.1f\", $age / 3600}")
      if [[ "$dry_run" == "true" ]]; then
        echo "[DRY-RUN] Would delete stale Spot TPU VM '$name' in '$zone' (age: ${age_h}h)"
      else
        echo "Reaping stale Spot TPU VM '$name' in '$zone' (age: ${age_h}h)..."
        gcloud compute tpus tpu-vm delete "$name" \
          --project="$ALLOWED_PROJECT" \
          --zone="$zone" \
          --quiet \
          --async
        reaped_count=$(( reaped_count + 1 ))
      fi
    fi
  done <<< "$vms_raw"

  if [[ $found_count -eq 0 ]]; then
    echo "No stale Spot TPU VMs found in $ALLOWED_PROJECT."
  fi

  for sock in /tmp/tpu_cm_* /tmp/tpu_ssh_control_*; do
    if [[ -S "$sock" || -e "$sock" ]]; then
      if ! ssh -O check -S "$sock" dummy 2>/dev/null; then
        rm -f "$sock"
      fi
    fi
  done
}

CLI_SUBCOMMAND=""
CLI_ZONE=""
CLI_NAME=""
CLI_PROJECT=""
CLI_MAX_AGE_HOURS=2
CLI_DRY_RUN=false
CLI_ON_DEMAND=false
REMAINING_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    up|down|status|with-tpu|reap)
      CLI_SUBCOMMAND="$1"
      shift
      break
      ;;
    -h|--help|help)
      show_help
      exit 0
      ;;
    --project)
      CLI_PROJECT="$2"
      shift 2
      ;;
    --project=*)
      CLI_PROJECT="${1#*=}"
      shift
      ;;
    *)
      echo "ERROR: Unknown option or subcommand: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CLI_SUBCOMMAND" ]]; then
  show_help
  exit 0
fi

if [[ "$CLI_SUBCOMMAND" == "with-tpu" ]]; then
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --on-demand)
        CLI_ON_DEMAND=true
        shift
        ;;
      --zone)
        CLI_ZONE="$2"
        shift 2
        ;;
      --zone=*)
        CLI_ZONE="${1#*=}"
        shift
        ;;
      --name)
        CLI_NAME="$2"
        shift 2
        ;;
      --name=*)
        CLI_NAME="${1#*=}"
        shift
        ;;
      --project)
        CLI_PROJECT="$2"
        shift 2
        ;;
      --project=*)
        CLI_PROJECT="${1#*=}"
        shift
        ;;
      --)
        shift
        break
        ;;
      -*)
        echo "ERROR: Unknown option for with-tpu: $1" >&2
        exit 1
        ;;
      *)
        break
        ;;
    esac
  done
  REMAINING_ARGS=("$@")
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --on-demand)
        CLI_ON_DEMAND=true
        shift
        ;;
      --zone)
        CLI_ZONE="$2"
        shift 2
        ;;
      --zone=*)
        CLI_ZONE="${1#*=}"
        shift
        ;;
      --name)
        CLI_NAME="$2"
        shift 2
        ;;
      --name=*)
        CLI_NAME="${1#*=}"
        shift
        ;;
      --project)
        CLI_PROJECT="$2"
        shift 2
        ;;
      --project=*)
        CLI_PROJECT="${1#*=}"
        shift
        ;;
      --max-age-hours)
        if [[ $# -lt 2 ]]; then
          echo "ERROR: --max-age-hours requires an integer argument." >&2
          exit 1
        fi
        CLI_MAX_AGE_HOURS="$2"
        shift 2
        if ! [[ "$CLI_MAX_AGE_HOURS" =~ ^-?[0-9]+$ ]]; then
          echo "ERROR: Invalid value for --max-age-hours: '$CLI_MAX_AGE_HOURS' is not an integer." >&2
          exit 1
        fi
        ;;
      --max-age-hours=*)
        CLI_MAX_AGE_HOURS="${1#*=}"
        shift
        if ! [[ "$CLI_MAX_AGE_HOURS" =~ ^-?[0-9]+$ ]]; then
          echo "ERROR: Invalid value for --max-age-hours: '$CLI_MAX_AGE_HOURS' is not an integer." >&2
          exit 1
        fi
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
        echo "ERROR: Unknown option: $1" >&2
        show_help >&2
        exit 1
        ;;
    esac
  done
fi

enforce_project_boundary "${CLI_PROJECT:-$ALLOWED_PROJECT}"

case "$CLI_SUBCOMMAND" in
  up)
    cmd_up
    ;;
  down)
    cmd_down ${CLI_NAME:+--name "$CLI_NAME"} ${CLI_ZONE:+--zone "$CLI_ZONE"} ${CLI_PROJECT:+--project "$CLI_PROJECT"}
    ;;
  status)
    cmd_status
    ;;
  with-tpu)
    cmd_with_tpu
    ;;
  reap)
    cmd_reap
    ;;
  *)
    echo "ERROR: Unrecognized subcommand: $CLI_SUBCOMMAND" >&2
    show_help >&2
    exit 1
    ;;
esac
