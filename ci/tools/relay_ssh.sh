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
# ci/tools/relay_ssh.sh
# Shared SSH plumbing for the host side of the relay. Source it; don't run it.
#
# Every hop to a TPU VM goes through an OpenSSH ControlMaster so that a test
# action costs one round trip instead of a fresh handshake. The master is
# started detached, because whoever provisioned the VM has usually exited by
# the time the tests run, and a master left in that process group dies with it.

# shellcheck shell=bash

RELAY_SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o ConnectTimeout=10
)

relay_ssh_identity() {
  echo "${SSH_IDENTITY:-${HOME}/.ssh/google_compute_engine}"
}

relay_ssh_master_alive() {
  local control_path="$1" remote="$2"
  [[ -S "$control_path" ]] && \
    ssh -O check -S "$control_path" "${RELAY_SSH_OPTS[@]}" "$remote" 2>/dev/null
}

# Brings the master back if it is missing, and reports whether the VM is
# reachable either way.
relay_ssh_ensure_master() {
  local control_path="$1" remote="$2"
  relay_ssh_master_alive "$control_path" "$remote" && return 0

  rm -f "$control_path"
  local identity_args=()
  local identity
  identity="$(relay_ssh_identity)"
  [[ -f "$identity" ]] && identity_args=(-i "$identity" -o IdentitiesOnly=yes)

  setsid ssh -M -N -f \
    -S "$control_path" \
    "${RELAY_SSH_OPTS[@]}" \
    -o ControlPersist=4h \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=4 \
    ${identity_args[@]+"${identity_args[@]}"} \
    "$remote" 2>/dev/null

  relay_ssh_master_alive "$control_path" "$remote"
}
