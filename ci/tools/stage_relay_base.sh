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
# ci/tools/stage_relay_base.sh
# Ships the shared half of the runfiles tree to the TPU VMs, so each test action
# only has to send its own files.
#
# Every torch_tpu test pulls in the same couple of gigabytes: the hermetic
# CPython that rules_python downloaded, the pip wheels bazel resolved (torch and
# libtpu are most of it), and the shared C++ solibs. Sending that per test
# action is not an option, and installing a second copy with the VM's own pip
# gives you the wrong interpreter and the wrong ABI. So it lands in
# /tmp/torch_tpu_relay/base once per VM and remote_tpu_executor.sh symlinks it
# into each sandbox.
#
# It goes over in two layers because they change at different rates:
#   deps   the interpreter and the wheels, only when MODULE.bazel.lock moves
#   solib  the shared C++ libraries, on every code change
#
# Both land under a content-addressed store, and each run gets a base directory
# of symlinks pointing into it. A fixed base path was fine while one person ran
# this by hand, but a shared VM pool means two runs can stage at once, and the
# second one would overwrite the shared libraries the first one's tests are
# already running against. Keying on content also means two runs of the same
# code stage nothing the second time.
set -euo pipefail

readonly ALLOWED_PROJECT="rbe-tpu-oss"
readonly REMOTE_RELAY_DIR="/tmp/torch_tpu_relay"
readonly REMOTE_STORE_DIR="${REMOTE_RELAY_DIR}/store"
readonly LOCAL_CACHE_DIR="${TORCH_TPU_BASE_TARBALL_DIR:-/tmp/torch_tpu_relay/base_tarballs}"
readonly LAYERS=(deps solib)
# Anything in the store older than this and not part of the run being staged is
# from a build nobody is waiting on. A day is far longer than any run.
readonly STORE_TTL_DAYS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=ci/tools/relay_ssh.sh
source "${SCRIPT_DIR}/relay_ssh.sh"
readonly SSH_OPTS=("${RELAY_SSH_OPTS[@]}")

CLI_POOL=""
CLI_SESSION=""
CLI_JOBS=12
CLI_EMIT_BASE_DIR=""

die() { echo "ERROR [stage_relay_base]: $*" >&2; exit 1; }
log() { echo "[stage_relay_base] $*"; }

show_help() {
  cat <<'EOF'
Usage:
  stage_relay_base.sh --pool DIR [--jobs N] [--emit-base-dir FILE]
  stage_relay_base.sh --session FILE [--emit-base-dir FILE]

  --pool DIR           Directory of *.env session files written by spot_tpu_fleet.sh.
  --session FILE       A single session file written by spot_tpu_manager.sh up.
  --jobs N             How many VMs to push to at once (default 12).
  --emit-base-dir FILE Write the remote base directory this staging produced to
                       FILE. The path is content-addressed, so the caller has to
                       be told it; pass it to the tests as
                       TORCH_TPU_RELAY_BASE_DIR.

Build the test targets before staging: the shared libraries only exist once
bazel has produced them. Re-running is cheap, a layer whose stamp already
matches is skipped.
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --pool)      CLI_POOL="${2:?--pool needs a directory}"; shift 2 ;;
      --pool=*)    CLI_POOL="${1#*=}"; shift ;;
      --session)   CLI_SESSION="${2:?--session needs a file}"; shift 2 ;;
      --session=*) CLI_SESSION="${1#*=}"; shift ;;
      --jobs)      CLI_JOBS="${2:?--jobs needs a number}"; shift 2 ;;
      --jobs=*)    CLI_JOBS="${1#*=}"; shift ;;
      --emit-base-dir)   CLI_EMIT_BASE_DIR="${2:?--emit-base-dir needs a file}"; shift 2 ;;
      --emit-base-dir=*) CLI_EMIT_BASE_DIR="${1#*=}"; shift ;;
      -h|--help)   show_help; exit 0 ;;
      *)           die "unknown argument: $1" ;;
    esac
  done

  [[ -n "$CLI_POOL" || -n "$CLI_SESSION" ]] || die "pass --pool or --session"
  [[ "$CLI_JOBS" =~ ^[0-9]+$ && "$CLI_JOBS" -gt 0 ]] || die "--jobs must be a positive integer"
}

# What to ship comes from the runfiles trees bazel just built, not from the
# whole external directory: that one holds 3.1 GB, most of it repositories no
# presubmit test ever imports.
#
# Each entry below is "parent_directory<TAB>member", which is what tar wants.
# A member that shows up in several runfiles trees is the same content every
# time, so the first tree to offer it wins.
resolve_sources() {
  local roots=()
  # -H so find walks through the bazel-bin convenience symlink.
  mapfile -t roots < <(find -H "${REPO_ROOT}/bazel-bin" -maxdepth 4 -name '*.runfiles' -type d 2>/dev/null)
  [[ ${#roots[@]} -gt 0 ]] \
    || die "no runfiles trees under bazel-bin; build the test targets first"

  DEP_MEMBERS=()
  SOLIB_MEMBERS=()
  local seen_deps=" " seen_solib=" "
  local root entry name

  for root in "${roots[@]}"; do
    for entry in "$root"/rules_python*; do
      [[ -d "$entry" ]] || continue
      name="$(basename "$entry")"
      [[ "$seen_deps" == *" ${name} "* ]] && continue
      seen_deps+="${name} "
      DEP_MEMBERS+=("${root}"$'\t'"${name}")
    done

    for entry in "$root"/_main/_solib_x86_64/*; do
      [[ -d "$entry" ]] || continue
      name="$(basename "$entry")"
      [[ "$seen_solib" == *" ${name} "* ]] && continue
      seen_solib+="${name} "
      SOLIB_MEMBERS+=("${root}/_main"$'\t'"_solib_x86_64/${name}")
    done

    # This repo's own extension modules. They sit in _main/csrc rather than
    # under _solib_x86_64, so the loop above walks past them, and they are the
    # same build outputs in every runfiles tree. libpywrap_torch_tpu_common.so
    # alone is 493 MB: outside the base cache it gets dereferenced into the
    # per-test payload and shipped again on every cache miss.
    if [[ -d "$root/_main/csrc" && "$seen_solib" != *" csrc "* ]]; then
      seen_solib+="csrc "
      SOLIB_MEMBERS+=("${root}/_main"$'\t'"csrc")
    fi
  done

  [[ ${#DEP_MEMBERS[@]} -gt 0 ]] \
    || die "runfiles trees carry no rules_python repositories; is the build complete?"
  log "found ${#DEP_MEMBERS[@]} dependency repositories and ${#SOLIB_MEMBERS[@]} solib directories across ${#roots[@]} runfiles tree(s)"
}

layer_members() {
  case "$1" in
    deps)  printf '%s\n' ${DEP_MEMBERS[@]+"${DEP_MEMBERS[@]}"} ;;
    solib) printf '%s\n' ${SOLIB_MEMBERS[@]+"${SOLIB_MEMBERS[@]}"} ;;
  esac
}

# The wheel set only moves when the lock file does. The C++ libraries move on
# every code change, so those are hashed by size and timestamp.
layer_stamp() {
  local layer="$1"
  {
    if [[ "$layer" == "deps" ]]; then
      sha256sum "${REPO_ROOT}/MODULE.bazel.lock" 2>/dev/null || true
      layer_members deps | cut -f2 | sort
    else
      local parent member
      while IFS=$'\t' read -r parent member; do
        [[ -n "$member" ]] || continue
        find -L "${parent}/${member}" -type f -printf '%P %s %T@\n' 2>/dev/null
      done < <(layer_members solib)
    fi
  } | sort | sha256sum | cut -d' ' -f1
}

# Both layers are dereferenced: bazel fills them with absolute symlinks into the
# local output base, which mean nothing on the VM.
build_layer() {
  local layer="$1"
  local stamp="$2"
  local tarball="${LOCAL_CACHE_DIR}/${layer}_${stamp}.tar.gz"

  if [[ -s "$tarball" ]]; then
    echo "$tarball"
    return 0
  fi

  mkdir -p "$LOCAL_CACHE_DIR"
  local compressor="gzip -1"
  command -v pigz >/dev/null 2>&1 && compressor="pigz -1"

  local -a tar_args=()
  local parent member
  while IFS=$'\t' read -r parent member; do
    [[ -n "$member" ]] || continue
    tar_args+=(-C "$parent" "$member")
  done < <(layer_members "$layer")

  log "packing ${layer} layer..." >&2
  tar -chf - --ignore-failed-read \
    --exclude='*.a' \
    --exclude='*.o' \
    --exclude='*.params' \
    --exclude='*.cppmap' \
    --exclude='*__pycache__*' \
    --exclude='*.pyc' \
    "${tar_args[@]}" 2>/dev/null | $compressor > "${tarball}.partial"
  mv -f "${tarball}.partial" "$tarball"
  log "${layer} layer is $(du -h "$tarball" | cut -f1)" >&2
  echo "$tarball"
}

stage_one() {
  local session="$1"
  local base_dir="$2"
  shift 2
  # Remaining arguments are "layer:stamp:tarball" triples.

  local TPU_IP="" SSH_USER="" SSH_CONTROL_PATH="" TPU_NAME="" SSH_IDENTITY=""
  # shellcheck disable=SC1090
  source "$session"
  local label="${TPU_NAME:-$(basename "$session")}"

  if [[ -z "$TPU_IP" || -z "$SSH_USER" || -z "$SSH_CONTROL_PATH" ]]; then
    echo "[stage_relay_base] ${label}: incomplete session file, skipping" >&2
    return 1
  fi
  local remote="${SSH_USER}@${TPU_IP}"

  if ! relay_ssh_ensure_master "$SSH_CONTROL_PATH" "$remote"; then
    echo "[stage_relay_base] ${label}: cannot open an SSH session" >&2
    return 1
  fi

  # tests/tpu_errors_test.py copies a 4 TB result back to the host and expects
  # the TPU to be the thing that reports the OOM. Under the kernel's default
  # heuristic a 47 GB v5litepod-1 refuses the host-side malloc first, so the
  # test sees "DefaultCPUAllocator: can't allocate memory" and fails on the
  # wrong error. A ct5lp CI runner has the RAM to get past it. Allowing the
  # overcommit lets the copy run far enough for the TPU to answer.
  ssh -S "$SSH_CONTROL_PATH" "${SSH_OPTS[@]}" "$remote" \
    "sudo sysctl -w vm.overcommit_memory=1" >/dev/null 2>&1 \
    || echo "[stage_relay_base] ${label}: could not relax memory overcommit;" \
            "tpu_errors_test will report a host OOM" >&2

  local spec layer_dirs=()
  for spec in "$@"; do
    local layer="${spec%%:*}"
    local rest="${spec#*:}"
    local stamp="${rest%%:*}"
    local tarball="${rest#*:}"
    local layer_dir="${REMOTE_STORE_DIR}/${layer}-${stamp}"
    layer_dirs+=("$layer_dir")

    # The marker goes in last, so a transfer that died halfway leaves a
    # directory that reads as incomplete and gets rebuilt rather than one that
    # reads as current and serves half a Python installation.
    if ssh -S "$SSH_CONTROL_PATH" "${SSH_OPTS[@]}" "$remote" \
        "[ -f '${layer_dir}/.complete' ]" 2>/dev/null; then
      echo "[stage_relay_base] ${label}: ${layer} already current"
      continue
    fi

    echo "[stage_relay_base] ${label}: pushing ${layer}..."
    if ! ssh -S "$SSH_CONTROL_PATH" "${SSH_OPTS[@]}" "$remote" \
        "rm -rf '${layer_dir}' && mkdir -p '${layer_dir}'" 2>/dev/null; then
      echo "[stage_relay_base] ${label}: cannot reach VM" >&2
      return 1
    fi
    if ! ssh -S "$SSH_CONTROL_PATH" "${SSH_OPTS[@]}" "$remote" \
        "tar -xzf - -C '${layer_dir}'" < "$tarball"; then
      echo "[stage_relay_base] ${label}: ${layer} transfer failed" >&2
      return 1
    fi
    ssh -S "$SSH_CONTROL_PATH" "${SSH_OPTS[@]}" "$remote" \
      "touch '${layer_dir}/.complete'"
  done

  # The base directory is just a view over the layers: one symlink per top-level
  # entry. remote_tpu_executor.sh links those into each sandbox, and a link to a
  # link resolves the same as a link to the directory.
  local link_script="set -e; rm -rf '${base_dir}'; mkdir -p '${base_dir}'"
  local layer_dir
  for layer_dir in "${layer_dirs[@]}"; do
    link_script+="; for entry in '${layer_dir}'/*; do"
    link_script+=" [ -e \"\$entry\" ] || continue;"
    link_script+=" ln -sfn \"\$entry\" '${base_dir}/'\"\$(basename \"\$entry\")\"; done"
    # Keep what this run needs out of reach of the sweep below.
    link_script+="; touch '${layer_dir}'"
  done
  if ! ssh -S "$SSH_CONTROL_PATH" "${SSH_OPTS[@]}" "$remote" "$link_script" 2>/dev/null; then
    echo "[stage_relay_base] ${label}: could not build the base directory" >&2
    return 1
  fi

  # Layers from older runs would otherwise pile up until the boot disk fills.
  ssh -S "$SSH_CONTROL_PATH" "${SSH_OPTS[@]}" "$remote" \
    "find '${REMOTE_STORE_DIR}' -maxdepth 1 -mindepth 1 -type d -mtime +${STORE_TTL_DAYS} \
       -exec rm -rf {} + 2>/dev/null;
     find '${REMOTE_RELAY_DIR}' -maxdepth 1 -mindepth 1 -name 'base-*' -mtime +${STORE_TTL_DAYS} \
       -exec rm -rf {} + 2>/dev/null" >/dev/null 2>&1 || true

  echo "[stage_relay_base] ${label}: ready"
}

main() {
  parse_args "$@"
  [[ "${CLOUDSDK_CORE_PROJECT:-$ALLOWED_PROJECT}" == "$ALLOWED_PROJECT" ]] \
    || die "CLOUDSDK_CORE_PROJECT must be $ALLOWED_PROJECT"

  local sessions=()
  if [[ -n "$CLI_SESSION" ]]; then
    sessions=("$CLI_SESSION")
  else
    mapfile -t sessions < <(ls "$CLI_POOL"/*.env 2>/dev/null || true)
    [[ ${#sessions[@]} -gt 0 ]] || die "no *.env session files in $CLI_POOL"
  fi

  resolve_sources

  local specs=()
  local layer stamp tarball
  for layer in "${LAYERS[@]}"; do
    if [[ -z "$(layer_members "$layer")" ]]; then
      log "${layer} layer is empty, skipping"
      continue
    fi
    stamp="$(layer_stamp "$layer")"
    tarball="$(build_layer "$layer" "$stamp")"
    specs+=("${layer}:${stamp}:${tarball}")
    log "${layer} stamp ${stamp:0:12}"
  done
  [[ ${#specs[@]} -gt 0 ]] || die "nothing to stage; build the test targets first"

  # Naming the base directory after both stamps is what keeps concurrent runs
  # apart. Same content, same directory, nothing to re-push; different content,
  # different directory, and neither run can overwrite the other's libraries.
  local base_key
  base_key=$(printf '%s\n' "${specs[@]}" | cut -d: -f1,2 | sort | sha256sum | cut -c1-16)
  local base_dir="${REMOTE_RELAY_DIR}/base-${base_key}"
  if [[ -n "$CLI_EMIT_BASE_DIR" ]]; then
    printf '%s\n' "$base_dir" > "$CLI_EMIT_BASE_DIR"
  fi

  local running=0
  local pids=()
  for session in "${sessions[@]}"; do
    stage_one "$session" "$base_dir" "${specs[@]}" &
    pids+=($!)
    running=$(( running + 1 ))
    if (( running >= CLI_JOBS )); then
      wait -n || true
      running=$(( running - 1 ))
    fi
  done

  local failures=0
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || failures=$(( failures + 1 ))
  done

  log "staged $(( ${#sessions[@]} - failures ))/${#sessions[@]} VM(s) at ${base_dir}"
  [[ "$failures" -eq 0 ]]
}

main "$@"
