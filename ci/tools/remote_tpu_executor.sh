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
# ci/tools/remote_tpu_executor.sh
# Remote execution engine running on physical Cloud TPU v5e VM silicon.
set -uo pipefail
set -m  # Enable job control for dedicated process group isolation

readonly ALLOWED_PROJECT="rbe-tpu-oss"

# Enforce GCP project boundary strictly to ALLOWED_PROJECT
enforce_project_boundary() {
  local target_project="${1:-$ALLOWED_PROJECT}"
  if [[ "$target_project" != "$ALLOWED_PROJECT" ]]; then
    echo "ERROR [remote_tpu_executor]: Project boundary violation: '$target_project' is not allowed." >&2
    echo "This executor is strictly restricted to project '$ALLOWED_PROJECT'." >&2
    exit 1
  fi

  local env_vars=("CLOUDSDK_CORE_PROJECT" "GOOGLE_CLOUD_PROJECT" "GCP_PROJECT" "GCLOUD_PROJECT" "TPU_PROJECT")
  for var in "${env_vars[@]}"; do
    if [[ -n "${!var:-}" && "${!var}" != "$ALLOWED_PROJECT" ]]; then
      echo "ERROR [remote_tpu_executor]: Project boundary violation: Environment variable $var is set to '${!var}' (violates allowed project '$ALLOWED_PROJECT')." >&2
      exit 1
    fi
  done
}

# Initial boundary check across environment variables
enforce_project_boundary "${TPU_PROJECT:-$ALLOWED_PROJECT}"

# relay_test_runner.sh invokes this over SSH as:
#   remote_tpu_executor.sh <sandbox_dir> <test_binary> [-- test_args...]
readonly SANDBOX_DIR="${1:-}"
readonly TEST_BIN_REL="${2:-}"

if [[ -z "$SANDBOX_DIR" || -z "$TEST_BIN_REL" ]]; then
  echo "ERROR [remote_tpu_executor]: Missing required arguments." >&2
  echo "Usage: $0 <sandbox_dir> <test_binary> [-- test_args...]" >&2
  exit 1
fi

shift 2
[[ "${1:-}" == "--" ]] && shift
declare -a TEST_ARGS=("$@")

mkdir -p "$SANDBOX_DIR"

# Phase timing, off unless TORCH_TPU_RELAY_TIMING is set. This is the VM half
# of the same instrumentation relay_test_runner.sh prints on the host.
EXEC_T0="${EPOCHREALTIME/,/.}"
EXEC_TPREV="$EXEC_T0"
vmark() {
  [[ -n "${TORCH_TPU_RELAY_TIMING:-}" ]] || return 0
  local now="${EPOCHREALTIME/,/.}"
  printf '[vm-timing] %-16s +%6.2fs  total %6.2fs\n' \
    "$1" \
    "$(awk -v a="$now" -v b="$EXEC_TPREV" 'BEGIN{printf "%.2f", a-b}')" \
    "$(awk -v a="$now" -v b="$EXEC_T0" 'BEGIN{printf "%.2f", a-b}')" >&2
  EXEC_TPREV="$now"
}

BASE_CACHE="${TORCH_TPU_BASE_CACHE:-/tmp/torch_tpu_relay/base}"
PAYLOAD_CACHE="${TORCH_TPU_PAYLOAD_CACHE:-/tmp/torch_tpu_relay/payloads}"
PAYLOAD_KEY="${TORCH_TPU_PAYLOAD_KEY:-}"
CACHED_TREE="${PAYLOAD_CACHE}/${PAYLOAD_KEY}"

workspace_root_for() {
  local root="$1/_main"
  [[ -d "$root" ]] || root="$1"
  printf '%s' "$root"
}

# 1. Get a prepared runfiles tree into the sandbox.
#
# Every shard of a target ships the same tree, and the big targets have 30-50
# shards each. Preparing it once per run and hardlinking the rest turns ~20s of
# unpack-and-relink into well under a second. The key is scoped to one presubmit
# run, so a rebuilt tree can never be served out of a stale cache.
if [[ -n "$PAYLOAD_KEY" && -d "$CACHED_TREE" ]]; then
  cp -al "${CACHED_TREE}/." "${SANDBOX_DIR}/" || {
    echo "ERROR [remote_tpu_executor]: Could not clone the cached payload ${PAYLOAD_KEY}." >&2
    exit 1
  }
  WORKSPACE_ROOT="$(workspace_root_for "$SANDBOX_DIR")"
  vmark tree_cache_hit
else
  if [[ ! -t 0 ]]; then
    # -i because relay_test_runner.sh sends several archives back to back.
    if ! tar -xzif - -C "$SANDBOX_DIR"; then
      echo "ERROR [remote_tpu_executor]: Could not unpack the test payload into ${SANDBOX_DIR}." >&2
      exit 1
    fi
  fi

  WORKSPACE_ROOT="$(workspace_root_for "$SANDBOX_DIR")"
  vmark tree_unpacked

  # 2. Wire in the shared runfiles that ci/tools/stage_relay_base.sh pushed. The
  # payload leaves these out because they are the same couple of gigabytes for
  # every test: the hermetic CPython, the pip wheels, and the shared C++ libs.
  # The venv symlinks inside the payload are relative to the sandbox root, so
  # linking the repositories in at that level is all it takes to resolve them.
  for dep in "$BASE_CACHE"/rules_python*; do
    [[ -d "$dep" ]] || continue
    ln -sfn "$dep" "${SANDBOX_DIR}/$(basename "$dep")"
  done
  if [[ -d "${BASE_CACHE}/_solib_x86_64" ]]; then
    mkdir -p "$WORKSPACE_ROOT"
    ln -sfn "${BASE_CACHE}/_solib_x86_64" "${WORKSPACE_ROOT}/_solib_x86_64"
  fi
  # Same deal for this repo's extension modules; the payload leaves the whole
  # csrc directory out because every test loads the identical build of it.
  if [[ -d "${BASE_CACHE}/csrc" ]]; then
    mkdir -p "$WORKSPACE_ROOT"
    ln -sfn "${BASE_CACHE}/csrc" "${WORKSPACE_ROOT}/csrc"
  fi

  # The venv arrives with links still aimed at absolute paths in the host's
  # output base. Any of them naming a repository staged here gets repointed at
  # the base cache; sending the real bytes instead would put 1.7 GB on the wire
  # for every test. Aiming at BASE_CACHE rather than at the sandbox is what lets
  # the prepared tree be cloned into a differently named sandbox.
  repaired=0
  while IFS= read -r link; do
    target="$(readlink "$link")"
    [[ "$target" == */external/* ]] || continue
    repo_rest="${target#*/external/}"
    [[ -e "${BASE_CACHE}/${repo_rest%%/*}" ]] || continue
    ln -sfn "${BASE_CACHE}/${repo_rest}" "$link"
    repaired=$(( repaired + 1 ))
  done < <(find "$SANDBOX_DIR" -xtype l 2>/dev/null)
  [[ "$repaired" -eq 0 ]] || echo "[remote_tpu_executor] Repointed ${repaired} venv link(s) at the base cache." >&2
  vmark links_repaired

  # Publish before the test runs, so nothing the test writes lands in the cache.
  if [[ -n "$PAYLOAD_KEY" ]]; then
    mkdir -p "$PAYLOAD_CACHE"
    staging="${PAYLOAD_CACHE}/.staging.$$"
    rm -rf "$staging"
    if cp -al "$SANDBOX_DIR" "$staging" 2>/dev/null; then
      mv -T "$staging" "$CACHED_TREE" 2>/dev/null || rm -rf "$staging"
    else
      rm -rf "$staging"
    fi
    # Trees from earlier runs can never be reused, so reclaim the space.
    for old in "$PAYLOAD_CACHE"/*; do
      [[ -d "$old" ]] || continue
      case "$(basename "$old")" in
        "${PAYLOAD_KEY%%_*}_"*) ;;
        *) rm -rf "$old" ;;
      esac
    done
    vmark payload_published
  fi
fi

# Bazel built the test against this interpreter, so its C extensions only load
# under this one. The image's own python3 is a different minor version.
PYTHON_BIN="$(ls -d "${BASE_CACHE}"/rules_python*python_3*/bin/python3 2>/dev/null | head -n 1)"
if [[ ! -x "${PYTHON_BIN:-}" ]]; then
  echo "WARNING [remote_tpu_executor]: no hermetic interpreter in ${BASE_CACHE}; falling back to the image's python3." >&2
  PYTHON_BIN="$(command -v python3 || true)"
fi


# 3. Configure hardware environment for Cloud TPU v5e (v5litepod-1)
export TPU_VISIBLE_DEVICES="${TPU_VISIBLE_DEVICES:-0}"
export TPU_VISIBLE_CHIPS="${TPU_VISIBLE_CHIPS:-0}"
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v5litepod-1}"
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-1,1,1}"
export TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-1}"
export ALLOW_MULTIPLE_LIBTPU_LOAD=true

# The libtpu wheel bazel resolved is part of the staged base cache. Falling
# back to the image's own copy would silently pair a different runtime with the
# torch build under test.
if [[ -z "${TPU_LIBRARY_PATH:-}" ]]; then
  _libtpu="$(ls -d "${BASE_CACHE}"/rules_python*libtpu/site-packages/libtpu/libtpu.so 2>/dev/null | head -n 1)"
  if [[ -f "${_libtpu:-}" ]]; then
    export TPU_LIBRARY_PATH="$_libtpu"
  elif [[ -f "/lib/libtpu.so" ]]; then
    export TPU_LIBRARY_PATH="/lib/libtpu.so"
  fi
fi

# 4. Set process and library search paths
export PYTHONUNBUFFERED=1
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:${PATH:-}"
export PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${WORKSPACE_ROOT}/_solib_x86_64:${LD_LIBRARY_PATH:-}"
export XML_OUTPUT_FILE="${SANDBOX_DIR}/test.xml"
export TEST_SRCDIR="${SANDBOX_DIR}"
export RUNFILES_DIR="${SANDBOX_DIR}"
export TEST_WORKSPACE="_main"
export TEST_TMPDIR="${SANDBOX_DIR}/tmp"
mkdir -p "${SANDBOX_DIR}/tmp"

# relay_test_runner.sh passes TEST_SHARD_INDEX and TEST_TOTAL_SHARDS in as
# command prefixes, so they are already in this process's environment. The
# status file has to be repointed though: bazel's value names a path on the
# host, and the test binary writes it here.
[[ -z "${TEST_TOTAL_SHARDS:-}" ]] || export TEST_SHARD_STATUS_FILE="${SANDBOX_DIR}/shard_status"

# 5. Clear stale accelerator device locks before launch
if [[ -d "/dev/vfio" ]] && command -v fuser >/dev/null 2>&1; then
  fuser -k /dev/vfio/* 2>/dev/null || true
fi

# State tracking for supervision and cleanup
TEST_PID=""
TEST_PGID=""
TEST_EXIT_CODE=0
CLEANUP_DONE=0
TARGET_NAME="${TEST_TARGET:-$(basename "$TEST_BIN_REL")}"
START_TIME_MS=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))' 2>/dev/null || echo "0")

synthesize_fallback_xml() {
  local code="$1"
  local msg="${2:-Test exited with code $code}"
  local xml_path="${SANDBOX_DIR}/test.xml"

  local duration_s="0.000"
  if [[ "$START_TIME_MS" != "0" ]]; then
    local now_ms
    now_ms=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))' 2>/dev/null || echo "0")
    if [[ "$now_ms" != "0" && "$now_ms" -ge "$START_TIME_MS" ]]; then
      duration_s=$(awk -v s="$START_TIME_MS" -v e="$now_ms" 'BEGIN { printf "%.3f", (e - s) / 1000 }' 2>/dev/null || echo "0.000")
    fi
  fi

  local failures=0
  local errors=0
  local failure_tag=""
  if [[ "$code" -ne 0 ]]; then
    failures=1
    failure_tag="<failure message=\"${msg}\"><![CDATA[Process failed with exit code ${code}. Check test.log for stderr details.]]></failure>"
  fi

  cat <<XML_EOF > "$xml_path"
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="${TARGET_NAME}" tests="1" failures="${failures}" errors="${errors}" time="${duration_s}">
    <testcase name="execution" classname="${TARGET_NAME}" time="${duration_s}">
      ${failure_tag}
    </testcase>
  </testsuite>
</testsuites>
XML_EOF
}

cleanup() {
  local trap_sig="${1:-EXIT}"
  local origin_rc="${2:-0}"
  if [[ "$CLEANUP_DONE" -eq 1 ]]; then
    return
  fi
  CLEANUP_DONE=1
  trap - EXIT INT TERM HUP

  case "$trap_sig" in
    INT)  TEST_EXIT_CODE=130 ;;
    TERM) TEST_EXIT_CODE=143 ;;
    HUP)  TEST_EXIT_CODE=129 ;;
    EXIT)
      if [[ "$TEST_EXIT_CODE" -eq 0 && "$origin_rc" -ne 0 ]]; then
        TEST_EXIT_CODE="$origin_rc"
      fi
      ;;
    *)    TEST_EXIT_CODE=1 ;;
  esac

  # Signal traps (EXIT INT TERM HUP) executing process group reaping
  if [[ -n "${TEST_PGID:-}" && "$TEST_PGID" -gt 1 ]]; then
    if kill -0 -"$TEST_PGID" 2>/dev/null; then
      kill -TERM -"$TEST_PGID" 2>/dev/null || true
      sleep 0.2
      kill -KILL -"$TEST_PGID" 2>/dev/null || true
      sleep 0.3
    fi
  fi

  # Free any dangling hardware file descriptor locks on /dev/vfio/*
  if [[ -d "/dev/vfio" ]] && command -v fuser >/dev/null 2>&1; then
    fuser -k /dev/vfio/* 2>/dev/null || true
  fi

  # Record exit code for host runner retrieval and disambiguation
  echo "$TEST_EXIT_CODE" > "${SANDBOX_DIR}/test.exitcode"

  # Ensure JUnit XML exists; synthesize fallback if missing or empty
  if [[ ! -s "${SANDBOX_DIR}/test.xml" ]]; then
    synthesize_fallback_xml "$TEST_EXIT_CODE" "Process terminated with exit code ${TEST_EXIT_CODE} (${trap_sig})"
  fi

  # Update latest pointer symlinks
  mkdir -p /tmp/torch_tpu_relay/sandboxes 2>/dev/null || true
  ln -sfn "${SANDBOX_DIR}/test.xml" "/tmp/torch_tpu_relay/sandboxes/$(basename "$SANDBOX_DIR")_latest.xml" 2>/dev/null || true
  ln -sfn "${SANDBOX_DIR}/test.xml" "/tmp/torch_tpu_relay/latest_test.xml" 2>/dev/null || true

  exit "$TEST_EXIT_CODE"
}

trap 'cleanup EXIT $?' EXIT
trap 'cleanup INT 130' INT
trap 'cleanup TERM 143' TERM
trap 'cleanup HUP 129' HUP

# 6. Resolve test command to execute
CLEAN_BIN="${TEST_BIN_REL#./}"
BIN_BASE="$(basename "$CLEAN_BIN")"
STAGE2_BOOTSTRAP="$(find "${WORKSPACE_ROOT}" -name "_${BIN_BASE}_stage2_bootstrap.py" 2>/dev/null | head -n 1 || true)"

# Bazel leaves the test binary in a different shape depending on the rule that
# built it. The py launcher comes first: it points itself at the venv and the
# hermetic interpreter, which now resolve because the base cache is linked in.
if [[ -x "${WORKSPACE_ROOT}/${CLEAN_BIN}" && "$CLEAN_BIN" != *.py ]]; then
  TEST_PATH="${WORKSPACE_ROOT}/${CLEAN_BIN}"
elif [[ -f "${WORKSPACE_ROOT}/${CLEAN_BIN}.py" ]]; then
  TEST_PATH="${WORKSPACE_ROOT}/${CLEAN_BIN}.py"
elif [[ -f "${WORKSPACE_ROOT}/${CLEAN_BIN}" && "$CLEAN_BIN" == *.py ]]; then
  TEST_PATH="${WORKSPACE_ROOT}/${CLEAN_BIN}"
elif [[ -f "$STAGE2_BOOTSTRAP" ]]; then
  TEST_PATH="$STAGE2_BOOTSTRAP"
elif [[ -x "${SANDBOX_DIR}/${CLEAN_BIN}" ]]; then
  TEST_PATH="${SANDBOX_DIR}/${CLEAN_BIN}"
elif [[ -f "${SANDBOX_DIR}/${CLEAN_BIN}" && "$CLEAN_BIN" == *.py ]]; then
  TEST_PATH="${SANDBOX_DIR}/${CLEAN_BIN}"
else
  TEST_PATH="$(find "$WORKSPACE_ROOT" \( -name "$BIN_BASE" -o -name "${BIN_BASE}.py" \) 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$TEST_PATH" ]]; then
  echo "ERROR [remote_tpu_executor]: Test executable '${CLEAN_BIN}' not found in sandbox." >&2
  exit 127
fi


# Anything that isn't already an executable needs the interpreter in front.
declare -a EXEC_CMD=()
[[ -x "$TEST_PATH" && "$TEST_PATH" != *.py ]] || EXEC_CMD=("$PYTHON_BIN")
EXEC_CMD+=("$TEST_PATH" ${TEST_ARGS[@]+"${TEST_ARGS[@]}"})

vmark launching_test

# 7. Execute test binary under job control (set -m) in dedicated process group ($PGID)
(
  cd "$WORKSPACE_ROOT"
  exec "${EXEC_CMD[@]}" < /dev/null
) &
TEST_PID=$!
TEST_PGID=$!

wait "$TEST_PID" 2>/dev/null
TEST_EXIT_CODE=$?
TEST_PID=""
vmark test_process_done

exit "$TEST_EXIT_CODE"
