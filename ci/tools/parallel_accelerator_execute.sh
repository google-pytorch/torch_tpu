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
# ==============================================================================
#
# Coordinates access to TPU accelerators between concurrent Bazel tests.
#
# Assigns each test an accelerator and ensures that tests are evenly distributed
# between available devices.
#
# Example usage:
# bazel test --run_under=/path/to/parallel_accelerator_execute.sh //tests/...
#
# Environment variables:
#     TORCH_TPU_ACCELERATOR_COUNT =
#         Number of TPU accelerators available.
#     TORCH_TPU_TESTS_PER_ACCELERATOR =
#         Number of concurrent test runs permitted per TPU accelerator.

TORCH_TPU_ACCELERATOR_COUNT=${TORCH_TPU_ACCELERATOR_COUNT:-8}
TORCH_TPU_TESTS_PER_ACCELERATOR=${TORCH_TPU_TESTS_PER_ACCELERATOR:-1}

# rlocation is needed to find the test binary when running with Bazel 8+
set -uo pipefail
f=bazel_tools/tools/bash/runfiles/runfiles.bash
# Source the runfiles library. Use a flexible grep to handle both WORKSPACE and
# Bzlmod prefixes.
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -m1 "$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || true

if command -v rlocation >/dev/null; then
  TEST_BINARY="$(rlocation "${TEST_WORKSPACE:-_main}/${1#./}")"
else
  TEST_BINARY="$1"
fi
shift; set +uo pipefail

mkdir -p /var/lock

# Enable job control so spawned test processes receive a distinct Process Group
# (PGID == TEST_PID).
set -m

TEST_PID=""
TEST_PGID=""
LOCK_FDS=()

# Terminate the test's entire process group before releasing accelerator locks.
#
# 1. Process Leaks (Orphaned Worker Cleanup):
#    Multi-process tests (e.g., spawned via torch.multiprocessing.spawn) can
#    leave behind orphaned worker processes if the main test process exits or
#    crashes. Because flock locks are bound to the lifetime of the wrapper
#    script's file descriptors, exiting the wrapper immediately frees the lock
#    while lingering background workers may still hold open hardware device
#    handles (/dev/vfio/*, /dev/accel*). This causes device contention and state
#    collisions when the next test claims the freed slot.
#
#    Standard PID-tree walking (e.g. pgrep -P $$) misses these processes once
#    the immediate parent dies and grandchildren are reparented to PID 1.
#    Enabling job control (set -m) assigns test executions their own dedicated
#    Process Group (PGID == TEST_PID), allowing cleanup() to signal and drain
#    the entire process tree (-$TEST_PGID) before any hardware locks are
#    surrendered.
#
# 2. File Descriptor Exhaustion:
#    See the polling loop below: failed flock attempts should explicitly close
#    their file descriptors (exec {lock_fd}>&-) on each iteration. Otherwise,
#    continuous retries under lock contention leak open file descriptors until
#    the process hits system/ulimit FD thresholds. In addition, acquired lock FDs
#    are closed in child subshells before exec to avoid inheriting locks into
#    spawned workers.
cleanup() {
  local exit_code=$?
  local sig="${1:-}"
  [ -n "$sig" ] && exit_code="$sig"
  trap - EXIT INT TERM

  if [ -n "${TEST_PGID:-}" ] && [ "$TEST_PGID" -gt 1 ]; then
    # Check if any process in the group is still alive.
    if kill -0 -"$TEST_PGID" 2>/dev/null; then
      kill -TERM -"$TEST_PGID" 2>/dev/null || true
      # Grace period for shutdown and log flushing.
      for _ in $(seq 1 10); do
        kill -0 -"$TEST_PGID" 2>/dev/null || break
        sleep 0.1
      done
      # Force-kill surviving stragglers.
      kill -KILL -"$TEST_PGID" 2>/dev/null || true
    fi
    # Allow kernel TPU/VFIO drivers to finish unmapping device buffers.
    sleep 0.5
  fi
  exit "$exit_code"
}

trap cleanup EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

run_test() {
  (
    for fd in "${LOCK_FDS[@]}"; do
      eval "exec $fd>&-"
    done
    exec "$TEST_BINARY" "$@"
  ) &
  TEST_PID=$!
  TEST_PGID=$!
  wait "$TEST_PID"
  local rc=$?
  TEST_PID=""  # Direct binary exited; retain TEST_PGID so cleanup can purge orphaned stragglers.
  return "$rc"
}

# Multi-accelerator/distributed tests require exclusive access across all slots.
if [ "${TORCH_TPU_EXCLUSIVE_TEST:-0}" = "1" ]; then
  echo "Acquiring ALL accelerator locks for exclusive test $TEST_BINARY..."
  LOCK_FDS=()
  for j in $(seq 0 $((TORCH_TPU_TESTS_PER_ACCELERATOR-1))); do
    for i in $(seq 0 $((TORCH_TPU_ACCELERATOR_COUNT-1))); do
      # Use >> to prevent lock file truncation race conditions.
      exec {fd}>>"/var/lock/torch_tpu_accelerator_lock_${i}_${j}" || exit 1
      flock "$fd" || exit 1
      LOCK_FDS+=("$fd")
    done
  done

  export TPU_VISIBLE_DEVICES="$(seq -s, 0 $((TORCH_TPU_ACCELERATOR_COUNT-1)))"
  export TPU_VISIBLE_CHIPS="$TPU_VISIBLE_DEVICES"
  run_test "$@"
  exit $?
fi

# Try to acquire any of the
# TORCH_TPU_ACCELERATOR_COUNT * TORCH_TPU_TESTS_PER_ACCELERATOR
# slots to run a test at.
#
# Prefer to allocate 1 test per accelerator over multiple tests on 1 accelerator.
# Iterate over TORCH_TPU_TESTS_PER_ACCELERATOR first (j) then accelerators (i).
while true; do
  for j in $(seq 0 $((TORCH_TPU_TESTS_PER_ACCELERATOR-1))); do
    for i in $(seq 0 $((TORCH_TPU_ACCELERATOR_COUNT-1))); do
      # Use >> to prevent file truncation race conditions.
      exec {lock_fd}>>"/var/lock/torch_tpu_accelerator_lock_${i}_${j}" || exit 1
      if flock -n "$lock_fd"; then
        LOCK_FDS=("$lock_fd")
        export TPU_VISIBLE_DEVICES=$i
        export TPU_VISIBLE_CHIPS=$i
        echo "Running test $TEST_BINARY $* on accelerator $i"
        run_test "$@"
        exit $?
      else
        # Close file descriptor immediately to prevent FD leaks during polling.
        exec {lock_fd}>&-
      fi
    done
  done
  echo "All accelerator slots are busy, retrying in 1s..."
  sleep 1
done
