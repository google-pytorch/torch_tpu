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
# Coordinates access to accelerator TPUs between concurrent Bazel tests.
#
# Assigns each test an accelerator and ensures that tests are evenly distributed
# between accelerators.
#
# Example usage:
# bazel test --run_under=/path/to/parallel_accelerator_execute.sh //tests/...
#
# Environment variables:
#     TORCH_TPU_ACCELERATOR_COUNT =
#         Number of accelerator TPUs available.
#     TORCH_TPU_TESTS_PER_ACCELERATOR =
#         Number of test runs per accelerator TPUs available.

TORCH_TPU_ACCELERATOR_COUNT=${TORCH_TPU_ACCELERATOR_COUNT:-8}
TORCH_TPU_TESTS_PER_ACCELERATOR=${TORCH_TPU_TESTS_PER_ACCELERATOR:-1}

# rlocation is needed to find the test binary when running with bazel 8+
set -uo pipefail
f=bazel_tools/tools/bash/runfiles/runfiles.bash
# Source the runfiles library. We use a flexible grep to handle both WORKSPACE and Bzlmod prefixes.
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

# Enable job control so spawned test processes receive a distinct Process Group (PGID == TEST_PID).
set -m

TEST_PID=""
TEST_PGID=""

# Terminate the test's entire process group before releasing locks.
# This ensures that any surviving child/grandchild workers (e.g. torch.multiprocessing workers)
# are terminated even if the parent test process has already exited or crashed.
cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  if [ -n "${TEST_PGID:-}" ]; then
    # Check if any process in the group is still alive
    if kill -0 -"$TEST_PGID" 2>/dev/null; then
      kill -TERM -"$TEST_PGID" 2>/dev/null || true
      # Grace period for graceful shutdown / flushing logs
      for _ in $(seq 1 10); do
        kill -0 -"$TEST_PGID" 2>/dev/null || break
        sleep 0.1
      done
      # Force-kill surviving stragglers
      kill -KILL -"$TEST_PGID" 2>/dev/null || true
      # Allow kernel TPU/VFIO drivers to finish unmapping device buffers
      sleep 0.5
    fi
  fi
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

run_test() {
  "$TEST_BINARY" "$@" &
  TEST_PID=$!
  TEST_PGID=$!
  wait "$TEST_PID"
  local rc=$?
  TEST_PID=""  # Direct binary exited; keep TEST_PGID so cleanup can purge orphaned stragglers
  return "$rc"
}

# Multi-accelerator/distributed tests require exclusive access across all slots.
if [ "${TORCH_TPU_EXCLUSIVE_TEST:-0}" = "1" ]; then
  echo "Acquiring ALL accelerator locks for exclusive test $TEST_BINARY..."
  for j in $(seq 0 $((TORCH_TPU_TESTS_PER_ACCELERATOR-1))); do
    for i in $(seq 0 $((TORCH_TPU_ACCELERATOR_COUNT-1))); do
      # Use >> to prevent lock file truncation race conditions
      exec {fd}>>"/var/lock/torch_tpu_accelerator_lock_${i}_${j}" || exit 1
      flock "$fd" || exit 1
    done
  done

  export TPU_VISIBLE_CHIPS="$(seq -s, 0 $((TORCH_TPU_ACCELERATOR_COUNT-1)))"
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
      # Use >> to prevent file truncation.
      exec {lock_fd}>>"/var/lock/torch_tpu_accelerator_lock_${i}_${j}" || exit 1
      if flock -n "$lock_fd"; then
        export TPU_VISIBLE_CHIPS=$i
        echo "Running test $TEST_BINARY $* on accelerator $i"
        run_test "$@"
        exit $?
      else
        # Close file descriptor immediately to prevent FD leak during polling.
        exec {lock_fd}>&-
      fi
    done
  done
  echo "All accelerator slots are busy, retrying in 1s..."
  sleep 1
done