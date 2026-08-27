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
# Coordinates access to accelerators TPUs between concurrent Bazel tests.
#
# Assigns each test an accelerator and ensures that tests are evenly distributed
# between accelerators.
#
# Example usage:
# bazel test --run_under=/path/to/parallel_accelerator_execute.sh //tests/...
#
#
# Environment variables:
#     TORCH_TPU_ACCELERATOR_COUNT =
#         Number of accelerators TPUs available.
#     TORCH_TPU_TESTS_PER_ACCELERATOR =
#         Number of test runs per accelerators TPUs available.

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


# *******************************************************************

mkdir -p /var/lock

# Enable job control so background jobs get their own Process Group (PGID == TEST_PID).
set -m

TEST_PID=""

# Terminate the test's process group before releasing locks.
# This ensures all children and grandchildren (e.g. torch.multiprocessing workers)
# are terminated even if they were re-parented to PID 1.
cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [ -n "$TEST_PID" ]; then
    kill -TERM -"$TEST_PID" 2>/dev/null || true
    kill -KILL -"$TEST_PID" 2>/dev/null || true
  fi
  exit $exit_code
}
trap cleanup EXIT INT TERM

# Multi-accelerator/distributed tests require exclusive access across all slots.
if [ "${TORCH_TPU_EXCLUSIVE_TEST:-0}" = "1" ]; then
  echo "Acquiring ALL accelerator locks for exclusive test $TEST_BINARY..."
  for j in $(seq 0 $((TORCH_TPU_TESTS_PER_ACCELERATOR-1))); do
    for i in $(seq 0 $((TORCH_TPU_ACCELERATOR_COUNT-1))); do
      exec {fd}>"/var/lock/torch_tpu_accelerator_lock_${i}_${j}" || exit 1
      flock "$fd" || exit 1
    done
  done
  export TPU_VISIBLE_CHIPS="$(seq -s, 0 $((TORCH_TPU_ACCELERATOR_COUNT-1)))"
  "$TEST_BINARY" "$@" &
  TEST_PID=$!
  wait "$TEST_PID"
  return_code=$?
  exit $return_code
fi

# Try to acquire any of the
# TORCH_TPU_ACCELERATOR_COUNT * TORCH_TPU_TESTS_PER_ACCELERATOR
# slots to run a test at.
#
# Prefer to allocate 1 test per accelerator over 4 tests on 1 accelerator
# So, we iterate over TORCH_TPU_TESTS_PER_ACCELERATOR
# first (j) then accelerators (i).
#
# If all slots are busy, we loop and retry until one is free.
while true; do
  for j in `seq 0 $((TORCH_TPU_TESTS_PER_ACCELERATOR-1))`; do
    for i in `seq 0 $((TORCH_TPU_ACCELERATOR_COUNT-1))`; do
      exec {lock_fd}>"/var/lock/torch_tpu_accelerator_lock_${i}_${j}" || exit 1
      if flock -n "$lock_fd"; then
        export TPU_VISIBLE_CHIPS=$i
        echo "Running test $TEST_BINARY $* on accelerator $i"
        "$TEST_BINARY" "$@" &
        TEST_PID=$!
        wait "$TEST_PID"
        return_code=$?
        exit $return_code
      else
        # Close file descriptor immediately to prevent FD leak during polling.
        exec {lock_fd}>&-
      fi
    done
  done
  echo "All accelerator slots are busy, retrying in 1s..."
  sleep 1
done