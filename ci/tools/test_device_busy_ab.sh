#!/bin/bash
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

# Reproduction script for TPU Wrapper A/B testing (Process Cleanup & FD Leaks).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER_NEW="${SCRIPT_DIR}/parallel_accelerator_execute.sh"
WRAPPER_OLD="${SCRIPT_DIR}/old_wrapper.sh"

if [ ! -f "$WRAPPER_NEW" ]; then
  echo "Error: Cannot find $WRAPPER_NEW"
  exit 1
fi
chmod +x "$WRAPPER_NEW"

# 1. Generate OLD wrapper statically (original version with pkill -P and without FD closing)
cat << 'EOF' > "$WRAPPER_OLD"
#!/usr/bin/env bash
TORCH_TPU_ACCELERATOR_COUNT=${TORCH_TPU_ACCELERATOR_COUNT:-8}
TORCH_TPU_TESTS_PER_ACCELERATOR=${TORCH_TPU_TESTS_PER_ACCELERATOR:-1}

set -uo pipefail
f=bazel_tools/tools/bash/runfiles/runfiles.bash
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

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  pkill -P $$ 2>/dev/null || true
  exit $exit_code
}
trap cleanup EXIT INT TERM

if [ "${TORCH_TPU_EXCLUSIVE_TEST:-0}" = "1" ]; then
  echo "Acquiring ALL accelerator locks for exclusive test $TEST_BINARY..."
  for j in $(seq 0 $((TORCH_TPU_TESTS_PER_ACCELERATOR-1))); do
    for i in $(seq 0 $((TORCH_TPU_ACCELERATOR_COUNT-1))); do
      exec {fd}>"/var/lock/torch_tpu_accelerator_lock_${i}_${j}" || exit 1
      flock "$fd" || exit 1
    done
  done
  (
    export TPU_VISIBLE_CHIPS="$(seq -s, 0 $((TORCH_TPU_ACCELERATOR_COUNT-1)))"
    "$TEST_BINARY" "$@"
  )
  return_code=$?
  exit $return_code
fi

while true; do
  for j in $(seq 0 $((TORCH_TPU_TESTS_PER_ACCELERATOR-1))); do
    for i in $(seq 0 $((TORCH_TPU_ACCELERATOR_COUNT-1))); do
      exec {lock_fd}>"/var/lock/torch_tpu_accelerator_lock_${i}_${j}" || exit 1
      if flock -n "$lock_fd"; then
        (
          export TPU_VISIBLE_CHIPS=$i
          echo "Running test $TEST_BINARY $* on accelerator $i"
          "$TEST_BINARY" "$@"
        )
        return_code=$?
        exit $return_code
      fi
    done
  done
  echo "All accelerator slots are busy, retrying in 1s..."
  sleep 1
done
EOF
chmod +x "$WRAPPER_OLD"

# ==============================================================================
# TEST 1: Grandchild Process Cleanup A/B Test
# ==============================================================================
test_grandchild_cleanup() {
  local wrapper="$1"
  local label="$2"
  local marker="ab_test_token_${RANDOM}_$$"

  echo "=========================================================="
  echo "Testing Grandchild Cleanup: $label"
  echo "=========================================================="

  # Launch parent via wrapper; parent spawns a background grandchild and exits immediately.
  # Using python with $marker as extra argument so it sleeps 60s without invalid argument error.
  "$wrapper" python3 -c "
import subprocess, time
p = subprocess.Popen(['python3', '-c', 'import time; time.sleep(60)', '$marker'])
time.sleep(0.5)
" || true

  # Allow wrapper cleanup trap to complete
  sleep 1.5

  if pgrep -f "$marker" > /dev/null; then
    echo "[$label Result]: Grandchild survived (LEAKED / ORPHANED)."
    pkill -f "$marker" 2>/dev/null || true
    return 1
  else
    echo "[$label Result]: Grandchild was cleanly terminated."
    return 0
  fi
}

echo ""
echo ">>> [TEST 1-A] Running Grandchild Cleanup on OLD WRAPPER (Expect Failure)..."
status_1a=0
test_grandchild_cleanup "$WRAPPER_OLD" "OLD WRAPPER (pkill -P $$)" || status_1a=$?

if [ $status_1a -ne 0 ]; then
  echo ">>> [TEST 1-A RESULT]: FAILED as expected (Grandchild leaked)."
else
  echo ">>> [TEST 1-A RESULT]: PASSED (Unexpected)."
  exit 1
fi

echo ""
echo ">>> [TEST 1-B] Running Grandchild Cleanup on NEW WRAPPER (Expect Success)..."
status_1b=0
test_grandchild_cleanup "$WRAPPER_NEW" "NEW WRAPPER (set -m + kill -KILL -TEST_PGID)" || status_1b=$?

if [ $status_1b -eq 0 ]; then
  echo ">>> [TEST 1-B RESULT]: PASSED as expected (Grandchild cleanly killed)."
else
  echo ">>> [TEST 1-B RESULT]: FAILED (Unexpected)."
  exit 1
fi

# ==============================================================================
# TEST 2: FD Leak Under Contention A/B Test
# ==============================================================================
test_fd_leak() {
  local wrapper="$1"
  local label="$2"

  echo "=========================================================="
  echo "Testing FD Leaks under Contention: $label"
  echo "=========================================================="

  # Occupy all 8 accelerator locks
  for i in $(seq 0 7); do
    exec {fd}>>"/var/lock/torch_tpu_accelerator_lock_${i}_0"
    flock "$fd"
    eval "OCCUPY_FD_${i}=$fd"
  done

  # Run wrapper in background trying to acquire a slot
  "$wrapper" echo "Acquired slot" &
  local polling_pid=$!

  # Poll for 2.5 seconds (at least 2 rounds of 8 slots = ~16 lock failures)
  sleep 2.5

  local fd_count
  fd_count=$(ls -1 "/proc/$polling_pid/fd" 2>/dev/null | wc -l)
  echo "Open file descriptors during polling: $fd_count"

  # Clean up locks and background process
  kill -9 "$polling_pid" 2>/dev/null || true
  for i in $(seq 0 7); do
    eval "fd=\$OCCUPY_FD_${i}"
    flock -u "$fd"
    exec {fd}>&-
  done

  if [ "$fd_count" -gt 15 ]; then
    echo "[$label Result]: FAILED (FD Leak detected: $fd_count open FDs)."
    return 1
  else
    echo "[$label Result]: PASSED (No FD Leak: $fd_count open FDs)."
    return 0
  fi
}

echo ""
echo ">>> [TEST 2-A] Running FD Leak Test on OLD WRAPPER (Expect Failure)..."
status_2a=0
test_fd_leak "$WRAPPER_OLD" "OLD WRAPPER (No FD close)" || status_2a=$?

if [ $status_2a -ne 0 ]; then
  echo ">>> [TEST 2-A RESULT]: FAILED as expected (FDs leaked during polling)."
else
  echo ">>> [TEST 2-A RESULT]: PASSED (Unexpected)."
  exit 1
fi

echo ""
echo ">>> [TEST 2-B] Running FD Leak Test on NEW WRAPPER (Expect Success)..."
status_2b=0
test_fd_leak "$WRAPPER_NEW" "NEW WRAPPER (exec {lock_fd}>&-)" || status_2b=$?

if [ $status_2b -eq 0 ]; then
  echo ">>> [TEST 2-B RESULT]: PASSED as expected (FDs remained flat)."
else
  echo ">>> [TEST 2-B RESULT]: FAILED (Unexpected)."
  exit 1
fi

echo ""
echo "=========================================================="
echo "ALL A/B TESTS PASSED: Old wrapper failed, new wrapper succeeded!"
echo "=========================================================="
