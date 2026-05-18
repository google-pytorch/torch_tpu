#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Drives upstream PyTorch tests against the torch_tpu PrivateUse1 backend
# via the chunked_runner (one pytest invocation per chunk so a single
# crashing test does not blow away the entire run's signal).

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

if [[ -z "${PYTORCH_SRC_DIR:-}" ]]; then
  PIN="$(tr -d '[:space:]' < "$HERE/pytorch_pin.txt")"
  PYTORCH_SRC_DIR="/tmp/pytorch"
  if [[ ! -d "$PYTORCH_SRC_DIR/.git" ]]; then
    echo "Cloning pytorch at $PIN into $PYTORCH_SRC_DIR" >&2
    git clone --depth 1 --filter=blob:none https://github.com/pytorch/pytorch.git "$PYTORCH_SRC_DIR"
    if [[ "$PIN" != "main" ]]; then
      git -C "$PYTORCH_SRC_DIR" fetch --depth 1 origin "$PIN"
      git -C "$PYTORCH_SRC_DIR" checkout "$PIN"
    fi
  fi
  export PYTORCH_SRC_DIR
fi

export PYTHONPATH="$HERE:${PYTHONPATH:-}"

# Suite selection. UPSTREAM_SUITE picks a list of pytest targets from
# tests/upstream/test_suites.py; defaults to the suite's DEFAULT_SUITE.
SUITE="${UPSTREAM_SUITE:-}"
if [[ -z "$SUITE" ]]; then
  SUITE="$(python -c 'from test_suites import DEFAULT_SUITE; print(DEFAULT_SUITE)')"
fi
echo "Using upstream suite: $SUITE"

mapfile -t SUITE_TARGETS < <(
  python -c "from test_suites import get_suite; print('\n'.join(get_suite('$SUITE')))"
)

# Prefix each suite-relative target with ${PYTORCH_SRC_DIR}/test/.
TESTS=()
for t in "${SUITE_TARGETS[@]}"; do
  TESTS+=("${PYTORCH_SRC_DIR}/test/${t}")
done

TEST_FILTER="${UPSTREAM_K_FILTER:-test_dtypes}"
CHUNK_SIZE="${UPSTREAM_CHUNK_SIZE:-5}"

# Run pytest from /tmp so the installed torch_tpu wheel wins over the
# repo-root torch_tpu/ source directory (which lacks compiled .so files).
cd /tmp

JUNIT_OUT="${UPSTREAM_JUNIT_OUT:-/tmp/upstream_results.xml}"

TEST_TARGET_ARGS=()
for t in "${TESTS[@]}"; do
  TEST_TARGET_ARGS+=(--test-target "$t")
done

python "$HERE/chunked_runner.py" \
  --chunk-size "$CHUNK_SIZE" \
  --junit-out "$JUNIT_OUT" \
  "${TEST_TARGET_ARGS[@]}" \
  -- \
  -k "$TEST_FILTER" \
  -p upstream_plugin \
  -p no:pytestshardplugin \
  "$@"
