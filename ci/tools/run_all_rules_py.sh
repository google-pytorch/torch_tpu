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

# Runs every rule in custom_lint_rules.METADATA against every file in
# torch_tpu, writes a grouped report (the format of the Rust run_all_rules.sh
# report), and prints execution timings.
#
# This is a full-tree (`--all`) run, so rules restricted to file operations
# (e.g. CheckCopyrightYearUpdate, which only applies to EDITs) do not run.
#
# Usage:
#   ./run_all_rules_py.sh [--benchmark]
#
# Environment:
#   OUT   Report path (default: /tmp/custom_linter_report.txt).
#   JOBS  Worker processes (default: auto-scale to the CPU count).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCH_TPU="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SHIM_DIR=""
if [[ -f "${TORCH_TPU}/custom_lint_rules.METADATA" ]]; then
  # Monorepo layout: expose torch_tpu.oss as a PEP 420 namespace package so
  # bare python3 does not execute torch_tpu/__init__.py (which imports torch).
  SHIM_DIR="$(mktemp -d)"
  mkdir -p "${SHIM_DIR}/torch_tpu"
  ln -s "${TORCH_TPU}/oss" "${SHIM_DIR}/torch_tpu/oss"
  PYTHONPATH_ROOT="${SHIM_DIR}"
  MODULE="torch_tpu.oss.ci.tools.custom_linter"
else
  # OSS layout: ci/tools -> PYTHONPATH=<repo_root>
  TORCH_TPU="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  PYTHONPATH_ROOT="${TORCH_TPU}"
  MODULE="ci.tools.custom_linter"
fi
export PYTHONPATH="${PYTHONPATH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
OUT="${OUT:-/tmp/custom_linter_report.txt}"
JOBS_ARGS=()
if [[ -n "${JOBS:-}" ]]; then
  JOBS_ARGS=(--jobs "${JOBS}")
fi

TIME_FILE="$(mktemp)"
LOG_FILE="$(mktemp)"
trap 'rm -rf "${TIME_FILE}" "${LOG_FILE}" ${SHIM_DIR:+"${SHIM_DIR}"}' EXIT

/usr/bin/time -q -f "%e %U %S" -o "${TIME_FILE}" \
  python3 -m "${MODULE}" --root "${TORCH_TPU}" --all "${JOBS_ARGS[@]}" \
  --report-file "${OUT}" >"${LOG_FILE}" 2>&1
STATUS=$?

# custom_linter exits with 1 when there are ERROR findings, which is expected
# here; anything else (e.g. 2 for a rules file that fails to load) is fatal.
if [[ "${STATUS}" -ne 0 && "${STATUS}" -ne 1 ]]; then
  cat "${LOG_FILE}" >&2
  echo "ERROR: ${MODULE} failed with exit code ${STATUS}" >&2
  exit "${STATUS}"
fi

read -r REAL USER SYS <"${TIME_FILE}"

head -n 3 "${OUT}" | tail -n 2
grep -E '^INFO: Checked' "${LOG_FILE}"
echo "Elapsed time: REAL=${REAL}s | USER=${USER}s | SYS=${SYS}s"
echo "Report written to ${OUT} (view with: less ${OUT})"

if [[ "${1:-}" == "--benchmark" ]]; then
  echo ""
  echo "=========================================================================================="
  echo " PYTHON CUSTOM_LINT BENCHMARK SUITE"
  echo "=========================================================================================="
  printf "%-45s | %9s | %9s | %9s\n" "SCENARIO" "REAL (s)" "USER (s)" "SYS (s)"
  printf -- "----------------------------------------------+-----------+-----------+-----------\n"

  run_py_bench() {
    local label="$1"
    shift
    local tf
    tf="$(mktemp)"
    /usr/bin/time -q -f "%e %U %S" -o "${tf}" "$@" >/dev/null 2>&1
    local r u s
    read -r r u s <"${tf}"
    rm -f "${tf}"
    printf "%-45s | %8ss | %8ss | %8ss\n" "${label}" "${r}" "${u}" "${s}"
  }

  PY_PRELUDE="import pathlib; from ${MODULE} import engine as custom_lint; root = custom_lint.find_torch_tpu_root(pathlib.Path('${TORCH_TPU}')); rules = custom_lint.load_rules(root_dir=root).rules"

  run_py_bench "1. Load & compile rules in memory" \
    python3 -c "${PY_PRELUDE}; [r.prewarm() for r in rules]"

  run_py_bench "2. Single file in-process (BUILD, no VCS)" \
    python3 -c "${PY_PRELUDE}; txt = (root / 'BUILD').read_text(); [custom_lint.check_file_content(txt, 'BUILD', r) for r in rules]"

  run_py_bench "3. Single explicit file via CLI (BUILD)" \
    python3 -m "${MODULE}" --root "${TORCH_TPU}" --all "${TORCH_TPU}/BUILD"

  run_py_bench "4. Incremental CL diff (VCS mode)" \
    python3 -m "${MODULE}" --root "${TORCH_TPU}"

  run_py_bench "5. Full repo scan (--all --jobs 1)" \
    python3 -m "${MODULE}" --root "${TORCH_TPU}" --all --jobs 1

  run_py_bench "6. Full repo scan (--all --jobs 8)" \
    python3 -m "${MODULE}" --root "${TORCH_TPU}" --all --jobs 8

  echo "=========================================================================================="
fi
