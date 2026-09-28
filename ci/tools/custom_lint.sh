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

# Runs custom TorchTPU convention checks using the Rust `custom_linter` binary.
#
# Modes:
#   - 'lint'   : Used by CI. Compiles `custom_linter` via Cargo (in OSS) and
#                evaluates rules across the repository, printing build and
#                multi-core execution metrics.
#   - 'fix' / 'format' : Currently a no-op as custom convention rules do not
#                support automatic formatting/fixing.

set -e

BASE_SHA="${BASE_SHA:-origin/main}"
MODE="${1:-}"

case "$MODE" in
  lint)
    if [[ -f "ci/custom_lint_rules.textproto" ]]; then
      COMMON_ARGS=(
        --rules ci/custom_lint_rules.textproto
        --canonical-root torch_tpu/oss
        --project-root .
        --path-map ci/custom_lint_rules.textproto=torch_tpu/custom_lint_rules.textproto
        --path-map src/torch_tpu=torch_tpu
        --path-map LICENSE=torch_tpu/LICENSE
        --path-map build_files=torch_tpu/build_files
        --path-map csrc=torch_tpu/csrc
        --path-map docker=torch_tpu/docker
        --path-map docs=torch_tpu/docs
        --path-map examples=torch_tpu/examples
        --path-map perflab=torch_tpu/perflab
        --path-map tests=torch_tpu/tests
        --path-map csrc/common/contain.cc=torch_tpu/oss/torch_tpu/csrc/common/contain.cc
        --path-map csrc/common/environment.cc=torch_tpu/oss/torch_tpu/csrc/common/environment.cc
        --path-map csrc/common/environment_test.cc=torch_tpu/oss/torch_tpu/csrc/common/environment_test.cc
        --path-map src/torch_tpu/_internal/distributed/multiprocessing.py=torch_tpu/oss/torch_tpu/_internal/distributed/multiprocessing.py
        --path-map src/torch_tpu/_internal/benchmarks/export_adapter.py=torch_tpu/oss/torch_tpu/_internal/benchmarks/export_adapter.py
        --path-map src/torch_tpu/_internal/benchmarks/xprof_adapter.py=torch_tpu/oss/torch_tpu/_internal/benchmarks/xprof_adapter.py
      )
      echo "INFO: Building custom_linter (Rust release binary)..."
      time cargo build --release --manifest-path ci/tools/custom_linter/Cargo.toml
      echo "INFO: Running custom TorchTPU convention checks (full repo --all benchmark)..."
      time ./ci/tools/custom_linter/target/release/custom_linter check --all "${COMMON_ARGS[@]}" || true
      echo "INFO: Running custom TorchTPU convention checks on modified files against ${BASE_SHA}..."
      ./ci/tools/custom_linter/target/release/custom_linter check --base "${BASE_SHA}" "${COMMON_ARGS[@]}"
    else
      SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
      TORCH_TPU="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
      REPO_ROOT="$(cd "${TORCH_TPU}/../../.." && pwd)"
      LINTER_BIN="${LINTER_BIN:-${REPO_ROOT}/blaze-bin/${TORCH_TPU#"${REPO_ROOT}/"}/oss/ci/tools/custom_linter/custom_linter_cli}"
      echo "INFO: Running custom TorchTPU convention checks (full repo --all)..."
      time "${LINTER_BIN}" check --all \
        --rules "${TORCH_TPU}/custom_lint_rules.METADATA" \
        --canonical-root "//depot/$(basename "${REPO_ROOT}")/${TORCH_TPU#"${REPO_ROOT}/"}" \
        --project-root "${TORCH_TPU}" \
        || true
    fi
    ;;

  fix|format)
    echo "INFO: Custom convention checks currently do not support automatic fixing."
    ;;

  *)
    echo "ERROR: Unknown mode '$MODE'. Explicit arguments required: 'lint' or 'fix'/'format'." >&2
    echo "Usage: $0 {lint|fix|format}" >&2
    exit 2
    ;;
esac
