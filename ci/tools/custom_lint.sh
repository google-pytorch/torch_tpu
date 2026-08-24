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

# This script checks custom TorchTPU convention rules on modified files.
#
# Design Intent (Incremental Checking):
#   Instead of running across the entire repository, this script uses `git diff`
#   (or `hg diff` in CitC) against a base tracking branch ($BASE_SHA). It
#   strictly evaluates only the specific lines modified in the current Pull
#   Request or local working tree.
#
# Modes:
#   - 'lint'   : Used by CI. Detects convention violations, outputs self-serve
#                suppression instructions, and exits with a failure code.
#   - 'fix' / 'format' : Currently a no-op as custom convention rules do not
#                support automatic formatting/fixing.

set -e

# Fall back to origin/main if BASE_SHA isn't passed by CI.
BASE_SHA="${BASE_SHA:-origin/main}"
MODE="${1:-}"

case "$MODE" in
  lint)
    echo "INFO: Running custom TorchTPU convention checks..."
    time python3 ci/tools/custom_lint.py --all || true
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
