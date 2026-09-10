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
#
# Resolves the effective base commit SHA for diff evaluation across CI workflows.
#
# Usage:
#   resolve_base_sha.sh [CURRENT_SHA] [OVERRIDE_BASE_SHA]
#
# Rules:
# 1. If OVERRIDE_BASE_SHA is provided, fetch and return git merge-base.
# 2. Else if CURRENT_SHA is a merge commit (e.g. GitHub Actions synthetic merge commit
#    refs/pull/N/merge), return CURRENT_SHA^1 (the base branch tip at CI runtime).
#    This isolates PR changes from intermediate commits merged to main.
# 3. Else, return an empty string (signals non-PR event -> run full test/build suite).

set -euo pipefail

CURRENT_SHA="${1:-HEAD}"
OVERRIDE_BASE_SHA="${2:-}"

if [[ -n "${OVERRIDE_BASE_SHA}" ]]; then
  git fetch --no-tags --depth=1 origin "${OVERRIDE_BASE_SHA}" 2>/dev/null || true
  git merge-base "${OVERRIDE_BASE_SHA}" "${CURRENT_SHA}" 2>/dev/null || echo "${OVERRIDE_BASE_SHA}"
elif git rev-parse --verify "${CURRENT_SHA}^2" >/dev/null 2>&1; then
  git rev-parse "${CURRENT_SHA}^1"
else
  echo ""
fi
