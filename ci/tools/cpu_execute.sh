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
# Wrapper script for CPU-only tests under Bazel in OSS.
#
# Sets up the TPU_LIBRARY_PATH environment variable pointing to the libtpu.so
# in runfiles. This is required because the OpenXLA dynamic registration
# wrapper always attempts to load libtpu.so at process startup (during static
# initializers), even if the test only uses CPU and never initializes TPU.
#
# If TPU_LIBRARY_PATH is not set, the process will crash with:
# "Check failed: RegisterDynamicPjrtPlugin(...) is OK"
#
# By pointing it to the libtpu.so inside runfiles, the dynamic registration
# succeeds, and the CPU tests can run normally on CPU-only machines.

set -euo pipefail

# 1. Source the runfiles library.
# We use a flexible search to handle both WORKSPACE and Bzlmod prefixes.
f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -m1 "$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || true

# 2. Locate the test binary.
if command -v rlocation >/dev/null; then
  TEST_BINARY="$(rlocation "${TEST_WORKSPACE:-_main}/${1#./}")"
else
  TEST_BINARY="$1"
fi
shift; set +uo pipefail

# 3. Locate libtpu.so in runfiles.
# We use a glob to find the unpacked libtpu directory (handles bzlmod mangling).
RUNFILES_DIR_PATH="${RUNFILES_DIR:-.}"
LIBTPU_GLOB="${RUNFILES_DIR_PATH}"/*libtpu*/site-packages/libtpu/libtpu.so

# Resolve the glob to actual path.
resolved_path=""
for path in $LIBTPU_GLOB; do
  if [ -f "$path" ]; then
    resolved_path="$path"
    break
  fi
done

if [ -n "$resolved_path" ]; then
  export TPU_LIBRARY_PATH="$resolved_path"
  echo "Found libtpu.so in runfiles: $TPU_LIBRARY_PATH"
else
  echo "WARNING: libtpu.so not found in runfiles. C++ tests linking to pjrt will likely crash at startup on CPU-only environments."
fi

# 4. Execute the test.
exec "$TEST_BINARY" "$@"
