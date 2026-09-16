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

if [[ $# -eq 0 ]]; then
  echo "ERROR: No test binary provided."
  echo "Usage: $0 <path_to_test_binary> [args...]"
  exit 1
fi

# The first argument is the Bazel-provided path to the C++ test binary.
TEST_BINARY="$1"
shift # Remove the binary from the argument list so "$@" contains only test flags.

# If TEST_BINARY does not end in _bin (e.g. wrapper target name was passed),
# resolve to the underlying C++ binary ending in _bin.
if [[ ! "$TEST_BINARY" =~ _bin$ ]] && [[ -f "${TEST_BINARY}_bin" ]]; then
  TEST_BINARY="${TEST_BINARY}_bin"
fi

if [[ "$TEST_BINARY" != /* ]] && [[ "$TEST_BINARY" != ./* ]]; then
  TEST_BINARY="./$TEST_BINARY"
fi

if [[ -n "$TPU_LIBRARY_PATH" && -f "$TPU_LIBRARY_PATH" ]]; then
  echo "INFO: Using existing TPU_LIBRARY_PATH=$TPU_LIBRARY_PATH"
else
  echo "INFO: Setting environment var TPU_LIBRARY_PATH."

  # 1. Find the .whl file in the Runfiles Root (..)
  # Search '..' to catch sibling repositories. In modern Bazel with sibling
  # directory layout, repositories like 'pypi_libtpu' are placed next to the main
  # repository instead of being inside it.
  WHL_FILE=$(find -L .. -name "*.whl" | grep "libtpu" | head -n 1)

if [[ -z "$WHL_FILE" ]]; then
  echo "ERROR: Could not find libtpu .whl file in runfiles."
  echo "DEBUG: Searching in parent directory:"
  find -L .. -maxdepth 3 -name "*libtpu*"
  exit 1
fi

echo "INFO: Found wheel: $WHL_FILE"

# 2. Extract the wheel to a temp directory
EXTRACT_DIR="libtpu_extracted"
rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"

unzip -q -o "$WHL_FILE" -d "$EXTRACT_DIR"

# 3. Find libtpu.so inside the extracted directory
LIBTPU_SO=$(find "$EXTRACT_DIR" -name "libtpu.so" | head -n 1)

if [[ -z "$LIBTPU_SO" ]]; then
  echo "ERROR: Wheel extracted, but libtpu.so was not found inside."
  exit 1
fi

# 4. Get absolute path & Export
ABS_LIBTPU=$(readlink -f "$LIBTPU_SO")
export TPU_LIBRARY_PATH="$ABS_LIBTPU"
echo "INFO: Set TPU_LIBRARY_PATH=$TPU_LIBRARY_PATH"
fi

# 5. Run the binary with any remaining Bazel arguments
echo "INFO: Executing test binary: $TEST_BINARY $@"
"$TEST_BINARY" "$@"
