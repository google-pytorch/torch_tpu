#!/bin/bash
# Copyright 2025 Google LLC
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

set -euo pipefail

# This finds PyTorch shared libraries and symlinks them into torch/lib/
# inside the Bazel repository root.

SOURCE_DIR=$1
PREFIX=${2:-"."}
DEST_DIR="$PREFIX/torch/lib"

mkdir -p "$DEST_DIR"

# Iterate over common library locations in PyTorch
for dir in "$SOURCE_DIR/torch/lib" "$SOURCE_DIR/build/lib"; do
    if [ -d "$dir" ]; then
        # Use a shell glob to find all shared libraries
        for lib in "$dir"/*.so*; do
            # Ensure the glob matched something and it's a real file/symlink
            if [ -e "$lib" ]; then
                ln -sf "$lib" "$DEST_DIR/"
            fi
        done
    fi
done
