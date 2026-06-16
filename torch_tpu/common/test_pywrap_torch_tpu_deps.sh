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
OUTPUT_FILE=$1
if [ -s "$OUTPUT_FILE" ]; then
    echo "ERROR: pywrap_torch_tpu_deps contains targets that transitively depend on PyTorch!"
    echo "The following PyTorch-related targets were found in the dependency graph:"
    head -n 50 "$OUTPUT_FILE"
    echo "..."
    echo "To fix this, do not manually add PyTorch dependencies to pywrap_torch_tpu_deps."
    echo "Instead, run update_common_deps.py to regenerate the list of PyTorch-agnostic dependencies."
    exit 1
fi
echo "SUCCESS: pywrap_torch_tpu_deps does not depend on PyTorch."
exit 0
