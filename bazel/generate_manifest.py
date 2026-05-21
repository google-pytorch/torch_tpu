#!/usr/bin/env python3
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

"""Generates a manifest of source tree files for wheel test diagnostics."""

import json
import sys


def main():
  if len(sys.argv) < 2:
    print("Usage: generate_manifest.py <output_json> [input_files...]")
    sys.exit(1)

  output_file = sys.argv[1]
  input_files = sys.argv[2:]

  # We store a dictionary of files that exist in the source tree.
  # The keys are paths relative to the workspace root.
  manifest = {f: True for f in input_files}

  with open(output_file, "w") as f:
    json.dump(manifest, f, indent=2)


if __name__ == "__main__":
  main()
