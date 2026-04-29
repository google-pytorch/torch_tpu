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
"""Generates a manifest of registered Python files from build targets."""

import os
import sys


def main():
  # SRCS contains all files passed to the genrule.
  # We want to output their paths relative to the execution root (repo root).

  seen = set()
  for arg in sys.argv[1:]:
    # Standardize path: remove any leading ./
    path = os.path.normpath(arg)

    # We only care about .py files
    if not path.endswith(".py"):
      continue

    # If running in a context where sources are under a repo-specific prefix
    # (like when Bazel treats the root as a package), we might need to
    # standardize. But for a standard local BUILD, 'path' is already relative.

    if path not in seen:
      print(path)
      seen.add(path)


if __name__ == "__main__":
  main()
