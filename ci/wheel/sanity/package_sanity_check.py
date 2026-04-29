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
"""Ensures all repository Python files are explicitly registered for testing/shipping."""

import pathlib
import sys

# Directories to ignore when comparing the 'actual' files to 'registered' files.
# These directories contain code that is not intended to be shipped or tested.
IGNORE_DIRS = {
    "build_files",
    "ci",
    "docker",
    "docs",
    "third_party",
    "tools",
    "bazel",
    "examples",
    "venv",
    "dist",
}


def load_manifest(name):
  """Loads paths from a manifest file in the runfiles."""
  # Try various runfile path layouts
  search_paths = [
      pathlib.Path(f"ci/wheel/sanity/{name}"),
      pathlib.Path(__file__).parent / name,
  ]

  for p in search_paths:
    if p.exists():
      return set(p.read_text().splitlines())

  print(f"Error: Manifest {name} not found in any of {search_paths}")
  sys.exit(1)


def is_ignored(path_str):
  """Checks if a path should be ignored by the sanity check."""
  path = pathlib.PurePath(path_str)
  parts = path.parts
  if not parts:
    return True
  # Ignore junk/hidden files
  if any(p.startswith(".") for p in parts):
    return True
  # Ignore infrastructure directories
  if parts[0] in IGNORE_DIRS:
    return True
  return False


def main():
  print("Loading manifests...")
  # Manifest of every .py file matched by glob
  all_actual = load_manifest("actual_files.txt")
  # Manifest of every file in the transitive closure of wheel/test targets
  registered_files = load_manifest("registered_files.txt")

  # Filter actual files to only those we care about
  actual_files = {f for f in all_actual if not is_ignored(f)}

  # Find files that are in the source tree but NOT in any build target.
  missing_files = sorted(list(actual_files - registered_files))

  if missing_files:
    print("=" * 68)
    print("[!] ERROR: Unregistered Python files detected!")
    print("The following files are present in the repo but NOT registered")
    print("in any target that goes into the wheel or test environment:")
    for f in missing_files:
      print(f"  -> {f}")
    print("")
    print("ACTION REQUIRED:")
    print(
        " 1. If it's a test file or support code, add it to"
        " 'wheel_extra_test_deps'"
    )
    print("    in the root BUILD.bazel.")
    print(" 2. If it should be shipped in the pip package, add it to")
    print("    'all_torch_tpu_files' in ci/wheel/BUILD.bazel.")
    print("=" * 68)
    sys.exit(1)

  print(f"All {len(actual_files)} Python files are properly registered! PASS.")


if __name__ == "__main__":
  main()
