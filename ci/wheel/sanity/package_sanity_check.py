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

"""Ensures all top-level Python directories are registered in BUILD configuration."""

import pathlib
import re
import sys

# Directories that contain Python files but are not intended to be registered as
# wheel components or test helpers.
IGNORE_DIRS = {
    "build_files",
    "ci",
    "docker",
    "docs",
    "third_party",
    "tools",
    "oss",
    "bazel",
    "examples",
}


def get_root_dir():
  """Finds the root directory containing the WORKSPACE file."""
  current = pathlib.Path(__file__).resolve().parent
  while current != current.parent:
    has_workspace = (current / "WORKSPACE").exists()
    has_module = (current / "MODULE.bazel").exists()
    if has_workspace or has_module:
      return current
    current = current.parent
  return pathlib.Path(".").resolve()


def get_all_dirs(root):
  """Finds all top-level directories containing Python files."""
  all_dirs = set()
  # Search for all .py files up to 2 levels deep from root
  for path in root.rglob("*.py"):
    relative = path.relative_to(root)
    parts = relative.parts
    if len(parts) > 2:
      continue
    if any(p.startswith(".") for p in parts):
      continue

    if len(parts) >= 1:
      all_dirs.add(parts[0])

  return {d for d in all_dirs if d not in IGNORE_DIRS}


def extract_registered_packages(file_path, target_names):
  """Extracts package names from labels in the specified target blocks."""
  if not file_path.exists():
    print(f"Warning: {file_path} not found.")
    return set()

  content = file_path.read_text()
  packages = set()

  for target in target_names:
    # Find the start of the target definition
    # Matches something like: py_library( name = "target", ...
    # Handle potential whitespace and multiple targets in one file.
    # This regex looks for the rule call followed by the name attribute.
    target_re = re.compile(
        rf'\w+\(\s*(?:[\w\s=,"]*?)\bname\s*=\s*"{target}"', re.DOTALL
    )
    match = target_re.search(content)
    if not match:
      continue

    # Find the closing parenthesis of the target.
    # Simple balanced parenthesis counting starting from rule opening '('
    start_idx = match.start()
    opening_paren = content.find("(", start_idx)
    if opening_paren == -1:
      continue

    depth = 1
    end_idx = opening_paren + 1
    while depth > 0 and end_idx < len(content):
      if content[end_idx] == "(":
        depth += 1
      elif content[end_idx] == ")":
        depth -= 1
      end_idx += 1

    block = content[opening_paren:end_idx]

    # Extract all labels like //foo/bar:baz or :qux
    # We look for strings starting with // or : inside the block
    label_re = re.compile(r'"(//[^"]+?|:[^"]+?)"')
    for label in label_re.findall(block):
      if label.startswith("//"):
        # //foo/bar:baz -> foo, //foo:bar -> foo
        pkg = re.split(r"[:/]", label.lstrip("/"))[0]
        if pkg:
          packages.add(pkg)
      elif label.startswith(":"):
        rel_pkg = file_path.relative_to(get_root_dir()).parts[0]
        if rel_pkg and not rel_pkg.endswith(".bazel"):
          packages.add(rel_pkg)

  return packages


def main():
  root = get_root_dir()
  print(f"Root directory: {root}")

  print("Scanning for top-level directories containing Python files...")
  all_dirs = get_all_dirs(root)

  print("Identifying registered packages via manual BUILD parsing...")
  # Files to check
  root_build = root / "BUILD.bazel"
  wheel_build = root / "ci" / "wheel" / "BUILD.bazel"

  registered = set()
  registered |= extract_registered_packages(
      wheel_build, ["all_torch_tpu_files", "remapped_pywrap_binaries"]
  )
  registered |= extract_registered_packages(
      root_build, ["wheel_extra_test_deps"]
  )

  missing_dirs = sorted([d for d in all_dirs if d not in registered])

  if missing_dirs:
    print("=" * 68)
    print("[!] ERROR: Unregistered Python directories detected!")
    print("The following top-level folders were added but are NOT registered:")
    for d in missing_dirs:
      print(f"  -> {d}/")
    print("")
    print("ACTION REQUIRED:")
    print(
        " 1. If it's for testing support, add its recursive filegroup to "
        "'wheel_extra_test_deps' in BUILD.bazel."
    )
    print(
        " 2. If it should be shipped via pip, add it to "
        "'all_torch_tpu_files' in ci/wheel/BUILD.bazel."
    )
    print("=" * 68)
    sys.exit(1)

  print("All top-level Python directories are properly registered! PASS.")


if __name__ == "__main__":
  main()
