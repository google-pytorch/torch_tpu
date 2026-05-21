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

"""Diagnostic hook for wheel tests to provide actionable advice on import errors."""

import json
import os
import re
import sys


def install_hook():
  """Installs a sys.excepthook to catch and analyze import-related errors."""
  original_excepthook = sys.excepthook

  def diagnostic_excepthook(exc_type, value, traceback):
    # Recursively check the exception and its causes for ImportErrors
    current = value
    while current:
      if isinstance(current, (ImportError, ModuleNotFoundError)):
        analyze_import_error(current)
        break
      current = getattr(current, "__cause__", None) or getattr(
          current, "__context__", None
      )

    original_excepthook(exc_type, value, traceback)

  sys.excepthook = diagnostic_excepthook


def analyze_import_error(error):
  """Analyzes an ImportError and prints helpful advice if the module exists in source."""
  # Find the manifest file in runfiles
  manifest_path = find_runfile("bazel/source_tree_manifest.json")
  if not manifest_path or not os.path.exists(manifest_path):
    return

  try:
    with open(manifest_path, "r") as f:
      manifest = json.load(f)
  except (OSError, json.JSONDecodeError):
    return

  # Extract the module name that failed to load
  module_name = getattr(error, "name", None)

  # Special handling for "cannot import name 'Z' from 'X.Y'"
  match = re.search(r"cannot import name '([^']+)' from '([^']+)'", str(error))
  if match:
    module_name = f"{match.group(2)}.{match.group(1)}"

  # Special handling for missing shared objects or ABI errors
  so_match = re.search(r"([^/\s]+\.so)", str(error))
  if so_match:
    module_name = so_match.group(1)

  if not module_name:
    match = re.search(r"No module named '([^']+)'", str(error))
    if match:
      module_name = match.group(1)

  if not module_name:
    return

  # Convert module name to potential source paths
  base_path = module_name.replace(".", "/")
  candidates = [
      base_path + ".py",
      base_path + "/__init__.py",
      base_path + ".so",
      module_name,  # For cases like "libfoo.so"
  ]
  # Add common locations for .so files if they are just the basename
  if module_name.endswith(".so"):
    candidates.append("torch_tpu/common/" + module_name)

  for candidate in candidates:
    # Check if the candidate exists in the source tree manifest
    if candidate in manifest or ("_main/" + candidate) in manifest:
      provide_advice(module_name, candidate)
      return


def provide_advice(module_name, source_path):
  """Prints actionable advice based on the type of missing module."""
  print("\n" + "=" * 80, file=sys.stderr)
  print("TORCH_TPU WHEEL TEST DIAGNOSTICS", file=sys.stderr)
  print("-" * 80, file=sys.stderr)
  print(f"FAILED TO LOAD MODULE: {module_name}", file=sys.stderr)
  print(f"FOUND IN SOURCE TREE: {source_path}", file=sys.stderr)
  print("-" * 80, file=sys.stderr)

  if source_path.startswith("torch_tpu/") and "test" not in source_path:
    print(
        "ADVICE: This appears to be a production module missing from the"
        " wheel.",
        file=sys.stderr,
    )
    print(
        "Please add it to the 'srcs' list in ci/wheel/BUILD.bazel.",
        file=sys.stderr,
    )
  elif "test" in source_path:
    print(
        "ADVICE: This appears to be a test utility or helper missing from the"
        " test environment.",
        file=sys.stderr,
    )
    print(
        "Please add it to 'wheel_extra_test_deps' in the root BUILD.bazel.",
        file=sys.stderr,
    )
  elif source_path.endswith(".so"):
    print(
        "ADVICE: This appears to be a C++ extension (pybind) missing from the"
        " wheel.",
        file=sys.stderr,
    )
    print(
        "Check if it is correctly listed in torch_tpu/common/BUILD and"
        " ci/wheel/BUILD.bazel.",
        file=sys.stderr,
    )
  else:
    print(
        "ADVICE: This module exists in source but is missing from the current"
        " environment.",
        file=sys.stderr,
    )
    print(
        "Check if it should be added to the wheel or extra test dependencies.",
        file=sys.stderr,
    )

  print("=" * 80 + "\n", file=sys.stderr)


def find_runfile(relpath):
  """Finds a file in Bazel runfiles."""
  # Try the environment variable first
  runfiles_dir = os.environ.get("PYTHON_RUNFILES")
  if runfiles_dir:
    path = os.path.join(runfiles_dir, relpath)
    if os.path.exists(path):
      return path
    # Try with _main prefix (Bzlmod/Bazel 6+)
    path = os.path.join(runfiles_dir, "_main", relpath)
    if os.path.exists(path):
      return path

  # Fallback: search relative to sys.path
  for p in sys.path:
    path = os.path.join(p, relpath)
    if os.path.exists(path):
      return path

  return None


if __name__ == "__main__":
  install_hook()
