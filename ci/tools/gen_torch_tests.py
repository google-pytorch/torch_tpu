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

"""Generates Bazel test targets for PyTorch device-generic tests.

This script discovers test modules under test/ directory of a PyTorch checkout
that instantiate device-generic tests via `instantiate_device_type_tests`, and
writes corresponding test targets to a BUILD file, by default tests/torch/BUILD.

Command-line flags and environment variables:
  1. --pytorch_test_dir: The path to the test/ directory of a PyTorch checkout.
  2. --build_file: Optional path to the BUILD file where targets are written.
     Defaults to tests/torch/BUILD under the workspace root.

Example usage:
  $ bazel run //ci/tools:gen_torch_tests -- \
      --pytorch_test_dir=~/.github/pytorch/test

The script exits with 0 on success, and non-zero on failure.
"""

import argparse
from collections.abc import Sequence
import os
import pathlib
import re
import sys
from typing import Final

from ci.tools import torch_test_module_utils

BEGIN_MARKER: Final[str] = "# GENERATED_TEST_TARGETS_BEGIN"
END_MARKER: Final[str] = "# GENERATED_TEST_TARGETS_END"
# Matches the call that instantiates device-generic test classes in a module:
#   ^\s*                     start of a line (re.MULTILINE) plus any indent, so
#                            that mentions in comments, strings and argument
#                            lists are not mistaken for the call
#   (common_device_type\.)?  optional qualifier, since upstream imports the
#                            helper both bare and through its module
#   \(                       an actual call, not an import or a bare reference
DEVICE_TYPE_TESTS_INSTANTIATION: Final[re.Pattern[str]] = re.compile(
    r"^\s*(common_device_type\.)?instantiate_device_type_tests\(", re.MULTILINE
)


def build_file_path(build_file: str | pathlib.Path | None) -> pathlib.Path:
  """Returns the resolved BUILD file path, falling back to workspace default."""
  if build_file:
    return pathlib.Path(build_file).expanduser().resolve()

  workspace_dir = os.environ.get(
      "BUILD_WORKSPACE_DIRECTORY", pathlib.Path.cwd()
  )
  return pathlib.Path(workspace_dir) / "tests" / "torch" / "BUILD"


def discover_device_generic_test_modules(
    test_dir: pathlib.Path,
) -> frozenset[str]:
  """Returns test modules with device-generic test instantiations."""
  return frozenset(
      str(path.relative_to(test_dir).with_suffix(""))
      for path in test_dir.rglob("test_*.py")
      if DEVICE_TYPE_TESTS_INSTANTIATION.search(
          path.read_text(errors="replace")
      )
  )


def torch_test_target(test_module: str) -> str:
  """Returns the Bazel target definition for a PyTorch test module.

  For instance, given `dynamo/test_debug_utils.py`, the function returns:

  torch_tpu_oss_torch_test(
      name = "dynamo_debug_utils_test",
      torch_test_module = "dynamo/test_debug_utils",
  )

  Args:
    test_module: Relative path of the test module without the `.py` extension.

  Returns:
    The formatted Bazel target definition.
  """
  # Normalize the module path to a test target name.
  *dirs, basename = test_module.split("/")
  name = "_".join([*dirs, basename.removeprefix("test_"), "test"])

  notap = (
      '    notap = "b/563402379",  # NOTAP_OK=torch tests\n'
      if test_module in torch_test_module_utils.DISABLED_TEST_MODULES
      else ""
  )

  num_shards = torch_test_module_utils.TEST_MODULE_SHARDS.get(test_module)
  shard_count = f"    shard_count = {num_shards},\n" if num_shards else ""

  return (
      "torch_tpu_oss_torch_test(\n"
      f'    name = "{name}",\n'
      f"{notap}"
      f"{shard_count}"
      f'    torch_test_module = "{test_module}",\n'
      ")"
  )


def write_to_build_file(
    build_file: pathlib.Path, torch_test_targets: str
) -> None:
  """Writes generated test targets between markers in the given BUILD file."""
  text = build_file.read_text()
  head, _, rest = text.partition(BEGIN_MARKER)
  _, end, tail = rest.partition(END_MARKER)

  if not end:
    sys.exit(
        f"ERROR: No markers are found in {build_file}. Please use"
        f" {BEGIN_MARKER} and {END_MARKER} to specify where test targets"
        " should be written."
    )

  build_file.write_text(
      f"{head}{BEGIN_MARKER}\n{torch_test_targets}\n{END_MARKER}{tail}"
  )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
      "--pytorch_test_dir",
      default=None,
      help=(
          "The path to test/ directory of a"
          " https://github.com/pytorch/pytorch/ checkout."
      ),
  )
  parser.add_argument(
      "--build_file",
      default=None,
      help=(
          "The BUILD file where test targets are written to. "
          "By default, it writes to tests/torch/BUILD under the workspace root."
      ),
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)

  if not args.pytorch_test_dir:
    print(
        "ERROR: Must specify the path to PyTorch test/ directory via"
        " --pytorch_test_dir.",
        file=sys.stderr,
    )
    return 1

  test_dir = pathlib.Path(args.pytorch_test_dir).expanduser().resolve()
  if not test_dir.is_dir():
    print(
        f"ERROR: {test_dir} does not exist or is not a directory.",
        file=sys.stderr,
    )
    return 1

  build_file = build_file_path(args.build_file)
  if not build_file.is_file():
    print(
        f"ERROR: {build_file} does not exist or is not a file.",
        file=sys.stderr,
    )
    return 1

  device_generic_test_modules = sorted(
      discover_device_generic_test_modules(test_dir)
      - torch_test_module_utils.EXCLUDED_TEST_MODULES
  )
  if not device_generic_test_modules:
    print(
        f"ERROR: No device-generic test modules found in {test_dir}. "
        "To avoid wiping out test targets, exiting...\n"
        "Please make sure --pytorch_test_dir points to the test/ directory "
        "of a valid PyTorch checkout.",
        file=sys.stderr,
    )
    return 1

  torch_test_targets = "\n\n".join(
      torch_test_target(module) for module in device_generic_test_modules
  )

  write_to_build_file(build_file, torch_test_targets)

  num_disabled = len(
      set(device_generic_test_modules)
      & torch_test_module_utils.DISABLED_TEST_MODULES
  )
  num_enabled = len(device_generic_test_modules) - num_disabled
  print(
      f"INFO: Successfully wrote {len(device_generic_test_modules)} test"
      f" targets ({num_disabled} disabled, {num_enabled} enabled) to"
      f" {build_file}.",
      file=sys.stderr,
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
