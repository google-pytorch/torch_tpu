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

r"""Produces a `compile_commands.json` from the output of `bazel aquery`.

This tool requires that a build has been completed for all targets in the
query (e.g., for the example usage below `bazel build //...`). This is due
to generated files like proto headers and files generated via tablegen.

Example usage:
  bazel aquery "mnemonic(CppCompile, //...)" --output=jsonproto | \
      python3 ci/tools/generate_compile_commands.py
"""

import dataclasses
import json
import pathlib
import sys
from typing import Any

from absl import logging

_JSONDict = dict[Any, Any]  # Approximates parsed JSON

_DISALLOWED_ARGS = frozenset(["-fno-canonical-system-headers"])

# Assuming this script lives in ci/tools/
# Two levels up is the repository root.
_TORCH_TPU_SRC_ROOT = pathlib.Path(__file__).absolute().parent.parent


@dataclasses.dataclass
class CompileCommand:
  """Represents a compilation command with options on a specific file."""

  file: str
  arguments: list[str]

  @classmethod
  def from_args_list(cls, args_list: list[str]) -> "CompileCommand":
    """Alternative constructor which uses the args_list from `bazel aquery`."""
    cc_file = None
    filtered_args = []

    for arg in args_list:
      if arg in _DISALLOWED_ARGS:
        continue

      # Support standard C++ extensions
      if arg.endswith(".cc") or arg.endswith(".cpp"):
        cc_file = arg

      # Split generated commands, because otherwise they get wrapped
      # into "command with spaces" when passed to clangd/clang-tidy,
      # and they can't parse them correctly.
      for s in arg.split(" "):
        filtered_args.append(s)

    return cls(cc_file, filtered_args)

  def to_dumpable_json(self, directory: str) -> _JSONDict:
    return {
        "directory": directory,
        "file": self.file,
        "arguments": self.arguments,
    }


def extract_compile_commands(
    parsed_aquery_output: _JSONDict,
) -> list[CompileCommand]:
  """Gathers compile commands to run from `bazel aquery` JSON output."""
  actions = parsed_aquery_output.get("actions", [])

  commands = []
  for action in actions:
    if "arguments" in action:
      command = CompileCommand.from_args_list(action["arguments"])
      # Only append if we successfully identified a C++ source file
      if command.file:
        commands.append(command)
  return commands


def main():
  # Setup logging
  logging.basicConfig()
  logging.getLogger().setLevel(logging.INFO)

  # Setup external symlink if necessary so headers can be found in include
  # paths.
  if not (external := _TORCH_TPU_SRC_ROOT / "external").exists():
    bazel_external = _TORCH_TPU_SRC_ROOT / "bazel-torch_tpu" / "external"
    if bazel_external.exists():
      logging.info("Symlinking `%s` to `external`", bazel_external)
      external.symlink_to(bazel_external)

  logging.info("Reading `bazel aquery` output from stdin...")
  try:
    parsed_aquery_output = json.loads(sys.stdin.read())
  except json.JSONDecodeError:
    logging.error(
        "Failed to parse JSON. Ensure `bazel aquery` ran successfully."
    )
    sys.exit(1)

  commands = extract_compile_commands(parsed_aquery_output)

  out_file = _TORCH_TPU_SRC_ROOT / "compile_commands.json"
  with out_file.open("w") as f:
    json.dump(
        [
            command.to_dumpable_json(directory=str(_TORCH_TPU_SRC_ROOT))
            for command in commands
        ],
        f,
        indent=2,  # Added for easier debugging of the JSON output
    )

  logging.info(
      "Successfully wrote %d commands to %s",
      len(commands),
      out_file.name,
  )


if __name__ == "__main__":
  main()
