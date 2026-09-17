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

"""Reads the supported Python versions from the single source of truth.

The source of truth is //bazel:supported_python_versions.bzl. Starlark
consumers `load()` it directly; everything else -- GitHub Actions workflows,
shell scripts, noxfile.py -- goes through this module so that there is
exactly one parser rather than a scattering of ad-hoc `grep | cut` calls.

That file is constrained to be valid Python as well as valid Starlark (see
its docstring, and python_versions_test.py which enforces it), so it is
evaluated directly instead of being pattern-matched.

Examples:

  # Print a constant as a JSON array, for fromJSON() in a workflow matrix.
  $ python3 ci/tools/python_versions.py SUPPORTED_PYTHON_VERSIONS
  ["3.11", "3.12", "3.13", "3.14"]

  # Print a scalar for shell interpolation.
  $ python3 ci/tools/python_versions.py DEFAULT_PYTHON_VERSION --format=plain
  3.14

  # Feed a whole build_wheel.yml setup job.
  $ python3 ci/tools/python_versions.py --github-output --run-all
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Sequence
import json
import os
import pathlib
import sys
from typing import Final

_REPO_ROOT: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[2]

_SOURCE_OF_TRUTH: Final[pathlib.Path] = (
    _REPO_ROOT / "bazel" / "supported_python_versions.bzl"
)

Constants = dict[str, str | list[str]]


def _validate(tree: ast.Module, path: pathlib.Path) -> None:
  """Rejects anything that makes the file more than a bag of constants.

  Evaluating arbitrary code here would be a problem, and a `load()` statement
  would make the file valid Starlark but not valid Python. Rather than trying
  to pattern-match those cases, require the whole module to consist of a
  docstring plus `UPPER_CASE = <expression>` assignments, which is all the
  source of truth is ever meant to contain.

  Args:
    tree: The parsed source of truth.
    path: Only used to build error messages.

  Raises:
    ValueError: If the module contains anything else.
  """
  for node in tree.body:
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
      continue  # Module docstring.
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
      target = node.targets[0]
      if isinstance(target, ast.Name) and target.id.isupper():
        continue
    raise ValueError(
        f"{path}:{node.lineno}: expected only a docstring and UPPER_CASE "
        "assignments. This file is read as plain Python by "
        "ci/tools/python_versions.py, so it must not contain load() "
        "statements, function definitions or other executable code."
    )


def load_constants(path: pathlib.Path | None = None) -> Constants:
  """Returns the `UPPER_CASE` constants defined by the source of truth.

  Args:
    path: The .bzl file to read. Defaults to the source of truth; overridable
      for tests.

  Raises:
    ValueError: If the file contains anything but constant assignments.
  """
  path = path or _SOURCE_OF_TRUTH
  source = path.read_text()

  tree = ast.parse(source, filename=str(path))
  _validate(tree, path)

  namespace: dict[str, object] = {"__builtins__": {}}
  exec(compile(tree, str(path), "exec"), namespace)  # pylint: disable=exec-used
  return {
      name: value
      for name, value in namespace.items()
      if name.isupper() and isinstance(value, (str, list))
  }


def matrix_for_event(
    constants: Constants, *, run_all_versions: bool
) -> dict[str, list[str]]:
  """Returns the CPU/TPU build matrices for a build_wheel.yml run.

  Args:
    constants: As returned by `load_constants`.
    run_all_versions: True for postsubmit-equivalent runs (push, schedule, or a
      PR labelled ci:postsubmit / ci:run_all_versions / tpu-nightly), which
      sweep the whole supported range instead of the presubmit subset.
  """
  if run_all_versions:
    return {
        "cpu_versions": constants["POSTSUBMIT_PYTHON_VERSIONS"],
        "tpu_versions": constants["POSTSUBMIT_PYTHON_VERSIONS"],
    }
  return {
      "cpu_versions": constants["PRESUBMIT_CPU_PYTHON_VERSIONS"],
      "tpu_versions": constants["PRESUBMIT_TPU_PYTHON_VERSIONS"],
  }


def github_outputs(
    constants: Constants, *, run_all_versions: bool
) -> dict[str, str]:
  """Returns the rendered `name=value` outputs for a workflow `setup` job.

  Lists are rendered as JSON so that workflows can consume them with
  `fromJSON()`; scalars are rendered bare.
  """
  values: dict[str, str | list[str]] = dict(
      matrix_for_event(constants, run_all_versions=run_all_versions)
  )
  for name in (
      "DEFAULT_PYTHON_VERSION",
      "LATEST_PYTHON_VERSION",
      "OLDEST_PYTHON_VERSION",
  ):
    values[name.lower()] = constants[name]

  return {
      key: json.dumps(value) if isinstance(value, list) else value
      for key, value in values.items()
  }


def _render(value: str | list[str], output_format: str) -> str:
  if output_format == "json":
    return json.dumps(value)
  if isinstance(value, str):
    return value
  return "\n".join(value) if output_format == "lines" else " ".join(value)


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "name",
      nargs="?",
      help="Constant to print, e.g. SUPPORTED_PYTHON_VERSIONS.",
  )
  parser.add_argument(
      "--format",
      dest="output_format",
      choices=("json", "lines", "plain"),
      default="json",
      help=(
          "json: a JSON literal (default, for fromJSON() in workflows). "
          "lines: newline-separated, for bash readarray. "
          "plain: space-separated, for shell word splitting."
      ),
  )
  parser.add_argument(
      "--github-output",
      action="store_true",
      help="Append the full set of workflow outputs to $GITHUB_OUTPUT.",
  )
  parser.add_argument(
      "--run-all-versions",
      action="store_true",
      help="Emit the postsubmit matrix rather than the presubmit subset.",
  )
  args = parser.parse_args(argv)

  constants = load_constants()

  if args.github_output:
    outputs = github_outputs(constants, run_all_versions=args.run_all_versions)
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
      for key, value in outputs.items():
        output.write(f"{key}={value}\n")
    # Echo to the log so the resolved matrix is visible in the run's output.
    for key, value in outputs.items():
      print(f"{key}={value}")
    return 0

  if not args.name:
    parser.error("a constant name is required unless --github-output is used")
  if args.name not in constants:
    parser.error(
        f"unknown constant {args.name!r}; "
        f"available: {', '.join(sorted(constants))}"
    )

  print(_render(constants[args.name], args.output_format))
  return 0


if __name__ == "__main__":
  sys.exit(main())
