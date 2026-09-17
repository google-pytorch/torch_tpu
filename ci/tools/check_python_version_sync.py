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

"""Verifies the hand-maintained copies of the supported Python versions.

//bazel:supported_python_versions.bzl is the single source of truth, and
almost everything derives from it -- Starlark via `load()`, everything else
via ci/tools/python_versions.py.

A few consumers structurally cannot:

  * MODULE.bazel cannot `load()` Starlark.
  * `workflow_dispatch` `choice` options must be literals; GitHub rejects
    expressions there.
  * copy.bara.sky lists required GitHub check *names*, which embed the
    version because the workflow's job name does.
  * .bazelrc is not Starlark and has no include-from-Starlark mechanism.

Those keep a hand-written copy. This script is what makes that safe: it
fails if any of them drifts from the source of truth.

Run directly, or via `nox -s check_python_version_sync`.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
import pathlib
import re
import sys
from typing import Final

import yaml

_CI_TOOLS_DIR: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parent
_REPO_ROOT: Final[pathlib.Path] = _CI_TOOLS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from ci.tools import python_versions  # pylint: disable=g-import-not-at-top

# copy.bara.sky lives one level above the repository root and is not part of
# the GitHub export, so this check is a no-op wherever that file is absent.
_COPYBARA_CONFIG: Final[pathlib.Path] = _REPO_ROOT.parent / "copy.bara.sky"

_SOT_LABEL: Final[str] = "bazel/supported_python_versions.bzl"


class MissingInputError(Exception):
  """A file this script must inspect is not present."""


def _read_text(path: pathlib.Path) -> str:
  """Reads a file this script verifies, or explains why it could not.

  Under Bazel these files are only present if declared in the target's
  `data`, and the resulting FileNotFoundError points at pathlib rather than
  at the real problem. Translate it into something actionable.
  """
  try:
    return path.read_text()
  except FileNotFoundError as error:
    try:
      relative = path.relative_to(_REPO_ROOT)
    except ValueError:
      relative = path
    raise MissingInputError(
        f"{relative}: not found at {path}. If this is a Bazel run, add it to "
        "the `data` of //ci/tools:check_python_version_sync (and of its "
        "test); only declared files exist in the runfiles tree."
    ) from error


def _starlark_constants(path: pathlib.Path, names: Iterable[str]) -> dict:
  """Extracts top-level constant assignments without executing the file.

  Unlike the source of truth, MODULE.bazel is full of function calls, so it
  cannot simply be exec'd. Parse it and literal_eval only the assignments of
  interest.
  """
  wanted = set(names)
  found = {}
  tree = ast.parse(_read_text(path), filename=str(path))
  for node in tree.body:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
      continue
    target = node.targets[0]
    if isinstance(target, ast.Name) and target.id in wanted:
      try:
        found[target.id] = ast.literal_eval(node.value)
      except ValueError:
        pass  # Not a literal; treated as missing and reported below.
  return found


def check_module_bazel(constants: dict) -> list[str]:
  """MODULE.bazel duplicates the list and the default; both must match."""
  path = _REPO_ROOT / "MODULE.bazel"
  errors = []
  found = _starlark_constants(
      path, ("SUPPORTED_PYTHON_VERSIONS", "DEFAULT_PYTHON_VERSION")
  )

  for name in ("SUPPORTED_PYTHON_VERSIONS", "DEFAULT_PYTHON_VERSION"):
    expected = constants[name]
    actual = found.get(name)
    if actual is None:
      errors.append(
          f"MODULE.bazel: could not find a literal assignment for {name}."
      )
    elif actual != expected:
      errors.append(
          f"MODULE.bazel: {name} is {actual!r} but {_SOT_LABEL} says "
          f"{expected!r}. MODULE.bazel cannot load() the source of truth, so "
          "update this copy by hand."
      )
  return errors


def _workflow(path: pathlib.Path) -> dict:
  """Loads a workflow, normalising the `on:` key.

  YAML 1.1 parses a bare `on` key as the boolean True, so look it up both
  ways rather than depending on the loader's behaviour.
  """
  data = yaml.safe_load(_read_text(path))
  if "on" not in data and True in data:
    data["on"] = data[True]
  return data


def check_run_benchmarks_choices(constants: dict) -> list[str]:
  """The workflow_dispatch `choice` options cannot be expressions."""
  path = _REPO_ROOT / ".github" / "workflows" / "run_benchmarks.yml"
  errors = []
  spec = _workflow(path)["on"]["workflow_dispatch"]["inputs"]["python_version"]

  expected = constants["SUPPORTED_PYTHON_VERSIONS"]
  actual = [str(option) for option in spec["options"]]
  if actual != expected:
    errors.append(
        f"run_benchmarks.yml: python_version options are {actual} but "
        f"{_SOT_LABEL} says {expected}. GitHub requires literal choice "
        "options, so update this copy by hand."
    )

  default = str(spec["default"])
  if default not in expected:
    errors.append(
        f"run_benchmarks.yml: default python_version {default!r} is not a "
        f"supported version ({expected})."
    )
  return errors


def check_bazelrc(constants: dict) -> list[str]:
  """.bazelrc pins the hermetic interpreter to the default version."""
  path = _REPO_ROOT / ".bazelrc"
  expected = constants["DEFAULT_PYTHON_VERSION"]

  matches = re.findall(
      r"^\s*common\s+--repo_env=HERMETIC_PYTHON_VERSION=(\S+)\s*$",
      _read_text(path),
      re.MULTILINE,
  )
  if not matches:
    return [".bazelrc: no HERMETIC_PYTHON_VERSION setting found."]
  return [
      f".bazelrc: HERMETIC_PYTHON_VERSION is {found!r} but {_SOT_LABEL} says "
      f"the default is {expected!r}."
      for found in matches
      if found != expected
  ]


def check_workspace_requirements(constants: dict) -> list[str]:
  """WORKSPACE maps each supported version to its lock file."""
  path = _REPO_ROOT / "WORKSPACE"
  expected = constants["SUPPORTED_PYTHON_VERSIONS"]

  block = re.search(
      r"requirements\s*=\s*\{(.*?)\}", _read_text(path), re.DOTALL
  )
  if not block:
    return ["WORKSPACE: could not find the `requirements = {...}` mapping."]

  actual = re.findall(r'"(\d+\.\d+)"\s*:', block.group(1))
  if actual != expected:
    return [
        f"WORKSPACE: requirements mapping covers {actual} but {_SOT_LABEL} "
        f"says {expected}."
    ]
  return []


def check_copybara_required_checks(constants: dict) -> list[str]:
  """copy.bara.sky's required checks must match the presubmit matrix.

  The GitHub check name embeds the Python version because the workflow's job
  name does, so pruning a version from the presubmit matrix without editing
  this file leaves Copybara blocking on a check that will never run again --
  and adding one leaves the new job ungated.
  """
  if not _COPYBARA_CONFIG.is_file():
    # Not part of the GitHub export, so there is nothing to verify here.
    return []

  source = _read_text(_COPYBARA_CONFIG)
  names = re.findall(r'get_github_check_run\(\s*"([^"]+)"', source)

  cpu = constants["PRESUBMIT_CPU_PYTHON_VERSIONS"]
  tpu = constants["PRESUBMIT_TPU_PYTHON_VERSIONS"]
  default = constants["DEFAULT_PYTHON_VERSION"]

  # Job-name prefix -> the versions that job is expected to run for.
  expected_by_prefix = {
      "Build wheel on CPU": cpu,
      "Test wheel on CPU": cpu,
      "Test wheel on TPU v7 1-Chip": tpu,
      "Test wheel on TPU v7": tpu,
      "Test wheel on CPU (CUDA wheel)": [default],
      "Test wheel on CPU (PyTorch Nightly)": [default],
      "Test wheel on TPU v7 (PyTorch Nightly)": [default],
  }

  errors = []
  for prefix, expected_versions in expected_by_prefix.items():
    pattern = re.compile(rf"^{re.escape(prefix)} \(Python (\d+\.\d+)\)$")
    actual = sorted(
        match.group(1)
        for match in (pattern.match(name) for name in names)
        if match
    )
    if actual != sorted(expected_versions):
      errors.append(
          f"copy.bara.sky: required checks for {prefix!r} cover {actual} but "
          f"the presubmit matrix in {_SOT_LABEL} implies "
          f"{sorted(expected_versions)}. Update the get_github_check_run "
          "entries to match the jobs build_wheel.yml will actually produce."
      )
  return errors


_CHECKS: Final[tuple] = (
    check_module_bazel,
    check_run_benchmarks_choices,
    check_bazelrc,
    check_workspace_requirements,
    check_copybara_required_checks,
)


def run_all_checks() -> list[str]:
  constants = python_versions.load_constants()
  errors = []
  for check in _CHECKS:
    try:
      errors.extend(check(constants))
    except MissingInputError as error:
      # Report and keep going: one undeclared input should not mask drift
      # that the remaining checks would have caught.
      errors.append(str(error))
  return errors


def main() -> int:
  errors = run_all_checks()
  if errors:
    print(
        f"Python version definitions have drifted from {_SOT_LABEL}:\n",
        file=sys.stderr,
    )
    for error in errors:
      print(f"::error::{error}", file=sys.stderr)
    return 1
  print(f"All hand-maintained Python version copies match {_SOT_LABEL}.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
