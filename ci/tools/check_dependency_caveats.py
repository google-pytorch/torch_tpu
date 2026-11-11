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
"""Verifies that per-Python-version dependency workarounds are still needed.

Each configured `DependencyCaveat` represents an upstream package that
currently fails to resolve for a given Python version via `uv pip compile`,
requiring a temporary override in `requirements/overrides_<ver>.txt` (and any
associated Bazel shim or `pyproject.toml` bound).

When upstream begins publishing compatible wheels or releases, `uv pip compile`
succeeds and this script fails with a GitHub Actions `::error::` annotation
describing which workaround can now be removed.

Usage:
  ./ci/tools/check_dependency_caveats.py
  ./ci/tools/check_dependency_caveats.py --python-version 3.15
"""

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import enum
import logging  # PYTHON_LOGGING_OK=CI utility script running in standard Python environment
import os
import subprocess
import sys
from typing import Final

_DEFAULT_PYTHON_PLATFORM: Final[str] = "x86_64-manylinux_2_31"
_NO_SOLUTION_MARKER: Final[str] = (
    "No solution found when resolving dependencies"
)


class CaveatStatus(enum.Enum):
  """Outcome of probing a single `DependencyCaveat` with `uv pip compile`."""

  STILL_HOLDS = "still_holds"
  RESOLVED = "resolved"


@dataclasses.dataclass(frozen=True)
class DependencyCaveat:
  """A temporary dependency workaround expected to fail resolution via `uv`.

  Attributes:
    name: Short package identifier matching `requirements/overrides_<ver>.txt`.
    requirement: Requirement specifier passed via stdin to `uv pip compile -`.
    extra_args: Additional flags passed to `uv pip compile` (e.g.
      `--only-binary`).
    extra_remediation: Optional additional cleanup instructions beyond removing
      the entry from `requirements/overrides_<ver>.txt` and this script.
  """

  name: str
  requirement: str
  extra_args: tuple[str, ...] = ()
  extra_remediation: str = ""

  def remediation(self, python_version: str) -> str:
    """Returns the actionable remediation message when the caveat resolves."""
    version_und = python_version.replace(".", "_")
    overrides_path = f"requirements/overrides_{version_und}.txt"
    extra = f" {self.extra_remediation}" if self.extra_remediation else ""
    return (
        f"'{self.requirement}' now resolves on Python {python_version}. Remove"
        f" the '{self.name}' override from {overrides_path},{extra} and remove"
        " this caveat from ci/tools/check_dependency_caveats.py."
    )


CAVEATS_BY_PYTHON_VERSION: Final[Mapping[str, Sequence[DependencyCaveat]]] = {
    "3.15": (
        DependencyCaveat(
            name="grain",
            requirement="grain>=0.2.16",
        ),
        DependencyCaveat(
            name="tensorflow-datasets",
            requirement="tensorflow-datasets>=4.9.9",
        ),
        DependencyCaveat(
            name="flax",
            requirement="flax>=0.12.3",
            extra_args=("--only-binary", ":all:"),
            extra_remediation=(
                "remove the is_python_3.15 target_compatible_with select() in"
                " shims/flax/BUILD.bazel,"
            ),
        ),
        DependencyCaveat(
            name="dm-tree",
            requirement="dm-tree>=0.1.10",
            extra_args=("--only-binary", ":all:"),
        ),
        DependencyCaveat(
            name="torchax",
            requirement="torchax>=0.0.14",
            extra_remediation="update 'torchax>=0.0.14' in pyproject.toml,",
        ),
    ),
}


def build_uv_command(
    caveat: DependencyCaveat,
    python_version: str,
    python_platform: str,
) -> tuple[str, ...]:
  """Builds the `uv pip compile` command for probing `caveat`."""
  return (
      "uv",
      "pip",
      "compile",
      "-",
      "--python-version",
      python_version,
      "--python-platform",
      python_platform,
      "--resolution",
      "highest",
      *caveat.extra_args,
      "--output-file",
      os.devnull,
  )


def check_caveat(
    caveat: DependencyCaveat,
    python_version: str,
    python_platform: str,
) -> CaveatStatus:
  """Probes `caveat` with `uv pip compile` and returns its `CaveatStatus`."""
  cmd = build_uv_command(
      caveat,
      python_version=python_version,
      python_platform=python_platform,
  )
  result = subprocess.run(
      cmd,
      input=f"{caveat.requirement}\n",
      capture_output=True,
      text=True,
      check=False,
  )
  if result.returncode == 0:
    return CaveatStatus.RESOLVED

  if _NO_SOLUTION_MARKER not in result.stderr:
    raise RuntimeError(
        f"Unexpected error running {' '.join(cmd)} for caveat"
        f" {caveat.name!r} (exit code {result.returncode}):\n{result.stderr}"
    )

  logging.info(
      "Python %s caveat %r still holds (upstream not yet available).",
      python_version,
      caveat.name,
  )
  return CaveatStatus.STILL_HOLDS


def check_caveats_for_version(
    python_version: str,
    python_platform: str,
) -> tuple[DependencyCaveat, ...]:
  """Checks all caveats for `python_version` and returns any that now resolve."""
  caveats = CAVEATS_BY_PYTHON_VERSION.get(python_version, ())
  if not caveats:
    logging.info(
        "No dependency caveats configured for Python %s.", python_version
    )
    return ()

  resolved_caveats: list[DependencyCaveat] = []
  for caveat in caveats:
    status = check_caveat(
        caveat,
        python_version=python_version,
        python_platform=python_platform,
    )
    if status is CaveatStatus.RESOLVED:
      resolved_caveats.append(caveat)
  return tuple(resolved_caveats)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description=(
          "Verify that per-Python-version dependency workarounds are still"
          " needed."
      ),
  )
  parser.add_argument(
      "--python-version",
      default=None,
      help=(
          "Python version (X.Y) to check caveats for. If omitted, checks all"
          " configured Python versions."
      ),
  )
  parser.add_argument(
      "--python-platform",
      default=_DEFAULT_PYTHON_PLATFORM,
      help="Target platform passed to `uv pip compile --python-platform`.",
  )
  return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  parsed = parse_args(args)

  versions = (
      (parsed.python_version,)
      if parsed.python_version is not None
      else tuple(CAVEATS_BY_PYTHON_VERSION.keys())
  )
  any_resolved = False
  for version in versions:
    for caveat in check_caveats_for_version(
        version,
        python_platform=parsed.python_platform,
    ):
      print(f"::error::{caveat.remediation(version)}")
      any_resolved = True

  return 1 if any_resolved else 0


if __name__ == "__main__":
  sys.exit(main())
