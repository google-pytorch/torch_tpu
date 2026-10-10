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

Each configured `DependencyCaveat` represents an upstream package or index that
currently fails to resolve for a given Python version via `uv pip compile`,
requiring a temporary workaround in `MODULE.bazel`,
`requirements/lock_environments.sh`, `shims/`, or CI workflows.

When upstream begins publishing compatible wheels or releases, `uv pip compile`
succeeds and this script fails with a GitHub Actions `::error::` annotation
describing which workaround can now be removed.
"""

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import logging  # PYTHON_LOGGING_OK=CI utility script running in standard Python environment
import os
import subprocess
import sys
from typing import Final

DEFAULT_PYTHON_PLATFORM: Final[str] = "x86_64-manylinux_2_31"


@dataclasses.dataclass(frozen=True)
class DependencyCaveat:
  """A temporary dependency workaround expected to fail resolution via `uv`.

  Attributes:
    name: Short identifier for logging.
    remediation: Actionable error message emitted when the target now resolves.
    requirement: Inline requirement specifier passed via stdin to `uv pip
      compile -`. Mutually exclusive with `input_file`.
    input_file: Requirement or project file (e.g. `pyproject.toml`) passed to
      `uv pip compile`. Mutually exclusive with `requirement`.
    extra_args: Additional flags passed to `uv pip compile` (e.g. `--index-url`,
      `--no-sources`, `--only-binary`).
  """

  name: str
  remediation: str
  requirement: str | None = None
  input_file: str | None = None
  extra_args: Sequence[str] = ()

  def __post_init__(self) -> None:
    if (self.requirement is None) == (self.input_file is None):
      raise ValueError(
          "Exactly one of `requirement` or `input_file` must be specified."
      )


CAVEATS_BY_PYTHON_VERSION: Final[Mapping[str, Sequence[DependencyCaveat]]] = {
    "3.15": (
        DependencyCaveat(
            name="pytorch-cuda-pypi",
            input_file="pyproject.toml",
            extra_args=(
                "--no-sources",
                "--index-url",
                "https://pypi.org/simple",
            ),
            remediation=(
                "PyTorch now publishes Python 3.15 CUDA wheels on PyPI. Enable"
                " Python 3.15 in torch_tpu_cuda_pypi (MODULE.bazel) and"
                " requirements/lock_environments.sh, and remove this caveat"
                " from ci/tools/check_dependency_caveats.py."
            ),
        ),
        DependencyCaveat(
            name="pytorch-2.15-cpu",
            requirement="torch>=2.15.0",
            extra_args=(
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ),
            remediation=(
                "PyTorch 2.15 is now released on the CPU wheel index. Remove"
                " the PyTorch test index workaround for Python 3.15 in"
                " requirements/lock_environments.sh and"
                " .github/workflows/nightly_unpinned_deps.yml, and remove this"
                " caveat from ci/tools/check_dependency_caveats.py."
            ),
        ),
        DependencyCaveat(
            name="grain",
            requirement="grain>=0.2.16",
            remediation=(
                "'grain>=0.2.16' now resolves on Python 3.15. Remove it from"
                " the Python 3.15 overrides in"
                " requirements/lock_environments.sh and"
                " .github/workflows/nightly_unpinned_deps.yml, and remove this"
                " caveat from ci/tools/check_dependency_caveats.py."
            ),
        ),
        DependencyCaveat(
            name="tensorflow-datasets",
            requirement="tensorflow-datasets>=4.9.9",
            remediation=(
                "'tensorflow-datasets>=4.9.9' now resolves on Python 3.15."
                " Remove it from the Python 3.15 overrides in"
                " requirements/lock_environments.sh and"
                " .github/workflows/nightly_unpinned_deps.yml, and remove this"
                " caveat from ci/tools/check_dependency_caveats.py."
            ),
        ),
        DependencyCaveat(
            name="tensorstore",
            requirement="tensorstore",
            extra_args=("--only-binary", ":all:"),
            remediation=(
                "'tensorstore' now publishes Python 3.15 wheels on PyPI. Remove"
                " 'flax ; python_version < 3.15' from the Python 3.15 overrides"
                " in requirements/lock_environments.sh and"
                " .github/workflows/nightly_unpinned_deps.yml, remove the"
                " is_python_3.15 select() workaround in"
                " shims/flax_nnx/BUILD.bazel, and remove this caveat from"
                " ci/tools/check_dependency_caveats.py."
            ),
        ),
        DependencyCaveat(
            name="dm-tree",
            requirement="dm-tree>=0.1.10",
            extra_args=("--only-binary", ":all:"),
            remediation=(
                "'dm-tree>=0.1.10' now publishes Python 3.15 wheels on PyPI."
                " Remove it from the Python 3.15 overrides in"
                " requirements/lock_environments.sh and"
                " .github/workflows/nightly_unpinned_deps.yml, and remove this"
                " caveat from ci/tools/check_dependency_caveats.py."
            ),
        ),
        DependencyCaveat(
            name="datasets",
            requirement="datasets>=4.8",
            extra_args=("--only-binary", ":all:"),
            remediation=(
                "'datasets>=4.8' and all of its dependencies (including"
                " pyarrow and aiohttp) now publish Python 3.15 wheels on PyPI."
                " Remove 'datasets ; python_version < 3.15' from the Python"
                " 3.15 overrides in requirements/lock_environments.sh and"
                " .github/workflows/nightly_unpinned_deps.yml, and remove this"
                " caveat from ci/tools/check_dependency_caveats.py."
            ),
        ),
        DependencyCaveat(
            name="torchax",
            requirement="torchax>=0.0.13",
            remediation=(
                "'torchax>=0.0.13' is now released on PyPI. Update 'torchax'"
                " in pyproject.toml, remove the 'torchax' override for Python"
                " 3.15 in requirements/lock_environments.sh and"
                " .github/workflows/nightly_unpinned_deps.yml, and remove this"
                " caveat from ci/tools/check_dependency_caveats.py."
            ),
        ),
    ),
}


def build_uv_command(
    caveat: DependencyCaveat,
    python_version: str,
    python_platform: str = DEFAULT_PYTHON_PLATFORM,
) -> tuple[str, ...]:
  """Builds the `uv pip compile` command for probing `caveat`."""
  target = caveat.input_file if caveat.input_file is not None else "-"
  return (
      "uv",
      "pip",
      "compile",
      target,
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
    python_platform: str = DEFAULT_PYTHON_PLATFORM,
) -> bool:
  """Returns True if `caveat` still fails to resolve, or False if it resolves."""
  cmd = build_uv_command(
      caveat,
      python_version=python_version,
      python_platform=python_platform,
  )
  stdin_text = (
      f"{caveat.requirement}\n" if caveat.requirement is not None else None
  )
  result = subprocess.run(
      cmd,
      input=stdin_text,
      capture_output=True,
      text=True,
      check=False,
  )
  if result.returncode == 0:
    logging.error("::error::%s", caveat.remediation)
    return False

  logging.info(
      "Python %s caveat %r still holds (upstream not yet available).",
      python_version,
      caveat.name,
  )
  return True


def check_caveats_for_version(
    python_version: str,
    caveats_by_version: Mapping[
        str, Sequence[DependencyCaveat]
    ] = CAVEATS_BY_PYTHON_VERSION,
    python_platform: str = DEFAULT_PYTHON_PLATFORM,
) -> tuple[DependencyCaveat, ...]:
  """Checks all caveats for `python_version` and returns any that now resolve."""
  caveats = caveats_by_version.get(python_version, ())
  if not caveats:
    logging.info(
        "No dependency caveats configured for Python %s.", python_version
    )
    return ()

  resolved_caveats: list[DependencyCaveat] = []
  for caveat in caveats:
    if not check_caveat(
        caveat,
        python_version=python_version,
        python_platform=python_platform,
    ):
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
      default=DEFAULT_PYTHON_PLATFORM,
      help=(
          "Target platform passed to `uv pip compile --python-platform`"
          f" (default: {DEFAULT_PYTHON_PLATFORM})."
      ),
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
    if check_caveats_for_version(
        version,
        python_platform=parsed.python_platform,
    ):
      any_resolved = True

  return 1 if any_resolved else 0


if __name__ == "__main__":
  sys.exit(main())
