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
"""Unit tests for check_dependency_caveats."""

from collections.abc import Sequence
import os
import subprocess
import unittest
from unittest import mock

from ci.tools import check_dependency_caveats


class _FakeUv:
  """Callable fake for `subprocess.run` simulating `uv pip compile` calls."""

  def __init__(self, succeeding_targets: set[str] | None = None) -> None:
    self.succeeding_targets = succeeding_targets or set()
    self.calls: list[tuple[tuple[str, ...], str | None]] = []

  def __call__(
      self,
      cmd: Sequence[str],
      *,
      input: str | None = None,  # pylint: disable=redefined-builtin
      capture_output: bool = False,
      text: bool = False,
      check: bool = False,
  ) -> subprocess.CompletedProcess[str]:
    del capture_output, text, check
    cmd_tuple = tuple(cmd)
    self.calls.append((cmd_tuple, input))
    target = input.strip() if input is not None else cmd_tuple[3]
    if target in self.succeeding_targets:
      return subprocess.CompletedProcess(
          args=cmd_tuple, returncode=0, stdout="", stderr=""
      )
    return subprocess.CompletedProcess(
        args=cmd_tuple,
        returncode=1,
        stdout="",
        stderr="error: No solution found when resolving dependencies",
    )


class CheckDependencyCaveatsTest(unittest.TestCase):

  def test_dependency_caveat_requires_exactly_one_target(self) -> None:
    with self.assertRaises(ValueError):
      check_dependency_caveats.DependencyCaveat(
          name="neither",
          remediation="Remove workaround.",
      )
    with self.assertRaises(ValueError):
      check_dependency_caveats.DependencyCaveat(
          name="both",
          requirement="pkg>=1.0",
          input_file="pyproject.toml",
          remediation="Remove workaround.",
      )

  def test_build_uv_command_with_requirement(self) -> None:
    caveat = check_dependency_caveats.DependencyCaveat(
        name="tensorstore",
        requirement="tensorstore",
        extra_args=("--only-binary", ":all:"),
        remediation="Remove workaround.",
    )
    self.assertEqual(
        check_dependency_caveats.build_uv_command(
            caveat,
            python_version="3.15",
            python_platform="x86_64-manylinux_2_31",
        ),
        (
            "uv",
            "pip",
            "compile",
            "-",
            "--python-version",
            "3.15",
            "--python-platform",
            "x86_64-manylinux_2_31",
            "--resolution",
            "highest",
            "--only-binary",
            ":all:",
            "--output-file",
            os.devnull,
        ),
    )

  def test_build_uv_command_with_input_file(self) -> None:
    caveat = check_dependency_caveats.DependencyCaveat(
        name="pytorch-cuda-pypi",
        input_file="pyproject.toml",
        extra_args=("--no-sources", "--index-url", "https://pypi.org/simple"),
        remediation="Remove workaround.",
    )
    self.assertEqual(
        check_dependency_caveats.build_uv_command(
            caveat,
            python_version="3.15",
            python_platform="x86_64-manylinux_2_31",
        ),
        (
            "uv",
            "pip",
            "compile",
            "pyproject.toml",
            "--python-version",
            "3.15",
            "--python-platform",
            "x86_64-manylinux_2_31",
            "--resolution",
            "highest",
            "--no-sources",
            "--index-url",
            "https://pypi.org/simple",
            "--output-file",
            os.devnull,
        ),
    )

  def test_all_caveats_still_hold_returns_zero(self) -> None:
    fake_uv = _FakeUv()
    with mock.patch.object(
        check_dependency_caveats.subprocess, "run", side_effect=fake_uv
    ):
      rc = check_dependency_caveats.main(["--python-version", "3.15"])

    self.assertEqual(rc, 0)
    self.assertEqual(
        len(fake_uv.calls),
        len(check_dependency_caveats.CAVEATS_BY_PYTHON_VERSION["3.15"]),
    )

  def test_resolved_caveats_logged_and_returns_nonzero(self) -> None:
    # Even when an earlier caveat resolves, all caveats for the version are
    # checked so a single CI run surfaces every newly available dependency.
    fake_uv = _FakeUv(succeeding_targets={"grain>=0.2.16", "tensorstore"})
    with (
        mock.patch.object(
            check_dependency_caveats.subprocess, "run", side_effect=fake_uv
        ),
        self.assertLogs(level="ERROR") as logs,
    ):
      rc = check_dependency_caveats.main(["--python-version", "3.15"])

    self.assertEqual(rc, 1)
    self.assertEqual(
        len(fake_uv.calls),
        len(check_dependency_caveats.CAVEATS_BY_PYTHON_VERSION["3.15"]),
    )
    combined_logs = "\n".join(logs.output)
    self.assertIn("::error::'grain>=0.2.16' now resolves", combined_logs)
    self.assertIn("::error::'tensorstore' now publishes", combined_logs)

  def test_unconfigured_python_version_is_noop(self) -> None:
    fake_uv = _FakeUv()
    with mock.patch.object(
        check_dependency_caveats.subprocess, "run", side_effect=fake_uv
    ):
      rc = check_dependency_caveats.main(["--python-version", "3.12"])

    self.assertEqual(rc, 0)
    self.assertEqual(fake_uv.calls, [])


if __name__ == "__main__":
  unittest.main()
