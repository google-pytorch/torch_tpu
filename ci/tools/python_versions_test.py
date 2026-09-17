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

"""Tests for ci/tools/python_versions.py."""

import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

_CI_TOOLS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _CI_TOOLS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from ci.tools import python_versions


class LoadConstantsTest(
    unittest.TestCase  # UNITTEST_OK=CI utility script test without torch_tpu dependencies
):

  def _write(self, source: str) -> pathlib.Path:
    tmpdir = tempfile.mkdtemp()
    path = pathlib.Path(tmpdir) / "supported_python_versions.bzl"
    path.write_text(source)
    return path

  def test_reads_lists_and_scalars(self):
    path = self._write(
        '"""Docstring."""\nVERSIONS = ["3.11", "3.12"]\nDEFAULT = "3.12"\n'
    )
    self.assertEqual(
        python_versions.load_constants(path),
        {"VERSIONS": ["3.11", "3.12"], "DEFAULT": "3.12"},
    )

  def test_resolves_derived_constants(self):
    path = self._write(
        'VERSIONS = ["3.11", "3.12", "3.13"]\n'
        "OLDEST = VERSIONS[0]\n"
        "LATEST = VERSIONS[-1]\n"
        "SUBSET = [VERSIONS[0], VERSIONS[-1]]\n"
    )
    constants = python_versions.load_constants(path)
    self.assertEqual(constants["OLDEST"], "3.11")
    self.assertEqual(constants["LATEST"], "3.13")
    self.assertEqual(constants["SUBSET"], ["3.11", "3.13"])

  def test_rejects_load_statement(self):
    # A load() is valid Starlark but would break the Python read path, so it
    # must fail loudly rather than surfacing as a NameError.
    path = self._write('load("//bazel:other.bzl", "THING")\nA = "1"\n')
    with self.assertRaisesRegex(ValueError, "must not contain load"):
      python_versions.load_constants(path)

  def test_rejects_function_definition(self):
    path = self._write("def helper():\n  return 1\n")
    with self.assertRaisesRegex(ValueError, "UPPER_CASE"):
      python_versions.load_constants(path)

  def test_rejects_lowercase_assignment(self):
    path = self._write('sneaky = "value"\n')
    with self.assertRaisesRegex(ValueError, "UPPER_CASE"):
      python_versions.load_constants(path)

  def test_error_message_reports_line_number(self):
    path = self._write('A = "1"\nB = "2"\ndef bad():\n  pass\n')
    with self.assertRaisesRegex(ValueError, r":3:"):
      python_versions.load_constants(path)


class SourceOfTruthTest(
    unittest.TestCase  # UNITTEST_OK=CI utility script test without torch_tpu dependencies
):
  """Checks the invariants the real source of truth must uphold."""

  def setUp(self):
    super().setUp()
    self.constants = python_versions.load_constants()

  def test_declares_expected_constants(self):
    for name in (
        "SUPPORTED_PYTHON_VERSIONS",
        "DEFAULT_PYTHON_VERSION",
        "OLDEST_PYTHON_VERSION",
        "LATEST_PYTHON_VERSION",
        "PRESUBMIT_CPU_PYTHON_VERSIONS",
        "PRESUBMIT_TPU_PYTHON_VERSIONS",
        "POSTSUBMIT_PYTHON_VERSIONS",
    ):
      self.assertIn(name, self.constants)

  def test_versions_are_sorted_and_unique(self):
    versions = self.constants["SUPPORTED_PYTHON_VERSIONS"]
    as_tuples = [tuple(int(p) for p in v.split(".")) for v in versions]
    self.assertEqual(as_tuples, sorted(as_tuples))
    self.assertEqual(len(set(versions)), len(versions))

  def test_default_is_supported(self):
    self.assertIn(
        self.constants["DEFAULT_PYTHON_VERSION"],
        self.constants["SUPPORTED_PYTHON_VERSIONS"],
    )

  def test_every_subset_is_a_subset(self):
    supported = set(self.constants["SUPPORTED_PYTHON_VERSIONS"])
    for name in (
        "PRESUBMIT_CPU_PYTHON_VERSIONS",
        "PRESUBMIT_TPU_PYTHON_VERSIONS",
        "POSTSUBMIT_PYTHON_VERSIONS",
    ):
      self.assertTrue(set(self.constants[name]).issubset(supported), name)

  def test_presubmit_cpu_covers_both_wheel_flavors(self):
    # build_wheel.yml builds --config=wheel_release for the oldest version
    # and the nightly-glue flavor for the default one. If the presubmit set
    # ever stops containing both, a PR would silently stop exercising a
    # wheel flavor.
    presubmit = self.constants["PRESUBMIT_CPU_PYTHON_VERSIONS"]
    self.assertIn(self.constants["OLDEST_PYTHON_VERSION"], presubmit)
    self.assertIn(self.constants["DEFAULT_PYTHON_VERSION"], presubmit)

  def test_every_supported_version_has_lock_files(self):
    root = pathlib.Path(python_versions.__file__).resolve().parents[2]
    for version in self.constants["SUPPORTED_PYTHON_VERSIONS"]:
      suffix = version.replace(".", "_")
      for name in (
          f"requirements_{suffix}.txt",
          f"requirements_cuda_{suffix}.txt",
      ):
        self.assertTrue(
            (root / "requirements" / name).is_file(),
            f"{name} is missing for supported version {version}",
        )


class MatrixTest(
    unittest.TestCase  # UNITTEST_OK=CI utility script test without torch_tpu dependencies
):

  CONSTANTS = {
      "SUPPORTED_PYTHON_VERSIONS": ["3.11", "3.12", "3.13"],
      "POSTSUBMIT_PYTHON_VERSIONS": ["3.11", "3.12", "3.13"],
      "PRESUBMIT_CPU_PYTHON_VERSIONS": ["3.11", "3.13"],
      "PRESUBMIT_TPU_PYTHON_VERSIONS": ["3.13"],
      "DEFAULT_PYTHON_VERSION": "3.13",
      "LATEST_PYTHON_VERSION": "3.13",
      "OLDEST_PYTHON_VERSION": "3.11",
  }

  def test_presubmit_matrix(self):
    self.assertEqual(
        python_versions.matrix_for_event(
            self.CONSTANTS, run_all_versions=False
        ),
        {"cpu_versions": ["3.11", "3.13"], "tpu_versions": ["3.13"]},
    )

  def test_postsubmit_matrix(self):
    self.assertEqual(
        python_versions.matrix_for_event(self.CONSTANTS, run_all_versions=True),
        {
            "cpu_versions": ["3.11", "3.12", "3.13"],
            "tpu_versions": ["3.11", "3.12", "3.13"],
        },
    )

  def test_github_outputs_render_lists_as_json(self):
    outputs = python_versions.github_outputs(
        self.CONSTANTS, run_all_versions=False
    )
    # Workflows consume these with fromJSON(), so lists must be valid JSON
    # and scalars must be bare (no surrounding quotes).
    self.assertEqual(json.loads(outputs["cpu_versions"]), ["3.11", "3.13"])
    self.assertEqual(outputs["default_python_version"], "3.13")
    self.assertEqual(outputs["oldest_python_version"], "3.11")


class MainTest(
    unittest.TestCase  # UNITTEST_OK=CI utility script test without torch_tpu dependencies
):

  def test_prints_json_by_default(self):
    stdout = io.StringIO()
    with mock.patch("sys.stdout", stdout):
      self.assertEqual(python_versions.main(["SUPPORTED_PYTHON_VERSIONS"]), 0)
    self.assertIsInstance(json.loads(stdout.getvalue()), list)

  def test_plain_format_of_scalar_has_no_quotes(self):
    stdout = io.StringIO()
    with mock.patch("sys.stdout", stdout):
      python_versions.main(["DEFAULT_PYTHON_VERSION", "--format=plain"])
    self.assertNotIn('"', stdout.getvalue())

  def test_lines_format_is_newline_separated(self):
    stdout = io.StringIO()
    with mock.patch("sys.stdout", stdout):
      python_versions.main(["SUPPORTED_PYTHON_VERSIONS", "--format=lines"])
    lines = stdout.getvalue().split()
    self.assertGreater(len(lines), 1)
    self.assertTrue(all("." in line for line in lines))

  def test_unknown_constant_exits_nonzero(self):
    with self.assertRaises(SystemExit):
      with mock.patch("sys.stderr", io.StringIO()):
        python_versions.main(["NOT_A_CONSTANT"])

  def test_github_output_appends_to_file(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      output_path = pathlib.Path(tmpdir) / "gh_output"
      env = {"GITHUB_OUTPUT": str(output_path)}
      with mock.patch.dict(os.environ, env), mock.patch(
          "sys.stdout", io.StringIO()
      ):
        self.assertEqual(python_versions.main(["--github-output"]), 0)

      written = dict(
          line.split("=", 1) for line in output_path.read_text().splitlines()
      )
    self.assertIn("cpu_versions", written)
    self.assertIn("tpu_versions", written)
    self.assertIn("oldest_python_version", written)
    self.assertIsInstance(json.loads(written["cpu_versions"]), list)


if __name__ == "__main__":
  unittest.main()
