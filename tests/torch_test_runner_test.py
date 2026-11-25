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

"""Unit tests for torch_test_runner runfiles and env var discovery."""

import os
import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
from tests.torch import torch_test_runner


class TorchTestRunnerTest(absltest.TestCase):  # ABSLTEST_OK=runner unit test

  def test_find_runfiles_test_dir_matches_canonical_repo_names(self) -> None:
    for repo_name in (
        "+pytorch_test_repo+pytorch_test",
        "+_repo_rules+pytorch_test",
        "pytorch_test",
    ):
      with self.subTest(repo_name=repo_name):
        with tempfile.TemporaryDirectory() as tmp:
          test_dir = pathlib.Path(tmp) / repo_name / "test"
          test_dir.mkdir(parents=True)
          (test_dir / "test_decomp.py").write_text("", encoding="utf-8")
          with mock.patch.dict(os.environ, {"TEST_SRCDIR": tmp}, clear=True):
            found = torch_test_runner._find_runfiles_test_dir("test_decomp")
            self.assertEqual(found, test_dir)

  def test_find_runfiles_test_dir_supports_runfiles_dir_fallback(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      test_dir = pathlib.Path(tmp) / "+pytorch_test_repo+pytorch_test" / "test"
      test_dir.mkdir(parents=True)
      (test_dir / "test_decomp.py").write_text("", encoding="utf-8")
      with mock.patch.dict(os.environ, {"RUNFILES_DIR": tmp}, clear=True):
        self.assertEqual(
            torch_test_runner._find_runfiles_test_dir("test_decomp"),
            test_dir,
        )

  def test_find_runfiles_test_dir_ignores_suffix_collision(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      bogus_dir = pathlib.Path(tmp) / "+ext+my_pytorch_test" / "test"
      bogus_dir.mkdir(parents=True)
      (bogus_dir / "test_decomp.py").write_text("", encoding="utf-8")
      with mock.patch.dict(os.environ, {"TEST_SRCDIR": tmp}, clear=True):
        self.assertIsNone(
            torch_test_runner._find_runfiles_test_dir("test_decomp")
        )

  def test_parse_test_env_vars_falls_back_to_runfiles(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      test_dir = pathlib.Path(tmp) / "+pytorch_test_repo+pytorch_test" / "test"
      test_dir.mkdir(parents=True)
      mod_file = test_dir / "test_decomp.py"
      mod_file.write_text("", encoding="utf-8")
      env = {
          "TORCH_TPU_INTERNAL_TORCH_TEST_MODULE": "test_decomp",
          "TEST_SRCDIR": tmp,
      }
      with mock.patch.dict(os.environ, env, clear=True):
        mod, resolved_dir, resolved_file = (
            torch_test_runner._parse_test_env_vars()
        )
        self.assertEqual(mod, "test_decomp")
        self.assertEqual(resolved_dir, test_dir.absolute())
        self.assertEqual(resolved_file, mod_file.absolute())

  def test_parse_test_env_vars_env_override_wins_over_runfiles(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      runfiles_dir = (
          pathlib.Path(tmp) / "+pytorch_test_repo+pytorch_test" / "test"
      )
      runfiles_dir.mkdir(parents=True)
      (runfiles_dir / "test_decomp.py").write_text("", encoding="utf-8")

      override_dir = pathlib.Path(tmp) / "override_test"
      override_dir.mkdir(parents=True)
      override_file = override_dir / "test_decomp.py"
      override_file.write_text("", encoding="utf-8")

      env = {
          "TORCH_TPU_INTERNAL_TORCH_TEST_MODULE": "test_decomp",
          "TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR": str(override_dir),
          "TEST_SRCDIR": tmp,
      }
      with mock.patch.dict(os.environ, env, clear=True):
        mod, resolved_dir, resolved_file = (
            torch_test_runner._parse_test_env_vars()
        )
        self.assertEqual(mod, "test_decomp")
        self.assertEqual(resolved_dir, override_dir.absolute())
        self.assertEqual(resolved_file, override_file.absolute())

  def test_parse_test_env_vars_raises_when_missing_in_env_and_runfiles(
      self,
  ) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      env = {
          "TORCH_TPU_INTERNAL_TORCH_TEST_MODULE": "test_decomp",
          "TEST_SRCDIR": tmp,
      }
      with mock.patch.dict(os.environ, env, clear=True):
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing runner env error.
            EnvironmentError,
            "not found in Bazel"
            " runfiles.*--repo_env=TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR",
        ):
          torch_test_runner._parse_test_env_vars()


if __name__ == "__main__":
  absltest.main()
