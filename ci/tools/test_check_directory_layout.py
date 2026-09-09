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

"""Unit tests for check_directory_layout.py."""

import pathlib
import sys
import tempfile
import unittest

_CI_TOOLS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _CI_TOOLS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from ci.tools import check_directory_layout


class CheckDirectoryLayoutTest(
    unittest.TestCase  # UNITTEST_OK=CI utility script test without torch_tpu dependencies
):

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.repo_root = pathlib.Path(self.temp_dir.name)

  def tearDown(self):
    self.temp_dir.cleanup()
    super().tearDown()

  def _create_valid_layout(self) -> None:
    for d in check_directory_layout.EXPECTED_ROOT_DIRS:
      (self.repo_root / d).mkdir(parents=True, exist_ok=True)
    src_torch_tpu = self.repo_root / "src" / "torch_tpu"
    for d in check_directory_layout.EXPECTED_SRC_DIRS:
      (src_torch_tpu / d).mkdir(parents=True, exist_ok=True)

  def test_valid_layout_passes(self):
    self._create_valid_layout()
    self.assertTrue(
        check_directory_layout.validate_directory_layout(self.repo_root)
    )

  def test_hidden_directories_ignored(self):
    self._create_valid_layout()
    (self.repo_root / ".git").mkdir()
    (self.repo_root / ".github").mkdir()
    (self.repo_root / ".nox").mkdir()
    (self.repo_root / "src" / "torch_tpu" / ".cache").mkdir()
    self.assertTrue(
        check_directory_layout.validate_directory_layout(self.repo_root)
    )

  def test_pycache_and_symlinks_ignored(self):
    self._create_valid_layout()
    (self.repo_root / "__pycache__").mkdir()
    (self.repo_root / "src" / "torch_tpu" / "__pycache__").mkdir()
    # Create a symlink to a directory (e.g. simulating bazel-out or bazel-bin)
    target_dir = self.repo_root / "ci"
    (self.repo_root / "bazel-bin").symlink_to(
        target_dir, target_is_directory=True
    )
    self.assertTrue(
        check_directory_layout.validate_directory_layout(self.repo_root)
    )

  def test_missing_root_directory_fails(self):
    self._create_valid_layout()
    (self.repo_root / "csrc").rmdir()
    self.assertFalse(
        check_directory_layout.validate_directory_layout(self.repo_root)
    )

  def test_unexpected_root_directory_fails(self):
    self._create_valid_layout()
    (self.repo_root / "unexpected_root_dir").mkdir()
    self.assertFalse(
        check_directory_layout.validate_directory_layout(self.repo_root)
    )

  def test_missing_src_directory_fails(self):
    self._create_valid_layout()
    (self.repo_root / "src" / "torch_tpu" / "ops").rmdir()
    self.assertFalse(
        check_directory_layout.validate_directory_layout(self.repo_root)
    )

  def test_unexpected_src_directory_fails(self):
    self._create_valid_layout()
    (self.repo_root / "src" / "torch_tpu" / "unexpected_src_dir").mkdir()
    self.assertFalse(
        check_directory_layout.validate_directory_layout(self.repo_root)
    )

  def test_missing_src_torch_tpu_fails(self):
    for d in check_directory_layout.EXPECTED_ROOT_DIRS:
      if d != "src":
        (self.repo_root / d).mkdir(parents=True, exist_ok=True)
    self.assertFalse(
        check_directory_layout.validate_directory_layout(self.repo_root)
    )

  def test_main_cli_success(self):
    self._create_valid_layout()
    self.assertEqual(
        check_directory_layout.main(["--repo-root", str(self.repo_root)]),
        0,
    )

  def test_main_cli_failure(self):
    self.assertEqual(
        check_directory_layout.main(["--repo-root", str(self.repo_root)]),
        1,
    )


if __name__ == "__main__":
  unittest.main()
