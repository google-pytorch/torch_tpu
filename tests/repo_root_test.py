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

"""Tests for `repo_root`."""

import os
import pathlib
from unittest import mock

from absl.testing import absltest
from tests import repo_root
from tests import seed_test_utils

_MARKER = os.path.join("docs", "markdown", "marker.md")

# The environment variables that `repo_root.find` reads.
_ENV_VARS = ("BUILD_WORKSPACE_DIRECTORY", "TEST_SRCDIR", "RUNFILES_DIR")


def _make_repo(path: pathlib.Path) -> pathlib.Path:
  """Creates a fake TorchTPU repo with the marker file at `path`."""
  (path / _MARKER).parent.mkdir(parents=True)
  (path / _MARKER).write_text("")
  return path


class RepoRootTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.tmp = pathlib.Path(self.create_tempdir().full_path)
    # Start every test from an environment without any of the variables, so
    # that the ones Blaze or Bazel set for this test don't leak in.
    env = {k: v for k, v in os.environ.items() if k not in _ENV_VARS}
    self.enter_context(mock.patch.dict(os.environ, env, clear=True))

  def test_finds_blaze_runfiles_layout(self):
    root = _make_repo(self.tmp / "google3" / "third_party" / "py" / "torch_tpu")
    os.environ["TEST_SRCDIR"] = str(self.tmp)

    found = repo_root.find(_MARKER)

    self.assertEqual(found.path, root)
    self.assertEqual(found.layout, repo_root.Layout.GOOGLE3)
    self.assertFalse(found.in_workspace)

  def test_finds_bazel_bzlmod_runfiles_layout(self):
    root = _make_repo(self.tmp / "_main")
    os.environ["TEST_SRCDIR"] = str(self.tmp)

    found = repo_root.find(_MARKER)

    self.assertEqual(found.path, root)
    self.assertEqual(found.layout, repo_root.Layout.OSS)
    self.assertFalse(found.in_workspace)

  def test_finds_bazel_workspace_runfiles_layout(self):
    root = _make_repo(self.tmp / "torch_tpu")
    os.environ["RUNFILES_DIR"] = str(self.tmp)

    found = repo_root.find(_MARKER)

    self.assertEqual(found.path, root)
    self.assertEqual(found.layout, repo_root.Layout.OSS)
    self.assertFalse(found.in_workspace)

  def test_test_srcdir_takes_precedence_over_runfiles_dir(self):
    root = _make_repo(self.tmp / "srcdir" / "_main")
    _make_repo(self.tmp / "runfiles" / "_main")
    os.environ["TEST_SRCDIR"] = str(self.tmp / "srcdir")
    os.environ["RUNFILES_DIR"] = str(self.tmp / "runfiles")

    self.assertEqual(repo_root.find(_MARKER).path, root)

  def test_skips_candidates_without_the_marker(self):
    # `_main` exists, but only `torch_tpu` has the marker.
    (self.tmp / "_main" / "docs").mkdir(parents=True)
    root = _make_repo(self.tmp / "torch_tpu")
    os.environ["TEST_SRCDIR"] = str(self.tmp)

    self.assertEqual(repo_root.find(_MARKER).path, root)

  def test_finds_google3_workspace(self):
    root = _make_repo(self.tmp / "ws" / "third_party" / "py" / "torch_tpu")
    _make_repo(self.tmp / "runfiles" / "_main")
    os.environ["BUILD_WORKSPACE_DIRECTORY"] = str(self.tmp / "ws")
    os.environ["RUNFILES_DIR"] = str(self.tmp / "runfiles")

    found = repo_root.find(_MARKER, include_workspace=True)

    self.assertEqual(found.path, root)
    self.assertEqual(found.layout, repo_root.Layout.GOOGLE3)
    self.assertTrue(found.in_workspace)

  def test_finds_oss_workspace(self):
    root = _make_repo(self.tmp / "ws")
    os.environ["BUILD_WORKSPACE_DIRECTORY"] = str(root)

    found = repo_root.find(_MARKER, include_workspace=True)

    self.assertEqual(found.path, root)
    self.assertEqual(found.layout, repo_root.Layout.OSS)
    self.assertTrue(found.in_workspace)

  def test_falls_back_to_runfiles_if_workspace_lacks_marker(self):
    (self.tmp / "ws").mkdir()
    root = _make_repo(self.tmp / "runfiles" / "_main")
    os.environ["BUILD_WORKSPACE_DIRECTORY"] = str(self.tmp / "ws")
    os.environ["RUNFILES_DIR"] = str(self.tmp / "runfiles")

    found = repo_root.find(_MARKER, include_workspace=True)

    self.assertEqual(found.path, root)
    self.assertFalse(found.in_workspace)

  def test_ignores_workspace_by_default(self):
    _make_repo(self.tmp / "ws")
    root = _make_repo(self.tmp / "runfiles" / "_main")
    os.environ["BUILD_WORKSPACE_DIRECTORY"] = str(self.tmp / "ws")
    os.environ["RUNFILES_DIR"] = str(self.tmp / "runfiles")

    found = repo_root.find(_MARKER)

    self.assertEqual(found.path, root)
    self.assertFalse(found.in_workspace)

  def test_raises_if_not_found(self):
    _make_repo(self.tmp / "elsewhere")
    os.environ["TEST_SRCDIR"] = str(self.tmp)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Test tooling error.
        FileNotFoundError, "Could not find torch_tpu repository root"
    ):
      repo_root.find(_MARKER)

  def test_raises_without_environment(self):
    with self.assertRaises(  # ASSERT_RAISES_OK=Test tooling error.
        FileNotFoundError
    ):
      repo_root.find(_MARKER)

  def test_is_ignored_dir(self):
    for name in (
        "_cpp_style_test.venv",
        "site-packages",
        "dist-packages",
        "venv",
        ".venv",
        ".git",
        "torch_tpu_wheel_test_unpacked_wheel",
    ):
      with self.subTest(name=name):
        self.assertTrue(repo_root.is_ignored_dir(name))
    for name in ("csrc", "src", "torch_tpu", "packages"):
      with self.subTest(name=name):
        self.assertFalse(repo_root.is_ignored_dir(name))


if __name__ == "__main__":
  absltest.main()
