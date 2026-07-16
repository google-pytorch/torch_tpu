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

"""Tests for the version-dispatch selection logic."""

import pathlib
import sys
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from torch_tpu import _versioned_so_loader as loader


class ResolveSuffixTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # Unversioned build: dispatch is disabled regardless of running version.
      ("no_built_glues_disables_dispatch", "2_12_0", [], None),
      ("no_built_glues_and_no_version_disables_dispatch", None, [], None),
      (
          "exact_match_is_chosen",
          "2_12_0",
          {"2_11_0", "2_12_0", "2_13_0"},
          "2_12_0",
      ),
      # 2.12.5 postdates the 2.12.0 glue and predates the 2.13.0 one, so the
      # 2.12.0 glue (which serves everything up to the next ABI break) is
      # chosen.
      (
          "release_between_glues_gets_the_newest_older_glue",
          "2_12_5",
          {"2_11_0", "2_12_0", "2_13_0"},
          "2_12_0",
      ),
      (
          "release_newer_than_every_glue_gets_the_newest",
          "2_14_0",
          {"2_11_0", "2_12_0"},
          "2_12_0",
      ),
      # "2_11_10" > "2_11_9" numerically although not lexicographically.
      (
          "comparison_is_numeric_not_lexicographic",
          "2_11_10",
          {"2_11_9", "2_11_10"},
          "2_11_10",
      ),
      (
          "comparison_is_numeric_below_the_running_version",
          "2_11_12",
          {"2_11_2", "2_11_10"},
          "2_11_10",
      ),
  )
  def test_resolution(self, running, built, expected):
    self.assertEqual(loader.resolve_suffix(running, built), expected)

  @parameterized.named_parameters(
      ("release_older_than_every_glue", "2_9_0", "predates"),
      ("undetectable_running_version", None, "could not determine"),
  )
  def test_raises_import_error(self, running, message):
    with self.assertRaisesRegex(ImportError, message):
      loader.resolve_suffix(running, {"2_11_0", "2_12_0"})


class VersionSuffixTest(parameterized.TestCase):

  @parameterized.parameters(
      ("2.13.0", "2_13_0"),
      ("2.13.0+cpu", "2_13_0"),
      # A dev build's suffix is the release triple it leads up to, so it is
      # served by the glue built against the matching dev snapshot.
      ("2.14.0.dev20260711+cpu", "2_14_0"),
      ("2.11.10", "2_11_10"),
      # Undeterminable versions disable rather than misdirect the dispatch.
      ("2.13", None),
      ("main", None),
  )
  def test_version_suffix(self, version, expected):
    self.assertEqual(loader.version_suffix(version), expected)


class InstalledTorchVersionTest(absltest.TestCase):

  def test_prefers_the_imported_torch_over_install_metadata(self):
    # The glue must match the torch actually loaded in this process, which the
    # install metadata need not reflect.
    fake_torch = types.SimpleNamespace(__version__="2.12.5+fake")
    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
      self.assertEqual(loader.installed_torch_version(), "2.12.5+fake")


class DiscoverBuiltSuffixesTest(absltest.TestCase):

  def _touch(self, root: pathlib.Path, rel: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

  def test_finds_versioned_glues_recursively_and_ignores_others(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    self._touch(root, "_internal/env_2_11_0.so")
    self._touch(root, "_internal/env_2_12_0.so")
    self._touch(root, "_internal/sync/_tpu_torch_sync_2_11_0.so")
    self._touch(root, "_internal/env.py")  # not a .so
    self._touch(root, "common/libpywrap_torch_tpu_common.so")  # unversioned
    self._touch(root, "common/libpywrap_2_11_0_common.so")  # not a glue

    self.assertEqual(
        loader.discover_built_suffixes(root), frozenset({"2_11_0", "2_12_0"})
    )

  def test_no_versioned_glues_returns_empty(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    self._touch(root, "_internal/env.so")
    self.assertEqual(loader.discover_built_suffixes(root), frozenset())


if __name__ == "__main__":
  absltest.main()
