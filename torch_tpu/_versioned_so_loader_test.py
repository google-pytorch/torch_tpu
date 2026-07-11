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

from absl.testing import absltest
from absl.testing import parameterized
from torch_tpu import _versioned_so_loader as loader


class ResolveSuffixTest(parameterized.TestCase):

  def test_no_built_glues_disables_dispatch(self):
    # Unversioned build: dispatch is disabled regardless of running version.
    self.assertIsNone(loader.resolve_suffix("2_12_0", []))
    self.assertIsNone(loader.resolve_suffix(None, []))

  def test_exact_match_is_chosen(self):
    self.assertEqual(
        loader.resolve_suffix("2_12_0", {"2_11_0", "2_12_0", "2_13_0"}),
        "2_12_0",
    )

  def test_release_between_glues_gets_the_newest_older_glue(self):
    # 2.12.5 postdates the 2.12.0 glue and predates the 2.13.0 one, so the
    # 2.12.0 glue (which serves everything up to the next ABI break) is chosen.
    self.assertEqual(
        loader.resolve_suffix("2_12_5", {"2_11_0", "2_12_0", "2_13_0"}),
        "2_12_0",
    )

  def test_release_newer_than_every_glue_gets_the_newest(self):
    self.assertEqual(
        loader.resolve_suffix("2_14_0", {"2_11_0", "2_12_0"}), "2_12_0"
    )

  def test_release_older_than_every_glue_raises(self):
    with self.assertRaisesRegex(ImportError, "predates"):
      loader.resolve_suffix("2_9_0", {"2_11_0", "2_12_0"})

  def test_comparison_is_by_semver_not_lexicographic(self):
    # "2_11_10" > "2_11_9" numerically although not lexicographically.
    self.assertEqual(
        loader.resolve_suffix("2_11_10", {"2_11_9", "2_11_10"}), "2_11_10"
    )
    self.assertEqual(
        loader.resolve_suffix("2_11_12", {"2_11_2", "2_11_10"}), "2_11_10"
    )

  def test_undetectable_running_version_with_built_glues_raises(self):
    with self.assertRaises(ImportError):
      loader.resolve_suffix(None, {"2_11_0", "2_12_0"})


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

    self.assertEqual(
        loader.discover_built_suffixes(root), frozenset({"2_11_0", "2_12_0"})
    )

  def test_no_versioned_glues_returns_empty(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    self._touch(root, "_internal/env.so")
    self.assertEqual(loader.discover_built_suffixes(root), frozenset())


if __name__ == "__main__":
  absltest.main()
