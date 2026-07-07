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
import warnings

from absl.testing import absltest
from absl.testing import parameterized
from torch_tpu import _versioned_so_loader as loader


class ResolveSuffixTest(parameterized.TestCase):

  def test_no_built_glues_disables_dispatch(self):
    # Unversioned build: dispatch is disabled regardless of running version.
    self.assertIsNone(loader.resolve_suffix("2_12_0", []))
    self.assertIsNone(loader.resolve_suffix(None, []))

  def test_exact_match_is_preferred(self):
    self.assertEqual(
        loader.resolve_suffix("2_12_0", {"2_11_0", "2_12_0", "2_13_0"}),
        "2_12_0",
    )

  def test_no_exact_match_falls_back_to_newest_patch_of_same_minor(self):
    with self.assertWarns(RuntimeWarning):
      resolved = loader.resolve_suffix("2_12_5", {"2_11_0", "2_12_0"})
    self.assertEqual(resolved, "2_12_0")

  def test_same_minor_is_preferred_over_a_newer_other_minor(self):
    # On 2.11.3, a 2.11.x glue is chosen even though 2.13.0 is newer overall.
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      resolved = loader.resolve_suffix(
          "2_11_3", {"2_11_9", "2_12_0", "2_13_0"}
      )
    self.assertEqual(resolved, "2_11_9")

  def test_unsupported_minor_falls_back_to_newest_overall(self):
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      resolved = loader.resolve_suffix("2_9_0", {"2_11_0", "2_12_0"})
    self.assertEqual(resolved, "2_12_0")

  def test_newest_is_by_semver_not_lexicographic(self):
    # No same-minor match (2.20) -> newest overall, compared numerically.
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      resolved = loader.resolve_suffix("2_20_0", {"2_11_0", "2_11_10", "2_9_0"})
    self.assertEqual(resolved, "2_11_10")

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
