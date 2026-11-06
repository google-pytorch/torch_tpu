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

"""Unit tests for the test_utils module."""

import gzip
import importlib.resources.abc
import pathlib
import plistlib
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tests import goldens_pin
from tests import seed_test_utils
from tests import test_utils


class TestUtilsTest(seed_test_utils.RepeatableTest):

  def test_locate_goldens_pin_finds_file(self):
    pin_file = test_utils.locate_goldens_pinfile()
    self.assertTrue(pin_file.is_file())
    self.assertEqual(pin_file.name, "goldens.pin")

  def test_locate_goldens_pin_not_found(self):
    mock_traversable = mock.create_autospec(
        importlib.resources.abc.Traversable, instance=True
    )
    mock_traversable.joinpath.return_value = None
    with mock.patch.object(
        test_utils.importlib.resources,
        "files",
        autospec=True,
        return_value=mock_traversable,
    ) as mock_files:
      with self.assertRaises(FileNotFoundError):  # ASSERT_RAISES_OK=tool test
        test_utils.locate_goldens_pinfile()
      mock_files.assert_called_once_with(
          test_utils.goldens_pin.__package__ or test_utils.goldens_pin
      )

  @parameterized.named_parameters(
      (
          "eager_prefix",
          "ops_test_gpu_golden0.gz",
          "ops_test_gpu_golden",
      ),
      (
          "compiled_prefix",
          "ops_test_gpu_golden_compiled0.gz",
          "ops_test_gpu_golden_compiled",
      ),
      (
          "grad_prefix",
          "ops_test_gpu_golden_grad0.gz",
          "ops_test_gpu_golden_grad",
      ),
      ("eager_hopper", "ops_test_gpu_golden_hopper0.gz", "ops_test_gpu_golden"),
      (
          "compiled_hopper",
          "ops_test_gpu_golden_compiled_hopper0.gz",
          "ops_test_gpu_golden_compiled",
      ),
      (
          "grad_hopper",
          "ops_test_gpu_golden_grad_hopper0.gz",
          "ops_test_gpu_golden_grad",
      ),
  )
  def test_matches_prefix_matches(self, filename: str, prefix: str):
    self.assertTrue(test_utils.matches_prefix(filename, prefix))

  @parameterized.named_parameters(
      (
          "compiled_file_with_eager_prefix",
          "ops_test_gpu_golden_compiled0.gz",
          "ops_test_gpu_golden",
      ),
      (
          "grad_file_with_eager_prefix",
          "ops_test_gpu_golden_grad0.gz",
          "ops_test_gpu_golden",
      ),
      (
          "unrelated_prefix",
          "other_golden0.gz",
          "ops_test_gpu_golden",
      ),
  )
  def test_matches_prefix_doesnt_match(self, filename: str, prefix: str):
    self.assertFalse(test_utils.matches_prefix(filename, prefix))

  def test_goldens_candidate_local_dirs(self):
    dirs = test_utils.goldens_candidate_local_dirs(base_dir="/custom/path")
    self.assertIn(pathlib.Path("/custom/path"), dirs)

  def _pin_with(self, filenames: list[str]) -> goldens_pin.GoldenPathPinFile:
    """Builds a pin file listing `filenames`.

    The checksums are placeholders: goldens are verified against goldens.pin at
    build time, never by load_golden_files.
    """
    return goldens_pin.CONVERTER.structure(
        {
            "version": 1,
            "gcs_bucket": "torchtpu-goldens",
            "goldens": {
                fname: {
                    "sha256": "0" * 64,
                    "gcs_path": f"ops_test/test/{fname}",
                }
                for fname in filenames
            },
        },
        goldens_pin.GoldenPathPinFile,
    )

  def test_load_golden_files_local_files_exist_succeeds(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    dummy_plist = gzip.compress(plistlib.dumps({}, fmt=plistlib.FMT_BINARY))
    (temp_dir / "ops_test_gpu_golden0.gz").write_bytes(dummy_plist)

    with (
        mock.patch.object(
            test_utils,
            "goldens_candidate_local_dirs",
            return_value=[temp_dir],
        ),
        mock.patch.object(
            test_utils.goldens_pin,
            "parse_goldens_pin",
            return_value=self._pin_with(["ops_test_gpu_golden0.gz"]),
        ),
    ):
      results = test_utils.load_golden_files("ops_test_gpu_golden")
      self.assertEqual(
          results, [test_utils.LoadedGolden("ops_test_gpu_golden0.gz", {})]
      )

  def test_load_golden_files_missing_file_raises(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    dummy_plist = gzip.compress(plistlib.dumps({}, fmt=plistlib.FMT_BINARY))
    (temp_dir / "ops_test_gpu_golden0.gz").write_bytes(dummy_plist)
    nonexistent_dir = temp_dir / "does_not_exist"

    with mock.patch.object(
        test_utils.goldens_pin,
        "parse_goldens_pin",
        return_value=self._pin_with(
            ["ops_test_gpu_golden0.gz", "ops_test_gpu_golden1.gz"]
        ),
    ):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=tool test
          FileNotFoundError,
          rf"ops_test_gpu_golden1\.gz .* not found .*{temp_dir},"
          rf" {nonexistent_dir}",
      ):
        test_utils.load_golden_files(
            "ops_test_gpu_golden", candidate_dirs=[temp_dir, nonexistent_dir]
        )

  def test_load_golden_files_uses_first_candidate_dir_with_file(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    first_dir = temp_dir / "first"
    second_dir = temp_dir / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    def plist(tree):
      return gzip.compress(plistlib.dumps(tree, fmt=plistlib.FMT_BINARY))

    (first_dir / "ops_test_gpu_golden0.gz").write_bytes(plist({"dir": "first"}))
    (second_dir / "ops_test_gpu_golden0.gz").write_bytes(
        plist({"dir": "second"})
    )
    (second_dir / "ops_test_gpu_golden1.gz").write_bytes(
        plist({"dir": "second"})
    )

    with mock.patch.object(
        test_utils.goldens_pin,
        "parse_goldens_pin",
        return_value=self._pin_with(
            ["ops_test_gpu_golden0.gz", "ops_test_gpu_golden1.gz"]
        ),
    ):
      results = test_utils.load_golden_files(
          "ops_test_gpu_golden", candidate_dirs=[first_dir, second_dir]
      )
    self.assertEqual(
        results,
        [
            test_utils.LoadedGolden(
                "ops_test_gpu_golden0.gz", {"dir": "first"}
            ),
            test_utils.LoadedGolden(
                "ops_test_gpu_golden1.gz", {"dir": "second"}
            ),
        ],
    )

  def test_no_matching_prefix_in_pinfile_raises(self):
    with mock.patch.object(
        test_utils.goldens_pin,
        "parse_goldens_pin",
        return_value=self._pin_with(["unrelated_golden0.gz"]),
    ):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=tool test
          ValueError, "No golden files found matching"
      ):
        test_utils.load_golden_files("ops_test_gpu_golden")

  def test_find_per_test_case_goldens_dir(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    per_test_case_dir = temp_dir / "ops_test_gpu_golden_per_test_case"
    per_test_case_dir.mkdir()

    found = test_utils.find_per_test_case_goldens_dir(
        "ops_test_gpu_golden_per_test_case", candidate_dirs=[temp_dir]
    )
    self.assertEqual(found, per_test_case_dir)

    missing = test_utils.find_per_test_case_goldens_dir(
        "nonexistent", candidate_dirs=[temp_dir]
    )
    self.assertIsNone(missing)

  def test_load_single_test_case_golden(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    test_case_data = {"variant": "base"}
    with open(temp_dir / "test_sin.plist", "wb") as f:
      plistlib.dump(test_case_data, f, fmt=plistlib.FMT_BINARY)

    loaded = test_utils.load_single_test_case_golden("test_sin", temp_dir)
    self.assertEqual(loaded, test_case_data)

    missing = test_utils.load_single_test_case_golden("test_cos", temp_dir)
    self.assertIsNone(missing)


if __name__ == "__main__":
  absltest.main()
