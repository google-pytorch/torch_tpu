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
import hashlib
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

  def test_download_golden_file_success(self):
    payload = b"downloaded golden payload"
    expected_hash = hashlib.sha256(payload).hexdigest()

    mock_resp = mock.MagicMock()
    mock_resp.read.return_value = payload
    mock_resp.__enter__.return_value = mock_resp

    with mock.patch.object(
        test_utils.urllib.request, "urlopen", return_value=mock_resp
    ) as mock_urlopen:
      downloaded_data = test_utils.download_golden_file(
          filename="ops_test_gpu_golden0.gz",
          expected_sha256=expected_hash,
          gcs_bucket="torchtpu-goldens",
          gcs_path="ops_test/sample/ops_test_gpu_golden0.gz",
      )
      mock_urlopen.assert_called_once()
      call_args = mock_urlopen.call_args[0]
      req = call_args[0]
      self.assertEqual(
          req.full_url,
          "https://storage.googleapis.com/torchtpu-goldens/ops_test/sample/ops_test_gpu_golden0.gz",
      )
      self.assertEqual(downloaded_data, payload)

  def test_download_golden_file_checksum_mismatch_raises(self):
    corrupted_payload = b"corrupted payload"
    expected_hash = hashlib.sha256(b"correct payload").hexdigest()

    mock_resp = mock.MagicMock()
    mock_resp.read.return_value = corrupted_payload
    mock_resp.__enter__.return_value = mock_resp

    with mock.patch.object(
        test_utils.urllib.request, "urlopen", return_value=mock_resp
    ):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=tool test
          ValueError, "SHA-256 checksum mismatch"
      ):
        test_utils.download_golden_file(
            filename="ops_test_gpu_golden0.gz",
            expected_sha256=expected_hash,
            gcs_bucket="torchtpu-goldens",
            gcs_path="ops_test/sample/ops_test_gpu_golden0.gz",
        )

  def test_validate_sha256_success(self):
    payload = b"valid data"
    expected = hashlib.sha256(payload).hexdigest()
    test_utils.validate_sha256(payload, expected)
    test_utils.validate_sha256(payload, expected.upper())

  def test_validate_sha256_mismatch_raises(self):
    payload = b"actual data"
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=tool test
        ValueError,
        r"SHA-256 checksum mismatch for foo\.gz downloaded from"
        r" http://example\.com/foo\.gz: expected deadbeef, got [0-9a-f]{64}",
    ):
      test_utils.validate_sha256(
          payload,
          "deadbeef",
          filename="foo.gz",
          url="http://example.com/foo.gz",
      )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=tool test
        ValueError,
        r"^SHA-256 checksum mismatch: expected deadbeef, got [0-9a-f]{64}$",
    ):
      test_utils.validate_sha256(payload, "deadbeef")

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

  def test_fallback_local_files_exist_skips_gcs(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    dummy_plist = gzip.compress(plistlib.dumps({}, fmt=plistlib.FMT_BINARY))
    (temp_dir / "ops_test_gpu_golden0.gz").write_bytes(dummy_plist)
    sha256 = hashlib.sha256(dummy_plist).hexdigest()

    mock_pin_data = {
        "version": 1,
        "gcs_bucket": "torchtpu-goldens",
        "goldens": {
            "ops_test_gpu_golden0.gz": {
                "sha256": sha256,
                "gcs_path": "ops_test/test/ops_test_gpu_golden0.gz",
            },
        },
    }

    with (
        mock.patch.object(
            test_utils,
            "goldens_candidate_local_dirs",
            return_value=[temp_dir],
        ),
        mock.patch.object(
            test_utils.goldens_pin,
            "parse_goldens_pin",
            return_value=goldens_pin.CONVERTER.structure(
                mock_pin_data, goldens_pin.GoldenPathPinFile
            ),
        ),
        mock.patch.object(test_utils.urllib.request, "urlopen") as mock_urlopen,
    ):
      results = test_utils.load_golden_files("ops_test_gpu_golden")
      mock_urlopen.assert_not_called()
      self.assertEqual(
          results, [test_utils.LoadedGolden("ops_test_gpu_golden0.gz", {})]
      )

  def test_fallback_missing_local_files_triggers_gcs_download(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    empty_local_dir = temp_dir / "empty_local"
    empty_local_dir.mkdir()

    dummy_plist = gzip.compress(plistlib.dumps({}, fmt=plistlib.FMT_BINARY))
    dummy_sha = hashlib.sha256(dummy_plist).hexdigest()

    mock_pin_data = {
        "version": 1,
        "gcs_bucket": "torchtpu-goldens",
        "goldens": {
            "ops_test_gpu_golden0.gz": {
                "sha256": dummy_sha,
                "gcs_path": "ops_test/test/ops_test_gpu_golden0.gz",
            },
        },
    }

    mock_resp = mock.MagicMock()
    mock_resp.read.return_value = dummy_plist
    mock_resp.__enter__.return_value = mock_resp

    with (
        mock.patch.object(
            test_utils,
            "goldens_candidate_local_dirs",
            return_value=[empty_local_dir],
        ),
        mock.patch.object(
            test_utils.goldens_pin,
            "parse_goldens_pin",
            return_value=goldens_pin.CONVERTER.structure(
                mock_pin_data, goldens_pin.GoldenPathPinFile
            ),
        ),
        mock.patch.object(
            test_utils.urllib.request, "urlopen", return_value=mock_resp
        ) as mock_urlopen,
    ):
      results = test_utils.load_golden_files("ops_test_gpu_golden")
      mock_urlopen.assert_called_once()
      call_args = mock_urlopen.call_args[0]
      req = call_args[0]
      self.assertEqual(
          req.full_url,
          "https://storage.googleapis.com/torchtpu-goldens/ops_test/test/ops_test_gpu_golden0.gz",
      )
      self.assertEqual(
          results, [test_utils.LoadedGolden("ops_test_gpu_golden0.gz", {})]
      )

  def test_partial_local_files_downloads_only_missing_from_gcs(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    dummy_plist = gzip.compress(plistlib.dumps({}, fmt=plistlib.FMT_BINARY))
    (temp_dir / "ops_test_gpu_golden0.gz").write_bytes(dummy_plist)

    dummy_remote_plist = gzip.compress(
        plistlib.dumps({}, fmt=plistlib.FMT_BINARY)
    )
    dummy_remote_sha = hashlib.sha256(dummy_remote_plist).hexdigest()
    dummy_local_sha = hashlib.sha256(dummy_plist).hexdigest()

    mock_pin_data = {
        "version": 1,
        "gcs_bucket": "torchtpu-goldens",
        "goldens": {
            "ops_test_gpu_golden0.gz": {
                "sha256": dummy_local_sha,
                "gcs_path": "ops_test/test/ops_test_gpu_golden0.gz",
            },
            "ops_test_gpu_golden1.gz": {
                "sha256": dummy_remote_sha,
                "gcs_path": "ops_test/test/ops_test_gpu_golden1.gz",
            },
        },
    }

    mock_resp = mock.MagicMock()
    mock_resp.read.return_value = dummy_remote_plist
    mock_resp.__enter__.return_value = mock_resp

    with (
        mock.patch.object(
            test_utils,
            "goldens_candidate_local_dirs",
            return_value=[temp_dir],
        ),
        mock.patch.object(
            test_utils.goldens_pin,
            "parse_goldens_pin",
            return_value=goldens_pin.CONVERTER.structure(
                mock_pin_data, goldens_pin.GoldenPathPinFile
            ),
        ),
        mock.patch.object(
            test_utils.urllib.request, "urlopen", return_value=mock_resp
        ) as mock_urlopen,
    ):
      results = test_utils.load_golden_files("ops_test_gpu_golden")
      mock_urlopen.assert_called_once()
      call_args = mock_urlopen.call_args[0]
      req = call_args[0]
      self.assertEqual(
          req.full_url,
          "https://storage.googleapis.com/torchtpu-goldens/ops_test/test/ops_test_gpu_golden1.gz",
      )
      self.assertEqual(
          [golden.filename for golden in results],
          ["ops_test_gpu_golden0.gz", "ops_test_gpu_golden1.gz"],
      )

  def test_no_matching_prefix_in_pinfile_raises(self):
    mock_pin_data = {
        "version": 1,
        "gcs_bucket": "torchtpu-goldens",
        "goldens": {
            "unrelated_golden0.gz": {
                "sha256": "sha",
                "gcs_path": "ops_test/test/unrelated_golden0.gz",
            },
        },
    }
    with mock.patch.object(
        test_utils.goldens_pin,
        "parse_goldens_pin",
        return_value=goldens_pin.CONVERTER.structure(
            mock_pin_data, goldens_pin.GoldenPathPinFile
        ),
    ):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=tool test
          ValueError, "No golden files found matching"
      ):
        test_utils.load_golden_files("ops_test_gpu_golden")


if __name__ == "__main__":
  absltest.main()
