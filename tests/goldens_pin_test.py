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

"""Unit tests for goldens_pin data structures and parser."""

import json
import pathlib
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis
from hypothesis import strategies as st
from tests import goldens_pin
from tests import seed_test_utils

_GOLDEN_PATH_STRATEGY = st.builds(
    goldens_pin.GoldenPath,
    sha256=st.text(min_size=1),
    gcs_path=st.text(min_size=1),
)

_GOLDEN_PATH_PIN_FILE_STRATEGY = st.builds(
    goldens_pin.GoldenPathPinFile,
    version=st.just(1),
    gcs_bucket=st.text(min_size=1),
    goldens=st.dictionaries(
        keys=st.text(min_size=1),
        values=_GOLDEN_PATH_STRATEGY,
    ),
)


class GoldensPinTest(seed_test_utils.RepeatableTest):

  def test_parse_from_file(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    temp_pin = temp_dir / "test.pin"
    raw_data = {
        "version": 1,
        "gcs_bucket": "test-bucket",
        "goldens": {
            "test_golden.gz": {
                "sha256": "abcdef123456",
                "gcs_path": "test/path/test_golden.gz",
            },
        },
    }
    temp_pin.write_text(json.dumps(raw_data), encoding="utf-8")
    pin_file = goldens_pin.parse_goldens_pin(temp_pin)
    self.assertEqual(pin_file.version, 1)
    self.assertEqual(pin_file.gcs_bucket, "test-bucket")
    self.assertLen(pin_file.goldens, 1)
    self.assertIn("test_golden.gz", pin_file.goldens)

    golden = pin_file.goldens["test_golden.gz"]
    self.assertEqual(golden.sha256, "abcdef123456")
    self.assertEqual(golden.gcs_path, "test/path/test_golden.gz")

    # Also test str path (PathType support)
    pin_file_from_str = goldens_pin.parse_goldens_pin(str(temp_pin))
    self.assertEqual(pin_file_from_str, pin_file)

  @hypothesis.given(pin_file=_GOLDEN_PATH_PIN_FILE_STRATEGY)
  def test_round_trip_serialization(
      self, pin_file: goldens_pin.GoldenPathPinFile
  ) -> None:
    unstructured = goldens_pin.CONVERTER.unstructure(pin_file)
    reparsed = goldens_pin.CONVERTER.structure(
        unstructured, goldens_pin.GoldenPathPinFile
    )
    self.assertEqual(pin_file, reparsed)

  def test_dump_and_read(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    temp_pin = temp_dir / "custom.pin"

    golden = goldens_pin.GoldenPath(
        sha256="1234567890abcdef",
        gcs_path="dummy/path.gz",
    )
    pin_file = goldens_pin.GoldenPathPinFile(
        version=1,
        gcs_bucket="my-bucket",
        goldens={"dummy.gz": golden},
    )

    goldens_pin.dump_goldens_pin(pin_file, temp_pin)
    self.assertTrue(temp_pin.is_file())

    loaded = goldens_pin.parse_goldens_pin(temp_pin)
    self.assertEqual(loaded, pin_file)

    # Also test str target path (PathType support)
    temp_pin_str = temp_dir / "custom_str.pin"
    goldens_pin.dump_goldens_pin(pin_file, str(temp_pin_str))
    self.assertTrue(temp_pin_str.is_file())
    self.assertEqual(goldens_pin.parse_goldens_pin(temp_pin_str), pin_file)

  def test_parse_goldens_pin_file_not_found_raises(self):
    with self.assertRaises(FileNotFoundError):  # ASSERT_RAISES_OK=tool tests
      goldens_pin.parse_goldens_pin("non_existent_file.pin")

  def test_parse_goldens_pin_invalid_source_type_raises(self):
    with self.assertRaises(AttributeError):  # ASSERT_RAISES_OK=tool tests
      goldens_pin.parse_goldens_pin({"version": 1})  # pytype: disable=wrong-arg-types

  def test_dump_goldens_pin_invalid_target_type_raises(self):
    pin_file = goldens_pin.GoldenPathPinFile(
        version=1,
        gcs_bucket="my-bucket",
        goldens={},
    )
    with self.assertRaises(AttributeError):  # ASSERT_RAISES_OK=tool tests
      goldens_pin.dump_goldens_pin(pin_file, 123)  # pytype: disable=wrong-arg-types

  def test_ensure_path_type_from_str(self):
    self.assertEqual(
        goldens_pin._ensure_path_type("/foo/bar"), pathlib.Path("/foo/bar")
    )

  def test_ensure_path_type_from_path(self):
    p = pathlib.Path("/foo/bar")
    self.assertIs(goldens_pin._ensure_path_type(p), p)

  @parameterized.named_parameters(
      (
          "not_a_dict",
          ["not", "a", "dict"],
      ),
      (
          "missing_version",
          {"gcs_bucket": "b", "goldens": {}},
      ),
      (
          "unsupported_version",
          {"version": 2, "gcs_bucket": "b", "goldens": {}},
      ),
      (
          "missing_bucket",
          {"version": 1, "goldens": {}},
      ),
      (
          "invalid_bucket_type",
          {"version": 1, "gcs_bucket": 123, "goldens": {}},
      ),
      (
          "missing_goldens",
          {"version": 1, "gcs_bucket": "b"},
      ),
      (
          "invalid_goldens_type",
          {"version": 1, "gcs_bucket": "b", "goldens": "invalid"},
      ),
  )
  def test_invalid_schema_raises(self, invalid_data: Any) -> None:
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    temp_pin = temp_dir / "invalid.pin"
    temp_pin.write_text(json.dumps(invalid_data), encoding="utf-8")
    with self.assertRaises(  # ASSERT_RAISES_OK=tool tests
        goldens_pin.InvalidSchemaError
    ):
      goldens_pin.parse_goldens_pin(temp_pin)

  def test_invalid_schema_error_is_value_error_subclass(self):
    self.assertTrue(issubclass(goldens_pin.InvalidSchemaError, ValueError))

  def test_golden_path_pin_file_invalid_version_raises(self):
    with self.assertRaises(  # ASSERT_RAISES_OK=tool tests
        goldens_pin.InvalidSchemaError
    ):
      goldens_pin.GoldenPathPinFile(
          version=2,
          gcs_bucket="my-bucket",
          goldens={},
      )

  @parameterized.named_parameters(
      (
          "empty_sha256",
          "",
          "path",
      ),
      (
          "empty_gcs_path",
          "hash",
          "",
      ),
  )
  def test_golden_path_empty_fields_raise(
      self, sha256: str, gcs_path: str
  ) -> None:
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=tool tests
      goldens_pin.GoldenPath(sha256=sha256, gcs_path=gcs_path)


if __name__ == "__main__":
  absltest.main()
