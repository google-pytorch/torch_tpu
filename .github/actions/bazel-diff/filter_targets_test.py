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

import pathlib
import tempfile

from absl.testing import absltest
import filter_targets


class TestFilterTargets(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.TemporaryDirectory()
    self.bazelrc_path = pathlib.Path(self.test_dir.name) / ".bazelrc"

  def tearDown(self):
    self.test_dir.cleanup()
    super().tearDown()

  def test_get_test_tag_filters_file_not_found(self):
    fake_path = pathlib.Path(self.test_dir.name) / "does_not_exist.bazelrc"

    result = filter_targets.get_test_tag_filters(fake_path, "wheel_test_tpu_v5")

    self.assertEqual(result, "")

  def test_get_test_tag_filters_config_found(self):
    content = (
        "# Some comments\n"
        "test:wheel_test_tpu_v5 --test_tag_filters=-notest,presubmit-v5\n"
        "test:other_config --test_tag_filters=something_else\n"
    )
    self.bazelrc_path.write_text(content)

    result = filter_targets.get_test_tag_filters(
        self.bazelrc_path, "wheel_test_tpu_v5"
    )

    self.assertEqual(result, "-notest,presubmit-v5")

  def test_get_test_tag_filters_config_not_found(self):
    content = "test:other_config --test_tag_filters=something\n"
    self.bazelrc_path.write_text(content)

    result = filter_targets.get_test_tag_filters(
        self.bazelrc_path, "wheel_test_tpu_v5"
    )

    self.assertEqual(result, "")

  def test_get_test_tag_filters_with_extra_whitespace(self):
    content = (
        "test:wheel_test_tpu_v6    --test_tag_filters=-nobuild,presubmit-v6"
        "   \n"
    )
    self.bazelrc_path.write_text(content)

    result = filter_targets.get_test_tag_filters(
        self.bazelrc_path, "wheel_test_tpu_v6"
    )

    self.assertEqual(result, "-nobuild,presubmit-v6")

  def test_build_bazel_query_empty_tags(self):
    targets = ["//tests:ops_test.py", "//tests:utils_test.py"]

    query = filter_targets.build_bazel_query(targets, "")

    self.assertEqual(
        query, "tests(set(//tests:ops_test.py //tests:utils_test.py))"
    )

  def test_build_bazel_query_only_positive(self):
    targets = ["//tests:ops_test.py"]

    query = filter_targets.build_bazel_query(
        targets, "presubmit-v7,requires-tpu"
    )

    expected = (
        "let impacted = tests(set(//tests:ops_test.py)) in (attr('tags', '[\\["
        " ]presubmit-v7[,\\]]', $impacted) union attr('tags', '[\\["
        " ]requires-tpu[,\\]]', $impacted))"
    )
    self.assertEqual(query, expected)

  def test_build_bazel_query_only_negative(self):
    targets = ["//tests:ops_test.py"]

    query = filter_targets.build_bazel_query(targets, "-notest,-nobuild")

    expected = (
        "let impacted = tests(set(//tests:ops_test.py)) in $impacted except"
        " attr('tags', '[\\[ ]notest[,\\]]', $impacted) except attr('tags',"
        " '[\\[ ]nobuild[,\\]]', $impacted)"
    )
    self.assertEqual(query, expected)

  def test_build_bazel_query_mixed_tags(self):
    targets = ["//tests:ops_test.py", "//tests:utils_test.py"]

    query = filter_targets.build_bazel_query(targets, "presubmit-v7,-notest")

    expected = (
        "let impacted = tests(set(//tests:ops_test.py //tests:utils_test.py))"
        " in (attr('tags', '[\\[ ]presubmit-v7[,\\]]', $impacted)) except"
        " attr('tags', '[\\[ ]notest[,\\]]', $impacted)"
    )
    self.assertEqual(query, expected)


if __name__ == "__main__":
  absltest.main()
