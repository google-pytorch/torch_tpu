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

import os
import subprocess
import tempfile
from unittest import mock

from absl.testing import absltest
import filter_impacted_targets


class FilterImpactedTargetsTest(absltest.TestCase):

  def test_parse_tag_filters_no_tag_filter(self):
    required_tags, excluded_tags = filter_impacted_targets.parse_tag_filters(
        None
    )

    self.assertEqual(required_tags, set())
    self.assertEqual(excluded_tags, set())

  def test_parse_tag_filters_nonempty_tag_filters(self):
    tag_filters_line = (
        "common:my_config"
        " --test_tag_filters=requires-tpu,-notest,-nobuild,-nopresubmit"
    )

    required_tags, excluded_tags = filter_impacted_targets.parse_tag_filters(
        tag_filters_line
    )

    self.assertEqual(required_tags, {"requires-tpu"})
    self.assertEqual(
        excluded_tags,
        {"notest", "nobuild", "nopresubmit"},
    )

  @mock.patch("subprocess.run")
  def test_get_bazel_tag_filters_from_config_only(self, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="--test_tag_filters=requires-tpu,-wheel_test_excluded\n",
        stderr="",
    )

    req, excl = filter_impacted_targets.get_bazel_tag_filters(
        "my_config", extra_flags=""
    )

    self.assertEqual(req, {"requires-tpu"})
    self.assertEqual(excl, {"wheel_test_excluded"})
    mock_run.assert_called_once()
    cmd_called = mock_run.call_args[0][0]
    self.assertIn("canonicalize-flags", cmd_called)
    self.assertIn("--config=my_config", cmd_called)

  @mock.patch("subprocess.run")
  def test_get_bazel_tag_filters_overridden_by_extra_flags(self, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="--test_tag_filters=-flaky_test\n",
        stderr="",
    )

    req, excl = filter_impacted_targets.get_bazel_tag_filters(
        "my_config", extra_flags="--test_tag_filters=-flaky_test"
    )

    self.assertEqual(req, set())
    self.assertEqual(excl, {"flaky_test"})
    mock_run.assert_called_once()
    cmd_called = mock_run.call_args[0][0]
    self.assertIn("canonicalize-flags", cmd_called)
    self.assertIn("--config=my_config", cmd_called)
    self.assertIn("--test_tag_filters=-flaky_test", cmd_called)

  @mock.patch("subprocess.run")
  def test_get_targets_metadata_xml(self, mock_run):
    expected_xml = """<?xml version="1.1" encoding="UTF-8"?>
    <query version="2">
        <rule class="torch_tpu_py_test" location="/tests/BUILD:10:1" name="//tests:test_a">
            <list name="tags">
                <string value="requires-tpu"/>
                <string value="nopresubmit"/>
            </list>
        </rule>
        <rule class="torch_tpu_cc_library" location="/tests/BUILD:20:1" name="//tests:test_b">
            <list name="tags"/>
        </rule>
    </query>
    """
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=expected_xml, stderr=""
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      query_file = os.path.join(tmpdir, "test.query")
      targets = ["//tests:test_a", "//tests:test_b"]

      xml_content = filter_impacted_targets.get_targets_metadata_xml(
          targets, query_file
      )

      self.assertEqual(xml_content, expected_xml)
      with open(query_file, "r") as f:
        self.assertEqual(f.read(), "set(//tests:test_a //tests:test_b)")
      mock_run.assert_called_once()
      cmd_called = mock_run.call_args[0][0]
      self.assertIn("query", cmd_called)
      self.assertIn(f"--query_file={query_file}", cmd_called)
      self.assertIn("--output=xml", cmd_called)

  def test_filter_targets(self):
    xml_content = """<?xml version="1.1" encoding="UTF-8"?>
        <query version="2">
            <!-- Should be KEPT: It's a test and has the required tag 'requires-tpu', no excluded tags -->
            <rule class="torch_tpu_py_test" name="//tests:good_test">
                <list name="tags">
                    <string value="requires-tpu"/>
                    <string value="nightly"/>
                </list>
            </rule>

            <!-- Should be DROPPED: It has the excluded tag 'wheel_test_excluded'. -->
            <rule class="torch_tpu_py_test" name="//tests:bad_test">
                <list name="tags">
                    <string value="requires-tpu"/>
                    <string value="wheel_test_excluded"/>
                </list>
            </rule>

            <!-- Should be DROPPED: It is missing the required tag 'requires-tpu'. -->
            <rule class="torch_tpu_py_test" name="//tests:missing_req_test">
                <list name="tags">
                    <string value="nightly"/>
                </list>
            </rule>

            <!-- Should be KEPT: It is not a test, tags don't matter. -->
            <rule class="torch_tpu_cc_library" name="//libs:core_lib">
                <list name="tags"/>
            </rule>
        </query>
        """
    required_tags = {"requires-tpu"}
    excluded_tags = {"wheel_test_excluded"}

    filtered = filter_impacted_targets.filter_targets(
        xml_content, required_tags, excluded_tags
    )

    self.assertIn("//tests:good_test", filtered)
    self.assertNotIn("//tests:bad_test", filtered)
    self.assertNotIn("//tests:missing_req_test", filtered)
    self.assertIn("//libs:core_lib", filtered)
    self.assertLen(filtered, 2)


if __name__ == "__main__":
  absltest.main()
