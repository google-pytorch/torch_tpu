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

"""Tests for custom_lint presubmit checker."""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
try:
  from torch_tpu.oss.ci.tools import custom_lint  # pylint: disable=g-import-not-at-top
except ImportError:
  import custom_lint  # pylint: disable=g-import-not-at-top  # noqa: E402


class CustomLintTest(unittest.TestCase):

  def test_parse_unified_diff(self):
    diff_text = """+++ b/foo/BUILD.bazel
@@ -10,0 +11,3 @@
+cc_library(
+    name = "lib",
+)
+++ b/bar/BUILD
@@ -5 +5 @@
-py_library(name = "x")
+cc_test(name = "x")
+++ /dev/null
@@ -1,5 +0,0 @@
-deleted_line
"""
    modified = custom_lint.parse_unified_diff(diff_text)
    self.assertIn("foo/BUILD.bazel", modified)
    self.assertEqual(modified["foo/BUILD.bazel"], {11, 12, 13})
    self.assertIn("bar/BUILD", modified)
    self.assertEqual(modified["bar/BUILD"], {5})
    self.assertNotIn("/dev/null", modified)
    self.assertNotIn("dev/null", modified)

  def test_is_rule_disabled(self):
    rule = custom_lint.Rule(
        id="TestRule",
        path_regex=r".*",
        prohibited_regex=r"foo",
        failure_message="msg",
        disable_tags=["ALLOW_FOO", "ROLLBACK_OF"],
    )
    self.assertTrue(
        custom_lint.is_rule_disabled(rule, "Fix: something\nALLOW_FOO=needed")
    )
    self.assertTrue(
        custom_lint.is_rule_disabled(rule, "Automated ROLLBACK_OF cl/123")
    )
    self.assertFalse(custom_lint.is_rule_disabled(rule, "Normal commit"))

  def _get_cc_target_rule(self):
    return next(r for r in custom_lint.RULES if r.id == "CheckCcTargetUsage")

  def test_check_file_cc_library_violation(self):
    rule = self._get_cc_target_rule()  # CheckCcTargetUsage
    with tempfile.TemporaryDirectory() as tmpdir:
      build_file = pathlib.Path(tmpdir) / "BUILD.bazel"
      build_file.write_text('cc_library(name = "test")\n', encoding="utf-8")
      violations = custom_lint.check_file(
          build_file, "BUILD.bazel", rule, modified_lines={"BUILD.bazel": {1}}
      )
      self.assertEqual(len(violations), 1)
      self.assertEqual(violations[0].rule_id, "CheckCcTargetUsage")
      self.assertEqual(violations[0].line_number, 1)

  def test_check_file_cc_library_suppressed(self):
    rule = self._get_cc_target_rule()  # CheckCcTargetUsage
    with tempfile.TemporaryDirectory() as tmpdir:
      build_file = pathlib.Path(tmpdir) / "BUILD.bazel"
      build_file.write_text(
          'cc_library(name = "test")  # ALLOW_CC_TARGETS=shim\n',
          encoding="utf-8",
      )
      violations = custom_lint.check_file(
          build_file, "BUILD.bazel", rule, modified_lines={"BUILD.bazel": {1}}
      )
      self.assertEqual(violations, [])

  def test_check_file_cc_library_suppressed_multiline(self):
    rule = self._get_cc_target_rule()  # CheckCcTargetUsage
    with tempfile.TemporaryDirectory() as tmpdir:
      build_file = pathlib.Path(tmpdir) / "BUILD.bazel"
      build_file.write_text(
          'cc_library(\n    # ALLOW_CC_TARGETS=shim\n    name = "test",\n)\n',
          encoding="utf-8",
      )
      violations = custom_lint.check_file(
          build_file,
          "BUILD.bazel",
          rule,
          modified_lines={"BUILD.bazel": {1, 2}},
      )
      self.assertEqual(violations, [])

  def test_check_file_delta_only_ignores_untouched_lines(self):
    rule = self._get_cc_target_rule()  # CheckCcTargetUsage
    with tempfile.TemporaryDirectory() as tmpdir:
      build_file = pathlib.Path(tmpdir) / "BUILD.bazel"
      build_file.write_text(
          'cc_library(name = "test")\n# Untouched line 2\n',
          encoding="utf-8",
      )
      # Suppose only line 2 was modified in the diff
      violations = custom_lint.check_file(
          build_file, "BUILD.bazel", rule, modified_lines={"BUILD.bazel": {2}}
      )
      self.assertEqual(violations, [])

  def test_check_file_c10_tostring_violation(self):
    rule = next(r for r in custom_lint.RULES if r.id == "CheckC10ToStringUsage")
    with tempfile.TemporaryDirectory() as tmpdir:
      cc_file = pathlib.Path(tmpdir) / "foo.cc"
      cc_file.write_text("auto s = c10::toString(x);\n", encoding="utf-8")
      violations = custom_lint.check_file(
          cc_file, "foo.cc", rule, modified_lines={"foo.cc": {1}}
      )
      self.assertEqual(len(violations), 1)
      self.assertEqual(violations[0].rule_id, "CheckC10ToStringUsage")

  def test_check_file_relative_imports_violation(self):
    rule = next(r for r in custom_lint.RULES if r.id == "CheckRelativeImports")
    with tempfile.TemporaryDirectory() as tmpdir:
      py_file = pathlib.Path(tmpdir) / "foo.py"
      py_file.write_text("from .bar import baz\n", encoding="utf-8")
      violations = custom_lint.check_file(
          py_file, "foo.py", rule, modified_lines={"foo.py": {1}}
      )
      self.assertEqual(len(violations), 1)
      self.assertEqual(violations[0].rule_id, "CheckRelativeImports")

  def test_check_file_relative_imports_prototypes_exclusion(self):
    rule = next(r for r in custom_lint.RULES if r.id == "CheckRelativeImports")
    with tempfile.TemporaryDirectory() as tmpdir:
      py_file = pathlib.Path(tmpdir) / "foo.py"
      py_file.write_text("from .bar import baz\n", encoding="utf-8")
      violations = custom_lint.check_file(
          py_file,
          "google/prototypes/foo.py",
          rule,
          modified_lines={"google/prototypes/foo.py": {1}},
      )
      self.assertEqual(violations, [])

  def test_check_file_py_test_multiline(self):
    rule = next(r for r in custom_lint.RULES if r.id == "CheckPyTestUsage")
    with tempfile.TemporaryDirectory() as tmpdir:
      build_file = pathlib.Path(tmpdir) / "BUILD.bazel"
      build_file.write_text(
          'py_test(\n    # PY_TEST_OK=legacy\n    name = "foo",\n)\n',
          encoding="utf-8",
      )
      violations = custom_lint.check_file(
          build_file,
          "BUILD.bazel",
          rule,
          modified_lines={"BUILD.bazel": {1, 2}},
      )
      self.assertEqual(violations, [])

  def test_check_file_include_before_copyright_delta_false(self):
    rule = custom_lint.Rule(
        id="CheckIncludeBeforeCopyright",
        path_regex=r".*\.(cc|cpp|h|hpp)$",
        prohibited_regex=r"^(\s|\n)*#include",
        failure_message=(
            "Please ensure that all #includes are after the copyright notice."
        ),
        check_delta_only=False,
        disable_tags=["ROLLBACK_OF"],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      cc_file = pathlib.Path(tmpdir) / "foo.cc"
      cc_file.write_text(
          "#include <iostream>\n// Copyright 2026 Google LLC\n",
          encoding="utf-8",
      )
      # Even if modified_lines is for line 2, line 1 #include will
      # trigger because check_delta_only=False
      violations = custom_lint.check_file(
          cc_file, "foo.cc", rule, modified_lines={"foo.cc": {2}}
      )
      self.assertEqual(len(violations), 1)
      self.assertEqual(violations[0].rule_id, "CheckIncludeBeforeCopyright")

  def test_check_file_include_after_copyright_passes(self):
    rule = custom_lint.Rule(
        id="CheckIncludeBeforeCopyright",
        path_regex=r".*\.(cc|cpp|h|hpp)$",
        prohibited_regex=r"^(\s|\n)*#include",
        failure_message=(
            "Please ensure that all #includes are after the copyright notice."
        ),
        check_delta_only=False,
        disable_tags=["ROLLBACK_OF"],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      cc_file = pathlib.Path(tmpdir) / "foo.cc"
      cc_file.write_text(
          "// Copyright 2026 Google LLC\n#include <iostream>\n",
          encoding="utf-8",
      )
      violations = custom_lint.check_file(
          cc_file, "foo.cc", rule, modified_lines={"foo.cc": {2}}
      )
      self.assertEqual(violations, [])

  def test_check_file_absl_and_unittest_testcase(self):
    rule = next(r for r in custom_lint.RULES if r.id == "CheckAbslTestCase")
    with tempfile.TemporaryDirectory() as tmpdir:
      py_file = pathlib.Path(tmpdir) / "test_foo.py"

      # Trigger on unittest.TestCase
      py_file.write_text(
          "class MyTest(unittest.TestCase):\n  pass\n", encoding="utf-8"
      )
      violations = custom_lint.check_file(
          py_file, "test_foo.py", rule, modified_lines={"test_foo.py": {1}}
      )
      self.assertEqual(len(violations), 1)
      self.assertEqual(violations[0].rule_id, "CheckAbslTestCase")

      # Trigger on absltest.TestCase
      py_file.write_text(
          "class MyTest(absltest.TestCase):\n  pass\n", encoding="utf-8"
      )
      violations = custom_lint.check_file(
          py_file, "test_foo.py", rule, modified_lines={"test_foo.py": {1}}
      )
      self.assertEqual(len(violations), 1)

      # Suppressed with UNITTEST_OK
      py_file.write_text(
          "class MyTest(unittest.TestCase):  # UNITTEST_OK=standalone\n "
          " pass\n",
          encoding="utf-8",
      )
      violations = custom_lint.check_file(
          py_file, "test_foo.py", rule, modified_lines={"test_foo.py": {1}}
      )
      self.assertEqual(violations, [])

      # Suppressed with ABSLTEST_OK
      py_file.write_text(
          "class MyTest(absltest.TestCase):  # ABSLTEST_OK=standalone\n "
          " pass\n",
          encoding="utf-8",
      )
      violations = custom_lint.check_file(
          py_file, "test_foo.py", rule, modified_lines={"test_foo.py": {1}}
      )
      self.assertEqual(violations, [])

  def test_is_path_excluded(self):
    self.assertTrue(custom_lint.is_path_excluded("custom_lint.py"))
    self.assertTrue(custom_lint.is_path_excluded("oss/ci/tools/custom_lint.py"))
    self.assertTrue(
        custom_lint.is_path_excluded("oss/ci/tools/custom_lint_test.py")
    )
    self.assertTrue(
        custom_lint.is_path_excluded("google/tools/custom_lint_presubmit.py")
    )
    self.assertTrue(
        custom_lint.is_path_excluded(
            "google/tools/custom_lint_presubmit_test.py"
        )
    )
    self.assertTrue(custom_lint.is_path_excluded("copy.bara.sky"))
    self.assertTrue(custom_lint.is_path_excluded("examples/some_model.py"))
    self.assertFalse(custom_lint.is_path_excluded("torch_tpu/csrc/core/foo.cc"))

  def test_rule_matches_path(self):
    rule = custom_lint.Rule(
        id="TestRule",
        path_regex=r".*\.(cc|cpp|h|hpp)$",
        prohibited_regex=r"\bfoo\b",
        failure_message="No foo",
        path_exclusion_regex=r".*_test\.(cc|cpp|h)",
    )
    self.assertTrue(rule.matches_path("src/core/model.cc"))
    self.assertFalse(rule.matches_path("src/core/model_test.cc"))
    self.assertFalse(rule.matches_path("src/core/model.py"))

  def test_find_candidate_files(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root_path = pathlib.Path(tmpdir)
      cc_file = root_path / "foo.cc"
      cc_file.write_text("content\n", encoding="utf-8")
      py_file = root_path / "bar.py"
      py_file.write_text("content\n", encoding="utf-8")

      rule_cc = custom_lint.Rule(
          id="RuleCC",
          path_regex=r".*\.cc$",
          prohibited_regex=r"foo",
          failure_message="msg",
      )
      candidates = custom_lint.find_candidate_files(root_path, [rule_cc])
      self.assertEqual(len(candidates), 1)
      self.assertIsInstance(candidates[0], custom_lint.CandidateFile)
      self.assertEqual(candidates[0].file_path, cc_file)
      self.assertEqual(candidates[0].matching_rules, [rule_cc])


if __name__ == "__main__":
  unittest.main()
