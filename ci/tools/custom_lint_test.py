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
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).parent))
try:
  from torch_tpu.oss.ci.tools import custom_lint  # pylint: disable=g-import-not-at-top
except ImportError:
  import custom_lint  # pylint: disable=g-import-not-at-top  # noqa: E402

_G3 = custom_lint._G3  # pylint: disable=protected-access
_TP_PREFIX = custom_lint._TP_PREFIX  # pylint: disable=protected-access


class CustomLintTest(unittest.TestCase):  # UNITTEST_OK=linter_test

  def test_parse_unified_diff(self):
    diff_text = """+++ b/foo/BUILD.bazel\t2026-08-26 23:52:54.000000000 +0000
@@ -10,0 +11,3 @@
+cc_library(
+    name = "lib",
+)
+++ b/bar/BUILD
@@ -5 +5 @@
-py_library(name = "x")
+cc_test(name = "x")
+++ b/baz/file.txt   2026-08-26 18:14:30.000000000 -0600
@@ -1 +1 @@
-old
+new
+++ b/simple/main.py
@@ -10 +10,2 @@
-# old code
+# new code 1
+# new code 2
+++ b/deletion/file.py
@@ -10,3 +10,0 @@
-line 10
-line 11
-line 12
+++ /dev/null\tWed Aug 26 23:52:54 2026
@@ -1,5 +0,0 @@
-deleted_line
"""
    modified = custom_lint.parse_unified_diff(diff_text)
    self.assertIn("foo/BUILD.bazel", modified)
    self.assertEqual(modified["foo/BUILD.bazel"], {11, 12, 13})
    self.assertIn("bar/BUILD", modified)
    self.assertEqual(modified["bar/BUILD"], {5})
    self.assertIn("baz/file.txt", modified)
    self.assertEqual(modified["baz/file.txt"], {1})
    self.assertIn("simple/main.py", modified)
    self.assertEqual(modified["simple/main.py"], {10, 11})
    self.assertIn("deletion/file.py", modified)
    self.assertEqual(modified["deletion/file.py"], set())
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
    content = 'cc_library(name = "test")\n'
    violations = custom_lint.check_file_content(
        content=content,
        lines=content.splitlines(),
        rel_path="BUILD.bazel",
        rule=rule,
        modified_lines={"BUILD.bazel": {1}},
    )
    self.assertEqual(len(violations), 1)
    self.assertEqual(violations[0].rule_id, "CheckCcTargetUsage")
    self.assertEqual(violations[0].line_number, 1)

  def test_check_file_cc_library_suppressed(self):
    rule = self._get_cc_target_rule()  # CheckCcTargetUsage
    content = 'cc_library(name = "test")  # ALLOW_CC_TARGETS=shim\n'
    violations = custom_lint.check_file_content(
        content=content,
        lines=content.splitlines(),
        rel_path="BUILD.bazel",
        rule=rule,
        modified_lines={"BUILD.bazel": {1}},
    )
    self.assertEqual(violations, [])

  def test_check_file_cc_library_suppressed_multiline(self):
    rule = self._get_cc_target_rule()  # CheckCcTargetUsage
    content = (
        'cc_library(\n    # ALLOW_CC_TARGETS=shim\n    name = "test",\n)\n'
    )
    violations = custom_lint.check_file_content(
        content=content,
        lines=content.splitlines(),
        rel_path="BUILD.bazel",
        rule=rule,
        modified_lines={"BUILD.bazel": {1, 2}},
    )
    self.assertEqual(violations, [])

  def test_check_file_delta_only_ignores_untouched_lines(self):
    # Validates that violations on unmodified lines within a modified file are ignored.
    rule = self._get_cc_target_rule()  # CheckCcTargetUsage
    content = 'cc_library(name = "test")\n# Untouched line 2\n'
    # Suppose only line 2 was modified in the diff
    violations = custom_lint.check_file_content(
        content=content,
        lines=content.splitlines(),
        rel_path="BUILD.bazel",
        rule=rule,
        modified_lines={"BUILD.bazel": {2}},
    )
    self.assertEqual(violations, [])

  def test_check_file_delta_only_ignores_untouched_file(self):
    # Validates that an entire file not present in the diff (untouched file) is skipped.
    rule = self._get_cc_target_rule()  # CheckCcTargetUsage
    content = 'cc_library(name = "test")\n'
    # modified_lines does not contain "BUILD.bazel" (untouched file)
    violations = custom_lint.check_file_content(
        content=content,
        lines=content.splitlines(),
        rel_path="BUILD.bazel",
        rule=rule,
        modified_lines={"OTHER_FILE.txt": {1}},
    )
    self.assertEqual(violations, [])

  def test_is_path_excluded(self):
    self.assertTrue(custom_lint.is_path_excluded("custom_lint.py"))
    self.assertTrue(custom_lint.is_path_excluded("custom_lint_rules.py"))
    self.assertTrue(custom_lint.is_path_excluded("oss/ci/tools/custom_lint.py"))
    self.assertTrue(
        custom_lint.is_path_excluded("oss/ci/tools/custom_lint_rules.py")
    )
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

    cc_rule = self._get_cc_target_rule()
    self.assertTrue(cc_rule.matches_path("BUILD"))
    self.assertTrue(cc_rule.matches_path("BUILD.bazel"))
    self.assertTrue(cc_rule.matches_path("tools/BUILD"))
    self.assertTrue(cc_rule.matches_path("tools/BUILD.bazel"))
    self.assertTrue(cc_rule.matches_path("tools/build_defs.bzl"))
    self.assertFalse(cc_rule.matches_path("rebuild.cc"))
    self.assertFalse(cc_rule.matches_path("BUILD_GUIDE.md"))
    self.assertFalse(cc_rule.matches_path("builder_utils.py"))

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

  def test_parse_unified_diff_with_vcs_prefixes(self):
    if not _TP_PREFIX:
      self.skipTest("Monorepo depot prefixes only apply in internal depot")
    diff_text = f"""+++ b/{_TP_PREFIX}/foo/BUILD.bazel
@@ -10,0 +11,3 @@
+cc_library(
+    name = "lib",
+)
+++ {_TP_PREFIX}/bar/BUILD
@@ -5 +5 @@
-py_library(name = "x")
+cc_test(name = "x")
+++ //depot/{_G3}/{_TP_PREFIX}/baz/file.py
@@ -1 +1 @@
-old
+new
+++ {_G3}/{_TP_PREFIX}/qux/main.py
@@ -20 +20,2 @@
-# old
+# new 1
+# new 2
"""
    modified = custom_lint.parse_unified_diff(diff_text)
    self.assertIn("foo/BUILD.bazel", modified)
    self.assertEqual(modified["foo/BUILD.bazel"], {11, 12, 13})
    self.assertIn("bar/BUILD", modified)
    self.assertEqual(modified["bar/BUILD"], {5})
    self.assertIn("baz/file.py", modified)
    self.assertEqual(modified["baz/file.py"], {1})
    self.assertIn("qux/main.py", modified)
    self.assertEqual(modified["qux/main.py"], {20, 21})
    self.assertNotIn(f"{_TP_PREFIX}/foo/BUILD.bazel", modified)
    self.assertNotIn(f"{_TP_PREFIX}/bar/BUILD", modified)
    self.assertNotIn(f"//depot/{_G3}/{_TP_PREFIX}/baz/file.py", modified)
    self.assertNotIn(f"{_G3}/{_TP_PREFIX}/qux/main.py", modified)

  def test_get_modified_lines_from_vcs_git(self):
    prefix = f"{_TP_PREFIX}/" if _TP_PREFIX else ""
    git_diff_output = f"""+++ b/{prefix}foo/BUILD
@@ -10,0 +11,2 @@
+cc_library(
+    name = "lib",
+)
"""
    mock_result = mock.Mock(returncode=0, stdout=git_diff_output, stderr="")
    with mock.patch("subprocess.run", return_value=mock_result) as mock_run:
      modified = custom_lint.get_modified_lines_from_vcs("origin/main")
      self.assertIsNotNone(modified)
      self.assertIn("foo/BUILD", modified)
      self.assertEqual(modified["foo/BUILD"], {11, 12})
      if prefix:
        self.assertNotIn(f"{prefix}foo/BUILD", modified)
      mock_run.assert_called_once_with(
          ["git", "diff", "-U0", "--no-color", "origin/main..."],
          stdout=mock.ANY,
          stderr=mock.ANY,
          text=True,
          check=False,
      )

  def test_get_modified_lines_from_vcs_git_with_existing_dots(self):
    mock_result = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch("subprocess.run", return_value=mock_result) as mock_run:
      custom_lint.get_modified_lines_from_vcs("origin/main...")
      mock_run.assert_called_once_with(
          ["git", "diff", "-U0", "--no-color", "origin/main..."],
          stdout=mock.ANY,
          stderr=mock.ANY,
          text=True,
          check=False,
      )

  def test_get_commit_message(self):
    mock_result = mock.Mock(
        returncode=0, stdout="feat: new feature\n", stderr=""
    )
    with mock.patch("subprocess.run", return_value=mock_result) as mock_run:
      msg = custom_lint.get_commit_message("origin/main...")
      self.assertEqual(msg, "feat: new feature\n")
      mock_run.assert_called_once_with(
          ["git", "log", "origin/main..HEAD", "--format=%B"],
          capture_output=True,
          text=True,
          check=False,
      )

  def test_get_modified_lines_from_vcs_hg(self):
    prefix = f"{_TP_PREFIX}/" if _TP_PREFIX else "b/"
    hg_diff_output = f"""+++ {prefix}common/BUILD
@@ -5 +5 @@
-old
+new
"""

    def fake_run(cmd, **kwargs):
      del kwargs
      if cmd[0] == "git":
        return mock.Mock(
            returncode=1, stdout="", stderr="fatal: not a git repo"
        )
      if cmd[0] == "hg":
        return mock.Mock(returncode=0, stdout=hg_diff_output, stderr="")
      return mock.Mock(returncode=1, stdout="", stderr="")

    with mock.patch("subprocess.run", side_effect=fake_run):
      modified = custom_lint.get_modified_lines_from_vcs()
      self.assertIsNotNone(modified)
      self.assertIn("common/BUILD", modified)
      self.assertEqual(modified["common/BUILD"], {5})
      if _TP_PREFIX:
        self.assertNotIn(f"{_TP_PREFIX}/common/BUILD", modified)

  def test_normalize_diff_path(self):
    if _TP_PREFIX:
      self.assertEqual(
          custom_lint.normalize_diff_path(f"b/{_TP_PREFIX}/pkg/BUILD.bazel"),
          "pkg/BUILD.bazel",
      )
      self.assertEqual(
          custom_lint.normalize_diff_path(
              f"//depot/{_G3}/{_TP_PREFIX}/pkg/BUILD.bazel"
          ),
          "pkg/BUILD.bazel",
      )
    self.assertEqual(
        custom_lint.normalize_diff_path("b/pkg/BUILD.bazel"),
        "pkg/BUILD.bazel",
    )
    self.assertEqual(
        custom_lint.normalize_diff_path("a/pkg/BUILD.bazel"),
        "pkg/BUILD.bazel",
    )

  def test_delta_only_matches_vcs_normalized_path(self):
    rule = self._get_cc_target_rule()  # CheckCcTargetUsage
    content = 'cc_library(name = "test")\n# Untouched line 2\n# Line 3\n'
    prefix = f"{_TP_PREFIX}/" if _TP_PREFIX else ""
    # VCS diff reports modification on line 2 for BUILD
    diff_text = f"""+++ b/{prefix}BUILD
@@ -2 +2 @@
-# old comment
+# Untouched line 2
"""
    modified_lines = custom_lint.parse_unified_diff(diff_text)
    # The candidate file is evaluated with rel_path="BUILD"
    violations = custom_lint.check_file_content(
        content=content,
        lines=content.splitlines(),
        rel_path="BUILD",
        rule=rule,
        modified_lines=modified_lines,
    )
    # Line 1 has violation but wasn't modified -> ignored
    self.assertEqual(violations, [])

    # If line 1 was modified in diff:
    diff_text_line1 = f"""+++ b/{prefix}BUILD
@@ -1 +1 @@
-old
+cc_library(name = "test")
"""
    modified_lines_line1 = custom_lint.parse_unified_diff(diff_text_line1)
    violations_line1 = custom_lint.check_file_content(
        content=content,
        lines=content.splitlines(),
        rel_path="BUILD",
        rule=rule,
        modified_lines=modified_lines_line1,
    )
    self.assertEqual(len(violations_line1), 1)
    self.assertEqual(violations_line1[0].line_number, 1)


if __name__ == "__main__":
  unittest.main()
