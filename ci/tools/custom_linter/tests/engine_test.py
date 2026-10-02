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

"""Tests for custom_linter.engine and custom_linter.metadata."""

from collections.abc import Iterator, Sequence
import concurrent.futures
import dataclasses
import functools
import io
import json
import multiprocessing
import os
import pathlib
import random
import re
import sys
import textwrap
from unittest import mock

import pytest
from ci.tools.custom_linter import engine as custom_lint
from ci.tools.custom_linter import metadata as custom_lint_metadata

_BASE = "//depot/repo/proj"


@functools.lru_cache(maxsize=1)
def _project_rules() -> dict[str, custom_lint.Rule]:
  return {r.id: r for r in custom_lint.load_default_rules().rules}


def _rule(rule_id: str) -> custom_lint.Rule:
  return _project_rules()[rule_id]


def _load(text: str) -> custom_lint_metadata.RuleSet:
  return custom_lint_metadata.parse_metadata(text, "METADATA", _BASE)


def _lines(violations) -> list[int | None]:
  return [v.line_number for v in violations]


def _check(rule_id: str, content: str, rel_path: str, modified=None, op=None):
  return custom_lint.check_file_content(
      content=content,
      rel_path=rel_path,
      rule=_rule(rule_id),
      modified_lines=modified,
      file_operation=op,
  )


# ---------------------------------------------------------------------------
# Unified diff parsing tests
# ---------------------------------------------------------------------------


def test_parse_unified_diff() -> None:
  diff_text = textwrap.dedent("""\
      +++ b/foo/BUILD.bazel\t2026-08-26 23:52:54.000000000 +0000
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
      """)
  modified = custom_lint.parse_unified_diff(diff_text)
  assert "foo/BUILD.bazel" in modified
  assert modified["foo/BUILD.bazel"] == {11, 12, 13}
  assert "bar/BUILD" in modified
  assert modified["bar/BUILD"] == {5}
  assert "baz/file.txt" in modified
  assert modified["baz/file.txt"] == {1}
  assert "simple/main.py" in modified
  assert modified["simple/main.py"] == {10, 11}
  assert "deletion/file.py" in modified
  assert modified["deletion/file.py"] == set()
  assert "/dev/null" not in modified
  assert "dev/null" not in modified


def test_file_operations_from_git_diff() -> None:
  diff_text = textwrap.dedent("""\
      diff --git a/p/BUILD b/p/BUILD
      --- a/p/BUILD
      +++ b/p/BUILD
      @@ -43,0 +44,1 @@
      +a
      diff --git a/new.py b/new.py
      new file mode 100644
      --- /dev/null
      +++ b/new.py
      @@ -0,0 +1,2 @@
      +x
      +--- not a header
      diff --git a/gone.cc b/gone.cc
      deleted file mode 100644
      --- a/gone.cc
      +++ /dev/null
      @@ -1,2 +0,0 @@
      -a
      -b
      diff --git a/old.cc b/moved.cc
      similarity index 100%
      rename from old.cc
      rename to moved.cc
      diff --git a/src.h b/copy.h
      similarity index 90%
      copy from src.h
      copy to copy.h
      --- a/src.h
      +++ b/copy.h
      @@ -3 +3 @@
      -old
      +new
      """)
  modified = custom_lint.parse_unified_diff(diff_text)
  assert modified.operations == {
      "p/BUILD": "EDIT",
      "new.py": "ADD",
      "moved.cc": "BRANCH",
      "copy.h": "ADD",
  }
  assert modified["new.py"] == {1, 2}
  assert modified["moved.cc"] == set()
  assert "gone.cc" not in modified
  assert "not a header" not in "".join(modified)


def test_new_file_in_plain_diff() -> None:
  diff_text = "--- /dev/null\n+++ b/a.py\n@@ -0,0 +1 @@\n+x\n"
  modified = custom_lint.parse_unified_diff(diff_text)
  assert modified.operations == {"a.py": "ADD"}


def test_parse_unified_diff_with_prefix_strips_and_filters() -> None:
  prefix = "sub/proj/"
  diff_text = textwrap.dedent(f"""\
      diff --git a/{prefix}foo/BUILD.bazel b/{prefix}foo/BUILD.bazel
      --- a/{prefix}foo/BUILD.bazel
      +++ b/{prefix}foo/BUILD.bazel
      @@ -10,0 +11,3 @@
      +cc_library(
      +    name = "lib",
      +)
      +++ {prefix}bar/BUILD
      @@ -5 +5 @@
      -py_library(name = "x")
      +cc_test(name = "x")
      diff --git a/other_pkg/outside.py b/other_pkg/outside.py
      new file mode 100644
      --- /dev/null
      +++ b/other_pkg/outside.py
      @@ -0,0 +1,2 @@
      +x = 1
      +y = 2
      """)
  modified = custom_lint.parse_unified_diff(diff_text, prefix=prefix)
  assert set(modified) == {"foo/BUILD.bazel", "bar/BUILD"}
  assert modified["foo/BUILD.bazel"] == {11, 12, 13}
  assert modified["bar/BUILD"] == {5}
  assert "other_pkg/outside.py" not in modified
  assert "outside.py" not in modified


def test_normalize_diff_path(tmp_path: pathlib.Path) -> None:
  assert (
      custom_lint.normalize_diff_path("b/pkg/BUILD.bazel") == "pkg/BUILD.bazel"
  )
  assert (
      custom_lint.normalize_diff_path("a/pkg/BUILD.bazel") == "pkg/BUILD.bazel"
  )
  assert (
      custom_lint.normalize_diff_path(
          "b/sub/proj/pkg/BUILD.bazel", prefix="sub/proj/"
      )
      == "pkg/BUILD.bazel"
  )
  assert (
      custom_lint.normalize_diff_path(
          "a/sub/proj/pkg/BUILD.bazel", prefix="sub/proj"
      )
      == "pkg/BUILD.bazel"
  )
  assert not custom_lint.normalize_diff_path(
      "b/other/pkg/BUILD.bazel", prefix="sub/proj/"
  )
  root = tmp_path.resolve()
  assert (
      custom_lint.normalize_diff_path(
          str(root / "pkg/BUILD.bazel"), root_dir=root
      )
      == "pkg/BUILD.bazel"
  )
  assert not custom_lint.normalize_diff_path(
      "/outside/root/BUILD.bazel", root_dir=root
  )


# ---------------------------------------------------------------------------
# Textproto parser tests
# ---------------------------------------------------------------------------


def test_scalars_and_nested_messages() -> None:
  msg = custom_lint_metadata.parse_textproto("""
      # comment
      name: "a"  # trailing comment
      count: 3
      enabled: true
      status: ERROR
      child { inner: "x" }
      child: { inner: "y" }
      other < inner: "z" >
      """)
  assert msg.string("name") == "a"
  assert msg.ident("count") == "3"
  assert msg.bool("enabled")
  assert msg.ident("status") == "ERROR"
  assert [m.string("inner") for m, _ in msg.messages("child")] == ["x", "y"]
  other = msg.message("other")
  assert other is not None
  assert other.string("inner") == "z"


def test_adjacent_strings_are_concatenated() -> None:
  msg = custom_lint_metadata.parse_textproto(
      'msg:\n  "foo "\n  \'bar\'\n  "baz"\n'
  )
  assert msg.string("msg") == "foo barbaz"


def test_escapes() -> None:
  msg = custom_lint_metadata.parse_textproto(
      r're: "\\bos\\.(a|b)\\b\n\t\"q\" \101\x42 \'"'
  )
  assert msg.string("re") == '\\bos\\.(a|b)\\b\n\t"q" AB \''


def test_repeated_fields_lists_and_extensions() -> None:
  msg = custom_lint_metadata.parse_textproto(
      'tag: "a" tag: ["b", "c"] m [{x: 1}, {x: 2}] empty: [] '
      "ext { [a.b.C] { d: 1 } [type.googleapis.com/x.Y] { e: 2 } }"
  )
  assert msg.strings("tag") == ["a", "b", "c"]
  assert len(list(msg.messages("m"))) == 2
  assert not list(msg.fields_named("empty"))
  ext = msg.message("ext")
  assert ext is not None
  assert [f.name for f in ext.fields] == [
      "[a.b.C]",
      "[type.googleapis.com/x.Y]",
  ]


def test_field_lines() -> None:
  msg = custom_lint_metadata.parse_textproto("a: 1\n\nb {\n  c: 2\n}\n")
  assert [f.line for f in msg.fields] == [1, 3]
  b_msg = msg.message("b")
  assert b_msg is not None
  assert b_msg.fields[0].line == 4


@pytest.mark.parametrize(
    ("text", "line", "message"),
    [
        ('a {\n  b: "x"\n', 2, "expected `}`"),
        ('a: "unterminated\n', 1, "newline in string"),
        ('a "x"', 1, "expected `:`"),
        ('a: "\\q"', 1, "unknown escape"),
    ],
)
def test_errors_have_positions(text: str, line: int, message: str) -> None:
  with pytest.raises(
      custom_lint_metadata.ParseError, match=message
  ) as exc_info:
    custom_lint_metadata.parse_textproto(text)
  assert exc_info.value.line == line


# ---------------------------------------------------------------------------
# Metadata loader tests
# ---------------------------------------------------------------------------


def test_basic_prohibited_rule() -> None:
  rules = _load(r"""
      presubmit {
        path_regexp: ".*\\.py"
        path_expression_exclusion: "tests/..."
        check_contents {
          base { id: "CheckEnv" disable_tags: "ROLLBACK_OF" disable_tags: "OK" }
          failure_status: WARNING
          failure_message: "no " "env"
          prohibited_regexp: "\\bos\\.environ\\b"
          suppression_regexp: "OK=.+"
          check_delta_only: true
        }
      }""").rules
  assert len(rules) == 1
  r = rules[0]
  assert r.id == "CheckEnv"
  assert r.severity == "WARNING"
  assert r.failure_message == "no env"
  assert r.disable_tags == ("ROLLBACK_OF", "OK")
  assert r.check_delta_only
  assert r.mode == custom_lint_metadata.MODE_LINE
  assert r.matches_path(f"{_BASE}/a/b.py")
  assert not r.matches_path(f"{_BASE}/tests/b.py")
  assert not r.matches_path(f"{_BASE}/a/b.cc")
  assert not r.matches_path("//depot/repo/other/a.py")


def test_default_scope_and_defaults() -> None:
  r = _load('presubmit { check_contents { prohibited_regexp: "x" } }').rules[0]
  assert r.matches_path(f"{_BASE}/any/file.txt")
  assert not r.matches_path("//depot/repo/projx/file.txt")
  assert r.severity == "ERROR"
  assert r.id == "CheckContents@1"
  assert r.failure_message == "Check CheckContents@1 failed."
  assert not r.check_delta_only


def test_rooted_paths_are_not_prefixed() -> None:
  r = _load(r"""presubmit {
      path_regexp: "//depot/repo/proj/.*\\.(cc|h)"
      path_regexp_exclusion: ".*_test\\.cc"
      path_expression_exclusion: "//depot/.../*.METADATA"
      check_contents { prohibited_regexp: "x" }
    }""").rules[0]
  assert r.matches_path(f"{_BASE}/a/b.cc")
  assert not r.matches_path(f"{_BASE}/a/b_test.cc")
  assert not r.matches_path(f"{_BASE}/a.METADATA")


def test_path_regexp_is_a_full_match_and_base_is_escaped() -> None:
  r = _load(
      'presubmit { path_regexp: "foo" check_contents { prohibited_regexp:'
      ' "x" } }'
  ).rules[0]
  assert r.matches_path(f"{_BASE}/foo")
  assert not r.matches_path(f"{_BASE}/foo.txt")
  assert not r.matches_path(f"{_BASE}/a/foo")
  r = custom_lint_metadata.parse_metadata(
      'presubmit { path_regexp: "a" check_contents { prohibited_regexp:'
      ' "x" } }',
      "M",
      "//r/v1.0",
  ).rules[0]
  assert r.matches_path("//r/v1.0/a")
  assert not r.matches_path("//r/v1x0/a")


def test_match_modes() -> None:
  rules = _load("""presubmit {
      check_contents { base { id: "A" } required_regexp: "x" match_entire_file: true ignore_large_files: true }
      check_contents { base { id: "B" } prohibited_regexp: "x" check_delta_only: true match_multi_line: true }
      check_contents { base { id: "C" } prohibited_regexp: "x" match_multi_line: true }
    }""").rules
  assert [(r.id, r.mode) for r in rules] == [
      ("A", custom_lint_metadata.MODE_ENTIRE_FILE),
      ("B", custom_lint_metadata.MODE_MODIFIED_CHUNKS),
      ("C", custom_lint_metadata.MODE_LINE),
  ]
  assert rules[0].ignore_large_files


def test_operations_and_file_operation_checks() -> None:
  rules = _load("""
      presubmit {
        file_operation: ADD
        file_operation: BRANCH
        check_contents { base { id: "Block" } prohibited_regexp: "x" }
        check_contents { base { id: "Own" } prohibited_regexp: "x" operation: EDIT }
        check_file_operation { base { id: "Op" } operation: ADD }
      }
      presubmit { check_contents { base { id: "All" } prohibited_regexp: "x" } }
      """).rules
  assert [(r.id, r.operations) for r in rules] == [
      ("Block", ("ADD", "BRANCH")),
      ("Own", ("EDIT",)),
      ("Op", ("ADD",)),
      ("All", ()),
  ]
  assert rules[2].kind == custom_lint_metadata.CHECK_FILE_OPERATION
  own = rules[1]
  assert own.applies_to("EDIT")
  assert not own.applies_to("ADD")
  assert not own.applies_to(None)
  assert rules[3].applies_to(None)


def test_unsupported_checks_are_skipped() -> None:
  rule_set = _load("""presubmit {
      check_description { base { id: "Desc" } required_regexp: "x" }
      check_presubmit_service { base { id: "Svc" } extended_options { [a.b] { c: 1 } } }
      include_presubmit: "//depot/repo/x/y.METADATA"
      check_contents { base { id: "Del" } prohibited_regexp: "x" check_deleted_only: true }
    }
    presubmit {
      path_regexp: "(?<name>x)\\\\C"
      check_lint { action: REVIEW }
    }""")
  assert not rule_set.rules
  assert [(s.kind, s.id) for s in rule_set.skipped] == [
      ("check_description", "Desc"),
      ("check_presubmit_service", "Svc"),
      ("include_presubmit", "//depot/repo/x/y.METADATA"),
      ("check_contents.check_deleted_only", "Del"),
      ("check_lint", None),
  ]


@pytest.mark.parametrize(
    ("text", "line", "message"),
    [
        (
            'presubmit {\n  check_contents {\n    base { id: "Bad" }\n'
            + '    prohibited_regexp: "("\n  }\n}',
            2,
            "Bad: invalid regex in `prohibited_regexp`",
        ),
        (
            'presubmit { check_contents { failure_message: "m" } }',
            1,
            "needs `prohibited_regexp` or `required_regexp`",
        ),
        (
            'presubmit { check_contents { prohibited_regexp: "x"'
            + " check_delta_only: true match_entire_file: true } }",
            1,
            "incompatible",
        ),
        (
            'presubmit { check_contents { prohibited_regexp: "x"'
            + " failure_status: FATAL } }",
            1,
            "unknown `failure_status` `FATAL`",
        ),
        (
            'presubmit {\n  path_regexp: "("\n  check_contents {'
            + ' prohibited_regexp: "x" }\n}',
            1,
            "path_regexp",
        ),
        ('presubmit { check_contents { prohibited_regexp: "x" }', 1, "syntax"),
        (
            'presubmit { check_contents { prohibited_regexp: "x" operation:'
            + " MOVE } }",
            1,
            "unknown `operation` `MOVE`",
        ),
        (
            'presubmit { check_contents { base { id: "A" } prohibited_regexp:'
            + ' "x" } check_contents { base { id: "A" } prohibited_regexp: "y"'
            + " } }",
            1,
            "duplicate rule id `A`",
        ),
    ],
)
def test_errors_point_at_the_offending_check(
    text: str, line: int, message: str
) -> None:
  with pytest.raises(
      custom_lint_metadata.LoadError, match=re.escape(message)
  ) as exc_info:
    _load(text)
  assert exc_info.value.line == line


def test_project_rules_load() -> None:
  rule_set = custom_lint.load_default_rules()
  assert len(rule_set.rules) > 50
  assert not rule_set.skipped
  for rule in rule_set.rules:
    rule.prewarm()


# ---------------------------------------------------------------------------
# PathMapper tests
# ---------------------------------------------------------------------------


def test_prefix_maps_longest_first() -> None:
  g3 = "//depot/repo/tt"
  mapper = custom_lint_metadata.PathMapper(
      f"{g3}/oss/",
      [
          ("src/torch_tpu", g3),
          ("src/torch_tpu/_internal/special/", f"{g3}/oss/torch_tpu/special"),
          ("/csrc/", f"{g3}/csrc/"),
      ],
  )
  assert (
      mapper.to_canonical("src/torch_tpu/_internal/env.py")
      == f"{g3}/_internal/env.py"
  )
  assert (
      mapper.to_canonical("src/torch_tpu/_internal/special/x.py")
      == f"{g3}/oss/torch_tpu/special/x.py"
  )
  assert mapper.to_canonical("csrc/ops/add.cc") == f"{g3}/csrc/ops/add.cc"
  assert mapper.to_canonical("csrcx/a.cc") == f"{g3}/oss/csrcx/a.cc"
  assert mapper.to_canonical("ci/tools/x.py") == f"{g3}/oss/ci/tools/x.py"


@pytest.mark.skipif(
    not (
        custom_lint.find_torch_tpu_root() / custom_lint.MONOREPO_RULES_FILE
    ).is_file(),
    reason="Needs the monorepo rules file",
)
def test_oss_export_matches_the_same_files() -> None:
  g3_rules = custom_lint.load_default_rules().rules
  project_root = custom_lint.find_torch_tpu_root()
  rules_path = custom_lint.find_rules_file(project_root)
  g3_mapper = custom_lint.default_path_mapper(project_root)
  text = rules_path.read_text(encoding="utf-8")
  text = re.sub(
      rf'(path_\w+): "{re.escape(g3_mapper.canonical_root)}',
      r'\1: "torch_tpu',
      text,
  )
  text = re.sub(r'(path_\w+): "//depot/\.\.\./', r'\1: ".../', text)
  oss_rules = custom_lint_metadata.parse_metadata(
      text, "oss", "torch_tpu"
  ).rules
  oss_mapper = custom_lint_metadata.PathMapper(
      custom_lint.OSS_CANONICAL_ROOT, custom_lint.OSS_PATH_MAPS
  )
  pairs = [
      ("_internal/env.py", "src/torch_tpu/_internal/env.py"),
      ("_internal/env_test.py", "src/torch_tpu/_internal/env_test.py"),
      ("ops/foo.py", "src/torch_tpu/ops/foo.py"),
      ("BUILD", "src/torch_tpu/BUILD"),
      ("csrc/ops/add/add_kernels.cc", "csrc/ops/add/add_kernels.cc"),
      ("csrc/ops/add/helper.cc", "csrc/ops/add/helper.cc"),
      ("csrc/ops/addkernel.cc", "csrc/ops/addkernel.cc"),
      ("csrc/common/BUILD", "csrc/common/BUILD"),
      ("csrc/common/env_vars.h", "csrc/common/env_vars.h"),
      ("tests/ops_test.py", "tests/ops_test.py"),
      ("tests/errors_test.py", "tests/errors_test.py"),
      ("docs/guide.md", "docs/guide.md"),
      ("examples/train.py", "examples/train.py"),
      ("oss/ci/tools/gen.py", "ci/tools/gen.py"),
      ("oss/shims/py_rules/pytype.bzl", "shims/py_rules/pytype.bzl"),
      ("build_files/build_defs.bzl", "build_files/build_defs.bzl"),
      (
          "oss/torch_tpu/csrc/common/environment.cc",
          "csrc/common/environment.cc",
      ),
  ]
  assert [r.id for r in g3_rules] == [r.id for r in oss_rules]
  for g3_rel, oss_rel in pairs:
    g3_path = g3_mapper.to_canonical(g3_rel)
    oss_path = oss_mapper.to_canonical(oss_rel)
    for g3_rule, oss_rule in zip(g3_rules, oss_rules):
      assert g3_rule.matches_path(g3_path) == oss_rule.matches_path(
          oss_path
      ), f"{g3_rule.id} on {g3_rel}"


def test_default_path_mapper_and_find_rules_file_by_layout(
    tmp_path: pathlib.Path,
) -> None:
  g3_dir_name = custom_lint.MONOREPO_DIR_NAME
  g3_root = (tmp_path / g3_dir_name / "pkg" / "my_proj").resolve()
  g3_root.mkdir(parents=True)
  g3_rules = g3_root / custom_lint.MONOREPO_RULES_FILE
  g3_rules.write_text("", encoding="utf-8")
  assert custom_lint.find_rules_file(g3_root) == g3_rules
  g3_mapper = custom_lint.default_path_mapper(g3_root)
  assert g3_mapper.canonical_root == f"//depot/{g3_dir_name}/pkg/my_proj"

  oss_root = (tmp_path / "oss_repo").resolve()
  oss_rules = oss_root / custom_lint.OSS_RULES_FILE
  oss_rules.parent.mkdir(parents=True)
  oss_rules.write_text("", encoding="utf-8")
  assert custom_lint.find_rules_file(oss_root) == oss_rules
  oss_mapper = custom_lint.default_path_mapper(oss_root)
  assert oss_mapper.canonical_root == custom_lint.OSS_CANONICAL_ROOT


# ---------------------------------------------------------------------------
# Tags tests
# ---------------------------------------------------------------------------


def test_parse_tags() -> None:
  tags = custom_lint_metadata.parse_tags(
      "Summary line\n\nBody mentions ALLOW_CC_TARGETS in prose.\n"
      "BUG=123\n  ROLLBACK_OF=456\nMARKDOWN = true\nlower=x\nX1_Y=\n"
      "Do not set ALLOW_FOO=1 here\n"
  )
  assert tags == {"BUG", "ROLLBACK_OF", "MARKDOWN", "X1_Y"}


def test_is_rule_disabled() -> None:
  rule = custom_lint.Rule(
      id="TestRule",
      path_regex=r".*",
      prohibited_regex=r"foo",
      failure_message="msg",
      disable_tags=["ALLOW_FOO", "ROLLBACK_OF"],
  )
  assert custom_lint.is_rule_disabled(rule, "Fix: something\nALLOW_FOO=needed")
  assert custom_lint.is_rule_disabled(rule, "Revert\n\nROLLBACK_OF=1")
  assert not custom_lint.is_rule_disabled(rule, "Automated ROLLBACK_OF cl/123")
  assert not custom_lint.is_rule_disabled(rule, "Normal commit")


# ---------------------------------------------------------------------------
# CheckFileContent tests
# ---------------------------------------------------------------------------


def test_cc_library_violation() -> None:
  violations = _check(
      "CheckCcTargetUsage",
      'cc_library(name = "test")\n',
      "BUILD",
      {"BUILD": {1}},
  )
  assert len(violations) == 1
  assert violations[0].rule_id == "CheckCcTargetUsage"
  assert violations[0].line_number == 1
  assert violations[0].severity == "ERROR"


def test_cc_library_suppressed_on_same_line() -> None:
  violations = _check(
      "CheckCcTargetUsage",
      'cc_library(name = "test")  # ALLOW_CC_TARGETS=shim\n',
      "BUILD",
      {"BUILD": {1}},
  )
  assert not violations


def test_line_mode_suppression_must_be_on_the_same_line() -> None:
  content = 'cc_library(\n    # ALLOW_CC_TARGETS=shim\n    name = "test",\n)\n'
  violations = _check("CheckCcTargetUsage", content, "BUILD", {"BUILD": {1, 2}})
  assert _lines(violations) == [1]


def test_delta_only_ignores_untouched_lines_and_files() -> None:
  content = 'cc_library(name = "test")\n# Untouched line 2\n'
  assert not _check("CheckCcTargetUsage", content, "BUILD", {"BUILD": {2}})
  assert not _check(
      "CheckCcTargetUsage", content, "BUILD", {"OTHER_FILE.txt": {1}}
  )
  assert _lines(_check("CheckCcTargetUsage", content, "BUILD")) == [1]


def test_delta_only_matches_vcs_normalized_path() -> None:
  content = 'cc_library(name = "test")\n# Untouched line 2\n# Line 3\n'
  prefix = "sub/proj/"
  modified_lines = custom_lint.parse_unified_diff(
      f"+++ b/{prefix}BUILD\n@@ -2 +2 @@\n-# old comment\n+# Untouched line"
      " 2\n",
      prefix=prefix,
  )
  assert not _check("CheckCcTargetUsage", content, "BUILD", modified_lines)
  modified_lines = custom_lint.parse_unified_diff(
      f'+++ b/{prefix}BUILD\n@@ -1 +1 @@\n-old\n+cc_library(name = "test")\n',
      prefix=prefix,
  )
  assert _lines(
      _check("CheckCcTargetUsage", content, "BUILD", modified_lines)
  ) == [1]


def test_line_anchored_patterns_match_every_line() -> None:
  content = '"""Doc."""\n\nimport os\nfrom . import sibling\n'
  assert _lines(_check("CheckRelativeImports", content, "ops/foo.py")) == [4]
  content = "#include <x>\n\n  #define FOO 1\n#define TT_OK 1\n"
  assert _lines(_check("CheckMacroNamePrefix", content, "csrc/a.cc")) == [3]


def test_line_numbers_ignore_form_feeds() -> None:
  content = "a = 1\n\x0c\nfrom . import x\n"
  assert _lines(_check("CheckRelativeImports", content, "ops/foo.py")) == [3]


def test_required_regexp_by_line() -> None:
  rel = "csrc/ops/add/add_kernels.cc"
  violations = _check("CheckKernelMacro", "int x;\n", rel)
  assert _lines(violations) == [None]
  assert not _check("CheckKernelMacro", "// TT_KERNEL is not used.\n", rel)
  assert not _check(
      "CheckKernelMacro", "TT_KERNEL(Add)\nint x;\n", rel, {rel: {2}}
  )


def test_required_regexp_entire_file() -> None:
  ok = 'load("//x")\n\ntorch_tpu_package_end()\n'
  assert not _check("CheckTorchTpuPackageEnd", ok, "csrc/BUILD")
  bad = "torch_tpu_package_end()\nfoo()\n"
  assert _lines(_check("CheckTorchTpuPackageEnd", bad, "csrc/BUILD")) == [None]


def test_prohibited_entire_file() -> None:
  assert _lines(
      _check("CheckIncludeBeforeCopyright", "\n#include <x>\n", "a.h")
  ) == [1]
  assert not _check(
      "CheckIncludeBeforeCopyright", "// Copyright\n#include <x>\n", "a.h"
  )


def test_modified_chunks() -> None:
  rel = "pkg/BUILD"
  content = (
      'py_library(\n    # PY_LIBRARY_OK=shim\n    name = "a",\n)\n'
      'py_library(\n    name = "b",\n)\n'
  )
  assert not _check("CheckPyLibraryUsage", content, rel, {rel: {1, 2, 3, 4}})
  assert _lines(_check("CheckPyLibraryUsage", content, rel, {rel: {5, 6}})) == [
      5
  ]
  assert _lines(_check("CheckPyLibraryUsage", content, rel, {rel: {2, 5}})) == [
      5
  ]


def test_file_operation_rule() -> None:
  rel = "csrc/ops/addkernel.cc"
  assert _lines(_check("CheckOpFileName", "x\n", rel, {rel: {1}}, "ADD")) == [
      None
  ]
  assert not _check("CheckOpFileName", "x\n", rel, {rel: {1}}, "EDIT")
  assert not _check("CheckOpFileName", "x\n", rel)


def test_operation_is_read_from_modified_lines() -> None:
  rel = "csrc/ops/addkernel.cc"
  modified = custom_lint.parse_unified_diff(
      f"--- /dev/null\n+++ b/{rel}\n@@ -0,0 +1 @@\n+x\n"
  )
  assert _lines(
      custom_lint.check_file_content(
          "x\n", rel, _rule("CheckOpFileName"), modified
      )
  ) == [None]


@pytest.mark.parametrize(
    "path",
    [
        "custom_lint.py",
        "custom_lint_metadata.py",
        "custom_lint_vcs.py",
        "custom_lint_rules.METADATA",
        "ci/custom_lint_rules.textproto",
        "custom_linter/engine.py",
        "custom_linter/metadata.py",
        "custom_linter/vcs.py",
        "oss/ci/tools/custom_linter/__init__.py",
        "oss/ci/tools/custom_linter/__main__.py",
        "oss/ci/tools/custom_linter/engine.py",
        "oss/ci/tools/custom_linter/metadata.py",
        "oss/ci/tools/custom_linter/vcs.py",
        "oss/ci/tools/custom_linter/tests/engine_test.py",
        "oss/ci/tools/custom_linter/tests/vcs_test.py",
        "ci/tools/custom_linter/engine.py",
        "ci/tools/custom_linter/tests/engine_test.py",
        "copy.bara.sky",
        "tests/ops_test_gpu_golden2.gz",
        "docs/images/logo.png",
        "model/topology.pb",
    ],
)
def test_is_path_excluded_true(path: str) -> None:
  assert custom_lint.is_path_excluded(path)


@pytest.mark.parametrize(
    "path",
    [
        "torch_tpu/csrc/core/foo.cc",
        "examples/some_model.py",
        "google/examples/benchmarks/foo_test.py",
    ],
)
def test_is_path_excluded_false(path: str) -> None:
  assert not custom_lint.is_path_excluded(path)


def test_rule_matches_path() -> None:
  rule = custom_lint.Rule(
      id="TestRule",
      path_regex=r".*\.(cc|cpp|h|hpp)$",
      prohibited_regex=r"\bfoo\b",
      failure_message="No foo",
      path_exclusion_regex=r".*_test\.(cc|cpp|h)",
  )
  assert rule.matches_path("src/core/model.cc")
  assert not rule.matches_path("src/core/model_test.cc")
  assert not rule.matches_path("src/core/model.py")

  mapper = custom_lint.default_path_mapper()
  cc_rule = _rule("CheckCcTargetUsage")
  assert cc_rule.matches_path(mapper.to_canonical("BUILD"))
  assert cc_rule.matches_path(mapper.to_canonical("tools/BUILD"))
  assert cc_rule.matches_path(mapper.to_canonical("tools/build_defs.bzl"))
  assert not cc_rule.matches_path(mapper.to_canonical("rebuild.cc"))
  assert not cc_rule.matches_path(mapper.to_canonical("BUILD_GUIDE.md"))
  assert not cc_rule.matches_path(mapper.to_canonical("builder_utils.py"))


# ---------------------------------------------------------------------------
# Candidate files & directory-walk tests
# ---------------------------------------------------------------------------


def test_find_candidate_files(tmp_path: pathlib.Path) -> None:
  root_path = tmp_path.resolve()
  cc_file = root_path / "foo.cc"
  cc_file.write_text("content\n", encoding="utf-8")
  (root_path / "bar.py").write_text("content\n", encoding="utf-8")
  (root_path / "sub").mkdir()
  (root_path / "sub" / "baz.cc").write_text("content\n", encoding="utf-8")

  rule_cc = custom_lint.Rule(
      id="RuleCC",
      path_regex=r"//p/.*\.cc",
      prohibited_regex=r"foo",
      failure_message="msg",
  )
  mapper = custom_lint_metadata.PathMapper("//p")
  candidates = custom_lint.find_candidate_files(
      root_path, [rule_cc], mapper=mapper
  )
  assert [c.rel_path for c in candidates] == ["foo.cc", "sub/baz.cc"]
  assert isinstance(candidates[0], custom_lint.CandidateFile)
  assert candidates[0].file_path == cc_file
  assert candidates[0].matching_rules == [rule_cc]
  candidates = custom_lint.find_candidate_files(
      root_path / "sub",
      [rule_cc],
      mapper=mapper,
      rel_prefix="sub",
      max_workers=1,
  )
  assert [c.rel_path for c in candidates] == ["sub/baz.cc"]


def test_default_ignore_dirs(tmp_path: pathlib.Path) -> None:
  assert "node_modules" in custom_lint.DEFAULT_IGNORE_DIRS
  assert "bazel-bin" in custom_lint.DEFAULT_IGNORE_DIRS
  root_path = tmp_path.resolve()
  for d in (
      "src",
      "node_modules",
      "venv",
      "bazel-bin",
      "bazel-custom",
      "other-bin",
  ):
    (root_path / d).mkdir()
    (root_path / d / "x.cc").write_text("foo\n", encoding="utf-8")
  # Symlinked directories are never followed.
  (root_path / "linked").symlink_to(root_path / "src", target_is_directory=True)

  rule_any_cc = custom_lint.Rule(
      id="RuleCC",
      path_regex=r".*\.cc$",
      prohibited_regex=r"foo",
      failure_message="msg",
  )
  monorepo_mapper = custom_lint_metadata.PathMapper("//depot/repo/proj")
  oss_mapper = custom_lint_metadata.PathMapper(
      custom_lint.OSS_CANONICAL_ROOT, custom_lint.OSS_PATH_MAPS
  )
  # The default ignore set does not depend on the repository layout.
  for mapper in (monorepo_mapper, oss_mapper):
    candidates = custom_lint.find_candidate_files(
        root_path, [rule_any_cc], mapper=mapper
    )
    assert [c.rel_path for c in candidates] == ["other-bin/x.cc", "src/x.cc"]


def test_exclude_dir_and_no_default_excludes(tmp_path: pathlib.Path) -> None:
  root_path = tmp_path.resolve()
  (root_path / "pyproject.toml").write_text("", encoding="utf-8")
  rules_file = root_path / "rules.METADATA"
  rules_file.write_text(
      'presubmit { path_regexp: ".*\\\\.cc"\n'
      '  check_contents { base { id: "NoFoo" } prohibited_regexp: "foo"'
      " failure_status: ERROR }\n"
      "}\n",
      encoding="utf-8",
  )
  for d in ("keep", "skip_a", "skip_b", "venv"):
    (root_path / d).mkdir()
    (root_path / d / "file.cc").write_text("foo\n", encoding="utf-8")

  json_out = io.StringIO()
  with (
      mock.patch("sys.stdout", json_out),
      mock.patch("sys.stderr", io.StringIO()),
  ):
    rc = custom_lint.main([
        "--root",
        str(root_path),
        "--all",
        "--jobs",
        "1",
        "--format",
        "json",
        "--rules",
        str(rules_file),
        "--exclude-dir",
        "skip_a",
        "--exclude-dir",
        "skip_b",
    ])
  assert rc == 1
  parsed = json.loads(json_out.getvalue())
  assert [v["file_path"] for v in parsed] == ["keep/file.cc"]

  res = custom_lint.collect_violations(
      root_path,
      check_all=True,
      jobs=1,
      rules_file=str(rules_file),
      exclude_dirs=["skip_a", "skip_b"],
      no_default_excludes=True,
  )
  assert [v.file_path for v in res.violations] == [
      "keep/file.cc",
      "venv/file.cc",
  ]


def test_binary_files_skipped_by_default(tmp_path: pathlib.Path) -> None:
  tag = "copybara:" + "strip"
  root_path = tmp_path.resolve()
  (root_path / "golden.gz").write_bytes(
      b"\x1f\x8b\x08\x00" + tag.encode() + b"\n"
  )
  (root_path / "binary_blob.cc").write_bytes(
      b"int x = 0;\x00" + tag.encode() + b"\n"
  )
  (root_path / "valid.cc").write_text(f"// {tag}\n", encoding="utf-8")

  rule = _rule("CheckCopybaraTags")
  candidates = custom_lint.find_candidate_files(
      root_path, [rule], mapper=custom_lint.default_path_mapper()
  )
  assert [c.rel_path for c in candidates] == ["binary_blob.cc", "valid.cc"]
  violations = []
  for c in candidates:
    violations.extend(custom_lint._check_candidate((c, None)))  # pylint: disable=protected-access
  assert len(violations) == 1
  assert violations[0].file_path == "valid.cc"


def test_run_checks_end_to_end(tmp_path: pathlib.Path) -> None:
  root_path = tmp_path.resolve()
  (root_path / "pyproject.toml").write_text("", encoding="utf-8")
  rules_file = root_path / "rules.METADATA"
  rules_file.write_text(
      'presubmit { path_regexp: ".*\\\\.cc"\n'
      '  check_contents { base { id: "NoFoo" } prohibited_regexp: "foo"'
      " failure_status: WARNING }\n"
      '  check_contents { base { id: "NoBar" } prohibited_regexp: "bar"'
      " failure_status: ERROR }\n"
      "}\n",
      encoding="utf-8",
  )
  (root_path / "a.cc").write_text("foo\n", encoding="utf-8")

  text_out = io.StringIO()
  with mock.patch("sys.stdout", text_out):
    exit_code = custom_lint.run_checks(
        root_path, check_all=True, jobs=1, rules_file=str(rules_file)
    )
  assert exit_code == 0
  assert "[NoFoo] (WARNING) a.cc:1" in text_out.getvalue()

  (root_path / "a.cc").write_text("foo\nbar\n", encoding="utf-8")
  report = root_path / "report.txt"
  json_out = io.StringIO()
  stderr_out = io.StringIO()
  with (
      mock.patch("sys.stdout", json_out),
      mock.patch("sys.stderr", stderr_out),
  ):
    exit_code = custom_lint.main([
        "--root",
        str(root_path),
        "--all",
        "--jobs",
        "1",
        "--format",
        "json",
        "--rules",
        str(rules_file),
        "--report-file",
        str(report),
    ])
  assert exit_code == 1
  parsed = json.loads(json_out.getvalue())
  assert [v["rule_id"] for v in parsed] == ["NoFoo", "NoBar"]
  assert "INFO: Checked 1 files" in stderr_out.getvalue()
  assert "# NoFoo  [WARNING]  1 findings" in report.read_text()
  assert "# NoBar  [ERROR]  1 findings" in report.read_text()

  err_out = io.StringIO()
  with mock.patch("sys.stderr", err_out):
    assert (
        custom_lint.run_checks(
            root_path,
            check_all=True,
            jobs=1,
            rules_file=str(rules_file),
            only_rules=["UnknownRule"],
        )
        == 2
    )
  assert "unknown rule id(s): UnknownRule" in err_out.getvalue()


def test_root_is_authoritative_and_default_vcs_root_resolution(
    tmp_path: pathlib.Path,
) -> None:
  vcs_root = tmp_path.resolve()
  (vcs_root / ".git").mkdir()
  (vcs_root / "pyproject.toml").write_text("", encoding="utf-8")
  (vcs_root / "parent.cc").write_text("foo\n", encoding="utf-8")

  sub_root = vcs_root / "sub_pkg"
  sub_root.mkdir()
  (sub_root / custom_lint.MONOREPO_RULES_FILE).write_text(
      'presubmit { path_regexp: ".*\\\\.cc"\n'
      '  check_contents { base { id: "NoFoo" } prohibited_regexp: "foo" }\n'
      "}\n",
      encoding="utf-8",
  )
  (sub_root / "child.cc").write_text("foo\n", encoding="utf-8")

  res = custom_lint.collect_violations(sub_root, check_all=True, jobs=1)
  assert [v.file_path for v in res.violations] == ["child.cc"]

  # Without --root, the nearest directory with a rules file (up to the VCS
  # root) is the project root, e.g. a project inside a monorepo workspace whose
  # `.hg` is above the monorepo root.
  nested_cwd = sub_root / "deeper"
  nested_cwd.mkdir()
  out_sub = io.StringIO()
  with (
      mock.patch("sys.stdout", out_sub),
      mock.patch("sys.stderr", io.StringIO()),
      mock.patch.dict(os.environ, {"BUILD_WORKING_DIRECTORY": str(nested_cwd)}),
  ):
    assert custom_lint.main(["--all", "--jobs", "1", "--format", "json"]) == 1
  assert [v["file_path"] for v in json.loads(out_sub.getvalue())] == [
      "child.cc"
  ]

  # No rules file between the cwd and the VCS root: ask for --root.
  other_cwd = vcs_root / "other"
  other_cwd.mkdir()
  err_no_root = io.StringIO()
  with (
      mock.patch("sys.stderr", err_no_root),
      mock.patch.dict(os.environ, {"BUILD_WORKING_DIRECTORY": str(other_cwd)}),
  ):
    assert custom_lint.main(["--all", "--jobs", "1"]) == 2
  assert "pass --root" in err_no_root.getvalue()

  oss_rules = vcs_root / custom_lint.OSS_RULES_FILE
  oss_rules.parent.mkdir(parents=True)
  oss_rules.write_text(
      'presubmit { path_regexp: ".*\\\\.cc"\n'
      '  check_contents { base { id: "NoFoo" } prohibited_regexp: "foo" }\n'
      "}\n",
      encoding="utf-8",
  )
  out_vcs_root = io.StringIO()
  with (
      mock.patch("sys.stdout", out_vcs_root),
      mock.patch("sys.stderr", io.StringIO()),
      mock.patch.dict(os.environ, {"BUILD_WORKING_DIRECTORY": str(other_cwd)}),
  ):
    assert (
        custom_lint.main([
            "--all",
            "--jobs",
            "1",
            "--format",
            "json",
            "--exclude-dir",
            "sub_pkg",
        ])
        == 1
    )
  assert [v["file_path"] for v in json.loads(out_vcs_root.getvalue())] == [
      "parent.cc"
  ]


# ---------------------------------------------------------------------------
# Literal pre-filter tests
# ---------------------------------------------------------------------------

_LITERAL_CASES = (
    ("foo", {"foo"}, set(), ["foo", "xfoox"]),
    (r"\bLOG\(ERROR\)", {"LOG(ERROR)"}, set(), ["  LOG(ERROR) << x;"]),
    (r"\.\(\)", {".()"}, set(), ["a.()"]),
    (r"x\\y", {"x\\y"}, set(), ["x\\y"]),
    (r"\w+\.cc", {".cc"}, set(), ["a.cc"]),
    (r"^\s*from \.", {"from ."}, set(), ["  from . import x"]),
    (r"Copyright \d{4}", {"Copyright "}, set(), ["# Copyright 2025 G"]),
    (
        r"\bat::toAccumulateType\b",
        {"at::toAccumulateType"},
        set(),
        ["at::toAccumulateType<float>"],
    ),
    ("a.bc", {"bc"}, set(), ["axbc"]),
    ("[ab]cd", {"cd"}, set(), ["acd", "bcd"]),
    (r"\d{4}", set(), set(), ["2025"]),
    ("a|b", set(), set(), ["a", "b"]),
    ("(?i)todo", set(), set(), ["TODO", "todo"]),
    ("(?i:todo)bar", {"bar"}, set(), ["TODObar", "todobar"]),
    ("(?:abc|(?i:def))", set(), set(), ["abc", "DEF"]),
    ("(foo)?bar", {"bar"}, set(), ["bar", "foobar"]),
    ("(?:foo|)bar", {"bar"}, set(), ["bar", "foobar"]),
    ("(?:foo|)", set(), set(), ["", "foo"]),
    ("foo(?:bar)*baz", {"foo", "baz"}, set(), ["foobaz", "foobarbarbaz"]),
    (
        r"(?:foo|bar)(?:\s+baz)?",
        set(),
        {("bar", "foo")},
        ["foo", "bar baz"],
    ),
    (r"(tpu|gpu)\s*=\s*\(", set(), {("gpu", "tpu")}, ["tpu = (", "gpu=("]),
    ("(?:a|bc)de", set(), {("ade", "bcde")}, ["ade", "bcde"]),
    (
        "(?:ab|cd)(?:ef|gh)",
        set(),
        {("abef", "abgh", "cdef", "cdgh")},
        ["abef", "cdgh"],
    ),
    (r"(?:abc|de\w+)", set(), {("abc", "de")}, ["abc", "def"]),
    (r"(?:ab\w|cd)x", set(), {("ab", "cd")}, ["abzx", "cdx"]),
    (
        "x(?:ab|c(?:de|fg))y",
        set(),
        {("xaby", "xcdey", "xcfgy")},
        ["xaby", "xcdey", "xcfgy"],
    ),
    (
        "(?:(?:ab|cd)|ef)gh",
        set(),
        {("abgh", "cdgh", "efgh")},
        ["abgh", "efgh"],
    ),
    (
        "(?:ab(?:cd|ef)|gh)",
        set(),
        {("abcd", "abef", "gh")},
        ["abcd", "abef", "gh"],
    ),
    ("a(?:b|c(?:d|e))f", set(), set(), ["abf", "acdf", "acef"]),
    ("(?!foo)bar", {"bar"}, set(), ["bar"]),
    (r"(?=foo)\w+", set(), set(), ["foo"]),
    ("(", set(), set(), []),
)


@pytest.mark.parametrize(
    ("pattern", "singletons", "disjunctions", "texts"), _LITERAL_CASES
)
def test_required_literals_table(
    pattern: str,
    singletons: set[str],
    disjunctions: set[tuple[str, ...]],
    texts: list[str],
) -> None:
  got_s, got_d = custom_lint_metadata._extract_required_literals(pattern)  # pylint: disable=protected-access
  assert set(got_s) == singletons
  assert {tuple(sorted(c)) for c in got_d} == disjunctions
  for text in texts:
    assert re.search(pattern, text) is not None
    for s in got_s:
      assert s in text
    for clause in got_d:
      assert any(s in text for s in clause), (clause, text)


def test_rule_prefilter_skips_only_impossible_files() -> None:
  rule = custom_lint.Rule(
      id="R", path_regex=".*", prohibited_regex=r"(tpu|gpu)\s*=\s*\(x"
  )
  assert not custom_lint._literals_present(rule, "cpu = (x")  # pylint: disable=protected-access
  assert not custom_lint._literals_present(rule, "tpu = (y")  # pylint: disable=protected-access
  assert custom_lint._literals_present(rule, "a\ngpu=(x")  # pylint: disable=protected-access


# ---------------------------------------------------------------------------
# Differential & execution-path tests
# ---------------------------------------------------------------------------

_DIFF_RULES = r"""
presubmit {
  path_regexp: ".*\\.txt"
  check_contents { base { id: "Plain" } prohibited_regexp: "\\bfoo\\(" suppression_regexp: "FOO_OK" }
  check_contents { base { id: "Anchored" } prohibited_regexp: "^\\s*bar$" }
  check_contents { base { id: "IgnoreCase" } prohibited_regexp: "(?i)todo\\b" }
  check_contents { base { id: "Alternation" } prohibited_regexp: "(tpu|gpu)\\s*=\\s*\\(" }
  check_contents { base { id: "Optional" } prohibited_regexp: "(?:no)?baz\\d" }
  check_contents { base { id: "Lookahead" } prohibited_regexp: "qux(?!_ok)" }
  check_contents { base { id: "Delta" } prohibited_regexp: "foo" check_delta_only: true }
  check_contents {
    base { id: "Chunk" }
    prohibited_regexp: "foo\\(\\s*bar"
    suppression_regexp: "CHUNK_OK"
    check_delta_only: true
    match_multi_line: true
  }
  check_contents { base { id: "WholeFile" } prohibited_regexp: "\\Abar" match_entire_file: true }
  check_contents {
    base { id: "WholeFileSupp" }
    prohibited_regexp: "baz\\d\\nqux"
    suppression_regexp: "FILE_OK"
    match_entire_file: true
  }
  check_contents { base { id: "RequiredLine" } required_regexp: "^header$" }
  check_contents { base { id: "RequiredFile" } required_regexp: "header\\nfoo" match_entire_file: true }
  check_contents { base { id: "EditOnly" } prohibited_regexp: "TODO" operation: EDIT check_delta_only: true }
  check_file_operation { base { id: "Added" } operation: ADD }
}
"""

_DIFF_LINES = (
    "",
    "header",
    "foo(",
    "foo(x)",
    "  foo(1)  # FOO_OK",
    "bar",
    "  bar",
    "bar ",
    "TODO: x",
    "todo",
    "Todo later",
    "tpu = (",
    'gpu="x"',
    "baz1",
    "nobaz2",
    "qux",
    "qux_ok",
    "quxx",
    "\fform feed",
    "bar(",
    "CHUNK_OK",
    "FILE_OK",
    "  foo",
    "x = foo(",
    "   bar",
)

_PROJECT_LINES = (
    "",
    "x = 1",
    '"""',
    "cc_library(",
    'cc_test(name = "x")',
    "py_library(",
    "py_test(",
    '  LOG(ERROR) << "x";',
    "LOG(FATAL) << x;",
    "CHECK(x);",
    "TORCH_CHECK(x);",
    'throw std::runtime_error("x");',
    "from . import foo",
    "from .x import y",
    "import logging",
    "from logging import getLogger",
    "# import logging",
    'flags.DEFINE_string("a", "", "")',
    "class T(absltest.TestCase):",
    "class T(parameterized.TestCase):  # ABSLTEST_OK=x",
    "self.assertRaises(ValueError)",
    'with self.assertRaisesRegex(E, "m"):  # ASSERT_RAISES_OK=x',
    'tpu="msg"',
    "gpu=(",
    'tpu="""a\\b"""',
    'tpu=r"size \\[3\\]"',
    "_init_device()",  # INIT_DEVICE_OK=linter_test
    "def _init_device():",
    "at::toAccumulateType<float>(x)",
    "# Copyright 2025 Google LLC",
    '#include "x.h"',
    "torch_tpu_package_end()",
    'os.environ["X"]',
    'std::getenv("TORCH_TPU_X")',
    "torch.testing.assert_close(a, b)",  # TORCH_ASSERT_CLOSE_OK=linter_test
    "assert torch.allclose(a, b)",  # TORCH_ASSERT_CLOSE_OK=linter_test
    'ABSL_FLAG(bool, torch_tpu_internal_x, false, "");',
    'ABSL_FLAG(bool, x, false, "");',
    'tags = ["notap"]',
)


def _fixture_texts(lines: Sequence[str], count: int, seed: int) -> list[str]:
  rng = random.Random(seed)
  texts = []
  for _ in range(count):
    body = "\n".join(rng.choices(lines, k=rng.randint(0, 10)))
    texts.append(body + ("\n" if rng.random() < 0.7 else ""))
  return texts


def _fixture_changes(
    texts: Sequence[str], seed: int
) -> custom_lint.ModifiedLines:
  rng = random.Random(seed)
  modified = custom_lint.ModifiedLines()
  for i, text in enumerate(texts):
    if rng.random() < 0.2:
      continue
    n = text.count("\n") + 1
    modified[f"f{i}.txt"] = {j for j in range(1, n + 1) if rng.random() < 0.5}
    modified.operations[f"f{i}.txt"] = rng.choice(("ADD", "EDIT", "BRANCH"))
  return modified


def _naive_findings(
    rule: custom_lint.Rule,
    rel_path: str,
    text: str,
    modified: custom_lint.ModifiedLines | None,
) -> list[int | None]:
  op = modified.operations.get(rel_path) if modified is not None else None
  if rule.operations and op not in rule.operations:
    return []
  if rule.kind == custom_lint_metadata.CHECK_FILE_OPERATION:
    return [None]
  if rule.check_delta_only and modified is not None:
    if rel_path not in modified:
      return []
    changed = modified[rel_path]
  else:
    changed = None
  lines = text.split("\n")
  if lines and not lines[-1]:
    lines.pop()
  prohibited = rule.prohibited_regex and re.compile(rule.prohibited_regex)
  required = rule.required_regex and re.compile(rule.required_regex)
  supp = rule.suppression_regex and re.compile(rule.suppression_regex)
  out: list[int | None] = []
  if rule.mode == custom_lint_metadata.MODE_ENTIRE_FILE:
    if prohibited and not (supp and supp.search(text)):
      for m in prohibited.finditer(text):
        out.append(text.count("\n", 0, m.start()) + 1)
    if required and not required.search(text):
      out.append(None)
    return out
  if changed is None:
    numbers = list(range(1, len(lines) + 1))
  else:
    numbers = sorted(i for i in changed if 1 <= i <= len(lines))
  if prohibited and rule.mode == custom_lint_metadata.MODE_MODIFIED_CHUNKS:
    runs: list[list[int]] = []
    for i in numbers:
      if runs and runs[-1][1] + 1 == i:
        runs[-1][1] = i
      else:
        runs.append([i, i])
    for first, last in runs:
      chunk = "\n".join(lines[first - 1 : last])
      if supp and supp.search(chunk):
        continue
      for m in prohibited.finditer(chunk):
        out.append(first + chunk.count("\n", 0, m.start()))
  elif prohibited:
    for i in numbers:
      line = lines[i - 1]
      if prohibited.search(line) and not (supp and supp.search(line)):
        out.append(i)
  if required and not any(required.search(line) for line in lines):
    out.append(None)
  return out


def _sorted_findings(findings):
  return sorted(findings, key=lambda x: (x is not None, x or 0))


def _assert_agree(rules, texts, modified, prechecked: bool = False) -> int:
  checked = 0
  for i, text in enumerate(texts):
    rel = f"f{i}.txt"
    for rule in rules:
      engine_res = custom_lint.check_file_content(
          content=text,
          rel_path=rel,
          rule=rule,
          modified_lines=modified,
          canonical_path=f"{_BASE}/{rel}",
          _paths_prechecked=prechecked,
      )
      naive = _naive_findings(rule, rel, text, modified)
      if engine_res or naive:
        checked += 1
      assert _sorted_findings(
          v.line_number for v in engine_res
      ) == _sorted_findings(naive), (
          f"{rule.id} on {rel} (modified={modified and modified.get(rel)}):"
          f"\n{text!r}"
      )
  return checked


def test_chunk_ending_in_an_empty_line() -> None:
  modified = custom_lint.ModifiedLines({"BUILD": {1, 2}})
  violations = custom_lint.check_file_content(
      'py_test(\n\n    name = "x",\n',
      "BUILD",
      _rule("CheckPyTestUsage"),
      modified,
  )
  assert _lines(violations) == [1]
  text = custom_lint.TextFile("x\n\ny\n")
  assert text.text[slice(*text.span(1, 2))] == "x\n"
  assert text.text[slice(*text.span(2, 3))] == "\ny"


def test_synthetic_rules_all_mode() -> None:
  rules = _load(_DIFF_RULES).rules
  texts = _fixture_texts(_DIFF_LINES, 300, seed=1)
  assert _assert_agree(rules, texts, None) > 300


def test_synthetic_rules_diff_mode() -> None:
  rules = _load(_DIFF_RULES).rules
  texts = _fixture_texts(_DIFF_LINES, 300, seed=2)
  modified = _fixture_changes(texts, seed=3)
  assert _assert_agree(rules, texts, modified) > 300


def test_project_rules_differential() -> None:
  rules = list(_project_rules().values())
  lines = list(_PROJECT_LINES)
  for rule in rules:
    for literal in rule.required_singletons + tuple(
        s for clause in rule.required_disjunctions for s in clause  # pylint: disable=g-complex-comprehension
    ):
      lines += [literal, f"  x {literal}(y) # z"]
  texts = _fixture_texts(lines, 400, seed=4)
  assert _assert_agree(rules, texts, None, prechecked=True) > 400
  modified = _fixture_changes(texts, seed=5)
  _assert_agree(rules, texts, modified, prechecked=True)


@pytest.mark.parametrize("check_all", [True, False])
def test_run_checks_matches_naive(
    tmp_path: pathlib.Path, check_all: bool
) -> None:
  texts = _fixture_texts(_DIFF_LINES, 60, seed=6)
  modified = _fixture_changes(texts, seed=7)
  root = tmp_path.resolve()
  (root / "pyproject.toml").write_text("", encoding="utf-8")
  rules_file = root / "rules.METADATA"
  rules_file.write_text(_DIFF_RULES, encoding="utf-8")
  for i, text in enumerate(texts):
    (root / f"f{i}.txt").write_text(text, encoding="utf-8")
  rules = custom_lint.load_rules(rules_file, root_dir=root).rules

  expected = sorted(
      (rule.id, f"f{i}.txt", line or 0)  # pylint: disable=g-complex-comprehension
      for i, text in enumerate(texts)
      if check_all or f"f{i}.txt" in modified
      for rule in rules
      for line in _naive_findings(
          rule, f"f{i}.txt", text, None if check_all else modified
      )
  )
  with (
      mock.patch.object(
          custom_lint, "get_modified_lines_from_vcs", return_value=modified
      ),
      mock.patch.object(custom_lint, "get_commit_message", return_value=""),
  ):
    result = custom_lint.collect_violations(
        root, check_all=check_all, rules_file=str(rules_file)
    )
  actual = sorted(
      (v.rule_id, v.file_path, v.line_number or 0) for v in result.violations
  )
  assert actual == expected


@dataclasses.dataclass
class _ExecFixture:
  root: pathlib.Path
  rules_file: pathlib.Path
  expected: list[tuple[str, str, int]]
  requested_contexts: list[str | None]


@pytest.fixture(name="exec_fixture")
def _exec_fixture(tmp_path: pathlib.Path) -> Iterator[_ExecFixture]:
  root = tmp_path.resolve()
  (root / "pyproject.toml").write_text("", encoding="utf-8")
  rules_file = root / "rules.METADATA"
  rules_file.write_text(_DIFF_RULES, encoding="utf-8")
  texts = _fixture_texts(_DIFF_LINES, 50, seed=8)
  for i, text in enumerate(texts):
    (root / f"f{i}.txt").write_text(text, encoding="utf-8")
  rules = custom_lint.load_rules(rules_file, root_dir=root).rules
  expected = sorted(
      (rule.id, f"f{i}.txt", line or 0)  # pylint: disable=g-complex-comprehension
      for i, text in enumerate(texts)
      for rule in rules
      for line in _naive_findings(rule, f"f{i}.txt", text, None)
  )
  assert expected
  requested_contexts: list[str | None] = []
  real_get_context = multiprocessing.get_context
  can_spawn = bool(sys.executable) and os.path.isfile(sys.executable)

  def get_context(method=None):
    requested_contexts.append(method)
    return real_get_context(method if can_spawn else "fork")

  with mock.patch.object(
      custom_lint.multiprocessing, "get_context", side_effect=get_context
  ):
    yield _ExecFixture(
        root=root,
        rules_file=rules_file,
        expected=expected,
        requested_contexts=requested_contexts,
    )


def _run_exec(
    fx: _ExecFixture, *, jobs: int, gil: bool = True, active_threads: int = 1
) -> list[tuple[str, str, int]]:
  with (
      mock.patch.object(sys, "_is_gil_enabled", return_value=gil, create=True),
      mock.patch.object(
          custom_lint.threading, "active_count", return_value=active_threads
      ),
  ):
    result = custom_lint.collect_violations(
        fx.root, check_all=True, jobs=jobs, rules_file=str(fx.rules_file)
    )
  return sorted(
      (v.rule_id, v.file_path, v.line_number or 0) for v in result.violations
  )


def test_execution_serial(exec_fixture: _ExecFixture) -> None:
  with mock.patch.object(
      concurrent.futures, "ProcessPoolExecutor", side_effect=AssertionError
  ):
    assert _run_exec(exec_fixture, jobs=1) == exec_fixture.expected
  assert not exec_fixture.requested_contexts


def test_execution_thread_pool_on_free_threaded_python(
    exec_fixture: _ExecFixture,
) -> None:
  with mock.patch.object(
      concurrent.futures, "ProcessPoolExecutor", side_effect=AssertionError
  ):
    assert _run_exec(exec_fixture, jobs=2, gil=False) == exec_fixture.expected
  assert not exec_fixture.requested_contexts


def test_execution_fork_with_slices(exec_fixture: _ExecFixture) -> None:
  assert _run_exec(exec_fixture, jobs=2) == exec_fixture.expected
  assert exec_fixture.requested_contexts == ["fork"]


def test_execution_pickled_tasks_when_other_threads_run(
    exec_fixture: _ExecFixture,
) -> None:
  assert (
      _run_exec(exec_fixture, jobs=2, active_threads=2) == exec_fixture.expected
  )
  assert len(exec_fixture.requested_contexts) == 1
  assert exec_fixture.requested_contexts[0] in ("forkserver", "spawn")


# ---------------------------------------------------------------------------
# Lazy commit-message tests
# ---------------------------------------------------------------------------

_LAZY_RULES = r"""
  presubmit {
    path_regexp: ".*\\.txt"
    check_contents { base { id: "Tagged" disable_tags: "ALLOW_FOO" } prohibited_regexp: "foo" }
    check_contents { base { id: "Untagged" } prohibited_regexp: "bar" }
  }
"""


def _run_lazy(
    tmp_path: pathlib.Path,
    content: str,
    message: str = "",
    check_all: bool = False,
) -> tuple[list[str], int]:
  root = tmp_path.resolve()
  (root / "pyproject.toml").write_text("", encoding="utf-8")
  rules_file = root / "rules.METADATA"
  rules_file.write_text(_LAZY_RULES, encoding="utf-8")
  (root / "a.txt").write_text(content, encoding="utf-8")
  modified = custom_lint.ModifiedLines(
      {"a.txt": set(range(1, content.count("\n") + 2))}
  )
  modified.operations["a.txt"] = "EDIT"
  with (
      mock.patch.object(
          custom_lint, "get_modified_lines_from_vcs", return_value=modified
      ),
      mock.patch.object(
          custom_lint, "get_commit_message", return_value=message
      ) as get_commit_message,
  ):
    result = custom_lint.collect_violations(
        root, check_all=check_all, rules_file=str(rules_file)
    )
  ids = sorted(v.rule_id for v in result.violations)
  return ids, get_commit_message.call_count


@pytest.mark.parametrize(
    ("content", "message", "check_all", "expected_ids", "expected_calls"),
    [
        ("x\n", "", False, [], 0),
        ("bar\n", "", False, ["Untagged"], 0),
        ("foo\nbar\n", "", False, ["Tagged", "Untagged"], 1),
        ("foo\nbar\n", "Fix\n\nALLOW_FOO=reason\n", False, ["Untagged"], 1),
        ("foo\n", "ALLOW_FOO=reason\n", True, ["Tagged"], 0),
    ],
)
def test_lazy_commit_message(
    tmp_path: pathlib.Path,
    content: str,
    message: str,
    check_all: bool,
    expected_ids: list[str],
    expected_calls: int,
) -> None:
  assert _run_lazy(tmp_path, content, message, check_all) == (
      expected_ids,
      expected_calls,
  )


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
