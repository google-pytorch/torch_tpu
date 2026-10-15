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

"""Tests for custom_linter.metadata."""

import re
import sys

import pytest
from ci.tools.custom_linter import metadata as custom_lint_metadata

_BASE = "//depot/repo/proj"


def _load(text: str) -> custom_lint_metadata.RuleSet:
  return custom_lint_metadata.parse_metadata(text, "METADATA", _BASE)


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
# Metadata loader & path expression tests
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
  assert r.disable_tags == frozenset({"ROLLBACK_OF", "OK"})
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


def test_compiled_rule_regexes_use_ascii_classes() -> None:
  rule = _load(r"""
      presubmit {
        path_regexp: ".*\\.py"
        check_contents {
          base { id: "CheckAsciiClasses" }
          prohibited_regexp: "^\\w+\\s+\\d+$"
          suppression_regexp: "\\bOK_\\w+\\b"
        }
      }""").rules[0]
  assert rule.prohibited_re is not None
  assert rule.prohibited_re.flags & re.ASCII
  assert rule.prohibited_re.search("foo_bar 123")
  assert not rule.prohibited_re.search("αβγ 123")
  assert not rule.prohibited_re.search("foo ١٢٣")
  assert rule.suppression_re is not None
  assert rule.suppression_re.flags & re.ASCII
  assert rule.suppression_re.search("OK_ascii")
  assert not rule.suppression_re.search("OK_αβγ")


# ---------------------------------------------------------------------------
# PathMapper & tag parsing tests
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


def test_parse_tags() -> None:
  tags = custom_lint_metadata.parse_tags(
      "Summary line\n\nBody mentions ALLOW_CC_TARGETS in prose.\n"
      "BUG=123\n  ROLLBACK_OF=456\nMARKDOWN = true\nlower=x\nX1_Y=\n"
      "Do not set ALLOW_FOO=1 here\n"
  )
  assert tags == {"BUG", "ROLLBACK_OF", "MARKDOWN", "X1_Y"}


# ---------------------------------------------------------------------------
# Literal pre-filter extraction tests
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


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
