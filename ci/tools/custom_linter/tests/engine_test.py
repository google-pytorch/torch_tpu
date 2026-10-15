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

"""Tests for custom_linter.engine."""

from collections.abc import Iterable, Iterator, Mapping, Sequence, Set
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
from ci.tools.custom_linter import vcs as custom_lint_vcs

_BASE = "//depot/repo/proj"
_PROJECT_ROOT = custom_lint.find_project_root(pathlib.Path(__file__))
_PROJECT_CONFIG = custom_lint.load_config(_PROJECT_ROOT)
_PROJECT_MAPPER = custom_lint.default_path_mapper(
    _PROJECT_ROOT, _PROJECT_CONFIG
)


@functools.lru_cache(maxsize=1)
def _load_default_rules() -> custom_lint_metadata.RuleSet:
  return custom_lint.load_rules(
      root_dir=_PROJECT_ROOT,
      mapper=_PROJECT_MAPPER,
      config=_PROJECT_CONFIG,
  )


@functools.lru_cache(maxsize=1)
def _project_rules() -> dict[str, custom_lint_metadata.Rule]:
  return {r.id: r for r in _load_default_rules().rules}


def _rule(rule_id: str) -> custom_lint_metadata.Rule:
  return _project_rules()[rule_id]


def _load(text: str) -> custom_lint_metadata.RuleSet:
  return custom_lint_metadata.parse_metadata(text, "METADATA", _BASE)


def _lines(violations: Iterable[custom_lint.Violation]) -> list[int | None]:
  return [v.line_number for v in violations]


def _check(
    rule_id: str,
    content: str,
    rel_path: str,
    modified: (
        custom_lint_vcs.ModifiedLines | Mapping[str, Set[int]] | None
    ) = None,
    op: custom_lint_metadata.FileOperation | str | None = None,
) -> Sequence[custom_lint.Violation]:
  return custom_lint.check_file_content(
      content=content,
      rel_path=rel_path,
      rule=_rule(rule_id),
      modified_lines=modified,
      canonical_path=_PROJECT_MAPPER.to_canonical(rel_path),
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
  modified = custom_lint_vcs.parse_unified_diff(diff_text)
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
  modified = custom_lint_vcs.parse_unified_diff(diff_text)
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
  modified = custom_lint_vcs.parse_unified_diff(diff_text)
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
  modified = custom_lint_vcs.parse_unified_diff(diff_text, prefix=prefix)
  assert set(modified) == {"foo/BUILD.bazel", "bar/BUILD"}
  assert modified["foo/BUILD.bazel"] == {11, 12, 13}
  assert modified["bar/BUILD"] == {5}
  assert "other_pkg/outside.py" not in modified
  assert "outside.py" not in modified


def test_normalize_diff_path(tmp_path: pathlib.Path) -> None:
  assert (
      custom_lint_vcs.normalize_diff_path("b/pkg/BUILD.bazel")
      == "pkg/BUILD.bazel"
  )
  assert (
      custom_lint_vcs.normalize_diff_path("a/pkg/BUILD.bazel")
      == "pkg/BUILD.bazel"
  )
  assert (
      custom_lint_vcs.normalize_diff_path(
          "b/sub/proj/pkg/BUILD.bazel", prefix="sub/proj/"
      )
      == "pkg/BUILD.bazel"
  )
  assert (
      custom_lint_vcs.normalize_diff_path(
          "a/sub/proj/pkg/BUILD.bazel", prefix="sub/proj"
      )
      == "pkg/BUILD.bazel"
  )
  assert not custom_lint_vcs.normalize_diff_path(
      "b/other/pkg/BUILD.bazel", prefix="sub/proj/"
  )
  root = tmp_path.resolve()
  assert (
      custom_lint_vcs.normalize_diff_path(
          str(root / "pkg/BUILD.bazel"), root_dir=root
      )
      == "pkg/BUILD.bazel"
  )
  assert not custom_lint_vcs.normalize_diff_path(
      "/outside/root/BUILD.bazel", root_dir=root
  )


# ---------------------------------------------------------------------------
# Project rules & path mapping tests
# ---------------------------------------------------------------------------


def test_project_rules_load() -> None:
  rule_set = _load_default_rules()
  assert len(rule_set.rules) > 50
  assert not rule_set.skipped
  for rule in rule_set.rules:
    rule.prewarm()


@pytest.mark.skipif(
    not (_PROJECT_ROOT / custom_lint.MONOREPO_RULES_FILE).is_file(),
    reason="Needs the monorepo rules file",
)
def test_oss_export_matches_the_same_files() -> None:
  g3_rules = _load_default_rules().rules
  rules_path = custom_lint.find_rules_file(_PROJECT_ROOT, _PROJECT_CONFIG)
  g3_mapper = _PROJECT_MAPPER
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
      _PROJECT_CONFIG.oss_canonical_root, _PROJECT_CONFIG.oss_path_maps
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
  for oss_rule in oss_rules:
    example = _RULE_EXAMPLES[oss_rule.id]
    assert oss_rule.matches_path(
        oss_mapper.to_canonical(example.path)
    ), f"{oss_rule.id} example path {example.path!r} does not match in OSS"


def test_default_path_mapper_and_find_rules_file_by_layout(
    tmp_path: pathlib.Path,
) -> None:
  g3_root = (tmp_path / "pkg" / "my_proj").resolve()
  g3_root.mkdir(parents=True)
  (g3_root / "pyproject.toml").write_text(
      "[tool.custom_linter]\nmonorepo_canonical_root ="
      ' "//depot/repo/pkg/my_proj"\n',
      encoding="utf-8",
  )
  g3_rules = g3_root / custom_lint.MONOREPO_RULES_FILE
  g3_rules.write_text("", encoding="utf-8")
  assert custom_lint.find_rules_file(g3_root) == g3_rules
  g3_mapper = custom_lint.default_path_mapper(g3_root)
  assert g3_mapper.canonical_root == "//depot/repo/pkg/my_proj"

  oss_root = (tmp_path / "oss_repo").resolve()
  oss_rules = oss_root / custom_lint.OSS_RULES_FILE
  oss_rules.parent.mkdir(parents=True)
  oss_rules.write_text("", encoding="utf-8")
  (oss_root / "pyproject.toml").write_text(
      '[tool.custom_linter]\noss_canonical_root = "torch_tpu/oss"\n',
      encoding="utf-8",
  )
  assert custom_lint.find_rules_file(oss_root) == oss_rules
  oss_mapper = custom_lint.default_path_mapper(oss_root)
  assert oss_mapper.canonical_root == "torch_tpu/oss"


def test_is_rule_disabled() -> None:
  rule = custom_lint_metadata.Rule(
      id="TestRule",
      path_regex=r".*",
      prohibited_regex=r"foo",
      failure_message="msg",
      disable_tags=frozenset({"ALLOW_FOO", "ROLLBACK_OF"}),
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


def test_line_mode_suppression_on_next_line() -> None:
  content = 'cc_library(\n    # ALLOW_CC_TARGETS=shim\n    name = "test",\n)\n'
  assert not _check("CheckCcTargetUsage", content, "BUILD", {"BUILD": {1, 2}})
  too_far = 'cc_library(\n    name = "test",\n    # ALLOW_CC_TARGETS=shim\n)\n'
  violations = _check(
      "CheckCcTargetUsage", too_far, "BUILD", {"BUILD": {1, 2, 3}}
  )
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
  modified_lines = custom_lint_vcs.parse_unified_diff(
      f"+++ b/{prefix}BUILD\n@@ -2 +2 @@\n-# old comment\n+# Untouched line"
      " 2\n",
      prefix=prefix,
  )
  assert not _check("CheckCcTargetUsage", content, "BUILD", modified_lines)
  modified_lines = custom_lint_vcs.parse_unified_diff(
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
  modified = custom_lint_vcs.parse_unified_diff(
      f"--- /dev/null\n+++ b/{rel}\n@@ -0,0 +1 @@\n+x\n"
  )
  assert _lines(
      custom_lint.check_file_content(
          "x\n",
          rel,
          _rule("CheckOpFileName"),
          modified,
          canonical_path=_PROJECT_MAPPER.to_canonical(rel),
      )
  ) == [None]


@dataclasses.dataclass(frozen=True)
class _RuleExample:
  path: str
  bad: str
  good: str | None = None
  op: custom_lint_metadata.FileOperation | None = None


# Exhaustive golden examples for every rule in custom_lint_rules.METADATA.
# Some string literals are concatenated so that the monorepo's own presubmit
# regexes and Copybara do not match this test file itself.
_RULE_EXAMPLES: Mapping[str, _RuleExample] = {
    "CheckCopybaraTags": _RuleExample(
        path="a.py",
        bad="# copybara:" + "strip\n",
    ),
    "CheckPythonEnvAccess": _RuleExample(
        path="ops/a.py",
        bad='os.environ["X"]\n',
        good='os.environ["X"]  # PY_ENVIRON_OK=r\n',
    ),
    "CheckPythonWarningsWarn": _RuleExample(
        path="ops/a.py",
        bad='warnings.warn("x")\n',
        good='warnings.warn("x")  # RAW_WARNINGS_OK=r\n',
    ),
    "CheckOpFileName": _RuleExample(
        path="csrc/ops/addkernel.cc",
        bad="x\n",
        op=custom_lint_metadata.FileOperation.ADD,
    ),
    "CheckOpImplFilename": _RuleExample(
        path="csrc/ops/add/helper.cc",
        bad="x\n",
        op=custom_lint_metadata.FileOperation.ADD,
    ),
    "CheckKernelMacro": _RuleExample(
        path="csrc/ops/add/add_kernels.cc",
        bad="int x;\n",
        good="TT_KERNEL(Add)\n",
    ),
    "CheckAssertOkAndAssign": _RuleExample(
        path="csrc/a.cc",
        bad="TF_ASSERT_OK_AND_ASSIGN(auto x, y);\n",
        good="// TF_ASSERT_OK_AND_ASSIGN(auto x, y);\n",
    ),
    "CheckC10ToStringUsage": _RuleExample(
        path="csrc/a.cc",
        bad="c10::toString(x);\n",
        good="c10::toString(x);  // C10_TOSTRING_OK\n",
    ),
    "CheckDeprecatedMutexLock": _RuleExample(
        path="csrc/a.cc",
        bad="MutexLock lock(&mu);\n",
    ),
    "CheckIgnoreInCacheKeyLegacy": _RuleExample(
        path="csrc/a.cc",
        bad='IgnoreInCacheKey(x, "Legacy usage");\n',
    ),
    "CheckIncludeBeforeCopyright": _RuleExample(
        path="csrc/a.cc",
        bad="\n#include <x>\n",
    ),
    "CheckThreadLocalNoDestructor": _RuleExample(
        path="csrc/a.cc",
        bad="thread_local absl::NoDestructor<Foo> x;\n",
        good=(
            "thread_local absl::NoDestructor<Foo> x;"
            "  // THREAD_LOCAL_NO_DESTRUCTOR_OK=r\n"
        ),
    ),
    "CheckThreadLocalNoTag": _RuleExample(
        path="csrc/a.cc",
        bad="thread_local int x = 0;\n",
        good="thread_local int x = 0;  // CPP_THREAD_LOCAL_OK=r\n",
    ),
    "CheckMaybeWrapDim": _RuleExample(
        path="csrc/a.cc",
        bad="at::maybe_wrap_dim(d, 0);\n",
        good="at::maybe_wrap_dim(d, 0);  // MAYBE_WRAP_DIM_OK=r\n",
    ),
    "CheckDisallowedLogMacros": _RuleExample(
        path="csrc/a.cc",
        bad='LOG(INFO) << "x";\n',
        good='LOG(INFO) << "x";  // ALLOW_LOG_MACROS=r\n',
    ),
    "CheckStdEndl": _RuleExample(
        path="csrc/a.cc",
        bad="os << std::endl;\n",
        good="os << std::endl;  // ENDL_OK\n",
    ),
    "CheckUninitializedTensor": _RuleExample(
        path="csrc/a.cc",
        bad="at::Tensor foo;\n",
        good="at::Tensor foo;  // UNINITIALIZED_TENSOR_OK\n",
    ),
    "CheckErrorApi": _RuleExample(
        path="csrc/a.cc",
        bad="TORCH_CHECK(x);\n",
        good="TORCH_CHECK(x);  // ok\n",
    ),
    "CheckMacroNamePrefix": _RuleExample(
        path="csrc/a.cc",
        bad="#define FOO 1\n",
        good="#define FOO 1  // NON_TT_MACRO_OK=r\n",
    ),
    "CheckIntVec": _RuleExample(
        path="csrc/a.cc",
        bad="std::vector<int64_t> v;\n",
        good="std::vector<int64_t> v;  // INT_VEC_OK\n",
    ),
    "CheckVecMethod": _RuleExample(
        path="csrc/a.cc",
        bad="x.vec();\n",
        good="x.vec();  // VEC_OK\n",
    ),
    "CheckErrorSpace": _RuleExample(
        path="csrc/a.cc",
        bad="absl::StatusCode::kInvalidArgument\n",
        good="absl::StatusCode::kInvalidArgument  // STATUS_CODE_OK\n",
    ),
    "CheckLeadingColons": _RuleExample(
        path="csrc/a.cc",
        bad="::std::string s;\n",
        good="::std::string s;  // LEADING_COLONS_OK\n",
    ),
    "CheckUsingDecls": _RuleExample(
        path="csrc/a.cc",
        bad="using std::vector;\n",
        good="using std::vector;  // USING_DECL_OK\n",
    ),
    "CheckMlirSignedInt": _RuleExample(
        path="csrc/a.cc",
        bad="type.isSigned()\n",
        good="type.isSigned()  // MLIR_SIGNED_INT_OK=r\n",
    ),
    "CheckLambdaCapture": _RuleExample(
        path="csrc/a.cc",
        bad="auto fn = [=]() {};\n",
        good="auto fn = [=]() {};  // DEFAULT_CAPTURE_OK\n",
    ),
    "CheckAtInferSize": _RuleExample(
        path="csrc/a.cc",
        bad="at::infer_size(a, b);\n",
        good="at::infer_size(a, b);  // AT_INFER_SIZE_OK\n",
    ),
    "CheckStdFunction": _RuleExample(
        path="csrc/a.cc",
        bad="std::function<void()> f;\n",
        good="std::function<void()> f;  // STD_FUNCTION_OK\n",
    ),
    "CheckStringView": _RuleExample(
        path="csrc/a.cc",
        bad="absl::string_view s;\n",
        good="absl::string_view s;  // ABSL_STRING_VIEW_OK\n",
    ),
    "CheckTorchWarn": _RuleExample(
        path="csrc/a.cc",
        bad='TORCH_WARN("x");\n',
        good='TORCH_WARN("x");  // TORCH_WARN_OK=r\n',
    ),
    "CheckCheckFunctionName": _RuleExample(
        path="csrc/a.cc",
        bad="absl::Status CheckFoo(\n",
        good="absl::Status CheckFoo(  // CHECK_NAME_OK=r\n",
    ),
    "CheckHashVsFingerprint": _RuleExample(
        path="csrc/a.cc",
        bad="absl::HashOf(x);\n",
        good="absl::HashOf(x);  // HASH_OK=r\n",
    ),
    "CheckNoVoidCast": _RuleExample(
        path="csrc/a.cc",
        bad="(void)foo();\n",
        good="(void)foo();  // VOID_CAST_OK\n",
    ),
    "CheckIgnoreError": _RuleExample(
        path="csrc/a.cc",
        bad="s.IgnoreError();\n",
        good="s.IgnoreError();  // IGNORE_ERROR_OK\n",
    ),
    "CheckCppFlagNames": _RuleExample(
        path="csrc/a.cc",
        bad='ABSL_FLAG(int, bad_name, 0, "d");\n',
        good=(
            'ABSL_FLAG(int, bad_name, 0, "d");  // NONSTANDARD_FLAG_NAME_OK=r\n'
        ),
    ),
    "CheckEnvVarName": _RuleExample(
        path="csrc/a.cc",
        bad='setenv("FOO", "1", 1);\n',
        good='setenv("FOO", "1", 1);  // STRING_LITERAL_ENV_VAR_NAME_OK=r\n',
    ),
    "CheckGetenv": _RuleExample(
        path="csrc/a.cc",
        bad='getenv("FOO");\n',
        good='getenv("FOO");  // GETENV_OK=r\n',
    ),
    "CheckGetFlag": _RuleExample(
        path="csrc/a.cc",
        bad="absl::GetFlag(FLAGS_x);\n",
        good="absl::GetFlag(FLAGS_x);  // GET_FLAG_OK=r\n",
    ),
    "CheckCrashingMacros": _RuleExample(
        path="csrc/a.cc",
        bad="ABSL_CHECK(x);\n",
        good="ABSL_CHECK(x);  // CRASH_OK\n",
    ),
    "CheckRawOpName": _RuleExample(
        path="csrc/a.cc",
        bad='"add(): bad";\n',
        good='"add(): bad";  // RAW_OP_NAME_OK\n',
    ),
    "CheckMImpl": _RuleExample(
        path="csrc/a.cc",
        bad='m.impl("x", fn);\n',
        good='m.impl("x", fn);  // M_IMPL_OK=r\n',
    ),
    "CheckStdPair": _RuleExample(
        path="csrc/a.cc",
        bad="std::pair<int, int> p;\n",
        good="std::pair<int, int> p;  // STD_PAIR_OK=r\n",
    ),
    "CheckPyFunctionSuffix": _RuleExample(
        path="csrc/a.cc",
        bad="void FooPy();\n",
        good="void FooPy();  // PY_SUFFIX_OK=r\n",
    ),
    "CheckTorchThreadLocalStateGuards": _RuleExample(
        path="csrc/a.cc",
        bad="c10::DebugInfoGuard g;\n",
        good="c10::DebugInfoGuard g;  // TORCH_THREAD_LOCAL_STATE_GUARD_OK=r\n",
    ),
    "CheckRawPyBindUsage": _RuleExample(
        path="csrc/a.cc",
        bad="PYBIND11_MODULE(foo, m) {}\n",
        good="PYBIND11_MODULE(foo, m) {}  // RAW_PYBIND_OK=r\n",
    ),
    "CheckExplicitPrecisionUsage": _RuleExample(
        path="csrc/a.cc",
        bad="auto p = Precision::HIGH;\n",
        good="auto p = Precision::HIGH;  // EXPLICIT_PRECISION_OK\n",
    ),
    "CheckNotestOss": _RuleExample(
        path="BUILD",
        bad="notest_oss = True\n",
        good="notest_oss = True  # NOTEST_OSS_OK=r\n",
    ),
    "CheckNobuildOss": _RuleExample(
        path="BUILD",
        bad="nobuild_oss = True\n",
        good="nobuild_oss = True  # NOBUILD_OSS_OK=r\n",
    ),
    "CheckBannedApiDependency": _RuleExample(
        path="BUILD",
        bad='"//api:foo"\n',
    ),
    "CheckNoSan": _RuleExample(
        path="BUILD",
        bad='"noasan"\n',
        good='"noasan"  # reason\n',
    ),
    "CheckNoTapTag": _RuleExample(
        path="BUILD",
        bad='"notap"\n',
        good='"notap"  # NOTAP_OK=r\n',
    ),
    "CheckCcTargetUsage": _RuleExample(
        path="BUILD",
        bad='cc_library(name = "x")\n',
        good='cc_library(name = "x")  # ALLOW_CC_TARGETS=r\n',
    ),
    "CheckPyLibraryUsage": _RuleExample(
        path="BUILD",
        bad='py_library(\n    name = "x",\n)\n',
        good='py_library(\n    # PY_LIBRARY_OK=r\n    name = "x",\n)\n',
    ),
    "CheckPyTestUsage": _RuleExample(
        path="BUILD",
        bad='py_test(\n    name = "x",\n)\n',
        good='py_test(\n    # PY_TEST_OK=r\n    name = "x",\n)\n',
    ),
    "CheckTorchTpuPackageEnd": _RuleExample(
        path="csrc/BUILD",
        bad='torch_tpu_cc_library(name = "x")\n',
        good='torch_tpu_cc_library(name = "x")\ntorch_tpu_package_end()\n',
    ),
    "CheckPythonErrorTestArgs": _RuleExample(
        path="tests/errors_test.py",
        bad='self.assert_error(fn, tpu="msg")\n',
    ),
    "CheckPythonErrorTestArgsParentheses": _RuleExample(
        path="tests/errors_test.py",
        bad='self.assert_error(fn, tpu=("msg"))\n',
    ),
    "CheckPythonErrorTestBackslashes": _RuleExample(
        path="tests/errors_test.py",
        bad='self.assert_error(fn, tpu="a\\\\b")\n',
        good='self.assert_error(fn, tpu=r"a\\\\b")\n',
    ),
    "CheckAssertRaises": _RuleExample(
        path="tests/ops_test.py",
        bad="self.assertRaises(ValueError)\n",
        good="self.assertRaises(ValueError)  # ASSERT_RAISES_OK=r\n",
    ),
    "CheckRelativeImports": _RuleExample(
        path="ops/a.py",
        bad="from . import foo\n",
        good="from . import foo  # RELATIVE_IMPORT_OK=r\n",
    ),
    "CheckPythonFlags": _RuleExample(
        path="ops/a.py",
        bad='flags.DEFINE_bool("x", False, "d")\n',
        good='flags.DEFINE_bool("x", False, "d")  # PYTHON_FLAG_OK=r\n',
    ),
    "CheckAbslTestCase": _RuleExample(
        path="tests/ops_test.py",
        bad="class T(absltest.TestCase):\n",
        good="class T(absltest.TestCase):  # ABSLTEST_OK=r\n",
    ),
    "CheckAssertClose": _RuleExample(
        path="tests/ops_test.py",
        bad="testing." + "assert_close(a, b)\n",
        good="testing." + "assert_close(a, b)  # TORCH_ASSERT_CLOSE_OK\n",
    ),
    "CheckPythonLogging": _RuleExample(
        path="ops/a.py",
        bad="import " + "logging\n",
        good="import " + "logging  # PYTHON_LOGGING_OK\n",
    ),
    "CheckInitDeviceUsage": _RuleExample(
        path="ops/a.py",
        bad="_init" + "_device()\n",
        good="_init" + "_device()  # INIT_DEVICE_OK\n",
    ),
    "CheckToAccumulateTypeUsage": _RuleExample(
        path="csrc/a.cc",
        bad="at::toAccumulateType(t, true);\n",
        good="at::toAccumulateType(t, true);  // AT_TO_ACCUMULATE_TYPE_OK\n",
    ),
    "CheckCopyrightYearUpdate": _RuleExample(
        path="csrc/a.cc",
        bad="// Copyright " + "2026 Google LLC\n",
        op=custom_lint_metadata.FileOperation.EDIT,
    ),
}


def test_all_project_rules_have_examples() -> None:
  assert set(_RULE_EXAMPLES) == set(_project_rules())


@pytest.mark.parametrize("rule_id", list(_RULE_EXAMPLES))
def test_project_rule_examples(rule_id: str) -> None:
  rule = _rule(rule_id)
  example = _RULE_EXAMPLES[rule_id]
  assert rule.matches_path(_PROJECT_MAPPER.to_canonical(example.path))

  if rule.operations:
    assert example.op in rule.operations
  else:
    assert example.op is None

  bad_lines = {example.path: set(range(1, example.bad.count("\n") + 2))}
  violations = _check(rule_id, example.bad, example.path, bad_lines, example.op)
  assert len(violations) == 1
  assert violations[0].rule_id == rule_id

  if rule.operations:
    other_op = (
        custom_lint_metadata.FileOperation.EDIT
        if custom_lint_metadata.FileOperation.EDIT not in rule.operations
        else custom_lint_metadata.FileOperation.ADD
    )
    assert not _check(rule_id, example.bad, example.path, bad_lines, other_op)

  if rule.suppression_regex is not None or rule.required_regex is not None:
    assert example.good is not None
    good_lines = {example.path: set(range(1, example.good.count("\n") + 2))}
    assert not _check(
        rule_id, example.good, example.path, good_lines, example.op
    )
    if rule.suppression_regex is not None:
      unsuppressed_rule = dataclasses.replace(rule, suppression_regex=None)
      assert custom_lint.check_file_content(
          content=example.good,
          rel_path=example.path,
          rule=unsuppressed_rule,
          modified_lines=good_lines,
          canonical_path=_PROJECT_MAPPER.to_canonical(example.path),
          file_operation=example.op,
      )
  else:
    assert example.good is None

  for tag in rule.disable_tags:
    assert custom_lint.is_rule_disabled(rule, f"Summary\n\n{tag}=reason")


@pytest.mark.parametrize(
    "path",
    [
        "custom_lint.py",
        "custom_lint_metadata.py",
        "custom_lint_vcs.py",
        "custom_lint_rules.METADATA",
        "ci/custom_lint_rules.METADATA",
        "custom_linter/engine.py",
        "custom_linter/metadata.py",
        "custom_linter/vcs.py",
        "oss/ci/tools/custom_linter/__init__.py",
        "oss/ci/tools/custom_linter/__main__.py",
        "oss/ci/tools/custom_linter/engine.py",
        "oss/ci/tools/custom_linter/metadata.py",
        "oss/ci/tools/custom_linter/vcs.py",
        "oss/ci/tools/custom_linter/tests/engine_test.py",
        "oss/ci/tools/custom_linter/tests/metadata_test.py",
        "oss/ci/tools/custom_linter/tests/vcs_test.py",
        "ci/tools/custom_linter/engine.py",
        "ci/tools/custom_linter/tests/engine_test.py",
        "ci/tools/custom_linter/tests/metadata_test.py",
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
  rule = custom_lint_metadata.Rule(
      id="TestRule",
      path_regex=r".*\.(cc|cpp|h|hpp)$",
      prohibited_regex=r"\bfoo\b",
      failure_message="No foo",
      path_exclusion_regex=r".*_test\.(cc|cpp|h)",
  )
  assert rule.matches_path("src/core/model.cc")
  assert not rule.matches_path("src/core/model_test.cc")
  assert not rule.matches_path("src/core/model.py")

  mapper = _PROJECT_MAPPER
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

  rule_cc = custom_lint_metadata.Rule(
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

  rule_any_cc = custom_lint_metadata.Rule(
      id="RuleCC",
      path_regex=r".*\.cc$",
      prohibited_regex=r"foo",
      failure_message="msg",
  )
  monorepo_mapper = custom_lint_metadata.PathMapper("//depot/repo/proj")
  oss_mapper = custom_lint_metadata.PathMapper(
      _PROJECT_CONFIG.oss_canonical_root, _PROJECT_CONFIG.oss_path_maps
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
      root_path, [rule], mapper=_PROJECT_MAPPER
  )
  assert [c.rel_path for c in candidates] == ["binary_blob.cc", "valid.cc"]
  violations: list[custom_lint.Violation] = []
  for c in candidates:
    violations.extend(custom_lint._check_candidate((c, None)))  # pylint: disable=protected-access
  assert len(violations) == 1
  assert violations[0].file_path == "valid.cc"


def test_read_text_file_and_missing_paths_raise(tmp_path: pathlib.Path) -> None:
  root_path = tmp_path.resolve()
  with pytest.raises(FileNotFoundError):
    custom_lint._read_text_file(root_path / "missing.cc")  # pylint: disable=protected-access

  bad_utf8 = root_path / "bad_utf8.cc"
  bad_utf8.write_bytes(b"valid ascii\xff\xfe\n")
  with pytest.raises(UnicodeDecodeError):
    custom_lint._read_text_file(bad_utf8)  # pylint: disable=protected-access

  rules_file = root_path / "rules.METADATA"
  rules_file.write_text(
      'presubmit { path_regexp: ".*\\\\.cc"\n'
      '  check_contents { base { id: "NoFoo" } prohibited_regexp: "foo" }\n'
      "}\n",
      encoding="utf-8",
  )
  with pytest.raises(FileNotFoundError, match="nonexistent.cc"):
    custom_lint.collect_violations(
        root_path,
        files=["nonexistent.cc"],
        jobs=1,
        rules_file=str(rules_file),
    )


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


def test_rule_prefilter_skips_only_impossible_files() -> None:
  rule = custom_lint_metadata.Rule(
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
  texts: list[str] = []
  for _ in range(count):
    body = "\n".join(rng.choices(lines, k=rng.randint(0, 10)))
    texts.append(body + ("\n" if rng.random() < 0.7 else ""))
  return texts


def _fixture_changes(
    texts: Sequence[str], seed: int
) -> custom_lint_vcs.ModifiedLines:
  rng = random.Random(seed)
  lines: dict[str, set[int]] = {}
  operations: dict[str, str] = {}
  for i, text in enumerate(texts):
    if rng.random() < 0.2:
      continue
    n = text.count("\n") + 1
    lines[f"f{i}.txt"] = {j for j in range(1, n + 1) if rng.random() < 0.5}
    operations[f"f{i}.txt"] = rng.choice(("ADD", "EDIT", "BRANCH"))
  return custom_lint_vcs.ModifiedLines(lines=lines, operations=operations)


def _naive_findings(
    rule: custom_lint_metadata.Rule,
    rel_path: str,
    text: str,
    modified: custom_lint_vcs.ModifiedLines | None,
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
      if prohibited.search(line) and not (
          supp
          and (
              supp.search(line)
              or (i < len(lines) and supp.search(f"{line}\n{lines[i]}"))
          )
      ):
        out.append(i)
  if required and not any(required.search(line) for line in lines):
    out.append(None)
  return out


def _sorted_findings(findings: Iterable[int | None]) -> list[int | None]:
  return sorted(findings, key=lambda x: (x is not None, x or 0))


def _assert_agree(
    rules: Sequence[custom_lint_metadata.Rule],
    texts: Sequence[str],
    modified: custom_lint_vcs.ModifiedLines | None,
    prechecked: bool = False,
) -> int:
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
  modified = custom_lint_vcs.ModifiedLines({"BUILD": {1, 2}})
  violations = custom_lint.check_file_content(
      'py_test(\n\n    name = "x",\n',
      "BUILD",
      _rule("CheckPyTestUsage"),
      modified,
      canonical_path=_PROJECT_MAPPER.to_canonical("BUILD"),
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
    for literal in rule._prohibited_singletons + tuple(  # pylint: disable=protected-access
        s for clause in rule._prohibited_disjunctions for s in clause  # pylint: disable=g-complex-comprehension,protected-access
    ):
      lines += [literal, f"  x {literal}(y) # z"]
  texts = _fixture_texts(lines, 400, seed=4)
  assert _assert_agree(rules, texts, None, prechecked=True) > 400
  modified = _fixture_changes(texts, seed=5)
  _assert_agree(rules, texts, modified, prechecked=True)


def test_is_line_local_possessive_and_atomic_groups() -> None:
  assert not custom_lint_metadata.is_line_local(r"^foo\s++$")
  assert not custom_lint_metadata.is_line_local(r"^foo(?>\s+)$")
  for pat in (r"^foo\s++$", r"^foo(?>\s+)$"):
    rule = custom_lint_metadata.Rule(
        id="PossessiveOrAtomic",
        path_regex=r".*\.txt",
        prohibited_regex=pat,
    )
    violations = custom_lint.check_file_content("foo   \nbar\n", "a.txt", rule)
    assert _lines(violations) == [1]


def test_sre_parser_unavailable_fallback() -> None:
  custom_lint_metadata._extract_required_literals.cache_clear()  # pylint: disable=protected-access
  custom_lint_metadata.is_line_local.cache_clear()
  try:
    with (
        mock.patch.object(custom_lint_metadata, "_sre_parse", None),
        mock.patch.object(custom_lint_metadata, "_sre_constants", None),
    ):
      assert custom_lint_metadata._extract_required_literals("foo(bar|baz)") == (  # pylint: disable=protected-access
          (),
          (),
      )
      assert not custom_lint_metadata.is_line_local(r"^foo$")
      rules = _load(_DIFF_RULES).rules
      texts = _fixture_texts(_DIFF_LINES, 80, seed=9)
      assert _assert_agree(rules, texts, None) > 80
  finally:
    custom_lint_metadata._extract_required_literals.cache_clear()  # pylint: disable=protected-access
    custom_lint_metadata.is_line_local.cache_clear()


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
          custom_lint_vcs.Vcs,
          "get_modified_lines",
          return_value=modified,
          autospec=True,
      ),
      mock.patch.object(
          custom_lint_vcs.Vcs,
          "get_commit_message",
          return_value="",
          autospec=True,
      ),
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

  def get_context(
      method: str | None = None,
  ) -> multiprocessing.context.BaseContext:
    requested_contexts.append(method)
    return real_get_context(method if can_spawn else "fork")

  with mock.patch.object(
      custom_lint.multiprocessing,
      "get_context",
      side_effect=get_context,
      autospec=True,
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
          custom_lint.threading,
          "active_count",
          return_value=active_threads,
          autospec=True,
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
      concurrent.futures,
      "ProcessPoolExecutor",
      side_effect=AssertionError,
      autospec=True,
  ):
    assert _run_exec(exec_fixture, jobs=1) == exec_fixture.expected
  assert not exec_fixture.requested_contexts


def test_execution_thread_pool_on_free_threaded_python(
    exec_fixture: _ExecFixture,
) -> None:
  with mock.patch.object(
      concurrent.futures,
      "ProcessPoolExecutor",
      side_effect=AssertionError,
      autospec=True,
  ):
    assert _run_exec(exec_fixture, jobs=2, gil=False) == exec_fixture.expected
  assert not exec_fixture.requested_contexts
  assert custom_lint._WORKER_CANDIDATES == ()
  assert custom_lint._WORKER_MODIFIED_LINES is None


def test_execution_fork_with_slices(exec_fixture: _ExecFixture) -> None:
  assert _run_exec(exec_fixture, jobs=2) == exec_fixture.expected
  assert exec_fixture.requested_contexts == ["fork"]
  assert custom_lint._WORKER_CANDIDATES == ()
  assert custom_lint._WORKER_MODIFIED_LINES is None


@pytest.mark.parametrize("jobs", [0, -1])
def test_execution_jobs_non_positive_auto_detects_cpus(
    exec_fixture: _ExecFixture, jobs: int
) -> None:
  with mock.patch.object(
      custom_lint.os, "cpu_count", return_value=2, autospec=True
  ):
    result = custom_lint.collect_violations(
        exec_fixture.root,
        check_all=True,
        jobs=jobs,
        rules_file=str(exec_fixture.rules_file),
    )
  assert result.effective_jobs == 2
  actual = sorted(
      (v.rule_id, v.file_path, v.line_number or 0) for v in result.violations
  )
  assert actual == exec_fixture.expected


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
    check_contents {
      base { id: "TaggedWithSuppression" disable_tags: "ALLOW_FOO" }
      prohibited_regexp: "foo"
      suppression_regexp: "FOO_OK"
    }
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
  modified = custom_lint_vcs.ModifiedLines(
      lines={"a.txt": set(range(1, content.count("\n") + 2))},
      operations={"a.txt": "EDIT"},
  )
  with (
      mock.patch.object(
          custom_lint_vcs.Vcs,
          "get_modified_lines",
          return_value=modified,
          autospec=True,
      ),
      mock.patch.object(
          custom_lint_vcs.Vcs,
          "get_commit_message",
          return_value=message,
          autospec=True,
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
        (
            "foo\nbar\n",
            "",
            False,
            ["Tagged", "TaggedWithSuppression", "Untagged"],
            1,
        ),
        ("foo\nbar\n", "Fix\n\nALLOW_FOO=reason\n", False, ["Untagged"], 1),
        (
            "foo\nbar\n",
            "Public commit description\n\nGenerated from cl/123456789\n",
            False,
            ["TaggedWithSuppression", "Untagged"],
            1,
        ),
        (
            "foo\n",
            "ALLOW_FOO=reason\n",
            True,
            ["Tagged", "TaggedWithSuppression"],
            0,
        ),
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


def test_oss_ci_dir_is_not_a_monorepo_root(tmp_path: pathlib.Path) -> None:
  """The OSS rules file keeps the monorepo name, so `ci/` must not look like

  a monorepo project root when walking up from a file under it.
  """
  oss_rules = tmp_path / custom_lint.OSS_RULES_FILE
  oss_rules.parent.mkdir(parents=True)
  oss_rules.write_text("", encoding="utf-8")
  (tmp_path / "pyproject.toml").write_text(
      '[tool.custom_linter]\noss_canonical_root = "torch_tpu/oss"\n',
      encoding="utf-8",
  )
  assert oss_rules.name == custom_lint.MONOREPO_RULES_FILE
  start = tmp_path / "ci" / "tools" / "custom_linter"
  start.mkdir(parents=True)

  assert custom_lint.find_project_root(start) == tmp_path.resolve()
  assert custom_lint.find_rules_file(tmp_path) == oss_rules
  mapper = custom_lint.default_path_mapper(tmp_path)
  assert mapper.canonical_root == "torch_tpu/oss"


def test_config_overrides_max_entire_file_bytes_and_excluded_path_regex(
    tmp_path: pathlib.Path,
) -> None:
  root = tmp_path.resolve()
  (root / "pyproject.toml").write_text(
      textwrap.dedent("""\
          [tool.custom_linter]
          max_entire_file_bytes = 10
          excluded_path_regex = '(^|/)skip_me\\.txt$'
          """),
      encoding="utf-8",
  )
  rules_file = root / "rules.METADATA"
  rules_file.write_text(
      textwrap.dedent("""\
          presubmit {
            path_regexp: ".*\\\\.txt"
            check_contents {
              base { id: "WholeFileRule" }
              prohibited_regexp: "bad"
              match_entire_file: true
            }
          }
          """),
      encoding="utf-8",
  )
  (root / "skip_me.txt").write_text("bad\n", encoding="utf-8")
  (root / "small.txt").write_text("bad\n", encoding="utf-8")
  (root / "large.txt").write_text("0123456789abcdef\n", encoding="utf-8")

  res = custom_lint.collect_violations(
      root, check_all=True, jobs=1, rules_file=str(rules_file)
  )
  by_file = {v.file_path: v for v in res.violations}
  assert set(by_file) == {"small.txt", "large.txt"}
  assert by_file["small.txt"].line_number == 1
  assert by_file["large.txt"].line_number is None
  assert "larger than 10 bytes" in by_file["large.txt"].message

  bad_cfg = root / "bad_config.toml"
  bad_cfg.write_text(
      '[tool.custom_linter]\nunknown_option = "x"\n', encoding="utf-8"
  )
  err_out = io.StringIO()
  with mock.patch("sys.stderr", err_out):
    assert (
        custom_lint.run_checks(
            root,
            check_all=True,
            jobs=1,
            rules_file=str(rules_file),
            config_file=bad_cfg,
        )
        == 2
    )
  assert (
      "unknown [tool.custom_linter] key(s): unknown_option"
      in err_out.getvalue()
  )


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
