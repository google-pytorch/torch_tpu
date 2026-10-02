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
import json
import multiprocessing
import os
import pathlib
import random
import re
import subprocess
import sys
import textwrap
from typing import Final
from unittest import mock

import pytest
from ci.tools.custom_linter import engine as custom_lint
from ci.tools.custom_linter import metadata as custom_lint_metadata
from ci.tools.custom_linter import vcs as custom_lint_vcs

_BASE: Final[str] = "//depot/repo/proj"
_OSS_DIR: Final[pathlib.Path] = pathlib.Path(__file__).absolute().parents[4]
_MONOREPO_DIR: Final[pathlib.Path] = _OSS_DIR.parent
_MONOREPO_RULES_FILE: Final[pathlib.Path] = (
    _MONOREPO_DIR / custom_lint.DEFAULT_RULES_FILE
)
_IS_MONOREPO: Final[bool] = _MONOREPO_RULES_FILE.is_file()
_PROJECT_ROOT: Final[pathlib.Path] = _MONOREPO_DIR if _IS_MONOREPO else _OSS_DIR
_PROJECT_CONFIG_FILE: Final[pathlib.Path] = (
    _MONOREPO_DIR / "google" / "linter_config.toml"
    if _IS_MONOREPO
    else _OSS_DIR / "pyproject.toml"
)
_PROJECT_RULES_FILE: Final[pathlib.Path] = (
    _MONOREPO_RULES_FILE
    if _IS_MONOREPO
    else _OSS_DIR / "ci" / custom_lint.DEFAULT_RULES_FILE
)
_OSS_CONFIG_FILE: Final[pathlib.Path] = _OSS_DIR / "pyproject.toml"


@functools.cache
def _project_config() -> custom_lint.LinterConfig:
  return custom_lint.load_config(
      _PROJECT_ROOT, config_file=_PROJECT_CONFIG_FILE
  )


@functools.cache
def _project_mapper() -> custom_lint_metadata.PathMapper:
  return custom_lint.default_path_mapper(_PROJECT_ROOT, _project_config())


@functools.cache
def _oss_project_config() -> custom_lint.LinterConfig:
  return custom_lint.load_config(_OSS_DIR, config_file=_OSS_CONFIG_FILE)


@functools.cache
def _oss_project_mapper() -> custom_lint_metadata.PathMapper:
  cfg = _oss_project_config()
  return custom_lint_metadata.PathMapper(
      cfg.canonical_root,
      tuple(cfg.path_maps.items()),
  )


@functools.cache
def _load_default_rules() -> custom_lint_metadata.RuleSet:
  return custom_lint.load_rules(
      rules_file=_PROJECT_RULES_FILE,
      root_dir=_PROJECT_ROOT,
      mapper=_project_mapper(),
      config=_project_config(),
  )


@functools.cache
def _project_rules() -> Mapping[str, custom_lint_metadata.Rule]:
  return {r.id: r for r in _load_default_rules().rules}


def _rule(rule_id: str) -> custom_lint_metadata.Rule:
  return _project_rules()[rule_id]


def _load(text: str) -> custom_lint_metadata.RuleSet:
  return custom_lint_metadata.parse_metadata(text, "METADATA", _BASE)


def _lines(violations: Iterable[custom_lint.Violation]) -> Sequence[int | None]:
  return [v.line_number for v in violations]


def _check(
    rule_id: str,
    content: str,
    rel_path: str,
    modified: (
        custom_lint_vcs.ModifiedLines | Mapping[str, Set[int]] | None
    ) = None,
    op: custom_lint_metadata.FileOperation | None = None,
) -> Sequence[custom_lint.Violation]:
  if modified is not None and not isinstance(
      modified, custom_lint_vcs.ModifiedLines
  ):
    modified = custom_lint_vcs.ModifiedLines(lines=modified)
  return custom_lint.check_file_content(
      content=content,
      rel_path=rel_path,
      rule=_rule(rule_id),
      modified_lines=modified,
      canonical_path=_project_mapper().to_canonical(rel_path),
      file_operation=op,
  )


# ---------------------------------------------------------------------------
# Project rules & path mapping tests
# ---------------------------------------------------------------------------


def test_project_rules_load() -> None:
  rule_set = _load_default_rules()
  assert not rule_set.skipped
  for rule in rule_set.rules:
    rule.prewarm()


@pytest.mark.skipif(not _IS_MONOREPO, reason="Needs the monorepo rules file")
def test_oss_export_matches_the_same_files(tmp_path: pathlib.Path) -> None:
  g3_rules = _load_default_rules().rules
  g3_mapper = _project_mapper()
  text = _PROJECT_RULES_FILE.read_text(encoding="utf-8")
  # LINT.IfChange(custom_lint_rules_move)
  text = re.sub(
      rf'(path_\w+): "{re.escape(g3_mapper.canonical_root)}',
      r'\1: "torch_tpu',
      text,
  )
  text = re.sub(r'(path_\w+): "//depot/\.\.\./', r'\1: ".../', text)
  # LINT.ThenChange(../../../../../copy.bara.sky:custom_lint_rules_move)
  oss_cfg = _oss_project_config()
  oss_mapper = _oss_project_mapper()
  oss_rules_file = tmp_path / oss_cfg.rules_file
  oss_rules_file.parent.mkdir(parents=True, exist_ok=True)
  oss_rules_file.write_text(text, encoding="utf-8")
  oss_rules = custom_lint.load_rules(
      rules_file=oss_rules_file,
      root_dir=tmp_path,
      mapper=oss_mapper,
      config=oss_cfg,
  ).rules
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
    for g3_rule, oss_rule in zip(g3_rules, oss_rules, strict=True):
      assert g3_rule.matches_path(g3_path) == oss_rule.matches_path(
          oss_path
      ), f"{g3_rule.id} on {g3_rel}"
  for oss_rule in oss_rules:
    example = _RULE_EXAMPLES[oss_rule.id]
    oss_canonical = oss_mapper.to_canonical(example.path)
    assert oss_rule.matches_path(
        oss_canonical
    ), f"{oss_rule.id} example path {example.path!r} does not match in OSS"
    bad_lines = custom_lint_vcs.ModifiedLines(
        {example.path: set(range(1, example.bad.count("\n") + 2))}
    )
    assert (
        len(
            custom_lint.check_file_content(
                content=example.bad,
                rel_path=example.path,
                rule=oss_rule,
                modified_lines=bad_lines,
                canonical_path=oss_canonical,
                file_operation=example.op,
            )
        )
        == 1
    ), f"{oss_rule.id} bad example did not fire in OSS"
    if example.good is not None:
      good_lines = custom_lint_vcs.ModifiedLines(
          {example.path: set(range(1, example.good.count("\n") + 2))}
      )
      assert not custom_lint.check_file_content(
          content=example.good,
          rel_path=example.path,
          rule=oss_rule,
          modified_lines=good_lines,
          canonical_path=oss_canonical,
          file_operation=example.op,
      ), f"{oss_rule.id} good example fired in OSS"


def test_default_path_mapper_and_find_rules_file_by_layout(
    tmp_path: pathlib.Path,
) -> None:
  g3_root = (tmp_path / "pkg" / "my_proj").resolve()
  g3_root.mkdir(parents=True)
  (g3_root / "pyproject.toml").write_text(
      '[tool.custom_linter]\ncanonical_root = "//depot/repo/pkg/my_proj"\n',
      encoding="utf-8",
  )
  g3_rules = g3_root / custom_lint.DEFAULT_RULES_FILE
  g3_rules.write_text("", encoding="utf-8")
  assert custom_lint.find_rules_file(g3_root) == g3_rules
  g3_mapper = custom_lint.default_path_mapper(g3_root)
  assert g3_mapper.canonical_root == "//depot/repo/pkg/my_proj"

  oss_root = (tmp_path / "oss_repo").resolve()
  oss_rules = oss_root / "ci" / custom_lint.DEFAULT_RULES_FILE
  oss_rules.parent.mkdir(parents=True)
  oss_rules.write_text("", encoding="utf-8")
  (oss_root / "pyproject.toml").write_text(
      '[tool.custom_linter]\ncanonical_root = "torch_tpu/oss"\nrules_file ='
      ' "ci/custom_lint_rules.METADATA"\n',
      encoding="utf-8",
  )
  assert custom_lint.find_rules_file(oss_root) == oss_rules
  oss_mapper = custom_lint.default_path_mapper(oss_root)
  assert oss_mapper.canonical_root == "torch_tpu/oss"

  unconfigured_root = tmp_path.joinpath("repo", "pkg", "plain").resolve()
  unconfigured_root.mkdir(parents=True)
  unconfigured_mapper = custom_lint.default_path_mapper(unconfigured_root)
  assert (
      unconfigured_mapper.canonical_root == custom_lint.DEFAULT_CANONICAL_ROOT
  )

  standalone_cfg_file = unconfigured_root / "config" / "linter_config.toml"
  standalone_cfg_file.parent.mkdir(parents=True)
  standalone_cfg_file.write_text(
      '[tool.custom_linter]\ncanonical_root = "//depot/repo/pkg/plain"\n',
      encoding="utf-8",
  )
  standalone_cfg = custom_lint.load_config(
      unconfigured_root, config_file=standalone_cfg_file
  )
  assert (
      custom_lint.default_path_mapper(
          unconfigured_root, standalone_cfg
      ).canonical_root
      == "//depot/repo/pkg/plain"
  )


def test_rules_with_unmatched_or_nonexistent_paths_match_zero_files(
    tmp_path: pathlib.Path,
) -> None:
  rules_file = tmp_path / custom_lint.DEFAULT_RULES_FILE
  rules_file.write_text(
      'presubmit {\n  path_expression: "//depot/repo/pkg/csrc/..."\n '
      ' path_expression_exclusion: "//depot/other/pkg/..."\n  check_contents {'
      ' base { id: "RootedRule" } prohibited_regexp: "foo" }\n}\n',
      encoding="utf-8",
  )
  (tmp_path / "top.cc").write_text("foo\n", encoding="utf-8")

  # Unmatched canonical_root or missing `csrc/` directory matches 0 files
  # without raising ConfigError.
  assert not custom_lint.collect_violations(tmp_path, check_all=True).violations
  matching_cfg = custom_lint.LinterConfig(canonical_root="//depot/repo/pkg")
  assert not custom_lint.collect_violations(
      tmp_path, check_all=True, config=matching_cfg
  ).violations

  (tmp_path / "csrc").mkdir()
  (tmp_path / "csrc" / "a.cc").write_text("foo\n", encoding="utf-8")
  res = custom_lint.collect_violations(
      tmp_path, check_all=True, config=matching_cfg
  )
  assert [v.file_path for v in res.violations] == ["csrc/a.cc"]


def test_rules_base_dir_config_resolves_rooted_and_relative_rules(
    tmp_path: pathlib.Path,
) -> None:
  root = tmp_path.resolve()
  (root / "pyproject.toml").write_text(
      textwrap.dedent("""\
          [tool.custom_linter]
          canonical_root = "torch_tpu/oss"
          rules_base_dir = "torch_tpu/"
          rules_file = "ci/custom_lint_rules.METADATA"

          [tool.custom_linter.path_maps]
          "csrc" = "torch_tpu/csrc"
          """),
      encoding="utf-8",
  )
  cfg = custom_lint.load_config(root)
  assert cfg.rules_base_dir == "torch_tpu"

  rules_file = root / "ci" / custom_lint.DEFAULT_RULES_FILE
  rules_file.parent.mkdir(parents=True)
  rules_file.write_text(
      textwrap.dedent("""\
          presubmit {
            path_expression: "torch_tpu/csrc/..."
            check_contents {
              base { id: "RootedCsrcRule" }
              prohibited_regexp: "foo"
            }
          }
          presubmit {
            path_expression: "oss/shims/..."
            check_contents {
              base { id: "RelativeOssRule" }
              prohibited_regexp: "bar"
            }
          }
          """),
      encoding="utf-8",
  )
  (root / "csrc").mkdir()
  (root / "csrc" / "a.cc").write_text("foo\n", encoding="utf-8")
  (root / "shims").mkdir()
  (root / "shims" / "a.bzl").write_text("bar\n", encoding="utf-8")

  res = custom_lint.collect_violations(root, check_all=True, jobs=1)
  assert sorted((v.rule_id, v.file_path) for v in res.violations) == [
      ("RelativeOssRule", "shims/a.bzl"),
      ("RootedCsrcRule", "csrc/a.cc"),
  ]

  sub_rules = root / "shims" / "sub" / "METADATA"
  sub_rules.parent.mkdir(parents=True)
  sub_rules.write_text(
      'presubmit {\n  path_expression: "local/..."\n  check_contents {'
      ' base { id: "SubRule" } prohibited_regexp: "baz" }\n}\n',
      encoding="utf-8",
  )
  loaded_sub = custom_lint.load_rules(sub_rules, root_dir=root, config=cfg)
  assert loaded_sub.rules[0].matches_path("torch_tpu/oss/shims/sub/local/x.py")


def test_is_rule_disabled() -> None:
  rule = custom_lint_metadata.Rule(
      id="TestRule",
      path_regex=r".*",
      prohibited_regex=r"foo",
      failure_message="msg",
      disable_tags=frozenset({"ALLOW_FOO", "ROLLBACK_OF"}),
  )
  assert custom_lint.is_rule_disabled(
      rule, custom_lint_vcs.Commit("Fix: something\nALLOW_FOO=needed")
  )
  assert custom_lint.is_rule_disabled(
      rule, custom_lint_vcs.Commit("Revert\n\nROLLBACK_OF=1")
  )
  assert not custom_lint.is_rule_disabled(
      rule, custom_lint_vcs.Commit("Automated ROLLBACK_OF cl/123")
  )
  assert not custom_lint.is_rule_disabled(
      rule, custom_lint_vcs.Commit("Public desc\n\nGenerated from cl/123456")
  )
  assert not custom_lint.is_rule_disabled(
      rule, custom_lint_vcs.Commit("Normal commit")
  )


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
  # Line-mode rules (without match_multi_line) only honor suppressions on the
  # same line as the match.
  line_rule = custom_lint_metadata.Rule(
      id="LineOnlyRule",
      path_regex=r".*BUILD$",
      prohibited_regex=r"cc_library\(",
      suppression_regex=r"ALLOW_CC_TARGETS=",
      failure_message="msg",
      mode=custom_lint_metadata.MatchMode.LINE,
  )
  assert _lines(
      custom_lint.check_file_content(
          content,
          "BUILD",
          line_rule,
          custom_lint_vcs.ModifiedLines({"BUILD": {1, 2}}),
          canonical_path="//depot/repo/proj/BUILD",
      )
  ) == [1]
  assert not custom_lint.check_file_content(
      'cc_library(name = "test")  # ALLOW_CC_TARGETS=shim\n',
      "BUILD",
      line_rule,
      custom_lint_vcs.ModifiedLines({"BUILD": {1}}),
      canonical_path="//depot/repo/proj/BUILD",
  )


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
  assert _lines(_check("CheckPyLibraryUsage", content, rel)) == [5]
  assert _lines(
      _check("CheckPyLibraryUsage", content, rel, {rel: set(range(1, 8))})
  ) == [5]
  same_line = (
      'py_library(name = "a")\npy_library(name = "b")  # PY_LIBRARY_OK=shim\n'
  )
  assert not _check("CheckPyLibraryUsage", same_line, rel, {rel: {2}})
  assert _lines(
      _check("CheckPyLibraryUsage", same_line, rel, {rel: {1, 2}})
  ) == [1]
  reversed_content = (
      'py_test(\n    name = "a",\n)\n'
      'py_test(\n    # PY_TEST_OK=shim\n    name = "b",\n)\n'
  )
  assert _lines(_check("CheckPyTestUsage", reversed_content, rel)) == [1]
  assert _lines(
      _check("CheckPyTestUsage", reversed_content, rel, {rel: set(range(1, 8))})
  ) == [1]
  both_justified = (
      'py_test(\n    # PY_TEST_OK=shim_a\n    name = "a",\n)\n'
      'py_test(\n    # PY_TEST_OK=shim_b\n    name = "b",\n)\n'
  )
  assert not _check("CheckPyTestUsage", both_justified, rel)
  assert not _check(
      "CheckPyTestUsage", both_justified, rel, {rel: set(range(1, 9))}
  )


def test_file_operation_rule() -> None:
  rel = "csrc/ops/addkernel.cc"
  assert _lines(
      _check(
          "CheckOpFileName",
          "x\n",
          rel,
          {rel: {1}},
          custom_lint_metadata.FileOperation.ADD,
      )
  ) == [None]
  assert not _check(
      "CheckOpFileName",
      "x\n",
      rel,
      {rel: {1}},
      custom_lint_metadata.FileOperation.EDIT,
  )
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
          canonical_path=_project_mapper().to_canonical(rel),
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
_RULE_EXAMPLES: Final[Mapping[str, _RuleExample]] = {
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
    "CheckAtToOpMathType": _RuleExample(
        path="csrc/a.cc",
        bad="at::toOpMathType(t);\n",
        good="at::toOpMathType(t);  // AT_TO_OP_MATH_TYPE_OK=r\n",
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
    "CheckSetenv": _RuleExample(
        path="csrc/a.cc",
        bad='setenv(kName, "1", 1);\n',
        good='setenv(kName, "1", 1);  // SETENV_OK=r\n',
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
    "CheckSetFlag": _RuleExample(
        path="csrc/a.cc",
        bad="absl::SetFlag(&FLAGS_x, 1);\n",
        good="absl::SetFlag(&FLAGS_x, 1);  // SET_FLAG_OK=r\n",
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
    "CheckDefaultRefLambdaCapture": _RuleExample(
        path="csrc/a.cc",
        bad="auto fn = [&]() {};\n",
        good="auto fn = [&]() {};  // REF_CAPTURE_OK=local\n",
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
  assert rule.matches_path(_project_mapper().to_canonical(example.path))
  assert bool(rule.operations) == (example.op is not None)
  assert (
      rule.suppression_regex is not None or rule.required_regex is not None
  ) == (example.good is not None)

  bad_lines = {example.path: set(range(1, example.bad.count("\n") + 2))}
  violations = _check(rule_id, example.bad, example.path, bad_lines, example.op)
  assert len(violations) == 1
  assert violations[0].rule_id == rule_id


@pytest.mark.parametrize(
    "rule_id", [rid for rid in _RULE_EXAMPLES if _rule(rid).operations]
)
def test_project_rule_operation_filtering(rule_id: str) -> None:
  rule = _rule(rule_id)
  example = _RULE_EXAMPLES[rule_id]
  assert example.op in rule.operations
  bad_lines = {example.path: set(range(1, example.bad.count("\n") + 2))}
  other_op = next(
      (
          op
          for op in custom_lint_metadata.FileOperation
          if op not in rule.operations
      ),
      None,
  )
  assert other_op is not None, f"{rule_id} lists all FileOperations"
  assert not _check(rule_id, example.bad, example.path, bad_lines, other_op)


@pytest.mark.parametrize(
    "rule_id",
    [rid for rid, ex in _RULE_EXAMPLES.items() if ex.good is not None],
)
def test_project_rule_good_examples(rule_id: str) -> None:
  example = _RULE_EXAMPLES[rule_id]
  assert example.good is not None
  good_lines = custom_lint_vcs.ModifiedLines(
      {example.path: set(range(1, example.good.count("\n") + 2))}
  )
  assert not _check(rule_id, example.good, example.path, good_lines, example.op)


@pytest.mark.parametrize(
    "rule_id",
    [rid for rid in _RULE_EXAMPLES if _rule(rid).suppression_regex is not None],
)
def test_project_rule_suppression_examples(rule_id: str) -> None:
  rule = _rule(rule_id)
  example = _RULE_EXAMPLES[rule_id]
  assert example.good is not None
  good_lines = custom_lint_vcs.ModifiedLines(
      {example.path: set(range(1, example.good.count("\n") + 2))}
  )
  unsuppressed_rule = dataclasses.replace(rule, suppression_regex=None)
  assert custom_lint.check_file_content(
      content=example.good,
      rel_path=example.path,
      rule=unsuppressed_rule,
      modified_lines=good_lines,
      canonical_path=_project_mapper().to_canonical(example.path),
      file_operation=example.op,
  )


@pytest.mark.parametrize(
    ("rule_id", "tag"),
    [
        (rid, tag)
        for rid in _RULE_EXAMPLES
        for tag in sorted(_rule(rid).disable_tags)
    ],
)
def test_project_rule_disable_tags(rule_id: str, tag: str) -> None:
  assert custom_lint.is_rule_disabled(
      _rule(rule_id), custom_lint_vcs.Commit(f"Summary\n\n{tag}=reason")
  )


@pytest.mark.parametrize(
    "path",
    [
        "custom_lint_rules.METADATA",
        "ci/custom_lint_rules.METADATA",
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
        "copy.bara.sky",
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

  mapper = _project_mapper()
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
  (root_path / "linked.cc").symlink_to(cc_file)
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
  assert [c.rel_path for c in candidates] == [
      "foo.cc",
      "linked.cc",
      "sub/baz.cc",
  ]
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


@pytest.fixture(name="ignore_dirs_tree")
def _ignore_dirs_tree_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
  root_path = tmp_path.resolve()
  for d in (
      ".git",
      ".github",
      ".pytest_cache",
      ".ruff_cache",
      "src",
      "node_modules",
      "venv",
      "bazel-bin",
      "bazel-custom",
      "other-bin",
  ):
    (root_path / d).mkdir()
    (root_path / d / "x.cc").write_text("foo\n", encoding="utf-8")
  # Symlinked directories and broken symlinks are skipped (neither descended
  # into nor yielded as candidate files, even under catch-all path rules).
  (root_path / "linked").symlink_to(root_path / "src", target_is_directory=True)
  (root_path / "linked_dir.cc").symlink_to(
      root_path / "src", target_is_directory=True
  )
  (root_path / "broken.cc").symlink_to(root_path / "missing.cc")
  return root_path


@pytest.mark.parametrize("mapper_kind", ["monorepo", "oss"])
def test_default_ignore_dirs(
    ignore_dirs_tree: pathlib.Path,
    mapper_kind: str,
) -> None:
  mapper = (
      custom_lint_metadata.PathMapper("//depot/repo/proj")
      if mapper_kind == "monorepo"
      else _oss_project_mapper()
  )
  assert "node_modules" in custom_lint.DEFAULT_IGNORE_DIRS
  assert "bazel-bin" in custom_lint.DEFAULT_IGNORE_DIRS
  rule_any_cc = custom_lint_metadata.Rule(
      id="RuleCC",
      path_regex=r".*\.cc$",
      prohibited_regex=r"foo",
      failure_message="msg",
  )
  candidates = custom_lint.find_candidate_files(
      ignore_dirs_tree, [rule_any_cc], mapper=mapper
  )
  assert [c.rel_path for c in candidates] == [
      ".github/x.cc",
      "bazel-custom/x.cc",
      "other-bin/x.cc",
      "src/x.cc",
  ]
  result = custom_lint.collect_violations(
      ignore_dirs_tree,
      check_all=True,
      rules_file=_PROJECT_RULES_FILE,
      only_rules=["CheckCopybaraTags"],
      mapper=_project_mapper(),
      config=_project_config(),
  )
  assert not result.violations


@pytest.fixture(name="exclude_dir_tree")
def _exclude_dir_tree_fixture(
    tmp_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
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
  return root_path, rules_file


def test_exclude_dir_and_no_default_excludes(
    exclude_dir_tree: tuple[pathlib.Path, pathlib.Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
  root_path, rules_file = exclude_dir_tree
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
  parsed = json.loads(capsys.readouterr().out)
  assert [v["file_path"] for v in parsed] == ["keep/file.cc"]

  res = custom_lint.collect_violations(
      root_path,
      check_all=True,
      jobs=1,
      rules_file=str(rules_file),
      exclude_dirs=["skip_a", "skip_b"],
      use_default_excludes=False,
  )
  assert [v.file_path for v in res.violations] == [
      "keep/file.cc",
      "venv/file.cc",
  ]


def test_binary_files_skipped_by_default(tmp_path: pathlib.Path) -> None:
  tag = "copybara:" + "strip"
  root_path = tmp_path.resolve()
  proj_cfg = _project_config()
  (root_path / "pyproject.toml").write_text(
      "[tool.custom_linter]\ncanonical_root ="
      f' "{_project_mapper().canonical_root}"\nexcluded_path_regex ='
      f" {proj_cfg.excluded_path_regex!r}\n",
      encoding="utf-8",
  )
  (root_path / "golden.gz").write_bytes(
      b"\x1f\x8b\x08\x00" + tag.encode() + b"\n"
  )
  (root_path / "binary_blob.cc").write_bytes(
      b"int x = 0;\x00" + tag.encode() + b"\n"
  )
  (root_path / "large_binary_blob.cc").write_bytes(
      b"a" * (custom_lint.BINARY_SNIFF_BYTES + 16)
      + b"\x00"
      + tag.encode()
      + b"\n"
  )
  (root_path / "valid.cc").write_text(f"// {tag}\n", encoding="utf-8")

  res = custom_lint.collect_violations(
      root_path,
      check_all=True,
      jobs=1,
      rules_file=_PROJECT_RULES_FILE,
      only_rules=["CheckCopybaraTags"],
  )
  assert [v.file_path for v in res.violations] == ["valid.cc"]


def test_non_utf8_file_decoded_with_replacement_and_checked(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
  root_path = tmp_path.resolve()
  rules_file = root_path / "rules.METADATA"
  rules_file.write_text(
      'presubmit { path_regexp: ".*\\\\.cc"\n'
      '  check_contents { base { id: "NoFoo" } prohibited_regexp: "foo" }\n'
      "}\n",
      encoding="utf-8",
  )

  bad_utf8 = root_path / "bad_utf8.cc"
  bad_utf8.write_bytes(b"// caf\xe9 \xff\xfe\nfoo\n")
  clean_utf8 = root_path / "clean_non_utf8.cc"
  clean_utf8.write_bytes(b"// caf\xe9 \xff\xfe\nint x = 1;\n")

  res = custom_lint.collect_violations(
      root_path,
      check_all=True,
      jobs=1,
      rules_file=str(rules_file),
  )
  assert [(v.rule_id, v.file_path, v.line_number) for v in res.violations] == [
      ("NoFoo", "bad_utf8.cc", 2)
  ]

  assert (
      custom_lint.main([
          "--root",
          str(root_path),
          "--rules",
          str(rules_file),
          "--jobs",
          "1",
          "clean_non_utf8.cc",
      ])
      == 0
  )
  capsys.readouterr()
  assert (
      custom_lint.main([
          "--root",
          str(root_path),
          "--rules",
          str(rules_file),
          "--jobs",
          "1",
          "bad_utf8.cc",
      ])
      == 1
  )
  assert "[NoFoo] bad_utf8.cc:2" in capsys.readouterr().out


def test_missing_explicit_path_raises_and_vanished_candidate_is_skipped(
    tmp_path: pathlib.Path,
) -> None:
  root_path = tmp_path.resolve()
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

  vanished = root_path / "vanished.cc"
  vanished.write_text("foo\n", encoding="utf-8")
  real_read_text_file = custom_lint._read_text_file

  def unlink_then_read(path: pathlib.Path) -> str | None:
    if path == vanished:
      vanished.unlink()
    return real_read_text_file(path)

  with mock.patch.object(
      custom_lint,
      "_read_text_file",
      side_effect=unlink_then_read,
      autospec=True,
  ):
    res = custom_lint.collect_violations(
        root_path,
        check_all=True,
        jobs=1,
        rules_file=str(rules_file),
    )
  assert not res.violations


def test_main_maps_io_and_encoding_errors_to_exit_2(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
  root_path = tmp_path.resolve()
  rules_file = root_path / "rules.METADATA"
  rules_file.write_text(
      'presubmit { path_regexp: ".*\\\\.cc"\n'
      '  check_contents { base { id: "NoFoo" } prohibited_regexp: "foo" }\n'
      "}\n",
      encoding="utf-8",
  )
  candidate = root_path / "a.cc"
  candidate.write_text("clean\n", encoding="utf-8")

  bad_cfg = root_path / "bad_config.toml"
  bad_cfg.write_bytes(b"[tool.custom_linter]\n# \xff\xfe\n")
  assert (
      custom_lint.main([
          "--root",
          str(root_path),
          "--config",
          str(bad_cfg),
          "--rules",
          str(rules_file),
          "--all",
          "--jobs",
          "1",
      ])
      == 2
  )
  assert f"ERROR: {bad_cfg}: invalid config file:" in capsys.readouterr().err

  bad_rules = root_path / "bad_rules.METADATA"
  bad_rules.write_bytes(b"presubmit { \xff\xfe }\n")
  assert (
      custom_lint.main([
          "--root",
          str(root_path),
          "--rules",
          str(bad_rules),
          "--all",
          "--jobs",
          "1",
      ])
      == 2
  )
  assert (
      f"ERROR: {bad_rules}: cannot read rules file:" in capsys.readouterr().err
  )

  missing_report = root_path / "no_such_dir" / "report.txt"
  assert (
      custom_lint.main([
          "--root",
          str(root_path),
          "--rules",
          str(rules_file),
          "--report-file",
          str(missing_report),
          "--all",
          "--jobs",
          "1",
      ])
      == 2
  )
  err = capsys.readouterr().err
  assert "ERROR:" in err and str(missing_report) in err

  real_read_text_file = custom_lint._read_text_file

  def deny_candidate(path: pathlib.Path) -> str | None:
    if path == candidate:
      raise PermissionError(13, "Permission denied", str(candidate))
    return real_read_text_file(path)

  with mock.patch.object(
      custom_lint,
      "_read_text_file",
      side_effect=deny_candidate,
      autospec=True,
  ):
    assert (
        custom_lint.main([
            "--root",
            str(root_path),
            "--rules",
            str(rules_file),
            "--all",
            "--jobs",
            "1",
        ])
        == 2
    )
  err = capsys.readouterr().err
  assert (
      "ERROR:" in err and "Permission denied" in err and str(candidate) in err
  )


def test_main_unexpected_exception_prints_traceback_and_exits_2(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
  root_path = tmp_path.resolve()
  rules_file = root_path / "rules.METADATA"
  rules_file.write_text("", encoding="utf-8")

  with mock.patch.object(
      custom_lint,
      "run_checks",
      side_effect=RuntimeError("boom"),
      autospec=True,
  ):
    assert (
        custom_lint.main([
            "--root",
            str(root_path),
            "--rules",
            str(rules_file),
            "--all",
        ])
        == 2
    )
  err = capsys.readouterr().err
  assert "Traceback (most recent call last):" in err
  assert "ERROR: unexpected internal error (RuntimeError): boom" in err


@pytest.fixture(name="cc_rules_repo")
def _cc_rules_repo_fixture(
    tmp_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
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
  return root_path, rules_file


def test_run_checks_warning_only_exits_0(
    cc_rules_repo: tuple[pathlib.Path, pathlib.Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
  root_path, rules_file = cc_rules_repo
  (root_path / "a.cc").write_text("foo\n", encoding="utf-8")

  exit_code = custom_lint.run_checks(
      root_path, check_all=True, jobs=1, rules_file=str(rules_file)
  )
  captured = capsys.readouterr()
  assert exit_code == 0
  assert (
      "WARNING: 1 custom convention check violation(s) found:" in captured.out
  )
  assert "[NoFoo] (WARNING) a.cc:1" in captured.out


def test_main_json_and_report_file_output(
    cc_rules_repo: tuple[pathlib.Path, pathlib.Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
  root_path, rules_file = cc_rules_repo
  (root_path / "a.cc").write_text("foo\nbar\n", encoding="utf-8")
  report = root_path / "report.txt"

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
  captured = capsys.readouterr()
  assert exit_code == 1
  parsed = json.loads(captured.out)
  assert [v["rule_id"] for v in parsed] == ["NoFoo", "NoBar"]
  assert "INFO: Checked 1 files" in captured.err
  report_text = report.read_text(encoding="utf-8")
  assert "# NoFoo  [WARNING]  1 findings" in report_text
  assert "# NoBar  [ERROR]  1 findings" in report_text


def test_run_checks_unknown_rule_id_exits_2(
    cc_rules_repo: tuple[pathlib.Path, pathlib.Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
  root_path, rules_file = cc_rules_repo
  exit_code = custom_lint.run_checks(
      root_path,
      check_all=True,
      jobs=1,
      rules_file=str(rules_file),
      only_rules=["UnknownRule"],
  )
  captured = capsys.readouterr()
  assert exit_code == 2
  assert "unknown rule id(s): UnknownRule" in captured.err


@pytest.fixture(name="rules_repo")
def _rules_repo_fixture(
    tmp_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
  """Creates a minimal project root with a single `.txt` rule."""
  root_path = tmp_path.resolve()
  (root_path / "pyproject.toml").write_text("", encoding="utf-8")
  rules_file = root_path / "rules.METADATA"
  rules_file.write_text(
      'presubmit { path_regexp: ".*\\\\.txt" check_contents { base { id: "R"'
      ' disable_tags: "ALLOW_R" } prohibited_regexp: "x" } }\n',
      encoding="utf-8",
  )
  return root_path, rules_file


def test_run_checks_vcs_failure_raises_vcs_error(
    rules_repo: tuple[pathlib.Path, pathlib.Path],
) -> None:
  root_path, rules_file = rules_repo
  fail_result = subprocess.CompletedProcess(
      args=[], returncode=1, stdout="", stderr="fatal: error"
  )
  with mock.patch("subprocess.run", return_value=fail_result, autospec=True):
    with pytest.raises(custom_lint_vcs.VcsError) as exc_info:
      custom_lint.run_checks(
          root_dir=root_path,
          check_all=False,
          files=None,
          base_sha="origin/main",
          rules_file=str(rules_file),
      )
  err_str = str(exc_info.value)
  assert "could not compute the diff against origin/main: " in err_str
  assert "fatal: error" in err_str
  assert "; pass --all to scan everything" in err_str


def test_main_vcs_failure_with_base_sha_flag_exits_2(
    rules_repo: tuple[pathlib.Path, pathlib.Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  root_path, rules_file = rules_repo
  fail_result = subprocess.CompletedProcess(
      args=[], returncode=1, stdout="", stderr="fatal: error"
  )
  monkeypatch.setattr(
      sys,
      "argv",
      [
          "custom_lint",
          "--root",
          str(root_path),
          "--base-sha",
          "origin/main",
          "--rules",
          str(rules_file),
      ],
  )
  with mock.patch("subprocess.run", return_value=fail_result, autospec=True):
    code = custom_lint.main()
  captured = capsys.readouterr()
  assert code == 2
  assert "could not compute the diff against origin/main: " in captured.err
  assert "fatal: error" in captured.err
  assert "; pass --all to scan everything" in captured.err


def test_main_reads_base_sha_env_var(
    rules_repo: tuple[pathlib.Path, pathlib.Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  root_path, rules_file = rules_repo
  fail_result = subprocess.CompletedProcess(
      args=[], returncode=1, stdout="", stderr="fatal: error"
  )
  monkeypatch.setenv("BASE_SHA", "env_base_sha")
  with mock.patch("subprocess.run", return_value=fail_result, autospec=True):
    code = custom_lint.main([
        "--root",
        str(root_path),
        "--rules",
        str(rules_file),
    ])
  captured = capsys.readouterr()
  assert code == 2
  assert "could not compute the diff against env_base_sha: " in captured.err
  assert "fatal: error" in captured.err
  assert "; pass --all to scan everything" in captured.err


def test_main_reads_head_sha_env_var(
    rules_repo: tuple[pathlib.Path, pathlib.Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  root_path, rules_file = rules_repo
  (root_path / "a.txt").write_text("x\n", encoding="utf-8")
  monkeypatch.setenv("BASE_SHA", "base123")
  monkeypatch.setenv("HEAD_SHA", "head456")

  def fake_run(
      cmd: Sequence[str], **kwargs: object
  ) -> subprocess.CompletedProcess[str]:
    del kwargs
    if cmd[:2] == ["git", "merge-base"]:
      return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout="base123\n", stderr=""
      )
    if cmd[:2] == ["git", "diff"]:
      return subprocess.CompletedProcess(
          args=cmd,
          returncode=0,
          stdout="+++ b/a.txt\n@@ -0,0 +1 @@\n+x\n",
          stderr="",
      )
    if cmd == ["git", "log", "base123..head456", "--format=%B"]:
      return subprocess.CompletedProcess(
          args=cmd,
          returncode=0,
          stdout="PR commit\n\nALLOW_R=needed\n",
          stderr="",
      )
    return subprocess.CompletedProcess(
        args=cmd, returncode=1, stdout="", stderr="unexpected"
    )

  with mock.patch("subprocess.run", side_effect=fake_run, autospec=True):
    code = custom_lint.main([
        "--root",
        str(root_path),
        "--rules",
        str(rules_file),
        "--vcs",
        "git",
        "--jobs",
        "1",
        "--format",
        "json",
    ])
  captured = capsys.readouterr()
  assert code == 0
  assert json.loads(captured.out) == []


def test_run_checks_check_all_skips_vcs(
    rules_repo: tuple[pathlib.Path, pathlib.Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
  root_path, rules_file = rules_repo
  with mock.patch("subprocess.run", autospec=True) as mock_run:
    code = custom_lint.run_checks(
        root_dir=root_path,
        check_all=True,
        files=None,
        rules_file=str(rules_file),
    )
  capsys.readouterr()
  assert code == 0
  mock_run.assert_not_called()


def test_run_checks_explicit_files_skips_vcs(
    rules_repo: tuple[pathlib.Path, pathlib.Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
  root_path, rules_file = rules_repo
  test_file = root_path / "test.txt"
  test_file.write_text("clean\n", encoding="utf-8")
  with mock.patch("subprocess.run", autospec=True) as mock_run:
    code = custom_lint.run_checks(
        root_dir=root_path,
        check_all=False,
        files=[str(test_file)],
        rules_file=str(rules_file),
    )
  capsys.readouterr()
  assert code == 0
  mock_run.assert_not_called()


def test_root_is_authoritative_and_default_vcs_root_resolution(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  vcs_root = tmp_path.resolve()
  (vcs_root / ".git").mkdir()
  (vcs_root / "pyproject.toml").write_text("", encoding="utf-8")
  (vcs_root / "parent.cc").write_text("foo\n", encoding="utf-8")

  sub_root = vcs_root / "sub_pkg"
  sub_root.mkdir()
  (sub_root / custom_lint.DEFAULT_RULES_FILE).write_text(
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
  monkeypatch.setenv("BUILD_WORKING_DIRECTORY", str(nested_cwd))
  assert custom_lint.main(["--all", "--jobs", "1", "--format", "json"]) == 1
  assert [v["file_path"] for v in json.loads(capsys.readouterr().out)] == [
      "child.cc"
  ]

  # No rules file between the cwd and the VCS root: ask for --root.
  other_cwd = vcs_root / "other"
  other_cwd.mkdir()
  monkeypatch.setenv("BUILD_WORKING_DIRECTORY", str(other_cwd))
  assert custom_lint.main(["--all", "--jobs", "1"]) == 2
  assert "pass --root" in capsys.readouterr().err

  (vcs_root / "pyproject.toml").write_text(
      '[tool.custom_linter]\nrules_file = "ci/custom_lint_rules.METADATA"\n',
      encoding="utf-8",
  )
  oss_rules = vcs_root / "ci" / custom_lint.DEFAULT_RULES_FILE
  oss_rules.parent.mkdir(parents=True)
  oss_rules.write_text(
      'presubmit { path_regexp: ".*\\\\.cc"\n'
      '  check_contents { base { id: "NoFoo" } prohibited_regexp: "foo" }\n'
      "}\n",
      encoding="utf-8",
  )
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
  assert [v["file_path"] for v in json.loads(capsys.readouterr().out)] == [
      "parent.cc"
  ]


# ---------------------------------------------------------------------------
# Differential & execution-path tests
# ---------------------------------------------------------------------------

_DIFF_RULES: Final[str] = r"""
presubmit {
  path_regexp: ".*\\.txt"
  check_contents { base { id: "Plain" } prohibited_regexp: "\\bfoo\\(" suppression_regexp: "FOO_OK" }
  check_contents { base { id: "Anchored" } prohibited_regexp: "^\\s*bar$" }
  check_contents { base { id: "IgnoreCase" } prohibited_regexp: "(?i)todo\\b" }
  check_contents { base { id: "Alternation" } prohibited_regexp: "(tpu|gpu)\\s*=\\s*\\(" }
  check_contents { base { id: "Optional" } prohibited_regexp: "(?:no)?baz\\d" }
  check_contents { base { id: "TextAnchor" } prohibited_regexp: "qux\\z" }
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

_DIFF_LINES: Final[Sequence[str]] = (
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

_PROJECT_LINES: Final[Sequence[str]] = (
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


class _FakeVcs(custom_lint_vcs.Vcs):
  """In-memory VCS test double returning fixed diff and commit message."""

  def __init__(
      self,
      modified: custom_lint_vcs.ModifiedLines,
      message: str = "",
  ) -> None:
    super().__init__()
    self._modified = modified
    self._message = message
    self.commit_message_calls = 0

  def diff(self, base: str | None = None) -> str:
    del base
    return ""

  def get_commit_message(
      self, base: str | None = None, head: str | None = None
  ) -> str:
    del base, head
    self.commit_message_calls += 1
    return self._message

  def get_modified_lines(
      self,
      base: str | None = None,
      root_dir: pathlib.Path | None = None,
  ) -> custom_lint_vcs.ModifiedLines:
    del base, root_dir
    return self._modified


def _fixture_texts(
    lines: Sequence[str], count: int, seed: int
) -> Sequence[str]:
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
  operations: dict[str, custom_lint_metadata.FileOperation] = {}
  for i, text in enumerate(texts):
    if rng.random() < 0.2:
      continue
    n = text.count("\n") + 1
    lines[f"f{i}.txt"] = {j for j in range(1, n + 1) if rng.random() < 0.5}
    operations[f"f{i}.txt"] = rng.choice((
        custom_lint_metadata.FileOperation.ADD,
        custom_lint_metadata.FileOperation.EDIT,
        custom_lint_metadata.FileOperation.BRANCH,
    ))
  return custom_lint_vcs.ModifiedLines(lines=lines, operations=operations)


def _naive_findings(
    rule: custom_lint_metadata.Rule,
    rel_path: str,
    text: str,
    modified: custom_lint_vcs.ModifiedLines | None,
) -> Sequence[int | None]:
  op = modified.operations.get(rel_path) if modified is not None else None
  if not rule.applies_to(op):
    return []
  if rule.kind == custom_lint_metadata.CheckKind.CHECK_FILE_OPERATION:
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
  prohibited = rule.prohibited_re
  required = rule.required_re
  supp = rule.suppression_re
  out: list[int | None] = []
  if rule.mode == custom_lint_metadata.MatchMode.ENTIRE_FILE:
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
  if prohibited and rule.mode == custom_lint_metadata.MatchMode.MODIFIED_CHUNKS:
    runs: list[list[int]] = []
    for i in numbers:
      if runs and runs[-1][1] + 1 == i:
        runs[-1][1] = i
      else:
        runs.append([i, i])
    for first, last in runs:
      suffix = "\n" if (last < len(lines) or text.endswith("\n")) else ""
      chunk = "\n".join(lines[first - 1 : last]) + suffix
      match_lines = [
          min(last, first + chunk.count("\n", 0, m.start()))
          for m in prohibited.finditer(chunk)
      ]
      for i, seg_first in enumerate(match_lines):
        if supp:
          seg_last = next(
              (ln - 1 for ln in match_lines[i + 1 :] if ln > seg_first),
              last,
          )
          seg_suffix = (
              "\n" if (seg_last < len(lines) or text.endswith("\n")) else ""
          )
          seg_text = "\n".join(lines[seg_first - 1 : seg_last]) + seg_suffix
          if supp.search(seg_text):
            continue
        out.append(seg_first)
  elif prohibited:
    for i in numbers:
      line = lines[i - 1]
      if prohibited.search(line) and not (supp and supp.search(line)):
        out.append(i)
  if required and not any(required.search(line) for line in lines):
    out.append(None)
  return out


def _sorted_findings(
    findings: Iterable[int | None],
) -> Sequence[int | None]:
  return sorted(findings, key=lambda x: (x is not None, x or 0))


def _assert_agree(
    rules: Sequence[custom_lint_metadata.Rule],
    texts: Sequence[str],
    modified: custom_lint_vcs.ModifiedLines | None,
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
  canonical_build = _project_mapper().to_canonical("BUILD")
  modified = custom_lint_vcs.ModifiedLines({"BUILD": {1, 2}})
  violations = custom_lint.check_file_content(
      'py_test(\n\n    name = "x",\n',
      "BUILD",
      _rule("CheckPyTestUsage"),
      modified,
      canonical_path=canonical_build,
  )
  assert _lines(violations) == [1]
  single_line_modified = custom_lint_vcs.ModifiedLines({"BUILD": {1}})
  single_line_violations = custom_lint.check_file_content(
      'py_test(\n    name = "x",\n',
      "BUILD",
      _rule("CheckPyTestUsage"),
      single_line_modified,
      canonical_path=canonical_build,
  )
  assert _lines(single_line_violations) == [1]
  text = custom_lint.TextFile("x\n\ny\n")
  assert text.text[slice(*text.span(1, 2))] == "x\n\n"
  assert text.text[slice(*text.span(2, 3))] == "\ny\n"


@pytest.mark.parametrize(
    ("rule_id", "content"),
    [
        (
            "CheckPyTestUsage",
            'py_test(\n    # PY_TEST_OK=reason\n    name = "x",\n)\n',
        ),
        (
            "CheckPyLibraryUsage",
            'py_library(\n    # PY_LIBRARY_OK=reason\n    name = "x",\n)\n',
        ),
        (
            "CheckCcTargetUsage",
            'cc_library(\n    # ALLOW_CC_TARGETS=reason\n    name = "x",\n)\n',
        ),
    ],
)
def test_single_line_chunk_does_not_use_unchanged_next_line_suppression(
    rule_id: str, content: str
) -> None:
  rule = _rule(rule_id)
  canonical_path = _project_mapper().to_canonical("BUILD")
  single_line = custom_lint_vcs.ModifiedLines({"BUILD": {1}})
  violations = custom_lint.check_file_content(
      content,
      "BUILD",
      rule,
      single_line,
      canonical_path=canonical_path,
  )
  assert _lines(violations) == [1]

  both_lines = custom_lint_vcs.ModifiedLines({"BUILD": {1, 2}})
  assert not custom_lint.check_file_content(
      content,
      "BUILD",
      rule,
      both_lines,
      canonical_path=canonical_path,
  )


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
  rules = [
      dataclasses.replace(r, path_regex=r".*", path_exclusion_regex=None)
      for r in _project_rules().values()
  ]
  lines = list(_PROJECT_LINES)
  for rule in rules:
    ex = _RULE_EXAMPLES.get(rule.id)
    if ex:
      lines.extend(ex.bad.splitlines())
      if ex.good is not None:
        lines.extend(ex.good.splitlines())
  texts = _fixture_texts(lines, 400, seed=4)
  assert _assert_agree(rules, texts, None) > 400
  modified = _fixture_changes(texts, seed=5)
  _assert_agree(rules, texts, modified)


def test_is_line_local_non_local_constructs() -> None:
  assert not custom_lint_metadata.is_line_local(r"\Afoo")
  assert not custom_lint_metadata.is_line_local(r"foo\z")
  assert custom_lint_metadata.is_line_local(r"\\Afoo")
  assert custom_lint_metadata.is_line_local(r"foo\\z")
  for pat in (r"\Afoo", r"foo\z"):
    rule = custom_lint_metadata.Rule(
        id="TextAnchorInLineMode",
        path_regex=r".*\.txt",
        prohibited_regex=pat,
    )
    violations = custom_lint.check_file_content(
        "x\nfoo\ny\n", "a.txt", rule, canonical_path="a.txt"
    )
    assert _lines(violations) == [2]


def _expected_naive_findings(
    texts: Sequence[str],
    rules: Sequence[custom_lint_metadata.Rule],
    modified: custom_lint_vcs.ModifiedLines | None = None,
) -> list[tuple[str, str, int]]:
  findings: list[tuple[str, str, int]] = []
  for i, text in enumerate(texts):
    rel_path = f"f{i}.txt"
    if modified is not None and rel_path not in modified:
      continue
    for rule in rules:
      for line in _naive_findings(rule, rel_path, text, modified):
        findings.append((rule.id, rel_path, line or 0))
  findings.sort()
  return findings


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

  expected = _expected_naive_findings(
      texts, rules, None if check_all else modified
  )
  result = custom_lint.collect_violations(
      root,
      check_all=check_all,
      rules_file=str(rules_file),
      vcs=_FakeVcs(modified),
  )
  actual = sorted(
      (v.rule_id, v.file_path, v.line_number or 0) for v in result.violations
  )
  assert actual == expected


@dataclasses.dataclass(frozen=True)
class _ExecFixture:
  root: pathlib.Path
  rules_file: pathlib.Path
  expected: Sequence[tuple[str, str, int]]
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
  expected = tuple(_expected_naive_findings(texts, rules))
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
) -> Sequence[tuple[str, str, int]]:
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
  return tuple(
      sorted(
          (v.rule_id, v.file_path, v.line_number or 0)
          for v in result.violations
      )
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
  actual = tuple(
      sorted(
          (v.rule_id, v.file_path, v.line_number or 0)
          for v in result.violations
      )
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

_LAZY_RULES: Final[str] = r"""
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
) -> tuple[Sequence[str], int]:
  root = tmp_path.resolve()
  (root / "pyproject.toml").write_text("", encoding="utf-8")
  rules_file = root / "rules.METADATA"
  rules_file.write_text(_LAZY_RULES, encoding="utf-8")
  (root / "a.txt").write_text(content, encoding="utf-8")
  modified = custom_lint_vcs.ModifiedLines(
      lines={"a.txt": set(range(1, content.count("\n") + 2))},
      operations={"a.txt": custom_lint_metadata.FileOperation.EDIT},
  )
  fake_vcs = _FakeVcs(modified, message=message)
  result = custom_lint.collect_violations(
      root, check_all=check_all, rules_file=str(rules_file), vcs=fake_vcs
  )
  ids = sorted(v.rule_id for v in result.violations)
  return ids, fake_vcs.commit_message_calls


@pytest.mark.parametrize(
    ("content", "message", "check_all", "expected_ids", "expected_calls"),
    [
        pytest.param("x\n", "", False, [], 0, id="no_violations"),
        pytest.param(
            "bar\n", "", False, ["Untagged"], 0, id="untagged_rule_only"
        ),
        pytest.param(
            "foo\nbar\n",
            "",
            False,
            ["Tagged", "TaggedWithSuppression", "Untagged"],
            1,
            id="tagged_violation_empty_message",
        ),
        pytest.param(
            "foo\nbar\n",
            "Fix\n\nALLOW_FOO=reason\n",
            False,
            ["Untagged"],
            1,
            id="tagged_violation_suppressed_by_tag",
        ),
        pytest.param(
            "foo\nbar\n",
            "Public commit description\n\nGenerated from cl/123456789\n",
            False,
            ["Tagged", "TaggedWithSuppression", "Untagged"],
            1,
            id="copybara_footer_does_not_disable_rules",
        ),
        pytest.param(
            "foo\n",
            "ALLOW_FOO=reason\n",
            True,
            ["Tagged", "TaggedWithSuppression"],
            0,
            id="check_all_ignores_commit_tags",
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
  """Verifies `ci/` is not treated as a project root when walking up."""
  oss_rules = tmp_path / "ci" / custom_lint.DEFAULT_RULES_FILE
  oss_rules.parent.mkdir(parents=True)
  oss_rules.write_text("", encoding="utf-8")
  (tmp_path / "pyproject.toml").write_text(
      '[tool.custom_linter]\ncanonical_root = "torch_tpu/oss"\nrules_file ='
      ' "ci/custom_lint_rules.METADATA"\n',
      encoding="utf-8",
  )
  assert oss_rules.name == custom_lint.DEFAULT_RULES_FILE
  start = tmp_path / "ci" / "tools" / "custom_linter"
  start.mkdir(parents=True)

  assert custom_lint.find_project_root(start) == tmp_path.resolve()
  assert custom_lint.find_rules_file(tmp_path) == oss_rules
  mapper = custom_lint.default_path_mapper(tmp_path)
  assert mapper.canonical_root == "torch_tpu/oss"


def test_config_overrides_max_entire_file_bytes_and_excluded_path_regex(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
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
  assert (
      custom_lint.main([
          "--root",
          str(root),
          "--all",
          "--jobs",
          "1",
          "--rules",
          str(rules_file),
          "--config",
          str(bad_cfg),
      ])
      == 2
  )
  assert (
      "unknown [tool.custom_linter] key(s): unknown_option"
      in capsys.readouterr().err
  )

  (root / "pyproject.toml").write_text(
      '[tool.custom_linter]\nunknown_in_pyproject = "x"\n', encoding="utf-8"
  )
  assert (
      custom_lint.run_checks(
          root,
          check_all=True,
          jobs=1,
          rules_file=rules_file,
      )
      == 2
  )
  assert (
      "unknown [tool.custom_linter] key(s): unknown_in_pyproject"
      in capsys.readouterr().err
  )


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("[other]\nfoo = 1\n", id="other_table"),
        pytest.param(
            '[tool.custom_lint]\ncanonical_root = "//depot"\n',
            id="typo_tool_subtable",
        ),
        pytest.param(
            'canonical_root = "//depot"\n',
            id="top_level_without_section",
        ),
        pytest.param("", id="empty_file"),
    ],
)
def test_load_config_explicit_file_missing_section(
    tmp_path: pathlib.Path,
    content: str,
) -> None:
  cfg_file = tmp_path / "other.toml"
  cfg_file.write_text(content, encoding="utf-8")
  with pytest.raises(
      custom_lint.ConfigError, match=r"missing \[tool\.custom_linter\]"
  ):
    custom_lint.load_config(config_file=cfg_file)


def test_load_config_type_validations(tmp_path: pathlib.Path) -> None:
  bad_str = tmp_path / "bad_str.toml"
  bad_str.write_text(
      "[tool.custom_linter]\ncanonical_root = 123\n", encoding="utf-8"
  )
  with pytest.raises(
      custom_lint.ConfigError, match=r"canonical_root must be a string, got int"
  ):
    custom_lint.load_config(config_file=bad_str)

  bad_base_dir = tmp_path / "bad_base_dir.toml"
  bad_base_dir.write_text(
      "[tool.custom_linter]\nrules_base_dir = 123\n", encoding="utf-8"
  )
  with pytest.raises(
      custom_lint.ConfigError, match=r"rules_base_dir must be a string, got int"
  ):
    custom_lint.load_config(config_file=bad_base_dir)

  bad_int = tmp_path / "bad_int.toml"
  bad_int.write_text(
      '[tool.custom_linter]\nmax_entire_file_bytes = "abc"\n', encoding="utf-8"
  )
  with pytest.raises(
      custom_lint.ConfigError,
      match=r"max_entire_file_bytes must be a positive integer, got str",
  ):
    custom_lint.load_config(config_file=bad_int)

  bad_bool = tmp_path / "bad_bool.toml"
  bad_bool.write_text(
      "[tool.custom_linter]\nmax_entire_file_bytes = true\n", encoding="utf-8"
  )
  with pytest.raises(
      custom_lint.ConfigError,
      match=r"max_entire_file_bytes must be a positive integer, got bool",
  ):
    custom_lint.load_config(config_file=bad_bool)

  bad_maps = tmp_path / "bad_maps.toml"
  bad_maps.write_text(
      "[tool.custom_linter]\npath_maps = [1, 2]\n", encoding="utf-8"
  )
  with pytest.raises(
      custom_lint.ConfigError,
      match=r"path_maps must be a table of string pairs, got list",
  ):
    custom_lint.load_config(config_file=bad_maps)

  bad_map_val = tmp_path / "bad_map_val.toml"
  bad_map_val.write_text(
      "[tool.custom_linter.path_maps]\na = 123\n", encoding="utf-8"
  )
  with pytest.raises(
      custom_lint.ConfigError,
      match=r"path_maps entries must be string pairs",
  ):
    custom_lint.load_config(config_file=bad_map_val)


def test_root_discovery_with_custom_rules_file(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  root = tmp_path.resolve()
  (root / "pyproject.toml").write_text(
      '[tool.custom_linter]\nrules_file = "custom_rules.METADATA"\n',
      encoding="utf-8",
  )
  custom_rules = root / "custom_rules.METADATA"
  custom_rules.write_text(
      'presubmit { path_regexp: ".*\\\\.txt"\n'
      '  check_contents { base { id: "NoFoo" } prohibited_regexp: "foo" }\n'
      "}\n",
      encoding="utf-8",
  )
  sub = root / "subdir"
  sub.mkdir()
  (sub / "a.txt").write_text("foo\n", encoding="utf-8")

  assert custom_lint.find_project_root(sub) == root
  assert custom_lint.find_rules_file(root) == custom_rules

  monkeypatch.setenv("BUILD_WORKING_DIRECTORY", str(sub))
  assert custom_lint.main(["--all", "--jobs", "1", "--format", "json"]) == 1
  assert [v["file_path"] for v in json.loads(capsys.readouterr().out)] == [
      "subdir/a.txt"
  ]


def test_broken_pyproject_during_root_discovery_raises_config_error(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  root = tmp_path.resolve()
  (root / ".git").mkdir()
  (root / "pyproject.toml").write_text(
      "[tool.custom_linter]\nbad_key = 1\n",
      encoding="utf-8",
  )
  (root / custom_lint.DEFAULT_RULES_FILE).write_text("", encoding="utf-8")
  sub = root / "subdir"
  sub.mkdir()

  with pytest.raises(custom_lint.ConfigError, match="bad_key"):
    custom_lint.find_project_root(sub)

  # Also raises when checking parent pyproject.toml for a subdirectory rules
  # file rather than silently ignoring the broken parent config.
  (sub / custom_lint.DEFAULT_RULES_FILE).write_text("", encoding="utf-8")
  with pytest.raises(custom_lint.ConfigError, match="bad_key"):
    custom_lint.find_project_root(sub)

  monkeypatch.setenv("BUILD_WORKING_DIRECTORY", str(sub))
  assert custom_lint.main(["--all", "--jobs", "1"]) == 2
  assert "unknown [tool.custom_linter] key(s): bad_key" in (
      capsys.readouterr().err
  )


def test_deleted_file_operation_rules_fire_in_diff_mode(
    tmp_path: pathlib.Path,
) -> None:
  root = tmp_path.resolve()
  (root / "pyproject.toml").write_text("", encoding="utf-8")
  rules_file = root / "rules.METADATA"
  rules_file.write_text(
      textwrap.dedent("""\
          presubmit {
            path_regexp: ".*\\\\.lock$"
            check_file_operation {
              base { id: "NoLockDelete" }
              operation: DELETE
              failure_message: "Do not delete lockfiles."
            }
            check_contents {
              base { id: "ContentCheckOnLock" }
              prohibited_regexp: "bad"
            }
          }
          """),
      encoding="utf-8",
  )
  diff_text = textwrap.dedent("""\
      diff --git a/deps.lock b/deps.lock
      deleted file mode 100644
      --- a/deps.lock
      +++ /dev/null
      @@ -1 +0,0 @@
      -bad
      """)
  modified = custom_lint_vcs.parse_unified_diff(diff_text)
  assert modified.operations == {
      "deps.lock": custom_lint_metadata.FileOperation.DELETE
  }
  result = custom_lint.collect_violations(
      root, jobs=1, rules_file=str(rules_file), vcs=_FakeVcs(modified)
  )
  assert [
      (v.rule_id, v.file_path, v.line_number) for v in result.violations
  ] == [("NoLockDelete", "deps.lock", None)]


@pytest.fixture(name="bwd_repo")
def _bwd_repo_fixture(
    tmp_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
  root = (tmp_path / "repo").resolve()
  runfiles_cwd = (tmp_path / "runfiles_cwd").resolve()
  root.mkdir()
  runfiles_cwd.mkdir()
  (root / "pyproject.toml").write_text("", encoding="utf-8")
  (root / custom_lint.DEFAULT_RULES_FILE).write_text(
      'presubmit { path_regexp: "//depot/custom_root/.*\\\\.cc"\n'
      '  check_contents { base { id: "NoFoo" } prohibited_regexp: "foo" }\n'
      "}\n",
      encoding="utf-8",
  )
  sub = root / "pkg"
  sub.mkdir()
  (sub / "local.cc").write_text("foo\n", encoding="utf-8")
  (root / "a.cc").write_text("foo\n", encoding="utf-8")
  return root, sub, runfiles_cwd


def test_main_resolves_relative_files_against_build_working_directory(
    bwd_repo: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  root, sub, runfiles_cwd = bwd_repo
  (root / "pyproject.toml").write_text(
      '[tool.custom_linter]\ncanonical_root = "//depot/custom_root"\n',
      encoding="utf-8",
  )
  monkeypatch.chdir(runfiles_cwd)
  monkeypatch.setenv("BUILD_WORKING_DIRECTORY", str(sub))
  rc = custom_lint.main(["--jobs", "1", "--format", "json", "local.cc"])
  captured = capsys.readouterr()
  assert rc == 1
  assert [v["file_path"] for v in json.loads(captured.out)] == ["pkg/local.cc"]


def test_main_missing_explicit_file_exits_2(
    bwd_repo: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  root, sub, runfiles_cwd = bwd_repo
  (root / "pyproject.toml").write_text(
      '[tool.custom_linter]\ncanonical_root = "//depot/custom_root"\n',
      encoding="utf-8",
  )
  monkeypatch.chdir(runfiles_cwd)
  monkeypatch.setenv("BUILD_WORKING_DIRECTORY", str(sub))
  rc_missing = custom_lint.main(["--jobs", "1", "does_not_exist.cc"])
  captured_missing = capsys.readouterr()
  assert rc_missing == 2
  assert "ERROR: Path does not exist: does_not_exist.cc" in captured_missing.err


def test_main_resolves_relative_config_in_runfiles_cwd(
    bwd_repo: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  root, _, runfiles_cwd = bwd_repo
  runfiles_cfg = runfiles_cwd / "pkg" / "linter_config.toml"
  runfiles_cfg.parent.mkdir(parents=True)
  runfiles_cfg.write_text(
      '[tool.custom_linter]\ncanonical_root = "//depot/custom_root"\n',
      encoding="utf-8",
  )
  monkeypatch.chdir(runfiles_cwd)
  monkeypatch.setenv("BUILD_WORKING_DIRECTORY", str(root))

  rc_runfiles = custom_lint.main([
      "--config",
      "pkg/linter_config.toml",
      "--jobs",
      "1",
      "--format",
      "json",
      "a.cc",
  ])
  out_runfiles = json.loads(capsys.readouterr().out)
  assert rc_runfiles == 1
  assert len(out_runfiles) == 1
  assert out_runfiles[0]["rule_id"] == "NoFoo"


def test_main_resolves_relative_config_in_build_working_directory(
    bwd_repo: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  root, _, runfiles_cwd = bwd_repo
  runfiles_cfg = runfiles_cwd / "pkg" / "linter_config.toml"
  runfiles_cfg.parent.mkdir(parents=True)
  runfiles_cfg.write_text(
      '[tool.custom_linter]\ncanonical_root = "//depot/wrong_root"\n',
      encoding="utf-8",
  )
  local_cfg = root / "pkg" / "linter_config.toml"
  local_cfg.write_text(
      '[tool.custom_linter]\ncanonical_root = "//depot/custom_root"\n',
      encoding="utf-8",
  )
  monkeypatch.chdir(runfiles_cwd)
  monkeypatch.setenv("BUILD_WORKING_DIRECTORY", str(root))

  rc_local = custom_lint.main([
      "--config",
      "pkg/linter_config.toml",
      "--jobs",
      "1",
      "--format",
      "json",
      "a.cc",
  ])
  out_local = json.loads(capsys.readouterr().out)
  assert rc_local == 1
  assert len(out_local) == 1
  assert out_local[0]["rule_id"] == "NoFoo"


def test_main_ignores_syspath_for_relative_config(
    bwd_repo: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  root, _, _ = bwd_repo
  for subdir in ("site_packages", "custom_lint.runfiles/_main"):
    syspath_dir = (tmp_path / subdir).resolve()
    cfg = syspath_dir / "pkg" / "linter_config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        '[tool.custom_linter]\ncanonical_root = "//depot/custom_root"\n',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(syspath_dir))
  monkeypatch.chdir(root)
  monkeypatch.delenv("BUILD_WORKING_DIRECTORY", raising=False)

  rc = custom_lint.main([
      "--config",
      "pkg/linter_config.toml",
      "--jobs",
      "1",
      "a.cc",
  ])
  assert rc == 2
  assert "config file not found" in capsys.readouterr().err


def test_main_root_outside_cwd_prefers_root_relative_file(
    bwd_repo: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  root, _, _ = bwd_repo
  (root / "pyproject.toml").write_text(
      '[tool.custom_linter]\ncanonical_root = "//depot/custom_root"\n',
      encoding="utf-8",
  )
  outside_cwd = (tmp_path / "outside_cwd").resolve()
  outside_cwd.mkdir()
  (outside_cwd / "a.cc").write_text("clean\n", encoding="utf-8")
  monkeypatch.chdir(outside_cwd)
  monkeypatch.delenv("BUILD_WORKING_DIRECTORY", raising=False)

  rc = custom_lint.main([
      "--root",
      str(root),
      "--jobs",
      "1",
      "--format",
      "json",
      "a.cc",
  ])
  out = json.loads(capsys.readouterr().out)
  assert rc == 1
  assert [v["file_path"] for v in out] == ["a.cc"]


def test_format_summary_report_counts_notices_separately() -> None:
  violations = [
      custom_lint.Violation(
          rule_id="ErrRule",
          file_path="a.cc",
          line_number=1,
          line_content="x",
          message="err",
          severity=custom_lint_metadata.Severity.ERROR,
      ),
      custom_lint.Violation(
          rule_id="WarnRule",
          file_path="a.cc",
          line_number=2,
          line_content="y",
          message="warn",
          severity=custom_lint_metadata.Severity.WARNING,
      ),
      custom_lint.Violation(
          rule_id="NoticeRule",
          file_path="a.cc",
          line_number=3,
          line_content="z",
          message="note",
          severity=custom_lint_metadata.Severity.NOTICE,
      ),
  ]
  report = custom_lint.format_summary_report(violations, files_checked=1)
  assert " SUMMARY: 1 errors, 1 warnings, 1 notices\n" in report


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
