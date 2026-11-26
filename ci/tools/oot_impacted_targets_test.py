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

"""Unit tests for ci/tools/oot_impacted_targets.py."""

import contextlib
import io
import json
import os
import pathlib
import subprocess
import tempfile
import textwrap
import unittest

from ci.tools import oot_impacted_targets

_BUILD = textwrap.dedent("""\
    torch_tpu_torch_test(
        name = "autograd_test",
        shard_count = 4,
        torch_test_module = "test_autograd",
    )
    torch_tpu_torch_test(
        name = "dynamo_after_aot_test",
        torch_test_module = "dynamo/test_after_aot",
    )
    torch_tpu_torch_test(
        name = "dynamo_modules_test",
        torch_test_module = "dynamo/test_modules",
    )
    torch_tpu_torch_test(
        name = "inductor_utils_test",
        torch_test_module = "inductor/test_utils",
    )
    torch_tpu_torch_test(
        name = "disabled_test",
        notap = "b/1",
        torch_test_module = "test_disabled",
    )
    torch_tpu_torch_test(
        name = "notest_test",
        tags = ["notest"],
        torch_test_module = "test_notest",
    )
""")

_BZL = textwrap.dedent("""\
    _EXTRA_MODULE_GLOBS = {
        "dynamo/test_modules": ["test/dynamo/test_functions.py"],
        "test_autograd": ["test/autograd/**"],
    }
    _EXTRA_MODULE_SUPPORT_TARGETS = {
        "dynamo/test_after_aot": [":inductor_support"],
        "test_autograd": [":expect_support"],
    }
""")


def _git(repo: pathlib.Path, *args: str) -> str:
  env = {
      **os.environ,
      "GIT_CONFIG_GLOBAL": os.devnull,
      "GIT_CONFIG_SYSTEM": os.devnull,
  }
  result = subprocess.run(
      ["git", "-C", str(repo), *args],
      capture_output=True,
      text=True,
      encoding="utf-8",
      env=env,
      check=True,
  )
  return result.stdout.strip()


class ParseTest(unittest.TestCase):

  def test_targets(self):
    targets = oot_impacted_targets.parse_torch_test_targets(_BUILD)
    self.assertEqual(len(targets), 6)
    self.assertEqual(targets["//tests/torch:autograd_test"].shard_count, 4)
    self.assertFalse(targets["//tests/torch:disabled_test"].enabled)
    self.assertFalse(targets["//tests/torch:notest_test"].enabled)
    self.assertTrue(targets["//tests/torch:inductor_utils_test"].enabled)

  def test_empty_build_raises(self):
    with self.assertRaisesRegex(ValueError, "No torch_tpu_torch_test"):
      oot_impacted_targets.parse_torch_test_targets("# empty BUILD file\n")

  def test_nobuild_tag_disables_target(self):
    build = textwrap.dedent("""\
        torch_tpu_torch_test(
            name = "nobuild_test",
            tags = ["nobuild"],
            torch_test_module = "test_nobuild",
        )
    """)
    targets = oot_impacted_targets.parse_torch_test_targets(build)
    self.assertFalse(targets["//tests/torch:nobuild_test"].enabled)
    tables = oot_impacted_targets.parse_deps_tables(_BZL)
    self.assertEqual(
        oot_impacted_targets.compute_impacted(
            ["test/test_nobuild.py"], targets, tables
        ),
        {},
    )

  def test_none_tags_and_bool_shard_count(self):
    build = textwrap.dedent("""\
        torch_tpu_torch_test(
            name = "none_tags_test",
            shard_count = True,
            tags = None,
            notap = None,
            nobuild = None,
            notest_oss = False,
            nobuild_oss = False,
            torch_test_module = "test_none_tags",
        )
    """)
    targets = oot_impacted_targets.parse_torch_test_targets(build)
    target = targets["//tests/torch:none_tags_test"]
    self.assertTrue(target.enabled)
    self.assertEqual(target.shard_count, 1)

  def test_oss_disable_kwargs_disable_targets(self):
    build = textwrap.dedent("""\
        torch_tpu_torch_test(
            name = "nobuild_kwarg_test",
            nobuild = "b/2",
            torch_test_module = "test_nobuild_kwarg",
        )
        torch_tpu_torch_test(
            name = "notest_oss_test",
            notest_oss = True,
            torch_test_module = "test_notest_oss",
        )
        torch_tpu_torch_test(
            name = "nobuild_oss_test",
            nobuild_oss = True,
            torch_test_module = "test_nobuild_oss",
        )
    """)
    targets = oot_impacted_targets.parse_torch_test_targets(build)
    for name in ("nobuild_kwarg_test", "notest_oss_test", "nobuild_oss_test"):
      with self.subTest(name=name):
        self.assertFalse(targets[f"//tests/torch:{name}"].enabled)
    tables = oot_impacted_targets.parse_deps_tables(_BZL)
    self.assertEqual(
        oot_impacted_targets.compute_impacted(
            [
                "test/test_nobuild_kwarg.py",
                "test/test_notest_oss.py",
                "test/test_nobuild_oss.py",
            ],
            targets,
            tables,
        ),
        {},
    )

  def test_non_literal_rule_args_raise(self):
    bad_builds = {
        "non_literal_name": textwrap.dedent("""\
            torch_tpu_torch_test(
                name = TEST_NAME,
                torch_test_module = "test_foo",
            )
        """),
        "non_literal_module": textwrap.dedent("""\
            torch_tpu_torch_test(
                name = "foo_test",
                torch_test_module = "test_" + "foo",
            )
        """),
        "kwargs_unpacking": textwrap.dedent("""\
            torch_tpu_torch_test(
                name = "foo_test",
                torch_test_module = "test_foo",
                **EXTRA_KWARGS
            )
        """),
        "non_literal_tags_expr": textwrap.dedent("""\
            torch_tpu_torch_test(
                name = "foo_test",
                tags = ["foo"] + ["notest"],
                torch_test_module = "test_foo",
            )
        """),
        "non_literal_tags_element": textwrap.dedent("""\
            torch_tpu_torch_test(
                name = "foo_test",
                tags = [TAG_VAR],
                torch_test_module = "test_foo",
            )
        """),
        "empty_notap": textwrap.dedent("""\
            torch_tpu_torch_test(
                name = "foo_test",
                notap = "",
                torch_test_module = "test_foo",
            )
        """),
        "non_literal_nobuild": textwrap.dedent("""\
            torch_tpu_torch_test(
                name = "foo_test",
                nobuild = REASON,
                torch_test_module = "test_foo",
            )
        """),
        "non_bool_notest_oss": textwrap.dedent("""\
            torch_tpu_torch_test(
                name = "foo_test",
                notest_oss = "yes",
                torch_test_module = "test_foo",
            )
        """),
        "non_literal_nobuild_oss": textwrap.dedent("""\
            torch_tpu_torch_test(
                name = "foo_test",
                nobuild_oss = FLAG,
                torch_test_module = "test_foo",
            )
        """),
    }
    for case_name, build in bad_builds.items():
      with self.subTest(case=case_name):
        with self.assertRaises(ValueError):
          oot_impacted_targets.parse_torch_test_targets(build)

  def test_tables(self):
    tables = oot_impacted_targets.parse_deps_tables(_BZL)
    self.assertEqual(
        tables.extra_module_globs["test_autograd"], ["test/autograd/**"]
    )
    self.assertEqual(
        tables.extra_module_support_targets["dynamo/test_after_aot"],
        [":inductor_support"],
    )

  def test_missing_table_raises(self):
    with self.assertRaisesRegex(ValueError, "missing"):
      oot_impacted_targets.parse_deps_tables("_EXTRA_MODULE_GLOBS = {}\n")

  def test_validate_deps_tables_rejects_stale_key(self):
    targets = oot_impacted_targets.parse_torch_test_targets(_BUILD)
    stale_tables = oot_impacted_targets.DepsTables(
        extra_module_globs={"test_nonexistent": ["test/foo.py"]},
        extra_module_support_targets={},
    )
    with self.assertRaisesRegex(ValueError, "stale module entries"):
      oot_impacted_targets.validate_deps_tables(targets, stale_tables)

  def test_real_bzl_and_build_in_sync(self):
    # Resolves both OSS layout (`oss/ci/tools/...`) and Google3 layout
    # (`torch_tpu/oss/ci/tools/...` with `torch_tpu/tests/torch/BUILD`).
    oss_root = pathlib.Path(__file__).resolve().parents[2]
    bzl_path = oss_root / "bazel" / "pytorch_test_repo.bzl"
    build_path = oss_root / "tests" / "torch" / "BUILD"
    if not build_path.is_file():
      build_path = oss_root.parent / "tests" / "torch" / "BUILD"
    self.assertTrue(bzl_path.is_file(), f"Missing {bzl_path}")
    self.assertTrue(build_path.is_file(), f"Missing {build_path}")

    targets = oot_impacted_targets.parse_torch_test_targets(
        build_path.read_text(encoding="utf-8")
    )
    tables = oot_impacted_targets.parse_deps_tables(
        bzl_path.read_text(encoding="utf-8")
    )
    oot_impacted_targets.validate_deps_tables(targets, tables)
    self.assertIn("test_autograd", tables.extra_module_globs)


class GlobTest(unittest.TestCase):

  def test_star_does_not_cross_slash(self):
    self.assertTrue(
        oot_impacted_targets.glob_matches(
            "test/dynamo/test_*.py", "test/dynamo/test_x.py"
        )
    )
    self.assertFalse(
        oot_impacted_targets.glob_matches(
            "test/dynamo/test_*.py", "test/dynamo/a/test_x.py"
        )
    )

  def test_double_star(self):
    self.assertTrue(
        oot_impacted_targets.glob_matches("test/ao/**", "test/ao/a/b/c.py")
    )
    self.assertTrue(
        oot_impacted_targets.glob_matches(
            "test/**/__pycache__/**", "test/x/__pycache__/a.pyc"
        )
    )
    self.assertTrue(
        oot_impacted_targets.glob_matches(
            "test/inductor/**/test_*.py", "test/inductor/test_x.py"
        )
    )
    self.assertTrue(
        oot_impacted_targets.glob_matches(
            "test/inductor/**/test_*.py", "test/inductor/sub/test_x.py"
        )
    )
    self.assertTrue(
        oot_impacted_targets.glob_matches(
            "test/inductor/**/.*", "test/inductor/.hidden"
        )
    )
    self.assertTrue(
        oot_impacted_targets.glob_matches(
            "test/inductor/**/.*", "test/inductor/a/.hidden"
        )
    )
    self.assertFalse(
        oot_impacted_targets.glob_matches(
            "test/inductor/**/.*", "test/inductor/a.py"
        )
    )


class ImpactTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.targets = oot_impacted_targets.parse_torch_test_targets(_BUILD)
    self.tables = oot_impacted_targets.parse_deps_tables(_BZL)

  def impacted(self, *paths: str) -> list[str]:
    return sorted(
        oot_impacted_targets.compute_impacted(paths, self.targets, self.tables)
    )

  def test_own_module(self):
    self.assertEqual(
        self.impacted("test/inductor/test_utils.py"),
        ["//tests/torch:inductor_utils_test"],
    )

  def test_unrelated_test_file_in_same_subdir_not_selected(self):
    # test_*.py siblings are excluded from <subdir>_support.
    self.assertEqual(self.impacted("test/inductor/test_other.py"), [])

  def test_subdir_support_helper_selects_all_subdir_modules(self):
    self.assertEqual(
        self.impacted("test/dynamo/utils.py"),
        [
            "//tests/torch:dynamo_after_aot_test",
            "//tests/torch:dynamo_modules_test",
        ],
    )

  def test_cross_subsystem_support(self):
    # test_after_aot pulls :inductor_support.
    self.assertEqual(
        self.impacted("test/inductor/helpers/x.py"),
        [
            "//tests/torch:dynamo_after_aot_test",
            "//tests/torch:inductor_utils_test",
        ],
    )

  def test_extra_module_glob(self):
    self.assertEqual(
        self.impacted("test/dynamo/test_functions.py"),
        ["//tests/torch:dynamo_modules_test"],
    )
    self.assertEqual(
        self.impacted("test/autograd/deep/x.py"),
        ["//tests/torch:autograd_test"],
    )

  def test_expect_support(self):
    self.assertEqual(
        self.impacted("test/expect/Foo.expect"),
        ["//tests/torch:autograd_test"],
    )

  def test_exclusions(self):
    self.assertEqual(self.impacted("test/inductor/.gitignore"), [])
    self.assertEqual(self.impacted("test/autograd/.hidden"), [])
    self.assertEqual(self.impacted("test/inductor/__pycache__/a.pyc"), [])
    self.assertEqual(self.impacted("test/inductor/BUILD.bazel"), [])

  def test_disabled_targets_never_selected(self):
    self.assertEqual(
        self.impacted("test/test_disabled.py", "test/test_notest.py"), []
    )

  def test_outside_test_dir_ignored(self):
    self.assertEqual(self.impacted("torch/nn/functional.py"), [])

  def test_reports_hits(self):
    impacted = oot_impacted_targets.compute_impacted(
        ["test/dynamo/utils.py", "test/dynamo/test_modules.py"],
        self.targets,
        self.tables,
    )
    self.assertEqual(
        impacted["//tests/torch:dynamo_modules_test"],
        ("test/dynamo/test_modules.py", "test/dynamo/utils.py"),
    )

  def test_module_inputs_rejects_invalid_support_target(self):
    for bad_label in (":bad_target", "//other:foo_support", "foo_support"):
      with self.subTest(bad_label=bad_label):
        bad_tables = oot_impacted_targets.DepsTables(
            extra_module_globs={},
            extra_module_support_targets={"test_autograd": [bad_label]},
        )
        with self.assertRaisesRegex(ValueError, "unsupported support target"):
          oot_impacted_targets.module_inputs("test_autograd", bad_tables)


class GitAndCliTest(unittest.TestCase):

  def _init_repo(self, repo: pathlib.Path) -> tuple[str, str]:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "test" / "inductor").mkdir(parents=True)
    (repo / "test" / "inductor" / "test_utils.py").write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    base_sha = _git(repo, "rev-parse", "HEAD")

    (repo / "test" / "inductor" / "test_utils.py").write_text("x = 2\n")
    (repo / "test" / "expect").mkdir(parents=True)
    (repo / "test" / "expect" / "café space.expect").write_text("ok\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "head")
    head_sha = _git(repo, "rev-parse", "HEAD")
    return base_sha, head_sha

  def test_changed_test_paths_handles_unicode_and_spaces(self):
    with tempfile.TemporaryDirectory() as tmp:
      repo = pathlib.Path(tmp) / "pytorch"
      base_sha, head_sha = self._init_repo(repo)
      changed = oot_impacted_targets.changed_test_paths(
          repo, base_sha, head_sha
      )
      self.assertEqual(
          changed,
          ("test/expect/café space.expect", "test/inductor/test_utils.py"),
      )

  def test_cli_end_to_end(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      workspace_dir = root / "ws"
      (workspace_dir / "tests" / "torch").mkdir(parents=True)
      (workspace_dir / "bazel").mkdir(parents=True)
      (workspace_dir / "tests" / "torch" / "BUILD").write_text(
          _BUILD, encoding="utf-8"
      )
      (workspace_dir / "bazel" / "pytorch_test_repo.bzl").write_text(
          _BZL, encoding="utf-8"
      )
      repo = root / "pytorch"
      base_sha, head_sha = self._init_repo(repo)

      targets_out = root / "impacted.txt"
      github_out = root / "github_output.txt"
      markdown_out = root / "summary.md"
      markdown_out.write_text("# Prior summary\n", encoding="utf-8")
      json_out = root / "impact.json"
      return_code = oot_impacted_targets.main([
          "--workspace-dir",
          str(workspace_dir),
          "--pytorch-dir",
          str(repo),
          "--base-sha",
          base_sha,
          "--head-sha",
          head_sha,
          "--impacted-targets-output",
          str(targets_out),
          "--github-output",
          str(github_out),
          "--markdown-output",
          str(markdown_out),
          "--json-output",
          str(json_out),
      ])
      self.assertEqual(return_code, 0)
      self.assertEqual(
          targets_out.read_text(encoding="utf-8").splitlines(),
          [
              "//tests/torch:autograd_test",
              "//tests/torch:inductor_utils_test",
          ],
      )
      github_output_text = github_out.read_text(encoding="utf-8")
      self.assertIn("has_impacted_targets=true\n", github_output_text)
      self.assertIn("impacted_count=2\n", github_output_text)
      self.assertIn("selection_scope=test_files_only\n", github_output_text)
      payload = json.loads(json_out.read_text(encoding="utf-8"))
      self.assertEqual(payload["selection_scope"], "test_files_only")
      self.assertEqual(
          sorted(payload["impacted_targets"]),
          [
              "//tests/torch:autograd_test",
              "//tests/torch:inductor_utils_test",
          ],
      )
      markdown_text = markdown_out.read_text(encoding="utf-8")
      self.assertIn("# Prior summary\n", markdown_text)
      self.assertIn("test_files_only", markdown_text)

  def test_ensure_commit_fetches_missing_commit_from_origin(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      origin = root / "origin"
      base_sha, head_sha = self._init_repo(origin)
      _git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")
      _git(origin, "config", "uploadpack.allowFilter", "true")

      consumer = root / "consumer"
      consumer.mkdir()
      _git(consumer, "init", "-q")
      _git(consumer, "remote", "add", "origin", str(origin))

      oot_impacted_targets.ensure_commit(consumer, base_sha)
      oot_impacted_targets.ensure_commit(consumer, head_sha)
      changed = oot_impacted_targets.changed_test_paths(
          consumer, base_sha, head_sha
      )
      self.assertEqual(
          changed,
          ("test/expect/café space.expect", "test/inductor/test_utils.py"),
      )

  def test_rejects_option_injection_in_shas(self):
    with tempfile.TemporaryDirectory() as tmp:
      repo = pathlib.Path(tmp) / "pytorch"
      self._init_repo(repo)
      for bad_sha in (
          "--upload-pack=evil",
          "--output=/tmp/pwn",
          "",
          "nothex!",
          "abcdef1",
      ):
        with self.subTest(bad_sha=bad_sha):
          with self.assertRaisesRegex(ValueError, "Invalid commit SHA"):
            oot_impacted_targets.ensure_commit(repo, bad_sha)
          with self.assertRaisesRegex(ValueError, "Invalid commit SHA"):
            oot_impacted_targets.changed_test_paths(repo, bad_sha, "a" * 40)

  def test_render_markdown_zero_impact_and_truncation(self):
    targets = oot_impacted_targets.parse_torch_test_targets(_BUILD)
    zero_markdown = oot_impacted_targets.render_markdown(
        base_sha="a" * 40,
        head_sha="b" * 40,
        changed=(),
        impacted={},
        targets=targets,
    )
    self.assertIn(
        "No enabled `//tests/torch:*` TPU test targets", zero_markdown
    )

    hits = tuple(f"test/expect/file_{index}.expect" for index in range(7))
    truncated_markdown = oot_impacted_targets.render_markdown(
        base_sha="a" * 40,
        head_sha="b" * 40,
        changed=hits,
        impacted={"//tests/torch:autograd_test": hits},
        targets=targets,
    )
    self.assertIn("… 2 more", truncated_markdown)
    self.assertIn("file_4.expect", truncated_markdown)
    self.assertNotIn("file_5.expect", truncated_markdown)

  def test_cli_zero_impact_and_omitted_outputs(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      workspace_dir = root / "ws"
      (workspace_dir / "tests" / "torch").mkdir(parents=True)
      (workspace_dir / "bazel").mkdir(parents=True)
      (workspace_dir / "tests" / "torch" / "BUILD").write_text(
          _BUILD, encoding="utf-8"
      )
      (workspace_dir / "bazel" / "pytorch_test_repo.bzl").write_text(
          _BZL, encoding="utf-8"
      )
      repo = root / "pytorch"
      base_sha, _ = self._init_repo(repo)
      github_out = root / "nested" / "dir" / "github_output.txt"

      stdout_buf = io.StringIO()
      with contextlib.redirect_stdout(stdout_buf):
        return_code = oot_impacted_targets.main([
            "--workspace-dir",
            str(workspace_dir),
            "--pytorch-dir",
            str(repo),
            "--base-sha",
            base_sha,
            "--head-sha",
            base_sha,
            "--markdown-output=",
            "--github-output",
            str(github_out),
        ])
      self.assertEqual(return_code, 0)
      github_output_text = github_out.read_text(encoding="utf-8")
      self.assertIn("has_impacted_targets=false\n", github_output_text)
      self.assertIn("impacted_count=0\n", github_output_text)
      self.assertIn("selection_scope=test_files_only\n", github_output_text)
      self.assertIn(
          "No enabled `//tests/torch:*` TPU test targets were impacted.",
          stdout_buf.getvalue(),
      )

  def test_ensure_commit_raises_when_missing_from_origin(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      origin = root / "origin"
      self._init_repo(origin)
      _git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")
      _git(origin, "config", "uploadpack.allowFilter", "true")

      consumer = root / "consumer"
      consumer.mkdir()
      _git(consumer, "init", "-q")
      _git(consumer, "remote", "add", "origin", str(origin))

      with self.assertRaisesRegex(RuntimeError, "git fetch"):
        oot_impacted_targets.ensure_commit(consumer, "f" * 40)


if __name__ == "__main__":
  unittest.main()
