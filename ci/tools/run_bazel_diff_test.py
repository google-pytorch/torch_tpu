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

"""Unit tests for ci/tools/run_bazel_diff.py."""

import os
import pathlib
import shutil
import tempfile
import textwrap
import unittest
from unittest import mock

from ci.tools import run_bazel_diff

# A `bazel query 'tests(...)' --output=xml` result with one CPU-only test and
# one TPU v5 test, so the two can be told apart by --test_tag_filters.
_QUERY_XML = """\
<query version="2">
  <rule class="py_test" location="/w/tests/BUILD:1:1" name="//tests:cpu_test">
    <list name="tags"/>
  </rule>
  <rule class="py_test" location="/w/tests/BUILD:9:1" name="//tests:v5_test">
    <list name="tags">
      <string value="requires-tpu"/>
      <string value="requires-tpu-v5lite"/>
      <string value="presubmit-v5"/>
    </list>
  </rule>
</query>
"""

_NOPRESUBMIT_QUERY_XML = """\
<query version="2">
  <rule class="py_test" location="/w/tests/BUILD:1:1" name="//tests:nopresubmit_test">
    <list name="tags">
      <string value="nopresubmit"/>
    </list>
  </rule>
</query>
"""

_V6_QUERY_XML = """\
<query version="2">
  <rule class="py_test" location="/w/tests/BUILD:1:1" name="//tests:v6_test">
    <list name="tags">
      <string value="requires-tpu"/>
      <string value="presubmit-v6"/>
    </list>
  </rule>
</query>
"""

# The --test_tag_filters lines from the repository .bazelrc, including both
# active presubmit configs and non-presubmit / nightly / inactive configs.
_BAZELRC = """\
common:ci_cpu --config=ci
common:ci_cpu --test_tag_filters=-nobuild,-notest,-requires-gpu,-requires-tpu,-cuda-build-test
common:ci_cpu_presubmit --config=ci_cpu
common:ci_cpu_presubmit --test_tag_filters=-nobuild,-notest,-requires-gpu,-requires-tpu,-nopresubmit,-cuda-build-test
common:ci_cpu_nightly --config=ci_cpu
common:ci_cpu_nightly --test_tag_filters=-nobuild,-notest,-requires-gpu,-requires-tpu,-nonightly,-cuda-build-test
common:ci_tpu_nightly --config=ci_tpu_base
common:ci_tpu_nightly --test_tag_filters=requires-tpu,-notest,-nobuild,-nonightly
common:ci_tpu_v5_presubmit --config=ci_tpu_v5
common:ci_tpu_v5_presubmit --test_tag_filters=-notest,-nobuild,-nopresubmit,-fails-on-tpu-v5,presubmit-v5
common:ci_tpu_v6_presubmit --config=ci_tpu_v6
common:ci_tpu_v6_presubmit --test_tag_filters=-notest,-nobuild,-nopresubmit,-fails-on-tpu-v6,presubmit-v6
common:ci_tpu_v7_presubmit --config=ci_tpu_v7
common:ci_tpu_v7_presubmit --test_tag_filters=-notest,-nobuild,-nopresubmit,-fails-on-tpu-v7,presubmit-v7
"""


class RunBazelDiffTest(unittest.TestCase):
  """Tests argument parsing, file classification, and ops-test filtering."""

  def test_parse_args_named_flags(self):
    opts = run_bazel_diff.parse_args([
        "--workspace-dir=/workspace/repo",
        "--base-sha=abc1234",
        "--current-sha=def5678",
        "--bazel-config=ci_cpu",
        (
            "--extra-bazel-flags=--repo_env=HERMETIC_PYTHON_VERSION=3.11"
            " --define FOO=1"
        ),
        "--test-all=true",
        (
            "--workflow-ref=google-pytorch/torch_tpu/.github/workflows/presubmit.yml@refs/pull/1/merge"
        ),
        "--candidate-configs=ci_cpu,ci_tpu_v5",
        "--workflow-file=.github/workflows/presubmit.yml",
    ])
    expected = {
        "workspace_dir": pathlib.Path("/workspace/repo"),
        "base_sha": "abc1234",
        "current_sha": "def5678",
        "bazel_config": "ci_cpu",
        "extra_bazel_flags": (
            "--repo_env=HERMETIC_PYTHON_VERSION=3.11 --define FOO=1"
        ),
        "test_all": True,
        "workflow_ref": (
            "google-pytorch/torch_tpu/.github/workflows/presubmit.yml@refs/pull/1/merge"
        ),
        "candidate_configs": ("ci_cpu", "ci_tpu_v5"),
        "workflow_file": ".github/workflows/presubmit.yml",
    }
    for field, expected_val in expected.items():
      with self.subTest(field=field):
        self.assertEqual(getattr(opts, field), expected_val)

  def test_parse_args_defaults(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      opts = run_bazel_diff.parse_args([])
    expected = {
        "base_sha": "",
        "current_sha": "HEAD",
        "bazel_config": "",
        "extra_bazel_flags": "",
        "test_all": False,
        "workflow_ref": "",
        "candidate_configs": (),
        "workflow_file": "",
    }
    for field, expected_val in expected.items():
      with self.subTest(field=field):
        self.assertEqual(getattr(opts, field), expected_val)

  def test_is_global_bazel_config_modified(self):
    cases = [
        (["csrc/ops/add.cc", "MODULE.bazel"], True),
        ([".bazelrc"], True),
        (["csrc/ops/BUILD", "src/torch_tpu/__init__.py"], False),
        # Dependency locks are pip.parse inputs, so they invalidate everything.
        # Jobs that repin a lock in their checkout also rely on this: matching
        # here returns before the `git checkout` calls, which would otherwise
        # abort on the dirty lock.
        (["requirements/requirements_3_14.txt"], True),
        (["requirements/lock_environments.sh"], True),
        # Near-miss: the patterns match whole paths.
        (["tools/requirements_helper.py"], False),
    ]
    for files, expected in cases:
      with self.subTest(files=files):
        self.assertEqual(
            run_bazel_diff.is_global_bazel_config_modified(files), expected
        )

  def test_workflow_path_from_ref(self):
    cases = [
        (
            (
                "google-pytorch/torch_tpu/.github/workflows/presubmit.yml@refs/"
                "pull/4194/merge"
            ),
            ".github/workflows/presubmit.yml",
        ),
        # A ref is not guaranteed to carry the `@<git-ref>` suffix.
        (
            "google-pytorch/torch_tpu/.github/workflows/nightly_tpu.yml",
            ".github/workflows/nightly_tpu.yml",
        ),
        # Outside GitHub Actions the variable is unset.
        ("", ""),
        ("not-a-workflow-ref", ""),
    ]
    for ref, expected in cases:
      with self.subTest(ref=ref):
        self.assertEqual(run_bazel_diff.workflow_path_from_ref(ref), expected)

  def test_is_ci_definition_modified(self):
    presubmit_ref = (
        "google-pytorch/torch_tpu/.github/workflows/presubmit.yml@refs/heads/"
        "main"
    )
    cases = [
        # The job's own workflow file.
        ([".github/workflows/presubmit.yml"], presubmit_ref, True),
        # Shared CI infrastructure affects every job.
        ([".github/actions/bazel-diff/action.yml"], presubmit_ref, True),
        ([".github/workflows/presubmit_runner.yml"], presubmit_ref, True),
        # A different workflow says nothing about this job.
        ([".github/workflows/nightly_tpu.yml"], presubmit_ref, False),
        # Without a ref, only the shared paths can match.
        ([".github/workflows/presubmit.yml"], "", False),
        ([".github/actions/bazel-diff/action.yml"], "", True),
        # Ordinary source changes are left to bazel-diff.
        (["csrc/ops/add.cc"], presubmit_ref, False),
    ]
    for files, ref, expected in cases:
      with self.subTest(files=files, ref=ref):
        self.assertEqual(
            run_bazel_diff.is_ci_definition_modified(files, ref), expected
        )

  def test_are_all_files_doc_or_metadata(self):
    cases = [
        (
            [
                "README.md",
                "docs/guide.md",
                "csrc/OWNERS",
                "OWNERS",
                "LICENSE",
                ".clang-format",
                ".gitignore",
                ".vscode/settings.json",
                "assets/logo.png",
            ],
            True,
        ),
        (["README.md", "csrc/ops/add.cc"], False),
        # The patterns match whole paths, so these near-misses are code.
        (["csrc/ops/markdown_writer.cc"], False),
        (["tools/docs_generator.py"], False),
        (["src/torch_tpu/LICENSE_checker.py"], False),
    ]
    for files, expected in cases:
      with self.subTest(files=files):
        self.assertEqual(
            run_bazel_diff.are_all_files_doc_or_metadata(files), expected
        )

  def test_read_targets_file(self):
    with tempfile.TemporaryDirectory() as td:
      with self.subTest("valid_file"):
        path = pathlib.Path(td) / "targets.txt"
        path.write_text(
            "//tests:foo_test\n\n  //tests:bar_test  \n", encoding="utf-8"
        )
        self.assertEqual(
            list(run_bazel_diff.read_targets_file(path)),
            ["//tests:foo_test", "//tests:bar_test"],
        )
      with self.subTest("missing_file"):
        missing = pathlib.Path(td) / "does_not_exist.txt"
        with self.assertRaisesRegex(FileNotFoundError, "does_not_exist.txt"):
          run_bazel_diff.read_targets_file(missing)

  def test_run_bazel_diff_invokes_java_jar(self):
    with mock.patch("subprocess.run") as mock_run:
      run_bazel_diff.run_bazel_diff(
          "generate-hashes",
          "-w",
          pathlib.Path("/workspace"),
          pathlib.Path("/workspace/hashes.json"),
      )
      mock_run.assert_called_once_with(
          [
              "java",
              "-jar",
              "/tmp/bazel-diff.jar",
              "generate-hashes",
              "-w",
              "/workspace",
              "/workspace/hashes.json",
          ],
          check=True,
      )

  def test_run_bazel_test_config_flag(self):
    with mock.patch("subprocess.run") as mock_run:
      mock_run.return_value.returncode = 0
      with self.subTest("with_config"):
        run_bazel_diff._run_bazel_test("ci_cpu", "--keep_going", ["//..."])
        mock_run.assert_called_with(
            ["bash", "-c", "bazel test --config=ci_cpu --keep_going //..."],
            check=False,
        )
      with self.subTest("empty_config"):
        run_bazel_diff._run_bazel_test("", "--keep_going", ["//..."])
        mock_run.assert_called_with(
            ["bash", "-c", "bazel test --keep_going //..."],
            check=False,
        )

  def test_get_changed_files_normalizes_paths(self):
    with (
        mock.patch("subprocess.run") as mock_run,
        mock.patch("subprocess.check_output") as mock_check_output,
    ):
      mock_run.return_value.returncode = 0
      mock_check_output.return_value = " README.md \n\n  csrc/ops/add.cc\n  \n"
      changed = run_bazel_diff.get_changed_files("base123", "head456")
      self.assertEqual(changed, ["README.md", "csrc/ops/add.cc"])

  def test_main_raises_if_workspace_dir_missing(self):
    with tempfile.TemporaryDirectory() as td:
      missing_dir = pathlib.Path(td) / "nonexistent"
      with self.assertRaisesRegex(
          ValueError, "Workspace directory does not exist"
      ):
        run_bazel_diff.main([f"--workspace-dir={missing_dir}"])

  def test_should_run_ops_tests(self):
    opted_in_files = [
        "csrc/ops/add.cc",
        "csrc/ops/experimental/foo.h",
        "src/torch_tpu/ops/custom_op.py",
        "tests/op_testing.py",
        "tests/ops_test.py",
        "tests/ops_test_data.py",
        "tests/dynamism_test_utils.py",
        "tests/ops_test_gpu_golden_v1.json.gz",
    ]
    for opted_in_file in opted_in_files:
      with self.subTest(opted_in=opted_in_file):
        self.assertTrue(
            run_bazel_diff.should_run_ops_tests(["README.md", opted_in_file])
        )

    opted_out_files = [
        "csrc/eager/runtime.cc",
        "src/torch_tpu/compiler/pass.py",
        "tests/eager_test.py",
        # The patterns match whole paths, so these near-misses stay opted out.
        "tests/other_dynamism_test_utils.py",
        "tests/ops_test.py.orig",
        "csrc/ops_fusion/fuse.cc",
    ]
    with self.subTest(opted_out=opted_out_files):
      self.assertFalse(run_bazel_diff.should_run_ops_tests(opted_out_files))

  def test_filter_ops_test_targets(self):
    impacted = [
        "@@//tests:eager_test_tpu-v5lite",
        "@@//tests:ops_test_tpu-v5lite",
        "@@//tests:ops_test_tpu-v5lite_build_test",
        "//tests:foreach_ops_test_tpu-v7x",
        "@@//csrc/eager:traversal_test",
    ]
    ops_targets = [
        "//tests:ops_test_tpu-v5lite",
        "//tests:foreach_ops_test_tpu-v7x",
        "//tests:searchsorted_ops_test_tpu-v5lite",
    ]
    filtered = list(
        run_bazel_diff.filter_ops_test_targets(impacted, ops_targets)
    )
    self.assertEqual(
        filtered,
        [
            "@@//tests:eager_test_tpu-v5lite",
            "@@//tests:ops_test_tpu-v5lite_build_test",
            "@@//csrc/eager:traversal_test",
        ],
    )

  def test_parse_args_mode_and_impacted_targets_file(self):
    with self.subTest("explicit"):
      opts = run_bazel_diff.parse_args([
          "--mode=calculate",
          "--impacted-targets-file=impacted_targets.txt",
      ])
      self.assertEqual(opts.mode, run_bazel_diff.MODE_CALCULATE)
      self.assertEqual(opts.impacted_targets_file, "impacted_targets.txt")
    with self.subTest("defaults"):
      opts = run_bazel_diff.parse_args([])
      self.assertEqual(opts.mode, run_bazel_diff.MODE_ALL)
      self.assertEqual(opts.impacted_targets_file, "")

  def _options(self, **overrides) -> run_bazel_diff.BazelDiffOptions:
    """Builds BazelDiffOptions with test-friendly defaults."""
    defaults = {
        "workspace_dir": pathlib.Path("/workspace"),
        "base_sha": "base123",
        "current_sha": "head456",
        "bazel_config": "ci_tpu_v5_presubmit",
        "extra_bazel_flags": "",
        "test_all": False,
        "workflow_ref": "",
        "mode": run_bazel_diff.MODE_ALL,
        "impacted_targets_file": "",
    }
    return run_bazel_diff.BazelDiffOptions(**(defaults | overrides))

  def test_resolve_mode(self):
    cases = [
        # An explicit calculate request always wins.
        (
            {"mode": run_bazel_diff.MODE_CALCULATE},
            run_bazel_diff.MODE_CALCULATE,
        ),
        # Without a config we cannot run tests, so degrade to calculate.
        ({"bazel_config": ""}, run_bazel_diff.MODE_CALCULATE),
        # An explicit test mode request always resolves to test mode.
        (
            {"mode": run_bazel_diff.MODE_TEST},
            run_bazel_diff.MODE_TEST,
        ),
        (
            {"mode": run_bazel_diff.MODE_TEST, "test_all": True},
            run_bazel_diff.MODE_TEST,
        ),
        (
            {"mode": run_bazel_diff.MODE_TEST, "impacted_targets_file": ""},
            run_bazel_diff.MODE_TEST,
        ),
        # A precomputed targets file means the diff already happened upstream.
        (
            {"impacted_targets_file": "impacted_targets.txt"},
            run_bazel_diff.MODE_TEST,
        ),
        # `test_all` overrides the targets file in ALL mode and forces a full local run.
        (
            {
                "mode": run_bazel_diff.MODE_ALL,
                "impacted_targets_file": "impacted_targets.txt",
                "test_all": True,
            },
            run_bazel_diff.MODE_ALL,
        ),
        ({}, run_bazel_diff.MODE_ALL),
    ]
    for overrides, expected in cases:
      with self.subTest(overrides=overrides):
        self.assertEqual(
            run_bazel_diff.resolve_mode(self._options(**overrides)), expected
        )

  def test_write_github_output(self):
    with tempfile.TemporaryDirectory() as td:
      gh_output = pathlib.Path(td) / "github_output"
      gh_output.touch()
      with mock.patch.dict(os.environ, {"GITHUB_OUTPUT": str(gh_output)}):
        run_bazel_diff.write_github_output(
            has_impacted_targets=True, is_full_run=False
        )
      self.assertEqual(
          gh_output.read_text(encoding="utf-8"),
          "has_impacted_targets=true\nis_full_run=false\n",
      )

  def test_write_github_output_without_env_var(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      # Must not raise when running outside GitHub Actions.
      run_bazel_diff.write_github_output(
          has_impacted_targets=False, is_full_run=False
      )

  def test_run_precomputed_targets_runs_all_tests_when_test_all(self):
    with tempfile.TemporaryDirectory() as td:
      workspace = pathlib.Path(td)
      with mock.patch.object(
          run_bazel_diff, "_run_bazel_test", return_value=0
      ) as mock_test:
        exit_code = run_bazel_diff._run_precomputed_targets(
            self._options(test_all=True), workspace
        )
      self.assertEqual(exit_code, 0)
      mock_test.assert_called_once_with("ci_tpu_v5_presubmit", "", ["//..."])

  def test_run_precomputed_targets_skips_when_file_empty_or_missing(self):
    with tempfile.TemporaryDirectory() as td:
      workspace = pathlib.Path(td)
      (workspace / "empty.txt").write_text("\n", encoding="utf-8")
      for targets_file in ("", "empty.txt", "missing.txt"):
        with self.subTest(targets_file=targets_file):
          with mock.patch.object(
              run_bazel_diff, "_run_bazel_test"
          ) as mock_test:
            exit_code = run_bazel_diff._run_precomputed_targets(
                self._options(impacted_targets_file=targets_file), workspace
            )
          self.assertEqual(exit_code, 0)
          mock_test.assert_not_called()

  def test_run_precomputed_targets_runs_bazel_test(self):
    with tempfile.TemporaryDirectory() as td:
      workspace = pathlib.Path(td)
      targets_path = workspace / "impacted_targets.txt"
      targets_path.write_text("//tests:foo_test\n", encoding="utf-8")
      with mock.patch.object(
          run_bazel_diff, "_run_bazel_test", return_value=0
      ) as mock_test:
        exit_code = run_bazel_diff._run_precomputed_targets(
            self._options(impacted_targets_file="impacted_targets.txt"),
            workspace,
        )
      self.assertEqual(exit_code, 0)
      mock_test.assert_called_once_with(
          "ci_tpu_v5_presubmit",
          "",
          [
              "--skip_incompatible_explicit_targets",
              f"--target_pattern_file={targets_path}",
          ],
      )

  def test_run_bazel_test_preserves_shell_expansion(self):
    """`$(pwd)` in extra flags must reach bash unexpanded so bash resolves it."""
    with mock.patch("subprocess.run") as mock_run:
      mock_run.return_value.returncode = 0
      run_bazel_diff._run_bazel_test(
          "ci_tpu_v5_presubmit",
          '--run_under="$(pwd)/ci/tools/parallel_accelerator_execute.sh"',
          ["//..."],
      )
      mock_run.assert_called_with(
          [
              "bash",
              "-c",
              (
                  "bazel test --config=ci_tpu_v5_presubmit"
                  ' --run_under="$(pwd)/ci/tools/parallel_accelerator_execute.sh"'
                  " //..."
              ),
          ],
          check=False,
      )

  def test_normalize_label(self):
    cases = [
        ("@@//ci/tools:foo", "//ci/tools:foo"),
        ("@//ci/tools:foo", "//ci/tools:foo"),
        ("//ci/tools:foo", "//ci/tools:foo"),
        ("@rules_python//python:defs.bzl", "@rules_python//python:defs.bzl"),
    ]
    for label, expected in cases:
      with self.subTest(label=label):
        self.assertEqual(run_bazel_diff.normalize_label(label), expected)

  def test_query_impacted_tests_returns_empty_without_targets(self):
    with mock.patch("subprocess.run") as mock_run:
      self.assertEqual(run_bazel_diff.query_impacted_tests([]), [])
      mock_run.assert_not_called()

  def test_query_impacted_tests_normalizes_labels_in_query(self):
    queries = []

    def fake_run(argv, **kwargs):
      del kwargs
      query_file = next(
          a.removeprefix("--query_file=")
          for a in argv
          if a.startswith("--query_file=")
      )
      queries.append(pathlib.Path(query_file).read_text(encoding="utf-8"))
      return mock.Mock(returncode=0, stdout=_QUERY_XML, stderr="")

    with mock.patch("subprocess.run", side_effect=fake_run):
      tests = run_bazel_diff.query_impacted_tests(
          ["@@//tests:cpu_test", "@@//tests:v5_test"]
      )
      run_bazel_diff.query_impacted_tests(
          ["@@rules_python~0.31.0//python:defs.bzl"]
      )
    self.assertEqual(
        queries,
        [
            'tests(set("//tests:cpu_test" "//tests:v5_test"))',
            'tests(set("@@rules_python~0.31.0//python:defs.bzl"))',
        ],
    )
    self.assertEqual(
        [t.label for t in tests], ["//tests:cpu_test", "//tests:v5_test"]
    )

  def test_query_impacted_tests_tolerates_keep_going_partial_failure(self):
    """Exit code 3 still yields usable XML, e.g. when a target was deleted."""
    with mock.patch("subprocess.run") as mock_run:
      mock_run.return_value.returncode = 3
      mock_run.return_value.stdout = _QUERY_XML
      tests = run_bazel_diff.query_impacted_tests(["//tests:cpu_test"])
    self.assertIsNotNone(tests)
    self.assertEqual(len(tests), 2)

  def test_query_impacted_tests_distinguishes_failure_from_empty(self):
    """A failed query must not be mistaken for "no tests impacted"."""
    cases = [
        # (returncode, stdout, expected)
        (1, "", None),
        (0, "not xml at all", None),
        # A successful query with no matches is a genuine empty result.
        (0, "", []),
        (0, '<query version="2"></query>', []),
    ]
    for returncode, stdout, expected in cases:
      with self.subTest(returncode=returncode, stdout=stdout):
        with mock.patch("subprocess.run") as mock_run:
          mock_run.return_value.returncode = returncode
          mock_run.return_value.stdout = stdout
          mock_run.return_value.stderr = ""
          self.assertEqual(
              run_bazel_diff.query_impacted_tests(["//tests:foo"]), expected
          )

  def test_resolve_impacted_configs(self):
    with tempfile.TemporaryDirectory() as td:
      bazelrc = pathlib.Path(td) / ".bazelrc"
      bazelrc.write_text(_BAZELRC, encoding="utf-8")
      with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = _QUERY_XML
        configs = run_bazel_diff.resolve_impacted_configs(
            ["//tests:cpu_test", "//tests:v5_test"],
            bazelrc,
            allowed_configs={
                "ci_cpu_presubmit",
                "ci_tpu_v5_presubmit",
                "ci_tpu_v7_presubmit",
            },
        )
    # The v7 runner has nothing to do: neither test carries `presubmit-v7`.
    self.assertEqual(configs, ["ci_cpu_presubmit", "ci_tpu_v5_presubmit"])

  def test_resolve_impacted_configs_ignores_non_presubmit_configs(self):
    """Non-presubmit and disabled configs in .bazelrc must not match."""
    with tempfile.TemporaryDirectory() as td:
      bazelrc = pathlib.Path(td) / ".bazelrc"
      bazelrc.write_text(_BAZELRC, encoding="utf-8")
      with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        # A test tagged nopresubmit matches ci_cpu and ci_cpu_nightly, but not ci_cpu_presubmit.
        mock_run.return_value.stdout = _NOPRESUBMIT_QUERY_XML
        configs = run_bazel_diff.resolve_impacted_configs(
            ["//tests:nopresubmit_test"],
            bazelrc,
            allowed_configs={
                "ci_cpu_presubmit",
                "ci_tpu_v5_presubmit",
                "ci_tpu_v7_presubmit",
            },
        )
        self.assertEqual(configs, [])

        # A test tagged presubmit-v6 matches ci_tpu_v6_presubmit and ci_tpu_nightly, but v6 is inactive.
        mock_run.return_value.stdout = _V6_QUERY_XML
        configs = run_bazel_diff.resolve_impacted_configs(
            ["//tests:v6_test"],
            bazelrc,
            allowed_configs={
                "ci_cpu_presubmit",
                "ci_tpu_v5_presubmit",
                "ci_tpu_v7_presubmit",
            },
        )
        self.assertEqual(configs, [])

  def test_resolve_impacted_configs_all_when_allowed_configs_none(self):
    """When allowed_configs is None, all configs defined in .bazelrc are evaluated."""
    with tempfile.TemporaryDirectory() as td:
      bazelrc = pathlib.Path(td) / ".bazelrc"
      bazelrc.write_text(_BAZELRC, encoding="utf-8")
      with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = _V6_QUERY_XML
        configs = run_bazel_diff.resolve_impacted_configs(
            ["//tests:v6_test"], bazelrc, allowed_configs=None
        )
        self.assertEqual(configs, ["ci_tpu_nightly", "ci_tpu_v6_presubmit"])

  def test_resolve_impacted_configs_empty_when_no_target_is_a_test(self):
    with tempfile.TemporaryDirectory() as td:
      bazelrc = pathlib.Path(td) / ".bazelrc"
      bazelrc.write_text(_BAZELRC, encoding="utf-8")
      with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '<query version="2"></query>'
        configs = run_bazel_diff.resolve_impacted_configs(
            ["//ci/tools:run_bazel_diff.py"], bazelrc
        )
    self.assertEqual(configs, [])

  def test_resolve_impacted_configs_returns_none_when_undeterminable(self):
    """Callers must run every config rather than silently skipping runners."""
    with tempfile.TemporaryDirectory() as td:
      bazelrc = pathlib.Path(td) / ".bazelrc"
      with mock.patch("subprocess.run") as mock_run:
        # Query failure.
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "boom"
        self.assertIsNone(
            run_bazel_diff.resolve_impacted_configs(["//tests:foo"], bazelrc)
        )
        # Query succeeded, but the .bazelrc above was never created.
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = _QUERY_XML
        self.assertIsNone(
            run_bazel_diff.resolve_impacted_configs(["//tests:foo"], bazelrc)
        )

  def test_write_github_output_impacted_configs(self):
    cases = [
        # An undeterminable set is left unset so the workflow reads back the
        # empty string and conservatively runs every config.
        (None, "has_impacted_targets=true\nis_full_run=false\n"),
        (
            [],
            "has_impacted_targets=true\nis_full_run=false\nimpacted_configs=[]\n",
        ),
        (
            ["ci_cpu_presubmit"],
            (
                "has_impacted_targets=true\nis_full_run=false\n"
                'impacted_configs=["ci_cpu_presubmit"]\n'
            ),
        ),
    ]
    for impacted_configs, expected in cases:
      with self.subTest(impacted_configs=impacted_configs):
        with tempfile.TemporaryDirectory() as td:
          gh_output = pathlib.Path(td) / "github_output"
          gh_output.touch()
          with mock.patch.dict(os.environ, {"GITHUB_OUTPUT": str(gh_output)}):
            run_bazel_diff.write_github_output(
                has_impacted_targets=True,
                is_full_run=False,
                impacted_configs=impacted_configs,
            )
          self.assertEqual(gh_output.read_text(encoding="utf-8"), expected)


class MainTest(unittest.TestCase):
  """Tests `main`'s mode dispatch and its early returns.

  `main` shells out to git, curl, the bazel-diff jar and Bazel itself, so each
  of those is stubbed at its module-level seam. What is exercised here is the
  control flow between them: which mode short-circuits where, what reaches
  $GITHUB_OUTPUT, and -- most importantly -- when Bazel is invoked at all.
  """

  def setUp(self):
    super().setUp()
    self.workspace = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, self.workspace, ignore_errors=True)
    # `main` chdirs into the workspace; restore the cwd so tests stay isolated.
    self.addCleanup(os.chdir, os.getcwd())

    self.github_output = self.workspace / "github_output"
    self.github_output.touch()

    # Stub every external command `main` drives.
    self.addCleanup(mock.patch.stopall)
    self.mock_bazel_test = mock.patch.object(
        run_bazel_diff, "_run_bazel_test", return_value=0
    ).start()
    # A non-empty base SHA is what marks this as a PR; returning "" would send
    # every case down the "not a PR, run everything" branch instead.
    mock.patch.object(
        run_bazel_diff, "resolve_base_sha", return_value="base123"
    ).start()
    mock.patch.object(run_bazel_diff, "run_bazel_diff").start()
    # Covers the curl download plus the `git fetch`/`git checkout` calls.
    mock.patch.object(run_bazel_diff, "subprocess").start()

  def _set_impacted_targets(self, targets):
    """Writes the targets file the stubbed bazel-diff jar would have emitted."""
    (self.workspace / "impacted_targets.txt").write_text(
        "".join(f"{t}\n" for t in targets), encoding="utf-8"
    )

  def _run_main(
      self,
      *extra_argv,
      changed_files=("csrc/ops/add.cc",),
      bazel_config="ci_cpu_presubmit",
      candidate_configs=(
          "ci_cpu_presubmit",
          "ci_tpu_v5_presubmit",
          "ci_tpu_v7_presubmit",
      ),
  ):
    """Runs `main` against the temp workspace, returning its exit code.

    A bazel config is passed by default because `resolve_mode` degrades to
    `calculate` without one, which would quietly make the `all` mode cases
    below test nothing at all.
    """
    candidate_args = ()
    if candidate_configs and not any(
        arg.startswith("--candidate-configs")
        or arg.startswith("--workflow-file")
        for arg in extra_argv
    ):
      candidate_args = (f"--candidate-configs={','.join(candidate_configs)}",)
    with mock.patch.object(
        run_bazel_diff, "get_changed_files", return_value=list(changed_files)
    ):
      with mock.patch.dict(
          os.environ, {"GITHUB_OUTPUT": str(self.github_output)}
      ):
        return run_bazel_diff.main([
            f"--workspace-dir={self.workspace}",
            "--base-sha=base123",
            "--current-sha=head456",
            f"--bazel-config={bazel_config}",
            *candidate_args,
            *extra_argv,
        ])

  def _github_output(self) -> dict[str, str]:
    """Parses the key=value pairs `main` appended to $GITHUB_OUTPUT."""
    lines = self.github_output.read_text(encoding="utf-8").splitlines()
    return dict(line.split("=", 1) for line in lines if line)

  def test_all_mode_skips_bazel_when_no_target_is_impacted(self):
    """An empty impacted set must not reach Bazel.

    Regression test: invoking Bazel with an empty --target_pattern_file makes
    it fail with "no test targets were found, yet testing was requested",
    which is the exact waste this tooling exists to avoid.
    """
    self._set_impacted_targets([])

    exit_code = self._run_main()

    self.assertEqual(exit_code, 0)
    self.mock_bazel_test.assert_not_called()

  def test_all_mode_tests_impacted_targets(self):
    self._set_impacted_targets(["//tests:foo_test"])

    exit_code = self._run_main()

    self.assertEqual(exit_code, 0)
    self.mock_bazel_test.assert_called_once_with(
        "ci_cpu_presubmit",
        "",
        [
            "--skip_incompatible_explicit_targets",
            f"--target_pattern_file={self.workspace / 'impacted_targets.txt'}",
        ],
    )

  def test_calculate_mode_publishes_configs_without_running_bazel(self):
    self._set_impacted_targets(["//tests:cpu_test"])

    with mock.patch.object(
        run_bazel_diff,
        "resolve_impacted_configs",
        return_value=["ci_cpu_presubmit"],
    ):
      exit_code = self._run_main("--mode=calculate")

    self.assertEqual(exit_code, 0)
    self.mock_bazel_test.assert_not_called()
    self.assertEqual(
        self._github_output(),
        {
            "has_impacted_targets": "true",
            "is_full_run": "false",
            "impacted_configs": '["ci_cpu_presubmit"]',
        },
    )

  def test_calculate_mode_reports_no_work_when_no_config_matches(self):
    """Impacted targets that are not tests must still skip every runner."""
    self._set_impacted_targets(["//ci/tools:run_bazel_diff.py"])

    with mock.patch.object(
        run_bazel_diff, "resolve_impacted_configs", return_value=[]
    ):
      exit_code = self._run_main("--mode=calculate")

    self.assertEqual(exit_code, 0)
    self.mock_bazel_test.assert_not_called()
    self.assertEqual(
        self._github_output(),
        {
            "has_impacted_targets": "false",
            "is_full_run": "false",
            "impacted_configs": "[]",
        },
    )

  def test_calculate_mode_runs_every_config_when_undeterminable(self):
    """A failed query must fall back to running everything, not to skipping."""
    self._set_impacted_targets(["//tests:cpu_test"])

    with mock.patch.object(
        run_bazel_diff, "resolve_impacted_configs", return_value=None
    ):
      exit_code = self._run_main("--mode=calculate")

    self.assertEqual(exit_code, 0)
    self.mock_bazel_test.assert_not_called()
    # `impacted_configs` is omitted entirely, which callers read as "run all".
    self.assertEqual(
        self._github_output(),
        {"has_impacted_targets": "true", "is_full_run": "false"},
    )

  def test_calculate_mode_with_explicit_candidate_configs(self):
    self._set_impacted_targets(["//tests:cpu_test"])
    with mock.patch.object(
        run_bazel_diff,
        "resolve_impacted_configs",
        return_value=["ci_cpu_presubmit"],
    ) as mock_resolve:
      exit_code = self._run_main(
          "--mode=calculate",
          "--candidate-configs=ci_cpu_presubmit",
      )

    self.assertEqual(exit_code, 0)
    mock_resolve.assert_called_once_with(
        ["//tests:cpu_test"],
        self.workspace / ".bazelrc",
        allowed_configs=frozenset({"ci_cpu_presubmit"}),
        impacted_tests=mock.ANY,
    )
    self.assertEqual(
        self._github_output(),
        {
            "has_impacted_targets": "true",
            "is_full_run": "false",
            "impacted_configs": '["ci_cpu_presubmit"]',
        },
    )

  def test_calculate_mode_with_workflow_file(self):
    self._set_impacted_targets(["//tests:cpu_test"])
    wf_path = self.workspace / ".github" / "workflows" / "ci.yml"
    wf_path.parent.mkdir(parents=True, exist_ok=True)
    wf_path.write_text(
        textwrap.dedent("""\
            name: CI
            jobs:
              test:
                strategy:
                  matrix:
                    job_info:
                      - config: ci_cpu_presubmit
        """),
        encoding="utf-8",
    )
    with mock.patch.object(
        run_bazel_diff,
        "resolve_impacted_configs",
        return_value=["ci_cpu_presubmit"],
    ) as mock_resolve:
      exit_code = self._run_main(
          "--mode=calculate",
          "--workflow-file=.github/workflows/ci.yml",
      )

    self.assertEqual(exit_code, 0)
    mock_resolve.assert_called_once_with(
        ["//tests:cpu_test"],
        self.workspace / ".bazelrc",
        allowed_configs=frozenset({"ci_cpu_presubmit"}),
        impacted_tests=mock.ANY,
    )
    self.assertEqual(
        self._github_output(),
        {
            "has_impacted_targets": "true",
            "is_full_run": "false",
            "impacted_configs": '["ci_cpu_presubmit"]',
        },
    )

  def test_calculate_mode_with_missing_workflow_file_falls_back_to_run_all(
      self,
  ):
    self._set_impacted_targets(["//tests:cpu_test"])
    with mock.patch.object(
        run_bazel_diff,
        "resolve_impacted_configs",
    ) as mock_resolve:
      exit_code = self._run_main(
          "--mode=calculate",
          "--workflow-file=nonexistent.yml",
      )

    self.assertEqual(exit_code, 0)
    mock_resolve.assert_not_called()
    self.assertEqual(
        self._github_output(),
        {"has_impacted_targets": "true", "is_full_run": "false"},
    )

  def test_calculate_mode_signals_full_run_for_global_config_change(self):
    exit_code = self._run_main("--mode=calculate", changed_files=[".bazelrc"])

    self.assertEqual(exit_code, 0)
    self.mock_bazel_test.assert_not_called()
    self.assertEqual(
        self._github_output(),
        {"has_impacted_targets": "true", "is_full_run": "true"},
    )

  def test_calculate_mode_signals_full_run_for_ci_definition_change(self):
    exit_code = self._run_main(
        "--mode=calculate",
        changed_files=[".github/workflows/presubmit_runner.yml"],
    )

    self.assertEqual(exit_code, 0)
    self.mock_bazel_test.assert_not_called()
    self.assertEqual(
        self._github_output(),
        {"has_impacted_targets": "true", "is_full_run": "true"},
    )

  def test_calculate_mode_reports_no_work_for_doc_only_change(self):
    exit_code = self._run_main("--mode=calculate", changed_files=["README.md"])

    self.assertEqual(exit_code, 0)
    self.mock_bazel_test.assert_not_called()
    self.assertEqual(
        self._github_output(),
        {"has_impacted_targets": "false", "is_full_run": "false"},
    )

  def test_calculate_mode_ops_test_filtering_with_build_test(self):
    (self.workspace / ".bazelrc").write_text(_BAZELRC, encoding="utf-8")
    self._set_impacted_targets([
        "@@//tests:ops_test_tpu-v5lite",
        "@@//tests:ops_test_tpu-v5lite_build_test",
    ])

    build_test_xml = """\
<query version="2">
  <rule class="py_test" location="/w/tests/BUILD:1:1" name="//tests:ops_test_tpu-v5lite_build_test">
    <list name="tags"/>
  </rule>
</query>
"""

    def fake_run(cmd, *args, **kwargs):
      del args, kwargs
      if (
          isinstance(cmd, (list, tuple))
          and len(cmd) > 2
          and "attr(tags" in str(cmd[2])
      ):
        return mock.Mock(
            returncode=0, stdout="//tests:ops_test_tpu-v5lite\n", stderr=""
        )
      if isinstance(cmd, (list, tuple)) and any(
          isinstance(arg, str) and arg.startswith("--query_file=")
          for arg in cmd
      ):
        return mock.Mock(returncode=0, stdout=build_test_xml, stderr="")
      return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch.object(
        run_bazel_diff.subprocess, "run", side_effect=fake_run
    ):
      exit_code = self._run_main(
          "--mode=calculate", changed_files=["csrc/eager/runtime.cc"]
      )

    self.assertEqual(exit_code, 0)
    self.mock_bazel_test.assert_not_called()
    self.assertEqual(
        self._github_output(),
        {
            "has_impacted_targets": "true",
            "is_full_run": "false",
            "impacted_configs": '["ci_cpu_presubmit"]',
        },
    )
    self.assertEqual(
        (self.workspace / "impacted_targets.txt").read_text(encoding="utf-8"),
        "//tests:ops_test_tpu-v5lite_build_test\n",
    )

  def test_calculate_mode_narrows_impacted_targets_to_resolved_tests(self):
    self._set_impacted_targets([
        "@@//csrc/ops:add_lib",
        "@@//tests:deleted_test",
        "@@//tests:cpu_test",
    ])
    resolved_target = run_bazel_diff.list_ci_tests.TestTarget(
        name="cpu_test",
        package="tests",
        source_file="/workspace/tests/BUILD",
        tags=frozenset(),
    )
    with (
        mock.patch.object(
            run_bazel_diff,
            "query_impacted_tests",
            return_value=[resolved_target],
        ),
        mock.patch.object(
            run_bazel_diff,
            "resolve_impacted_configs",
            return_value=["ci_cpu_presubmit"],
        ),
    ):
      exit_code = self._run_main("--mode=calculate")

    self.assertEqual(exit_code, 0)
    self.mock_bazel_test.assert_not_called()
    self.assertEqual(
        (self.workspace / "impacted_targets.txt").read_text(encoding="utf-8"),
        "//tests:cpu_test\n",
    )

  def test_test_mode_uses_precomputed_targets_without_recomputing(self):
    self._set_impacted_targets(["//tests:foo_test"])

    with mock.patch.object(
        run_bazel_diff, "_run_precomputed_targets", return_value=0
    ) as mock_precomputed:
      with mock.patch.object(run_bazel_diff, "resolve_base_sha") as mock_base:
        exit_code = self._run_main(
            "--mode=test", "--impacted-targets-file=impacted_targets.txt"
        )

    self.assertEqual(exit_code, 0)
    mock_precomputed.assert_called_once()
    # The whole point of `test` mode: the diff is never recomputed.
    mock_base.assert_not_called()

  def test_test_mode_with_test_all_runs_all_tests_directly(self):
    with (
        mock.patch.object(run_bazel_diff, "resolve_base_sha") as mock_base,
        mock.patch.object(run_bazel_diff, "run_bazel_diff") as mock_bazel_diff,
    ):
      exit_code = self._run_main("--mode=test", "--test-all=true")

    self.assertEqual(exit_code, 0)
    mock_base.assert_not_called()
    mock_bazel_diff.assert_not_called()
    self.mock_bazel_test.assert_called_once_with(
        "ci_cpu_presubmit",
        "",
        ["//..."],
    )

  def test_test_mode_without_targets_file_skips_bazel(self):
    with (
        mock.patch.object(run_bazel_diff, "resolve_base_sha") as mock_base,
        mock.patch.object(run_bazel_diff, "run_bazel_diff") as mock_bazel_diff,
    ):
      exit_code = self._run_main(
          "--mode=test", "--impacted-targets-file=", "--test-all=false"
      )

    self.assertEqual(exit_code, 0)
    mock_base.assert_not_called()
    mock_bazel_diff.assert_not_called()
    self.mock_bazel_test.assert_not_called()


if __name__ == "__main__":
  unittest.main()
