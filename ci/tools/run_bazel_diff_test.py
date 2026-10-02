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

import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from ci.tools import run_bazel_diff
from ci.tools.testing import fakes


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
    }
    for field, expected_val in expected.items():
      with self.subTest(field=field):
        self.assertEqual(getattr(opts, field), expected_val)

  def test_parse_args_defaults(self):
    opts = run_bazel_diff.parse_args([])
    expected = {
        "base_sha": "",
        "current_sha": "HEAD",
        "bazel_config": "",
        "extra_bazel_flags": "",
        "test_all": False,
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
    runner = fakes.FakeSubprocessRunner()
    runner.on(["java", "-jar"])
    with mock.patch.object(subprocess, "run", runner):
      run_bazel_diff.run_bazel_diff(
          "generate-hashes",
          "-w",
          pathlib.Path("/workspace"),
          pathlib.Path("/workspace/hashes.json"),
      )
    self.assertEqual(
        runner.commands("java"),
        [(
            "java",
            "-jar",
            "/tmp/bazel-diff.jar",
            "generate-hashes",
            "-w",
            "/workspace",
            "/workspace/hashes.json",
        )],
    )

  def test_bazel_test_command_config_flag(self):
    with self.subTest("with_config"):
      self.assertEqual(
          run_bazel_diff._bazel_test_command(
              "ci_cpu", "--keep_going", ["/" + "/..."]
          ),
          "bazel test --config=ci_cpu --keep_going /" + "/...",
      )
    with self.subTest("empty_config"):
      self.assertEqual(
          run_bazel_diff._bazel_test_command(
              "", "--keep_going", ["/" + "/..."]
          ),
          "bazel test --keep_going /" + "/...",
      )

  def test_bazel_analysis_command_config_flag(self):
    with self.subTest("with_config"):
      self.assertEqual(
          run_bazel_diff._bazel_analysis_command("wheel_test_cpu"),
          [
              "bazel",
              "cquery",
              "--config=wheel_test_cpu",
              "--output=label",
              "/" + "/...",
          ],
      )
    with self.subTest("empty_config"):
      self.assertEqual(
          run_bazel_diff._bazel_analysis_command(""),
          ["bazel", "cquery", "--output=label", "/" + "/..."],
      )

  def test_main_test_all_analyzes_before_testing(self):
    with (
        tempfile.TemporaryDirectory() as td,
        mock.patch.object(run_bazel_diff, "resolve_base_sha", return_value=""),
        mock.patch("os.chdir"),
    ):
      args = [f"--workspace-dir={td}", "--bazel-config=wheel_test_cpu"]
      with self.subTest("analysis_passes"):
        runner = fakes.FakeSubprocessRunner()
        runner.on(["bazel", "cquery"], returncode=0)
        runner.on(["bash", "-c"], returncode=0)
        with mock.patch.object(subprocess, "run", runner):
          self.assertEqual(run_bazel_diff.main(args), 0)
        self.assertEqual(len(runner.commands("bazel", "cquery")), 1)
        self.assertEqual(len(runner.commands("bash", "-c")), 1)
      with self.subTest("analysis_fails"):
        runner = fakes.FakeSubprocessRunner()
        runner.on(["bazel", "cquery"], returncode=1)
        with mock.patch.object(subprocess, "run", runner):
          self.assertEqual(run_bazel_diff.main(args), 1)
        self.assertEqual(len(runner.commands("bazel", "cquery")), 1)
        self.assertEqual(runner.commands("bash", "-c"), [])

  def test_get_changed_files_normalizes_paths(self):
    runner = fakes.FakeSubprocessRunner()
    runner.on(["git", "fetch"])
    runner.on(
        ["git", "diff", "--name-only"],
        stdout=" README.md \n\n  csrc/ops/add.cc\n  \n",
    )
    runner.on(["git", "diff", "--name-status"])
    runner.on(["git", "diff", "--stat"])
    with mock.patch.object(subprocess, "run", runner):
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
        "tests/goldens.pin",
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


if __name__ == "__main__":
  unittest.main()
