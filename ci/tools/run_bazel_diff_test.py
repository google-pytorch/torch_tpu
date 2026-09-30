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

import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from ci.tools import run_bazel_diff


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
      self.assertIs(opts.mode, run_bazel_diff.Mode.CALCULATE)
      self.assertEqual(opts.impacted_targets_file, "impacted_targets.txt")
    with self.subTest("defaults"):
      opts = run_bazel_diff.parse_args([])
      self.assertIs(opts.mode, run_bazel_diff.Mode.ALL)
      self.assertEqual(opts.impacted_targets_file, "")
    with self.subTest("invalid"):
      with (
          mock.patch("sys.stderr"),
          self.assertRaises(SystemExit),
      ):
        run_bazel_diff.parse_args(["--mode=bogus"])

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
        "mode": run_bazel_diff.Mode.ALL,
        "impacted_targets_file": "",
    }
    return run_bazel_diff.BazelDiffOptions(**(defaults | overrides))

  def test_resolve_mode(self):
    mode = run_bazel_diff.Mode
    cases = [
        # An explicit mode always wins.
        ({"mode": mode.CALCULATE}, mode.CALCULATE),
        ({"mode": mode.TEST}, mode.TEST),
        ({"mode": mode.TEST, "test_all": True}, mode.TEST),
        # A precomputed targets file means the diff already happened upstream.
        ({"impacted_targets_file": "impacted_targets.txt"}, mode.TEST),
        # `test_all` overrides the targets file and forces a full local run.
        (
            {"impacted_targets_file": "impacted_targets.txt", "test_all": True},
            mode.ALL,
        ),
        ({}, mode.ALL),
    ]
    for overrides, expected in cases:
      with self.subTest(overrides=overrides):
        self.assertIs(
            run_bazel_diff.resolve_mode(self._options(**overrides)), expected
        )

  def test_resolve_mode_requires_bazel_config(self):
    """Every mode hashes or tests with the runner's config, so it is required."""
    for mode in run_bazel_diff.Mode:
      with self.subTest(mode=mode):
        with self.assertRaises(ValueError):
          run_bazel_diff.resolve_mode(self._options(mode=mode, bazel_config=""))

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

  def test_run_precomputed_targets_skips_when_file_empty(self):
    with tempfile.TemporaryDirectory() as td:
      workspace = pathlib.Path(td)
      (workspace / "empty.txt").write_text("\n", encoding="utf-8")
      with mock.patch.object(run_bazel_diff, "_run_bazel_test") as mock_test:
        exit_code = run_bazel_diff._run_precomputed_targets(
            self._options(impacted_targets_file="empty.txt"), workspace
        )
      self.assertEqual(exit_code, 0)
      mock_test.assert_not_called()

  def test_run_precomputed_targets_fails_when_hand_off_is_broken(self):
    """A missing targets file must fail, not pass having tested nothing."""
    cases = [
        ("", ValueError),
        ("missing.txt", FileNotFoundError),
    ]
    with tempfile.TemporaryDirectory() as td:
      workspace = pathlib.Path(td)
      for targets_file, error in cases:
        with self.subTest(targets_file=targets_file):
          with (
              mock.patch.object(run_bazel_diff, "_run_bazel_test") as mock_test,
              self.assertRaises(error),
          ):
            run_bazel_diff._run_precomputed_targets(
                self._options(impacted_targets_file=targets_file), workspace
            )
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

  def _fake_bazel_build(self, events, returncode=0):
    """Returns a `subprocess.run` fake that writes `events` as BEP JSON."""
    calls = []

    def fake_run(args, check):
      del check  # Unused.
      calls.append(args)
      bep_path = args[-1].rsplit("--build_event_json_file=", 1)[1]
      pathlib.Path(bep_path).write_text(
          "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
      )
      return mock.Mock(returncode=returncode)

    return fake_run, calls

  @staticmethod
  def _configured(label, test_size=None):
    configured = {"targetKind": "py_test rule" if test_size else "filegroup"}
    if test_size:
      configured["testSize"] = test_size
    return {
        "id": {"targetConfigured": {"label": label}},
        "configured": configured,
    }

  def test_config_selects_tests_runs_bazel_with_the_runner_config(self):
    """Bazel, not Python, applies the config's tag filters."""
    fake_run, calls = self._fake_bazel_build([])
    with mock.patch("subprocess.run", side_effect=fake_run):
      run_bazel_diff.config_selects_tests(
          "ci_tpu_v5_presubmit",
          '--run_under="$(pwd)/ci/tools/parallel_accelerator_execute.sh"',
          pathlib.Path("/w/impacted_targets.txt"),
      )
    self.assertEqual(len(calls), 1)
    self.assertEqual(calls[0][:2], ["bash", "-c"])
    self.assertRegex(
        calls[0][2],
        r"^bazel build --config=ci_tpu_v5_presubmit"
        r' --run_under="\$\(pwd\)/ci/tools/parallel_accelerator_execute.sh"'
        r" --nobuild --build_tests_only --skip_incompatible_explicit_targets"
        r" --target_pattern_file=/w/impacted_targets.txt"
        r" --build_event_json_file=\S+/bep.json$",
    )

  def test_config_selects_tests_true_when_a_test_is_configured(self):
    fake_run, _ = self._fake_bazel_build([
        {"id": {"started": {}}, "started": {}},
        self._configured("//tests/eager:strictly_ordering_test", "SMALL"),
    ])
    with mock.patch("subprocess.run", side_effect=fake_run):
      self.assertTrue(
          run_bazel_diff.config_selects_tests(
              "ci_tpu_v5_presubmit", "", pathlib.Path("t.txt")
          )
      )

  def test_config_selects_tests_false_when_filters_leave_no_test(self):
    """Covers v7 skipping a v5-only test, which `test --nobuild` misreported."""
    fake_run, _ = self._fake_bazel_build([
        {"id": {"started": {}}, "started": {}},
        self._configured("//tests/eager:data"),
    ])
    with mock.patch("subprocess.run", side_effect=fake_run):
      self.assertFalse(
          run_bazel_diff.config_selects_tests(
              "ci_tpu_v7_presubmit", "", pathlib.Path("t.txt")
          )
      )

  def test_config_selects_tests_raises_on_analysis_error(self):
    """An error must never read as "no tests selected" and skip the runner."""
    for returncode in (1, 2, 4, 7, 37):
      with self.subTest(returncode=returncode):
        fake_run, _ = self._fake_bazel_build(
            [self._configured("//t:t_test", "SMALL")], returncode=returncode
        )
        with (
            mock.patch("subprocess.run", side_effect=fake_run),
            self.assertRaises(RuntimeError),
        ):
          run_bazel_diff.config_selects_tests(
              "ci_cpu_presubmit", "", pathlib.Path("t.txt")
          )

  def test_configured_test_targets_keeps_only_tests(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      bep_file = pathlib.Path(tmp_dir) / "bep.json"
      bep_file.write_text(
          "\n".join(
              json.dumps(e)
              for e in [
                  {"id": {"started": {}}, "started": {}},
                  self._configured("//a:lib"),
                  self._configured("//a:a_test", "MEDIUM"),
                  self._configured("//b:b_test", "SMALL"),
              ]
          )
          + "\n\n",
          encoding="utf-8",
      )
      self.assertEqual(
          run_bazel_diff.configured_test_targets(bep_file),
          ["//a:a_test", "//b:b_test"],
      )


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
  ):
    """Runs `main` against the temp workspace, returning its exit code.

    A bazel config is passed by default because `resolve_mode` rejects a
    missing one.
    """
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

  def test_main_requires_bazel_config(self):
    with self.assertRaises(ValueError):
      self._run_main(bazel_config="")

  def test_calculate_mode_reports_tests_selected_by_the_runner_config(self):
    self._set_impacted_targets(["//tests:cpu_test"])

    with mock.patch.object(
        run_bazel_diff, "config_selects_tests", return_value=True
    ) as mock_selects:
      exit_code = self._run_main("--mode=calculate")

    self.assertEqual(exit_code, 0)
    mock_selects.assert_called_once_with(
        "ci_cpu_presubmit", "", self.workspace / "impacted_targets.txt"
    )
    self.mock_bazel_test.assert_not_called()
    self.assertEqual(
        self._github_output(),
        {"has_impacted_targets": "true", "is_full_run": "false"},
    )

  def test_calculate_mode_hashes_with_the_runner_config(self):
    self._set_impacted_targets([])

    self._run_main("--mode=calculate", bazel_config="ci_tpu_v7_presubmit")

    hash_calls = [
        call
        for call in run_bazel_diff.run_bazel_diff.call_args_list
        if call.args[0] == "generate-hashes"
    ]
    # One call for the base SHA, one for the PR.
    self.assertEqual(len(hash_calls), 2)
    for call in hash_calls:
      self.assertIn(
          "--cqueryCommandOptions=--config=ci_tpu_v7_presubmit", call.args
      )

  def test_calculate_mode_reports_no_work_when_config_selects_no_test(self):
    """Impacted targets this config filters out must skip the runner."""
    self._set_impacted_targets(["//tests:v5_test"])

    with mock.patch.object(
        run_bazel_diff, "config_selects_tests", return_value=False
    ):
      exit_code = self._run_main("--mode=calculate")

    self.assertEqual(exit_code, 0)
    self.mock_bazel_test.assert_not_called()
    self.assertEqual(
        self._github_output(),
        {"has_impacted_targets": "false", "is_full_run": "false"},
    )

  def test_calculate_mode_skips_config_check_without_impacted_targets(self):
    self._set_impacted_targets([])

    with mock.patch.object(
        run_bazel_diff, "config_selects_tests"
    ) as mock_selects:
      exit_code = self._run_main("--mode=calculate")

    self.assertEqual(exit_code, 0)
    mock_selects.assert_not_called()
    self.assertEqual(
        self._github_output(),
        {"has_impacted_targets": "false", "is_full_run": "false"},
    )

  def test_calculate_mode_fails_when_config_check_fails(self):
    """A Bazel error must fail the job, not report "nothing to run"."""
    self._set_impacted_targets(["//tests:cpu_test"])

    with (
        mock.patch.object(
            run_bazel_diff,
            "config_selects_tests",
            side_effect=RuntimeError("bazel exited with code 1"),
        ),
        self.assertRaises(RuntimeError),
    ):
      self._run_main("--mode=calculate")

    self.assertEqual(self._github_output(), {})

  def test_calculate_mode_fails_when_bazel_diff_fails(self):
    run_bazel_diff.run_bazel_diff.side_effect = subprocess.CalledProcessError(
        1, ["java", "-jar", "bazel-diff.jar"]
    )

    with self.assertRaises(subprocess.CalledProcessError):
      self._run_main("--mode=calculate")

    self.assertEqual(self._github_output(), {})

  def test_calculate_mode_fails_when_ops_test_query_fails(self):
    """A failed ops-test query must fail instead of keeping every target."""
    self._set_impacted_targets(["@@//tests:cpu_test"])
    run_bazel_diff.subprocess.run.side_effect = subprocess.CalledProcessError(
        1, ["bazel", "query"]
    )

    with (
        mock.patch.object(
            run_bazel_diff, "config_selects_tests"
        ) as mock_selects,
        self.assertRaises(subprocess.CalledProcessError),
    ):
      # A change outside OPS_TEST_PATHS triggers the ops-test query.
      self._run_main(
          "--mode=calculate", changed_files=["csrc/eager/runtime.cc"]
      )

    mock_selects.assert_not_called()
    self.assertEqual(self._github_output(), {})

  def test_calculate_mode_signals_full_run_for_global_config_change(self):
    exit_code = self._run_main("--mode=calculate", changed_files=[".bazelrc"])

    self.assertEqual(exit_code, 0)
    self.mock_bazel_test.assert_not_called()
    self.assertEqual(
        self._github_output(),
        {"has_impacted_targets": "true", "is_full_run": "true"},
    )

  def test_calculate_mode_signals_full_run_without_base_sha(self):
    """`workflow_dispatch` has no base SHA, so everything must run."""
    run_bazel_diff.resolve_base_sha.return_value = ""

    with mock.patch.object(
        run_bazel_diff, "config_selects_tests"
    ) as mock_selects:
      exit_code = self._run_main("--mode=calculate")

    self.assertEqual(exit_code, 0)
    run_bazel_diff.run_bazel_diff.assert_not_called()
    mock_selects.assert_not_called()
    self.assertFalse((self.workspace / "impacted_targets.txt").exists())
    self.assertEqual(
        self._github_output(),
        {"has_impacted_targets": "true", "is_full_run": "true"},
    )

  def test_calculate_mode_signals_full_run_for_disable_bazel_diff_label(self):
    """The `ci:disable-bazel-diff` label reaches the script as --test-all."""
    with mock.patch.object(
        run_bazel_diff, "config_selects_tests"
    ) as mock_selects:
      exit_code = self._run_main("--mode=calculate", "--test-all=true")

    self.assertEqual(exit_code, 0)
    run_bazel_diff.run_bazel_diff.assert_not_called()
    mock_selects.assert_not_called()
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

  def test_calculate_mode_drops_ops_tests_before_checking_config(self):
    self._set_impacted_targets([
        "@@//tests:ops_test_tpu-v5lite",
        "@@//tests:ops_test_tpu-v5lite_build_test",
    ])
    run_bazel_diff.subprocess.run.return_value = mock.Mock(
        returncode=0, stdout="//tests:ops_test_tpu-v5lite\n", stderr=""
    )

    with mock.patch.object(
        run_bazel_diff, "config_selects_tests", return_value=True
    ):
      exit_code = self._run_main(
          "--mode=calculate", changed_files=["csrc/eager/runtime.cc"]
      )

    self.assertEqual(exit_code, 0)
    self.mock_bazel_test.assert_not_called()
    self.assertEqual(
        (self.workspace / "impacted_targets.txt").read_text(encoding="utf-8"),
        "@@//tests:ops_test_tpu-v5lite_build_test\n",
    )
    self.assertEqual(
        self._github_output(),
        {"has_impacted_targets": "true", "is_full_run": "false"},
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

  def test_test_mode_without_targets_file_fails(self):
    with self.assertRaises(ValueError):
      self._run_main(
          "--mode=test", "--impacted-targets-file=", "--test-all=false"
      )
    self.mock_bazel_test.assert_not_called()


if __name__ == "__main__":
  unittest.main()
