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
import pathlib
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
        "shard_index": 0,
        "shard_count": 1,
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


_BAZELRC = """\
common --announce_rc
common:base --test_tag_filters=-notest
common:base --local_test_jobs=2
common:v5 --config=base
common:v5 --test_tag_filters=presubmit-v5,-notest  # trailing comment
common:v5 --local_test_jobs=8
common:multichip --test_tag_filters="requires-tpu-v5lite:8,-notest"
common:loop --config=loop
test:loop --test_tag_filters=looped
"""


def _test(label, *tags):
  return run_bazel_diff.QueriedTest(label, frozenset(tags))


class ShardSplitTest(unittest.TestCase):
  """Tests splitting a job's tests across several runners."""

  def test_parse_args_shard_flags(self):
    opts = run_bazel_diff.parse_args(["--shard-index=2", "--shard-count=3"])
    self.assertEqual((opts.shard_index, opts.shard_count), (2, 3))

  def test_parse_args_rejects_bad_shard_flags(self):
    for argv in (
        ["--shard-index=3", "--shard-count=3"],
        ["--shard-index=-1", "--shard-count=3"],
        ["--shard-count=0"],
    ):
      with self.subTest(argv=argv):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
          run_bazel_diff.parse_args(argv)

  def test_effective_flag(self):
    cases = [
        # A nested config's value is overridden by a later line of the parent.
        (["--config=v5"], "test_tag_filters", "presubmit-v5,-notest"),
        (["--config=v5"], "local_test_jobs", "8"),
        # Later --config flags win, and quotes are stripped.
        (
            ["--config=v5", "--config=multichip"],
            "test_tag_filters",
            "requires-tpu-v5lite:8,-notest",
        ),
        # A plain flag on the command line wins over an earlier config.
        (["--config=v5", "--local_test_jobs=4"], "local_test_jobs", "4"),
        (["--config=base"], "local_test_jobs", "2"),
        (["--config=missing"], "test_tag_filters", None),
        # A config that includes itself doesn't recurse forever.
        (["--config=loop"], "test_tag_filters", "looped"),
    ]
    for command_line, flag, expected in cases:
      with self.subTest(command_line=command_line, flag=flag):
        self.assertEqual(
            run_bazel_diff.effective_flag(_BAZELRC, command_line, flag),
            expected,
        )

  def test_effective_flag_reads_config_less_lines(self):
    bazelrc = "test --test_tag_filters=a\ncommon:x --test_tag_filters=b\n"
    self.assertEqual(
        run_bazel_diff.effective_flag(bazelrc, [], "test_tag_filters"), "a"
    )
    self.assertEqual(
        run_bazel_diff.effective_flag(
            bazelrc, ["--config=x"], "test_tag_filters"
        ),
        "b",
    )

  def test_matches_tag_filters(self):
    cases = [
        ({"a"}, [], True),
        ({"a"}, ["-b"], True),
        ({"a", "b"}, ["-b"], False),
        ({"a"}, ["a", "c"], True),
        ({"b"}, ["a", "c"], False),
        # Negative filters win over positive ones.
        ({"a", "notest"}, ["a", "-notest"], False),
    ]
    for tags, filters, expected in cases:
      with self.subTest(tags=tags, filters=filters):
        self.assertEqual(
            run_bazel_diff.matches_tag_filters(tags, filters), expected
        )

  def test_tests_query_expression(self):
    with self.subTest("wildcard"):
      self.assertEqual(
          run_bazel_diff.tests_query_expression(["//..."]),
          'tests(set(//...)) except attr(tags, "\\bmanual\\b", set(//...))',
      )
    with self.subTest("explicit_labels"):
      self.assertEqual(
          run_bazel_diff.tests_query_expression(
              ["@@//tests:a_test", "//tests:b_test"]
          ),
          "tests(set(//tests:a_test //tests:b_test))",
      )

  def test_parse_query_xml(self):
    xml_text = """<?xml version="1.1" encoding="UTF-8"?>
<query version="2">
  <rule class="py_test" location="/w/tests/BUILD:1:1" name="//tests:a_test">
    <string name="name" value="a_test"/>
    <list name="tags">
      <string value="exclusive"/>
      <string value="requires-tpu"/>
    </list>
  </rule>
  <rule class="sh_test" location="/w/tests/BUILD:9:1" name="//tests:b_test">
    <string name="name" value="b_test"/>
  </rule>
</query>
"""
    self.assertEqual(
        run_bazel_diff.parse_query_xml(xml_text),
        [
            _test("//tests:a_test", "exclusive", "requires-tpu"),
            _test("//tests:b_test"),
        ],
    )

  def test_load_test_durations_adds_label_prefix(self):
    with tempfile.TemporaryDirectory() as td:
      path = pathlib.Path(td) / "durations.json"
      path.write_text(
          '{"tests/a:b_test": 22, "//tests:c_test": 7}', encoding="utf-8"
      )
      self.assertEqual(
          run_bazel_diff.load_test_durations(path),
          {"//tests/a:b_test": 22, "//tests:c_test": 7},
      )

  def test_checked_in_durations_skip_label_prefix(self):
    raw = json.loads(
        run_bazel_diff._TEST_DURATIONS_FILE.read_text(encoding="utf-8")
    )
    self.assertTrue(raw)
    self.assertEqual([k for k in raw if k.startswith("//")], [])

  def test_estimated_seconds(self):
    durations = {"//t:excl": 80.0, "//t:shared": 80.0}
    cases = [
        (_test("//t:excl", "exclusive"), 80.0),
        (_test("//t:shared"), 10.0),
        (_test("//t:unknown"), run_bazel_diff._DEFAULT_TEST_SECONDS / 8),
    ]
    for test, expected in cases:
      with self.subTest(label=test.label):
        self.assertEqual(
            run_bazel_diff.estimated_seconds(test, durations, 8), expected
        )

  def test_partition_tests_balances_estimated_time(self):
    durations = {"//t:a": 300, "//t:b": 200, "//t:c": 150, "//t:d": 100}
    tests = [_test(label, "exclusive") for label in durations]
    shards = run_bazel_diff.partition_tests(tests, 2, durations, 8)
    self.assertEqual(shards, [["//t:a", "//t:d"], ["//t:b", "//t:c"]])

  def test_partition_tests_is_order_independent(self):
    durations = {"//t:a": 50, "//t:b": 50, "//t:c": 50}
    tests = [_test(label, "exclusive") for label in durations]
    self.assertEqual(
        run_bazel_diff.partition_tests(tests, 2, durations, 8),
        run_bazel_diff.partition_tests(tests[::-1], 2, durations, 8),
    )

  def test_partition_tests_more_shards_than_tests(self):
    shards = run_bazel_diff.partition_tests([_test("//t:a")], 3, {}, 8)
    self.assertEqual(shards, [["//t:a"], [], []])

  def test_shard_target_patterns(self):
    shards = [["//t:a"], ["//t:b", "//t:c"], ["//t:d"]]
    with self.subTest("shard_0_keeps_patterns_minus_other_shards"):
      self.assertEqual(
          run_bazel_diff.shard_target_patterns(["//..."], shards, 0),
          ["//...", "-//t:b", "-//t:c", "-//t:d"],
      )
    with self.subTest("other_shards_run_their_own_tests"):
      self.assertEqual(
          run_bazel_diff.shard_target_patterns(["//..."], shards, 1),
          ["//t:b", "//t:c"],
      )

  def _select(self, shard_index, extra_flags, queried):
    with tempfile.TemporaryDirectory() as td:
      workspace = pathlib.Path(td)
      (workspace / ".bazelrc").write_text(_BAZELRC, encoding="utf-8")
      durations = workspace / "durations.json"
      durations.write_text(
          '{"t:big": 600, "t:mid": 300, "t:ops": 800}', encoding="utf-8"
      )
      opts = run_bazel_diff.BazelDiffOptions(
          workspace_dir=workspace,
          base_sha="",
          current_sha="HEAD",
          bazel_config="v5",
          extra_bazel_flags=extra_flags,
          test_all=True,
          workflow_ref="",
          shard_index=shard_index,
          shard_count=2,
      )
      with (
          mock.patch.object(
              run_bazel_diff, "_query_tests", return_value=queried
          ),
          mock.patch.object(run_bazel_diff, "_TEST_DURATIONS_FILE", durations),
          mock.patch("builtins.print"),
      ):
        return run_bazel_diff.select_shard_patterns(opts, ["//..."])

  def test_select_shard_patterns_applies_the_job_tag_filters(self):
    queried = [
        _test("//t:big", "exclusive", "presubmit-v5", "requires-tpu-v5lite:8"),
        _test("//t:mid", "exclusive", "presubmit-v5", "requires-tpu-v5lite:8"),
        # Filtered out by the multichip overlay, so it must not take a shard.
        _test("//t:ops", "presubmit-v5"),
    ]
    flags = '--run_under="$(pwd)/wrapper.sh" --config=multichip'
    self.assertEqual(self._select(0, flags, queried), ["//...", "-//t:mid"])
    self.assertEqual(self._select(1, flags, queried), ["//t:mid"])

  def test_select_shard_patterns_falls_back_to_shard_0(self):
    self.assertEqual(self._select(0, "", None), ["//..."])
    self.assertEqual(self._select(1, "", None), [])


class RunTestsTest(unittest.TestCase):
  """Tests how _run_tests hands targets to Bazel."""

  def _opts(self, workspace, shard_index=0, shard_count=1):
    return run_bazel_diff.BazelDiffOptions(
        workspace_dir=workspace,
        base_sha="",
        current_sha="HEAD",
        bazel_config="ci",
        extra_bazel_flags="",
        test_all=True,
        workflow_ref="",
        shard_index=shard_index,
        shard_count=shard_count,
    )

  def test_unsharded_passes_patterns_through(self):
    with mock.patch.object(
        run_bazel_diff, "_run_bazel_test", return_value=0
    ) as run:
      run_bazel_diff._run_tests(self._opts(pathlib.Path("/w")), ["//..."])
      run.assert_called_once_with("ci", "", ["//..."])

  def test_unsharded_uses_existing_patterns_file(self):
    with mock.patch.object(
        run_bazel_diff, "_run_bazel_test", return_value=0
    ) as run:
      run_bazel_diff._run_tests(
          self._opts(pathlib.Path("/w")),
          ["//t:a"],
          pathlib.Path("/w/impacted_targets.txt"),
      )
      run.assert_called_once_with(
          "ci",
          "",
          [
              "--skip_incompatible_explicit_targets",
              "--target_pattern_file=/w/impacted_targets.txt",
          ],
      )

  def test_sharded_writes_this_shards_patterns(self):
    with tempfile.TemporaryDirectory() as td:
      workspace = pathlib.Path(td)
      with (
          mock.patch.object(
              run_bazel_diff,
              "select_shard_patterns",
              return_value=["//t:b", "//t:c"],
          ),
          mock.patch.object(
              run_bazel_diff, "_run_bazel_test", return_value=0
          ) as run,
      ):
        run_bazel_diff._run_tests(self._opts(workspace, 1, 2), ["//..."])
      patterns_file = workspace / "shard_targets.txt"
      self.assertEqual(
          patterns_file.read_text(encoding="utf-8"), "//t:b\n//t:c\n"
      )
      run.assert_called_once_with(
          "ci",
          "",
          [
              "--skip_incompatible_explicit_targets",
              f"--target_pattern_file={patterns_file}",
          ],
      )

  def test_sharded_with_no_tests_skips_bazel(self):
    with (
        mock.patch.object(
            run_bazel_diff, "select_shard_patterns", return_value=[]
        ),
        mock.patch.object(run_bazel_diff, "_run_bazel_test") as run,
        mock.patch("builtins.print"),
    ):
      rc = run_bazel_diff._run_tests(
          self._opts(pathlib.Path("/w"), 1, 2), ["//..."]
      )
    self.assertEqual(rc, 0)
    run.assert_not_called()


if __name__ == "__main__":
  unittest.main()
