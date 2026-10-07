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

"""Unit tests for the open-source list_ci_tests tool."""

import io
import os
import pathlib
import platform
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

try:
  from ci.tools import list_ci_tests  # pylint: disable=g-import-not-at-top
except ImportError:
  import list_ci_tests  # pylint: disable=g-import-not-at-top

_SAMPLE_BAZELRC = textwrap.dedent("""\
    # Sample CI test filter definitions
    test:ci_cpu --test_tag_filters=-nobuild,-notest,-requires-gpu,-requires-tpu
    test:ci_cpu_presubmit --test_tag_filters=-nobuild,-notest,-requires-gpu,-requires-tpu,-nopresubmit
    test:ci_tpu_v5 --test_tag_filters=requires-tpu,-notest,-nobuild,-fails-on-tpu-v5
    test:wheel_test_tpu_v5 --test_tag_filters=-notest,-nobuild,-nopresubmit,-wheel_test_excluded,-fails-on-tpu-v5,presubmit-v5
    """)

_SAMPLE_BAZEL_XML = textwrap.dedent("""\
    <?xml version="1.1" encoding="UTF-8" standalone="no"?>
    <query version="2">
        <rule class="py_test" location="/workspace/tests/BUILD:10:1" name="//tests:ops_test_tpu-v5lite">
            <string name="name" value="ops_test_tpu-v5lite"/>
            <list name="tags">
                <string value="presubmit-v5"/>
                <string value="requires-tpu"/>
                <string value="requires-tpu-v5lite"/>
                <string value="fails-on-tpu-v6"/>
            </list>
        </rule>
        <rule class="cc_test" location="/workspace/csrc/ops/BUILD:20:1" name="//csrc/ops:kernel_test">
            <string name="name" value="kernel_test"/>
            <list name="tags">
                <string value="cpu"/>
            </list>
        </rule>
    </query>
    """)

_SAMPLE_WORKFLOW = textwrap.dedent("""\
    jobs:
      run_tests:
        strategy:
          matrix:
            job_info:
              - config: "ci_cpu"
                runner: "linux-x86-n4-16"
                machine_type: "CPU"
              - config: "ci_cpu_presubmit"
                runner: "linux-x86-n4-16"
                machine_type: "CPU"
              - config: "ci_tpu_v5"
                runner: "linux-x86-ct5lp-224-8tpu"
                machine_type: "TPU v5e"
      rbe:
        runs-on: linux-x86-n4-16
        steps:
          - uses: ./.github/actions/bazel-diff
            with:
              bazel_config: "ci_rbe_presubmit"
    """)


class ListCiTestsTest(unittest.TestCase):  # UNITTEST_OK=testing tools
  """Test suite for OSS list_ci_tests."""

  def setUp(self):
    super().setUp()
    self.repo_root = list_ci_tests.find_repo_root(pathlib.Path(__file__))

  def _write_sample_workflow(self, root_path: pathlib.Path) -> None:
    wf_dir = root_path / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "presubmit.yml").write_text(_SAMPLE_WORKFLOW, encoding="utf-8")

  def test_get_bazelrc_path(self):
    """Verifies that .bazelrc is discovered correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
      root_path = pathlib.Path(temp_dir)
      bazelrc = root_path / ".bazelrc"
      bazelrc.write_text("test:ci --test_tag_filters=cpu\n", encoding="utf-8")

      discovered = list_ci_tests.get_bazelrc_path(root_path)
      self.assertEqual(discovered, bazelrc)

  def test_get_bazelrc_path_missing(self):
    """Verifies FileNotFoundError when .bazelrc is missing."""
    with tempfile.TemporaryDirectory() as temp_dir:
      root_path = pathlib.Path(temp_dir)
      with self.assertRaises(FileNotFoundError):  # ASSERT_RAISES_OK=tools
        list_ci_tests.get_bazelrc_path(root_path)

  def test_find_repo_root_env_override(self):
    """Verifies TORCH_TPU_REPO_DIR environment variable override."""
    with tempfile.TemporaryDirectory() as temp_dir:
      with mock.patch.dict(os.environ, {"TORCH_TPU_REPO_DIR": temp_dir}):
        resolved = list_ci_tests.find_repo_root()
        self.assertEqual(resolved, pathlib.Path(temp_dir).resolve())

  def test_parse_bazelrc_tag_filters(self):
    """Verifies parsing of --test_tag_filters from .bazelrc content."""
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".bazelrc", delete=False
    ) as f:
      f.write(_SAMPLE_BAZELRC)
      f.flush()
      f_path = pathlib.Path(f.name)

    try:
      configs = list_ci_tests.parse_bazelrc_tag_filters(f_path)
      self.assertIn("ci_cpu", configs)
      self.assertEqual(
          configs["ci_cpu"],
          ["-nobuild", "-notest", "-requires-gpu", "-requires-tpu"],
      )
      self.assertIn("ci_tpu_v5", configs)
      self.assertEqual(
          configs["ci_tpu_v5"],
          ["requires-tpu", "-notest", "-nobuild", "-fails-on-tpu-v5"],
      )
    finally:
      f_path.unlink()

  def test_parse_bazelrc_tag_filters_multiline_and_quotes(self):
    """Verifies that multiple filter lines for the same job accumulate and strip quotes."""
    content = textwrap.dedent("""\
        test:ci_multi --test_tag_filters="foo, bar"
        test:ci_multi --test_tag_filters='baz, bar'
        """)
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".bazelrc", delete=False
    ) as f:
      f.write(content)
      f.flush()
      f_path = pathlib.Path(f.name)

    try:
      configs = list_ci_tests.parse_bazelrc_tag_filters(f_path)
      self.assertEqual(configs["ci_multi"], ["foo", "bar", "baz"])
    finally:
      f_path.unlink()

  def test_matches_tag_filters(self):
    """Verifies tag filtering matching positive and negative filters."""
    tags = {"requires-tpu", "presubmit-v5"}
    self.assertTrue(list_ci_tests.matches_tag_filters(tags, ["presubmit-v5"]))
    self.assertTrue(
        list_ci_tests.matches_tag_filters(tags, ["-fails-on-tpu-v5"])
    )
    self.assertFalse(list_ci_tests.matches_tag_filters(tags, ["-requires-tpu"]))
    self.assertFalse(list_ci_tests.matches_tag_filters(tags, ["presubmit-v6"]))

  def test_parse_bazel_query_xml(self):
    """Verifies parsing of Bazel query XML output."""
    targets = list_ci_tests.parse_bazel_query_xml(_SAMPLE_BAZEL_XML)
    self.assertEqual(len(targets), 2)

    target_map = {t.name: t for t in targets}
    self.assertIn("ops_test_tpu-v5lite", target_map)
    self.assertIn("kernel_test", target_map)

    v5_target = target_map["ops_test_tpu-v5lite"]
    self.assertEqual(v5_target.label, "//tests:ops_test_tpu-v5lite")
    self.assertEqual(v5_target.package, "tests")
    self.assertEqual(v5_target.source_file, "/workspace/tests/BUILD")
    self.assertIn("presubmit-v5", v5_target.tags)
    self.assertIn("requires-tpu", v5_target.tags)

    kernel_target = target_map["kernel_test"]
    self.assertEqual(kernel_target.label, "//csrc/ops:kernel_test")
    self.assertEqual(kernel_target.package, "csrc/ops")
    self.assertIn("cpu", kernel_target.tags)

  def test_find_bazel_binary_explicit(self):
    """Verifies using an explicit Bazel binary."""
    fake_bin = pathlib.Path("/bin/true")
    with mock.patch.object(
        list_ci_tests, "_verify_bazel_binary", return_value=True
    ):
      resolved = list_ci_tests.find_bazel_binary(
          self.repo_root, explicit_bazel=fake_bin
      )
      self.assertEqual(resolved, fake_bin.resolve())

  def test_find_bazel_binary_env(self):
    """Verifies resolution via $BAZEL_BIN environment variable."""
    fake_bin = "/custom/bin/bazel"
    with mock.patch.dict(os.environ, {"BAZEL_BIN": fake_bin}):
      with mock.patch.object(
          list_ci_tests, "_verify_bazel_binary", return_value=True
      ):
        resolved = list_ci_tests.find_bazel_binary(self.repo_root)
        self.assertEqual(resolved, pathlib.Path(fake_bin).resolve())

  def test_find_bazel_binary_from_path(self):
    """Verifies resolution via `bazel` on PATH."""
    fake_bin = "/usr/bin/bazel"
    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch.object(shutil, "which", return_value=fake_bin):
        with mock.patch.object(
            list_ci_tests, "_verify_bazel_binary", return_value=True
        ):
          resolved = list_ci_tests.find_bazel_binary(self.repo_root)
          self.assertEqual(resolved, pathlib.Path(fake_bin))

  def test_find_bazel_binary_cached_bazelisk(self):
    """Verifies fallback to a cached Bazelisk when PATH has no usable Bazel."""
    cached = list_ci_tests._get_bazelisk_cache_path()
    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch.object(shutil, "which", return_value=None):
        with mock.patch.object(
            list_ci_tests,
            "_verify_bazel_binary",
            side_effect=lambda p, root=None: p == cached,
        ):
          resolved = list_ci_tests.find_bazel_binary(self.repo_root)
          self.assertEqual(resolved, cached)

  def test_find_bazel_binary_downloads_bazelisk(self):
    """Verifies Bazelisk is downloaded when no usable Bazel is available."""
    downloaded = pathlib.Path("/fake/cache/bazelisk-1.27.0")
    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch.object(shutil, "which", return_value=None):
        with mock.patch.object(
            list_ci_tests,
            "_download_bazelisk",
            return_value=downloaded,
        ) as mock_download:
          # Every candidate fails until the freshly downloaded Bazelisk is
          # checked, so the download path is the one exercised here.
          with mock.patch.object(
              list_ci_tests,
              "_verify_bazel_binary",
              side_effect=lambda p, root=None: p == downloaded,
          ):
            resolved = list_ci_tests.find_bazel_binary(self.repo_root)
            self.assertEqual(resolved, downloaded)
            mock_download.assert_called_once()

  def test_find_bazel_binary_rejects_unverifiable_download(self):
    """Verifies a downloaded Bazelisk that does not run is not returned."""
    downloaded = pathlib.Path("/fake/cache/bazelisk-1.27.0")
    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch.object(shutil, "which", return_value=None):
        with mock.patch.object(
            list_ci_tests, "_download_bazelisk", return_value=downloaded
        ):
          with mock.patch.object(
              list_ci_tests, "_verify_bazel_binary", return_value=False
          ):
            with self.assertRaises(FileNotFoundError):  # ASSERT_RAISES_OK=tools
              list_ci_tests.find_bazel_binary(self.repo_root)

  def test_get_bazelisk_download_url_is_platform_specific(self):
    """Verifies the Bazelisk URL encodes the OS and CPU architecture."""
    with mock.patch.object(sys, "platform", "linux"):
      with mock.patch.object(platform, "machine", return_value="x86_64"):
        self.assertEqual(
            list_ci_tests._get_bazelisk_download_url("1.27.0"),
            "https://github.com/bazelbuild/bazelisk/releases/download/"
            "v1.27.0/bazelisk-linux-amd64",
        )
    with mock.patch.object(sys, "platform", "darwin"):
      with mock.patch.object(platform, "machine", return_value="arm64"):
        self.assertIn(
            "bazelisk-darwin-arm64",
            list_ci_tests._get_bazelisk_download_url("1.27.0"),
        )

  def test_get_bazelisk_download_url_unsupported_arch(self):
    """Verifies an unsupported CPU architecture is reported clearly."""
    with mock.patch.object(sys, "platform", "linux"):
      with mock.patch.object(platform, "machine", return_value="mips"):
        with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=tools
          list_ci_tests._get_bazelisk_download_url("1.27.0")

  def test_discover_test_targets_executes_query(self):
    """Verifies discover_test_targets executes Bazel query and parses XML."""
    mock_proc = mock.Mock(returncode=0, stdout=_SAMPLE_BAZEL_XML, stderr="")
    with mock.patch("subprocess.run", return_value=mock_proc) as mock_run:
      with mock.patch.object(
          list_ci_tests,
          "find_bazel_binary",
          return_value=pathlib.Path("/bin/bazel"),
      ):
        targets = list_ci_tests.discover_test_targets(self.repo_root)
        self.assertEqual(len(targets), 2)
        mock_run.assert_called_once_with(
            [
                "/bin/bazel",
                "query",
                'kind(".*_test rule", //...)',
                "--output=xml",
            ],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

  def test_map_jobs_to_tests(self):
    """Verifies mapping of test targets to CI jobs with tag filters."""
    targets = [
        list_ci_tests.TestTarget(
            name="cpu_test",
            package="tests",
            tags=set(),
            source_file="BUILD",
        ),
        list_ci_tests.TestTarget(
            name="tpu_v5_test",
            package="tests",
            tags={"requires-tpu", "presubmit-v5"},
            source_file="BUILD",
        ),
    ]

    configs = {
        "ci_cpu": ["-requires-tpu"],
        "ci_tpu_v5_presubmit": ["presubmit-v5"],
    }

    job_map = list_ci_tests.map_jobs_to_tests(targets, configs)
    self.assertEqual(len(job_map["ci_cpu"]), 1)
    self.assertEqual(job_map["ci_cpu"][0].name, "cpu_test")
    self.assertEqual(len(job_map["ci_tpu_v5_presubmit"]), 1)
    self.assertEqual(job_map["ci_tpu_v5_presubmit"][0].name, "tpu_v5_test")

  def test_map_test_to_jobs(self):
    """Verifies target querying across multiple jobs."""
    targets = [
        list_ci_tests.TestTarget(
            name="ops_test_tpu-v5lite",
            package="tests",
            tags={"requires-tpu", "presubmit-v5"},
            source_file="tests/BUILD",
        ),
    ]

    configs = {
        "ci_cpu": ["-requires-tpu"],
        "ci_tpu_v5": ["requires-tpu"],
        "ci_tpu_v5_presubmit": ["presubmit-v5"],
    }

    target, jobs = list_ci_tests.map_test_to_jobs(
        "ops_test_tpu-v5lite", targets, configs
    )
    self.assertIsNotNone(target)
    self.assertCountEqual(jobs, ["ci_tpu_v5", "ci_tpu_v5_presubmit"])

  def test_format_text_output(self):
    """Verifies that format_text_output formats targets with indentation and no bullet."""
    targets = [
        list_ci_tests.TestTarget(
            name="cpu_test",
            package="tests",
            tags=set(),
            source_file="BUILD",
        ),
    ]
    job_to_targets = {"ci_cpu": targets}
    output = list_ci_tests.format_text_output(job_to_targets)
    self.assertIn("CI Job: ci_cpu (1 tests)", output)
    self.assertIn("  //tests:cpu_test", output)
    self.assertNotIn("  - //tests:cpu_test", output)

  def test_cli_execution_list_jobs(self):
    """Verifies the --list_jobs CLI flag."""
    captured_stdout = io.StringIO()
    old_stdout = sys.stdout
    with tempfile.TemporaryDirectory() as temp_dir:
      root_path = pathlib.Path(temp_dir)
      (root_path / ".bazelrc").write_text(_SAMPLE_BAZELRC, encoding="utf-8")
      self._write_sample_workflow(root_path)
      try:
        sys.stdout = captured_stdout
        with mock.patch.dict(
            os.environ, {"TORCH_TPU_REPO_DIR": str(root_path)}
        ):
          ret = list_ci_tests.main(["--list_jobs"])
          self.assertEqual(ret, 0)
          output = captured_stdout.getvalue()
          self.assertIn("Available CI jobs", output)
          self.assertIn("[Machine: CPU (linux-x86-n4-16)]", output)
          self.assertIn("[Machine: TPU v5e (linux-x86-ct5lp-224-8tpu)]", output)
      finally:
        sys.stdout = old_stdout

  def test_cli_execution_no_workflows_dir(self):
    """Verifies CLI execution succeeds when .github/workflows is absent."""
    captured_stdout = io.StringIO()
    old_stdout = sys.stdout
    with tempfile.TemporaryDirectory() as temp_dir:
      root_path = pathlib.Path(temp_dir)
      (root_path / ".bazelrc").write_text(_SAMPLE_BAZELRC, encoding="utf-8")
      try:
        sys.stdout = captured_stdout
        with mock.patch.dict(
            os.environ, {"TORCH_TPU_REPO_DIR": str(root_path)}
        ):
          ret = list_ci_tests.main(["--list_jobs"])
          self.assertEqual(ret, 0)
          output = captured_stdout.getvalue()
          self.assertIn("Available CI jobs", output)
          self.assertNotIn("[Machine:", output)
      finally:
        sys.stdout = old_stdout

  def test_cli_execution_test_target_text(self):
    """Verifies querying a test target with text output format."""
    mock_targets = [
        list_ci_tests.TestTarget(
            name="my_target",
            package="tests",
            tags={"requires-tpu"},
            source_file="tests/BUILD",
        )
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
      root_path = pathlib.Path(temp_dir)
      (root_path / ".bazelrc").write_text(_SAMPLE_BAZELRC, encoding="utf-8")
      self._write_sample_workflow(root_path)

      captured_stdout = io.StringIO()
      old_stdout = sys.stdout
      try:
        sys.stdout = captured_stdout
        with mock.patch.dict(
            os.environ, {"TORCH_TPU_REPO_DIR": str(root_path)}
        ):
          with mock.patch.object(
              list_ci_tests, "discover_test_targets", return_value=mock_targets
          ):
            ret = list_ci_tests.main(["--test=my_target"])
            self.assertEqual(ret, 0)
            output = captured_stdout.getvalue()
            self.assertIn("Test Target: //tests:my_target", output)
            self.assertIn("Runs in 1 CI job(s):", output)
            self.assertIn("ci_tpu_v5", output)
            self.assertIn("TPU v5e (linux-x86-ct5lp-224-8tpu)", output)
      finally:
        sys.stdout = old_stdout

  def test_filter_targets_by_patterns(self):
    """Verifies glob filtering by target name and by full label."""
    ops_target = list_ci_tests.TestTarget(
        name="ops_test_tpu-v5lite",
        package="tests",
        tags=set(),
        source_file="tests/BUILD",
    )
    kernel_target = list_ci_tests.TestTarget(
        name="kernel_test",
        package="csrc/ops",
        tags=set(),
        source_file="csrc/ops/BUILD",
    )
    targets = [ops_target, kernel_target]

    matched, unmatched = list_ci_tests.filter_targets_by_patterns(
        targets, ["ops_test*"]
    )
    self.assertEqual(matched, [ops_target])
    self.assertEqual(unmatched, [])

    matched, unmatched = list_ci_tests.filter_targets_by_patterns(
        targets, ["//csrc/ops:*"]
    )
    self.assertEqual(matched, [kernel_target])
    self.assertEqual(unmatched, [])

    # Multiple patterns are unioned, and each target is listed at most once.
    matched, unmatched = list_ci_tests.filter_targets_by_patterns(
        targets, ["*_test*", "ops_test_tpu-v5lite"]
    )
    self.assertEqual(matched, targets)
    self.assertEqual(unmatched, [])

    matched, unmatched = list_ci_tests.filter_targets_by_patterns(
        targets, ["ops_test*", "no_such_test*"]
    )
    self.assertEqual(matched, [ops_target])
    self.assertEqual(unmatched, ["no_such_test*"])

  def test_cli_execution_tests_filter_text(self):
    """Verifies --tests limits the listed targets to matching globs."""
    mock_targets = [
        list_ci_tests.TestTarget(
            name="my_target",
            package="tests",
            tags=set(),
            source_file="tests/BUILD",
        ),
        list_ci_tests.TestTarget(
            name="other_target",
            package="tests",
            tags=set(),
            source_file="tests/BUILD",
        ),
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
      root_path = pathlib.Path(temp_dir)
      (root_path / ".bazelrc").write_text(_SAMPLE_BAZELRC, encoding="utf-8")
      self._write_sample_workflow(root_path)

      captured_stdout = io.StringIO()
      old_stdout = sys.stdout
      try:
        sys.stdout = captured_stdout
        with mock.patch.dict(
            os.environ, {"TORCH_TPU_REPO_DIR": str(root_path)}
        ):
          with mock.patch.object(
              list_ci_tests, "discover_test_targets", return_value=mock_targets
          ):
            ret = list_ci_tests.main([
                "--jobs=ci_cpu",
                "--tests=my_*",
            ])
            self.assertEqual(ret, 0)
            output = captured_stdout.getvalue()
            self.assertIn("//tests:my_target", output)
            self.assertNotIn("//tests:other_target", output)
      finally:
        sys.stdout = old_stdout

  def test_cli_execution_jobs_filter_count(self):
    """Verifies filtering by --jobs with count format."""
    mock_targets = [
        list_ci_tests.TestTarget(
            name="my_target",
            package="tests",
            tags=set(),
            source_file="tests/BUILD",
        )
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
      root_path = pathlib.Path(temp_dir)
      (root_path / ".bazelrc").write_text(_SAMPLE_BAZELRC, encoding="utf-8")
      self._write_sample_workflow(root_path)

      captured_stdout = io.StringIO()
      old_stdout = sys.stdout
      try:
        sys.stdout = captured_stdout
        with mock.patch.dict(
            os.environ, {"TORCH_TPU_REPO_DIR": str(root_path)}
        ):
          with mock.patch.object(
              list_ci_tests, "discover_test_targets", return_value=mock_targets
          ):
            ret = list_ci_tests.main([
                "--jobs=ci_cpu,ci_cpu_presubmit",
                "--format=count",
            ])
            self.assertEqual(ret, 0)
            output = captured_stdout.getvalue()
            self.assertIn("ci_cpu", output)
            self.assertIn("ci_cpu_presubmit", output)
            self.assertIn("Machine Type", output)
            self.assertIn("CPU (linux-x86-n4-16)", output)
      finally:
        sys.stdout = old_stdout

  def test_decode_runner_machine_type(self):
    """Verifies runner string to canonical machine type mapping."""
    self.assertEqual(
        list_ci_tests.decode_runner_machine_type("linux-x86-n4-16"), "CPU"
    )
    self.assertEqual(
        list_ci_tests.decode_runner_machine_type("linux-x86-ct5lp-224-8tpu"),
        "TPU v5e",
    )
    self.assertEqual(
        list_ci_tests.decode_runner_machine_type("linux-x86-ct6e-180-8tpu"),
        "TPU v6e",
    )
    self.assertEqual(
        list_ci_tests.decode_runner_machine_type("linux-x86-tpu7x-224-4tpu"),
        "TPU v7x",
    )
    self.assertEqual(
        list_ci_tests.decode_runner_machine_type("linux-x86-gpu-h100"), "GPU"
    )
    self.assertEqual(
        list_ci_tests.decode_runner_machine_type("custom-runner"), "Unknown"
    )

  def test_job_machine_info_description(self):
    """Verifies JobMachineInfo description formatting."""
    info_full = list_ci_tests.JobMachineInfo(
        machine_type="TPU v5e", runner="linux-x86-ct5lp-224-8tpu"
    )
    self.assertEqual(
        info_full.description, "TPU v5e (linux-x86-ct5lp-224-8tpu)"
    )

    info_no_runner = list_ci_tests.JobMachineInfo(machine_type="CPU", runner="")
    self.assertEqual(info_no_runner.description, "CPU")

    info_unknown_runner = list_ci_tests.JobMachineInfo(
        machine_type="CPU", runner="Unknown"
    )
    self.assertEqual(info_unknown_runner.description, "CPU")

  def test_parse_workflow_machine_types_mock(self):
    """Verifies explicit machine_type overrides runner inference in workflows."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = pathlib.Path(temp_dir)
      mock_workflow = temp_path / "mock.yml"
      mock_workflow.write_text(
          textwrap.dedent("""\
              jobs:
                test_matrix:
                  strategy:
                    matrix:
                      job_info:
                        - config: "ci_tpu_mock"
                          runner: "linux-x86-n4-16"
                          machine_type: "TPU v6e"
                        - config: "ci_cpu_mock"
                          runner: "linux-x86-n4-16"
              """),
          encoding="utf-8",
      )
      parsed = list_ci_tests.parse_workflow_machine_types(temp_path)
      self.assertIn("ci_tpu_mock", parsed)
      self.assertEqual(parsed["ci_tpu_mock"].machine_type, "TPU v6e")
      self.assertEqual(parsed["ci_tpu_mock"].runner, "linux-x86-n4-16")
      self.assertIn("ci_cpu_mock", parsed)
      self.assertEqual(parsed["ci_cpu_mock"].machine_type, "CPU")
      self.assertEqual(parsed["ci_cpu_mock"].runner, "linux-x86-n4-16")

  def test_parse_workflow_machine_types_any_config_key(self):
    """Verifies every `*_config` key of a matrix entry maps to its runner."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = pathlib.Path(temp_dir)
      (temp_path / "postsubmit.yml").write_text(
          textwrap.dedent("""\
              jobs:
                test_matrix:
                  strategy:
                    matrix:
                      job_info:
                        - runner: "linux-x86-tpu7x-224-4tpu"
                          bazel_config: "ci_tpu_v7"
                          newest_deps_config: "ci_tpu_v7_newest_deps"
                          config_file: "not_a_config"
              """),
          encoding="utf-8",
      )
      parsed = list_ci_tests.parse_workflow_machine_types(temp_path)
      self.assertCountEqual(parsed, ["ci_tpu_v7", "ci_tpu_v7_newest_deps"])
      for config in ("ci_tpu_v7", "ci_tpu_v7_newest_deps"):
        with self.subTest(config=config):
          self.assertEqual(parsed[config].machine_type, "TPU v7x")
          self.assertEqual(parsed[config].runner, "linux-x86-tpu7x-224-4tpu")

  def test_parse_workflow_machine_types_step_command(self):
    """Verifies parsing step-level bazel test --config in single-job workflows."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = pathlib.Path(temp_dir)
      mock_workflow = temp_path / "nightly.yml"
      mock_workflow.write_text(
          """
jobs:
  nightly_tpu:
    runs-on: linux-x86-ct6e-180-8tpu
    steps:
      - name: Run Bazel tests
        run: |
          bazel test --config=ci_tpu_nightly //tests:my_test
  rbe:
    name: "Presubmit on RBE"
    runs-on: linux-x86-n4-16
    steps:
      - name: Run Impacted Tests via bazel-diff
        uses: ./.github/actions/bazel-diff
        with:
          bazel_config: "ci_rbe_presubmit"
""",
          encoding="utf-8",
      )
      parsed = list_ci_tests.parse_workflow_machine_types(temp_path)
      self.assertIn("ci_tpu_nightly", parsed)
      self.assertEqual(parsed["ci_tpu_nightly"].machine_type, "TPU v6e")
      self.assertEqual(
          parsed["ci_tpu_nightly"].runner, "linux-x86-ct6e-180-8tpu"
      )
      self.assertIn("ci_rbe_presubmit", parsed)
      self.assertEqual(parsed["ci_rbe_presubmit"].machine_type, "CPU")
      self.assertEqual(parsed["ci_rbe_presubmit"].runner, "linux-x86-n4-16")

  def test_parse_workflow_machine_types_ignores_misleading_names(self):
    """Verifies a TPU runner wins over a job name that decodes to another type."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = pathlib.Path(temp_dir)
      (temp_path / "names.yml").write_text(
          textwrap.dedent("""\
              jobs:
                matrix_job:
                  strategy:
                    matrix:
                      job_info:
                        - name: "TPU v7 1-Chip"
                          runner: "linux-x86-tpu7x-56-1tpu"
                          config: "ci_single_chip"
                        - name: "Nightly v6 vs v5 compare"
                          runner: "linux-x86-ct6e-180-8tpu"
                          config: "ci_compare"
                qwen2_eval:
                  name: "Qwen2 eval"
                  runs-on: linux-x86-tpu7x-224-4tpu
                  steps:
                    - run: bazel test --config=ci_eval //tests:eval_test
              """),
          encoding="utf-8",
      )
      parsed = list_ci_tests.parse_workflow_machine_types(temp_path)
      cases = [
          ("ci_single_chip", "TPU v7x 1-Chip"),
          ("ci_compare", "TPU v6e"),
          ("ci_eval", "TPU v7x"),
      ]
      for config, expected in cases:
        with self.subTest(config=config):
          self.assertEqual(parsed[config].machine_type, expected)

  def test_find_workflows_dir(self):
    """Verifies that get_workflows_dir locates .github/workflows."""
    with tempfile.TemporaryDirectory() as temp_dir:
      root_path = pathlib.Path(temp_dir)
      self.assertIsNone(list_ci_tests.get_workflows_dir(root_path))
      wf_dir = root_path / ".github" / "workflows"
      wf_dir.mkdir(parents=True)
      self.assertEqual(list_ci_tests.get_workflows_dir(root_path), wf_dir)

  def test_real_workflows_machine_types_integrity(self):
    """Verifies that actual CI workflow definitions map to machine types."""
    workflows_dir = list_ci_tests.get_workflows_dir(self.repo_root)
    self.assertIsNotNone(
        workflows_dir, f"Workflows directory not found from {self.repo_root}"
    )
    job_machines = list_ci_tests.parse_workflow_machine_types(workflows_dir)

    expected_machines = {
        "ci_cpu_presubmit": ("CPU", "linux-x86-n4-16"),
        "ci_tpu_v5_presubmit": ("TPU v5e", "linux-x86-ct5lp-224-8tpu"),
        "ci_rbe_presubmit": ("CPU", "linux-x86-n4-16"),
        "ci_tpu_v7_presubmit": ("TPU v7x", "linux-x86-tpu7x-224-4tpu"),
        "ci_cpu_nightly": ("CPU", "linux-x86-n4-16"),
        "ci_tpu_nightly": ("TPU v6e", "linux-x86-ct6e-180-8tpu"),
        "wheel_test_cpu": ("CPU", "linux-x86-n4-16"),
        "wheel_test_tpu_v7": ("TPU v7x", "linux-x86-tpu7x-224-4tpu"),
    }

    for job, (expected_machine, expected_runner) in expected_machines.items():
      self.assertIn(
          job, job_machines, f"Expected CI job '{job}' not found in workflows"
      )
      self.assertEqual(
          job_machines[job].machine_type,
          expected_machine,
          f"Job '{job}' expected machine_type '{expected_machine}', got"
          f" '{job_machines[job].machine_type}'",
      )
      self.assertEqual(
          job_machines[job].runner,
          expected_runner,
          f"Job '{job}' expected runner '{expected_runner}', got"
          f" '{job_machines[job].runner}'",
      )

    # `ci_tpu_v6_presubmit` is commented out in presubmit.yml until more quota
    # is available, so it must not be reported as a job that runs tests.
    self.assertNotIn("ci_tpu_v6_presubmit", job_machines)

  def test_real_bazelrc_rbe_configs(self):
    """Verifies that ci_rbe_presubmit is the only RBE job in the real .bazelrc."""
    bazelrc_path = list_ci_tests.get_bazelrc_path(self.repo_root)
    rbe_configs = list_ci_tests.parse_bazelrc_rbe_configs(bazelrc_path)
    ci_configs = list_ci_tests.parse_bazelrc_tag_filters(bazelrc_path)

    self.assertEqual(rbe_configs & ci_configs.keys(), {"ci_rbe_presubmit"})

  def test_format_output_with_machine_info(self):
    """Verifies output formatters incorporate machine type details."""
    targets = [
        list_ci_tests.TestTarget(
            name="test_1",
            package="tests",
            tags=set(),
            source_file="BUILD",
        )
    ]
    job_to_targets = {"ci_tpu_v5": targets}
    job_machines = {
        "ci_tpu_v5": list_ci_tests.JobMachineInfo(
            machine_type="TPU v5e", runner="linux-x86-ct5lp-224-8tpu"
        )
    }

    # Text format
    text_out = list_ci_tests.format_text_output(job_to_targets, job_machines)
    self.assertIn(
        "CI Job: ci_tpu_v5 [Machine: TPU v5e (linux-x86-ct5lp-224-8tpu)]",
        text_out,
    )

    # Count format
    count_out = list_ci_tests.format_count_output(job_to_targets, job_machines)
    self.assertIn("Machine Type", count_out)
    self.assertIn("TPU v5e (linux-x86-ct5lp-224-8tpu)", count_out)

  def test_parse_workflow_machine_types_multi_job_isolation(self):
    """Verifies step commands in multi-job workflows don't leak runners across jobs."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = pathlib.Path(temp_dir)
      mock_workflow = temp_path / "multi_job.yml"
      mock_workflow.write_text(
          """
jobs:
  job_cpu:
    runs-on: linux-x86-n4-16
    steps:
      - name: Run CPU tests
        run: |
          bazel test --config=ci_cpu_isolated //tests:cpu_test

  job_tpu:
    runs-on: linux-x86-ct6e-180-8tpu
    steps:
      - name: Run TPU tests
        run: |
          bazel test --config=ci_tpu_isolated //tests:tpu_test
""",
          encoding="utf-8",
      )
      parsed = list_ci_tests.parse_workflow_machine_types(temp_path)
      self.assertIn("ci_cpu_isolated", parsed)
      self.assertEqual(parsed["ci_cpu_isolated"].machine_type, "CPU")
      self.assertEqual(parsed["ci_cpu_isolated"].runner, "linux-x86-n4-16")

      self.assertIn("ci_tpu_isolated", parsed)
      self.assertEqual(parsed["ci_tpu_isolated"].machine_type, "TPU v6e")
      self.assertEqual(
          parsed["ci_tpu_isolated"].runner, "linux-x86-ct6e-180-8tpu"
      )

  def test_parse_workflow_machine_types_skips_unparseable_workflow(self):
    """Verifies malformed workflows are skipped without failing the parse."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = pathlib.Path(temp_dir)
      (temp_path / "broken.yml").write_text(
          "jobs: [unclosed\n", encoding="utf-8"
      )
      (temp_path / "not_a_mapping.yml").write_text(
          "- just\n- a\n- list\n", encoding="utf-8"
      )
      (temp_path / "good.yml").write_text(
          """
jobs:
  job_cpu:
    runs-on: linux-x86-n4-16
    steps:
      - run: bazel test --config=ci_cpu_ok //tests:cpu_test
""",
          encoding="utf-8",
      )
      parsed = list_ci_tests.parse_workflow_machine_types(temp_path)
      self.assertEqual(list(parsed), ["ci_cpu_ok"])

  def test_format_count_output_dynamic_column_width(self):
    """Verifies that format_count_output dynamically pads long machine descriptions."""
    targets = [
        list_ci_tests.TestTarget(
            name="test_1",
            package="tests",
            tags=set(),
            source_file="BUILD",
        )
    ]
    job_to_targets = {"ci_long_machine_job": targets}
    long_desc_runner = (
        "linux-x86-very-very-long-custom-runner-name-exceeding-38-chars"
    )
    job_machines = {
        "ci_long_machine_job": list_ci_tests.JobMachineInfo(
            machine_type="TPU v7x", runner=long_desc_runner
        )
    }
    output = list_ci_tests.format_count_output(job_to_targets, job_machines)
    expected_desc = f"TPU v7x ({long_desc_runner})"
    self.assertIn(expected_desc, output)
    for line in output.splitlines():
      if "ci_long_machine_job" in line:
        self.assertTrue(line.endswith("1"))
        self.assertIn(expected_desc, line)

  def test_parse_bazelrc_rbe_configs(self):
    """Verifies rbe_test_execution and configs that include it directly are RBE."""
    with tempfile.TemporaryDirectory() as temp_dir:
      path = pathlib.Path(temp_dir) / ".bazelrc"
      path.write_text(
          textwrap.dedent("""\
              common:rbe_test_execution --remote_executor=grpcs://example
              common:ci_rbe --config=resultstore_base --config=rbe_test_execution
              common:ci_cpu --config=resultstore_base
              common:ci_other --config=rbe_test_execution_other
              # common:ci_commented_out --config=rbe_test_execution
              """),
          encoding="utf-8",
      )

      self.assertEqual(
          list_ci_tests.parse_bazelrc_rbe_configs(path),
          {"rbe_test_execution", "ci_rbe"},
      )

  def test_resolve_rbe_target_machine_type(self):
    """Verifies RBE tests map to CPU or their presubmit-v<N> TPU."""
    resolve = list_ci_tests.resolve_rbe_target_machine_type
    cases = [
        (set(), "CPU"),
        ({"presubmit-v5"}, "CPU"),
        ({"requires-tpu", "presubmit-v5"}, "TPU v5e"),
        ({"requires-tpu", "presubmit-v6"}, "TPU v6e"),
        ({"requires-tpu", "presubmit-v7"}, "TPU v7x"),
        ({"requires-tpu"}, "Unknown"),
    ]
    for tags, expected in cases:
      with self.subTest(tags=sorted(tags)):
        self.assertEqual(resolve(tags), expected)

  def test_cli_execution_rbe_per_target_machine_types(self):
    """Verifies main() labels RBE jobs as per-test and annotates each RBE target."""
    mock_targets = [
        list_ci_tests.TestTarget(
            name="cpu_test",
            package="tests",
            tags=set(),
            source_file="tests/BUILD",
        ),
        list_ci_tests.TestTarget(
            name="cpu_with_presubmit_tag",
            package="tests",
            tags={"presubmit-v5"},
            source_file="tests/BUILD",
        ),
        list_ci_tests.TestTarget(
            name="v5_test",
            package="tests",
            tags={"requires-tpu", "presubmit-v5"},
            source_file="tests/BUILD",
        ),
        list_ci_tests.TestTarget(
            name="v6_test",
            package="tests",
            tags={"requires-tpu", "presubmit-v6"},
            source_file="tests/BUILD",
        ),
        list_ci_tests.TestTarget(
            name="v7_test",
            package="tests",
            tags={"requires-tpu", "presubmit-v7"},
            source_file="tests/BUILD",
        ),
        list_ci_tests.TestTarget(
            name="unrouted_tpu_test",
            package="tests",
            tags={"requires-tpu"},
            source_file="tests/BUILD",
        ),
    ]
    bazelrc_content = textwrap.dedent("""\
        common:rbe_test_execution --remote_executor=grpcs://example
        common:ci_rbe_presubmit --config=resultstore_base --config=rbe_test_execution
        common:ci_rbe_presubmit --test_tag_filters=-notest,-nobuild
        common:ci_tpu_v5 --config=rbe_test_execution_other
        # common:ci_tpu_v5 --config=rbe_test_execution
        common:ci_tpu_v5 --test_tag_filters=requires-tpu,-notest,-nobuild
        """)
    with tempfile.TemporaryDirectory() as temp_dir:
      root_path = pathlib.Path(temp_dir)
      (root_path / ".bazelrc").write_text(bazelrc_content, encoding="utf-8")
      self._write_sample_workflow(root_path)

      with mock.patch.dict(os.environ, {"TORCH_TPU_REPO_DIR": str(root_path)}):
        with mock.patch.object(
            list_ci_tests, "discover_test_targets", return_value=mock_targets
        ):
          text_stdout = io.StringIO()
          with mock.patch.object(sys, "stdout", text_stdout):
            self.assertEqual(list_ci_tests.main([]), 0)
          text_out = text_stdout.getvalue()

          expected_rbe_block = textwrap.dedent("""\
              CI Job: ci_rbe_presubmit [Machine: RBE (per test)] (6 tests)
                //tests:cpu_test [CPU]
                //tests:cpu_with_presubmit_tag [CPU]
                //tests:v5_test [TPU v5e]
                //tests:v6_test [TPU v6e]
                //tests:v7_test [TPU v7x]
                //tests:unrouted_tpu_test [Unknown]""")
          expected_non_rbe_block = textwrap.dedent("""\
              CI Job: ci_tpu_v5 [Machine: TPU v5e (linux-x86-ct5lp-224-8tpu)] (4 tests)
                //tests:v5_test
                //tests:v6_test
                //tests:v7_test
                //tests:unrouted_tpu_test""")
          self.assertIn(expected_rbe_block, text_out)
          self.assertIn(expected_non_rbe_block, text_out)

          list_jobs_stdout = io.StringIO()
          with mock.patch.object(sys, "stdout", list_jobs_stdout):
            self.assertEqual(list_ci_tests.main(["--list_jobs"]), 0)
          self.assertIn(
              "ci_rbe_presubmit [Machine: RBE (per test)]:",
              list_jobs_stdout.getvalue(),
          )


if __name__ == "__main__":
  unittest.main()
