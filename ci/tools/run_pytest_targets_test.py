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

"""Unit tests for run_pytest_targets.py."""

import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

_CI_TOOLS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _CI_TOOLS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from ci.tools import run_pytest_targets

_EXAMPLES_PREFIX = f"//{'examples'}"


class RunPytestTargetsTest(
    unittest.TestCase  # UNITTEST_OK=CI utility script test without torch_tpu
):

  def setUp(self) -> None:
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.workspace = pathlib.Path(self.temp_dir.name)

  def tearDown(self) -> None:
    self.temp_dir.cleanup()
    super().tearDown()

  def test_infer_device_and_generation(self) -> None:
    test_cases = [
        ("wheel_test_cpu", "cpu", None),
        ("ci_cpu", "cpu", None),
        ("wheel_test_tpu_v5", "tpu", "v5"),
        ("wheel_test_tpu_v6", "tpu", "v6"),
        ("wheel_test_tpu_v7", "tpu", "v7"),
        ("ci_tpu_v5e", "tpu", "v5"),
        ("ci_tpu_v6e", "tpu", "v6"),
        (None, "cpu", None),
    ]
    for config, expected_device, expected_gen in test_cases:
      self.assertEqual(run_pytest_targets.infer_device(config), expected_device)
      self.assertEqual(
          run_pytest_targets.infer_tpu_generation(config), expected_gen
      )
    self.assertEqual(
        run_pytest_targets.infer_device("wheel_test_cpu", "tpu"), "tpu"
    )
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=CI tool arg validation
        ValueError, "Invalid explicit device"
    ):
      run_pytest_targets.infer_device(None, explicit_device="gpu")

  def test_get_tag_excludes_regex(self) -> None:
    cpu_tags = run_pytest_targets.get_tag_excludes_regex("cpu")
    self.assertIn("requires-tpu.*", cpu_tags)
    self.assertIn("requires-gpu.*", cpu_tags)

    tpu_tags = run_pytest_targets.get_tag_excludes_regex("tpu")
    self.assertIn("fails-on-tpu.*", tpu_tags)
    self.assertNotIn("requires-tpu.*", tpu_tags)

    tpu_v7_tags = run_pytest_targets.get_tag_excludes_regex("tpu", tpu_gen="v7")
    self.assertIn("fails-on-tpu-v7", tpu_v7_tags)
    self.assertNotIn("fails-on-tpu.*", tpu_v7_tags)

  def test_sanitize_targets(self) -> None:
    raw_targets = [
        "  ",
        "# comment",
        "//tests:autoload_cpu_test",
        "@@//tests:precision_test",
        "@//tests:rng_test",
        "@local_torch//:torch",
        f"{_EXAMPLES_PREFIX}/mnist:mnist_test",
    ]
    self.assertEqual(
        run_pytest_targets.sanitize_targets(raw_targets),
        [
            "//tests:autoload_cpu_test",
            "//tests:precision_test",
            "//tests:rng_test",
            f"{_EXAMPLES_PREFIX}/mnist:mnist_test",
        ],
    )

  def test_build_bazel_query(self) -> None:
    full_query = run_pytest_targets.build_bazel_query("full", None, "manual")
    self.assertIn(f"//tests/... + {_EXAMPLES_PREFIX}/...", full_query)
    self.assertNotIn("presubmit-v7", full_query)

    v7_query = run_pytest_targets.build_bazel_query(
        "full", None, "manual", positive_tag="presubmit-v7"
    )
    self.assertIn("attr('tags', 'presubmit-v7',", v7_query)

    diff_query = run_pytest_targets.build_bazel_query(
        "diff", ["//tests:foo"], "manual", positive_tag="presubmit-v7"
    )
    self.assertIn("let impacted = set(", diff_query)
    self.assertIn("attr('tags', 'presubmit-v7',", diff_query)
    self.assertEqual(
        run_pytest_targets.build_bazel_query("diff", [], "manual"), ""
    )

  def test_convert_and_filter_files(self) -> None:
    labels = [
        "//tests:autoload_cpu_test.py",
        "//tests:device_tpu_test.py",
        "//tests/pallas:pallas_test.py",
        "//tests:cpp_style_test.py",
        f"{_EXAMPLES_PREFIX}/gemma4:correctness_test.py",
        f"{_EXAMPLES_PREFIX}/benchmarks:bench.py",
        "//csrc:kernel.cc",
    ]
    cpu_files = run_pytest_targets.filter_test_files(
        labels, device="cpu", check_exists=False
    )
    self.assertEqual(
        cpu_files,
        [
            "tests/autoload_cpu_test.py",
            "tests/pallas/pallas_test.py",
            "examples/gemma4/correctness_test.py",
        ],
    )
    tpu_files = run_pytest_targets.filter_test_files(
        labels, device="tpu", check_exists=False
    )
    self.assertEqual(
        tpu_files,
        [
            "tests/device_tpu_test.py",
            "tests/pallas/pallas_test.py",
            "examples/gemma4/correctness_test.py",
        ],
    )

  def test_resolve_pytest_cmd(self) -> None:
    custom_bin = self.workspace / "custom_env" / "bin"
    custom_bin.mkdir(parents=True)
    custom_py = custom_bin / "python"
    custom_py.touch()
    custom_py.chmod(0o755)
    with (
        mock.patch.dict(os.environ, {"VIRTUAL_ENV": str(custom_bin.parent)}),
        mock.patch.object(
            run_pytest_targets.shutil, "which", return_value="/usr/bin/pytest"
        ),
        mock.patch.object(
            run_pytest_targets, "_is_executable_file", return_value=False
        ),
    ):
      self.assertEqual(
          run_pytest_targets.resolve_pytest_cmd(self.workspace),
          ["/usr/bin/pytest"],
      )
    custom_pytest = custom_bin / "pytest"
    custom_pytest.touch()
    custom_pytest.chmod(0o755)
    with mock.patch.dict(os.environ, {"VIRTUAL_ENV": str(custom_bin.parent)}):
      self.assertEqual(
          run_pytest_targets.resolve_pytest_cmd(self.workspace),
          [str(custom_py), "-m", "pytest"],
      )
    self.assertEqual(
        run_pytest_targets.resolve_pytest_cmd(self.workspace, "/bin/pytest"),
        ["/bin/pytest"],
    )

  @mock.patch.dict(os.environ, {}, clear=True)
  @mock.patch.object(run_pytest_targets.subprocess, "run")
  def test_run_pytest_cpu_env_injection(
      self, mock_subproc: mock.MagicMock
  ) -> None:
    mock_subproc.return_value = mock.MagicMock(returncode=0)
    test_files = [
        "tests/autoload_cpu_test.py",
        "tests/autoload_disabled_backend_test.py",
    ]
    run_pytest_targets.run_pytest(test_files, self.workspace, device="cpu")
    self.assertEqual(mock_subproc.call_count, 2)
    env_autoload = mock_subproc.call_args_list[0].kwargs["env"]
    env_disabled = mock_subproc.call_args_list[1].kwargs["env"]
    self.assertEqual(
        env_autoload.get("TORCH_TPU_INTERNAL_ALLOW_XLA_BACKEND"), "1"
    )
    self.assertNotIn("TORCH_TPU_INTERNAL_ALLOW_XLA_BACKEND", env_disabled)

  @mock.patch.dict(os.environ, {}, clear=True)
  @mock.patch.object(run_pytest_targets.subprocess, "run")
  def test_run_pytest_tpu_pallas_env_injection(
      self, mock_subproc: mock.MagicMock
  ) -> None:
    mock_subproc.return_value = mock.MagicMock(returncode=0)
    test_files = [
        "tests/pallas/pallas_test.py",
        "tests/device_tpu_test.py",
    ]
    run_pytest_targets.run_pytest(test_files, self.workspace, device="tpu")
    self.assertEqual(mock_subproc.call_count, 2)
    env_pallas = mock_subproc.call_args_list[0].kwargs["env"]
    env_other = mock_subproc.call_args_list[1].kwargs["env"]
    self.assertEqual(env_pallas.get("TPU_PREMAPPED_BUFFER_SIZE"), "0")
    self.assertEqual(env_pallas.get("IS_OSS"), "1")
    self.assertNotIn("TPU_PREMAPPED_BUFFER_SIZE", env_other)
    self.assertNotIn("IS_OSS", env_other)

  @mock.patch.object(run_pytest_targets, "run_bazel_query")
  def test_dry_run_main(self, mock_query: mock.MagicMock) -> None:
    mock_query.return_value = ["//tests:autoload_cpu_test.py"]
    test_file = self.workspace / "tests" / "autoload_cpu_test.py"
    test_file.parent.mkdir(parents=True)
    test_file.touch()
    targets_file = self.workspace / "impacted.txt"
    targets_file.write_text("//tests:autoload_cpu_test\n", encoding="utf-8")
    code = run_pytest_targets.main([
        "--mode=diff",
        f"--targets-file={targets_file}",
        "--bazel-config=wheel_test_cpu",
        f"--workspace-dir={self.workspace}",
        "--dry-run",
    ])
    self.assertEqual(code, 0)

  def test_missing_targets_file_returns_error(self) -> None:
    missing_file = self.workspace / "nonexistent.txt"
    code = run_pytest_targets.main([
        "--mode=diff",
        f"--targets-file={missing_file}",
        "--bazel-config=wheel_test_cpu",
        f"--workspace-dir={self.workspace}",
    ])
    self.assertEqual(code, 1)


if __name__ == "__main__":
  unittest.main()
