#!/usr/bin/env python3
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

"""Unit tests for mailbox presubmit and freeform bazel test CLI tools."""

from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile
from typing import Final
import unittest

from rbe_tpu.cli import bazel_test

REPO_ROOT: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS_DIR: Final[pathlib.Path] = REPO_ROOT / "deploy"
MAILBOX_BAZEL_TEST: Final[pathlib.Path] = SCRIPTS_DIR / "rbe_tpu_bazel_test.sh"
MAILBOX_PRESUBMIT_PR: Final[pathlib.Path] = (
    SCRIPTS_DIR / "rbe_tpu_presubmit_pr.sh"
)


class TestRBETpuBazelTestCLI(
    unittest.TestCase  # UNITTEST_OK=CLI script test with no torch_tpu deps
):
  """Tests command-line handling of deploy/rbe_tpu_bazel_test.sh."""

  def run_cmd(self, *args: str, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(MAILBOX_BAZEL_TEST), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

  def test_help_flag(self):
    res = self.run_cmd("--help")
    self.assertEqual(res.returncode, 0)
    self.assertIn("Usage: rbe_tpu/deploy/rbe_tpu_bazel_test.sh", res.stdout)
    self.assertIn("--bucket", res.stdout)
    self.assertIn("--jobs", res.stdout)
    self.assertIn("--dry-run", res.stdout)

  def test_dry_run_single_target(self):
    res = self.run_cmd("--dry-run", "//tests/device:device_module_test_tpu")
    self.assertEqual(res.returncode, 0, res.stderr)
    self.assertIn("[DRY RUN] Execution plan confirmed", res.stdout)
    self.assertIn("//tests/device:device_module_test_tpu", res.stdout)

  def test_dry_run_multiple_targets(self):
    res = self.run_cmd(
        "--dry-run",
        "//tests:fp4_test",
        "//tests:view_test",
    )
    self.assertEqual(res.returncode, 0, res.stderr)
    self.assertIn("//tests:fp4_test", res.stdout)
    self.assertIn("//tests:view_test", res.stdout)

  def test_dry_run_extra_bazel_flags(self):
    res = self.run_cmd(
        "--dry-run",
        "//tests:fp4_test",
        "--",
        "--runs_per_test=10",
        "--test_arg=--verbose",
    )
    self.assertEqual(res.returncode, 0, res.stderr)
    self.assertIn("[DRY RUN] Extra bazel flags:", res.stdout)
    self.assertIn("--runs_per_test=10", res.stdout)
    self.assertIn("--test_arg=--verbose", res.stdout)

  def test_reject_invalid_jobs(self):
    res = self.run_cmd("//tests:fp4_test", "--jobs", "invalid")
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("--jobs must be a number", res.stderr)

  def test_reject_invalid_timeout(self):
    res = self.run_cmd("//tests:fp4_test", "--timeout", "invalid")
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("--timeout must be a number", res.stderr)

  def test_reject_unknown_option(self):
    res = self.run_cmd("//tests:fp4_test", "--unsupported-option")
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("Unknown option: --unsupported-option", res.stderr)

  def test_keep_flag_accepted(self):
    res = self.run_cmd(
        "--dry-run",
        "--keep",
        "//tests:fp4_test",
    )
    self.assertEqual(res.returncode, 0, res.stderr)
    self.assertIn("//tests:fp4_test", res.stdout)

  def test_custom_bucket_flag(self):
    res = self.run_cmd(
        "--dry-run",
        "--bucket",
        "custom-eval-bucket",
        "//tests:fp4_test",
    )
    self.assertEqual(res.returncode, 0, res.stderr)
    self.assertIn("gs://custom-eval-bucket/fleet/", res.stdout)

  def test_custom_bazel_config_flag(self):
    res = self.run_cmd(
        "--dry-run",
        "--bazel-config",
        "custom_config",
        "//tests:fp4_test",
    )
    self.assertEqual(res.returncode, 0, res.stderr)


class TestBazelTestPythonModule(
    unittest.TestCase  # UNITTEST_OK=Pure Python unit tests for bazel_test CLI
):
  """Unit tests for pure functions in rbe_tpu.cli.bazel_test."""

  def test_build_query_expr_excludes_manual_and_build_test(self):
    expr = bazel_test.build_query_expr("//csrc/...", "")
    self.assertEqual(
        expr,
        'tests(//csrc/...) except attr(tags, "manual|notest", //csrc/...)'
        ' except attr(name, ".*_build_test$", //csrc/...)',
    )

  def test_build_query_expr_with_positive_and_negative_tags(self):
    expr = bazel_test.build_query_expr(
        "//tests/...", "requires-tpu-v5lite,-flaky,-slow"
    )
    self.assertIn(
        'intersect attr(tags, "requires-tpu-v5lite", //tests/...)', expr
    )
    self.assertIn('except attr(tags, "flaky|slow", //tests/...)', expr)

  def test_filter_single_chip_targets_excludes_multichip(self):
    targets = [
        "//tests:ops_test_tpu-v5lite",
        "//tests:pallas_test:8",
        "//tests:dist_8tpu_test",
    ]
    filtered = bazel_test.filter_single_chip_targets(targets)
    self.assertEqual(filtered, ["//tests:ops_test_tpu-v5lite"])

  def test_build_bazel_args_orders_config_before_jobs(self):
    opts = bazel_test.parse_args([
        "--bazel-config",
        "custom_cfg",
        "--jobs",
        "14",
        "//tests:fp4_test",
    ])
    args = bazel_test.build_bazel_args(opts, opts.targets)
    cfg_idx = args.index("--config=custom_cfg")
    jobs_idx = args.index("--jobs=14")
    local_jobs_idx = args.index("--local_test_jobs=14")
    self.assertLess(cfg_idx, jobs_idx)
    self.assertLess(cfg_idx, local_jobs_idx)

  def test_extract_critical_path_parses_last_occurrence(self):
    log_text = (
        "INFO: Elapsed time: 12.3s, Critical Path: 5.4s\n"
        "INFO: Elapsed time: 45.6s, Critical Path: 31.2s\n"
    )
    self.assertEqual(bazel_test.extract_critical_path(log_text), "31.2s")
    self.assertEqual(bazel_test.extract_critical_path("no timing line"), "N/A")

  def test_format_report_includes_stage_timing_and_failures(self):
    report = bazel_test.format_report(
        verdict="FAILED",
        total_elapsed=40,
        bucket="my-bucket",
        target_count=2,
        fleet_count=28,
        now_tag="20260917_120000Z",
        bazel_rc=3,
        fleet_check_elapsed=1,
        query_elapsed=2,
        bazel_elapsed=37,
        critical_path="35.1s",
        failure_lines=["FAIL: //csrc/common:compilation_test"],
    )
    self.assertIn("## Stage Timing Breakdown", report)
    self.assertIn("35.1s", report)
    self.assertIn("### Failures", report)
    self.assertIn("FAIL: //csrc/common:compilation_test", report)


class TestRBETpuPresubmitPRCLI(
    unittest.TestCase  # UNITTEST_OK=CLI script test with no torch_tpu deps
):
  """Tests command-line handling of deploy/rbe_tpu_presubmit_pr.sh."""

  def run_cmd(self, *args: str, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(env_overrides)
    cmd_args = list(args)
    if "--help" not in cmd_args and "--output-dir" not in cmd_args:
      cmd_args = ["--output-dir", tempfile.mkdtemp(), *cmd_args]
    return subprocess.run(
        ["bash", str(MAILBOX_PRESUBMIT_PR), *cmd_args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

  def test_help_flag(self):
    res = self.run_cmd("--help")
    self.assertEqual(res.returncode, 0)
    self.assertIn("Usage: rbe_tpu/deploy/rbe_tpu_presubmit_pr.sh", res.stdout)
    self.assertIn("--mode", res.stdout)
    self.assertIn("--add-label", res.stdout)
    self.assertIn("--skip-head-check", res.stdout)

  def test_dry_run_default_shadow_mode(self):
    res = self.run_cmd(
        "--dry-run",
        "--target",
        "//tests:view_test",
    )
    self.assertEqual(res.returncode, 0, res.stderr)
    self.assertIn("Mode:            shadow", res.stdout)
    self.assertIn("Context:         TPU v5e relay (shadow)", res.stdout)
    self.assertIn("Target Label:    ci:relay-tpu-v5", res.stdout)
    self.assertIn(
        "[DRY RUN] Would report under context: 'TPU v5e relay (shadow)'",
        res.stdout,
    )

  def test_dry_run_replacement_mode_with_add_label(self):
    res = self.run_cmd(
        "--dry-run",
        "--pr",
        "3862",
        "--mode",
        "replacement",
        "--add-label",
        "--target",
        "//tests:view_test",
    )
    self.assertEqual(res.returncode, 0, res.stderr)
    self.assertIn("Mode:            replacement", res.stdout)
    self.assertIn(
        "Context:         Presubmit on linux-x86-ct5lp-224-8tpu", res.stdout
    )
    self.assertIn("Target Label:    ci:replace-tpu-v5", res.stdout)
    self.assertIn(
        "[DRY RUN] Would add label 'ci:replace-tpu-v5' to PR #3862", res.stdout
    )
    self.assertIn(
        "[DRY RUN] Would report under context: 'Presubmit on"
        " linux-x86-ct5lp-224-8tpu'",
        res.stdout,
    )

  def test_dry_run_shadow_mode_with_add_label(self):
    res = self.run_cmd(
        "--dry-run",
        "--pr",
        "3862",
        "--mode",
        "shadow",
        "--add-label",
        "--target",
        "//tests:view_test",
    )
    self.assertEqual(res.returncode, 0, res.stderr)
    self.assertIn(
        "[DRY RUN] Would add label 'ci:relay-tpu-v5' to PR #3862", res.stdout
    )

  def test_reject_invalid_mode(self):
    res = self.run_cmd("--mode", "invalid")
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("--mode must be 'shadow' or 'replacement'", res.stderr)

  def test_reject_invalid_jobs(self):
    res = self.run_cmd("--jobs", "not-a-number")
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("--jobs must be a number", res.stderr)

  def test_reject_unknown_argument(self):
    res = self.run_cmd("--bad-flag")
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("Unknown argument: --bad-flag", res.stderr)

  def test_config_flag_precedes_jobs_flags_in_presubmit_script(self):
    body = MAILBOX_PRESUBMIT_PR.read_text(encoding="utf-8")
    config_idx = body.find("--config=${CLI_BAZEL_CONFIG}")
    jobs_idx = body.find("--jobs=${CLI_JOBS}")
    local_jobs_idx = body.find("--local_test_jobs=${CLI_JOBS}")
    self.assertGreater(config_idx, 0)
    self.assertGreater(jobs_idx, config_idx)
    self.assertGreater(local_jobs_idx, config_idx)

  def test_bazelrc_decouples_mailbox_from_presubmit_filters(self):
    bazelrc_path = REPO_ROOT / ".bazelrc"
    if not bazelrc_path.exists():
      bazelrc_path = REPO_ROOT.parent / ".bazelrc"
    body = bazelrc_path.read_text(encoding="utf-8")
    self.assertIn("common:ci_tpu_v5_mailbox --config=ci_tpu_base", body)
    self.assertNotIn(
        "common:ci_tpu_v5_mailbox --config=ci_tpu_v5_presubmit", body
    )

  def test_presubmit_script_adds_presubmit_config_for_full_suite(self):
    body = MAILBOX_PRESUBMIT_PR.read_text(encoding="utf-8")
    self.assertIn('BAZEL_ARGS+=("--config=ci_tpu_v5_presubmit")', body)

  def test_label_is_verified_by_read_back_not_exit_code(self):
    body = MAILBOX_PRESUBMIT_PR.read_text(encoding="utf-8")
    self.assertIn("gh pr view", body)
    self.assertIn("--json labels", body)
    self.assertIn("label_applied=true", body)

  def test_replacement_mode_aborts_when_label_cannot_be_applied(self):
    body = MAILBOX_PRESUBMIT_PR.read_text(encoding="utf-8")
    abort_block = body.split('if [[ "$label_applied" == "false" ]]; then')[1]
    guard, _, remainder = abort_block.partition(
        'if [[ "$CLI_MODE" == "replacement" ]]; then'
    )
    self.assertEqual(guard.strip(), "")
    self.assertIn("exit 1", remainder.split("fi")[0])

  def test_shadow_mode_only_warns_when_label_cannot_be_applied(self):
    body = MAILBOX_PRESUBMIT_PR.read_text(encoding="utf-8")
    abort_block = body.split('if [[ "$label_applied" == "false" ]]; then')[1]
    self.assertIn("WARNING: failed to add label", abort_block)

  def test_label_check_runs_before_any_status_is_posted(self):
    body = MAILBOX_PRESUBMIT_PR.read_text(encoding="utf-8")
    self.assertLess(
        body.index('if [[ "$label_applied" == "false" ]]; then'),
        body.index('post_status "$CLI_SHA" "pending"'),
    )


if __name__ == "__main__":
  unittest.main()
