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

"""Unit tests for mailbox presubmit and freeform bazel test CLI scripts."""

import os
import subprocess
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAILBOX_BAZEL_TEST = os.path.join(REPO_ROOT, "scripts", "mailbox_bazel_test.sh")
MAILBOX_PRESUBMIT_PR = os.path.join(
    REPO_ROOT, "scripts", "mailbox_presubmit_pr.sh"
)


class TestMailboxBazelTestCLI(unittest.TestCase):
  """Tests command-line handling of scripts/mailbox_bazel_test.sh."""

  def run_cmd(self, *args, **env_overrides):
    env = dict(os.environ)
    env.update(env_overrides)
    return subprocess.run(
        ["bash", MAILBOX_BAZEL_TEST, *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

  def test_help_flag(self):
    res = self.run_cmd("--help")
    self.assertEqual(res.returncode, 0)
    self.assertIn("Usage: scripts/mailbox_bazel_test.sh", res.stdout)
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


class TestMailboxPresubmitPRCLI(unittest.TestCase):
  """Tests command-line handling of scripts/mailbox_presubmit_pr.sh."""

  def run_cmd(self, *args, **env_overrides):
    env = dict(os.environ)
    env.update(env_overrides)
    return subprocess.run(
        ["bash", MAILBOX_PRESUBMIT_PR, *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

  def test_help_flag(self):
    res = self.run_cmd("--help")
    self.assertEqual(res.returncode, 0)
    self.assertIn("Usage: scripts/mailbox_presubmit_pr.sh", res.stdout)
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


  def test_config_flag_precedes_jobs_flags_in_scripts(self):
    for script_name in ["mailbox_bazel_test.sh", "mailbox_presubmit_pr.sh"]:
      script_path = os.path.join(REPO_ROOT, "scripts", script_name)
      with open(script_path) as f:
        body = f.read()
      config_idx = body.find("--config=${CLI_BAZEL_CONFIG}")
      jobs_idx = body.find("--jobs=${CLI_JOBS}")
      local_jobs_idx = body.find("--local_test_jobs=${CLI_JOBS}")
      self.assertGreater(config_idx, 0, f"--config missing in {script_name}")
      self.assertGreater(jobs_idx, config_idx, f"--jobs must follow --config in {script_name}")
      self.assertGreater(local_jobs_idx, config_idx, f"--local_test_jobs must follow --config in {script_name}")

  def test_stage_timing_breakdown_in_mailbox_bazel_test(self):
    script_path = os.path.join(REPO_ROOT, "scripts", "mailbox_bazel_test.sh")
    with open(script_path) as f:
      body = f.read()
    self.assertIn("=== Stage Timing Breakdown ===", body)
    self.assertIn("FLEET_CHECK_ELAPSED", body)
    self.assertIn("QUERY_ELAPSED", body)
    self.assertIn("BAZEL_CRITICAL_PATH", body)


if __name__ == "__main__":
  unittest.main()
