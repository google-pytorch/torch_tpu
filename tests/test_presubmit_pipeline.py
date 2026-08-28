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

"""test_presubmit_pipeline.py

Unit, integration, and pipeline test suite for generate_presubmit_report.py
and run_presubmit_v5_relay.sh reporting capabilities.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTER_SCRIPT = os.path.join(
    REPO_ROOT, "scripts", "generate_presubmit_report.py"
)
RUNNER_SCRIPT = os.path.join(REPO_ROOT, "scripts", "run_presubmit_v5_relay.sh")

sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from generate_presubmit_report import (
    collect_xml_paths,
    extract_diagnostics,
    parse_bazel_statuses,
    parse_target_reports,
    summarize_pool,
    parse_xml_report,
)


class PresubmitPipelineReporterTest(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests report generation.
):

  def setUp(self):
    self.td = tempfile.TemporaryDirectory()
    self.tmp_dir = self.td.name
    self.testlogs_dir = os.path.join(self.tmp_dir, "bazel-testlogs")
    self.output_dir = os.path.join(self.tmp_dir, "reports")
    self.targets_file = os.path.join(self.tmp_dir, "targets.txt")
    self.session_env = os.path.join(self.tmp_dir, "session.env")

    os.makedirs(self.testlogs_dir, exist_ok=True)
    os.makedirs(self.output_dir, exist_ok=True)

    with open(self.session_env, "w") as f:
      f.write("""export TPU_NAME="test-spot-vm-1"
export TPU_ZONE="europe-west4-b"
export TPU_PROJECT="rbe-tpu-oss"
""")

  def tearDown(self):
    self.td.cleanup()

  def _create_mock_testlog(
      self,
      target: str,
      passes: bool,
      error_msg: str = "",
      log_text: str = "",
      duration: float = 1.5,
  ):
    clean = target.lstrip("/").replace(":", "/")
    target_dir = os.path.join(self.testlogs_dir, clean)
    os.makedirs(target_dir, exist_ok=True)

    xml_path = os.path.join(target_dir, "test.xml")
    log_path = os.path.join(target_dir, "test.log")

    if passes:
      xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="{target}" tests="2" failures="0" errors="0" time="{duration}">
    <testcase classname="{target}" name="test1" time="0.7"/>
    <testcase classname="{target}" name="test2" time="0.8"/>
  </testsuite>
</testsuites>
"""
    else:
      xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="{target}" tests="2" failures="1" errors="0" time="{duration}">
    <testcase classname="{target}" name="test1" time="0.7"/>
    <testcase classname="{target}" name="test2" time="0.8">
      <failure message="{error_msg}"><![CDATA[{log_text}]]></failure>
    </testcase>
  </testsuite>
</testsuites>
"""
    with open(xml_path, "w") as f:
      f.write(xml_content)

    with open(log_path, "w") as f:
      f.write(log_text or "Default test execution log.")

  def test_all_passing_report_generation(self):
    targets = ["//tests:empty_test", "//tests:fused_adam_test"]
    with open(self.targets_file, "w") as f:
      f.write("\n".join(targets))

    for t in targets:
      self._create_mock_testlog(t, passes=True)

    cmd = [
        sys.executable,
        REPORTER_SCRIPT,
        f"--testlogs-dir={self.testlogs_dir}",
        f"--targets-file={self.targets_file}",
        f"--session-env={self.session_env}",
        f"--output-dir={self.output_dir}",
        "--duration=12.5",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    self.assertEqual(proc.returncode, 0, proc.stderr)

    summary_file = os.path.join(self.output_dir, "presubmit_summary.json")
    report_file = os.path.join(self.output_dir, "presubmit_report.md")
    self.assertTrue(os.path.isfile(summary_file))
    self.assertTrue(os.path.isfile(report_file))

    with open(summary_file, "r") as f:
      data = json.load(f)

    self.assertEqual(data["status"], "PASSED")
    self.assertEqual(data["total_targets"], 2)
    self.assertEqual(data["targets_total"], 2)
    self.assertEqual(data["passed_targets"], 2)
    self.assertEqual(data["failed_targets"], 0)
    self.assertEqual(data["project"], "rbe-tpu-oss")
    self.assertEqual(data["tpu_vm"], "test-spot-vm-1")
    self.assertEqual(len(data["targets"]), 2)

    with open(report_file, "r") as f:
      md = f.read()

    self.assertIn("🟢 PASSED", md)
    self.assertIn("100.0%", md)
    self.assertNotIn("Failure Diagnostics", md)

  def test_failing_target_diagnostic_traceback(self):
    targets = ["//tests:empty_test", "//tests:broken_test"]
    with open(self.targets_file, "w") as f:
      f.write("\n".join(targets))

    self._create_mock_testlog("//tests:empty_test", passes=True)

    traceback_log = """
[INFO] Starting test execution
Traceback (most recent call last):
  File "tests/broken_test.py", line 45, in test_mismatch
    assert a == b
AssertionError: 1 != 2
"""
    self._create_mock_testlog(
        "//tests:broken_test",
        passes=False,
        error_msg="AssertionError: 1 != 2",
        log_text=traceback_log,
    )

    cmd = [
        sys.executable,
        REPORTER_SCRIPT,
        f"--testlogs-dir={self.testlogs_dir}",
        f"--targets-file={self.targets_file}",
        f"--session-env={self.session_env}",
        f"--output-dir={self.output_dir}",
        "--duration=18.3",
        "--bazel-exit-code=3",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    self.assertEqual(proc.returncode, 1)

    summary_file = os.path.join(self.output_dir, "presubmit_summary.json")
    report_file = os.path.join(self.output_dir, "presubmit_report.md")

    with open(summary_file, "r") as f:
      data = json.load(f)

    self.assertEqual(data["status"], "FAILED")
    self.assertEqual(data["passed_targets"], 1)
    self.assertEqual(data["failed_targets"], 1)

    broken_target = [
        t for t in data["targets"] if t["target"] == "//tests:broken_test"
    ][0]
    self.assertEqual(broken_target["status"], "FAILED")
    self.assertIn("AssertionError: 1 != 2", broken_target["error_message"])
    self.assertIn(
        "Traceback (most recent call last):", broken_target["failure_details"]
    )

    with open(report_file, "r") as f:
      md = f.read()

    self.assertIn("🔴 FAILED", md)
    self.assertIn("## Failure Diagnostics", md)
    self.assertIn("### ❌ `//tests:broken_test`", md)
    self.assertIn("AssertionError: 1 != 2", md)

  def test_diagnostic_extraction_sigsegv_fatal(self):
    targets = ["//tests:crash_test"]
    with open(self.targets_file, "w") as f:
      f.write("\n".join(targets))

    crash_log = """
[INFO] Initializing runtime
F20260910 23:10:00.123456 1234 torch_tpu.cc:42] Check failed: tensor.is_valid()
*** SIGSEGV (@0x0) received by PID 1234 ***
    @ 0x7f89abcd0000 (unknown)
    @ 0x7f89abcd0100 (unknown)
"""
    self._create_mock_testlog(
        "//tests:crash_test",
        passes=False,
        error_msg="Fatal C++ check failure",
        log_text=crash_log,
    )

    cmd = [
        sys.executable,
        REPORTER_SCRIPT,
        f"--testlogs-dir={self.testlogs_dir}",
        f"--targets-file={self.targets_file}",
        f"--session-env={self.session_env}",
        f"--output-dir={self.output_dir}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    self.assertEqual(proc.returncode, 1)

    summary_file = os.path.join(self.output_dir, "presubmit_summary.json")
    with open(summary_file, "r") as f:
      data = json.load(f)

    target_info = data["targets"][0]
    self.assertIn("Check failed:", target_info["failure_details"])
    self.assertIn("SIGSEGV", target_info["failure_details"])

  def test_diagnostic_extraction_tpu_device_error(self):
    targets = ["//tests:tpu_busy_test"]
    with open(self.targets_file, "w") as f:
      f.write("\n".join(targets))

    tpu_error_log = """
[INFO] Attempting hardware initialization
RuntimeError: InitializePjrtPlugin failed: The TPU is already in use by process 9999
Failed to open /dev/vfio/0: Device or resource busy
"""
    self._create_mock_testlog(
        "//tests:tpu_busy_test",
        passes=False,
        error_msg="TPU in use",
        log_text=tpu_error_log,
    )

    cmd = [
        sys.executable,
        REPORTER_SCRIPT,
        f"--testlogs-dir={self.testlogs_dir}",
        f"--targets-file={self.targets_file}",
        f"--session-env={self.session_env}",
        f"--output-dir={self.output_dir}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    self.assertEqual(proc.returncode, 1)

    summary_file = os.path.join(self.output_dir, "presubmit_summary.json")
    with open(summary_file, "r") as f:
      data = json.load(f)

    self.assertIn(
        "InitializePjrtPlugin failed", data["targets"][0]["failure_details"]
    )

  def test_diagnostic_extraction_relay_runner_error(self):
    targets = ["//tests:preemption_test"]
    with open(self.targets_file, "w") as f:
      f.write("\n".join(targets))

    relay_log = """
[relay] Starting test action
ERROR [relay_test_runner]: Preemption detected for Spot TPU VM spot-tpu-v5e-123
GCP instance state: PREEMPTED
"""
    self._create_mock_testlog(
        "//tests:preemption_test",
        passes=False,
        error_msg="Spot preemption",
        log_text=relay_log,
    )

    cmd = [
        sys.executable,
        REPORTER_SCRIPT,
        f"--testlogs-dir={self.testlogs_dir}",
        f"--targets-file={self.targets_file}",
        f"--session-env={self.session_env}",
        f"--output-dir={self.output_dir}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    self.assertEqual(proc.returncode, 1)

    summary_file = os.path.join(self.output_dir, "presubmit_summary.json")
    with open(summary_file, "r") as f:
      data = json.load(f)

    self.assertIn(
        "ERROR [relay_test_runner]: Preemption detected",
        data["targets"][0]["failure_details"],
    )

  def test_ansi_escape_code_stripping(self):
    targets = ["//tests:color_test"]
    with open(self.targets_file, "w") as f:
      f.write("\n".join(targets))

    ansi_log = (
        "\x1b[31mTraceback (most recent call last):\x1b[0m\n  \x1b[33mFile"
        " 'test.py', line 1\x1b[0m\n\x1b[31mAssertionError: colored\x1b[0m"
    )
    self._create_mock_testlog(
        "//tests:color_test",
        passes=False,
        error_msg="Colored error",
        log_text=ansi_log,
    )

    cmd = [
        sys.executable,
        REPORTER_SCRIPT,
        f"--testlogs-dir={self.testlogs_dir}",
        f"--targets-file={self.targets_file}",
        f"--session-env={self.session_env}",
        f"--output-dir={self.output_dir}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    self.assertEqual(proc.returncode, 1)

    summary_file = os.path.join(self.output_dir, "presubmit_summary.json")
    with open(summary_file, "r") as f:
      data = json.load(f)

    details = data["targets"][0]["failure_details"]
    self.assertNotIn("\x1b[", details)
    self.assertIn("AssertionError: colored", details)

  def test_missing_xml_build_failure(self):
    targets = ["//tests:unbuilt_test"]
    with open(self.targets_file, "w") as f:
      f.write("\n".join(targets))

    # Do not create test.xml or test.log
    cmd = [
        sys.executable,
        REPORTER_SCRIPT,
        f"--testlogs-dir={self.testlogs_dir}",
        f"--targets-file={self.targets_file}",
        f"--session-env={self.session_env}",
        f"--output-dir={self.output_dir}",
        "--bazel-exit-code=1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    self.assertEqual(proc.returncode, 1)

    summary_file = os.path.join(self.output_dir, "presubmit_summary.json")
    with open(summary_file, "r") as f:
      data = json.load(f)

    self.assertEqual(data["status"], "FAILED")
    self.assertEqual(data["failed_targets"], 1)
    # Bazel printed no status line for this target, so we cannot say it failed
    # to build; all we know is that nothing reported on it.
    self.assertEqual(data["targets"][0]["status"], "NO_STATUS")

  def test_dry_run_mode(self):
    targets = ["//tests:test_a", "//tests:test_b"]
    with open(self.targets_file, "w") as f:
      f.write("\n".join(targets))

    cmd = [
        sys.executable,
        REPORTER_SCRIPT,
        f"--testlogs-dir={self.testlogs_dir}",
        f"--targets-file={self.targets_file}",
        f"--session-env={self.session_env}",
        f"--output-dir={self.output_dir}",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    self.assertEqual(proc.returncode, 0)

    summary_file = os.path.join(self.output_dir, "presubmit_summary.json")
    report_file = os.path.join(self.output_dir, "presubmit_report.md")
    with open(summary_file, "r") as f:
      data = json.load(f)

    self.assertEqual(data["status"], "DRY_RUN")
    self.assertEqual(data["total_targets"], 2)
    # Nothing ran, so nothing passed. Counting these would report a 100% pass
    # rate for a run that provisioned no hardware.
    self.assertEqual(data["passed_targets"], 0)
    self.assertEqual(data["failed_targets"], 0)
    self.assertEqual(
        [t["status"] for t in data["targets"]], ["DRY_RUN", "DRY_RUN"]
    )

    with open(report_file, "r") as f:
      md = f.read()
    self.assertIn("⚪ DRY_RUN", md)

  def test_json_and_markdown_schema_conformance(self):
    targets = ["//tests:empty_test"]
    with open(self.targets_file, "w") as f:
      f.write("\n".join(targets))
    self._create_mock_testlog("//tests:empty_test", passes=True, duration=2.4)

    cmd = [
        sys.executable,
        REPORTER_SCRIPT,
        f"--testlogs-dir={self.testlogs_dir}",
        f"--targets-file={self.targets_file}",
        f"--session-env={self.session_env}",
        f"--output-dir={self.output_dir}",
        "--duration=5.0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    self.assertEqual(proc.returncode, 0)

    summary_file = os.path.join(self.output_dir, "presubmit_summary.json")
    with open(summary_file, "r") as f:
      data = json.load(f)

    required_top_keys = [
        "schema_version",
        "timestamp",
        "duration_seconds",
        "execution_time",
        "project",
        "tpu_zone",
        "tpu_vm",
        "status",
        "total_targets",
        "targets_total",
        "passed_targets",
        "targets_passed",
        "failed_targets",
        "targets_failed",
        "skipped_targets",
        "targets_skipped",
        "targets",
    ]
    for k in required_top_keys:
      self.assertIn(k, data, f"Missing required JSON key: {k}")

    target_rec = data["targets"][0]
    required_target_keys = [
        "target",
        "status",
        "duration_seconds",
        "exit_code",
        "tests_count",
        "failures_count",
        "errors_count",
        "skipped_count",
        "log_file",
        "xml_file",
        "error_message",
        "failure_details",
    ]
    for k in required_target_keys:
      self.assertIn(k, target_rec, f"Missing required target JSON key: {k}")

    self.assertIsInstance(data["duration_seconds"], float)
    self.assertIsInstance(data["total_targets"], int)
    self.assertIsInstance(data["targets"], list)

  def test_dynamic_discovery_from_bazel_log(self):
    bazel_log = os.path.join(self.tmp_dir, "bazel.log")
    with open(bazel_log, "w") as f:
      f.write("""
INFO: Running tests...
//tests:alpha_test PASSED in 1.1s
//tests:beta_test FAILED in 2.2s
INFO: Build completed.
""")

    self._create_mock_testlog("//tests:alpha_test", passes=True)
    self._create_mock_testlog(
        "//tests:beta_test", passes=False, error_msg="Assertion failed"
    )

    cmd = [
        sys.executable,
        REPORTER_SCRIPT,
        f"--testlogs-dir={self.testlogs_dir}",
        f"--bazel-log={bazel_log}",
        f"--session-env={self.session_env}",
        f"--output-dir={self.output_dir}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    self.assertEqual(proc.returncode, 1)

    summary_file = os.path.join(self.output_dir, "presubmit_summary.json")
    with open(summary_file, "r") as f:
      data = json.load(f)

    target_names = [t["target"] for t in data["targets"]]
    self.assertIn("//tests:alpha_test", target_names)
    self.assertIn("//tests:beta_test", target_names)

  def test_dynamic_discovery_from_testlogs_directory(self):
    self._create_mock_testlog("//tests/compile:dtensor_test", passes=True)
    self._create_mock_testlog("//tests:patch_test", passes=True)

    cmd = [
        sys.executable,
        REPORTER_SCRIPT,
        f"--testlogs-dir={self.testlogs_dir}",
        f"--session-env={self.session_env}",
        f"--output-dir={self.output_dir}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    self.assertEqual(proc.returncode, 0)

    summary_file = os.path.join(self.output_dir, "presubmit_summary.json")
    with open(summary_file, "r") as f:
      data = json.load(f)

    self.assertEqual(data["total_targets"], 2)
    self.assertEqual(data["passed_targets"], 2)

  def test_xml_parsing_empty_attributes(self):
    targets = ["//tests:empty_attr_test"]
    with open(self.targets_file, "w") as f:
      f.write("\n".join(targets))

    target_dir = os.path.join(self.testlogs_dir, "tests", "empty_attr_test")
    os.makedirs(target_dir, exist_ok=True)
    xml_path = os.path.join(target_dir, "test.xml")
    log_path = os.path.join(target_dir, "test.log")

    with open(xml_path, "w") as f:
      f.write("""<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="//tests:empty_attr_test" tests="" failures="" errors="" skipped="" time="">
  <testcase classname="//tests:empty_attr_test" name="test_dummy"/>
</testsuite>
""")
    with open(log_path, "w") as f:
      f.write("Empty attributes test completed successfully.")

    res = parse_xml_report(xml_path)
    self.assertIsNotNone(res)
    self.assertIsInstance(res["tests"], int)
    self.assertIsInstance(res["failures"], int)
    self.assertIsInstance(res["errors"], int)
    self.assertIsInstance(res["skipped"], int)
    self.assertIsInstance(res["time"], float)
    self.assertGreaterEqual(res["tests"], 1)
    self.assertEqual(res["failures"], 0)
    self.assertEqual(res["errors"], 0)

    cmd = [
        sys.executable,
        REPORTER_SCRIPT,
        f"--testlogs-dir={self.testlogs_dir}",
        f"--targets-file={self.targets_file}",
        f"--session-env={self.session_env}",
        f"--output-dir={self.output_dir}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    self.assertEqual(
        proc.returncode,
        0,
        f"Expected clean exit 0, got {proc.returncode}: {proc.stderr}",
    )

    summary_file = os.path.join(self.output_dir, "presubmit_summary.json")
    self.assertTrue(os.path.isfile(summary_file))
    with open(summary_file, "r") as f:
      data = json.load(f)
    self.assertEqual(data["status"], "PASSED")
    self.assertEqual(data["passed_targets"], 1)
    self.assertEqual(data["failed_targets"], 0)
    target_rec = data["targets"][0]
    self.assertIsInstance(target_rec["tests_count"], int)
    self.assertIsInstance(target_rec["failures_count"], int)

  def test_xml_parsing_non_numeric_attributes(self):
    targets = ["//tests:invalid_attr_test"]
    with open(self.targets_file, "w") as f:
      f.write("\n".join(targets))

    target_dir = os.path.join(self.testlogs_dir, "tests", "invalid_attr_test")
    os.makedirs(target_dir, exist_ok=True)
    xml_path = os.path.join(target_dir, "test.xml")
    log_path = os.path.join(target_dir, "test.log")

    with open(xml_path, "w") as f:
      f.write("""<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="//tests:invalid_attr_test" tests="invalid" failures="NaN" errors="none" skipped="disabled" time="timeout">
  <testcase classname="//tests:invalid_attr_test" name="test_invalid"/>
</testsuite>
""")
    with open(log_path, "w") as f:
      f.write("Non-numeric attributes test log.")

    res = parse_xml_report(xml_path)
    self.assertIsNotNone(res)
    self.assertIsInstance(res["tests"], int)
    self.assertIsInstance(res["failures"], int)
    self.assertIsInstance(res["errors"], int)
    self.assertIsInstance(res["skipped"], int)
    self.assertIsInstance(res["time"], float)
    self.assertGreaterEqual(res["tests"], 1)
    self.assertEqual(res["failures"], 0)

    cmd = [
        sys.executable,
        REPORTER_SCRIPT,
        f"--testlogs-dir={self.testlogs_dir}",
        f"--targets-file={self.targets_file}",
        f"--session-env={self.session_env}",
        f"--output-dir={self.output_dir}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    self.assertEqual(
        proc.returncode,
        0,
        f"Expected clean exit 0, got {proc.returncode}: {proc.stderr}",
    )

    summary_file = os.path.join(self.output_dir, "presubmit_summary.json")
    with open(summary_file, "r") as f:
      data = json.load(f)
    self.assertEqual(data["status"], "PASSED")
    self.assertEqual(data["passed_targets"], 1)

  def test_xml_parsing_missing_attributes_with_child_fallback(self):
    targets = ["//tests:missing_attr_test"]
    with open(self.targets_file, "w") as f:
      f.write("\n".join(targets))

    target_dir = os.path.join(self.testlogs_dir, "tests", "missing_attr_test")
    os.makedirs(target_dir, exist_ok=True)
    xml_path = os.path.join(target_dir, "test.xml")
    log_path = os.path.join(target_dir, "test.log")

    with open(xml_path, "w") as f:
      f.write("""<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="//tests:missing_attr_test">
    <testcase classname="//tests:missing_attr_test" name="test_ok" time="0.5"/>
    <testcase classname="//tests:missing_attr_test" name="test_failing" time="0.6">
      <failure message="Expected True but got False">Traceback: assert False</failure>
    </testcase>
  </testsuite>
</testsuites>
""")
    with open(log_path, "w") as f:
      f.write("Traceback: assert False")

    res = parse_xml_report(xml_path)
    self.assertIsNotNone(res)
    self.assertEqual(res["tests"], 2)
    self.assertEqual(res["failures"], 1)
    self.assertEqual(res["errors"], 0)
    self.assertEqual(res["error_message"], "Expected True but got False")

    cmd = [
        sys.executable,
        REPORTER_SCRIPT,
        f"--testlogs-dir={self.testlogs_dir}",
        f"--targets-file={self.targets_file}",
        f"--session-env={self.session_env}",
        f"--output-dir={self.output_dir}",
        "--bazel-exit-code=3",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    self.assertEqual(proc.returncode, 1)

    summary_file = os.path.join(self.output_dir, "presubmit_summary.json")
    with open(summary_file, "r") as f:
      data = json.load(f)
    self.assertEqual(data["status"], "FAILED")
    self.assertEqual(data["failed_targets"], 1)
    target_rec = data["targets"][0]
    self.assertEqual(target_rec["tests_count"], 2)
    self.assertEqual(target_rec["failures_count"], 1)
    self.assertIn("Expected True but got False", target_rec["error_message"])

  def test_xml_parsing_nested_testsuites_mixed_malformed_attributes(self):
    targets = ["//tests:multi_suite_test"]
    with open(self.targets_file, "w") as f:
      f.write("\n".join(targets))

    target_dir = os.path.join(self.testlogs_dir, "tests", "multi_suite_test")
    os.makedirs(target_dir, exist_ok=True)
    xml_path = os.path.join(target_dir, "test.xml")
    log_path = os.path.join(target_dir, "test.log")

    with open(xml_path, "w") as f:
      f.write("""<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="suite1" tests="" failures="" errors="" skipped=""/>
  <testsuite name="suite2" tests="invalid" failures="bad"/>
  <testsuite name="suite3">
    <testcase name="c1"/>
    <testcase name="c2"/>
  </testsuite>
  <testsuite name="suite4" tests="3" failures="0" errors="0" skipped="1" time="1.2">
    <testcase name="c3"/>
    <testcase name="c4"/>
    <testcase name="c5"><skipped/></testcase>
  </testsuite>
</testsuites>
""")
    with open(log_path, "w") as f:
      f.write("Multi-suite test log.")

    res = parse_xml_report(xml_path)
    self.assertIsNotNone(res)
    self.assertGreaterEqual(res["tests"], 5)
    self.assertEqual(res["failures"], 0)
    self.assertEqual(res["errors"], 0)
    self.assertGreaterEqual(res["skipped"], 1)

    cmd = [
        sys.executable,
        REPORTER_SCRIPT,
        f"--testlogs-dir={self.testlogs_dir}",
        f"--targets-file={self.targets_file}",
        f"--session-env={self.session_env}",
        f"--output-dir={self.output_dir}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    self.assertEqual(
        proc.returncode,
        0,
        f"Expected clean exit 0, got {proc.returncode}: {proc.stderr}",
    )


class PipelineMockToolchainTestCase(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """Runs the relay driver against stub tools instead of real hardware."""

  def setUp(self):
    self.td = tempfile.TemporaryDirectory()
    self.tmp_dir = self.td.name
    self.output_dir = os.path.join(self.tmp_dir, "e2e_reports")

  def tearDown(self):
    self.td.cleanup()

  def _write_exe(self, path, body):
    with open(path, "w") as f:
      f.write("#!/usr/bin/env bash\n" + body)
    os.chmod(path, 0o755)

  def _mock_env(self, name, bazel_body):
    """Builds a fake toolchain the driver can run against with no hardware.

    Stubs bazel, the TPU manager, and the base-cache stager. The stager is
    stubbed because the real one opens an SSH session to the address in the
    session file, and the mock VM's address goes nowhere.
    """
    mock_bin = os.path.join(self.tmp_dir, name)
    os.makedirs(mock_bin, exist_ok=True)
    session_env = os.path.join(self.tmp_dir, f"{name}_session.env")

    self._write_exe(os.path.join(mock_bin, "bazel"), bazel_body)

    mock_mgr = os.path.join(mock_bin, "spot_tpu_manager.sh")
    self._write_exe(
        mock_mgr,
        f"""
case "${{1:-}}" in
  up)
    cat << 'SESSION_EOF' > "{session_env}"
export TPU_NAME='mock-e2e-vm'
export TPU_ZONE='europe-west4-b'
export TPU_PROJECT='rbe-tpu-oss'
export TPU_IP='10.0.0.1'
export SSH_USER='mockuser'
export SSH_CONTROL_PATH='{os.path.join(self.tmp_dir, "mock_ctl")}'
SESSION_EOF
    ;;
  down)   rm -f "{session_env}" ;;
  status) exit 1 ;;
esac
exit 0
""",
    )

    mock_stager = os.path.join(mock_bin, "stage_relay_base.sh")
    self._write_exe(
        mock_stager, 'echo "[stage_relay_base] mock: up to date"\nexit 0\n'
    )

    env = os.environ.copy()
    env["PATH"] = f"{mock_bin}:{env.get('PATH', '')}"
    env["SPOT_TPU_MANAGER_BIN"] = mock_mgr
    env["STAGE_RELAY_BASE_BIN"] = mock_stager
    env["TPU_SESSION_ENV"] = session_env
    return env


class PresubmitPipelineEndToEndTest(PipelineMockToolchainTestCase):

  def test_pipeline_dry_run_e2e(self):
    cmd = [
        RUNNER_SCRIPT,
        "--dry-run",
        f"--output-dir={self.output_dir}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    self.assertEqual(proc.returncode, 0, proc.stderr)
    self.assertIn(
        "torch_tpu Presubmit-v5 Relay Execution Pipeline", proc.stdout
    )
    self.assertIn("--local_test_jobs=1", proc.stdout)

    summary_file = os.path.join(self.output_dir, "presubmit_summary.json")
    report_file = os.path.join(self.output_dir, "presubmit_report.md")
    self.assertTrue(os.path.isfile(summary_file))
    self.assertTrue(os.path.isfile(report_file))

    with open(summary_file, "r") as f:
      data = json.load(f)

    self.assertEqual(data["status"], "DRY_RUN")
    self.assertEqual(data["project"], "rbe-tpu-oss")

    # The literal count moves whenever anyone adds a tagged test, so check the
    # things that must hold instead: the banner agrees with the summary, and
    # the suite is not empty.
    count_match = re.search(r"Target Count:\s+(\d+)", proc.stdout)
    self.assertIsNotNone(count_match, proc.stdout)
    self.assertEqual(int(count_match.group(1)), data["total_targets"])
    self.assertGreater(data["total_targets"], 0)

    # One chip per VM, so anything asking for a full 8-chip pod must be
    # filtered out before bazel ever sees it.
    self.assertIn("-requires-tpu-v5lite:8", proc.stdout)
    resolved = {t["target"] for t in data["targets"]}
    self.assertNotIn("//tests/distributed:torchcomm_multi_tpu_test", resolved)

    # The plan is generated from the same list main() passes to bazel, so this
    # also proves the real invocation keeps compilation eligible for RBE.
    self.assertIn("--strategy=TestRunner=local", proc.stdout)
    self.assertNotIn("--spawn_strategy", proc.stdout)

  def test_pipeline_mock_execution_pass(self):
    env = self._mock_env(
        "mock_bin",
        """
if [[ "$1" == "query" ]]; then
  echo "//tests:empty_test"
  exit 0
fi
echo "//tests:empty_test PASSED in 0.4s"
exit 0
""",
    )

    cmd = [
        RUNNER_SCRIPT,
        f"--output-dir={self.output_dir}",
        "--targets=//tests:empty_test",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=REPO_ROOT, env=env
    )
    self.assertEqual(proc.returncode, 0, proc.stderr)

    summary_file = os.path.join(self.output_dir, "presubmit_summary.json")
    self.assertTrue(os.path.isfile(summary_file))
    with open(summary_file, "r") as f:
      data = json.load(f)
    self.assertEqual(data["status"], "PASSED")
    self.assertEqual(data["passed_targets"], 1)

  def test_pipeline_mock_execution_fail(self):
    env = self._mock_env(
        "mock_bin_fail",
        """
if [[ "$1" == "query" ]]; then
  echo "//tests:broken_test"
  exit 0
fi
if [[ "$1" == "build" ]]; then
  exit 0
fi
echo "//tests:broken_test FAILED in 0.5s" >&2
exit 3
""",
    )

    fail_out_dir = os.path.join(self.tmp_dir, "e2e_fail_reports")
    cmd = [
        RUNNER_SCRIPT,
        f"--output-dir={fail_out_dir}",
        "--targets=//tests:broken_test",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=REPO_ROOT, env=env
    )
    self.assertEqual(proc.returncode, 3)

    summary_file = os.path.join(fail_out_dir, "presubmit_summary.json")
    self.assertTrue(os.path.isfile(summary_file))
    with open(summary_file, "r") as f:
      data = json.load(f)
    self.assertEqual(data["status"], "FAILED")
    self.assertEqual(data["failed_targets"], 1)


class BazelStatusParsingTest(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests report generation.
):
  """Bazel's own summary is the only record of what ran in this invocation."""

  def setUp(self):
    self.td = tempfile.TemporaryDirectory()
    self.addCleanup(self.td.cleanup)

  def write_log(self, text):
    path = os.path.join(self.td.name, "bazel.log")
    with open(path, "w", encoding="utf-8") as fh:
      fh.write(text)
    return path

  def test_reads_the_padded_summary_column(self):
    """Bazel pads the status column, so target and verdict aren't adjacent."""
    log = self.write_log(
        "//tests:gru_test                          PASSED in 27.9s\n"
        "//tests:tpu_errors_test                   FAILED in 57.3s\n"
    )
    self.assertEqual(
        parse_bazel_statuses(log),
        {"//tests:gru_test": "PASSED", "//tests:tpu_errors_test": "FAILED"},
    )

  def test_reads_partial_shard_failures(self):
    log = self.write_log(
        "//tests:errors_test_tpu     FAILED in 3 out of 4 in 32.9s\n"
    )
    self.assertEqual(
        parse_bazel_statuses(log), {"//tests:errors_test_tpu": "FAILED"}
    )

  def test_reads_cached_and_timed_out_and_statusless_targets(self):
    log = self.write_log(
        "//tests:a          (cached) PASSED in 0.0s\n"
        "//tests:b          TIMEOUT in 900.0s\n"
        "//tests:c          NO STATUS\n"
    )
    self.assertEqual(
        parse_bazel_statuses(log),
        {
            "//tests:a": "PASSED",
            "//tests:b": "TIMEOUT",
            "//tests:c": "NO STATUS",
        },
    )

  def test_ignores_progress_lines_that_mention_a_target(self):
    log = self.write_log(
        "[40,924 / 41,219] 15 / 57 tests; Testing //tests:gru_test; 159s\r"
        "//tests:gru_test        PASSED in 27.9s\n"
    )
    self.assertEqual(parse_bazel_statuses(log), {"//tests:gru_test": "PASSED"})

  def test_a_missing_log_is_not_an_error(self):
    self.assertEqual(parse_bazel_statuses("/nonexistent/bazel.log"), {})


class ShardedReportCollectionTest(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests report generation.
):
  """A sharded target writes shard_N_of_M/test.xml and nothing at the top.

  Looking only for test.xml found nothing for every sharded target, and the
  reporter called that a build failure. It turned a 53/57 green run into a
  claimed 43/14.
  """

  SUITE = (
      '<?xml version="1.0" encoding="UTF-8"?>'
      '<testsuites><testsuite name="t" tests="{n}" failures="{f}" errors="0" '
      'time="1.0">{cases}</testsuite></testsuites>'
  )

  def setUp(self):
    self.td = tempfile.TemporaryDirectory()
    self.addCleanup(self.td.cleanup)
    self.logs = os.path.join(self.td.name, "bazel-testlogs")

  def write_shard(self, target_dir, shard, tests=2, failures=0, age=0.0):
    path = (
        os.path.join(self.logs, target_dir, shard)
        if shard
        else os.path.join(self.logs, target_dir)
    )
    os.makedirs(path, exist_ok=True)
    xml = os.path.join(path, "test.xml")
    cases = "".join(
        '<testcase name="c%d" time="0.5">%s</testcase>'
        % (i, '<failure message="boom"></failure>' if i < failures else "")
        for i in range(tests)
    )
    with open(xml, "w", encoding="utf-8") as fh:
      fh.write(self.SUITE.format(n=tests, f=failures, cases=cases))
    if age:
      os.utime(xml, (age, age))
    return xml

  def test_finds_every_shard(self):
    for i in range(1, 5):
      self.write_shard("tests/gru_test", f"shard_{i}_of_4")
    paths, stale = collect_xml_paths(self.logs, "//tests:gru_test")
    self.assertEqual(len(paths), 4)
    self.assertEqual(stale, 0)

  def test_finds_an_unsharded_report(self):
    self.write_shard("tests/amp_test", None)
    paths, _ = collect_xml_paths(self.logs, "//tests:amp_test")
    self.assertEqual(len(paths), 1)

  def test_reports_from_an_earlier_run_are_not_counted(self):
    """bazel-testlogs survives across runs; last week's green is not today's."""
    self.write_shard("tests/gru_test", "shard_1_of_2", age=1000)
    self.write_shard("tests/gru_test", "shard_2_of_2")
    paths, stale = collect_xml_paths(
        self.logs, "//tests:gru_test", min_mtime=5000
    )
    self.assertEqual(len(paths), 1)
    self.assertEqual(stale, 1)

  def test_a_target_that_never_ran_yields_nothing(self):
    paths, stale = collect_xml_paths(self.logs, "//tests:absent")
    self.assertEqual((paths, stale), ([], 0))

  def test_shard_counts_are_summed(self):
    for i in range(1, 4):
      self.write_shard(
          "tests/ops_test", f"shard_{i}_of_3", tests=10, failures=1
      )
    paths, _ = collect_xml_paths(self.logs, "//tests:ops_test")
    totals = parse_target_reports(paths)
    self.assertEqual(totals["tests"], 30)
    self.assertEqual(totals["failures"], 3)


class DiagnosticsOrderingTest(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests report generation.
):
  """An infra fault explains the traceback under it, so report it instead."""

  def setUp(self):
    self.td = tempfile.TemporaryDirectory()
    self.addCleanup(self.td.cleanup)

  def write_log(self, text):
    path = os.path.join(self.td.name, "test.log")
    with open(path, "w", encoding="utf-8") as fh:
      fh.write(text)
    return path

  def test_an_infra_fault_wins_over_the_traceback_it_caused(self):
    log = self.write_log(
        "ERROR [relay_test_runner]: Cannot reach TPU VM at 10.0.0.1\n"
        "Traceback (most recent call last):\n"
        "  File 'x.py', line 1\n"
        "RuntimeError: downstream noise\n"
    )
    self.assertIn("Cannot reach TPU VM", extract_diagnostics(log))

  def test_the_first_traceback_wins(self):
    """Chained exceptions print the root cause first, the handler's last."""
    log = self.write_log(
        "Traceback (most recent call last):\n"
        "ModuleNotFoundError: No module named 'torch_tpu'\n"
        "During handling of the above exception, another exception occurred:\n"
        "Traceback (most recent call last):\n"
        "AttributeError: downstream symptom\n"
    )
    diag = extract_diagnostics(log)
    self.assertIn("No module named 'torch_tpu'", diag)


if __name__ == "__main__":
  unittest.main(verbosity=2)


class PoolSessionSummaryTest(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests report generation.
):
  """A fleet run has no single session file.

  Every pool report used to say the VM was "unknown", which is precisely the
  field you reach for when one bad VM is failing shards faster than the healthy
  ones can pick them up.
  """

  def setUp(self):
    self.td = tempfile.TemporaryDirectory()
    self.addCleanup(self.td.cleanup)
    self.pool = os.path.join(self.td.name, "pool")
    os.makedirs(self.pool)

  def write_session(self, slot, name, zone="europe-west4-b"):
    path = os.path.join(self.pool, f"vm_{slot}.env")
    with open(path, "w") as f:
      f.write(
          f'export TPU_NAME="{name}"\n'
          f'export TPU_ZONE="{zone}"\n'
          'export TPU_PROJECT="rbe-tpu-oss"\n'
      )
    return path

  def test_a_fleet_is_named_by_size_and_listed_by_member(self):
    self.write_session(0, "spot-tpu-v5e-1-a")
    self.write_session(1, "spot-tpu-v5e-2-b")

    info = summarize_pool(self.pool)
    self.assertEqual(info["TPU_NAME"], "2 VM pool")
    self.assertEqual(info["TPU_VMS"], ["spot-tpu-v5e-1-a", "spot-tpu-v5e-2-b"])
    self.assertEqual(info["TPU_ZONE"], "europe-west4-b")

  def test_every_zone_the_fleet_spans_is_reported(self):
    self.write_session(0, "spot-tpu-v5e-1-a", zone="europe-west4-b")
    self.write_session(1, "spot-tpu-v5e-2-b", zone="us-central1-a")

    self.assertEqual(
        summarize_pool(self.pool)["TPU_ZONE"], "europe-west4-b, us-central1-a"
    )

  def test_an_empty_or_missing_pool_yields_nothing(self):
    """Falsy so the caller falls back to the single-session path."""
    self.assertFalse(summarize_pool(self.pool))
    self.assertFalse(summarize_pool(os.path.join(self.td.name, "absent")))
    self.assertFalse(summarize_pool(""))

  def test_session_files_with_no_vm_name_are_not_counted(self):
    with open(os.path.join(self.pool, "vm_0.env"), "w") as f:
      f.write('export TPU_ZONE="europe-west4-b"\n')
    self.assertFalse(summarize_pool(self.pool))

  def test_the_report_names_the_fleet_instead_of_unknown(self):
    self.write_session(0, "spot-tpu-v5e-1-a")
    targets_file = os.path.join(self.td.name, "targets.txt")
    with open(targets_file, "w") as f:
      f.write("//tests:a\n")
    out_dir = os.path.join(self.td.name, "reports")

    subprocess.run(
        [
            sys.executable,
            REPORTER_SCRIPT,
            f"--testlogs-dir={os.path.join(self.td.name, 'logs')}",
            f"--targets-file={targets_file}",
            f"--session-pool={self.pool}",
            f"--output-dir={out_dir}",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    with open(os.path.join(out_dir, "presubmit_summary.json")) as f:
      data = json.load(f)
    self.assertEqual(data["tpu_vm"], "1 VM pool")
    self.assertEqual(data["tpu_vms"], ["spot-tpu-v5e-1-a"])


class BazelFlagPassthroughTest(PipelineMockToolchainTestCase):
  """--bazel-flag has to reach both bazel invocations, not just the test one.

  The ci configs chain down to --config=resultstore_base, which uploads build
  events to an instance only the CI service account can write to. A Googler
  running the relay by hand needs a way to switch that upload back off, and it
  has to apply to the build as well: the build runs first, and its failed
  upload sinks the run before a single test starts.
  """

  LOGGING_BAZEL = """
log="${BAZEL_ARGV_LOG}"
printf '%s\\n' "$*" >> "$log"
if [[ "$1" == "query" ]]; then
  echo "//tests:empty_test"
  exit 0
fi
echo "//tests:empty_test PASSED in 0.4s"
exit 0
"""

  def _run(self, *extra_args):
    env = self._mock_env("flag_bin", self.LOGGING_BAZEL)
    self.argv_log = os.path.join(self.tmp_dir, "bazel_argv.log")
    env["BAZEL_ARGV_LOG"] = self.argv_log
    proc = subprocess.run(
        [
            RUNNER_SCRIPT,
            f"--output-dir={self.output_dir}",
            "--targets=//tests:empty_test",
            *extra_args,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )
    self.assertEqual(proc.returncode, 0, proc.stderr)
    return proc

  def _invocation(self, verb):
    with open(self.argv_log) as f:
      calls = [line.rstrip("\n") for line in f]
    matching = [call for call in calls if call.startswith(f"{verb} ")]
    self.assertEqual(len(matching), 1, f"expected one `bazel {verb}`: {calls}")
    return matching[0]

  def test_an_extra_flag_reaches_the_build(self):
    self._run("--bazel-flag=--bes_backend=")
    self.assertIn("--bes_backend=", self._invocation("build"))

  def test_an_extra_flag_reaches_the_test_run(self):
    self._run("--bazel-flag=--bes_backend=")
    self.assertIn("--bes_backend=", self._invocation("test"))

  def test_the_flag_is_optional(self):
    self._run()
    self.assertNotIn("--bes_backend=", self._invocation("build"))
    self.assertNotIn("--bes_backend=", self._invocation("test"))

  def test_flags_repeat(self):
    self._run("--bazel-flag=--bes_backend=", "--bazel-flag=--verbose_failures")
    for verb in ("build", "test"):
      with self.subTest(verb=verb):
        self.assertIn("--bes_backend=", self._invocation(verb))
        self.assertIn("--verbose_failures", self._invocation(verb))

  def test_the_space_separated_spelling_works_too(self):
    self._run("--bazel-flag", "--bes_backend=")
    self.assertIn("--bes_backend=", self._invocation("test"))

  def test_an_extra_flag_lands_after_the_config_it_overrides(self):
    """Bazel takes the last value of a repeated flag, so order decides who wins.

    A flag placed before --config would be overwritten by whatever that config
    expands to.
    """
    self._run("--bazel-config=ci_tpu_v5_relay", "--bazel-flag=--bes_backend=")
    for verb in ("build", "test"):
      with self.subTest(verb=verb):
        call = self._invocation(verb)
        self.assertLess(
            call.index("--config=ci_tpu_v5_relay"),
            call.index("--bes_backend="),
            call,
        )

  def test_a_missing_value_is_an_error(self):
    env = self._mock_env("missing_bin", self.LOGGING_BAZEL)
    proc = subprocess.run(
        [RUNNER_SCRIPT, "--bazel-flag"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )
    self.assertNotEqual(proc.returncode, 0)
    self.assertIn("--bazel-flag requires an argument", proc.stderr)

  def test_the_dry_run_plan_shows_it(self):
    proc = self._run("--dry-run", "--bazel-flag=--bes_backend=")
    self.assertIn("--bes_backend=", proc.stdout)
