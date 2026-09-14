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

"""generate_presubmit_report.py

Parses test logs, JUnit XML artifacts, and Bazel execution metadata
to produce presubmit_summary.json and presubmit_report.md.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET


def strip_ansi(text: str) -> str:
  ansi_regex = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\([a-zA-Z]")
  return ansi_regex.sub("", text)


def extract_diagnostics(log_path: str, max_lines: int = 50) -> str:
  if not log_path or not os.path.isfile(log_path):
    return "Log file not found."

  try:
    with open(log_path, "r", errors="replace") as f:
      raw_text = f.read()
  except OSError as e:
    return f"Failed to read log file: {e}"

  lines = strip_ansi(raw_text).splitlines()
  if not lines:
    return "Empty log file."

  # 1. Relay and infrastructure faults come first. They explain every traceback
  # printed after them, so reporting the traceback instead just points the
  # reader at a test that was never the problem.
  relay_patterns = [
      "ERROR [relay_test_runner]",
      "ERROR [remote_tpu_executor]",
      "Project boundary violation",
      "Preemption detected",
      "ControlMaster connection failed",
  ]
  relay_indices = [
      i
      for i, line in enumerate(lines)
      if any(pat in line for pat in relay_patterns)
  ]
  if relay_indices:
    return "\n".join(lines[relay_indices[0] : relay_indices[0] + max_lines])

  # 2. Search for Python Traceback
  #
  # The first one, not the last: chained exceptions print the root cause first
  # and the handler's own failure last, and the root cause is what's wanted.
  tb_indices = [
      i
      for i, line in enumerate(lines)
      if "Traceback (most recent call last):" in line
  ]
  if tb_indices:
    start = tb_indices[0]
    return "\n".join(lines[start : start + max_lines])

  # 3. Search for C++ Fatal Errors, SIGSEGV, or Check failed
  fatal_patterns = [
      "Check failed:",
      "SIGSEGV",
      "SIGABRT",
      "SIGBUS",
      "SIGFPE",
      "Segmentation fault",
      "Fatal error",
      "terminate called",
      "Assertion failed",
  ]
  fatal_indices = [
      i
      for i, line in enumerate(lines)
      if any(pat in line for pat in fatal_patterns)
  ]
  if fatal_indices:
    start = max(0, fatal_indices[-1] - 3)
    return "\n".join(lines[start : start + max_lines])

  # 4. Search for TPU device or runtime errors
  tpu_patterns = [
      "InitializePjrtPlugin failed",
      "The TPU is already in use",
      "No such device or address",
      "TPU_VISIBLE_DEVICES",
      "Failed to open /dev/vfio",
      "ERROR: TPU",
  ]
  tpu_indices = [
      i
      for i, line in enumerate(lines)
      if any(pat in line for pat in tpu_patterns)
  ]
  if tpu_indices:
    start = max(0, tpu_indices[0] - 2)
    return "\n".join(lines[start : start + max_lines])

  # 5. Fallback: last 40 lines
  return "\n".join(lines[-40:] if len(lines) > 40 else lines)


def target_to_testlog_relpath(target: str) -> str:
  clean = target.lstrip("/")
  if ":" in clean:
    pkg, name = clean.split(":", 1)
    return os.path.join(pkg, name)
  return clean


def testlog_relpath_to_target(rel_path: str) -> str:
  parts = rel_path.strip("/").split(os.sep)
  if len(parts) == 1:
    return f"//:{parts[0]}"
  pkg = "/".join(parts[:-1])
  name = parts[-1]
  return f"//{pkg}:{name}"


def _safe_int(val, default=0):
  if val is None:
    return default
  try:
    return int(float(str(val).strip()))
  except (ValueError, TypeError, OverflowError):
    return default


def _safe_float(val, default=0.0):
  if val is None:
    return default
  try:
    f = float(str(val).strip())
    if math.isnan(f) or math.isinf(f):
      return default
    return f
  except (ValueError, TypeError):
    return default


def parse_xml_report(xml_path: str):
  if not os.path.isfile(xml_path):
    return None

  try:
    with open(xml_path, "r", errors="replace") as f:
      content = f.read()
    clean_content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)
    root = ET.fromstring(clean_content)
  except (ET.ParseError, OSError):
    return None

  testsuites = []
  if root.tag == "testsuite":
    testsuites.append(root)
  else:
    testsuites.extend(root.findall(".//testsuite"))

  if not testsuites:
    if root.findall(".//testcase") or root.tag == "testsuites":
      testsuites.append(root)
    else:
      return None

  total_tests = 0
  total_failures = 0
  total_errors = 0
  total_skipped = 0
  total_time = 0.0
  error_msg = ""

  for suite in testsuites:
    testcases = suite.findall(".//testcase")
    tc_count = len(testcases)

    tc_failures = sum(1 for tc in testcases if tc.find("failure") is not None)
    tc_errors = sum(1 for tc in testcases if tc.find("error") is not None)
    tc_skipped = sum(
        1
        for tc in testcases
        if tc.find("skipped") is not None or tc.find("disabled") is not None
    )
    tc_time = sum(_safe_float(tc.attrib.get("time"), 0.0) for tc in testcases)

    attr_tests = _safe_int(suite.attrib.get("tests"), default=None)
    if attr_tests is not None and attr_tests >= 0:
      suite_tests = (
          tc_count if (attr_tests == 0 and tc_count > 0) else attr_tests
      )
    else:
      suite_tests = tc_count if tc_count > 0 else 1

    attr_failures = _safe_int(suite.attrib.get("failures"), default=None)
    if attr_failures is not None and attr_failures >= 0:
      suite_failures = max(attr_failures, tc_failures)
    else:
      suite_failures = tc_failures
    if suite_failures == 0 and suite.find(".//failure") is not None:
      suite_failures = 1

    attr_errors = _safe_int(suite.attrib.get("errors"), default=None)
    if attr_errors is not None and attr_errors >= 0:
      suite_errors = max(attr_errors, tc_errors)
    else:
      suite_errors = tc_errors
    if suite_errors == 0 and suite.find(".//error") is not None:
      suite_errors = 1

    attr_skipped = _safe_int(suite.attrib.get("skipped"), default=None)
    if attr_skipped is None:
      attr_skipped = _safe_int(suite.attrib.get("disabled"), default=None)
    if attr_skipped is not None and attr_skipped >= 0:
      suite_skipped = max(attr_skipped, tc_skipped)
    else:
      suite_skipped = tc_skipped

    attr_time = _safe_float(suite.attrib.get("time"), default=None)
    if attr_time is not None and attr_time >= 0.0:
      suite_time = attr_time
    else:
      suite_time = tc_time

    total_tests += suite_tests
    total_failures += suite_failures
    total_errors += suite_errors
    total_skipped += suite_skipped
    total_time += suite_time

    if not error_msg:
      fail_el = suite.find(".//failure")
      if fail_el is not None:
        error_msg = fail_el.attrib.get("message", "").strip()
        if not error_msg and fail_el.text:
          lines = fail_el.text.strip().splitlines()
          if lines:
            error_msg = lines[0].strip()
      if not error_msg:
        err_el = suite.find(".//error")
        if err_el is not None:
          error_msg = err_el.attrib.get("message", "").strip()
          if not error_msg and err_el.text:
            lines = err_el.text.strip().splitlines()
            if lines:
              error_msg = lines[0].strip()

  if (total_failures > 0 or total_errors > 0) and not error_msg:
    error_msg = "Test assertion or execution failure"

  return {
      "tests": total_tests,
      "failures": total_failures,
      "errors": total_errors,
      "skipped": total_skipped,
      "time": total_time,
      "error_message": error_msg,
  }


# Bazel prints one of these per target once the run ends. The status column is
# padded out to a fixed width, so the target and the verdict are separated by a
# run of spaces rather than a single one.
BAZEL_STATUS_LINE = re.compile(
    r"^(?P<target>//\S+)\s+(?:\(cached\)\s+)?"
    r"(?P<status>PASSED|FAILED|TIMEOUT|FLAKY|NO STATUS)\b"
)

RELAY_INFRA_FAILURES = frozenset({
    "Spot TPU Preempted",
    "SSH Connection Failed",
    "TPU State Unknown",
    "TPU Base Cache Missing",
    "No free TPU in pool",
    "No active TPU session",
    "Incomplete TPU session",
    "Relay executor missing",
    "Test Report Not Retrieved",
    "Test Timeout",
})


def parse_bazel_statuses(bazel_log: str) -> dict:
  """Reads the verdict bazel itself reached for each target.

  This is the only source that knows whether an action ran at all. The xml is
  for counts and diagnostics; it cannot answer "did this target run in *this*
  invocation", and a sharded target leaves no top-level test.xml to find.
  """
  statuses = {}
  if not bazel_log or not os.path.isfile(bazel_log):
    return statuses

  try:
    with open(bazel_log, "r", errors="replace") as f:
      content = f.read()
  except OSError:
    return statuses

  # Bazel redraws its progress line with carriage returns.
  for line in strip_ansi(content).replace("\r", "\n").splitlines():
    match = BAZEL_STATUS_LINE.match(line.strip())
    if match:
      statuses[match.group("target")] = match.group("status")
  return statuses


def collect_xml_paths(testlogs_dir: str, target: str, min_mtime: float = 0.0):
  """Returns this target's report files, newest run only.

  A sharded target writes shard_1_of_N/test.xml and has nothing at the top
  level, so looking only for test.xml finds nothing and the target reads as
  though it never built.
  """
  base = os.path.join(testlogs_dir, target_to_testlog_relpath(target))
  if not os.path.isdir(base):
    return [], 0

  candidates = []
  top = os.path.join(base, "test.xml")
  if os.path.isfile(top):
    candidates.append(top)
  try:
    entries = sorted(os.listdir(base))
  except OSError:
    entries = []
  for entry in entries:
    shard = os.path.join(base, entry, "test.xml")
    if os.path.isfile(shard):
      candidates.append(shard)

  fresh, stale = [], 0
  for path in candidates:
    try:
      if os.path.getmtime(path) < min_mtime:
        stale += 1
        continue
    except OSError:
      continue
    fresh.append(path)
  return fresh, stale


def parse_target_reports(xml_paths):
  """Sums every shard's report into one record for the target."""
  totals = None
  for path in xml_paths:
    shard = parse_xml_report(path)
    if shard is None:
      continue
    if totals is None:
      totals = dict(shard)
      continue
    for key in ("tests", "failures", "errors", "skipped"):
      totals[key] += shard[key]
    totals["time"] += shard["time"]
    if not totals["error_message"]:
      totals["error_message"] = shard["error_message"]
  return totals


def parse_session_env(session_env_path: str) -> dict:
  res = {
      "TPU_NAME": "unknown",
      "TPU_ZONE": "europe-west4-b",
      "TPU_PROJECT": "rbe-tpu-oss",
  }
  if not session_env_path or not os.path.isfile(session_env_path):
    return res

  try:
    with open(session_env_path, "r", errors="replace") as f:
      for line in f:
        line = line.strip()
        if line.startswith("export "):
          line = line[7:]
        if "=" in line:
          k, v = line.split("=", 1)
          k = k.strip()
          v = v.strip().strip("\"'")
          if k in res and v:
            res[k] = v
  except OSError:
    pass
  return res


def summarize_pool(pool_dir: str) -> dict:
  """Describes a fleet the way parse_session_env describes one VM.

  A pool run has no single session file, so the header names the fleet and the
  JSON keeps the individual VMs. Which VM served which shard is the first thing
  you need when a run goes wrong.
  """
  sessions = []
  if pool_dir and os.path.isdir(pool_dir):
    sessions = sorted(
        os.path.join(pool_dir, name)
        for name in os.listdir(pool_dir)
        if name.endswith(".env")
    )
  if not sessions:
    return {}

  parsed = [parse_session_env(path) for path in sessions]
  names = sorted(p["TPU_NAME"] for p in parsed if p["TPU_NAME"] != "unknown")
  if not names:
    return {}

  return {
      "TPU_NAME": f"{len(names)} VM pool",
      "TPU_VMS": names,
      "TPU_ZONE": ", ".join(sorted({p["TPU_ZONE"] for p in parsed})),
      "TPU_PROJECT": parsed[0]["TPU_PROJECT"],
  }


def discover_targets_from_testlogs(testlogs_dir: str) -> list[str]:
  targets = []
  if not os.path.isdir(testlogs_dir):
    return targets

  for root, _, files in os.walk(testlogs_dir):
    if "test.xml" in files or "test.log" in files:
      rel = os.path.relpath(root, testlogs_dir)
      targets.append(testlog_relpath_to_target(rel))

  targets.sort()
  return targets


def discover_targets_from_bazel_log(bazel_log_path: str) -> list[str]:
  targets = []
  if not bazel_log_path or not os.path.isfile(bazel_log_path):
    return targets

  target_regex = re.compile(
      r"^\s*(//[a-zA-Z0-9_\-./]+:[a-zA-Z0-9_\-.]+)\s+(PASSED|FAILED|FLAKY|TIMEOUT|TIMED"
      r" OUT)"
  )
  try:
    with open(bazel_log_path, "r", errors="replace") as f:
      for line in f:
        m = target_regex.search(line)
        if m:
          t = m.group(1)
          if t not in targets:
            targets.append(t)
  except OSError:
    pass
  return targets


def parse_args():
  parser = argparse.ArgumentParser(
      description="Generate presubmit summary JSON and Markdown reports"
  )
  parser.add_argument(
      "--output-dir",
      required=True,
      help=(
          "Directory to deposit presubmit_summary.json and presubmit_report.md"
      ),
  )
  parser.add_argument(
      "--bazel-log",
      "--bazel-output",
      dest="bazel_log",
      default="",
      help="Captured Bazel console output log",
  )
  parser.add_argument(
      "--workspace-root",
      default=os.getcwd(),
      help="Root directory of the workspace",
  )
  parser.add_argument(
      "--testlogs-dir",
      default="",
      help=(
          "Path to bazel-testlogs directory (default:"
          " <workspace-root>/bazel-testlogs)"
      ),
  )
  parser.add_argument(
      "--targets-file",
      default="",
      help="File containing list of executed test targets",
  )
  parser.add_argument(
      "--targets",
      default="",
      help="Comma- or space-separated list of targets",
  )
  parser.add_argument(
      "--session-env",
      default="/tmp/tpu_active_session.env",
      help="Path to active TPU session file",
  )
  parser.add_argument(
      "--session-pool",
      default="",
      help="Directory of session files, when the run spread across a fleet",
  )
  parser.add_argument(
      "--duration",
      type=float,
      default=0.0,
      help="Execution duration in seconds",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Dry run mode; generate mock/simulated report",
  )
  parser.add_argument(
      "--bazel-exit-code",
      type=int,
      default=0,
      help="Exit code returned by Bazel process",
  )
  parser.add_argument(
      "--run-started-at",
      type=float,
      default=0.0,
      help=(
          "Unix time this run started. bazel-testlogs survives across runs, so "
          "anything older than this is from a previous run and is ignored."
      ),
  )
  return parser.parse_args()


def main():
  args = parse_args()
  os.makedirs(args.output_dir, exist_ok=True)

  session_info = summarize_pool(args.session_pool) or parse_session_env(
      args.session_env
  )
  if args.dry_run and session_info["TPU_NAME"] == "unknown":
    session_info["TPU_NAME"] = "simulated-dry-run-vm"

  testlogs_dir = args.testlogs_dir
  if not testlogs_dir:
    testlogs_dir = os.path.join(args.workspace_root, "bazel-testlogs")

  # Resolve target list
  targets = []
  if args.targets_file and os.path.isfile(args.targets_file):
    with open(args.targets_file, "r") as f:
      targets = [line.strip() for line in f if line.strip()]
  elif args.targets:
    raw = args.targets.replace(",", " ")
    targets = [t.strip() for t in raw.split() if t.strip()]
  else:
    targets = discover_targets_from_bazel_log(args.bazel_log)
    if not targets:
      targets = discover_targets_from_testlogs(testlogs_dir)

  bazel_statuses = parse_bazel_statuses(args.bazel_log)

  target_records = []
  total_targets = len(targets)
  passed_targets = 0
  failed_targets = 0
  skipped_targets = 0
  infra_failures = 0

  for target in targets:
    rel_path = target_to_testlog_relpath(target)
    xml_path = os.path.join(testlogs_dir, rel_path, "test.xml")
    log_path = os.path.join(testlogs_dir, rel_path, "test.log")

    if args.dry_run:
      target_records.append({
          "target": target,
          "status": "DRY_RUN",
          "duration_seconds": 0.0,
          "exit_code": 0,
          "tests_count": 0,
          "failures_count": 0,
          "errors_count": 0,
          "skipped_count": 0,
          "log_file": log_path,
          "xml_file": xml_path,
          "error_message": "",
          "failure_details": "",
      })
      continue

    xml_paths, stale_count = collect_xml_paths(
        testlogs_dir, target, args.run_started_at
    )
    xml_data = parse_target_reports(xml_paths)
    bazel_status = bazel_statuses.get(target)

    # Bazel decides pass or fail. It is the only party that knows whether the
    # action ran in this invocation; the xml only says what a test process
    # wrote, whenever that was. Deriving status from the xml alone reported a
    # green run as 14 failures, because a sharded target has no top-level
    # test.xml and "no xml" was being read as "did not build".
    if bazel_status == "PASSED":
      status, exit_code = "PASSED", 0
    elif bazel_status in ("FAILED", "TIMEOUT", "FLAKY"):
      status, exit_code = bazel_status, 1
    elif bazel_status == "NO STATUS":
      status, exit_code = "NO_STATUS", 1
    elif xml_paths:
      # No summary line, but this run did leave a report behind.
      has_fail = xml_data is not None and (
          xml_data["failures"] > 0 or xml_data["errors"] > 0
      )
      status, exit_code = ("FAILED", 1) if has_fail else ("PASSED", 0)
    else:
      status, exit_code = "NO_STATUS", 1

    err_msg = xml_data["error_message"] if xml_data else ""

    # An infra failure is not a code regression, and reporting it as a plain
    # FAILED sends whoever reads this hunting through a test they never broke.
    if err_msg in RELAY_INFRA_FAILURES and status != "PASSED":
      status = "INFRA_FAILED"

    failed = status not in ("PASSED", "DRY_RUN")
    if failed:
      failed_targets += 1
      if status == "INFRA_FAILED":
        infra_failures += 1
    else:
      passed_targets += 1

    if not err_msg and status == "NO_STATUS":
      err_msg = (
          "No result for this target in this run "
          f"({stale_count} report(s) on disk from an earlier run)"
          if stale_count
          else "No result for this target in this run"
      )

    diag = ""
    if failed:
      if os.path.isfile(log_path):
        diag = extract_diagnostics(log_path)
      elif xml_paths:
        diag = extract_diagnostics(
            os.path.join(os.path.dirname(xml_paths[0]), "test.log")
        )
      elif args.bazel_log and os.path.isfile(args.bazel_log):
        diag = extract_diagnostics(args.bazel_log)

    if xml_data:
      skipped_targets += xml_data["skipped"]

    target_records.append({
        "target": target,
        "status": status,
        "duration_seconds": round(xml_data["time"], 2) if xml_data else 0.0,
        "exit_code": exit_code,
        "tests_count": xml_data["tests"] if xml_data else 0,
        "failures_count": xml_data["failures"] if xml_data else 0,
        "errors_count": xml_data["errors"] if xml_data else 0,
        "skipped_count": xml_data["skipped"] if xml_data else 0,
        "shards": len(xml_paths),
        "log_file": log_path,
        "xml_file": xml_path,
        "error_message": err_msg,
        "failure_details": diag,
    })

  overall_status = "PASSED"
  if args.dry_run:
    overall_status = "DRY_RUN"
  elif failed_targets > 0 or args.bazel_exit_code != 0:
    overall_status = "FAILED"

  iso_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

  summary = {
      "schema_version": "1.0.0",
      "timestamp": iso_timestamp,
      "duration_seconds": round(args.duration, 2),
      "execution_time": round(args.duration, 2),
      "project": session_info["TPU_PROJECT"],
      "tpu_zone": session_info["TPU_ZONE"],
      "tpu_vm": session_info["TPU_NAME"],
      "tpu_vms": session_info.get("TPU_VMS", []),
      "status": overall_status,
      "total_targets": total_targets,
      "targets_total": total_targets,
      "passed_targets": passed_targets,
      "targets_passed": passed_targets,
      "failed_targets": failed_targets,
      "targets_failed": failed_targets,
      "skipped_targets": skipped_targets,
      "targets_skipped": skipped_targets,
      "targets": target_records,
  }

  json_path = os.path.join(args.output_dir, "presubmit_summary.json")
  with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)

  pass_rate = (passed_targets / total_targets * 100.0) if total_targets else 0.0
  status_badge = "🟢 PASSED"
  if overall_status == "FAILED":
    status_badge = "🔴 FAILED"
  elif overall_status == "DRY_RUN":
    status_badge = "⚪ DRY_RUN"

  md_lines = [
      "# Torch TPU Presubmit Report: v5e Spot Relay",
      "",
      f"**Status**: {status_badge}",
      f"**Timestamp**: {iso_timestamp}",
      f"**Total Duration**: {round(args.duration, 1)}s",
      f"**Project**: `{session_info['TPU_PROJECT']}`",
      f"**Zone**: `{session_info['TPU_ZONE']}`",
      f"**TPU VM**: `{session_info['TPU_NAME']}`",
      "",
      "## Summary Metrics",
      "",
      "| Metric | Value |",
      "| :--- | :--- |",
      f"| Total Targets | {total_targets} |",
      f"| Passed Targets | {passed_targets} |",
      f"| Failed Targets | {failed_targets} |",
      f"| Pass Rate | {pass_rate:.1f}% |",
      f"| Total Execution Time | {round(args.duration, 1)}s |",
      "",
      "## Target Execution Results",
      "",
      "| # | Target | Status | Duration | Tests | Failures | Logs |",
      "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
  ]

  for idx, rec in enumerate(target_records, 1):
    dur_str = f"{rec['duration_seconds']:.1f}s"
    log_name = os.path.basename(rec["log_file"])
    log_link = f"[{log_name}]({rec['log_file']})"
    md_lines.append(
        f"| {idx} | `{rec['target']}` | {rec['status']} | {dur_str} |"
        f" {rec['tests_count']} | {rec['failures_count']} | {log_link} |"
    )

  failures_exist = any(
      r["status"] not in ["PASSED", "DRY_RUN"] for r in target_records
  )
  if failures_exist:
    md_lines.extend(["", "## Failure Diagnostics", ""])
    for rec in target_records:
      if rec["status"] in ["PASSED", "DRY_RUN"]:
        continue
      md_lines.append(f"### ❌ `{rec['target']}`")
      md_lines.append(
          f"- **Status**: {rec['status']} (exit code {rec['exit_code']})"
      )
      md_lines.append(f"- **Duration**: {rec['duration_seconds']:.1f}s")
      if rec["error_message"]:
        md_lines.append(f"- **Error Summary**: `{rec['error_message']}`")
      md_lines.append(f"- **Log Path**: `{rec['log_file']}`")
      md_lines.append("")
      md_lines.append("<details open>")
      md_lines.append("<summary>Diagnostic Excerpt</summary>")
      md_lines.append("")
      md_lines.append("```")
      md_lines.append(
          rec["failure_details"] or "No detailed diagnostic excerpt available."
      )
      md_lines.append("```")
      md_lines.append("</details>")
      md_lines.append("")

  md_path = os.path.join(args.output_dir, "presubmit_report.md")
  with open(md_path, "w") as f:
    f.write("\n".join(md_lines) + "\n")

  return 0 if overall_status in ["PASSED", "DRY_RUN"] else 1


if __name__ == "__main__":
  sys.exit(main())
