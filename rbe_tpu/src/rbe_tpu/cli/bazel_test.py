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

"""CLI runner for executing Bazel tests across the Cloud TPU v5e fleet via GCS mailbox."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import datetime
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Final

DEFAULT_BUCKET: Final[str] = "torch-tpu-mailbox-eval"
DEFAULT_SCOPE: Final[str] = "//tests/..."
DEFAULT_JOBS: Final[int] = 28
DEFAULT_BAZEL_CONFIG: Final[str] = "rbe_tpu"
DEFAULT_TIMEOUT_SECONDS: Final[int] = 900

HELP_TEXT: Final[str] = """Usage: rbe_tpu/deploy/rbe_tpu_bazel_test.sh [options] [TARGET...] [-- BAZEL_FLAG...]

Runs bazel test on the standing Cloud TPU v5e fleet in rbe-tpu-oss via the GCS
mailbox runner. Supports arbitrary target labels, package patterns, and bazel flags.

Examples:
  rbe_tpu/deploy/rbe_tpu_bazel_test.sh //tests/pallas:pallas_test
  rbe_tpu/deploy/rbe_tpu_bazel_test.sh //tests/pallas:pallas_test -- --runs_per_test=20
  rbe_tpu/deploy/rbe_tpu_bazel_test.sh //tests/pallas/...
  rbe_tpu/deploy/rbe_tpu_bazel_test.sh --scope //tests/pallas/... --filter presubmit-v5

Targets can be exact labels or wildcard patterns (e.g. //tests/pallas/...).
Wildcard patterns are expanded with `bazel query tests(...)`.
With no targets specified, --scope and --filter determine what runs.

Options:
  --bucket NAME         GCS mailbox bucket (default: torch-tpu-mailbox-eval).
  --jobs N              Concurrent test jobs (default: 28).
  --bazel-config NAME   Bazel config to inherit (default: ci_tpu_v5_mailbox).
                        Pass "" for standalone execution.
  --bazel-flag FLAG     Extra flag passed to bazel test. Can be repeated.
  --scope PATTERN       Where to look when no targets are named (default: //tests/...).
  --filter TAG_EXPR     Tag filter when scope is used. Default: none.
  --timeout SECONDS     Per-test timeout in seconds (default: 900).
  --output-dir DIR      Destination directory for logs and reports.
  --dry-run             Expand targets, show execution plan, and exit.
  --keep                Retained for CLI parity (mailbox fleet is always available).
  -h, --help            Show this help message.

Flags after `--` are passed directly to `bazel test` (e.g. --runs_per_test=20)."""


class CLIError(Exception):
  """Raised when invalid command-line arguments are supplied."""


@dataclasses.dataclass(frozen=True)
class Options:
  targets: tuple[str, ...] = ()
  bazel_flags: tuple[str, ...] = ()
  bucket: str = DEFAULT_BUCKET
  jobs: int = DEFAULT_JOBS
  bazel_config: str = DEFAULT_BAZEL_CONFIG
  scope: str = ""
  tag_filter: str = ""
  output_dir: pathlib.Path | None = None
  dry_run: bool = False
  test_timeout: int = DEFAULT_TIMEOUT_SECONDS
  keep: bool = False
  show_help: bool = False


def log(msg: str) -> None:
  print(f"[rbe_tpu_test] {msg}", flush=True)


def parse_args(argv: Sequence[str]) -> Options:
  targets: list[str] = []
  bazel_flags: list[str] = []
  bucket = DEFAULT_BUCKET
  jobs = DEFAULT_JOBS
  bazel_config = DEFAULT_BAZEL_CONFIG
  scope = ""
  tag_filter = ""
  output_dir: pathlib.Path | None = None
  dry_run = False
  test_timeout = DEFAULT_TIMEOUT_SECONDS
  keep = False

  idx = 0
  n = len(argv)
  while idx < n:
    arg = argv[idx]
    if arg == "--":
      bazel_flags.extend(argv[idx + 1 :])
      break
    if arg in ("-h", "--help"):
      return Options(show_help=True)
    if arg == "--bucket":
      if idx + 1 >= n:
        raise CLIError("--bucket requires an argument.")
      bucket = argv[idx + 1]
      idx += 2
    elif arg == "--jobs":
      if idx + 1 >= n:
        raise CLIError("--jobs requires an argument.")
      if not argv[idx + 1].isdigit():
        raise CLIError("--jobs must be a number.")
      jobs = int(argv[idx + 1])
      idx += 2
    elif arg == "--bazel-config":
      if idx + 1 >= n:
        raise CLIError("--bazel-config requires an argument.")
      bazel_config = argv[idx + 1]
      idx += 2
    elif arg == "--bazel-flag":
      if idx + 1 >= n:
        raise CLIError("--bazel-flag requires an argument.")
      bazel_flags.append(argv[idx + 1])
      idx += 2
    elif arg == "--scope":
      if idx + 1 >= n:
        raise CLIError("--scope requires an argument.")
      scope = argv[idx + 1]
      idx += 2
    elif arg == "--filter":
      if idx + 1 >= n:
        raise CLIError("--filter requires an argument.")
      tag_filter = argv[idx + 1]
      idx += 2
    elif arg == "--timeout":
      if idx + 1 >= n:
        raise CLIError("--timeout requires an argument.")
      if not argv[idx + 1].isdigit():
        raise CLIError("--timeout must be a number.")
      test_timeout = int(argv[idx + 1])
      idx += 2
    elif arg == "--output-dir":
      if idx + 1 >= n:
        raise CLIError("--output-dir requires an argument.")
      output_dir = pathlib.Path(argv[idx + 1])
      idx += 2
    elif arg == "--dry-run":
      dry_run = True
      idx += 1
    elif arg == "--keep":
      keep = True
      idx += 1
    elif arg.startswith("-"):
      raise CLIError(f"Unknown option: {arg}")
    else:
      targets.append(arg)
      idx += 1

  return Options(
      targets=tuple(targets),
      bazel_flags=tuple(bazel_flags),
      bucket=bucket,
      jobs=jobs,
      bazel_config=bazel_config,
      scope=scope,
      tag_filter=tag_filter,
      output_dir=output_dir,
      dry_run=dry_run,
      test_timeout=test_timeout,
      keep=keep,
  )


def build_query_expr(scope: str, tag_filter: str) -> str:
  expr = (
      f'tests({scope}) except attr(tags, "manual|notest", {scope}) except'
      f' attr(name, ".*_build_test$", {scope})'
  )
  if not tag_filter:
    return expr

  pos_tags: list[str] = []
  neg_tags: list[str] = []
  for raw_item in tag_filter.split(","):
    item = raw_item.strip()
    if not item:
      continue
    if item.startswith("-"):
      neg_tags.append(item[1:])
    else:
      pos_tags.append(item)

  if pos_tags:
    pos_pattern = "|".join(pos_tags)
    expr = f'{expr} intersect attr(tags, "{pos_pattern}", {scope})'
  if neg_tags:
    neg_pattern = "|".join(neg_tags)
    expr = f'{expr} except attr(tags, "{neg_pattern}", {scope})'
  return expr


def run_bazel_query(query_expr: str) -> list[str]:
  res = subprocess.run(
      ["bazel", "query", query_expr, "--output=label"],
      capture_output=True,
      text=True,
      check=False,
  )
  return [
      line.strip()
      for line in res.stdout.splitlines()
      if line.strip().startswith("//")
  ]


def filter_single_chip_targets(targets: Sequence[str]) -> list[str]:
  final: list[str] = []
  for tgt in targets:
    if ":8" in tgt or "8tpu" in tgt:
      log(
          f"Skipping multi-chip target: {tgt} (only single-chip v5litepod-1"
          " supported)"
      )
      continue
    final.append(tgt)
  return final


def resolve_targets(opts: Options) -> list[str]:
  resolved: list[str] = []
  if opts.targets:
    for tgt in opts.targets:
      if ":" in tgt and "..." not in tgt and not tgt.endswith(":all"):
        resolved.append(tgt)
      else:
        log(f"Expanding pattern: {tgt}")
        expr = build_query_expr(tgt, opts.tag_filter)
        resolved.extend(run_bazel_query(expr))
  else:
    target_scope = opts.scope or DEFAULT_SCOPE
    log(f"Querying test targets in {target_scope}...")
    expr = build_query_expr(target_scope, opts.tag_filter)
    resolved.extend(run_bazel_query(expr))
  return filter_single_chip_targets(resolved)


def check_fleet_count(bucket: str) -> int:
  log(f"Checking fleet status in gs://{bucket}/fleet/...")
  res = subprocess.run(
      ["gcloud", "storage", "ls", f"gs://{bucket}/fleet/"],
      capture_output=True,
      text=True,
      check=False,
  )
  count = sum(
      1 for line in res.stdout.splitlines() if line.strip().endswith(".json")
  )
  if count == 0:
    print(
        f"WARNING: No healthy TPU VMs currently registered in"
        f" gs://{bucket}/fleet/.",
        file=sys.stderr,
    )
  else:
    log(f"Found {count} registered TPU VM(s) in fleet.")
  return count


def build_bazel_args(opts: Options, targets: Sequence[str]) -> list[str]:
  args = ["test"]
  if opts.bazel_config:
    args.append(f"--config={opts.bazel_config}")
  args.extend([
      f"--test_env=TORCH_TPU_RELAY_BUCKET={opts.bucket}",
      "--run_under=//rbe_tpu:worker_main",
      f"--test_timeout={opts.test_timeout}",
      f"--jobs={opts.jobs}",
      f"--local_test_jobs={opts.jobs}",
      "--test_output=errors",
      "--nocache_test_results",
  ])
  args.extend(opts.bazel_flags)
  args.extend(targets)
  return args


def extract_critical_path(log_text: str) -> str:
  matches = re.findall(
      r"INFO: Elapsed time: .* Critical Path: ([0-9.]+)s", log_text
  )
  return f"{matches[-1]}s" if matches else "N/A"


def extract_failure_lines(log_text: str, limit: int = 30) -> list[str]:
  failures = [
      line
      for line in log_text.splitlines()
      if any(token in line for token in ("FAIL:", "TIMEOUT:", "ERROR:"))
  ]
  return failures[:limit]


def format_report(
    *,
    verdict: str,
    total_elapsed: int,
    bucket: str,
    target_count: int,
    fleet_count: int,
    now_tag: str,
    bazel_rc: int,
    fleet_check_elapsed: int,
    query_elapsed: int,
    bazel_elapsed: int,
    critical_path: str,
    failure_lines: Sequence[str],
) -> str:
  report = f"""# TPU Mailbox Test Run Report

**Verdict:** {verdict}
**Wall Clock Time:** {total_elapsed}s
**Bucket:** `gs://{bucket}`
**Targets Evaluated:** {target_count}
**Fleet Capacity:** {fleet_count} chips
**Timestamp:** {now_tag}

## Summary

| Metric | Value |
| :--- | :--- |
| **Status** | **{verdict}** |
| **Total Duration** | {total_elapsed}s |
| **Exit Code** | {bazel_rc} |
| **Targets** | {target_count} |
| **Fleet Capacity** | {fleet_count} chips registered |

## Stage Timing Breakdown

| Stage | Duration |
| :--- | :--- |
| **1. Fleet Discovery (GCS)** | {fleet_check_elapsed}s |
| **2. Target Resolution** | {query_elapsed}s |
| **3. Bazel Execution Total** | {bazel_elapsed}s |
| **3a. Test Critical Path** | {critical_path} |
| **Total Wall Clock** | **{total_elapsed}s** |
"""
  if bazel_rc != 0 and failure_lines:
    joined = "\n".join(failure_lines)
    report += f"\n### Failures\n```\n{joined}\n```\n"
  return report


def default_repo_root() -> pathlib.Path:
  here = pathlib.Path(__file__).resolve()
  for parent in here.parents:
    if (parent / "deploy" / "rbe_tpu_bazel_test.sh").exists():
      return parent
  return here.parents[3]


def main(argv: Sequence[str] | None = None) -> int:
  args = sys.argv[1:] if argv is None else argv
  try:
    opts = parse_args(args)
  except CLIError as err:
    print(f"ERROR [rbe_tpu_test]: {err}", file=sys.stderr)
    return 1

  if opts.show_help:
    print(HELP_TEXT)
    return 0

  script_start = int(time.time())
  now_tag = datetime.datetime.now(datetime.timezone.utc).strftime(
      "%Y%m%d_%H%M%SZ"
  )
  output_dir = opts.output_dir or (
      default_repo_root() / "mailbox_reports" / f"test_{now_tag}"
  )
  try:
    output_dir.mkdir(parents=True, exist_ok=True)
  except OSError:
    fallback_root = pathlib.Path(os.environ.get("TEST_TMPDIR", "/tmp"))
    output_dir = fallback_root / "mailbox_reports" / f"test_{now_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

  fleet_start = int(time.time())
  fleet_count = check_fleet_count(opts.bucket)
  fleet_elapsed = int(time.time()) - fleet_start

  query_start = int(time.time())
  final_targets = resolve_targets(opts)
  query_elapsed = int(time.time()) - query_start

  target_count = len(final_targets)
  log(f"Resolved {target_count} target(s) to execute.")
  if target_count == 0:
    print("ERROR [rbe_tpu_test]: No test targets resolved.", file=sys.stderr)
    return 1

  if opts.dry_run:
    print(
        f"[DRY RUN] Execution plan confirmed. Resolved targets ({target_count}):"
    )
    for tgt in final_targets:
      print(f"  {tgt}")
    if opts.bazel_flags:
      print("[DRY RUN] Extra bazel flags:")
      for flag in opts.bazel_flags:
        print(f"  {flag}")
    return 0

  bazel_args = build_bazel_args(opts, final_targets)
  log_file = output_dir / "bazel_test.log"
  report_file = output_dir / "report.md"

  exec_start = int(time.time())
  log(f"Starting Bazel test execution (logs -> {log_file})...")

  captured_lines: list[str] = []
  with open(log_file, "w", encoding="utf-8") as log_handle:
    proc = subprocess.Popen(
        ["bazel", *bazel_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
      sys.stdout.write(line)
      sys.stdout.flush()
      log_handle.write(line)
      captured_lines.append(line)
    bazel_rc = proc.wait()

  end_time = int(time.time())
  bazel_elapsed = end_time - exec_start
  total_elapsed = end_time - script_start

  log_text = "".join(captured_lines)
  critical_path = extract_critical_path(log_text)
  verdict = "PASSED" if bazel_rc == 0 else "FAILED"

  log(
      f"Bazel execution completed in {bazel_elapsed}s with verdict: {verdict}"
      f" (rc={bazel_rc})"
  )
  log("=== Stage Timing Breakdown ===")
  log(f"  1. Fleet discovery (GCS):  {fleet_elapsed}s")
  log(f"  2. Target resolution:      {query_elapsed}s")
  log(
      f"  3. Bazel execution total:  {bazel_elapsed}s (Critical path:"
      f" {critical_path})"
  )
  log(f"  Total wall clock:          {total_elapsed}s")

  report_content = format_report(
      verdict=verdict,
      total_elapsed=total_elapsed,
      bucket=opts.bucket,
      target_count=target_count,
      fleet_count=fleet_count,
      now_tag=now_tag,
      bazel_rc=bazel_rc,
      fleet_check_elapsed=fleet_elapsed,
      query_elapsed=query_elapsed,
      bazel_elapsed=bazel_elapsed,
      critical_path=critical_path,
      failure_lines=extract_failure_lines(log_text),
  )
  report_file.write_text(report_content, encoding="utf-8")
  log(f"Report generated at: {report_file}")
  return bazel_rc


if __name__ == "__main__":
  sys.exit(main())
