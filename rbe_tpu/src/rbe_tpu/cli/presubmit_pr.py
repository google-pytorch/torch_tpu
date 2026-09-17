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

"""GitHub PR presubmit runner for Cloud TPU v5e tests via GCS mailbox."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Final

from rbe_tpu.cli import bazel_test

DEFAULT_BUCKET: Final[str] = "torch-tpu-mailbox-eval"
DEFAULT_REPO: Final[str] = "google-pytorch/torch_tpu"
DEFAULT_FILTER: Final[str] = (
    "presubmit-v5,-fails-on-tpu-v5,-nopresubmit,-notest,-nobuild,-requires-tpu-v5lite:8"
)
DEFAULT_SCOPE: Final[str] = "//tests/..."

REPLACE_LABEL: Final[str] = "ci:replace-tpu-v5"
SHADOW_LABEL: Final[str] = "ci:relay-tpu-v5"
COMMENT_MARKER: Final[str] = "<!-- torch-tpu-v5e-mailbox -->"
GATING_CONTEXT: Final[str] = "Presubmit on linux-x86-ct5lp-224-8tpu"
SHADOW_CONTEXT: Final[str] = "TPU v5e relay (shadow)"

HELP_TEXT: Final[str] = """Usage: rbe_tpu/deploy/rbe_tpu_presubmit_pr.sh [options]

Executes the torch_tpu single-chip presubmit suite on Cloud TPU v5e hardware
via the GCS mailbox runner.

Modes:
  shadow       Advisory run. Reports under "TPU v5e relay (shadow)" and
               does not block PR merge (default).
  replacement  Gating run. Reports under "Presubmit on linux-x86-ct5lp-224-8tpu",
               resolving the required status check.

Options:
  --pr N                Pull request number to test.
  --repo OWNER/NAME     GitHub repository (default: google-pytorch/torch_tpu).
  --mode MODE           shadow (default) or replacement.
  --sha SHA             Commit SHA to report against (default: PR head SHA).
  --bucket NAME         GCS mailbox bucket (default: torch-tpu-mailbox-eval).
  --target TARGET       Specific test target to run. Can be repeated.
  --jobs N              Concurrent test jobs (default: 28).
  --bazel-config NAME   Bazel config to inherit (default: ci_tpu_v5_mailbox).
  --bazel-flag FLAG     Extra flag passed to bazel test. Can be repeated.
  --output-dir DIR      Directory to write execution reports and logs.
  --add-label           Add the mode's label to the PR (ci:replace-tpu-v5 or ci:relay-tpu-v5).
  --skip-head-check     Run even if the local checkout is not at the PR head commit.
  --no-report           Do not post commit statuses, labels, or comments to GitHub.
  --dry-run             Resolve targets and show execution plan without running.
  -h, --help            Show this help."""


class CLIError(Exception):
  """Raised when invalid CLI arguments are given."""


@dataclasses.dataclass(frozen=True)
class Options:
  pr: str = ""
  repo: str = DEFAULT_REPO
  mode: str = "shadow"
  sha: str = ""
  bucket: str = DEFAULT_BUCKET
  targets: tuple[str, ...] = ()
  jobs: int = 28
  bazel_config: str = "rbe_tpu"
  bazel_flags: tuple[str, ...] = ()
  output_dir: pathlib.Path | None = None
  add_label: bool = False
  skip_head_check: bool = False
  no_report: bool = False
  dry_run: bool = False
  test_timeout: int = 900
  scope: str = DEFAULT_SCOPE
  tag_filter: str = DEFAULT_FILTER
  show_help: bool = False


def parse_args(argv: Sequence[str]) -> Options:
  pr = ""
  repo = DEFAULT_REPO
  mode = "shadow"
  sha = ""
  bucket = DEFAULT_BUCKET
  targets: list[str] = []
  jobs = 28
  bazel_config = "rbe_tpu"
  bazel_flags: list[str] = []
  output_dir: pathlib.Path | None = None
  add_label = False
  skip_head_check = False
  no_report = False
  dry_run = False
  test_timeout = 900
  scope = DEFAULT_SCOPE
  tag_filter = DEFAULT_FILTER

  idx = 0
  n = len(argv)
  while idx < n:
    arg = argv[idx]
    if arg in ("-h", "--help"):
      return Options(show_help=True)
    if arg == "--pr":
      if idx + 1 >= n:
        raise CLIError("--pr requires an argument.")
      pr = argv[idx + 1]
      idx += 2
    elif arg == "--repo":
      if idx + 1 >= n:
        raise CLIError("--repo requires an argument.")
      repo = argv[idx + 1]
      idx += 2
    elif arg == "--mode":
      if idx + 1 >= n:
        raise CLIError("--mode requires an argument.")
      mode = argv[idx + 1]
      idx += 2
    elif arg == "--sha":
      if idx + 1 >= n:
        raise CLIError("--sha requires an argument.")
      sha = argv[idx + 1]
      idx += 2
    elif arg == "--bucket":
      if idx + 1 >= n:
        raise CLIError("--bucket requires an argument.")
      bucket = argv[idx + 1]
      idx += 2
    elif arg == "--target":
      if idx + 1 >= n:
        raise CLIError("--target requires an argument.")
      targets.append(argv[idx + 1])
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
    elif arg == "--output-dir":
      if idx + 1 >= n:
        raise CLIError("--output-dir requires an argument.")
      output_dir = pathlib.Path(argv[idx + 1])
      idx += 2
    elif arg == "--add-label":
      add_label = True
      idx += 1
    elif arg == "--skip-head-check":
      skip_head_check = True
      idx += 1
    elif arg == "--no-report":
      no_report = True
      idx += 1
    elif arg == "--dry-run":
      dry_run = True
      idx += 1
    else:
      raise CLIError(f"Unknown argument: {arg}")

  if mode not in ("shadow", "replacement"):
    raise CLIError(f"--mode must be 'shadow' or 'replacement', got '{mode}'")

  return Options(
      pr=pr,
      repo=repo,
      mode=mode,
      sha=sha,
      bucket=bucket,
      targets=tuple(targets),
      jobs=jobs,
      bazel_config=bazel_config,
      bazel_flags=tuple(bazel_flags),
      output_dir=output_dir,
      add_label=add_label,
      skip_head_check=skip_head_check,
      no_report=no_report,
      dry_run=dry_run,
      test_timeout=test_timeout,
      scope=scope,
      tag_filter=tag_filter,
  )


def get_target_context(mode: str) -> str:
  return GATING_CONTEXT if mode == "replacement" else SHADOW_CONTEXT


def get_target_label(mode: str) -> str:
  return REPLACE_LABEL if mode == "replacement" else SHADOW_LABEL


def fetch_pr_head_sha(repo: str, pr: str) -> str:
  res = subprocess.run(
      ["gh", "api", f"repos/{repo}/pulls/{pr}", "--jq", ".head.sha"],
      capture_output=True,
      text=True,
      check=False,
  )
  return res.stdout.strip() if res.returncode == 0 else ""


def apply_and_verify_label(repo: str, pr: str, label: str) -> bool:
  subprocess.run(
      ["gh", "pr", "edit", pr, "-R", repo, "--add-label", label],
      capture_output=True,
      text=True,
      check=False,
  )
  res = subprocess.run(
      [
          "gh",
          "pr",
          "view",
          pr,
          "-R",
          repo,
          "--json",
          "labels",
          "--jq",
          ".labels[].name",
      ],
      capture_output=True,
      text=True,
      check=False,
  )
  labels = [line.strip() for line in res.stdout.splitlines()]
  return label in labels


def post_status(
    repo: str, sha: str, state: str, description: str, context: str
) -> None:
  desc = description[:140]
  subprocess.run(
      [
          "gh",
          "api",
          "-X",
          "POST",
          f"repos/{repo}/statuses/{sha}",
          "-f",
          f"state={state}",
          "-f",
          f"context={context}",
          "-f",
          f"description={desc}",
      ],
      capture_output=True,
      text=True,
      check=False,
  )


def build_bazel_args(opts: Options, targets: Sequence[str]) -> list[str]:
  args = ["test"]
  if opts.bazel_config:
    if not opts.targets and opts.bazel_config == "ci_tpu_v5_mailbox":
      args.append("--config=ci_tpu_v5_presubmit")
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


def format_presubmit_report(
    *,
    verdict: str,
    mode: str,
    context: str,
    elapsed: int,
    bucket: str,
    target_count: int,
    fleet_count: int,
    now_tag: str,
    bazel_rc: int,
    failure_lines: Sequence[str],
) -> str:
  report = f"""{COMMENT_MARKER}
# TPU Mailbox Presubmit Report

**Verdict:** {verdict}
**Mode:** {mode}
**Context:** `{context}`
**Wall Clock Time:** {elapsed}s
**Bucket:** `gs://{bucket}`
**Targets Evaluated:** {target_count}
**Fleet Capacity:** {fleet_count} chips
**Timestamp:** {now_tag}

## Summary

| Metric | Value |
| :--- | :--- |
| **Status** | **{verdict}** |
| **Duration** | {elapsed}s |
| **Exit Code** | {bazel_rc} |
| **Targets** | {target_count} |
| **Fleet Capacity** | {fleet_count} chips registered |
"""
  if bazel_rc != 0 and failure_lines:
    joined = "\n".join(failure_lines)
    report += f"\n### Failures\n```\n{joined}\n```\n"
  return report


def update_or_post_pr_comment(
    repo: str, pr: str, report_content: str
) -> None:
  res = subprocess.run(
      ["gh", "api", f"repos/{repo}/issues/{pr}/comments"],
      capture_output=True,
      text=True,
      check=False,
  )
  existing_id = ""
  if res.returncode == 0:
    try:
      comments = json.loads(res.stdout)
      for c in comments:
        if COMMENT_MARKER in c.get("body", ""):
          existing_id = str(c.get("id"))
    except Exception:  # pylint: disable=broad-except
      pass

  if existing_id:
    subprocess.run(
        [
            "gh",
            "api",
            "-X",
            "PATCH",
            f"repos/{repo}/issues/comments/{existing_id}",
            "-f",
            f"body={report_content}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
  else:
    subprocess.run(
        ["gh", "pr", "comment", pr, "-R", repo, "--body", report_content],
        capture_output=True,
        text=True,
        check=False,
    )


def default_repo_root() -> pathlib.Path:
  here = pathlib.Path(__file__).resolve()
  for parent in here.parents:
    if (parent / "deploy" / "rbe_tpu_presubmit_pr.sh").exists():
      return parent
  return here.parents[3]


def main(argv: Sequence[str] | None = None) -> int:
  args = sys.argv[1:] if argv is None else argv
  try:
    opts = parse_args(args)
  except CLIError as err:
    print(f"ERROR: {err}", file=sys.stderr)
    return 2

  if opts.show_help:
    print(HELP_TEXT)
    return 0

  target_context = get_target_context(opts.mode)
  target_label = get_target_label(opts.mode)

  now_tag = datetime.datetime.now(datetime.timezone.utc).strftime(
      "%Y%m%d_%H%M%SZ"
  )
  prefix = f"pr{opts.pr}" if opts.pr else "run"
  output_dir = opts.output_dir or (
      default_repo_root() / "mailbox_reports" / f"{prefix}_{now_tag}"
  )
  try:
    output_dir.mkdir(parents=True, exist_ok=True)
  except OSError:
    fallback_root = pathlib.Path(os.environ.get("TEST_TMPDIR", "/tmp"))
    output_dir = fallback_root / "mailbox_reports" / f"{prefix}_{now_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

  sha = opts.sha
  if opts.pr and not sha:
    sha = fetch_pr_head_sha(opts.repo, opts.pr)

  if sha and not opts.skip_head_check:
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    local_sha = res.stdout.strip()
    if local_sha and local_sha != sha:
      print(
          f"WARNING: Local checkout ({local_sha[:10]}) does not match PR"
          f" #{opts.pr} head ({sha[:10]}).\nTests will run against local files"
          " rather than the PR commit.\nPass --skip-head-check to proceed"
          " anyway.",
          file=sys.stderr,
      )

  print("========================================================================")
  print("TPU Mailbox Presubmit Pipeline")
  print("========================================================================")
  print(f"Bucket:          gs://{opts.bucket}")
  print(f"Output Directory:{output_dir}")
  print(f"Mode:            {opts.mode}")
  if opts.pr:
    print(f"Pull Request:    #{opts.pr} ({opts.repo})")
  if sha:
    print(f"Commit SHA:      {sha}")
  print(f"Context:         {target_context}")
  print(f"Target Label:    {target_label}")
  print("========================================================================")

  fleet_count = bazel_test.check_fleet_count(opts.bucket)
  if opts.targets:
    final_targets = list(opts.targets)
  else:
    print(
        f"Querying test targets in {opts.scope} matching {opts.tag_filter}..."
    )
    expr = bazel_test.build_query_expr(opts.scope, opts.tag_filter)
    final_targets = bazel_test.filter_single_chip_targets(
        bazel_test.run_bazel_query(expr)
    )

  target_count = len(final_targets)
  print(f"Resolved {target_count} target(s) to execute.")
  if target_count == 0:
    print("ERROR: No test targets resolved.", file=sys.stderr)
    return 1

  if opts.dry_run:
    print(
        f"[DRY RUN] Execution plan confirmed. Resolved targets ({target_count}):"
    )
    for tgt in final_targets:
      print(f"  {tgt}")
    if opts.add_label and opts.pr:
      print(f"[DRY RUN] Would add label '{target_label}' to PR #{opts.pr}")
    print(f"[DRY RUN] Would report under context: '{target_context}'")
    return 0

  if opts.pr and opts.add_label and not opts.no_report:
    print(f"Adding label '{target_label}' to PR #{opts.pr}...")
    label_ok = apply_and_verify_label(opts.repo, opts.pr, target_label)
    if not label_ok:
      if opts.mode == "replacement":
        print(
            f"ERROR: could not apply '{target_label}' to PR #{opts.pr}.\n"
            f"Replacement mode publishes under '{GATING_CONTEXT}', the context"
            " branch\nprotection requires. Without the label the normal"
            " workflow never learns a\nfleet run is happening, so a verdict"
            " posted here would sit next to the real\nct5lp job wearing its"
            " name. Refusing to post.\nAdd the label by hand and re-run, or use"
            " --mode shadow.",
            file=sys.stderr,
        )
        return 1
      print(
          f"WARNING: failed to add label '{target_label}' to PR #{opts.pr}.",
          file=sys.stderr,
      )

  if sha and not opts.no_report:
    post_status(
        opts.repo,
        sha,
        "pending",
        f"TPU mailbox presubmit running across {fleet_count} chips",
        target_context,
    )

  bazel_args = build_bazel_args(opts, final_targets)
  log_file = output_dir / "bazel_test.log"
  report_file = output_dir / "report.md"

  exec_start = int(time.time())
  print(f"Starting Bazel test execution (logs -> {log_file})...")

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

  elapsed = int(time.time()) - exec_start
  verdict = "PASSED" if bazel_rc == 0 else "FAILED"
  status_state = "success" if bazel_rc == 0 else "failure"

  print(
      f"Bazel execution completed in {elapsed}s with verdict: {verdict}"
      f" (rc={bazel_rc})"
  )

  log_text = "".join(captured_lines)
  report_content = format_presubmit_report(
      verdict=verdict,
      mode=opts.mode,
      context=target_context,
      elapsed=elapsed,
      bucket=opts.bucket,
      target_count=target_count,
      fleet_count=fleet_count,
      now_tag=now_tag,
      bazel_rc=bazel_rc,
      failure_lines=bazel_test.extract_failure_lines(log_text),
  )
  report_file.write_text(report_content, encoding="utf-8")
  print(f"Report generated at: {report_file}")

  if sha and not opts.no_report:
    post_status(
        opts.repo,
        sha,
        status_state,
        f"TPU mailbox presubmit {verdict} in {elapsed}s ({target_count}"
        " targets)",
        target_context,
    )

  if opts.pr and not opts.no_report:
    update_or_post_pr_comment(opts.repo, opts.pr, report_content)

  return bazel_rc


if __name__ == "__main__":
  sys.exit(main())
