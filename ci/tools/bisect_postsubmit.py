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

"""Drives automated git bisection for postsubmit CI breakages."""

import argparse
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
import os
import re
import subprocess
import sys
from typing import Any, Final

_MAX_ITERATIONS_DEFAULT: Final[int] = 5


@dataclasses.dataclass(frozen=True)
class BisectionResult:
  """Result of a bisection execution."""

  culprit_sha: str | None
  skipped: bool
  skip_reason: str = ""
  iterations: int = 0
  narrowed_sha: str | None = None


def set_github_output(name: str, value: str) -> None:
  """Writes a key-value output pair to $GITHUB_OUTPUT if set."""
  output_file = os.environ.get("GITHUB_OUTPUT")
  if output_file:
    with open(output_file, "a", encoding="utf-8") as f:
      f.write(f"{name}={value}\n")


def build_bazel_args(
    bazel_config: str,
    fail_fast: bool,
    seed: str | None = None,
    repo_root: str | None = None,
) -> tuple[list[str], list[str]]:
  """Builds bazel build and bazel test arguments.

  Args:
    bazel_config: Bazel config name (e.g. ci_tpu_v7, ci_cpu).
    fail_fast: If True, stop execution on the first failure.
    seed: Random seed for reproducible test execution.
    repo_root: Path to repository root directory.

  Returns:
    A tuple of (build_args, test_args).
  """
  build_args = [f"--config={bazel_config}"]
  test_args = [f"--config={bazel_config}"]

  if fail_fast:
    build_args.append("--nokeep_going")
    test_args.extend(["--nokeep_going", "--test_runner_fail_fast"])

  if seed:
    test_args.extend([
        f"--test_arg=--test_random_seed={seed}",
        f"--test_env=PYTHONHASHSEED={seed}",
    ])

  if "tpu" in bazel_config:
    root = repo_root or os.getcwd()
    run_under = os.path.join(root, "ci/tools/parallel_accelerator_execute.sh")
    test_args.append(f"--run_under={run_under}")

  return build_args, test_args


def job_matches_config(job_name: str, bazel_config: str, runner: str) -> bool:
  """Returns True if the workflow job corresponds to the given config/runner."""
  name_lower = job_name.lower()
  config_lower = bazel_config.lower()
  runner_lower = runner.lower()

  if runner and runner_lower in name_lower:
    return True
  if bazel_config and config_lower in name_lower:
    return True

  if "v7" in config_lower and ("tpu v7" in name_lower or "tpu7" in name_lower):
    return True
  if "v6" in config_lower and (
      "tpu v6" in name_lower or "tpu6" in name_lower or "ct6e" in name_lower
  ):
    return True
  if "v5" in config_lower and (
      "tpu v5" in name_lower or "tpu5" in name_lower or "ct5lp" in name_lower
  ):
    return True
  if "cpu" in config_lower and ("cpu" in name_lower or "n2-64" in name_lower):
    return True

  return False


def get_job_test_status(job: Mapping[str, Any]) -> str | None:
  """Returns the test execution conclusion (success, failure, skipped, etc.)."""
  steps = job.get("steps") or []
  for step in steps:
    step_name = step.get("name", "")
    if "Run Bazel Test" in step_name:
      return step.get("conclusion")
  return job.get("conclusion")


def check_prior_run(
    current_run_id: int | None,
    bazel_config: str,
    runner: str,
    run_cmd_fn: Callable[..., Any] = subprocess.run,
) -> tuple[bool, str]:
  """Checks if bisection should be skipped based on prior runs and current jobs.

  Args:
    current_run_id: GitHub Actions workflow_run.id of the failed run.
    bazel_config: Target Bazel config.
    runner: Target runner label.
    run_cmd_fn: Subprocess runner function for testing.

  Returns:
    Tuple of (should_skip, reason).
  """
  if not current_run_id:
    return False, "Manual dispatch or no current run ID provided"

  # 1. Check if failure in CURRENT_RUN_ID was isolated to other configs.
  view_cmd = ["gh", "run", "view", str(current_run_id), "--json", "jobs"]
  view_res = run_cmd_fn(view_cmd, capture_output=True, text=True)
  if view_res.returncode == 0 and view_res.stdout.strip():
    try:
      data = json.loads(view_res.stdout)
      jobs = data.get("jobs", [])
      failed_jobs = [j for j in jobs if j.get("conclusion") == "failure"]
      if failed_jobs:
        matching_failed = [
            j
            for j in failed_jobs
            if job_matches_config(j.get("name", ""), bazel_config, runner)
        ]
        if not matching_failed:
          failed_names = [j.get("name", "") for j in failed_jobs]
          return (
              True,
              (
                  f"Failure in run {current_run_id} was isolated to other"
                  f" configs ({failed_names}); skipping bisection on"
                  f" {bazel_config}."
              ),
          )
    except json.JSONDecodeError:
      pass

  # 2. Check if main branch was already failing before CURRENT_RUN_ID.
  list_cmd = [
      "gh",
      "run",
      "list",
      "--workflow=CI - Postsubmit",
      "--branch=main",
      "--limit=20",
      "--json",
      "databaseId,status,conclusion,headSha",
  ]
  list_res = run_cmd_fn(list_cmd, capture_output=True, text=True)
  if list_res.returncode != 0 or not list_res.stdout.strip():
    return False, "Could not fetch prior postsubmit runs"

  try:
    runs = json.loads(list_res.stdout)
  except json.JSONDecodeError:
    return False, "Failed to parse prior runs JSON"

  prior_completed = [
      r
      for r in runs
      if r.get("databaseId", 0) < current_run_id
      and r.get("status") == "completed"
  ]

  for r in prior_completed:
    prior_id = r.get("databaseId")
    p_view_cmd = ["gh", "run", "view", str(prior_id), "--json", "jobs"]
    p_view_res = run_cmd_fn(p_view_cmd, capture_output=True, text=True)
    if p_view_res.returncode != 0 or not p_view_res.stdout.strip():
      continue
    try:
      p_jobs = json.loads(p_view_res.stdout).get("jobs", [])
    except json.JSONDecodeError:
      continue

    for job in p_jobs:
      if job_matches_config(job.get("name", ""), bazel_config, runner):
        test_status = get_job_test_status(job)
        if test_status in ("success", "failure"):
          if test_status == "failure":
            return (
                True,
                (
                    "Main branch was already failing prior to run"
                    f" {current_run_id} (in run {prior_id}); skipping bisect."
                ),
            )
          return (
              False,
              f"Prior run {prior_id} succeeded on {bazel_config}; proceeding.",
          )

  return False, "No prior run with executed tests found; proceeding."


def find_good_sha(
    bad_sha: str,
    bazel_config: str,
    runner: str,
    run_cmd_fn: Callable[..., Any] = subprocess.run,
) -> str | None:
  """Finds successful postsubmit commit that is an ancestor of BAD_SHA."""
  list_cmd = [
      "gh",
      "run",
      "list",
      "--workflow=CI - Postsubmit",
      "--branch=main",
      "--status=success",
      "--limit=20",
      "--json",
      "databaseId,headSha,conclusion,status",
  ]
  res = run_cmd_fn(list_cmd, capture_output=True, text=True)
  if res.returncode != 0 or not res.stdout.strip():
    return None

  try:
    runs = json.loads(res.stdout)
  except json.JSONDecodeError:
    return None

  for r in runs:
    candidate_sha = r.get("headSha")
    if not candidate_sha or candidate_sha == bad_sha:
      continue

    ancestor_cmd = [
        "git",
        "merge-base",
        "--is-ancestor",
        candidate_sha,
        bad_sha,
    ]
    anc_res = run_cmd_fn(ancestor_cmd, capture_output=True, text=True)
    if anc_res.returncode != 0:
      continue

    run_id = r.get("databaseId")
    if run_id:
      view_cmd = ["gh", "run", "view", str(run_id), "--json", "jobs"]
      view_res = run_cmd_fn(view_cmd, capture_output=True, text=True)
      if view_res.returncode == 0 and view_res.stdout.strip():
        try:
          jobs = json.loads(view_res.stdout).get("jobs", [])
          matching_jobs = [
              j
              for j in jobs
              if job_matches_config(j.get("name", ""), bazel_config, runner)
          ]
          if matching_jobs:
            test_status = get_job_test_status(matching_jobs[0])
            if test_status != "success":
              continue
        except json.JSONDecodeError:
          pass

    return candidate_sha

  return None


def resolve_good_sha(
    input_good_sha: str | None,
    bad_sha: str,
    bazel_config: str,
    runner: str,
    run_cmd_fn: Callable[..., Any] = subprocess.run,
) -> str:
  """Resolves and validates GOOD_SHA as a valid ancestor of BAD_SHA."""
  good_sha = (input_good_sha or "").strip()
  if not good_sha:
    found = find_good_sha(bad_sha, bazel_config, runner, run_cmd_fn=run_cmd_fn)
    if found:
      good_sha = found

  if not good_sha:
    rev_cmd = ["git", "rev-parse", f"{bad_sha}~5"]
    rev_res = run_cmd_fn(rev_cmd, capture_output=True, text=True)
    if rev_res.returncode == 0 and rev_res.stdout.strip():
      good_sha = rev_res.stdout.strip()
    else:
      rev_main_cmd = ["git", "rev-parse", "origin/main~5"]
      rev_main_res = run_cmd_fn(rev_main_cmd, capture_output=True, text=True)
      if rev_main_res.returncode == 0 and rev_main_res.stdout.strip():
        good_sha = rev_main_res.stdout.strip()
      else:
        raise ValueError(
            f"Could not resolve GOOD_SHA fallback for BAD_SHA ({bad_sha})"
        )

  anc_cmd = ["git", "merge-base", "--is-ancestor", good_sha, bad_sha]
  anc_res = run_cmd_fn(anc_cmd, capture_output=True, text=True)
  if anc_res.returncode != 0:
    raise ValueError(f"GOOD_SHA ({good_sha}) is not an ancestor of {bad_sha}.")

  return good_sha


def run_commit_test(
    sha: str,
    build_args: Sequence[str],
    test_args: Sequence[str],
    test_targets: Sequence[str],
    run_cmd_fn: Callable[..., Any] = subprocess.run,
) -> bool:
  """Checks out sha, runs bazel build, and runs bazel test."""
  checkout_cmd = ["git", "checkout", sha, "--quiet"]
  run_cmd_fn(checkout_cmd, check=True)

  build_cmd = ["bazel", "build", *build_args, *test_targets]
  print(f"Building targets on {sha}: {' '.join(build_cmd)}")
  build_res = run_cmd_fn(build_cmd)
  if build_res.returncode != 0:
    print(f"Build failed on {sha} with returncode {build_res.returncode}")
    return False

  test_cmd = ["bazel", "test", *test_args, *test_targets]
  print(f"Testing targets on {sha}: {' '.join(test_cmd)}")
  test_res = run_cmd_fn(test_cmd)
  if test_res.returncode != 0:
    print(f"Test failed on {sha} with returncode {test_res.returncode}")
    return False

  return True


def run_bisection(
    bad_sha: str,
    good_sha: str,
    build_args: Sequence[str],
    test_args: Sequence[str],
    test_targets: Sequence[str],
    max_iterations: int = _MAX_ITERATIONS_DEFAULT,
    run_cmd_fn: Callable[..., Any] = subprocess.run,
) -> BisectionResult:
  """Runs the automated bisection workflow."""
  print(f"Starting automated bisection: {good_sha} .. {bad_sha}")
  print(f"Target tests: {' '.join(test_targets)}")
  print(f"Build args: {' '.join(build_args)}")
  print(f"Test args: {' '.join(test_args)}")

  # 1. Pre-verify failure reproducibility on BAD_SHA first
  print("==========================================================")
  print(f"Pre-verifying failure reproducibility on BAD_SHA ({bad_sha})...")
  print("==========================================================")
  bad_passes = run_commit_test(
      bad_sha, build_args, test_args, test_targets, run_cmd_fn=run_cmd_fn
  )
  if bad_passes:
    reason = f"Failure on {bad_sha} not reproduced; skipping."
    print(f"::warning::{reason}")
    return BisectionResult(culprit_sha=None, skipped=True, skip_reason=reason)

  print(
      f"Failure reproduced on {bad_sha}; proceeding to baseline verification."
  )

  # 2. Pre-verify baseline on GOOD_SHA
  print("==========================================================")
  print(f"Pre-verifying baseline passes on GOOD_SHA ({good_sha})...")
  print("==========================================================")
  good_passes = run_commit_test(
      good_sha, build_args, test_args, test_targets, run_cmd_fn=run_cmd_fn
  )
  if not good_passes:
    raise RuntimeError(f"Baseline failed on GOOD_SHA ({good_sha}).")

  print(f"Baseline verified on {good_sha}. Tests pass.")

  # 3. Check commit count between GOOD_SHA and BAD_SHA (short-circuit 1-commit)
  count_cmd = ["git", "rev-list", "--count", f"{good_sha}..{bad_sha}"]
  count_res = run_cmd_fn(count_cmd, capture_output=True, text=True, check=True)
  commit_count = int(count_res.stdout.strip())
  if commit_count <= 1:
    print(
        f"Commit count between {good_sha} and {bad_sha} is {commit_count}."
        f" Short-circuiting directly to culprit {bad_sha}."
    )
    return BisectionResult(culprit_sha=bad_sha, skipped=False, iterations=0)

  # 4. Multi-commit git bisect
  run_cmd_fn(["git", "bisect", "start", bad_sha, good_sha], check=True)
  try:
    for iteration in range(1, max_iterations + 1):
      rev_res = run_cmd_fn(
          ["git", "rev-parse", "HEAD"],
          capture_output=True,
          text=True,
          check=True,
      )
      current_sha = rev_res.stdout.strip()
      print("==========================================================")
      print(f"Testing candidate [{iteration}/{max_iterations}]: {current_sha}")
      print("==========================================================")

      passed = run_commit_test(
          current_sha,
          build_args,
          test_args,
          test_targets,
          run_cmd_fn=run_cmd_fn,
      )
      bisect_action = "good" if passed else "bad"
      print(
          f"Commit {current_sha} {'passed' if passed else 'failed'}."
          f" Marking as {bisect_action}."
      )
      res = run_cmd_fn(
          ["git", "bisect", bisect_action],
          capture_output=True,
          text=True,
          check=True,
      )
      bisect_output = (res.stdout or "") + (res.stderr or "")
      print(bisect_output)

      first_bad_re = r"(\S+)\s+is the first '?bad'? commit"
      m = re.search(first_bad_re, bisect_output)
      if m:
        culprit = m.group(1)
        print(f"Culprit commit identified: {culprit}")
        return BisectionResult(
            culprit_sha=culprit, skipped=False, iterations=iteration
        )

    print(f"::warning::Reached max iterations ({max_iterations}).")
    narrow_res = run_cmd_fn(
        ["git", "rev-parse", "refs/bisect/bad"],
        capture_output=True,
        text=True,
    )
    narrowed = (
        narrow_res.stdout.strip() if narrow_res.returncode == 0 else "unknown"
    )
    print(f"Current narrowed bad commit: {narrowed}")
    return BisectionResult(
        culprit_sha=None,
        skipped=False,
        iterations=max_iterations,
        narrowed_sha=narrowed,
    )
  finally:
    run_cmd_fn(["git", "bisect", "reset"], capture_output=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Drive automated git bisection for postsubmit CI breakages."
  )
  parser.add_argument(
      "--good-sha", default="", help="Last known good commit SHA."
  )
  parser.add_argument(
      "--bad-sha", default="", help="First known bad commit SHA."
  )
  parser.add_argument(
      "--event-bad-sha",
      default="",
      help="Fallback bad SHA from workflow_run event.",
  )
  parser.add_argument(
      "--test-targets",
      default="//...",
      help="Space-separated list of test targets.",
  )
  parser.add_argument(
      "--fail-fast",
      default="true",
      help="Stop test execution on first failure.",
  )
  parser.add_argument(
      "--seed",
      default="",
      help="Random seed for reproducible test execution.",
  )
  parser.add_argument(
      "--bazel-config", default="ci_tpu_v7", help="Bazel config to use."
  )
  parser.add_argument(
      "--runner",
      default="linux-x86-tpu7x-224-4tpu",
      help="Runner label.",
  )
  parser.add_argument(
      "--current-run-id",
      type=int,
      default=None,
      help="Current workflow run database ID (for prior-run inspection).",
  )
  parser.add_argument(
      "--max-iterations",
      type=int,
      default=_MAX_ITERATIONS_DEFAULT,
      help="Maximum bisection iterations.",
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  fail_fast = str(args.fail_fast).lower() in ("true", "1", "yes")

  # 1. Check prior run guard if triggered by workflow_run
  if args.current_run_id:
    should_skip, reason = check_prior_run(
        args.current_run_id, args.bazel_config, args.runner
    )
    if should_skip:
      print(f"::notice::{reason}")
      set_github_output("skip", "true")
      return 0

  # 2. Resolve BAD_SHA
  bad_sha = (args.bad_sha or args.event_bad_sha or "").strip()
  if not bad_sha:
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    bad_sha = res.stdout.strip()

  # 3. Resolve GOOD_SHA
  try:
    good_sha = resolve_good_sha(
        args.good_sha, bad_sha, args.bazel_config, args.runner
    )
  except ValueError as err:
    print(f"::error::{err}")
    return 1

  # 4. Construct Bazel arguments
  build_args, test_args = build_bazel_args(
      args.bazel_config, fail_fast, args.seed or None
  )

  test_targets = [t for t in args.test_targets.split() if t]
  if not test_targets:
    test_targets = ["//..."]

  # 5. Run bisection
  try:
    result = run_bisection(
        bad_sha=bad_sha,
        good_sha=good_sha,
        build_args=build_args,
        test_args=test_args,
        test_targets=test_targets,
        max_iterations=args.max_iterations,
    )
  except RuntimeError as err:
    print(f"::error::{err}")
    return 1

  if result.skipped:
    set_github_output("skip", "true")
    return 0

  set_github_output("skip", "false")
  if result.culprit_sha:
    set_github_output("CULPRIT_SHA", result.culprit_sha)
    print(f"Bisection complete! Culprit commit: {result.culprit_sha}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
