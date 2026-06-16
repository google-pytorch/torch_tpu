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

"""Script to find the latest successful workflow run (build + TPU test) and return GCS path."""

import argparse
from collections.abc import Mapping, Sequence
import json
import os
import subprocess
import sys
import time
from typing import Any, Final
import urllib.request

# Global configuration
GITHUB_TOKEN: Final[str | None] = os.environ.get("GITHUB_TOKEN")
GITHUB_REPOSITORY: Final[str] = os.environ.get(
    "GITHUB_REPOSITORY", "google-pytorch/torch_tpu"
)


DEFAULT_PYTHON_VERSION: Final[str] = "cp312"


def check_env() -> None:
  """Validates that the GITHUB_TOKEN environment variable is set."""
  if not GITHUB_TOKEN:
    print("ERROR: GITHUB_TOKEN environment variable not set.", file=sys.stderr)
    sys.exit(1)


def get_runs(target_sha: str) -> Mapping[str, Any]:
  """Queries the GitHub API for build_wheel runs matching target_sha."""
  url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/actions/workflows/build_wheel.yml/runs?head_sha={target_sha}"
  req = urllib.request.Request(
      url,
      headers={
          "Authorization": f"Bearer {GITHUB_TOKEN}",
          "Accept": "application/vnd.github+json",
          "User-Agent": "torch-tpu-ci-agent",
      },
  )
  with urllib.request.urlopen(req) as response:
    return json.loads(response.read().decode())


def get_jobs(run_id: int) -> Mapping[str, Any]:
  """Queries the GitHub API for jobs of a specific workflow run."""
  url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/actions/runs/{run_id}/jobs"
  req = urllib.request.Request(
      url,
      headers={
          "Authorization": f"Bearer {GITHUB_TOKEN}",
          "Accept": "application/vnd.github+json",
          "User-Agent": "torch-tpu-ci-agent",
      },
  )
  with urllib.request.urlopen(req) as response:
    return json.loads(response.read().decode())


def check_jobs_status(jobs: Sequence[Mapping[str, Any]]) -> str:
  """Checks that the required build and TPU test jobs completed successfully."""
  build_job = None
  tpu_job = None

  for j in jobs:
    if j["name"] == "Build wheel on CPU":
      build_job = j
    elif j["name"] == "Test wheel on linux-x86-ct5lp-224-8tpu":
      tpu_job = j

  # 1. Verify build job
  if not build_job or build_job["status"] != "completed":
    return "active"
  if build_job["conclusion"] != "success":
    return "failed"

  # 2. Verify TPU test job (which depends on build job)
  if not tpu_job or tpu_job["status"] != "completed":
    return "active"
  if tpu_job["conclusion"] != "success":
    return "failed"

  return "success"


def verify_gcs_path(
    run_number: int, run_attempt: int, python_version: str
) -> tuple[bool, str | None]:
  """Verifies GCS path existence and returns the exact wheel path."""
  gcs_path = (
      "gs://torch-tpu-ci-transient/artifacts/Build and Test"
      f" Wheels/{run_number}/{run_attempt}"
  )
  cmd = ["gcloud", "storage", "ls", f"{gcs_path}/*{python_version}*.whl"]
  try:
    res = subprocess.run(cmd, capture_output=True, check=True, text=True)
    lines = res.stdout.strip().split("\n")
    if lines and lines[0]:
      return True, lines[0]
    return False, None
  except subprocess.CalledProcessError as e:
    print(
        f"Warning: GCS path check failed for {gcs_path} with {python_version}:"
        f" {e.stderr}",
        file=sys.stderr,
    )
    return False, None


def find_run_once(
    target_sha: str, python_version: str
) -> tuple[str, str | None]:
  """Checks runs once. Returns (status, result_value)."""
  data = get_runs(target_sha)
  runs = data.get("workflow_runs", [])

  if not runs:
    return "no_runs", None

  matched_run_exists = False
  active_run_exists = False
  failed_runs = []

  for r in runs:
    matched_run_exists = True
    jobs_data = get_jobs(r["id"])
    jobs = jobs_data.get("jobs", [])
    run_status = check_jobs_status(jobs)

    print(
        f"Run {r['run_number']} (Attempt {r['run_attempt']}):"
        f" status={r['status']}, conclusion={r['conclusion']},"
        f" jobs_status={run_status}",
        file=sys.stderr,
    )

    if run_status == "success":
      exists, wheel_path = verify_gcs_path(
          r["run_number"], r["run_attempt"], python_version
      )
      if exists:
        return "success", wheel_path
      else:
        print(
            f"Run {r['run_number']} jobs succeeded, but wheel for"
            f" {python_version} not found in GCS yet.",
            file=sys.stderr,
        )
        active_run_exists = True
    elif run_status == "active":
      active_run_exists = True
    else:
      failed_runs.append(r)

  if not matched_run_exists:
    return "no_runs", None
  if active_run_exists:
    return "active", None

  # All matched runs failed
  for r in failed_runs:
    print(
        f"ERROR: Build run {r['run_number']} completed with job failures.",
        file=sys.stderr,
    )
  return "failed", None


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("target_sha", help="Enforce matching this head SHA.")
  parser.add_argument(
      "--wait", action="store_true", help="Wait and retry up to 30 minutes."
  )
  parser.add_argument(
      "-p",
      "--python-version",
      help="Specific python version tag to look for (e.g. cp312).",
  )
  args = parser.parse_args()

  check_env()
  target_sha = args.target_sha

  python_version = args.python_version or DEFAULT_PYTHON_VERSION

  timeout_minutes = 60
  sleep_seconds = 60
  start_time = time.monotonic()

  status = "active"
  result = None

  while True:
    try:
      status, result = find_run_once(target_sha, python_version)
      if status == "success":
        print(result)
        sys.exit(0)
      elif status == "failed":
        print(
            "ERROR: build_wheel workflow failed for this commit.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"Warning: Temporary error fetching workflow status: {e}",
          file=sys.stderr,
      )
      status = "active"

    elapsed_minutes = (time.monotonic() - start_time) // 60
    if args.wait and elapsed_minutes < timeout_minutes:
      print(
          "Waiting for build_wheel workflow to complete successfully..."
          f" ({int(elapsed_minutes)}/{timeout_minutes})",
          file=sys.stderr,
      )
      time.sleep(sleep_seconds)
    else:
      break

  print(
      "Error: build_wheel workflow not found or failed after timeout.",
      file=sys.stderr,
  )
  sys.exit(1)


if __name__ == "__main__":
  main()
