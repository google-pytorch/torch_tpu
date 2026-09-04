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

"""Evaluates pull request state to gate CI presubmit runs.

Prevents redundant presubmit jobs (e.g. TPU v5 and TPU v7 test runs) from
queueing or consuming runner quota on pull requests that have already been
closed or merged, or are in the process of merging via Copybara.
"""

import argparse
from collections.abc import Mapping
import json
import os
import sys
import time
from typing import Any, Optional
import urllib.error
import urllib.request


def is_pr_closed_or_merged(pr_data: Mapping[str, Any]) -> bool:
  """Returns True if the PR data indicates it is closed or merged."""
  state = str(pr_data.get("state", "")).strip().lower()
  merged = pr_data.get("merged", False)
  merged_at = pr_data.get("merged_at")
  return state == "closed" or bool(merged) or bool(merged_at)


def fetch_pr_data(
    repo: str,
    pr_number: str,
    token: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Optional[dict[str, Any]]:
  """Fetches pull request details from GitHub REST API."""
  if api_url:
    url = f"{api_url.rstrip('/')}/repos/{repo}/pulls/{pr_number}"
  else:
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"

  headers = {
      "Accept": "application/vnd.github+json",
      "User-Agent": "torch-tpu-pr-gate",
      "X-GitHub-Api-Version": "2022-11-28",
  }
  if token:
    headers["Authorization"] = f"Bearer {token}"

  req = urllib.request.Request(url, headers=headers)
  try:
    with urllib.request.urlopen(req, timeout=15) as response:
      content = response.read().decode("utf-8")
      return json.loads(content)
  except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as err:
    print(
        f"Warning: Failed to query GitHub API for PR #{pr_number}: {err}",
        file=sys.stderr,
    )
    return None


def should_run_presubmit(
    event_name: str,
    repo: str,
    pr_number: str,
    token: Optional[str] = None,
    settle_seconds: float = 6.0,
    api_url: Optional[str] = None,
) -> bool:
  """Determines whether presubmit jobs should execute for the event."""
  if event_name != "pull_request":
    print(
        f"Event '{event_name}' is not pull_request. Gate passed"
        " (should_run=true)."
    )
    return True

  if not pr_number or not repo:
    print(
        "Missing PR number or repository name. Failing open (should_run=true)."
    )
    return True

  pr_data = fetch_pr_data(repo, pr_number, token=token, api_url=api_url)
  if pr_data is None:
    print("Unable to fetch initial PR status. Failing open (should_run=true).")
    return True

  if is_pr_closed_or_merged(pr_data):
    print(
        f"PR #{pr_number} is closed or merged (state={pr_data.get('state')}, "
        f"merged={pr_data.get('merged')}). Skipping presubmit"
        " (should_run=false)."
    )
    return False

  if settle_seconds > 0:
    print(
        f"PR #{pr_number} currently open. Waiting {settle_seconds}s for"
        " potential submit-time merge completion..."
    )
    time.sleep(settle_seconds)
    settled_data = fetch_pr_data(repo, pr_number, token=token, api_url=api_url)
    if settled_data is not None and is_pr_closed_or_merged(settled_data):
      print(
          f"PR #{pr_number} closed or merged during settling window "
          f"(state={settled_data.get('state')}, "
          f"merged={settled_data.get('merged')}). "
          "Skipping presubmit (should_run=false)."
      )
      return False

  print(
      f"PR #{pr_number} remains open. Proceeding with presubmit"
      " (should_run=true)."
  )
  return True


def record_output(output_file: Optional[str], should_run: bool) -> None:
  """Appends should_run output to GitHub output file if configured."""
  val = "true" if should_run else "false"
  print(f"Gate evaluation: should_run={val}")
  if output_file:
    with open(output_file, "a", encoding="utf-8") as f:
      f.write(f"should_run={val}\n")


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
  """Parses command line arguments with environment variable fallbacks."""
  parser = argparse.ArgumentParser(description="PR pre-flight gate check.")
  parser.add_argument(
      "--event-name", default=os.environ.get("EVENT_NAME", "").strip()
  )
  parser.add_argument("--repo", default=os.environ.get("GH_REPO", "").strip())
  parser.add_argument(
      "--pr-number", default=os.environ.get("PR_NUMBER", "").strip()
  )
  parser.add_argument(
      "--token",
      default=(
          os.environ.get("GH_TOKEN", "").strip()
          or os.environ.get("GITHUB_TOKEN", "").strip()
          or None
      ),
  )
  parser.add_argument(
      "--settle-seconds",
      type=float,
      default=float(os.environ.get("GATE_SETTLE_SECONDS", "6")),
  )
  parser.add_argument(
      "--api-url",
      default=os.environ.get("GITHUB_API_URL", "").strip() or None,
  )
  parser.add_argument(
      "--output-file",
      default=os.environ.get("GITHUB_OUTPUT", "").strip() or None,
  )
  return parser.parse_args(args)


def main(argv: Optional[list[str]] = None) -> int:
  args = parse_args(argv)
  should_run = should_run_presubmit(
      event_name=args.event_name,
      repo=args.repo,
      pr_number=args.pr_number,
      token=args.token,
      settle_seconds=args.settle_seconds,
      api_url=args.api_url,
  )
  record_output(args.output_file, should_run)
  return 0


if __name__ == "__main__":
  sys.exit(main())
