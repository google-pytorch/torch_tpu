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

"""Enforces that TorchTPU PR assignees are quality champions only.

Reads the allowed GitHub usernames from `.github/quality_champions.txt`.
On open, non-Copybara pull requests, removes any assignee not in that list,
comments explaining how to assign reviewers vs. champions, and sets the
`assignee-is-quality-champion` commit status on the PR head SHA so Copybara
blocks import until a quality champion is assigned.
"""

from collections.abc import Callable, Mapping, Sequence, Set
import json
import os
import pathlib
import sys
from typing import Any, NamedTuple

from tools import pr_champions

STATUS_CONTEXT = pr_champions.STATUS_CONTEXT
DEFAULT_CHAMPIONS_PATH = pr_champions.DEFAULT_CHAMPIONS_PATH


class EnforcementPlan(NamedTuple):
  """Planned GitHub API actions for a pull request event.

  Attributes:
    remove_assignees: GitHub logins to unassign (non-champions or displaced
      champions when enforcing at most one assignee).
    non_champion_assignees: Subset of `remove_assignees` that are not quality
      champions (used to post an explanation comment).
    displaced_champions: Subset of `remove_assignees` that are quality champions
      displaced by enforcing at most one assignee.
    kept_champion: Quality champion login retained when enforcing at most one
      assignee, if any.
    has_valid_assignee: Whether at least one quality champion will remain
      assigned.
    restore_assignee: Quality champion login to re-add if they were the only
      champion assigned and just got unassigned.
    pr_number: Pull request number.
    head_sha: Head commit SHA to receive the commit status.
  """

  remove_assignees: tuple[str, ...] = ()
  non_champion_assignees: tuple[str, ...] = ()
  displaced_champions: tuple[str, ...] = ()
  kept_champion: str = ""
  has_valid_assignee: bool = False
  restore_assignee: str = ""
  pr_number: int = 0
  head_sha: str = ""


def load_champions(
    path: pathlib.Path = DEFAULT_CHAMPIONS_PATH,
) -> Set[str]:
  """Loads lowercase GitHub logins from the quality champions config file."""
  return frozenset(login.lower() for login in pr_champions.load_champions(path))


def plan_enforcement(
    payload: Mapping[str, Any], champions: Set[str]
) -> EnforcementPlan | None:
  """Computes assignee removals and status update for a pull_request event.

  Args:
    payload: Parsed GitHub webhook event JSON.
    champions: Lowercase GitHub logins of allowed quality champions.

  Returns:
    An EnforcementPlan for open, non-Copybara PRs with an integer PR number, or
    None when the event should be ignored (missing or closed PR,
    `copybara-service[bot]` author, or non-integer PR number).
  """
  pr = payload.get("pull_request")
  if not pr or pr.get("state") != "open":
    return None

  author = ((pr.get("user") or {}).get("login") or "").lower()
  if author == "copybara-service[bot]":
    return None

  pr_number = pr.get("number")
  if not isinstance(pr_number, int):
    return None

  head_sha = str((pr.get("head") or {}).get("sha") or "")
  non_champions: list[str] = []
  valid: list[str] = []
  for user in pr.get("assignees") or ():
    login = (user or {}).get("login")
    if not login:
      continue
    if login.lower() in champions:
      valid.append(login)
    else:
      non_champions.append(login)

  remove = list(non_champions)
  displaced_champions: list[str] = []
  kept_champion = ""
  if len(valid) > 1:
    added = ((payload.get("assignee") or {}).get("login") or "").lower()
    if payload.get("action") == "assigned" and any(
        champion.lower() == added for champion in valid
    ):
      kept_champion = next(
          champion for champion in valid if champion.lower() == added
      )
      displaced_champions = [
          champion for champion in valid if champion.lower() != added
      ]
    else:
      kept_champion = valid[0]
      displaced_champions = list(valid[1:])
    remove.extend(displaced_champions)

  has_valid = bool(valid)
  restore_assignee = ""
  if payload.get("action") == "unassigned" and not has_valid:
    removed = ((payload.get("assignee") or {}).get("login")) or ""
    if removed and removed.lower() in champions:
      restore_assignee = removed

  return EnforcementPlan(
      remove_assignees=tuple(remove),
      non_champion_assignees=tuple(non_champions),
      displaced_champions=tuple(displaced_champions),
      kept_champion=kept_champion,
      has_valid_assignee=has_valid,
      restore_assignee=restore_assignee,
      pr_number=pr_number,
      head_sha=head_sha,
  )


def format_removal_comment(removed: Sequence[str], repository: str) -> str:
  """Formats the PR comment explaining why non-champion assignees were removed."""
  mentions = ", ".join(f"@{login}" for login in removed)
  verb = "was" if len(removed) == 1 else "were"
  return (
      f"{mentions} {verb} unassigned because PR assignees are reserved for "
      "quality champions (listed in "
      "[`.github/quality_champions.txt`]("
      f"https://github.com/{repository}/blob/main/.github/quality_champions.txt"
      ")) — Copybara uses the assignee to gate internal CL submission.\n\n"
      "Please add domain experts as **reviewers** instead of assignees; see "
      "[the PR review guide]("
      f"https://github.com/{repository}/blob/main/CONTRIBUTING.md#code-reviews"
      ")."
  )


def _github_request(
    url: str,
    token: str,
    method: str,
    data: Mapping[str, Any],
) -> None:
  pr_champions.github_request(
      url,
      token,
      method=method,
      data=data,
      user_agent="torch-tpu-enforce-pr-assignee",
  )


def apply_plan(
    plan: EnforcementPlan,
    repository: str,
    token: str,
    api_url: str = "https://api.github.com",
    request_fn: Callable[[str, str, str, Mapping[str, Any]], None] = (
        _github_request
    ),
) -> None:
  """Applies assignee updates, posts comments if needed, and sets status.

  Args:
    plan: Enforcement actions computed by `plan_enforcement`.
    repository: Target GitHub repository in `owner/repo` format.
    token: GitHub API bearer token.
    api_url: Base GitHub REST API URL.
    request_fn: HTTP request callable taking `(url, token, method, data)`.
  """
  base = f"{api_url.rstrip('/')}/repos/{repository}"

  if plan.remove_assignees:
    request_fn(
        f"{base}/issues/{plan.pr_number}/assignees",
        token,
        "DELETE",
        {"assignees": list(plan.remove_assignees)},
    )
  if plan.non_champion_assignees:
    request_fn(
        f"{base}/issues/{plan.pr_number}/comments",
        token,
        "POST",
        {
            "body": format_removal_comment(
                plan.non_champion_assignees, repository
            )
        },
    )

  if plan.displaced_champions:
    mentions = ", ".join(f"@{login}" for login in plan.displaced_champions)
    request_fn(
        f"{base}/issues/{plan.pr_number}/comments",
        token,
        "POST",
        {
            "body": (
                f"Unassigned {mentions}: a PR can have at most one assignee, "
                f"so @{plan.kept_champion} is now the sole assignee."
            )
        },
    )

  if plan.restore_assignee:
    request_fn(
        f"{base}/issues/{plan.pr_number}/assignees",
        token,
        "POST",
        {"assignees": [plan.restore_assignee]},
    )
    request_fn(
        f"{base}/issues/{plan.pr_number}/comments",
        token,
        "POST",
        {
            "body": (
                f"Re-assigned @{plan.restore_assignee}: an open PR with a "
                "quality champion assignee cannot be left without one. To hand "
                "off to another quality champion, assign the new champion "
                f"directly (@{plan.restore_assignee} will be unassigned "
                "automatically)."
            )
        },
    )

  if plan.head_sha:
    if plan.has_valid_assignee or plan.restore_assignee:
      state = "success"
      description = "Assigned to a quality champion."
    else:
      state = "pending"
      description = (
          "Waiting for a quality champion to be assigned as PR assignee."
      )
    request_fn(
        f"{base}/statuses/{plan.head_sha}",
        token,
        "POST",
        {
            "state": state,
            "context": STATUS_CONTEXT,
            "description": description,
        },
    )


def main(
    request_fn: Callable[[str, str, str, Mapping[str, Any]], None] = (
        _github_request
    ),
) -> int:
  token = os.environ.get("GITHUB_TOKEN", "")
  event_path = os.environ.get("GITHUB_EVENT_PATH", "")
  repository = os.environ.get("GITHUB_REPOSITORY", "")
  api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
  if not token or not event_path or not repository:
    print(
        "Missing required GITHUB_TOKEN, GITHUB_EVENT_PATH, or "
        "GITHUB_REPOSITORY",
        file=sys.stderr,
    )
    return 1

  champions_path = pathlib.Path(
      os.environ.get("QUALITY_CHAMPIONS_PATH", str(DEFAULT_CHAMPIONS_PATH))
  )
  champions = load_champions(champions_path)
  payload = json.loads(pathlib.Path(event_path).read_text(encoding="utf-8"))
  plan = plan_enforcement(payload, champions)
  if plan is not None:
    apply_plan(plan, repository, token, api_url=api_url, request_fn=request_fn)
  return 0


if __name__ == "__main__":
  sys.exit(main())
