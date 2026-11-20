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

"""Auto-assigns a random quality champion when a PR is approved."""

from collections.abc import Callable, Mapping, Sequence
import json
import os
import pathlib
import random
import sys
from typing import Any
import urllib.error
import urllib.parse

from tools import pr_champions

STATUS_CONTEXT = pr_champions.STATUS_CONTEXT
DEFAULT_CHAMPIONS_PATH = pr_champions.DEFAULT_CHAMPIONS_PATH
APPROVER_PERMISSIONS = frozenset({"write", "maintain", "admin"})
REVIEWERS_LABEL = "stat:awaiting-reviewers"
REQUESTER_LABEL = "stat:awaiting-requester"
COPYBARA_PREFIX = "copybara"

load_champions = pr_champions.load_champions
RequestFn = Callable[[str, str, str, Mapping[str, Any] | None], Any]
GhFn = Callable[[str, str, Mapping[str, Any] | None], Any]


def choose_champion(
    champions: Sequence[str],
    author: str,
    approver: str,
    rng: random.Random | None = None,
) -> str | None:
  """Selects a random champion excluding `author` and `approver`.

  Args:
    champions: Configured quality champion GitHub logins.
    author: Pull request author's GitHub login.
    approver: Approving reviewer's GitHub login.
    rng: Optional `random.Random` instance for deterministic testing.

  Returns:
    Selected champion login, or `None` when no champion remains after excluding
    `author` and `approver`.
  """
  excluded = {author.lower(), approver.lower()}
  candidates = [login for login in champions if login.lower() not in excluded]
  if not candidates:
    return None
  return (rng if rng is not None else random).choice(candidates)


def is_approved_on_commit(
    reviews: Sequence[Mapping[str, Any]], actor: str, head_sha: str
) -> bool:
  """Returns whether `actor`'s latest non-PENDING review on `head_sha` approves."""
  latest_state = ""
  actor_lower = actor.lower()
  for review in reviews:
    user = (review.get("user") or {}).get("login", "")
    state = review.get("state", "")
    if (
        user.lower() == actor_lower
        and review.get("commit_id") == head_sha
        and state
        and state != "PENDING"
    ):
      latest_state = state
  return latest_state == "APPROVED"


def _has_approver_permission(gh: GhFn, actor: str) -> bool:
  """Returns whether `actor` is a human with write/maintain/admin access."""
  if actor.lower().endswith("[bot]"):
    return False
  encoded_actor = urllib.parse.quote(actor, safe="")
  try:
    perm_data = gh(f"/collaborators/{encoded_actor}/permission", "GET", None)
  except urllib.error.HTTPError as err:
    if err.code == 404:
      return False
    raise
  return (perm_data or {}).get("permission", "") in APPROVER_PERMISSIONS


def _resolve_pr_numbers(
    gh: GhFn, repo: str, workflow_run: Mapping[str, Any]
) -> tuple[list[int], str]:
  """Resolves candidate PR numbers and optional expected head SHA."""
  pr_numbers = [
      pr_ref["number"]
      for pr_ref in workflow_run.get("pull_requests") or []
      if pr_ref.get("number")
  ]
  if pr_numbers:
    return pr_numbers, ""
  head_sha = workflow_run.get("head_sha") or ""
  if not head_sha:
    return [], ""
  query = urllib.parse.urlencode(
      {"q": f"repo:{repo} type:pr state:open {head_sha}"}
  )
  search = gh(f"/search/issues?{query}", "GET", None) or {}
  return [
      item["number"] for item in search.get("items") or [] if item.get("number")
  ], head_sha


def _maybe_assign_pr(
    gh: GhFn,
    pr_number: int,
    actor: str,
    champions: Sequence[str],
    rng: random.Random | None,
    expected_head_sha: str,
) -> str | None:
  """Assigns a random champion to `pr_number` if eligible."""
  pr_data = gh(f"/pulls/{pr_number}", "GET", None)
  if (
      not pr_data
      or pr_data.get("state") != "open"
      or pr_data.get("draft")
      or pr_data.get("assignees")
  ):
    return None
  author = (pr_data.get("user") or {}).get("login", "")
  head_sha = (pr_data.get("head") or {}).get("sha", "")
  if (
      author.lower().startswith(COPYBARA_PREFIX)
      or not head_sha
      or (expected_head_sha and head_sha != expected_head_sha)
  ):
    return None
  reviews = gh(f"/pulls/{pr_number}/reviews?per_page=100", "GET", None) or []
  if not is_approved_on_commit(reviews, actor, head_sha):
    return None
  chosen = choose_champion(champions, author=author, approver=actor, rng=rng)
  if not chosen:
    return None

  gh(f"/issues/{pr_number}/assignees", "POST", {"assignees": [chosen]})
  gh(f"/pulls/{pr_number}/requested_reviewers", "POST", {"reviewers": [chosen]})
  gh(
      f"/statuses/{head_sha}",
      "POST",
      {
          "state": "success",
          "context": STATUS_CONTEXT,
          "description": f"Assigned to quality champion @{chosen}.",
      },
  )
  gh(f"/issues/{pr_number}/labels", "POST", {"labels": [REVIEWERS_LABEL]})
  encoded_requester = urllib.parse.quote(REQUESTER_LABEL, safe="")
  try:
    gh(f"/issues/{pr_number}/labels/{encoded_requester}", "DELETE", None)
  except urllib.error.HTTPError as err:
    if err.code != 404:
      raise
  return chosen


def process_workflow_run(
    payload: Mapping[str, Any],
    champions: Sequence[str],
    token: str,
    repo: str,
    api_url: str = "https://api.github.com",
    request_fn: RequestFn = pr_champions.github_request,
    rng: random.Random | None = None,
) -> list[tuple[int, str]]:
  """Processes a `workflow_run` event and assigns a champion if approved."""
  workflow_run = payload.get("workflow_run") or {}
  if (
      payload.get("action") != "completed"
      or workflow_run.get("event") != "pull_request_review"
  ):
    return []

  def gh(
      path: str,
      method: str = "GET",
      data: Mapping[str, Any] | None = None,
  ) -> Any:
    url = (
        f"{api_url}{path}"
        if path.startswith("/search/")
        else f"{api_url}/repos/{repo}{path}"
    )
    return request_fn(url, token, method, data)

  actor = (workflow_run.get("actor") or {}).get("login", "")
  if not actor or not _has_approver_permission(gh, actor):
    return []
  pr_numbers, expected_head_sha = _resolve_pr_numbers(gh, repo, workflow_run)
  assigned: list[tuple[int, str]] = []
  for pr_number in pr_numbers:
    chosen = _maybe_assign_pr(
        gh, pr_number, actor, champions, rng, expected_head_sha
    )
    if chosen:
      assigned.append((pr_number, chosen))
  return assigned


def main() -> int:
  event_path = os.environ.get("GITHUB_EVENT_PATH")
  token = os.environ.get("GITHUB_TOKEN")
  repo = os.environ.get("GITHUB_REPOSITORY")
  if not event_path or not token or not repo:
    print(
        "Missing GITHUB_EVENT_PATH, GITHUB_TOKEN, or GITHUB_REPOSITORY.",
        file=sys.stderr,
    )
    return 1

  payload = json.loads(pathlib.Path(event_path).read_text(encoding="utf-8"))
  champions_path = pathlib.Path(
      os.environ.get("QUALITY_CHAMPIONS_PATH", str(DEFAULT_CHAMPIONS_PATH))
  )
  assigned = process_workflow_run(
      payload=payload,
      champions=load_champions(champions_path),
      token=token,
      repo=repo,
      api_url=os.environ.get("GITHUB_API_URL", "https://api.github.com"),
  )
  for pr_number, champion in assigned:
    print(f"PR #{pr_number}: assigned quality champion @{champion}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
