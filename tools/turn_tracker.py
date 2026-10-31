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

"""Two-party ping-pong turn tracking for TorchTPU issues and pull requests.

Contract:
- Inputs:
    Environment variables:
      GITHUB_TOKEN: Bearer token for GitHub REST API authentication.
      GITHUB_EVENT_PATH: Path to the JSON event payload file from GitHub
        Actions.
      GITHUB_REPOSITORY: Repository name in 'owner/repo' format.
      GITHUB_EVENT_NAME: Event name ('issues', 'issue_comment',
        'pull_request', 'pull_request_target', 'pull_request_review',
        'pull_request_review_comment', or 'workflow_run').
      GITHUB_API_URL: (Optional) Base API URL, defaults to
        'https://api.github.com'.
    Event payloads:
      'issues': Triggered on 'opened', 'reopened', 'transferred', 'assigned',
        or 'unassigned'.
      'issue_comment': Triggered on 'created' (on issues or pull requests).
      'pull_request' / 'pull_request_target': Triggered on 'opened',
        'reopened', 'ready_for_review', 'review_requested',
        'review_request_removed', 'assigned', or 'unassigned'.
      'pull_request_review': Triggered on 'submitted'.
      'pull_request_review_comment': Triggered on 'created'.
      'workflow_run': Relays a 'pull_request_review' or
        'pull_request_review_comment' run. GitHub hands those events a
        read-only token on PRs from forks, so a no-op workflow runs on them
        and this script handles its completion with a write token, looking
        open PRs up via the Search API and verifying their head SHA.

- Outputs and Return Code:
    Exit 0: Normal completion, whether labels were modified or the event was
      ignored as a no-op.
    Exit 1: Terminated due to missing required environment variables.

- Side Effects:
    Executes REST API requests (GET, DELETE, POST) against GitHub's pull
    request and issue label endpoints:
      - Adds 'stat:awaiting-assignee' on issues and 'stat:awaiting-reviewers'
        on pull requests when the item is new, reopened, transferred, or
        ready for review, when the reviewer list or assignee field changes,
        or when the original poster comments or submits a review.
      - Adds 'stat:awaiting-requester' when a responder comments or submits
        a review. Responders are assignees on issues, and assignees plus
        reviewers (requested or already reviewed) on pull requests.
      - Deletes conflicting turn labels on turn flips (including
        'stat:awaiting-assignee' on pull requests).

- Failure Cases and Ignored Events:
    - Closed or draft items: Activity on closed issues/PRs or draft PRs is
      ignored (no-op).
    - Third-party activity: Comments or reviews from users who are neither the
      original poster nor a responder are ignored (no-op).
    - Bot activity: Items authored by bots (e.g. Copybara) and comments or
      reviews from bots are ignored (no-op).
    - Self-assigned items: If the creator is also assigned to the item,
      comments or reviews are ignored to prevent self-ping-pong (no-op).
    - HTTP 404 on label deletion: Swallowed gracefully (label was already
      absent).
    - Other HTTP errors and network timeouts: Raised as exceptions causing the
      run to fail.
"""

import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Final, Mapping, NamedTuple, Sequence
import urllib.error
import urllib.parse
import urllib.request

ASSIGNEE_LABEL: Final[str] = "stat:awaiting-assignee"
REVIEWERS_LABEL: Final[str] = "stat:awaiting-reviewers"
REQUESTER_LABEL: Final[str] = "stat:awaiting-requester"

_ISSUE_RESPONDER_ACTIONS: Final[frozenset[str]] = frozenset({
    # go/keep-sorted start
    "assigned",
    "opened",
    "reopened",
    "transferred",
    "unassigned",
    # go/keep-sorted end
})
_PR_RESPONDER_ACTIONS: Final[frozenset[str]] = frozenset({
    # go/keep-sorted start
    "assigned",
    "opened",
    "ready_for_review",
    "reopened",
    "review_request_removed",
    "review_requested",
    "unassigned",
    # go/keep-sorted end
})


class LabelChanges(NamedTuple):
  add: Sequence[str] = ()
  remove: Sequence[str] = ()


def _is_bot(user: Mapping[str, Any] | None) -> bool:
  if not user:
    return False
  return user.get("type") == "Bot" or (user.get("login") or "").endswith(
      "[bot]"
  )


def _logins(users: Sequence[Mapping[str, Any] | None]) -> frozenset[str]:
  return frozenset(u["login"].lower() for u in users if u and u.get("login"))


def determine_label_changes(
    event_name: str,
    payload: Mapping[str, Any],
    reviewers: frozenset[str] = frozenset(),
) -> LabelChanges:
  """Determines labels to add and remove based on the event.

  `reviewers` holds lowercase logins of the PR's reviewers (requested or
  already reviewed). They count as responders alongside assignees, matching
  the review process where a reviewer, not an assignee, owns the first pass.
  """
  target = payload.get("issue") or payload.get("pull_request")
  if not target:
    return LabelChanges()

  if target.get("state") == "closed" or target.get("draft"):
    return LabelChanges()

  if _is_bot(target.get("user")) or _is_bot(payload.get("sender")):
    return LabelChanges()

  existing_labels: dict[str, str] = {}
  for item in target.get("labels", []):
    name = item if isinstance(item, str) else item.get("name")
    if name:
      existing_labels[name.lower()] = name

  is_pr = "pull_request" in payload or "pull_request" in target
  responder_label = REVIEWERS_LABEL if is_pr else ASSIGNEE_LABEL

  def awaiting_responder() -> LabelChanges:
    to_add = []
    to_remove = []
    if responder_label.lower() not in existing_labels:
      to_add.append(responder_label)
    if REQUESTER_LABEL.lower() in existing_labels:
      to_remove.append(existing_labels[REQUESTER_LABEL.lower()])
    if is_pr and ASSIGNEE_LABEL.lower() in existing_labels:
      to_remove.append(existing_labels[ASSIGNEE_LABEL.lower()])
    return LabelChanges(add=tuple(to_add), remove=tuple(to_remove))

  def awaiting_requester() -> LabelChanges:
    to_add = []
    to_remove = []
    if REQUESTER_LABEL.lower() not in existing_labels:
      to_add.append(REQUESTER_LABEL)
    if responder_label.lower() in existing_labels:
      to_remove.append(existing_labels[responder_label.lower()])
    if is_pr and ASSIGNEE_LABEL.lower() in existing_labels:
      to_remove.append(existing_labels[ASSIGNEE_LABEL.lower()])
    return LabelChanges(add=tuple(to_add), remove=tuple(to_remove))

  action = payload.get("action")
  if (event_name == "issues" and action in _ISSUE_RESPONDER_ACTIONS) or (
      event_name in ("pull_request", "pull_request_target")
      and action in _PR_RESPONDER_ACTIONS
  ):
    return awaiting_responder()

  is_comment_or_review = (
      event_name in ("issue_comment", "pull_request_review_comment")
      and action == "created"
  ) or (event_name == "pull_request_review" and action == "submitted")
  if not is_comment_or_review:
    return LabelChanges()

  sender_login = ((payload.get("sender") or {}).get("login") or "").lower()
  if not sender_login:
    return LabelChanges()

  op_login = ((target.get("user") or {}).get("login") or "").lower()
  # Replying to an inline diff thread submits a COMMENTED review, putting the
  # PR author into `reviewers`.
  responders = _logins(
      [target.get("assignee"), *(target.get("assignees") or [])]
  ) | (reviewers - {op_login})

  is_op = bool(op_login and sender_login == op_login)
  is_responder = sender_login in responders

  if is_op and is_responder:
    return LabelChanges()
  if is_responder:
    return awaiting_requester()
  if is_op:
    return awaiting_responder()
  return LabelChanges()


RELAYED_EVENTS: Final[Mapping[str, str]] = {
    "pull_request_review": "submitted",
    "pull_request_review_comment": "created",
}


def _headers(token: str) -> dict[str, str]:
  return {
      "Authorization": f"Bearer {token}",
      "Accept": "application/vnd.github+json",
      "User-Agent": "TorchTPU-Turn-Tracker",
      "X-GitHub-Api-Version": "2022-11-28",
  }


def _get_json(
    url: str, token: str, *, timeout: float, opener: Callable[..., Any]
) -> Any:
  req = urllib.request.Request(url, headers=_headers(token))
  with opener(req, timeout=timeout) as resp:
    return json.loads(resp.read().decode("utf-8"))


def resolve_workflow_run(
    api_url: str,
    repo: str,
    token: str,
    payload: Mapping[str, Any],
    *,
    timeout: float = 30.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> list[tuple[str, dict[str, Any]]]:
  """Rebuilds relayed review events from a 'workflow_run' payload.

  Returns one (event_name, payload) pair per open PR whose head is the
  relayed run's head SHA, with the run's actor as the sender. Queries the
  Search API because `/commits/{sha}/pulls` omits unmerged fork commits, then
  fetches each candidate PR to check its actual head SHA.
  """
  run = payload.get("workflow_run") or {}
  event_name = run.get("event")
  head_sha = run.get("head_sha")
  actor = run.get("actor")
  if event_name not in RELAYED_EVENTS or not head_sha or not actor:
    return []

  query = urllib.parse.urlencode(
      {"q": f"repo:{repo} type:pr state:open {head_sha}"}
  )
  search = _get_json(
      f"{api_url}/search/issues?{query}",
      token,
      timeout=timeout,
      opener=opener,
  )
  pulls = [
      _get_json(
          f"{api_url}/repos/{repo}/pulls/{item['number']}",
          token,
          timeout=timeout,
          opener=opener,
      )
      for item in search["items"]
  ]

  return [
      (
          event_name,
          {
              "action": RELAYED_EVENTS[event_name],
              "pull_request": pr,
              "sender": actor,
          },
      )
      for pr in pulls
      if pr.get("state") == "open"
      and (pr.get("head") or {}).get("sha") == head_sha
  ]


def fetch_pr_reviewers(
    api_url: str,
    repo: str,
    token: str,
    pr_number: int,
    *,
    timeout: float = 30.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> frozenset[str]:
  """Returns lowercase logins of the PR's reviewers.

  Mirrors the "Reviewers" sidebar: pending review requests plus anyone who
  has already submitted a review. GitHub drops a reviewer from the pending
  list once they submit, so both sources are needed.
  """
  base = f"{api_url}/repos/{repo}/pulls/{pr_number}"
  requested = _get_json(
      f"{base}/requested_reviewers", token, timeout=timeout, opener=opener
  )
  reviews = _get_json(
      f"{base}/reviews?per_page=100", token, timeout=timeout, opener=opener
  )
  return _logins(requested.get("users") or []) | _logins(
      [r.get("user") for r in reviews if r.get("state") != "PENDING"]
  )


def apply_label_changes(
    api_url: str,
    repo: str,
    token: str,
    issue_number: int,
    to_add: Sequence[str],
    to_remove: Sequence[str],
    *,
    timeout: float = 30.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> None:
  """Executes GitHub REST API requests to update labels."""
  headers = _headers(token)

  for label in to_remove:
    encoded_name = urllib.parse.quote(label)
    url = f"{api_url}/repos/{repo}/issues/{issue_number}/labels/{encoded_name}"
    req = urllib.request.Request(url, headers=headers, method="DELETE")
    try:
      with opener(req, timeout=timeout):
        pass
    except urllib.error.HTTPError as err:
      if err.code != 404:
        raise

  if to_add:
    url = f"{api_url}/repos/{repo}/issues/{issue_number}/labels"
    data = json.dumps({"labels": list(to_add)}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    with opener(req, timeout=timeout):
      pass


def main() -> None:
  token = os.environ.get("GITHUB_TOKEN")  # PY_ENVIRON_OK=ci
  event_path = os.environ.get("GITHUB_EVENT_PATH")  # PY_ENVIRON_OK=ci
  repo = os.environ.get("GITHUB_REPOSITORY")  # PY_ENVIRON_OK=ci
  event_name = os.environ.get("GITHUB_EVENT_NAME")  # PY_ENVIRON_OK=ci
  api_url = os.environ.get(  # PY_ENVIRON_OK=ci
      "GITHUB_API_URL", "https://api.github.com"
  )

  if not (token and event_path and repo and event_name):
    print("Missing required environment variables", file=sys.stderr)
    sys.exit(1)

  event_payload_path = Path(event_path)
  payload = json.loads(event_payload_path.read_text(encoding="utf-8"))

  if event_name == "workflow_run":
    events = resolve_workflow_run(api_url, repo, token, payload)
  else:
    events = [(event_name, payload)]

  for name, event_payload in events:
    target = (
        event_payload.get("issue") or event_payload.get("pull_request") or {}
    )
    issue_number = target.get("number")
    if not issue_number:
      continue

    is_pr = "pull_request" in event_payload or "pull_request" in target
    reviewers = frozenset()
    if is_pr and name in (
        "issue_comment",
        "pull_request_review",
        "pull_request_review_comment",
    ):
      reviewers = fetch_pr_reviewers(api_url, repo, token, issue_number)

    changes = determine_label_changes(name, event_payload, reviewers)
    if not changes.add and not changes.remove:
      continue

    apply_label_changes(
        api_url, repo, token, issue_number, changes.add, changes.remove
    )


if __name__ == "__main__":
  main()
