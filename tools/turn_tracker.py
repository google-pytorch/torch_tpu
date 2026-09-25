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

"""Two-party ping-pong turn tracking for TorchTPU.

Contract:
- Inputs:
    Environment variables:
      GITHUB_TOKEN: Bearer token for GitHub REST API authentication.
      GITHUB_EVENT_PATH: Path to the JSON event payload file from GitHub
      Actions.
      GITHUB_REPOSITORY: Repository name in 'owner/repo' format.
      GITHUB_EVENT_NAME: Event name ('issues' or 'issue_comment').
      GITHUB_API_URL: (Optional) Base API URL, defaults to
        'https://api.github.com'.
    Event payloads:
      'issues': Triggered on 'opened', 'reopened', or 'transferred'.
      'issue_comment': Triggered on 'created'.

- Outputs and Return Code:
    Exit 0: Normal completion, whether labels were modified or the event was
      ignored as a no-op.
    Exit 1: Terminated due to missing required environment variables.

- Side Effects:
    Executes REST API requests (DELETE, POST) against GitHub's issue label
    endpoints:
      - Adds 'stat:awaiting-assignee' on new, reopened, transferred
        issues, or when the original poster comments.
      - Adds 'stat:awaiting-requester' when an assignee comments.
      - Deletes conflicting turn labels on turn flips.

- Failure Cases and Ignored Events:
    - Pull requests: Ignored (no-op in phase 1).
    - Closed issues: Comments on closed issues are ignored (no-op).
    - Third-party comments: Comments from users who are neither the original
      poster nor an assigned engineer are ignored (no-op).
    - Bot comments: Ignored (no-op).
    - Self-assigned issues: If the creator is also the assignee, comments are
      ignored to prevent self-ping-pong (no-op).
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
REQUESTER_LABEL: Final[str] = "stat:awaiting-requester"


class LabelChanges(NamedTuple):
  add: Sequence[str] = ()
  remove: Sequence[str] = ()


def determine_label_changes(
    event_name: str, payload: Mapping[str, Any]
) -> LabelChanges:
  """Determines labels to add and remove based on the event."""
  issue = payload.get("issue")
  if not issue:
    return LabelChanges()

  if issue.get("pull_request"):
    return LabelChanges()

  if issue.get("state") == "closed" and event_name != "issues":
    return LabelChanges()

  existing_labels: dict[str, str] = {}
  for item in issue.get("labels", []):
    name = item if isinstance(item, str) else item.get("name")
    if name:
      existing_labels[name.lower()] = name

  if event_name == "issues":
    action = payload.get("action")
    if action in ("opened", "reopened", "transferred"):
      to_add = []
      to_remove = []
      if ASSIGNEE_LABEL.lower() not in existing_labels:
        to_add.append(ASSIGNEE_LABEL)
      if REQUESTER_LABEL.lower() in existing_labels:
        to_remove.append(existing_labels[REQUESTER_LABEL.lower()])
      return LabelChanges(add=tuple(to_add), remove=tuple(to_remove))
    return LabelChanges()

  if event_name == "issue_comment":
    action = payload.get("action")
    if action != "created":
      return LabelChanges()

    sender = payload.get("sender") or {}
    if sender.get("type") == "Bot" or (sender.get("login") or "").endswith(
        "[bot]"
    ):
      return LabelChanges()

    sender_login = (sender.get("login") or "").lower()
    if not sender_login:
      return LabelChanges()

    op_login = ((issue.get("user") or {}).get("login") or "").lower()

    raw_assignees = [issue.get("assignee"), *(issue.get("assignees") or [])]
    assignees: frozenset[str] = frozenset(
        a["login"].lower() for a in raw_assignees if a and a.get("login")
    )

    is_op = bool(op_login and sender_login == op_login)
    is_assignee = sender_login in assignees

    if is_op and is_assignee:
      return LabelChanges()

    to_add = []
    to_remove = []

    if is_assignee:
      if REQUESTER_LABEL.lower() not in existing_labels:
        to_add.append(REQUESTER_LABEL)
      if ASSIGNEE_LABEL.lower() in existing_labels:
        to_remove.append(existing_labels[ASSIGNEE_LABEL.lower()])
      return LabelChanges(add=tuple(to_add), remove=tuple(to_remove))

    if is_op:
      if ASSIGNEE_LABEL.lower() not in existing_labels:
        to_add.append(ASSIGNEE_LABEL)
      if REQUESTER_LABEL.lower() in existing_labels:
        to_remove.append(existing_labels[REQUESTER_LABEL.lower()])
      return LabelChanges(add=tuple(to_add), remove=tuple(to_remove))

    return LabelChanges()

  return LabelChanges()


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
  headers = {
      "Authorization": f"Bearer {token}",
      "Accept": "application/vnd.github+json",
      "User-Agent": "TorchTPU-Turn-Tracker",
      "X-GitHub-Api-Version": "2022-11-28",
  }

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

  issue_number = (payload.get("issue") or {}).get("number")
  if not issue_number:
    return

  changes = determine_label_changes(event_name, payload)
  if not changes.add and not changes.remove:
    return

  apply_label_changes(
      api_url, repo, token, issue_number, changes.add, changes.remove
  )


if __name__ == "__main__":
  main()
