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

"""Unit tests for turn_tracker.py."""

import io
import json
import os
import tempfile
from typing import Any
import unittest
from unittest import mock
import urllib.error
import urllib.parse

from turn_tracker import (
    ASSIGNEE_LABEL,
    REQUESTER_LABEL,
    REVIEWERS_LABEL,
    apply_label_changes,
    determine_label_changes,
    fetch_pr_reviewers,
    main,
    resolve_workflow_run,
)


class TurnTrackerTest(unittest.TestCase):  # UNITTEST_OK=standalone

  def _make_issue(
      self,
      number=101,
      author="alice",
      assignee="bob",
      assignees=None,
      labels=None,
      state="open",
      is_pr=False,
      draft=False,
  ):
    issue = {
        "number": number,
        "state": state,
        "user": {"login": author},
        "labels": labels or [],
    }
    if draft:
      issue["draft"] = True
    if is_pr:
      issue["pull_request"] = {
          "url": f"https://api.github.com/repos/o/r/pulls/{number}"
      }
    if assignee:
      issue["assignee"] = {"login": assignee}
    if assignees is not None:
      issue["assignees"] = [{"login": a} for a in assignees]
    elif assignee:
      issue["assignees"] = [{"login": assignee}]
    else:
      issue["assignees"] = []
    return issue

  def test_issue_opened_adds_assignee_label(self):
    payload = {
        "action": "opened",
        "issue": self._make_issue(author="alice", assignee="bob"),
    }
    changes = determine_label_changes("issues", payload)
    self.assertEqual(changes.add, (ASSIGNEE_LABEL,))
    self.assertEqual(changes.remove, ())

  def test_issue_reopened_clears_requester_and_adds_assignee(self):
    payload = {
        "action": "reopened",
        "issue": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REQUESTER_LABEL}, {"name": "stale"}],
        ),
    }
    changes = determine_label_changes("issues", payload)
    self.assertEqual(changes.add, (ASSIGNEE_LABEL,))
    self.assertEqual(changes.remove, (REQUESTER_LABEL,))

  def test_issue_transferred_adds_assignee_label(self):
    payload = {
        "action": "transferred",
        "issue": self._make_issue(author="alice"),
    }
    changes = determine_label_changes("issues", payload)
    self.assertEqual(changes.add, (ASSIGNEE_LABEL,))

  def test_issue_assigned_switches_to_assignee(self):
    payload = {
        "action": "assigned",
        "issue": self._make_issue(
            author="alice",
            assignee="charlie",
            labels=[{"name": REQUESTER_LABEL}],
        ),
        "sender": {"login": "bob"},
    }
    changes = determine_label_changes("issues", payload)
    self.assertEqual(changes.add, (ASSIGNEE_LABEL,))
    self.assertEqual(changes.remove, (REQUESTER_LABEL,))

  def test_issue_unassigned_switches_to_assignee(self):
    payload = {
        "action": "unassigned",
        "issue": self._make_issue(
            author="alice",
            assignee=None,
            labels=[{"name": REQUESTER_LABEL}],
        ),
        "sender": {"login": "bob"},
    }
    changes = determine_label_changes("issues", payload)
    self.assertEqual(changes.add, (ASSIGNEE_LABEL,))
    self.assertEqual(changes.remove, (REQUESTER_LABEL,))

  def test_issue_opened_already_has_assignee_label(self):
    payload = {
        "action": "opened",
        "issue": self._make_issue(
            author="alice", labels=[{"name": ASSIGNEE_LABEL}]
        ),
    }
    changes = determine_label_changes("issues", payload)
    self.assertEqual(changes.add, ())
    self.assertEqual(changes.remove, ())

  def test_pr_opened_adds_reviewers_label(self):
    payload = {
        "action": "opened",
        "pull_request": self._make_issue(author="alice", assignee="bob"),
    }
    changes = determine_label_changes("pull_request_target", payload)
    self.assertEqual(changes.add, (REVIEWERS_LABEL,))
    self.assertEqual(changes.remove, ())

  def test_draft_pr_opened_ignored(self):
    payload = {
        "action": "opened",
        "pull_request": self._make_issue(
            author="alice", assignee="bob", draft=True
        ),
    }
    changes = determine_label_changes("pull_request_target", payload)
    self.assertEqual(changes.add, ())
    self.assertEqual(changes.remove, ())

  def test_draft_pr_comment_ignored(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(
            author="alice",
            assignee="bob",
            is_pr=True,
            draft=True,
        ),
        "sender": {"login": "bob"},
    }
    changes = determine_label_changes("issue_comment", payload)
    self.assertEqual(changes.add, ())
    self.assertEqual(changes.remove, ())

  def test_bot_authored_pr_opened_ignored(self):
    payload = {
        "action": "opened",
        "pull_request": self._make_issue(
            author="copybara-service[bot]", assignee="bob"
        ),
    }
    changes = determine_label_changes("pull_request_target", payload)
    self.assertEqual(changes.add, ())
    self.assertEqual(changes.remove, ())

  def test_pr_reopened_clears_requester_and_adds_reviewers(self):
    payload = {
        "action": "reopened",
        "pull_request": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REQUESTER_LABEL}],
        ),
    }
    changes = determine_label_changes("pull_request_target", payload)
    self.assertEqual(changes.add, (REVIEWERS_LABEL,))
    self.assertEqual(changes.remove, (REQUESTER_LABEL,))

  def test_pr_ready_for_review_adds_reviewers_label(self):
    payload = {
        "action": "ready_for_review",
        "pull_request": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REQUESTER_LABEL}],
        ),
    }
    changes = determine_label_changes("pull_request", payload)
    self.assertEqual(changes.add, (REVIEWERS_LABEL,))
    self.assertEqual(changes.remove, (REQUESTER_LABEL,))

  def test_pr_review_requested_by_op_switches_to_reviewers(self):
    payload = {
        "action": "review_requested",
        "pull_request": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REQUESTER_LABEL}],
        ),
        "sender": {"login": "alice"},
    }
    changes = determine_label_changes("pull_request_target", payload)
    self.assertEqual(changes.add, (REVIEWERS_LABEL,))
    self.assertEqual(changes.remove, (REQUESTER_LABEL,))

  def test_pr_review_requested_by_reviewer_handoff_switches_to_reviewers(
      self,
  ):
    payload = {
        "action": "review_requested",
        "pull_request": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REQUESTER_LABEL}],
        ),
        "sender": {"login": "bob"},
    }
    changes = determine_label_changes("pull_request_target", payload)
    self.assertEqual(changes.add, (REVIEWERS_LABEL,))
    self.assertEqual(changes.remove, (REQUESTER_LABEL,))

  def test_pr_review_request_removed_switches_to_reviewers(self):
    payload = {
        "action": "review_request_removed",
        "pull_request": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REQUESTER_LABEL}],
        ),
        "sender": {"login": "bob"},
    }
    changes = determine_label_changes("pull_request_target", payload)
    self.assertEqual(changes.add, (REVIEWERS_LABEL,))
    self.assertEqual(changes.remove, (REQUESTER_LABEL,))

  def test_pr_assigned_switches_to_reviewers(self):
    payload = {
        "action": "assigned",
        "pull_request": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REQUESTER_LABEL}],
        ),
        "sender": {"login": "bob"},
    }
    changes = determine_label_changes("pull_request_target", payload)
    self.assertEqual(changes.add, (REVIEWERS_LABEL,))
    self.assertEqual(changes.remove, (REQUESTER_LABEL,))

  def test_pr_unassigned_switches_to_reviewers(self):
    payload = {
        "action": "unassigned",
        "pull_request": self._make_issue(
            author="alice",
            assignee=None,
            labels=[{"name": REQUESTER_LABEL}],
        ),
        "sender": {"login": "bob"},
    }
    changes = determine_label_changes("pull_request_target", payload)
    self.assertEqual(changes.add, (REVIEWERS_LABEL,))
    self.assertEqual(changes.remove, (REQUESTER_LABEL,))

  def test_pr_responder_turn_removes_legacy_assignee_label(self):
    payload = {
        "action": "review_request_removed",
        "pull_request": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REVIEWERS_LABEL}, {"name": ASSIGNEE_LABEL}],
        ),
        "sender": {"login": "bob"},
    }
    changes = determine_label_changes("pull_request_target", payload)
    self.assertEqual(changes.add, ())
    self.assertEqual(changes.remove, (ASSIGNEE_LABEL,))

  def test_pr_requester_turn_removes_legacy_assignee_label(self):
    payload = {
        "action": "submitted",
        "pull_request": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": ASSIGNEE_LABEL}],
        ),
        "sender": {"login": "bob"},
    }
    changes = determine_label_changes("pull_request_review", payload)
    self.assertEqual(changes.add, (REQUESTER_LABEL,))
    self.assertEqual(changes.remove, (ASSIGNEE_LABEL,))

  def test_pr_review_by_unassigned_reviewer_switches_to_requester(self):
    payload = {
        "action": "submitted",
        "pull_request": self._make_issue(
            author="alice",
            assignee=None,
            labels=[{"name": REVIEWERS_LABEL}],
        ),
        "sender": {"login": "Dana"},
    }
    changes = determine_label_changes(
        "pull_request_review", payload, reviewers=frozenset({"dana"})
    )
    self.assertEqual(changes.add, (REQUESTER_LABEL,))
    self.assertEqual(changes.remove, (REVIEWERS_LABEL,))

  def test_pr_comment_by_reviewer_switches_to_requester(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REVIEWERS_LABEL}],
            is_pr=True,
        ),
        "sender": {"login": "dana"},
    }
    changes = determine_label_changes(
        "issue_comment", payload, reviewers=frozenset({"dana"})
    )
    self.assertEqual(changes.add, (REQUESTER_LABEL,))
    self.assertEqual(changes.remove, (REVIEWERS_LABEL,))

  def test_pr_review_by_non_reviewer_ignored(self):
    payload = {
        "action": "submitted",
        "pull_request": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REVIEWERS_LABEL}],
        ),
        "sender": {"login": "eve"},
    }
    changes = determine_label_changes(
        "pull_request_review", payload, reviewers=frozenset({"dana"})
    )
    self.assertEqual(changes.add, ())
    self.assertEqual(changes.remove, ())

  def test_pr_issue_comment_by_assignee_switches_to_requester(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REVIEWERS_LABEL}],
            is_pr=True,
        ),
        "sender": {"login": "bob"},
    }
    changes = determine_label_changes("issue_comment", payload)
    self.assertEqual(changes.add, (REQUESTER_LABEL,))
    self.assertEqual(changes.remove, (REVIEWERS_LABEL,))

  def test_pr_issue_comment_by_op_switches_to_reviewers(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REQUESTER_LABEL}],
            is_pr=True,
        ),
        "sender": {"login": "alice"},
    }
    changes = determine_label_changes("issue_comment", payload)
    self.assertEqual(changes.add, (REVIEWERS_LABEL,))
    self.assertEqual(changes.remove, (REQUESTER_LABEL,))

  def test_pr_review_submitted_by_assignee_switches_to_requester(self):
    payload = {
        "action": "submitted",
        "pull_request": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REVIEWERS_LABEL}],
        ),
        "sender": {"login": "bob"},
    }
    changes = determine_label_changes("pull_request_review", payload)
    self.assertEqual(changes.add, (REQUESTER_LABEL,))
    self.assertEqual(changes.remove, (REVIEWERS_LABEL,))

  def test_pr_review_submitted_by_op_switches_to_reviewers(self):
    payload = {
        "action": "submitted",
        "pull_request": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REQUESTER_LABEL}],
        ),
        "sender": {"login": "alice"},
    }
    changes = determine_label_changes("pull_request_review", payload)
    self.assertEqual(changes.add, (REVIEWERS_LABEL,))
    self.assertEqual(changes.remove, (REQUESTER_LABEL,))

  def test_pr_review_submitted_by_third_party_ignored(self):
    payload = {
        "action": "submitted",
        "pull_request": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REVIEWERS_LABEL}],
        ),
        "sender": {"login": "eve"},
    }
    changes = determine_label_changes("pull_request_review", payload)
    self.assertEqual(changes.add, ())
    self.assertEqual(changes.remove, ())

  def test_pr_review_comment_by_assignee_switches_to_requester(self):
    payload = {
        "action": "created",
        "pull_request": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REVIEWERS_LABEL}],
        ),
        "sender": {"login": "bob"},
    }
    changes = determine_label_changes("pull_request_review_comment", payload)
    self.assertEqual(changes.add, (REQUESTER_LABEL,))
    self.assertEqual(changes.remove, (REVIEWERS_LABEL,))

  def test_pr_review_comment_by_op_switches_to_reviewers(self):
    payload = {
        "action": "created",
        "pull_request": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REQUESTER_LABEL}],
        ),
        "sender": {"login": "alice"},
    }
    changes = determine_label_changes("pull_request_review_comment", payload)
    self.assertEqual(changes.add, (REVIEWERS_LABEL,))
    self.assertEqual(changes.remove, (REQUESTER_LABEL,))

  def test_pr_op_in_reviewers_still_switches_to_reviewers(self):
    payload = {
        "action": "created",
        "pull_request": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REQUESTER_LABEL}],
        ),
        "sender": {"login": "alice"},
    }
    changes = determine_label_changes(
        "pull_request_review_comment",
        payload,
        reviewers=frozenset({"alice", "bob"}),
    )
    self.assertEqual(changes.add, (REVIEWERS_LABEL,))
    self.assertEqual(changes.remove, (REQUESTER_LABEL,))

  def test_self_assigned_pr_comment_ignored(self):
    payload = {
        "action": "created",
        "pull_request": self._make_issue(
            author="alice",
            assignee="alice",
            labels=[{"name": REVIEWERS_LABEL}],
        ),
        "sender": {"login": "alice"},
    }
    changes = determine_label_changes(
        "pull_request_review_comment", payload, reviewers=frozenset({"alice"})
    )
    self.assertEqual(changes.add, ())
    self.assertEqual(changes.remove, ())

  def test_closed_issue_comment_ignored(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(
            author="alice", assignee="bob", state="closed"
        ),
        "sender": {"login": "alice"},
    }
    changes = determine_label_changes("issue_comment", payload)
    self.assertEqual(changes.add, ())
    self.assertEqual(changes.remove, ())

  def test_closed_pr_review_ignored(self):
    payload = {
        "action": "submitted",
        "pull_request": self._make_issue(
            author="alice",
            assignee="bob",
            state="closed",
            labels=[{"name": REVIEWERS_LABEL}],
        ),
        "sender": {"login": "bob"},
    }
    changes = determine_label_changes("pull_request_review", payload)
    self.assertEqual(changes.add, ())
    self.assertEqual(changes.remove, ())

  def test_bot_comment_ignored(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(author="alice", assignee="bob"),
        "sender": {"login": "github-actions[bot]", "type": "Bot"},
    }
    changes = determine_label_changes("issue_comment", payload)
    self.assertEqual(changes.add, ())
    self.assertEqual(changes.remove, ())

  def test_bot_pr_review_ignored(self):
    payload = {
        "action": "submitted",
        "pull_request": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REVIEWERS_LABEL}],
        ),
        "sender": {"login": "gemini-code-assist[bot]", "type": "Bot"},
    }
    changes = determine_label_changes("pull_request_review", payload)
    self.assertEqual(changes.add, ())
    self.assertEqual(changes.remove, ())

  def test_assignee_comment_switches_to_requester(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": ASSIGNEE_LABEL}],
        ),
        "sender": {"login": "bob"},
    }
    changes = determine_label_changes("issue_comment", payload)
    self.assertEqual(changes.add, (REQUESTER_LABEL,))
    self.assertEqual(changes.remove, (ASSIGNEE_LABEL,))

  def test_op_comment_switches_to_assignee(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REQUESTER_LABEL}, {"name": "stale"}],
        ),
        "sender": {"login": "alice"},
    }
    changes = determine_label_changes("issue_comment", payload)
    self.assertEqual(changes.add, (ASSIGNEE_LABEL,))
    self.assertEqual(changes.remove, (REQUESTER_LABEL,))

  def test_multiple_assignees_any_assignee_comments(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(
            author="alice",
            assignee=None,
            assignees=["bob", "charlie"],
            labels=[{"name": ASSIGNEE_LABEL}],
        ),
        "sender": {"login": "charlie"},
    }
    changes = determine_label_changes("issue_comment", payload)
    self.assertEqual(changes.add, (REQUESTER_LABEL,))
    self.assertEqual(changes.remove, (ASSIGNEE_LABEL,))

  def test_third_party_comment_disregarded(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REQUESTER_LABEL}],
        ),
        "sender": {"login": "eve"},
    }
    changes = determine_label_changes("issue_comment", payload)
    self.assertEqual(changes.add, ())
    self.assertEqual(changes.remove, ())

  def test_self_assigned_issue_comment_ignored(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(
            author="alice",
            assignee="alice",
            labels=[{"name": ASSIGNEE_LABEL}],
        ),
        "sender": {"login": "alice"},
    }
    changes = determine_label_changes("issue_comment", payload)
    self.assertEqual(changes.add, ())
    self.assertEqual(changes.remove, ())

  def test_unassigned_issue_op_comment(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(
            author="alice",
            assignee=None,
            assignees=[],
            labels=[{"name": REQUESTER_LABEL}],
        ),
        "sender": {"login": "alice"},
    }
    changes = determine_label_changes("issue_comment", payload)
    self.assertEqual(changes.add, (ASSIGNEE_LABEL,))
    self.assertEqual(changes.remove, (REQUESTER_LABEL,))

  def test_unassigned_issue_non_op_comment_disregarded(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(
            author="alice",
            assignee=None,
            assignees=[],
            labels=[{"name": ASSIGNEE_LABEL}],
        ),
        "sender": {"login": "bob"},
    }
    changes = determine_label_changes("issue_comment", payload)
    self.assertEqual(changes.add, ())
    self.assertEqual(changes.remove, ())

  def test_case_insensitive_logins(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(
            author="AliceSmith",
            assignee="BobJones",
            labels=[{"name": ASSIGNEE_LABEL}],
        ),
        "sender": {"login": "bobjones"},
    }
    changes = determine_label_changes("issue_comment", payload)
    self.assertEqual(changes.add, (REQUESTER_LABEL,))
    self.assertEqual(changes.remove, (ASSIGNEE_LABEL,))

  def test_case_insensitive_labels_preserved_on_delete(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": "STAT:AWAITING-ASSIGNEE"}],
        ),
        "sender": {"login": "bob"},
    }
    changes = determine_label_changes("issue_comment", payload)
    self.assertEqual(changes.add, (REQUESTER_LABEL,))
    self.assertEqual(changes.remove, ("STAT:AWAITING-ASSIGNEE",))

  def test_idempotent_when_target_label_already_set(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(
            author="alice",
            assignee="bob",
            labels=[{"name": REQUESTER_LABEL}],
        ),
        "sender": {"login": "bob"},
    }
    changes = determine_label_changes("issue_comment", payload)
    self.assertEqual(changes.add, ())
    self.assertEqual(changes.remove, ())


class _FakeResponse:

  def __init__(self, body: bytes = b""):
    self._body = body

  def __enter__(self):
    return self

  def __exit__(self, *unused_exc):
    return None

  def read(self) -> bytes:
    return self._body


class _FakeOpener:
  """Records urllib.request.Request calls and optionally raises an error.

  GET requests for a URL in `responses` return that value as a JSON body.
  """

  def __init__(
      self,
      error: Exception | None = None,
      responses: dict[str, Any] | None = None,
  ):
    self.calls: list[tuple[str, str, dict[str, str], bytes | None, float]] = []
    self._error = error
    self._responses = responses or {}

  def __call__(self, req, timeout: float = 30.0):
    self.calls.append((
        req.get_method(),
        req.full_url,
        dict(req.header_items()),
        req.data,
        timeout,
    ))
    if self._error is not None:
      raise self._error
    if req.get_method() == "GET" and req.full_url in self._responses:
      return _FakeResponse(
          json.dumps(self._responses[req.full_url]).encode("utf-8")
      )
    return _FakeResponse()

  def requests(self, method: str) -> list[str]:
    """Returns the URLs of the received requests with `method`, in order."""
    return [url for m, url, _, _, _ in self.calls if m == method]


class RestApiExecutionTest(unittest.TestCase):  # UNITTEST_OK=standalone

  def test_apply_label_changes_calls(self):
    opener = _FakeOpener()

    apply_label_changes(
        api_url="https://api.github.com",
        repo="google-pytorch/torch_tpu",
        token="test-token",
        issue_number=42,
        to_add=[REQUESTER_LABEL],
        to_remove=[ASSIGNEE_LABEL],
        timeout=15.0,
        opener=opener,
    )

    self.assertEqual(len(opener.calls), 2)
    # First call: DELETE assignee label with timeout
    method, url, headers, _, timeout = opener.calls[0]
    self.assertEqual(method, "DELETE")
    self.assertIn("stat%3Aawaiting-assignee", url)
    self.assertEqual(headers["Authorization"], "Bearer test-token")
    self.assertEqual(timeout, 15.0)

    # Second call: POST requester label with timeout
    method, url, _, data, timeout = opener.calls[1]
    self.assertEqual(method, "POST")
    self.assertIn("/issues/42/labels", url)
    self.assertIsNotNone(data)
    self.assertEqual(
        json.loads(data.decode("utf-8")), {"labels": [REQUESTER_LABEL]}
    )
    self.assertEqual(timeout, 15.0)

  def test_apply_label_changes_swallows_404_on_delete(self):
    err = urllib.error.HTTPError(
        "https://url", 404, "Not Found", {}, fp=io.BytesIO()
    )
    opener = _FakeOpener(error=err)

    # Should not raise
    apply_label_changes(
        api_url="https://api.github.com",
        repo="google-pytorch/torch_tpu",
        token="test-token",
        issue_number=42,
        to_add=[],
        to_remove=["non-existent-label"],
        opener=opener,
    )
    self.assertEqual(len(opener.calls), 1)

  def test_apply_label_changes_raises_non_404(self):
    err = urllib.error.HTTPError(
        "https://url", 403, "Forbidden", {}, fp=io.BytesIO()
    )
    opener = _FakeOpener(error=err)

    with self.assertRaises(urllib.error.HTTPError):  # ASSERT_RAISES_OK=err
      apply_label_changes(
          api_url="https://api.github.com",
          repo="google-pytorch/torch_tpu",
          token="test-token",
          issue_number=42,
          to_add=[],
          to_remove=["label"],
          opener=opener,
      )


class WorkflowRunRelayTest(unittest.TestCase):  # UNITTEST_OK=standalone

  HEAD_SHA = "abc123"

  def _make_opener(self, pulls):
    self.requests = []
    responses = iter([
        {"items": [{"number": pr["number"]} for pr in pulls]},
        *pulls,
    ])

    def opener(req, timeout=None):
      del timeout
      self.requests.append(req)
      return io.BytesIO(json.dumps(next(responses)).encode("utf-8"))

    return opener

  def _make_run_payload(self, event="pull_request_review", actor="bob"):
    return {
        "action": "completed",
        "workflow_run": {
            "event": event,
            "head_sha": self.HEAD_SHA,
            "actor": {"login": actor, "type": "User"},
        },
    }

  def _make_pr(self, number=7, state="open", sha=HEAD_SHA):
    return {
        "number": number,
        "state": state,
        "head": {"sha": sha},
        "user": {"login": "alice"},
        "assignees": [{"login": "bob"}],
        "labels": [{"name": REVIEWERS_LABEL}],
    }

  def test_rebuilds_review_event_for_matching_open_pr(self):
    opener = self._make_opener([self._make_pr()])

    events = resolve_workflow_run(
        "https://api.github.com",
        "google-pytorch/torch_tpu",
        "tok",
        self._make_run_payload(),
        opener=opener,
    )

    self.assertEqual(len(self.requests), 2)
    self.assertEqual(self.requests[0].get_method(), "GET")
    self.assertEqual(
        self.requests[0].full_url,
        "https://api.github.com/search/issues?q=repo%3Agoogle-pytorch"
        f"%2Ftorch_tpu+type%3Apr+state%3Aopen+{self.HEAD_SHA}",
    )
    self.assertEqual(self.requests[0].headers["Authorization"], "Bearer tok")
    self.assertEqual(
        self.requests[1].full_url,
        "https://api.github.com/repos/google-pytorch/torch_tpu/pulls/7",
    )
    self.assertEqual(len(events), 1)
    event_name, payload = events[0]
    self.assertEqual(event_name, "pull_request_review")
    self.assertEqual(payload["action"], "submitted")
    self.assertEqual(payload["sender"]["login"], "bob")
    self.assertEqual(payload["pull_request"]["number"], 7)

  def test_returns_empty_when_search_finds_no_prs(self):
    opener = self._make_opener([])

    events = resolve_workflow_run(
        "https://api.github.com",
        "o/r",
        "tok",
        self._make_run_payload(),
        opener=opener,
    )

    self.assertEqual(events, [])
    self.assertEqual(len(self.requests), 1)

  def test_review_comment_maps_to_created(self):
    opener = self._make_opener([self._make_pr()])
    events = resolve_workflow_run(
        "https://api.github.com",
        "o/r",
        "tok",
        self._make_run_payload(event="pull_request_review_comment"),
        opener=opener,
    )
    self.assertEqual(events[0][0], "pull_request_review_comment")
    self.assertEqual(events[0][1]["action"], "created")

  def test_skips_closed_and_stale_head_prs(self):
    opener = self._make_opener([
        self._make_pr(number=1, state="closed"),
        self._make_pr(number=2, sha="other"),
        self._make_pr(number=3),
    ])
    events = resolve_workflow_run(
        "https://api.github.com",
        "o/r",
        "tok",
        self._make_run_payload(),
        opener=opener,
    )
    self.assertEqual([p["pull_request"]["number"] for _, p in events], [3])

  def test_ignores_unrelated_upstream_event(self):
    opener = self._make_opener([self._make_pr()])
    events = resolve_workflow_run(
        "https://api.github.com",
        "o/r",
        "tok",
        self._make_run_payload(event="push"),
        opener=opener,
    )
    self.assertEqual(events, [])
    self.assertEqual(self.requests, [])

  def test_rebuilt_event_flips_turn(self):
    _, payload = resolve_workflow_run(
        "https://api.github.com",
        "o/r",
        "tok",
        self._make_run_payload(actor="bob"),
        opener=self._make_opener([self._make_pr()]),
    )[0]
    changes = determine_label_changes("pull_request_review", payload)
    self.assertEqual(changes.add, (REQUESTER_LABEL,))
    self.assertEqual(changes.remove, (REVIEWERS_LABEL,))


class FetchPrReviewersTest(unittest.TestCase):  # UNITTEST_OK=standalone

  def _make_opener(self, requested, reviews):
    responses = [json.dumps(requested), json.dumps(reviews)]
    opener = mock.MagicMock()
    opener.return_value.__enter__.return_value.read.side_effect = [
        r.encode("utf-8") for r in responses
    ]
    return opener

  def test_unions_pending_requests_and_submitted_reviews(self):
    opener = self._make_opener(
        requested={"users": [{"login": "Dana"}], "teams": [{"slug": "qa"}]},
        reviews=[{"user": {"login": "erin"}}, {"user": {"login": "dana"}}],
    )

    reviewers = fetch_pr_reviewers(
        "https://api.github.com", "o/r", "tok", 42, opener=opener
    )

    self.assertEqual(reviewers, frozenset({"dana", "erin"}))
    urls = [c[0][0].full_url for c in opener.call_args_list]
    self.assertEqual(
        urls,
        [
            "https://api.github.com/repos/o/r/pulls/42/requested_reviewers",
            "https://api.github.com/repos/o/r/pulls/42/reviews?per_page=100",
        ],
    )

  def test_ignores_pending_reviews(self):
    opener = self._make_opener(
        requested={"users": []},
        reviews=[
            {"user": {"login": "erin"}, "state": "PENDING"},
            {"user": {"login": "dana"}, "state": "COMMENTED"},
        ],
    )
    reviewers = fetch_pr_reviewers(
        "https://api.github.com", "o/r", "tok", 42, opener=opener
    )
    self.assertEqual(reviewers, frozenset({"dana"}))

  def test_empty_when_no_reviewers(self):
    opener = self._make_opener(requested={"users": []}, reviews=[])
    reviewers = fetch_pr_reviewers(
        "https://api.github.com", "o/r", "tok", 42, opener=opener
    )
    self.assertEqual(reviewers, frozenset())


class MainEntryPointTest(unittest.TestCase):  # UNITTEST_OK=standalone

  def test_main_exits_on_missing_env(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch("sys.stderr", new_callable=io.StringIO):
        with self.assertRaises(SystemExit) as ctx:  # ASSERT_RAISES_OK=exit
          main()
        self.assertEqual(ctx.exception.code, 1)

  _API = "https://api.github.com/repos/google-pytorch/torch_tpu"

  def _run_main(self, event_name, payload, opener):
    with tempfile.TemporaryDirectory() as tmp_dir:
      event_path = os.path.join(tmp_dir, "event.json")
      with open(event_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
      env = {
          "GITHUB_TOKEN": "secret",
          "GITHUB_EVENT_PATH": event_path,
          "GITHUB_REPOSITORY": "google-pytorch/torch_tpu",
          "GITHUB_EVENT_NAME": event_name,
          "GITHUB_API_URL": "https://api.github.com",
      }
      with mock.patch.dict(os.environ, env):
        main(opener=opener)

  def _reviewer_responses(self, pr_number, requested=(), reviewed=()):
    base = f"{self._API}/pulls/{pr_number}"
    return {
        f"{base}/requested_reviewers": {
            "users": [{"login": login} for login in requested]
        },
        f"{base}/reviews?per_page=100": [
            {"user": {"login": login}} for login in reviewed
        ],
    }

  def _assert_labels_flipped(self, opener, pr_number, added, removed):
    labels_url = f"{self._API}/issues/{pr_number}/labels"
    self.assertEqual(
        opener.requests("DELETE"),
        [f"{labels_url}/{urllib.parse.quote(label)}" for label in removed],
    )
    self.assertEqual(opener.requests("POST"), [labels_url])
    (data,) = [d for m, _, _, d, _ in opener.calls if m == "POST"]
    self.assertEqual(json.loads(data.decode("utf-8")), {"labels": added})

  def test_main_success_flow(self):
    payload = {
        "action": "opened",
        "issue": {
            "number": 123,
            "user": {"login": "alice"},
            "labels": [],
        },
    }
    opener = _FakeOpener()

    self._run_main("issues", payload, opener)

    # An issue has no reviewers to fetch.
    self.assertEqual(opener.requests("GET"), [])
    self.assertEqual(len(opener.calls), 1)
    method, url, headers, data, _ = opener.calls[0]
    self.assertEqual(method, "POST")
    self.assertEqual(url, f"{self._API}/issues/123/labels")
    self.assertEqual(headers["Authorization"], "Bearer secret")
    self.assertEqual(
        json.loads(data.decode("utf-8")), {"labels": [ASSIGNEE_LABEL]}
    )

  def test_main_success_flow_pull_request(self):
    payload = {
        "action": "submitted",
        "pull_request": {
            "number": 456,
            "state": "open",
            "user": {"login": "alice"},
            "assignee": {"login": "bob"},
            "assignees": [{"login": "bob"}],
            "labels": [{"name": REVIEWERS_LABEL}],
        },
        "sender": {"login": "bob"},
    }
    responses = self._reviewer_responses(456)
    opener = _FakeOpener(responses=responses)

    self._run_main("pull_request_review", payload, opener)

    self.assertEqual(opener.requests("GET"), list(responses))
    self._assert_labels_flipped(
        opener, 456, added=[REQUESTER_LABEL], removed=[REVIEWERS_LABEL]
    )

  def test_main_pr_comment_uses_fetched_reviewers(self):
    payload = {
        "action": "created",
        "issue": {
            "number": 457,
            "state": "open",
            "user": {"login": "alice"},
            "assignees": [],
            "labels": [{"name": REVIEWERS_LABEL}],
            "pull_request": {
                "url": "https://api.github.com/repos/o/r/pulls/457"
            },
        },
        "sender": {"login": "dana"},
    }
    # dana is a reviewer only according to the fetched review requests.
    opener = _FakeOpener(
        responses=self._reviewer_responses(457, requested=["dana"])
    )

    self._run_main("issue_comment", payload, opener)

    self._assert_labels_flipped(
        opener, 457, added=[REQUESTER_LABEL], removed=[REVIEWERS_LABEL]
    )

  def test_main_workflow_run_relays_each_pr(self):
    head_sha = "abc123"
    payload = {
        "action": "completed",
        "workflow_run": {
            "event": "pull_request_review",
            "head_sha": head_sha,
            "actor": {"login": "bob", "type": "User"},
        },
    }
    pr = {
        "number": 789,
        "state": "open",
        "head": {"sha": head_sha},
        "user": {"login": "alice"},
        "assignees": [{"login": "bob"}],
        "labels": [{"name": REVIEWERS_LABEL}],
    }
    responses = {
        f"{self._API}/commits/{head_sha}/pulls": [pr],
        **self._reviewer_responses(789),
    }
    opener = _FakeOpener(responses=responses)

    self._run_main("workflow_run", payload, opener)

    self.assertEqual(opener.requests("GET"), list(responses))
    self._assert_labels_flipped(
        opener, 789, added=[REQUESTER_LABEL], removed=[REVIEWERS_LABEL]
    )


if __name__ == "__main__":
  unittest.main()
