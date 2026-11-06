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
import unittest
from unittest import mock
import urllib.error

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


class RestApiExecutionTest(unittest.TestCase):  # UNITTEST_OK=standalone

  def test_apply_label_changes_calls(self):
    mock_opener = mock.MagicMock()
    mock_opener.return_value.__enter__.return_value = None

    apply_label_changes(
        api_url="https://api.github.com",
        repo="google-pytorch/torch_tpu",
        token="mock-token",
        issue_number=42,
        to_add=[REQUESTER_LABEL],
        to_remove=[ASSIGNEE_LABEL],
        timeout=15.0,
        opener=mock_opener,
    )

    self.assertEqual(mock_opener.call_count, 2)
    # First call: DELETE assignee label with timeout
    delete_call = mock_opener.call_args_list[0]
    req = delete_call[0][0]
    self.assertEqual(req.get_method(), "DELETE")
    self.assertIn("stat%3Aawaiting-assignee", req.full_url)
    self.assertEqual(req.headers["Authorization"], "Bearer mock-token")
    self.assertEqual(delete_call[1]["timeout"], 15.0)

    # Second call: POST requester label with timeout
    post_call = mock_opener.call_args_list[1]
    req = post_call[0][0]
    self.assertEqual(req.get_method(), "POST")
    self.assertIn("/issues/42/labels", req.full_url)
    self.assertEqual(
        json.loads(req.data.decode("utf-8")), {"labels": [REQUESTER_LABEL]}
    )
    self.assertEqual(post_call[1]["timeout"], 15.0)

  def test_apply_label_changes_swallows_404_on_delete(self):
    fp = mock.MagicMock()
    err = urllib.error.HTTPError("https://url", 404, "Not Found", {}, fp)
    mock_opener = mock.MagicMock(side_effect=err)

    # Should not raise
    apply_label_changes(
        api_url="https://api.github.com",
        repo="google-pytorch/torch_tpu",
        token="mock-token",
        issue_number=42,
        to_add=[],
        to_remove=["non-existent-label"],
        opener=mock_opener,
    )

  def test_apply_label_changes_raises_non_404(self):
    fp = mock.MagicMock()
    err = urllib.error.HTTPError("https://url", 403, "Forbidden", {}, fp)
    mock_opener = mock.MagicMock(side_effect=err)

    with self.assertRaises(urllib.error.HTTPError):  # ASSERT_RAISES_OK=err
      apply_label_changes(
          api_url="https://api.github.com",
          repo="google-pytorch/torch_tpu",
          token="mock-token",
          issue_number=42,
          to_add=[],
          to_remove=["label"],
          opener=mock_opener,
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
      with mock.patch("sys.stderr"):
        with self.assertRaises(SystemExit) as ctx:  # ASSERT_RAISES_OK=exit
          main()
        self.assertEqual(ctx.exception.code, 1)

  def _run_main(self, event_name, payload):
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
        main()

  @mock.patch("turn_tracker.fetch_pr_reviewers")
  @mock.patch("turn_tracker.apply_label_changes")
  def test_main_success_flow(self, mock_apply, mock_fetch):
    payload = {
        "action": "opened",
        "issue": {
            "number": 123,
            "user": {"login": "alice"},
            "labels": [],
        },
    }

    self._run_main("issues", payload)

    mock_fetch.assert_not_called()
    mock_apply.assert_called_once_with(
        "https://api.github.com",
        "google-pytorch/torch_tpu",
        "secret",
        123,
        (ASSIGNEE_LABEL,),
        (),
    )

  @mock.patch("turn_tracker.fetch_pr_reviewers", return_value=frozenset())
  @mock.patch("turn_tracker.apply_label_changes")
  def test_main_success_flow_pull_request(self, mock_apply, mock_fetch):
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

    self._run_main("pull_request_review", payload)

    mock_fetch.assert_called_once_with(
        "https://api.github.com", "google-pytorch/torch_tpu", "secret", 456
    )
    mock_apply.assert_called_once_with(
        "https://api.github.com",
        "google-pytorch/torch_tpu",
        "secret",
        456,
        (REQUESTER_LABEL,),
        (REVIEWERS_LABEL,),
    )

  @mock.patch(
      "turn_tracker.fetch_pr_reviewers", return_value=frozenset({"dana"})
  )
  @mock.patch("turn_tracker.apply_label_changes")
  def test_main_pr_comment_uses_fetched_reviewers(self, mock_apply, mock_fetch):
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

    self._run_main("issue_comment", payload)

    mock_fetch.assert_called_once()
    mock_apply.assert_called_once_with(
        "https://api.github.com",
        "google-pytorch/torch_tpu",
        "secret",
        457,
        (REQUESTER_LABEL,),
        (REVIEWERS_LABEL,),
    )

  @mock.patch("turn_tracker.fetch_pr_reviewers", return_value=frozenset())
  @mock.patch("turn_tracker.apply_label_changes")
  @mock.patch("turn_tracker.resolve_workflow_run")
  def test_main_workflow_run_relays_each_pr(
      self, mock_resolve, mock_apply, mock_fetch
  ):
    mock_resolve.return_value = [(
        "pull_request_review",
        {
            "action": "submitted",
            "pull_request": {
                "number": 789,
                "state": "open",
                "user": {"login": "alice"},
                "assignees": [{"login": "bob"}],
                "labels": [{"name": REVIEWERS_LABEL}],
            },
            "sender": {"login": "bob"},
        },
    )]
    payload = {"workflow_run": {"event": "pull_request_review"}}

    self._run_main("workflow_run", payload)

    mock_resolve.assert_called_once_with(
        "https://api.github.com",
        "google-pytorch/torch_tpu",
        "secret",
        payload,
    )
    mock_apply.assert_called_once_with(
        "https://api.github.com",
        "google-pytorch/torch_tpu",
        "secret",
        789,
        (REQUESTER_LABEL,),
        (REVIEWERS_LABEL,),
    )


if __name__ == "__main__":
  unittest.main()
