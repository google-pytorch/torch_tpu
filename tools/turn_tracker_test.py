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
    apply_label_changes,
    determine_label_changes,
    main,
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
  ):
    issue = {
        "number": number,
        "state": state,
        "user": {"login": author},
        "labels": labels or [],
    }
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

  def test_pull_request_ignored(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(author="alice", assignee="bob", is_pr=True),
        "sender": {"login": "alice"},
    }
    changes = determine_label_changes("issue_comment", payload)
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

  def test_bot_comment_ignored(self):
    payload = {
        "action": "created",
        "issue": self._make_issue(author="alice", assignee="bob"),
        "sender": {"login": "github-actions[bot]", "type": "Bot"},
    }
    changes = determine_label_changes("issue_comment", payload)
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

  def __enter__(self):
    return self

  def __exit__(self, *unused_exc):
    return None


class _FakeOpener:
  """Records urllib.request.Request calls and optionally raises an error."""

  def __init__(self, error: Exception | None = None):
    self.calls: list[tuple[str, str, dict[str, str], bytes | None, float]] = []
    self._error = error

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
    return _FakeResponse()


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


class MainEntryPointTest(unittest.TestCase):  # UNITTEST_OK=standalone

  def test_main_exits_on_missing_env(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch("sys.stderr", new_callable=io.StringIO):
        with self.assertRaises(SystemExit) as ctx:  # ASSERT_RAISES_OK=exit
          main()
        self.assertEqual(ctx.exception.code, 1)

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
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_path = os.path.join(tmp_dir, "test_event.json")
      with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

      env = {
          "GITHUB_TOKEN": "secret",
          "GITHUB_EVENT_PATH": tmp_path,
          "GITHUB_REPOSITORY": "google-pytorch/torch_tpu",
          "GITHUB_EVENT_NAME": "issues",
          "GITHUB_API_URL": "https://api.github.com",
      }
      with mock.patch.dict(os.environ, env):
        main(opener=opener)

    self.assertEqual(len(opener.calls), 1)
    method, url, headers, data, _ = opener.calls[0]
    self.assertEqual(method, "POST")
    self.assertEqual(
        url,
        "https://api.github.com/repos/google-pytorch/torch_tpu/issues/123/labels",
    )
    self.assertEqual(headers["Authorization"], "Bearer secret")
    self.assertEqual(
        json.loads(data.decode("utf-8")), {"labels": [ASSIGNEE_LABEL]}
    )


if __name__ == "__main__":
  unittest.main()
