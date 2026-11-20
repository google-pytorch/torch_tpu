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

"""Tests for `tools/assign_quality_champion.py`."""

import email.message
import pathlib
import random
import tempfile
import unittest
import urllib.error

from tools import assign_quality_champion as aqc


def _payload(
    *,
    action="completed",
    event="pull_request_review",
    actor="alice",
    pr_numbers=(42,),
    head_sha="",
):
  workflow_run = {
      "event": event,
      "actor": {"login": actor},
      "pull_requests": [{"number": num} for num in pr_numbers],
  }
  if head_sha:
    workflow_run["head_sha"] = head_sha
  return {"action": action, "workflow_run": workflow_run}


def _open_pr(**overrides):
  pr = {
      "state": "open",
      "draft": False,
      "assignees": [],
      "user": {"login": "external-author"},
      "head": {"sha": "abc123"},
  }
  pr.update(overrides)
  return pr


class AssignQualityChampionTest(unittest.TestCase):  # UNITTEST_OK=standalone

  def _fake_github(
      self,
      *,
      pulls=None,
      reviews=None,
      permission="write",
      search_items=(),
      delete_404=False,
      perm_404=False,
  ):
    calls = []
    pulls = pulls if pulls is not None else {42: _open_pr()}
    reviews = (
        reviews
        if reviews is not None
        else [{
            "user": {"login": "alice"},
            "commit_id": "abc123",
            "state": "APPROVED",
        }]
    )

    def request_fn(url, token, method="GET", data=None):
      del token
      calls.append((method, url, data))
      if method == "DELETE" and delete_404:
        raise urllib.error.HTTPError(
            url, 404, "Not Found", email.message.Message(), None
        )
      if "/collaborators/" in url:
        if perm_404:
          raise urllib.error.HTTPError(
              url, 404, "Not Found", email.message.Message(), None
          )
        return {"permission": permission}
      if "/search/issues?" in url:
        return {"items": [{"number": num} for num in search_items]}
      if url.endswith("/reviews?per_page=100"):
        return reviews
      for num, pr_data in pulls.items():
        if url.endswith(f"/pulls/{num}"):
          return pr_data
      return {}

    return calls, request_fn

  def test_load_champions_ignores_comments_and_preserves_case(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = pathlib.Path(tmpdir) / "quality_champions.txt"
      path.write_text("# comment\nAlice  # inline\n\nbob\n", encoding="utf-8")
      self.assertEqual(aqc.load_champions(path), ("Alice", "bob"))

  def test_choose_champion_excludes_author_and_approver(self):
    self.assertEqual(
        aqc.choose_champion(("Alice", "Bob", "Carol"), "alice", "bob"), "Carol"
    )
    self.assertIsNone(aqc.choose_champion(("Alice", "Bob"), "alice", "bob"))

  def test_is_approved_on_commit(self):
    def rev(user, state, commit_id="sha123"):
      return {"user": {"login": user}, "commit_id": commit_id, "state": state}

    self.assertTrue(
        aqc.is_approved_on_commit(
            [
                rev("Alice", "CHANGES_REQUESTED"),
                rev("alice", "APPROVED"),
                rev("alice", "PENDING"),
            ],
            "alice",
            "sha123",
        )
    )
    self.assertFalse(
        aqc.is_approved_on_commit(
            [rev("alice", "APPROVED"), rev("alice", "CHANGES_REQUESTED")],
            "alice",
            "sha123",
        )
    )
    self.assertFalse(
        aqc.is_approved_on_commit(
            [rev("alice", "APPROVED", "old"), rev("bob", "APPROVED")],
            "alice",
            "sha123",
        )
    )

  def test_assigns_random_champion_and_updates_reviewers_status_and_labels(
      self,
  ):
    calls, request_fn = self._fake_github()
    assigned = aqc.process_workflow_run(
        payload=_payload(),
        champions=("alice", "bob", "carol"),
        token="tok",
        repo="google-pytorch/torch_tpu",
        request_fn=request_fn,
        rng=random.Random(0),
    )
    self.assertEqual(len(assigned), 1)
    pr_number, chosen = assigned[0]
    self.assertEqual(pr_number, 42)
    self.assertIn(chosen, ("bob", "carol"))
    base = "https://api.github.com/repos/google-pytorch/torch_tpu"
    self.assertEqual(
        [call for call in calls if call[0] != "GET"],
        [
            ("POST", f"{base}/issues/42/assignees", {"assignees": [chosen]}),
            (
                "POST",
                f"{base}/pulls/42/requested_reviewers",
                {"reviewers": [chosen]},
            ),
            (
                "POST",
                f"{base}/statuses/abc123",
                {
                    "state": "success",
                    "context": aqc.STATUS_CONTEXT,
                    "description": f"Assigned to quality champion @{chosen}.",
                },
            ),
            (
                "POST",
                f"{base}/issues/42/labels",
                {"labels": [aqc.REVIEWERS_LABEL]},
            ),
            (
                "DELETE",
                f"{base}/issues/42/labels/stat%3Aawaiting-requester",
                None,
            ),
        ],
    )

  def test_ignores_404_when_deleting_requester_label(self):
    _, request_fn = self._fake_github(permission="maintain", delete_404=True)
    assigned = aqc.process_workflow_run(
        payload=_payload(),
        champions=("bob",),
        token="tok",
        repo="google-pytorch/torch_tpu",
        request_fn=request_fn,
    )
    self.assertEqual(assigned, [(42, "bob")])

  def test_skips_bot_actor_and_404_permission(self):
    calls, request_fn = self._fake_github()
    self.assertEqual(
        aqc.process_workflow_run(
            payload=_payload(actor="dependabot[bot]"),
            champions=("bob",),
            token="tok",
            repo="google-pytorch/torch_tpu",
            request_fn=request_fn,
        ),
        [],
    )
    self.assertEqual(calls, [])

    _, not_found_fn = self._fake_github(perm_404=True)
    self.assertEqual(
        aqc.process_workflow_run(
            payload=_payload(actor="external-user"),
            champions=("bob",),
            token="tok",
            repo="google-pytorch/torch_tpu",
            request_fn=not_found_fn,
        ),
        [],
    )

  def test_skips_ineligible_prs_and_non_approvals(self):
    cases = [
        (
            "already_assigned",
            _open_pr(assignees=[{"login": "bob"}]),
            "APPROVED",
        ),
        ("draft_pr", _open_pr(draft=True), "APPROVED"),
        (
            "copybara_pr",
            _open_pr(user={"login": "copybara-service[bot]"}),
            "APPROVED",
        ),
        ("closed_pr", _open_pr(state="closed"), "APPROVED"),
        ("non_approval_review", _open_pr(), "COMMENTED"),
    ]
    for name, pr_data, review_state in cases:
      with self.subTest(name=name):
        calls, request_fn = self._fake_github(
            pulls={42: pr_data},
            reviews=[{
                "user": {"login": "alice"},
                "commit_id": "abc123",
                "state": review_state,
            }],
        )
        assigned = aqc.process_workflow_run(
            payload=_payload(),
            champions=("bob", "carol"),
            token="tok",
            repo="google-pytorch/torch_tpu",
            request_fn=request_fn,
        )
        self.assertEqual(assigned, [])
        self.assertEqual([call for call in calls if call[0] != "GET"], [])

  def test_resolves_fork_pr_via_search_when_pull_requests_empty(self):
    _, request_fn = self._fake_github(
        pulls={
            99: _open_pr(head={"sha": "other_sha"}),
            42: _open_pr(user={"login": "fork-author"}),
        },
        search_items=(99, 42),
    )
    assigned = aqc.process_workflow_run(
        payload=_payload(pr_numbers=(), head_sha="abc123"),
        champions=("bob",),
        token="tok",
        repo="google-pytorch/torch_tpu",
        request_fn=request_fn,
    )
    self.assertEqual(assigned, [(42, "bob")])


if __name__ == "__main__":
  unittest.main()
