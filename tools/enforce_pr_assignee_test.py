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

"""Unit tests for enforce_pr_assignee.py."""

import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from tools import enforce_pr_assignee as epa


class EnforcePrAssigneeTest(unittest.TestCase):  # UNITTEST_OK=standalone

  def _make_payload(
      self,
      assignees=(),
      state="open",
      author="contributor",
      number=42,
      sha="deadbeef",
      action="assigned",
      event_assignee="",
  ):
    payload = {
        "action": action,
        "pull_request": {
            "number": number,
            "state": state,
            "user": {"login": author},
            "head": {"sha": sha},
            "assignees": [{"login": assignee} for assignee in assignees],
        },
    }
    if event_assignee:
      payload["assignee"] = {"login": event_assignee}
    return payload

  def _run_plan(self, payload, champions):
    plan = epa.plan_enforcement(payload, champions)
    if plan is None:
      self.fail("Expected EnforcementPlan, got None")
    calls = []

    def fake_req(url, token, method, data):
      calls.append((method, url, token, data))

    epa.apply_plan(plan, "org/repo", "tok", request_fn=fake_req)
    return calls

  def test_load_champions_parses_comments_and_lowercases(self):
    with tempfile.TemporaryDirectory() as tmp:
      cfg = pathlib.Path(tmp) / "quality_champions.txt"
      cfg.write_text(
          "# comment\n\n  LeeHowes  \nzhanyong-wan # inline comment\n",
          encoding="utf-8",
      )
      self.assertEqual(
          epa.load_champions(cfg),
          frozenset({"leehowes", "zhanyong-wan"}),
      )

  def test_valid_champion_case_insensitive_sets_success(self):
    champions = frozenset({"zhanyong-wan", "leehowes"})
    calls = self._run_plan(
        self._make_payload(assignees=["Zhanyong-Wan"]), champions
    )

    self.assertEqual(len(calls), 1)
    method, url, _, data = calls[0]
    self.assertEqual(method, "POST")
    self.assertEqual(
        url, "https://api.github.com/repos/org/repo/statuses/deadbeef"
    )
    self.assertEqual(data["state"], "success")
    self.assertEqual(data["context"], epa.STATUS_CONTEXT)

  def test_unassigned_pr_sets_pending_without_removal_or_comment(self):
    champions = frozenset({"zhanyong-wan"})
    calls = self._run_plan(self._make_payload(assignees=[]), champions)

    self.assertEqual(len(calls), 1)
    method, url, _, data = calls[0]
    self.assertEqual(method, "POST")
    self.assertEqual(
        url, "https://api.github.com/repos/org/repo/statuses/deadbeef"
    )
    self.assertEqual(data["state"], "pending")
    self.assertEqual(data["context"], epa.STATUS_CONTEXT)

  def test_non_champion_removed_commented_and_pending(self):
    champions = frozenset({"zhanyong-wan"})
    calls = self._run_plan(
        self._make_payload(assignees=["expert-dev"]), champions
    )

    self.assertEqual(
        [(method, url) for method, url, _, _ in calls],
        [
            (
                "DELETE",
                "https://api.github.com/repos/org/repo/issues/42/assignees",
            ),
            (
                "POST",
                "https://api.github.com/repos/org/repo/issues/42/comments",
            ),
            ("POST", "https://api.github.com/repos/org/repo/statuses/deadbeef"),
        ],
    )
    self.assertEqual(calls[0][3], {"assignees": ["expert-dev"]})
    self.assertIn("@expert-dev", calls[1][3]["body"])
    self.assertIn("quality_champions.txt", calls[1][3]["body"])
    self.assertEqual(calls[2][3]["state"], "pending")

  def test_mixed_assignees_removes_only_non_champion_and_sets_success(self):
    champions = frozenset({"cbasile-g"})
    calls = self._run_plan(
        self._make_payload(assignees=["cbasile-g", "other-user"]), champions
    )

    self.assertEqual(calls[0][3], {"assignees": ["other-user"]})
    self.assertEqual(calls[2][3]["state"], "success")

  def test_skips_closed_and_copybara_prs(self):
    champions = frozenset({"zhanyong-wan"})
    self.assertIsNone(
        epa.plan_enforcement(
            self._make_payload(assignees=["anyone"], state="closed"), champions
        )
    )
    self.assertIsNone(
        epa.plan_enforcement(
            self._make_payload(
                assignees=["anyone"], author="copybara-service[bot]"
            ),
            champions,
        )
    )

  def test_unassigning_only_champion_restores_them_and_keeps_success(self):
    champions = frozenset({"zhanyong-wan"})
    calls = self._run_plan(
        self._make_payload(
            assignees=[],
            action="unassigned",
            event_assignee="zhanyong-wan",
        ),
        champions,
    )

    self.assertEqual(
        [(method, url) for method, url, _, _ in calls],
        [
            (
                "POST",
                "https://api.github.com/repos/org/repo/issues/42/assignees",
            ),
            (
                "POST",
                "https://api.github.com/repos/org/repo/issues/42/comments",
            ),
            ("POST", "https://api.github.com/repos/org/repo/statuses/deadbeef"),
        ],
    )
    self.assertEqual(calls[0][3], {"assignees": ["zhanyong-wan"]})
    self.assertIn("Re-assigned @zhanyong-wan", calls[1][3]["body"])
    self.assertEqual(calls[2][3]["state"], "success")

  def test_replacing_only_champion_with_non_champion_is_rejected(self):
    champions = frozenset({"zhanyong-wan"})
    calls = self._run_plan(
        self._make_payload(
            assignees=["non-champion"],
            action="unassigned",
            event_assignee="zhanyong-wan",
        ),
        champions,
    )

    self.assertEqual(
        [(method, url) for method, url, _, _ in calls],
        [
            (
                "DELETE",
                "https://api.github.com/repos/org/repo/issues/42/assignees",
            ),
            (
                "POST",
                "https://api.github.com/repos/org/repo/issues/42/comments",
            ),
            (
                "POST",
                "https://api.github.com/repos/org/repo/issues/42/assignees",
            ),
            (
                "POST",
                "https://api.github.com/repos/org/repo/issues/42/comments",
            ),
            ("POST", "https://api.github.com/repos/org/repo/statuses/deadbeef"),
        ],
    )
    self.assertEqual(calls[0][3], {"assignees": ["non-champion"]})
    self.assertEqual(calls[2][3], {"assignees": ["zhanyong-wan"]})
    self.assertEqual(calls[4][3]["state"], "success")

  def test_unassigning_champion_when_another_champion_remains_is_allowed(self):
    champions = frozenset({"zhanyong-wan", "cbasile-g"})
    calls = self._run_plan(
        self._make_payload(
            assignees=["cbasile-g"],
            action="unassigned",
            event_assignee="zhanyong-wan",
        ),
        champions,
    )

    self.assertEqual(len(calls), 1)
    self.assertEqual(calls[0][0], "POST")
    self.assertEqual(calls[0][3]["state"], "success")

  def test_assigning_second_champion_unassigns_previous_champion(self):
    champions = frozenset({"zhanyong-wan", "cbasile-g"})
    calls = self._run_plan(
        self._make_payload(
            assignees=["cbasile-g", "zhanyong-wan"],
            action="assigned",
            event_assignee="zhanyong-wan",
        ),
        champions,
    )

    self.assertEqual(
        [(method, url) for method, url, _, _ in calls],
        [
            (
                "DELETE",
                "https://api.github.com/repos/org/repo/issues/42/assignees",
            ),
            (
                "POST",
                "https://api.github.com/repos/org/repo/issues/42/comments",
            ),
            ("POST", "https://api.github.com/repos/org/repo/statuses/deadbeef"),
        ],
    )
    self.assertEqual(calls[0][3], {"assignees": ["cbasile-g"]})
    self.assertIn("at most one assignee", calls[1][3]["body"])
    self.assertIn("@zhanyong-wan", calls[1][3]["body"])
    self.assertEqual(calls[2][3]["state"], "success")

  def test_main_missing_env_returns_1(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertEqual(epa.main(), 1)

  def test_main_reads_repo_config(self):
    payload = self._make_payload(assignees=["LeeHowes"])
    calls = []

    def fake_req(url, token, method, data):
      calls.append((method, url, token, data))

    with tempfile.TemporaryDirectory() as tmp:
      event_file = pathlib.Path(tmp) / "event.json"
      event_file.write_text(json.dumps(payload), encoding="utf-8")
      with mock.patch.dict(
          os.environ,
          {
              "GITHUB_TOKEN": "tok",
              "GITHUB_EVENT_PATH": str(event_file),
              "GITHUB_REPOSITORY": "google-pytorch/torch_tpu",
          },
          clear=True,
      ):
        self.assertEqual(epa.main(request_fn=fake_req), 0)

    self.assertEqual(len(calls), 1)
    method, url, token, data = calls[0]
    self.assertEqual(method, "POST")
    self.assertEqual(
        url,
        "https://api.github.com/repos/google-pytorch/torch_tpu/statuses/deadbeef",
    )
    self.assertEqual(token, "tok")
    self.assertEqual(data["state"], "success")
    self.assertEqual(data["context"], epa.STATUS_CONTEXT)


if __name__ == "__main__":
  unittest.main()
