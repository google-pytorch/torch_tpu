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

"""Tests for tools/stray_refs.py against a local bare remote and a fake API."""

import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest
import urllib.error
from typing import Any

import stray_refs

REPO = "example/repo"
_GIT_ENV = {
  "GIT_AUTHOR_NAME": "Test User",
  "GIT_AUTHOR_EMAIL": "test@example.com",
  "GIT_COMMITTER_NAME": "Test User",
  "GIT_COMMITTER_EMAIL": "test@example.com",
  "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
  "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
}


def _git(repo: pathlib.Path, *args: str) -> str:
  env = {**os.environ, **_GIT_ENV}
  return subprocess.check_output(
    ["git", "-c", "safe.bareRepository=all", "-C", str(repo), *args],
    text=True,
    env=env,
    stderr=subprocess.STDOUT,
  ).strip()


def _commit(repo: pathlib.Path, name: str) -> str:
  (repo / name).write_text(name + "\n", encoding="utf-8")
  _git(repo, "add", name)
  _git(repo, "commit", "-q", "-m", name)
  return _git(repo, "rev-parse", "HEAD")


def _http_error(code: int) -> urllib.error.HTTPError:
  return urllib.error.HTTPError("", code, "", {}, None)  # type: ignore


class FakeApi:
  """Scripted GitHub API. `PATCH git/refs/*` is applied to the local remote."""

  def __init__(self, remote: pathlib.Path) -> None:
    self.remote = remote
    self.activity: dict[str, list[dict[str, Any]]] = {}
    self.open_prs: dict[str, int] = {}
    self.issues: list[dict[str, Any]] = []
    self.pr_comments: dict[int, list[dict[str, Any]]] = {}
    self.calls: list[tuple[str, str, Any]] = []
    self.reject_assignees = False

  def __call__(self, method: str, path: str, body: Any) -> Any:
    self.calls.append((method, path, body))
    if method == "GET" and "/activity?" in path:
      ref = path.split("ref=", 1)[1].split("&", 1)[0]
      return self.activity.get(ref.replace("%2F", "/"), [])
    if method == "GET" and "/pulls?" in path:
      branch = path.split("%3A", 1)[1] if "%3A" in path else path.split(":")[1]
      return (
        [{"number": self.open_prs[branch]}] if branch in self.open_prs else []
      )
    if method == "GET" and path.endswith("/comments?per_page=100"):
      number = int(path.split("/issues/")[1].split("/")[0])
      return self.pr_comments.get(number, [])
    if method == "GET" and "/issues?" in path:
      return self.issues
    if method == "GET" and "/git/ref/" in path:
      ref = "refs/" + path.split("/git/ref/", 1)[1]
      return {"object": {"sha": _git(self.remote, "rev-parse", ref)}}
    if method == "PATCH" and "/git/refs/" in path:
      ref = "refs/" + path.split("/git/refs/", 1)[1]
      _git(self.remote, "update-ref", ref, body["sha"])
      return {"object": {"sha": body["sha"]}}
    if method == "POST" and path.endswith("/issues"):
      if self.reject_assignees and "assignees" in body:
        raise _http_error(422)
      issue = {"number": 100 + len(self.issues), **body}
      self.issues.append(issue)
      return issue
    if method == "POST" and "/comments" in path:
      number = int(path.split("/issues/")[1].split("/")[0])
      self.pr_comments.setdefault(number, []).append(body)
      return {"id": 1}
    raise AssertionError(f"unexpected API call {method} {path}")


@unittest.skipIf(shutil.which("git") is None, "git not available")
class StrayRefsTest(unittest.TestCase):  # UNITTEST_OK=standalone
  """Remote layout: old_root->old1 is pre-rewrite, new_root->new1 rewritten.

  stray         old1->s1              archive + delete
  with-pr       old1->p1 (open PR 7)  archive + delete, PR comment
  release/1.0   old1, force-pushed from r1 (rewritten) -> restore to r1
  fine          new1->f1              untouched
  """

  def setUp(self) -> None:
    self.tmp = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
    work = self.tmp / "work"
    work.mkdir()
    _git(work, "init", "-q", "-b", "old")
    self.old_root = _commit(work, "old_root")
    self.old1 = _commit(work, "old1")
    _git(work, "update-ref", "refs/archive/main", self.old1)
    _git(work, "update-ref", "refs/archive/release/1.0", self.old1)
    _git(work, "branch", "release/1.0", "old")
    _git(work, "checkout", "-q", "-b", "stray")
    self.s1 = _commit(work, "s1")
    _git(work, "checkout", "-q", "-b", "with-pr", "old")
    self.p1 = _commit(work, "p1")
    _git(work, "checkout", "-q", "--orphan", "main")
    _git(work, "rm", "-rfq", ".")
    self.new_root = _commit(work, "new_root")
    self.new1 = _commit(work, "new1")
    _git(work, "checkout", "-q", "-b", "rr")
    self.r1 = _commit(work, "r1")
    _git(work, "checkout", "-q", "-b", "fine", "main")
    self.f1 = _commit(work, "f1")
    _git(work, "branch", "-D", "old")

    self.remote = self.tmp / "remote.git"
    _git(self.tmp, "init", "-q", "--bare", str(self.remote))
    _git(self.remote, "config", "uploadpack.allowFilter", "true")
    _git(self.remote, "config", "uploadpack.allowAnySHA1InWant", "true")
    _git(
      work,
      "push",
      "-q",
      str(self.remote),
      "+refs/heads/*:refs/heads/*",
      "+refs/archive/*:refs/archive/*",
    )
    _git(self.remote, "update-ref", "-d", "refs/heads/rr")  # r1: unreachable

    self.git_dir = stray_refs.ensure_scratch_repo(
      self.tmp / "scratch.git", str(self.remote)
    )
    self.api = FakeApi(self.remote)
    self.api.activity = {
      "refs/heads/stray": [
        {
          "activity_type": "push",
          "actor": {"login": "alice"},
          "before": self.old1,
          "after": self.s1,
          "timestamp": "2026-10-08T01:00:00Z",
        }
      ],
      "refs/heads/with-pr": [
        {
          "activity_type": "branch_creation",
          "actor": {"login": "copybara-service[bot]"},
          "before": "0" * 40,
          "after": self.p1,
          "timestamp": "2026-10-08T01:00:00Z",
        }
      ],
      "refs/heads/release/1.0": [
        {
          "activity_type": "force_push",
          "actor": {"login": "bob"},
          "before": self.r1,
          "after": self.old1,
          "timestamp": "2026-10-08T02:00:00Z",
        }
      ],
    }
    self.api.open_prs = {"with-pr": 7}

  def _refs(self) -> dict[str, str]:
    out = _git(self.remote, "for-each-ref", "--format=%(refname) %(objectname)")
    return dict(line.split(" ", 1) for line in out.splitlines())

  def _sweep(self, **kw: Any) -> stray_refs.SweepStats:
    return stray_refs.sweep(self.git_dir, self.api, REPO, token=None, **kw)

  def test_dry_run_changes_nothing(self) -> None:
    before = self._refs()
    stats = self._sweep(dry_run=True)
    self.assertEqual(stats.stray_refs_found, 3)
    self.assertEqual(stats.refs_archived, 2)
    self.assertEqual(stats.protected_restored, 1)
    self.assertEqual(self._refs(), before)
    self.assertEqual([c for c in self.api.calls if c[0] != "GET"], [])

  def test_live_sweep(self) -> None:
    stats = self._sweep()
    self.assertEqual(
      (
        stats.refs_archived,
        stats.protected_restored,
        stats.refs_failed,
        stats.notifications_sent,
      ),
      (2, 1, 0, 2),
    )
    refs = self._refs()
    self.assertEqual(refs["refs/archive/stray"], self.s1)
    self.assertEqual(refs["refs/archive/with-pr"], self.p1)
    self.assertNotIn("refs/heads/stray", refs)
    self.assertNotIn("refs/heads/with-pr", refs)
    self.assertEqual(refs["refs/heads/release/1.0"], self.r1)
    self.assertEqual(refs["refs/heads/fine"], self.f1)
    self.assertEqual(refs["refs/heads/main"], self.new1)

    titles = sorted(i["title"] for i in self.api.issues)
    self.assertEqual(
      titles,
      [
        "Protected branch `release/1.0` was reset to pre-rewrite history",
        "Stray pre-rewrite ref `stray` was archived",
      ],
    )
    alice = next(i for i in self.api.issues if "alice" in i["body"])
    self.assertEqual(alice["assignees"], ["alice"])
    self.assertIn("rebase --onto origin/main", alice["body"])
    self.assertIn(
      f"<!-- stray-ref refs/heads/stray@{self.s1} -->", alice["body"]
    )
    self.assertEqual(self.api.pr_comments, {})  # bot-created PR branch

    # Idempotent: a second pass finds nothing and posts nothing.
    again = self._sweep()
    self.assertEqual(again.stray_refs_found, 0)

  def test_push_event_attribution_and_pr_comment(self) -> None:
    payload = {
      "ref": "refs/heads/with-pr",
      "before": self.old1,
      "after": self.p1,
      "forced": False,
      "deleted": False,
      "sender": {"login": "carol"},
      "head_commit": {"timestamp": "2026-10-08T03:00:00Z"},
    }
    self._sweep(event_name="push", payload=payload)
    self.assertIn(7, self.api.pr_comments)
    body = self.api.pr_comments[7][0]["body"]
    self.assertIn("@carol", body)
    self.assertIn("2026-10-08T03:00:00Z", body)
    activity_calls = [
      c for c in self.api.calls if "/activity?" in c[1] and "with-pr" in c[1]
    ]
    self.assertEqual(activity_calls, [])

  def test_notification_dedupe_via_marker(self) -> None:
    stray_refs.fetch_snapshot(self.git_dir, None)
    plan = [
      m
      for m in stray_refs.plan_migrations(self.git_dir, None)
      if m.refname == "refs/heads/stray"
    ]
    self.assertEqual(len(plan), 1)
    m = plan[0]
    self.api.issues.append({
      "number": 1,
      "title": "x",
      "body": f"old\n<!-- stray-ref {m.refname}@{m.sha} -->",
    })
    sent = stray_refs.notify(
      self.api, REPO, m, None, stray_refs.Outcome.ARCHIVED
    )
    self.assertFalse(sent)
    self.assertEqual(len(self.api.issues), 1)

  def test_issue_falls_back_without_assignee(self) -> None:
    self.api.reject_assignees = True
    stats = self._sweep()
    self.assertEqual(stats.notifications_sent, 2)
    self.assertTrue(all("assignees" not in i for i in self.api.issues))

  def test_protected_refuses_when_before_is_old_history(self) -> None:
    self.api.activity["refs/heads/release/1.0"][0]["before"] = self.old_root
    stats = self._sweep()
    self.assertEqual(stats.protected_restored, 0)
    self.assertEqual(stats.refs_failed, 1)
    self.assertEqual(self._refs()["refs/heads/release/1.0"], self.old1)
    escalation = next(
      i for i in self.api.issues if "ACTION NEEDED" in i["title"]
    )
    self.assertIn("nothing has been changed", escalation["body"])

  def test_protected_refuses_plain_push(self) -> None:
    self.api.activity["refs/heads/release/1.0"][0]["activity_type"] = "push"
    stats = self._sweep()
    self.assertEqual(stats.protected_restored, 0)
    self.assertEqual(self._refs()["refs/heads/release/1.0"], self.old1)

  def test_lease_rejects_concurrent_push(self) -> None:
    stray_refs.fetch_snapshot(self.git_dir, None)
    m = next(
      p
      for p in stray_refs.plan_migrations(self.git_dir, None)
      if p.refname == "refs/heads/stray"
    )
    _git(self.remote, "update-ref", "refs/heads/stray", self.old1)
    cmd = stray_refs.push_command(self.git_dir, m, None, dry_run=False)
    self.assertNotEqual(
      subprocess.run(cmd, check=False, capture_output=True).returncode, 0
    )
    self.assertNotIn("refs/archive/stray", self._refs())


if __name__ == "__main__":
  unittest.main()
