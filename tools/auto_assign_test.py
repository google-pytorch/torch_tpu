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

"""Unit tests for `tools/auto_assign.py`."""

from collections.abc import Mapping, Sequence
import contextlib
import http.server
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
from typing import Any, Final
import unittest
from unittest import mock
import urllib.error

from tools import auto_assign

_PR_EVENT: Final[str] = "pull_request_target"
_ISSUE_EVENT: Final[str] = "issues"
_REPO: Final[str] = "google-pytorch/torch_tpu"


class _FakeGitHubServer:
  """Loopback HTTP fake for GitHub REST GET/POST endpoints."""

  def __init__(
      self,
      *,
      pr_files: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
      pr_reviews: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
      open_pulls: Sequence[Sequence[Mapping[str, Any]]] = ((),),
      open_issues: Sequence[Sequence[Mapping[str, Any]]] = ((),),
      unassignable: set[str] | None = None,
      post_status: int | None = None,
      cross_origin_next_link: str | None = None,
  ) -> None:
    self.pr_files = dict(pr_files or {})
    self.pr_reviews = dict(pr_reviews or {})
    self.open_pulls = list(open_pulls)
    self.open_issues = list(open_issues)
    self.unassignable = {u.lower() for u in unassignable or set()}
    self.post_status = post_status
    self.cross_origin_next_link = cross_origin_next_link
    self.posts: list[tuple[str, dict[str, Any], str]] = []
    self._server: http.server.ThreadingHTTPServer | None = None
    self._thread: threading.Thread | None = None

  @property
  def api_url(self) -> str:
    assert self._server is not None
    return f"http://127.0.0.1:{self._server.server_address[1]}"

  def start(self) -> None:
    fake = self

    class _Handler(http.server.BaseHTTPRequestHandler):

      def log_message(self, format: str, *args: object) -> None:
        del format, args

      def _send_json(
          self,
          status: int,
          body: Any,
          extra_headers: Mapping[str, str] | None = None,
      ) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        for name, value in (extra_headers or {}).items():
          self.send_header(name, value)
        self.end_headers()
        self.wfile.write(raw)

      def _send_page(self, pages: list[Sequence[Mapping[str, Any]]]) -> None:
        match = re.search(r"[?&]page=(\d+)", self.path)
        page = int(match.group(1)) if match else 1
        idx = page - 1
        body = list(pages[idx]) if idx < len(pages) else []
        headers: dict[str, str] = {}
        if fake.cross_origin_next_link and page == 1:
          headers["Link"] = f'<{fake.cross_origin_next_link}>; rel="next"'
        elif page < len(pages):
          base = self.path.split("?")[0]
          next_url = (
              f"{fake.api_url}{base}?state=open&per_page=100&page={page + 1}"
          )
          headers["Link"] = f'<{next_url}>; rel="next"'
        self._send_json(200, body, extra_headers=headers)

      def do_GET(self) -> None:
        if "/reviews" in self.path:
          number = int(self.path.split("/pulls/")[1].split("/")[0])
          self._send_json(200, list(fake.pr_reviews.get(number, ())))
        elif "/files" in self.path:
          number = int(self.path.split("/pulls/")[1].split("/")[0])
          self._send_json(200, list(fake.pr_files.get(number, ())))
        elif "/pulls?" in self.path:
          self._send_page(fake.open_pulls)
        elif "/issues?" in self.path:
          self._send_page(fake.open_issues)
        else:
          self._send_json(404, {"message": f"Unexpected GET: {self.path}"})

      def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        auth = self.headers.get("Authorization", "")
        fake.posts.append((self.path, payload, auth))

        if fake.post_status is not None:
          self._send_json(fake.post_status, {"message": "forced error"})
          return

        targets = payload.get("reviewers") or payload.get("assignees") or ()
        for target in targets:
          if target.lower() in fake.unassignable:
            self._send_json(422, {"message": "Unprocessable Entity"})
            return
        self._send_json(201, {"ok": True})

    self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    self._thread = threading.Thread(
        target=self._server.serve_forever,
        kwargs={"poll_interval": 0.01},
        daemon=True,
    )
    self._thread.start()

  def stop(self) -> None:
    if self._server is not None:
      self._server.shutdown()
      self._server.server_close()
      self._server = None

  def post_paths_and_bodies(self) -> list[tuple[str, dict[str, Any]]]:
    return [(path, body) for path, body, _ in self.posts]


class AutoAssignTest(unittest.TestCase):  # UNITTEST_OK=standalone

  def setUp(self) -> None:
    super().setUp()
    self._temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(self._temp_dir.cleanup)
    self.repo_root = Path(self._temp_dir.name)
    self._write_file(
        "BUILD.bazel",
        """\
# Domain experts for this package:
# root-expert
# jonathan1920
""",
    )
    self._write_file(
        "csrc/ops/BUILD",
        """\
# Domain experts for this package:
# ops-lead
# cbasile-g
# ops-backup
""",
    )
    self._write_file(
        "csrc/ops/group_norm/BUILD",
        """\
# Domain experts for this package:
# zhanyong-wan
# jonathan1920
# ghpvnist
""",
    )
    self._write_file(
        "src/torch_tpu/_internal/pallas/BUILD",
        """\
# Domain experts for this package:
# pallas-expert
""",
    )

  def _write_file(self, rel_path: str, content: str) -> Path:
    path = self.repo_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path

  def _start_server(self, **kwargs: Any) -> _FakeGitHubServer:
    server = _FakeGitHubServer(**kwargs)
    server.start()
    self.addCleanup(server.stop)
    return server

  def _client(self, server: _FakeGitHubServer) -> auto_assign.GitHubClient:
    return auto_assign.GitHubClient(
        api_url=server.api_url, repo=_REPO, token="tok"
    )

  def _config(self, **kwargs: Any) -> auto_assign.ActionConfig:
    defaults: dict[str, Any] = {
        "repo_root": self.repo_root,
        "exclude_logins": frozenset({"cbasile-g", "jonathan1920"}),
    }
    defaults.update(kwargs)
    return auto_assign.ActionConfig(**defaults)

  def test_load_excludes_strips_comments_and_normalizes_case(self) -> None:
    exclude_path = self._write_file(
        ".github/auto_assign_excludes.txt",
        """\
# Team opt-outs
  CBasile-G  # inline comment
jonathan1920

""",
    )

    excludes = auto_assign.load_excludes(exclude_path)

    self.assertEqual(excludes, frozenset({"cbasile-g", "jonathan1920"}))

  def test_find_experts_walks_up_handles_eof_and_guards_traversal(self) -> None:
    self._write_file(
        "docs/BUILD",
        "# Domain experts for this package:\n# docs-lead\n# docs-backup",
    )

    self.assertEqual(
        auto_assign.find_experts(
            self.repo_root, "csrc/ops/group_norm/kernels.cc"
        ),
        ("csrc/ops/group_norm", ("zhanyong-wan", "jonathan1920", "ghpvnist")),
    )
    # New subdirectory without a BUILD file falls back to csrc/ops/BUILD.
    self.assertEqual(
        auto_assign.find_experts(self.repo_root, "csrc/ops/new_op/kernels.cc"),
        ("csrc/ops", ("ops-lead", "cbasile-g", "ops-backup")),
    )
    # Trailing expert line at EOF without a final newline is still matched.
    self.assertEqual(
        auto_assign.find_experts(self.repo_root, "docs/guide.md"),
        ("docs", ("docs-lead", "docs-backup")),
    )
    self.assertIsNone(
        auto_assign.find_experts(self.repo_root, "../../etc/passwd")
    )

  def test_pr_opened_assigns_least_loaded_near_top_expert(self) -> None:
    server = self._start_server(
        pr_files={
            101: [{
                "filename": "csrc/ops/group_norm/a.cc",
                "additions": 20,
                "deletions": 5,
            }]
        },
        open_pulls=(
            [
                {
                    "number": 1,
                    "requested_reviewers": [{"login": "zhanyong-wan"}],
                    "assignees": [],
                },
                {
                    "number": 2,
                    "requested_reviewers": [],
                    "assignees": [{"login": "zhanyong-wan"}],
                },
            ],
        ),
        open_issues=([{"number": 3, "assignees": [{"login": "ghpvnist"}]}],),
    )
    payload = {
        "action": "opened",
        "pull_request": {
            "number": 101,
            "state": "open",
            "draft": False,
            "user": {"login": "external-dev"},
            "body": "Fix group_norm dtype handling",
            "requested_reviewers": [],
            "requested_teams": [],
            "assignees": [],
        },
    }

    assigned = auto_assign.process_event(
        _PR_EVENT, payload, self._client(server), self._config()
    )

    # jonathan1920 is excluded; ghpvnist (score >= 1/3 * top, load 1) beats
    # zhanyong-wan (load 2).
    self.assertEqual(assigned, "ghpvnist")
    self.assertEqual(
        server.post_paths_and_bodies(),
        [(
            f"/repos/{_REPO}/pulls/101/requested_reviewers",
            {"reviewers": ["ghpvnist"]},
        )],
    )

  def test_pr_skips_draft_copybara_bots_or_existing_review(self) -> None:
    cases = [
        ("draft", {"draft": True}),
        ("closed", {"state": "closed"}),
        ("copybara_bot", {"user": {"login": "copybara-service[bot]"}}),
        ("copybara_generated", {"body": "Title\n\nGenerated from cl/12345678"}),
        ("copybara_exports", {"body": "Exports cl/12345678"}),
        (
            "other_bot",
            {"user": {"login": "github-actions[bot]", "type": "Bot"}},
        ),
        ("has_req_reviewer", {"requested_reviewers": [{"login": "someone"}]}),
        ("has_req_team", {"requested_teams": [{"slug": "tpu-team"}]}),
        ("has_assignee", {"assignees": [{"login": "someone"}]}),
    ]
    for label, override in cases:
      with self.subTest(label=label):
        server = self._start_server()
        pr: dict[str, Any] = {
            "number": 50,
            "state": "open",
            "draft": False,
            "user": {"login": "ext"},
            "body": "",
            "requested_reviewers": [],
            "requested_teams": [],
            "assignees": [],
        }
        pr.update(override)

        assigned = auto_assign.process_event(
            _PR_EVENT,
            {"action": "opened", "pull_request": pr},
            self._client(server),
            self._config(),
        )

        self.assertIsNone(assigned)
        self.assertEqual(server.posts, [])

  def test_external_pr_mentioning_cl_in_prose_is_still_assigned(self) -> None:
    server = self._start_server(
        pr_files={
            51: [{
                "filename": "csrc/ops/group_norm/a.cc",
                "additions": 4,
                "deletions": 1,
            }]
        }
    )
    payload = {
        "action": "opened",
        "pull_request": {
            "number": 51,
            "state": "open",
            "draft": False,
            "user": {"login": "external-dev"},
            "body": "Fixes regression introduced by cl/1012238398.",
            "requested_reviewers": [],
            "requested_teams": [],
            "assignees": [],
        },
    }

    assigned = auto_assign.process_event(
        _PR_EVENT, payload, self._client(server), self._config()
    )

    self.assertEqual(assigned, "zhanyong-wan")

  def test_pr_skips_when_already_reviewed(self) -> None:
    server = self._start_server(
        pr_reviews={
            77: [{"state": "COMMENTED", "user": {"login": "other-googler"}}],
            78: [
                {"state": "PENDING", "user": {"login": "other-googler"}},
                {"state": "COMMENTED", "user": {"login": "ext"}},
            ],
        },
        pr_files={
            78: [{
                "filename": "csrc/ops/group_norm/a.cc",
                "additions": 5,
                "deletions": 0,
            }]
        },
    )
    pr_base = {
        "state": "open",
        "draft": False,
        "user": {"login": "ext"},
        "body": "",
        "requested_reviewers": [],
        "requested_teams": [],
        "assignees": [],
    }

    self.assertIsNone(
        auto_assign.process_event(
            _PR_EVENT,
            {
                "action": "ready_for_review",
                "pull_request": {**pr_base, "number": 77},
            },
            self._client(server),
            self._config(),
        )
    )
    self.assertEqual(server.posts, [])

    # Pending or self-authored reviews do not count as an external review.
    assigned = auto_assign.process_event(
        _PR_EVENT,
        {
            "action": "ready_for_review",
            "pull_request": {**pr_base, "number": 78},
        },
        self._client(server),
        self._config(),
    )

    self.assertEqual(assigned, "zhanyong-wan")

  def test_issue_skips_closed_bot_or_already_assigned(self) -> None:
    cases = [
        ("closed", {"state": "closed"}),
        (
            "bot_author",
            {"user": {"login": "copybara-service[bot]", "type": "Bot"}},
        ),
        ("has_assignees", {"assignees": [{"login": "someone"}]}),
        ("has_assignee", {"assignee": {"login": "someone"}}),
    ]
    for label, override in cases:
      with self.subTest(label=label):
        server = self._start_server()
        issue: dict[str, Any] = {
            "number": 201,
            "state": "open",
            "title": "group_norm bug",
            "body": "",
            "user": {"login": "ext"},
            "assignees": [],
            "labels": [],
        }
        issue.update(override)

        assigned = auto_assign.process_event(
            _ISSUE_EVENT,
            {"action": "opened", "issue": issue},
            self._client(server),
            self._config(),
        )

        self.assertIsNone(assigned)
        self.assertEqual(server.posts, [])

  def test_dry_run_skips_post_and_non_422_errors_propagate(self) -> None:
    server = self._start_server(
        pr_files={
            103: [{
                "filename": "csrc/ops/group_norm/a.cc",
                "additions": 5,
                "deletions": 0,
            }]
        }
    )
    payload = {
        "action": "opened",
        "pull_request": {
            "number": 103,
            "state": "open",
            "draft": False,
            "user": {"login": "ext"},
            "body": "",
            "requested_reviewers": [],
            "requested_teams": [],
            "assignees": [],
        },
    }

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      chosen = auto_assign.process_event(
          _PR_EVENT, payload, self._client(server), self._config(dry_run=True)
      )

    self.assertEqual(chosen, "zhanyong-wan")
    self.assertEqual(server.posts, [])
    report = json.loads(stdout.getvalue())
    self.assertEqual(report["chosen"], "zhanyong-wan")
    self.assertEqual(report["open_load"], 0)

    error_server = self._start_server(
        pr_files=server.pr_files,
        post_status=500,
    )
    with self.assertRaises(urllib.error.HTTPError) as ctx:
      auto_assign.process_event(
          _PR_EVENT, payload, self._client(error_server), self._config()
      )
    self.assertEqual(ctx.exception.code, 500)

  def test_dependabot_pr_routes_to_chizhang529_or_falls_back(self) -> None:
    server = self._start_server(
        pr_files={
            88: [{"filename": "MODULE.bazel", "additions": 2, "deletions": 2}]
        }
    )
    payload = {
        "action": "opened",
        "pull_request": {
            "number": 88,
            "state": "open",
            "draft": False,
            "user": {"login": "dependabot[bot]", "type": "Bot"},
            "body": "Bump ruff from 0.1 to 0.2",
            "requested_reviewers": [],
            "requested_teams": [],
            "assignees": [],
        },
    }

    assigned = auto_assign.process_event(
        _PR_EVENT, payload, self._client(server), self._config()
    )

    self.assertEqual(assigned, "chizhang529")
    self.assertEqual(
        server.post_paths_and_bodies(),
        [(
            f"/repos/{_REPO}/pulls/88/requested_reviewers",
            {"reviewers": ["chizhang529"]},
        )],
    )

    fallback_server = self._start_server(pr_files=server.pr_files)
    fallback_assigned = auto_assign.process_event(
        _PR_EVENT,
        payload,
        self._client(fallback_server),
        self._config(exclude_logins=frozenset({"chizhang529", "jonathan1920"})),
    )

    self.assertEqual(fallback_assigned, "root-expert")

  def test_pr_falls_back_on_http_422_unassignable_candidate(self) -> None:
    server = self._start_server(
        pr_files={
            102: [{
                "filename": "csrc/ops/group_norm/a.cc",
                "additions": 5,
                "deletions": 0,
            }]
        },
        unassignable={"zhanyong-wan"},
    )
    payload = {
        "action": "opened",
        "pull_request": {
            "number": 102,
            "state": "open",
            "draft": False,
            "user": {"login": "ext"},
            "body": "",
            "requested_reviewers": [],
            "requested_teams": [],
            "assignees": [],
        },
    }

    assigned = auto_assign.process_event(
        _PR_EVENT, payload, self._client(server), self._config()
    )

    self.assertEqual(assigned, "ghpvnist")
    self.assertEqual(
        [body["reviewers"][0] for _, body, _ in server.posts],
        ["zhanyong-wan", "ghpvnist"],
    )

  def test_issue_opened_infers_op_and_subsystem_paths_or_skips_ambiguous(
      self,
  ) -> None:
    cases = [
        (
            "op_dir_in_title",
            {"title": "group_norm fails on bf16", "body": ""},
            "zhanyong-wan",
        ),
        (
            "pallas_keyword",
            {
                "title": "Kernel lowering crash",
                "body": "Happens in pallas splash",
            },
            "pallas-expert",
        ),
        (
            "generic_op_fallback",
            {"title": "aten::hardshrink wrong result", "body": ""},
            "ops-lead",
        ),
        (
            "ambiguous_stays_unassigned",
            {
                "title": "Question about release timeline",
                "body": "When is 0.3 out?",
            },
            None,
        ),
        (
            "word_with_op_substring_stays_unassigned",
            {
                "title": "Option to build on macOS fails when open",
                "body": "Traceback during setup; please support stopping.",
            },
            None,
        ),
    ]
    for label, issue_fields, expected in cases:
      with self.subTest(label=label):
        server = self._start_server()
        issue = {
            "number": 200,
            "state": "open",
            "user": {"login": "ext"},
            "assignees": [],
            "labels": [],
            **issue_fields,
        }

        assigned = auto_assign.process_event(
            _ISSUE_EVENT,
            {"action": "opened", "issue": issue},
            self._client(server),
            self._config(),
        )

        self.assertEqual(assigned, expected)
        if expected:
          self.assertEqual(
              server.post_paths_and_bodies(),
              [(
                  f"/repos/{_REPO}/issues/200/assignees",
                  {"assignees": [expected]},
              )],
          )
        else:
          self.assertEqual(server.posts, [])

  def test_count_open_loads_paginates_and_ignores_prs_in_issues_endpoint(
      self,
  ) -> None:
    server = self._start_server(
        open_pulls=(
            [
                {
                    "number": 1,
                    "requested_reviewers": [{"login": "Alice"}],
                    "assignees": [{"login": "alice"}],
                },
            ],
            [
                {
                    "number": 2,
                    "requested_reviewers": [{"login": "bob"}],
                    "assignees": [],
                },
            ],
        ),
        open_issues=(
            [
                {
                    "number": 1,
                    "pull_request": {},
                    "assignees": [{"login": "bob"}],
                },
                {"number": 10, "assignees": [{"login": "alice"}]},
            ],
        ),
    )

    loads = auto_assign.count_open_loads(self._client(server), {"alice", "bob"})

    # PR #1 & issue #10 -> alice=2; PR #2 -> bob=1; PR #1 in /issues skipped.
    self.assertEqual(loads, {"alice": 2, "bob": 1})

  def test_get_paginated_rejects_cross_origin_next_link(self) -> None:
    server = self._start_server(
        open_pulls=([{"number": 1}],),
        cross_origin_next_link="https://evil.example/steal-token?page=2",
    )

    with self.assertRaisesRegex(ValueError, "cross-origin"):
      self._client(server).get_paginated(
          f"/repos/{_REPO}/pulls?state=open&per_page=100"
      )

  def test_rank_by_load_preserves_unrounded_score_tiebreak(self) -> None:
    candidates = (
        auto_assign.Candidate(github="lower", score=0.451, packages=("a",)),
        auto_assign.Candidate(github="higher", score=0.454, packages=("b",)),
    )

    ranked = auto_assign.rank_by_load(candidates, {"lower": 1, "higher": 1})

    self.assertEqual([c.github for c in ranked], ["higher", "lower"])

  def test_main_reads_event_and_excludes_from_disk(self) -> None:
    exclude_path = self._write_file(
        ".github/auto_assign_excludes.txt", "zhanyong-wan\n"
    )
    event_path = self._write_file(
        "event.json",
        json.dumps({
            "action": "opened",
            "issue": {
                "number": 303,
                "state": "open",
                "title": "group_norm bug",
                "body": "",
                "user": {"login": "ext"},
                "assignees": [],
                "labels": [],
            },
        }),
    )
    server = self._start_server()
    env = {
        "GITHUB_TOKEN": "tok",
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_REPOSITORY": _REPO,
        "GITHUB_EVENT_NAME": _ISSUE_EVENT,
        "GITHUB_API_URL": server.api_url,
    }

    with mock.patch.dict(os.environ, env, clear=True):
      rc = auto_assign.main([
          "--repo-root",
          str(self.repo_root),
          "--exclude-file",
          str(exclude_path),
          "--exclude",
          "jonathan1920",
      ])

    self.assertEqual(rc, 0)
    self.assertEqual(
        server.post_paths_and_bodies(),
        [(
            f"/repos/{_REPO}/issues/303/assignees",
            {"assignees": ["ghpvnist"]},
        )],
    )

  def test_issue_explicit_path_and_stopword_qualification(self) -> None:
    self._write_file(
        "csrc/ops/equal/BUILD",
        "# Domain experts for this package:\n# equal-expert\n",
    )
    self._write_file(
        "csrc/ops/logical/BUILD",
        "# Domain experts for this package:\n# logical-expert\n",
    )
    self._write_file(
        "csrc/ops/reductions/BUILD",
        "# Domain experts for this package:\n# red-expert\n",
    )
    self._write_file(
        "csrc/common/BUILD",
        "# Domain experts for this package:\n# common-expert\n",
    )
    self._write_file(
        "csrc/eager/BUILD",
        "# Domain experts for this package:\n# eager-expert\n",
    )
    self._write_file(
        "src/torch_tpu/_internal/export/BUILD",
        "# Domain experts for this package:\n# export-expert\n",
    )
    cases = [
        (
            "explicit_path_beats_incidental_eager",
            {
                "title": "[Ops][var] wrong output with bounded dynamic",
                "body": (
                    "See csrc/ops/reductions/reduction_utils.cc; fails "
                    "with `torch.compile` unlike eager."
                ),
            },
            "red-expert",
        ),
        (
            "explicit_common_beats_prose_logical",
            {
                "title": "Pool sized from logical CPUs",
                "body": "See csrc/common/compilation_cache.cc.",
            },
            "common-expert",
        ),
        (
            "qualified_stopword_op_matches",
            {"title": "`torch.equal` returns False on TPU", "body": ""},
            "equal-expert",
        ),
        (
            "qualified_aten_stopword_op_matches",
            {"title": "Incorrect aten::equal gradient", "body": ""},
            "equal-expert",
        ),
        (
            "subsystem_word_boundary_ignores_exporting",
            {
                "title": "Failure while exporting wheels with logical cores",
                "body": "Running eager_mode test failed.",
            },
            None,
        ),
    ]
    for label, issue_fields, expected in cases:
      with self.subTest(label=label):
        server = self._start_server()
        issue = {
            "number": 401,
            "state": "open",
            "user": {"login": "ext"},
            "assignees": [],
            "labels": [],
            **issue_fields,
        }

        assigned = auto_assign.process_event(
            _ISSUE_EVENT,
            {"action": "opened", "issue": issue},
            self._client(server),
            self._config(),
        )

        self.assertEqual(assigned, expected)

  def test_edge_cases_null_fields_events_and_all_422(self) -> None:
    server = self._start_server(
        pr_files={
            104: [{
                "filename": "csrc/ops/group_norm/a.cc",
                "additions": 0,
                "deletions": 0,
            }],
            105: [],
        },
        unassignable={"zhanyong-wan", "ghpvnist"},
    )

    self.assertIsNone(
        auto_assign.process_event(
            "push", {"action": "opened"}, self._client(server), self._config()
        )
    )
    self.assertIsNone(
        auto_assign.process_event(
            _PR_EVENT,
            {"action": "synchronize", "pull_request": {"number": 104}},
            self._client(server),
            self._config(),
        )
    )
    # Null body/user/labels do not crash and ambiguous text stays unassigned.
    self.assertIsNone(
        auto_assign.process_event(
            _ISSUE_EVENT,
            {
                "action": "reopened",
                "issue": {
                    "number": 402,
                    "state": "open",
                    "title": "Build fails",
                    "body": None,
                    "user": None,
                    "assignees": None,
                    "labels": None,
                },
            },
            self._client(server),
            self._config(),
        )
    )
    # Empty PR files list returns None cleanly.
    self.assertIsNone(
        auto_assign.process_event(
            _PR_EVENT,
            {
                "action": "opened",
                "pull_request": {
                    "number": 105,
                    "state": "open",
                    "draft": False,
                    "user": None,
                    "body": None,
                    "requested_reviewers": None,
                    "requested_teams": None,
                    "assignees": None,
                },
            },
            self._client(server),
            self._config(),
        )
    )
    # All candidates failing with HTTP 422 returns None without raising.
    self.assertIsNone(
        auto_assign.process_event(
            "pull_request",
            {
                "action": "opened",
                "pull_request": {
                    "number": 104,
                    "state": "open",
                    "draft": False,
                    "user": {"login": "ext"},
                    "body": "",
                    "requested_reviewers": [],
                    "requested_teams": [],
                    "assignees": [],
                },
            },
            self._client(server),
            self._config(),
        )
    )
    self.assertEqual(
        [body["reviewers"][0] for _, body, _ in server.posts],
        ["zhanyong-wan", "ghpvnist"],
    )

  def test_missing_env_vars_and_bad_repo_root_return_nonzero(self) -> None:
    stderr = io.StringIO()
    with (
        mock.patch.dict(os.environ, {}, clear=True),
        contextlib.redirect_stderr(stderr),
    ):
      rc = auto_assign.main([])

    self.assertEqual(rc, 1)
    self.assertIn("GITHUB_TOKEN", stderr.getvalue())
    self.assertIn("GITHUB_EVENT_PATH", stderr.getvalue())

    bad_root = self.repo_root / "missing"
    env = {
        "GITHUB_TOKEN": "tok",
        "GITHUB_EVENT_PATH": str(self.repo_root / "event.json"),
        "GITHUB_REPOSITORY": _REPO,
        "GITHUB_EVENT_NAME": _ISSUE_EVENT,
    }
    stderr = io.StringIO()
    with (
        mock.patch.dict(os.environ, env, clear=True),
        contextlib.redirect_stderr(stderr),
    ):
      rc = auto_assign.main(["--repo-root", str(bad_root)])

    self.assertEqual(rc, 1)
    self.assertIn("does not exist", stderr.getvalue())

  def test_cli_script_end_to_end_over_loopback_http(self) -> None:
    server = self._start_server(
        pr_files={
            505: [{
                "filename": "csrc/ops/group_norm/a.cc",
                "additions": 10,
                "deletions": 2,
            }]
        }
    )
    exclude_path = self._write_file(
        "excludes.txt", "zhanyong-wan\n# comment\nJonathan1920\n"
    )
    event_path = self._write_file(
        "event.json",
        json.dumps({
            "action": "opened",
            "pull_request": {
                "number": 505,
                "state": "open",
                "draft": False,
                "user": {"login": "ext"},
                "body": "",
                "requested_reviewers": [],
                "requested_teams": [],
                "assignees": [],
            },
        }),
    )
    script = Path(auto_assign.__file__).resolve()
    self.assertTrue(os.access(script, os.X_OK))

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(self.repo_root),
            "--exclude-file",
            str(exclude_path),
        ],
        env={
            "GITHUB_TOKEN": "secret-tok",
            "GITHUB_EVENT_PATH": str(event_path),
            "GITHUB_REPOSITORY": _REPO,
            "GITHUB_EVENT_NAME": _PR_EVENT,
            "GITHUB_API_URL": server.api_url,
        },
        check=False,
        capture_output=True,
        text=True,
    )

    self.assertEqual(result.returncode, 0, msg=result.stderr)
    self.assertIn("Assigned ghpvnist as reviewer on #505.", result.stdout)
    self.assertEqual(
        server.posts,
        [(
            f"/repos/{_REPO}/pulls/505/requested_reviewers",
            {"reviewers": ["ghpvnist"]},
            "Bearer secret-tok",
        )],
    )


if __name__ == "__main__":
  unittest.main()
