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

"""Unit tests for the delete_cancelled_pr_runs CI cleanup tool."""

from collections.abc import Mapping, Sequence
import http
import json
import subprocess
from typing import Final
import unittest
from unittest import mock
import urllib.parse

from ci.tools import delete_cancelled_pr_runs

_REPO: Final[str] = "google-pytorch/torch_tpu"
_BRANCH: Final[str] = "some-pr-branch"
_CURRENT_RUN_ID: Final[int] = 999


def _run(
    run_id: int, status: str = "completed", conclusion: str | None = "cancelled"
) -> Mapping[str, object]:
  """Builds a workflow run entry as returned by the REST runs endpoint."""
  return {"id": run_id, "status": status, "conclusion": conclusion}


def _summary(
    *,
    deleted: Sequence[int] = (),
    failed: Sequence[int] = (),
    unfinished: Sequence[int] = (),
) -> delete_cancelled_pr_runs.CleanupSummary:
  """Builds an expected CleanupSummary, defaulting each category to empty."""
  return delete_cancelled_pr_runs.CleanupSummary(deleted, failed, unfinished)


class _FakeGh:
  """Stands in for `subprocess.run`, serving canned `gh api` responses.

  `list_pages` supplies successive pages of the workflow-runs listing.
  `statuses` maps a run ID to the sequence of statuses successive polls should
  report; the final entry is repeated once exhausted. `missing_runs` are runs
  the API reports as 404. `list_error`, `status_errors` and `delete_errors`
  give the HTTP status that the listing, a status query or a deletion should
  fail with.
  """

  def __init__(
      self,
      *,
      list_pages: Sequence[Sequence[Mapping[str, object]]] = (),
      list_error: http.HTTPStatus | None = None,
      statuses: Mapping[int, Sequence[str]] | None = None,
      missing_runs: Sequence[int] = (),
      status_errors: Mapping[int, http.HTTPStatus] | None = None,
      delete_errors: Mapping[int, http.HTTPStatus] | None = None,
  ):
    self._list_pages = list_pages
    self._list_error = list_error
    self._statuses = dict(statuses or {})
    self._missing_runs = frozenset(missing_runs)
    self._status_errors = dict(status_errors or {})
    self._delete_errors = dict(delete_errors or {})
    self._poll_counts: dict[int, int] = {}
    self.commands: list[tuple[str, ...]] = []

  @property
  def paths(self) -> tuple[str, ...]:
    return tuple(cmd[5] for cmd in self.commands)

  def _reply(
      self, cmd: Sequence[str], status: http.HTTPStatus, body: str = ""
  ) -> subprocess.CompletedProcess[str]:
    """Renders what `gh api --include` writes for `status`."""
    stdout = (
        f"HTTP/2.0 {status.value} {status.phrase}\n"
        "Content-Type: application/json\n"
        f"\n{body}"
    )
    if 200 <= status < 300:
      return subprocess.CompletedProcess(list(cmd), 0, stdout, "")
    stderr = f"gh: {status.phrase} (HTTP {status.value})\n"
    return subprocess.CompletedProcess(list(cmd), 1, stdout, stderr)

  def _run_id(self, path: str) -> int:
    return int(path.rsplit("/", 1)[-1])

  def _status_for(self, run_id: int) -> str:
    seen = self._poll_counts.get(run_id, 0)
    self._poll_counts[run_id] = seen + 1
    sequence = self._statuses.get(run_id, ("completed",))
    return sequence[min(seen, len(sequence) - 1)]

  def __call__(
      self, cmd: Sequence[str], **kwargs: object
  ) -> subprocess.CompletedProcess[str]:
    del kwargs  # Unused; stdout/stderr/text are fixed by the caller.
    self.commands.append(tuple(cmd))
    method, path = cmd[4], cmd[5]

    if method == "DELETE":
      run_id = self._run_id(path)
      if run_id in self._missing_runs:
        return self._reply(cmd, http.HTTPStatus.NOT_FOUND)
      if run_id in self._delete_errors:
        return self._reply(cmd, self._delete_errors[run_id])
      return self._reply(cmd, http.HTTPStatus.NO_CONTENT)

    if "/actions/runs?" in path:
      if self._list_error is not None:
        return self._reply(cmd, self._list_error)
      page = int(urllib.parse.parse_qs(path.split("?", 1)[1])["page"][0])
      runs = self._list_pages[page - 1] if page <= len(self._list_pages) else []
      return self._reply(
          cmd, http.HTTPStatus.OK, json.dumps({"workflow_runs": list(runs)})
      )

    run_id = self._run_id(path)
    if run_id in self._missing_runs:
      return self._reply(cmd, http.HTTPStatus.NOT_FOUND)
    if run_id in self._status_errors:
      return self._reply(cmd, self._status_errors[run_id])
    return self._reply(
        cmd,
        http.HTTPStatus.OK,
        json.dumps({"id": run_id, "status": self._status_for(run_id)}),
    )


class DeleteCancelledPrRunsTest(
    unittest.TestCase  # UNITTEST_OK=CI utility script test without torch_tpu dependencies
):
  """Validates workflow run cleanup for closed pull requests."""

  def setUp(self):
    super().setUp()
    self.slept: list[int] = []
    sleep_patch = mock.patch.object(
        delete_cancelled_pr_runs.time, "sleep", self.slept.append
    )
    sleep_patch.start()
    self.addCleanup(sleep_patch.stop)
    # The module logs at INFO/WARNING/ERROR; keep test output readable.
    logging_patch = mock.patch.object(delete_cancelled_pr_runs, "logging")
    self.logging = logging_patch.start()
    self.addCleanup(logging_patch.stop)

  def _last_logged(self, level: str) -> str:
    """Renders the most recent message logged at `level`, as `logging` would.

    Rendering with `%` also catches a format string whose placeholders do not
    match its arguments, which the `logging` mock would otherwise hide.
    """
    call = getattr(self.logging, level).call_args
    self.assertIsNotNone(call, f"Nothing was logged at level {level!r}.")
    return call.args[0] % call.args[1:]

  def _install(self, fake: _FakeGh) -> _FakeGh:
    patch = mock.patch.object(delete_cancelled_pr_runs.subprocess, "run", fake)
    patch.start()
    self.addCleanup(patch.stop)
    return fake

  def _cleanup(
      self, fake: _FakeGh, **kwargs: object
  ) -> delete_cancelled_pr_runs.CleanupSummary:
    self._install(fake)
    return delete_cancelled_pr_runs.delete_cancelled_pr_runs(
        repository=_REPO,
        head_ref=_BRANCH,
        current_run_id=_CURRENT_RUN_ID,
        **kwargs,
    )

  # --- Argument parsing ---------------------------------------------------

  def test_parse_args_defaults(self):
    parsed = delete_cancelled_pr_runs.parse_args([
        "--head-ref",
        _BRANCH,
        "--current-run-id",
        str(_CURRENT_RUN_ID),
        "--repository",
        _REPO,
    ])
    self.assertEqual(parsed.head_ref, _BRANCH)
    self.assertEqual(parsed.current_run_id, _CURRENT_RUN_ID)
    self.assertEqual(parsed.repository, _REPO)
    self.assertEqual(parsed.poll_attempts, 8)
    self.assertEqual(parsed.poll_interval_seconds, 15)

  # --- Listing target runs ------------------------------------------------

  def test_get_target_run_ids_selects_only_cleanup_candidates(self):
    cases = (
        ("in-flight run", _run(1, status="in_progress", conclusion=None), True),
        ("queued run", _run(2, status="queued", conclusion=None), True),
        ("cancelled run", _run(3, conclusion="cancelled"), True),
        ("successful run", _run(4, conclusion="success"), False),
        ("failed run", _run(5, conclusion="failure"), False),
        ("skipped run", _run(6, conclusion="skipped"), False),
    )
    for name, run, expected in cases:
      with self.subTest(name):
        self._install(_FakeGh(list_pages=([run],)))
        ids = delete_cancelled_pr_runs.get_target_run_ids(
            _REPO, _BRANCH, _CURRENT_RUN_ID
        )
        self.assertEqual(ids, (run["id"],) if expected else ())

  def test_get_target_run_ids_excludes_the_current_run(self):
    self._install(
        _FakeGh(
            list_pages=([_run(_CURRENT_RUN_ID, status="in_progress"), _run(7)],)
        )
    )
    ids = delete_cancelled_pr_runs.get_target_run_ids(
        _REPO, _BRANCH, _CURRENT_RUN_ID
    )
    self.assertEqual(ids, (7,))

  def test_get_target_run_ids_paginates_until_a_short_page(self):
    full_page = [_run(i) for i in range(100)]
    fake = self._install(_FakeGh(list_pages=(full_page, [_run(100)])))
    ids = delete_cancelled_pr_runs.get_target_run_ids(
        _REPO, _BRANCH, _CURRENT_RUN_ID
    )
    self.assertEqual(ids, tuple(range(101)))
    self.assertEqual(len(fake.commands), 2)

  def test_get_target_run_ids_url_encodes_the_branch_name(self):
    fake = self._install(_FakeGh(list_pages=([],)))
    delete_cancelled_pr_runs.get_target_run_ids(
        _REPO, "user/feature-branch", _CURRENT_RUN_ID
    )
    self.assertIn("branch=user%2Ffeature-branch", fake.paths[0])

  def test_get_target_run_ids_raises_when_listing_fails(self):
    self._install(_FakeGh(list_error=http.HTTPStatus.FORBIDDEN))
    with self.assertRaises(RuntimeError):
      delete_cancelled_pr_runs.get_target_run_ids(
          _REPO, _BRANCH, _CURRENT_RUN_ID
      )

  # --- Per-run status and deletion ---------------------------------------

  def test_get_run_status_maps_api_responses(self):
    cases = (
        (
            _FakeGh(statuses={1: ("completed",)}),
            delete_cancelled_pr_runs.RunStatus.COMPLETED,
        ),
        (
            _FakeGh(statuses={1: ("in_progress",)}),
            delete_cancelled_pr_runs.RunStatus.IN_FLIGHT,
        ),
        (
            _FakeGh(missing_runs=(1,)),
            delete_cancelled_pr_runs.RunStatus.NONEXISTENT,
        ),
        (
            _FakeGh(status_errors={1: http.HTTPStatus.FORBIDDEN}),
            delete_cancelled_pr_runs.RunStatus.UNKNOWN,
        ),
    )
    for fake, expected in cases:
      with self.subTest(expected.name):
        self._install(fake)
        self.assertIs(
            delete_cancelled_pr_runs.get_run_status(_REPO, 1), expected
        )

  def test_get_run_status_is_unknown_when_there_was_no_http_reply(self):
    # `gh` can fail before it reaches GitHub at all, in which case there is no
    # status line to read. That is a retry, not a missing run.
    def no_reply(cmd, **kwargs):
      del kwargs
      return subprocess.CompletedProcess(list(cmd), 1, "", "gh: no such host\n")

    self._install(no_reply)
    self.assertIs(
        delete_cancelled_pr_runs.get_run_status(_REPO, 1),
        delete_cancelled_pr_runs.RunStatus.UNKNOWN,
    )

  def test_unreadable_status_line_raises(self):
    # A reply we cannot parse means `gh` changed its output format. Treating
    # that as "no reply" would silently turn every call into a retry, so it
    # propagates instead.
    def garbled(cmd, **kwargs):
      del kwargs
      return subprocess.CompletedProcess(
          list(cmd), 0, "not a status line\n\n{}", ""
      )

    self._install(garbled)
    with self.assertRaises(ValueError):
      delete_cancelled_pr_runs.get_run_status(_REPO, 1)

  def test_delete_run_maps_api_responses(self):
    cases = (
        (_FakeGh(), delete_cancelled_pr_runs.DeletionResult.SUCCESS),
        (
            _FakeGh(missing_runs=(1,)),
            delete_cancelled_pr_runs.DeletionResult.NONEXISTENT,
        ),
        (
            _FakeGh(delete_errors={1: http.HTTPStatus.FORBIDDEN}),
            delete_cancelled_pr_runs.DeletionResult.FAILURE,
        ),
    )
    for fake, expected in cases:
      with self.subTest(expected.name):
        self._install(fake)
        self.assertIs(delete_cancelled_pr_runs.delete_run(_REPO, 1), expected)

  # --- End-to-end cleanup -------------------------------------------------

  def test_deletes_completed_runs(self):
    fake = _FakeGh(list_pages=([_run(1), _run(2)],))
    summary = self._cleanup(fake)
    self.assertEqual(summary, _summary(deleted=(1, 2)))
    self.assertIn(f"repos/{_REPO}/actions/runs/1", fake.paths)
    self.assertIn(f"repos/{_REPO}/actions/runs/2", fake.paths)

  def test_waits_for_in_flight_runs_to_complete(self):
    fake = _FakeGh(
        list_pages=([_run(1, status="in_progress", conclusion=None)],),
        statuses={1: ("in_progress", "in_progress", "completed")},
    )
    summary = self._cleanup(fake, poll_interval_seconds=7)
    self.assertEqual(summary.deleted, (1,))
    self.assertEqual(summary.unfinished, ())
    self.assertEqual(self.slept, [7, 7])

  def test_reports_runs_that_never_complete(self):
    fake = _FakeGh(
        list_pages=(
            [
                _run(2, status="in_progress", conclusion=None),
                _run(1, status="in_progress", conclusion=None),
            ],
        ),
        statuses={1: ("in_progress",), 2: ("in_progress",)},
    )
    summary = self._cleanup(fake, poll_attempts=3)
    self.assertEqual(summary, _summary(unfinished=(1, 2)))
    # Sleeps happen between attempts, not after the last one.
    self.assertEqual(len(self.slept), 2)
    self.assertTrue(
        self._last_logged("warning").endswith(
            f"\nhttps://github.com/{_REPO}/actions/runs/1"
            f"\nhttps://github.com/{_REPO}/actions/runs/2"
        )
    )

  def test_treats_an_already_deleted_run_as_cleaned_up(self):
    fake = _FakeGh(list_pages=([_run(1)],), missing_runs=(1,))
    summary = self._cleanup(fake)
    self.assertEqual(summary, _summary(deleted=(1,)))

  def test_retries_a_run_whose_status_cannot_be_read(self):
    fake = _FakeGh(
        list_pages=([_run(1)],), status_errors={1: http.HTTPStatus.FORBIDDEN}
    )
    summary = self._cleanup(fake, poll_attempts=2)
    self.assertEqual(summary, _summary(unfinished=(1,)))

  def test_records_runs_that_cannot_be_deleted(self):
    fake = _FakeGh(
        list_pages=([_run(1), _run(2)],),
        delete_errors={2: http.HTTPStatus.FORBIDDEN},
    )
    summary = self._cleanup(fake)
    self.assertEqual(summary, _summary(deleted=(1,), failed=(2,)))

  def test_empty_head_ref_makes_no_api_calls(self):
    fake = self._install(_FakeGh())
    summary = delete_cancelled_pr_runs.delete_cancelled_pr_runs(
        repository=_REPO, head_ref="", current_run_id=_CURRENT_RUN_ID
    )
    self.assertEqual(summary, _summary())
    self.assertEqual(fake.commands, [])

  def test_no_candidate_runs_deletes_nothing(self):
    fake = _FakeGh(list_pages=([_run(1, conclusion="success")],))
    summary = self._cleanup(fake)
    self.assertEqual(summary, _summary())
    self.assertEqual(len(fake.commands), 1)

  def test_never_invokes_the_gh_run_subcommands(self):
    # Regression test: `gh run list/view/delete` resolve each run's parent
    # workflow, which 404s for runs from workflows absent from this repository
    # and takes the whole listing down with it. Everything must go through the
    # run-scoped REST endpoints instead.
    fake = _FakeGh(list_pages=([_run(1), _run(2)],))
    self._cleanup(fake)
    for cmd in fake.commands:
      self.assertEqual(cmd[:2], ("gh", "api"))
    self.assertNotIn("workflows", " ".join(path for path in fake.paths))

  # --- Process exit status ------------------------------------------------

  def _main(self, fake: _FakeGh) -> int:
    self._install(fake)
    return delete_cancelled_pr_runs.main([
        "--head-ref",
        _BRANCH,
        "--current-run-id",
        str(_CURRENT_RUN_ID),
        "--repository",
        _REPO,
        "--poll-attempts",
        "2",
        "--poll-interval-seconds",
        "0",
    ])

  def test_main_succeeds_when_everything_is_deleted(self):
    self.assertEqual(self._main(_FakeGh(list_pages=([_run(1)],))), 0)

  def test_main_fails_when_a_delete_fails(self):
    fake = _FakeGh(
        list_pages=([_run(1), _run(2)],),
        delete_errors={
            1: http.HTTPStatus.FORBIDDEN,
            2: http.HTTPStatus.FORBIDDEN,
        },
    )
    self.assertEqual(self._main(fake), 1)
    # `delete_run` logs an error per run first; the summary from `main` is last.
    self.assertEqual(
        self._last_logged("error"),
        "Failed to delete 2 workflow run(s):\n"
        f"https://github.com/{_REPO}/actions/runs/1\n"
        f"https://github.com/{_REPO}/actions/runs/2",
    )

  def test_main_fails_when_listing_fails(self):
    self.assertEqual(
        self._main(_FakeGh(list_error=http.HTTPStatus.FORBIDDEN)), 1
    )

  def test_main_succeeds_when_runs_merely_did_not_finish(self):
    # A run still shutting down is a timing artifact, not an error.
    fake = _FakeGh(
        list_pages=([_run(1, status="in_progress", conclusion=None)],),
        statuses={1: ("in_progress",)},
    )
    self.assertEqual(self._main(fake), 0)


if __name__ == "__main__":
  unittest.main()
