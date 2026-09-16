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

"""Deletes cancelled GitHub Actions workflow runs for a closed PR once completed."""

import argparse
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import enum
import http
import json
import logging  # PYTHON_LOGGING_OK=CI utility script running in standard Python environment
import subprocess
import sys
import time
from typing import Final
import urllib.parse

# GitHub's maximum page size for the workflow runs endpoint.
_RUNS_PER_PAGE: Final[int] = 100

# Upper bound on pages fetched, to keep a pathological branch from looping
# forever. 100 pages is far more runs than any real pull request accumulates.
_MAX_RUN_PAGES: Final[int] = 100


class RunStatus(enum.Enum):
  """Cleanup-relevant state of a workflow run."""

  # The run has finished and is ready to be deleted.
  COMPLETED = enum.auto()
  # The run is still queued or executing; it is not deletable yet.
  IN_FLIGHT = enum.auto()
  # The run no longer exists, so there is nothing left to clean up.
  NONEXISTENT = enum.auto()
  # The status could not be determined; the caller should retry.
  UNKNOWN = enum.auto()


class DeletionResult(enum.Enum):
  """Outcome of an attempt to delete a workflow run."""

  # The API accepted the deletion.
  SUCCESS = enum.auto()
  # The run was not there to delete, so there is nothing left to clean up.
  NONEXISTENT = enum.auto()
  # The run is still present and could not be deleted.
  FAILURE = enum.auto()


@dataclasses.dataclass(frozen=True)
class CleanupSummary:
  """Per-run outcomes of a cleanup pass."""

  # Runs that are no longer present, whether this pass deleted them or not.
  deleted: Sequence[int]
  # Runs that exist but could not be deleted. These are hard errors.
  failed: Sequence[int]
  # Runs that never reached a completed status within the polling budget.
  unfinished: Sequence[int]


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description=(
          "Delete cancelled workflow runs for a closed PR once completed."
      )
  )
  parser.add_argument(
      "--head-ref",
      required=True,
      help="The head reference/branch name of the closed pull request.",
  )
  parser.add_argument(
      "--current-run-id",
      type=int,
      required=True,
      help="The run ID of the current cleanup workflow to exclude.",
  )
  parser.add_argument(
      "--repository",
      required=True,
      help="The GitHub repository name in 'owner/repo' format.",
  )
  parser.add_argument(
      "--poll-attempts",
      type=int,
      default=8,
      help=(
          "Maximum polling attempts waiting for runs to reach completed status."
      ),
  )
  parser.add_argument(
      "--poll-interval-seconds",
      type=int,
      default=15,
      help="Seconds to wait between polling attempts.",
  )
  return parser.parse_args(args)


@dataclasses.dataclass(frozen=True)
class _ApiResponse:
  """Result of a single `gh api` invocation.

  Attributes:
    status_code: The HTTP status GitHub replied with, or None if `gh` never got
      a reply (bad credentials, no network, malformed request).
    body: The response body, empty for a 204 and when there was no reply.
    error: Whatever `gh` wrote to stderr, empty on success.
  """

  status_code: int | None
  body: str
  error: str

  @property
  def ok(self) -> bool:
    """Whether GitHub replied with a 2xx."""
    return self.status_code is not None and 200 <= self.status_code < 300

  @property
  def not_found(self) -> bool:
    """Whether GitHub replied 404, meaning the run is not there."""
    return self.status_code == http.HTTPStatus.NOT_FOUND


def _parse_status_line(headers: str) -> int | None:
  """Reads the status code out of a status line like `HTTP/2.0 404 Not Found`.

  Args:
    headers: The header block that `gh api --include` writes ahead of the body,
      or the empty string if `gh` never got a reply at all.

  Returns:
    The status code, or None if there was no reply to read one from.

  Raises:
    ValueError: if `headers` is non-empty but does not begin with a status
      line. That means `gh` has changed its output format, which we would
      rather hear about than quietly treat as a request that got no reply.
  """
  if not headers.strip():
    return None
  fields = headers.split(maxsplit=2)
  if len(fields) >= 2 and fields[0].startswith("HTTP/") and fields[1].isdigit():
    return int(fields[1])
  raise ValueError(
      f"Could not read an HTTP status line from `gh api --include`: {headers!r}"
  )


def _gh_run_api(
    repository: str,
    run_id: int | None = None,
    *,
    method: str = "GET",
    query: Mapping[str, str | int] | None = None,
) -> _ApiResponse:
  """Calls the workflow-runs REST API through `gh api`.

  Only the run-scoped endpoints are reachable from here, by construction: the
  path is always `repos/{repository}/actions/runs`, optionally addressing a
  single run by ID.

  These are preferred over invocations of `gh run` because `gh run` attempts to
  resolve the workflow that generated them, which does not work for org-level
  workflows like `GitHub Actions Scan` (zizmor). Concretely: runs of that
  workflow carry a `workflow_id` that is not in this repository, so
  `gh run view <one of those runs>` fails with `HTTP 404` on
  `/actions/workflows/<workflow_id>`. `gh run list` resolves the parent of every
  run it lists, so one such run fails the entire listing rather than just
  itself, and the cleanup job ends up deleting nothing at all.

  Args:
    repository: Target repository, in `owner/repo` form.
    run_id: Workflow run to address. If omitted, the call targets the collection
      of runs rather than an individual one.
    method: HTTP method to invoke.
    query: Query-string parameters. Values are URL-encoded.

  Returns:
    The HTTP status, response body and any `gh` error output. Callers should
    branch on the status rather than on the error text.
  """
  path = f"repos/{repository}/actions/runs"
  if run_id is not None:
    path += f"/{run_id}"
  if query:
    path += f"?{urllib.parse.urlencode(query)}"

  # `--include` prefixes the body with the response headers. It is the only way
  # to recover the status code: `gh api` otherwise just exits non-zero, leaving
  # the stderr text as the sole clue about what went wrong.
  result = subprocess.run(
      ["gh", "api", "--include", "--method", method, path],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
  )
  headers, _, body = result.stdout.partition("\n\n")
  return _ApiResponse(
      status_code=_parse_status_line(headers),
      body=body,
      error=result.stderr.strip(),
  )


def _run_urls(repository: str, run_ids: Iterable[int]) -> str:
  """Formats workflow runs as browser URLs, one per line.

  Used for the log lines a human may need to act on: the Actions log viewer
  turns each URL into a link, and one per line keeps list brackets and commas
  out of the link text.
  """
  return "\n".join(
      f"https://github.com/{repository}/actions/runs/{run_id}"
      for run_id in sorted(run_ids)
  )


def get_target_run_ids(
    repository: str, head_ref: str, current_run_id: int
) -> tuple[int, ...]:
  """Queries workflow runs to clean up on the branch (in-flight or cancelled).

  Raises:
    RuntimeError: if the runs for `head_ref` could not be listed.
  """
  target_ids: list[int] = []
  for page in range(1, _MAX_RUN_PAGES + 1):
    response = _gh_run_api(
        repository,
        query={
            "branch": head_ref,
            "per_page": _RUNS_PER_PAGE,
            "page": page,
        },
    )
    if not response.ok:
      raise RuntimeError(
          f"Failed to list workflow runs for branch '{head_ref}': "
          f"{response.error}"
      )
    runs = json.loads(response.body).get("workflow_runs", [])
    target_ids.extend(
        run["id"]
        for run in runs
        if run.get("id") != current_run_id
        and (
            run.get("status") != "completed"
            or run.get("conclusion") == "cancelled"
        )
    )
    if len(runs) < _RUNS_PER_PAGE:
      break

  return tuple(target_ids)


def get_run_status(repository: str, run_id: int) -> RunStatus:
  """Fetches the current status of the given run_id."""
  response = _gh_run_api(repository, run_id)
  if response.not_found:
    return RunStatus.NONEXISTENT
  if not response.ok:
    logging.warning(
        "Failed to get status for run %s: %s", run_id, response.error
    )
    return RunStatus.UNKNOWN

  if json.loads(response.body).get("status") == "completed":
    return RunStatus.COMPLETED
  return RunStatus.IN_FLIGHT


def delete_run(repository: str, run_id: int) -> DeletionResult:
  """Deletes the specified workflow run."""
  response = _gh_run_api(repository, run_id, method="DELETE")
  if response.ok:
    return DeletionResult.SUCCESS
  if response.not_found:
    return DeletionResult.NONEXISTENT

  logging.error("Failed to delete run %s: %s", run_id, response.error)
  return DeletionResult.FAILURE


def delete_cancelled_pr_runs(
    repository: str,
    head_ref: str,
    current_run_id: int,
    poll_attempts: int = 8,
    poll_interval_seconds: int = 15,
) -> CleanupSummary:
  """Deletes cancelled workflow runs on head_ref once they complete."""
  if not head_ref:
    logging.info("No head_ref specified; nothing to clean up.")
    return CleanupSummary((), (), ())

  logging.info(
      "Searching for workflow runs to clean up on branch '%s'...", head_ref
  )
  run_ids = get_target_run_ids(repository, head_ref, current_run_id)

  if not run_ids:
    logging.info(
        "No workflow runs found to clean up for branch '%s'.", head_ref
    )
    return CleanupSummary((), (), ())

  logging.info(
      "Found workflow runs to clean up for branch '%s': %s",
      head_ref,
      sorted(run_ids),
  )

  deleted: list[int] = []
  failed: list[int] = []
  remaining = set(run_ids)
  for attempt in range(poll_attempts):
    for run_id in sorted(remaining):
      match get_run_status(repository, run_id):
        case RunStatus.NONEXISTENT:
          logging.info("Run %s is already gone.", run_id)
          deleted.append(run_id)
          remaining.discard(run_id)
        case RunStatus.COMPLETED:
          logging.info("Deleting completed run %s...", run_id)
          result = delete_run(repository, run_id)
          if result is DeletionResult.FAILURE:
            failed.append(run_id)
          else:
            deleted.append(run_id)
          remaining.discard(run_id)
        case RunStatus.IN_FLIGHT | RunStatus.UNKNOWN:
          # Not deletable yet (or not queryable right now); retry next pass.
          pass

    if not remaining:
      break

    if attempt < poll_attempts - 1:
      logging.info(
          "Waiting for remaining runs %s to reach completed status (attempt"
          " %d/%d, sleeping %ds)...",
          sorted(remaining),
          attempt + 1,
          poll_attempts,
          poll_interval_seconds,
      )
      time.sleep(poll_interval_seconds)

  if remaining:
    logging.warning(
        "Did not delete the following runs because they did not finish"
        " shutting down in time:\n%s",
        _run_urls(repository, remaining),
    )

  logging.info("Cleanup complete for branch '%s'.", head_ref)
  return CleanupSummary(
      deleted=tuple(sorted(deleted)),
      failed=tuple(sorted(failed)),
      unfinished=tuple(sorted(remaining)),
  )


def main(args: Sequence[str] | None = None) -> int:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  parsed = parse_args(args)
  try:
    summary = delete_cancelled_pr_runs(
        repository=parsed.repository,
        head_ref=parsed.head_ref,
        current_run_id=parsed.current_run_id,
        poll_attempts=parsed.poll_attempts,
        poll_interval_seconds=parsed.poll_interval_seconds,
    )
  except Exception as e:
    logging.exception("Failed to clean up workflow runs: %s", e)
    return 1

  if summary.failed:
    # If we got an error when requesting that GitHub delete a run, this job's
    # status should be failure, not success. `unfinished` runs are not counted
    # since they just haven't stopped yet.
    logging.error(
        "Failed to delete %d workflow run(s):\n%s",
        len(summary.failed),
        _run_urls(parsed.repository, summary.failed),
    )
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
