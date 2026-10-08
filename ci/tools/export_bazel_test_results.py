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

"""Exports per-target Bazel test results from a BEP JSON file to BigQuery.

Bazel writes its Build Event Protocol (BEP) stream to a local file when invoked
with `--build_event_json_file=<path>`. This is the same stream that is uploaded
to ResultStore through BES, so no ResultStore integration is needed to obtain
test-level telemetry.

This tool flattens the `TestSummary`, `TestResult`, and `TargetComplete` events
into one newline-delimited JSON row per test target, enriches each row with the
GitHub Actions run context, and optionally appends the rows to a BigQuery table
with `bq load`.

Example:
  python3 ci/tools/export_bazel_test_results.py \
    --bep_file="${RUNNER_TEMP}/bazel_bep.json" \
    --output="${RUNNER_TEMP}/bazel_test_results.ndjson" \
    --bq_table=ml-velocity-actions-telemetry:ml_velocity.bazel_test_results
"""

import argparse
from collections.abc import Iterable, Iterator, Mapping, Sequence
import dataclasses
import datetime
import json
import logging  # PYTHON_LOGGING_OK=standalone CI script
import os
import pathlib
import re
import subprocess
import sys
from typing import Any, Final

# BigQuery schema for the exported rows. Keep in sync with `TestTargetRow`.
# Append new columns at the end; `bq_load` adds them to the existing table.
BQ_SCHEMA: Final[str] = ",".join((
    "invocation_id:STRING",
    "invocation_start_time:TIMESTAMP",
    "bazel_command:STRING",
    "repository:STRING",
    "workflow_name:STRING",
    "job_key:STRING",
    "job_name:STRING",
    "run_id:STRING",
    "run_attempt:INTEGER",
    "event_name:STRING",
    "ref:STRING",
    "head_sha:STRING",
    "runner_label:STRING",
    "bazel_config:STRING",
    "target:STRING",
    "overall_status:STRING",
    "test_timeout_s:FLOAT",
    "total_run_count:INTEGER",
    "shard_count:INTEGER",
    "attempt_count:INTEGER",
    "cached_count:INTEGER",
    "is_cached:BOOLEAN",
    "strategy:STRING",
    "total_run_duration_s:FLOAT",
    "max_attempt_duration_s:FLOAT",
    "first_start_time:TIMESTAMP",
    "last_stop_time:TIMESTAMP",
))

# Partitioning on the invocation start time (rather than the test start time)
# keeps cached results, whose test timestamps predate the build, in the
# partition of the run that reported them.
_BQ_PARTITION_FIELD: Final[str] = "invocation_start_time"
_BQ_CLUSTERING_FIELDS: Final[str] = "repository,target"
_BQ_TIMEOUT_SECONDS: Final[int] = 300

# `overall_status` of test targets skipped as incompatible with the target
# platform. Matches both the BEP `Aborted.AbortReason` and Bazel's summary.
_SKIPPED_STATUS: Final[str] = "SKIPPED"
# `overall_status` of a `TestSummary` that omits the field.
_NO_STATUS: Final[str] = "NO_STATUS"

# Matches RFC 3339 timestamps as emitted by the protobuf JSON printer, e.g.
# "2026-09-25T17:02:26.123456789Z" or "2026-09-25T17:02:26Z".
_TIMESTAMP_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<base>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<frac>\d+))?"
    r"(?P<tz>Z|[+-]\d{2}:\d{2})$"
)

# (label, configuration id) of a configured target.
_TargetKey = tuple[str, str]


@dataclasses.dataclass(frozen=True, kw_only=True)
class RunContext:
  """GitHub Actions context attached to every exported row."""

  repository: str | None = None
  workflow_name: str | None = None
  job_key: str | None = None
  # Display name of the matrix job; `job_key` (GITHUB_JOB) is shared by all
  # legs of a matrix, and two legs may share a runner label and Bazel config.
  job_name: str | None = None
  run_id: str | None = None
  run_attempt: int | None = None
  event_name: str | None = None
  ref: str | None = None
  head_sha: str | None = None
  runner_label: str | None = None
  bazel_config: str | None = None

  @classmethod
  def from_env(
      cls,
      env: Mapping[str, str],
      *,
      job_name: str | None = None,
      runner_label: str | None = None,
      bazel_config: str | None = None,
  ) -> "RunContext":
    """Builds the context from GitHub Actions default environment variables.

    Args:
      env: Environment variables, e.g. `os.environ`.
      job_name: Display name of the matrix job.
      runner_label: GitHub Actions runner label.
      bazel_config: Bazel `--config` of the run.

    Returns:
      The context. Fields whose variable is unset are None.
    """
    run_attempt = env.get("GITHUB_RUN_ATTEMPT")
    return cls(
        repository=env.get("GITHUB_REPOSITORY"),
        workflow_name=env.get("GITHUB_WORKFLOW"),
        job_key=env.get("GITHUB_JOB"),
        job_name=job_name,
        run_id=env.get("GITHUB_RUN_ID"),
        run_attempt=int(run_attempt) if run_attempt else None,
        event_name=env.get("GITHUB_EVENT_NAME"),
        ref=env.get("GITHUB_REF"),
        head_sha=env.get("GITHUB_SHA"),
        runner_label=runner_label,
        bazel_config=bazel_config,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class TestTargetRow:
  """One exported row: the summarized result of a single test target."""

  invocation_id: str | None
  invocation_start_time: str | None
  bazel_command: str | None
  repository: str | None
  workflow_name: str | None
  job_key: str | None
  job_name: str | None
  run_id: str | None
  run_attempt: int | None
  event_name: str | None
  ref: str | None
  head_sha: str | None
  runner_label: str | None
  bazel_config: str | None
  target: str
  overall_status: str
  test_timeout_s: float | None
  total_run_count: int
  shard_count: int
  attempt_count: int
  cached_count: int
  is_cached: bool
  strategy: str | None
  total_run_duration_s: float | None
  max_attempt_duration_s: float | None
  first_start_time: str | None
  last_stop_time: str | None


@dataclasses.dataclass
class _AttemptStats:
  """Aggregated stats over all `TestResult` events of one target."""

  max_attempt: int = 0
  cached_count: int = 0
  max_attempt_duration_s: float | None = None
  strategies: set[str] = dataclasses.field(default_factory=set)


def _duration_to_seconds(value: str | float) -> float:
  """Converts a protobuf JSON Duration (e.g. "1.5s") to seconds."""
  return float(str(value).removesuffix("s"))


def parse_duration_seconds(
    msg: Mapping[str, Any], field: str, legacy_millis_field: str
) -> float | None:
  """Returns a protobuf Duration field (e.g. "1.5s") in seconds.

  Falls back to the deprecated `*_millis` field emitted by older Bazel versions.

  Args:
    msg: The BEP JSON message containing the field.
    field: Name of the `google.protobuf.Duration` field.
    legacy_millis_field: Name of the deprecated int64 milliseconds field.

  Returns:
    The duration in seconds, or None if neither field is set.
  """
  if (value := msg.get(field)) is not None:
    return _duration_to_seconds(value)
  if (millis := msg.get(legacy_millis_field)) is not None:
    return int(millis) / 1000.0
  return None


def normalize_timestamp(value: str | None) -> str | None:
  """Normalizes an RFC 3339 timestamp to a BigQuery-loadable UTC string.

  The protobuf JSON printer may emit nanosecond precision, which BigQuery
  rejects, so the fractional part is truncated to microseconds.

  Args:
    value: An RFC 3339 timestamp string, or None.

  Returns:
    A "YYYY-MM-DD HH:MM:SS.ffffff UTC" string, or None if unparsable.
  """
  if not value:
    return None
  match = _TIMESTAMP_RE.match(value)
  if not match:
    logging.warning("Ignoring unparsable timestamp: %r", value)
    return None
  frac = (match["frac"] or "")[:6].ljust(6, "0")
  tz = "+00:00" if match["tz"] == "Z" else match["tz"]
  parsed = datetime.datetime.fromisoformat(f"{match['base']}.{frac}{tz}")
  return parsed.astimezone(datetime.timezone.utc).strftime(
      "%Y-%m-%d %H:%M:%S.%f UTC"
  )


def _parse_timestamp_field(
    msg: Mapping[str, Any], field: str, legacy_millis_field: str
) -> str | None:
  """Returns a Timestamp field, falling back to a legacy epoch-millis field."""
  if (value := msg.get(field)) is not None:
    return normalize_timestamp(value)
  if (millis := msg.get(legacy_millis_field)) is not None:
    parsed = datetime.datetime.fromtimestamp(
        int(millis) / 1000.0, tz=datetime.timezone.utc
    )
    return parsed.strftime("%Y-%m-%d %H:%M:%S.%f UTC")
  return None


def iter_bep_events(lines: Iterable[str]) -> Iterator[Mapping[str, Any]]:
  """Yields parsed BEP events, skipping blank or malformed lines.

  A Bazel process that is killed mid-write (e.g. on job cancellation) can leave
  a truncated final line; that must not discard the events preceding it.

  Args:
    lines: Lines of a file written by `--build_event_json_file`.

  Yields:
    Each BEP event as a parsed JSON object, in file order.
  """
  for line_number, line in enumerate(lines, start=1):
    if not (line := line.strip()):
      continue
    try:
      yield json.loads(line)
    except json.JSONDecodeError:
      logging.warning("Skipping malformed BEP line %d.", line_number)


def _target_key(event_id: Mapping[str, Any]) -> _TargetKey:
  """Returns the (label, configuration id) key of a target-scoped event id."""
  return (
      event_id.get("label", ""),
      event_id.get("configuration", {}).get("id", ""),
  )


@dataclasses.dataclass
class _BepState:
  """The parts of one invocation's BEP stream needed to build rows."""

  invocation_id: str | None = None
  invocation_start_time: str | None = None
  bazel_command: str | None = None
  summaries: dict[_TargetKey, Mapping[str, Any]] = dataclasses.field(
      default_factory=dict
  )
  attempts: dict[_TargetKey, _AttemptStats] = dataclasses.field(
      default_factory=dict
  )
  test_timeouts: dict[_TargetKey, float] = dataclasses.field(
      default_factory=dict
  )
  # Labels of configured test rules; `TargetConfigured.testSize` is only set
  # for tests, which distinguishes skipped tests from skipped non-test targets.
  test_labels: set[str] = dataclasses.field(default_factory=set)
  skipped: set[_TargetKey] = dataclasses.field(default_factory=set)


def _record_test_result(
    stats: _AttemptStats,
    result_id: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
  """Adds one `TestResult` event to its target's attempt stats."""
  execution_info = result.get("executionInfo", {})
  stats.max_attempt = max(stats.max_attempt, result_id.get("attempt", 1))
  if result.get("cachedLocally") or execution_info.get("cachedRemotely"):
    stats.cached_count += 1
  if strategy := execution_info.get("strategy"):
    stats.strategies.add(strategy)
  duration_s = parse_duration_seconds(
      result, "testAttemptDuration", "testAttemptDurationMillis"
  )
  if duration_s is not None:
    stats.max_attempt_duration_s = max(
        stats.max_attempt_duration_s or 0.0, duration_s
    )


def _parse_test_timeout(completed: Mapping[str, Any]) -> float | None:
  """Returns a `TargetComplete` test timeout in seconds, or None if unset."""
  if (timeout := completed.get("testTimeout")) is not None:
    return _duration_to_seconds(timeout)
  # Older Bazel versions only emit the deprecated whole-seconds field.
  if (timeout := completed.get("testTimeoutSeconds")) is not None:
    return float(timeout)
  return None


def _collect_bep_state(events: Iterable[Mapping[str, Any]]) -> _BepState:
  """Collects the invocation and per-target events from a BEP stream."""
  state = _BepState()
  for event in events:
    event_id = event.get("id", {})
    if "started" in event_id:
      started = event.get("started", {})
      state.invocation_id = started.get("uuid")
      state.bazel_command = started.get("command")
      state.invocation_start_time = _parse_timestamp_field(
          started, "startTime", "startTimeMillis"
      )
    elif "targetConfigured" in event_id:
      if event.get("configured", {}).get("testSize"):
        state.test_labels.add(event_id["targetConfigured"].get("label", ""))
    elif "testSummary" in event_id:
      key = _target_key(event_id["testSummary"])
      state.summaries[key] = event.get("testSummary", {})
    elif "testResult" in event_id:
      result_id = event_id["testResult"]
      stats = state.attempts.setdefault(_target_key(result_id), _AttemptStats())
      _record_test_result(stats, result_id, event.get("testResult", {}))
    elif "targetCompleted" in event_id:
      key = _target_key(event_id["targetCompleted"])
      # Targets incompatible with the target platform (`target_compatible_with`)
      # complete with an `aborted` payload instead of a `TestSummary`.
      if event.get("aborted", {}).get("reason") == _SKIPPED_STATUS:
        state.skipped.add(key)
        continue
      timeout_s = _parse_test_timeout(event.get("completed", {}))
      if timeout_s is not None:
        state.test_timeouts[key] = timeout_s
  return state


def _build_row(
    key: _TargetKey,
    summary: Mapping[str, Any],
    state: _BepState,
    context: RunContext,
) -> TestTargetRow:
  """Builds the row of one test target from its `TestSummary`."""
  label, _ = key
  stats = state.attempts.get(key, _AttemptStats())
  total_run_count = int(summary.get("totalRunCount", 0))
  cached_count = int(summary.get("totalNumCached", stats.cached_count))
  return TestTargetRow(
      invocation_id=state.invocation_id,
      invocation_start_time=state.invocation_start_time,
      bazel_command=state.bazel_command,
      repository=context.repository,
      workflow_name=context.workflow_name,
      job_key=context.job_key,
      job_name=context.job_name,
      run_id=context.run_id,
      run_attempt=context.run_attempt,
      event_name=context.event_name,
      ref=context.ref,
      head_sha=context.head_sha,
      runner_label=context.runner_label,
      bazel_config=context.bazel_config,
      target=label,
      overall_status=summary.get("overallStatus", _NO_STATUS),
      test_timeout_s=state.test_timeouts.get(key),
      total_run_count=total_run_count,
      shard_count=int(summary.get("shardCount", 0)),
      attempt_count=int(summary.get("attemptCount", stats.max_attempt)),
      cached_count=cached_count,
      is_cached=total_run_count > 0 and cached_count >= total_run_count,
      strategy=",".join(sorted(stats.strategies)) or None,
      total_run_duration_s=parse_duration_seconds(
          summary, "totalRunDuration", "totalRunDurationMillis"
      ),
      max_attempt_duration_s=stats.max_attempt_duration_s,
      first_start_time=_parse_timestamp_field(
          summary, "firstStartTime", "firstStartTimeMillis"
      ),
      last_stop_time=_parse_timestamp_field(
          summary, "lastStopTime", "lastStopTimeMillis"
      ),
  )


def extract_rows(
    events: Iterable[Mapping[str, Any]], context: RunContext
) -> list[TestTargetRow]:
  """Flattens BEP events into one row per summarized test target.

  Test targets skipped as incompatible with the target platform have no
  `TestSummary`; they are emitted with `overall_status` "SKIPPED" and no runs.

  Args:
    events: Parsed BEP JSON events.
    context: GitHub Actions context attached to every row.

  Returns:
    Rows ordered by target label.
  """
  state = _collect_bep_state(events)
  summaries = dict(state.summaries)
  for key in state.skipped:
    label, _ = key
    if label in state.test_labels:
      summaries.setdefault(key, {"overallStatus": _SKIPPED_STATUS})
  return [
      _build_row(key, summaries[key], state, context)
      for key in sorted(summaries)
  ]


def write_ndjson(rows: Sequence[TestTargetRow], output: pathlib.Path) -> None:
  """Writes rows to `output` as newline-delimited JSON.

  Args:
    rows: Rows to write, one JSON object per line.
    output: Path of the file to create or overwrite.
  """
  with output.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(dataclasses.asdict(row), sort_keys=True))
      f.write("\n")


def bq_load(ndjson_path: pathlib.Path, bq_table: str) -> None:
  """Appends the NDJSON file to `bq_table`, creating the table if needed.

  New NULLABLE columns in `BQ_SCHEMA` are added to an existing table.

  Args:
    ndjson_path: Path to the newline-delimited JSON file to load.
    bq_table: Destination table as `project:dataset.table`.

  Raises:
    subprocess.CalledProcessError: If `bq load` fails.
    subprocess.TimeoutExpired: If `bq load` does not finish in time.
    FileNotFoundError: If the `bq` CLI is not installed.
  """
  project_id, _, table = bq_table.rpartition(":")
  command = ["bq"]
  if project_id:
    command.append(f"--project_id={project_id}")
  command.extend([
      "load",
      "--source_format=NEWLINE_DELIMITED_JSON",
      "--schema_update_option=ALLOW_FIELD_ADDITION",
      f"--time_partitioning_field={_BQ_PARTITION_FIELD}",
      f"--clustering_fields={_BQ_CLUSTERING_FIELDS}",
      table,
      str(ndjson_path),
      BQ_SCHEMA,
  ])
  subprocess.run(
      command,
      check=True,
      capture_output=True,
      text=True,
      timeout=_BQ_TIMEOUT_SECONDS,
  )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(__doc__ or "").partition("\n")[0]
  )
  parser.add_argument(
      "--bep_file",
      type=pathlib.Path,
      required=True,
      help="Path written by `bazel test --build_event_json_file=<path>`.",
  )
  parser.add_argument(
      "--output",
      type=pathlib.Path,
      required=True,
      help="Path of the newline-delimited JSON file to write.",
  )
  parser.add_argument(
      "--bq_table",
      default=None,
      help=(
          "Optional BigQuery table (`project:dataset.table`) to append the rows"
          " to. When unset, only the NDJSON file is written."
      ),
  )
  parser.add_argument(
      "--job_name",
      default=None,
      help=(
          "Display name of the GitHub Actions matrix job, e.g. `TPU v5 (newest"
          " deps)`. Distinguishes matrix legs that share a runner and config."
      ),
  )
  parser.add_argument(
      "--runner_label", default=None, help="GitHub Actions runner label."
  )
  parser.add_argument(
      "--bazel_config", default=None, help="Bazel `--config` of the run."
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
  """Exports the BEP file's test results and optionally uploads them.

  Args:
    argv: Command-line arguments, excluding the program name.

  Returns:
    The process exit code: 0 if the rows were exported or there was nothing to
    export, 1 if the BigQuery upload failed.
  """
  logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
  args = _parse_args(argv)

  if not args.bep_file.is_file():
    # Bazel may exit before writing the file, e.g. on a flag parsing error.
    logging.warning("BEP file %s not found; nothing to export.", args.bep_file)
    return 0

  with args.bep_file.open(encoding="utf-8") as f:
    rows = extract_rows(
        iter_bep_events(f),
        RunContext.from_env(
            os.environ,
            job_name=args.job_name,
            runner_label=args.runner_label,
            bazel_config=args.bazel_config,
        ),
    )
  write_ndjson(rows, args.output)
  logging.info("Wrote %d test target rows to %s.", len(rows), args.output)

  if not rows or not args.bq_table:
    return 0
  try:
    bq_load(args.output, args.bq_table)
  except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as err:
    logging.error(
        "bq load failed:\nstdout: %s\nstderr: %s", err.stdout, err.stderr
    )
    return 1
  except FileNotFoundError:
    logging.error("`bq` CLI not found; cannot upload to %s.", args.bq_table)
    return 1
  logging.info("Loaded %d rows into %s.", len(rows), args.bq_table)
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
