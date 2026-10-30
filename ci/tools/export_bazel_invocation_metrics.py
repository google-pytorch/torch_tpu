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

r"""Exports per-invocation Bazel build metrics from BEP JSON files to BigQuery.

Bazel ends every Build Event Protocol (BEP) stream with a `BuildMetrics` event
that summarizes how actions were executed (remote cache hit, remote execution,
local sandbox, ...), the action cache hit rate, phase timings, and network
traffic. This tool flattens the `BuildStarted`, `BuildMetrics`, and
`BuildFinished` events of each BEP file into one row per Bazel invocation,
enriches it with the GitHub Actions run context, and optionally appends the
rows to a BigQuery table with `bq load`.

Example:
  python3 ci/tools/export_bazel_invocation_metrics.py \
    --bep_file="${RUNNER_TEMP}/bazel_bep_build.json" \
    --bep_file="${RUNNER_TEMP}/bazel_bep.json" \
    --output="${RUNNER_TEMP}/bazel_invocations.ndjson" \
    --bq_table=ml-velocity-actions-telemetry:ml_velocity.bazel_invocations
"""

import argparse
from collections.abc import Iterable, Iterator, Mapping, Sequence, Set
import dataclasses
import datetime
import enum
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import types
import typing
from typing import Any, Final

_TIMESTAMP_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<base>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<frac>\d+))?"
    r"(?P<tz>Z|[+-]\d{2}:\d{2})$"
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class RunContext:
  """GitHub Actions context attached to every exported row.

  Attributes:
    repository: `owner/name` of the repository (GITHUB_REPOSITORY).
    workflow_name: Workflow name (GITHUB_WORKFLOW).
    job_key: Job ID in the workflow file (GITHUB_JOB).
    job_name: Display name of the matrix job; `job_key` is shared by all legs of
      a matrix, and two legs may share a runner label and Bazel config.
    run_id: Workflow run ID (GITHUB_RUN_ID).
    run_attempt: Attempt number of the run (GITHUB_RUN_ATTEMPT).
    event_name: Event that triggered the run (GITHUB_EVENT_NAME).
    ref: Git ref of the run (GITHUB_REF).
    head_sha: Commit SHA of the run (GITHUB_SHA).
    runner_label: GitHub Actions runner label.
    bazel_config: Bazel `--config` of the run.
  """

  repository: str | None = None
  workflow_name: str | None = None
  job_key: str | None = None
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


class BqType(enum.StrEnum):
  """BigQuery column types of the exported rows."""

  STRING = "STRING"
  INTEGER = "INTEGER"
  FLOAT = "FLOAT"
  TIMESTAMP = "TIMESTAMP"
  RECORD = "RECORD"


_PYTHON_TO_BQ_TYPE: Final[Mapping[type[Any], BqType]] = types.MappingProxyType({
    str: BqType.STRING,
    int: BqType.INTEGER,
    float: BqType.FLOAT,
})
# Field metadata key overriding the BigQuery type derived from the annotation.
_BQ_TYPE: Final[str] = "bq_type"
# Metadata of `str` fields holding BigQuery TIMESTAMP strings.
_TIMESTAMP_METADATA: Final[Mapping[str, BqType]] = types.MappingProxyType(
    {_BQ_TYPE: BqType.TIMESTAMP}
)

_BQ_PARTITION_FIELD: Final[str] = "invocation_start_time"
_BQ_CLUSTERING_FIELDS: Final[str] = "repository,bazel_command"
_BQ_TIMEOUT_SECONDS: Final[int] = 300

# `BuildMetrics.ActionSummary.RunnerCount` names. Bazel reports actions that
# hit the remote cache as a runner rather than as an execution.
_TOTAL_RUNNER: Final[str] = "total"
_REMOTE_CACHE_HIT_RUNNER: Final[str] = "remote cache hit"
_DISK_CACHE_HIT_RUNNER: Final[str] = "disk cache hit"
_REMOTE_EXECUTION_RUNNER: Final[str] = "remote"
# `execKind` values of runners that execute actions on the CI machine, e.g.
# "processwrapper-sandbox", "linux-sandbox", "local", or "worker".
_LOCAL_EXEC_KINDS: Final[frozenset[str]] = frozenset({"Local", "Worker"})


@dataclasses.dataclass(frozen=True, kw_only=True)
class RunnerCount:
  """Number of actions executed by one Bazel spawn runner.

  Attributes:
    name: Runner name, e.g. "remote cache hit" or "linux-sandbox".
    exec_kind: Runner kind, e.g. "Remote", "Local", or "Worker".
    count: Number of actions the runner executed.
  """

  name: str
  exec_kind: str | None
  count: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class ActionData:
  """Execution stats of all actions sharing one mnemonic.

  Attributes:
    mnemonic: Action mnemonic, e.g. "CppCompile".
    actions_created: Actions created in the analysis phase.
    actions_executed: Actions executed, i.e. not served by any cache.
    user_time_s: User CPU time of the executed actions.
    system_time_s: System CPU time of the executed actions.
  """

  mnemonic: str
  actions_created: int | None
  actions_executed: int | None
  user_time_s: float | None
  system_time_s: float | None


@dataclasses.dataclass(frozen=True, kw_only=True)
class InvocationRow:
  """One exported row: the build metrics of a single Bazel invocation.

  `BQ_SCHEMA` is derived from these fields, so rows and schema cannot diverge.
  Append new fields at the end; `bq_load` adds them to the existing table.

  Attributes:
    invocation_id: Bazel invocation ID (`BuildStarted.uuid`).
    invocation_start_time: When the invocation started.
    bazel_command: Bazel command, e.g. "build" or "test".
    repository: GitHub repository, `owner/name`.
    workflow_name: GitHub Actions workflow name.
    job_key: Job ID in the workflow file.
    job_name: Display name of the matrix job.
    run_id: Workflow run ID.
    run_attempt: Attempt number of the run.
    event_name: Event that triggered the run, e.g. "push" or "schedule".
    ref: Git ref of the run.
    head_sha: Commit SHA of the run.
    runner_label: GitHub Actions runner label.
    bazel_config: `--config` values of the invocation's command line, or the
      job's `RunContext.bazel_config` if it has none.
    bazel_version: Bazel release.
    finish_time: When the invocation finished.
    exit_code_name: Bazel exit code name, e.g. "SUCCESS" or "TESTS_FAILED".
    exit_code: Bazel exit code.
    wall_time_s: Wall time of the invocation.
    cpu_time_s: CPU time of the invocation.
    analysis_phase_s: Duration of the analysis phase.
    execution_phase_s: Duration of the execution phase.
    critical_path_s: Duration of the critical path.
    actions_created: Actions created in the analysis phase.
    actions_executed: Actions executed, i.e. not served by the action cache.
    remote_cache_hit_count: Actions served from the remote cache.
    disk_cache_hit_count: Actions served from the disk cache.
    remote_execution_count: Actions executed remotely.
    local_execution_count: Actions executed on the CI machine (sandboxes and
      persistent workers).
    action_cache_hits: Local action cache hits.
    action_cache_misses: Local action cache misses.
    bytes_sent: Network bytes sent by the machine during the invocation.
    bytes_recv: Network bytes received by the machine during the invocation.
    output_artifacts_seen_bytes: Size of all output artifacts seen.
    output_artifacts_from_action_cache_bytes: Size of the output artifacts
      served from the action cache.
    runner_counts: Actions per spawn runner.
    action_data: Action stats per mnemonic.

  Metrics are None if the BEP stream has no `BuildMetrics` event.
  """

  invocation_id: str | None
  invocation_start_time: str | None = dataclasses.field(
      metadata=_TIMESTAMP_METADATA
  )
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
  bazel_version: str | None
  finish_time: str | None = dataclasses.field(metadata=_TIMESTAMP_METADATA)
  exit_code_name: str | None
  exit_code: int | None
  wall_time_s: float | None
  cpu_time_s: float | None
  analysis_phase_s: float | None
  execution_phase_s: float | None
  critical_path_s: float | None
  actions_created: int | None
  actions_executed: int | None
  remote_cache_hit_count: int | None
  disk_cache_hit_count: int | None
  remote_execution_count: int | None
  local_execution_count: int | None
  action_cache_hits: int | None
  action_cache_misses: int | None
  bytes_sent: int | None
  bytes_recv: int | None
  output_artifacts_seen_bytes: int | None
  output_artifacts_from_action_cache_bytes: int | None
  runner_counts: tuple[RunnerCount, ...]
  action_data: tuple[ActionData, ...]


def _bq_type(annotation: Any) -> BqType:
  """Returns the BigQuery type of a `T` or `T | None` field annotation."""
  if isinstance(annotation, types.UnionType):
    (annotation,) = (
        arg for arg in typing.get_args(annotation) if arg is not types.NoneType
    )
  return _PYTHON_TO_BQ_TYPE[annotation]


def _bq_schema(row_type: type[Any]) -> tuple[dict[str, Any], ...]:
  """Returns the BigQuery JSON schema of a row dataclass, in field order.

  `tuple[Dataclass, ...]` fields become REPEATED RECORD columns. All other
  columns are NULLABLE, so the schema matches tables created before a field was
  added.

  Args:
    row_type: Dataclass whose fields are the table columns.

  Returns:
    The schema in the JSON format accepted by `bq load --schema`.
  """
  type_hints = typing.get_type_hints(row_type)
  schema = []
  for field in dataclasses.fields(row_type):
    annotation = type_hints[field.name]
    if typing.get_origin(annotation) is tuple:
      item_type, _ = typing.get_args(annotation)
      schema.append({
          "name": field.name,
          "type": BqType.RECORD,
          "mode": "REPEATED",
          "fields": _bq_schema(item_type),
      })
      continue
    schema.append({
        "name": field.name,
        "type": field.metadata.get(_BQ_TYPE) or _bq_type(annotation),
        "mode": "NULLABLE",
    })
  return tuple(schema)


# A JSON schema is required because the inline `bq` schema syntax cannot express
# the REPEATED RECORD columns.
BQ_SCHEMA: Final[Sequence[Mapping[str, Any]]] = _bq_schema(InvocationRow)


def _int_or_none(value: Any) -> int | None:
  """Parses a protobuf JSON int64 (emitted as a string) or int32."""
  return None if value is None else int(value)


def _millis_to_seconds(value: Any) -> float | None:
  """Converts a protobuf JSON int64 milliseconds value to seconds."""
  return None if value is None else int(value) / 1000.0


def _duration_or_none(msg: Mapping[str, Any], field: str) -> float | None:
  """Returns a Duration field in seconds, or None if unset."""
  return parse_duration_seconds(msg, field, f"{field}Millis")


def _parse_runner_counts(
    action_summary: Mapping[str, Any],
) -> tuple[RunnerCount, ...]:
  """Returns the per-runner action counts, excluding the "total" pseudo-runner.

  "total" equals `actionsExecuted` and would double count every action when
  summed with the real runners.

  Args:
    action_summary: The `BuildMetrics.actionSummary` message.

  Returns:
    The runner counts in BEP order.
  """
  return tuple(
      RunnerCount(
          name=runner.get("name", ""),
          exec_kind=runner.get("execKind"),
          count=int(runner.get("count", 0)),
      )
      for runner in action_summary.get("runnerCount", [])
      if runner.get("name") != _TOTAL_RUNNER
  )


def _parse_action_data(
    action_summary: Mapping[str, Any],
) -> tuple[ActionData, ...]:
  """Returns the per-mnemonic action stats of a `BuildMetrics.actionSummary`."""
  return tuple(
      ActionData(
          mnemonic=stats.get("mnemonic", ""),
          actions_created=_int_or_none(stats.get("actionsCreated")),
          actions_executed=_int_or_none(stats.get("actionsExecuted")),
          user_time_s=_duration_or_none(stats, "userTime"),
          system_time_s=_duration_or_none(stats, "systemTime"),
      )
      for stats in action_summary.get("actionData", [])
  )


def _sum_runner_counts(
    runner_counts: Sequence[RunnerCount] | None,
    *,
    name: str | None = None,
    exec_kinds: Set[str] = frozenset(),
) -> int | None:
  """Sums the counts of runners matching `name` or one of `exec_kinds`.

  Args:
    runner_counts: Runner counts of the invocation, or None if the BEP stream
      has no `BuildMetrics` event.
    name: Runner name to match, e.g. "remote cache hit".
    exec_kinds: Runner `execKind` values to match, e.g. {"Local"}.

  Returns:
    The summed count, or None if `runner_counts` is None.
  """
  if runner_counts is None:
    return None
  return sum(
      runner.count
      for runner in runner_counts
      if runner.name == name or runner.exec_kind in exec_kinds
  )


def _cache_count(cache_stats: Mapping[str, Any], field: str) -> int | None:
  """Returns an `ActionCacheStatistics` counter, or None if not reported."""
  if not cache_stats:
    return None
  # proto3 JSON omits zero-valued fields, so a missing counter means zero.
  return int(cache_stats.get(field, 0))


def _artifact_bytes(artifacts: Mapping[str, Any], field: str) -> int | None:
  """Returns the size of an `ArtifactMetrics` file set, or None if absent."""
  if not artifacts:
    return None
  return int(artifacts.get(field, {}).get("sizeInBytes", 0))


@dataclasses.dataclass
class _InvocationEvents:
  """The invocation-level events of one BEP stream."""

  started: Mapping[str, Any] | None = None
  options: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  finished: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  metrics: Mapping[str, Any] | None = None


def _collect_invocation_events(
    events: Iterable[Mapping[str, Any]],
) -> _InvocationEvents:
  """Collects the invocation-level events of a BEP stream."""
  collected = _InvocationEvents()
  for event in events:
    event_id = event.get("id", {})
    if "started" in event_id:
      collected.started = event.get("started", {})
    elif "optionsParsed" in event_id:
      collected.options = event.get("optionsParsed", {})
    elif "buildFinished" in event_id:
      collected.finished = event.get("finished", {})
    elif "buildMetrics" in event_id:
      collected.metrics = event.get("buildMetrics", {})
  return collected


def _bazel_config(options: Mapping[str, Any]) -> str | None:
  """Returns the `--config` values of an `OptionsParsed` event, if any.

  Args:
    options: The `OptionsParsed` event payload.

  Returns:
    The comma-separated `--config` values passed on the command line, or None if
    there are none.
  """
  configs = [
      arg.removeprefix("--config=")
      for arg in options.get("explicitCmdLine", [])
      if arg.startswith("--config=")
  ]
  return ",".join(configs) or None


def extract_invocation_row(
    events: Iterable[Mapping[str, Any]], context: RunContext
) -> InvocationRow | None:
  """Flattens the invocation-level events of one BEP stream into a row.

  A stream cut short (e.g. by job cancellation) has no `BuildMetrics` event;
  its row is still emitted, with NULL metrics, so that the invocation counts.

  Args:
    events: Parsed BEP JSON events of a single Bazel invocation.
    context: GitHub Actions context attached to the row. Its `bazel_config` is
      used only if the stream has no `--config` on its command line.

  Returns:
    The row, or None if the stream has no `BuildStarted` event.
  """
  collected = _collect_invocation_events(events)
  if collected.started is None:
    return None
  started = collected.started
  finished = collected.finished
  metrics = collected.metrics or {}
  action_summary = metrics.get("actionSummary", {})
  timing = metrics.get("timingMetrics", {})
  network = metrics.get("networkMetrics", {}).get("systemNetworkStats", {})
  artifacts = metrics.get("artifactMetrics", {})
  cache_stats = action_summary.get("actionCacheStatistics", {})
  exit_code = finished.get("exitCode", {})
  runner_counts = _parse_runner_counts(action_summary)
  # Runner-derived counts are NULL, not zero, when `BuildMetrics` is missing.
  metric_runners = runner_counts if collected.metrics is not None else None

  return InvocationRow(
      invocation_id=started.get("uuid"),
      invocation_start_time=normalize_timestamp(started.get("startTime")),
      bazel_command=started.get("command"),
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
      # One job can run Bazel with several configs (e.g. newest-deps tests).
      bazel_config=_bazel_config(collected.options) or context.bazel_config,
      bazel_version=started.get("buildToolVersion"),
      finish_time=normalize_timestamp(finished.get("finishTime")),
      exit_code_name=exit_code.get("name"),
      # `ExitCode.code` is omitted for SUCCESS (0) by the proto3 JSON printer.
      exit_code=int(exit_code.get("code", 0)) if exit_code else None,
      wall_time_s=_millis_to_seconds(timing.get("wallTimeInMs")),
      cpu_time_s=_millis_to_seconds(timing.get("cpuTimeInMs")),
      analysis_phase_s=_millis_to_seconds(timing.get("analysisPhaseTimeInMs")),
      execution_phase_s=_millis_to_seconds(
          timing.get("executionPhaseTimeInMs")
      ),
      critical_path_s=_duration_or_none(timing, "criticalPathTime"),
      actions_created=_int_or_none(action_summary.get("actionsCreated")),
      actions_executed=_int_or_none(action_summary.get("actionsExecuted")),
      remote_cache_hit_count=_sum_runner_counts(
          metric_runners, name=_REMOTE_CACHE_HIT_RUNNER
      ),
      disk_cache_hit_count=_sum_runner_counts(
          metric_runners, name=_DISK_CACHE_HIT_RUNNER
      ),
      remote_execution_count=_sum_runner_counts(
          metric_runners, name=_REMOTE_EXECUTION_RUNNER
      ),
      local_execution_count=_sum_runner_counts(
          metric_runners, exec_kinds=_LOCAL_EXEC_KINDS
      ),
      action_cache_hits=_cache_count(cache_stats, "hits"),
      action_cache_misses=_cache_count(cache_stats, "misses"),
      bytes_sent=_int_or_none(network.get("bytesSent")),
      bytes_recv=_int_or_none(network.get("bytesRecv")),
      output_artifacts_seen_bytes=_artifact_bytes(
          artifacts, "outputArtifactsSeen"
      ),
      output_artifacts_from_action_cache_bytes=_artifact_bytes(
          artifacts, "outputArtifactsFromActionCache"
      ),
      runner_counts=runner_counts,
      action_data=_parse_action_data(action_summary),
  )


def write_ndjson(rows: Sequence[InvocationRow], output: pathlib.Path) -> None:
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
  with tempfile.NamedTemporaryFile(
      "w", suffix=".json", encoding="utf-8"
  ) as schema_file:
    json.dump(BQ_SCHEMA, schema_file)
    schema_file.flush()
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
        schema_file.name,
    ])
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=_BQ_TIMEOUT_SECONDS,
    )


def _read_rows(
    bep_files: Iterable[pathlib.Path], context: RunContext
) -> Sequence[InvocationRow]:
  """Reads one row per BEP file.

  Args:
    bep_files: Files written by `--build_event_json_file`, one per invocation.
    context: GitHub Actions context attached to the rows.

  Returns:
    The rows of the files that exist and have a `BuildStarted` event, in
    `bep_files` order.
  """
  rows = []
  for bep_file in bep_files:
    if not bep_file.is_file():
      # The Bazel step may have been skipped or failed before writing the file.
      logging.warning("BEP file %s not found; skipping.", bep_file)
      continue
    with bep_file.open(encoding="utf-8") as f:
      row = extract_invocation_row(iter_bep_events(f), context)
    if row is None:
      logging.warning("BEP file %s has no BuildStarted event.", bep_file)
      continue
    rows.append(row)
  return rows


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
  """Parses the command-line arguments."""
  parser = argparse.ArgumentParser(
      description=(__doc__ or "").partition("\n")[0]
  )
  parser.add_argument(
      "--bep_file",
      type=pathlib.Path,
      action="append",
      required=True,
      help=(
          "Path written by `bazel --build_event_json_file=<path>`. Repeat for"
          " each Bazel invocation of the job; missing files are skipped."
      ),
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
      help="Display name of the GitHub Actions matrix job.",
  )
  parser.add_argument(
      "--runner_label", default=None, help="GitHub Actions runner label."
  )
  parser.add_argument(
      "--bazel_config", default=None, help="Bazel `--config` of the run."
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
  """Exports the BEP files' invocation metrics and optionally uploads them.

  Args:
    argv: Command-line arguments, excluding the program name.

  Returns:
    The process exit code: 0 if the rows were exported or there was nothing to
    export, 1 if the BigQuery upload failed.
  """
  logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
  args = _parse_args(argv)
  context = RunContext.from_env(
      os.environ,
      job_name=args.job_name,
      runner_label=args.runner_label,
      bazel_config=args.bazel_config,
  )

  rows = _read_rows(args.bep_file, context)
  write_ndjson(rows, args.output)
  logging.info("Wrote %d invocation rows to %s.", len(rows), args.output)

  if not rows or not args.bq_table:
    return 0
  try:
    bq_load(args.output, args.bq_table)
  except subprocess.CalledProcessError as err:
    logging.error(
        "bq load failed: %s\nstdout: %s\nstderr: %s",
        err,
        err.stdout,
        err.stderr,
    )
    return 1
  except subprocess.TimeoutExpired as err:
    # `TimeoutExpired` carries bytes or None output even with `text=True`, and
    # its message already names the timeout, so log the exception itself.
    logging.error("bq load timed out: %s", err)
    return 1
  except FileNotFoundError as err:
    logging.error(
        "`bq` CLI not found (%s); cannot upload to %s.", err, args.bq_table
    )
    return 1
  logging.info("Loaded %d rows into %s.", len(rows), args.bq_table)
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
