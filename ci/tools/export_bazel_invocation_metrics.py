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

"""Exports per-invocation Bazel build metrics from BEP JSON files to BigQuery.

Bazel ends every Build Event Protocol (BEP) stream with a `BuildMetrics` event
that summarizes how actions were executed (remote cache hit, remote execution,
local sandbox, ...), the action cache hit rate, phase timings, and network
traffic. This tool flattens the `BuildStarted`, `BuildMetrics`, and
`BuildFinished` events of each BEP file into one row per Bazel invocation,
enriches it with the GitHub Actions run context, and optionally appends the
rows to a BigQuery table with `bq load`.

Run it from the repository root as a module so that it can import the shared
BEP helpers:

  python3 -m ci.tools.export_bazel_invocation_metrics \
    --bep_file="${RUNNER_TEMP}/bazel_bep_build.json" \
    --bep_file="${RUNNER_TEMP}/bazel_bep.json" \
    --output="${RUNNER_TEMP}/bazel_invocations.ndjson" \
    --bq_table=ml-velocity-actions-telemetry:ml_velocity.bazel_invocations
"""

import argparse
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import json
import logging  # PYTHON_LOGGING_OK=standalone CI script
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any, Final

from ci.tools import export_bazel_test_results as bep


def _field(name: str, field_type: str) -> dict[str, Any]:
  """Returns a NULLABLE BigQuery JSON schema field."""
  return {"name": name, "type": field_type, "mode": "NULLABLE"}


def _repeated(name: str, fields: Sequence[dict[str, Any]]) -> dict[str, Any]:
  """Returns a REPEATED RECORD BigQuery JSON schema field."""
  return {
      "name": name,
      "type": "RECORD",
      "mode": "REPEATED",
      "fields": list(fields),
  }


# BigQuery schema for the exported rows. Keep in sync with `InvocationRow`.
# Append new columns at the end; `bq_load` adds them to the existing table.
# A JSON schema is required because the inline `bq` schema syntax cannot
# express the REPEATED RECORD columns.
BQ_SCHEMA: Final[list[dict[str, Any]]] = [
    _field("invocation_id", "STRING"),
    _field("invocation_start_time", "TIMESTAMP"),
    _field("bazel_command", "STRING"),
    _field("repository", "STRING"),
    _field("workflow_name", "STRING"),
    _field("job_key", "STRING"),
    _field("job_name", "STRING"),
    _field("run_id", "STRING"),
    _field("run_attempt", "INTEGER"),
    _field("event_name", "STRING"),
    _field("ref", "STRING"),
    _field("head_sha", "STRING"),
    _field("runner_label", "STRING"),
    _field("bazel_config", "STRING"),
    _field("bazel_version", "STRING"),
    _field("finish_time", "TIMESTAMP"),
    _field("exit_code_name", "STRING"),
    _field("exit_code", "INTEGER"),
    _field("wall_time_s", "FLOAT"),
    _field("cpu_time_s", "FLOAT"),
    _field("analysis_phase_s", "FLOAT"),
    _field("execution_phase_s", "FLOAT"),
    _field("critical_path_s", "FLOAT"),
    _field("actions_created", "INTEGER"),
    _field("actions_executed", "INTEGER"),
    _field("remote_cache_hit_count", "INTEGER"),
    _field("disk_cache_hit_count", "INTEGER"),
    _field("remote_execution_count", "INTEGER"),
    _field("local_execution_count", "INTEGER"),
    _field("action_cache_hits", "INTEGER"),
    _field("action_cache_misses", "INTEGER"),
    _field("bytes_sent", "INTEGER"),
    _field("bytes_recv", "INTEGER"),
    _field("output_artifacts_seen_bytes", "INTEGER"),
    _field("output_artifacts_from_action_cache_bytes", "INTEGER"),
    _repeated(
        "runner_counts",
        (
            _field("name", "STRING"),
            _field("exec_kind", "STRING"),
            _field("count", "INTEGER"),
        ),
    ),
    _repeated(
        "action_data",
        (
            _field("mnemonic", "STRING"),
            _field("actions_created", "INTEGER"),
            _field("actions_executed", "INTEGER"),
            _field("user_time_s", "FLOAT"),
            _field("system_time_s", "FLOAT"),
        ),
    ),
]

_BQ_PARTITION_FIELD: Final[str] = "invocation_start_time"
_BQ_CLUSTERING_FIELDS: Final[str] = "repository,bazel_command"
_BQ_TIMEOUT_SECONDS: Final[int] = 300

# `BuildMetrics.ActionSummary.RunnerCount` names. Bazel reports actions that
# hit the remote cache as a runner rather than as an execution.
_REMOTE_CACHE_HIT_RUNNER: Final[str] = "remote cache hit"
_DISK_CACHE_HIT_RUNNER: Final[str] = "disk cache hit"
_REMOTE_EXECUTION_RUNNER: Final[str] = "remote"
# `execKind` values of runners that execute actions on the CI machine, e.g.
# "processwrapper-sandbox", "linux-sandbox", "local", or "worker".
_LOCAL_EXEC_KINDS: Final[frozenset[str]] = frozenset({"Local", "Worker"})


@dataclasses.dataclass(frozen=True, kw_only=True)
class RunnerCount:
  """Number of actions executed by one Bazel spawn runner."""

  name: str
  exec_kind: str | None
  count: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class ActionData:
  """Execution stats of all actions sharing one mnemonic."""

  mnemonic: str
  actions_created: int | None
  actions_executed: int | None
  user_time_s: float | None
  system_time_s: float | None


@dataclasses.dataclass(frozen=True, kw_only=True)
class InvocationRow:
  """One exported row: the build metrics of a single Bazel invocation."""

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
  bazel_version: str | None
  finish_time: str | None
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
  runner_counts: list[RunnerCount]
  action_data: list[ActionData]


def _int_or_none(value: Any) -> int | None:
  """Parses a protobuf JSON int64 (emitted as a string) or int32."""
  return None if value is None else int(value)


def _millis_to_seconds(value: Any) -> float | None:
  """Converts a protobuf JSON int64 milliseconds value to seconds."""
  return None if value is None else int(value) / 1000.0


def _duration_or_none(msg: Mapping[str, Any], field: str) -> float | None:
  """Returns a Duration field in seconds, or None if unset."""
  return bep.parse_duration_seconds(msg, field, f"{field}Millis")


def _parse_runner_counts(
    action_summary: Mapping[str, Any],
) -> list[RunnerCount]:
  """Returns the per-runner action counts, excluding the "total" pseudo-runner.

  "total" equals `actionsExecuted` and would double count every action when
  summed with the real runners.

  Args:
    action_summary: The `BuildMetrics.actionSummary` message.

  Returns:
    The runner counts in BEP order.
  """
  return [
      RunnerCount(
          name=runner.get("name", ""),
          exec_kind=runner.get("execKind"),
          count=int(runner.get("count", 0)),
      )
      for runner in action_summary.get("runnerCount", [])
      if runner.get("name") != "total"
  ]


def _parse_action_data(action_summary: Mapping[str, Any]) -> list[ActionData]:
  """Returns the per-mnemonic action stats of a `BuildMetrics.actionSummary`."""
  return [
      ActionData(
          mnemonic=stats.get("mnemonic", ""),
          actions_created=_int_or_none(stats.get("actionsCreated")),
          actions_executed=_int_or_none(stats.get("actionsExecuted")),
          user_time_s=_duration_or_none(stats, "userTime"),
          system_time_s=_duration_or_none(stats, "systemTime"),
      )
      for stats in action_summary.get("actionData", [])
  ]


def _sum_runner_counts(
    runner_counts: Sequence[RunnerCount] | None,
    *,
    name: str | None = None,
    exec_kinds: frozenset[str] = frozenset(),
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
    elif "buildFinished" in event_id:
      collected.finished = event.get("finished", {})
    elif "buildMetrics" in event_id:
      collected.metrics = event.get("buildMetrics", {})
  return collected


def extract_invocation_row(
    events: Iterable[Mapping[str, Any]], context: bep.RunContext
) -> InvocationRow | None:
  """Flattens the invocation-level events of one BEP stream into a row.

  A stream cut short (e.g. by job cancellation) has no `BuildMetrics` event;
  its row is still emitted, with NULL metrics, so that the invocation counts.

  Args:
    events: Parsed BEP JSON events of a single Bazel invocation.
    context: GitHub Actions context attached to the row.

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
      invocation_start_time=bep.normalize_timestamp(started.get("startTime")),
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
      bazel_config=context.bazel_config,
      bazel_version=started.get("buildToolVersion"),
      finish_time=bep.normalize_timestamp(finished.get("finishTime")),
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
  context = bep.RunContext.from_env(
      os.environ,
      job_name=args.job_name,
      runner_label=args.runner_label,
      bazel_config=args.bazel_config,
  )

  rows: list[InvocationRow] = []
  for bep_file in args.bep_file:
    if not bep_file.is_file():
      # The Bazel step may have been skipped or failed before writing the file.
      logging.warning("BEP file %s not found; skipping.", bep_file)
      continue
    with bep_file.open(encoding="utf-8") as f:
      row = extract_invocation_row(bep.iter_bep_events(f), context)
    if row is None:
      logging.warning("BEP file %s has no BuildStarted event.", bep_file)
      continue
    rows.append(row)
  write_ndjson(rows, args.output)
  logging.info("Wrote %d invocation rows to %s.", len(rows), args.output)

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
