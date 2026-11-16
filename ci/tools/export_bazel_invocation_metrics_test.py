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

"""Unit tests for the export_bazel_invocation_metrics CI telemetry tool."""

from collections.abc import Mapping, Sequence
import copy
import dataclasses
import json
import pathlib
import subprocess
import tempfile
from typing import Any, Final
import unittest
from unittest import mock

from ci.tools import export_bazel_invocation_metrics as exporter

_INVOCATION_ID: Final[str] = "9b5c0f1e-0000-4000-8000-000000000001"
_CONTEXT: Final[exporter.RunContext] = exporter.RunContext(
    repository="google-pytorch/torch_tpu",
    workflow_name="CI - Postsubmit",
    job_key="run_tests",
    job_name="CPU",
    run_id="1234567890",
    run_attempt=2,
    event_name="push",
    ref="refs/heads/main",
    head_sha="deadbeef",
    runner_label="linux-x86-n2-16",
    bazel_config="ci_cpu",
)


def _started(command: str = "build") -> dict[str, Any]:
  """Returns a `BuildStarted` event for a Bazel `command`."""
  return {
      "id": {"started": {}},
      "started": {
          "uuid": _INVOCATION_ID,
          "command": command,
          "startTime": "2026-09-25T21:47:12.475123456Z",
          "buildToolVersion": "9.2.0",
      },
  }


def _finished(exit_code: Mapping[str, Any]) -> dict[str, Any]:
  """Returns a `BuildFinished` event with the given `ExitCode` message."""
  return {
      "id": {"buildFinished": {}},
      "finished": {
          "exitCode": dict(exit_code),
          "finishTime": "2026-09-25T21:47:40.260Z",
      },
  }


# Shaped after a real CI `BuildMetrics` event (Bazel 9.2.0).
_BUILD_METRICS: Final[dict[str, Any]] = {
    "id": {"buildMetrics": {}},
    "buildMetrics": {
        "actionSummary": {
            "actionsCreated": "246495",
            "actionsExecuted": "944",
            "actionData": [
                {
                    "mnemonic": "CppCompile",
                    "actionsExecuted": "21",
                    "actionsCreated": "4552",
                    "userTime": "5.286s",
                    "systemTime": "0.778s",
                },
                {"mnemonic": "SymlinkTree", "actionsCreated": "4"},
            ],
            "runnerCount": [
                {"name": "total", "count": 944},
                {"name": "internal", "count": 16},
                {
                    "name": "remote cache hit",
                    "count": 1722,
                    "execKind": "Remote",
                },
                {
                    "name": "processwrapper-sandbox",
                    "count": 19,
                    "execKind": "Local",
                },
                {"name": "worker", "count": 3, "execKind": "Worker"},
                {"name": "remote", "count": 2, "execKind": "Remote"},
            ],
            "actionCacheStatistics": {"hits": 68300, "misses": 944},
        },
        "timingMetrics": {
            "cpuTimeInMs": "139440",
            "wallTimeInMs": "27804",
            "analysisPhaseTimeInMs": "5083",
            "executionPhaseTimeInMs": "19345",
            "criticalPathTime": "0.634931642s",
        },
        "networkMetrics": {
            "systemNetworkStats": {
                "bytesSent": "51352441",
                "bytesRecv": "4287504273",
            }
        },
        "artifactMetrics": {
            "outputArtifactsSeen": {"sizeInBytes": "3484100591", "count": 1},
            "outputArtifactsFromActionCache": {
                "sizeInBytes": "3481570616",
                "count": 1,
            },
        },
    },
    "lastMessage": True,
}

# Rows parsed from `_BUILD_METRICS["buildMetrics"]["actionSummary"]`.
_EXPECTED_RUNNER_COUNTS: Final[tuple[exporter.RunnerCount, ...]] = (
    exporter.RunnerCount(name="internal", exec_kind=None, count=16),
    exporter.RunnerCount(
        name="remote cache hit", exec_kind="Remote", count=1722
    ),
    exporter.RunnerCount(
        name="processwrapper-sandbox", exec_kind="Local", count=19
    ),
    exporter.RunnerCount(name="worker", exec_kind="Worker", count=3),
    exporter.RunnerCount(name="remote", exec_kind="Remote", count=2),
)
_EXPECTED_ACTION_DATA: Final[tuple[exporter.ActionData, ...]] = (
    exporter.ActionData(
        mnemonic="CppCompile",
        actions_created=4552,
        actions_executed=21,
        user_time_s=5.286,
        system_time_s=0.778,
    ),
    exporter.ActionData(
        mnemonic="SymlinkTree",
        actions_created=4,
        actions_executed=None,
        user_time_s=None,
        system_time_s=None,
    ),
)


class ExtractInvocationRowTest(unittest.TestCase):

  def _extract(
      self, events: Sequence[Mapping[str, Any]]
  ) -> exporter.InvocationRow:
    """Extracts the row of `events`, failing the test if there is none."""
    row = exporter.extract_invocation_row(events, _CONTEXT)
    if row is None:
      self.fail("extract_invocation_row returned None.")
    return row

  def test_full_stream(self):
    row = self._extract(
        [_started(), _finished({"name": "SUCCESS"}), _BUILD_METRICS]
    )
    self.assertEqual(
        row,
        exporter.InvocationRow(
            invocation_id=_INVOCATION_ID,
            invocation_start_time="2026-09-25 21:47:12.475123 UTC",
            bazel_command="build",
            repository="google-pytorch/torch_tpu",
            workflow_name="CI - Postsubmit",
            job_key="run_tests",
            job_name="CPU",
            run_id="1234567890",
            run_attempt=2,
            event_name="push",
            ref="refs/heads/main",
            head_sha="deadbeef",
            runner_label="linux-x86-n2-16",
            bazel_config="ci_cpu",
            bazel_version="9.2.0",
            finish_time="2026-09-25 21:47:40.260000 UTC",
            exit_code_name="SUCCESS",
            exit_code=0,
            wall_time_s=27.804,
            cpu_time_s=139.44,
            analysis_phase_s=5.083,
            execution_phase_s=19.345,
            critical_path_s=0.634931642,
            actions_created=246495,
            actions_executed=944,
            remote_cache_hit_count=1722,
            disk_cache_hit_count=0,
            remote_execution_count=2,
            # Local sandbox and persistent worker executions.
            local_execution_count=22,
            action_cache_hits=68300,
            action_cache_misses=944,
            bytes_sent=51352441,
            bytes_recv=4287504273,
            output_artifacts_seen_bytes=3484100591,
            output_artifacts_from_action_cache_bytes=3481570616,
            runner_counts=_EXPECTED_RUNNER_COUNTS,
            action_data=_EXPECTED_ACTION_DATA,
        ),
    )

  def test_runner_counts_exclude_total(self):
    row = self._extract([_started(), _BUILD_METRICS])
    self.assertEqual(row.runner_counts, _EXPECTED_RUNNER_COUNTS)

  def test_derived_runner_counts(self):
    row = self._extract([_started(), _BUILD_METRICS])
    self.assertEqual(row.remote_cache_hit_count, 1722)
    self.assertEqual(row.disk_cache_hit_count, 0)
    self.assertEqual(row.remote_execution_count, 2)
    # Local sandbox and persistent worker executions.
    self.assertEqual(row.local_execution_count, 22)

  def test_action_data(self):
    row = self._extract([_started(), _BUILD_METRICS])
    self.assertEqual(row.action_data, _EXPECTED_ACTION_DATA)

  def test_row_is_hashable(self):
    row = self._extract([_started(), _BUILD_METRICS])
    self.assertEqual(hash(row), hash(dataclasses.replace(row)))

  def test_success_exit_code_defaults_to_zero(self):
    row = self._extract([_started(), _finished({"name": "SUCCESS"})])
    self.assertEqual(row.exit_code_name, "SUCCESS")
    self.assertEqual(row.exit_code, 0)

  def test_failure_exit_code(self):
    row = self._extract(
        [_started("test"), _finished({"name": "TESTS_FAILED", "code": 3})]
    )
    self.assertEqual(row.bazel_command, "test")
    self.assertEqual(row.exit_code_name, "TESTS_FAILED")
    self.assertEqual(row.exit_code, 3)

  def test_truncated_stream_has_null_metrics(self):
    row = self._extract([_started()])
    self.assertEqual(row.invocation_id, _INVOCATION_ID)
    self.assertIsNone(row.exit_code)
    self.assertIsNone(row.finish_time)
    self.assertIsNone(row.wall_time_s)
    self.assertIsNone(row.remote_cache_hit_count)
    self.assertIsNone(row.local_execution_count)
    self.assertIsNone(row.action_cache_hits)
    self.assertEqual(row.runner_counts, ())
    self.assertEqual(row.action_data, ())

  def test_omitted_cache_counter_is_zero(self):
    metrics = copy.deepcopy(_BUILD_METRICS)
    del metrics["buildMetrics"]["actionSummary"]["actionCacheStatistics"][
        "hits"
    ]
    row = self._extract([_started(), metrics])
    self.assertEqual(row.action_cache_hits, 0)
    self.assertEqual(row.action_cache_misses, 944)

  def test_missing_critical_path_is_none(self):
    # Bazel 8 does not report `criticalPathTime` in `timingMetrics`.
    metrics = copy.deepcopy(_BUILD_METRICS)
    del metrics["buildMetrics"]["timingMetrics"]["criticalPathTime"]
    row = self._extract([_started(), metrics])
    self.assertIsNone(row.critical_path_s)

  def test_bazel_config_from_command_line(self):
    options = {
        "id": {"optionsParsed": {}},
        "optionsParsed": {
            "explicitCmdLine": [
                "--config=ci_tpu_v5_newest_deps",
                "--build_event_json_file=/tmp/bep.json",
            ]
        },
    }
    row = self._extract([_started("test"), options, _BUILD_METRICS])
    self.assertEqual(row.bazel_config, "ci_tpu_v5_newest_deps")

  def test_bazel_config_falls_back_to_context(self):
    row = self._extract([_started(), _BUILD_METRICS])
    self.assertEqual(row.bazel_config, "ci_cpu")

  def test_no_started_event_returns_none(self):
    self.assertIsNone(
        exporter.extract_invocation_row([_BUILD_METRICS], _CONTEXT)
    )


class SchemaTest(unittest.TestCase):

  def test_schema_matches_row_fields(self):
    self.assertEqual(
        [field["name"] for field in exporter.BQ_SCHEMA],
        [field.name for field in dataclasses.fields(exporter.InvocationRow)],
    )

  def test_repeated_schema_matches_nested_fields(self):
    nested = {
        field["name"]: [f["name"] for f in field["fields"]]
        for field in exporter.BQ_SCHEMA
        if field["mode"] == "REPEATED"
    }
    self.assertEqual(
        nested,
        {
            "runner_counts": [
                f.name for f in dataclasses.fields(exporter.RunnerCount)
            ],
            "action_data": [
                f.name for f in dataclasses.fields(exporter.ActionData)
            ],
        },
    )

  def test_column_types_derive_from_annotations(self):
    types_by_name = {
        field["name"]: (field["type"], field["mode"])
        for field in exporter.BQ_SCHEMA
    }
    self.assertEqual(types_by_name["invocation_id"], ("STRING", "NULLABLE"))
    self.assertEqual(
        types_by_name["invocation_start_time"], ("TIMESTAMP", "NULLABLE")
    )
    self.assertEqual(types_by_name["finish_time"], ("TIMESTAMP", "NULLABLE"))
    self.assertEqual(types_by_name["run_attempt"], ("INTEGER", "NULLABLE"))
    self.assertEqual(types_by_name["wall_time_s"], ("FLOAT", "NULLABLE"))
    self.assertEqual(types_by_name["runner_counts"], ("RECORD", "REPEATED"))

  def test_nested_column_types(self):
    (runner_counts,) = (
        field
        for field in exporter.BQ_SCHEMA
        if field["name"] == "runner_counts"
    )
    self.assertEqual(
        runner_counts["fields"],
        (
            {"name": "name", "type": "STRING", "mode": "NULLABLE"},
            {"name": "exec_kind", "type": "STRING", "mode": "NULLABLE"},
            {"name": "count", "type": "INTEGER", "mode": "NULLABLE"},
        ),
    )


class ParseDurationSecondsTest(unittest.TestCase):

  def test_parses_protobuf_duration(self):
    self.assertEqual(
        exporter.parse_duration_seconds({"d": "12.25s"}, "d", "dMillis"),
        12.25,
    )

  def test_falls_back_to_legacy_millis(self):
    self.assertEqual(
        exporter.parse_duration_seconds({"dMillis": "1500"}, "d", "dMillis"),
        1.5,
    )

  def test_returns_none_when_missing(self):
    self.assertIsNone(exporter.parse_duration_seconds({}, "d", "dMillis"))


class NormalizeTimestampTest(unittest.TestCase):

  def test_normalizes_to_utc_microseconds(self):
    cases = (
        # Nanoseconds are truncated to microseconds.
        ("2026-09-25T17:00:00.123456789Z", "2026-09-25 17:00:00.123456 UTC"),
        ("2026-09-25T17:00:00Z", "2026-09-25 17:00:00.000000 UTC"),
        # Offsets are converted to UTC.
        ("2026-09-25T10:00:00.5-07:00", "2026-09-25 17:00:00.500000 UTC"),
    )
    for value, expected in cases:
      with self.subTest(value=value):
        self.assertEqual(exporter.normalize_timestamp(value), expected)

  def test_returns_none_for_invalid_input(self):
    for value in ("bogus", "", None):
      with self.subTest(value=value):
        self.assertIsNone(exporter.normalize_timestamp(value))


class IterBepEventsTest(unittest.TestCase):

  def test_skips_blank_and_truncated_lines(self):
    lines = ['{"id": {"started": {}}}\n', "\n", '{"id": {"testSumm']
    self.assertEqual(
        list(exporter.iter_bep_events(lines)),
        [{"id": {"started": {}}}],
    )


class RunContextTest(unittest.TestCase):

  def test_from_env(self):
    context = exporter.RunContext.from_env(
        {
            "GITHUB_REPOSITORY": "google-pytorch/torch_tpu",
            "GITHUB_WORKFLOW": "CI - Postsubmit",
            "GITHUB_JOB": "run_tests",
            "GITHUB_RUN_ID": "1234567890",
            "GITHUB_RUN_ATTEMPT": "2",
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_SHA": "deadbeef",
        },
        job_name="CPU",
        runner_label="linux-x86-n2-16",
        bazel_config="ci_cpu",
    )

    self.assertEqual(context, _CONTEXT)

  def test_from_empty_env(self):
    context = exporter.RunContext.from_env({})

    self.assertEqual(context, exporter.RunContext())


class MainTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._tmp = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
    self.enterContext(mock.patch.dict("os.environ", {}, clear=True))
    self._output = self._tmp / "out.ndjson"

  def _write_bep(self, name: str, events: Sequence[Mapping[str, Any]]) -> str:
    """Writes `events` as a BEP JSON file and returns its path."""
    path = self._tmp / name
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    return str(path)

  def _read_output(self) -> list[dict[str, Any]]:
    """Returns the rows of the NDJSON output file."""
    return [json.loads(line) for line in self._output.read_text().splitlines()]

  def test_writes_one_row_per_bep_file_and_skips_missing(self):
    build = self._write_bep("build.json", [_started("build"), _BUILD_METRICS])
    test = self._write_bep("test.json", [_started("test"), _BUILD_METRICS])
    exit_code = exporter.main([
        f"--bep_file={build}",
        f"--bep_file={self._tmp / 'missing.json'}",
        f"--bep_file={test}",
        f"--output={self._output}",
        "--job_name=CPU",
    ])
    self.assertEqual(exit_code, 0)
    rows = self._read_output()
    self.assertEqual([r["bazel_command"] for r in rows], ["build", "test"])
    self.assertEqual(rows[0]["job_name"], "CPU")
    self.assertEqual(
        rows[0]["runner_counts"][1],
        {"name": "remote cache hit", "exec_kind": "Remote", "count": 1722},
    )

  def test_no_rows_skips_upload(self):
    with mock.patch.object(subprocess, "run", autospec=True) as mock_run:
      exit_code = exporter.main([
          f"--bep_file={self._tmp / 'missing.json'}",
          f"--output={self._output}",
          "--bq_table=project:dataset.table",
      ])
    self.assertEqual(exit_code, 0)
    mock_run.assert_not_called()
    self.assertEqual(self._read_output(), [])

  def test_uploads_with_json_schema(self):
    bep_file = self._write_bep("build.json", [_started(), _BUILD_METRICS])
    schemas = []

    def fake_run(command: Sequence[str], **kwargs: Any) -> None:
      del kwargs
      schemas.append(json.loads(pathlib.Path(command[-1]).read_text()))

    with mock.patch.object(
        subprocess, "run", autospec=True, side_effect=fake_run
    ) as mock_run:
      exit_code = exporter.main([
          f"--bep_file={bep_file}",
          f"--output={self._output}",
          "--bq_table=project:dataset.table",
      ])
    self.assertEqual(exit_code, 0)
    mock_run.assert_called_once()
    command = mock_run.call_args.args[0]
    self.assertEqual(command[:3], ["bq", "--project_id=project", "load"])
    self.assertIn("--clustering_fields=repository,bazel_command", command)
    self.assertIn("--time_partitioning_field=invocation_start_time", command)
    self.assertEqual(command[-3:-1], ["dataset.table", str(self._output)])
    self.assertEqual(schemas, [json.loads(json.dumps(exporter.BQ_SCHEMA))])

  def _main_with_failing_upload(self, error: Exception) -> int:
    """Runs `main` with an upload that raises `error`; returns the exit code."""
    bep_file = self._write_bep("build.json", [_started(), _BUILD_METRICS])
    with mock.patch.object(subprocess, "run", autospec=True, side_effect=error):
      return exporter.main([
          f"--bep_file={bep_file}",
          f"--output={self._output}",
          "--bq_table=project:dataset.table",
      ])

  def test_upload_failure_returns_1(self):
    cases = (
        (
            subprocess.CalledProcessError(1, "bq", "out", "err"),
            (
                "bq load failed: Command 'bq' returned non-zero exit status"
                " 1.\nstdout: out\nstderr: err"
            ),
        ),
        (
            # `TimeoutExpired` output is bytes even with `text=True`.
            subprocess.TimeoutExpired("bq", 300, output=b"out", stderr=b"err"),
            "bq load timed out: Command 'bq' timed out after 300 seconds",
        ),
        (
            FileNotFoundError("bq"),
            "`bq` CLI not found (bq); cannot upload to project:dataset.table.",
        ),
    )
    for error, expected_log in cases:
      with self.subTest(error=type(error).__name__):
        with self.assertLogs(level="ERROR") as logs:
          exit_code = self._main_with_failing_upload(error)
        self.assertEqual(exit_code, 1)
        self.assertIn(expected_log, logs.output[0])


if __name__ == "__main__":
  unittest.main()
