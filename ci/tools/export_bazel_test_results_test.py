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

"""Unit tests for the export_bazel_test_results CI telemetry tool."""

from collections.abc import Mapping, Sequence
import json
import pathlib
import subprocess
import tempfile
from typing import Any, Final
import unittest
from unittest import mock

from ci.tools import export_bazel_test_results

_CONFIG_ID: Final[str] = "abc123"
_INVOCATION_ID: Final[str] = "9b5c0f1e-0000-4000-8000-000000000001"
_CONTEXT: Final[export_bazel_test_results.RunContext] = (
    export_bazel_test_results.RunContext(
        repository="google-pytorch/torch_tpu",
        workflow_name="CI - Postsubmit",
        job_key="run_tests",
        job_name="CPU",
        run_id="1234567890",
        run_attempt=2,
        event_name="push",
        ref="refs/heads/main",
        head_sha="deadbeef",
        runner_label="linux-x86-n4-16",
        bazel_config="ci_cpu",
    )
)


def _started_event() -> dict[str, Any]:
  return {
      "id": {"started": {}},
      "started": {
          "uuid": _INVOCATION_ID,
          "startTime": "2026-09-25T17:00:00.123456789Z",
          "command": "test",
      },
  }


def _test_result_event(
    label: str,
    *,
    status: str = "PASSED",
    attempt: int = 1,
    duration: str = "1.5s",
    cached_locally: bool = False,
    cached_remotely: bool = False,
    strategy: str = "remote",
) -> dict[str, Any]:
  result: dict[str, Any] = {
      "status": status,
      "testAttemptDuration": duration,
      "executionInfo": {"strategy": strategy},
  }
  if cached_locally:
    result["cachedLocally"] = True
  if cached_remotely:
    result["executionInfo"]["cachedRemotely"] = True
  return {
      "id": {
          "testResult": {
              "label": label,
              "run": 1,
              "shard": 1,
              "attempt": attempt,
              "configuration": {"id": _CONFIG_ID},
          }
      },
      "testResult": result,
  }


def _test_summary_event(label: str, **summary_fields: Any) -> dict[str, Any]:
  summary = {
      "overallStatus": "PASSED",
      "totalRunCount": 1,
      "runCount": 1,
      "attemptCount": 1,
      "shardCount": 1,
      "totalRunDuration": "1.5s",
      "firstStartTime": "2026-09-25T17:00:01Z",
      "lastStopTime": "2026-09-25T17:00:02.5Z",
  }
  summary.update(summary_fields)
  return {
      "id": {
          "testSummary": {
              "label": label,
              "configuration": {"id": _CONFIG_ID},
          }
      },
      "testSummary": summary,
  }


def _target_completed_event(
    label: str, **completed_fields: Any
) -> dict[str, Any]:
  return {
      "id": {
          "targetCompleted": {
              "label": label,
              "configuration": {"id": _CONFIG_ID},
          }
      },
      "completed": {"success": True, **completed_fields},
  }


def _target_configured_event(
    label: str, *, test_size: str | None = "SMALL"
) -> dict[str, Any]:
  configured: dict[str, Any] = {"targetKind": "py_test rule"}
  if test_size:
    configured["testSize"] = test_size
  return {
      "id": {"targetConfigured": {"label": label}},
      "configured": configured,
  }


def _target_skipped_event(label: str) -> dict[str, Any]:
  return {
      "id": {
          "targetCompleted": {
              "label": label,
              "configuration": {"id": _CONFIG_ID},
          }
      },
      "aborted": {
          "reason": "SKIPPED",
          "description": f"Target {label} build was skipped.",
      },
  }


def _to_lines(events: Sequence[Mapping[str, Any]]) -> list[str]:
  return [json.dumps(event) + "\n" for event in events]


class ParseDurationSecondsTest(unittest.TestCase):

  def test_parses_protobuf_duration(self):
    self.assertEqual(
        export_bazel_test_results.parse_duration_seconds(
            {"d": "12.25s"}, "d", "dMillis"
        ),
        12.25,
    )

  def test_falls_back_to_legacy_millis(self):
    self.assertEqual(
        export_bazel_test_results.parse_duration_seconds(
            {"dMillis": "1500"}, "d", "dMillis"
        ),
        1.5,
    )

  def test_returns_none_when_missing(self):
    self.assertIsNone(
        export_bazel_test_results.parse_duration_seconds({}, "d", "dMillis")
    )


class NormalizeTimestampTest(unittest.TestCase):

  def test_truncates_nanoseconds_to_microseconds(self):
    self.assertEqual(
        export_bazel_test_results.normalize_timestamp(
            "2026-09-25T17:00:00.123456789Z"
        ),
        "2026-09-25 17:00:00.123456 UTC",
    )

  def test_handles_missing_fraction(self):
    self.assertEqual(
        export_bazel_test_results.normalize_timestamp("2026-09-25T17:00:00Z"),
        "2026-09-25 17:00:00.000000 UTC",
    )

  def test_converts_offsets_to_utc(self):
    self.assertEqual(
        export_bazel_test_results.normalize_timestamp(
            "2026-09-25T10:00:00.5-07:00"
        ),
        "2026-09-25 17:00:00.500000 UTC",
    )

  def test_returns_none_for_invalid_input(self):
    self.assertIsNone(export_bazel_test_results.normalize_timestamp("bogus"))
    self.assertIsNone(export_bazel_test_results.normalize_timestamp(None))


class IterBepEventsTest(unittest.TestCase):

  def test_skips_blank_and_truncated_lines(self):
    lines = ['{"id": {"started": {}}}\n', "\n", '{"id": {"testSumm']
    self.assertEqual(
        list(export_bazel_test_results.iter_bep_events(lines)),
        [{"id": {"started": {}}}],
    )


class ExtractRowsTest(unittest.TestCase):

  def test_passing_target(self):
    events = [
        _started_event(),
        _target_completed_event("//tests:ops_test", testTimeout="300s"),
        _test_result_event("//tests:ops_test"),
        _test_summary_event("//tests:ops_test"),
    ]

    rows = export_bazel_test_results.extract_rows(events, _CONTEXT)

    self.assertEqual(
        rows,
        [
            export_bazel_test_results.TestTargetRow(
                invocation_id=_INVOCATION_ID,
                invocation_start_time="2026-09-25 17:00:00.123456 UTC",
                bazel_command="test",
                repository="google-pytorch/torch_tpu",
                workflow_name="CI - Postsubmit",
                job_key="run_tests",
                job_name="CPU",
                run_id="1234567890",
                run_attempt=2,
                event_name="push",
                ref="refs/heads/main",
                head_sha="deadbeef",
                runner_label="linux-x86-n4-16",
                bazel_config="ci_cpu",
                target="//tests:ops_test",
                overall_status="PASSED",
                test_timeout_s=300.0,
                total_run_count=1,
                shard_count=1,
                attempt_count=1,
                cached_count=0,
                is_cached=False,
                strategy="remote",
                total_run_duration_s=1.5,
                max_attempt_duration_s=1.5,
                first_start_time="2026-09-25 17:00:01.000000 UTC",
                last_stop_time="2026-09-25 17:00:02.500000 UTC",
            )
        ],
    )

  def test_flaky_target_aggregates_attempts(self):
    events = [
        _started_event(),
        _test_result_event(
            "//tests:flaky_test", status="FAILED", attempt=1, duration="3s"
        ),
        _test_result_event(
            "//tests:flaky_test", status="PASSED", attempt=2, duration="2s"
        ),
        _test_summary_event(
            "//tests:flaky_test",
            overallStatus="FLAKY",
            totalRunCount=2,
            attemptCount=2,
        ),
    ]

    (row,) = export_bazel_test_results.extract_rows(events, _CONTEXT)

    self.assertEqual(row.overall_status, "FLAKY")
    self.assertEqual(row.attempt_count, 2)
    self.assertEqual(row.total_run_count, 2)
    self.assertEqual(row.max_attempt_duration_s, 3.0)

  def test_cached_target(self):
    events = [
        _started_event(),
        _test_result_event("//tests:cached_test", cached_remotely=True),
        _test_summary_event("//tests:cached_test", totalNumCached=1),
    ]

    (row,) = export_bazel_test_results.extract_rows(events, _CONTEXT)

    self.assertEqual(row.cached_count, 1)
    self.assertTrue(row.is_cached)

  def test_cached_count_falls_back_to_test_results(self):
    events = [
        _test_result_event("//tests:cached_test", cached_locally=True),
        _test_summary_event("//tests:cached_test"),
    ]

    (row,) = export_bazel_test_results.extract_rows(events, _CONTEXT)

    self.assertEqual(row.cached_count, 1)
    self.assertTrue(row.is_cached)

  def test_target_that_failed_to_build_has_no_test_results(self):
    events = [
        _test_summary_event(
            "//tests:broken_test",
            overallStatus="FAILED_TO_BUILD",
            totalRunCount=0,
            attemptCount=0,
            totalRunDuration=None,
            firstStartTime=None,
            lastStopTime=None,
        ),
    ]

    (row,) = export_bazel_test_results.extract_rows(events, _CONTEXT)

    self.assertEqual(row.overall_status, "FAILED_TO_BUILD")
    self.assertFalse(row.is_cached)
    self.assertIsNone(row.strategy)
    self.assertIsNone(row.max_attempt_duration_s)
    self.assertIsNone(row.total_run_duration_s)
    self.assertIsNone(row.first_start_time)

  def test_supports_legacy_millis_fields(self):
    summary = _test_summary_event("//tests:ops_test")
    summary["testSummary"] = {
        "overallStatus": "PASSED",
        "totalRunCount": 1,
        "totalRunDurationMillis": "2500",
        "firstStartTimeMillis": "1790355600000",
        "lastStopTimeMillis": "1790355602500",
    }

    (row,) = export_bazel_test_results.extract_rows([summary], _CONTEXT)

    self.assertEqual(row.total_run_duration_s, 2.5)
    self.assertEqual(row.first_start_time, "2026-09-25 17:00:00.000000 UTC")
    self.assertEqual(row.last_stop_time, "2026-09-25 17:00:02.500000 UTC")

  def test_legacy_test_timeout_is_in_seconds(self):
    events = [
        _target_completed_event("//tests:ops_test", testTimeoutSeconds="900"),
        _test_summary_event("//tests:ops_test"),
    ]

    (row,) = export_bazel_test_results.extract_rows(events, _CONTEXT)

    self.assertEqual(row.test_timeout_s, 900.0)

  def test_rows_are_sorted_by_target(self):
    events = [
        _test_summary_event("//tests:b_test"),
        _test_summary_event("//tests:a_test"),
    ]

    rows = export_bazel_test_results.extract_rows(events, _CONTEXT)

    self.assertEqual(
        [row.target for row in rows], ["//tests:a_test", "//tests:b_test"]
    )

  def test_skipped_test_target(self):
    events = [
        _started_event(),
        _target_configured_event("//ci/wheel:pywrap_split_test"),
        _target_skipped_event("//ci/wheel:pywrap_split_test"),
    ]

    (row,) = export_bazel_test_results.extract_rows(events, _CONTEXT)

    self.assertEqual(row.target, "//ci/wheel:pywrap_split_test")
    self.assertEqual(row.overall_status, "SKIPPED")
    self.assertEqual(row.invocation_id, _INVOCATION_ID)
    self.assertEqual(row.run_id, "1234567890")
    self.assertEqual(row.total_run_count, 0)
    self.assertEqual(row.attempt_count, 0)
    self.assertFalse(row.is_cached)
    self.assertIsNone(row.strategy)
    self.assertIsNone(row.test_timeout_s)
    self.assertIsNone(row.total_run_duration_s)

  def test_skipped_non_test_target_is_ignored(self):
    events = [
        _target_configured_event("//ci/wheel:pywrap_lib", test_size=None),
        _target_skipped_event("//ci/wheel:pywrap_lib"),
    ]

    rows = export_bazel_test_results.extract_rows(events, _CONTEXT)

    self.assertEqual(rows, [])


class RunContextTest(unittest.TestCase):

  def test_from_env(self):
    context = export_bazel_test_results.RunContext.from_env(
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
        runner_label="linux-x86-n4-16",
        bazel_config="ci_cpu",
    )

    self.assertEqual(context, _CONTEXT)

  def test_from_empty_env(self):
    context = export_bazel_test_results.RunContext.from_env({})

    self.assertEqual(context, export_bazel_test_results.RunContext())


class MainTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp_dir = pathlib.Path(
        self.enterContext(tempfile.TemporaryDirectory())
    )
    self.bep_file = self.tmp_dir / "bep.json"
    self.output = self.tmp_dir / "rows.ndjson"

  def _write_bep(self, events: Sequence[Mapping[str, Any]]) -> None:
    self.bep_file.write_text("".join(_to_lines(events)), encoding="utf-8")

  def test_missing_bep_file_is_not_an_error(self):
    exit_code = export_bazel_test_results.main(
        [f"--bep_file={self.bep_file}", f"--output={self.output}"]
    )

    self.assertEqual(exit_code, 0)
    self.assertFalse(self.output.exists())

  @mock.patch.object(subprocess, "run", autospec=True)
  def test_writes_ndjson_without_upload(self, mock_run):
    self._write_bep([_started_event(), _test_summary_event("//tests:t")])

    exit_code = export_bazel_test_results.main(
        [f"--bep_file={self.bep_file}", f"--output={self.output}"]
    )

    self.assertEqual(exit_code, 0)
    mock_run.assert_not_called()
    (line,) = self.output.read_text(encoding="utf-8").splitlines()
    row = json.loads(line)
    self.assertEqual(row["target"], "//tests:t")
    self.assertEqual(row["invocation_id"], _INVOCATION_ID)
    # Every exported column must be declared in the BigQuery schema.
    schema_columns = [
        field.split(":")[0]
        for field in export_bazel_test_results.BQ_SCHEMA.split(",")
    ]
    self.assertCountEqual(row.keys(), schema_columns)

  def test_job_name_flag_is_exported(self):
    self._write_bep([_test_summary_event("//tests:t")])

    exit_code = export_bazel_test_results.main([
        f"--bep_file={self.bep_file}",
        f"--output={self.output}",
        "--job_name=TPU v5 (newest deps)",
    ])

    self.assertEqual(exit_code, 0)
    (line,) = self.output.read_text(encoding="utf-8").splitlines()
    self.assertEqual(json.loads(line)["job_name"], "TPU v5 (newest deps)")

  @mock.patch.object(subprocess, "run", autospec=True)
  def test_uploads_to_bigquery(self, mock_run):
    self._write_bep([_test_summary_event("//tests:t")])

    exit_code = export_bazel_test_results.main([
        f"--bep_file={self.bep_file}",
        f"--output={self.output}",
        "--bq_table=my-project:my_dataset.my_table",
    ])

    self.assertEqual(exit_code, 0)
    mock_run.assert_called_once()
    command = mock_run.call_args.args[0]
    self.assertEqual(command[:3], ["bq", "--project_id=my-project", "load"])
    self.assertIn("my_dataset.my_table", command)
    self.assertIn(str(self.output), command)
    self.assertIn("--schema_update_option=ALLOW_FIELD_ADDITION", command)
    self.assertEqual(command[-1], export_bazel_test_results.BQ_SCHEMA)

  @mock.patch.object(subprocess, "run", autospec=True)
  def test_skips_upload_when_no_rows(self, mock_run):
    self._write_bep([_started_event()])

    exit_code = export_bazel_test_results.main([
        f"--bep_file={self.bep_file}",
        f"--output={self.output}",
        "--bq_table=my-project:my_dataset.my_table",
    ])

    self.assertEqual(exit_code, 0)
    mock_run.assert_not_called()

  @mock.patch.object(subprocess, "run", autospec=True)
  def test_upload_failure_returns_non_zero(self, mock_run):
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="bq", output="", stderr="Access Denied"
    )
    self._write_bep([_test_summary_event("//tests:t")])

    exit_code = export_bazel_test_results.main([
        f"--bep_file={self.bep_file}",
        f"--output={self.output}",
        "--bq_table=my-project:my_dataset.my_table",
    ])

    self.assertEqual(exit_code, 1)

  @mock.patch.object(subprocess, "run", autospec=True)
  def test_upload_timeout_returns_non_zero(self, mock_run):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="bq", timeout=300)
    self._write_bep([_test_summary_event("//tests:t")])

    exit_code = export_bazel_test_results.main([
        f"--bep_file={self.bep_file}",
        f"--output={self.output}",
        "--bq_table=my-project:my_dataset.my_table",
    ])

    self.assertEqual(exit_code, 1)

  @mock.patch.object(subprocess, "run", autospec=True)
  def test_missing_bq_cli_returns_non_zero(self, mock_run):
    mock_run.side_effect = FileNotFoundError("bq")
    self._write_bep([_test_summary_event("//tests:t")])

    exit_code = export_bazel_test_results.main([
        f"--bep_file={self.bep_file}",
        f"--output={self.output}",
        "--bq_table=my-project:my_dataset.my_table",
    ])

    self.assertEqual(exit_code, 1)


if __name__ == "__main__":
  unittest.main()
