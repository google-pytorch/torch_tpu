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

"""Tests for harness/export.py.

The MLCompass upload (`export_lib_borg.export_results_to_mlcompass`) is the
only external boundary; it is replaced with a fake that records the uploaded
records so the tests can check what would reach MLCompass.
"""

from collections.abc import Sequence
import datetime
import json
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from torch_tpu._internal.benchmarks import export_adapter
from perflab import common
from perflab import export as export_lib
from perflab import known_failures
from perflab import metrics as metrics_lib
from perflab import mode as mode_lib
from perflab import target as target_lib
from tests import seed_test_utils


class _FakeMlCompass:
  """Records the BenchmarkData records uploaded to MLCompass."""

  def __init__(self):
    self.uploaded = []

  def export_results_to_mlcompass(self, data, **unused_kwargs):
    self.uploaded.append(data)


# The MLCompass upload is the I/O edge only in google3; the OSS export_adapter
# writes TensorBoard events instead, so these tests are skipped there rather
# than faking a different boundary per platform.
@absltest.skipUnless(
    hasattr(export_adapter, "export_lib_borg"),
    "MLCompass export is only available in google3.",
)
class ExportTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.mlcompass = _FakeMlCompass()
    self.enter_context(
        mock.patch.object(
            export_adapter.export_lib_borg,
            "export_results_to_mlcompass",
            self.mlcompass.export_results_to_mlcompass,
        )
    )
    self.enter_context(
        flagsaver.flagsaver(
            (export_adapter.MLCOMPASS_TRACKING_ID, "uuid-1234"),
            (export_adapter.TEST_SUITE, "model"),
        )
    )

  def test_export_uploads_record_keyed_by_test_name(self):
    perf_metrics = metrics_lib.PerformanceMetrics(
        e2e_wall_time_seconds=10.0,
        post_warmup_step_time_seconds=0.5,
        peak_device_memory_mb=1024.0,
    )

    export_lib.export(
        export_lib.BenchmarkData(
            spec_name="test_model",
            platform=target_lib.Platform.CPU,
            framework=mode_lib.Framework.TORCH,
            run_mode=common.RunMode.EAGER_DEFAULT,
            succeeded=True,
            metrics=perf_metrics,
        )
    )

    self.assertLen(self.mlcompass.uploaded, 1)
    record = self.mlcompass.uploaded[0]
    self.assertEqual(
        record.test_name, "torch_tpu/cpu/test_model/eager_default/torch"
    )
    self.assertEqual(record.platform, "cpu")
    self.assertEqual(record.parameters, "torch")
    self.assertEqual(record.implementation, "eager_default")
    self.assertEqual(record.metrics, perf_metrics.metric_map())
    self.assertEqual(record.wall_time, 10.0)
    self.assertTrue(record.succeeded)
    self.assertFalse(record.micro_result_key)

  def test_export_with_microbenchmark(self):
    perf_metrics = metrics_lib.PerformanceMetrics(
        e2e_wall_time_seconds=12.5,
        post_warmup_step_time_seconds=0.25,
    )

    export_lib.export(
        export_lib.BenchmarkData(
            spec_name="llama3_8b_forward",
            platform=target_lib.Platform.V6E_1X1,
            framework=mode_lib.Framework.TORCHAX,
            run_mode=common.RunMode.COMPILED,
            succeeded=False,
            metrics=perf_metrics,
            microbenchmark_name="prefill",
        )
    )

    self.assertLen(self.mlcompass.uploaded, 1)
    record = self.mlcompass.uploaded[0]
    self.assertEqual(
        record.test_name,
        "torch_tpu/v6e_1x1/llama3_8b_forward/compiled/torchax",
    )
    self.assertEqual(record.micro_result_key, "prefill")
    self.assertEqual(record.wall_time, 12.5)
    self.assertFalse(record.succeeded)

  @mock.patch.object(export_adapter, "export_benchmark_skip")
  def test_export_skip(self, mock_export_benchmark_skip):
    export_lib.export_skip(
        spec_name="test_model",
        platform=target_lib.Platform.V7_1X1X1,
        framework=mode_lib.Framework.TORCH,
        run_mode=common.RunMode.EAGER_OPTIMIZED,
    )
    mock_export_benchmark_skip.assert_called_once_with(
        spec_name="test_model",
        platform="v7_1x1x1",
        framework="torch",
        run_mode="eager_optimized",
    )

  @mock.patch.object(export_adapter, "export_benchmark_skip")
  def test_export_known_failure_skip(self, mock_export_benchmark_skip):
    failures = (
        known_failures.KnownFailure(
            benchmark="known_bad",
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
            run_modes=frozenset({common.RunMode.COMPILED}),
            reason="simulated TPU compiled failure",
        ),
    )
    with mock.patch.object(known_failures, "KNOWN_FAILURES", failures):
      export_lib.export_known_failure_skip(
          spec_name="known_bad",
          platform=target_lib.Platform.V7_1X1X1,
          framework=mode_lib.Framework.TORCH,
          run_mode=common.RunMode.COMPILED,
      )
    mock_export_benchmark_skip.assert_called_once_with(
        spec_name="known_bad",
        platform="v7_1x1x1",
        framework="torch",
        run_mode="compiled",
    )

  @mock.patch.object(export_adapter, "export_benchmark_skip")
  def test_export_known_failure_skip_ignores_other_skips(
      self, mock_export_benchmark_skip
  ):
    failures = (
        known_failures.KnownFailure(
            benchmark="known_bad",
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
            run_modes=frozenset({common.RunMode.COMPILED}),
            reason="simulated TPU compiled failure",
        ),
    )
    with mock.patch.object(known_failures, "KNOWN_FAILURES", failures):
      for platform, run_mode in (
          # Another run mode, and another device kind.
          (target_lib.Platform.V7_1X1X1, common.RunMode.EAGER_DEFAULT),
          (target_lib.Platform.B200_1, common.RunMode.COMPILED),
      ):
        export_lib.export_known_failure_skip(
            spec_name="known_bad",
            platform=platform,
            framework=mode_lib.Framework.TORCH,
            run_mode=run_mode,
        )
    mock_export_benchmark_skip.assert_not_called()

  def test_make_benchmark_data(self):
    spec = types.SimpleNamespace(name="layer", microbenchmark_name="shape_1")
    perf_metrics = metrics_lib.PerformanceMetrics(e2e_wall_time_seconds=3.0)

    data = export_lib.make_benchmark_data(
        spec,
        platform=target_lib.Platform.V7_1X1X1,
        framework=mode_lib.Framework.TORCH,
        run_mode=common.RunMode.COMPILED,
        succeeded=True,
        metrics=perf_metrics,
    )

    self.assertEqual(
        data,
        export_lib.BenchmarkData(
            spec_name="layer",
            platform=target_lib.Platform.V7_1X1X1,
            framework=mode_lib.Framework.TORCH,
            run_mode=common.RunMode.COMPILED,
            succeeded=True,
            metrics=perf_metrics,
            microbenchmark_name="shape_1",
        ),
    )

  def test_make_benchmark_data_defaults_to_empty_metrics(self):
    spec = types.SimpleNamespace(name="model", microbenchmark_name=None)

    data = export_lib.make_benchmark_data(
        spec,
        platform=target_lib.Platform.B200_8,
        framework=mode_lib.Framework.TORCH,
        run_mode=common.RunMode.COMPILED,
        succeeded=False,
    )

    self.assertFalse(data.succeeded)
    self.assertEqual(data.metrics, metrics_lib.PerformanceMetrics())
    self.assertIsNone(data.microbenchmark_name)
    self.assertIsNone(data.batch_size)

  def test_make_benchmark_data_batch_sweep_case(self):
    data = [
        export_lib.make_benchmark_data(
            types.SimpleNamespace(name="layer", microbenchmark_name=name),
            platform=target_lib.Platform.V7_1X1X1,
            framework=mode_lib.Framework.TORCH,
            run_mode=common.RunMode.COMPILED,
            succeeded=True,
            batch_size=16,
            out_of_memory=out_of_memory,
            skipped=skipped,
        )
        for name, out_of_memory, skipped in (
            ("shape_1", False, False),
            (None, True, False),
            (None, False, True),
        )
    ]

    self.assertEqual(
        [d.microbenchmark_name for d in data], ["shape_1_bs16", "bs16", "bs16"]
    )
    self.assertEqual([d.batch_size for d in data], [16, 16, 16])
    self.assertEqual([d.out_of_memory for d in data], [False, True, False])
    self.assertEqual([d.skipped for d in data], [False, False, True])

  @mock.patch.object(export_adapter, "export_benchmark_results")
  def test_export_batch_sweep_case(self, mock_export_benchmark_results):
    perf_metrics = metrics_lib.PerformanceMetrics(
        e2e_wall_time_seconds=3.0, post_warmup_step_time_seconds=0.5
    )
    export_lib.export(
        export_lib.BenchmarkData(
            spec_name="test_model",
            platform=target_lib.Platform.V7_1X1X1,
            framework=mode_lib.Framework.TORCH,
            run_mode=common.RunMode.COMPILED,
            succeeded=True,
            metrics=perf_metrics,
            microbenchmark_name=export_lib.batch_size_key(16),
            batch_size=16,
        )
    )
    mock_export_benchmark_results.assert_called_once_with(
        spec_name="test_model",
        platform="v7_1x1x1",
        framework="torch",
        run_mode="compiled",
        metrics=perf_metrics.metric_map(),
        succeeded=True,
        wall_time=3.0,
        microbenchmark_name="bs16",
    )

  @mock.patch.object(export_adapter, "export_benchmark_results")
  def test_export_out_of_memory(self, mock_export_benchmark_results):
    export_lib.export(
        export_lib.BenchmarkData(
            spec_name="test_model",
            platform=target_lib.Platform.B200_1,
            framework=mode_lib.Framework.TORCH,
            run_mode=common.RunMode.COMPILED,
            succeeded=True,
            metrics=metrics_lib.PerformanceMetrics(),
            microbenchmark_name=export_lib.batch_size_key(64),
            batch_size=64,
            out_of_memory=True,
        )
    )
    mock_export_benchmark_results.assert_called_once_with(
        spec_name="test_model",
        platform="b200_1",
        framework="torch",
        run_mode="compiled",
        metrics={"out_of_memory": 1.0, "batch_size": 64.0},
        succeeded=True,
        wall_time=None,
        microbenchmark_name="bs64",
    )

  @mock.patch.object(export_adapter, "export_benchmark_results")
  def test_export_skipped_after_out_of_memory(
      self, mock_export_benchmark_results
  ):
    export_lib.export(
        export_lib.BenchmarkData(
            spec_name="test_model",
            platform=target_lib.Platform.B200_1,
            framework=mode_lib.Framework.TORCH,
            run_mode=common.RunMode.COMPILED,
            succeeded=True,
            metrics=metrics_lib.PerformanceMetrics(),
            microbenchmark_name=export_lib.batch_size_key(128),
            batch_size=128,
            skipped=True,
        )
    )
    mock_export_benchmark_results.assert_called_once_with(
        spec_name="test_model",
        platform="b200_1",
        framework="torch",
        run_mode="compiled",
        metrics={"skipped": 1.0, "batch_size": 128.0},
        succeeded=True,
        wall_time=None,
        microbenchmark_name="bs128",
    )


def _result_lines(log_output: Sequence[str]) -> list[str]:
  """Returns the JSON payloads of `PERFLAB_RESULT_JSON` INFO log lines."""
  prefix = f"{export_lib.RESULT_LOG_PREFIX} "
  lines = []
  for line in log_output:
    # assertLogs formats records as "<LEVEL>:<logger>:<message>".
    parts = line.split(":", 2)
    if len(parts) == 3 and parts[0] == "INFO" and parts[2].startswith(prefix):
      lines.append(parts[2][len(prefix) :])
  return lines


class ResultLogTest(seed_test_utils.RepeatableTest):
  """Tests for PERFLAB_RESULT_JSON structured log records (shared with OSS)."""

  def _read_results(self, log_output: Sequence[str]) -> list[dict[str, object]]:
    records = [json.loads(line) for line in _result_lines(log_output)]
    for record in records:
      # Each record has a timezone-aware ISO 8601 timestamp.
      timestamp = datetime.datetime.fromisoformat(record.pop("timestamp"))
      self.assertIsNotNone(timestamp.tzinfo)
    return records

  @mock.patch.object(export_adapter, "export_benchmark_results")
  def test_export_logs_structured_json_records(self, unused_mock_export):
    perf_metrics = metrics_lib.PerformanceMetrics(
        e2e_wall_time_seconds=10.0,
        post_warmup_step_time_seconds=0.5,
        warmup_session_xprof_url="http://xprof/warmup",
        post_warmup_run_session_xprof_url="http://xprof/post_warmup",
    )
    with self.assertLogs(level="INFO") as logs:
      export_lib.export(
          export_lib.BenchmarkData(
              spec_name="model",
              platform=target_lib.Platform.V7_2X2X1,
              framework=mode_lib.Framework.TORCH,
              run_mode=common.RunMode.COMPILED,
              succeeded=True,
              metrics=perf_metrics,
              microbenchmark_name="prefill",
          )
      )
      export_lib.export(
          export_lib.BenchmarkData(
              spec_name="model",
              platform=target_lib.Platform.B200_1,
              framework=mode_lib.Framework.TORCH,
              run_mode=common.RunMode.EAGER_DEFAULT,
              succeeded=False,
              metrics=metrics_lib.PerformanceMetrics(),
          )
      )
      export_lib.export(
          export_lib.BenchmarkData(
              spec_name="model",
              platform=target_lib.Platform.B200_1,
              framework=mode_lib.Framework.TORCH,
              run_mode=common.RunMode.COMPILED,
              succeeded=True,
              metrics=metrics_lib.PerformanceMetrics(),
              microbenchmark_name="bs64",
              batch_size=64,
              out_of_memory=True,
          )
      )

    ok, failed, oom = self._read_results(logs.output)
    self.assertEqual(
        ok,
        {
            "schema_version": 1,
            "spec_name": "model",
            "microbenchmark_name": "prefill",
            "batch_size": None,
            "status": "ok",
            "succeeded": True,
            "metrics": dict(perf_metrics.metric_map()),
            "test_name": "torch_tpu/v7_2x2x1/model/compiled/torch",
            "warmup_xprof_url": "http://xprof/warmup",
            "post_warmup_xprof_url": "http://xprof/post_warmup",
        },
    )
    self.assertEqual((failed["status"], failed["succeeded"]), ("failed", False))
    self.assertIsNone(failed["warmup_xprof_url"])
    self.assertEqual(
        failed["test_name"],
        f"torch_tpu/b200_1/model/{common.RunMode.EAGER_DEFAULT.value}/torch",
    )
    self.assertEqual((oom["status"], oom["succeeded"]), ("oom", True))
    self.assertEqual(oom["metrics"], {"out_of_memory": 1.0, "batch_size": 64.0})
    self.assertEqual(oom["batch_size"], 64)

  @mock.patch.object(export_adapter, "export_benchmark_skip")
  def test_export_skip_logs_skipped_result(self, unused_mock_export):
    with self.assertLogs(level="INFO") as logs:
      export_lib.export_skip(
          spec_name="model",
          platform=target_lib.Platform.V7_1X1X1,
          framework=mode_lib.Framework.TORCH,
          run_mode=common.RunMode.COMPILED,
      )

    (skipped,) = self._read_results(logs.output)
    self.assertEqual(skipped["spec_name"], "model")
    self.assertEqual(skipped["status"], "skipped")
    self.assertTrue(skipped["succeeded"])
    self.assertEqual(skipped["metrics"], {"skipped": 1.0})
    self.assertEqual(
        skipped["test_name"], "torch_tpu/v7_1x1x1/model/compiled/torch"
    )

  @mock.patch.object(export_adapter, "export_benchmark_skip")
  def test_export_known_failure_skip_logs_nothing_for_other_skips(
      self, mock_export_benchmark_skip
  ):
    with (
        mock.patch.object(known_failures, "KNOWN_FAILURES", ()),
        self.assertNoLogs(level="INFO"),
    ):
      export_lib.export_known_failure_skip(
          spec_name="model",
          platform=target_lib.Platform.V7_1X1X1,
          framework=mode_lib.Framework.TORCH,
          run_mode=common.RunMode.COMPILED,
      )
    mock_export_benchmark_skip.assert_not_called()

  @mock.patch.object(export_adapter, "export_benchmark_results")
  def test_export_logs_non_finite_metrics_as_null(self, unused_mock_export):
    with self.assertLogs(level="INFO") as logs:
      export_lib.export(
          export_lib.BenchmarkData(
              spec_name="model",
              platform=target_lib.Platform.CPU,
              framework=mode_lib.Framework.TORCH,
              run_mode=common.RunMode.COMPILED,
              succeeded=True,
              metrics=metrics_lib.PerformanceMetrics(
                  post_warmup_step_time_seconds=float("nan"),
                  first_step_time_seconds=float("inf"),
                  warmup_overhead_seconds=float("-inf"),
                  peak_device_memory_mb=12.0,
              ),
          )
      )

    (line,) = _result_lines(logs.output)
    # Strict JSON: no NaN or Infinity tokens.
    self.assertNotIn("NaN", line)
    self.assertNotIn("Infinity", line)
    (record,) = self._read_results(logs.output)
    metrics = record["metrics"]
    self.assertIsNone(metrics["post_warmup_step_time_seconds"])
    self.assertIsNone(metrics["first_step_time_seconds"])
    self.assertIsNone(metrics["warmup_overhead_seconds"])
    self.assertEqual(metrics["peak_device_memory_mb"], 12.0)

  @mock.patch.object(export_adapter, "export_benchmark_results")
  def test_export_survives_result_serialization_failure(self, mock_export):
    with (
        mock.patch.object(
            export_lib.json, "dumps", side_effect=TypeError("not serializable")
        ),
        self.assertLogs(level="INFO") as logs,
    ):
      export_lib.export(
          export_lib.BenchmarkData(
              spec_name="model",
              platform=target_lib.Platform.CPU,
              framework=mode_lib.Framework.TORCH,
              run_mode=common.RunMode.COMPILED,
              succeeded=True,
              metrics=metrics_lib.PerformanceMetrics(),
          )
      )
    # The results still reach the export adapter.
    mock_export.assert_called_once()
    self.assertEmpty(_result_lines(logs.output))

  @mock.patch.object(
      export_adapter,
      "export_benchmark_results",
      side_effect=RuntimeError("adapter failed"),
  )
  def test_export_logs_result_before_adapter_failure(self, unused_mock_export):
    with (
        self.assertLogs(level="INFO") as logs,
        self.assertRaisesRegex(  # ASSERT_RAISES_OK=export propagates adapter errors after logging the record.
            RuntimeError, "adapter failed"
        ),
    ):
      export_lib.export(
          export_lib.BenchmarkData(
              spec_name="model",
              platform=target_lib.Platform.CPU,
              framework=mode_lib.Framework.TORCH,
              run_mode=common.RunMode.COMPILED,
              succeeded=True,
              metrics=metrics_lib.PerformanceMetrics(),
          )
      )
    (record,) = self._read_results(logs.output)
    self.assertEqual(record["spec_name"], "model")
    self.assertEqual(record["status"], "ok")


if __name__ == "__main__":
  absltest.main()
