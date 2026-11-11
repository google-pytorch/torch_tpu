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

"""Tests for harness/export.py."""

import types
from unittest import mock

from absl.testing import absltest
from torch_tpu._internal.benchmarks import export_adapter
from perflab import common
from perflab import export as export_lib
from perflab import known_failures
from perflab import metrics as metrics_lib
from perflab import mode as mode_lib
from perflab import target as target_lib
from tests import seed_test_utils


class ExportTest(seed_test_utils.RepeatableTest):

  def test_benchmark_data_creation(self):
    perf_metrics = metrics_lib.PerformanceMetrics(
        e2e_wall_time_seconds=10.0,
        post_warmup_step_time_seconds=0.5,
    )
    data = export_lib.BenchmarkData(
        spec_name="test_model",
        platform=target_lib.Platform.CPU,
        framework=mode_lib.Framework.TORCH,
        run_mode=common.RunMode.EAGER_DEFAULT,
        succeeded=True,
        metrics=perf_metrics,
    )
    self.assertEqual(data.spec_name, "test_model")
    self.assertEqual(data.platform, target_lib.Platform.CPU)
    self.assertEqual(data.framework, mode_lib.Framework.TORCH)
    self.assertEqual(data.run_mode, common.RunMode.EAGER_DEFAULT)
    self.assertTrue(data.succeeded)
    self.assertEqual(data.metrics, perf_metrics)
    self.assertIsNone(data.microbenchmark_name)

  def test_benchmark_data_with_microbenchmark(self):
    perf_metrics = metrics_lib.PerformanceMetrics(
        e2e_wall_time_seconds=12.5,
    )
    data = export_lib.BenchmarkData(
        spec_name="llama3_8b_forward",
        platform=target_lib.Platform.V6E_1X1,
        framework=mode_lib.Framework.TORCHAX,
        run_mode=common.RunMode.COMPILED,
        succeeded=False,
        metrics=perf_metrics,
        microbenchmark_name="prefill",
    )
    self.assertEqual(data.spec_name, "llama3_8b_forward")
    self.assertEqual(data.platform, target_lib.Platform.V6E_1X1)
    self.assertEqual(data.framework, mode_lib.Framework.TORCHAX)
    self.assertEqual(data.run_mode, common.RunMode.COMPILED)
    self.assertFalse(data.succeeded)
    self.assertEqual(data.metrics, perf_metrics)
    self.assertEqual(data.microbenchmark_name, "prefill")

  @mock.patch.object(export_adapter, "export_benchmark_results")
  def test_export(self, mock_export_benchmark_results):
    perf_metrics = metrics_lib.PerformanceMetrics(
        e2e_wall_time_seconds=10.0,
        post_warmup_step_time_seconds=0.5,
        peak_device_memory_mb=1024.0,
    )
    data = export_lib.BenchmarkData(
        spec_name="test_model",
        platform=target_lib.Platform.CPU,
        framework=mode_lib.Framework.TORCH,
        run_mode=common.RunMode.EAGER_DEFAULT,
        succeeded=True,
        metrics=perf_metrics,
    )
    export_lib.export(data)
    mock_export_benchmark_results.assert_called_once_with(
        spec_name="test_model",
        platform="cpu",
        framework="torch",
        run_mode="eager_default",
        metrics=perf_metrics.metric_map(),
        succeeded=True,
        wall_time=10.0,
        microbenchmark_name=None,
    )

  @mock.patch.object(export_adapter, "export_benchmark_results")
  def test_export_with_microbenchmark(self, mock_export_benchmark_results):
    perf_metrics = metrics_lib.PerformanceMetrics(
        e2e_wall_time_seconds=12.5,
        post_warmup_step_time_seconds=0.25,
    )
    data = export_lib.BenchmarkData(
        spec_name="llama3_8b_forward",
        platform=target_lib.Platform.V6E_1X1,
        framework=mode_lib.Framework.TORCHAX,
        run_mode=common.RunMode.COMPILED,
        succeeded=False,
        metrics=perf_metrics,
        microbenchmark_name="prefill",
    )
    export_lib.export(data)
    mock_export_benchmark_results.assert_called_once_with(
        spec_name="llama3_8b_forward",
        platform="v6e_1x1",
        framework="torchax",
        run_mode="compiled",
        metrics=perf_metrics.metric_map(),
        succeeded=False,
        wall_time=12.5,
        microbenchmark_name="prefill",
    )

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


if __name__ == "__main__":
  absltest.main()
