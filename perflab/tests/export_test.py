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

from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from torch_tpu._internal.benchmarks import export_adapter
from perflab import common
from perflab import export as export_lib
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


# TODO(wan): revisit DECISION (b/568592967): the MLCompass upload is the I/O
# edge only in google3; the OSS export_adapter writes TensorBoard events
# instead, so these tests are skipped there rather than faking a different
# boundary per platform.
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


if __name__ == "__main__":
  absltest.main()
