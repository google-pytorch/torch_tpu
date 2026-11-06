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

"""Tests for perflab/workload.py and the throughput metrics."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from perflab import metrics as metrics_lib
from perflab import step_lib
from perflab import workload as workload_lib
from tests import seed_test_utils

_FORWARD = step_lib.StepperType.FORWARD
_TRAINING = step_lib.StepperType.TRAINING
_DECODE = step_lib.StepperType.DECODER_ONLY_DECODE


class WorkloadTest(seed_test_utils.RepeatableTest):

  def test_with_batch_size(self):
    self.assertEqual(
        workload_lib.Workload(16, 2048).with_batch_size(4),
        workload_lib.Workload(4, 2048),
    )

  def test_require_seq_len(self):
    self.assertEqual(workload_lib.Workload(2, 128).require_seq_len(), 128)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=test
        ValueError, "declares no seq_len"
    ):
      workload_lib.Workload(2).require_seq_len()

  def test_invalid_workload(self):
    with self.assertRaisesRegex(
        ValueError, "batch_size"
    ):  # ASSERT_RAISES_OK=test
      workload_lib.Workload(batch_size=0)
    with self.assertRaisesRegex(ValueError, "seq_len"):  # ASSERT_RAISES_OK=test
      workload_lib.Workload(batch_size=1, seq_len=0)


class StepWorkTest(seed_test_utils.RepeatableTest):

  def test_forward(self):
    workload = workload_lib.Workload(batch_size=4, seq_len=128)
    self.assertEqual(
        workload_lib.step_work(workload, _FORWARD, {}),
        workload_lib.StepWork(
            workload=workload, samples_per_step=4, tokens_per_step=512
        ),
    )

  def test_training_accumulates_micro_batches(self):
    workload = workload_lib.Workload(batch_size=4, seq_len=128)
    self.assertEqual(
        workload_lib.step_work(workload, _TRAINING, {"accum_steps": 3}),
        workload_lib.StepWork(
            workload=workload, samples_per_step=12, tokens_per_step=1536
        ),
    )

  def test_samples_without_tokens(self):
    workload = workload_lib.Workload(batch_size=32)
    self.assertEqual(
        workload_lib.step_work(workload, _FORWARD, {}),
        workload_lib.StepWork(
            workload=workload, samples_per_step=32, tokens_per_step=None
        ),
    )

  def test_decode_counts_generated_tokens(self):
    workload = workload_lib.Workload(batch_size=4, seq_len=2048)
    self.assertEqual(
        workload_lib.step_work(workload, _DECODE, {}),
        workload_lib.StepWork(
            workload=workload,
            samples_per_step=None,
            tokens_per_step=4 * step_lib.DEFAULT_DECODE_OUTPUT_TOKENS,
        ),
    )
    self.assertEqual(
        workload_lib.step_work(workload, _DECODE, {"output_tokens": 8}),
        workload_lib.StepWork(
            workload=workload, samples_per_step=None, tokens_per_step=32
        ),
    )


class CheckBuiltInputsTest(seed_test_utils.RepeatableTest):

  @parameterized.named_parameters(
      (
          "token_ids",
          workload_lib.Workload(4, 128),
          ("model", (), {"input_ids": torch.zeros(4, 128, dtype=torch.long)}),
      ),
      (
          "inputs_embeds",
          workload_lib.Workload(2, 16),
          ("model", (), {"inputs_embeds": torch.zeros(2, 16, 32)}),
      ),
      (
          "decoder_token_ids",
          workload_lib.Workload(16, 448),
          (
              "model",
              (),
              {
                  "input_features": torch.zeros(16, 80, 3000),
                  "decoder_input_ids": torch.zeros(16, 448),
              },
          ),
      ),
      (
          "packed_token_ids",
          workload_lib.Workload(2, 1024),
          ("model", (torch.zeros(2048, dtype=torch.long),), {}),
      ),
      (
          "images",
          workload_lib.Workload(32),
          ("model", (torch.zeros(32, 3, 224, 224),), {}, "optimizer"),
      ),
      (
          "nested",
          workload_lib.Workload(8),
          ("model", (), {"inputs": [{"pixel_values": torch.zeros(8, 3)}]}),
      ),
  )
  def test_matches(self, workload, benchmark_args):
    self.assertIsNone(workload_lib.check_built_inputs(workload, benchmark_args))

  @parameterized.named_parameters(
      (
          "other_batch_size",
          workload_lib.Workload(16, 128),
          ("model", (), {"input_ids": torch.zeros(4, 128, dtype=torch.long)}),
      ),
      (
          "other_seq_len",
          workload_lib.Workload(4, 2048),
          ("model", (), {"input_ids": torch.zeros(4, 512, dtype=torch.long)}),
      ),
      (
          "other_image_batch",
          workload_lib.Workload(8),
          ("model", (torch.zeros(32, 3, 224, 224),), {}),
      ),
      ("no_tensors", workload_lib.Workload(1), ("model", (1, 2), {"x": "y"})),
  )
  def test_mismatches(self, workload, benchmark_args):
    mismatch = workload_lib.check_built_inputs(workload, benchmark_args)
    self.assertIsNotNone(mismatch)
    self.assertIn(str(workload), mismatch)


class ThroughputMetricsTest(seed_test_utils.RepeatableTest):

  def test_throughput_from_wall_and_device_time(self):
    metrics = metrics_lib.PerformanceMetrics(
        post_warmup_step_time_seconds=0.5,
        average_post_warmup_device_time_seconds=0.25,
        step_work=workload_lib.StepWork(
            workload=workload_lib.Workload(batch_size=4, seq_len=128),
            samples_per_step=4,
            tokens_per_step=512,
        ),
    )
    metric_map = metrics.metric_map()
    expected = {
        "batch_size": 4.0,
        "seq_len": 128.0,
        "samples_per_step": 4.0,
        "tokens_per_step": 512.0,
        "samples_per_second": 8.0,
        "device_samples_per_second": 16.0,
        "tokens_per_second": 1024.0,
        "device_tokens_per_second": 2048.0,
    }
    self.assertEqual({key: metric_map.get(key) for key in expected}, expected)

  def test_no_device_throughput_without_device_time(self):
    metrics = metrics_lib.PerformanceMetrics(
        post_warmup_step_time_seconds=2.0,
        step_work=workload_lib.StepWork(
            workload=workload_lib.Workload(batch_size=8),
            samples_per_step=8,
            tokens_per_step=None,
        ),
    )
    metric_map = metrics.metric_map()
    self.assertEqual(metric_map["samples_per_second"], 4.0)
    for key in (
        "seq_len",
        "tokens_per_step",
        "tokens_per_second",
        "device_samples_per_second",
    ):
      self.assertNotIn(key, metric_map)

  def test_decode_exports_only_tokens(self):
    metrics = metrics_lib.PerformanceMetrics(
        post_warmup_step_time_seconds=0.5,
        step_work=workload_lib.StepWork(
            workload=workload_lib.Workload(batch_size=2, seq_len=2048),
            samples_per_step=None,
            tokens_per_step=32,
        ),
    )
    metric_map = metrics.metric_map()
    self.assertEqual(metric_map["tokens_per_second"], 64.0)
    self.assertNotIn("samples_per_second", metric_map)

  def test_no_throughput_without_step_work(self):
    metric_map = metrics_lib.PerformanceMetrics(
        post_warmup_step_time_seconds=0.5
    ).metric_map()
    for key in ("batch_size", "samples_per_second", "tokens_per_second"):
      self.assertNotIn(key, metric_map)


if __name__ == "__main__":
  absltest.main()
