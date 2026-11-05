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

"""Tests for perflab/workload.py."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from perflab import workload as workload_lib
from tests import seed_test_utils


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


if __name__ == "__main__":
  absltest.main()
