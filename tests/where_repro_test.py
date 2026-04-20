# Copyright 2025 Google LLC
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

"""Reproduction of TPU RuntimeError in torch.where."""

import torch
from torch_tpu import api
from absl.testing import absltest
from absl.testing import parameterized

class WhereReproTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.device = api.tpu_device()

  @parameterized.parameters(
      (128, 10), # Typical global_num_experts and local
      (1024, 100),
      (1, 0),    # Scalar/small case
      (128, 0),  # Empty case (no non-zero)
      (128, 128), # Full case (all non-zero)
  )
  def test_where_condition_only_1d(self, global_num_experts, local_num_experts):
    # expert_map logic from vLLM fused_moe
    expert_map_cpu = torch.full((global_num_experts,), -1, dtype=torch.int32)
    if local_num_experts > 0:
      expert_map_cpu[0:local_num_experts] = torch.arange(local_num_experts, dtype=torch.int32)

    expert_map = expert_map_cpu.to(self.device)

    # Logic from vLLM: global_indices = torch.where(expert_map != -1)[0]
    def compute(t):
      mask = (t != -1)
      # torch.where(mask) returns a tuple of tensors (one per dimension)
      res = torch.where(mask)
      return res[0]

    expected = compute(expert_map_cpu)
    actual = compute(expert_map).to("cpu")

    torch.testing.assert_close(actual, expected)

  def test_where_2d_mask(self):
    # expert_mask logic from MoE snippets
    # (num_experts, top_k, batch_seq_len)
    mask_cpu = (torch.randn(128, 8, 10) > 0)
    mask = mask_cpu.to(self.device)

    # common pattern: _, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
    res_cpu = torch.where(mask_cpu)
    res_tpu = torch.where(mask)

    self.assertEqual(len(res_cpu), len(res_tpu))
    for r_cpu, r_tpu in zip(res_cpu, res_tpu):
      torch.testing.assert_close(r_tpu.to("cpu"), r_cpu)

if __name__ == '__main__':
  absltest.main()
