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

from absl.testing import absltest
import torch
import torch_tpu  # noqa: F401

from tests import seed_test_utils


class ExperimentalCollectivesTest(seed_test_utils.RepeatableTest):

  def test_meta_shapes_preserve_collective_dimension(self):
    value = torch.empty((2, 3, 4), device="meta")

    gathered = torch.ops.tpu.experimental_all_gather_dim(
        value, 1, 8, "unused"
    )
    scattered = torch.ops.tpu.experimental_reduce_scatter_dim(
        gathered, -2, 8, "unused"
    )

    self.assertEqual(gathered.shape, (2, 24, 4))
    self.assertEqual(scattered.shape, value.shape)

  def test_meta_reduce_scatter_rejects_an_uneven_dimension(self):
    value = torch.empty((2, 3, 4), device="meta")

    with self.assertRaisesRegex(RuntimeError, "not divisible"):
      torch.ops.tpu.experimental_reduce_scatter_dim(value, 1, 2, "unused")

  def test_meta_rejects_an_invalid_dimension(self):
    value = torch.empty((2, 3, 4), device="meta")

    with self.assertRaisesRegex(RuntimeError, "invalid"):
      torch.ops.tpu.experimental_all_gather_dim(value, 3, 2, "unused")

  def test_meta_rejects_a_nonpositive_group_size(self):
    value = torch.empty((2, 3, 4), device="meta")

    with self.assertRaisesRegex(RuntimeError, "must be positive"):
      torch.ops.tpu.experimental_all_gather_dim(value, 1, 0, "unused")


if __name__ == "__main__":
  absltest.main()
