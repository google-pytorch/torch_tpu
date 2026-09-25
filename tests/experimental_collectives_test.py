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

  def test_meta_shapes(self):
    value = torch.empty((2, 3, 4), device="meta")
    gathered = torch.ops.tpu.experimental_all_gather_dim(value, 1, 8, "unused")
    scattered = torch.ops.tpu.experimental_reduce_scatter_dim(
        gathered, -2, 8, "unused"
    )
    self.assertEqual(gathered.shape, (2, 24, 4))
    self.assertEqual(scattered.shape, value.shape)

  def test_meta_errors(self):
    value = torch.empty((2, 3, 4), device="meta")
    all_gather = torch.ops.tpu.experimental_all_gather_dim
    reduce_scatter = torch.ops.tpu.experimental_reduce_scatter_dim
    for op, x, dim, group_size, error in (
        (reduce_scatter, value, 1, 2, "not divisible"),
        (all_gather, value, 3, 2, "invalid"),
        (reduce_scatter, value, -4, 2, "invalid"),
        (all_gather, value, 1, 0, "must be positive"),
        (all_gather, torch.empty((), device="meta"), 0, 2, "non-scalar"),
    ):
      with self.subTest(op=op, dim=dim, group_size=group_size):
        with self.assertRaisesRegex(RuntimeError, error):
          op(x, dim, group_size, "unused")


if __name__ == "__main__":
  absltest.main()
