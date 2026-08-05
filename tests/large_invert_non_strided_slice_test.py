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

"""Regression test for >32-bit indexing in InvertNonStridedSliceShlo."""

import random
from absl.testing import absltest
import torch
from tests import seed_test_utils


class LargeInvertNonStridedSliceTest(seed_test_utils.RepeatableTest):
  """Verifies that >INT_MAX 1D views compile via static HLO Slice/Concat ops.

  This bypasses DynamicUpdateSlice to prevent an XLA >32-bit dynamic index
  crash.
  """

  def setUp(self):
    super().setUp()
    random.seed(1234)
    torch.manual_seed(1234)
    self.device = torch.device('tpu')

  def test_exceeds_32bit_indexing(self):
    size = 2**31

    # Use bool to prevent OOM
    x = torch.zeros(size, dtype=torch.bool, device=self.device)
    grad = torch.ones(10, dtype=torch.bool, device=self.device)

    # as_strided_scatter is the explicit ATen operator for as_strided's backward
    # pass (as_strided_inverse).
    y = torch.ops.aten.as_strided_scatter.default(x, grad, (10,), (1,), 0)
    torch.tpu.synchronize()

    self.assertIsNotNone(y)
    self.assertEqual(y[0].item(), True)
    self.assertEqual(y[:10].sum().item(), 10)


if __name__ == '__main__':
  absltest.main()
