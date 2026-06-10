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

import os
from absl.testing import absltest
import torch
from torch_tpu._internal.utils import utils


class SortLoopedTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Force the use of looped sort for all tests in this class
    os.environ['TORCH_TPU_INTERNAL_USE_STABLEHLO_SORT'] = '0'

  def tearDown(self):
    # Restore default behavior
    os.environ.pop('TORCH_TPU_INTERNAL_USE_STABLEHLO_SORT', None)
    super().tearDown()

  def test_sort_looped_basic(self):
    device = torch.device('tpu')
    # Use a power-of-2 size since our LLO implementation requires it.
    n = 1024

    # Generate random float32 tensor on CPU and copy to TPU
    x_cpu = torch.randn(n, dtype=torch.float32)
    x_tpu = x_cpu.to(device)

    # Call standard torch.sort (which should dispatch to looped sort)
    values_tpu, indices_tpu = torch.sort(x_tpu)

    # Copy results back to CPU for verification
    values_cpu = values_tpu.to('cpu')
    indices_cpu = indices_tpu.to('cpu')

    # Verify values are sorted
    expected_values_cpu, _ = torch.sort(x_cpu)
    utils.assert_close(values_cpu, expected_values_cpu)

    # Verify indices are dummy (iota: 0, 1, 2, ..., N-1)
    expected_indices_cpu = torch.arange(n, dtype=torch.int64)
    utils.assert_close(indices_cpu, expected_indices_cpu)

  def test_sort_looped_large(self):
    device = torch.device('tpu')
    n = 65536  # 64K

    x_cpu = torch.randn(n, dtype=torch.float32)
    x_tpu = x_cpu.to(device)

    values_tpu, indices_tpu = torch.sort(x_tpu)

    values_cpu = values_tpu.to('cpu')
    indices_cpu = indices_tpu.to('cpu')

    expected_values_cpu, _ = torch.sort(x_cpu)
    utils.assert_close(values_cpu, expected_values_cpu)

    expected_indices_cpu = torch.arange(n, dtype=torch.int64)
    utils.assert_close(indices_cpu, expected_indices_cpu)


if __name__ == '__main__':
  absltest.main()
