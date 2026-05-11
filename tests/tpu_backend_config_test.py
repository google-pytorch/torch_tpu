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

"""Tests for torch.backends.tpu related configs.

The global settings tests in this file may interfere with other tests if run in
parallel.
"""

from absl.testing import absltest
import torch
from torch.testing._internal import common_utils
from torch_tpu._internal import execution_mode as em


class TpuBackendConfigTest(absltest.TestCase, common_utils.TestCase):
  """Tests for torch.backends.tpu related configs."""

  def test_allow_excess_precision_backend(self):
    """Tests torch.backends.tpu.allow_excess_precision."""
    device = torch.device("tpu")
    eps: float = torch.finfo(torch.bfloat16).eps
    x_element = 1.0 + eps
    x = torch.tensor(
        [[x_element, 0.0], [0.0, x_element]],
        device=device,
        dtype=torch.bfloat16,
    )

    def run_compute():
      with em.eager_mode(em.EagerMode.DEFER_AND_FUSE):
        intermediate = torch.mm(x, x)
        output = intermediate * torch.tensor(
            [1.0], dtype=torch.float32, device=device
        )
      return output[0, 0]

    value_for_true_flag = 1.01568603515625
    value_for_false_flag = 1.015625

    # 1. Default should be True
    # pytype: disable=module-attr
    self.assertTrue(torch.backends.tpu.allow_excess_precision)
    self.assertEqual(run_compute().cpu().item(), value_for_true_flag)

    # 2. Set to False
    torch.backends.tpu.allow_excess_precision = False
    self.assertFalse(torch.backends.tpu.allow_excess_precision)
    self.assertEqual(run_compute().cpu().item(), value_for_false_flag)

    # 3. Set to True
    torch.backends.tpu.allow_excess_precision = True
    self.assertTrue(torch.backends.tpu.allow_excess_precision)
    self.assertEqual(run_compute().cpu().item(), value_for_true_flag)
    # pytype: enable=module-attr


if __name__ == "__main__":
  absltest.main()
