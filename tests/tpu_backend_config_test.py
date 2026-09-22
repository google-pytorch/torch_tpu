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

"""Tests for torch.tpu excess precision and verification that torch.backends.tpu is removed."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal import execution_mode as em
from tests import seed_test_utils


class TpuBackendConfigTest(seed_test_utils.RepeatableTest):
  """Tests for torch.tpu related configs."""

  @parameterized.named_parameters(
      dict(
          testcase_name="true",
          allow=True,
          expected_value=1.01568603515625,
      ),
      dict(testcase_name="false", allow=False, expected_value=1.015625),
  )
  def test_allow_excess_precision(self, allow: bool, expected_value: float):
    """Tests global torch.tpu.allow_excess_precision API."""
    # Default value is False.
    self.assertIs(torch.tpu.allow_excess_precision, False)

    torch.tpu.allow_excess_precision = allow
    self.assertIs(torch.tpu.allow_excess_precision, allow)

    device = torch.device("tpu")
    eps: float = torch.finfo(torch.bfloat16).eps
    x_element = 1.0 + eps
    x = torch.tensor(
        [[x_element, 0.0], [0.0, x_element]],
        device=device,
        dtype=torch.bfloat16,
    )

    def run_compute():
      with em.set_eager_mode(em.EagerMode.DEFER_AND_FUSE):
        intermediate = torch.mm(x, x)
        output = intermediate * torch.tensor(
            [1.0], dtype=torch.float32, device=device
        )
      return output[0, 0]

    self.assertEqual(run_compute().cpu().item(), expected_value)


if __name__ == "__main__":
  absltest.main()
