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

"""Tests that zero-size tensors work correctly.

'Zero-size' means tensors with 0 total elements. This is distinct from both
0-dimensional tensors (e.g. scalars) which have 1 element, and empty
tensors (e.g. torch.empty(1, 2, 3)), which represent a non-zero number of
elements but have uninitialized memory.
"""

from absl.testing import absltest
import torch


class ZeroSizeTest(absltest.TestCase):

  def test_copy_round_trip(self):
    """Tests that we can move a zero-size tensor between devices."""
    tpu_device = torch.accelerator.current_accelerator()
    cpu_tensor_original = torch.zeros(0)
    tpu_tensor = cpu_tensor_original.to(tpu_device)
    cpu_tensor_return = tpu_tensor.to("cpu")
    self.assertEqual(cpu_tensor_return.numel(), 0)

  def test_construct_on_tpu(self):
    """Tests that we can construct a zero-size tensor on TPU."""
    tpu_device = torch.accelerator.current_accelerator()
    tpu_tensor = torch.zeros(0, device=tpu_device)
    self.assertEqual(tpu_tensor.numel(), 0)

  def test_zero_size_output(self):
    """Tests that we can get a zero-size tensor as output."""
    tpu_device = torch.accelerator.current_accelerator()
    lhs = torch.ones(0, 2, device=tpu_device)
    rhs = torch.ones(2, 3, device=tpu_device)
    out = lhs.mm(rhs)
    self.assertEqual(out.shape, (0, 3))
    self.assertEqual(out.numel(), 0)


if __name__ == "__main__":
  absltest.main()
