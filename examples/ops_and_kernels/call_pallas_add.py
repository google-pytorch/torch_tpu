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

"""Example demonstrating calling a Pallas kernel from PyTorch on TPU."""

from absl.testing import absltest
import torch
from examples.ops_and_kernels import torch_pallas_add
from tests import seed_test_utils


class CallPallasAddTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")

  def test_eager_execution(self):
    a = torch.tensor([1.0, 2.0, 3.0], device=self.device)
    b = torch.tensor([4.0, 5.0, 6.0], device=self.device)
    result = torch_pallas_add.pallas_add_vectors(a, b)
    expected = torch.tensor([5.0, 7.0, 9.0], device=self.device)
    torch.testing.assert_close(result, expected)

  def test_compiled_execution(self):
    @torch.compile(backend="tpu")
    def my_model(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
      return torch_pallas_add.pallas_add_vectors(x, y) * 2.0

    a = torch.tensor([1.0, 2.0, 3.0], device=self.device)
    b = torch.tensor([4.0, 5.0, 6.0], device=self.device)
    result = my_model(a, b)
    expected = torch.tensor([10.0, 14.0, 18.0], device=self.device)
    torch.testing.assert_close(result, expected)


if __name__ == "__main__":
  absltest.main()
