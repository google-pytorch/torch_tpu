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


class IssueTensorInForwardTest(absltest.TestCase):

  def test_tensor_in_forward(self):
    device = torch.device("tpu")

    class MinModel(torch.nn.Module):

      def forward(self, x):
        # Creating a tensor inside the forward pass of a compiled model
        t_steps = torch.tensor([0.1, 0.5, 0.9], device=x.device)
        return x * t_steps[0]

    model = MinModel().to(device)
    model = torch.compile(model)

    x = torch.randn(2, 2, device=device)
    out = model(x)
    self.assertIsNotNone(out)


if __name__ == "__main__":
  absltest.main()
