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
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class SymExprArgTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    self.device = torch.accelerator.current_accelerator()

  def test_tpu_backend_execution(self):
    class CustomOp(torch.autograd.Function):

      @staticmethod
      def forward(ctx, x, split_size):
        ctx.split_size = split_size
        return x.clone()

      @staticmethod
      def backward(ctx, grad_out):
        split_size = ctx.split_size
        z = torch.ones(split_size, device=grad_out.device)
        return grad_out + z.sum(), None

    class Model(torch.nn.Module):

      def forward(self, x):
        return CustomOp.apply(x, x.shape[0] // 2)

    m = Model()
    compiled = torch.compile(
        m, backend="tpu", options={"bounded_dynamism": True}
    )

    # CPU Reference
    x_cpu = torch.ones(4, requires_grad=True)
    out_cpu = m(x_cpu)
    loss_cpu = out_cpu.sum()
    loss_cpu.backward()

    # TPU Execution
    x_tpu = torch.ones(4, device=self.device, requires_grad=True)
    torch._dynamo.mark_dynamic(x_tpu, 0, min=4, max=8)
    out_tpu = compiled(x_tpu)
    loss_tpu = out_tpu.sum()
    loss_tpu.backward()

    # Compare forward and backward outputs
    utils.assert_close(out_tpu.cpu(), out_cpu)
    utils.assert_close(x_tpu.grad.cpu(), x_cpu.grad)

  def test_symint_expression_placeholder_backward(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        half_s0 = x.shape[0] // 2
        sub_x = x[:half_s0]
        return sub_x * 2.0

    m = Model()

    # CPU Reference
    x_cpu = torch.ones(8, 4, requires_grad=True)
    out_cpu = m(x_cpu)
    out_cpu.sum().backward()

    # TPU Execution
    compiled = torch.compile(
        m, backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.ones(8, 4, device=self.device, requires_grad=True)
    torch._dynamo.mark_dynamic(x1, 0)

    out1 = compiled(x1)
    loss = out1.sum()
    loss.backward()

    expected_grad = torch.cat(
        [torch.ones(4, 4) * 2.0, torch.zeros(4, 4)], dim=0
    )

    utils.assert_close(out1.to("cpu"), out_cpu)
    utils.assert_close(x1.grad.to("cpu"), x_cpu.grad)
    utils.assert_close(x1.grad.to("cpu"), expected_grad)


if __name__ == "__main__":
  absltest.main()
