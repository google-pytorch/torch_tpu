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

from absl import flags  # pylint: disable=unused-import  # required for VLOG
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.nn import attention
from torch_tpu._internal.utils import utils


class BackwardsTest(parameterized.TestCase):

  def test_sdpa_backward_math(self):
    torch.manual_seed(42)
    device = torch.accelerator.current_accelerator()
    cpu_device = torch.device('cpu')

    # create inputs
    query_cpu = torch.randn(2, 2, requires_grad=True, device=cpu_device)
    key_cpu = torch.randn(2, 2, requires_grad=True, device=cpu_device)
    value_cpu = torch.randn(2, 2, requires_grad=True, device=cpu_device)
    query_tpu = query_cpu.to(device).detach().requires_grad_(True)
    key_tpu = key_cpu.to(device).detach().requires_grad_(True)
    value_tpu = value_cpu.to(device).detach().requires_grad_(True)

    # compute forward pass
    # cpu (using math backend)
    with attention.sdpa_kernel(attention.SDPBackend.MATH):
      sdpa_cpu = torch.nn.functional.scaled_dot_product_attention(
          query_cpu, key_cpu, value_cpu, is_causal=True
      )

    # tpu (using math backend)
    with attention.sdpa_kernel(attention.SDPBackend.MATH):
      sdpa_tpu = torch.nn.functional.scaled_dot_product_attention(
          query_tpu, key_tpu, value_tpu, is_causal=True
      )

    # Create fake gradients
    sdpa_fake_grad_cpu = torch.randn_like(sdpa_cpu, device=cpu_device)
    sdpa_fake_grad_tpu = sdpa_fake_grad_cpu.to(device)

    # Do backwards pass from fake gradients
    sdpa_cpu.backward(gradient=sdpa_fake_grad_cpu)
    sdpa_tpu.backward(gradient=sdpa_fake_grad_tpu)

    # Check that gradients are close
    self.assertIsNotNone(query_cpu.grad)
    self.assertIsNotNone(query_tpu.grad)
    utils.assert_close(
        query_cpu.grad,
        query_tpu.grad.cpu(),
        rtol=7.7e-3,
        atol=0.3,
        check_value=utils.CheckValueMode.LOOSE,
    )

    self.assertIsNotNone(key_cpu.grad)
    self.assertIsNotNone(key_tpu.grad)
    utils.assert_close(
        key_cpu.grad,
        key_tpu.grad.cpu(),
        rtol=7e-3,
        atol=2e-2,
        check_value=utils.CheckValueMode.LOOSE,
    )

    self.assertIsNotNone(value_cpu.grad)
    self.assertIsNotNone(value_tpu.grad)
    utils.assert_close(
        value_cpu.grad,
        value_tpu.grad.cpu(),
        rtol=1.7e-3,
        atol=2.2e-3,
    )

  @parameterized.product(
      dtype=[torch.float32, torch.bfloat16], is_causal=[True, False]
  )
  def test_sdpa_backward_overrideable(self, dtype, is_causal):
    device = torch.accelerator.current_accelerator()
    cpu_device = torch.device('cpu')

    # create inputs
    query_cpu = torch.randn(
        16, 4, 1024, 128, requires_grad=True, device=cpu_device, dtype=dtype
    )
    key_cpu = torch.randn(
        16, 4, 1024, 128, requires_grad=True, device=cpu_device, dtype=dtype
    )
    value_cpu = torch.randn(
        16, 4, 1024, 128, requires_grad=True, device=cpu_device, dtype=dtype
    )
    query_tpu = query_cpu.to(device).detach().requires_grad_(True)
    key_tpu = key_cpu.to(device).detach().requires_grad_(True)
    value_tpu = value_cpu.to(device).detach().requires_grad_(True)

    # compute forward pass
    # cpu (using math backend)
    with attention.sdpa_kernel(attention.SDPBackend.MATH):
      sdpa_cpu = torch.nn.functional.scaled_dot_product_attention(
          query_cpu, key_cpu, value_cpu, is_causal=is_causal
      )

    # tpu (using overrideable backend)
    with attention.sdpa_kernel(attention.SDPBackend.OVERRIDEABLE):
      sdpa_tpu = torch.nn.functional.scaled_dot_product_attention(
          query_tpu, key_tpu, value_tpu, is_causal=is_causal
      )

    # Create fake gradients
    sdpa_fake_grad_cpu = torch.ones_like(sdpa_cpu, device=cpu_device)
    sdpa_fake_grad_tpu = torch.ones_like(sdpa_tpu, device=device)

    # CPU backwards pass works
    sdpa_cpu.backward(gradient=sdpa_fake_grad_cpu)

    # TPU backwards pass works
    sdpa_tpu.backward(gradient=sdpa_fake_grad_tpu)

    utils.assert_close(
        query_tpu.grad.cpu(),
        query_cpu.grad,
        rtol=1e-2,
        atol=3e-2,
        check_value=utils.CheckValueMode.LOOSE,
    )

    utils.assert_close(
        key_tpu.grad.cpu(),
        key_cpu.grad,
        rtol=1e-2,
        atol=3e-2,
        check_value=utils.CheckValueMode.LOOSE,
    )
    utils.assert_close(
        value_tpu.grad.cpu(),
        value_cpu.grad,
        rtol=1e-2,
        atol=2e-2,
        check_value=utils.CheckValueMode.LOOSE,
    )

  @absltest.skip(
      "b/437527594 - This won't run until we support generators for rng."
  )
  def test_matmul_backward(self):
    torch.manual_seed(42)
    device = torch.accelerator.current_accelerator()
    cpu_device = torch.device('cpu')

    # tpu
    x = torch.randn(2, 2, requires_grad=True, device=device)
    y = torch.randn(2, 2, requires_grad=True, device=device)
    z = x.matmul(y)
    z_rand = torch.randn_like(z)
    z.backward(z_rand)
    self.assertIsNotNone(x.grad)
    self.assertIsNotNone(y.grad)

    # cpu
    x_cpu = x.detach().to(cpu_device).requires_grad_(True)
    y_cpu = y.detach().to(cpu_device).requires_grad_(True)
    z_cpu = x_cpu.matmul(y_cpu)
    z_rand_cpu = z_rand.to(cpu_device)
    z_cpu.backward(z_rand_cpu)
    self.assertIsNotNone(x_cpu.grad)
    self.assertIsNotNone(y_cpu.grad)

    x_grad_cpu = x.grad.cpu()
    y_grad_cpu = y.grad.cpu()
    utils.assert_close(x_grad_cpu, x_cpu.grad)
    utils.assert_close(y_grad_cpu, y_cpu.grad)

  @absltest.skip(
      "b/437527594 - This won't run until we support generators for rng."
  )
  def test_copy_grad(self):
    torch.manual_seed(42)
    device = torch.accelerator.current_accelerator()
    cpu_device = torch.device('cpu')

    x = torch.randn(2, 2, requires_grad=True, device=device)
    y = torch.randn(2, 2, requires_grad=True, device=device)
    z = x.matmul(y)
    z_rand = torch.randn_like(z)
    z.backward(z_rand)
    x_cpu = x.to(cpu_device)
    y_cpu = y.to(cpu_device)
    x_grad_cpu = x.grad.cpu()
    y_grad_cpu = y.grad.cpu()
    self.assertIsNotNone(x_cpu.grad)
    self.assertIsNotNone(y_cpu.grad)
    utils.assert_close(x_grad_cpu, x_cpu.grad)
    utils.assert_close(y_grad_cpu, y_cpu.grad)

  @absltest.skip(
      "b/437527594 - This won't run until we support generators for rng."
  )
  def test_optimizer_loop(self):
    torch.manual_seed(42)
    device = torch.accelerator.current_accelerator()
    cpu_device = torch.device('cpu')

    # TODO(b/435215740): Remove these lines.
    x = torch.randn(2, 2, requires_grad=True, device=device)
    x.matmul(x).backward(x)
    x.grad.cpu()

    tpu_model = torch.nn.Linear(10, 5).to(device)
    tpu_optimizer = torch.optim.SGD(tpu_model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    tpu_input = torch.randn(16, 10, device=device)
    tpu_target = torch.randn(16, 5, device=device)

    cpu_model = torch.nn.Linear(10, 5).to(cpu_device)
    cpu_model.load_state_dict(tpu_model.state_dict())
    cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=0.1)
    cpu_input = tpu_input.to(cpu_device)
    cpu_target = tpu_target.to(cpu_device)

    num_steps = 5
    for _ in range(num_steps):
      tpu_optimizer.zero_grad()
      tpu_output = tpu_model(tpu_input)
      tpu_loss = loss_fn(tpu_output, tpu_target)
      tpu_loss.backward()
      tpu_optimizer.step()

      cpu_optimizer.zero_grad()
      cpu_output = cpu_model(cpu_input)
      cpu_loss = loss_fn(cpu_output, cpu_target)
      cpu_loss.backward()
      cpu_optimizer.step()

    tpu_weight_cpu = tpu_model.weight.to(cpu_device)
    tpu_bias_cpu = tpu_model.bias.to(cpu_device)

    utils.assert_close(tpu_weight_cpu, cpu_model.weight, atol=1e-3, rtol=6e-5)
    utils.assert_close(tpu_bias_cpu, cpu_model.bias, atol=5e-4, rtol=4e-3)


if __name__ == '__main__':
  absltest.main()
