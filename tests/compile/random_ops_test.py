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

"""Test for activation checkpointing with RNG state under compile mode."""

from typing import Callable
import unittest
from absl.testing import absltest
import torch
import torch.accelerator
from torch.utils import checkpoint
from torch_tpu._internal.utils import utils


class RandomOpsTest(absltest.TestCase):

  def assert_outputs_not_close(
      self,
      runner: Callable[[], torch.Tensor],
      atol: float = 1e-4,
      rtol: float = 1e-6,
      num_samples: int = 5,
  ):
    prev_out = None
    for _ in range(num_samples):
      out = runner()
      out = out.cpu()
      if prev_out is not None:
        self.assertFalse(torch.allclose(out, prev_out, atol=atol, rtol=rtol))
      prev_out = out

  @unittest.skip("b/496168350")
  def test_dropout_compile(self):
    torch.manual_seed(42)
    device = torch.accelerator.current_accelerator()

    class MyModule(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(5, 5)
        self.dropout = torch.nn.Dropout(p=0.5)

      def forward(self, x):
        return self.dropout(self.lin(x))

    module = MyModule().to(device)
    compiled_module = torch.compile(module)
    x = torch.randn(2, 5, device=device)

    def runner():
      nonlocal x
      return compiled_module(x)

    self.assert_outputs_not_close(runner)

  @unittest.skip("b/496168350")
  def test_dropout_compile_with_former_eager_run(self):
    torch.manual_seed(42)
    device = torch.accelerator.current_accelerator()

    class MyModule(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(5, 5)
        self.dropout = torch.nn.Dropout(p=0.5)

      def forward(self, x):
        return self.dropout(self.lin(x))

    module = MyModule().to(device)

    # Run the module eagerly before torch.compile
    out = module(torch.randn(2, 5, device=device))
    del out

    compiled_module = torch.compile(module)
    x = torch.randn(2, 5, device=device)

    def runner():
      nonlocal x
      return compiled_module(x)

    self.assert_outputs_not_close(runner)

  @unittest.skip("b/496168350")
  def test_checkpoint_rng_compile(self):
    # Arrange
    device = torch.accelerator.current_accelerator()

    class MyModule(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(5, 5)
        self.dropout = torch.nn.Dropout(p=0.5)

      def forward(self, x):
        return self.dropout(self.lin(x))

    module = MyModule().to(device)

    def run_model(model, x, use_checkpoint):
      if use_checkpoint:
        return checkpoint.checkpoint(model, x, use_reentrant=False)
      else:
        return model(x)

    x = torch.randn(2, 5, device=device).requires_grad_(True)

    # Create expected results
    torch.manual_seed(42)
    out_eager = run_model(module, x, use_checkpoint=False)
    loss_eager = out_eager.sum()
    loss_eager.backward()
    grad_x_eager = x.grad.clone()
    grad_W_eager = module.lin.weight.grad.clone()  # pylint: disable=invalid-name

    # Act. Run with compiled and activation checkpointed.
    # Reset gradients
    x.grad.zero_()
    module.lin.weight.grad.zero_()
    module.lin.bias.grad.zero_()

    # Compiled checkpointed
    compiled_run = torch.compile(run_model)

    torch.manual_seed(42)
    out_compiled = compiled_run(module, x, use_checkpoint=True)
    loss_compiled = out_compiled.sum()
    loss_compiled.backward()
    grad_x_compiled = x.grad.clone()
    grad_W_compiled = module.lin.weight.grad.clone()  # pylint: disable=invalid-name

    # Assert
    # Compare Compiled Checkpointed with Eager Non-Checkpointed
    utils.assert_close(out_compiled.cpu(), out_eager.cpu())
    utils.assert_close(grad_x_compiled.cpu(), grad_x_eager.cpu())
    utils.assert_close(grad_W_compiled.cpu(), grad_W_eager.cpu())


if __name__ == "__main__":
  absltest.main()
