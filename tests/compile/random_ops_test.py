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

"""Tests for RNG state and random operations under compile mode."""

from typing import Any, Callable
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.utils import _pytree
from torch_tpu._internal.compile.debug import TpuCompileDebug
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


def _get_cpu_tensors(tree: Any):
  outs, _ = _pytree.tree_flatten(tree)
  return [o.cpu() for o in outs]


class RandomOpsTest(seed_test_utils.RepeatableTest):

  def assert_random_outputs_close_with_same_seed(
      self,
      runner: Callable[[], torch.Tensor],
      generator: torch.Generator | None = None,
      atol: float = 1e-4,
      rtol: float = 1e-6,
      num_samples: int = 5,
  ):
    for seed in range(num_samples):
      if generator:
        generator.manual_seed(seed)
      else:
        torch.manual_seed(seed)
      prev_outs = _get_cpu_tensors(runner())

      if generator:
        generator.manual_seed(seed)
      else:
        torch.manual_seed(seed)
      curr_outs = _get_cpu_tensors(runner())

      if seed < 2:
        continue
      utils.assert_close(curr_outs, prev_outs, atol=atol, rtol=rtol)

  def assert_random_outputs_not_close_with_different_seeds(
      self,
      runner: Callable[[], torch.Tensor],
      generator: torch.Generator | None = None,
      atol: float = 1e-4,
      rtol: float = 1e-6,
      num_samples: int = 5,
  ):
    prev_outs = None
    for seed in range(num_samples):
      if generator:
        generator.manual_seed(seed)
      else:
        torch.manual_seed(seed)
      curr_outs = _get_cpu_tensors(runner())
      if prev_outs is not None:
        with self.assertRaises(AssertionError):
          utils.assert_close(curr_outs, prev_outs, atol=atol, rtol=rtol)
      prev_outs = curr_outs

  def assert_random_outputs(
      self,
      runner: Callable[[], torch.Tensor],
      generator: torch.Generator | None = None,
      atol: float = 1e-4,
      rtol: float = 1e-6,
      num_samples: int = 5,
  ):
    self.assert_random_outputs_close_with_same_seed(
        runner,
        generator=generator,
        atol=atol,
        rtol=rtol,
        num_samples=num_samples,
    )
    self.assert_random_outputs_not_close_with_different_seeds(
        runner,
        generator=generator,
        atol=atol,
        rtol=rtol,
        num_samples=num_samples,
    )

  def test_dropout_compile(self):
    torch.manual_seed(42)
    device = torch.device("tpu")

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

    self.assert_random_outputs(runner)

  def test_dropout_compile_with_former_eager_run(self):
    torch.manual_seed(42)
    device = torch.device("tpu")

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

    self.assert_random_outputs(runner)

  @parameterized.named_parameters(
      ("rand", lambda x: torch.rand(x.shape, device=x.device)),
      ("randn", lambda x: torch.randn(x.shape, device=x.device)),
      ("bernoulli", lambda x: x.bernoulli_(0.5)),
      ("bernoulli_tensor", lambda x: x.bernoulli_(torch.full_like(x, 0.5))),
      ("bernoulli_functional", torch.bernoulli),
      ("bernoulli_out", lambda x: torch.bernoulli(x, out=torch.zeros_like(x))),
      ("exponential", lambda x: x.exponential_()),
  )
  def test_random_ops_compile(self, op_fn):
    device = torch.device("tpu")

    class MyModule(torch.nn.Module):

      def forward(self, x):
        return op_fn(x)

    module = MyModule().to(device)
    compiled_module = torch.compile(module)
    x = torch.full((2, 5), 0.5, device=device)

    def runner():
      nonlocal x
      return compiled_module(x)

    self.assert_random_outputs(runner)

  def test_multiple_random_ops_compile(self):
    device = torch.device("tpu")

    class MyModule(torch.nn.Module):

      def forward(self, x):
        y = torch.rand(x.shape, device=x.device)
        z = torch.randn(x.shape, device=x.device)
        return x + y + z

    module = MyModule().to(device)
    compiled_module = torch.compile(module)
    x = torch.zeros(2, 5, device=device)

    def runner():
      return compiled_module(x)

    self.assert_random_outputs(runner)

  def test_explicit_generator_compile(self):
    device = torch.device("tpu")
    gen = torch.Generator(device=device)

    class MyModule(torch.nn.Module):

      def forward(self, x, gen):
        return torch.randn(x.shape, device=x.device, generator=gen) + x

    module = MyModule().to(device)

    # NOTE: Passing a custom generator to a compiled module causes an expected
    # graph break in Dynamo because there is no `as_proxy()` implementation
    # for custom `torch.Generator` objects. As a result, Dynamo falls back to
    # eager mode for the random operation, and the test passes using eager.
    compiled_module = torch.compile(module)
    x = torch.zeros(2, 5, device=device)

    def runner():
      return compiled_module(x, gen)

    self.assert_random_outputs(runner, generator=gen)

  def test_no_rng_state_update_compile(self):
    debugs: list[TpuCompileDebug] = []

    class MyModule(torch.nn.Module):

      def forward(self, x):
        return x + 10.0

    device = torch.accelerator.current_accelerator()
    module = MyModule().to(device)
    compiled_module = torch.compile(
        module,
        backend="tpu",
        options={"serializable": False, "debug_callback": debugs.append},
    )

    x = torch.zeros(2, 5, device=device)
    compiled_module(x)

    self.assertLen(debugs, 1)
    self.assertNotEmpty(debugs[0].compiled_executables)
    for executable in debugs[0].compiled_executables:
      self.assertFalse(executable.updates_default_generator_state())

  def test_eager_vs_compile_numerics(self):
    device = torch.device("tpu")
    x = torch.randn(2, 5, device=device)

    class MyModule(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)

      def forward(self, x):
        return self.dropout(x)

    module = MyModule().to(device)
    compiled_module = torch.compile(module)

    # 1. Run eager mode seeded with a specific seed
    torch.manual_seed(42)
    eager_out = module(x)

    # 2. Run compiled mode seeded with the same seed
    torch.manual_seed(42)
    compiled_out = compiled_module(x)

    # The outputs must be numerically identical.
    utils.assert_close(eager_out.cpu(), compiled_out.cpu())


if __name__ == "__main__":
  absltest.main()
