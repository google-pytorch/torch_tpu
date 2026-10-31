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

"""Unit tests for ops."""

from collections.abc import Callable, Sequence
import concurrent
import contextlib
import dataclasses
import enum
import functools
import itertools
import math
import os
import plistlib
import re
import threading
import time
import typing
from typing import Any
import unittest
from unittest import mock
import warnings

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from scipy import stats
import torch
from torch._dynamo import testing as dynamo_testing
from torch._subclasses import fake_tensor
from torch.fx.experimental import symbolic_shapes
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.attention import flex_attention
from torch.testing._internal import common_methods_invocations
from torch_tpu._internal import dynamism
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync
from torch_tpu._internal.compile import compiler as compile_compiler
from torch_tpu._internal.compile import flex_attention_redirect
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.utils import test_utils as utils
from tests import op_testing
from tests import ops_test_data
from tests import oss_utils
from tests import quantize_utils

# In this file, we use the following naming convention for variables:
# - golden_*: a value for the device used for computing the golden results
#   (either CPU or GPU)
# - tpu_*: a value for the TPU device


OpInput = op_testing.OpInput
TorchTpuVsCpuTestBase = op_testing.TorchTpuVsCpuTestBase
op_db = common_methods_invocations.op_db
to = op_testing.to
CheckValueMode = utils.CheckValueMode


class FakeSample:
  """A fake test sample for an op. Used for testing op_testing.to()."""

  def __init__(
      self,
      name: str,
      input_value: torch.Tensor,
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
  ):
    self.name = name
    self.input = input_value
    self.args = args
    self.kwargs = kwargs


@contextlib.contextmanager
def set_default_dtype(dtype):
  original_dtype = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    yield
  finally:
    torch.set_default_dtype(original_dtype)


def run_with_dynamic_dims(
    fn: Callable[..., Any],
    inputs: Sequence[tuple[torch.Tensor, Sequence[bool]]],
    no_grad: bool = False,
) -> set[str]:
  """Runs `fn` on FakeTensors with backed symbolic dims.

  Args:
    fn: The function to run.
    inputs: `(tensor, dynamic)` pairs. `fn` gets a FakeTensor made from
      `tensor`, whose dim `j` is symbolic iff `dynamic[j]` is true.
    no_grad: Whether to run `fn` under `torch.no_grad()`.

  Returns:
    The names (`arg{i}.dim{j}`) of the symbolic dims that running `fn`
    specialized to a constant.
  """
  shape_env = symbolic_shapes.ShapeEnv()
  mode = fake_tensor.FakeTensorMode(shape_env=shape_env)
  fakes = []
  symbols = {}
  for i, (tensor, dynamic) in enumerate(inputs):
    context = symbolic_shapes.StatelessSymbolicContext(
        dynamic_sizes=[
            symbolic_shapes.DimDynamic.DYNAMIC
            if d
            else symbolic_shapes.DimDynamic.STATIC
            for d in dynamic
        ]
    )
    fake = mode.from_tensor(tensor, symbolic_context=context)
    fakes.append(fake)
    for j, size in enumerate(fake.shape):
      if isinstance(size, torch.SymInt):
        symbols[f"arg{i}.dim{j}"] = size.node.expr
  grad_context = torch.no_grad() if no_grad else contextlib.nullcontext()
  with mode, grad_context:
    fn(*fakes)
  return {
      name
      for name, expr in symbols.items()
      if shape_env.replace(expr).is_number
  }


class OpsUnitTest(TorchTpuVsCpuTestBase):
  """Tests for ops using custom values.

  If a bug is found that's not covered by do_test_op() in ops_test.py, please
  add it here.
  """

  @parameterized.product(
      dtype=[torch.float32, torch.float64],
      op_fn=[
          ("add", lambda s, t1, t2: torch.add(t1, t2)),
          ("div", lambda s, t1, t2: torch.div(t1, t2)),
          ("addcdiv", lambda s, t1, t2: torch.addcdiv(s, t1, t2, value=3.14)),
          ("addcmul", lambda s, t1, t2: torch.addcmul(s, t1, t2, value=3.14)),
      ],
  )
  def test_dynamic_float64_elementwise_and_fused_ops(self, dtype, op_fn):
    """Tests dynamic shapes for float64 (and f32 baseline) ops.

    Covers add, div, addcdiv, and addcmul ops (X64 split layout handling).
    """
    torch.manual_seed(42)

    # Case 1: 2D tensors (2, 4) with dimension 0 dynamic
    with self.subTest(op=op_fn[0], dtype=dtype, case="2x4_dim0_dynamic"):
      s_tpu = torch.tensor(
          [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
          dtype=dtype,
          device="tpu",
      )
      t1_tpu = torch.tensor(
          [[2.0, 8.0, 1.0, 4.0], [1.0, 0.0, 3.0, 4.0]],
          dtype=dtype,
          device="tpu",
      )
      t2_tpu = torch.tensor(
          [[1.0, 3.0, 3.0, 1.0], [1.0, 0.0, 0.0, 2.0]],
          dtype=dtype,
          device="tpu",
      )

      s_cpu = s_tpu.cpu()
      t1_cpu = t1_tpu.cpu()
      t2_cpu = t2_tpu.cpu()

      dynamism.mark_dynamic(s_tpu, dimension=1, lower_bound=2, upper_bound=15)
      dynamism.mark_dynamic(t1_tpu, dimension=1, lower_bound=2, upper_bound=15)
      dynamism.mark_dynamic(t2_tpu, dimension=1, lower_bound=2, upper_bound=15)

      out_tpu = op_fn[1](s_tpu, t1_tpu, t2_tpu)
      out_cpu = op_fn[1](s_cpu, t1_cpu, t2_cpu)

      utils.assert_close(actual=out_tpu.cpu(), expected=out_cpu)

    # Case 2: Broadcast dynamic (5, 1) with static (1, 4) -> output (5, 4)
    with self.subTest(op=op_fn[0], dtype=dtype, case="broadcast_5x1_with_1x4"):
      s_tpu = torch.rand(5, 4, dtype=dtype, device="tpu") + 0.5
      t1_tpu = torch.rand(5, 1, dtype=dtype, device="tpu") + 0.5
      t2_tpu = torch.rand(1, 4, dtype=dtype, device="tpu") + 0.5

      s_cpu = s_tpu.cpu()
      t1_cpu = t1_tpu.cpu()
      t2_cpu = t2_tpu.cpu()

      dynamism.mark_dynamic(s_tpu, dimension=0, lower_bound=2, upper_bound=15)
      dynamism.mark_dynamic(t1_tpu, dimension=0, lower_bound=2, upper_bound=15)

      out_tpu = op_fn[1](s_tpu, t1_tpu, t2_tpu)
      out_cpu = op_fn[1](s_cpu, t1_cpu, t2_cpu)

      utils.assert_close(actual=out_tpu.cpu(), expected=out_cpu)

    # Case 3: Broadcast dynamic (5, 1) with 1D static (4,) -> output (5, 4)
    with self.subTest(op=op_fn[0], dtype=dtype, case="broadcast_5x1_with_1d_4"):
      s_tpu = torch.rand(5, 4, dtype=dtype, device="tpu") + 0.5
      t1_tpu = torch.rand(5, 1, dtype=dtype, device="tpu") + 0.5
      t2_tpu = torch.rand(4, dtype=dtype, device="tpu") + 0.5

      s_cpu = s_tpu.cpu()
      t1_cpu = t1_tpu.cpu()
      t2_cpu = t2_tpu.cpu()

      dynamism.mark_dynamic(s_tpu, dimension=0, lower_bound=2, upper_bound=15)
      dynamism.mark_dynamic(t1_tpu, dimension=0, lower_bound=2, upper_bound=15)

      out_tpu = op_fn[1](s_tpu, t1_tpu, t2_tpu)
      out_cpu = op_fn[1](s_cpu, t1_cpu, t2_cpu)

      utils.assert_close(actual=out_tpu.cpu(), expected=out_cpu)

  def test_clone_inplace_mutation_preserves_original(self):
    """Verifies that an in-place mutation on a cloned tensor does not mutate or

    invalidate the original tensor's shared device buffer.
    """
    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_NEVER):

      def compute(device):
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
        x_clone = x.clone()
        x_clone.add_(1.0)
        y = x * 2.0
        return x, x_clone, y

      self.assert_close_tpu_vs_cpu(compute)

  def test_clone_original_inplace_mutation_preserves_clone(self):
    """Verifies that an in-place mutation on the original tensor does not mutate

    or invalidate a live clone that shares the same underlying buffer.
    """
    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_NEVER):

      def compute(device):
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
        x_clone = x.clone()
        x.add_(1.0)
        y = x_clone * 2.0
        return x, x_clone, y

      self.assert_close_tpu_vs_cpu(compute)

  def test_clone_inplace_mutation_with_original_as_operand(self):
    """Verifies that an in-place op taking both the clone and original tensor as

    operands (e.g. x_clone.add_(x)) does not donate the shared buffer and
    corrupt
    either operand during execution.
    """
    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_NEVER):

      def compute(device):
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
        x_clone = x.clone()
        x_clone.add_(x)
        y = x * 2.0
        return x, x_clone, y

      self.assert_close_tpu_vs_cpu(compute)

  def test_clone_deletion_allows_inplace_donation_on_original(self):
    """Verifies that deleting a clone drops live_data_ptrs back to 1, safely

    re-enabling in-place buffer donation for subsequent operations on the
    original.
    """
    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_NEVER):

      def compute(device):
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
        x_clone = x.clone()
        del x_clone
        x.add_(1.0)
        return x

      self.assert_close_tpu_vs_cpu(compute)

  def test_prelu_kernel_direct(self):
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
      # 0D scalar self, 0D scalar weight
      self_cpu_0d = torch.randn((), dtype=dtype)
      weight_cpu_0d = torch.randn((), dtype=dtype)

      def compute_0d(device, self_cpu=self_cpu_0d, weight_cpu=weight_cpu_0d):
        self_tensor = self_cpu.to(device).detach().requires_grad_(True)
        weight = weight_cpu.to(device).detach().requires_grad_(True)
        out = torch.ops.aten._prelu_kernel(self_tensor, weight)
        out.sum().backward()
        return out, self_tensor.grad, weight.grad

      self.assert_close_tpu_vs_cpu(compute_0d)

      # 1D self, 1D weight (single scalar weight)
      self_cpu_1d = torch.randn(10, dtype=dtype)
      weight_cpu_1d = torch.tensor([0.25], dtype=dtype)

      def compute_1d(device, self_cpu=self_cpu_1d, weight_cpu=weight_cpu_1d):
        self_tensor = self_cpu.to(device).detach().requires_grad_(True)
        weight = weight_cpu.to(device).detach().requires_grad_(True)
        out = torch.ops.aten._prelu_kernel(self_tensor, weight)
        out.sum().backward()
        return out, self_tensor.grad, weight.grad

      self.assert_close_tpu_vs_cpu(compute_1d)

      # 2D self (N, C), 1D weight (C)
      self_cpu_2d = torch.randn(4, 5, dtype=dtype)
      weight_cpu_2d = torch.randn(5, dtype=dtype)

      def compute_2d(device, self_cpu=self_cpu_2d, weight_cpu=weight_cpu_2d):
        self_tensor = self_cpu.to(device).detach().requires_grad_(True)
        weight = weight_cpu.to(device).detach().requires_grad_(True)
        out = torch.ops.aten._prelu_kernel(self_tensor, weight.reshape(1, 5))
        out.sum().backward()
        return out, self_tensor.grad, weight.grad

      self.assert_close_tpu_vs_cpu(compute_2d)

      # 3D self (N, C, L), 1D weight (C) reshaped to (1, C, 1)
      self_cpu_3d = torch.randn(3, 4, 5, dtype=dtype)
      weight_cpu_3d = torch.randn(4, dtype=dtype)

      def compute_3d(device, self_cpu=self_cpu_3d, weight_cpu=weight_cpu_3d):
        self_tensor = self_cpu.to(device).detach().requires_grad_(True)
        weight = weight_cpu.to(device).detach().requires_grad_(True)
        out = torch.ops.aten._prelu_kernel(self_tensor, weight.reshape(1, 4, 1))
        out.sum().backward()
        return out, self_tensor.grad, weight.grad

      self.assert_close_tpu_vs_cpu(compute_3d)

      # 4D self (N, C, H, W), 1D weight (C) reshaped to (1, C, 1, 1)
      self_cpu_4d = torch.randn(2, 3, 4, 4, dtype=dtype)
      weight_cpu_4d = torch.randn(3, dtype=dtype)

      def compute_4d(device, self_cpu=self_cpu_4d, weight_cpu=weight_cpu_4d):
        self_tensor = self_cpu.to(device).detach().requires_grad_(True)
        weight = weight_cpu.to(device).detach().requires_grad_(True)
        out = torch.ops.aten._prelu_kernel(
            self_tensor, weight.reshape(1, 3, 1, 1)
        )
        out.sum().backward()
        return out, self_tensor.grad, weight.grad

      self.assert_close_tpu_vs_cpu(compute_4d)

      # 0-element empty tensor
      self_cpu_empty = torch.randn(0, 3, 4, dtype=dtype)
      weight_cpu_empty = torch.randn(3, dtype=dtype)

      def compute_empty(
          device, self_cpu=self_cpu_empty, weight_cpu=weight_cpu_empty
      ):
        self_tensor = self_cpu.to(device)
        weight = weight_cpu.to(device)
        return torch.ops.aten._prelu_kernel(
            self_tensor, weight.reshape(1, 3, 1)
        )

      self.assert_close_tpu_vs_cpu(compute_empty)

      # Direct _prelu_kernel_backward call
      grad_out_cpu = torch.randn(2, 3, 4, dtype=dtype)
      self_cpu_bwd = torch.randn(2, 3, 4, dtype=dtype)
      weight_cpu_bwd = torch.tensor([0.25], dtype=dtype)

      def compute_bwd_direct(
          device,
          grad_out_cpu=grad_out_cpu,
          self_cpu=self_cpu_bwd,
          weight_cpu=weight_cpu_bwd,
      ):
        grad_out = grad_out_cpu.to(device)
        self_tensor = self_cpu.to(device)
        weight = weight_cpu.to(device)
        grad_self, grad_weight = torch.ops.aten._prelu_kernel_backward(
            grad_out, self_tensor, weight
        )
        if device == "cpu":
          grad_weight = grad_weight.sum().reshape(1)
        return grad_self, grad_weight

      self.assert_close_tpu_vs_cpu(compute_bwd_direct, atol=2e-4)

      # NaN inputs test
      self_cpu_nan = torch.tensor([1.0, -2.0, float("nan")], dtype=dtype)
      weight_cpu_nan = torch.tensor([0.25], dtype=dtype)
      grad_out_nan = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)

      def compute_nan(
          device,
          grad_out_nan=grad_out_nan,
          self_cpu_nan=self_cpu_nan,
          weight_cpu_nan=weight_cpu_nan,
      ):
        grad_out = grad_out_nan.to(device)
        self_tensor = self_cpu_nan.to(device)
        weight = weight_cpu_nan.to(device)
        out = torch.ops.aten._prelu_kernel(self_tensor, weight)
        grad_self, grad_weight = torch.ops.aten._prelu_kernel_backward(
            grad_out, self_tensor, weight
        )
        if device == "cpu":
          grad_weight = grad_weight.sum().reshape(1)
        return out, grad_self, grad_weight

      self.assert_close_tpu_vs_cpu(compute_nan)

  def _run_addmm_activation_test(
      self,
      self_shape,
      m1_shape,
      m2_shape,
      beta,
      alpha,
      use_gelu,
      dtype,
      rtol,
      atol,
  ):
    def _make_tensor(shape, device, offset=0.0):
      numel = 1
      for s in shape:
        numel *= s
      return (
          torch.arange(numel, dtype=dtype, device=device) * 0.15 - offset
      ).reshape(shape)

    def compute(device):
      self_val = _make_tensor(self_shape, device, offset=0.5)
      m1_val = _make_tensor(m1_shape, device, offset=0.2)
      m2_val = _make_tensor(m2_shape, device, offset=0.8)
      return torch.ops.aten._addmm_activation(
          self_val,
          m1_val,
          m2_val,
          beta=beta,
          alpha=alpha,
          use_gelu=use_gelu,
      )

    self.assert_close_tpu_vs_cpu(compute, rtol=rtol, atol=atol)

  def _run_addmm_activation_out_test(
      self,
      self_shape,
      m1_shape,
      m2_shape,
      target_out_shape,
      beta,
      alpha,
      use_gelu,
      dtype,
      rtol,
      atol,
  ):
    def _make_tensor(shape, device, offset=0.0):
      numel = 1
      for s in shape:
        numel *= s
      return (
          torch.arange(numel, dtype=dtype, device=device) * 0.15 - offset
      ).reshape(shape)

    def compute(device):
      self_val = _make_tensor(self_shape, device, offset=0.5)
      m1_val = _make_tensor(m1_shape, device, offset=0.2)
      m2_val = _make_tensor(m2_shape, device, offset=0.8)
      out_val = torch.empty(target_out_shape, dtype=dtype, device=device)
      torch.ops.aten._addmm_activation.out(
          self_val,
          m1_val,
          m2_val,
          beta=beta,
          alpha=alpha,
          use_gelu=use_gelu,
          out=out_val,
      )
      return out_val

    self.assert_close_tpu_vs_cpu(compute, rtol=rtol, atol=atol)

  @parameterized.product(
      use_gelu=[True, False],
      dtype=[torch.float32, torch.bfloat16, torch.float16],
  )
  def test__addmm_activation(self, use_gelu, dtype):
    """Tests torch.ops.aten._addmm_activation with deterministic values."""
    shapes = [
        ((3, 4), (3, 5), (5, 4)),  # matching 2D
        ((4,), (3, 5), (5, 4)),  # 1D bias broadcast
        ((1, 4), (3, 5), (5, 4)),  # 2D bias broadcast col
        ((3, 1), (3, 5), (5, 4)),  # 2D bias broadcast row
        ((1,), (3, 5), (5, 4)),  # 1D scalar broadcast
    ]
    multipliers = [
        (0.0, 1.0),
        (1.0, 1.0),
        (0.5, 2.0),
        (2.0, 0.5),
        (-1.0, 1.5),
    ]
    rtol = 1.3e-2 if dtype == torch.float32 else 1.5e-2
    atol = 1.3e-1 if dtype == torch.float32 else 1.6e-1

    for self_shape, m1_shape, m2_shape in shapes:
      for beta, alpha in multipliers:
        self._run_addmm_activation_test(
            self_shape,
            m1_shape,
            m2_shape,
            beta,
            alpha,
            use_gelu,
            dtype,
            rtol,
            atol,
        )

  @parameterized.product(
      use_gelu=[True, False],
      dtype=[torch.float32, torch.bfloat16, torch.float16],
  )
  def test__addmm_activation_out(self, use_gelu, dtype):
    """Tests torch.ops.aten._addmm_activation.out with deterministic values."""
    shapes = [
        ((3, 4), (3, 5), (5, 4)),
        ((4,), (2, 6), (6, 4)),
        ((1, 3), (2, 5), (5, 3)),
    ]
    multipliers = [
        (0.0, 1.0),
        (1.0, 2.0),
        (0.5, 0.5),
    ]
    out_shapes = [
        None,  # Same shape as expected output
        (1,),  # Smaller 1D shape (must resize up)
        (10, 10),  # Larger 2D shape (must resize down/differently)
        (2, 3, 4),  # 3D shape (must change rank and dimensions)
    ]
    rtol = 1.3e-2 if dtype == torch.float32 else 1.5e-2
    atol = 1.3e-1 if dtype == torch.float32 else 1.6e-1

    for self_shape, m1_shape, m2_shape in shapes:
      expected_shape = (m1_shape[0], m2_shape[1])
      for beta, alpha in multipliers:
        for out_shape in out_shapes:
          target_out_shape = expected_shape if out_shape is None else out_shape
          self._run_addmm_activation_out_test(
              self_shape,
              m1_shape,
              m2_shape,
              target_out_shape,
              beta,
              alpha,
              use_gelu,
              dtype,
              rtol,
              atol,
          )

  def test_ldexp_overflow_float16(self):
    """Tests that ldexp avoids intermediate overflow for float16."""
    # 2^16 overflows float16 (max 65504), but 1e-4 * 2^16 = 6.5536 doesn't.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ldexp(
            torch.tensor([1e-4], dtype=torch.float16, device=device),
            torch.tensor([16], dtype=torch.int32, device=device),
        )
    )

  def test_ldexp_underflow_float16(self):
    """Tests that ldexp avoids intermediate underflow for float16."""
    # 2^-25 underflows float16 to 0, but 1e4 * 2^-25 = 2.98e-4 doesn't.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ldexp(
            torch.tensor([10000.0], dtype=torch.float16, device=device),
            torch.tensor([-25], dtype=torch.int32, device=device),
        )
    )

  def test_ldexp_overflow_float32(self):
    """Tests that ldexp avoids intermediate overflow for float32."""
    # 2^130 overflows float32, but 1e-10 * 2^130 = 1.36e29 doesn't.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ldexp(
            torch.tensor([1e-10], dtype=torch.float32, device=device),
            torch.tensor([130], dtype=torch.int32, device=device),
        )
    )

  def test_ldexp_underflow_float32(self):
    """Tests that ldexp avoids intermediate underflow for float32."""
    # 2^-150 underflows float32 to 0, but 1e10 * 2^-150 = 7e-36 doesn't.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ldexp(
            torch.tensor([1e10], dtype=torch.float32, device=device),
            torch.tensor([-150], dtype=torch.int32, device=device),
        )
    )

  def test_ldexp_inplace(self):
    """Tests the inplace ldexp_ variant."""

    def test_inplace(device):
      x = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
      exp = torch.tensor([1, 2], dtype=torch.int32, device=device)
      x.ldexp_(exp)
      return x

    self.assert_close_tpu_vs_cpu(test_inplace)

  def test_ldexp_out(self):
    """Tests the out-variant ldexp.out."""

    def test_out(device):
      x = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
      exp = torch.tensor([1, 2], dtype=torch.int32, device=device)
      out = torch.empty_like(x)
      torch.ldexp(x, exp, out=out)
      return out

    self.assert_close_tpu_vs_cpu(test_out)

  @parameterized.product(
      reduction=[0, 1, 2],
      dtype=[torch.float32, torch.bfloat16, torch.float16],
      has_weight=[True, False],
  )
  def test_binary_cross_entropy_simple(self, reduction, dtype, has_weight):
    """Tests binary_cross_entropy."""
    input_val = torch.rand(3, 3, dtype=dtype)
    target_val = torch.rand(3, 3, dtype=dtype)

    if has_weight:
      # Test 1D weight broadcasting to 2D input
      weight_val = torch.rand(3, dtype=dtype)
    else:
      weight_val = None

    def compute(device):
      w = weight_val.to(device) if weight_val is not None else None
      return torch.ops.aten.binary_cross_entropy(
          input_val.to(device),
          target_val.to(device),
          weight=w,
          reduction=reduction,
      )

    rtol = 1e-2 if dtype != torch.float32 else 4e-5
    atol = 1e-2 if dtype != torch.float32 else 4e-5

    self.assert_close_tpu_vs_cpu(compute, rtol=rtol, atol=atol)

  @parameterized.product(
      reduction=[0, 1, 2],
      dtype=[torch.float32, torch.bfloat16, torch.float16],
      has_weight=[True, False],
  )
  def test_binary_cross_entropy_out(self, reduction, dtype, has_weight):
    """Tests binary_cross_entropy.out."""
    input_val = torch.rand(3, 3, dtype=dtype)
    target_val = torch.rand(3, 3, dtype=dtype)

    if has_weight:
      weight_val = torch.rand(3, dtype=dtype)
    else:
      weight_val = None

    if reduction != 0:
      # Skip reduction != 0 for .out variant due to PyTorch CPU
      # Bugs/Discrepancies:
      # 1. Shape mismatch when has_weight=True (expects weight's broadcast
      # shape).
      # 2. Wrong/Inconsistent values (e.g. SUM returns MEAN values, MEAN
      # differs from non-out).
      return

    def compute(device):
      w = weight_val.to(device) if weight_val is not None else None
      out_shape = input_val.shape
      out = torch.empty(out_shape, dtype=dtype).to(device)
      torch.ops.aten.binary_cross_entropy.out(
          input_val.to(device),
          target_val.to(device),
          weight=w,
          reduction=reduction,
          out=out,
      )
      return out

    rtol = 1e-2 if dtype != torch.float32 else 4e-5
    atol = 1e-2 if dtype != torch.float32 else 4e-5

    self.assert_close_tpu_vs_cpu(compute, rtol=rtol, atol=atol)

  @parameterized.product(
      reduction=[0, 1, 2],
      dtype=[torch.float32, torch.bfloat16, torch.float16],
      shape=[(0,), (0, 3)],
  )
  def test_binary_cross_entropy_empty(self, reduction, dtype, shape):
    """Tests binary_cross_entropy with empty tensors."""
    input_val = torch.zeros(shape, dtype=dtype)
    target_val = torch.zeros(shape, dtype=dtype)

    def compute(device):
      return torch.ops.aten.binary_cross_entropy(
          input_val.to(device),
          target_val.to(device),
          reduction=reduction,
      )

    self.assert_close_tpu_vs_cpu(compute)

  @parameterized.product(
      reduction=[0, 1, 2],
      dtype=[torch.float32, torch.bfloat16, torch.float16],
  )
  def test_binary_cross_entropy_boundary(self, reduction, dtype):
    """Tests binary_cross_entropy with boundary values (0, 1)."""

    def compute(device):
      input_val = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=dtype, device=device)
      target_val = torch.tensor(
          [1.0, 0.0, 0.0, 1.0], dtype=dtype, device=device
      )
      return torch.ops.aten.binary_cross_entropy(
          input_val, target_val, reduction=reduction
      )

    rtol = 1e-2 if dtype != torch.float32 else 4e-5
    atol = 1e-2 if dtype != torch.float32 else 4e-5

    self.assert_close_tpu_vs_cpu(compute, rtol=rtol, atol=atol)

  @parameterized.product(
      reduction=[0, 1, 2],
      dtype=[torch.float32, torch.bfloat16, torch.float16],
      shape=[(0,), (0, 3)],
  )
  def test_binary_cross_entropy_backward_empty(self, reduction, dtype, shape):
    """Tests binary_cross_entropy_backward with empty tensors."""
    input_val = torch.zeros(shape, dtype=dtype)
    target_val = torch.zeros(shape, dtype=dtype)
    if reduction == 0:
      grad_output_val = torch.zeros(shape, dtype=dtype)
    else:
      grad_output_val = torch.zeros((), dtype=dtype)

    def compute(device):
      return torch.ops.aten.binary_cross_entropy_backward(
          grad_output_val.to(device),
          input_val.to(device),
          target_val.to(device),
          reduction=reduction,
      )

    self.assert_close_tpu_vs_cpu(compute)

  @parameterized.product(
      reduction=[0, 1, 2],
      dtype=[torch.float32, torch.bfloat16],
  )
  def test_binary_cross_entropy_backward_boundary(self, reduction, dtype):
    """Tests binary_cross_entropy_backward with boundary values (0, 1)."""
    input_val = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=dtype)
    target_val = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=dtype)

    if reduction == 0:
      grad_output_val = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=dtype)
    else:
      grad_output_val = torch.tensor(1.0, dtype=dtype)

    def compute(device):
      return torch.ops.aten.binary_cross_entropy_backward(
          grad_output_val.to(device),
          input_val.to(device),
          target_val.to(device),
          reduction=reduction,
      )

    rtol = 1e-2 if dtype != torch.float32 else 4e-5
    atol = 1e-2 if dtype != torch.float32 else 4e-5

    self.assert_close_tpu_vs_cpu(compute, rtol=rtol, atol=atol)

  @parameterized.product(
      reduction=[0, 1, 2],
      dtype=[torch.float32, torch.bfloat16, torch.float16],
      has_weight=[True, False],
  )
  def test_binary_cross_entropy_backward_out(
      self, reduction, dtype, has_weight
  ):
    """Tests binary_cross_entropy_backward.grad_input."""
    input_val = torch.rand(3, 3, dtype=dtype)
    target_val = torch.rand(3, 3, dtype=dtype)

    if has_weight:
      weight_val = torch.rand(3, dtype=dtype)
    else:
      weight_val = None

    if reduction == 0:
      grad_output_val = torch.rand(3, 3, dtype=dtype)
    else:
      grad_output_val = torch.rand((), dtype=dtype)

    def compute(device):
      w = weight_val.to(device) if weight_val is not None else None
      out_shape = input_val.shape
      out = torch.empty(out_shape, dtype=dtype).to(device)
      torch.ops.aten.binary_cross_entropy_backward.grad_input(
          grad_output_val.to(device),
          input_val.to(device),
          target_val.to(device),
          weight=w,
          reduction=reduction,
          grad_input=out,
      )
      return out

    rtol = 1e-2 if dtype != torch.float32 else 4e-5
    atol = 1e-2 if dtype != torch.float32 else 4e-5

    self.assert_close_tpu_vs_cpu(compute, rtol=rtol, atol=atol)

  # --- Fused RNN-cell ops (_thnn_fused_lstm_cell / _thnn_fused_gru_cell) -----
  # These are the ops stock nn.LSTM / nn.GRU lower to on the TPU backend. They
  # are CUDA-only in ATen (no CPU kernel), so they are exercised through the
  # public modules -- which run comparable math on both devices -- rather than
  # called directly. Params + inputs are seeded and built on CPU, then moved,
  # so the TPU and CPU runs use identical values.

  def _check_rnn_tpu_vs_cpu(
      self,
      ctor,
      *,
      bias,
      bidirectional,
      num_layers,
      backward,
      input_size=16,
      hidden_size=8,
  ):
    def compute(device):
      torch.manual_seed(0)
      module = ctor(
          input_size,
          hidden_size,
          num_layers=num_layers,
          bias=bias,
          bidirectional=bidirectional,
      ).to(device)
      x = torch.randn(6, 3, input_size).to(device)
      if backward:
        x.requires_grad_(True)
        output, _ = module(x)
        output.pow(2).sum().backward()
        parts = [x.grad.reshape(-1)]
        parts += [p.grad.reshape(-1) for p in module.parameters()]
        return torch.cat(parts)
      output, state = module(x)
      states = state if isinstance(state, tuple) else (state,)
      return torch.cat([output.reshape(-1)] + [s.reshape(-1) for s in states])

    # rtol for TPU matmul drift over the recurrence; atol so the handful of
    # near-zero gradient entries (where relative error is meaningless) pass.
    self.assert_close_tpu_vs_cpu(compute, rtol=2e-2, atol=1e-2)

  @parameterized.product(bidirectional=[False, True], num_layers=[1, 2])
  def test__thnn_fused_lstm_cell(self, bidirectional, num_layers):
    """nn.LSTM forward lowers to _thnn_fused_lstm_cell on TPU."""
    self._check_rnn_tpu_vs_cpu(
        torch.nn.LSTM,
        bias=True,
        bidirectional=bidirectional,
        num_layers=num_layers,
        backward=False,
    )

  def test__thnn_fused_lstm_cell_no_bias(self):
    """LSTM cell with bias=False (no input/hidden bias inputs)."""
    self._check_rnn_tpu_vs_cpu(
        torch.nn.LSTM,
        bias=False,
        bidirectional=False,
        num_layers=1,
        backward=False,
    )

  def test__thnn_fused_lstm_cell_backward(self):
    """nn.LSTM backward lowers to _thnn_fused_lstm_cell_backward_impl on TPU."""
    self._check_rnn_tpu_vs_cpu(
        torch.nn.LSTM,
        bias=True,
        bidirectional=True,
        num_layers=2,
        backward=True,
    )

  @parameterized.product(bidirectional=[False, True], num_layers=[1, 2])
  def test__thnn_fused_gru_cell(self, bidirectional, num_layers):
    """nn.GRU forward lowers to _thnn_fused_gru_cell on TPU."""
    self._check_rnn_tpu_vs_cpu(
        torch.nn.GRU,
        bias=True,
        bidirectional=bidirectional,
        num_layers=num_layers,
        backward=False,
    )

  def test__thnn_fused_gru_cell_no_bias(self):
    """GRU cell with bias=False (no input/hidden bias inputs)."""
    self._check_rnn_tpu_vs_cpu(
        torch.nn.GRU,
        bias=False,
        bidirectional=False,
        num_layers=1,
        backward=False,
    )

  def test__thnn_fused_gru_cell_backward(self):
    """nn.GRU backward lowers to _thnn_fused_gru_cell_backward on TPU."""
    self._check_rnn_tpu_vs_cpu(
        torch.nn.GRU,
        bias=True,
        bidirectional=True,
        num_layers=2,
        backward=True,
    )

  def test_thnn_fused_gru_cell_zero_batch(self):
    """Verifies that _thnn_fused_gru_cell handles zero batch size."""
    device = torch.device("tpu")
    ig = torch.randn(0, 24, device=device)
    hg = torch.randn(0, 24, device=device)
    hx = torch.randn(0, 8, device=device)
    hy, ws = torch.ops.aten._thnn_fused_gru_cell(ig, hg, hx)
    self.assertEqual(hy.shape, (0, 8))
    self.assertEqual(ws.shape, (0, 40))

  def test_ldexp_integer_promotion(self):
    """Tests that ldexp promotes integer base and exponent to float."""
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ldexp(
            torch.tensor([1, 2], dtype=torch.int32, device=device),
            torch.tensor([2, 3], dtype=torch.int32, device=device),
        )
    )

  def test_ldexp_extreme_exponents_float16(self):
    """Tests extreme exponent cases with zero base for float16."""
    # Float16: exp=50 (exceeds max_exp=14). scale must not overflow to inf.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ldexp(
            torch.tensor([0.0], dtype=torch.float16, device=device),
            torch.tensor([50], dtype=torch.int32, device=device),
        )
    )

  def test_ldexp_extreme_exponents_float32(self):
    """Tests extreme exponent cases with zero base for float32."""
    # Float32: exp=400 (exceeds max_exp=120). scale must not overflow to inf.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ldexp(
            torch.tensor([0.0], dtype=torch.float32, device=device),
            torch.tensor([400], dtype=torch.int32, device=device),
        )
    )

  def test_ldexp_extreme_exponents_float64(self):
    """Tests extreme exponent cases with zero base for float64."""
    if torch.tensor([1e100], dtype=torch.float64, device="tpu").isinf().any():
      self.skipTest("float64 is downcasted to float32 on this device")
    # Float64: exp=3000 (exceeds max_exp=1000). scale must not overflow to inf.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ldexp(
            torch.tensor([0.0], dtype=torch.float64, device=device),
            torch.tensor([3000], dtype=torch.int32, device=device),
        )
    )

  def test_ldexp_large_exponent_float64(self):
    """Tests large exponent scaling for float64 to ensure it covers up to 3000."""
    if torch.tensor([1e100], dtype=torch.float64, device="tpu").isinf().any():
      self.skipTest("float64 is downcasted to float32 on this device")
    # 2^900 is ~ 8.41e270. 1e-100 * 2^900 is ~ 8.41e170, which is representable.
    # If scaling limit is 360, it will return 1e-100 * 2^360 = 2.34e8,
    # which is wrong.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ldexp(
            torch.tensor([1e-100], dtype=torch.float64, device=device),
            torch.tensor([900], dtype=torch.int32, device=device),
        )
    )

  def test_addmm_input_broadcasting(self):
    """Tests torch.addmm input broadcasting rules."""
    mat1 = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    mat2 = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    self_shapes = [[1, 4], [2, 1], [1, 1], [1]]

    for self_shape in self_shapes:
      numel = math.prod(self_shape)
      self_input = torch.arange(numel, dtype=torch.float32).reshape(self_shape)
      self.assert_close_tpu_vs_cpu(
          lambda device, self_input=self_input, m1=mat1, m2=mat2: (
              torch.addmm(self_input.to(device), m1.to(device), m2.to(device))
          )
      )

  def test_foreach_add_sub_alpha(self):
    """Tests foreach_add and foreach_sub with different alphas."""
    tensors1 = [
        torch.arange(6, dtype=torch.float32).reshape(2, 3) + i for i in range(3)
    ]
    tensors2 = [
        torch.arange(6, dtype=torch.float32).reshape(2, 3) * 2 + i
        for i in range(3)
    ]
    single_tensor = torch.tensor(1.5, dtype=torch.float32)

    # Out-of-place tests (list-list)
    # Test alpha = 1
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_add(
            [t.to(device) for t in tensors1],
            [t.to(device) for t in tensors2],
            alpha=1.0,
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_sub(
            [t.to(device) for t in tensors1],
            [t.to(device) for t in tensors2],
            alpha=1.0,
        )
    )

    # Test alpha = -1
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_add(
            [t.to(device) for t in tensors1],
            [t.to(device) for t in tensors2],
            alpha=-1.0,
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_sub(
            [t.to(device) for t in tensors1],
            [t.to(device) for t in tensors2],
            alpha=-1.0,
        )
    )

    # Test alpha = 2.5
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_add(
            [t.to(device) for t in tensors1],
            [t.to(device) for t in tensors2],
            alpha=2.5,
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_sub(
            [t.to(device) for t in tensors1],
            [t.to(device) for t in tensors2],
            alpha=2.5,
        )
    )

    # Out-of-place tests (list-tensor)
    # Test alpha = 1
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_add(
            [t.to(device) for t in tensors1],
            single_tensor.to(device),
            alpha=1.0,
        )
    )
    # Test alpha = -1
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_add(
            [t.to(device) for t in tensors1],
            single_tensor.to(device),
            alpha=-1.0,
        )
    )
    # Test alpha = 2.5
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_add(
            [t.to(device) for t in tensors1],
            single_tensor.to(device),
            alpha=2.5,
        )
    )

    # Inplace tests (list-list)
    # Test alpha = 1
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_add_(
            [t.clone().to(device) for t in tensors1],
            [t.to(device) for t in tensors2],
            alpha=1.0,
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_sub_(
            [t.clone().to(device) for t in tensors1],
            [t.to(device) for t in tensors2],
            alpha=1.0,
        )
    )

    # Test alpha = -1
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_add_(
            [t.clone().to(device) for t in tensors1],
            [t.to(device) for t in tensors2],
            alpha=-1.0,
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_sub_(
            [t.clone().to(device) for t in tensors1],
            [t.to(device) for t in tensors2],
            alpha=-1.0,
        )
    )

    # Test alpha = 2.5
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_add_(
            [t.clone().to(device) for t in tensors1],
            [t.to(device) for t in tensors2],
            alpha=2.5,
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_sub_(
            [t.clone().to(device) for t in tensors1],
            [t.to(device) for t in tensors2],
            alpha=2.5,
        )
    )

    # Inplace tests (list-tensor)
    # Test alpha = 1
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_add_(
            [t.clone().to(device) for t in tensors1],
            single_tensor.to(device),
            alpha=1.0,
        )
    )
    # Test alpha = -1
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_add_(
            [t.clone().to(device) for t in tensors1],
            single_tensor.to(device),
            alpha=-1.0,
        )
    )
    # Test alpha = 2.5
    self.assert_close_tpu_vs_cpu(
        lambda device: torch._foreach_add_(
            [t.clone().to(device) for t in tensors1],
            single_tensor.to(device),
            alpha=2.5,
        )
    )

  def test_lu_factor_ex_empty_tensors(self):
    # Unit test verifying that LU decomposition supports empty tensors across
    # all dtypes (including bool and integer dtypes), comparing against CPU
    # reference numerics.
    dtypes = [
        *op_testing.INTEGRAL_DTYPES,
        *op_testing.FLOAT_DTYPES,
    ]
    for dtype in dtypes:
      t_tpu = torch.empty(0, 3, 5, dtype=dtype, device="tpu")
      t_cpu = t_tpu.cpu()
      lu_tpu, pivots_tpu, info_tpu = torch.ops.aten.linalg_lu_factor_ex(t_tpu)
      lu_cpu, pivots_cpu, info_cpu = torch.ops.aten.linalg_lu_factor_ex(t_cpu)
      self.assertEqual(lu_tpu, lu_cpu)
      self.assertEqual(pivots_tpu, pivots_cpu)
      self.assertEqual(info_tpu, info_cpu)

  def test_linalg_lu_zero_sized_matrix(self):
    device = torch.device("tpu")
    shapes = [
        (0, 0),
        (0, 3),
        (3, 0),
        (2, 3, 0),
        (2, 0, 3),
        (0, 3, 5),
        (0, 0, 0),
    ]
    for shape in shapes:
      with self.subTest(shape=shape):
        a_cpu = torch.empty(shape, dtype=torch.float32)
        a_tpu = a_cpu.to(device)
        p_cpu, l_cpu, u_cpu = torch.linalg.lu(a_cpu)
        p_tpu, l_tpu, u_tpu = torch.linalg.lu(a_tpu)
        self.assertEqual(p_tpu.shape, p_cpu.shape)
        self.assertEqual(l_tpu.shape, l_cpu.shape)
        self.assertEqual(u_tpu.shape, u_cpu.shape)
        utils.assert_close(p_tpu.cpu(), p_cpu)
        utils.assert_close(l_tpu.cpu(), l_cpu)
        utils.assert_close(u_tpu.cpu(), u_cpu)

        lu_cpu, piv_cpu = torch.linalg.lu_factor(a_cpu)
        lu_tpu, piv_tpu = torch.linalg.lu_factor(a_tpu)
        for unpack_data, unpack_pivots in [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ]:
          with self.subTest(
              shape=shape,
              unpack_data=unpack_data,
              unpack_pivots=unpack_pivots,
          ):
            p_cpu, l_cpu, u_cpu = torch.lu_unpack(
                lu_cpu,
                piv_cpu,
                unpack_data=unpack_data,
                unpack_pivots=unpack_pivots,
            )
            p_tpu, l_tpu, u_tpu = torch.lu_unpack(
                lu_tpu,
                piv_tpu,
                unpack_data=unpack_data,
                unpack_pivots=unpack_pivots,
            )
            self.assertEqual(p_tpu.shape, p_cpu.shape)
            self.assertEqual(l_tpu.shape, l_cpu.shape)
            self.assertEqual(u_tpu.shape, u_cpu.shape)
            utils.assert_close(p_tpu.cpu(), p_cpu)
            utils.assert_close(l_tpu.cpu(), l_cpu)
            utils.assert_close(u_tpu.cpu(), u_cpu)

  def test_assert_close_partial_override(self):
    """Tests that utils.assert_close works with partial numeric overrides.

    A partial override means specifying only `rtol` or only `atol`, leaving
    the other as `None` to be resolved to the PyTorch default.

    Verifies it works in LOOSE mode.
    """
    t1 = torch.tensor([1.0, 2.0])
    t2 = torch.tensor([1.0001, 2.0001])  # diff is 1e-4

    # Default float32 tolerance is rtol=1.3e-6, atol=1e-5.
    # This diff (1e-4) exceeds default atol (1e-5).
    # If we override only atol to 2e-4, it should pass.
    # If we didn't support partial overrides, it would crash with ValueError
    # in LOOSE mode.
    utils.assert_close(t1, t2, atol=2e-4)

    # Similarly, override only rtol
    # rel diff is 1e-4 / 1 = 1e-4.
    # If we override only rtol to 2e-4, it should pass.
    utils.assert_close(t1, t2, rtol=2e-4)

  def test_assert_close_tolerances_must_be_positive(self):
    """Tests that specifying rtol or atol <= 0 raises ValueError."""
    t1 = torch.tensor([1.0, 2.0])
    t2 = torch.tensor([1.0, 2.0])

    with self.assertRaisesRegex(ValueError, "rtol must be > 0"):
      self.assert_close(golden_result=t1, torch_tpu_result=t2, rtol=0)
    with self.assertRaisesRegex(ValueError, "rtol must be > 0"):
      self.assert_close(golden_result=t1, torch_tpu_result=t2, rtol=-1e-5)
    with self.assertRaisesRegex(ValueError, "atol must be > 0"):
      self.assert_close(golden_result=t1, torch_tpu_result=t2, atol=0)
    with self.assertRaisesRegex(ValueError, "atol must be > 0"):
      self.assert_close(golden_result=t1, torch_tpu_result=t2, atol=-1e-5)

    with self.assertRaisesRegex(ValueError, "rtol must be > 0"):
      self.assert_close_tpu_vs_cpu(
          lambda device: torch.tensor([1.0], device=device), rtol=0
      )
    with self.assertRaisesRegex(ValueError, "atol must be > 0"):
      self.assert_close_tpu_vs_cpu(
          lambda device: torch.tensor([1.0], device=device), atol=0
      )

    with self.assertRaisesRegex(ValueError, "rtol must be > 0"):
      self.set_accuracy_overrides(
          tpu_gpu_overrides={"op": {torch.float32: {"rtol": 0}}},
          grad_gpu_overrides={},
      )
    with self.assertRaisesRegex(ValueError, "atol must be > 0"):
      self.set_accuracy_overrides(
          tpu_gpu_overrides={"op": {torch.float32: {"atol": 0}}},
          grad_gpu_overrides={},
      )

  def test_topk_sorted_false(self):
    """Tests torch.topk with sorted=False."""
    device = torch.device("tpu")
    t = torch.tensor([[1.0, 5.0, 2.0], [4.0, 3.0, 6.0]], device=device)
    tpu_values, tpu_indices = torch.topk(t, k=2, sorted=False)
    golden_values, _ = torch.topk(t, k=2, sorted=True)
    tpu_values_sorted, _ = torch.sort(tpu_values, descending=True)
    utils.assert_close(tpu_values_sorted, golden_values)
    gathered = torch.gather(t, 1, tpu_indices)
    utils.assert_close(gathered, tpu_values)

  def test_topk_chlo(self):
    """Tests torch.topk lowering across dtypes, dims, k, and edge cases."""

    def _verify_topk(cpu_t, k, dim, largest, check_indices=True):
      def compute(device):
        t = cpu_t.to(device)
        values, indices = torch.topk(
            t, k=k, dim=dim, largest=largest, sorted=True
        )
        self.assertEqual(indices.dtype, torch.int64)
        gathered = torch.gather(t, dim, indices)
        if check_indices:
          return values, indices, gathered
        else:
          return values, gathered

      self.assert_close_tpu_vs_cpu(compute)

    for dtype in [torch.bfloat16, torch.float32]:
      # Distinct values (2D tensor) where tie-breaking is not an issue.
      cpu_t = torch.tensor(
          [
              [1.0, 5.0, 2.0, 8.0, 3.0],
              [10.0, -1.0, 4.0, 6.0, 7.0],
              [-5.0, 0.0, 9.0, 2.5, 1.5],
          ],
          dtype=dtype,
      )
      for largest in [True, False]:
        for dim in [0, 1, -1]:
          dim_size = cpu_t.size(dim)
          for k in [0, 1, 2, dim_size]:
            _verify_topk(
                cpu_t, k=k, dim=dim, largest=largest, check_indices=True
            )

      # Duplicates and zeroes (check values and gather,
      # skip indices check due to ties).
      cpu_dup = torch.tensor(
          [
              [0.0, 0.0, 0.0, 5.0, 5.0],
              [2.0, 2.0, -1.0, -1.0, 0.0],
          ],
          dtype=dtype,
      )
      for largest in [True, False]:
        for dim in [0, 1, -1]:
          dim_size = cpu_dup.size(dim)
          for k in [0, 1, 2, dim_size]:
            _verify_topk(
                cpu_dup, k=k, dim=dim, largest=largest, check_indices=False
            )

      # Varied dimensions (1D, 3D), single-element, and all-zero tensors.
      cpu_1d = torch.tensor([4.0, 1.0, 3.0, 2.0, 0.0], dtype=dtype)
      for largest in [True, False]:
        for dim in [0, -1]:
          for k in [0, 1, 3, 5]:
            _verify_topk(
                cpu_1d, k=k, dim=dim, largest=largest, check_indices=True
            )

      cpu_3d = torch.arange(24, dtype=dtype).reshape(2, 3, 4)
      for largest in [True, False]:
        for dim in [0, 1, 2, -1, -2]:
          for k in [0, 1, 2]:
            _verify_topk(
                cpu_3d, k=k, dim=dim, largest=largest, check_indices=True
            )

      cpu_single = torch.tensor([[-42.0]], dtype=dtype)
      for largest in [True, False]:
        for dim in [0, 1, -1]:
          _verify_topk(
              cpu_single, k=1, dim=dim, largest=largest, check_indices=True
          )

      cpu_zeroes = torch.zeros((2, 3, 4), dtype=dtype)
      for largest in [True, False]:
        for dim in [0, 1, 2, -1]:
          for k in [0, 1, 2]:
            _verify_topk(
                cpu_zeroes, k=k, dim=dim, largest=largest, check_indices=False
            )

  def test_max_pool2d_no_indices(self):
    """Tests nn.functional.max_pool2d without indices."""
    device = torch.device("tpu")
    maxpool_input = torch.tensor(
        [[
            [
                [-7.7435, -8.8254, 7.2097, 4.3371, 2.8040, -3.4491],
                [-8.5819, 3.9336, -6.2229, 1.1184, -6.0094, 7.3457],
                [-2.0333, 5.7398, 1.8601, 8.6590, 0.6541, 0.0145],
            ],
            [
                [2.8523, -5.7473, 2.1480, -0.3480, 2.5668, -8.3042],
                [-1.1508, -8.2351, 4.4935, -0.0096, -2.7059, -5.8874],
                [5.4567, 0.2254, -3.6194, -6.1967, 8.8962, -1.7928],
            ],
        ]],
        dtype=torch.float32,
        device=device,
    )

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nn.functional.max_pool2d(
            maxpool_input.to(device),
            kernel_size=3,
            stride=(2, 1),
            padding=1,
            dilation=(1, 2),
            ceil_mode=True,
            return_indices=False,
        )
    )

  def test_max_pool1d_no_indices(self):
    """Tests nn.functional.max_pool1d without indices."""
    device = torch.device("tpu")
    maxpool_input = torch.tensor(
        [[
            [-7.7435, -8.8254, 7.2097, 4.3371, 2.8040, -3.4491, 5.1234, -1.23],
            [2.8523, -5.7473, 2.1480, -0.3480, 2.5668, -8.3042, 3.456, 0.789],
        ]],
        dtype=torch.float32,
        device=device,
    )

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nn.functional.max_pool1d(
            maxpool_input.to(device),
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            ceil_mode=True,
            return_indices=False,
        )
    )

  def test_max_pool3d_no_indices(self):
    """Tests nn.functional.max_pool3d without indices."""
    maxpool_input = torch.randn(
        (2, 3, 4, 6, 6),
        dtype=torch.float32,
        generator=torch.manual_seed(42),
    )

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nn.functional.max_pool3d(
            maxpool_input.to(device),
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            ceil_mode=True,
            return_indices=False,
        )
    )

  def test_adaptive_max_pool2d(self):
    """Tests nn.functional.adaptive_max_pool2d and aten.adaptive_max_pool2d.out."""
    for shape, out_size in [
        # Divisible cases
        ((2, 3, 4, 4), (2, 2)),
        ((1, 2, 6, 6), (3, 3)),
        ((2, 4, 8, 8), (4, 4)),
        # Non-divisible cases
        ((1, 2, 5, 7), (3, 2)),
        ((2, 3, 7, 7), (3, 3)),
        ((1, 1, 10, 10), (3, 3)),
        # Single int output_size
        ((2, 3, 6, 6), 3),
        ((1, 2, 7, 7), (4, 3)),
        # 3D tensor (C, H, W)
        ((3, 7, 7), (3, 3)),
        ((2, 5, 7), (2, 3)),
        # Zero-batch tensor (0, 3, 8, 8)
        ((0, 3, 8, 8), (4, 4)),
        # 3D zero-channel tensor (0, 4, 4)
        ((0, 4, 4), (2, 2)),
        ((0, 4, 4), (0, 0)),
        # Zero output dimension cases (4D and 3D)
        ((2, 3, 4, 4), 0),
        ((2, 3, 4, 4), (0, 0)),
        ((2, 3, 4, 4), (0, 3)),
        ((2, 3, 4, 4), (3, 0)),
        ((3, 4, 4), 0),
        ((3, 4, 4), (0, 0)),
        ((3, 4, 4), (0, 3)),
        ((3, 4, 4), (3, 0)),
    ]:
      for dtype in (
          torch.float32,
          torch.bfloat16,
          torch.float16,
          torch.float64,
      ):
        x_cpu = torch.randn(shape, dtype=dtype)
        out_size_pair = (
            (out_size, out_size) if isinstance(out_size, int) else out_size
        )

        def compute_functional(device, x=x_cpu, os=out_size):
          return torch.nn.functional.adaptive_max_pool2d(
              x.to(device), os, return_indices=True
          )

        self.assert_close_tpu_vs_cpu(
            compute_functional, rtol=1e-3, atol=1e-3, check_dtype=True
        )

        def compute_out(device, x=x_cpu, os=out_size_pair, dt=dtype):
          out_buf = torch.empty((0,), dtype=dt, device=device)
          idx_buf = torch.empty((0,), dtype=torch.int64, device=device)
          torch.ops.aten.adaptive_max_pool2d.out(
              x.to(device), os, out=out_buf, indices=idx_buf
          )
          return out_buf, idx_buf

        self.assert_close_tpu_vs_cpu(
            compute_out, rtol=1e-3, atol=1e-3, check_dtype=True
        )

    # Tie-breaking test with duplicate maxima (both divisible and non-divisible shapes)
    for dt in (torch.float32, torch.float64):
      for tie_data, out_sz in [
          # Non-divisible (3, 3) -> (2, 2)
          ([[[[1.0, 5.0, 5.0], [5.0, 2.0, 3.0], [4.0, 5.0, 1.0]]]], (2, 2)),
          # Divisible (4, 4) -> (2, 2)
          (
              [[[
                  [5.0, 5.0, 1.0, 2.0],
                  [5.0, 5.0, 3.0, 4.0],
                  [1.0, 2.0, 6.0, 6.0],
                  [3.0, 4.0, 6.0, 6.0],
              ]]],
              (2, 2),
          ),
      ]:
        tie_x_cpu = torch.tensor(tie_data, dtype=dt)
        self.assert_close_tpu_vs_cpu(
            lambda device, x=tie_x_cpu, os=out_sz: torch.nn.functional.adaptive_max_pool2d(
                x.to(device), os, return_indices=True
            )
        )

  def test_rsqrt_complex_grad(self):
    device = torch.device("tpu")
    x = torch.randn(
        2, 2, dtype=torch.complex64, device=device, requires_grad=True
    )
    y = torch.rsqrt(x)
    y.abs().sum().backward()
    print(f"rsqrt(complex64) grad: {x.grad}")

  def test_rsqrt_complex(self):
    device = torch.device("tpu")
    x = torch.randn(2, 2, dtype=torch.complex64, device=device)
    y = torch.rsqrt(x)
    print(f"rsqrt(complex64) result: {y}")

  def test_complex_abs_precision(self):
    for dtype in [torch.complex64, torch.complex128]:
      vals = [1e30, 1e-35]

      for val in vals:
        input_tensor = torch.tensor([complex(val, val)], dtype=dtype)

        def run(device, input_tensor=input_tensor):
          return torch.abs(input_tensor.to(device))

        self.assert_close_tpu_vs_cpu(run)

  def test_complex_abs_out_dtype(self):
    device = torch.device("tpu")
    x = torch.tensor([3.0 + 4.0j], dtype=torch.complex64, device=device)
    # Valid cast to real float32
    out_f32 = torch.empty(1, dtype=torch.float32, device=device)
    torch.abs(x, out=out_f32)
    utils.assert_close(
        out_f32, torch.tensor([5.0], dtype=torch.float32, device=device)
    )

    # Valid upcast to real float64
    out_f64 = torch.empty(1, dtype=torch.float64, device=device)
    torch.abs(x, out=out_f64)
    utils.assert_close(
        out_f64, torch.tensor([5.0], dtype=torch.float64, device=device)
    )

  def test_logcumsumexp_out_dtype(self):
    device = torch.device("tpu")
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    out_f64 = torch.empty(3, dtype=torch.float64, device=device)
    torch.ops.aten._logcumsumexp.out(x, 0, out=out_f64)
    expected = torch.logcumsumexp(torch.tensor([1.0, 2.0, 3.0]), dim=0).to(
        torch.float64
    )
    self.assertEqual(out_f64.dtype, torch.float64)
    utils.assert_close(out_f64.cpu(), expected)

    # 0-D scalar tensor test
    x0 = torch.tensor(2.5, dtype=torch.float32, device=device)
    out0 = torch.empty((), dtype=torch.float64, device=device)
    torch.ops.aten._logcumsumexp.out(x0, 0, out=out0)
    self.assertEqual(out0.dtype, torch.float64)
    self.assertEqual(out0.shape, ())
    utils.assert_close(out0.cpu(), torch.tensor(2.5, dtype=torch.float64))

    # 0-D scalar integer tensor test: aten._logcumsumexp.out supports 0-D,
    # whereas the public torch.logcumsumexp composite rejects integer dtypes
    # (tested in errors_test.py).
    x_int = torch.tensor(5, dtype=torch.int32, device=device)
    out_int = torch.empty((), dtype=torch.int32, device=device)
    torch.ops.aten._logcumsumexp.out(x_int, 0, out=out_int)
    self.assertEqual(out_int.dtype, torch.int32)
    self.assertEqual(out_int.shape, ())
    utils.assert_close(out_int.cpu(), torch.tensor(5, dtype=torch.int32))

    # Empty tensor test with shape resizing
    xe = torch.empty(0, 4, dtype=torch.float32, device=device)
    oute = torch.empty(2, 3, dtype=torch.float64, device=device)
    torch.ops.aten._logcumsumexp.out(xe, 1, out=oute)
    self.assertEqual(oute.shape, (0, 4))
    self.assertEqual(oute.dtype, torch.float64)

    # Shape resizing from incorrect shape
    x_matrix = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device
    )
    out_resized = torch.empty(0, dtype=torch.float32, device=device)
    torch.ops.aten._logcumsumexp.out(x_matrix, 0, out=out_resized)
    self.assertEqual(out_resized.shape, (2, 2))
    expected_matrix = torch.logcumsumexp(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=0
    )
    utils.assert_close(out_resized.cpu(), expected_matrix)

  def test_cumsum_out_dtype(self):
    device = torch.device("tpu")
    x = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device)
    out_f32 = torch.empty(4, dtype=torch.float32, device=device)
    torch.cumsum(x, dim=0, out=out_f32)
    utils.assert_close(
        out_f32,
        torch.tensor([1.0, 3.0, 6.0, 10.0], dtype=torch.float32, device=device),
    )

    # Complex to float out casting
    xc = torch.tensor(
        [1.0 + 2.0j, 3.0 + 4.0j], dtype=torch.complex64, device=device
    )
    out_f64 = torch.empty(2, dtype=torch.float64, device=device)
    torch.cumsum(xc, dim=0, out=out_f64)
    utils.assert_close(
        out_f64,
        torch.tensor([1.0, 4.0], dtype=torch.float64, device=device),
    )

    # 0-D scalar tensor test (including boolean scalar parity with ATen)
    x0 = torch.tensor(5, dtype=torch.int32, device=device)
    out0 = torch.empty((), dtype=torch.float32, device=device)
    torch.cumsum(x0, dim=0, out=out0)
    utils.assert_close(
        out0, torch.tensor(5.0, dtype=torch.float32, device=device)
    )
    x0_bool = torch.tensor(True, dtype=torch.bool, device=device)
    out0_bool = torch.empty((), dtype=torch.bool, device=device)
    torch.cumsum(x0_bool, dim=0, out=out0_bool)
    self.assertTrue(out0_bool.item())

    # Empty tensor test
    xe = torch.empty(0, 4, dtype=torch.int32, device=device)
    oute = torch.empty(0, 4, dtype=torch.float32, device=device)
    torch.cumsum(xe, dim=1, out=oute)
    self.assertEqual(oute.shape, (0, 4))
    self.assertEqual(oute.dtype, torch.float32)

    # Integer default promotion to int64 (functional and out-variant)
    for int_dtype in [torch.int8, torch.int16, torch.int32]:
      xi_cpu = torch.tensor([1, 2, 3], dtype=int_dtype, device="cpu")
      xi_tpu = torch.tensor([1, 2, 3], dtype=int_dtype, device=device)

      # Functional variant
      res_cpu = torch.cumsum(xi_cpu, dim=0)
      res_tpu = torch.cumsum(xi_tpu, dim=0)
      self.assertEqual(res_cpu.dtype, torch.int64)
      self.assertEqual(res_tpu.dtype, torch.int64)
      utils.assert_close(res_tpu.cpu(), res_cpu)

      # Out-variant without dtype
      out_tpu = torch.empty(3, dtype=torch.int64, device=device)
      out_cpu = torch.empty(3, dtype=torch.int64, device="cpu")
      torch.cumsum(xi_tpu, dim=0, out=out_tpu)
      torch.cumsum(xi_cpu, dim=0, out=out_cpu)
      self.assertEqual(out_tpu.dtype, torch.int64)
      utils.assert_close(out_tpu.cpu(), out_cpu)

  def test_cumsum_pre_scan_cast(self):
    """Tests that cumsum casts inputs to target element type before scanning."""
    # 1. float32 -> float16 overflow: 70000.0 becomes inf in float16,
    # -70000.0 becomes -inf, and inf + (-inf) = nan.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.cumsum(
            torch.tensor(
                [70000.0, -70000.0], dtype=torch.float32, device=device
            ),
            dim=0,
            dtype=torch.float16,
        )
    )
    # 2. int32 -> bfloat16 precision: 257 casts to 256.0 in bfloat16.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.cumsum(
            torch.full((4,), 257, dtype=torch.int32, device=device),
            dim=0,
            dtype=torch.bfloat16,
        )
    )

    # 3. out-variant: float32 -> float16 overflow with out=
    def test_out(device):
      out = torch.empty(2, dtype=torch.float16, device=device)
      torch.cumsum(
          torch.tensor([70000.0, -70000.0], dtype=torch.float32, device=device),
          dim=0,
          out=out,
      )
      return out

    self.assert_close_tpu_vs_cpu(test_out)
    # 4. Multi-dimensional tensor along dim=1
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.cumsum(
            torch.tensor(
                [[70000.0, -70000.0], [1.0, 2.0]],
                dtype=torch.float32,
                device=device,
            ),
            dim=1,
            dtype=torch.float16,
        )
    )

  def test_cumprod_out_dtype(self):
    device = torch.device("tpu")
    x = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device)
    out_f32 = torch.empty(4, dtype=torch.float32, device=device)
    torch.cumprod(x, dim=0, out=out_f32)
    utils.assert_close(
        out_f32,
        torch.tensor([1.0, 2.0, 6.0, 24.0], dtype=torch.float32, device=device),
    )

    # Complex to float out casting
    xc = torch.tensor(
        [2.0 + 3.0j, 4.0 + 5.0j], dtype=torch.complex64, device=device
    )
    out_f64 = torch.empty(2, dtype=torch.float64, device=device)
    torch.cumprod(xc, dim=0, out=out_f64)
    utils.assert_close(
        out_f64,
        torch.tensor([2.0, 8.0], dtype=torch.float64, device=device),
    )

    # 0-D scalar tensor test (including boolean scalar parity with ATen)
    x0 = torch.tensor(7, dtype=torch.int32, device=device)
    out0 = torch.empty((), dtype=torch.float32, device=device)
    torch.cumprod(x0, dim=0, out=out0)
    utils.assert_close(
        out0, torch.tensor(7.0, dtype=torch.float32, device=device)
    )
    x0_bool = torch.tensor(False, dtype=torch.bool, device=device)
    out0_bool = torch.empty((), dtype=torch.bool, device=device)
    torch.cumprod(x0_bool, dim=0, out=out0_bool)
    self.assertFalse(out0_bool.item())

    # 0-D complex to float test
    x0_c = torch.tensor(3.0 + 4.0j, dtype=torch.complex64, device=device)
    out0_f = torch.empty((), dtype=torch.float32, device=device)
    torch.cumprod(x0_c, dim=0, out=out0_f)
    self.assertEqual(out0_f.item(), 3.0)

    # Empty tensor test
    xe = torch.empty(0, 4, dtype=torch.int32, device=device)
    oute = torch.empty(0, 4, dtype=torch.float32, device=device)
    torch.cumprod(xe, dim=1, out=oute)
    self.assertEqual(oute.shape, (0, 4))
    self.assertEqual(oute.dtype, torch.float32)

    # Integer default promotion to int64 (functional and out-variant)
    for int_dtype in [torch.int8, torch.int16, torch.int32]:
      xi_cpu = torch.tensor([1, 2, 3], dtype=int_dtype, device="cpu")
      xi_tpu = torch.tensor([1, 2, 3], dtype=int_dtype, device=device)

      # Functional variant
      res_cpu = torch.cumprod(xi_cpu, dim=0)
      res_tpu = torch.cumprod(xi_tpu, dim=0)
      self.assertEqual(res_cpu.dtype, torch.int64)
      self.assertEqual(res_tpu.dtype, torch.int64)
      utils.assert_close(res_tpu.cpu(), res_cpu)

      # Out-variant without dtype
      out_tpu = torch.empty(3, dtype=torch.int64, device=device)
      out_cpu = torch.empty(3, dtype=torch.int64, device="cpu")
      torch.cumprod(xi_tpu, dim=0, out=out_tpu)
      torch.cumprod(xi_cpu, dim=0, out=out_cpu)
      self.assertEqual(out_tpu.dtype, torch.int64)
      utils.assert_close(out_tpu.cpu(), out_cpu)

  def test_addcmul_mixed_dtypes(self):
    device = torch.device("tpu")
    # 1. int32 / bool self with float32 tensor1/tensor2
    s_i32 = torch.ones(2, dtype=torch.int32, device=device)
    t1_f32 = torch.full((2,), 2.0, dtype=torch.float32, device=device)
    t2_f32 = torch.full((2,), 3.0, dtype=torch.float32, device=device)
    res_i32 = torch.addcmul(s_i32, t1_f32, t2_f32)
    self.assertEqual(res_i32.dtype, torch.float32)
    utils.assert_close(
        res_i32.cpu(), torch.tensor([7.0, 7.0], dtype=torch.float32)
    )

    s_bool = torch.ones(2, dtype=torch.bool, device=device)
    res_bool = torch.addcmul(s_bool, t1_f32, t2_f32)
    self.assertEqual(res_bool.dtype, torch.float32)
    utils.assert_close(
        res_bool.cpu(), torch.tensor([7.0, 7.0], dtype=torch.float32)
    )

    # 2. float32 self with float64 tensor1/tensor2 (retaining float64 precision)
    s_f32 = torch.tensor(
        [-16777216.0, -33554432.0], dtype=torch.float32, device=device
    )
    t1_f64 = torch.tensor(
        [16777217.0, 33554433.0], dtype=torch.float64, device=device
    )
    t2_f64 = torch.ones(2, dtype=torch.float64, device=device)
    res_f64 = torch.addcmul(s_f32, t1_f64, t2_f64)
    self.assertEqual(res_f64.dtype, torch.float64)
    utils.assert_close(
        res_f64.cpu(), torch.tensor([1.0, 1.0], dtype=torch.float64)
    )

    # 3. int8 self with int32 tensor1/tensor2 (100 * 100 = 10000 in int32)
    s_i8 = torch.zeros(2, dtype=torch.int8, device=device)
    t_i32 = torch.full((2,), 100, dtype=torch.int32, device=device)
    res_i8_i32 = torch.addcmul(s_i8, t_i32, t_i32)
    self.assertEqual(res_i8_i32.dtype, torch.int32)
    utils.assert_close(
        res_i8_i32.cpu(), torch.tensor([10000, 10000], dtype=torch.int32)
    )

  def test_addcdiv_default_value_build_mlir(self):
    """Verifies addcdiv with value=1 traces without extra parameter."""
    for value_kwargs in ({}, {"value": 1}, {"value": 1.0}):
      with self.subTest(value_kwargs=value_kwargs):
        s = torch.ones(2, 2, device="tpu")
        t1 = torch.ones(2, 2, device="tpu")
        t2 = torch.ones(2, 2, device="tpu")
        torch.accelerator.synchronize()
        with execution_mode.set_eager_mode(
            execution_mode.EagerMode.DEFER_AND_FUSE
        ):
          z = torch.addcdiv(s, t1, t2, **value_kwargs)
        mlir_text = tpu_torch_compile.serialize_mlir_text(
            tpu_torch_compile.build_mlir([z], [s, t1, t2])
        )
        self.assertIn("stablehlo.divide", mlir_text)
        self.assertNotIn("stablehlo.multiply", mlir_text)
        utils.assert_close(
            z.cpu(), torch.full((2, 2), 2.0, dtype=torch.float32)
        )

  def test_addcmul_default_value_build_mlir(self) -> None:
    """Verifies addcmul with default value=1 traces without extra parameter."""
    for value_kwargs in ({}, {"value": 1}, {"value": 1.0}):
      with self.subTest(value_kwargs=value_kwargs):
        s = torch.ones(2, 2, device="tpu")
        t1 = torch.ones(2, 2, device="tpu")
        t2 = torch.ones(2, 2, device="tpu")
        torch.accelerator.synchronize()
        with execution_mode.set_eager_mode(
            execution_mode.EagerMode.DEFER_AND_FUSE
        ):
          z = torch.addcmul(s, t1, t2, **value_kwargs)
        mlir_text = tpu_torch_compile.serialize_mlir_text(
            tpu_torch_compile.build_mlir([z], [s, t1, t2])
        )
        self.assertEqual(mlir_text.count("stablehlo.multiply"), 1)
        self.assertEqual(mlir_text.count("stablehlo.add"), 1)
        self.assertNotIn("%arg3", mlir_text)
        utils.assert_close(
            z.cpu(), torch.full((2, 2), 2.0, dtype=torch.float32)
        )
    s = torch.ones(2, 2, device="tpu")
    t1 = torch.ones(2, 2, device="tpu")
    t2 = torch.ones(2, 2, device="tpu")
    z_val2 = torch.addcmul(s, t1, t2, value=2)
    utils.assert_close(
        z_val2.cpu(), torch.full((2, 2), 3.0, dtype=torch.float32)
    )

  def test_binary_arithmetic_out_upcast(self):
    device = torch.device("tpu")
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32, device=device)

    # add: float32 -> float64 upcast
    out_add = torch.empty(3, dtype=torch.float64, device=device)
    torch.add(a, b, out=out_add)
    self.assertEqual(out_add.dtype, torch.float64)
    utils.assert_close(
        out_add,
        torch.tensor([5.0, 7.0, 9.0], dtype=torch.float64, device=device),
    )

    # sub: float32 -> float64 upcast
    out_sub = torch.empty(3, dtype=torch.float64, device=device)
    torch.sub(a, b, out=out_sub)
    self.assertEqual(out_sub.dtype, torch.float64)
    utils.assert_close(
        out_sub,
        torch.tensor([-3.0, -3.0, -3.0], dtype=torch.float64, device=device),
    )

    # mul: float32 -> float64 upcast
    out_mul = torch.empty(3, dtype=torch.float64, device=device)
    torch.mul(a, b, out=out_mul)
    self.assertEqual(out_mul.dtype, torch.float64)
    utils.assert_close(
        out_mul,
        torch.tensor([4.0, 10.0, 18.0], dtype=torch.float64, device=device),
    )

    # div: int32 -> float32 promotion and upcast
    a_int = torch.tensor([1, 2, 4], dtype=torch.int32, device=device)
    b_int = torch.tensor([2, 4, 8], dtype=torch.int32, device=device)
    out_div = torch.empty(3, dtype=torch.float32, device=device)
    torch.div(a_int, b_int, out=out_div)
    self.assertEqual(out_div.dtype, torch.float32)
    utils.assert_close(
        out_div,
        torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=device),
    )

    # pow: float32 -> float64 upcast
    out_pow = torch.empty(3, dtype=torch.float64, device=device)
    torch.pow(a, b, out=out_pow)
    self.assertEqual(out_pow.dtype, torch.float64)
    utils.assert_close(
        out_pow,
        torch.tensor([1.0, 32.0, 729.0], dtype=torch.float64, device=device),
    )

  def test_ldexp_out_upcast(self):
    device = torch.device("tpu")
    # ldexp: float32 -> float64 upcast
    x = torch.tensor([2.0, 3.0], dtype=torch.float32, device=device)
    exp = torch.tensor([2, 3], dtype=torch.int32, device=device)
    out_ldexp = torch.empty(2, dtype=torch.float64, device=device)
    torch.ldexp(x, exp, out=out_ldexp)
    self.assertEqual(out_ldexp.dtype, torch.float64)
    utils.assert_close(
        out_ldexp,
        torch.tensor([8.0, 24.0], dtype=torch.float64, device=device),
    )

  def test_sigmoid_out_upcast(self):
    device = torch.device("tpu")
    x = torch.tensor([0.0, 1.0], dtype=torch.float32, device=device)
    out = torch.empty(2, dtype=torch.float64, device=device)
    torch.sigmoid(x, out=out)
    self.assertEqual(out.dtype, torch.float64)
    utils.assert_close(
        out,
        torch.sigmoid(x).to(torch.float64),
    )

  @parameterized.parameters(torch.float32, torch.float16, torch.bfloat16)
  def test_tanh_edge_cases(self, dtype):
    small_val = 1e-15 if dtype == torch.float32 else 1e-3
    x_init = torch.tensor(
        [
            -float("inf"),
            -100.0,
            -20.0,
            -small_val,
            -0.0,
            0.0,
            small_val,
            20.0,
            100.0,
            float("inf"),
            float("nan"),
        ],
        dtype=dtype,
    )
    grad_out = torch.ones_like(x_init)

    def compute(device):
      x = to(x_init.clone(), device=device).requires_grad_(True)
      out = torch.tanh(x)
      out.backward(to(grad_out, device=device))
      return out, x.grad

    self.assert_close_tpu_vs_cpu(compute)

  def test_logit_out_upcast(self):
    device = torch.device("tpu")
    x = torch.tensor([0.2, 0.8], dtype=torch.float32, device=device)
    out = torch.empty(2, dtype=torch.float64, device=device)
    torch.logit(x, out=out)
    self.assertEqual(out.dtype, torch.float64)
    utils.assert_close(
        out,
        torch.logit(x).to(torch.float64),
    )

  def test_logit_backward_nan_propagation(self):
    device = torch.device("tpu")
    grad_out = torch.tensor([1.0, 1.0, float("nan")], dtype=torch.float32)
    x = torch.tensor([float("nan"), 0.5, 0.0], dtype=torch.float32)
    for eps in (1e-6, 0.0):
      res_tpu = torch.ops.aten.logit_backward(
          grad_out.to(device), x.to(device), eps
      ).cpu()
      res_cpu = torch.ops.aten.logit_backward(grad_out, x, eps)
      self.assertTrue(torch.isnan(res_tpu[0]))
      utils.assert_close(res_tpu, res_cpu)

  def test_complex_and_polar_out(self):
    device = torch.device("tpu")
    # complex: float32 -> complex64 out
    real_f32 = torch.tensor([3.0, 1.0], dtype=torch.float32, device=device)
    imag_f32 = torch.tensor([4.0, 2.0], dtype=torch.float32, device=device)
    out_c64 = torch.empty(2, dtype=torch.complex64, device=device)
    torch.complex(real_f32, imag_f32, out=out_c64)
    self.assertEqual(out_c64.dtype, torch.complex64)
    utils.assert_close(
        out_c64,
        torch.tensor(
            [3.0 + 4.0j, 1.0 + 2.0j], dtype=torch.complex64, device=device
        ),
    )

    # polar: float32 -> complex64 out
    abs_f32 = torch.tensor([2.0], dtype=torch.float32, device=device)
    angle_f32 = torch.tensor([0.0], dtype=torch.float32, device=device)
    out_polar_c64 = torch.empty(1, dtype=torch.complex64, device=device)
    torch.polar(abs_f32, angle_f32, out=out_polar_c64)
    self.assertEqual(out_polar_c64.dtype, torch.complex64)
    utils.assert_close(
        out_polar_c64,
        torch.tensor([2.0 + 0.0j], dtype=torch.complex64, device=device),
    )

  def test_bool_abs(self):
    device = torch.device("tpu")
    x = torch.tensor(
        [[True, False], [False, True]], dtype=torch.bool, device=device
    )
    utils.assert_close(torch.abs(x), x.cpu())

    out = torch.empty_like(x)
    torch.abs(x, out=out)
    # torch.abs() on booleans is an identity function.
    utils.assert_close(out, x.cpu())

    y = torch.tensor(
        [[False, True], [True, False]], dtype=torch.bool, device=device
    )
    res = torch._foreach_abs([x, y])
    utils.assert_close(res[0], x.cpu())
    utils.assert_close(res[1], y.cpu())

  def test_gather_scalar_grad(self):
    del self  # self is unused in this test.
    device = torch.device("tpu")
    input_val = 4.901_066_981_864_172
    input_tensor_tpu = torch.tensor(
        input_val, requires_grad=True, device=device
    )
    dim = 0
    index_tpu = torch.tensor([0], device=device, dtype=torch.int64)

    output_tpu = torch.gather(input_tensor_tpu, dim, index_tpu)
    output_tpu.sum().backward()

    input_tensor_cpu = torch.tensor(input_val, requires_grad=True, device="cpu")
    index_cpu = torch.tensor([0], device="cpu", dtype=torch.int64)
    output_cpu = torch.gather(input_tensor_cpu, dim, index_cpu)
    output_cpu.sum().backward()

    utils.assert_close(input_tensor_tpu.grad.cpu(), input_tensor_cpu.grad)

  def test_gather_scalar_multi_index_grad(self):
    del self  # self is unused in this test.
    device = torch.device("tpu")
    input_val = 4.901_066_981_864_172
    self_tpu = torch.tensor([input_val], requires_grad=True, device=device)
    index_tpu = torch.tensor([0, 0], device=device, dtype=torch.int64)
    dim = 0

    output_tpu = torch.gather(self_tpu, dim, index_tpu)
    output_tpu.sum().backward()

    expected_grad = torch.tensor([2.0])
    utils.assert_close(self_tpu.grad.cpu(), expected_grad)

  def test_gather_empty_dim(self):
    def test_fn(device):
      self_tensor = torch.randn(0, 7, device=device)
      index = torch.empty(0, 7, dtype=torch.long, device=device)
      return torch.gather(self_tensor, 0, index)

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_gather_empty_batch_dim(self):
    """Tests gather where index has size 0 along a batch dimension (d != dim)."""

    def test_fn(device):
      self_tensor = torch.randn(3, 5, device=device)
      index = torch.empty(0, 2, dtype=torch.long, device=device)
      return torch.gather(self_tensor, 1, index)

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_gather_scatter_zero_stride_and_size1_broadcast(self):
    """Tests eager and compiled gather/scatter_add on expanded broadcast indices."""
    for dtype in (torch.float32, torch.bfloat16):
      cpu_x = torch.randn(1, 4, 16, 32, dtype=dtype, requires_grad=True)
      cpu_base_idx = torch.randint(0, 32, (1, 1, 16, 16), dtype=torch.long)
      cpu_exp_idx = cpu_base_idx.expand(1, 4, 16, 16)
      cpu_out = torch.gather(cpu_x, -1, cpu_exp_idx)
      cpu_grad_out = torch.randn_like(cpu_out)
      cpu_out.backward(cpu_grad_out)

      tpu_x = cpu_x.detach().to("tpu").requires_grad_(True)
      tpu_base_idx = cpu_base_idx.to("tpu")
      tpu_exp_idx = tpu_base_idx.expand(1, 4, 16, 16)
      self.assertEqual(tpu_exp_idx.stride(1), 0)
      tpu_out = torch.gather(tpu_x, -1, tpu_exp_idx)
      tpu_out.backward(cpu_grad_out.to("tpu"))
      utils.assert_close(actual=tpu_out.cpu(), expected=cpu_out)
      utils.assert_close(actual=tpu_x.grad.cpu(), expected=cpu_x.grad)

      @torch.compile(dynamic=False)
      def compiled_fn(x, base_idx, grad_out):
        idx = base_idx.expand(1, 4, 16, 16)
        out = torch.gather(x, -1, idx)
        grad_in = torch.zeros_like(x).scatter_add_(-1, idx, grad_out)
        return out, grad_in

      comp_out, comp_grad = compiled_fn(
          cpu_x.detach().to("tpu"),
          cpu_base_idx.to("tpu"),
          cpu_grad_out.to("tpu"),
      )
      utils.assert_close(actual=comp_out.cpu(), expected=cpu_out)
      utils.assert_close(actual=comp_grad.cpu(), expected=cpu_x.grad)

  @parameterized.product(
      input_dtype=[torch.int32, torch.int64],
      op_fn=[
          torch.ops.aten.__ilshift__.Scalar,
          torch.ops.aten.__irshift__.Scalar,
      ],
  )
  def test__ishift__Scalar(self, input_dtype, op_fn):
    """Tests the __ilshift__.Scalar and __irshift__.Scalar op."""
    tpu_device = torch.device("tpu")
    x = torch.tensor([1, 2, 3], dtype=input_dtype)
    x_tpu = x.to(tpu_device)
    op_fn(x, 2)
    op_fn(x_tpu, 2)
    self.assertEqual(x, x_tpu.cpu())

  @parameterized.product(
      self_dtype=[torch.int32, torch.int64],
      other_dtype=[torch.int32, torch.int64],
      op_fn=[
          torch.ops.aten.__ilshift__.Tensor,
          torch.ops.aten.__irshift__.Tensor,
      ],
  )
  def test__ishift__Tensor(self, self_dtype, other_dtype, op_fn):
    """Tests the __ilshift__.Tensor and __irshift__.Tensor op."""
    tpu_device = torch.device("tpu")
    self_tensor = torch.tensor([1, 2, 3], dtype=self_dtype)
    self_tensor_tpu = self_tensor.to(tpu_device)
    other_tensor = torch.tensor([1, 2, 3], dtype=other_dtype)
    other_tensor_tpu = other_tensor.to(tpu_device)
    op_fn(self_tensor, other_tensor)
    op_fn(self_tensor_tpu, other_tensor_tpu)
    self.assert_close(
        golden_result=self_tensor, torch_tpu_result=self_tensor_tpu.cpu()
    )

  @parameterized.product(
      input_dtype=[torch.uint8, torch.int32, torch.int64],
      op_fn=[
          torch.ops.aten.__lshift__.Scalar,
          torch.ops.aten.__rshift__.Scalar,
      ],
  )
  def test__shift__Scalar(self, input_dtype, op_fn):
    """Tests the __lshift__.Scalar and __rshift__.Scalar op."""
    tpu_device = torch.device("tpu")
    x = torch.tensor([1, 2, 3, 128], dtype=input_dtype)
    x_tpu = x.to(tpu_device)
    out = op_fn(x, 2)
    out_tpu = op_fn(x_tpu, 2)
    self.assert_close(golden_result=out, torch_tpu_result=out_tpu.cpu())

  @parameterized.product(
      self_dtype=[torch.uint8, torch.int32, torch.int64],
      other_dtype=[torch.int32, torch.int64],
      op_fn=[
          torch.ops.aten.__rshift__.Tensor,
          torch.ops.aten.__lshift__.Tensor,
      ],
  )
  def test__shift__Tensor(self, self_dtype, other_dtype, op_fn):
    """Tests the __lshift__.Tensor and __rshift__.Tensor op."""
    tpu_device = torch.device("tpu")
    self_tensor = torch.tensor([128, 128, 128], dtype=self_dtype)
    self_tensor_tpu = self_tensor.to(tpu_device)
    other_tensor = torch.tensor([1, 2, 3], dtype=other_dtype)
    other_tensor_tpu = other_tensor.to(tpu_device)
    golden_result = op_fn(self_tensor, other_tensor)
    tpu_result = op_fn(self_tensor_tpu, other_tensor_tpu)
    self.assert_close(
        golden_result=golden_result, torch_tpu_result=tpu_result.cpu()
    )

  @parameterized.product(
      self_dtype=[torch.int32, torch.float32],
      other_dtype=[torch.int32, torch.float32],
      op_fn=[
          torch.ops.aten.__ilshift__.Tensor,
          torch.ops.aten.__irshift__.Tensor,
          torch.ops.aten.__lshift__.Tensor,
          torch.ops.aten.__rshift__.Tensor,
      ],
  )
  def test_unsupported_shift_Tensor_dtypes(
      self, self_dtype, other_dtype, op_fn
  ):
    """Tests the bitwise shift ops with unsupported dtypes."""
    self.assert_close_tpu_vs_cpu(
        lambda device: op_fn(
            torch.tensor([1, 2, 3], dtype=self_dtype).to(device),
            torch.tensor([1, 2, 3], dtype=other_dtype).to(device),
        ),
        check_exception_type=False,
        allow_failure=True,
    )

  @parameterized.product(
      self_dtype=[torch.int32, torch.float32],
      other_value=[2, 2.0],
      op_fn=[
          torch.ops.aten.__ilshift__.Scalar,
          torch.ops.aten.__irshift__.Scalar,
          torch.ops.aten.__lshift__.Scalar,
          torch.ops.aten.__rshift__.Scalar,
      ],
  )
  def test_unsupported_shift_Scalar_dtypes(
      self, self_dtype, other_value, op_fn
  ):
    """Tests the bitwise shift ops with unsupported dtypes."""
    self.assert_close_tpu_vs_cpu(
        lambda device: op_fn(
            torch.tensor([1, 2, 3], dtype=self_dtype).to(device),
            other_value,
        ),
        check_exception_type=False,
        allow_failure=True,
    )

  def test_empty_tensor_empty_index_in_take(self):
    """Tests that torch.take() works when the input tensor and index are both empty."""
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.take(
            torch.tensor([], dtype=torch.float32, device=device),
            torch.tensor([], dtype=torch.int64, device=device),
        )
    )

  def test__add_relu_Scalar(self):
    """Tests _add_relu.Scalar with various inputs and dtypes."""
    # Case 1: Standard positive
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ops.aten._add_relu.Scalar(x1.to(device), 2.0, 1.0)
    )

    # Case 2: Negative values (ReLU clamping)
    x2 = torch.tensor([[-1.0, -5.0], [1.0, 2.0]], dtype=torch.float32)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ops.aten._add_relu.Scalar(x2.to(device), -1.0, 1.0)
    )

    # Case 3: bfloat16 (PyTorch CPU doesn't support _add_relu.Scalar,
    # use separate ops on CPU for reference)
    x3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    self.assert_close_tpu_vs_cpu(
        lambda device: (
            torch.relu(x3 + 2.0 * 0.5)
            if device == "cpu"
            else torch.ops.aten._add_relu.Scalar(x3.to(device), 2.0, 0.5)
        )
    )

    # Case 4: Non-contiguous
    x4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).t()
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ops.aten._add_relu.Scalar(x4.to(device), 2.0, 1.0)
    )

  def test__add_relu_Tensor(self):
    """Tests _add_relu.Tensor with various inputs and dtypes."""
    # Case 1: Standard positive
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    y1 = torch.tensor([[2.0, 3.0], [1.0, 5.0]], dtype=torch.float32)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ops.aten._add_relu.Tensor(
            x1.to(device), y1.to(device), alpha=1.0
        )
    )

    # Case 2: Negative values (ReLU clamping)
    x2 = torch.tensor([[-1.0, -5.0], [1.0, 2.0]], dtype=torch.float32)
    y2 = torch.tensor([[-2.0, 1.0], [-1.0, -3.0]], dtype=torch.float32)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ops.aten._add_relu.Tensor(
            x2.to(device), y2.to(device), alpha=1.0
        )
    )

    # Case 3: bfloat16
    x3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    y3 = torch.tensor([[2.0, 3.0], [1.0, 5.0]], dtype=torch.bfloat16)

    def compute_add_relu_tensor(device):
      if device == "cpu":
        # CPU backend might not support _add_relu.Tensor directly
        return torch.relu(x3 + y3 * 0.5)
      else:
        return torch.ops.aten._add_relu.Tensor(
            x3.to(device), y3.to(device), alpha=0.5
        )

    self.assert_close_tpu_vs_cpu(compute_add_relu_tensor)

  def test__add_relu_out(self):
    """Tests _add_relu.out with various inputs and dtypes."""
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    y1 = torch.tensor([[2.0, 3.0], [1.0, 5.0]], dtype=torch.float32)

    def test_fn(device):
      out = torch.empty(2, 2, dtype=torch.float32, device=device)
      if device == "cpu":
        # CPU backend might not support _add_relu.out directly
        return torch.relu(x1 + y1 * 1.0)
      torch.ops.aten._add_relu.out(
          x1.to(device), y1.to(device), alpha=1.0, out=out
      )
      return out

    self.assert_close_tpu_vs_cpu(test_fn)

  def test__sub_out(self):
    """Tests sub.out with various inputs and dtypes."""
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    y1 = torch.tensor([[2.0, 3.0], [1.0, 5.0]], dtype=torch.float32)

    def test_fn(device):
      out = torch.empty(2, 2, dtype=torch.float32, device=device)
      torch.sub(x1.to(device), y1.to(device), alpha=1.0, out=out)
      return out

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_rsub_with_alpha(self):
    """Tests torch.rsub operator with alpha scaling factor != 1."""
    self_tensor = torch.tensor([10.0, 20.0, 30.0])
    other_tensor = torch.tensor([100.0, 200.0, 300.0])
    alpha = 2.5

    def test_fn(device):
      return torch.rsub(
          self_tensor.to(device), other_tensor.to(device), alpha=alpha
      )

    self.assert_close_tpu_vs_cpu(test_fn)

  def test__add_relu__Scalar(self):
    """Tests _add_relu_.Scalar with various inputs and dtypes."""
    x1 = torch.tensor([[1.0, -2.0], [-3.0, 4.0]], dtype=torch.float32)

    def test_fn(device):
      t = x1.clone().to(device)
      if device == "cpu":
        # CPU backend might not support add_relu_.Scalar directly
        return torch.relu(t + 2.0 * 1.0)
      torch.ops.aten._add_relu_.Scalar(t, 2.0, alpha=1.0)
      return t

    self.assert_close_tpu_vs_cpu(test_fn)

  def test__add_relu__Tensor(self):
    """Tests _add_relu_.Tensor with various inputs and dtypes."""
    x1 = torch.tensor([[1.0, -2.0], [-3.0, 4.0]], dtype=torch.float32)
    y1 = torch.tensor([[2.0, 3.0], [-1.0, -5.0]], dtype=torch.float32)

    def test_fn(device):
      t = x1.clone().to(device)
      if device == "cpu":
        # CPU backend might not support _add_relu_.Tensor directly
        return torch.relu(t + y1.to(device) * 1.0)
      torch.ops.aten._add_relu_.Tensor(t, y1.to(device), alpha=1.0)
      return t

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_ones_grad(self):
    """Tests autograd for ones()."""

    def get_grad(device):
      t = torch.ones(2, 3, device=device)
      t.requires_grad = True
      loss = torch.sum(t)
      loss.backward()
      return t.grad

    self.assert_close_tpu_vs_cpu(get_grad)

  def test_to(self):
    """Tests the op_testing.to() function."""

    # Test to() with simple types and the device argument being a torch.device.
    gpu = torch.device("cuda")
    tpu = to(gpu, torch.device("tpu"))
    self.assertEqual(tpu, torch.device("tpu"))
    gpu_dict = {"device": gpu}
    tpu_dict = to(gpu_dict, torch.device("tpu"))
    self.assertEqual(tpu_dict, {"device": torch.device("tpu")})

    # Test to() with simple types and the device argument being a string.
    gpu = "cuda"
    tpu = to(gpu, "tpu")
    self.assertEqual(tpu, "tpu")
    gpu_dict = {"device": gpu}
    tpu_dict = to(gpu_dict, "tpu")
    self.assertEqual(tpu_dict, {"device": "tpu"})

    # Test to() with a complex type we care about (OpInput).
    gpu_input = OpInput(
        FakeSample(
            "sample1", torch.zeros(1, device="cpu"), (1,), {"device": gpu}
        )
    )
    tpu_input = to(gpu_input, torch.device("tpu"))
    self.assertEqual(tpu_input.name, "sample1")
    self.assert_devices_equivalent(
        tpu_input.input_value.device, torch.device("tpu")
    )
    self.assertEqual(tpu_input.args, (1,))
    self.assertEqual(tpu_input.kwargs, {"device": "tpu"})

  def test_batch_norm_forward_mixed_dtype_training(self):
    """Tests batch_norm forward in training mode with mixed BF16 input and F32 stats."""
    input_dtype = torch.bfloat16
    stats_dtype = torch.float32
    n, c, h, w = 2, 4, 4, 4

    # Create inputs
    input_val = torch.randn(n, c, h, w, dtype=input_dtype)
    weight = torch.randn(c, dtype=stats_dtype)
    bias = torch.randn(c, dtype=stats_dtype)
    running_mean = torch.randn(c, dtype=stats_dtype)
    running_var = torch.rand(c, dtype=stats_dtype)

    def run_forward(device):
      return torch.ops.aten.native_batch_norm(
          input_val.to(device),
          weight.to(device),
          bias.to(device),
          running_mean.to(device),
          running_var.to(device),
          training=True,
          momentum=0.1,
          eps=1e-5,
      )

    # Comparison results: (output, save_mean, save_invstd)
    # output should be BF16, stats should be F32
    # Relax tolerances because we are running in BF16 on TPU for performance.
    self.assert_close_tpu_vs_cpu(run_forward, rtol=3e-2, atol=2e-2)

  @parameterized.parameters(0.1, None)
  def test_batch_norm_training_momentum_does_not_recompile(self, momentum):
    """Tests that BatchNorm training steps reuse the compiled program.

    With `momentum=None`, `nn.BatchNorm2d` passes `1 / num_batches_tracked` as
    the momentum, so its value changes on every step. The momentum must not
    cause a new compilation on every step.

    Args:
      momentum: The momentum of the BatchNorm module.
    """
    x = torch.randn(8, 3, 16, 16).to("tpu")
    bn = torch.nn.BatchNorm2d(3, momentum=momentum).to("tpu").train()

    compilations_per_step = []
    for _ in range(6):
      before = torch.tpu._get_cache_stats().num_compilations
      bn(x).cpu()
      compilations_per_step.append(
          torch.tpu._get_cache_stats().num_compilations - before
      )

    # The first step may compile; later steps must hit the compilation cache.
    self.assertEqual(
        compilations_per_step[1:],
        [0] * 5,
        f"compilations per step: {compilations_per_step}",
    )

  @parameterized.product(training=[True, False], has_weight=[True, False])
  def test_batch_norm_backward_mixed_dtype(self, training, has_weight):
    """Tests batch_norm_backward with mixed BF16 input and F32 stats."""
    # This scenario happens often in mixed precision training where
    # inputs/grads are BF16 but running stats are kept in F32.
    # The optimization in TpuBatchNormBackward avoids upcasting inputs to F32.
    input_dtype = torch.bfloat16
    stats_dtype = torch.float32
    n, c, h, w = 2, 4, 4, 4

    # Create inputs
    grad_out = torch.randn(n, c, h, w, dtype=input_dtype)
    input_val = torch.randn(n, c, h, w, dtype=input_dtype)
    weight = torch.randn(c, dtype=stats_dtype) if has_weight else None
    running_mean = torch.randn(c, dtype=stats_dtype)
    running_var = torch.rand(c, dtype=stats_dtype)  # Positive variance
    save_mean = torch.randn(c, dtype=stats_dtype)
    save_invstd = torch.rand(c, dtype=stats_dtype)  # Positive invstd

    def run_backward(device):
      return torch.ops.aten.native_batch_norm_backward(
          grad_out.to(device),
          input_val.to(device),
          weight.to(device) if weight is not None else None,
          running_mean.to(device),
          running_var.to(device),
          save_mean.to(device),
          save_invstd.to(device),
          training,
          1e-5,  # eps
          [True, True, True],  # output_mask
      )

    # Relax tolerances because we are running in BF16 on TPU for performance.
    self.assert_close_tpu_vs_cpu(run_backward, rtol=6e-2, atol=4e-2)

  @parameterized.product(
      dtype=[torch.float32, torch.bfloat16, torch.float16],
      stats_dtype=[torch.float32, None],
  )
  def test_batch_norm_backward_training_no_weight_with_running_stats(
      self, dtype, stats_dtype
  ):
    """Tests training backward with weight=None and running statistics present."""
    effective_stats_dtype = stats_dtype if stats_dtype is not None else dtype
    n, c, h, w = 2, 4, 8, 8
    grad_out = torch.randn(n, c, h, w, dtype=dtype)
    input_val = torch.randn(n, c, h, w, dtype=dtype)
    running_mean = torch.randn(c, dtype=effective_stats_dtype)
    running_var = torch.rand(c, dtype=effective_stats_dtype) + 0.1
    save_mean = torch.randn(c, dtype=effective_stats_dtype)
    save_invstd = torch.rand(c, dtype=effective_stats_dtype) + 0.1

    def run_backward(device):
      return torch.ops.aten.native_batch_norm_backward(
          grad_out.to(device),
          input_val.to(device),
          None,
          running_mean.to(device),
          running_var.to(device),
          save_mean.to(device),
          save_invstd.to(device),
          True,  # training
          1e-5,  # eps
          [True, True, True],
      )

    tpu_grads = run_backward("tpu")
    self.assertEqual(tpu_grads[1].dtype, effective_stats_dtype)
    self.assertEqual(tpu_grads[2].dtype, effective_stats_dtype)
    rtol = 5e-2 if dtype != torch.float32 else 1e-4
    atol = 5e-2 if dtype != torch.float32 else 1e-4
    self.assert_close_tpu_vs_cpu(run_backward, rtol=rtol, atol=atol)

  @parameterized.product(
      dtype=[torch.float32, torch.bfloat16, torch.float16],
      has_weight=[True, False],
  )
  def test_native_batch_norm_backward_inference_grad_weight_precision(
      self, dtype, has_weight
  ):
    """Tests inference native_batch_norm_backward grad_weight accumulation precision."""
    n, c, h, w = 4, 4, 16, 16
    stats_dtype = torch.float32
    grad_out = torch.randn(n, c, h, w, dtype=dtype)
    input_val = torch.randn(n, c, h, w, dtype=dtype) + 50.0
    weight = torch.randn(c, dtype=stats_dtype) if has_weight else None
    running_mean = torch.full((c,), 50.0, dtype=stats_dtype)
    running_var = torch.rand(c, dtype=stats_dtype) * 0.05 + 1e-3

    def run_backward(device):
      return torch.ops.aten.native_batch_norm_backward(
          grad_out.to(device),
          input_val.to(device),
          weight.to(device) if weight is not None else None,
          running_mean.to(device),
          running_var.to(device),
          None,
          None,
          False,  # training
          1e-5,  # eps
          [True, True, True],
      )

    rtol = 5e-2 if dtype != torch.float32 else 1e-4
    atol = 5e-2 if dtype != torch.float32 else 1e-4
    self.assert_close_tpu_vs_cpu(run_backward, rtol=rtol, atol=atol)

  @parameterized.product(training=[True, False])
  def test_batch_norm_backward_extreme_stats(self, training):
    """Tests batch_norm_backward with mixed BF16 input and extreme F32 stats."""
    # Large mean and small variance can expose numerical precision issues.
    input_dtype = torch.bfloat16
    stats_dtype = torch.float32
    n, c, h, w = 2, 4, 4, 4

    # Create inputs
    grad_out = torch.randn(n, c, h, w, dtype=input_dtype)
    input_val = torch.randn(n, c, h, w, dtype=input_dtype)
    weight = torch.randn(c, dtype=stats_dtype)
    running_mean = torch.randn(c, dtype=stats_dtype)
    running_var = torch.rand(c, dtype=stats_dtype) * 0.1  # Small variance

    # save_mean/invstd are what's actually used in training backward.
    save_mean = torch.randn(c, dtype=stats_dtype) + 2000.0  # Large mean
    save_invstd = torch.rand(c, dtype=stats_dtype) * 10.0 + 5.0  # Small var

    def run_backward(device):
      return torch.ops.aten.native_batch_norm_backward(
          grad_out.to(device),
          input_val.to(device),
          weight.to(device),
          running_mean.to(device),
          running_var.to(device),
          save_mean.to(device),
          save_invstd.to(device),
          training,
          1e-5,  # eps
          [True, True, True],  # output_mask
      )

    # Relax tolerances because we are running in BF16 on TPU for performance.
    self.assert_close_tpu_vs_cpu(run_backward, rtol=6e-2, atol=4e-2)

  @parameterized.product(
      shape=[(2, 4, 8, 8), (4, 3, 16), (2, 5, 4, 4, 4)],
      dtype=[torch.float32, torch.bfloat16],
      has_weight=[True, False],
      mask=[
          [True, True, True],
          [True, False, False],
          [False, True, False],
          [False, False, True],
          [True, True, False],
      ],
  )
  def test_native_batch_norm_backward_comprehensive(
      self, shape, dtype, has_weight, mask
  ):
    """Tests training batch_norm_backward across shapes, dtypes, weights, and masks."""
    c = shape[1]
    stats_dtype = torch.float32
    grad_out = torch.randn(*shape, dtype=dtype)
    input_val = torch.randn(*shape, dtype=dtype)
    weight = torch.randn(c, dtype=stats_dtype) if has_weight else None
    save_mean = torch.randn(c, dtype=stats_dtype)
    save_invstd = torch.rand(c, dtype=stats_dtype) + 0.1

    def run_backward(device):
      return torch.ops.aten.native_batch_norm_backward(
          grad_out.to(device),
          input_val.to(device),
          weight.to(device) if weight is not None else None,
          None,
          None,
          save_mean.to(device),
          save_invstd.to(device),
          True,  # training
          1e-5,  # eps
          mask,
      )

    rtol = 5e-2 if dtype == torch.bfloat16 else 1e-4
    atol = 5e-2 if dtype == torch.bfloat16 else 1e-4
    self.assert_close_tpu_vs_cpu(run_backward, rtol=rtol, atol=atol)

  @parameterized.product(
      dtype=[torch.float32, torch.bfloat16],
      has_weight=[True, False],
  )
  def test_native_batch_norm_backward_zero_batch(self, dtype, has_weight):
    """Tests training batch_norm_backward on zero-batch inputs (where forward stats are NaN)."""
    shape = (0, 4, 8, 8)
    c = shape[1]
    stats_dtype = torch.float32
    grad_out = torch.empty(*shape, dtype=dtype, device="tpu")
    input_val = torch.empty(*shape, dtype=dtype, device="tpu")
    weight = (
        torch.randn(c, dtype=stats_dtype, device="tpu") if has_weight else None
    )
    save_mean = torch.full((c,), float("nan"), dtype=stats_dtype, device="tpu")
    save_invstd = torch.full(
        (c,), float("nan"), dtype=stats_dtype, device="tpu"
    )

    grad_input, grad_weight, grad_bias = (
        torch.ops.aten.native_batch_norm_backward(
            grad_out,
            input_val,
            weight,
            None,
            None,
            save_mean,
            save_invstd,
            True,
            1e-5,
            [True, True, True],
        )
    )
    self.assertEqual(grad_input.shape, shape)
    self.assertEqual(grad_input.dtype, dtype)
    expected_zeros = torch.zeros(c, dtype=stats_dtype)
    self.assert_close(
        golden_result=expected_zeros, torch_tpu_result=grad_weight.cpu()
    )
    self.assert_close(
        golden_result=expected_zeros, torch_tpu_result=grad_bias.cpu()
    )

  @parameterized.product(
      has_weight=[True, False],
      training=[True, False],
      dynamic_dim=[0, 2],
  )
  def test_native_batch_norm_backward_dynamic(
      self, has_weight, training, dynamic_dim
  ):
    """Tests batch_norm_backward with bounded dynamic batch or spatial dimension."""
    shape = (4, 3, 6, 6)
    c = shape[1]
    grad_out = torch.randn(*shape, dtype=torch.float32)
    input_val = torch.randn(*shape, dtype=torch.float32)
    weight = torch.randn(c, dtype=torch.float32) if has_weight else None
    running_mean = torch.randn(c, dtype=torch.float32)
    running_var = torch.rand(c, dtype=torch.float32) + 0.1
    save_mean = torch.randn(c, dtype=torch.float32)
    save_invstd = torch.rand(c, dtype=torch.float32) + 0.1

    def run_backward(device):
      g = grad_out.to(device)
      x = input_val.to(device)
      w = weight.to(device) if weight is not None else None
      if str(device).startswith("tpu"):
        dynamism.mark_dynamic(
            g, dimension=dynamic_dim, lower_bound=2, upper_bound=10
        )
        dynamism.mark_dynamic(
            x, dimension=dynamic_dim, lower_bound=2, upper_bound=10
        )
      return torch.ops.aten.native_batch_norm_backward(
          g,
          x,
          w,
          running_mean.to(device),
          running_var.to(device),
          save_mean.to(device),
          save_invstd.to(device),
          training,
          1e-5,
          [True, True, True],
      )

    self.assert_close_tpu_vs_cpu(run_backward, rtol=1e-4, atol=1e-4)

  def test_native_batch_norm_legit_out(self):
    input_dtype = torch.float32
    stats_dtype = torch.float32
    n, c, h, w = 2, 4, 4, 4
    input_val = torch.randn(n, c, h, w, dtype=input_dtype)
    weight = torch.randn(c, dtype=stats_dtype)
    bias = torch.randn(c, dtype=stats_dtype)
    running_mean = torch.randn(c, dtype=stats_dtype)
    running_var = torch.rand(c, dtype=stats_dtype).abs() + 1e-5

    def run_op(device):
      out = torch.empty(n, c, h, w, device=device, dtype=input_dtype)
      save_mean = torch.empty(c, device=device, dtype=stats_dtype)
      save_invstd = torch.empty(c, device=device, dtype=stats_dtype)
      return torch.ops.aten._native_batch_norm_legit.out(
          input_val.to(device),
          weight.to(device),
          bias.to(device),
          running_mean.to(device),
          running_var.to(device),
          True,
          0.1,
          1e-5,
          out=out,
          save_mean=save_mean,
          save_invstd=save_invstd,
      )

    self.assert_close_tpu_vs_cpu(run_op)

  @parameterized.product(
      op_name=["native_batch_norm", "_native_batch_norm_legit"],
      dtypes=[
          (torch.float32, torch.float32),
          (torch.bfloat16, torch.float32),
          (torch.bfloat16, torch.bfloat16),
      ],
  )
  def test_batch_norm_training_updates_running_stats_in_place(
      self, op_name, dtypes
  ):
    """Tests that training-mode batch norm updates running stats in place."""
    input_dtype, stats_dtype = dtypes
    c = 3
    input_val = torch.randn(4, c, 5, 6, dtype=input_dtype)
    weight = torch.randn(c, dtype=stats_dtype)
    bias = torch.randn(c, dtype=stats_dtype)
    running_mean = torch.randn(c, dtype=stats_dtype)
    running_var = torch.rand(c, dtype=stats_dtype) + 0.5

    def run_forward(device):
      # Clone, since `.to("cpu")` returns the same tensor and the op updates it
      # in place.
      mean = running_mean.clone().to(device)
      var = running_var.clone().to(device)
      outputs = getattr(torch.ops.aten, op_name)(
          input_val.to(device),
          weight.to(device),
          bias.to(device),
          mean,
          var,
          True,  # training
          0.3,  # momentum
          1e-5,  # eps
      )
      return (*outputs, mean, var)

    # Results: (output, save_mean, save_invstd, running_mean, running_var).
    # The running stats must match the CPU's in-place update.
    if stats_dtype == torch.float32:
      self.assert_close_tpu_vs_cpu(run_forward)
      return

    def pick(*indices, to_float=False):
      def fn(device):
        results = run_forward(device)
        return tuple(
            results[i].float() if to_float else results[i] for i in indices
        )

      return fn

    # With bfloat16 weight and bias, the CPU computes in bfloat16, while
    # TorchTPU computes in float32 and rounds once, so the bfloat16 results can
    # differ by one bfloat16 step.
    self.assert_close_tpu_vs_cpu(pick(0, 3, 4), rtol=1e-2, atol=1e-2)
    # Like CUDA, TorchTPU returns save_mean and save_invstd in float32; the CPU
    # returns them in bfloat16 here. Compare the values.
    self.assertEqual(run_forward("tpu")[1].dtype, torch.float32)
    self.assert_close_tpu_vs_cpu(
        pick(1, 2, to_float=True), rtol=1e-2, atol=1e-2
    )

  @parameterized.parameters("native_batch_norm", "_native_batch_norm_legit")
  def test_batch_norm_training_out_resizes_empty_outputs(self, op_name):
    """Tests training-mode batch norm with empty out tensors to resize."""
    c = 3
    input_val = torch.randn(4, c, 5, 6)
    weight = torch.randn(c)
    bias = torch.randn(c)
    running_mean = torch.randn(c)
    running_var = torch.rand(c) + 0.5

    def run_op(device):
      mean = running_mean.clone().to(device)
      var = running_var.clone().to(device)
      out = torch.empty(0, device=device)
      save_mean = torch.empty(0, device=device)
      save_invstd = torch.empty(0, device=device)
      getattr(torch.ops.aten, op_name).out(
          input_val.to(device),
          weight.to(device),
          bias.to(device),
          mean,
          var,
          True,  # training
          0.3,  # momentum
          1e-5,  # eps
          out=out,
          save_mean=save_mean,
          save_invstd=save_invstd,
      )
      return out, save_mean, save_invstd, mean, var

    self.assert_close_tpu_vs_cpu(run_op)

  @parameterized.product(momentum=[0.1, None], compiled=[False, True])
  def test_batch_norm_module_training_updates_running_stats(
      self, momentum, compiled
  ):
    """Tests that nn.BatchNorm2d running stats match the CPU after training.

    Each step runs the backward too: autograd saves the running stats, so
    updating them must not bump their version counters. Also tests that later
    steps do not compile again, although `momentum=None` passes a new momentum
    (1 / num_batches_tracked) each step. The first two steps are warm-up: the
    first TPU step in a process can compile a one-off program.
    """
    inputs = [torch.randn(8, 3, 16, 16) for _ in range(4)]
    cache_misses_after_warm_up = []

    def run_training(device):
      bn = torch.nn.BatchNorm2d(3, momentum=momentum).to(device).train()
      on_tpu = torch.device(device).type == "tpu"
      step = bn
      if compiled and on_tpu:
        step = torch.compile(bn, backend="tpu")
      for i, x in enumerate(inputs):
        output = step(x.to(device))
        output.sum().backward()
        if on_tpu and i == 1:
          start_misses = torch.tpu._get_cache_misses()
      if on_tpu:
        cache_misses_after_warm_up.append(
            torch.tpu._get_cache_misses() - start_misses
        )
      # Inference uses the running stats.
      bn.eval()
      with torch.no_grad():
        eval_output = bn(inputs[0].to(device))
      return (
          output.detach(),
          bn.running_mean,
          bn.running_var,
          bn.num_batches_tracked,
          bn.weight.grad,
          bn.bias.grad,
          eval_output,
      )

    self.assert_close_tpu_vs_cpu(run_training)
    self.assertEqual(cache_misses_after_warm_up, [0])

  def test_native_batch_norm_legit_no_stats_out(self):
    input_dtype = torch.float32
    stats_dtype = torch.float32
    n, c, h, w = 2, 4, 4, 4
    input_val = torch.randn(n, c, h, w, dtype=input_dtype)
    weight = torch.randn(c, dtype=stats_dtype)
    bias = torch.randn(c, dtype=stats_dtype)

    def run_op(device):
      out = torch.empty(n, c, h, w, device=device, dtype=input_dtype)
      save_mean = torch.empty(c, device=device, dtype=stats_dtype)
      save_invstd = torch.empty(c, device=device, dtype=stats_dtype)
      return torch.ops.aten._native_batch_norm_legit.no_stats_out(
          input_val.to(device),
          weight.to(device),
          bias.to(device),
          True,
          0.1,
          1e-5,
          out=out,
          save_mean=save_mean,
          save_invstd=save_invstd,
      )

    self.assert_close_tpu_vs_cpu(run_op)

  @parameterized.product(
      has_weight=[True, False],
      has_bias=[True, False],
      has_running_stats=[True, False],
      training=[True, False],
  )
  def test_native_batch_norm_optional_args(
      self, has_weight, has_bias, has_running_stats, training
  ):
    input_dtype = torch.float32
    stats_dtype = torch.float32
    n, c, h, w = 2, 4, 4, 4
    input_val = torch.randn(n, c, h, w, dtype=input_dtype)
    weight = torch.randn(c, dtype=stats_dtype) if has_weight else None
    bias = torch.randn(c, dtype=stats_dtype) if has_bias else None
    running_mean = (
        torch.randn(c, dtype=stats_dtype) if has_running_stats else None
    )
    eps = 1e-5
    running_var = (
        torch.rand(c, dtype=stats_dtype).abs() + eps
        if has_running_stats
        else None
    )

    def run_op(device):
      is_cpu_eval = not training and str(device) == "cpu"
      rm = (
          running_mean.to(device)
          if running_mean is not None
          else (
              torch.zeros(c, dtype=stats_dtype, device=device)
              if is_cpu_eval
              else None
          )
      )
      rv = (
          running_var.to(device)
          if running_var is not None
          else (
              torch.ones(c, dtype=stats_dtype, device=device)
              if is_cpu_eval
              else None
          )
      )
      out, save_mean, save_invstd = torch.ops.aten.native_batch_norm(
          input_val.to(device),
          weight.to(device) if weight is not None else None,
          bias.to(device) if bias is not None else None,
          rm,
          rv,
          training=training,
          momentum=0.1,  # momentum
          eps=eps,  # eps
      )
      if is_cpu_eval:
        # Match TPU/CUDA inference semantics where save_mean and save_invstd
        # return the effective running_mean and rsqrt(running_var + eps).
        save_mean = rm
        save_invstd = torch.rsqrt(rv + eps)
      return out, save_mean, save_invstd

    self.assert_close_tpu_vs_cpu(run_op)

  def test_bernoulli_distribution(self):
    """Tests bernoulli to produce the correct distribution."""
    n = 1000
    tpu_device = torch.device("tpu")

    torch.manual_seed(123)
    p = 0.7
    t = torch.empty(n, n, dtype=torch.float32, device=tpu_device)
    t = torch.bernoulli(t, p)
    mean_value = t.mean()
    expected_mean = p
    self.assert_close(
        golden_result=torch.tensor(expected_mean),
        torch_tpu_result=mean_value.to("cpu"),
        atol=1e-2,
        rtol=1e-2,
    )

  def test_binary_op_dtype_mismatch(self):
    """Test that binary ops properly promote all dtypes in binary ops."""
    # All dtypes except C128, no TPU support for C128
    dtypes = op_testing.all_xla_supported_dtypes()

    for a_dtype, b_dtype in list(itertools.combinations(dtypes, 2)):

      def test_fn(device, a_dtype=a_dtype, b_dtype=b_dtype):
        return torch.add(
            torch.tensor(5, dtype=a_dtype).to(device),
            torch.tensor([3, 1, 5], dtype=b_dtype).to(device),
        )

      self.assert_close_tpu_vs_cpu(test_fn)

  def test_binary_op_shape_broadcast(self):
    """Test that binary ops properly broadcast operands to the same shape."""
    # All shapes are compatible with `5x8x10`
    shapes = [
        torch.Size([]),
        torch.Size([1]),
        torch.Size([10]),
        torch.Size([8, 1]),
        torch.Size([1, 10]),
        torch.Size([8, 10]),
        torch.Size([1, 1, 1]),
        torch.Size([5, 1, 10]),
        torch.Size([5, 8, 10]),
    ]

    for a_shape, b_shape in list(itertools.combinations(shapes, 2)):

      def test_fn(device, a_shape=a_shape, b_shape=b_shape):
        return torch.add(
            torch.arange(a_shape.numel()).reshape(a_shape).to(device),
            torch.arange(b_shape.numel()).reshape(b_shape).to(device),
        )

      self.assert_close_tpu_vs_cpu(test_fn)

  def test_bincount(self):
    inputs = torch.tensor([0, 1, 3, 5, 1])
    weights = torch.tensor([0.1, 0.1, 0.3, 0.5, 0.1])
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bincount(
            inputs.to(device=device),
            minlength=8,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bincount(
            inputs.to(device=device),
            weights.to(device=device),
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bincount(
            inputs.to(dtype=torch.uint8).to(device=device),
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bincount(
            inputs.to(device=device),
            weights.to(dtype=torch.float64).to(device=device),
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bincount(
            inputs.to(device=device),
            weights.to(dtype=torch.bfloat16).to(device=device),
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bincount(
            inputs.to(device=device),
            weights.to(dtype=torch.float16).to(device=device),
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bincount(
            inputs.to(device=device),
            torch.ones_like(inputs, dtype=torch.uint8).to(device=device),
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bincount(
            inputs.to(device=device),
            minlength=8,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bincount(
            torch.tensor([], dtype=torch.int32).to(device=device),
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bincount(
            torch.tensor([], dtype=torch.int32).to(device=device),
            minlength=8,
        ),
    )

  @parameterized.named_parameters(
      ("small", 5, 2, 3, 3, [5, 4], [3, 2], torch.int32, False),
      (
          "large",
          50,
          4,
          20,
          30,
          [50, 45, 40, 35],
          [30, 25, 20, 15],
          torch.int32,
          False,
      ),
      ("zero_target_length", 10, 2, 5, 0, [10, 8], [0, 0], torch.int32, False),
      ("2d_lengths", 5, 2, 3, 3, [[5, 4]], [[3, 2]], torch.int32, False),
      ("int64_lengths", 5, 2, 3, 3, [5, 4], [3, 2], torch.int64, False),
      ("1d_targets", 5, 2, 3, 3, [5, 4], [3, 2], torch.int32, True),
  )
  def test_ctc_loss(
      self,
      t,
      n,
      c,
      s,
      in_lens,
      tgt_lens,
      lengths_dtype=torch.int32,
      is_1d_targets=False,
  ):
    """Tests _ctc_loss.Tensor with various configurations."""
    blank = 0

    # Generate random log probs
    probs = torch.randn(t, n, c)
    log_probs = torch.nn.functional.log_softmax(probs, dim=2)

    # Generate random targets in range [1, C-1] since blank is 0
    if is_1d_targets:
      targets = torch.randint(1, c, (sum(tgt_lens),), dtype=torch.int32)
    else:
      targets = torch.randint(1, c, (n, s), dtype=torch.int32)

    input_lengths = torch.tensor(in_lens, dtype=lengths_dtype)
    target_lengths = torch.tensor(tgt_lens, dtype=lengths_dtype)

    def compute(
        device,
        log_probs=log_probs,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
    ):
      loss, log_alpha = torch.ops.aten._ctc_loss.Tensor(
          log_probs.to(device),
          targets.to(device),
          input_lengths.to(device),
          target_lengths.to(device),
          blank,
          zero_infinity=False,
      )

      # Mask invalid states to avoid comparing CPU garbage values (close to 0)
      # against TPU's -inf
      n_batch, t_len, c_len = log_alpha.shape

      t_idx = (
          torch.arange(t_len, device=device).unsqueeze(0).expand(n_batch, t_len)
      )
      time_mask = t_idx < input_lengths.to(device).flatten().unsqueeze(1)

      c_idx = (
          torch.arange(c_len, device=device).unsqueeze(0).expand(n_batch, c_len)
      )
      state_mask = c_idx <= (2 * target_lengths.to(device).flatten()).unsqueeze(
          1
      )

      valid_mask = time_mask.unsqueeze(2) & state_mask.unsqueeze(1)
      log_alpha = torch.where(
          valid_mask, log_alpha, torch.tensor(float("-inf"), device=device)
      )

      return loss, log_alpha

    self.assert_close_tpu_vs_cpu(compute, rtol=2e-4, atol=3e-4)

  @parameterized.named_parameters(
      ("mean_reduction", "mean", False),
      ("sum_reduction", "sum", False),
      ("none_reduction", "none", False),
      ("zero_infinity", "mean", True),
  )
  def test_ctc_loss_functional_1d_targets(self, reduction, zero_infinity):
    logits = torch.randn(2, 8, 5)
    log_probs = torch.nn.functional.log_softmax(
        logits, dim=-1, dtype=torch.float32
    ).transpose(0, 1)
    targets = torch.randint(1, 5, (6,), dtype=torch.long)
    input_lengths = torch.full((2,), 8, dtype=torch.long)
    target_lengths = torch.tensor([3, 3], dtype=torch.long)

    def compute(
        device,
        log_probs=log_probs,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
    ):
      lp = log_probs.to(device).detach().requires_grad_(True)
      loss = torch.nn.functional.ctc_loss(
          lp,
          targets.to(device),
          input_lengths.to(device),
          target_lengths.to(device),
          blank=0,
          reduction=reduction,
          zero_infinity=zero_infinity,
      )
      if reduction == "none":
        loss.sum().backward()
      else:
        loss.backward()
      return loss, lp.grad

    self.assert_close_tpu_vs_cpu(compute, rtol=1e-3, atol=1e-4)

  def test_ctc_loss_functional_1d_targets_int32_and_cpu_lengths(self):
    logits = torch.randn(2, 8, 5)
    log_probs = torch.nn.functional.log_softmax(
        logits, dim=-1, dtype=torch.float32
    ).transpose(0, 1)
    targets = torch.randint(1, 5, (6,), dtype=torch.long)
    input_lengths_i32 = torch.tensor([8, 8], dtype=torch.int32)
    target_lengths_i32 = torch.tensor([3, 3], dtype=torch.int32)

    def compute_i32(device):
      lp = log_probs.to(device).detach().requires_grad_(True)
      loss = torch.nn.functional.ctc_loss(
          lp,
          targets.to(device),
          input_lengths_i32.to(device),
          target_lengths_i32.to(device),
      )
      loss.backward()
      return loss, lp.grad

    self.assert_close_tpu_vs_cpu(compute_i32, rtol=1e-3, atol=1e-4)

    def compute_list_lengths(device):
      lp = log_probs.to(device).detach().requires_grad_(True)
      loss_list = torch.nn.functional.ctc_loss(
          lp,
          targets.to(device),
          [8, 8],
          [3, 3],
      )
      loss_list.backward()
      return loss_list, lp.grad

    self.assert_close_tpu_vs_cpu(compute_list_lengths, rtol=1e-3, atol=1e-4)

    def compute_list_lengths_uneven(device):
      lp = log_probs.to(device).detach().requires_grad_(True)
      loss_list = torch.nn.functional.ctc_loss(
          lp,
          torch.tensor([1, 2, 3, 4, 2], dtype=torch.long).to(device),
          [8, 8],
          [4, 1],
      )
      loss_list.backward()
      return loss_list, lp.grad

    self.assert_close_tpu_vs_cpu(
        compute_list_lengths_uneven, rtol=1e-3, atol=1e-4
    )

  def test_ctc_loss_functional_1d_targets_uneven_lengths(self):
    logits = torch.randn(2, 8, 5)
    log_probs = torch.nn.functional.log_softmax(
        logits, dim=-1, dtype=torch.float32
    ).transpose(0, 1)
    targets = torch.tensor([1, 2, 3, 4, 2], dtype=torch.long)
    input_lengths = torch.tensor([8, 8], dtype=torch.long)
    target_lengths = torch.tensor([4, 1], dtype=torch.long)

    def compute(device):
      lp = log_probs.to(device).detach().requires_grad_(True)
      loss = torch.nn.functional.ctc_loss(
          lp,
          targets.to(device),
          input_lengths.to(device),
          target_lengths.to(device),
          blank=0,
      )
      loss.backward()
      return loss, lp.grad

    self.assert_close_tpu_vs_cpu(compute, rtol=1e-3, atol=1e-4)

  def test_ctc_loss_functional_1d_targets_empty(self):
    logits = torch.randn(2, 8, 5)
    log_probs = torch.nn.functional.log_softmax(
        logits, dim=-1, dtype=torch.float32
    ).transpose(0, 1)
    targets = torch.empty((0,), dtype=torch.long)
    input_lengths = torch.tensor([8, 8], dtype=torch.long)
    target_lengths = torch.tensor([0, 0], dtype=torch.long)

    def compute(device):
      lp = log_probs.to(device).detach().requires_grad_(True)
      loss = torch.nn.functional.ctc_loss(
          lp,
          targets.to(device),
          input_lengths.to(device),
          target_lengths.to(device),
          blank=0,
      )
      loss.backward()
      return loss, lp.grad

    self.assert_close_tpu_vs_cpu(compute, rtol=1e-3, atol=1e-4)

  def test_ctc_loss_2d_targets_with_trailing_padding(self):
    logits = torch.randn(2, 8, 5)
    log_probs = torch.nn.functional.log_softmax(
        logits, dim=-1, dtype=torch.float32
    ).transpose(0, 1)
    targets = torch.tensor([[1, 2, 0, 0, 0], [2, 3, 1, 0, 0]], dtype=torch.long)
    input_lengths = torch.tensor([8, 8], dtype=torch.long)
    target_lengths = torch.tensor([2, 3], dtype=torch.long)

    def compute(device):
      lp = log_probs.to(device).detach().requires_grad_(True)
      loss = torch.nn.functional.ctc_loss(
          lp,
          targets.to(device),
          input_lengths.to(device),
          target_lengths.to(device),
          blank=0,
      )
      loss.backward()
      return loss, lp.grad

    self.assert_close_tpu_vs_cpu(compute, rtol=1e-3, atol=1e-4)

  @parameterized.named_parameters(
      ("small", 5, 2, 3, 3, [5, 4], [3, 2], torch.int32, False),
      (
          "large",
          50,
          4,
          20,
          30,
          [50, 45, 40, 35],
          [30, 25, 20, 15],
          torch.int32,
          False,
      ),
      ("zero_target_length", 10, 2, 5, 0, [10, 8], [0, 0], torch.int32, False),
      ("2d_lengths", 5, 2, 3, 3, [[5, 4]], [[3, 2]], torch.int32, False),
      ("int64_lengths", 5, 2, 3, 3, [5, 4], [3, 2], torch.int64, False),
      ("1d_targets", 5, 2, 3, 3, [5, 4], [3, 2], torch.int32, True),
  )
  @absltest.skip("b/515048424")
  def test_ctc_loss_backward(
      self,
      t,
      n,
      c,
      s,
      in_lens,
      tgt_lens,
      lengths_dtype=torch.int32,
      is_1d_targets=False,
  ):
    """Tests _ctc_loss_backward.Tensor with various configurations."""
    blank = 0

    # Generate random log probs
    probs = torch.randn(t, n, c)
    log_probs = torch.nn.functional.log_softmax(probs, dim=2)

    # Generate random targets in range [1, C-1] since blank is 0
    if is_1d_targets:
      targets = torch.randint(1, c, (sum(tgt_lens),), dtype=torch.int32)
    else:
      targets = torch.randint(1, c, (n, s), dtype=torch.int32)

    input_lengths = torch.tensor(in_lens, dtype=lengths_dtype)
    target_lengths = torch.tensor(tgt_lens, dtype=lengths_dtype)

    # Calculate forward pass on CPU to get the inputs for backward
    loss, log_alpha = torch.ops.aten._ctc_loss.Tensor(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        zero_infinity=False,
    )

    grad_out = torch.randn_like(loss)

    def compute(
        device,
        grad_out=grad_out,
        log_probs=log_probs,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        loss=loss,
        log_alpha=log_alpha,
    ):
      return torch.ops.aten._ctc_loss_backward.Tensor(
          grad_out.to(device),
          log_probs.to(device),
          targets.to(device),
          input_lengths.to(device),
          target_lengths.to(device),
          loss.to(device),
          log_alpha.to(device),
          blank,
          zero_infinity=False,
      )

    self.assert_close_tpu_vs_cpu(compute, rtol=3.7e-3, atol=4.2e-5)

  @parameterized.named_parameters(
      dict(
          testcase_name="clone",
          op_fn=lambda t: t.clone(),
      ),
      dict(
          testcase_name="to_copy",
          op_fn=lambda t: torch.ops.aten._to_copy(t, dtype=t.dtype),
      ),
      dict(
          testcase_name="view",
          op_fn=lambda t: t.view(4, 2),
      ),
      dict(
          testcase_name="reshape",
          op_fn=lambda t: t.reshape(4, 2),
      ),
      dict(
          testcase_name="slice",
          op_fn=lambda t: t[:1, :],
      ),
      dict(
          testcase_name="cat",
          op_fn=lambda t: torch.cat([t, t], dim=0),
      ),
      dict(
          testcase_name="stack",
          op_fn=lambda t: torch.stack([t, t], dim=0),
      ),
      dict(
          testcase_name="zero_",
          op_fn=lambda t: t.clone().zero_(),
      ),
      dict(
          testcase_name="eq",
          op_fn=lambda t: torch.eq(t, t),
      ),
      dict(
          testcase_name="ne",
          op_fn=lambda t: torch.ne(t, t),
      ),
      dict(
          testcase_name="isnan",
          op_fn=lambda t: torch.isnan(t),
      ),
  )
  def test_low_precision_float_kept_ops(self, op_fn):
    """Tests non-regression for allowlisted data movement and predicate ops on FP8."""
    device = torch.device("tpu")
    t_cpu = torch.randn(2, 4, dtype=torch.float32).to(torch.float8_e4m3fn)
    t_tpu = t_cpu.to(device)

    res_tpu = op_fn(t_tpu)
    try:
      res_cpu = op_fn(t_cpu)
    except NotImplementedError:
      res_cpu = op_fn(t_cpu.float())

    self.assertEqual(res_tpu.dtype, res_cpu.dtype)
    self.assertEqual(res_tpu.shape, res_cpu.shape)
    utils.assert_close(
        actual=res_tpu.cpu().view(torch.uint8),
        expected=res_cpu.view(torch.uint8),
    )

  def test_fp8_e5m2_abs_allowed(self):
    """Tests that abs() is allowed on float8_e5m2 per CUDA parity."""
    device = torch.device("tpu")
    raw = torch.tensor([[0x81, 0x02], [0x83, 0x04]], dtype=torch.uint8)
    t_cpu = raw.view(torch.float8_e5m2)
    t_tpu = t_cpu.to(device)

    res_tpu = torch.abs(t_tpu)
    res_cpu = torch.abs(t_cpu)

    self.assertEqual(res_tpu.dtype, torch.float8_e5m2)
    utils.assert_close(
        actual=res_tpu.cpu().view(torch.uint8),
        expected=res_cpu.view(torch.uint8),
    )

  @parameterized.product(
      m=[16, 32],
      n=[16, 32],
      k=[16, 32],
      dtype=[torch.float8_e4m3fn, torch.float8_e5m2],
  )
  def test_scaled_mm_numeric(self, m, n, k, dtype):
    """Tests torch._scaled_mm with numerical verification against F32 CPU."""
    self_float = torch.randn(m, k, dtype=torch.float32)
    self_fp8 = self_float.to(dtype)
    mat2_float = torch.randn(k, n, dtype=torch.float32)
    mat2_fp8 = mat2_float.to(dtype)

    scale_a = torch.tensor([1.5], dtype=torch.float32)
    scale_b = torch.tensor([2.0], dtype=torch.float32)

    def compute(device):
      if device == "cpu":
        # Simulate in F32 on CPU
        s_f32 = self_fp8.float()
        m_f32 = mat2_fp8.float()
        out_f32 = torch.mm(s_f32, m_f32)
        return (out_f32 * scale_a * scale_b).to(dtype)
      else:
        return torch._scaled_mm(
            self_fp8.to(device),
            mat2_fp8.to(device),
            scale_a.to(device),
            scale_b.to(device),
        )

    self.assert_close_tpu_vs_cpu(
        compute,
    )

  @parameterized.product(
      m=[32, 128],
      n=[32, 128],
      k=[128, 256],
      fp8_dtype=[torch.float8_e4m3fn, torch.float8_e5m2],
      swizzle_a=[0, 1],
      swizzle_b=[0, 1],
  )
  def test_scaled_mm_v2_blockwise(
      self, m, n, k, fp8_dtype, swizzle_a, swizzle_b
  ):
    """Tests torch._scaled_mm_v2 with BlockWise1x32 scaling and swizzling."""
    torch.set_printoptions(threshold=10000, profile="full")
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
      if swizzle_a == 1 and m < 128:
        self.skipTest("LHS Swizzling requires M >= 128")
      if swizzle_b == 1 and n < 128:
        self.skipTest("RHS Swizzling requires N >= 128")

      self_float = torch.randn(m, k, dtype=torch.float32)
      self_fp8 = self_float.to(fp8_dtype)
      mat2_float = torch.randn(k, n, dtype=torch.float32)
      mat2_fp8 = mat2_float.to(fp8_dtype)

      scale_a_logical = torch.randn(m, k // 32, dtype=torch.float32).abs() + 0.1
      scale_b_logical = torch.randn(k // 32, n, dtype=torch.float32).abs() + 0.1

      if swizzle_a == 1:
        scale_a_tpu = op_testing._swizzle_lhs_scale(scale_a_logical)
      else:
        scale_a_tpu = scale_a_logical

      if swizzle_b == 1:
        scale_b_tpu = op_testing._swizzle_rhs_scale(scale_b_logical)
      else:
        scale_b_tpu = scale_b_logical

      recipe_a = [3]
      recipe_b = [3]
      swiz_a = [swizzle_a]
      swiz_b = [swizzle_b]

      def compute(device):
        if device == "cpu":
          s_f32 = self_fp8.float()
          m_f32 = mat2_fp8.float()
          sa_expanded = (
              scale_a_logical.unsqueeze(-1).expand(m, k // 32, 32).reshape(m, k)
          )
          s_dequant = s_f32 * sa_expanded
          sb_expanded = (
              scale_b_logical.unsqueeze(1).expand(k // 32, 32, n).reshape(k, n)
          )
          m_dequant = m_f32 * sb_expanded
          out_f32 = torch.mm(s_dequant, m_dequant)
          return out_f32
        else:
          return torch._scaled_mm_v2(
              self_fp8.to(device),
              mat2_fp8.to(device),
              [scale_a_tpu.to(device)],
              recipe_a,
              swiz_a,
              [scale_b_tpu.to(device)],
              recipe_b,
              swiz_b,
              None,
              torch.float32,
          )

      self.assert_close_tpu_vs_cpu(
          compute,
          rtol=1e-2,
          atol=1.001e-2,
      )
    finally:
      torch.set_float32_matmul_precision(old_precision)

  def test_scaled_mm_v2_valid_inputs_pass(self):
    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = op_testing._get_scaled_mm_v2_default_inputs("tpu")
    out = torch._scaled_mm_v2(
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
        None,
        None,
        contraction_dim=[1, 0],
    )
    self.assertEqual(out.shape, (16, 16))

  def test_scaled_mm_v2_use_fast_accum(self):
    """Tests torch._scaled_mm_v2 with use_fast_accum=True."""
    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = op_testing._get_scaled_mm_v2_default_inputs("tpu")
    out = torch._scaled_mm_v2(
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
        None,
        None,
        use_fast_accum=True,
    )
    self.assertEqual(out.shape, (16, 16))

  @parameterized.product(
      scale_kind=["tensorwise", "rowwise"], use_out=[False, True]
  )
  def test_scaled_mm_use_fast_accum_matches_default(self, scale_kind, use_out):
    """Tests that use_fast_accum=True reuses the default's kernel and result."""
    m, k, n = 32, 64, 48
    device = "tpu"
    self_fp8 = torch.randn(m, k).to(torch.float8_e4m3fn).to(device)
    mat2_fp8 = torch.randn(k, n).to(torch.float8_e4m3fn).to(device)
    if scale_kind == "tensorwise":
      scale_a = torch.tensor(1.5, dtype=torch.float32)
      scale_b = torch.tensor(0.5, dtype=torch.float32)
    else:
      scale_a = torch.rand(m, 1, dtype=torch.float32) + 0.5
      scale_b = torch.rand(1, n, dtype=torch.float32) + 0.5
    scale_a = scale_a.to(device)
    scale_b = scale_b.to(device)

    def run(use_fast_accum):
      args = (self_fp8, mat2_fp8, scale_a, scale_b)
      kwargs = dict(out_dtype=torch.bfloat16, use_fast_accum=use_fast_accum)
      if use_out:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=device)
        torch._scaled_mm(*args, **kwargs, out=out)
        return out.cpu()
      return torch._scaled_mm(*args, **kwargs).cpu()

    expected = run(use_fast_accum=False)
    start_misses = torch.tpu._get_cache_misses()
    actual = run(use_fast_accum=True)
    # use_fast_accum doesn't change the lowering, so the kernel is reused.
    self.assertEqual(torch.tpu._get_cache_misses() - start_misses, 0)
    utils.assert_close(actual, expected, rtol=0, atol=0)

  def test_swizzle_math(self):
    # More than one tile in each direction, so that the ordering of the tile
    # grid is exercised and not just the layout within a single tile.
    logical_lhs = torch.randn(256, 8)
    swizzled_lhs = op_testing._swizzle_lhs_scale(logical_lhs)
    unswizzled_lhs = op_testing._unswizzle_lhs_scale(swizzled_lhs, 256, 8)
    self.assertTrue(torch.equal(logical_lhs, unswizzled_lhs))

    logical_rhs = torch.randn(8, 256)
    swizzled_rhs = op_testing._swizzle_rhs_scale(logical_rhs)
    unswizzled_rhs = op_testing._unswizzle_rhs_scale(swizzled_rhs, 8, 256)
    self.assertTrue(torch.equal(logical_rhs, unswizzled_rhs))

  def test_swizzle_matches_cublas_layout(self):
    # 2 row-blocks of 128 and 2 col-blocks of 4 -> 4 tiles of [128,4].
    rows, cols = 256, 8
    logical = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    swizzled = op_testing._swizzle_lhs_scale(logical)
    self.assertEqual(swizzled.shape, (rows * cols,))

    row = torch.arange(rows).unsqueeze(1).expand(rows, cols)
    col = torch.arange(cols).unsqueeze(0).expand(rows, cols)
    outer = row % 128  # row within the 128x4 tile
    # Compute the indices that would recover "swizzled"
    offsets = (
        512 * ((row // 128) * (cols // 4) + col // 4)  # 128x4 tile base index
        + (outer % 32) * 16  # row offset mod 32 (16-elem packet)
        + (outer // 32) * 4  # 32-row chunk index (4 cols each)
        + col % 4  # column offset within chunk
    )
    # Fancy-indexing with offsets should reconstruct the logical 2D tensor.
    self.assertTrue(torch.equal(swizzled[offsets], logical))

  @parameterized.product(
      m=[16, 32],
      n=[16, 32],
      k=[16, 32],
      dtype=[torch.float8_e4m3fn, torch.float8_e5m2],
  )
  def test_scaled_mm_with_bias_and_scale_result(self, m, n, k, dtype):
    """Tests torch._scaled_mm with bias and scale_result."""
    self_float = torch.randn(m, k, dtype=torch.float32)
    self_fp8 = self_float.to(dtype)
    mat2_float = torch.randn(k, n, dtype=torch.float32)
    mat2_fp8 = mat2_float.to(dtype)

    scale_a = torch.tensor([1.5], dtype=torch.float32)
    scale_b = torch.tensor([2.0], dtype=torch.float32)
    bias = torch.randn(m, n, dtype=torch.float32)
    scale_result = torch.tensor([0.5], dtype=torch.float32)

    def compute(device):
      if device == "cpu":
        # Simulate in F32 on CPU
        s_f32 = self_fp8.float()
        m_f32 = mat2_fp8.float()
        out_f32 = torch.mm(s_f32, m_f32)
        res = (out_f32 * scale_a * scale_b) + bias
        return (res * scale_result).to(dtype)
      else:
        return torch._scaled_mm(
            self_fp8.to(device),
            mat2_fp8.to(device),
            scale_a.to(device),
            scale_b.to(device),
            bias=bias.to(device),
            scale_result=scale_result.to(device),
        )

    self.assert_close_tpu_vs_cpu(
        compute,
    )

  @parameterized.product(
      m=[16, 32],
      n=[16, 32],
      k=[16, 32],
      scale_ndim=[1, 2],
      dtype=[torch.float8_e4m3fn, torch.float8_e5m2],
  )
  def test_scaled_mm_rowwise_channelwise(self, m, n, k, scale_ndim, dtype):
    """Tests torch._scaled_mm with rowwise scale_a and per-channel scale_b.

    Covers https://github.com/google-pytorch/torch_tpu/issues/1820: scale_a is
    per-token ([m, 1] or [m])
    and scale_b is per-output-channel ([1, n] or [n]).

    Output dtype is F32 and compared against an F32 dequant reference with an
    FP8-appropriate tolerance: the TPU FP8 dot does not bit-match CPU F32 mm.

    Args:
      m: Number of rows in self tensor.
      n: Number of columns in mat2 tensor.
      k: Inner dimension of self and mat2 tensors.
      scale_ndim: Number of dimensions for the scale tensors (1 or 2).
      dtype: Input tensor data type.
    """
    self_fp8 = torch.randn(m, k, dtype=torch.float32).to(dtype)
    mat2_fp8 = torch.randn(k, n, dtype=torch.float32).to(dtype)

    if scale_ndim == 2:
      scale_a = torch.rand(m, 1, dtype=torch.float32) + 0.5
      scale_b = torch.rand(1, n, dtype=torch.float32) + 0.5
    else:
      scale_a = torch.rand(m, dtype=torch.float32) + 0.5
      scale_b = torch.rand(n, dtype=torch.float32) + 0.5

    def compute(device):
      if device == "cpu":
        a_deq = self_fp8.float() * scale_a.reshape(m, 1)
        b_deq = mat2_fp8.float() * scale_b.reshape(1, n)
        return torch.mm(a_deq, b_deq)
      return torch._scaled_mm(
          self_fp8.to(device),
          mat2_fp8.to(device),
          scale_a.to(device),
          scale_b.to(device),
          out_dtype=torch.float32,
      )

    # Loose FP8 tolerance: the TPU FP8 dot does not bit-match the CPU F32
    # reference, and per-element relative error is large near zero.
    self.assert_close_tpu_vs_cpu(compute, rtol=5e-1, atol=5e-1)

  @parameterized.product(
      shape=[
          (1, 32, 32),  # decode: M=1
          (5, 32, 32),  # ragged M
          (17, 32, 32),  # ragged M
          (32, 17, 48),  # ragged K (contraction)
          (32, 24, 48),  # ragged K
          (32, 64, 17),  # ragged N (output)
          (32, 64, 40),  # ragged N
          (1, 17, 17),  # all dims ragged
      ],
      dtype=[torch.float8_e4m3fn, torch.float8_e5m2],
  )
  def test_scaled_mm_ragged_dims(self, shape, dtype):
    """Tests torch._scaled_mm with M, K, or N not a multiple of 16.

    Covers https://github.com/google-pytorch/torch_tpu/issues/1823: no dimension
    needs to be aligned; XLA
    pads M, K, and N to the MXU tile internally (e.g. decode M=1, unaligned
    K/N).

    Args:
      shape: Tuple of (m, k, n) dimensions for self and mat2 tensors.
      dtype: Input tensor data type.
    """
    m, k, n = shape
    self_fp8 = torch.randn(m, k, dtype=torch.float32).to(dtype)
    mat2_fp8 = torch.randn(k, n, dtype=torch.float32).to(dtype)
    scale_a = torch.rand(m, 1, dtype=torch.float32) + 0.5
    scale_b = torch.rand(1, n, dtype=torch.float32) + 0.5

    def compute(device):
      if device == "cpu":
        a_deq = self_fp8.float() * scale_a
        b_deq = mat2_fp8.float() * scale_b
        return torch.mm(a_deq, b_deq)
      return torch._scaled_mm(
          self_fp8.to(device),
          mat2_fp8.to(device),
          scale_a.to(device),
          scale_b.to(device),
          out_dtype=torch.float32,
      )

    # Loose FP8 tolerance: the TPU FP8 dot does not bit-match the CPU F32
    # reference, and per-element relative error is large near zero.
    self.assert_close_tpu_vs_cpu(compute, rtol=5e-1, atol=5e-1)

  @parameterized.product(
      scale_kind=["row_x_scalar", "scalar_x_chan", "row_x_chan"],
      use_bias=[False, True],
      dtype=[torch.float8_e4m3fn, torch.float8_e5m2],
  )
  def test_scaled_mm_mixed_scales_and_bias(self, scale_kind, use_bias, dtype):
    """Tests torch._scaled_mm with mixed scale shapes and optional bias.

    Covers the rowwise x scalar / scalar x per-channel paths (a scalar scale
    broadcasts unchanged while the other operand is reshaped) and rowwise +
    bias, which the rowwise/ragged tests do not exercise.

    Args:
      scale_kind: String specifying the scaling configuration.
      use_bias: Boolean indicating whether to include bias.
      dtype: Input tensor data type.
    """
    m, k, n = 32, 64, 48
    self_fp8 = torch.randn(m, k, dtype=torch.float32).to(dtype)
    mat2_fp8 = torch.randn(k, n, dtype=torch.float32).to(dtype)
    scale_a = (
        torch.rand(m, 1, dtype=torch.float32) + 0.5
        if scale_kind != "scalar_x_chan"
        else torch.tensor([1.5], dtype=torch.float32)
    )
    scale_b = (
        torch.rand(1, n, dtype=torch.float32) + 0.5
        if scale_kind != "row_x_scalar"
        else torch.tensor([2.0], dtype=torch.float32)
    )
    bias = torch.randn(n, dtype=torch.float32) if use_bias else None

    def compute(device):
      if device == "cpu":
        sa = scale_a.reshape(m, 1) if scale_a.numel() == m else scale_a
        sb = scale_b.reshape(1, n) if scale_b.numel() == n else scale_b
        out = (self_fp8.float() * sa) @ (mat2_fp8.float() * sb)
        return out + bias if use_bias else out
      return torch._scaled_mm(
          self_fp8.to(device),
          mat2_fp8.to(device),
          scale_a.to(device),
          scale_b.to(device),
          bias=bias.to(device) if use_bias else None,
          out_dtype=torch.float32,
      )

    # Loose FP8 tolerance: the TPU FP8 dot does not bit-match the CPU F32
    # reference, and per-element relative error is large near zero.
    self.assert_close_tpu_vs_cpu(compute, rtol=5e-1, atol=5e-1)

  @parameterized.product(
      scale_kind=["tensorwise", "rowwise"],
      dtypes=[
          # dgrad: grad_output (E5M2) @ weight.T (E4M3)
          (torch.float8_e5m2, torch.float8_e4m3fn),
          # wgrad: input.T (E4M3) @ grad_output (E5M2)
          (torch.float8_e4m3fn, torch.float8_e5m2),
          (torch.float8_e5m2, torch.float8_e5m2),  # pure e5m2 matmul
      ],
      shape=[(16, 32, 16), (32, 64, 48)],
      out_dtype=[torch.bfloat16, torch.float32],
  )
  def test_scaled_mm_mixed_fp8_dtypes(
      self, scale_kind, dtypes, shape, out_dtype
  ):
    """Tests torch._scaled_mm with mixed float8 dtypes and e5m2 matmuls.

    Covers forward and backward gradient combinations:
    - a=e5m2, b=e4m3fn (dgrad: grad_output @ weight.T)
    - a=e4m3fn, b=e5m2 (wgrad: input.T @ grad_output)
    - a=e5m2, b=e5m2 (not supported in CUDA cuBLASLt, supported natively on TPU)
    Tested with both tensorwise (1D/scalar) and rowwise ((M,1) x (1,N)) scaling,
    contiguous and non-contiguous column-major mat2 layouts, the .out overload
    for tensorwise scaling, and out_dtype in (bfloat16, float32) against CPU
    FP32 reference.

    Args:
      scale_kind: Scaling configuration ("tensorwise" or "rowwise").
      dtypes: Tuple of (dtype_a, dtype_b) for the two input matrices.
      shape: Tuple of (m, k, n) matrix dimensions.
      out_dtype: Output tensor data type.
    """
    m, k, n = shape
    dtype_a, dtype_b = dtypes
    self_fp8 = torch.randn(m, k, dtype=torch.float32).to(dtype_a)
    mat2_fp8 = torch.randn(k, n, dtype=torch.float32).to(dtype_b)
    mat2_t_fp8 = torch.randn(n, k, dtype=torch.float32).to(dtype_b)

    if scale_kind == "rowwise":
      scale_a = torch.rand(m, 1, dtype=torch.float32) + 0.5
      scale_b = torch.rand(1, n, dtype=torch.float32) + 0.5
    else:
      scale_a = torch.tensor([1.2], dtype=torch.float32)
      scale_b = torch.tensor([0.8], dtype=torch.float32)

    def compute(device):
      if device == "cpu":
        res = (self_fp8.float() * scale_a) @ (mat2_fp8.float() * scale_b)
        return res.to(out_dtype)
      return torch._scaled_mm(
          self_fp8.to(device),
          mat2_fp8.to(device),
          scale_a.to(device),
          scale_b.to(device),
          out_dtype=out_dtype,
      )

    self.assert_close_tpu_vs_cpu(compute, rtol=1.6e-2, atol=5e-2)

    def compute_col_major(device):
      mat2_nt = mat2_t_fp8.to(device).t()
      if device == "cpu":
        res = (self_fp8.float() * scale_a) @ (mat2_nt.float() * scale_b)
        return res.to(out_dtype)
      return torch._scaled_mm(
          self_fp8.to(device),
          mat2_nt,
          scale_a.to(device),
          scale_b.to(device),
          out_dtype=out_dtype,
      )

    self.assert_close_tpu_vs_cpu(compute_col_major, rtol=1.6e-2, atol=5e-2)

    if scale_kind == "tensorwise":

      def compute_out(device):
        if device == "cpu":
          res = (self_fp8.float() * scale_a) @ (mat2_fp8.float() * scale_b)
          return res.to(out_dtype)
        out = torch.empty((m, n), dtype=out_dtype, device=device)
        return torch.ops.aten._scaled_mm.out(  # pylint: disable=protected-access
            self_fp8.to(device),
            mat2_fp8.to(device),
            scale_a.to(device),
            scale_b.to(device),
            out_dtype=out_dtype,
            out=out,
        )

      self.assert_close_tpu_vs_cpu(compute_out, rtol=1.6e-2, atol=5e-2)

  @parameterized.product(
      compiled=[False, True],
      out_dtype=[torch.bfloat16, torch.float32],
      dtype=[torch.float8_e4m3fn, torch.float8_e5m2],
  )
  def test_scaled_mm_rank0_tensorwise_scales(self, compiled, out_dtype, dtype):
    """Tests torch._scaled_mm with rank-0 tensorwise scales.

    torchao's tensorwise float8 training passes 0-d scales, while the other
    tensorwise tests here use shape-[1] scales.

    Args:
      compiled: Whether to run under `torch.compile(fullgraph=True)`.
      out_dtype: Output dtype of the scaled matmul.
      dtype: Input tensor data type.
    """
    torch._dynamo.reset()
    m, k, n = 32, 64, 48
    self_fp8 = torch.randn(m, k, dtype=torch.float32).to(dtype)
    mat2_fp8 = torch.randn(k, n, dtype=torch.float32).to(dtype)
    scale_a = torch.tensor(1.5, dtype=torch.float32)
    scale_b = torch.tensor(0.25, dtype=torch.float32)

    def scaled_mm(a, b, sa, sb):
      return torch._scaled_mm(a, b, sa, sb, out_dtype=out_dtype)

    def compute(device):
      if device == "cpu":
        out = (self_fp8.float() * scale_a) @ (mat2_fp8.float() * scale_b)
        return out.to(out_dtype)
      fn = (
          torch.compile(scaled_mm, backend="tpu", fullgraph=True)
          if compiled
          else scaled_mm
      )
      return fn(
          self_fp8.to(device),
          mat2_fp8.to(device),
          scale_a.to(device),
          scale_b.to(device),
      )

    self.assert_close_tpu_vs_cpu(compute, rtol=1.6e-2, atol=5e-2)

  @parameterized.product(
      out_dtype=[torch.bfloat16, torch.float32],
      dtype=[torch.float8_e4m3fn, torch.float8_e5m2],
  )
  def test_scaled_mm_rowwise_compiled_without_graph_breaks(
      self, out_dtype, dtype
  ):
    """Tests rowwise torch._scaled_mm under torch.compile in a single graph.

    torchao's rowwise float8 training calls torch._scaled_mm with (M, 1) and
    (1, N) scales from inside compiled models.

    Args:
      out_dtype: Output dtype of the scaled matmul.
      dtype: Input tensor data type.
    """
    torch._dynamo.reset()
    m, k, n = 32, 64, 48
    self_fp8 = torch.randn(m, k, dtype=torch.float32).to(dtype)
    mat2_fp8 = torch.randn(k, n, dtype=torch.float32).to(dtype)
    scale_a = torch.rand(m, 1, dtype=torch.float32) + 0.5
    scale_b = torch.rand(1, n, dtype=torch.float32) + 0.5

    def scaled_mm(a, b, sa, sb):
      return torch._scaled_mm(a, b, sa, sb, out_dtype=out_dtype)

    def compute(device):
      if device == "cpu":
        out = (self_fp8.float() * scale_a) @ (mat2_fp8.float() * scale_b)
        return out.to(out_dtype)
      compiled = torch.compile(scaled_mm, backend="tpu", fullgraph=True)
      return compiled(
          self_fp8.to(device),
          mat2_fp8.to(device),
          scale_a.to(device),
          scale_b.to(device),
      )

    self.assert_close_tpu_vs_cpu(compute, rtol=1.6e-2, atol=5e-2)

    explanation = torch._dynamo.explain(scaled_mm)(
        self_fp8.to("tpu"),
        mat2_fp8.to("tpu"),
        scale_a.to("tpu"),
        scale_b.to("tpu"),
    )
    self.assertEqual(explanation.graph_count, 1)
    self.assertEqual(explanation.graph_break_count, 0)

  @parameterized.product(
      compiled=[False, True],
      orig_dtype=[torch.bfloat16, torch.float32],
      dtype=[torch.float8_e4m3fn, torch.float8_e5m2],
  )
  def test_scaled_mm_unit_scales_then_rowwise_post_scale(
      self, compiled, orig_dtype, dtype
  ):
    """Tests the sequence torchao runs for rowwise scales without fast accum.

    In that case torchao's `addmm_float8_unwrapped` calls torch._scaled_mm with
    rank-0 ones scales and a bf16 output, multiplies the output in place by the
    (M, 1) x (1, N) inverse scales, and casts to the original dtype.

    Args:
      compiled: Whether to run under `torch.compile(fullgraph=True)`.
      orig_dtype: Dtype the result is cast to after the post-scale.
      dtype: Input tensor data type.
    """
    torch._dynamo.reset()
    m, k, n = 32, 64, 48
    self_fp8 = torch.randn(m, k, dtype=torch.float32).to(dtype)
    mat2_fp8 = torch.randn(k, n, dtype=torch.float32).to(dtype)
    inv_scale_a = torch.rand(m, 1, dtype=torch.float32) + 0.5
    inv_scale_b = torch.rand(1, n, dtype=torch.float32) + 0.5

    def rowwise_as_tensorwise(a, b, inv_sa, inv_sb):
      post_inverse_scale = inv_sa * inv_sb
      ones = inv_sa.new_ones(())
      out = torch._scaled_mm(
          a,
          b,
          scale_a=ones,
          scale_b=ones,
          out_dtype=torch.bfloat16,
          use_fast_accum=False,
      )
      out *= post_inverse_scale
      return out.to(orig_dtype)

    def compute(device):
      if device == "cpu":
        out = (self_fp8.float() @ mat2_fp8.float()).to(torch.bfloat16)
        out *= inv_scale_a * inv_scale_b
        return out.to(orig_dtype)
      fn = (
          torch.compile(rowwise_as_tensorwise, backend="tpu", fullgraph=True)
          if compiled
          else rowwise_as_tensorwise
      )
      return fn(
          self_fp8.to(device),
          mat2_fp8.to(device),
          inv_scale_a.to(device),
          inv_scale_b.to(device),
      )

    self.assert_close_tpu_vs_cpu(compute, rtol=1.6e-2, atol=5e-2)

  @parameterized.product(
      m=[16, 32],
      n=[16, 32],
      k=[16, 32],
      out_dtype=[torch.bfloat16, torch.float32, torch.float4_e2m1fn_x2],
  )
  def test_scaled_mm_fp4_numeric(self, m, n, k, out_dtype):
    """Tests torch._scaled_mm with FP4 inputs."""

    if "TPU_NAME" not in os.environ:
      raise unittest.SkipTest(
          "FP4 scaled_mm test requires TPU hardware to avoid PyTorch OpMathType"
          " CPU fallback crash."
      )

    device = "tpu"
    self_float_tpu = torch.randn(m, k, dtype=torch.float32).to(device)
    mat2_float_tpu = torch.randn(k, n, dtype=torch.float32).to(device)

    self_fp4_tpu = self_float_tpu.to(torch.float4_e2m1fn_x2)
    mat2_fp4_tpu = mat2_float_tpu.to(torch.float4_e2m1fn_x2)

    scale_a = torch.tensor([1.5], dtype=torch.float32).to(device)
    scale_b = torch.tensor([2.0], dtype=torch.float32).to(device)

    out = torch._scaled_mm(
        self_fp4_tpu,
        mat2_fp4_tpu,
        scale_a,
        scale_b,
        out_dtype=out_dtype,
    )
    expected_shape = torch.Size([m, n])
    self.assertEqual(out.shape, expected_shape)
    self.assertEqual(out.dtype, out_dtype)
    self.assertEqual(out.device.type, "tpu")

    # Verify numeric correctness by dequantizing and running float32 matmul
    self_float_dequant = self_fp4_tpu.to(torch.float32)
    mat2_float_dequant = mat2_fp4_tpu.to(torch.float32)
    expected_out = (
        torch.matmul(self_float_dequant, mat2_float_dequant) * scale_a * scale_b
    ).to(out_dtype)
    utils.assert_close(
        out.to(torch.float32),
        expected_out.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )

  @parameterized.parameters(
      (2, 32, 64, 32, torch.bfloat16),
  )
  def test_scaled_grouped_mm_fp4_numeric(self, num_groups, m, n, k, out_dtype):
    """Tests torch._scaled_grouped_mm with FP4 inputs."""

    if "TPU_NAME" not in os.environ:
      raise unittest.SkipTest(
          "FP4 scaled_grouped_mm test requires TPU hardware to avoid PyTorch"
          " OpMathType CPU fallback crash."
      )

    device = "tpu"
    self_float_tpu = torch.randn(num_groups, m, k, dtype=torch.float32).to(
        device
    )
    mat2_float_tpu = torch.randn(num_groups, k, n, dtype=torch.float32).to(
        device
    )

    self_fp4_tpu = self_float_tpu.to(torch.float4_e2m1fn_x2)
    mat2_fp4_tpu = mat2_float_tpu.to(torch.float4_e2m1fn_x2)

    scale_a = torch.rand(num_groups, dtype=torch.float32).to(device)
    scale_b = torch.rand(num_groups, dtype=torch.float32).to(device)

    out = torch._scaled_grouped_mm(
        self_fp4_tpu,
        mat2_fp4_tpu,
        scale_a,
        scale_b,
        out_dtype=out_dtype,
    )
    expected_shape = torch.Size([num_groups, m, n])
    self.assertEqual(out.shape, expected_shape)
    self.assertEqual(out.dtype, out_dtype)
    self.assertEqual(out.device.type, "tpu")

    # Verify numeric correctness by dequantizing and running float32 matmul
    self_float_dequant = self_fp4_tpu.to(torch.float32)
    mat2_float_dequant = mat2_fp4_tpu.to(torch.float32)

    expected_out = torch.empty(num_groups, m, n, dtype=out_dtype, device=device)
    for g in range(num_groups):
      group_out = (
          torch.matmul(self_float_dequant[g], mat2_float_dequant[g])
          * scale_a[g]
          * scale_b[g]
      )
      expected_out[g] = group_out.to(out_dtype)

    utils.assert_close(
        out.to(torch.float32),
        expected_out.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )

  def test_col2im_fold(self):
    """Tests col2im via torch.nn.Fold.

    torch.nn.Fold is a module wrapper around col2im. We test it here because
    ops_test.py does not support testing modules via OpInfoDB.
    """
    fold = torch.nn.Fold(output_size=(4, 5), kernel_size=(2, 2))
    img = torch.arange(1 * 12 * 12, dtype=torch.float32).reshape(1, 12, 12)

    def test(device):
      output = fold(img.to(device))
      return output

    self.assert_close_tpu_vs_cpu(
        test,
        rtol=9.4e-2,
        atol=8e-3,
    )

  @parameterized.parameters(
      (torch.float32,),
      (torch.float64,),
      (torch.float16,),
      (torch.bfloat16,),
      (torch.complex64,),
  )
  def test_col2im_dtypes(self, dtype):
    """Tests col2im with various dtypes."""
    n, c, h, w = 1, 2, 4, 4
    kernel_size = (2, 2)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (1, 1)
    output_size = (h, w)
    k_h, k_w = kernel_size
    d_h, d_w = dilation
    p_h, p_w = padding
    s_h, s_w = stride
    o_h, o_w = output_size
    l_h = (o_h + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
    l_w = (o_w + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1
    l = l_h * l_w
    col_shape = (n, c * k_h * k_w, l)
    col = torch.randn(*col_shape, dtype=torch.float32)

    def test_fn(device):
      col_dev = col.to(dtype=dtype, device=device)
      return torch.ops.aten.col2im(
          col_dev, output_size, kernel_size, dilation, padding, stride
      )

    self.assert_close_tpu_vs_cpu(test_fn, rtol=7.2e-03, atol=5.9e-03)

  @parameterized.parameters(
      # (kernel_size, dilation, padding, stride, output_size)
      ((1, 1), (1, 1), (0, 0), (1, 1), (4, 4)),  # 1x1
      ((2, 3), (1, 1), (0, 0), (1, 1), (4, 4)),  # asymmetric kernel
      ((2, 2), (2, 2), (0, 0), (1, 1), (8, 8)),  # dilation
      ((2, 2), (1, 1), (1, 1), (1, 1), (4, 4)),  # padding
      ((2, 2), (1, 1), (0, 0), (2, 2), (4, 4)),  # stride
      ((3, 3), (1, 1), (1, 1), (2, 2), (28, 28)),  # volo outlooker
      ((3, 3), (2, 1), (1, 0), (2, 1), (10, 10)),  # asymmetric everything
  )
  def test_col2im_geometries(
      self, kernel_size, dilation, padding, stride, output_size
  ):
    """Tests col2im with various geometries."""
    n, c = 2, 2
    k_h, k_w = kernel_size
    d_h, d_w = dilation
    p_h, p_w = padding
    s_h, s_w = stride
    o_h, o_w = output_size
    l_h = (o_h + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
    l_w = (o_w + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1
    l = l_h * l_w
    col_shape = (n, c * k_h * k_w, l)
    col = torch.randn(*col_shape)

    def test_fn(device):
      col_dev = col.to(device)
      return torch.ops.aten.col2im(
          col_dev, output_size, kernel_size, dilation, padding, stride
      )

    self.assert_close_tpu_vs_cpu(test_fn, rtol=5e-2, atol=5e-2)

  @parameterized.parameters(
      (torch.float32,),
      (torch.float64,),
      (torch.float16,),
      (torch.bfloat16,),
  )
  def test_im2col_dtypes(self, dtype):
    """Tests im2col with various dtypes."""
    n, c, h, w = 1, 2, 4, 4
    kernel_size = (2, 2)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (1, 1)

    input_val = torch.randn(n, c, h, w, dtype=torch.float32)

    def test_fn(device):
      input_dev = input_val.to(dtype=dtype, device=device)
      return torch.nn.functional.unfold(
          input_dev, kernel_size, dilation, padding, stride
      )

    self.assert_close_tpu_vs_cpu(test_fn)

  @parameterized.parameters(
      # (kernel_size, dilation, padding, stride)
      ((1, 1), (1, 1), (0, 0), (1, 1)),  # 1x1
      ((2, 3), (1, 1), (0, 0), (1, 1)),  # asymmetric kernel
      ((2, 2), (2, 2), (0, 0), (1, 1)),  # dilation
      ((2, 2), (1, 1), (1, 1), (1, 1)),  # padding
      ((2, 2), (1, 1), (0, 0), (2, 2)),  # stride
      ((3, 3), (2, 1), (1, 0), (2, 1)),  # asymmetric everything
  )
  def test_im2col_geometries(self, kernel_size, dilation, padding, stride):
    """Tests im2col with various geometries."""
    n, c, h, w = 2, 2, 8, 8
    input_val = torch.randn(n, c, h, w)

    def test_fn(device):
      input_dev = input_val.to(device)
      return torch.nn.functional.unfold(
          input_dev, kernel_size, dilation, padding, stride
      )

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_im2col_errors(self):
    """Tests im2col with invalid inputs that trigger error messages."""
    tpu_device = torch.device("tpu")

    # Test invalid input dimensions (e.g. 2D tensor)
    # C++ check: input.dim() == 4 (after unsqueeze if 3D)
    x_2d = torch.randn(2, 2).to(tpu_device)
    with self.assertRaisesRegex(
        RuntimeError, "expected input to have 3 or 4 dimensions"
    ):
      torch.ops.aten.im2col(
          x_2d,
          kernel_size=(2, 2),
          dilation=(1, 1),
          padding=(0, 0),
          stride=(1, 1),
      )

    # Test non-positive output shape
    # kernel_size > input_size without padding
    x_small = torch.randn(1, 1, 2, 2).to(tpu_device)
    with self.assertRaisesRegex(
        RuntimeError, "expected output shape to be positive"
    ):
      torch.ops.aten.im2col(
          x_small,
          kernel_size=(3, 3),
          dilation=(1, 1),
          padding=(0, 0),
          stride=(1, 1),
      )

  def test_conj(self):
    def test(device):
      x = torch.tensor([[1 + 1j, 2 - 2j], [3 + 3j, 4 - 4j]]).to(device=device)
      self.assertFalse(x.is_conj())
      y = torch.conj(x)
      self.assertTrue(y.is_conj())
      return y

    self.assert_close_tpu_vs_cpu(test)

  def test_conj_in_place(self):
    def test(device):
      x = torch.tensor([[1 + 1j, 2 - 2j], [3 + 3j, 4 - 4j]]).to(device=device)
      self.assertFalse(x.is_conj())
      y = torch._conj(x)
      self.assertTrue(y.is_conj())
      return y

    self.assert_close_tpu_vs_cpu(test)

  def test_conj_view_inplace(self):
    def test(device):
      x = torch.tensor([[1 + 1j, 2 - 2j], [3 + 3j, 4 - 4j]]).to(device=device)
      y = x.conj()
      # In-place modification of y (the view).
      # Since y = conj(x), y += 1 means conj(x) += 1.
      # Conjugating both sides: x += conj(1) = x + 1.
      # So x should increase by 1.
      y.add_(1)
      # Check that x is modified.
      return x, y

    self.assert_close_tpu_vs_cpu(test)

  def test_conj_transpose_view(self):
    def test(device):
      x = torch.tensor([[1 + 1j, 2 - 2j], [3 + 3j, 4 - 4j]], device=device)
      y = torch.tensor([[2 + 1j, 4 + 3j], [3 - 2j, 5 - 4j]], device=device)

      x1 = x.clone()
      # Use transpose to make the view non-contiguous to avoid bypassing
      # compilation via direct buffer pointer swap. Similar below.
      x1.t().copy_(y)

      x2 = x.clone()
      x2.conj().t().copy_(y)

      # Ensure writes to conjugate and non-conjugate views both succeed.
      return x1, x2

    self.assert_close_tpu_vs_cpu(test)

  def test_conj_chain(self):
    def test(device):
      x = torch.tensor([[1 + 1j, 2 - 2j], [3 + 3j, 4 - 4j]]).to(device=device)
      # Double conjugation should be identity.
      y = x.conj().conj()
      return y

    self.assert_close_tpu_vs_cpu(test)

  def test_conj_compose(self):
    def test(device):
      x = torch.tensor([[1 + 1j, 2 - 2j], [3 + 3j, 4 - 4j]]).to(device=device)
      # Test interaction with transpose.
      # y = conj(x)^T
      y = x.conj().transpose(0, 1)
      return y

    self.assert_close_tpu_vs_cpu(test)

  def test_conj_real(self):
    def test(device):
      x = torch.tensor([[1, 2], [3, 4]]).to(device=device)
      # Conj on real tensor should be identity/no-op.
      y = x.conj()
      return y

    self.assert_close_tpu_vs_cpu(test)

  def test_conj_bitcast(self):
    def test(device):
      # float32 -> complex64 is a bitcast.
      # 4 float32s -> 2 complex64 elements.
      x = torch.ones(4, dtype=torch.float32, device=device)
      y = x.view(torch.complex64).conj()
      return y

    self.assert_close_tpu_vs_cpu(test)

  def test_conj_copy(self):
    def test(device):
      x = torch.tensor([[1 + 1j, 2 - 2j], [3 + 3j, 4 - 4j]]).to(device=device)
      self.assertFalse(x.is_conj())
      y = torch.zeros_like(x).to(device=device)
      torch._conj_copy(x, out=y)
      self.assertFalse(y.is_conj())
      return y

    self.assert_close_tpu_vs_cpu(test)

  def test_copy_within_device(self):
    """Tests tensor.copy_() within the same device."""
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.empty(3, device=device).copy_(
            torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
        )
    )

  def test_copy_cpu_to_tpu(self):
    """Tests CPU -> TPU copy."""
    tpu_device = torch.device("tpu")
    src_cpu = torch.tensor([1, 2, 3], dtype=torch.int32)
    dst_tpu = torch.empty(3, dtype=torch.int32, device=tpu_device)
    dst_tpu.copy_(src_cpu)
    self.assertEqual(dst_tpu.cpu(), src_cpu)

  def test_copy_tpu_to_cpu(self):
    """Tests TPU -> CPU copy."""
    tpu_device = torch.device("tpu")
    src_tpu = torch.tensor([4, 5, 6], dtype=torch.int32, device=tpu_device)
    dst_cpu = torch.empty(3, dtype=torch.int32)
    dst_cpu.copy_(src_tpu)
    self.assertEqual(dst_cpu, src_tpu.cpu())

  def test_copy_broadcasting_tpu_to_tpu(self):
    """Tests TPU -> TPU broadcasting copy."""
    tpu_device = torch.device("tpu")
    # Case A: Scalar -> 1D
    src_tpu = torch.tensor(3.14, dtype=torch.float32, device=tpu_device)
    dst_tpu = torch.empty(3, dtype=torch.float32, device=tpu_device)
    dst_tpu.copy_(src_tpu)
    self.assertEqual(dst_tpu.cpu(), torch.tensor([3.14, 3.14, 3.14]))

    # Case B: 1D -> 2D
    src_tpu = torch.tensor(
        [1.0, 2.0, 3.0], dtype=torch.float32, device=tpu_device
    )
    dst_tpu = torch.empty(2, 3, dtype=torch.float32, device=tpu_device)
    dst_tpu.copy_(src_tpu)
    self.assertEqual(dst_tpu.cpu(), src_tpu.cpu().expand(2, 3))

  def test_copy_broadcasting_cpu_to_tpu(self):
    """Tests CPU -> TPU broadcasting copy."""
    tpu_device = torch.device("tpu")
    # Case A: Scalar -> 1D
    src_cpu = torch.tensor(3.14, dtype=torch.float32)
    dst_tpu = torch.empty(3, dtype=torch.float32, device=tpu_device)
    dst_tpu.copy_(src_cpu)
    self.assertEqual(dst_tpu.cpu(), torch.tensor([3.14, 3.14, 3.14]))

    # Case B: 1D -> 2D
    src_cpu = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    dst_tpu = torch.empty(2, 3, dtype=torch.float32, device=tpu_device)
    dst_tpu.copy_(src_cpu)
    self.assertEqual(dst_tpu.cpu(), src_cpu.expand(2, 3))

  def test_copy_broadcasting_tpu_to_cpu(self):
    """Tests TPU -> CPU broadcasting copy."""
    tpu_device = torch.device("tpu")
    # Case A: Scalar -> 1D
    src_tpu = torch.tensor(3.14, dtype=torch.float32, device=tpu_device)
    dst_cpu = torch.empty(3, dtype=torch.float32)
    dst_cpu.copy_(src_tpu)
    self.assertEqual(dst_cpu, torch.tensor([3.14, 3.14, 3.14]))

    # Case B: 1D -> 2D
    src_tpu = torch.tensor(
        [1.0, 2.0, 3.0], dtype=torch.float32, device=tpu_device
    )
    dst_cpu = torch.empty(2, 3, dtype=torch.float32)
    dst_cpu.copy_(src_tpu)
    self.assertEqual(dst_cpu, src_tpu.cpu().expand(2, 3))

  def test_copy_casting(self):
    """Tests copy with type casting."""
    tpu_device = torch.device("tpu")
    src_cpu = torch.tensor([1.5, 2.5], dtype=torch.float32)
    dst_tpu = torch.empty(2, dtype=torch.int32, device=tpu_device)
    dst_tpu.copy_(src_cpu)
    self.assertEqual(dst_tpu.cpu(), src_cpu.int())

  def test_copy_aten_op(self):
    """Tests torch.ops.aten.copy."""
    tpu_device = torch.device("tpu")
    src_cpu = torch.tensor([1, 2, 3], dtype=torch.int32)
    dst_tpu = torch.empty(3, dtype=torch.int32, device=tpu_device)
    dst_tpu = torch.ops.aten.copy(dst_tpu, src_cpu)
    self.assertEqual(dst_tpu.cpu(), src_cpu)

  def test_copy_gradients(self):
    """Tests gradient copying behavior."""
    tpu_device = torch.device("tpu")

    # --- TPU -> TPU Gradient Copy ---
    src_tpu = torch.tensor([1.0, 2.0], device=tpu_device, requires_grad=True)
    src_tpu.grad = torch.tensor([0.1, 0.2], device=tpu_device)
    dst_tpu = torch.tensor([3.0, 4.0], device=tpu_device, requires_grad=True)
    with torch.no_grad():
      dst_tpu.copy_(src_tpu)
    self.assertIsNotNone(dst_tpu.grad)
    self.assertEqual(dst_tpu.grad.cpu(), torch.tensor([0.1, 0.2]))

    # --- TPU -> CPU Gradient Copy ---
    src_tpu = torch.tensor([1.0, 2.0], device=tpu_device, requires_grad=True)
    src_tpu.grad = torch.tensor([0.1, 0.2], device=tpu_device)
    dst_cpu = torch.tensor([3.0, 4.0], device="cpu", requires_grad=True)
    with torch.no_grad():
      dst_cpu.copy_(src_tpu)
    self.assertIsNotNone(dst_cpu.grad)
    self.assertEqual(dst_cpu.grad, torch.tensor([0.1, 0.2]))

    # --- CPU -> TPU Gradient Copy ---
    src_cpu = torch.tensor([1.0, 2.0], device="cpu", requires_grad=True)
    src_cpu.grad = torch.tensor([0.1, 0.2], device="cpu")
    dst_tpu = torch.tensor([3.0, 4.0], device=tpu_device, requires_grad=True)
    with torch.no_grad():
      dst_tpu.copy_(src_cpu)
    self.assertIsNotNone(dst_tpu.grad)
    self.assertEqual(dst_tpu.grad.cpu(), torch.tensor([0.1, 0.2]))

  def test_copy_equivalent_transposes(self):
    tpu_device = torch.device("tpu")

    def _test_fn(device) -> torch.Tensor:
      src = torch.tensor([[1, 2], [3, 4]], device=device)
      dst = torch.tensor([[-1, -1], [-1, -1]], device=device)
      dst.t().copy_(src.t())
      return dst.cpu()

    expected = torch.tensor([[1, 2], [3, 4]], device="cpu")
    actual_tpu = _test_fn(tpu_device)
    actual_cpu = _test_fn(torch.device("cpu"))
    utils.assert_close(actual_tpu, expected)
    utils.assert_close(actual_cpu, expected)

  def test_copy_src_transposed(self):
    tpu_device = torch.device("tpu")

    def _test_fn(device) -> torch.Tensor:
      src = torch.tensor([[1, 2], [3, 4]], device=device)
      dst = torch.tensor([[-1, -1], [-1, -1]], device=device)
      dst.copy_(src.t())
      return dst.cpu()

    expected = torch.tensor([[1, 3], [2, 4]], device="cpu")
    actual_tpu = _test_fn(tpu_device)
    actual_cpu = _test_fn(torch.device("cpu"))
    utils.assert_close(actual_tpu, expected)
    utils.assert_close(actual_cpu, expected)

  def test_copy_dst_transposed(self):
    tpu_device = torch.device("tpu")

    def _test_fn(device) -> torch.Tensor:
      src = torch.tensor([[1, 2], [3, 4]], device=device)
      dst = torch.tensor([[-1, -1], [-1, -1]], device=device)
      dst.t().copy_(src)
      return dst.cpu()

    expected = torch.tensor([[1, 3], [2, 4]], device="cpu")
    actual_tpu = _test_fn(tpu_device)
    actual_cpu = _test_fn(torch.device("cpu"))
    utils.assert_close(actual_tpu, expected)
    utils.assert_close(actual_cpu, expected)

  def test_copy_broadcast_into_transposed(self):
    tpu_device = torch.device("tpu")

    def _test_fn(device) -> torch.Tensor:
      src = torch.tensor([[1], [2]], device=device)
      dst = torch.tensor([[-1, -1], [-1, -1]], device=device)
      dst.t().copy_(src)
      return dst.cpu()

    expected = torch.tensor([[1, 2], [1, 2]], device="cpu")
    actual_tpu = _test_fn(tpu_device)
    actual_cpu = _test_fn(torch.device("cpu"))
    utils.assert_close(actual_tpu, expected)
    utils.assert_close(actual_cpu, expected)

  def test_copy_non_contiguous_non_blocking(self):
    tpu_device = torch.device("tpu")
    cpu_source = torch.randn(400, device="cpu")
    tpu_buffers = []
    for i in range(100):
      cpu_noncontiguous_slice = cpu_source[i : i + 4].view(2, 2).t()
      tpu_buffers.append(
          cpu_noncontiguous_slice.to(
              device=tpu_device,
              non_blocking=True,
              memory_format=torch.contiguous_format,
          )
      )
    torch.accelerator.synchronize()
    results = []
    for i in range(100):
      results.append(
          tpu_buffers[i].to(
              device="cpu",
              non_blocking=True,
              memory_format=torch.contiguous_format,
          )
      )
    torch.accelerator.synchronize()
    for i in range(100):
      utils.assert_close(results[i], cpu_source[i : i + 4].view(2, 2).t())

  def test_copy_different_dtype_non_blocking(self):
    tpu_device = torch.device("tpu")
    cpu_source = torch.randn(100, dtype=torch.float32, device="cpu")
    tpu_buffers = []
    for i in range(100):
      cpu_noncontiguous_slice = cpu_source[i].view(1)
      tpu_target = torch.empty(1, dtype=torch.bfloat16, device=tpu_device)
      tpu_target.copy_(cpu_noncontiguous_slice, non_blocking=True)
      tpu_buffers.append(tpu_target)
    torch.accelerator.synchronize()
    results = []
    for i in range(100):
      results.append(
          tpu_buffers[i].to(
              device="cpu",
              non_blocking=True,
              memory_format=torch.contiguous_format,
          )
      )
    torch.accelerator.synchronize()
    for i in range(100):
      utils.assert_close(results[i], cpu_source[i].view(1).to(torch.bfloat16))

  def test_copy_zero_sized_non_blocking(self):
    tpu_device = torch.device("tpu")
    cpu_source = torch.tensor([], device="cpu")
    tpu_buffers = []
    for _ in range(100):
      tpu_buffers.append(
          cpu_source.to(
              device=tpu_device,
              non_blocking=True,
              memory_format=torch.contiguous_format,
          )
      )
    torch.accelerator.synchronize()
    results = []
    for i in range(100):
      results.append(
          tpu_buffers[i].to(
              device="cpu",
              non_blocking=True,
              memory_format=torch.contiguous_format,
          )
      )
    torch.accelerator.synchronize()
    for i in range(100):
      self.assertEqual(results[i], cpu_source)

  def test_histc_dynamic_bounds(self):
    """Test cases that cause the min and max computation the input data.

    This is done when min == max and the input is not empty.
    """
    inputs = torch.tensor([0.1, 1.1, 3.1, 5.1, 1.1])
    bins = 8
    minimum = 0.1
    maximum = 5.1

    # Test with all combinations of min and max.
    for min_val, max_val in itertools.product([0.0, minimum], [0.0, maximum]):
      if min_val > max_val:
        continue
      self.assert_close_tpu_vs_cpu(
          lambda device, mv=min_val, mx=max_val: torch.histc(
              torch.tensor(inputs, dtype=torch.float32).to(device=device),
              bins=bins,
              min=mv,
              max=mx,
          ),
      )

    # Test with min == max.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.histc(
            torch.tensor(inputs, dtype=torch.float32).to(device=device),
            bins=bins,
            min=0.1,
            max=0.1,
        ),
    )

    # Test with min == max == 0.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.histc(
            torch.tensor(inputs, dtype=torch.float32).to(device=device),
            bins=bins,
            min=0,
            max=0,
        ),
    )

  def test_histc_explicit_bounds(self):
    inputs = torch.tensor([0.1, 1.1, 3.1, 5.1, 1.1])
    bins = 8
    minimum = 0.1
    maximum = 5.1

    # Test with out of bounds data.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.histc(
            torch.tensor(inputs, dtype=torch.float32).to(device=device),
            bins=8,
            min=1.0,
            max=4.0,
        ),
    )

    # Test with empty input tensor.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.histc(
            torch.tensor([], dtype=torch.float32).to(device=device),
            bins=bins,
            min=minimum,
            max=maximum,
        ),
    )

    # Test with / without bins.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.histc(
            torch.tensor(inputs, dtype=torch.float32).to(device=device),
            min=minimum,
            max=maximum,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.histc(
            torch.tensor(inputs, dtype=torch.float32).to(device=device),
            bins=bins,
            min=minimum,
            max=maximum,
        ),
    )

  def test_histc_dtypes(self):
    inputs = torch.tensor([0.1, 1.1, 3.1, 5.1, 1.1])
    bins = 8

    # Test with float16 dtype, which were excluded from the general op tests due
    # to expected binning errors.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.histc(
            torch.tensor(inputs, dtype=torch.float16).to(device=device),
            bins=bins,
        ),
    )

    # Test with integer dtypes.
    for int_dtype in [torch.int32, torch.int64]:
      for int_input, int_result in [
          ([0, 1, 2, 3, 4, 3, 1, 2], [1, 2, 2, 2, 1, 0, 0, 0]),
          ([0, 0, 0, 0, 1, 1, 1, 1], [4, 4, 0, 0, 0, 0, 0, 0]),
      ]:
        tpu_input = torch.tensor(int_input, dtype=int_dtype).to(
            device=torch.device("tpu")
        )
        tpu_result = torch.histc(tpu_input, bins=8, min=0, max=7)
        self.assertEqual(tpu_result, torch.tensor(int_result, dtype=int_dtype))

  @parameterized.named_parameters(
      (
          "int8",
          [-7, 9, -7, -9, 6],
          torch.int8,
          10,
          -10,
          10,
          [1, 2, 0, 0, 0, 0, 0, 0, 1, 1],
      ),
      (
          "uint8",
          [1, 9, 1, 0, 6],
          torch.uint8,
          10,
          0,
          10,
          [1, 2, 0, 0, 0, 0, 1, 0, 0, 1],
      ),
  )
  def test_histc_int8_uint8(
      self, input_data, dtype, bins, min_val, max_val, expected
  ):
    # Regression test for b/487653210: histc returns incorrect output on int8
    # and uint8 dtypes.
    tpu_input = torch.tensor(input_data, dtype=dtype, device="tpu")
    tpu_result = torch.histc(tpu_input, bins=bins, min=min_val, max=max_val)
    expected_tensor = torch.tensor(expected, dtype=dtype, device="tpu")
    self.assertEqual(tpu_result, expected_tensor)

  def test_lerp_complex128(self):
    # Regression test for b/451684563: lerp() supports complex128.
    # TODO(b/451684563): Switch to testing TPU-vs-GPU in ops_test.py once GPU
    # goldens are regenerated with complex128 for lerp.
    device = torch.device("tpu")
    start = torch.tensor(
        [1.0 + 2.0j, 3.0 + 4.0j], dtype=torch.complex128, device=device
    )
    end = torch.tensor(
        [5.0 + 6.0j, 7.0 + 8.0j], dtype=torch.complex128, device=device
    )
    weight = torch.tensor(
        [0.5 + 0.0j, 0.25 + 0.0j], dtype=torch.complex128, device=device
    )
    expected = torch.tensor(
        [3.0 + 4.0j, 4.0 + 5.0j], dtype=torch.complex128, device=device
    )
    self.assertEqual(torch.lerp(start, end, weight), expected)

    # Test in-place variant with complex128 weight.
    start_inplace = start.clone()
    start_inplace.lerp_(end, weight)
    self.assertEqual(start_inplace, expected)

    # Test with 0D float64 tensor weight (f64[] -> c128[]).
    weight_f64_0d = torch.tensor(0.5, dtype=torch.float64, device=device)
    expected_scalar = torch.tensor(
        [3.0 + 4.0j, 5.0 + 6.0j], dtype=torch.complex128, device=device
    )
    self.assertEqual(torch.lerp(start, end, weight_f64_0d), expected_scalar)

    # Test in-place variant with 0D float64 tensor weight.
    start_inplace_0d = start.clone()
    start_inplace_0d.lerp_(end, weight_f64_0d)
    self.assertEqual(start_inplace_0d, expected_scalar)

    # Test with scalar weight (eager and compiled).
    self.assertEqual(torch.lerp(start, end, 0.5), expected_scalar)
    compiled_fn = torch.compile(lambda a, b: torch.lerp(a, b, 0.5))
    self.assertEqual(compiled_fn(start, end), expected_scalar)

    # Test in-place variant with scalar weight.
    start_inplace_scalar = start.clone()
    start_inplace_scalar.lerp_(end, 0.5)
    self.assertEqual(start_inplace_scalar, expected_scalar)

    # Compare TPU vs CPU execution.
    self.assert_close_tpu_vs_cpu(
        lambda dev: torch.lerp(
            torch.tensor(
                [1.0 + 2.0j, 3.0 + 4.0j],
                dtype=torch.complex128,
                device=dev,
            ),
            torch.tensor(
                [5.0 + 6.0j, 7.0 + 8.0j],
                dtype=torch.complex128,
                device=dev,
            ),
            torch.tensor(
                [0.5 + 0.0j, 0.25 + 0.0j],
                dtype=torch.complex128,
                device=dev,
            ),
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda dev: torch.lerp(
            torch.tensor(
                [1.0 + 2.0j, 3.0 + 4.0j],
                dtype=torch.complex128,
                device=dev,
            ),
            torch.tensor(
                [5.0 + 6.0j, 7.0 + 8.0j],
                dtype=torch.complex128,
                device=dev,
            ),
            torch.tensor(0.5, dtype=torch.float64, device=dev),
        )
    )

  def test_conj_physical(self):
    def test_out(device):
      x = torch.tensor([[1 + 1j, 2 - 2j], [3 + 3j, 4 - 4j]]).to(device=device)
      y = torch.zeros_like(x).to(device=device)
      torch.conj_physical(x, out=y)
      return y

    self.assert_close_tpu_vs_cpu(test_out)

    def test(device):
      x = torch.tensor([[1 + 1j, 2 - 2j], [3 + 3j, 4 - 4j]]).to(device=device)
      return torch.conj_physical(x)

    self.assert_close_tpu_vs_cpu(test)

    # TODO: b/448907643 - there is a problem with the plumbing of the inplace
    # variant:
    #
    # def test_in_place(device):
    #   x = torch.tensor([[1 + 1j, 2 - 2j], [3 + 3j, 4 - 4j]]).to(device=device)
    #   torch.conj_physical_(x)
    #   return x
    # self.assert_close_tpu_vs_cpu(test_in_place)

    def _test(device):
      x = torch.tensor([[1 + 1j, 2 - 2j], [3 + 3j, 4 - 4j]]).to(device=device)
      return torch._conj_physical(x)

    self.assert_close_tpu_vs_cpu(_test)

    # TODO: b/448907643 - can't exercise the out variant of the internal op.
    # def _test_out(device):
    #   x = torch.tensor([[1 + 1j, 2 - 2j], [3 + 3j, 4 - 4j]]).to(device=device)
    #   y = torch.zeros_like(x).to(device=device)
    #   torch._conj_physical(x, out=y)
    #   return y
    # self.assert_close_tpu_vs_cpu(_test_out)

  def test_reshape_with_zero_numel(self):
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.reshape(
            torch.empty(0, 2).to(device),
            [3, 0, 4],
        ),
    )

  def test_torch_manual_seed_same_seed_same_result(self):
    torch.manual_seed(321)
    x = torch.rand(1, dtype=torch.float32, device=torch.device("tpu"))
    torch.manual_seed(321)
    y = torch.rand(1, dtype=torch.float32, device=torch.device("tpu"))
    self.assertEqual(x.cpu(), y.cpu())

  def test_torch_manual_seed_different_seed_different_result(self):
    torch.manual_seed(321)
    x = torch.rand(1, dtype=torch.float32, device=torch.device("tpu"))
    torch.manual_seed(123)
    y = torch.rand(1, dtype=torch.float32, device=torch.device("tpu"))
    self.assertNotEqual(x.cpu(), y.cpu())

  def test_bitwise_left_shift(self):
    tpu_device = torch.device("tpu")
    shift_by_int32 = torch.tensor([1, 2, 4, 5], dtype=torch.int32)
    shift_by_int32_tpu = shift_by_int32.to(tpu_device)
    shift_by_int64 = torch.tensor([1, 2, 4, 5], dtype=torch.int64)
    shift_by_int64_tpu = shift_by_int64.to(tpu_device)

    # Basic scalar tensor test.
    tpu_result = torch.bitwise_left_shift(1, shift_by_int32_tpu).cpu()
    golden_result = torch.bitwise_left_shift(1, shift_by_int32)
    self.assert_close(golden_result=golden_result, torch_tpu_result=tpu_result)

    tpu_result = torch.bitwise_left_shift(1, shift_by_int64_tpu).cpu()
    golden_result = torch.bitwise_left_shift(1, shift_by_int64)
    self.assert_close(golden_result=golden_result, torch_tpu_result=tpu_result)

    # Test with the largest int32 value.
    tpu_result = torch.bitwise_left_shift(2147483647, shift_by_int32_tpu).cpu()
    golden_result = torch.bitwise_left_shift(2147483647, shift_by_int32)
    self.assert_close(golden_result=golden_result, torch_tpu_result=tpu_result)

    # Test with int64 input and shift by int32 tensor.
    input_long = torch.tensor([37, 128, 200, 511], dtype=torch.int64)
    input_long_tpu = input_long.to(tpu_device)
    tpu_result = torch.bitwise_left_shift(
        input_long_tpu, shift_by_int32_tpu
    ).cpu()
    golden_result = torch.bitwise_left_shift(input_long, shift_by_int32)
    self.assert_close(golden_result=golden_result, torch_tpu_result=tpu_result)

    # Test with int32 input and shift by int64 tensor.
    input_int32 = torch.tensor([37, 128, 200, 511], dtype=torch.int32)
    input_int32_tpu = input_int32.to(tpu_device)
    tpu_result = torch.bitwise_left_shift(
        input_int32_tpu, shift_by_int64_tpu
    ).cpu()
    golden_result = torch.bitwise_left_shift(input_int32, shift_by_int64)
    self.assert_close(golden_result=golden_result, torch_tpu_result=tpu_result)

    # Basic tensor scalar test.
    tpu_result = torch.bitwise_left_shift(input_long_tpu, 1).cpu()
    golden_result = torch.bitwise_left_shift(input_long, 1)
    self.assert_close(golden_result=golden_result, torch_tpu_result=tpu_result)

    # Test bitwise shift with mixed bool and integer inputs (b/478955517).
    int_a = torch.tensor([1, 2, 4], dtype=torch.int32)
    bool_b = torch.tensor([True, False, True], dtype=torch.bool)
    tpu_res = torch.bitwise_left_shift(
        int_a.to(tpu_device), bool_b.to(tpu_device)
    ).cpu()
    golden_res = torch.bitwise_left_shift(int_a, bool_b)
    self.assert_close(golden_result=golden_res, torch_tpu_result=tpu_res)

    tpu_res = torch.bitwise_left_shift(
        bool_b.to(tpu_device), int_a.to(tpu_device)
    ).cpu()
    golden_res = torch.bitwise_left_shift(bool_b, int_a)
    self.assert_close(golden_result=golden_res, torch_tpu_result=tpu_res)

    int64_a = torch.tensor([1, 2, 4], dtype=torch.int64)
    tpu_res = (int64_a.to(tpu_device) << bool_b.to(tpu_device)).cpu()
    golden_res = int64_a << bool_b
    self.assert_close(golden_result=golden_res, torch_tpu_result=tpu_res)

    tpu_res = (bool_b.to(tpu_device) << int64_a.to(tpu_device)).cpu()
    golden_res = bool_b << int64_a
    self.assert_close(golden_result=golden_res, torch_tpu_result=tpu_res)

  def test_upsample_bilinear2d_precision_edge_case(self):
    # This test targets a specific case where floating point precision issues
    # in coordinate calculation can cause out-of-bounds access if not clamped.
    # Case: in_size=4, out_size=22.
    # stride = (4 - 1) / (22 - 1) = 3 / 21 = 1/7
    # last_idx = 21
    # src_idx = 21 * (1/7) in float32 is 3.0000002... > 3.0
    # If not clamped, ceil(src_idx) becomes 4, which is OOB (valid: 0, 1, 2, 3).

    tpu_device = torch.device("tpu")
    n, c = 1, 1
    h_in, w_in = 4, 4
    # We use a large output size that triggers the precision issue.
    # We only need one dimension to trigger it, but we'll scale both.
    h_out, w_out = 22, 22

    # Create input with known values (e.g., indices) to easily check correctness
    x = torch.arange(h_in * w_in, dtype=torch.float32).reshape(n, c, h_in, w_in)
    x_tpu = x.to(tpu_device)

    # We expect the last pixel to be exactly the last pixel of the input
    # because align_corners=True maps corner to corner.
    out_tpu = torch.nn.functional.interpolate(
        x_tpu, size=(h_out, w_out), mode="bilinear", align_corners=True
    )
    out_cpu = torch.nn.functional.interpolate(
        x, size=(h_out, w_out), mode="bilinear", align_corners=True
    )

    self.assert_close(golden_result=out_cpu, torch_tpu_result=out_tpu.cpu())

  def test_upsample_bilinear2d_align_corners_false_edge_case(self):
    # This test targets the lower bound edge case for align_corners=False.
    # Where src_idx can be negative.
    # Case: in_size=2, out_size=4.
    # scale = 2 / 4 = 0.5.
    # dst_idx = 0.
    # src_idx = (0 + 0.5) * 0.5 - 0.5 = 0.25 - 0.5 = -0.25.
    # floor(-0.25) = -1. ceil(-0.25) = 0.
    # If not clamped to 0:
    #   We gather at -1 and 0.
    #   If -1 wraps to last element (1), we get mixed result.
    #   val[0]=10, val[1]=20.
    #   lambda = -0.25 - (-1) = 0.75.
    #   res = val[-1]*(1-0.75) + val[0]*0.75 = 20*0.25 + 10*0.75 = 12.5.
    # Expected (clamped): val[0] = 10.

    tpu_device = torch.device("tpu")
    n, c = 1, 1
    h_in, w_in = 1, 2
    h_out, w_out = 1, 4

    x = torch.tensor([10.0, 20.0], dtype=torch.float32).reshape(
        n, c, h_in, w_in
    )
    x_tpu = x.to(tpu_device)

    out_tpu = torch.nn.functional.interpolate(
        x_tpu, size=(h_out, w_out), mode="bilinear", align_corners=False
    )
    out_cpu = torch.nn.functional.interpolate(
        x, size=(h_out, w_out), mode="bilinear", align_corners=False
    )

    self.assert_close(golden_result=out_cpu, torch_tpu_result=out_tpu.cpu())

  @parameterized.product(
      dtype=(torch.float32, torch.bfloat16, torch.float16, torch.float64),
      case=(
          # (input_shape_or_hw, size, scale_factor)
          ((7, 5), (19, 13), None),
          ((16, 12), (6, 5), None),
          ((6, 10), (14, 4), None),
          ((6, 5), None, (2.5, 2.4)),
          ((10, 10), None, (0.4, 0.4)),
          ((5, 6), None, (1.1, 2.5)),
          ((1, 4), (5, 4), None),
          ((5, 6), (1, 1), None),
          ((6, 6), (6, 6), None),
          ((513, 4), (520, 8), None),
          ((0, 3, 5, 5), (10, 10), None),
      ),
      align_corners=(False, True),
  )
  def test_upsample_bilinear2d_forward_backward(
      self, dtype, case, align_corners
  ):
    # Exercises the separable (matmul) bilinear lowering for static shapes:
    # up/down sampling, non-integer and explicit scale factors, mixed
    # directions, identity dims, empty batch, and larger spatial extents.
    atol, rtol = {
        torch.float32: (1e-4, 1e-4),
        torch.bfloat16: (1e-2, 1.6e-2),
        torch.float16: (5e-3, 5e-3),
        torch.float64: (1e-10, 1e-10),
    }[dtype]
    input_shape, size, scale_factor = case
    if len(input_shape) == 2:
      input_shape = (2, 3, *input_shape)
    tpu_device = torch.device("tpu")
    torch.manual_seed(0)
    x_cpu = torch.randn(*input_shape, dtype=dtype, requires_grad=True)
    x_tpu = x_cpu.detach().to(tpu_device).requires_grad_(True)
    kwargs = {
        "size": size,
        "scale_factor": scale_factor,
        "mode": "bilinear",
        "align_corners": align_corners,
    }
    out_cpu = torch.nn.functional.interpolate(x_cpu, **kwargs)
    out_tpu = torch.nn.functional.interpolate(x_tpu, **kwargs)
    self.assert_close(
        golden_result=out_cpu,
        torch_tpu_result=out_tpu.cpu(),
        atol=atol,
        rtol=rtol,
    )
    grad_out = torch.randn_like(out_cpu)
    out_cpu.backward(grad_out)
    out_tpu.backward(grad_out.to(tpu_device))
    # Each input element accumulates up to 4 weighted gradient terms per
    # output element it feeds, so allow more absolute error than the forward.
    self.assert_close(
        golden_result=x_cpu.grad,
        torch_tpu_result=x_tpu.grad.cpu(),
        atol=atol * 4,
        rtol=rtol,
    )

  @parameterized.named_parameters(
      # Both dims keep their size: returns the input unchanged.
      ("identity", (1, 2, 5, 5), (5, 5), 1.3, 1.3),
      # W keeps its size while H is resized: W's scale must be ignored.
      ("partial_identity", (1, 1, 600, 5), (1200, 5), None, 1.3),
  )
  def test_upsample_bilinear2d_identity_with_scale_factor(
      self, input_size, output_size, scale_h, scale_w
  ):
    # PyTorch's CPU kernel ignores the scale of a dim whose size is unchanged.
    tpu_device = torch.device("tpu")
    align_corners = False
    torch.manual_seed(0)
    x = torch.randn(input_size)
    out_cpu = torch.ops.aten.upsample_bilinear2d(
        x, output_size, align_corners, scale_h, scale_w
    )
    out_tpu = torch.ops.aten.upsample_bilinear2d(
        x.to(tpu_device), output_size, align_corners, scale_h, scale_w
    )
    self.assert_close(golden_result=out_cpu, torch_tpu_result=out_tpu.cpu())
    grad_out = torch.randn_like(out_cpu)
    grad_cpu = torch.ops.aten.upsample_bilinear2d_backward(
        grad_out, output_size, input_size, align_corners, scale_h, scale_w
    )
    grad_tpu = torch.ops.aten.upsample_bilinear2d_backward(
        grad_out.to(tpu_device),
        output_size,
        input_size,
        align_corners,
        scale_h,
        scale_w,
    )
    self.assert_close(golden_result=grad_cpu, torch_tpu_result=grad_tpu.cpu())

  def test_upsample_linear1d_align_corners(self):
    tpu_device = torch.device("tpu")
    for align_corners in (True, False):
      x = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
      x_tpu = x.to(tpu_device)
      out_tpu = torch.nn.functional.interpolate(
          x_tpu, size=10, mode="linear", align_corners=align_corners
      )
      out_cpu = torch.nn.functional.interpolate(
          x, size=10, mode="linear", align_corners=align_corners
      )
      self.assert_close(golden_result=out_cpu, torch_tpu_result=out_tpu.cpu())

  def test_upsample_linear1d_with_scale_factor(self):
    tpu_device = torch.device("tpu")
    for align_corners in (True, False):
      x = torch.randn(2, 3, 16, dtype=torch.float32)
      x_tpu = x.to(tpu_device)
      out_tpu = torch.nn.functional.interpolate(
          x_tpu, scale_factor=2.5, mode="linear", align_corners=align_corners
      )
      out_cpu = torch.nn.functional.interpolate(
          x, scale_factor=2.5, mode="linear", align_corners=align_corners
      )
      self.assert_close(golden_result=out_cpu, torch_tpu_result=out_tpu.cpu())

  def test_upsample_linear1d_dtypes(self):
    tpu_device = torch.device("tpu")
    for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.float64):
      x = torch.randn(2, 2, 8, dtype=dtype)
      x_tpu = x.to(tpu_device)
      out_tpu = torch.nn.functional.interpolate(
          x_tpu, size=12, mode="linear", align_corners=False
      )
      out_cpu = torch.nn.functional.interpolate(
          x, size=12, mode="linear", align_corners=False
      )
      self.assert_close(
          golden_result=out_cpu,
          torch_tpu_result=out_tpu.cpu(),
          atol=1e-2,
          rtol=1e-2,
      )

  def test_upsample_linear1d_edge_cases(self):
    tpu_device = torch.device("tpu")
    # Output size 1
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32)
    x_tpu = x.to(tpu_device)
    out_tpu = torch.nn.functional.interpolate(
        x_tpu, size=1, mode="linear", align_corners=False
    )
    out_cpu = torch.nn.functional.interpolate(
        x, size=1, mode="linear", align_corners=False
    )
    self.assert_close(golden_result=out_cpu, torch_tpu_result=out_tpu.cpu())

    # Empty batch
    x_empty = torch.empty(0, 2, 4, dtype=torch.float32)
    x_empty_tpu = x_empty.to(tpu_device)
    out_empty_tpu = torch.nn.functional.interpolate(
        x_empty_tpu, size=8, mode="linear", align_corners=True
    )
    out_empty_cpu = torch.nn.functional.interpolate(
        x_empty, size=8, mode="linear", align_corners=True
    )
    self.assert_close(
        golden_result=out_empty_cpu, torch_tpu_result=out_empty_tpu.cpu()
    )

    # Out variant
    out_tensor = torch.empty(2, 1, 10, dtype=torch.float32, device=tpu_device)
    torch.ops.aten.upsample_linear1d.out(
        x_tpu.repeat(2, 1, 1), [10], False, None, out=out_tensor
    )
    expected_out = torch.nn.functional.interpolate(
        x.repeat(2, 1, 1), size=10, mode="linear", align_corners=False
    )
    self.assert_close(
        golden_result=expected_out, torch_tpu_result=out_tensor.cpu()
    )

  def test_upsample_linear1d_backward_out(self):
    tpu_device = torch.device("tpu")
    shape_cases = [
        # Upsampling backward (in_w < out_w), downsampling backward (in_w > out_w),
        # and identity (in_w == out_w)
        ((2, 3, 4), (8,)),
        ((2, 3, 8), (4,)),
        ((2, 3, 5), (5,)),
        # Multiple overlapping inputs accumulating into the same index
        ((1, 2, 1), (5,)),
        ((2, 2, 2), (7,)),
        ((1, 3, 3), (10,)),
        # Non-divisible sizes
        ((2, 3, 3), (7,)),
        ((2, 3, 7), (3,)),
        ((1, 2, 5), (11,)),
        # Single-element spatial edge cases
        ((1, 1, 1), (1,)),
        ((2, 3, 1), (4,)),
        ((2, 3, 4), (1,)),
        # Empty batch edge case
        ((0, 2, 4), (8,)),
    ]
    for input_size, output_size in shape_cases:
      for align_corners in (True, False):
        grad_out_shape = (input_size[0], input_size[1], output_size[0])
        grad_out_cpu = torch.randn(grad_out_shape, dtype=torch.float32)
        grad_out_tpu = grad_out_cpu.to(tpu_device)
        grad_in_cpu = torch.empty(input_size, dtype=torch.float32)
        grad_in_tpu = torch.empty(
            input_size, dtype=torch.float32, device=tpu_device
        )
        torch.ops.aten.upsample_linear1d_backward.grad_input(
            grad_out_cpu,
            list(output_size),
            list(input_size),
            align_corners,
            None,
            grad_input=grad_in_cpu,
        )
        torch.ops.aten.upsample_linear1d_backward.grad_input(
            grad_out_tpu,
            list(output_size),
            list(input_size),
            align_corners,
            None,
            grad_input=grad_in_tpu,
        )
        self.assert_close(
            golden_result=grad_in_cpu, torch_tpu_result=grad_in_tpu.cpu()
        )

    # Explicit scales (scales=2.5, scales=0.5, scales=None)
    scale_cases = [
        ((2, 3, 4), (10,), 2.5),
        ((2, 3, 8), (4,), 0.5),
        ((2, 3, 4), (8,), None),
    ]
    for input_size, output_size, scales in scale_cases:
      for align_corners in (True, False):
        grad_out_shape = (input_size[0], input_size[1], output_size[0])
        grad_out_cpu = torch.randn(grad_out_shape, dtype=torch.float32)
        grad_out_tpu = grad_out_cpu.to(tpu_device)
        grad_in_cpu = torch.empty(input_size, dtype=torch.float32)
        grad_in_tpu = torch.empty(
            input_size, dtype=torch.float32, device=tpu_device
        )
        torch.ops.aten.upsample_linear1d_backward.grad_input(
            grad_out_cpu,
            list(output_size),
            list(input_size),
            align_corners,
            scales,
            grad_input=grad_in_cpu,
        )
        torch.ops.aten.upsample_linear1d_backward.grad_input(
            grad_out_tpu,
            list(output_size),
            list(input_size),
            align_corners,
            scales,
            grad_input=grad_in_tpu,
        )
        self.assert_close(
            golden_result=grad_in_cpu, torch_tpu_result=grad_in_tpu.cpu()
        )

    # Out-variant buffer reuse: matching shape, wrong shape requiring resize,
    # and empty(0)
    input_size = (2, 3, 4)
    output_size = (8,)
    grad_out_cpu = torch.randn(2, 3, 8, dtype=torch.float32)
    grad_out_tpu = grad_out_cpu.to(tpu_device)
    for init_shape in (input_size, (1, 1, 1), (0,)):
      grad_in_cpu = torch.full(init_shape, 99.0, dtype=torch.float32)
      grad_in_tpu = torch.full(
          init_shape, 99.0, dtype=torch.float32, device=tpu_device
      )
      ret_cpu = torch.ops.aten.upsample_linear1d_backward.grad_input(
          grad_out_cpu,
          list(output_size),
          list(input_size),
          False,
          None,
          grad_input=grad_in_cpu,
      )
      ret_tpu = torch.ops.aten.upsample_linear1d_backward.grad_input(
          grad_out_tpu,
          list(output_size),
          list(input_size),
          False,
          None,
          grad_input=grad_in_tpu,
      )
      self.assertIs(ret_tpu, grad_in_tpu)
      self.assertEqual(grad_in_tpu.shape, input_size)
      self.assert_close(
          golden_result=ret_cpu, torch_tpu_result=grad_in_tpu.cpu()
      )

    # Dtypes: float32, float64, bfloat16, float16
    for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
      for input_size, output_size in (((2, 2, 4), (9,)), ((2, 2, 9), (4,))):
        for align_corners in (True, False):
          grad_out_cpu = torch.randn(
              input_size[0], input_size[1], output_size[0], dtype=dtype
          )
          grad_out_tpu = grad_out_cpu.to(tpu_device)
          grad_in_cpu = torch.empty(input_size, dtype=dtype)
          grad_in_tpu = torch.empty(input_size, dtype=dtype, device=tpu_device)
          torch.ops.aten.upsample_linear1d_backward.grad_input(
              grad_out_cpu,
              list(output_size),
              list(input_size),
              align_corners,
              None,
              grad_input=grad_in_cpu,
          )
          torch.ops.aten.upsample_linear1d_backward.grad_input(
              grad_out_tpu,
              list(output_size),
              list(input_size),
              align_corners,
              None,
              grad_input=grad_in_tpu,
          )
          if dtype in (torch.bfloat16, torch.float16):
            self.assert_close(
                golden_result=grad_in_cpu,
                torch_tpu_result=grad_in_tpu.cpu(),
                atol=1e-2,
                rtol=1e-2,
            )
          else:
            self.assert_close(
                golden_result=grad_in_cpu,
                torch_tpu_result=grad_in_tpu.cpu(),
            )

  def test_upsample_bicubic2d_aa_precision_and_large_dim_fallback(self):
    tpu_device = torch.device("tpu")
    x_cpu = torch.randn(2, 3, 8, 8, dtype=torch.float32)
    x_tpu = x_cpu.to(tpu_device)
    out_cpu = torch.ops.aten._upsample_bicubic2d_aa(
        x_cpu, [16, 16], False, None, None
    )

    out_default_tpu = torch.ops.aten._upsample_bicubic2d_aa(
        x_tpu, [16, 16], False, None, None
    )
    self.assert_close(
        golden_result=out_cpu, torch_tpu_result=out_default_tpu.cpu()
    )

    for prec in (
        torch.tpu.Precision.DEFAULT,
        torch.tpu.Precision.HIGH,
        torch.tpu.Precision.HIGHEST,
    ):
      with torch.tpu.precision(prec):
        out_tpu = torch.ops.aten._upsample_bicubic2d_aa(
            x_tpu, [16, 16], False, None, None
        )
      tol = 2e-2 if prec == torch.tpu.Precision.DEFAULT else 1e-4
      self.assert_close(
          golden_result=out_cpu,
          torch_tpu_result=out_tpu.cpu(),
          atol=tol,
          rtol=tol,
      )

    # Spatial dimension > 4096 exercises the 1D Gather fallback path.
    large_cpu = torch.randn(1, 1, 2, 4097, dtype=torch.float32)
    large_tpu = large_cpu.to(tpu_device)
    large_out_cpu = torch.ops.aten._upsample_bicubic2d_aa(
        large_cpu, [4, 2048], False, None, None
    )
    large_out_tpu = torch.ops.aten._upsample_bicubic2d_aa(
        large_tpu, [4, 2048], False, None, None
    )
    self.assert_close(
        golden_result=large_out_cpu, torch_tpu_result=large_out_tpu.cpu()
    )

    # Identity fast-path (in_h == out_h and in_w == out_w).
    for align_corners, scales_h, scales_w in (
        (False, None, None),
        (True, None, None),
        (False, 1.0, 1.0),
    ):
      id_cpu = torch.ops.aten._upsample_bicubic2d_aa(
          x_cpu, [8, 8], align_corners, scales_h, scales_w
      )
      id_tpu = torch.ops.aten._upsample_bicubic2d_aa(
          x_tpu, [8, 8], align_corners, scales_h, scales_w
      )
      self.assert_close(golden_result=id_cpu, torch_tpu_result=id_tpu.cpu())

    # .out variant: mismatched shape resizing and pre-allocated buffer writing.
    for init_shape in ((0,), (2, 3, 2, 2), (2, 3, 16, 16)):
      out_buf_cpu = torch.empty(init_shape, dtype=torch.float32)
      out_buf_tpu = torch.empty(
          init_shape, dtype=torch.float32, device=tpu_device
      )
      ret_cpu = torch.ops.aten._upsample_bicubic2d_aa.out(
          x_cpu, [16, 16], False, None, None, out=out_buf_cpu
      )
      ret_tpu = torch.ops.aten._upsample_bicubic2d_aa.out(
          x_tpu, [16, 16], False, None, None, out=out_buf_tpu
      )
      self.assertEqual(out_buf_tpu.shape, torch.Size([2, 3, 16, 16]))
      self.assertIs(ret_tpu, out_buf_tpu)
      self.assert_close(
          golden_result=ret_cpu, torch_tpu_result=out_buf_tpu.cpu()
      )

    # Downsampling vs. upsampling with align_corners=False/True and explicit
    # scales_h/scales_w.
    for out_size, explicit_scales in (
        ([4, 5], (0.5, 0.625)),
        ([13, 19], (1.625, 2.375)),
    ):
      for align_corners, scales_h, scales_w in (
          (False, None, None),
          (True, None, None),
          (False, explicit_scales[0], explicit_scales[1]),
      ):
        res_cpu = torch.ops.aten._upsample_bicubic2d_aa(
            x_cpu, out_size, align_corners, scales_h, scales_w
        )
        res_tpu = torch.ops.aten._upsample_bicubic2d_aa(
            x_tpu, out_size, align_corners, scales_h, scales_w
        )
        self.assert_close(golden_result=res_cpu, torch_tpu_result=res_tpu.cpu())

  def test_upsample_bilinear2d_aa_precision_and_large_dim_fallback(self):
    tpu_device = torch.device("tpu")
    x_cpu = torch.randn(2, 3, 8, 8, dtype=torch.float32)
    x_tpu = x_cpu.to(tpu_device)
    out_cpu = torch.ops.aten._upsample_bilinear2d_aa(
        x_cpu, [16, 16], False, None, None
    )

    out_default_tpu = torch.ops.aten._upsample_bilinear2d_aa(
        x_tpu, [16, 16], False, None, None
    )
    self.assert_close(
        golden_result=out_cpu, torch_tpu_result=out_default_tpu.cpu()
    )

    # Downsampling with align_corners=False, align_corners=True, and explicit scales.
    down_cpu = torch.randn(2, 3, 16, 16, dtype=torch.float32)
    down_tpu = down_cpu.to(tpu_device)
    for align_corners, scales_h, scales_w in (
        (False, None, None),
        (True, None, None),
        (False, 7.0 / 16.0, 7.0 / 16.0),
    ):
      down_out_cpu = torch.ops.aten._upsample_bilinear2d_aa(
          down_cpu, [7, 7], align_corners, scales_h, scales_w
      )
      down_out_tpu = torch.ops.aten._upsample_bilinear2d_aa(
          down_tpu, [7, 7], align_corners, scales_h, scales_w
      )
      self.assert_close(
          golden_result=down_out_cpu, torch_tpu_result=down_out_tpu.cpu()
      )

    # Identity fast-path (input spatial size == output spatial size).
    id_out_cpu = torch.ops.aten._upsample_bilinear2d_aa(
        x_cpu, [8, 8], False, None, None
    )
    id_out_tpu = torch.ops.aten._upsample_bilinear2d_aa(
        x_tpu, [8, 8], False, None, None
    )
    self.assert_close(
        golden_result=id_out_cpu, torch_tpu_result=id_out_tpu.cpu()
    )

    for prec in (
        torch.tpu.Precision.DEFAULT,
        torch.tpu.Precision.HIGH,
        torch.tpu.Precision.HIGHEST,
    ):
      with torch.tpu.precision(prec):
        out_tpu = torch.ops.aten._upsample_bilinear2d_aa(
            x_tpu, [16, 16], False, None, None
        )
      tol = 2e-2 if prec == torch.tpu.Precision.DEFAULT else 1e-4
      self.assert_close(
          golden_result=out_cpu,
          torch_tpu_result=out_tpu.cpu(),
          atol=tol,
          rtol=tol,
      )

    # Spatial dimension > 4096 exercises the 1D Gather fallback path.
    for shape, out_size in (
        ((1, 1, 2, 4097), [4, 2048]),
        ((1, 1, 4097, 2), [2048, 4]),
    ):
      large_cpu = torch.randn(*shape, dtype=torch.float32)
      large_tpu = large_cpu.to(tpu_device)
      large_out_cpu = torch.ops.aten._upsample_bilinear2d_aa(
          large_cpu, out_size, False, None, None
      )
      large_out_tpu = torch.ops.aten._upsample_bilinear2d_aa(
          large_tpu, out_size, False, None, None
      )
      self.assert_close(
          golden_result=large_out_cpu, torch_tpu_result=large_out_tpu.cpu()
      )

    # Adversarial spatial shapes: 1x1 inputs/outputs, prime dimensions,
    # extreme downsampling (support > input size), and asymmetric H/W scaling.
    for in_shape, out_size in (
        ((1, 2, 1, 1), [1, 1]),
        ((1, 2, 1, 1), [4, 4]),
        ((1, 2, 4, 4), [1, 1]),
        ((2, 3, 13, 17), [7, 11]),
        ((2, 3, 7, 11), [13, 17]),
        ((1, 2, 64, 64), [2, 2]),
        ((1, 2, 8, 32), [24, 6]),
    ):
      adv_cpu = torch.randn(*in_shape, dtype=torch.float32)
      adv_tpu = adv_cpu.to(tpu_device)
      for align_corners in (False, True):
        adv_out_cpu = torch.ops.aten._upsample_bilinear2d_aa(
            adv_cpu, out_size, align_corners, None, None
        )
        adv_out_tpu = torch.ops.aten._upsample_bilinear2d_aa(
            adv_tpu, out_size, align_corners, None, None
        )
        self.assert_close(
            golden_result=adv_out_cpu, torch_tpu_result=adv_out_tpu.cpu()
        )

    # Zero-batch empty tensor.
    empty_cpu = torch.empty(0, 3, 8, 8, dtype=torch.float32)
    empty_tpu = empty_cpu.to(tpu_device)
    empty_out_cpu = torch.ops.aten._upsample_bilinear2d_aa(
        empty_cpu, [4, 4], False, None, None
    )
    empty_out_tpu = torch.ops.aten._upsample_bilinear2d_aa(
        empty_tpu, [4, 4], False, None, None
    )
    self.assertEqual(empty_out_tpu.shape, empty_out_cpu.shape)
    self.assert_close(
        golden_result=empty_out_cpu, torch_tpu_result=empty_out_tpu.cpu()
    )

    # Non-contiguous channels-last input.
    cl_cpu = torch.randn(2, 3, 12, 10, dtype=torch.float32).to(
        memory_format=torch.channels_last
    )
    cl_tpu = cl_cpu.to(tpu_device)
    cl_out_cpu = torch.ops.aten._upsample_bilinear2d_aa(
        cl_cpu, [6, 15], False, None, None
    )
    cl_out_tpu = torch.ops.aten._upsample_bilinear2d_aa(
        cl_tpu, [6, 15], False, None, None
    )
    self.assert_close(
        golden_result=cl_out_cpu, torch_tpu_result=cl_out_tpu.cpu()
    )

    # Pre-allocated out= tensor with mismatched initial shape resized in-place.
    prealloc_cpu = torch.empty(0, dtype=torch.float32)
    prealloc_tpu = torch.empty(0, dtype=torch.float32, device=tpu_device)
    torch.ops.aten._upsample_bilinear2d_aa.out(
        down_cpu, [7, 7], False, None, None, out=prealloc_cpu
    )
    torch.ops.aten._upsample_bilinear2d_aa.out(
        down_tpu, [7, 7], False, None, None, out=prealloc_tpu
    )
    self.assertEqual(prealloc_tpu.shape, prealloc_cpu.shape)
    self.assert_close(
        golden_result=prealloc_cpu, torch_tpu_result=prealloc_tpu.cpu()
    )

    # Dtypes: bfloat16, float16, float32, float64.
    for dtype, tol in (
        (torch.bfloat16, 2e-2),
        (torch.float16, 1e-2),
        (torch.float32, 1e-4),
        (torch.float64, 1e-5),
    ):
      dt_cpu = torch.randn(1, 2, 10, 12, dtype=dtype)
      dt_tpu = dt_cpu.to(tpu_device)
      cpu_in = (
          dt_cpu.float() if dtype in (torch.bfloat16, torch.float16) else dt_cpu
      )
      dt_out_cpu = torch.ops.aten._upsample_bilinear2d_aa(
          cpu_in, [5, 18], False, None, None
      ).to(dtype)
      dt_out_tpu = torch.ops.aten._upsample_bilinear2d_aa(
          dt_tpu, [5, 18], False, None, None
      )
      self.assertEqual(dt_out_tpu.dtype, dtype)
      self.assert_close(
          golden_result=dt_out_cpu,
          torch_tpu_result=dt_out_tpu.cpu(),
          atol=tol,
          rtol=tol,
      )

  def test_bitwise_right_shift(self):
    tpu_device = torch.device("tpu")
    shift_by_int32 = torch.tensor([1, 2, 4, 5], dtype=torch.int32)
    shift_by_int32_tpu = shift_by_int32.to(tpu_device)
    shift_by_int64 = torch.tensor([1, 2, 4, 5], dtype=torch.int64)
    shift_by_int64_tpu = shift_by_int64.to(tpu_device)

    # Basic scalar tensor test.
    tpu_result = torch.bitwise_right_shift(128, shift_by_int32_tpu).cpu()
    golden_result = torch.bitwise_right_shift(128, shift_by_int32)
    self.assert_close(golden_result=golden_result, torch_tpu_result=tpu_result)

    tpu_result = torch.bitwise_right_shift(128, shift_by_int64_tpu).cpu()
    golden_result = torch.bitwise_right_shift(128, shift_by_int64)
    self.assert_close(golden_result=golden_result, torch_tpu_result=tpu_result)

    # Test with corner case of not shifting beyond 0.
    tpu_result = torch.bitwise_right_shift(1, shift_by_int32_tpu).cpu()
    golden_result = torch.bitwise_right_shift(1, shift_by_int32)
    self.assert_close(golden_result=golden_result, torch_tpu_result=tpu_result)

    # Test with int64 input and shift by int32 tensor.
    input_long = torch.tensor([37, 128, 200, 511], dtype=torch.int64)
    input_long_tpu = input_long.to(tpu_device)
    tpu_result = torch.bitwise_right_shift(
        input_long_tpu, shift_by_int32_tpu
    ).cpu()
    golden_result = torch.bitwise_right_shift(input_long, shift_by_int32)
    self.assert_close(golden_result=golden_result, torch_tpu_result=tpu_result)

    # Test with int32 input and shift by int64 tensor.
    input_int32 = torch.tensor([37, 128, 200, 511], dtype=torch.int32)
    input_int32_tpu = input_int32.to(tpu_device)
    tpu_result = torch.bitwise_right_shift(
        input_int32_tpu, shift_by_int64_tpu
    ).cpu()
    golden_result = torch.bitwise_right_shift(input_int32, shift_by_int64)
    self.assert_close(golden_result=golden_result, torch_tpu_result=tpu_result)

    # Basic tensor scalar test.
    tpu_result = torch.bitwise_right_shift(input_long_tpu, 1).cpu()
    golden_result = torch.bitwise_right_shift(input_long, 1)
    self.assert_close(golden_result=golden_result, torch_tpu_result=tpu_result)

    # Test bitwise shift with mixed bool and integer inputs (b/478955517).
    int_a = torch.tensor([8, 16, 32], dtype=torch.int32)
    bool_b = torch.tensor([True, False, True], dtype=torch.bool)
    tpu_res = torch.bitwise_right_shift(
        int_a.to(tpu_device), bool_b.to(tpu_device)
    ).cpu()
    golden_res = torch.bitwise_right_shift(int_a, bool_b)
    self.assert_close(golden_result=golden_res, torch_tpu_result=tpu_res)

    tpu_res = torch.bitwise_right_shift(
        bool_b.to(tpu_device), int_a.to(tpu_device)
    ).cpu()
    golden_res = torch.bitwise_right_shift(bool_b, int_a)
    self.assert_close(golden_result=golden_res, torch_tpu_result=tpu_res)

    int64_a = torch.tensor([8, 16, 32], dtype=torch.int64)
    tpu_res = (int64_a.to(tpu_device) >> bool_b.to(tpu_device)).cpu()
    golden_res = int64_a >> bool_b
    self.assert_close(golden_result=golden_res, torch_tpu_result=tpu_res)

    tpu_res = (bool_b.to(tpu_device) >> int64_a.to(tpu_device)).cpu()
    golden_res = bool_b >> int64_a
    self.assert_close(golden_result=golden_res, torch_tpu_result=tpu_res)

  def test_bitwise_or(self):
    # tensor-tensor
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_or(
            torch.tensor([1, 2, 3], dtype=torch.int32).to(device),
            torch.tensor([3, 1, 5], dtype=torch.int32).to(device),
        )
    )
    # tensor-scalar
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_or(
            torch.tensor([True, False, True], dtype=torch.bool).to(device),
            False,
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_or(
            torch.tensor([1, 2, 3], dtype=torch.int32).to(device),
            2,
        )
    )
    # scalar-tensor
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_or(
            True,
            torch.tensor([1, 0, 1], dtype=torch.bool).to(device),
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_or(
            10,
            torch.tensor([1, 2, 3], dtype=torch.int64).to(device),
        )
    )
    # out param
    out_cpu = torch.empty(3, dtype=torch.int32)
    out_tpu = torch.empty(3, dtype=torch.int32, device=torch.device("tpu"))
    a = torch.tensor([1, 2, 3], dtype=torch.int32)
    b = torch.tensor([3, 1, 5], dtype=torch.int32)
    torch.bitwise_or(a, b, out=out_cpu)
    torch.bitwise_or(
        a.to(torch.device("tpu")),
        b.to(torch.device("tpu")),
        out=out_tpu,
    )
    self.assertEqual(out_cpu, out_tpu.cpu())

    # empty input
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_or(
            torch.tensor([], dtype=torch.int32).to(device),
            torch.tensor([], dtype=torch.int32).to(device),
        )
    )
    # mismatched dtypes
    # int32 and int64
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_or(
            torch.tensor([1, 2, 3], dtype=torch.int32).to(device),
            torch.tensor([3, 1, 5], dtype=torch.int64).to(device),
        )
    )
    # bool and int32
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_or(
            torch.tensor([True, False, True], dtype=torch.bool).to(device),
            torch.tensor([0, 1, 1], dtype=torch.int32).to(device),
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_or(
            torch.tensor([0, 1, 1], dtype=torch.int32).to(device),
            torch.tensor([True, False, True], dtype=torch.bool).to(device),
        )
    )

  def test_bitwise_xor(self):
    # tensor-tensor
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_xor(
            torch.tensor([1, 2, 3], dtype=torch.int32).to(device),
            torch.tensor([3, 1, 5], dtype=torch.int32).to(device),
        )
    )
    # tensor-scalar
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_xor(
            torch.tensor([True, False, True], dtype=torch.bool).to(device),
            False,
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_xor(
            torch.tensor([1, 2, 3], dtype=torch.int32).to(device),
            2,
        )
    )
    # scalar-tensor
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_xor(
            True,
            torch.tensor([1, 0, 1], dtype=torch.bool).to(device),
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_xor(
            10,
            torch.tensor([1, 2, 3], dtype=torch.int64).to(device),
        )
    )
    # out param
    out_cpu = torch.empty(3, dtype=torch.int32)
    out_tpu = torch.empty(3, dtype=torch.int32, device=torch.device("tpu"))
    a = torch.tensor([1, 2, 3], dtype=torch.int32)
    b = torch.tensor([3, 1, 5], dtype=torch.int32)
    torch.bitwise_xor(a, b, out=out_cpu)
    torch.bitwise_xor(
        a.to(torch.device("tpu")),
        b.to(torch.device("tpu")),
        out=out_tpu,
    )
    self.assertEqual(out_cpu, out_tpu.cpu())

    # empty input
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_xor(
            torch.tensor([], dtype=torch.int32).to(device),
            torch.tensor([], dtype=torch.int32).to(device),
        )
    )
    # mismatched dtypes
    # int32 and int64
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_xor(
            torch.tensor([1, 2, 3], dtype=torch.int32).to(device),
            torch.tensor([3, 1, 5], dtype=torch.int64).to(device),
        )
    )
    # bool and int32
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_xor(
            torch.tensor([True, False, True], dtype=torch.bool).to(device),
            torch.tensor([0, 1, 1], dtype=torch.int32).to(device),
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.bitwise_xor(
            torch.tensor([0, 1, 1], dtype=torch.int32).to(device),
            torch.tensor([True, False, True], dtype=torch.bool).to(device),
        )
    )

  def test_cat(self):
    # Additional test cases to ensure coverage.
    t0 = torch.tensor([], dtype=torch.float32).to("tpu")
    t3 = torch.tensor([1, 2, 3], dtype=torch.int32).to("tpu")
    t1x3 = torch.tensor([[4, 5, 6]], dtype=torch.float16).to("tpu")
    t1x3_f64 = torch.tensor([[7, 8, 9]], dtype=torch.float64).to("tpu")
    t2x2 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int16).to("tpu")
    t2x3 = torch.tensor([[4, 5, 6], [7, 8, 9]], dtype=torch.int16).to("tpu")
    t3x2 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int16).to("tpu")
    t2x0 = torch.empty(2, 0, dtype=torch.int16).to("tpu")
    t2x0x3 = torch.empty(2, 0, 3, dtype=torch.int16).to("tpu")
    t1x0x3 = torch.empty(1, 0, 3, dtype=torch.int16).to("tpu")

    # All tensors being concatenated are 1D with size (0,).
    self.assert_close(golden_result=t0, torch_tpu_result=torch.cat([t0]))
    self.assert_close(golden_result=t0, torch_tpu_result=torch.cat([t0, t0]))

    # Concatenating t0 with other 1D tensors.
    self.assert_close(
        golden_result=torch.tensor([1.0, 2.0, 3.0], device="tpu"),
        torch_tpu_result=torch.cat([t0, t3]),
    )
    self.assert_close(
        golden_result=torch.tensor([1.0, 2.0, 3.0], device="tpu"),
        torch_tpu_result=torch.cat([t3, t0]),
    )
    self.assert_close(
        golden_result=t1x3.to(dtype=torch.float32),
        torch_tpu_result=torch.cat([t0, t1x3], dim=1),
    )

    # Concatenating t1x3 with 2D tensors.
    self.assert_close(
        golden_result=t1x3.to(dtype=torch.float32),
        torch_tpu_result=torch.cat([t1x3, t0]),
    )
    self.assert_close(
        golden_result=t1x3_f64, torch_tpu_result=torch.cat([t1x3_f64, t0])
    )

    # Concatenating same-shaped tensors with default dim (=0).
    self.assert_close(
        golden_result=torch.tensor(
            [1, 2, 3, 1, 2, 3], dtype=torch.int32, device="tpu"
        ),
        torch_tpu_result=torch.cat([t3, t3]),
    )
    self.assert_close(
        golden_result=torch.tensor(
            [[7.0, 8.0, 9.0], [4.0, 5.0, 6.0]],
            dtype=torch.float64,
            device="tpu",
        ),
        torch_tpu_result=torch.cat([t1x3_f64, t1x3]),
    )

    # Concatenating different-shaped tensors.
    self.assert_close(
        golden_result=torch.tensor(
            [[1, 2], [3, 4], [1, 2], [3, 4], [5, 6]],
            dtype=torch.int16,
            device="tpu",
        ),
        torch_tpu_result=torch.cat([t2x2, t3x2]),
    )
    self.assert_close(
        golden_result=torch.tensor(
            [[1, 2, 4, 5, 6], [3, 4, 7, 8, 9]],
            dtype=torch.int16,
            device="tpu",
        ),
        torch_tpu_result=torch.cat([t2x2, t2x3], dim=1),
    )
    self.assert_close(
        golden_result=torch.tensor(
            [[1, 2, 4, 5, 6], [3, 4, 7, 8, 9]],
            dtype=torch.int16,
            device="tpu",
        ),
        torch_tpu_result=torch.cat([t2x2, t2x3], dim=-1),
    )

    # Concatenating tensors with a 0-sized dimension.
    self.assert_close(
        golden_result=torch.empty(4, 0, dtype=torch.int16).to("tpu"),
        torch_tpu_result=torch.cat([t2x0, t2x0]),
    )
    self.assert_close(
        golden_result=t2x2, torch_tpu_result=torch.cat([t2x0, t2x2], 1)
    )
    self.assert_close(
        golden_result=t1x0x3.to(dtype=torch.float32),
        torch_tpu_result=torch.cat([t0, t1x0x3]),
    )
    self.assert_close(
        golden_result=torch.empty(3, 0, 3, dtype=torch.int16).to("tpu"),
        torch_tpu_result=torch.cat([t1x0x3, t2x0x3]),
    )

  def test_cat_out_invalid_cast(self):
    """Tests that an invalid cast during torch.cat throws an error."""

    def test_fn(device):
      t_f32 = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
      out_int32 = torch.zeros(2, dtype=torch.int32, device=device)
      return torch.cat([t_f32], out=out_int32)

    self.assert_close_tpu_vs_cpu(
        test_fn,
        check_exception_type=False,
        allow_failure=True,
    )

  def test_cat_out_mismatched_dtypes(self):
    t_f32 = torch.tensor([1.0, 2.0], dtype=torch.float32)
    t_f16 = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)

    def test_fn(device):
      out_f16 = torch.zeros(4, dtype=torch.bfloat16, device=device)
      torch.cat([t_f32.to(device), t_f16.to(device)], out=out_f16)

      out_f32 = torch.zeros(4, dtype=torch.float32, device=device)
      torch.cat([t_f32.to(device), t_f16.to(device)], out=out_f32)
      return out_f16, out_f32

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_cat_out_resized(self):
    t1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    t2 = torch.tensor([[5, 6]], dtype=torch.float32)

    def fn(device):
      # Expected output shape is (3, 2)
      # out is initialized with a different shape (1, 1)
      out = torch.zeros((1, 1), dtype=torch.float32, device=device)
      torch.cat([t1.to(device), t2.to(device)], dim=0, out=out)
      return out

    self.assert_close_tpu_vs_cpu(fn)

  def test_stack(self):
    def fn(device):
      s1 = torch.tensor(1.0, dtype=torch.float32, device=device)
      s2 = torch.tensor(2, dtype=torch.int32, device=device)
      t1 = torch.tensor(
          [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device
      )
      t2 = torch.tensor(
          [[5.0, 6.0], [7.0, 8.0]], dtype=torch.bfloat16, device=device
      )
      t_empty = torch.empty(2, 0, 3, dtype=torch.float32, device=device)
      out_empty = torch.zeros(1, dtype=torch.float32, device=device)
      torch.stack([t_empty, t_empty], dim=1, out=out_empty)

      out_scalars = torch.zeros(2, dtype=torch.float32, device=device)
      torch.stack([s1, s2], dim=0, out=out_scalars)

      out_single = torch.zeros((2, 1, 2), dtype=torch.float32, device=device)
      torch.stack([t1], dim=1, out=out_single)

      out_neg_dim = torch.zeros((2, 2, 2), dtype=torch.float32, device=device)
      torch.stack([t1, t2], dim=-1, out=out_neg_dim)

      out_matching = torch.zeros((2, 2, 2), dtype=torch.float32, device=device)
      torch.stack([t1, t2], dim=0, out=out_matching)

      t1_view = t1.t()
      t2_view = t2.t()
      out_view_base = torch.zeros((2, 4, 2), dtype=torch.float32, device=device)
      out_view = out_view_base[:, ::2, :]
      torch.stack([t1_view, t2_view], dim=1, out=out_view)

      return (
          torch.stack([s1]),
          torch.stack([s1, s2], dim=0),
          torch.stack([s1, s2], dim=-1),
          torch.stack([t1], dim=1),
          torch.stack([t1, t2], dim=0),
          torch.stack([t1, t2], dim=1),
          torch.stack([t1, t2], dim=2),
          torch.stack([t1, t2], dim=-1),
          torch.stack([t1_view, t2_view], dim=1),
          torch.stack([t_empty, t_empty], dim=0),
          out_empty,
          out_scalars,
          out_single,
          out_neg_dim,
          out_matching,
          out_view_base,
      )

    self.assert_close_tpu_vs_cpu(fn)

  def test_stack_single_dispatch(self):
    tensors = [
        torch.tensor(float(i), dtype=torch.float32, device="tpu")
        for i in range(16)
    ]
    torch.accelerator.synchronize()
    start_reqs = torch.tpu._get_cache_stats().num_cache_reqs
    actual = torch.stack(tensors).cpu()
    reqs = torch.tpu._get_cache_stats().num_cache_reqs - start_reqs
    self.assertEqual(reqs, 1)
    utils.assert_close(actual, torch.arange(16, dtype=torch.float32))

  def test_stack_out_mismatched_dtypes(self):
    t_f32 = torch.tensor([1.0, 2.0], dtype=torch.float32)
    t_bf16 = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)

    def test_fn(device):
      out_bf16 = torch.zeros((2, 2), dtype=torch.bfloat16, device=device)
      torch.stack([t_f32.to(device), t_bf16.to(device)], out=out_bf16)

      out_f32 = torch.zeros((2, 2), dtype=torch.float32, device=device)
      torch.stack([t_f32.to(device), t_bf16.to(device)], out=out_f32)
      return out_bf16, out_f32

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_stack_out_resized(self):
    t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    t2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)

    def fn(device):
      out = torch.zeros((1, 1), dtype=torch.float32, device=device)
      torch.stack([t1.to(device), t2.to(device)], dim=1, out=out)
      return out

    self.assert_close_tpu_vs_cpu(fn)

  @parameterized.product(
      dtype=[torch.bfloat16, torch.float32], world_size=[1, 2, 8]
  )
  def test_split_with_sizes_copy_fsdp2_copy_out(self, dtype, world_size):
    """Tests split_with_sizes_copy as FSDP2 (fully_shard) calls it.

    After every all-gather, FSDP2 views the flat buffer as (world_size, N) and
    copies its columns into (world_size, split_size) views of flat
    per-parameter buffers.

    Args:
      dtype: The dtype of the buffers.
      world_size: The number of rows of the all-gather buffer.
    """
    split_sizes = [512, 128, 1, 0, 130]
    gathered = torch.randn(world_size * sum(split_sizes)).to(dtype)

    def fn(device):
      out = [
          torch.empty(world_size * size, dtype=dtype, device=device).view(
              world_size, size
          )
          for size in split_sizes
      ]
      torch.split_with_sizes_copy(
          gathered.to(device).view(world_size, -1), split_sizes, dim=1, out=out
      )
      return tuple(out)

    self.assert_close_tpu_vs_cpu(fn, check_value=CheckValueMode.STRICT)

  def test_split_with_sizes_copy_fsdp2_ops_do_not_grow_with_world_size(self):
    """FSDP2's copy-out must not record more ATen ops for more ranks."""
    split_sizes = [512, 128, 1, 130]

    def count_aten_ops(world_size):
      gathered = torch.randn(world_size * sum(split_sizes), device="tpu")
      out = [
          torch.empty(world_size * size, device="tpu").view(world_size, size)
          for size in split_sizes
      ]
      with torch.profiler.profile(
          activities=[torch.profiler.ProfilerActivity.CPU]
      ) as prof:
        torch.split_with_sizes_copy(
            gathered.view(world_size, -1), split_sizes, dim=1, out=out
        )
      return sum(
          event.count
          for event in prof.key_averages()
          if event.key.startswith("aten::")
      )

    self.assertEqual(count_aten_ops(2), count_aten_ops(16))

  def test_split_with_sizes_copy_dim1_split_sizes_mismatch(self):
    def fn(device):
      out = [torch.empty(2, 2, device=device), torch.empty(2, 3, device=device)]
      torch.split_with_sizes_copy(
          torch.zeros(2, 6, device=device), [2, 3], dim=1, out=out
      )
      return tuple(out)

    self.assert_close_tpu_vs_cpu(fn, allow_failure=True)

  def test_cumprod_default_dtype(self):
    """Tests torch.cumprod with default dtype (None)."""
    x = torch.randn(2, 3, dtype=torch.float32)
    self.assert_close_tpu_vs_cpu(
        lambda device, x=x: torch.cumprod(x.to(device), dim=0)
    )

  def test_arange_start_step_float32(self):
    golden_result = torch.arange(
        0, 10, 2, dtype=torch.float32, device=self.golden_device
    )

    tpu_result = torch.arange(
        0,
        10,
        2,
        dtype=torch.float32,
        device=torch.device("tpu"),
    ).cpu()
    self.assert_close(golden_result=golden_result, torch_tpu_result=tpu_result)

  def test_atan2(self):
    x = torch.tensor([2], dtype=torch.int32)
    y = torch.tensor([3], dtype=torch.float32)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.atan2(x.to(device=device), y.to(device=device))
    )

    x = torch.tensor([2], dtype=torch.int32)
    y = torch.tensor([3], dtype=torch.int32)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.atan2(x.to(device=device), y.to(device=device))
    )

    x = torch.tensor([False])
    y = torch.tensor([3], dtype=torch.int32)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.atan2(x.to(device=device), y.to(device=device))
    )

    x = torch.tensor([2.0])
    y = torch.tensor([True])
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.atan2(x.to(device=device), y.to(device=device))
    )

    x = torch.tensor([True])
    y = torch.tensor([False])
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.atan2(x.to(device=device), y.to(device=device))
    )

  def test_atan2_dtype_broadcasts(self):
    x = torch.tensor([1.0, 2.0], dtype=torch.int32, device=self.golden_device)
    y = torch.tensor([1.0, 2.0], dtype=torch.float32, device=self.golden_device)
    x_tpu = x.to("tpu")
    y_tpu = y.to("tpu")
    golden_result = torch.atan2(x, y)
    tpu_result = torch.atan2(x_tpu, y_tpu)
    self.assert_close(
        golden_result=golden_result, torch_tpu_result=tpu_result.cpu()
    )

    golden_result = torch.atan2(y, x)
    tpu_result = torch.atan2(y_tpu, x_tpu)
    self.assert_close(
        golden_result=golden_result, torch_tpu_result=tpu_result.cpu()
    )

  def test_clamp_manual(self):
    x = torch.tensor([[0, -1], [-2, -3]])
    min_ = torch.tensor([0, 1])
    max_ = torch.tensor([2, 3])

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.clamp(
            x.to(device=device),
            min=min_.to(device=device),
            max=max_.to(device=device),
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.clamp(
            x.to(device=device, dtype=torch.float32),
            min=min_.to(device=device, dtype=torch.long),
            max=max_.to(device=device, dtype=torch.short),
        )
    )

    y = x.clone()
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.clamp_(
            y.to(device=device, dtype=torch.int8),
            min=min_.to(device=device, dtype=torch.int16),
            max=max_.to(device=device, dtype=torch.int32),
        )
    )

    try:
      a = torch.rand([2, 2]).to("tpu")
      b = torch.rand([2, 3]).to("tpu")
      c = torch.rand([3, 2]).to("tpu")
      torch.clamp(a, b, c)
      assert (
          False
      ), "This test should fail because input shapes are not compatible"
    except RuntimeError:
      # Expected exception.
      pass

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.clamp(
            x.to(dtype=torch.float32, device=device), min=1, max=2
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.clamp(
            x.to(dtype=torch.bfloat16, device=device), min=1
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.clamp(
            x.to(dtype=torch.float64, device=device), max=2
        )
    )

    # Test multi-way tensor broadcasting where min and max have different
    # shapes.
    input_t = torch.randn(1, 3)
    min_t = torch.tensor([[0.0]])
    max_t = torch.tensor([[1.0], [1.0]])
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.clamp(
            input_t.to(device=device),
            min=min_t.to(device=device),
            max=max_t.to(device=device),
        )
    )

  def test_clamp_out(self):
    """Tests the out-variant clamp.out with scalar and tensor bounds."""

    # Scalar bounds (same dtype)
    def test_scalar_out(device):
      x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
      out = torch.empty_like(x)
      torch.clamp(x, 1.5, 2.5, out=out)
      return out

    self.assert_close_tpu_vs_cpu(test_scalar_out)

    # Tensor bounds with promotional upcast (float32 -> float64)
    def test_tensor_upcast_out(device):
      x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
      min_t = torch.tensor([1.5, 1.5, 1.5], dtype=torch.float32, device=device)
      max_t = torch.tensor([2.5, 2.5, 2.5], dtype=torch.float32, device=device)
      out = torch.empty(3, dtype=torch.float64, device=device)
      torch.clamp(x, min_t, max_t, out=out)
      return out

    self.assert_close_tpu_vs_cpu(test_tensor_upcast_out)

  def test_clamp_min_out(self):
    """Tests the out-variant clamp_min.out with scalar and tensor bounds."""

    # Scalar bound (same dtype)
    def test_scalar_out(device):
      x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
      out = torch.empty_like(x)
      torch.clamp_min(x, 2.0, out=out)
      return out

    self.assert_close_tpu_vs_cpu(test_scalar_out)

    # Tensor bound with promotional upcast (float32 -> float64)
    def test_tensor_upcast_out(device):
      x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
      min_t = torch.tensor([2.0, 1.5, 0.5], dtype=torch.float32, device=device)
      out = torch.empty(3, dtype=torch.float64, device=device)
      torch.clamp_min(x, min_t, out=out)
      return out

    self.assert_close_tpu_vs_cpu(test_tensor_upcast_out)

  def test_clamp_max_out(self):
    """Tests the out-variant clamp_max.out with scalar and tensor bounds."""

    # Scalar bound (same dtype)
    def test_scalar_out(device):
      x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
      out = torch.empty_like(x)
      torch.clamp_max(x, 2.0, out=out)
      return out

    self.assert_close_tpu_vs_cpu(test_scalar_out)

    # Tensor bound with promotional upcast (float32 -> float64)
    def test_tensor_upcast_out(device):
      x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
      max_t = torch.tensor([2.0, 1.5, 0.5], dtype=torch.float32, device=device)
      out = torch.empty(3, dtype=torch.float64, device=device)
      torch.clamp_max(x, max_t, out=out)
      return out

    self.assert_close_tpu_vs_cpu(test_tensor_upcast_out)

  def test_tril_out(self):
    """Tests the out-variant tril.out with matching dtype."""

    def test_out(device):
      x = torch.tensor(
          [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device
      )
      out = torch.empty_like(x)
      torch.tril(x, diagonal=0, out=out)
      return out

    self.assert_close_tpu_vs_cpu(test_out)

  def test_triu_out(self):
    """Tests the out-variant triu.out with matching dtype."""

    def test_out(device):
      x = torch.tensor(
          [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device
      )
      out = torch.empty_like(x)
      torch.triu(x, diagonal=0, out=out)
      return out

    self.assert_close_tpu_vs_cpu(test_out)

  def test_concurrent_ops(self):
    """Tests that ops can run concurrently without correctness issues."""

    def run_op(op):
      arg = torch.tensor(2, dtype=torch.float32, device=torch.device("tpu"))
      res = op(arg)
      return res.to("cpu").item()

    # Start 100 threads to run log2 and exp ops concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
      log2_futures = [executor.submit(run_op, torch.log2) for _ in range(100)]
      exp_futures = [executor.submit(run_op, torch.exp) for _ in range(100)]
      for future in log2_futures:
        self.assertEqual(future.result(), 1)
      for future in exp_futures:
        self.assertEqual(future.result(), 7.389056)

  def test_concurrent_ops_deferred_to_different_thread(self):
    """Tests deferring an op to a different thread."""
    device = torch.device("tpu")
    lock = threading.Lock()
    num_tensors = 100
    tensors = [None for _ in range(num_tensors)]
    # Threads for producing elements of tensors.
    producers = [None for _ in range(num_tensors)]
    results = [None for _ in range(num_tensors)]
    # Threads for computing results from elements of tensors.
    consumers = [None for _ in range(num_tensors)]

    # Create the producer threads.

    def run_producer(i):
      with lock:
        tensors[i] = torch.zeros(1, dtype=torch.float32, device=device)

    for i in range(num_tensors):
      producers[i] = threading.Thread(
          target=run_producer, args=(i,), daemon=True
      )

    # Create the consumer threads.

    def run_consumer(i):
      done = False
      while not done:  # Wait for producer #i to finish setting tensors[i].
        with lock:
          t = tensors[i]
          if t is not None:
            done = True
            # The producer has set tensors[i] to zeros, so we can use it.
            results[i] = t.add(i).to("cpu").item()

        if not done:
          time.sleep(0)  # Yield to another thread.

    for i in range(num_tensors):
      consumers[i] = threading.Thread(
          target=run_consumer, args=(i,), daemon=True
      )

    # Start all threads.
    for i in range(num_tensors):
      producers[i].start()
      consumers[i].start()

    # Wait for all threads to finish.
    for i in range(num_tensors):
      producers[i].join()
      consumers[i].join()
      self.assertEqual(results[i], i)

  def test_dropout_mean_of_entries(self):
    n = 5000
    p = 0.5
    t = torch.rand(
        n,
        n,
        dtype=torch.float32,
        device=torch.device("tpu"),
    )
    t = torch.dropout(t, p, train=True)
    # If X = average of entries of t = (1/n^2) sum_{ij} U_ij * B_ij / (1 - p)
    #   U_ij ~ Uniform[0,1], B_ij ~ Bernoulli(1 - p)
    # then
    #   E(X) = 0.5
    #   V(X) = (1/n^2) *(1/ (1-p)^2) V(U_00 * B_00)
    #        = (3 + p) / 12 / (1 - p) / n / n
    mean_value = t.mean()
    expected_mean = 0.5
    pop_variance = (3 + p) / 12 / (1 - p) / n / n
    # P(|X - mean| >= atol) <= V(X) / atol^2 = pop_variance / atol^2
    # make atol big enough so above prob is <= 10^-6
    atol = torch.sqrt(torch.tensor(pop_variance)).item() * 1e3
    self.assert_close(
        golden_result=torch.tensor(expected_mean),
        torch_tpu_result=mean_value.to("cpu"),
        atol=atol,
    )

  def test_dropout_equal_to_zero_or_scaled_original(self):
    t = torch.rand(
        10,
        10,
        dtype=torch.float32,
        device=torch.device("tpu"),
    )
    z = torch.dropout(t, 0.5, train=True)
    mask = z != (2 * t)
    w = torch.masked_select(z, mask)
    zeros = torch.zeros_like(w)
    self.assert_close(
        golden_result=zeros.to("cpu"), torch_tpu_result=w.to("cpu")
    )

  def test_dropout_reproducible(self):
    t = torch.ones(
        10,
        10,
        dtype=torch.float32,
        device=torch.device("tpu"),
    )
    torch.manual_seed(1234)
    z = torch.dropout(t, 0.5, train=True)
    w = torch.dropout(t, 0.5, train=True)
    # with probability 1 - (1/2)^(100)
    self.assertNotEqual(z.to("cpu"), w.to("cpu"))
    torch.manual_seed(1234)
    z_again = torch.dropout(t, 0.5, train=True)
    self.assert_close(
        golden_result=z.to("cpu"), torch_tpu_result=z_again.to("cpu")
    )

  def test_exponential(self):
    """Tests the exponential_ op."""
    tpu_device = torch.device("tpu")
    shape = (10, 10)
    lambd = 0.5

    # Test determinism
    torch.manual_seed(123)
    t1 = torch.empty(shape, device=tpu_device).exponential_(lambd)

    torch.manual_seed(123)
    t2 = torch.empty(shape, device=tpu_device).exponential_(lambd)
    self.assert_close(golden_result=t1.cpu(), torch_tpu_result=t2.cpu())

    # Test strict positivity on (0, inf) and no -0.0
    self.assertGreater(t1.min().item(), 0)
    self.assertFalse(torch.signbit(t1).any().item())

  @parameterized.named_parameters(
      ("float32", torch.float32),
      ("bfloat16", torch.bfloat16),
      ("float16", torch.float16),
      ("float64", torch.float64),
  )
  def test_exponential_support_and_precision(self, dtype: torch.dtype) -> None:
    """Tests exponential_ strict (0, inf) support, acc_type precision, and edge cases."""
    tpu_device = torch.device("tpu")
    torch.manual_seed(42)

    num_elements = 100_000
    q = torch.empty(num_elements, device=tpu_device, dtype=dtype).exponential_(
        1.0
    )
    self.assertTrue(torch.all(q > 0).item())
    self.assertFalse(torch.signbit(q).any().item())

    if dtype == torch.bfloat16:
      # With f32 computation precision, bfloat16 draws must have far more than
      # 2^7 = 128 distinct values and reach past the 7 * ln(2) ~= 4.85 ceiling.
      self.assertGreater(q.unique().numel(), 128)
      self.assertGreater(q.max().item(), 5.0)

      rate = 1e30
      q_rate = torch.empty(
          100_000, device=tpu_device, dtype=dtype
      ).exponential_(rate)
      self.assertTrue(torch.all(q_rate > 0).item())
      self.assertFalse(torch.signbit(q_rate).any().item())

    if dtype == torch.float16:
      # Verify lambd >= 2.0 and large lambd (1e5 > f16 max 65504) do not
      # underflow to 0.0 on cast or overflow lambd to inf on the host.
      for rate in (2.0, 1e5):
        q_rate = torch.empty(
            100_000, device=tpu_device, dtype=dtype
        ).exponential_(rate)
        self.assertTrue(torch.all(q_rate > 0).item())
        self.assertFalse(torch.signbit(q_rate).any().item())

    # lambd = inf must still produce 0.0 (upstream PyTorch contract).
    q_inf = torch.empty(16, device=tpu_device, dtype=dtype).exponential_(
        float("inf")
    )
    self.assertTrue(torch.all(q_inf == 0).item())

    # Non-contiguous strided slice test
    base = torch.zeros((16, 32), device=tpu_device, dtype=dtype)
    strided_view = base[:, ::2]
    strided_view.exponential_(1.0)
    self.assertTrue(torch.all(strided_view > 0).item())
    self.assertFalse(torch.signbit(strided_view).any().item())
    self.assertTrue(torch.all(base[:, 1::2] == 0).item())

    # Downstream Gumbel-max / exponential race sampler: -log(q) and argmax(p / q)
    q_race = torch.empty(
        (4096, 2), device=tpu_device, dtype=dtype
    ).exponential_(1.0)
    self.assertTrue(torch.isfinite(-torch.log(q_race)).all().item())
    probs = torch.tensor([0.0, 1.0], device=tpu_device, dtype=dtype).expand(
        4096, 2
    )
    scores = probs / q_race
    self.assertFalse(torch.isnan(scores).any().item())
    sampled = torch.argmax(scores, dim=-1)
    self.assertTrue(torch.all(sampled == 1).item())

    # Generator offset advancement must use compute_dtype bitwidth (32 for
    # float16/bfloat16/float32, 64 for float64).
    gen = torch.Generator(device=tpu_device).manual_seed(1234)
    self.assertEqual(gen.get_offset(), 0)
    n_gen = 256
    torch.empty(n_gen, device=tpu_device, dtype=dtype).exponential_(
        1.0, generator=gen
    )
    compute_bits = 64 if dtype == torch.float64 else 32
    expected_offset = (n_gen * compute_bits + 127) // 128
    self.assertEqual(gen.get_offset(), expected_offset)

  def test_random_distribution(self):
    for dtype in [
        torch.int32,
        torch.int64,
        torch.uint32,
        torch.uint64,
    ]:
      n = 10000
      t = torch.zeros(n, n, dtype=dtype, device=torch.device("tpu"))
      t = t.random_(0, 16)
      # If X = avg of entries of t that are equal to zero, then
      #   E(X) = 1/16
      #   V(X) = (1/n^2) * V(B_00) = 15 / 256 / n / n
      mean_num_zeros = t.eq(0).sum() / n / n
      expected_mean = 1 / 16
      # Using Chebyshev's inequality:
      #   P(|X - mean| >= atol) <= V(X) / atol^2 = 15 / 256 / n^2 / atol^2
      # 1) To make this probability less than 10^-6 we need
      #   atol >= sqrt(15) * 10^3 / (16 * n)
      # 2) To make the test meaningful, we want atol < 1/16, so we need
      #   n >= sqrt(15) * 10^3 ~= 4000
      atol = (15**0.5) * 1e3 / (16 * n)  # ~ 0.024
      self.assert_close(
          golden_result=torch.tensor(expected_mean),
          torch_tpu_result=mean_num_zeros.to("cpu"),
          atol=atol,
      )

  def test_uniform_complex(self):
    def compute(device):
      z = torch.zeros(100, dtype=torch.complex64, device=device).uniform_(0, 10)
      with self.subTest("is_complex"):
        self.assertTrue(z.is_complex())

      with self.subTest("real_part_non_zero"):
        self.assertTrue((z.real != 0).any())

      with self.subTest("imag_part_non_zero"):
        self.assertTrue((z.imag != 0).any())

      with self.subTest("real_part_in_range"):
        self.assertTrue((z.real >= 0).all())
        self.assertTrue((z.real < 10).all())

      with self.subTest("imag_part_in_range"):
        self.assertTrue((z.imag >= 0).all())
        self.assertTrue((z.imag < 10).all())
      return z

    self.assert_close_tpu_vs_cpu(compute, check_value=CheckValueMode.SKIP)

  def test_random_reproducible(self):
    t = torch.zeros(
        10,
        10,
        dtype=torch.int32,
        device=torch.device("tpu"),
    )
    w = torch.zeros(
        10,
        10,
        dtype=torch.int32,
        device=torch.device("tpu"),
    )
    torch.manual_seed(1234)
    t = t.random_(10)
    w = w.random_(10)
    self.assertNotEqual(t.to("cpu"), w.to("cpu"))
    torch.manual_seed(1234)
    t_again = t.random_(10)
    self.assert_close(
        golden_result=t.to("cpu"), torch_tpu_result=t_again.to("cpu")
    )

  def test_random_reproducible_default_to(self):
    t = torch.zeros(
        10,
        10,
        dtype=torch.int32,
        device=torch.device("tpu"),
    )
    w = torch.zeros(
        10,
        10,
        dtype=torch.int32,
        device=torch.device("tpu"),
    )
    torch.manual_seed(1234)
    t = t.random_()
    w = w.random_()
    self.assertNotEqual(t.to("cpu"), w.to("cpu"))
    torch.manual_seed(1234)
    t_again = t.random_(10)
    self.assert_close(
        golden_result=t.to("cpu"), torch_tpu_result=t_again.to("cpu")
    )

  def test_roll(self):
    # 0-shift.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.roll(torch.arange(10).to(device), shifts=0)
    )
    # Flattened tensor.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.roll(
            torch.arange(10).to(device),
            shifts=2,
        )
    )
    # Flattened tensor, tuple of one.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.roll(
            torch.arange(10).to(device),
            shifts=(2,),
        )
    )
    # Multiple dimensions.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.roll(
            torch.arange(24).reshape(2, 3, 4).to(device),
            shifts=(1, 2),
            dims=(0, 2),
        )
    )
    # Negative shift.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.roll(
            torch.arange(10).to(device),
            shifts=-2,
        )
    )
    # shift > dimension size.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.roll(
            torch.arange(5).to(device),
            shifts=7,
        )
    )
    # abs(shift) > dimension size.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.roll(
            torch.arange(5).to(device),
            shifts=-7,
        )
    )

  def test_round(self):
    # TODO(b/489136147): Fix test case & disable allow_failure
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.round(
            torch.tensor([-2, -1, 0, 1, 2], dtype=torch.int32).to(device),
            decimals=0,
        ),
        check_exception_type=False,
        allow_failure=True,
    )
    # TODO(b/489136147): Fix test case & disable allow_failure
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.round(
            torch.tensor([-2, -1, 0, 1, 2], dtype=torch.int32).to(device),
            decimals=1,
        ),
        check_exception_type=False,
        allow_failure=True,
    )
    # TODO(b/489136147): Fix test case & disable allow_failure
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.round(
            torch.tensor([-2, -1, 0, 1, 2], dtype=torch.int32).to(device),
            decimals=-1,
        ),
        check_exception_type=False,
        allow_failure=True,
    )

  def test_solve_ex_singular(self):
    torch.manual_seed(1234)
    # Create a singular matrix to trigger the unsolvable case.
    # [[1, 1], [1, 1]] is singular (det=0).
    a = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    b = torch.randn(2, 1)

    # CPU
    # check_errors=False is required to get the 'info' tensor with a non-zero
    # value instead of raising an error.
    _, cpu_info = torch.linalg.solve_ex(a, b, check_errors=False)

    # TPU
    tpu_device = torch.device("tpu")
    _, tpu_info = torch.linalg.solve_ex(
        a.to(tpu_device), b.to(tpu_device), check_errors=False
    )

    self.assertEqual(cpu_info.item(), tpu_info.cpu().item())

  def test_solve_ex_singular_check_errors_true(self):
    """Tests that singular inputs raise an error when check_errors=True."""
    torch.manual_seed(1234)
    # Create a singular matrix to trigger the unsolvable case.
    # [[1, 1], [1, 1]] is singular (det=0).
    a = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    b = torch.randn(2, 1)

    # CPU
    with self.assertRaises(RuntimeError):
      torch.linalg.solve_ex(a, b, check_errors=True)

    # TPU
    tpu_device = torch.device("tpu")
    with self.assertRaises(RuntimeError):
      torch.linalg.solve_ex(
          a.to(tpu_device), b.to(tpu_device), check_errors=True
      )

  def test_solve_ex_solvable(self):
    """Tests solve_ex with a solvable system."""
    torch.manual_seed(1234)
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.randn(2, 1)

    def fn(device):
      return torch.linalg.solve_ex(a.to(device), b.to(device))

    self.assert_close_tpu_vs_cpu(fn)

  def test_fill_tensor(self):
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.fill(
            torch.tensor([1, 2, 3], device=device),
            torch.tensor(2, device=device),
        ),
    )

  def test_fill_tensor_empty(self):
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.fill(
            torch.tensor([], device=device),
            torch.tensor(2, device=device),
        ),
    )

  def test_fill_scalar(self):
    def test_fn(device):
      t = torch.tensor([1.0, 2.0, 3.0], device=device)
      t.fill_(2.5)
      return t

    self.assert_close_tpu_vs_cpu(test_fn)

    def test_fn_int(device):
      t = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
      t.fill_(5)
      return t

    self.assert_close_tpu_vs_cpu(test_fn_int)

  def test_fill_complex128(self):
    t = torch.empty((2, 2), dtype=torch.complex128, device=torch.device("tpu"))
    with self.assertRaisesRegex(
        RuntimeError, "complex128 dtype is not supported"
    ):
      t.fill_(1.0)
      t.cpu()

  def test_floor_divide_tensor(self):
    sample_input = (
        torch.tensor([10.0, -10.0, 25.5, -25.5, 0.0]).repeat(12).view(3, 4, 5)
    )
    other_tensor = (
        torch.tensor([3.0, 3.0, 5.0, 5.0, 5.0]).repeat(12).view(3, 4, 5)
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.floor_divide(
            sample_input.to(device=device),
            other_tensor.to(device=device),
        ),
    )

  def test_floor_divide_inplace_tensor(self):
    sample_input = (
        torch.tensor([10.0, -10.0, 25.5, -25.5, 0.0]).repeat(12).view(3, 4, 5)
    )
    other_tensor = (
        torch.tensor([3.0, 3.0, 5.0, 5.0, 5.0]).repeat(12).view(3, 4, 5)
    )
    tpu_device = torch.device("tpu")
    cpu_result = sample_input.clone().floor_divide_(other_tensor)
    tpu_result = (
        sample_input.to(tpu_device)
        .floor_divide_(other_tensor.to(tpu_device))
        .to("cpu")
    )
    self.assert_close(
        golden_result=cpu_result,
        torch_tpu_result=tpu_result,
        rtol=None,
        atol=None,
    )

  def test_device_gen(self):
    gen = torch.Generator(device=torch.device("tpu"))
    self.assertEqual(gen.device.type, torch.device("tpu").type)

    gen.manual_seed(42)
    state = gen.get_state()

    self.assertEqual(state.dtype, torch.uint8)
    self.assertEqual(state[0].item(), 42)
    self.assertEqual(state[1:8].sum().item(), 0)

    gen2 = gen.clone_state()

    new_state = torch.zeros(16, device="cpu", dtype=torch.uint8)
    new_state[0] = 4
    new_state[8] = 3
    gen.set_state(new_state)

    state2 = gen.get_state()
    self.assertEqual(state2[0].item(), 4)
    self.assertEqual(state2[8].item(), 3)
    self.assertEqual(gen2.get_state()[0].item(), 42)

  def test_floor_divide_scalar(self):
    sample_input = (
        torch.tensor([10.0, -10.0, 25.5, -25.5, 0.0]).repeat(12).view(3, 4, 5)
    )
    other_tensor = torch.tensor(3.0)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.floor_divide(
            sample_input.to(device=device),
            other_tensor.to(device=device),
        ),
    )

  def test_floor_divide_inplace_scalar(self):
    sample_input = torch.tensor([10.0, -10.0, 25.5, -25.5]).repeat(3).view(3, 4)
    other_tensor = torch.tensor(3.0)
    tpu_device = torch.device("tpu")
    cpu_result = sample_input.clone().floor_divide_(other_tensor)
    tpu_result = (
        sample_input.to(tpu_device)
        .floor_divide_(other_tensor.to(tpu_device))
        .to("cpu")
    )
    self.assert_close(
        golden_result=cpu_result,
        torch_tpu_result=tpu_result,
        rtol=None,
        atol=None,
    )

  # https://numpy.org/devdocs/user/basics.indexing.html#advanced-indexing
  def test_index_tensor_1(self):
    t = torch.randn(3, 4, 5)
    # Length same as rank, with undefined index
    self.assert_close_tpu_vs_cpu(
        lambda device: t.to(device=device)[
            torch.tensor([0, 2], device=device),
            :,
            torch.tensor([0, 1], device=device),
        ],
    )

  def test_index_tensor_2(self):
    t = torch.randn(3, 4, 5)
    # Length same as rank, with undefined index at the end
    self.assert_close_tpu_vs_cpu(
        lambda device: t.to(device=device)[
            torch.tensor([0, 2], device=device),
            torch.tensor([0, 1], device=device),
            :,
        ],
    )

  def test_index_tensor_3(self):
    t = torch.randn(3, 4, 5)
    # Length less rank, with undefined index at the beginning
    self.assert_close_tpu_vs_cpu(
        lambda device: t.to(device=device)[
            :,
            torch.tensor([0, 1], device=device),
        ],
    )

  def test_index_tensor_4(self):
    # Length less rank, with undefined index at the end
    t = torch.randn(2, 3, 4, 5, 6)
    print(t, flush=True)
    self.assert_close_tpu_vs_cpu(
        lambda device: t.to(device=device)[
            :,
            torch.tensor([0, 2], device=device),
            :,
            torch.tensor([0, 1], device=device),
        ],
    )

  def test_index_tensor_negative(self):
    t = torch.arange(60).reshape(3, 4, 5)
    self.assert_close_tpu_vs_cpu(
        lambda device: t.to(device=device)[
            torch.tensor([-1, 1], device=device),
            :,
            torch.tensor([-2, -1], device=device),
        ],
    )

  def test_negative_indexing_put_scalar(self):

    def test_fn(device: torch.device) -> torch.Tensor:
      t = torch.arange(8, dtype=torch.float32, device=device)
      idx = torch.tensor([-1], dtype=torch.int, device=device)
      t[idx] = torch.tensor(99.0, device=device)
      return t

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_index_tensor_multidim_1(self):
    t = torch.arange(60).reshape(3, 4, 5)
    i1 = torch.tensor([[0, 1], [1, 2]])
    i2 = torch.tensor([2, 3])
    self.assert_close_tpu_vs_cpu(
        lambda device: t.to(device=device)[
            i1.to(device=device),
            :,
            i2.to(device=device),
        ],
    )

  def test_index_tensor_multidim_2(self):
    t = torch.arange(60).reshape(3, 4, 5)
    i1 = torch.tensor([0, 1])
    i2 = torch.tensor([[2], [3]])
    self.assert_close_tpu_vs_cpu(
        lambda device: t.to(device=device)[
            i1.to(device=device),
            :,
            i2.to(device=device),
        ],
    )

  def test_index_tensor_adjacent_adv_index(self):
    t = torch.arange(35).reshape(5, 7)
    i1 = torch.tensor([0, 2, 4])
    i2 = torch.tensor([0, 1, 2])
    self.assert_close_tpu_vs_cpu(
        lambda device: t.to(device=device)[
            i1.to(device=device),
            i2.to(device=device),
        ],
    )

  def test_index_tensor_broadcast_adv_index(self):
    t = torch.arange(12).reshape(3, 4)
    i1 = torch.tensor([[0], [1]])  # shape (2,1)
    i2 = torch.tensor([1, 2])  # shape (2)
    # i1, i2 broadcast to (2,2)
    # result shape should be (2,2)
    self.assert_close_tpu_vs_cpu(
        lambda device: t.to(device=device)[
            i1.to(device=device),
            i2.to(device=device),
        ],
    )

  def test_index_tensor_broadcast_adv_index_slice(self):
    t = torch.arange(60).reshape(3, 4, 5)
    i1 = torch.tensor([[0], [1]])  # shape (2,1)
    i2 = torch.tensor([1, 2])  # shape (2)
    # i1, i2 broadcast to (2,2) for dims 0 and 2.
    # dim 1 is slice.
    # result shape should be (2,2,4) in numpy-like advanced indexing
    # if i1 and i2 are not consecutive.
    self.assert_close_tpu_vs_cpu(
        lambda device: t.to(device=device)[
            i1.to(device=device),
            :,
            i2.to(device=device),
        ],
    )

  def test_index_tensor_diag(self):
    t = torch.arange(9).reshape(3, 3)
    indices = torch.tensor([0, 2])
    self.assert_close_tpu_vs_cpu(
        lambda device: t.to(device=device)[
            indices.to(device=device),
            indices.to(device=device),
        ],
    )

  def test_index_int_adv_mix(self):
    t = torch.arange(60).reshape(3, 4, 5)
    i1 = torch.tensor([[0], [1]])
    i2 = torch.tensor([1, 2])
    self.assert_close_tpu_vs_cpu(
        lambda device: t.to(device=device)[i1.to(device), 0, i2.to(device)],
    )

  def test_index_nd_indexing(self):
    # t = torch.randn(3, 4, 5)
    t = torch.arange(60).reshape(3, 4, 5)
    i1 = torch.tensor([[0, 1], [1, 2]])
    i2 = torch.tensor([2, 3])
    self.assert_close_tpu_vs_cpu(
        lambda device: t.to(device=device)[
            i1.to(device=device),
            :,
            i2.to(device=device),
        ],
    )

  def test_index_non_increasing_indices(self):
    t = torch.randn(5, 5)
    indices = torch.tensor([2, 1, 0])
    self.assert_close_tpu_vs_cpu(
        lambda device: t.to(device=device)[
            indices.to(device=device),
            :,
        ],
    )

  @parameterized.named_parameters(
      (
          "1d_long_mask",
          lambda: torch.tensor([0, 1, 2, 5, 10, -1, 3, 255], dtype=torch.long),
          lambda: torch.tensor(
              [True, True, True, True, False, False, True, False]
          ),
          lambda x, m: x[m],
      ),
      (
          "same_rank_mask",
          lambda: torch.randn(8, 128, 128),
          lambda: torch.rand(8, 128, 128) > 0.5,
          lambda x, m: x[m],
      ),
      (
          "smaller_rank_mask",
          lambda: torch.randn(8, 128, 128),
          lambda: torch.rand(8, 128) > 0.5,
          lambda x, m: x[m],
      ),
      (
          "leading_slices_all_true",
          lambda: torch.randn(2, 3, 7, 7),
          lambda: torch.ones(7, 7, dtype=torch.bool),
          lambda x, m: x[:, :, m],
      ),
      (
          "leading_slices_partial_mask",
          lambda: torch.randn(2, 3, 7, 7),
          lambda: torch.rand(7, 7) > 0.5,
          lambda x, m: x[:, :, m],
      ),
      (
          "leading_slices_empty_mask",
          lambda: torch.randn(2, 3, 7, 7),
          lambda: torch.zeros(7, 7, dtype=torch.bool),
          lambda x, m: x[:, :, m],
      ),
      (
          "leading_and_trailing_slices",
          lambda: torch.randn(2, 3, 7, 7),
          lambda: torch.tensor([True, False, True, False, True, True, False]),
          lambda x, m: x[:, :, m, :],
      ),
      (
          "0d_true_mask_with_slices",
          lambda: torch.randn(2, 3),
          lambda: torch.tensor(True),
          lambda x, m: x[:, m, :],
      ),
      (
          "0d_false_mask",
          lambda: torch.randn(2, 3),
          lambda: torch.tensor(False),
          lambda x, m: x[m],
      ),
      (
          "0d_true_mask_aten_index_out_with_slices",
          lambda: torch.randn(2, 3),
          lambda: torch.tensor(True),
          lambda x, m: (
              torch.ops.aten.index.Tensor_out(
                  x,
                  [None, m, None],
                  out=torch.empty(0, dtype=x.dtype, device=x.device),
              )
              if x.device.type == "tpu"
              else x[:, m, :]
          ),
      ),
      (
          "0d_false_mask_aten_index_out",
          lambda: torch.randn(2, 3),
          lambda: torch.tensor(False),
          lambda x, m: (
              torch.ops.aten.index.Tensor_out(
                  x,
                  [m],
                  out=torch.empty(0, dtype=x.dtype, device=x.device),
              )
              if x.device.type == "tpu"
              else x[m]
          ),
      ),
      (
          "0d_true_mask_aten_index_out_trailing",
          lambda: torch.randn(2, 3),
          lambda: torch.tensor(True),
          lambda x, m: (
              torch.ops.aten.index.Tensor_out(
                  x,
                  [None, None, m],
                  out=torch.empty(0, dtype=x.dtype, device=x.device),
              )
              if x.device.type == "tpu"
              else x[:, :, m]
          ),
      ),
      (
          "0d_false_mask_aten_index_out_with_slices",
          lambda: torch.randn(2, 3),
          lambda: torch.tensor(False),
          lambda x, m: (
              torch.ops.aten.index.Tensor_out(
                  x,
                  [None, m, None],
                  out=torch.empty(0, dtype=x.dtype, device=x.device),
              )
              if x.device.type == "tpu"
              else x[:, m, :]
          ),
      ),
      (
          "0d_scalar_self_with_0d_mask_aten_index_out",
          lambda: torch.tensor(42.0),
          lambda: torch.tensor(True),
          lambda x, m: (
              torch.ops.aten.index.Tensor_out(
                  x,
                  [m],
                  out=torch.empty(0, dtype=x.dtype, device=x.device),
              )
              if x.device.type == "tpu"
              else x[m]
          ),
      ),
      (
          "0d_scalar_self_with_0d_false_mask_aten_index_out",
          lambda: torch.tensor(42.0),
          lambda: torch.tensor(False),
          lambda x, m: (
              torch.ops.aten.index.Tensor_out(
                  x,
                  [m],
                  out=torch.empty(0, dtype=x.dtype, device=x.device),
              )
              if x.device.type == "tpu"
              else x[m]
          ),
      ),
      (
          "3d_mask_with_leading_slice",
          lambda: torch.randn(2, 3, 4, 5),
          lambda: torch.rand(3, 4, 5) > 0.5,
          lambda x, m: x[:, m],
      ),
      (
          "2d_mask_with_leading_and_trailing_slices",
          lambda: torch.randn(2, 3, 4, 5),
          lambda: torch.rand(3, 4) > 0.5,
          lambda x, m: x[:, m, :],
      ),
      (
          "zero_sized_2d_mask_with_leading_slice",
          lambda: torch.randn(2, 0, 4),
          lambda: torch.zeros(0, 4, dtype=torch.bool),
          lambda x, m: x[:, m],
      ),
      (
          "zero_sized_1d_mask_with_slices",
          lambda: torch.randn(2, 0, 4),
          lambda: torch.zeros(0, dtype=torch.bool),
          lambda x, m: x[:, m, :],
      ),
      (
          "zero_sized_unindexed_dim_with_2d_mask",
          lambda: torch.randn(0, 3, 4),
          lambda: torch.tensor([
              [True, False, True, False],
              [False, True, False, True],
              [True, True, False, False],
          ]),
          lambda x, m: x[:, m],
      ),
      (
          "non_contiguous_self_and_mask",
          lambda: torch.randn(3, 2, 5, 4),
          lambda: torch.rand(4, 3) > 0.5,
          lambda x, m: x.transpose(0, 1).transpose(2, 3)[:, m.t(), :],
      ),
  )
  def test_boolean_indexing(self, x_fn, mask_fn, index_fn):
    x = x_fn()
    mask = mask_fn()

    def boolean_indexing(device):
      return index_fn(x.to(device), mask.to(device))

    self.assert_close_tpu_vs_cpu(boolean_indexing)

  # PyTorch allows indexing a device tensor with index tensors that live on
  # CPU (`dev_tensor[{cpu,dev}_tensor]`), so the index tensors are deliberately
  # kept on CPU here while `self` is moved to the device under test.
  @parameterized.named_parameters(
      (
          "1d_bool_mask",
          lambda: torch.randn(4, 3),
          lambda: torch.tensor([True, False, True, True]),
          lambda x, i: x[i],
      ),
      (
          "all_true_1d_bool_mask",
          lambda: torch.arange(4),
          lambda: torch.ones(4, dtype=torch.bool),
          lambda x, i: x[i],
      ),
      (
          "all_false_1d_bool_mask",
          lambda: torch.randn(4, 3),
          lambda: torch.zeros(4, dtype=torch.bool),
          lambda x, i: x[i],
      ),
      (
          "1d_bool_mask_with_leading_slice",
          lambda: torch.randn(2, 5, 3),
          lambda: torch.tensor([False, True, True, False, True]),
          lambda x, i: x[:, i],
      ),
      (
          "2d_bool_mask_with_leading_slice",
          lambda: torch.randn(2, 3, 4),
          lambda: torch.tensor([
              [True, False, True, False],
              [False, False, True, True],
              [True, True, False, False],
          ]),
          lambda x, i: x[:, i],
      ),
      (
          "non_contiguous_2d_bool_mask",
          lambda: torch.randn(4, 3, 2),
          lambda: torch.tensor([
              [True, False, True, False],
              [False, False, True, True],
              [True, True, False, False],
          ]).t(),
          lambda x, i: x[i],
      ),
      (
          "0d_bool_mask",
          lambda: torch.randn(2, 3),
          lambda: torch.tensor(True),
          lambda x, i: x[i],
      ),
      (
          "empty_bool_mask",
          lambda: torch.randn(0, 3),
          lambda: torch.zeros(0, dtype=torch.bool),
          lambda x, i: x[i],
      ),
      (
          "1d_bool_mask_aten_index_out",
          lambda: torch.randn(4, 3),
          lambda: torch.tensor([True, False, True, True]),
          lambda x, i: torch.ops.aten.index.Tensor_out(
              x, [i], out=torch.empty(0, dtype=x.dtype, device=x.device)
          ),
      ),
      (
          "long_indices",
          lambda: torch.randn(5, 3),
          lambda: torch.tensor([2, 0, -1, 2]),
          lambda x, i: x[i],
      ),
      (
          "long_indices_with_leading_slice",
          lambda: torch.randn(2, 5, 3),
          lambda: torch.tensor([[4, 0], [1, -2]]),
          lambda x, i: x[:, i],
      ),
      (
          "int_indices",
          lambda: torch.randn(5, 3),
          lambda: torch.tensor([3, 1], dtype=torch.int32),
          lambda x, i: x[i],
      ),
  )
  def test_index_with_cpu_indices(self, x_fn, index_fn, index_op):
    x = x_fn()
    index = index_fn()
    self.assert_close_tpu_vs_cpu(lambda device: index_op(x.to(device), index))

  def test_index_with_mixed_cpu_and_device_indices(self):
    x = torch.randn(4, 5, 3)
    cpu_index = torch.tensor([3, 0, -1])
    device_index = torch.tensor([1, 4, 2])
    self.assert_close_tpu_vs_cpu(
        lambda device: x.to(device)[cpu_index, device_index.to(device)]
    )

  @parameterized.named_parameters(
      (
          "bool_mask_broadcast_value",
          lambda: torch.randn(4, 3),
          lambda: [torch.tensor([True, False, True, True])],
          lambda: torch.tensor(2.5),
      ),
      (
          "bool_mask_per_row_values",
          lambda: torch.randn(4, 3),
          lambda: [torch.tensor([True, False, True, True])],
          lambda: torch.randn(3, 3),
      ),
      (
          "bool_mask_with_leading_slice",
          lambda: torch.randn(2, 4, 3),
          lambda: [None, torch.tensor([True, False, True, True])],
          lambda: torch.randn(2, 3, 3),
      ),
      (
          "2d_bool_mask",
          lambda: torch.randn(3, 4),
          lambda: [
              torch.tensor([
                  [True, False, True, False],
                  [False, False, True, True],
                  [True, True, False, False],
              ])
          ],
          lambda: torch.randn(6),
      ),
      (
          "non_contiguous_2d_bool_mask_per_element_values",
          lambda: torch.randn(4, 3),
          lambda: [
              torch.tensor([
                  [True, False, True, False],
                  [False, False, True, True],
                  [True, True, False, False],
              ]).t()
          ],
          lambda: torch.randn(6),
      ),
      (
          "long_indices",
          lambda: torch.randn(5, 3),
          lambda: [torch.tensor([-1, 0, 2])],
          lambda: torch.randn(3, 3),
      ),
      (
          "long_indices_with_leading_slice",
          lambda: torch.randn(2, 5),
          lambda: [None, torch.tensor([4, 1])],
          lambda: torch.randn(2, 2),
      ),
      (
          "multiple_long_indices",
          lambda: torch.randn(4, 5),
          lambda: [torch.tensor([0, 3]), torch.tensor([-1, 2])],
          lambda: torch.randn(2),
      ),
      (
          "int_indices",
          lambda: torch.randn(5, 3),
          lambda: [torch.tensor([3, -1], dtype=torch.int32)],
          lambda: torch.randn(2, 3),
      ),
      (
          "empty_bool_mask_broadcast_value",
          lambda: torch.randn(0, 3),
          lambda: [torch.zeros(0, dtype=torch.bool)],
          lambda: torch.tensor(2.5),
      ),
      (
          "empty_bool_mask_per_row_values",
          lambda: torch.randn(0, 3),
          lambda: [torch.zeros(0, dtype=torch.bool)],
          lambda: torch.randn(0, 3),
      ),
      (
          "bool_mask_and_long_index",
          lambda: torch.randn(4, 5),
          lambda: [
              torch.tensor([True, False, True, True]),
              torch.tensor([0, 4, -2]),
          ],
          lambda: torch.randn(3),
      ),
  )
  def test_index_put_with_cpu_indices(self, x_fn, indices_fn, values_fn):
    x = x_fn()
    indices = indices_fn()
    values = values_fn()
    for accumulate in (False, True):
      with self.subTest(accumulate=accumulate):
        # `x` is cloned since `x.to("cpu")` aliases `x`, which `index_put_`
        # would otherwise modify in place. `torch.ops.aten.index_put_` is used
        # since the `Tensor.index_put_` Python binding rejects `None` indices.
        self.assert_close_tpu_vs_cpu(
            lambda device, accumulate=accumulate: torch.ops.aten.index_put_(
                x.clone().to(device), indices, values.to(device), accumulate
            )
        )

  # CPU PyTorch only supports a 0-d bool mask in `index_put_` through its
  # `masked_fill_` fast path (scalar value, `accumulate=False`); otherwise its
  # CPU index kernel hits an internal assert, so only that case is compared.
  @parameterized.parameters(True, False)
  def test_index_put_with_cpu_0d_bool_mask(self, mask_value):
    x = torch.randn(2, 3)
    mask = torch.tensor(mask_value)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ops.aten.index_put_(
            x.clone().to(device), [mask], torch.tensor(2.5).to(device), False
        )
    )

  @parameterized.named_parameters(
      (
          "long_indices",
          lambda: torch.randn(4, 5),
          lambda: torch.tensor([0, 3]),
          lambda: torch.tensor([-1, 2]),
          lambda: torch.randn(2),
      ),
      (
          "bool_mask_and_long_index",
          lambda: torch.randn(4, 5),
          lambda: torch.tensor([True, False, True, True]),
          lambda: torch.tensor([0, 4, -2]),
          lambda: torch.randn(3),
      ),
  )
  def test_index_put_with_mixed_cpu_and_device_indices(
      self, x_fn, cpu_index_fn, device_index_fn, values_fn
  ):
    x = x_fn()
    cpu_index = cpu_index_fn()
    device_index = device_index_fn()
    values = values_fn()
    for accumulate in (False, True):
      with self.subTest(accumulate=accumulate):
        self.assert_close_tpu_vs_cpu(
            lambda device, accumulate=accumulate: x.clone()
            .to(device)
            .index_put_(
                [cpu_index, device_index.to(device)],
                values.to(device),
                accumulate=accumulate,
            )
        )

  def test_is_nonzero(self):
    def assert_is_nonzero_equal_on_cpu_vs_tpu(tensor: torch.Tensor):
      cpu_result = torch.is_nonzero(tensor.to("cpu"))
      tpu_result = torch.is_nonzero(tensor.to(torch.device("tpu")))
      self.assertEqual(cpu_result, tpu_result)

    assert_is_nonzero_equal_on_cpu_vs_tpu(torch.tensor([0.0]))
    assert_is_nonzero_equal_on_cpu_vs_tpu(torch.tensor([1.5]))
    assert_is_nonzero_equal_on_cpu_vs_tpu(torch.tensor([False]))
    assert_is_nonzero_equal_on_cpu_vs_tpu(torch.tensor([3]))
    assert_is_nonzero_equal_on_cpu_vs_tpu(torch.scalar_tensor(0.0))
    assert_is_nonzero_equal_on_cpu_vs_tpu(torch.scalar_tensor(1.5))
    assert_is_nonzero_equal_on_cpu_vs_tpu(torch.tensor([[[0.0]]]))
    assert_is_nonzero_equal_on_cpu_vs_tpu(torch.tensor([[[1.5]]]))

  def test_isin(self):
    elements = torch.tensor([[1, 2], [3, 4]])
    test_elements = torch.tensor([3, 4])
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.isin(
            to(elements, device=device),
            to(test_elements, device=device),
            assume_unique=False,
            invert=False,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.isin(
            to(elements, device=device),
            to(test_elements, device=device),
            assume_unique=True,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.isin(
            to(elements, device=device),
            to(test_elements, device=device),
            invert=True,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.isin(
            to(elements, device=device),
            3,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.isin(
            3,
            to(test_elements, device=device),
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.isin(
            -3,
            to(test_elements, device=device),
        ),
    )

  def test_kron_unit(self):
    """Extra tests for kron."""

    # Scalar x Scalar
    # TODO: torch cpu implementation doesn't support scalars, but numpy does.
    # a = 2
    # b = 3
    # self.assert_close_tpu_vs_cpu(
    #     lambda device: torch.kron(to(a, device=device),
    #                               to(b, device=device)),
    # )

    # Scalar x 1D
    # TODO: torch cpu implementation doesn't support scalars, but numpy does.
    # a = 2
    # b = torch.randn(3)
    # golden_result = torch.tensor([6])
    # tpu_result = torch.kron(
    #     to(a, device=torch.device("tpu")),
    #     to(b, device=torch.device("tpu")),
    # )
    # tpu_result = to(tpu_result, device="cpu")
    # self.assert_close(golden_result, tpu_result)

    # 1D x Scalar
    # TODO: torch cpu implementation doesn't support scalars, but numpy does.
    # a = torch.randn(4)
    # b = 3
    # golden_result = torch.tensor([12])
    # tpu_result = torch.kron(
    #     to(a, device=torch.device("tpu")),
    #     to(b, device=torch.device("tpu")),
    # )
    # tpu_result = to(tpu_result, device="cpu")
    # self.assert_close(golden_result, tpu_result)

    # 1D x 1D
    a = torch.randn(5)
    b = torch.randn(4)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.kron(to(a, device=device), to(b, device=device)),
    )

    # 1D x 2D
    a = torch.randn(3)
    b = torch.randn(4, 5)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.kron(to(a, device=device), to(b, device=device)),
    )

    # 2D x 1D
    a = torch.randn(3, 4)
    b = torch.randn(5)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.kron(to(a, device=device), to(b, device=device)),
    )

    # 2D x 2D
    a = torch.randn(3, 4)
    b = torch.randn(5, 6)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.kron(to(a, device=device), to(b, device=device)),
    )

  def test_mul(self):
    x = torch.tensor([1, 2])
    y = torch.tensor([3, 4])
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.mul(x.to(device=device), y.to(device=device)),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.mul(
            x.to(device=device, dtype=torch.int32),
            y.to(device=device, dtype=torch.float),
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.mul(12.3, y.to(device=device, dtype=torch.int32)),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.mul(x.to(device=device, dtype=torch.float), 123),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.mul(
            torch.tensor([[1], [2]]).to(device=device),
            torch.tensor([3, 4]).to(device=device),
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.mul(
            torch.tensor([1, 2]).to(device=device),
            torch.tensor([[3], [4]]).to(device=device),
        ),
    )

  def test_ne(self):
    # tensor-tensor
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ne(
            torch.tensor([1, 2, 3], dtype=torch.int32).to(device),
            torch.tensor([3, 2, 1], dtype=torch.int32).to(device),
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ne(
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).to(device),
            torch.tensor([1.0, 0.0, 3.1], dtype=torch.float32).to(device),
        )
    )
    # tensor-scalar
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ne(
            torch.tensor([True, False, True], dtype=torch.bool).to(device),
            False,
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ne(
            torch.tensor([1, 2, 3], dtype=torch.int32).to(device),
            2,
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ne(
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).to(device),
            2.0,
        )
    )
    # out param
    out_cpu = torch.empty(3, dtype=torch.bool)
    out_tpu = torch.empty(3, dtype=torch.bool, device=torch.device("tpu"))
    a = torch.tensor([1, 2, 3], dtype=torch.int32)
    b = torch.tensor([3, 2, 1], dtype=torch.int32)
    torch.ne(a, b, out=out_cpu)
    torch.ne(
        a.to(torch.device("tpu")),
        b.to(torch.device("tpu")),
        out=out_tpu,
    )
    self.assertEqual(out_cpu, out_tpu.cpu())

    # empty input
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ne(
            torch.tensor([], dtype=torch.int32).to(device),
            torch.tensor([], dtype=torch.int32).to(device),
        )
    )
    # mismatched dtypes
    # int32 and int64
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ne(
            torch.tensor([1, 2, 3], dtype=torch.int32).to(device),
            torch.tensor([3, 2, 1], dtype=torch.int64).to(device),
        )
    )
    # bool and int32
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ne(
            torch.tensor([True, False, True], dtype=torch.bool).to(device),
            torch.tensor([0, 1, 1], dtype=torch.int32).to(device),
        )
    )
    # float32 and int32
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ne(
            torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32).to(device),
            torch.tensor([1, 0, 1], dtype=torch.int32).to(device),
        )
    )

  def test_nonzero(self):
    # empty tensor
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nonzero(torch.tensor([]).to(device=device))
    )

    # scalar tensor
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nonzero(torch.tensor(1).to(device=device))
    )

    # no non-zeros
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nonzero(torch.tensor([1, 2, 3]).to(device=device))
    )

    # non-zeros in the first dimension
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nonzero(
            torch.tensor([1, 0, 3]).to(device=device),
        ),
    )

    # multiple non-zeros, multiple dimensions
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nonzero(
            torch.tensor([[0, 1, 2, 0, 3], [0, 0, -1, 0, 0]]).to(device=device),
        ),
    )

    # non-zeros with out param
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nonzero(
            torch.tensor([[1, 2, 3, 4, 0]]).to(device=device),
            out=torch.empty(4, 2, dtype=torch.int64).to(device=device),
        ),
    )

    # tests from torch docs
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nonzero(
            torch.tensor([1, 1, 1, 0, 1]).to(device=device),
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nonzero(
            torch.tensor([
                [0.6, 0.0, 0.0, 0.0],
                [0.0, 0.4, 0.0, 0.0],
                [0.0, 0.0, 1.2, 0.0],
                [0.0, 0.0, 0.0, -0.4],
            ]).to(device=device),
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nonzero(
            torch.tensor(5).to(device=device),
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nonzero(
            torch.tensor([1, 1, 1, 0, 1]).to(device=device),
            as_tuple=True,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nonzero(
            torch.tensor([
                [0.6, 0.0, 0.0, 0.0],
                [0.0, 0.4, 0.0, 0.0],
                [0.0, 0.0, 1.2, 0.0],
                [0.0, 0.0, 0.0, -0.4],
            ]).to(device=device),
            as_tuple=True,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nonzero(
            torch.tensor(5).to(device=device),
            as_tuple=True,
        ),
    )

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nonzero(
            torch.tensor([
                [True, False, False, False],
                [False, True, False, False],
                [False, False, True, False],
                [False, False, False, True],
            ]).to(device=device),
        ),
    )

  @parameterized.parameters(torch.float32, torch.float16)
  def test_nonzero_subnormal(self, dtype):
    # TPU float comparators flush subnormals to zero, unlike CPU and CUDA.
    # TODO: b/562066667 - cover bfloat16. TPU flushes its subnormals as the
    # operand enters the computation, so the lowering never sees them.
    subnormal = torch.finfo(dtype).smallest_normal / 2

    def compute(device):
      return torch.nonzero(
          torch.tensor([0.0, subnormal, -0.0, -subnormal, 1.0], dtype=dtype).to(
              device=device
          ),
      )

    self.assert_close_tpu_vs_cpu(compute)

  @parameterized.parameters(torch.float32, torch.float16)
  def test_count_nonzero_subnormal(self, dtype):
    # Same flush-to-zero gap as test_nonzero_subnormal above.
    subnormal = torch.finfo(dtype).smallest_normal / 2

    def compute(device):
      return torch.count_nonzero(
          torch.tensor([0.0, subnormal, -0.0, -subnormal, 1.0], dtype=dtype).to(
              device=device
          ),
      )

    self.assert_close_tpu_vs_cpu(compute)

  @parameterized.named_parameters(("functional", False), ("out", True))
  def test_nonzero_reuses_static_kernel_for_same_num_nonzero(self, use_out):
    # nonzero compiles one static kernel per number of nonzero elements. That
    # count is the leading extent of the output, which is already part of the
    # cache key, so the kernels must stay distinct without an op param cache
    # key for it.
    shape = (7, 11)  # Unused by other nonzero tests, so the cache starts cold.

    def make_input(num_nonzero, offset):
      x = torch.zeros(shape).flatten()
      x[offset : offset + 3 * num_nonzero : 3] = 1.0
      return x.reshape(shape)

    def run_nonzero(x):
      """Checks nonzero(x) on TPU and returns the cache (hits, misses) it adds."""
      tpu_x = x.to("tpu")
      tpu_out = torch.empty(0, x.dim(), dtype=torch.int64, device="tpu")
      start_hits = torch.tpu._get_cache_hits()
      start_misses = torch.tpu._get_cache_misses()
      if use_out:
        torch.nonzero(tpu_x, out=tpu_out)
        actual = tpu_out.cpu()
      else:
        actual = torch.nonzero(tpu_x).cpu()
      hits = torch.tpu._get_cache_hits() - start_hits
      misses = torch.tpu._get_cache_misses() - start_misses
      utils.assert_close(actual, torch.nonzero(x))
      return hits, misses

    _, misses = run_nonzero(make_input(num_nonzero=3, offset=0))
    self.assertGreaterEqual(misses, 1, "first count must compile a kernel")

    # Same count at different positions: the count-3 kernel is reused.
    hits, misses = run_nonzero(make_input(num_nonzero=3, offset=1))
    self.assertEqual(misses, 0, "same count must not recompile")
    self.assertGreater(hits, 0)

    # New count: only the count-5 nonzero kernel is compiled. The size kernel
    # depends on the input shape alone, so it is still a hit.
    _, misses = run_nonzero(make_input(num_nonzero=5, offset=0))
    self.assertEqual(misses, 1, "new count must not reuse the count-3 kernel")

  def test_normal_float_float_out(self):
    golden_result = torch.normal(
        mean=2.0,
        std=3.0,
        size=(1, 4),
        dtype=torch.float32,
        device=self.golden_device,
    )
    out = torch.empty(
        1,
        4,
        dtype=torch.float32,
        device=torch.device("tpu"),
    )
    tpu_result = torch.normal(mean=2.0, std=3.0, size=(1, 4), out=out).cpu()

    self.assertEqual(golden_result.shape, tpu_result.shape)
    self.assertEqual(golden_result.dtype, tpu_result.dtype)

  def test_normal_broadcasting_real(self):
    def compute(device):
      mean = torch.zeros((2, 1), device=device)
      std = torch.ones((1, 3), device=device)
      return torch.normal(mean, std)

    self.assert_close_tpu_vs_cpu(compute, check_value=CheckValueMode.SKIP)

  def test_normal_broadcasting_complex(self):
    def compute(device):
      mean_c = torch.complex(
          torch.zeros((2, 1), device=device), torch.zeros((2, 1), device=device)
      )
      std = torch.ones((1, 3), device=device)
      return torch.normal(mean_c, std)

    self.assert_close_tpu_vs_cpu(compute, check_value=CheckValueMode.SKIP)

  def test_normal_inplace_complex(self):
    device = torch.device("tpu")
    t = torch.zeros(10, dtype=torch.complex64, device=device)
    t.normal_(mean=1.0, std=0.5)
    self.assertTrue(t.is_complex())
    self.assertNotEqual(t.abs().sum().item(), 0.0)

  def test_normal_integer_std(self):
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.normal(
            mean=torch.zeros((2, 1), device=device),
            std=torch.ones((1, 3), device=device, dtype=torch.int32),
        ),
        check_value=CheckValueMode.SKIP,
    )

  def test_normal_empty_std(self):
    def compute(device):
      out_empty = torch.empty(0, device=device)
      torch.normal(mean=0.0, std=torch.tensor([], device=device), out=out_empty)
      return out_empty

    self.assert_close_tpu_vs_cpu(compute, check_value=CheckValueMode.SKIP)

  def test_normal_resize_out(self):
    def compute(device):
      mean = torch.zeros(2, device=device)
      std = torch.ones(2, device=device)
      out = torch.empty(3, device=device)
      torch.normal(mean, std, out=out)
      return out

    self.assert_close_tpu_vs_cpu(compute, check_value=CheckValueMode.SKIP)

  def test_normal_resize_float_tensor_out(self):
    def compute(device):
      std = torch.ones(5, device=device)
      out = torch.empty(2, device=device)
      torch.normal(mean=0.0, std=std, out=out)
      return out

    self.assert_close_tpu_vs_cpu(compute, check_value=CheckValueMode.SKIP)

  def test_normal_resize_tensor_float_out(self):
    def compute(device):
      mean = torch.zeros(4, device=device)
      out = torch.empty(1, device=device)
      torch.normal(mean, std=1.0, out=out)
      return out

    self.assert_close_tpu_vs_cpu(compute, check_value=CheckValueMode.SKIP)

  def test_normal_resize_complex_out(self):
    def compute(device):
      mean = torch.zeros(3, dtype=torch.complex64, device=device)
      out = torch.empty(1, dtype=torch.complex64, device=device)
      torch.normal(mean, std=1.0, out=out)
      return out

    self.assert_close_tpu_vs_cpu(compute, check_value=CheckValueMode.SKIP)

  def test_normal_real_mean_complex_out(self):
    def compute(device):
      mean = torch.zeros(100, device=device)
      std = torch.ones(100, device=device)
      out = torch.empty(100, dtype=torch.complex64, device=device)
      torch.normal(mean, std, out=out)
      # Having a real mean does not mean the output should be real; it just
      # means the complex output will center around mean+0j.
      self.assertTrue((out.imag != 0.0).any())
      return out

    self.assert_close_tpu_vs_cpu(compute, check_value=CheckValueMode.SKIP)

  def test_normal_complex_scaling_deterministic_scalar(self):
    def compute(device):
      # Sample complex normal.
      torch.manual_seed(42)
      z = torch.normal(
          mean=0.0, std=1.0, size=(10,), dtype=torch.complex64, device=device
      )

      # Sample real normal with 1/sqrt(2) scaling.
      torch.manual_seed(42)
      r = torch.normal(
          mean=0.0,
          std=1.0 / math.sqrt(2.0),
          size=(10, 2),
          dtype=torch.float32,
          device=device,
      )

      self.assert_close(
          golden_result=r.cpu(), torch_tpu_result=torch.view_as_real(z).cpu()
      )
      return z

    self.assert_close_tpu_vs_cpu(compute, check_value=CheckValueMode.SKIP)

  def test_normal_complex_scaling_deterministic_tensor_scalar(self):
    def compute(device):
      mean = torch.full((10,), 1.0 + 2.0j, dtype=torch.complex64, device=device)
      torch.manual_seed(42)
      z = torch.normal(mean, std=1.0)

      torch.manual_seed(42)
      mean_real = torch.tensor([1.0, 2.0], device=device).expand(10, 2)
      r = torch.normal(
          mean_real,
          std=1.0 / math.sqrt(2.0),
      )
      self.assert_close(
          golden_result=r.cpu(), torch_tpu_result=torch.view_as_real(z).cpu()
      )
      return z

    self.assert_close_tpu_vs_cpu(compute, check_value=CheckValueMode.SKIP)

  def test_normal_complex_scaling_deterministic_tensor_tensor(self):
    def compute(device):
      mean = torch.full((10,), 1.0 + 2.0j, dtype=torch.complex64, device=device)
      std = torch.full((10,), 3.0, dtype=torch.float32, device=device)
      torch.manual_seed(42)
      z = torch.normal(mean, std)

      torch.manual_seed(42)
      mean_real = torch.tensor([1.0, 2.0], device=device).expand(10, 2)
      # The real std is unsqueezed and broadcasted to both real/imag parts.
      std_real = torch.full((10, 2), 3.0, device=device)
      r = torch.normal(
          mean_real,
          std_real / math.sqrt(2.0),
      )
      self.assert_close(
          golden_result=r.cpu(), torch_tpu_result=torch.view_as_real(z).cpu()
      )
      return z

    self.assert_close_tpu_vs_cpu(compute, check_value=CheckValueMode.SKIP)

  def test_pow(self):
    rtol, atol = 1.3e-6, 1.2e-5
    # tensor-tensor
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.pow(
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).to(device),
            torch.tensor([3.0, 1.0, 2.0], dtype=torch.float32).to(device),
        ),
        rtol=rtol,
        atol=atol,
    )
    # tensor-scalar
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.pow(
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).to(device),
            2.0,
        ),
        rtol=rtol,
        atol=atol,
    )
    # scalar-tensor
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.pow(
            2.0,
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).to(device),
        ),
        rtol=rtol,
        atol=atol,
    )
    # out param (tensor/tensor)
    out_cpu = torch.empty(3, dtype=torch.float32)
    out_tpu = torch.empty(3, dtype=torch.float32, device=torch.device("tpu"))
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b = torch.tensor([3.0, 1.0, 2.0], dtype=torch.float32)
    torch.pow(a, b, out=out_cpu)
    torch.pow(
        a.to(torch.device("tpu")),
        b.to(torch.device("tpu")),
        out=out_tpu,
    )
    self.assertEqual(out_cpu, out_tpu.cpu())

    # out param (scalar/tensor)
    out_cpu = torch.empty(3, dtype=torch.float32)
    out_tpu = torch.empty(3, dtype=torch.float32, device=torch.device("tpu"))
    a = 3.0
    b = torch.tensor([3.0, 1.0, 2.0], dtype=torch.float32)
    torch.pow(a, b, out=out_cpu)
    torch.pow(a, b.to(torch.device("tpu")), out=out_tpu)
    self.assertEqual(out_cpu, out_tpu.cpu())

    # empty input
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.pow(
            torch.tensor([], dtype=torch.float32).to(device),
            torch.tensor([], dtype=torch.float32).to(device),
        )
    )
    # mismatched dtypes
    # float32 and float64
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.pow(
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).to(device),
            torch.tensor([3.0, 1.0, 2.0], dtype=torch.float64).to(device),
        )
    )
    # float and int
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.pow(
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).to(device),
            torch.tensor([3, 1, 2], dtype=torch.int32).to(device),
        ),
        rtol=rtol,
        atol=atol,
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.pow(
            torch.tensor([1, 2, 3], dtype=torch.int32).to(device),
            torch.tensor([3.0, 1.0, 2.0], dtype=torch.float32).to(device),
        ),
        rtol=rtol,
        atol=atol,
    )
    # inplace pow_.Tensor
    a_cpu = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b_cpu = torch.tensor([3.0, 1.0, 2.0], dtype=torch.float32)
    a_tpu = a_cpu.clone().to(torch.device("tpu"))
    b_tpu = b_cpu.clone().to(torch.device("tpu"))
    a_cpu.pow_(b_cpu)
    a_tpu.pow_(b_tpu)
    self.assert_close(
        golden_result=a_cpu, torch_tpu_result=a_tpu.cpu(), rtol=rtol, atol=atol
    )

    # inplace pow_.Scalar
    a_cpu = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    a_tpu = a_cpu.clone().to(torch.device("tpu"))
    a_cpu.pow_(2.0)
    a_tpu.pow_(2.0)
    self.assert_close(
        golden_result=a_cpu, torch_tpu_result=a_tpu.cpu(), rtol=rtol, atol=atol
    )

  @parameterized.parameters(
      torch.float32,
      torch.float64,
      torch.bfloat16,
      torch.float16,
      torch.complex64,
      torch.complex128,
  )
  def test_pow_exponent_two(self, dtype):
    if dtype.is_complex:
      x_init = torch.tensor(
          [-2.5 + 1.25j, -0.99 - 0.5j, 0.0 + 0.0j, 1.5 - 2.25j], dtype=dtype
      )
    else:
      x_init = torch.tensor([-2.5, -0.99, 0.0, 0.99, 3.25], dtype=dtype)
    grad_out = torch.ones_like(x_init)

    for exp in (2, 2.0):

      def compute(device, exp_val=exp):
        x = to(x_init.clone(), device=device).requires_grad_(True)
        out = torch.pow(x, exp_val)
        out.backward(to(grad_out, device=device))
        return out, x.grad

      self.assert_close_tpu_vs_cpu(compute)

  @parameterized.parameters(
      torch.float32,
      torch.float64,
      torch.bfloat16,
      torch.float16,
  )
  def test_atanh_backward_near_unit_boundary(self, dtype):
    vals = (
        [-0.99, -0.95, -0.9, 0.9, 0.95, 0.99]
        if dtype in (torch.float16, torch.bfloat16)
        else [-0.9995, -0.999, -0.99, -0.9, 0.9, 0.99, 0.999, 0.9995]
    )
    x_init = torch.tensor(vals, dtype=dtype)
    grad_out = torch.ones_like(x_init)

    def compute(device):
      x = to(x_init.clone(), device=device).requires_grad_(True)
      out = torch.atanh(x)
      out.backward(to(grad_out, device=device))
      return x.grad

    self.assert_close_tpu_vs_cpu(compute)

  def test_tanh_backward_grad_input_cross_dtype(self):
    g = torch.ones((4,), dtype=torch.float32)
    o = torch.full((4,), 0.5, dtype=torch.float32)

    def compute(device):
      grad_in = torch.empty((4,), dtype=torch.bfloat16, device=device)
      return torch.ops.aten.tanh_backward.grad_input(
          to(g, device=device), to(o, device=device), grad_input=grad_in
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_repeat_interleave_self_int(self):
    t = torch.tensor([1, 2, 3])
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.repeat_interleave(
            input=to(t, device=device),
            repeats=2,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.repeat_interleave(
            input=to(t, device=device),
            repeats=2,
            dim=0,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.repeat_interleave(
            input=to(t, device=device),
            repeats=1,
            dim=0,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.repeat_interleave(
            input=to(t, device=device),
            repeats=0,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.repeat_interleave(
            input=to(t, device=device),
            repeats=0,
            dim=0,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.repeat_interleave(
            input=to(t, device=device),
            repeats=2,
            dim=0,
            output_size=6,
        ),
    )

    # Multi-dimensional shapes (2D, 3D, 4D) with positive, negative, and None dims.
    t_3d = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    for dim in (None, 0, 1, 2, -1, -2):
      self.assert_close_tpu_vs_cpu(
          lambda device, d=dim: torch.repeat_interleave(
              input=to(t_3d, device=device),
              repeats=3,
              dim=d,
          ),
      )

    # 4D GQA head expansion shape with explicit output_size and 1-element tensor repeat.
    t_4d = torch.randn(2, 4, 8, 16, dtype=torch.float32)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.repeat_interleave(
            input=to(t_4d, device=device),
            repeats=2,
            dim=2,
            output_size=16,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.repeat_interleave(
            input=to(t_4d, device=device),
            repeats=to(torch.tensor([2]), device=device),
            dim=2,
            output_size=16,
        ),
    )

    # Backward / Autograd verification across dim=1, dim=-1, dim=None, repeats=1, repeats=0.
    grad_3d = torch.randn(2, 9, 4, dtype=torch.float32)

    def _backward_dim1(device):
      x = to(t_3d.clone(), device=device).requires_grad_(True)
      out = torch.repeat_interleave(x, repeats=3, dim=1)
      out.backward(to(grad_3d, device=device))
      return x.grad

    self.assert_close_tpu_vs_cpu(_backward_dim1)

    grad_flat = torch.randn(24 * 2, dtype=torch.float32)

    def _backward_flat(device):
      x = to(t_3d.clone(), device=device).requires_grad_(True)
      out = torch.repeat_interleave(x, repeats=2, dim=None)
      out.backward(to(grad_flat, device=device))
      return x.grad

    self.assert_close_tpu_vs_cpu(_backward_flat)

    def _backward_repeats1(device):
      x = to(t_3d.clone(), device=device).requires_grad_(True)
      out = torch.repeat_interleave(x, repeats=1, dim=2)
      out.backward(to(t_3d, device=device))
      return x.grad

    self.assert_close_tpu_vs_cpu(_backward_repeats1)

    def _backward_repeats0(device):
      x = to(t_3d.clone(), device=device).requires_grad_(True)
      out = torch.repeat_interleave(x, repeats=0, dim=1)
      out.backward(torch.zeros_like(out))
      return x.grad

    self.assert_close_tpu_vs_cpu(_backward_repeats0)

  def test_repeat_interleave_tensor(self):
    t = torch.tensor([1, 2, 3])

    # Case 1: Multi-element repeat tensor
    repeats = torch.tensor([2, 1, 3])
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.repeat_interleave(
            input=to(t, device=device),
            repeats=to(repeats, device=device),
        ),
    )

    # Case 2: Single-element 1D repeat tensor (Broadcasting check)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.repeat_interleave(
            input=to(t, device=device),
            repeats=to(torch.tensor([2]), device=device),
        ),
    )

    # Case 3: 0D repeat tensor / scalar (0D check)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.repeat_interleave(
            input=to(t, device=device),
            repeats=to(torch.tensor(2), device=device),
        ),
    )

    # Case 4: Repetitions containing zero elements (pruning check)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.repeat_interleave(
            input=to(t, device=device),
            repeats=to(torch.tensor([2, 0, 1]), device=device),
        ),
    )

    # Case 5: 2D inputs along different dimensions (rank and dim checks)
    t_2d = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    repeats_2d = torch.tensor([2, 3])
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.repeat_interleave(
            input=to(t_2d, device=device),
            repeats=to(repeats_2d, device=device),
            dim=0,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.repeat_interleave(
            input=to(t_2d, device=device),
            repeats=to(repeats_2d, device=device),
            dim=1,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.repeat_interleave(
            input=to(t_2d, device=device),
            repeats=to(repeats_2d, device=device),
            dim=-1,
        ),
    )

    # Case 6: Dtype variations (bfloat16, int32, bool)
    for dtype in [torch.bfloat16, torch.int32, torch.bool]:
      t_dtype = torch.tensor([1, 0, 1], dtype=dtype)
      repeats_dtype = torch.tensor([2, 1, 3])
      self.assert_close_tpu_vs_cpu(
          lambda device, t_dtype=t_dtype, repeats_dtype=repeats_dtype: torch.repeat_interleave(
              input=to(t_dtype, device=device),
              repeats=to(repeats_dtype, device=device),
          ),
      )

    # Case 7: Explicit output_size parameter (avoids host sync)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.repeat_interleave(
            input=to(t, device=device),
            repeats=to(repeats, device=device),
            output_size=6,
        ),
    )

  def test_repeat_interleave_repeats_only(self):
    # `torch.repeat_interleave(repeats)` returns the index vector.
    for dtype in [torch.int64, torch.int32]:
      for repeats in [[2, 1, 3], [2, 0, 1], [0, 0, 0], [5], []]:
        r = torch.tensor(repeats, dtype=dtype)
        self.assert_close_tpu_vs_cpu(
            lambda device, r=r: torch.repeat_interleave(to(r, device=device)),
        )

    # Explicit output_size skips the host sync.
    r = torch.tensor([2, 1, 3])
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.repeat_interleave(
            to(r, device=device), output_size=6
        ),
    )

    # Same repeats shape, different totals: each total gets its own output.
    for repeats in [[1, 1, 1], [4, 4, 4]]:
      r = torch.tensor(repeats)
      self.assert_close_tpu_vs_cpu(
          lambda device, r=r: torch.repeat_interleave(to(r, device=device)),
      )

    # Non-contiguous repeats input (sliced with stride).
    for dtype in [torch.int64, torch.int32]:
      r = torch.tensor([2, 0, 1, 0, 3], dtype=dtype)[::2]
      self.assertFalse(r.is_contiguous())
      self.assert_close_tpu_vs_cpu(
          lambda device, r=r: torch.repeat_interleave(to(r, device=device)),
      )

    # Out-variant (`repeat_interleave.Tensor_out`): pre-allocated, resized,
    # zero-length output, and in-place aliasing (`out=repeats` buffer donation).
    for dtype in [torch.int64, torch.int32]:
      for repeats, out_init_size, out_size in [
          ([2, 1, 3], (6,), None),
          ([2, 1, 3], (2,), 6),
          ([0, 0, 0], (4,), None),
          ([], (3,), None),
      ]:
        r = torch.tensor(repeats, dtype=dtype)

        def _run_out(
            device, r=r, dtype=dtype, init_sz=out_init_size, out_sz=out_size
        ):
          out = torch.empty(init_sz, dtype=dtype, device=device)
          res = torch.ops.aten.repeat_interleave.Tensor_out(
              to(r, device=device), output_size=out_sz, out=out
          )
          self.assertIs(res, out)
          return out

        self.assert_close_tpu_vs_cpu(_run_out)

      # In-place aliasing (`out=repeats` with matching shape `{3}` -> `{3}`).
      def _run_inplace_out(device, dtype=dtype):
        r_dev = to(torch.tensor([1, 0, 2], dtype=dtype), device=device)
        res = torch.ops.aten.repeat_interleave.Tensor_out(
            r_dev, output_size=3, out=r_dev
        )
        self.assertIs(res, r_dev)
        return r_dev

      self.assert_close_tpu_vs_cpu(_run_inplace_out)

  def test_repeat_interleave_tensor_backward_variable_repeats(self):
    # Variable repeats with autograd go through
    # `repeat_interleave(repeats) -> index_select`.
    t = torch.randn(3, 4, dtype=torch.float32)
    repeats = torch.tensor([2, 1, 3])
    grad = torch.randn(6, 4, dtype=torch.float32)

    def _backward(device):
      x = to(t.clone(), device=device).requires_grad_(True)
      out = torch.repeat_interleave(x, to(repeats, device=device), dim=0)
      out.backward(to(grad, device=device))
      return x.grad

    self.assert_close_tpu_vs_cpu(_backward)

  @parameterized.named_parameters(
      dict(
          testcase_name="constant_output_size_no_grad",
          fn=lambda x, r: torch.repeat_interleave(x, r, dim=0, output_size=6),
          x_dynamic=(True, True),
          repeats=[2],
          no_grad=True,
      ),
      dict(
          testcase_name="constant_output_size_requires_grad",
          fn=lambda x, r: torch.repeat_interleave(x, r, dim=0, output_size=6),
          x_dynamic=(True, True),
          repeats=[2],
          requires_grad=True,
      ),
      dict(
          testcase_name="constant_output_size_no_dim",
          fn=lambda x, r: torch.repeat_interleave(x, r, output_size=24),
          x_dynamic=(True, True),
          repeats=[2],
      ),
      dict(
          testcase_name="symbolic_output_size",
          fn=lambda x, r: torch.repeat_interleave(
              x, r, dim=0, output_size=x.shape[0] * 2
          ),
          x_dynamic=(True, True),
          repeats=[2],
      ),
      dict(
          testcase_name="dynamic_repeats_length",
          fn=lambda x, r: torch.repeat_interleave(
              x, r, dim=0, output_size=x.shape[0] * 2
          ),
          x_dynamic=(True, False),
          repeats=[2, 2, 2],
      ),
      dict(
          testcase_name="no_output_size",
          fn=lambda x, r: torch.repeat_interleave(x, r, dim=0),
          x_dynamic=(True, True),
          repeats=[2],
      ),
  )
  def test_repeat_interleave_tensor_symbolic_shapes(
      self, fn, x_dynamic, repeats, no_grad=False, requires_grad=False
  ):
    # `repeat_interleave.self_Tensor` must neither throw on nor specialize the
    # symbolic dims of `self`, `repeats` and `output_size`. CPU is the control.
    for device in ("cpu", "tpu"):
      with self.subTest(device=device):
        x = torch.randn(3, 4, device=device, requires_grad=requires_grad)
        r = torch.tensor(repeats, dtype=torch.int64, device=device)
        specialized = run_with_dynamic_dims(
            fn,
            [(x, x_dynamic), (r, (len(repeats) > 1,))],
            no_grad=no_grad,
        )
        self.assertEmpty(specialized)

  @parameterized.named_parameters(
      ("one_element_repeats", False),
      ("per_row_repeats", True),
  )
  def test_repeat_interleave_tensor_symbolic_output_size_compile(
      self, per_row_repeats
  ):
    # Automatic dynamic shapes compile one static and one dynamic frame. A
    # specialized `output_size` would recompile for every new size.
    def fn(x, r):
      return torch.repeat_interleave(x, r, dim=0, output_size=x.shape[0] * 2)

    self.addCleanup(torch._dynamo.reset)
    for device in ("cpu", "tpu"):
      with self.subTest(device=device):
        torch._dynamo.reset()
        counter = dynamo_testing.CompileCounter()
        compiled_fn = torch.compile(fn, backend=counter, fullgraph=True)
        for n in (3, 5, 7):
          x = torch.randn(n, 4, device=device)
          r = torch.full(
              (n if per_row_repeats else 1,),
              2,
              dtype=torch.int64,
              device=device,
          )
          utils.assert_close(
              compiled_fn(x, r).cpu(),
              torch.repeat_interleave(x.cpu(), r.cpu(), dim=0),
          )
        self.assertEqual(counter.frame_count, 2)

  def test_tensor_to_tpu_with_default_device(self):
    """Tests that a tensor can be moved to TPU with a default device set."""
    device = torch.device("tpu")
    with torch.device(device):
      a = torch.tensor(1)
      # This should not raise an error.
      a.to(device=device, dtype=torch.float)

  def test_to_tpu_with_dtype(self):
    device = torch.device("tpu")
    cpu_device = torch.device("cpu")
    x = torch.arange(0, 1024, 2, dtype=torch.int64).to(cpu_device)
    x_tpu = x.to(device)
    x_none_with_dtype_from_cpu = x.to(None, dtype=torch.float32)
    x_none_with_dtype_from_tpu = x_tpu.to(None, dtype=torch.float32).cpu()
    self.assertEqual(x_none_with_dtype_from_cpu, x_none_with_dtype_from_tpu)

  def test_chained_ops_and_views(self):
    dtype = torch.float32
    tpu_device = torch.device("tpu")

    b, c, h, w = 2, 12, 16, 16  # C == K_mid
    k_mid, n_final = c, 8

    golden_a = torch.randn((b, c, h, w), dtype=dtype, device=self.golden_device)
    golden_b = torch.randn(
        (b, 1, h, w), dtype=dtype, device=self.golden_device
    )  # broadcasts over C
    golden_d = torch.randn((b, c, h, w), dtype=dtype, device=self.golden_device)
    golden_e_matmul = torch.randn(
        (k_mid, n_final), dtype=dtype, device=self.golden_device
    )

    alpha1, scalar1 = 0.7, 1.5

    # ---------- the golden device path ----------
    golden_y = (golden_a + alpha1 * golden_b) * scalar1  # (B,C,H,W)
    golden_y = golden_y.view(b, c, h * w)  # (B,C,256)
    golden_y = golden_y[:, :, h * w // 4 : h * w // 2]  # keep len 64
    golden_y = golden_y.permute(0, 2, 1).contiguous()  # (B,64,C)

    rows = golden_y.numel() // k_mid  # B*64
    golden_y = golden_y.view(rows, k_mid)  # (rows,K_mid)
    golden_y += golden_d.view(-1, c).mean(dim=0)  # broadcast add

    # Build a 2×-wider buffer so we can step by 2 in the col stride.
    width = k_mid * 2  # 24
    golden_buf = torch.arange(
        rows * width, dtype=dtype, device=self.golden_device
    ).view(rows, width)
    # TODO(haifengj): Use as_strided() when the op is implemented.
    # golden_x = torch.as_strided(
    #     golden_buf, (rows, K_mid), (golden_buf.stride()[0], 2)
    # )  # (rows,K_mid)
    golden_x = golden_buf[:, ::2]  # (rows,K_mid)

    golden_result = golden_x.matmul(golden_e_matmul)  # (rows,N_final)

    # ---------- TPU path ----------
    tpu_a, tpu_b = to(golden_a, tpu_device), to(golden_b, tpu_device)
    tpu_d = to(golden_d, tpu_device)
    tpu_e = to(golden_e_matmul, tpu_device)

    tpu_y = (tpu_a + alpha1 * tpu_b) * scalar1  # (B,C,H,W)
    tpu_y = tpu_y.view(b, c, h * w)[:, :, h * w // 4 : h * w // 2]
    tpu_y = tpu_y.permute(0, 2, 1).contiguous().view(rows, k_mid)
    tpu_y += tpu_d.view(-1, c).mean(dim=0)

    tpu_buf = to(golden_buf, tpu_device)
    # TODO(haifengj): Use as_strided() when the op is implemented.
    # tpu_x = torch.as_strided(tpu_buf, (rows, K_mid), (tpu_buf.stride()[0], 2))
    tpu_x = tpu_buf[:, ::2]

    tpu_result = tpu_x.matmul(tpu_e).cpu()

    self.assert_close(
        golden_result=golden_result,
        torch_tpu_result=tpu_result,
        rtol=4e-2,
        atol=6e1,
    )

  def test_default_dtype_change_after_deferred_op(self):
    """Tests that the default dtype is captured when the op is enqueued."""
    with set_default_dtype(torch.float32):
      arg = torch.tensor(2, dtype=torch.int32, device=torch.device("tpu"))
      # Enqueue log2 op. The expected output type is float32.
      res = torch.log2(arg)
      # The tensor's dtype is set at the time of op creation.
      self.assertEqual(res.dtype, torch.float32)

      # Change the default dtype *after* the op is enqueued.
      with set_default_dtype(torch.float16):
        # Materialization happens here. Even though the *current* default dtype
        # (float16) has changed, the one captured when the op was enqueued
        # (float32) is used, as if the op had been executed immediately when
        # the default dtype was float32.
        res.to("cpu")

  @parameterized.product(dtype=[torch.float32, torch.bfloat16])
  def test_hardsigmoid(self, dtype):
    # Generate deterministic inputs spanning the boundaries (-3 and 3)
    input_value = torch.linspace(-6, 6, 16, dtype=dtype).reshape(4, 4)

    def compute(device):
      x = input_value.clone().detach().to(device).requires_grad_(True)
      y = torch.nn.functional.hardsigmoid(x)
      y.sum().backward()
      return y, x.grad

    if dtype == torch.bfloat16:
      self.assert_close_tpu_vs_cpu(compute, atol=2e-2, rtol=1e-2)
    else:
      self.assert_close_tpu_vs_cpu(compute)

  @parameterized.product(dtype=[torch.float32, torch.bfloat16])
  def test_hardsigmoid_inplace(self, dtype):
    # Generate deterministic inputs spanning the boundaries (-3 and 3)
    input_value = torch.linspace(-6, 6, 16, dtype=dtype).reshape(4, 4)

    def compute(device):
      x = input_value.clone().detach().to(device)
      y = torch.nn.functional.hardsigmoid(x, inplace=True)
      return y, x

    if dtype == torch.bfloat16:
      self.assert_close_tpu_vs_cpu(compute, atol=2e-2, rtol=1e-2)
    else:
      self.assert_close_tpu_vs_cpu(compute)

  @parameterized.product(dtype=[torch.float32, torch.bfloat16])
  def test_hardsigmoid_out(self, dtype):
    # Generate deterministic inputs spanning the boundaries (-3 and 3)
    input_value = torch.linspace(-6, 6, 16, dtype=dtype).reshape(4, 4)

    def compute(device):
      x = input_value.clone().detach().to(device)
      out = torch.empty(4, 4, dtype=dtype, device=device)
      # Call the ATen out variant directly
      torch.ops.aten.hardsigmoid.out(x, out=out)
      return out

    if dtype == torch.bfloat16:
      self.assert_close_tpu_vs_cpu(compute, atol=2e-2, rtol=1e-2)
    else:
      self.assert_close_tpu_vs_cpu(compute)

  def test_hardsigmoid_backward_boundary(self):
    """Tests hardsigmoid backward at the boundary x = -3.0 and x = 3.0."""

    def compute(device):
      x = torch.tensor([-3.0, 3.0], device=device, requires_grad=True)
      y = torch.nn.functional.hardsigmoid(x)
      y.sum().backward()
      return x.grad

    self.assert_close_tpu_vs_cpu(compute)

  def test_hardsigmoid_backward_nan(self):
    """Tests hardsigmoid backward returns 0.0 for NaN and Inf inputs."""

    def compute(device) -> torch.Tensor:
      x = torch.tensor(
          [
              float("nan"),
              -4.0,
              -3.0,
              0.0,
              3.0,
              4.0,
              float("inf"),
              float("-inf"),
          ],
          device=device,
          requires_grad=True,
      )
      y = torch.nn.functional.hardsigmoid(x)
      y.backward(torch.full_like(x, 2.0))
      return x.grad

    self.assert_close_tpu_vs_cpu(compute)

    def compute_direct(device) -> torch.Tensor:
      x = torch.tensor(
          [
              float("nan"),
              -4.0,
              -3.0,
              0.0,
              3.0,
              4.0,
              float("inf"),
              float("-inf"),
          ],
          device=device,
      )
      grad_output = torch.full_like(x, 2.0)
      return torch.ops.aten.hardsigmoid_backward(grad_output, x)

    self.assert_close_tpu_vs_cpu(compute_direct)

  def test_hardsigmoid_backward_broadcast(self):
    """Tests hardsigmoid backward with broadcastable inputs."""

    def compute_grad_output_broadcast(device) -> torch.Tensor:
      grad_output = torch.tensor([2.0], device=device)
      self_tensor = torch.tensor([-4.0, 0.0, 4.0], device=device)
      grad_input = torch.empty(3, device=device)
      torch.ops.aten.hardsigmoid_backward.grad_input(
          grad_output, self_tensor, grad_input=grad_input
      )
      return grad_input

    self.assert_close_tpu_vs_cpu(compute_grad_output_broadcast)

    def compute_self_broadcast(device) -> torch.Tensor:
      grad_output = torch.tensor([1.0, 2.0, 3.0], device=device)
      self_tensor = torch.tensor([0.0], device=device)
      grad_input = torch.empty(3, device=device)
      torch.ops.aten.hardsigmoid_backward.grad_input(
          grad_output, self_tensor, grad_input=grad_input
      )
      return grad_input

    self.assert_close_tpu_vs_cpu(compute_self_broadcast)

    def compute_multidimensional(device) -> torch.Tensor:
      grad_output = torch.tensor([[1.0], [2.0]], device=device)
      self_tensor = torch.tensor([[-4.0, 0.0, 4.0]], device=device)
      return torch.ops.aten.hardsigmoid_backward(grad_output, self_tensor)

    self.assert_close_tpu_vs_cpu(compute_multidimensional)

    def compute_mixed_dtype(device) -> torch.Tensor:
      grad_output = torch.tensor([2.0], dtype=torch.float32, device=device)
      self_tensor = torch.tensor(
          [0.0, 1.0], dtype=torch.bfloat16, device=device
      )
      return torch.ops.aten.hardsigmoid_backward(grad_output, self_tensor)

    self.assert_close_tpu_vs_cpu(compute_mixed_dtype)

  @parameterized.product(dtype=[torch.float32, torch.bfloat16])
  def test_hardswish(self, dtype):
    # Scale by 5 to have coverage of the x < -3 and x >= 3 piecewise branches
    input_value = torch.randn(4, 4, dtype=dtype) * 5

    def compute(device):
      x = input_value.clone().detach().to(device).requires_grad_(True)
      y = torch.nn.functional.hardswish(x)
      y.sum().backward()
      return y, x.grad

    if dtype == torch.bfloat16:
      self.assert_close_tpu_vs_cpu(compute, atol=2e-2, rtol=1e-2)
    else:
      self.assert_close_tpu_vs_cpu(compute)

  @parameterized.product(dtype=[torch.float32, torch.bfloat16])
  def test_hardswish_inplace(self, dtype):
    # Scale by 5 to have coverage of the x < -3 and x >= 3 piecewise branches
    input_value = torch.randn(4, 4, dtype=dtype) * 5

    def compute(device):
      x = input_value.clone().detach().to(device)
      y = torch.nn.functional.hardswish(x, inplace=True)
      return y, x

    if dtype == torch.bfloat16:
      self.assert_close_tpu_vs_cpu(compute, atol=2e-2, rtol=1e-2)
    else:
      self.assert_close_tpu_vs_cpu(compute)

  @parameterized.product(dtype=[torch.float32, torch.bfloat16])
  def test_hardswish_out(self, dtype):
    # Scale by 5 to have coverage of the x < -3 and x >= 3 piecewise branches
    input_value = torch.randn(4, 4, dtype=dtype) * 5

    def compute(device):
      x = input_value.clone().detach().to(device)
      out = torch.empty(4, 4, dtype=dtype, device=device)
      # Call the ATen out variant directly
      torch.ops.aten.hardswish.out(x, out=out)
      return out

    if dtype == torch.bfloat16:
      self.assert_close_tpu_vs_cpu(compute, atol=2e-2, rtol=1e-2)
    else:
      self.assert_close_tpu_vs_cpu(compute)

  def test_hardswish_backward_boundary(self):
    """Tests hardswish backward at the boundaries x = -3.0 / x = 3.0."""

    def compute(device):
      x = torch.tensor([-3.0, 3.0], device=device, requires_grad=True)
      y = torch.nn.functional.hardswish(x)
      y.sum().backward()
      return x.grad

    self.assert_close_tpu_vs_cpu(compute)

  def test_default_dtype_consistent(self):
    """Tests the torch_tpu respects the default dtype."""
    with set_default_dtype(torch.float32):
      arg = torch.tensor(2, dtype=torch.int32, device=torch.device("tpu"))
      res = torch.log2(arg)
      self.assertEqual(res.dtype, torch.float32)
      # This should not raise an error.
      res.to("cpu")
      self.assertEqual(res.dtype, torch.float32)

  def test_embedding_scalar_index(self):
    """Tests that embedding works with a scalar index in forward and backward."""
    with set_default_dtype(torch.float32):
      vocab_size = 3
      embedding_size = 7
      embedding_table = torch.randn(vocab_size, embedding_size)
      index = torch.randint(0, vocab_size, ())
      self.assert_close_tpu_vs_cpu(
          lambda device: torch.nn.functional.embedding(
              to(index, device=device),
              to(embedding_table, device=device),
          )
      )

      def compute_grad(device):
        w = to(embedding_table.clone(), device=device).requires_grad_(True)
        idx = to(index, device=device)
        out = torch.nn.functional.embedding(idx, w)
        loss = out.sum()
        loss.backward()
        return w.grad

      self.assert_close_tpu_vs_cpu(compute_grad)

  def test_randn_scalar(self):
    with set_default_dtype(torch.float32):
      torch.manual_seed(46)
      x = torch.randn((), device=torch.device("tpu"))
      y = torch.randn((), device=torch.device("tpu"))
      torch.manual_seed(47)
      z = torch.randn((), device=torch.device("tpu"))
      torch.manual_seed(46)
      w = torch.randn((), device=torch.device("tpu"))
      with self.subTest("same_seed_same_result"):
        self.assertEqual(x, w)

      with self.subTest("different_call_different_result"):
        self.assertNotEqual(x, y)

      with self.subTest("different_seed_different_result"):
        self.assertNotEqual(x, z)

  def test_randn_isotropy(self):
    with set_default_dtype(torch.float32):
      torch.manual_seed(46)
      n = 1000
      x = torch.randn((n, n), device=torch.device("tpu"))
      y = torch.randn((n, 1), device=torch.device("tpu"))
      count = (x @ y > 0).sum()
      # Count should be close to N/2.
      n_tensor = torch.tensor(
          n, dtype=torch.float32, device=torch.device("tpu")
      )
      self.assertGreater(count, n_tensor / 2 - 3 * torch.sqrt(n_tensor) / 2)
      self.assertLess(count, n_tensor / 2 + 3 * torch.sqrt(n_tensor) / 2)

  def test_randn_magnitude(self):
    with set_default_dtype(torch.float32):
      torch.manual_seed(48)
      n = 1000
      x = torch.randn((n, n), device=torch.device("tpu"))
      norm_squared = (x**2).sum(dim=1)
      mean_sq_norm = torch.mean(norm_squared).item()
      self.assertGreater(mean_sq_norm, n - 6)
      self.assertLess(mean_sq_norm, n + 6)

  @parameterized.named_parameters(
      ("scalar", ()),
      ("1d", (10,)),
      ("all_odds", (3, 5)),
      ("3d_1", (10, 5, 6)),
      ("3d_2", (11, 5, 6)),
  )
  def test_randn_shape(self, shape):
    with set_default_dtype(torch.float32):
      x = torch.randn(shape, device=torch.device("tpu"))
      self.assertEqual(x.shape, shape)

  @parameterized.named_parameters(
      ("both_scalar", 3.0, 9.0),
      ("mean_scalar_only", 3.0, torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
      ("std_scalar_only", torch.tensor([[0.0, 1.0], [2.0, 3.0]]), 9.0),
      (
          "none_scalar",
          torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
          torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
      ),
  )
  def test_randn_broadcast(self, mean, std):
    with set_default_dtype(torch.float32):
      torch.manual_seed(49)
      if isinstance(mean, torch.Tensor):
        mean = mean.to(torch.device("tpu"))
      if isinstance(std, torch.Tensor):
        std = std.to(torch.device("tpu"))
      if isinstance(mean, float) and isinstance(std, float):
        x = torch.normal(mean, std, (3, 3), device=torch.device("tpu"))
      else:
        x = torch.normal(mean, std)
      self.assertIsInstance(x, torch.Tensor)

  def test_randn_gaussianity(self):
    with set_default_dtype(torch.float32):
      samples = torch.randn((10000,), device=torch.device("tpu")).cpu()

      # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
      with self.subTest("shapiro_wilk_test"):
        _, p_value = stats.shapiro(samples)
        self.assertGreater(p_value, 0.01)

      # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
      with self.subTest("d_agostino_k_squared_test"):
        _, p = stats.normaltest(samples)
        self.assertGreater(p, 0.01)

  @parameterized.named_parameters(
      ("bfloat16", torch.bfloat16),
      ("float16", torch.float16),
      ("float32", torch.float32),
  )
  def test_randn_float(self, dtype: torch.dtype):
    numel = 100_000
    t = torch.randn(numel, dtype=dtype, device=torch.device("tpu"))
    t_cpu = t.cpu().float()

    num_nonfinite = (~torch.isfinite(t_cpu)).sum().item()
    self.assertEqual(
        num_nonfinite,
        0,
        f"randn {dtype}: {num_nonfinite}/{numel} non-finite values",
    )

    mean = t_cpu.mean().item()
    std = t_cpu.std().item()
    self.assertAlmostEqual(
        mean, 0.0, delta=0.05, msg=f"randn {dtype} mean={mean}, expected ~0.0"
    )
    self.assertAlmostEqual(
        std, 1.0, delta=0.05, msg=f"randn {dtype} std={std}, expected ~1.0"
    )

  @parameterized.named_parameters(
      ("complex64", torch.complex64),
      ("complex128", torch.complex128),
  )
  def test_randn_complex(self, dtype: torch.dtype):
    numel = 100_000
    t = torch.randn(numel, dtype=dtype, device=torch.device("tpu"))
    t_cpu = t.cpu()

    num_nonfinite = (~torch.isfinite(t_cpu)).sum().item()
    self.assertEqual(
        num_nonfinite,
        0,
        f"randn {dtype}: {num_nonfinite}/{numel} non-finite values",
    )

    real_mean = t_cpu.real.float().mean().item()
    real_std = t_cpu.real.float().std().item()
    imag_mean = t_cpu.imag.float().mean().item()
    imag_std = t_cpu.imag.float().std().item()

    self.assertAlmostEqual(
        real_mean,
        0.0,
        delta=0.05,
        msg=f"randn {dtype} real mean={real_mean}, expected ~0.0",
    )
    self.assertAlmostEqual(
        real_std,
        0.707,
        delta=0.05,
        msg=f"randn {dtype} real std={real_std}, expected ~0.707",
    )
    self.assertAlmostEqual(
        imag_mean,
        0.0,
        delta=0.05,
        msg=f"randn {dtype} imag mean={imag_mean}, expected ~0.0",
    )
    self.assertAlmostEqual(
        imag_std,
        0.707,
        delta=0.05,
        msg=f"randn {dtype} imag std={imag_std}, expected ~0.707",
    )

  def test_embedding_vector_indices(self):
    """Tests that embedding works with a vector of indices."""
    with set_default_dtype(torch.float32):
      vocab_size = 3
      embedding_size = 7
      embedding_table = torch.randn(vocab_size, embedding_size)
      indices = torch.randint(0, vocab_size, size=(10,))

      self.assert_close_tpu_vs_cpu(
          lambda device: torch.nn.functional.embedding(
              to(indices, device=device),
              to(embedding_table, device=device),
          )
      )

  def test_embedding_5D_tensor_indices(self):
    """Tests that embedding works with a 5D tensor of indices."""
    with set_default_dtype(torch.float32):
      vocab_size = 10_000
      embedding_size = 2**10
      embedding_table = torch.randn(vocab_size, embedding_size)
      indices = torch.randint(0, vocab_size, size=(3, 5, 7, 11, 32))

      self.assert_close_tpu_vs_cpu(
          lambda device: torch.nn.functional.embedding(
              to(indices, device=device),
              to(embedding_table, device=device),
          )
      )

  def test_embedding_with_2d_indices(self):
    """Tests that embedding works with a 5D tensor of indices."""

    def fn(device):
      table = torch.tensor(
          [
              [8.5625, -10.5625, -1.8281, 7.5938, 6.6250],
              [-12.9375, -6.1875, 15.1875, 13.0625, -0.4219],
              [2.6719, 4.7812, 13.9375, -2.3906, -1.4062],
              [13.2500, -2.3906, 8.8750, -14.3750, 14.0625],
              [-14.7500, 2.8125, -14.6250, 16.2500, 14.1875],
          ],
          dtype=torch.bfloat16,
          device=device,
      )
      indices = torch.tensor([[0, 2], [1, 2]], device=device)

      out = torch.nn.functional.embedding(indices, table, max_norm=1.0)
      return out

    def fn_1(device):
      table = torch.tensor(
          [
              [-10.5503, -7.3657, -17.6449, -10.3086, -10.0450],
              [-6.8466, -13.8767, 5.4327, 8.9219, 2.2766],
              [-16.0622, 13.1197, 14.9397, 2.4518, 8.4597],
              [-4.8560, 11.6760, -9.6373, 11.8647, 9.1323],
              [-15.0545, 11.5629, 5.4394, -10.2997, 9.2878],
          ],
          device=device,
      )
      indices = torch.tensor([[0, 2], [1, 2]], device=device)

      out = torch.nn.functional.embedding(
          indices, table, max_norm=1.0, norm_type=1.0
      )
      return out

    fn(torch.device("tpu"))
    fn_1(torch.device("tpu"))

  @parameterized.named_parameters(
      ("float32", torch.float32),
      ("bfloat16", torch.bfloat16),
  )
  def test_embedding_aten(self, dtype):
    """Tests native aten.embedding forward and backward."""
    vocab_size = 32
    embedding_dim = 16
    batch_size = 4
    seq_len = 8

    indices = torch.randint(0, vocab_size, size=(batch_size, seq_len))
    weight = torch.randn(vocab_size, embedding_dim, dtype=dtype)

    # Test direct aten.embedding forward
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ops.aten.embedding(
            to(weight, device=device),
            to(indices, device=device),
        )
    )

    # Test backward gradient
    def compute_grad(device):
      w = to(weight.clone(), device=device).requires_grad_(True)
      idx = to(indices, device=device)
      out = torch.ops.aten.embedding(w, idx)
      loss = out.sum()
      loss.backward()
      return w.grad

    self.assert_close_tpu_vs_cpu(compute_grad)

  @parameterized.named_parameters(
      ("float32", torch.float32),
      ("bfloat16", torch.bfloat16),
  )
  def test_embedding_out(self, dtype):
    """Tests native aten.embedding.out with pre-allocated and resizing out tensor."""
    vocab_size = 32
    embedding_dim = 16
    batch_size = 4
    seq_len = 8

    indices = torch.randint(0, vocab_size, size=(batch_size, seq_len))
    weight = torch.randn(vocab_size, embedding_dim, dtype=dtype)

    # 1. Matching pre-allocated out shape
    def run_preallocated(device):
      w = to(weight, device=device)
      idx = to(indices, device=device)
      out = torch.empty(
          batch_size, seq_len, embedding_dim, dtype=dtype, device=device
      )
      res = torch.ops.aten.embedding.out(w, idx, out=out)
      self.assertIs(res, out)
      return out

    self.assert_close_tpu_vs_cpu(run_preallocated)

    # 2. Differing / empty out shape requiring resize
    def run_resize(device):
      w = to(weight, device=device)
      idx = to(indices, device=device)
      out = torch.empty(0, dtype=dtype, device=device)
      res = torch.ops.aten.embedding.out(w, idx, out=out)
      self.assertIs(res, out)
      return out

    self.assert_close_tpu_vs_cpu(run_resize)

    # 3. 0-D scalar index with out tensor
    scalar_idx = torch.tensor(3, dtype=torch.long)

    def run_scalar(device):
      w = to(weight, device=device)
      idx = to(scalar_idx, device=device)
      out = torch.empty(embedding_dim, dtype=dtype, device=device)
      res = torch.ops.aten.embedding.out(w, idx, out=out)
      self.assertIs(res, out)
      return out

    self.assert_close_tpu_vs_cpu(run_scalar)

  @parameterized.named_parameters(
      ("float32", torch.float32),
      ("bfloat16", torch.bfloat16),
      ("float16", torch.float16),
  )
  def test_embedding_sparse(self, dtype):
    """Tests aten.embedding and aten.embedding.out with sparse=True."""
    vocab_size = 16
    embedding_dim = 8
    # Include repeated indices and padding_idx=2 to test uncoalesced sparse
    # gradients and padding masking.
    indices = torch.tensor([[1, 2, 5, 1], [5, 0, 2, 5]], dtype=torch.long)
    weight = torch.randn(vocab_size, embedding_dim, dtype=dtype)
    grad_out = torch.randn(2, 4, embedding_dim, dtype=dtype)
    tol = 2e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-5

    # 1. Forward and backward with padding_idx and repeated indices
    def run_sparse_fwd_bwd(device):
      w = to(weight.clone(), device=device).requires_grad_(True)
      idx = to(indices, device=device)
      go = to(grad_out, device=device)
      out = torch.nn.functional.embedding(idx, w, padding_idx=2, sparse=True)
      out.backward(go)
      self.assertTrue(w.grad.is_sparse)
      self.assertEqual(w.grad.layout, torch.sparse_coo)
      # Verify both to_dense() and coalesce() work on the sparse gradient
      dense_from_to_dense = w.grad.to_dense()
      dense_from_coalesce = w.grad.coalesce().to_dense()
      self.assertEqual(
          dense_from_to_dense, dense_from_coalesce, atol=tol, rtol=tol
      )
      return out, dense_from_to_dense

    tol_opt = 2e-2 if dtype in (torch.bfloat16, torch.float16) else None
    self.assert_close_tpu_vs_cpu(run_sparse_fwd_bwd, atol=tol_opt, rtol=tol_opt)

    # 2. Multiple backward passes (sparse + sparse gradient accumulation)
    # and dense parameter update (dense + sparse add_ / sub_)
    def run_accum_and_step(device):
      # PyTorch CPU does not implement cadd_sparse for Half (float16), so
      # compute the CPU reference in float32 and cast back to float16.
      is_cpu_fp16 = (
          torch.device(device).type == "cpu" and dtype == torch.float16
      )
      compute_dtype = torch.float32 if is_cpu_fp16 else dtype
      w = (
          to(weight.clone(), device=device)
          .to(compute_dtype)
          .requires_grad_(True)
      )
      idx = to(indices, device=device)
      go1 = to(grad_out, device=device).to(compute_dtype)
      go2 = to(grad_out * 0.5, device=device).to(compute_dtype)
      out1 = torch.nn.functional.embedding(idx, w, padding_idx=2, sparse=True)
      out1.backward(go1)
      out2 = torch.nn.functional.embedding(idx, w, padding_idx=2, sparse=True)
      out2.backward(go2)
      self.assertTrue(w.grad.is_sparse)
      grad_dense = w.grad.to_dense()
      # Simulate optimizer step: dense.add_(sparse, alpha=-lr)
      with torch.no_grad():
        w.add_(w.grad, alpha=-0.1)
      return grad_dense.to(dtype), w.detach().to(dtype)

    self.assert_close_tpu_vs_cpu(run_accum_and_step, atol=tol_opt, rtol=tol_opt)

    # 3. aten.embedding.out with sparse=True
    def run_sparse_out(device):
      w = to(weight, device=device)
      idx = to(indices, device=device)
      out = torch.empty(2, 4, embedding_dim, dtype=dtype, device=device)
      res = torch.ops.aten.embedding.out(
          w, idx, padding_idx=2, sparse=True, out=out
      )
      self.assertIs(res, out)
      return out

    self.assert_close_tpu_vs_cpu(run_sparse_out)

    # 4. End-to-end nn.Embedding(sparse=True) with torch.optim.SGD
    def run_sgd_steps(device):
      is_cpu_fp16 = (
          torch.device(device).type == "cpu" and dtype == torch.float16
      )
      compute_dtype = torch.float32 if is_cpu_fp16 else dtype
      emb = torch.nn.Embedding(
          vocab_size,
          embedding_dim,
          padding_idx=2,
          sparse=True,
          device=device,
          dtype=compute_dtype,
      )
      with torch.no_grad():
        emb.weight.copy_(to(weight, device=device).to(compute_dtype))
      opt = torch.optim.SGD(emb.parameters(), lr=0.1)
      idx = to(indices, device=device)
      go1 = to(grad_out, device=device).to(compute_dtype)
      go2 = to(grad_out * 0.5, device=device).to(compute_dtype)

      opt.zero_grad()
      out1 = emb(idx)
      out1.backward(go1)
      opt.step()

      opt.zero_grad()
      out2 = emb(idx)
      out2.backward(go2)
      opt.step()
      return emb.weight.detach().to(dtype)

    self.assert_close_tpu_vs_cpu(run_sgd_steps, atol=tol_opt, rtol=tol_opt)

  @parameterized.parameters(torch.float32, torch.bfloat16, torch.float16)
  def test_embedding_bag_sparse(self, dtype):
    """Tests nn.EmbeddingBag and aten._embedding_bag_backward with sparse=True."""
    vocab_size = 12
    embedding_dim = 4
    weight = torch.randn(vocab_size, embedding_dim, dtype=dtype)
    indices = torch.tensor([1, 2, 4, 2, 5, 1, 0, 2], dtype=torch.long)
    offsets = torch.tensor([0, 3, 5], dtype=torch.long)
    per_sample_weights = torch.tensor(
        [0.5, 1.5, 2.0, 0.8, 1.2, 0.3, 1.0, 0.7], dtype=dtype
    )
    grad_out = torch.randn(3, embedding_dim, dtype=dtype)
    tol = 1e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-5

    # 1. Forward + backward for mode='sum' (with/without padding_idx and per_sample_weights)
    for mode, psw in [
        ("sum", None),
        ("sum", per_sample_weights),
        ("mean", None),
    ]:
      for pad_idx in [None, 2]:

        def run_bag_sparse(device, mode=mode, psw=psw, pad_idx=pad_idx):
          is_cpu_fp16 = (
              torch.device(device).type == "cpu" and dtype == torch.float16
          )
          compute_dtype = torch.float32 if is_cpu_fp16 else dtype
          w = (
              to(weight, device=device)
              .to(compute_dtype)
              .detach()
              .requires_grad_(True)
          )
          idx = to(indices, device=device)
          off = to(offsets, device=device)
          psw_dev = (
              to(psw, device=device).to(compute_dtype)
              if psw is not None
              else None
          )
          go = to(grad_out, device=device).to(compute_dtype)
          out = torch.nn.functional.embedding_bag(
              idx,
              w,
              off,
              mode=mode,
              sparse=True,
              per_sample_weights=psw_dev,
              padding_idx=pad_idx,
          )
          out.backward(go)
          self.assertIsNotNone(w.grad)
          self.assertTrue(w.grad.is_sparse)
          return out.to(dtype), w.grad.to_dense().to(dtype)

        self.assert_close_tpu_vs_cpu(run_bag_sparse, atol=tol, rtol=tol)

    # 2. End-to-end nn.EmbeddingBag(sparse=True) multi-step SGD optimization
    def run_bag_sgd_steps(device):
      is_cpu_fp16 = (
          torch.device(device).type == "cpu" and dtype == torch.float16
      )
      compute_dtype = torch.float32 if is_cpu_fp16 else dtype
      bag = torch.nn.EmbeddingBag(
          vocab_size,
          embedding_dim,
          mode="sum",
          padding_idx=2,
          sparse=True,
          device=device,
          dtype=compute_dtype,
      )
      with torch.no_grad():
        bag.weight.copy_(to(weight, device=device).to(compute_dtype))
      opt = torch.optim.SGD(bag.parameters(), lr=0.1)
      idx = to(indices, device=device)
      off = to(offsets, device=device)
      go1 = to(grad_out, device=device).to(compute_dtype)
      go2 = to(grad_out * 0.5, device=device).to(compute_dtype)

      opt.zero_grad()
      out1 = bag(idx, off)
      out1.backward(go1)
      self.assertTrue(bag.weight.grad.is_sparse)
      opt.step()

      opt.zero_grad()
      out2 = bag(idx, off)
      out2.backward(go2)
      self.assertTrue(bag.weight.grad.is_sparse)
      opt.step()
      return bag.weight.detach().to(dtype)

    self.assert_close_tpu_vs_cpu(run_bag_sgd_steps, atol=tol, rtol=tol)

    # 3. Direct aten._embedding_bag_backward with empty offset2bag / empty indices
    def run_direct_sparse_bwd(device):
      is_cpu_fp16 = (
          torch.device(device).type == "cpu" and dtype == torch.float16
      )
      compute_dtype = torch.float32 if is_cpu_fp16 else dtype
      go = to(grad_out, device=device).to(compute_dtype)
      idx = to(indices, device=device)
      off = to(offsets, device=device)
      empty_long = torch.empty(0, dtype=torch.long, device=device)
      bwd_auto_offset2bag = torch.ops.aten._embedding_bag_backward(
          go,
          idx,
          off,
          empty_long,
          empty_long,
          empty_long,
          vocab_size,
          False,
          0,
          True,
          None,
          2,
      )
      self.assertTrue(bwd_auto_offset2bag.is_sparse)
      go_empty = torch.zeros(
          1, embedding_dim, device=device, dtype=compute_dtype
      )
      off_empty = torch.tensor([0], dtype=torch.long, device=device)
      bwd_empty = torch.ops.aten._embedding_bag_backward(
          go_empty,
          empty_long,
          off_empty,
          empty_long,
          empty_long,
          empty_long,
          vocab_size,
          False,
          0,
          True,
          None,
          -1,
      )
      self.assertTrue(bwd_empty.is_sparse)
      return (
          bwd_auto_offset2bag.to_dense().to(dtype),
          bwd_empty.to_dense().to(dtype),
      )

    self.assert_close_tpu_vs_cpu(run_direct_sparse_bwd, atol=tol, rtol=tol)

    # 5. Edge case: all indices equal to padding_idx (produces nnz=0 gradient)
    def run_all_padding_indices(device):
      is_cpu_fp16 = (
          torch.device(device).type == "cpu" and dtype == torch.float16
      )
      compute_dtype = torch.float32 if is_cpu_fp16 else dtype
      w = (
          to(weight.clone(), device=device)
          .to(compute_dtype)
          .requires_grad_(True)
      )
      pad_idx = torch.full((2, 3), 2, dtype=torch.long, device=device)
      go_pad = to(
          torch.randn(2, 3, embedding_dim, dtype=dtype), device=device
      ).to(compute_dtype)
      out = torch.nn.functional.embedding(
          pad_idx, w, padding_idx=2, sparse=True
      )
      out.backward(go_pad)
      self.assertTrue(w.grad.is_sparse)
      self.assertEqual(w.grad._nnz(), 0)
      dense_grad = w.grad.to_dense()
      coalesced_grad = w.grad.coalesce().to_dense()
      with torch.no_grad():
        w.add_(w.grad, alpha=-0.1)
      return (
          out.to(dtype),
          dense_grad.to(dtype),
          coalesced_grad.to(dtype),
          w.detach().to(dtype),
      )

    self.assert_close_tpu_vs_cpu(run_all_padding_indices)

  def test_sparse_coo_tensor_creation_and_metadata(self):
    """Tests creation, metadata inspection, copy, clone, zero_, and device/dtype conversions for SparsePrivateUse1 COO tensors."""
    indices_cpu = torch.tensor([[0, 2, 1], [1, 0, 2]], dtype=torch.long)
    values_cpu = torch.tensor(
        [[1.5, -2.0], [3.0, 4.5], [-1.0, 0.5]], dtype=torch.float32
    )
    size = (3, 4, 2)

    def run_creation_and_ops(device):
      sp = torch.sparse_coo_tensor(
          to(indices_cpu, device=device),
          to(values_cpu, device=device),
          size,
          device=device,
      )
      self.assertTrue(sp.is_sparse)
      self.assertEqual(sp.layout, torch.sparse_coo)
      self.assertEqual(sp.sparse_dim(), 2)
      self.assertEqual(sp.dense_dim(), 1)
      self.assertEqual(sp._nnz(), 3)
      self.assertFalse(sp.is_coalesced())

      # Test clone and to_dense
      sp_clone = sp.clone()
      self.assertTrue(sp_clone.is_sparse)
      self.assertEqual(sp_clone._nnz(), 3)

      # Test coalesce and coalesced accessors (indices() and values())
      sp_coal = sp.coalesce()
      self.assertTrue(sp_coal.is_coalesced())
      idx_coal = sp_coal.indices()
      val_coal = sp_coal.values()

      # Test copy_ between sparse tensors and zero_()
      sp_empty = torch.empty(size, layout=torch.sparse_coo, device=device)
      self.assertTrue(sp_empty.is_sparse)
      self.assertEqual(sp_empty._nnz(), 0)
      sp_empty.copy_(sp)
      dense_copied = sp_empty.to_dense()
      sp_empty.zero_()
      self.assertEqual(sp_empty._nnz(), 0)
      dense_zeroed = sp_empty.to_dense()

      # Test empty_like
      sp_like = torch.empty_like(sp)
      self.assertTrue(sp_like.is_sparse)
      self.assertEqual(sp_like.shape, sp.shape)

      # Test dtype conversion
      sp_bf16 = sp.to(torch.bfloat16)
      self.assertEqual(sp_bf16.dtype, torch.bfloat16)
      dense_bf16 = sp_bf16.to_dense().to(torch.float32)

      return (
          sp.to_dense(),
          sp_clone.to_dense(),
          idx_coal,
          val_coal,
          dense_copied,
          dense_zeroed,
          dense_bf16,
      )

    self.assert_close_tpu_vs_cpu(run_creation_and_ops)

  @parameterized.named_parameters(
      ("float32", torch.float32),
      ("bfloat16", torch.bfloat16),
      ("float16", torch.float16),
  )
  def test_sparse_coo_coalesce(self, dtype):
    """Tests _coalesce for 1D, 2D, hybrid, already-coalesced, and empty sparse COO tensors."""
    tol = 2e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-5
    tol_opt = 2e-2 if dtype in (torch.bfloat16, torch.float16) else None

    # 1. 1D sparse + 1D dense with duplicates and unsorted indices
    idx_1d = torch.tensor([[3, 1, 3, 0, 1]], dtype=torch.long)
    val_1d = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        dtype=dtype,
    )

    # 2. 2D sparse + 1D dense (hybrid) with duplicates and unsorted coordinates
    idx_2d = torch.tensor([[2, 0, 2, 1, 0], [1, 3, 1, 2, 3]], dtype=torch.long)
    val_2d = torch.tensor(
        [[1.0, -1.0], [2.0, 3.0], [4.0, 5.0], [-2.0, 0.5], [1.5, -0.5]],
        dtype=dtype,
    )

    def run_coalesce_cases(device):
      sp1 = torch.sparse_coo_tensor(
          to(idx_1d, device=device),
          to(val_1d, device=device),
          (5, 2),
          device=device,
      )
      self.assertFalse(sp1.is_coalesced())
      c1 = sp1.coalesce()
      self.assertTrue(c1.is_coalesced())
      self.assertEqual(c1._nnz(), 3)
      self.assertEqual(sp1.to_dense(), c1.to_dense(), atol=tol, rtol=tol)

      # Calling coalesce() on an already-coalesced tensor
      c1_again = c1.coalesce()
      self.assertTrue(c1_again.is_coalesced())

      sp2 = torch.sparse_coo_tensor(
          to(idx_2d, device=device),
          to(val_2d, device=device),
          (3, 4, 2),
          device=device,
      )
      c2 = sp2.coalesce()
      self.assertTrue(c2.is_coalesced())
      self.assertEqual(c2._nnz(), 3)
      self.assertEqual(sp2.to_dense(), c2.to_dense(), atol=tol, rtol=tol)

      # Empty sparse tensor coalesce
      sp_empty = torch.sparse_coo_tensor(
          torch.empty((2, 0), dtype=torch.long, device=device),
          torch.empty((0, 2), dtype=dtype, device=device),
          (3, 4, 2),
          device=device,
      )
      c_empty = sp_empty.coalesce()
      self.assertTrue(c_empty.is_coalesced())
      self.assertEqual(c_empty._nnz(), 0)

      return (
          c1.indices(),
          c1.values(),
          c2.indices(),
          c2.values(),
          c_empty.to_dense(),
      )

    self.assert_close_tpu_vs_cpu(run_coalesce_cases, atol=tol_opt, rtol=tol_opt)

  @parameterized.named_parameters(
      ("float32", torch.float32),
      ("bfloat16", torch.bfloat16),
      ("float16", torch.float16),
  )
  def test_sparse_coo_add_sub(self, dtype):
    """Tests dense + sparse, sparse + sparse addition/subtraction, non-contiguous out, and empty operand edge cases."""
    tol_opt = 2e-2 if dtype in (torch.bfloat16, torch.float16) else None

    dense_base = torch.randn(4, 3, dtype=dtype)
    idx1 = torch.tensor([[2, 0, 2]], dtype=torch.long)
    val1 = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-0.5, 1.5, -2.0]], dtype=dtype
    )
    idx2 = torch.tensor([[1, 2]], dtype=torch.long)
    val2 = torch.tensor([[0.5, -1.0, 2.0], [3.0, 0.0, -1.0]], dtype=dtype)

    # 2D sparse + 1D dense
    dense_3d_base = torch.randn(3, 4, 2, dtype=dtype)
    idx_2d = torch.tensor([[2, 0, 2], [1, 3, 1]], dtype=torch.long)
    val_2d = torch.tensor([[1.0, -1.0], [2.0, 3.0], [0.5, 1.5]], dtype=dtype)

    def run_add_sub_cases(device):
      is_cpu_fp16 = (
          torch.device(device).type == "cpu" and dtype == torch.float16
      )
      compute_dtype = torch.float32 if is_cpu_fp16 else dtype
      d = to(dense_base.clone(), device=device).to(compute_dtype)
      sp1 = torch.sparse_coo_tensor(
          to(idx1, device=device),
          to(val1, device=device).to(compute_dtype),
          (4, 3),
          device=device,
      )
      sp2 = torch.sparse_coo_tensor(
          to(idx2, device=device),
          to(val2, device=device).to(compute_dtype),
          (4, 3),
          device=device,
      )
      sp_empty = torch.sparse_coo_tensor(
          torch.empty((1, 0), dtype=torch.long, device=device),
          torch.empty((0, 3), dtype=compute_dtype, device=device),
          (4, 3),
          device=device,
      )

      # 1. dense + sparse (out-of-place, in-place, and with alpha)
      res_dense_add = torch.add(d, sp1, alpha=0.5)
      res_dense_sub = torch.sub(d, sp1, alpha=0.5)
      d_inplace = d.clone()
      d_inplace.add_(sp1, alpha=-0.25)
      d_sub_inplace = d.clone()
      d_sub_inplace.sub_(sp2, alpha=0.5)

      # 2. Out-of-place add into distinct output tensor and non-contiguous output tensor
      out_distinct = torch.empty_like(d)
      torch.add(d, sp1, alpha=1.0, out=out_distinct)

      d_transposed_base = to(torch.randn(3, 4, dtype=dtype), device=device).to(
          compute_dtype
      )
      non_contig_out = d_transposed_base.t()  # shape (4, 3), non-contiguous
      self.assertFalse(non_contig_out.is_contiguous())
      torch.add(d, sp1, alpha=0.5, out=non_contig_out)

      # 3. 2D sparse + 3D dense
      d3 = to(dense_3d_base.clone(), device=device).to(compute_dtype)
      sp_2d = torch.sparse_coo_tensor(
          to(idx_2d, device=device),
          to(val_2d, device=device).to(compute_dtype),
          (3, 4, 2),
          device=device,
      )
      res_3d = d3.add(sp_2d, alpha=-0.5)

      # 4. sparse + sparse and sparse - sparse
      sp_sum = torch.add(sp1, sp2, alpha=0.5)
      sp_diff = torch.sub(sp1, sp2, alpha=0.5)
      self.assertTrue(sp_sum.is_sparse)
      self.assertTrue(sp_diff.is_sparse)

      # 5. Empty operand edge cases for sparse + sparse and dense + sparse
      res_dense_plus_empty = torch.add(d, sp_empty, alpha=2.0)
      sp_empty_plus_sp1_alpha1 = torch.add(sp_empty, sp1, alpha=1.0)
      sp_empty_plus_sp1_alpha2 = torch.add(sp_empty, sp1, alpha=2.0)
      sp_sp1_plus_empty = torch.add(sp1, sp_empty, alpha=2.0)

      return (
          res_dense_add.to(dtype),
          res_dense_sub.to(dtype),
          d_inplace.to(dtype),
          d_sub_inplace.to(dtype),
          out_distinct.to(dtype),
          non_contig_out.clone().to(dtype),
          res_3d.to(dtype),
          sp_sum.to_dense().to(dtype),
          sp_diff.to_dense().to(dtype),
          res_dense_plus_empty.to(dtype),
          sp_empty_plus_sp1_alpha1.to_dense().to(dtype),
          sp_empty_plus_sp1_alpha2.to_dense().to(dtype),
          sp_sp1_plus_empty.to_dense().to(dtype),
      )

    self.assert_close_tpu_vs_cpu(run_add_sub_cases, atol=tol_opt, rtol=tol_opt)

  @parameterized.named_parameters(
      ("float32", torch.float32),
      ("bfloat16", torch.bfloat16),
      ("float16", torch.float16),
  )
  def test_sparse_coo_mul_neg(self, dtype):
    """Tests scalar multiplication (mul, mul_, mul.out) and negation (neg, neg_, neg.out) on sparse COO tensors."""
    tol_opt = 2e-2 if dtype in (torch.bfloat16, torch.float16) else None
    idx = torch.tensor([[0, 2, 1], [1, 0, 2]], dtype=torch.long)
    val = torch.tensor([[1.5, -2.0], [3.0, 4.5], [-1.0, 0.5]], dtype=dtype)

    def run_mul_neg_cases(device):
      is_cpu_fp16 = (
          torch.device(device).type == "cpu" and dtype == torch.float16
      )
      compute_dtype = torch.float32 if is_cpu_fp16 else dtype
      sp = torch.sparse_coo_tensor(
          to(idx, device=device),
          to(val, device=device).to(compute_dtype),
          (3, 3, 2),
          device=device,
      )
      scalar_t = torch.tensor(2.5, dtype=compute_dtype, device=device)

      # 1. sp * scalar_t and scalar_t * sp
      res_mul1 = sp * scalar_t
      res_mul2 = scalar_t * sp
      self.assertTrue(res_mul1.is_sparse)
      self.assertTrue(res_mul2.is_sparse)

      # 2. In-place mul_ and mul.out
      sp_inplace = sp.clone()
      sp_inplace.mul_(scalar_t)
      sp_out = torch.empty_like(sp)
      torch.mul(sp, scalar_t, out=sp_out)

      # 3. Negation (-sp, neg_(), neg.out)
      res_neg = -sp
      sp_neg_inplace = sp.clone()
      sp_neg_inplace.neg_()
      sp_neg_out = torch.empty_like(sp)
      torch.neg(sp, out=sp_neg_out)

      # 4. Empty sparse tensor mul and neg
      sp_empty = torch.sparse_coo_tensor(
          torch.empty((2, 0), dtype=torch.long, device=device),
          torch.empty((0, 2), dtype=compute_dtype, device=device),
          (3, 3, 2),
          device=device,
      )
      res_empty_mul = sp_empty * scalar_t
      res_empty_neg = -sp_empty

      return (
          res_mul1.to_dense().to(dtype),
          res_mul2.to_dense().to(dtype),
          sp_inplace.to_dense().to(dtype),
          sp_out.to_dense().to(dtype),
          res_neg.to_dense().to(dtype),
          sp_neg_inplace.to_dense().to(dtype),
          sp_neg_out.to_dense().to(dtype),
          res_empty_mul.to_dense().to(dtype),
          res_empty_neg.to_dense().to(dtype),
      )

    self.assert_close_tpu_vs_cpu(run_mul_neg_cases, atol=tol_opt, rtol=tol_opt)

  def test_sparse_coo_corner_cases(self):
    """Tests non-redundant corner cases for sparse COO ops on TPU.

    Covers:
    1. Out-variant shape resizing from (0, 0) / (0,) and promotional upcasting
       (float32 -> float64) for add, sub, mul, neg, and dense+sparse add.
    2. In-place self-aliasing (sp.add_(sp) and sp.sub_(sp)).
    3. Duplicate coordinates whose values cancel to exact 0.0 during coalesce().
    4. Cross-device copy_ (CPU <-> TPU) for uncoalesced hybrid sparse tensors.
    5. High-dimensional hybrid (sparse_dim=3, dense_dim=2) and purely sparse
       (sparse_dim=3, dense_dim=0) tensors.
    6. Direct aten.embedding_sparse_backward with padding_idx=-1 and boundary
       indices (0 and num_weights - 1).
    """
    idx1 = torch.tensor([[0, 2, 1], [1, 0, 2]], dtype=torch.long)
    val1 = torch.tensor(
        [[1.5, -2.0], [3.0, 4.5], [-1.0, 0.5]], dtype=torch.float32
    )
    idx2 = torch.tensor([[1, 2], [2, 0]], dtype=torch.long)
    val2 = torch.tensor([[2.0, -0.5], [-1.5, 1.0]], dtype=torch.float32)
    dense_fp32 = torch.randn(3, 3, 2, dtype=torch.float32)

    # 3. Duplicate coordinates cancelling to exact zero
    idx_cancel = torch.tensor([[1, 0, 1], [2, 1, 2]], dtype=torch.long)
    val_cancel = torch.tensor([3.5, 2.0, -3.5], dtype=torch.float32)

    # 5. High-dimensional 5D (3 sparse + 2 dense) and purely sparse 3D (0 dense)
    idx_5d = torch.tensor([[0, 1, 0], [1, 0, 1], [2, 1, 2]], dtype=torch.long)
    val_5d = torch.randn(3, 2, 2, dtype=torch.float32)
    dense_5d = torch.randn(2, 2, 3, 2, 2, dtype=torch.float32)

    idx_3d_pure = torch.tensor(
        [[1, 0, 1], [2, 1, 2], [0, 2, 0]], dtype=torch.long
    )
    val_3d_pure = torch.tensor([1.25, -2.5, 0.75], dtype=torch.float32)

    # 6. Direct embedding_sparse_backward inputs
    emb_grad = torch.randn(2, 3, 4, dtype=torch.float32)
    emb_indices = torch.tensor([[0, 7, 3], [7, 0, 2]], dtype=torch.long)

    def run_corner_cases(device):
      sp1 = torch.sparse_coo_tensor(
          to(idx1, device=device),
          to(val1, device=device),
          (3, 3, 2),
          device=device,
      )
      sp2 = torch.sparse_coo_tensor(
          to(idx2, device=device),
          to(val2, device=device),
          (3, 3, 2),
          device=device,
      )
      d = to(dense_fp32.clone(), device=device)

      # Case 1: Out-variant shape resizing from empty (0, 0, 0) + promotional upcasting to float64
      sp_out_add = torch.ops.aten._sparse_coo_tensor_with_dims(
          2,
          1,
          (0, 0, 0),
          dtype=torch.float64,
          layout=torch.sparse_coo,
          device=device,
      )
      torch.add(sp1, sp2, alpha=0.5, out=sp_out_add)
      self.assertEqual(sp_out_add.dtype, torch.float64)
      self.assertEqual(sp_out_add.shape, (3, 3, 2))

      sp_out_sub = torch.ops.aten._sparse_coo_tensor_with_dims(
          2,
          1,
          (0, 0, 0),
          dtype=torch.float64,
          layout=torch.sparse_coo,
          device=device,
      )
      torch.sub(sp1, sp2, alpha=0.5, out=sp_out_sub)

      sp_out_mul = torch.ops.aten._sparse_coo_tensor_with_dims(
          2,
          1,
          (0, 0, 0),
          dtype=torch.float64,
          layout=torch.sparse_coo,
          device=device,
      )
      torch.mul(sp1, 2.5, out=sp_out_mul)

      sp_out_neg = torch.ops.aten._sparse_coo_tensor_with_dims(
          2,
          1,
          (0, 0, 0),
          dtype=torch.float64,
          layout=torch.sparse_coo,
          device=device,
      )
      torch.neg(sp1, out=sp_out_neg)

      dense_out_add = torch.empty((0,), dtype=torch.float64, device=device)
      torch.add(d, sp1, alpha=1.5, out=dense_out_add)
      self.assertEqual(dense_out_add.dtype, torch.float64)
      self.assertEqual(dense_out_add.shape, (3, 3, 2))

      # Case 2: In-place self-aliasing (add_ and sub_ with self)
      sp_self_add = sp1.clone()
      sp_self_add.add_(sp_self_add, alpha=2.0)
      sp_self_sub = sp1.clone()
      sp_self_sub.sub_(sp_self_sub)

      # Case 3: Duplicate coordinates cancelling to exact 0.0 during coalesce()
      sp_cancel = torch.sparse_coo_tensor(
          to(idx_cancel, device=device),
          to(val_cancel, device=device),
          (3, 3),
          device=device,
      )
      sp_cancel_coal = sp_cancel.coalesce()
      self.assertTrue(sp_cancel_coal.is_coalesced())

      # Case 5: High-dimensional 5D hybrid and 3D purely sparse tensors
      sp5 = torch.sparse_coo_tensor(
          to(idx_5d, device=device),
          to(val_5d, device=device),
          (2, 2, 3, 2, 2),
          device=device,
      )
      d5 = to(dense_5d.clone(), device=device)
      sp5_coal = sp5.coalesce()
      res_5d_dense_add = torch.add(d5, sp5)

      sp3_pure = torch.sparse_coo_tensor(
          to(idx_3d_pure, device=device),
          to(val_3d_pure, device=device),
          (2, 3, 3),
          device=device,
      )
      sp3_pure_coal = sp3_pure.coalesce()

      # Case 6: Direct aten.embedding_sparse_backward (padding_idx=-1 and boundary padding_idx=0)
      eg = to(emb_grad, device=device)
      ei = to(emb_indices, device=device)
      bwd_no_pad = torch.ops.aten.embedding_sparse_backward(
          eg, ei, num_weights=8, padding_idx=-1, scale_grad_by_freq=False
      )
      bwd_pad0 = torch.ops.aten.embedding_sparse_backward(
          eg, ei, num_weights=8, padding_idx=0, scale_grad_by_freq=False
      )

      return (
          sp_out_add.to_dense(),
          sp_out_sub.to_dense(),
          sp_out_mul.to_dense(),
          sp_out_neg.to_dense(),
          dense_out_add,
          sp_self_add.to_dense(),
          sp_self_sub.to_dense(),
          sp_cancel_coal.indices(),
          sp_cancel_coal.values(),
          sp_cancel_coal.to_dense(),
          sp5_coal.to_dense(),
          res_5d_dense_add,
          sp3_pure_coal.indices(),
          sp3_pure_coal.values(),
          bwd_no_pad.to_dense(),
          bwd_pad0.to_dense(),
      )

    self.assert_close_tpu_vs_cpu(run_corner_cases)

    # Case 4: Cross-device copy_ (CPU <-> TPU) for uncoalesced hybrid sparse tensor
    sp_cpu_src = torch.sparse_coo_tensor(idx1, val1, (3, 3, 2), device="cpu")
    self.assertFalse(sp_cpu_src.is_coalesced())
    sp_tpu_dst = torch.empty((3, 3, 2), layout=torch.sparse_coo, device="tpu")
    sp_tpu_dst.copy_(sp_cpu_src)
    self.assertFalse(sp_tpu_dst.is_coalesced())
    self.assertEqual(sp_tpu_dst._nnz(), 3)

    sp_cpu_roundtrip = torch.empty(
        (3, 3, 2), layout=torch.sparse_coo, device="cpu"
    )
    sp_cpu_roundtrip.copy_(sp_tpu_dst)
    self.assertFalse(sp_cpu_roundtrip.is_coalesced())
    self.assertEqual(sp_cpu_roundtrip.to_dense(), sp_cpu_src.to_dense())

  @parameterized.named_parameters(
      ("1d_empty_indices", 16, (0,)),
      ("2d_empty_indices", 16, (2, 0)),
      ("0_row_weight", 0, (0,)),
  )
  def test_embedding_empty_tensors(self, vocab_size, indices_shape):
    """Tests embedding with 0-element tensors in forward and backward."""
    embedding_dim = 8
    weight = torch.randn(vocab_size, embedding_dim)
    indices = torch.empty(indices_shape, dtype=torch.long)

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ops.aten.embedding(
            to(weight, device=device),
            to(indices, device=device),
        )
    )

    def compute_grad(device):
      w = to(weight.clone(), device=device).requires_grad_(True)
      idx = to(indices, device=device)
      out = torch.ops.aten.embedding(w, idx)
      out.backward(torch.zeros_like(out))
      return w.grad

    self.assert_close_tpu_vs_cpu(compute_grad)

  @parameterized.named_parameters(
      ("float32", torch.float32),
      ("bfloat16", torch.bfloat16),
  )
  def test_embedding_duplicate_indices_and_options(self, dtype):
    """Tests backward gradient accumulation with duplicate indices, padding_idx, and scale_grad_by_freq."""
    vocab_size = 8
    embedding_dim = 4
    indices = torch.tensor([2, 5, 2, 2, 5, 0, 2, 5, 0, 2], dtype=torch.long)
    weight = torch.randn(vocab_size, embedding_dim, dtype=dtype)

    def compute_grad(device):
      w = to(weight.clone(), device=device).requires_grad_(True)
      idx = to(indices, device=device)
      out = torch.ops.aten.embedding(
          w, idx, padding_idx=0, scale_grad_by_freq=True
      )
      scale = torch.arange(
          1, out.numel() + 1, device=device, dtype=dtype
      ).view_as(out)
      loss = (out * scale).sum()
      loss.backward()
      return w.grad

    self.assert_close_tpu_vs_cpu(compute_grad)

  def test_embedding_multidim_int32(self):
    """Tests embedding with 3-D indices and int32 index dtype."""
    vocab_size = 32
    embedding_dim = 16
    indices = torch.randint(0, vocab_size, size=(2, 3, 4), dtype=torch.int32)
    weight = torch.randn(vocab_size, embedding_dim)

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ops.aten.embedding(
            to(weight, device=device),
            to(indices, device=device),
        )
    )

    def compute_grad(device):
      w = to(weight.clone(), device=device).requires_grad_(True)
      idx = to(indices, device=device)
      out = torch.ops.aten.embedding(w, idx)
      loss = out.sum()
      loss.backward()
      return w.grad

    self.assert_close_tpu_vs_cpu(compute_grad)

  def test_embedding_non_contiguous(self):
    """Tests embedding with non-contiguous weight and indices."""
    vocab_size = 16
    embedding_dim = 8
    weight = torch.randn(vocab_size, embedding_dim * 2)[:, ::2]
    self.assertFalse(weight.is_contiguous())
    indices = torch.randint(0, vocab_size, size=(4, 6)).t()
    self.assertFalse(indices.is_contiguous())

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ops.aten.embedding(
            to(weight, device=device),
            to(indices, device=device),
        )
    )

    def compute_grad(device):
      w = to(weight.clone(), device=device).requires_grad_(True)
      idx = to(indices, device=device)
      out = torch.ops.aten.embedding(w, idx)
      loss = out.sum()
      loss.backward()
      return w.grad

    self.assert_close_tpu_vs_cpu(compute_grad)

  def test_embedding_renorm(self):
    """Tests that embedding renorm works."""

    def fn(device):
      vocab_size = 16
      embedding_size = 32
      # Initialize the table with all 1.0, so the 2-norm will be greater than
      # 1.0 guarantees the renorm on rows are triggered.
      embedding_table = torch.ones(
          vocab_size, embedding_size, dtype=torch.float32, device=device
      )
      indices = torch.tensor([0, 1, 2, 3, 10, 11, 15], device=device)
      torch.embedding_renorm_(
          embedding_table, indices, max_norm=1.0, norm_type=2
      )
      return embedding_table

    self.assert_close_tpu_vs_cpu(fn)

  def test_embedding_renorm_complex(self):
    """Tests that embedding renorm works with complex types."""

    def fn(device):
      vocab_size = 16
      embedding_size = 32
      # Initialize with complex values
      embedding_table = torch.complex(
          torch.ones(
              vocab_size, embedding_size, dtype=torch.float32, device=device
          ),
          torch.ones(
              vocab_size, embedding_size, dtype=torch.float32, device=device
          ),
      )
      indices = torch.tensor([0, 1, 2, 3, 10, 11, 15], device=device)
      torch.embedding_renorm_(
          embedding_table, indices, max_norm=1.0, norm_type=2
      )
      return embedding_table

    self.assert_close_tpu_vs_cpu(fn)

  def test_materialize_empty_tensor(self):
    """Tests that materializing an empty tensor works."""
    prev_deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    prev_deterministic_fill = (
        torch.utils.deterministic.fill_uninitialized_memory
    )
    exception = None
    try:
      # Enable deterministic mode to force CPU kernels to fill empty float32
      # tensors with NaNs. Otherwise, the buffers are not comparable.
      torch.use_deterministic_algorithms(True)
      torch.utils.deterministic.fill_uninitialized_memory = True

      # Create three empty tensors: one directly on CPU, one by materializing
      # empty TPU to CPU, and one by filling a CPU tensor with NaNs manually.
      empty_tpu = torch.empty(
          1,
          2,
          3,
          device=torch.device("tpu"),
          dtype=torch.float32,
      )
      empty_tpu_to_cpu = empty_tpu.to("cpu")
      empty_cpu = torch.empty(
          1, 2, 3, device=torch.device("cpu"), dtype=torch.float32
      )
      empty_nans = torch.full_like(
          empty_cpu,
          float("nan"),
          device=torch.device("cpu"),
          dtype=torch.float32,
      )

      # All three tensors should have the same shape and be filled with NaNs.
      self.assert_close(golden_result=empty_cpu, torch_tpu_result=empty_nans)
      self.assert_close(
          golden_result=empty_tpu_to_cpu, torch_tpu_result=empty_nans
      )
    except Exception as e:  # pylint: disable=broad-except
      exception = e
    finally:
      # Restore the previous deterministic mode settings.
      torch.use_deterministic_algorithms(prev_deterministic_algorithms)
      torch.utils.deterministic.fill_uninitialized_memory = (
          prev_deterministic_fill
      )
    if exception:
      raise exception

  def test_min(self):
    # empty tensor on reduction dim
    with self.assertRaisesRegex(IndexError, "min"):
      torch.min(torch.tensor([], device="cpu"), dim=0)
    with self.assertRaisesRegex(IndexError, "min"):
      torch.min(
          torch.tensor([], device=torch.device("tpu")),
          dim=0,
      )

    # empty tensor on reduction dim
    with self.assertRaisesRegex(IndexError, "min"):
      torch.min(torch.empty(0, 2, device="cpu"), dim=0)
    with self.assertRaisesRegex(IndexError, "min"):
      torch.min(
          torch.empty(0, 2, device=torch.device("tpu")),
          dim=0,
      )

    # nested empty tensor
    cpu_result = torch.min(torch.empty(1, 0, device="cpu"), dim=0)
    tpu_result = torch.min(torch.empty(1, 0, device=torch.device("tpu")), dim=0)
    self.assert_close(
        golden_result=cpu_result[0], torch_tpu_result=tpu_result[0].cpu()
    )
    self.assert_close(
        golden_result=cpu_result[1], torch_tpu_result=tpu_result[1].cpu()
    )

    cpu_result = torch.min(torch.empty(0, 2, device="cpu"), dim=1)
    tpu_result = torch.min(torch.empty(0, 2, device=torch.device("tpu")), dim=1)
    self.assert_close(
        golden_result=cpu_result[0], torch_tpu_result=tpu_result[0].cpu()
    )
    self.assert_close(
        golden_result=cpu_result[1], torch_tpu_result=tpu_result[1].cpu()
    )

    # test min reduced to scalar
    cpu_result = torch.min(torch.tensor([1, 3, 2], device="cpu"), dim=0)
    tpu_result = torch.min(
        torch.tensor([1, 3, 2], device=torch.device("tpu")),
        dim=0,
    )
    self.assert_close(
        golden_result=cpu_result[0], torch_tpu_result=tpu_result[0].cpu()
    )
    self.assert_close(
        golden_result=cpu_result[1], torch_tpu_result=tpu_result[1].cpu()
    )

    # test min
    cpu_result = torch.min(
        torch.tensor([1, 3, 2], device="cpu"), dim=0, keepdim=True
    )
    tpu_result = torch.min(
        torch.tensor([1, 3, 2], device=torch.device("tpu")),
        dim=0,
        keepdim=True,
    )
    self.assert_close(
        golden_result=cpu_result[0], torch_tpu_result=tpu_result[0].cpu()
    )
    self.assert_close(
        golden_result=cpu_result[1], torch_tpu_result=tpu_result[1].cpu()
    )

    cpu_result = torch.min(torch.tensor([-1.0, 3.0, 2.0], device="cpu"), dim=0)
    tpu_result = torch.min(
        torch.tensor([-1.0, 3.0, 2.0], device=torch.device("tpu")),
        dim=0,
    )
    self.assert_close(
        golden_result=cpu_result[0], torch_tpu_result=tpu_result[0].cpu()
    )
    self.assert_close(
        golden_result=cpu_result[1], torch_tpu_result=tpu_result[1].cpu()
    )

    # dim=0
    cpu_result = torch.min(
        torch.tensor([[1, 3, 2], [4, 6, 5]], device="cpu"), dim=0
    )
    tpu_result = torch.min(
        torch.tensor(
            [[1, 3, 2], [4, 6, 5]],
            device=torch.device("tpu"),
        ),
        dim=0,
    )
    self.assert_close(
        golden_result=cpu_result[0], torch_tpu_result=tpu_result[0].cpu()
    )
    self.assert_close(
        golden_result=cpu_result[1], torch_tpu_result=tpu_result[1].cpu()
    )

    # dim=1, keepdim=True
    cpu_result = torch.min(
        torch.tensor([[1, 3, 2], [4, 6, 5]], device="cpu"), dim=1, keepdim=True
    )
    tpu_result = torch.min(
        torch.tensor(
            [[1, 3, 2], [4, 6, 5]],
            device=torch.device("tpu"),
        ),
        dim=1,
        keepdim=True,
    )
    self.assert_close(
        golden_result=cpu_result[0], torch_tpu_result=tpu_result[0].cpu()
    )
    self.assert_close(
        golden_result=cpu_result[1], torch_tpu_result=tpu_result[1].cpu()
    )

    # dim=-1
    cpu_result = torch.min(
        torch.tensor([[1, 3, 2], [4, 6, 5]], device="cpu"), dim=-1
    )
    tpu_result = torch.min(
        torch.tensor(
            [[1, 3, 2], [4, 6, 5]],
            device=torch.device("tpu"),
        ),
        dim=-1,
    )
    self.assert_close(
        golden_result=cpu_result[0], torch_tpu_result=tpu_result[0].cpu()
    )
    self.assert_close(
        golden_result=cpu_result[1], torch_tpu_result=tpu_result[1].cpu()
    )

    # dim=-1, keepdim=True
    cpu_result = torch.min(
        torch.tensor([[1, 3, 2], [4, 6, 5]], device="cpu"), dim=-1, keepdim=True
    )
    tpu_result = torch.min(
        torch.tensor(
            [[1, 3, 2], [4, 6, 5]],
            device=torch.device("tpu"),
        ),
        dim=-1,
        keepdim=True,
    )
    self.assert_close(
        golden_result=cpu_result[0], torch_tpu_result=tpu_result[0].cpu()
    )
    self.assert_close(
        golden_result=cpu_result[1], torch_tpu_result=tpu_result[1].cpu()
    )

    # out param
    a = torch.tensor([[1.0, 2.0], [4.0, 3.0]])
    v_cpu = torch.empty(2, dtype=a.dtype)
    i_cpu = torch.empty(2, dtype=torch.int64)
    torch.min(a, dim=1, out=(v_cpu, i_cpu))
    v_tpu = torch.empty(2, dtype=a.dtype, device=torch.device("tpu"))
    i_tpu = torch.empty(2, dtype=torch.int64, device=torch.device("tpu"))
    torch.min(
        a.clone().to(torch.device("tpu")),
        dim=1,
        out=(v_tpu, i_tpu),
    )
    self.assert_close(golden_result=v_cpu, torch_tpu_result=v_tpu.cpu())
    self.assert_close(golden_result=i_cpu, torch_tpu_result=i_tpu.cpu())

    # out param with different dtypes for indices
    a = torch.tensor([[1, 2], [4, 3]], dtype=torch.int16)
    v_cpu = torch.empty(2, dtype=a.dtype)
    i_cpu = torch.empty(2, dtype=torch.int64)
    torch.min(a, dim=1, out=(v_cpu, i_cpu))
    v_tpu = torch.empty(2, dtype=a.dtype, device=torch.device("tpu"))
    i_tpu = torch.empty(2, dtype=torch.int64, device=torch.device("tpu"))
    torch.min(a.to(torch.device("tpu")), dim=1, out=(v_tpu, i_tpu))
    self.assert_close(golden_result=v_cpu, torch_tpu_result=v_tpu.cpu())
    self.assert_close(golden_result=i_cpu, torch_tpu_result=i_tpu.cpu())

  def test_all(self):
    def compute(dim, keep_dim, device):
      x = torch.tensor(
          [[True, True], [True, False], [True, True], [True, True]],
          dtype=torch.bool,
          device=device,
      )
      out = torch.empty(0, dtype=torch.bool, device=device)
      torch.all(x, dim=dim, keepdim=keep_dim, out=out)
      return out

    self.assert_close_tpu_vs_cpu(functools.partial(compute, 0, False))
    self.assert_close_tpu_vs_cpu(functools.partial(compute, 0, True))
    self.assert_close_tpu_vs_cpu(functools.partial(compute, 1, False))
    self.assert_close_tpu_vs_cpu(functools.partial(compute, 1, True))

  def test_any(self):
    def compute(dim, keep_dim, device):
      x = torch.tensor(
          [[False, False], [True, True]], dtype=torch.bool, device=device
      )
      out = torch.empty(0, dtype=torch.bool, device=device)
      torch.any(x, dim=dim, keepdim=keep_dim, out=out)
      return out

    self.assert_close_tpu_vs_cpu(functools.partial(compute, 0, False))
    self.assert_close_tpu_vs_cpu(functools.partial(compute, 0, True))
    self.assert_close_tpu_vs_cpu(functools.partial(compute, 1, False))
    self.assert_close_tpu_vs_cpu(functools.partial(compute, 1, True))

  def test_all_byte(self):
    device = torch.device("tpu")
    # For uint8 inputs, PyTorch returns uint8 for backward compatibility.
    x = torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.uint8, device=device)
    out = torch.empty(0, dtype=torch.uint8, device=device)
    torch.all(x, dim=0, out=out)
    self.assertEqual(out.dtype, torch.uint8)
    utils.assert_close(
        out, torch.tensor([1, 0], dtype=torch.uint8, device=device)
    )

    out_all = torch.empty((), dtype=torch.uint8, device=device)
    torch.all(x, out=out_all)
    self.assertEqual(out_all.dtype, torch.uint8)
    utils.assert_close(
        out_all, torch.tensor(0, dtype=torch.uint8, device=device)
    )

    def compute(dim, keep_dim, dev):
      x_dev = torch.tensor(
          [[1, 1], [1, 0], [1, 1]], dtype=torch.uint8, device=dev
      )
      out_dev = torch.empty(0, dtype=torch.uint8, device=dev)
      torch.all(x_dev, dim=dim, keepdim=keep_dim, out=out_dev)
      return out_dev

    self.assert_close_tpu_vs_cpu(functools.partial(compute, 0, False))
    self.assert_close_tpu_vs_cpu(functools.partial(compute, 0, True))

    # Float input with uint8 output is supported.
    x_float = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    out_byte = torch.empty(0, dtype=torch.uint8, device=device)
    torch.all(x_float, out=out_byte)
    self.assertEqual(out_byte.dtype, torch.uint8)
    utils.assert_close(
        out_byte, torch.tensor(1, dtype=torch.uint8, device=device)
    )

  def test_any_byte(self):
    device = torch.device("tpu")
    # For uint8 inputs, PyTorch returns uint8 for backward compatibility.
    x = torch.tensor([[0, 0], [1, 0], [0, 0]], dtype=torch.uint8, device=device)
    out = torch.empty(0, dtype=torch.uint8, device=device)
    torch.any(x, dim=0, out=out)
    self.assertEqual(out.dtype, torch.uint8)
    utils.assert_close(
        out, torch.tensor([1, 0], dtype=torch.uint8, device=device)
    )

    out_any = torch.empty((), dtype=torch.uint8, device=device)
    torch.any(x, out=out_any)
    self.assertEqual(out_any.dtype, torch.uint8)
    utils.assert_close(
        out_any, torch.tensor(1, dtype=torch.uint8, device=device)
    )

    def compute(dim, keep_dim, dev):
      x_dev = torch.tensor(
          [[0, 0], [1, 0], [0, 0]], dtype=torch.uint8, device=dev
      )
      out_dev = torch.empty(0, dtype=torch.uint8, device=dev)
      torch.any(x_dev, dim=dim, keepdim=keep_dim, out=out_dev)
      return out_dev

    self.assert_close_tpu_vs_cpu(functools.partial(compute, 0, False))
    self.assert_close_tpu_vs_cpu(functools.partial(compute, 0, True))

    # Float input with uint8 output is supported.
    x_float = torch.tensor([0.0, 1.0], dtype=torch.float32, device=device)
    out_byte = torch.empty(0, dtype=torch.uint8, device=device)
    torch.any(x_float, out=out_byte)
    self.assertEqual(out_byte.dtype, torch.uint8)
    utils.assert_close(
        out_byte, torch.tensor(1, dtype=torch.uint8, device=device)
    )

  def test_multinomial_output_properties(self):
    device = torch.device("tpu")

    # 2D input
    probs = torch.rand(4, 10, device=device)

    # With replacement
    num_samples = 5
    tpu_result = torch.multinomial(probs, num_samples, replacement=True)
    self.assertEqual(tpu_result.device.type, device.type)
    self.assertTrue((tpu_result >= 0).all())
    self.assertTrue((tpu_result < 10).all())
    self.assertEqual(tpu_result.shape, (4, num_samples))

    # Without replacement
    num_samples = 5
    tpu_result = torch.multinomial(probs, num_samples, replacement=False)
    self.assertEqual(tpu_result.device.type, device.type)
    self.assertTrue((tpu_result >= 0).all())
    self.assertTrue((tpu_result < 10).all())
    self.assertEqual(tpu_result.shape, (4, num_samples))
    for row in tpu_result.cpu():
      self.assertLen(torch.unique(row), len(row))

    # 1D input
    probs = torch.rand(10, device=device)

    # With replacement
    num_samples = 5
    tpu_result = torch.multinomial(probs, num_samples, replacement=True)
    self.assertEqual(tpu_result.device.type, device.type)
    self.assertTrue((tpu_result >= 0).all())
    self.assertTrue((tpu_result < 10).all())
    self.assertEqual(tpu_result.shape, (num_samples,))

    # Without replacement
    num_samples = 5
    tpu_result = torch.multinomial(probs, num_samples, replacement=False)
    self.assertEqual(tpu_result.device.type, device.type)
    self.assertTrue((tpu_result >= 0).all())
    self.assertTrue((tpu_result < 10).all())
    self.assertEqual(tpu_result.shape, (num_samples,))
    self.assertLen(torch.unique(tpu_result.cpu()), len(tpu_result))

  def test_multinomial_skewed_distribution(self):
    device = torch.device("tpu")

    # 2D input
    probs = torch.tensor([[1e10, 1.0], [1.0, 1e10]], device=device)

    # With replacement
    num_samples = 10
    tpu_result = torch.multinomial(probs, num_samples, replacement=True).cpu()
    self.assertTrue((tpu_result[0] == 0).all())
    self.assertTrue((tpu_result[1] == 1).all())

    # Without replacement
    num_samples = 1
    tpu_result = torch.multinomial(probs, num_samples, replacement=False).cpu()
    self.assertEqual(tpu_result[0, 0], 0)
    self.assertEqual(tpu_result[1, 0], 1)

    # 1D input
    probs = torch.tensor([1.0, 1e10], device=device)

    # With replacement
    num_samples = 10
    tpu_result = torch.multinomial(probs, num_samples, replacement=True).cpu()
    self.assertTrue((tpu_result == 1).all())

    # Without replacement
    num_samples = 1
    tpu_result = torch.multinomial(probs, num_samples, replacement=False).cpu()
    self.assertEqual(tpu_result[0], 1)

  def test_empty_like(self):
    device = torch.device("tpu")

    nonempty_cpu_tensor = torch.randn(2, 3, 4, device="cpu")
    nonempty_contiguous_tensor = nonempty_cpu_tensor.to(device)

    # Contiguous input tensor
    # Empty tensor has identical metadata but no values
    empty_like_contiguous = torch.empty_like(nonempty_cpu_tensor)
    self.assertEqual(empty_like_contiguous.shape, nonempty_cpu_tensor.shape)
    self.assertEqual(
        empty_like_contiguous.stride(), nonempty_contiguous_tensor.stride()
    )
    self.assertEqual(empty_like_contiguous.dtype, nonempty_cpu_tensor.dtype)
    self.assertEqual(empty_like_contiguous.device, nonempty_cpu_tensor.device)

    # Discontiguous input tensor
    # Empty tensor has identical metadata but no values
    nonempty_discontiguous_tensor = nonempty_contiguous_tensor.transpose(0, 2)
    empty_like_discontiguous = torch.empty_like(nonempty_discontiguous_tensor)
    self.assertEqual(
        empty_like_discontiguous.shape, nonempty_discontiguous_tensor.shape
    )
    self.assertEqual(
        empty_like_discontiguous.stride(),
        nonempty_discontiguous_tensor.stride(),
    )
    self.assertEqual(
        empty_like_discontiguous.dtype, nonempty_discontiguous_tensor.dtype
    )
    self.assertEqual(
        empty_like_discontiguous.device, nonempty_discontiguous_tensor.device
    )

  def test_copy_from_and_resize(self):
    # Going through the ordinary "copy_" method doesn't reach this aten op,
    # so we have to access it manually.
    copy_from_and_resize = torch.ops.aten._copy_from_and_resize
    src = torch.arange(20, device=torch.device("tpu"))
    dst = torch.empty(10, dtype=src.dtype, device=torch.device("tpu"))
    copy_from_and_resize(src, dst)
    self.assertEqual(dst.shape, (20,))
    self.assertEqual(dst.cpu(), torch.arange(20, device="cpu"))

  def test_empty_overwrite_all(self):
    def test_fn(device):
      x = torch.empty(10, dtype=torch.int64, device=device)
      x.copy_(torch.arange(10, dtype=torch.int64, device=device))
      return x

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_randint_reproducible(self):
    low, high = 10, 20
    torch.manual_seed(1234)
    z = torch.randint(
        low=low,
        high=high,
        size=(3, 3),
        device=torch.device("tpu"),
    )
    w = torch.randint(
        low=low,
        high=high,
        size=(3, 3),
        device=torch.device("tpu"),
    )
    # with probability 1 - (1/2)^(100)
    self.assertNotEqual(z.to("cpu"), w.to("cpu"))
    torch.manual_seed(1234)
    z_again = torch.randint(
        low=low,
        high=high,
        size=(3, 3),
        device=torch.device("tpu"),
    )
    self.assertEqual(z.to("cpu"), z_again.to("cpu"))

  def test_randint_range_u32_pow2(self):
    low, high = 0, 2**16
    x = torch.randint(
        low=low,
        high=high,
        size=(3, 3),
        device=torch.device("tpu"),
        dtype=torch.int64,
    )
    res = x.cpu()
    self.assertLess(res.max(), high)
    self.assertGreaterEqual(res.min(), low)

  def test_randint_range_u32_not_pow2(self):
    low, high = 0, 2**32 - 1
    x = torch.randint(
        low=low,
        high=high,
        size=(3, 3),
        device=torch.device("tpu"),
        dtype=torch.int64,
    )
    res = x.cpu()
    self.assertLess(res.max(), high)
    self.assertGreaterEqual(res.min(), low)

  def test_randint_range_u64_pow2(self):
    low, high = 0, 2**37
    x = torch.randint(
        low=low,
        high=high,
        size=(3, 3),
        device=torch.device("tpu"),
        dtype=torch.int64,
    )
    res = x.cpu()
    self.assertLess(res.max(), high)
    self.assertGreaterEqual(res.min(), low)

  def test_randint_range_u64_not_pow2(self):
    low, high = 0, 2**32 + 1
    x = torch.randint(
        low=low,
        high=high,
        size=(3, 3),
        device=torch.device("tpu"),
        dtype=torch.int64,
    )
    res = x.cpu()
    self.assertLess(res.max(), high)
    self.assertGreaterEqual(res.min(), low)

  def test_randint_range_u32_pow2_high_low(self):
    low, high = 2**40, 2**40 + 2**16
    x = torch.randint(
        low=low,
        high=high,
        size=(3, 3),
        device=torch.device("tpu"),
        dtype=torch.int64,
    )
    res = x.cpu()
    self.assertLess(res.max(), high)
    self.assertGreaterEqual(res.min(), low)

  def test_randint_range_u32_not_pow2_high_low(self):
    low, high = 2**40, 2**40 + 101
    x = torch.randint(
        low=low,
        high=high,
        size=(3, 3),
        device=torch.device("tpu"),
        dtype=torch.int64,
    )
    res = x.cpu()
    self.assertLess(res.max(), high)
    self.assertGreaterEqual(res.min(), low)

  def test_empty_write_prefix_dense(self):
    def test_fn(device):
      x = torch.empty(20, dtype=torch.int64, device=device)
      view = x[:10]
      view.copy_(torch.arange(10, dtype=torch.int64, device=device))
      return view

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_empty_write_nondense(self):
    def test_fn(device):
      x = torch.empty(20, dtype=torch.int64, device=device)
      view = x[::2]
      view.copy_(torch.arange(10, dtype=torch.int64, device=device))
      return view

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_resize_overwrite_all(self):
    def test_fn(device):
      x = torch.arange(10, dtype=torch.int64, device=device)
      x.resize_(20)
      x.copy_(torch.arange(10, end=30, dtype=torch.int64, device=device))
      return x

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_resize_write_dense_overlapped(self):
    def test_fn(device):
      x = torch.arange(10, dtype=torch.int64, device=device)
      x.resize_(20)
      view = x[5:15]
      view.copy_(
          torch.arange(start=10, end=20, dtype=torch.int64, device=device)
      )
      return view

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_resize_write_dense_sequential(self):
    def test_fn(device):
      x = torch.arange(10, dtype=torch.int64, device=device)
      x.resize_(20)
      view = x[10:15]
      view.copy_(
          torch.arange(start=10, end=15, dtype=torch.int64, device=device)
      )
      return view

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_resize_write_dense_nonsequential(self):
    def test_fn(device):
      x = torch.arange(10, dtype=torch.int64, device=device)
      x.resize_(20)
      view = x[11:16]
      view.copy_(
          torch.arange(start=10, end=15, dtype=torch.int64, device=device)
      )
      return view

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_resize_write_nondense(self):
    def test_fn(device):
      x = torch.arange(10, dtype=torch.int64, device=device)
      x.resize_(20)
      view = x[::2]
      view.copy_(
          torch.arange(start=10, end=20, dtype=torch.int64, device=device)
      )
      return view

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_empty_channels_last(self):
    # Example adapted from
    # https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html
    shape = (10, 3, 32, 32)
    x = torch.empty(
        *shape,
        device=torch.device("tpu"),
        memory_format=torch.channels_last,
    )
    self.assertTrue(x.is_contiguous(memory_format=torch.channels_last))
    self.assertEqual(x.shape, shape)
    self.assertEqual(x.stride(), (32 * 32 * 3, 1, 32 * 3, 3))

  def test_empty_channels_last_3d(self):
    # Example adapted from
    # https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html
    shape = (10, 3, 32, 32, 32)
    x = torch.empty(
        *shape,
        device=torch.device("tpu"),
        memory_format=torch.channels_last_3d,
    )
    self.assertTrue(x.is_contiguous(memory_format=torch.channels_last_3d))
    self.assertEqual(x.shape, shape)
    self.assertEqual(x.stride(), (32 * 32 * 32 * 3, 1, 32 * 32 * 3, 32 * 3, 3))

  def test_uniform_reproducible(self):
    torch.manual_seed(4321)
    t = torch.zeros(10, 10, dtype=torch.float32).to(torch.device("tpu"))
    t = torch.Tensor.uniform_(t, 0, 1)
    w = torch.zeros(10, 10, dtype=torch.float32).to(torch.device("tpu"))
    w = torch.Tensor.uniform_(w, 0, 1)
    self.assertNotEqual(t.cpu(), w.cpu())
    torch.manual_seed(4321)
    t_again = torch.zeros(10, 10, dtype=torch.float32).to(torch.device("tpu"))
    t_again = torch.Tensor.uniform_(t_again, 0, 1)
    self.assert_close(golden_result=t.cpu(), torch_tpu_result=t_again.cpu())

  @parameterized.named_parameters(
      ("float64", torch.float64),
      ("float32", torch.float32),
      # ("float16", torch.float16), Disabled because some conversions produce
      # NaNs for f16.
      ("bfloat16", torch.bfloat16),
  )
  def test_uniform_distribution(self, dtype: torch.dtype):
    # To make sure atol is small, and the test is meaningful
    n = 1000
    t = torch.zeros(n, n, dtype=dtype, device=torch.device("tpu"))
    t = torch.Tensor.uniform_(t, 0, 1)
    # P(|mean(t) - 0.5| > atol) < 1 / 12 / n / n / a^2
    # Make P < 1e-6, by picking a = sqrt(1e6 / 12 / n / n) = sqrt(1 / 12)
    atol = torch.sqrt(torch.tensor(1e6 / 12 / n / n))
    self.assert_close(
        golden_result=torch.tensor(0.5, dtype=dtype),
        torch_tpu_result=t.mean().cpu(),
        atol=atol,
    )

  @parameterized.named_parameters(
      ("float32", torch.float32),
      ("float64", torch.float64),
      ("bfloat16", torch.bfloat16),
      ("int64", torch.int64),
      ("complex64", torch.complex64),
  )
  def test_power_negative_base(self, dtype):
    """ops_test only checks positive values for both base and power."""

    def test_fn(device):
      base = torch.tensor([-2.0, -2.0, -2.0, -2.0], dtype=dtype, device=device)
      power = torch.tensor([2, 3, 0.5, -2], dtype=torch.float32, device=device)
      return torch.pow(base, power)

    rtol, atol = (1.7e-6, 2.3e-6) if dtype == torch.complex64 else (None, None)
    self.assert_close_tpu_vs_cpu(test_fn, rtol=rtol, atol=atol)

  @parameterized.named_parameters(
      (f'{op}_{"include" if include_self else "no"}_self', op, include_self)
      for op, include_self in itertools.product(
          ["sum", "prod", "mean", "amax", "amin"], [True, False]
      )
  )
  def test_scatter_reduce(self, reduce_op, include_self):
    """Tests torch.ops.aten.scatter_reduce with various reduction ops."""

    def test_fn(device: torch.device) -> torch.Tensor:
      dtype = torch.float32

      if reduce_op == "prod":
        arg = torch.ones(3, 5, dtype=dtype, device=device)
      elif reduce_op == "amin":
        arg = torch.full((3, 5), 100.0, dtype=dtype, device=device)
      elif reduce_op == "amax":
        arg = torch.full((3, 5), -100.0, dtype=dtype, device=device)
      else:  # sum, mean
        arg = torch.zeros(3, 5, dtype=dtype, device=device)

      index = torch.tensor([[0, 1, 2], [0, 1, 2]], device=device)
      # src values: [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]
      src = (
          torch.arange(6, device=device, dtype=dtype).reshape(2, 3) + 1.0
      ) * 2.0

      return arg.scatter_reduce(
          1, index, src, reduce=reduce_op, include_self=include_self
      )

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_scatter_src_larger_than_index(self):
    """Tests scatter where src is larger than index."""

    def _test(device):
      arg = torch.zeros(3, 5, device=device)
      index = torch.tensor([[0, 1, 2], [0, 1, 2]], device=device)
      src = torch.ones(2, 5, device=device)  # Larger than index (2, 3)
      return arg.scatter(1, index, src)

    self.assert_close_tpu_vs_cpu(_test)

  @parameterized.product(
      dtype=[torch.float32, torch.bfloat16],
      dim=[0, 1],
  )
  def test_scatter_add_broadcast_index(self, dtype: torch.dtype, dim: int):
    """scatter_add with a broadcast (expanded) index matches CPU.

    This is the top-k MoE token-combine idiom
    `out.scatter_add_(0, ids[:, None].expand_as(src), src)`, which exercises the
    row-broadcast fast path in BuildScatterShlo that lowers to a row/window
    scatter (index_add) instead of a scalar scatter. Each id appears multiple
    times, so accumulation of duplicates is covered.

    Args:
      dtype: Input tensor data type.
      dim: Dimension along which to index.
    """
    t, k, feat = 8, 4, 6

    def test_fn(device: torch.device) -> torch.Tensor:
      ids = torch.arange(t * k, device=device) % t  # each id repeats k times
      if dim == 0:
        out = torch.zeros(t, feat, dtype=dtype, device=device)
        src = (
            torch.arange(t * k * feat, device=device, dtype=dtype).reshape(
                t * k, feat
            )
            * 0.01
        )
        index = ids.unsqueeze(-1).expand_as(src)
      else:
        out = torch.zeros(feat, t, dtype=dtype, device=device)
        src = (
            torch.arange(feat * t * k, device=device, dtype=dtype).reshape(
                feat, t * k
            )
            * 0.01
        )
        index = ids.unsqueeze(0).expand_as(src)
      return out.scatter_add(dim, index, src)

    rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (None, None)
    self.assert_close_tpu_vs_cpu(test_fn, rtol=rtol, atol=atol)

  @parameterized.product(include_self=[True, False])
  def test_scatter_reduce_sum_broadcast_index(self, include_self: bool):
    """scatter_reduce(sum) with a broadcast index matches CPU.

    include_self=True hits the row-broadcast fast path; include_self=False must
    fall through to the general scalar scatter and stay correct.

    Args:
      include_self: Whether to include the self tensor values.
    """
    t, k, feat = 8, 4, 6

    def test_fn(device: torch.device) -> torch.Tensor:
      ids = torch.arange(t * k, device=device) % t
      out = torch.zeros(t, feat, dtype=torch.float32, device=device)
      src = (
          torch.arange(
              t * k * feat, device=device, dtype=torch.float32
          ).reshape(t * k, feat)
          * 0.01
      )
      index = ids.unsqueeze(-1).expand_as(src)
      return out.scatter_reduce(
          0, index, src, reduce="sum", include_self=include_self
      )

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_scatter_add_broadcast_index_lowers_to_row_scatter(self):
    """The broadcast-index scatter_add lowers to a row (window) scatter.

    The numerics tests above pass whether or not the fast path fires, so this
    guards the lowering itself: assert the emitted StableHLO is the efficient
    window scatter (`update_window_dims = [1]`) rather than the slow scalar
    scatter (`update_window_dims = []`, one index entry per element).
    """
    t, k, feat = 8, 4, 6
    ids = (torch.arange(t * k) % t).to("tpu")
    src = torch.ones(t * k, feat, device="cpu").to("tpu")
    with execution_mode.set_eager_mode(
        execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
    ):
      out = torch.zeros(t, feat, device="tpu")
      result = out.scatter_add(0, ids.unsqueeze(-1).expand_as(src), src)
    mlir_text = tpu_torch_compile.serialize_mlir_text(
        tpu_torch_compile.build_mlir([result], [ids, src])
    )
    self.assertIn("stablehlo.scatter", mlir_text)
    self.assertIn("update_window_dims = [1]", mlir_text)
    self.assertNotIn("update_window_dims = []", mlir_text)

  @parameterized.product(
      dtype=[torch.float32, torch.bfloat16],
      rank=[3, 4],
      dim=[-2, -1],
  )
  def test_gather_multidim_broadcast_index(
      self, dtype: torch.dtype, rank: int, dim: int
  ):
    """gather with a multi-dimensional broadcast index matches CPU."""
    b, h, q, v, k = 2, 4, 8, 12, 6
    torch.manual_seed(42)
    lead_shape = (h,) if rank == 3 else (b, h)
    if dim == -1:
      x_cpu = torch.randn(*lead_shape, q, v, dtype=dtype)
      base_idx_cpu = torch.randint(0, v, (q, k))
      idx_view = (1,) * len(lead_shape) + (q, k)
      target_idx_shape = (*lead_shape, q, k)
    else:
      x_cpu = torch.randn(*lead_shape, v, q, dtype=dtype)
      base_idx_cpu = torch.randint(0, v, (k, q))
      idx_view = (1,) * len(lead_shape) + (k, q)
      target_idx_shape = (*lead_shape, k, q)

    def test_fn(device: torch.device) -> torch.Tensor:
      x = x_cpu.to(device)
      base_idx = base_idx_cpu.to(device)
      index = base_idx.view(idx_view).expand(*target_idx_shape)
      return torch.gather(x, dim, index)

    rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (None, None)
    self.assert_close_tpu_vs_cpu(test_fn, rtol=rtol, atol=atol)

  @parameterized.product(
      dtype=[torch.float32, torch.bfloat16],
      rank=[3, 4],
      dim=[-2, -1],
      nonzero_self=[False, True],
  )
  def test_scatter_add_multidim_broadcast_index(
      self, dtype: torch.dtype, rank: int, dim: int, nonzero_self: bool
  ):
    """scatter_add with a multi-dimensional broadcast index matches CPU."""
    b, h, q, v, k = 2, 4, 8, 6, 12
    torch.manual_seed(42)
    lead_shape = (h,) if rank == 3 else (b, h)
    if dim == -1:
      out_shape = (*lead_shape, q, v)
      src_shape = (*lead_shape, q, k)
      base_idx_cpu = torch.arange(q * k).reshape(q, k) % v
    else:
      out_shape = (*lead_shape, v, q)
      src_shape = (*lead_shape, k, q)
      base_idx_cpu = torch.arange(k * q).reshape(k, q) % v
    out_cpu = (
        torch.randn(*out_shape, dtype=dtype)
        if nonzero_self
        else torch.zeros(*out_shape, dtype=dtype)
    )
    src_cpu = torch.randn(*src_shape, dtype=dtype) * 0.1
    idx_view = (1,) * len(lead_shape) + base_idx_cpu.shape

    def test_fn(device: torch.device) -> torch.Tensor:
      out = out_cpu.to(device)
      src = src_cpu.to(device)
      base_idx = base_idx_cpu.to(device)
      index = base_idx.view(idx_view).expand(*src_shape)
      return out.scatter_add(dim, index, src)

    rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (None, None)
    self.assert_close_tpu_vs_cpu(test_fn, rtol=rtol, atol=atol)

  @parameterized.product(dtype=[torch.float32, torch.bfloat16])
  def test_gather_multidim_broadcast_index_backward(self, dtype: torch.dtype):
    """Forward and backward of broadcast-index gather match CPU."""
    b, h, q, v, k = 2, 4, 8, 10, 8
    torch.manual_seed(42)
    x_cpu = torch.randn(b, h, q, v, dtype=dtype)
    grad_out_cpu = torch.randn(b, h, q, k, dtype=dtype)
    base_idx_cpu = torch.arange(q * k).reshape(q, k) % v

    def test_fn(device: torch.device):
      x = x_cpu.to(device).detach().requires_grad_(True)
      grad_out = grad_out_cpu.to(device)
      base_idx = base_idx_cpu.to(device)
      index = base_idx.unsqueeze(0).unsqueeze(0).expand(b, h, q, k)
      out = torch.gather(x, -1, index)
      out.backward(grad_out)
      return out.detach(), x.grad.detach()

    rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (None, None)
    self.assert_close_tpu_vs_cpu(test_fn, rtol=rtol, atol=atol)

  def test_gather_and_scatter_add_multidim_broadcast_index_lowers_to_dot_general(
      self,
  ):
    """Multi-dim broadcast-index gather and scatter_add lower to dot_general."""
    h, q, v, k = 4, 8, 12, 8
    base_idx = (torch.arange(q * k).reshape(q, k) % v).to("tpu")
    x = torch.ones(h, q, v, device="cpu").to("tpu")
    src = torch.ones(h, q, k, device="cpu").to("tpu")

    with execution_mode.set_eager_mode(
        execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
    ):
      idx = base_idx.unsqueeze(0).expand(h, q, k)
      gathered = torch.gather(x, -1, idx)
    gather_mlir = tpu_torch_compile.serialize_mlir_text(
        tpu_torch_compile.build_mlir([gathered], [base_idx, x])
    )
    self.assertIn("stablehlo.dot_general", gather_mlir)
    self.assertNotIn("stablehlo.gather", gather_mlir)

    with execution_mode.set_eager_mode(
        execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
    ):
      out = torch.zeros(h, q, v, device="tpu")
      idx = base_idx.unsqueeze(0).expand(h, q, k)
      scattered = out.scatter_add(-1, idx, src)
    scatter_mlir = tpu_torch_compile.serialize_mlir_text(
        tpu_torch_compile.build_mlir([scattered], [base_idx, src])
    )
    self.assertIn("stablehlo.dot_general", scatter_mlir)
    self.assertNotIn("stablehlo.scatter", scatter_mlir)

  @parameterized.product(
      reduce_op=["add", "multiply"],
      dtype=[torch.float32, torch.int32, torch.bfloat16],
      value=[2.0, -1.5, 0.0],
  )
  def test_scatter_value_reduce(self, reduce_op, dtype, value):
    """Tests torch.ops.aten.scatter.value_reduce with various reduction ops."""
    if dtype.is_floating_point:
      scalar_value = float(value)
    else:
      scalar_value = int(value)

    def test_fn(device: torch.device) -> torch.Tensor:
      self_tensor = torch.ones(3, 5, dtype=dtype, device=device) * 3.0
      index = torch.tensor(
          [[0, 1, 2], [0, 1, 2]], device=device, dtype=torch.int64
      )
      return torch.ops.aten.scatter.value_reduce(
          self_tensor, dim=1, index=index, value=scalar_value, reduce=reduce_op
      )

    rtol, atol = None, None
    if dtype == torch.bfloat16:
      rtol, atol = 1e-2, 1e-2

    self.assert_close_tpu_vs_cpu(test_fn, rtol=rtol, atol=atol)

  def test_foreach_add_different_dtypes_with_alpha(self):
    """Tests _foreach_add with different dtypes and an alpha parameter."""

    def foreach_add_inside_fn(device):
      self_list = [
          torch.tensor([1.0, 2.0], dtype=torch.float32, device=device),
          torch.tensor([3, 4], dtype=torch.int32, device=device),
          torch.tensor([5.0, 6.0], dtype=torch.bfloat16, device=device),
      ]
      other_list = [
          torch.tensor([1, 2], dtype=torch.int64, device=device),
          torch.tensor([3.0, 4.0], dtype=torch.float32, device=device),
          torch.tensor([5.0, 6.0], dtype=torch.float64, device=device),
      ]
      return torch._foreach_add(self_list, other_list, alpha=1.5)

    self.assert_close_tpu_vs_cpu(foreach_add_inside_fn)

  def test_foreach_add_different_dtypes_no_alpha(self):
    """Tests _foreach_add with different dtypes and no alpha parameter."""

    def foreach_add_inside_fn(device):
      self_list = [
          torch.tensor([1.0, 2.0], dtype=torch.float32, device=device),
          torch.tensor([3, 4], dtype=torch.int32, device=device),
          torch.tensor([5.0, 6.0], dtype=torch.bfloat16, device=device),
      ]
      other_list = [
          torch.tensor([1, 2], dtype=torch.int64, device=device),
          torch.tensor([3.0, 4.0], dtype=torch.float32, device=device),
          torch.tensor([5.0, 6.0], dtype=torch.float64, device=device),
      ]
      return torch._foreach_add(self_list, other_list)

    self.assert_close_tpu_vs_cpu(foreach_add_inside_fn)

  def test_foreach_div_with_zero_scalar(self):
    """Tests _foreach_div with a zero scalar."""

    def foreach_div_inside_fn(device):
      self_list = [
          torch.tensor([1.0, 2.0], dtype=torch.float32, device=device),
          torch.tensor([3, 4], dtype=torch.int32, device=device),
          torch.tensor([5.0, 6.0], dtype=torch.bfloat16, device=device),
      ]
      return torch._foreach_div(self_list, 0)

    # Both TPU and CPU should return infinite values.
    self.assert_close_tpu_vs_cpu(foreach_div_inside_fn)

  def test_foreach_div_different_dtypes(self):
    """Tests _foreach_div with different dtypes (e.g. bfloat16 and float32)."""

    def foreach_div_inside_fn(device):
      self_list = [
          torch.tensor([1.0, 2.0], dtype=torch.bfloat16, device=device),
      ]
      other_list = [
          torch.tensor([2.0, 4.0], dtype=torch.float32, device=device),
      ]
      return torch._foreach_div(self_list, other_list)

    self.assert_close_tpu_vs_cpu(foreach_div_inside_fn)

  def test_foreach_div_inplace_different_dtypes(self):
    """Tests _foreach_div_ with different dtypes (e.g. bfloat16 and float32)."""

    def foreach_div_inplace_inside_fn(device):
      self_list = [
          torch.tensor([1.0, 2.0], dtype=torch.bfloat16, device=device),
      ]
      other_list = [
          torch.tensor([2.0, 4.0], dtype=torch.float32, device=device),
      ]
      torch._foreach_div_(self_list, other_list)
      return self_list

    self.assert_close_tpu_vs_cpu(foreach_div_inplace_inside_fn)

  def test_upsample_nearest_with_size_parameters(self):
    """Tests that the upsample nearest op works with size parameters."""

    device = torch.device("tpu")
    upsample_float32 = torch.tensor(
        [
            [
                [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                ],
            ],
        ],
        dtype=torch.float32,
        device=device,
    )

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nn.functional.interpolate(
            upsample_float32.to(device),
            size=(3, 4),
            mode="nearest",
        )
    )

  def test_arange_fractional_step(self):
    """Tests arange with fractional start, end, and step."""

    def _test_positive(device):
      return torch.arange(0.5, 3.5, 0.5, device=device)

    def _test_negative(device):
      return torch.arange(3.5, 0.5, -0.5, device=device)

    self.assert_close_tpu_vs_cpu(_test_positive)
    self.assert_close_tpu_vs_cpu(_test_negative)

  def test_arange_large_int(self):
    """Tests arange where start and end are large integers.

    This test makes sure that we correctly check the inputs even with integer
    values that can't be exactly represented in double precision.
    Both 2**53 and 2**53+1 are mapped to 2**53 when converted to a double
    precision floating point value.
    """
    i = 2**53

    def _test(device):
      return torch.arange(start=i, end=i + 1, step=1, device=device)

    self.assert_close_tpu_vs_cpu(_test)

  @absltest.skip("b/535216136")
  def test_arange_large_int_with_float_step(self):
    """Tests arange with large integers and a float step.

    Slight change from the test_arange_large_int, but that uses a step value
    where type(step) == float and forces the output dtype to be torch.int64.
    This would result in an incorrect result in the previous implementation
    because not all inputs are integers.
    """
    i = 2**53

    def _test(device):
      return torch.arange(
          start=i, end=i + 1, step=1.0, device=device, dtype=torch.int64
      )

    self.assert_close_tpu_vs_cpu(_test)

  def test_randperm(self):
    n = 1000
    seed = 4321
    device = torch.device("tpu")

    # Check that same seed produces same result
    torch.manual_seed(seed)
    t1 = torch.randperm(n, device=device)
    torch.manual_seed(seed)
    t2 = torch.randperm(n, device=device)
    self.assert_close(golden_result=t1.cpu(), torch_tpu_result=t2.cpu())

    # Check that different calls (without reset) produce different results
    t3 = torch.randperm(n, device=device)
    self.assertNotEqual(t2.cpu(), t3.cpu())

    # Check shape
    self.assertEqual(t1.shape, (n,))

    # Check that the sorted tensor is the same as the expected tensor
    t1_sorted = torch.sort(t1)
    expected_asc = torch.arange(n, device=torch.device("tpu"))
    self.assertEqual(t1_sorted.values.cpu(), expected_asc.cpu())
    self.assertNotEqual(t1.cpu(), expected_asc.cpu())

  def test_randperm_rng(self):
    """Verifies RNG results and state update for randperm."""
    device = torch.device("tpu")
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    # Golden Philox bits for seed=42, offset=0.
    golden_sequence = [8, 7, 0, 5, 9, 4, 6, 3, 1, 2]
    res = torch.randperm(10, generator=gen, device=device)
    self.assertEqual(res.cpu().tolist(), golden_sequence)
    state = gen.get_state()
    self.assertEqual(state[8].item(), 5)

  @absltest.skip("large n times out")
  def test_randperm_dtypes(self):
    device = torch.device("tpu")

    test_configs = [
        # Note: we run the "too large" test first to test that its (expected)
        # failure does not affect subsequent tests.
        (torch.int64, 2**31 + 1024),
        (torch.int64, 1000),
        (torch.float32, 1000),
        (torch.int32, 1000),
        (torch.bfloat16, 256),
    ]

    for dtype, n in test_configs:
      with self.subTest(dtype=dtype, n=n):
        t = torch.randperm(n, device=device, dtype=dtype)

        # Check that the output tensor has the expected shape and dtype
        self.assertEqual(t.dtype, dtype)
        self.assertEqual(t.shape, (n,))

        if n > 1000:
          # For large n, we check the max value
          try:
            max_val = t.max().item()
            self.assertEqual(
                max_val,
                n - 1,
                f"Max value mismatch: expected {n-1}, got {max_val}",
            )

            min_val = t.min().item()
            self.assertEqual(min_val, 0, "Min value should be 0.")
          except RuntimeError as e:
            if "ran out of memory" in str(e).lower():
              # Skip validation for large n due to host transfer limits
              print(
                  f"Warning: Skipping validation for n={n}"
                  f" due to host transfer limits: {e}"
              )
            else:
              raise e
        else:
          # For small n, we check that all values are unique and sorted
          unique_count = len(torch.unique(t.cpu()))
          self.assertEqual(
              unique_count,
              n,
              f"Duplicate values found in randperm for {dtype}. ",
          )

          t_sorted = torch.sort(t).values
          expected = torch.arange(n, device=device, dtype=dtype)
          self.assert_close(
              golden_result=t_sorted.cpu(), torch_tpu_result=expected.cpu()
          )

  def test_fft_rfft_norm_modes(self):
    n = 100
    input_cpu = torch.ones(1, n, dtype=torch.float32)
    input_tpu = input_cpu.to(torch.device("tpu"))
    dim = -1
    onesided = True

    # fft_rfft passes norm int 0 for 'backward', 1 for 'ortho',2 for 'forward'
    # to _fft_r2c implementation.
    norm_map = {"backward": 0, "ortho": 1, "forward": 2}

    for norm_str, norm_int in norm_map.items():
      with self.subTest(norm=norm_str, norm_int=norm_int):
        out_tpu = torch.ops.aten._fft_r2c(input_tpu, [dim], norm_int, onesided)

        # scale = dc_tpu / n, where DC component is n*scale for torch.ones(n)
        # scale = 1.0 for backward, 1.0/sqrt(n) for ortho, 1.0/n for forward
        dc_tpu = out_tpu[0, 0].cpu().abs().item()
        ratio_tpu = dc_tpu / n

        if norm_str == "backward":
          expected_ratio = 1.0
        elif norm_str == "ortho":
          expected_ratio = 1.0 / math.sqrt(n)
        elif norm_str == "forward":
          expected_ratio = 1.0 / n
        else:
          raise ValueError(f"Unknown norm_str: {norm_str}")

        self.assertTrue(math.isclose(ratio_tpu, expected_ratio))

        golden_result = torch.fft.rfft(input_cpu, n=n, dim=dim, norm=norm_str)
        tpu_result = torch.fft.rfft(input_tpu, n=n, dim=dim, norm=norm_str)
        self.assert_close(
            golden_result=golden_result.cpu(),
            torch_tpu_result=tpu_result.cpu(),
        )

  def test_fft_out_resizing(self):
    device = torch.device("tpu")

    # 1. fft.fft with cfloat input and 0-element / mismatched shape resize
    x_c64 = torch.randn(8, dtype=torch.complex64, device=device)
    out_fft_0 = torch.empty(0, dtype=torch.complex64, device=device)
    res_fft_0 = torch.fft.fft(x_c64, out=out_fft_0)
    self.assertEqual(res_fft_0.shape, torch.Size([8]))
    self.assertEqual(res_fft_0.dtype, torch.complex64)
    expected_fft = torch.fft.fft(x_c64.cpu())
    self.assert_close(
        golden_result=expected_fft,
        torch_tpu_result=res_fft_0.cpu(),
        rtol=1e-5,
        atol=1e-5,
    )

    out_fft_mismatch = torch.empty((2, 3), dtype=torch.complex64, device=device)
    res_fft_mismatch = torch.fft.fft(x_c64, out=out_fft_mismatch)
    self.assertEqual(res_fft_mismatch.shape, torch.Size([8]))
    self.assert_close(
        golden_result=expected_fft,
        torch_tpu_result=res_fft_mismatch.cpu(),
        rtol=1e-5,
        atol=1e-5,
    )

    # 2. fft.ifft with cfloat input and 0-element / mismatched shape resize
    out_ifft_0 = torch.empty(0, dtype=torch.complex64, device=device)
    res_ifft_0 = torch.fft.ifft(x_c64, out=out_ifft_0)
    self.assertEqual(res_ifft_0.shape, torch.Size([8]))
    self.assertEqual(res_ifft_0.dtype, torch.complex64)
    expected_ifft = torch.fft.ifft(x_c64.cpu())
    self.assert_close(
        golden_result=expected_ifft,
        torch_tpu_result=res_ifft_0.cpu(),
        rtol=1e-5,
        atol=1e-5,
    )

    out_ifft_mismatch = torch.empty(
        (4, 2), dtype=torch.complex64, device=device
    )
    res_ifft_mismatch = torch.fft.ifft(x_c64, out=out_ifft_mismatch)
    self.assertEqual(res_ifft_mismatch.shape, torch.Size([8]))
    self.assert_close(
        golden_result=expected_ifft,
        torch_tpu_result=res_ifft_mismatch.cpu(),
        rtol=1e-5,
        atol=1e-5,
    )

    # 3. fft.rfft with float32 input and 0-element / mismatched shape resize
    x_f32 = torch.randn(8, dtype=torch.float32, device=device)
    out_rfft_0 = torch.empty(0, dtype=torch.complex64, device=device)
    res_rfft_0 = torch.fft.rfft(x_f32, out=out_rfft_0)
    self.assertEqual(res_rfft_0.shape, torch.Size([5]))
    self.assertEqual(res_rfft_0.dtype, torch.complex64)
    expected_rfft = torch.fft.rfft(x_f32.cpu())
    self.assert_close(
        golden_result=expected_rfft,
        torch_tpu_result=res_rfft_0.cpu(),
        rtol=1e-5,
        atol=1e-5,
    )

    out_rfft_mismatch = torch.empty(
        (3, 2), dtype=torch.complex64, device=device
    )
    res_rfft_mismatch = torch.fft.rfft(x_f32, out=out_rfft_mismatch)
    self.assertEqual(res_rfft_mismatch.shape, torch.Size([5]))
    self.assert_close(
        golden_result=expected_rfft,
        torch_tpu_result=res_rfft_mismatch.cpu(),
        rtol=1e-5,
        atol=1e-5,
    )

    # 4. fft.irfft with cfloat input and 0-element / mismatched shape resize
    x_irfft = torch.randn(5, dtype=torch.complex64, device=device)
    out_irfft_0 = torch.empty(0, dtype=torch.float32, device=device)
    res_irfft_0 = torch.fft.irfft(x_irfft, n=8, out=out_irfft_0)
    self.assertEqual(res_irfft_0.shape, torch.Size([8]))
    self.assertEqual(res_irfft_0.dtype, torch.float32)
    expected_irfft = torch.fft.irfft(x_irfft.cpu(), n=8)
    self.assert_close(
        golden_result=expected_irfft,
        torch_tpu_result=res_irfft_0.cpu(),
        rtol=1e-5,
        atol=1e-5,
    )

    out_irfft_mismatch = torch.empty((10,), dtype=torch.float32, device=device)
    res_irfft_mismatch = torch.fft.irfft(x_irfft, n=8, out=out_irfft_mismatch)
    self.assertEqual(res_irfft_mismatch.shape, torch.Size([8]))
    self.assert_close(
        golden_result=expected_irfft,
        torch_tpu_result=res_irfft_mismatch.cpu(),
        rtol=1e-5,
        atol=1e-5,
    )

    # 5. Direct backend kernels: _fft_c2c.out, _fft_r2c.out, _fft_c2r.out
    out_c2c = torch.empty(0, dtype=torch.complex64, device=device)
    res_c2c = torch.ops.aten._fft_c2c.out(x_c64, [0], 0, True, out=out_c2c)
    self.assertEqual(res_c2c.shape, torch.Size([8]))
    self.assertEqual(res_c2c.dtype, torch.complex64)

    out_r2c = torch.empty(0, dtype=torch.complex64, device=device)
    res_r2c = torch.ops.aten._fft_r2c.out(x_f32, [0], 0, True, out=out_r2c)
    self.assertEqual(res_r2c.shape, torch.Size([5]))
    self.assertEqual(res_r2c.dtype, torch.complex64)

    out_c2r = torch.empty(0, dtype=torch.float32, device=device)
    res_c2r = torch.ops.aten._fft_c2r.out(x_irfft, [0], 0, 8, out=out_c2r)
    self.assertEqual(res_c2r.shape, torch.Size([8]))
    self.assertEqual(res_c2r.dtype, torch.float32)

  def test_fake_quantize_per_tensor_affine_cachemask(self):
    scale = 1 / 255
    zero_point = 10
    quant_min = -127
    quant_max = 127

    input_tensor = torch.tensor(
        [
            0,  # zero point
            (quant_min - zero_point - 5)
            * scale,  # dequantized value smaller than dequantized quant_min
            (quant_max - zero_point + 5)
            * scale,  # dequantized value larger than dequantized quant_max
            (37 - zero_point - 1) * scale,  # arbitrary value within range
        ],
        dtype=torch.float32,
    )

    def test_fn(device):
      input_tensor_device = input_tensor.to(device)
      fake_quantized_tensor, mask = (
          torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
              input_tensor_device,
              scale=scale,
              zero_point=zero_point,
              quant_min=quant_min,
              quant_max=quant_max,
          )
      )
      return fake_quantized_tensor, mask

    cpu_res_tensor, cpu_mask = test_fn("cpu")
    tpu_res_tensor, tpu_mask = test_fn(torch.device("tpu"))

    self.assert_close(
        golden_result=cpu_res_tensor.cpu(),
        torch_tpu_result=tpu_res_tensor.cpu(),
    )

    self.assertEqual(cpu_mask.cpu(), tpu_mask.cpu())

  def test_embedding_bag_max_indices(self):
    """Tests that max_indices is computed correctly in _embedding_bag."""
    weight = torch.tensor(
        [[1.0, 5.0], [10.0, 2.0], [10.0, 8.0]], dtype=torch.float32
    )
    indices = torch.tensor([0, 1, 2], dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)

    def test_fn(device):
      # returns output, grad_input, grad_weight, max_indices
      output, _, _, max_indices = torch.ops.aten._embedding_bag(
          weight.to(device), indices.to(device), offsets.to(device), mode=2
      )

      expected_output = torch.tensor([[10.0, 8.0]])
      expected_max_indices = torch.tensor([[1, 2]], dtype=torch.long)
      self.assert_close(
          golden_result=expected_output.cpu(),
          torch_tpu_result=output.cpu(),
      )
      self.assert_close(
          golden_result=expected_max_indices.cpu(),
          torch_tpu_result=max_indices.cpu(),
      )
      return output, max_indices

    self.assert_close_tpu_vs_cpu(test_fn)

  @parameterized.product(mode=[0, 1])
  def test_embedding_bag_max_indices_sum_mean(self, mode):
    """Tests that max_indices has shape [batch_size] when mode is sum or mean."""
    weight = torch.tensor(
        [[1.0, 5.0], [10.0, 2.0], [10.0, 8.0]], dtype=torch.float32
    )
    indices = torch.tensor([0, 1, 2], dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)

    def test_fn(device):
      w = weight.clone().detach().to(device).requires_grad_(True)
      output, _, _, max_indices = torch.ops.aten._embedding_bag(
          w, indices.to(device), offsets.to(device), mode=mode
      )
      expected_shape = (offsets.numel(),)
      self.assertEqual(max_indices.shape, expected_shape)
      output.sum().backward()
      return output, max_indices, w.grad

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_embedding_bag_empty_bag(self):
    """Tests that empty bags are handled correctly in _embedding_bag."""
    weight = torch.randn(5, 2).to(torch.device("tpu"))
    indices = torch.tensor([0, 1, 2], dtype=torch.long).to(torch.device("tpu"))
    # The first bag is empty
    offsets = torch.tensor([0, 0], dtype=torch.long).to(torch.device("tpu"))

    output_sum, _, _, _ = torch.ops.aten._embedding_bag(
        weight, indices, offsets, mode=0
    )
    self.assertTrue(torch.all(output_sum[0] == 0))

    output_mean, _, _, _ = torch.ops.aten._embedding_bag(
        weight, indices, offsets, mode=1
    )
    self.assertTrue(torch.all(output_mean[0] == 0))

    output_max, _, _, max_indices = torch.ops.aten._embedding_bag(
        weight, indices, offsets, mode=2
    )
    self.assertTrue(torch.all(output_max[0] == 0))
    self.assertTrue(torch.all(max_indices[0] == 0))

  def test_zero_sized_to_device(self):
    tensor = torch.ones(2, 0, 3, dtype=torch.int32, device="cpu")
    tensor_tpu = tensor.to(torch.device("tpu"))

    # Zero-sized tensors are constructed as deferred constants, rather than
    # actually transferring 0 bytes.
    self.assertFalse(sync.is_materializing(tensor_tpu))

  @parameterized.product(
      dtype=[
          torch.float32,
          torch.float64,
          torch.float16,
          torch.bfloat16,
      ],
      dim=[0, 3],
  )
  def test_weight_norm_interface_dim(self, dtype, dim):
    """Tests torch.ops.aten._weight_norm_interface with various dims."""
    v = torch.randn(2, 3, 4, 5, dtype=dtype)
    g = torch.randn(v.shape[dim], dtype=dtype)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ops.aten._weight_norm_interface(
            v.to(device), g.to(device), dim
        ),
    )

  @parameterized.product(
      dtype=[
          torch.float32,
          torch.float64,
          torch.float16,
          torch.bfloat16,
      ],
  )
  def test_weight_norm_interface_scalar_g(self, dtype):
    """Tests torch.ops.aten._weight_norm_interface with scalar g."""
    v = torch.randn(1, 3, 4, dtype=dtype)
    g = torch.randn((), dtype=dtype)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ops.aten._weight_norm_interface(
            v.to(device), g.to(device), 0
        ),
    )

  def test_weight_norm_interface_m(self):
    """Tests torch.ops.aten._weight_norm_interface with 1D g."""
    v = torch.randn(2, 3, 4, dtype=torch.float32)
    g = torch.randn(2, dtype=torch.float32)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ops.aten._weight_norm_interface(
            v.to(device), g.to(device), 0
        ),
    )

  def test_weight_norm_interface_all_inputs(self):
    """Tests torch.ops.aten._weight_norm_interface with all possible inputs."""
    test_cases = [
        # Last dim
        ((2, 3, 4), (4,), 2),
        # First dim
        ((2, 3, 4), (2,), 0),
        # Dimensions match, but g is not 1D (first dim)
        ((5, 10, 7), (5, 1, 1), 0),
        # Dimensions match, but g is not 1D (last dim)
        ((2, 3, 4), (1, 1, 4), 2),
        # Minimal dimension (Rank 1)
        ((10,), (10,), 0),
        # Rank-2 v with same-rank g
        ((2, 3), (2, 1), 0),
        ((2, 3), (1, 3), 1),
    ]

    for v_shape, g_shape, dim in test_cases:
      with self.subTest(v_shape=v_shape, g_shape=g_shape, dim=dim):
        v = torch.randn(v_shape, dtype=torch.float32)
        g = torch.randn(g_shape, dtype=torch.float32)
        self.assert_close_tpu_vs_cpu(
            lambda device, v=v, g=g, dim=dim: torch.ops.aten._weight_norm_interface(
                v.to(device), g.to(device), dim
            ),
        )

  def test_weight_norm_interface_norm_shape(self):
    """Tests that the returned norm has the expected shape (matching g)."""
    device = torch.device("tpu")
    v = torch.randn(2, 3, 4, device=device)
    g = torch.randn(2, device=device)

    _, norm = torch.ops.aten._weight_norm_interface(v, g, 0)
    self.assertEqual(norm.shape, (2,))

    g_last = torch.randn(4, device=device)
    _, norm_last = torch.ops.aten._weight_norm_interface(v, g_last, 2)
    self.assertEqual(norm_last.shape, (4,))

    g_md = torch.randn(2, 1, 1, device=device)
    _, norm_md = torch.ops.aten._weight_norm_interface(v, g_md, 0)
    self.assertEqual(norm_md.shape, (2, 1, 1))

    g_md_last = torch.randn(1, 1, 4, device=device)
    _, norm_md_last = torch.ops.aten._weight_norm_interface(v, g_md_last, 2)
    self.assertEqual(norm_md_last.shape, (1, 1, 4))

  @parameterized.parameters(
      ((8, 4, 5), 2, torch.float32),
      ((2, 3, 4), 0, torch.bfloat16),
      ((5, 10, 7), 0, torch.float16),
      ((10,), 0, torch.float64),
      ((2, 3), 0, torch.float32),
      ((2, 3), 1, torch.bfloat16),
      ((2, 3, 4, 5), 3, torch.float16),
      ((2, 3, 4, 5), 0, torch.float64),
  )
  def test_weight_norm_backward(self, shape, dim, dtype):
    """Tests torch.nn.utils.weight_norm backward across shapes, dims, and dtypes."""
    weight = torch.randn(shape, dtype=dtype)
    grad_w = torch.randn(shape, dtype=dtype)

    class InlineWeightNormModule(torch.nn.Module):

      def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w)

      def forward(self):
        return self.weight

    def run_module(device):
      m = InlineWeightNormModule(weight.clone()).to(device)
      torch.nn.utils.weight_norm(m, name="weight", dim=dim)
      w = m()
      w.backward(grad_w.to(device))
      return m.weight_v.grad, m.weight_g.grad

    self.assert_close_tpu_vs_cpu(run_module)

  @parameterized.product(
      input_dtype=[
          torch.float32,
          torch.bfloat16,
          torch.float16,
      ],
      target_dtype=[
          torch.float32,
      ],
  )
  def test_mm_dtype(self, input_dtype, target_dtype):
    """Tests torch.ops.aten.mm.dtype with various dtypes."""
    lhs = torch.randn(2, 3, dtype=input_dtype)
    rhs = torch.randn(3, 4, dtype=input_dtype)

    def test_fn(device):
      l = lhs.to(device)
      r = rhs.to(device)

      if device == "cpu":
        # CPU backend does not support mm.dtype directly
        return torch.mm(l.to(target_dtype), r.to(target_dtype))
      else:
        return torch.ops.aten.mm.dtype(l, r, target_dtype)

    self.assert_close_tpu_vs_cpu(
        test_fn,
        rtol=2e-2,
        atol=6e-3,
    )

  def test_bucketize_scalar_empty_boundaries(self):
    """Tests torch.bucketize with empty boundaries."""

    def run(device):
      return torch.bucketize(
          self=1.234,
          boundaries=torch.tensor([], dtype=torch.float32).to(device),
      )

    self.assert_close_tpu_vs_cpu(run)

  @parameterized.product(
      input_scalar=[0, 1, 1.5, 2, 3, 4.5, 5, 6],
      out_int32=[True, False],
      right=[True, False],
      dtype=[
          torch.uint8,
          torch.int32,
          torch.bfloat16,
          torch.float32,
          torch.float64,
      ],
  )
  def test_bucketize_scalar(self, input_scalar, out_int32, right, dtype):
    def run(device):
      return torch.bucketize(
          self=input_scalar,
          boundaries=torch.tensor([1, 2, 4, 5], dtype=dtype).to(device),
          out_int32=out_int32,
          right=right,
      )

    self.assert_close_tpu_vs_cpu(run)

  @parameterized.product(
      input_shape=[(7,), (3, 7), (2, 3, 7)],
      out_int32=[True, False],
      right=[True, False],
      dtype=[
          torch.uint8,
          torch.int32,
          torch.bfloat16,
          torch.float32,
          torch.float64,
      ],
  )
  def test_bucketize(self, input_shape, out_int32, right, dtype):
    input_tensor = torch.tensor([3, 5, 6, 0, 1, 1.5, 2], dtype=dtype).repeat(
        input_shape
    )
    boundaries = torch.tensor([1, 2, 4, 5], dtype=dtype)

    def run(device):
      return torch.bucketize(
          input_tensor.to(device),
          boundaries.to(device),
          out_int32=out_int32,
          right=right,
      )

    self.assert_close_tpu_vs_cpu(run)

  def test_bucketize_out(self):
    tpu_device = torch.device("tpu")
    input_tensor = torch.tensor(
        [[0, 2, 3], [3, 4, 6]], dtype=torch.float32, device=tpu_device
    )
    boundaries = torch.tensor([1, 3, 5], dtype=torch.float32, device=tpu_device)
    out = torch.empty((2, 3), dtype=torch.int64, device=tpu_device)
    torch.bucketize(input_tensor, boundaries, out=out)

    self.assertEqual(
        out.cpu(), torch.tensor([[0, 1, 1], [1, 2, 3]], dtype=torch.int64)
    )

  @parameterized.product(
      shape=[
          (4, 4),
          (10, 4),
          (4, 10),
          (2, 3, 4),
          (0, 4),
          (4, 0),
          (0, 0),
      ],
      dtype=[
          torch.float32,
          torch.float64,
          torch.complex64,
      ],
  )
  def test_geqrf(self, shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype)

    def run(device):
      return torch.geqrf(input_tensor.to(device))

    self.assert_close_tpu_vs_cpu(run, rtol=4e-6, atol=1e-6)

  @parameterized.product(
      shape=[
          (4, 4),
          (10, 4),
          (4, 10),
          (2, 3, 4),
          (0, 4),
          (4, 0),
          (0, 0),
      ],
      dtype=[
          torch.float32,
          torch.float64,
          torch.complex64,
      ],
      mode=["reduced", "complete", "r"],
  )
  def test_linalg_qr(self, shape, dtype, mode):
    input_tensor = torch.randn(shape, dtype=dtype)

    def run(device):
      return torch.linalg.qr(input_tensor.to(device), mode=mode)

    self.assert_close_tpu_vs_cpu(run, rtol=7e-6, atol=1e-6)

  def test_linalg_qr_out(self):
    shape = (5, 3)
    input_tensor = torch.randn(shape, dtype=torch.float32)

    def run(device):
      m, n = shape
      k = min(m, n)
      q = torch.empty((m, k), dtype=torch.float32, device=device)
      r = torch.empty((k, n), dtype=torch.float32, device=device)
      torch.linalg.qr(input_tensor.to(device), mode="reduced", out=(q, r))
      return q, r

    self.assert_close_tpu_vs_cpu(run)

  @parameterized.product(
      shape=[
          (4, 4),
          (10, 10),
          (2, 3, 3),
          (2, 3, 4, 4),
          (0, 0),
          (0, 3, 3),
          (2, 0, 0),
      ],
      dtype=[
          torch.float32,
          torch.float64,
          torch.complex64,
          torch.complex128,
      ],
      upper=[False, True],
  )
  def test_linalg_cholesky_ex(self, shape, dtype, upper):
    n = shape[-1]
    a = torch.randn(shape, dtype=dtype)
    eye = torch.eye(n, dtype=dtype).expand(shape)
    spd = a @ a.mH + eye * max(n, 1)
    if (
        dtype == torch.complex128
        and oss_utils.is_oss()
        and (ver := utils.libtpu_version()) is not None
        and ver < (0, 0, 52)
    ):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=OSS libtpu < 0.0.52 version guard.
          RuntimeError, "complex128 requires libtpu >= 0.0.52"
      ):
        torch.linalg.cholesky_ex(spd.to("tpu"), upper=upper)
      return

    def run(device):
      l_func, info_func = torch.linalg.cholesky_ex(spd.to(device), upper=upper)
      l_out = torch.empty_like(spd, device=device)
      info_out = torch.empty(shape[:-2], dtype=torch.int32, device=device)
      torch.linalg.cholesky_ex(
          spd.to(device), upper=upper, out=(l_out, info_out)
      )
      # Also test pre-allocated non-contiguous (Fortran-strided)
      # destination `l`.
      l_noncontig = torch.empty_like(spd, device=device).mT
      info_noncontig = torch.empty(shape[:-2], dtype=torch.int32, device=device)
      torch.linalg.cholesky_ex(
          spd.to(device), upper=upper, out=(l_noncontig, info_noncontig)
      )
      return (
          l_func,
          info_func,
          l_out,
          info_out,
          l_noncontig,
          info_noncontig,
      )

    rtol = 1e-6 if dtype in (torch.float64, torch.complex128) else 1e-4
    atol = 1e-6 if dtype in (torch.float64, torch.complex128) else 1e-4
    self.assert_close_tpu_vs_cpu(run, rtol=rtol, atol=atol)

  @parameterized.product(
      dtype=[
          torch.float32,
          torch.float64,
          torch.complex64,
          torch.complex128,
      ],
      upper=[False, True],
  )
  def test_linalg_cholesky_ex_non_positive_definite(self, dtype, upper):
    # Batch of 3 matrices:
    # - batch 0: positive-definite (succeeds, info == 0)
    # - batch 1: fails at 1st leading minor (A[0, 0] == -1, info == 1)
    # - batch 2: fails at 2nd leading minor (A[1, 1] == -1, info > 0)
    batch = torch.stack([
        torch.tensor([[4.0, 0.0], [0.0, 9.0]], dtype=dtype),
        torch.tensor([[-1.0, 0.0], [0.0, 4.0]], dtype=dtype),
        torch.tensor([[4.0, 0.0], [0.0, -1.0]], dtype=dtype),
    ]).to("tpu")
    if (
        dtype == torch.complex128
        and oss_utils.is_oss()
        and (ver := utils.libtpu_version()) is not None
        and ver < (0, 0, 52)
    ):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=OSS libtpu < 0.0.52 version guard.
          RuntimeError, "complex128 requires libtpu >= 0.0.52"
      ):
        torch.linalg.cholesky_ex(batch, upper=upper, check_errors=False)
      return

    l_tpu, info_tpu = torch.linalg.cholesky_ex(
        batch, upper=upper, check_errors=False
    )
    info_cpu = info_tpu.cpu()
    self.assertEqual(info_cpu[0].item(), 0)
    self.assertEqual(info_cpu[1].item(), 1)
    self.assertGreater(info_cpu[2].item(), 0)

    # Verify the valid batch element (index 0) still matches CPU factor.
    l_expected, _ = torch.linalg.cholesky_ex(batch[0].cpu(), upper=upper)
    self.assertEqual(l_tpu[0].cpu(), l_expected)

  def test_linalg_cholesky_multivariate_normal_and_constraint(self):
    from torch.distributions import constraints, MultivariateNormal

    covariance = torch.eye(4, device="tpu") * 2.0
    self.assertTrue(constraints.positive_definite.check(covariance).item())
    non_pd = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], device="tpu")
    self.assertFalse(constraints.positive_definite.check(non_pd).item())

    mean = torch.zeros(4, device="tpu")
    dist = MultivariateNormal(mean, covariance_matrix=covariance)
    sample = dist.sample((10,))
    self.assertEqual(sample.shape, (10, 4))
    self.assertEqual(sample.device.type, "tpu")
    log_prob = dist.log_prob(sample)
    self.assertEqual(log_prob.shape, (10,))

    # Test the exact HuggingFace transformers resize_token_embeddings pattern:
    old_embeddings_weight = torch.randn(100, 32, device="tpu")
    mean_embeddings = old_embeddings_weight.mean(dim=0)
    old_centered_embeddings = old_embeddings_weight - mean_embeddings
    cov = old_centered_embeddings.T @ old_centered_embeddings / 100
    eps = 1e-4
    cov_reg = eps * cov + torch.eye(32, device="tpu") * 1e-6
    self.assertTrue(constraints.positive_definite.check(cov_reg).all().item())
    mvn = MultivariateNormal(mean_embeddings, covariance_matrix=cov_reg)
    new_weights = mvn.sample((15,))
    self.assertEqual(new_weights.shape, (15, 32))
    self.assertEqual(new_weights.device.type, "tpu")

  def test_linalg_cholesky_ex_out_views_and_strides(self):
    device = "tpu"
    # 1. Sliced view with non-zero storage_offset
    base_l = torch.zeros(2, 2, 2, dtype=torch.float32, device=device)
    l_slice = base_l[1]  # offset = 4
    base_info = torch.full((2,), -1, dtype=torch.int32, device=device)
    info_slice = base_info[1]  # offset = 1

    a = torch.tensor(
        [[4.0, 0.0], [0.0, 9.0]], dtype=torch.float32, device=device
    )
    torch.linalg.cholesky_ex(a, out=(l_slice, info_slice))

    self.assertEqual(l_slice.storage_offset(), 4)
    self.assertEqual(info_slice.storage_offset(), 1)
    self.assertTrue(torch.equal(base_l[0].cpu(), torch.zeros(2, 2)))
    self.assertTrue(
        torch.equal(base_l[1].cpu(), torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
    )
    self.assertTrue(
        torch.equal(base_info.cpu(), torch.tensor([-1, 0], dtype=torch.int32))
    )

    # 2. Transposed out tensor strides preserved
    l_transposed = torch.empty(2, 2, dtype=torch.float32, device=device).t()
    self.assertEqual(l_transposed.stride(), (1, 2))
    info = torch.empty((), dtype=torch.int32, device=device)
    torch.linalg.cholesky_ex(a, out=(l_transposed, info))
    self.assertEqual(l_transposed.stride(), (1, 2))
    self.assertTrue(
        torch.equal(l_transposed.cpu(), torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
    )

  def _assert_all_nan(self, *tensors):
    for t in tensors:
      self.assertTrue(torch.isnan(t.to("cpu")).all())

  def _assert_no_nan(self, *tensors):
    for t in tensors:
      self.assertFalse(torch.isnan(t.to("cpu")).any())

  def _verify_svd_invariants(self, u, s, vh):
    u = u.to("cpu")
    s = s.to("cpu")
    vh = vh.to("cpu")

    k = s.shape[-1]
    u_sliced = u[..., :k]
    vh_sliced = vh[..., :k, :]

    u_ortho = u_sliced.adjoint() @ u_sliced
    vh_ortho = vh_sliced @ vh_sliced.adjoint()
    s_expanded = s.unsqueeze(-2)
    reconstructed = (u_sliced * s_expanded) @ vh_sliced

    return s, reconstructed, u_ortho, vh_ortho

  def _test_svd_helper(
      self, input_tensor, full_matrices=True, atol=None, rtol=None
  ):
    def run(device):
      u, s, vh = torch.linalg.svd(
          input_tensor.to(device), full_matrices=full_matrices
      )
      return self._verify_svd_invariants(u, s, vh)

    self.assert_close_tpu_vs_cpu(run, atol=atol, rtol=rtol)

  @parameterized.product(
      shape=[
          (4, 4),
          (10, 4),
          (4, 10),
          (2, 3, 4),
      ],
      dtype=[
          torch.float32,
          torch.complex64,
      ],
      full_matrices=[True, False],
  )
  def test_linalg_svd(self, shape, dtype, full_matrices):
    input_tensor = torch.randn(shape, dtype=dtype)
    tolerances = {
        torch.float32: (1.1e-02, 1.8e-02),
        torch.complex64: (2.3e-02, 1.2e-02),
    }
    rtol, atol = tolerances[dtype]
    self._test_svd_helper(input_tensor, full_matrices, rtol=rtol, atol=atol)

  @parameterized.product(
      shape=[
          (4, 4),
          (2, 3, 4),
      ],
      full_matrices=[True, False],
  )
  def test_linalg_svd_complex128(self, shape, full_matrices):
    input_tensor = torch.randn(shape, dtype=torch.complex128)

    def run(device):
      if torch.device(device).type == "cpu":
        # TorchTPU computes complex128 SVD by downcasting to complex64 and
        # upcasting results back. We mirror this on CPU for a fair comparison.
        u, s, vh = torch.linalg.svd(
            input_tensor.to(torch.complex64), full_matrices=full_matrices
        )
        u = u.to(torch.complex128)
        s = s.to(torch.float64)
        vh = vh.to(torch.complex128)
      else:
        u, s, vh = torch.linalg.svd(
            input_tensor.to(device), full_matrices=full_matrices
        )
      return self._verify_svd_invariants(u, s, vh)

    self.assert_close_tpu_vs_cpu(run, rtol=2.3e-02, atol=1.2e-02)

  @parameterized.product(
      shape=[
          (4, 4),
          (2, 3, 4),
      ],
      full_matrices=[True, False],
  )
  def test_linalg_svd_float64(self, shape, full_matrices):
    input_tensor = torch.randn(shape, dtype=torch.float64)
    self._test_svd_helper(input_tensor, full_matrices=full_matrices)

  @parameterized.product(
      shape=[
          (9, 8),  # Tall, no QR preprocess (9 <= 1.15 * 8)
          (8, 9),  # Wide, no QR preprocess
      ],
      dtype=[
          torch.float32,
          torch.complex64,
      ],
      full_matrices=[True, False],
  )
  def test_linalg_svd_extra_shapes(self, shape, dtype, full_matrices):
    input_tensor = torch.randn(shape, dtype=dtype)
    tolerances = {
        torch.float32: (1.4e-02, 1.6e-02),
        torch.complex64: (2.3e-02, 2.2e-02),
    }
    rtol, atol = tolerances[dtype]
    self._test_svd_helper(input_tensor, full_matrices, rtol=rtol, atol=atol)

  @parameterized.parameters(
      {"some": True},
      {"some": False},
  )
  def test_torch_legacy_svd(self, some):
    input_tensor = torch.randn(5, 3, dtype=torch.float64)

    def run(device):
      u, s, v = torch.svd(input_tensor.to(device), some=some)
      vh = v.adjoint() if v.is_complex() else v.transpose(-2, -1)
      return self._verify_svd_invariants(u, s, vh)

    self.assert_close_tpu_vs_cpu(run)

  @parameterized.product(
      shape=[
          (0, 3),
          (5, 0),
          (0, 0),
          (0, 5, 3),
          (2, 0, 3),
          (2, 5, 0),
      ],
      full_matrices=[True, False],
  )
  def test_linalg_svd_zero_sized(self, shape, full_matrices):
    input_tensor = torch.randn(shape, dtype=torch.float32)

    def run(device):
      u, s, vh = torch.linalg.svd(
          input_tensor.to(device), full_matrices=full_matrices
      )
      # For zero-sized inputs, outputs are deterministic (empty or identity),
      # so we can compare them directly.
      return u, s, vh

    self.assert_close_tpu_vs_cpu(run)

  @parameterized.product(
      shape=[
          (4, 4),
          (2, 3, 4),
      ]
  )
  def test_linalg_svdvals(self, shape):
    input_tensor = torch.randn(shape, dtype=torch.float64)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.linalg.svdvals(input_tensor.to(device))
    )

  @parameterized.product(
      shape=[
          (0, 3),
          (5, 0),
          (0, 0),
          (2, 0, 3),
          (2, 5, 0),
      ],
  )
  def test_linalg_svdvals_zero_sized(self, shape):
    input_tensor = torch.randn(shape, dtype=torch.float32)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.linalg.svdvals(input_tensor.to(device))
    )

  def test_linalg_svd_all_zeros(self):
    input_tensor = torch.zeros(5, 3, dtype=torch.float32)
    self._test_svd_helper(input_tensor, full_matrices=False)

  def test_linalg_svd_nan(self):
    input_tensor = torch.randn(5, 3, dtype=torch.float32)
    input_tensor[2, 1] = float("nan")

    device = torch.device("tpu")
    u, s, vh = torch.linalg.svd(input_tensor.to(device), full_matrices=False)
    self._assert_all_nan(u, s, vh)

  def test_linalg_svd_nan_batched(self):
    input_tensor = torch.randn(2, 5, 3, dtype=torch.float64)
    input_tensor[0, 2, 1] = float("nan")

    device = torch.device("tpu")
    u, s, vh = torch.linalg.svd(input_tensor.to(device), full_matrices=False)
    self._assert_all_nan(u[0], s[0], vh[0])
    self._assert_no_nan(u[1], s[1], vh[1])

  def test_linalg_svd_tiny_element(self):
    dtype = torch.float32
    tiny_element = torch.finfo(dtype).tiny
    input_tensor = torch.zeros(5, 3, dtype=dtype)
    input_tensor[2, 1] = tiny_element
    self._test_svd_helper(input_tensor, full_matrices=False)

  def test_linalg_svd_inf(self):
    input_tensor = torch.randn(5, 3, dtype=torch.float32)
    input_tensor[2, 1] = float("inf")

    device = torch.device("tpu")
    u, s, vh = torch.linalg.svd(input_tensor.to(device), full_matrices=False)
    self._assert_all_nan(u, s, vh)

  def test_linalg_svd_inf_batched(self):
    input_tensor = torch.randn(2, 5, 3, dtype=torch.float64)
    input_tensor[0, 2, 1] = float("inf")

    device = torch.device("tpu")
    u, s, vh = torch.linalg.svd(input_tensor.to(device), full_matrices=False)
    self._assert_all_nan(u[0], s[0], vh[0])
    self._assert_no_nan(u[1], s[1], vh[1])

  @parameterized.parameters(
      torch.complex64,
      torch.complex128,
  )
  def test_linalg_svd_nan_complex(self, dtype):
    input_tensor = torch.randn(5, 3, dtype=dtype)
    # Set real part to NaN
    input_tensor[2, 1] = complex(float("nan"), 1.0)
    device = torch.device("tpu")
    u, s, vh = torch.linalg.svd(input_tensor.to(device), full_matrices=False)
    self._assert_all_nan(u, s, vh)

    # Set imag part to NaN
    input_tensor = torch.randn(5, 3, dtype=dtype)
    input_tensor[2, 1] = complex(1.0, float("nan"))
    u, s, vh = torch.linalg.svd(input_tensor.to(device), full_matrices=False)
    self._assert_all_nan(u, s, vh)

  @parameterized.parameters(
      torch.complex64,
      torch.complex128,
  )
  def test_linalg_svd_inf_complex(self, dtype):
    input_tensor = torch.randn(5, 3, dtype=dtype)
    # Set real part to Inf
    input_tensor[2, 1] = complex(float("inf"), 1.0)
    device = torch.device("tpu")
    u, s, vh = torch.linalg.svd(input_tensor.to(device), full_matrices=False)
    self._assert_all_nan(u, s, vh)

    # Set imag part to Inf
    input_tensor = torch.randn(5, 3, dtype=dtype)
    input_tensor[2, 1] = complex(1.0, float("inf"))
    u, s, vh = torch.linalg.svd(input_tensor.to(device), full_matrices=False)
    self._assert_all_nan(u, s, vh)

  def test_linalg_svd_identity(self):
    input_tensor = torch.eye(5, dtype=torch.float32)
    self._test_svd_helper(input_tensor, full_matrices=True)

  def test_linalg_svd_ill_conditioned(self):
    dtype = torch.float64
    u_rand = torch.randn(5, 5, dtype=dtype)
    u, _ = torch.linalg.qr(u_rand)
    vh_rand = torch.randn(5, 5, dtype=dtype)
    vh, _ = torch.linalg.qr(vh_rand)
    s = torch.tensor([1.0, 1e-3, 1e-6, 1e-10, 1e-15], dtype=dtype)
    input_tensor = u @ torch.diag(s) @ vh
    self._test_svd_helper(input_tensor, full_matrices=True)

  @parameterized.product(
      shape=[
          (3, 5),
          (4, 4),
      ],
      zero_dims=[1, 2],
  )
  def test_linalg_svd_rank_deficient(self, shape, zero_dims):
    input_tensor = torch.randn(shape, dtype=torch.float64)
    m, n = shape
    if m >= n:
      input_tensor[..., :, n - zero_dims :] = 0.0
    else:
      input_tensor[..., m - zero_dims :, :] = 0.0

    self._test_svd_helper(input_tensor, full_matrices=True)

  @parameterized.product(
      zero_dims_batch=[(0, 1), (2, 1)],
  )
  def test_linalg_svd_rank_deficient_batched(self, zero_dims_batch):
    shape = (2, 5, 3)
    input_tensor = torch.randn(shape, dtype=torch.float64)
    m, n = shape[-2:]
    for b in range(shape[0]):
      zero_dims = zero_dims_batch[b]
      if zero_dims > 0:
        if m >= n:
          input_tensor[b, :, n - zero_dims :] = 0.0
        else:
          input_tensor[b, m - zero_dims :, :] = 0.0

    self._test_svd_helper(input_tensor, full_matrices=True)

  @parameterized.product(
      shape=[
          (1000, 2),  # Large tall, QR preprocess
          (2, 1000),  # Large wide, QR preprocess
      ]
  )
  def test_linalg_svd_large(self, shape):
    input_tensor = torch.randn(shape, dtype=torch.float64)
    self._test_svd_helper(input_tensor, full_matrices=False)

  def test_linalg_svd_out(self):
    dtype = torch.float64
    shape = (5, 3)
    input_tensor = torch.randn(shape, dtype=dtype)

    def run(device):
      m, n = shape
      k = min(m, n)
      u = torch.empty((m, k), dtype=dtype, device=device)
      s = torch.empty((k,), dtype=dtype, device=device)
      vh = torch.empty((k, n), dtype=dtype, device=device)
      torch.linalg.svd(
          input_tensor.to(device), full_matrices=False, out=(u, s, vh)
      )
      return self._verify_svd_invariants(u, s, vh)

    self.assert_close_tpu_vs_cpu(run)

  def test_linalg_svd_variants(self):
    dtype = torch.float64
    shape = (5, 3)
    input_tensor = torch.randn(shape, dtype=dtype)

    # 1. linalg_svd (public functional)
    def run_linalg_svd(device):
      u, s, vh = torch.linalg.svd(input_tensor.to(device), full_matrices=False)
      return self._verify_svd_invariants(u, s, vh)

    self.assert_close_tpu_vs_cpu(run_linalg_svd)

    # 2. linalg_svd.U (public out)
    def run_linalg_svd_out(device):
      m, n = shape
      k = min(m, n)
      u = torch.empty((m, k), dtype=dtype, device=device)
      s = torch.empty((k,), dtype=dtype, device=device)
      vh = torch.empty((k, n), dtype=dtype, device=device)
      torch.linalg.svd(
          input_tensor.to(device), full_matrices=False, out=(u, s, vh)
      )
      return self._verify_svd_invariants(u, s, vh)

    self.assert_close_tpu_vs_cpu(run_linalg_svd_out)

    # 3. _linalg_svd (private functional)
    def run_private_svd(device):
      u, s, vh = torch._linalg_svd(
          input_tensor.to(device), full_matrices=False, compute_uv=True
      )
      return self._verify_svd_invariants(u, s, vh)

    self.assert_close_tpu_vs_cpu(run_private_svd)

    # 4. _linalg_svd.U (private out)
    def run_private_svd_out(device):
      m, n = shape
      k = min(m, n)
      u = torch.empty((m, k), dtype=dtype, device=device)
      s = torch.empty((k,), dtype=dtype, device=device)
      vh = torch.empty((k, n), dtype=dtype, device=device)
      torch._linalg_svd(
          input_tensor.to(device),
          full_matrices=False,
          compute_uv=True,
          out=(u, s, vh),
      )
      return self._verify_svd_invariants(u, s, vh)

    self.assert_close_tpu_vs_cpu(run_private_svd_out)

    # 5. linalg_svdvals (public functional)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.linalg.svdvals(input_tensor.to(device))
    )

    # 6. linalg_svdvals.out (public out)
    def run_svdvals_out(device):
      m, n = shape
      k = min(m, n)
      s = torch.empty((k,), dtype=dtype, device=device)
      torch.linalg.svdvals(input_tensor.to(device), out=s)
      return s

    self.assert_close_tpu_vs_cpu(run_svdvals_out)

  def test_geqrf_out(self):
    input_tensor = torch.randn(8, 4)
    a = torch.empty(8, 4)
    tau = torch.empty(4)

    def run(device):
      return torch.geqrf(
          input_tensor.to(device), out=(a.to(device), tau.to(device))
      )

    self.assert_close_tpu_vs_cpu(run)

  def test_xlogy(self):
    x_data = [0.0, 0.0, 1.0, 0.0, 2.0, -1.0]
    y_data = [0.0, -1.0, float("nan"), float("nan"), 2.0, 3.0]

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.xlogy(
            torch.tensor(x_data, device=device).to(device),
            torch.tensor(y_data, device=device).to(device),
        ),
        rtol=5.7e-05,
        atol=6.3e-05,
    )

    # Test broadcasting
    def test_broadcasting_fn(device):
      x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
      y = torch.tensor([1.0, 2.0], device=device)
      return torch.xlogy(x, y)

    self.assert_close_tpu_vs_cpu(test_broadcasting_fn)

  def test_xlogy_scalar_and_cpu_scalar_tensors(self):
    """Tests xlogy scalar overloads and 0-D CPU scalar tensors."""
    # Scalar self
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.xlogy(2.0, torch.tensor([1.0, 2.0], device=device))
    )

    # Scalar other
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.xlogy(torch.tensor([1.0, 2.0], device=device), 2.0)
    )

    # In-place scalar other
    def run_inplace(device):
      t = torch.tensor([1.0, 2.0], device=device)
      t.xlogy_(2.0)
      return t

    self.assert_close_tpu_vs_cpu(run_inplace)

    # Out scalar self
    def run_out_self(device):
      out = torch.empty(2, device=device)
      torch.xlogy(2.0, torch.tensor([1.0, 2.0], device=device), out=out)
      return out

    self.assert_close_tpu_vs_cpu(run_out_self)

    # Out scalar other
    def run_out_other(device):
      out = torch.empty(2, device=device)
      torch.xlogy(torch.tensor([1.0, 2.0], device=device), 2.0, out=out)
      return out

    self.assert_close_tpu_vs_cpu(run_out_other)

    # 0-D CPU tensor self
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.xlogy(
            torch.tensor(2.0), torch.tensor([1.0, 2.0], device=device)
        )
    )

    # 0-D CPU tensor other
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.xlogy(
            torch.tensor([1.0, 2.0], device=device), torch.tensor(2.0)
        )
    )

    # Integer tensor with scalar promotes to default float
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.xlogy(2, torch.tensor([1, 2], device=device))
    )

  # ============================================================================
  # aten::_assert_async and aten::_assert_async.msg tests
  # ============================================================================

  def test_assert_async_eager_true(self):
    """Tests that _assert_async succeeds without error on truthy conditions.

    Checks:
      1. Calling `aten._assert_async` and `aten._assert_async.msg` with a
         boolean `True` tensor completes successfully and does not latch any
         sticky error.
      2. Non-boolean numeric scalar tensors that evaluate to true (e.g. integer
         1, float 1.0) also evaluate as truthy without raising errors.
      3. An explicit `torch.accelerator.synchronize()` flushes all background
         assertion workers and confirms that no deferred assertion failures
         occurred.
    """
    device = torch.device("tpu")

    # Verify that a boolean True condition does not latch any sticky error.
    true_cond = torch.tensor(True, device=device)
    torch.ops.aten._assert_async(true_cond)
    torch.ops.aten._assert_async.msg(true_cond, "should not fail")

    # Verify that numeric scalar truthy values (e.g. integer 1, float 1.0) also pass.
    int_cond = torch.tensor(1, device=device)
    torch.ops.aten._assert_async(int_cond)
    float_cond = torch.tensor(1.0, device=device)
    torch.ops.aten._assert_async.msg(float_cond, "float true condition")

    # Synchronize to drain the background thread pool and verify no errors were set.
    torch.accelerator.synchronize()

  def test_assert_async_eager_false_msg(self):
    """Tests that _assert_async.msg fails and preserves custom error message.

    Checks:
      1. Calling `aten._assert_async.msg` with a boolean `False` tensor returns
         immediately to host execution without blocking.
      2. The background thread detects the false condition after DtoH copy and
         latches a sticky error containing the specified custom message.
      3. A subsequent explicit `torch.accelerator.synchronize()` drains pending
         checks and raises a `RuntimeError` matching the user's custom
         assertion message.
    """
    import torch_tpu._internal.testing as tpu_testing

    device = torch.device("tpu")
    false_cond = torch.tensor(False, device=device)

    # Schedule the assertion failure on the background worker thread.
    torch.ops.aten._assert_async.msg(false_cond, "intended failure")

    try:
      # Synchronizing forces completion of all pending async assertion checks
      # and checks HasStickyError(), rethrowing the latched assertion failure.
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing assert_async failure behavior.
          RuntimeError, "intended failure"
      ):
        torch.accelerator.synchronize()
    finally:
      # Clear the latched error state so subsequent test cases are not poisoned.
      tpu_testing.clear_sticky_error()

  def test_assert_async_eager_false_no_msg(self):
    """Tests that _assert_async without a custom message uses the default message.

    Checks:
      1. Calling `aten._assert_async` (no message argument) with a `False`
         condition delegates to `_assert_async.msg` with the default string
         "assertion failed".
      2. Calling `torch.accelerator.synchronize()` raises a `RuntimeError`
         matching the default error message "assertion failed".
    """
    import torch_tpu._internal.testing as tpu_testing

    device = torch.device("tpu")
    false_cond = torch.tensor(False, device=device)

    # When no message string is provided, the kernel defaults to "assertion failed".
    torch.ops.aten._assert_async(false_cond)

    try:
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing assert_async failure behavior.
          RuntimeError, "assertion failed"
      ):
        torch.accelerator.synchronize()
    finally:
      # Reset sticky error state for subsequent tests.
      tpu_testing.clear_sticky_error()

  def test_assert_async_eager_op_dispatch_failure(self):
    """Tests that subsequent op dispatches detect a latched sticky error.

    Checks:
      1. When an asynchronous assertion fails in the background thread pool, it
         latches an error into the global sticky error state (`SetStickyError`).
      2. Subsequent PyTorch eager op dispatches (e.g. `torch.ops.aten.add`),
      which
         check `HasStickyError()` on their fast path, detect the latched error
         and raise `RuntimeError` without requiring an explicit synchronize
         call.
    """
    import time
    import torch_tpu._internal.testing as tpu_testing

    device = torch.device("tpu")
    false_cond = torch.tensor(False, device=device)

    # Trigger async assertion failure.
    torch.ops.aten._assert_async.msg(false_cond, "sticky error caught in op")

    try:
      # In eager mode, DispatchOp checks HasStickyError() on the critical path.
      # Because assertion evaluation runs asynchronously on a background thread,
      # repeatedly dispatch an op until the background thread latches the error.
      for _ in range(50):
        try:
          torch.ops.aten.add(false_cond, 1)
          time.sleep(0.1)
        except RuntimeError as e:
          if "sticky error caught in op" in str(e):
            break
      else:
        self.fail("Expected RuntimeError with sticky error on subsequent op")
    finally:
      # Reset sticky error state for subsequent tests.
      tpu_testing.clear_sticky_error()

  def test_assert_async_ambiguous_empty(self):
    """Tests that _assert_async rejects empty 0-element tensors.

    Checks:
      1. Calling `_assert_async` or `_assert_async.msg` with an empty tensor
         (`numel == 0`) immediately raises a `RuntimeError` indicating that the
         boolean value of an empty tensor is ambiguous.
      2. Validation occurs synchronously on the host during kernel invocation,
         matching standard PyTorch behavior before any device work is scheduled.
    """
    device = torch.device("tpu")
    # Empty tensor with 0 elements cannot be converted to a scalar boolean value.
    empty_cond = torch.empty(0, dtype=torch.bool, device=device)

    # Verify input validation throws synchronously on host for both variants.
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing input validation.
        RuntimeError, "is ambiguous"
    ):
      torch.ops.aten._assert_async(empty_cond)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing input validation.
        RuntimeError, "is ambiguous"
    ):
      torch.ops.aten._assert_async.msg(empty_cond, "empty condition")

  def test_assert_async_ambiguous_multiple(self):
    """Tests that _assert_async rejects multi-element tensors.

    Checks:
      1. Calling `_assert_async` or `_assert_async.msg` with a tensor containing
         more than one element (`numel > 1`) immediately raises a `RuntimeError`
         indicating that the boolean value of a multi-element tensor is
         ambiguous.
      2. Validation occurs synchronously on the host during kernel invocation,
         matching standard PyTorch behavior before any device work is scheduled.
    """
    device = torch.device("tpu")
    # Tensor with >1 elements cannot be converted to a scalar boolean value.
    multi_cond = torch.tensor([True, False], device=device)

    # Verify input validation throws synchronously on host for both variants.
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing input validation.
        RuntimeError, "is ambiguous"
    ):
      torch.ops.aten._assert_async(multi_cond)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing input validation.
        RuntimeError, "is ambiguous"
    ):
      torch.ops.aten._assert_async.msg(multi_cond, "multi condition")

  def test_assert_async_non_bool_false(self):
    """Tests that _assert_async fails on numeric zero tensors.

    Checks:
      1. Calling `_assert_async.msg` with a numeric non-boolean scalar whose
         value is 0 (e.g., integer 0) evaluates to boolean False on device.
      2. The background worker detects the false evaluation and latches an
         error, which is subsequently raised as `RuntimeError` upon
         `torch.accelerator.synchronize()`.
    """
    import torch_tpu._internal.testing as tpu_testing

    device = torch.device("tpu")
    # Integer scalar with value 0 evaluates to boolean False.
    zero_int = torch.tensor(0, device=device)
    torch.ops.aten._assert_async.msg(zero_int, "zero int failed")

    try:
      # Synchronization rethrows the assertion failure from the zero integer value.
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing assert_async failure behavior.
          RuntimeError, "zero int failed"
      ):
        torch.accelerator.synchronize()
    finally:
      # Reset sticky error state for subsequent tests.
      tpu_testing.clear_sticky_error()

  def test_assert_async_compile(self):
    """Tests _assert_async behavior during graph compilation and tracing.

    Checks:
      1. During FX graph compilation (`EagerMode::kInternalCompileFxGraph`),
         host asynchronous assertion scheduling is skipped because
         placeholder/symbolic tensors cannot be copied to the host CPU during
         tracing.
      2. The resulting MLIR graph compiles cleanly without inserting unsupported
         or dummy `@shape_assertion` custom calls.
    """
    device = torch.device("tpu")
    cond = torch.tensor(True, device=device)

    # In compile mode (FxGraph tracing), host async evaluation is skipped because
    # tensors are symbolic/placeholders and cannot be copied to CPU at trace time.
    with execution_mode.set_eager_mode(
        execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
    ):
      torch.ops.aten._assert_async(cond)
      dummy = cond.to(torch.float32) + 1.0

    # Ensure valid MLIR generation without dummy custom calls.
    mlir = tpu_torch_compile.build_mlir([dummy], [cond])
    mlir_text = tpu_torch_compile.serialize_mlir_text(mlir)
    self.assertNotIn("stablehlo.custom_call @shape_assertion", mlir_text)

  def test_assert_async_torch_public_api(self):
    """Tests public torch._assert_async API for success and failure cases.

    Checks:
      1. The higher-level public Python API `torch._assert_async(cond, [msg])`
         correctly routes to the TPU `aten::_assert_async` kernels.
      2. Truthy conditions pass cleanly when flushed via
         `torch.accelerator.synchronize()`.
      3. Falsy conditions latch a sticky error and raise `RuntimeError` matching
         the user message upon calling `torch.accelerator.synchronize()`.
    """
    import torch_tpu._internal.testing as tpu_testing

    device = torch.device("tpu")

    # Test that the public torch._assert_async API succeeds for True conditions.
    true_cond = torch.tensor(True, device=device)
    torch._assert_async(true_cond)
    torch._assert_async(true_cond, "should not fail")
    torch.accelerator.synchronize()

    # Test that public torch._assert_async API fails and reports error on synchronize().
    false_cond = torch.tensor(False, device=device)
    torch._assert_async(false_cond, "public api failure")
    try:
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing assert_async failure behavior.
          RuntimeError, "public api failure"
      ):
        torch.accelerator.synchronize()
    finally:
      # Reset sticky error state for subsequent tests.
      tpu_testing.clear_sticky_error()

  def test_assert_async_deferred_computation(self):
    """Tests _assert_async on lazily computed / deferred tensor expressions.

    Checks:
      1. When `_assert_async` is called on a condition produced by deferred ops
         (e.g., `(a + b == 3.0).squeeze()`), `MaterializeAndReturn` forces lazy
         graph execution and materializes the condition's device buffer on TPU.
      2. True expressions succeed without error upon
         `torch.accelerator.synchronize()`.
      3. False expressions trigger the background failure and raise
         `RuntimeError` with the custom message upon
         `torch.accelerator.synchronize()`.
    """
    import torch_tpu._internal.testing as tpu_testing

    device = torch.device("tpu")
    a = torch.tensor([1.0], device=device)
    b = torch.tensor([2.0], device=device)

    # Test deferred boolean expression that evaluates to True.
    # MaterializeAndReturn triggers lazy execution so the buffer can be copied to host.
    torch._assert_async((a + b == 3.0).squeeze())
    torch.accelerator.synchronize()

    # Test deferred boolean expression that evaluates to False.
    torch._assert_async((a + b == 4.0).squeeze(), "deferred mismatch")
    try:
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing assert_async failure behavior.
          RuntimeError, "deferred mismatch"
      ):
        torch.accelerator.synchronize()
    finally:
      # Reset sticky error state for subsequent tests.
      tpu_testing.clear_sticky_error()

  def test_assert_async_stream_synchronize(self):
    """Tests that stream-level synchronization catches pending assertion failures.

    Checks:
      1. Synchronizing a stream via
      `torch.accelerator.current_stream().synchronize()`
         invokes the TPU hook `synchronizeStream`.
      2. `synchronizeStream` calls `SyncAndCheckStickyError()`, ensuring pending
         assertion background checks are drained and any latched sticky error is
         re-thrown as a `RuntimeError`.
    """
    import torch_tpu._internal.testing as tpu_testing

    device = torch.device("tpu")
    false_cond = torch.tensor(False, device=device)

    # Schedule assertion failure.
    torch._assert_async(false_cond, "stream sync failure")
    try:
      # Stream-level synchronization (torch.accelerator.current_stream().synchronize())
      # invokes synchronizeStream, which flushes assertion checks and rethrows sticky error.
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing assert_async failure behavior.
          RuntimeError, "stream sync failure"
      ):
        torch.accelerator.current_stream().synchronize()
    finally:
      # Reset sticky error state for subsequent tests.
      tpu_testing.clear_sticky_error()

  def test_assert_async_fifo_ordering(self):
    """Tests deterministic FIFO ordering of sequential assertion failures.

    Checks:
      1. When multiple assertions are enqueued sequentially, the single
      background
         worker thread (`/*num_threads=*/1`) processes them in strict FIFO
         order.
      2. The error message from the *first* failing assertion is latched into
      the
         sticky error state, ensuring predictable error reporting instead of a
         race
         condition between multiple failures.
    """
    import torch_tpu._internal.testing as tpu_testing

    device = torch.device("tpu")

    # Allocate both condition tensors before enqueuing assertions to avoid
    # race conditions where the first async assertion sets sticky error before
    # the second tensor is created.
    cond1 = torch.tensor(False, device=device)
    cond2 = torch.tensor(False, device=device)
    try:
      torch._assert_async(cond1, "first error")
      torch._assert_async(cond2, "second error")
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing assert_async failure behavior.
          RuntimeError, "first error"
      ):
        torch.accelerator.synchronize()
    finally:
      # Reset sticky error state for subsequent tests.
      tpu_testing.clear_sticky_error()

  def test_matmul_fp8_bf16(self):
    """Tests torch.matmul on TPU with FP8 inputs and BF16 output."""
    device = torch.device("tpu")

    a_fp32 = torch.randn(16, 16, dtype=torch.float32, device=device)
    b_fp32 = torch.randn(16, 16, dtype=torch.float32, device=device)

    a_fp8 = a_fp32.to(torch.float8_e4m3fn)
    b_fp8 = b_fp32.to(torch.float8_e4m3fn)

    out_bf16 = torch.empty(16, 16, dtype=torch.bfloat16, device=device)

    torch.mm(a_fp8, b_fp8, out_dtype=torch.bfloat16, out=out_bf16)

    a_cpu_fp8 = a_fp8.cpu()
    b_cpu_fp8 = b_fp8.cpu()
    expected_cpu = torch.matmul(a_cpu_fp8.float(), b_cpu_fp8.float()).to(
        torch.bfloat16
    )

    self.assert_close(
        golden_result=expected_cpu,
        torch_tpu_result=out_bf16.cpu(),
        rtol=1e-2,
        atol=1e-2,
    )

  def test_baddbmm_zero_beta_ignores_nan_input(self):
    """torch.baddbmm() with zero beta should ignore NaN input."""
    nan_input = torch.full((2, 2, 4), float("nan"), dtype=torch.float32)
    batch1 = torch.ones(2, 2, 3, dtype=torch.float32)
    batch2 = torch.ones(2, 3, 4, dtype=torch.float32)
    beta = 0.0
    alpha = 1.0

    # Internal check.
    assert torch.all(torch.isnan(nan_input))

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.baddbmm(
            nan_input.to(device),
            batch1.to(device),
            batch2.to(device),
            beta=beta,
            alpha=alpha,
        )
    )

  def test_baddbmm_zero_beta_ignores_inf_input(self):
    """torch.baddbmm() with zero beta should ignore Inf input."""
    inf_input = torch.full((2, 2, 4), float("inf"), dtype=torch.float32)
    batch1 = torch.ones(2, 2, 3, dtype=torch.float32)
    batch2 = torch.ones(2, 3, 4, dtype=torch.float32)
    beta = 0.0
    alpha = 1.0

    # Internal check.
    assert torch.all(torch.isinf(inf_input))

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.baddbmm(
            inf_input.to(device),
            batch1.to(device),
            batch2.to(device),
            beta=beta,
            alpha=alpha,
        )
    )

  def test_baddbmm_zero_alpha_ignores_nan_mat(self):
    """torch.baddbmm() with zero alpha should ignore NaN mat input."""
    self_input = torch.ones(2, 2, 4, dtype=torch.float32)
    nan_batch1 = torch.full((2, 2, 3), float("nan"), dtype=torch.float32)
    batch2 = torch.ones(2, 3, 4, dtype=torch.float32)
    beta = 1.0
    alpha = 0.0

    # Internal check.
    assert torch.all(torch.isnan(nan_batch1))

    device = torch.device("tpu")
    tpu_res = torch.baddbmm(
        self_input.to(device),
        nan_batch1.to(device),
        batch2.to(device),
        beta=beta,
        alpha=alpha,
    )
    expected = self_input * beta
    self.assert_close(golden_result=expected, torch_tpu_result=tpu_res.cpu())

  def test_baddbmm_zero_alpha_ignores_inf_mat(self):
    """torch.baddbmm() with zero alpha should ignore Inf mat input."""
    self_input = torch.ones(2, 2, 4, dtype=torch.float32)
    inf_batch1 = torch.full((2, 2, 3), float("inf"), dtype=torch.float32)
    batch2 = torch.ones(2, 3, 4, dtype=torch.float32)
    beta = 1.0
    alpha = 0.0

    # Internal check.
    assert torch.all(torch.isinf(inf_batch1))

    device = torch.device("tpu")
    tpu_res = torch.baddbmm(
        self_input.to(device),
        inf_batch1.to(device),
        batch2.to(device),
        beta=beta,
        alpha=alpha,
    )
    expected = self_input * beta
    self.assert_close(golden_result=expected, torch_tpu_result=tpu_res.cpu())

  def test_elu_backward_is_result(self):
    """Tests torch.ops.aten.elu_backward with is_result=True and negative values."""
    grad_output = torch.tensor([-1.0, 2.0, -3.0], dtype=torch.float32)
    self_or_result = torch.tensor([-2.0, -1.5, 0.5], dtype=torch.float32)

    def compute(device):
      return torch.ops.aten.elu_backward(
          to(grad_output, device),
          alpha=1.0,
          scale=1.0,
          input_scale=1.0,
          is_result=True,
          self_or_result=to(self_or_result, device),
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_elu_small_negative_input_precision_float32(self):
    """Tests elu, selu and celu in float32 for negative inputs close to 0.

    exp(x) - 1 loses precision when x is close to 0, so the kernel uses
    expm1(x), as PyTorch does on CPU and CUDA.
    """
    x = -torch.logspace(-8.0, 0.0, 801, dtype=torch.float64)
    selu_alpha = 1.6732632423543772848170429916717
    selu_scale = 1.0507009873554804934193349852946
    cases = [
        (torch.nn.functional.elu, torch.expm1(x)),
        (torch.nn.functional.selu, selu_scale * selu_alpha * torch.expm1(x)),
        (
            functools.partial(torch.nn.functional.celu, alpha=2.0),
            2.0 * torch.expm1(x / 2.0),
        ),
    ]
    x_tpu = x.to(torch.float32).to("tpu")
    for op, want in cases:
      got = op(x_tpu)
      # TPU's float32 expm1 is within about 1.2e-4 of the reference here; the
      # cancellation in exp(x) - 1 gives errors of 100% and more.
      utils.assert_close(got.cpu().double(), want, rtol=1e-3, atol=0.0)

  def test_elu_input_scale_only_scales_negative_inputs(self):
    """Tests that elu applies input_scale to negative inputs only.

    celu(x, alpha) calls elu(x, alpha, scale=1, input_scale=1 / alpha), so
    celu(x) must equal x for positive x.
    """
    x = torch.tensor([-3.0, -0.5, -1e-3, 0.0, 1e-3, 0.5, 3.0])
    for alpha, scale, input_scale in [
        (1.0, 1.0, 0.5),
        (2.0, 1.5, 0.5),
        (1.0, 1.0, -2.0),
    ]:

      def compute(device, alpha=alpha, scale=scale, input_scale=input_scale):
        return torch.ops.aten.elu(
            to(x, device), alpha, scale=scale, input_scale=input_scale
        )

      # rtol covers TPU's float32 expm1 error (e.g. expm1(6) with
      # input_scale=-2); a wrongly scaled branch is off by 50% or more.
      self.assert_close_tpu_vs_cpu(compute, rtol=1e-3, atol=1e-6)

    for alpha in [2.0, 0.5, -1.0]:
      self.assert_close_tpu_vs_cpu(
          lambda device, alpha=alpha: torch.nn.functional.celu(
              to(x, device), alpha=alpha
          ),
          rtol=1e-3,
          atol=1e-6,
      )

  def test_ceil_integral(self):
    for dtype in [torch.int32, torch.uint8]:
      x = torch.tensor([-2, -1, 0, 1, 2], dtype=dtype)
      self.assert_close_tpu_vs_cpu(lambda device, x=x: torch.ceil(x.to(device)))
      self.assert_close_tpu_vs_cpu(
          lambda device, x=x: torch.ceil_(x.clone().to(device))
      )

  def test_floor_integral(self):
    for dtype in [torch.int32, torch.uint8]:
      x = torch.tensor([-2, -1, 0, 1, 2], dtype=dtype)
      self.assert_close_tpu_vs_cpu(
          lambda device, x=x: torch.floor(x.to(device))
      )
      self.assert_close_tpu_vs_cpu(
          lambda device, x=x: torch.floor_(x.clone().to(device))
      )

  def test_foreach_ceil_integral(self):
    for dtype in [torch.int32, torch.uint8]:
      x = [
          torch.tensor([-2, -1, 0, 1, 2], dtype=dtype),
          torch.tensor([10, -5, 3], dtype=dtype),
      ]
      self.assert_close_tpu_vs_cpu(
          lambda device, x=x: torch._foreach_ceil([t.to(device) for t in x])
      )
      self.assert_close_tpu_vs_cpu(
          lambda device, x=x: torch._foreach_ceil_(
              [t.clone().to(device) for t in x]
          )
      )

  def test_foreach_floor_integral(self):
    for dtype in [torch.int32, torch.uint8]:
      x = [
          torch.tensor([-2, -1, 0, 1, 2], dtype=dtype),
          torch.tensor([10, -5, 3], dtype=dtype),
      ]
      self.assert_close_tpu_vs_cpu(
          lambda device, x=x: torch._foreach_floor([t.to(device) for t in x])
      )
      self.assert_close_tpu_vs_cpu(
          lambda device, x=x: torch._foreach_floor_(
              [t.clone().to(device) for t in x]
          )
      )

  def test_sort_indices(self):
    """Tests that torch.sort returns correct indices (dtype and values)."""
    x = torch.tensor([5.0, 1.0, 3.0, 2.0, 4.0], dtype=torch.float32)

    def run(device):
      values, indices = torch.sort(x.to(device))
      return values, indices

    self.assert_close_tpu_vs_cpu(run)

  @parameterized.product(
      batch_first=[True, False],
      norm_first=[True, False],
      use_gelu=[True, False],
      dtype=[torch.float32, torch.bfloat16],
      mask_kind=[None, "2d_bool", "2d_float"],
  )
  def test_transformer_encoder_layer_fwd(
      self, batch_first, norm_first, use_gelu, dtype, mask_kind
  ):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    eps = 1e-5

    dim0, dim1 = (2, 8) if batch_first else (8, 2)
    batch_size, seq_len = dim0, dim1

    scale = 1.0 / math.sqrt(embed_dim)
    cpu_src = (
        torch.linspace(-1.0, 1.0, dim0 * dim1 * embed_dim, dtype=dtype).reshape(
            dim0, dim1, embed_dim
        )
        * 0.5
    )
    cpu_qkv_w = (
        torch.linspace(
            -1.0, 1.0, 3 * embed_dim * embed_dim, dtype=dtype
        ).reshape(3 * embed_dim, embed_dim)
        * scale
    )
    cpu_qkv_b = torch.linspace(-0.1, 0.1, 3 * embed_dim, dtype=dtype)
    cpu_proj_w = (
        torch.linspace(-1.0, 1.0, embed_dim * embed_dim, dtype=dtype).reshape(
            embed_dim, embed_dim
        )
        * scale
    )
    cpu_proj_b = torch.linspace(-0.1, 0.1, embed_dim, dtype=dtype)
    cpu_norm_w1 = torch.ones(embed_dim, dtype=dtype)
    cpu_norm_b1 = torch.zeros(embed_dim, dtype=dtype)
    cpu_norm_w2 = torch.ones(embed_dim, dtype=dtype)
    cpu_norm_b2 = torch.zeros(embed_dim, dtype=dtype)
    cpu_ffn_w1 = (
        torch.linspace(-1.0, 1.0, d_ff * embed_dim, dtype=dtype).reshape(
            d_ff, embed_dim
        )
        * scale
    )
    cpu_ffn_b1 = torch.linspace(-0.1, 0.1, d_ff, dtype=dtype)
    cpu_ffn_w2 = torch.linspace(
        -1.0, 1.0, embed_dim * d_ff, dtype=dtype
    ).reshape(embed_dim, d_ff) * (1.0 / math.sqrt(d_ff))
    cpu_ffn_b2 = torch.linspace(-0.1, 0.1, embed_dim, dtype=dtype)

    cpu_mask = None
    mask_type = None
    if mask_kind == "2d_bool":
      cpu_mask = ~torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
      cpu_mask = cpu_mask.view(1, 1, seq_len, seq_len).expand(
          batch_size, num_heads, seq_len, seq_len
      )
      mask_type = 2
    elif mask_kind == "2d_float":
      cpu_mask = torch.zeros(seq_len, seq_len, dtype=dtype)
      cpu_mask.masked_fill_(
          ~torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)), -10000.0
      )
      cpu_mask = cpu_mask.view(1, 1, seq_len, seq_len).expand(
          batch_size, num_heads, seq_len, seq_len
      )
      mask_type = 2

    def compute(device):
      src = cpu_src.to(device)
      qkv_w = cpu_qkv_w.to(device)
      qkv_b = cpu_qkv_b.to(device)
      proj_w = cpu_proj_w.to(device)
      proj_b = cpu_proj_b.to(device)
      norm_w1 = cpu_norm_w1.to(device)
      norm_b1 = cpu_norm_b1.to(device)
      norm_w2 = cpu_norm_w2.to(device)
      norm_b2 = cpu_norm_b2.to(device)
      ffn_w1 = cpu_ffn_w1.to(device)
      ffn_b1 = cpu_ffn_b1.to(device)
      ffn_w2 = cpu_ffn_w2.to(device)
      ffn_b2 = cpu_ffn_b2.to(device)
      mask = cpu_mask.to(device) if cpu_mask is not None else None

      return torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          use_gelu,
          norm_first,
          eps,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
          mask,
          mask_type,
      )

    rtol = 3e-2
    atol = 3e-2
    self.assert_close_tpu_vs_cpu(compute, rtol=rtol, atol=atol)

  def test_segment_reduce_modes(self):
    """Tests segment_reduce with different reduction modes."""
    data_cpu = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=self.golden_device
    )
    lengths_cpu = torch.tensor(
        [2, 3, 1], dtype=torch.long, device=self.golden_device
    )
    data_tpu = data_cpu.to("tpu")
    lengths_tpu = lengths_cpu.to("tpu")

    for reduce_mode in ("sum", "mean", "max", "min", "prod"):
      golden = torch.segment_reduce(
          data_cpu, reduce=reduce_mode, lengths=lengths_cpu
      )
      actual = torch.segment_reduce(
          data_tpu, reduce=reduce_mode, lengths=lengths_tpu
      )
      self.assert_close(golden_result=golden, torch_tpu_result=actual)

  def test_segment_reduce_lengths_vs_offsets(self):
    """Tests segment_reduce with lengths vs offsets."""
    data_cpu = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=self.golden_device
    )
    data_tpu = data_cpu.to("tpu")

    lengths_cpu = torch.tensor(
        [2, 3, 1], dtype=torch.long, device=self.golden_device
    )
    lengths_tpu = lengths_cpu.to("tpu")
    golden_lengths = torch.segment_reduce(
        data_cpu, reduce="sum", lengths=lengths_cpu
    )
    actual_lengths = torch.segment_reduce(
        data_tpu, reduce="sum", lengths=lengths_tpu
    )
    self.assert_close(
        golden_result=golden_lengths, torch_tpu_result=actual_lengths
    )

    offsets_cpu = torch.tensor(
        [0, 2, 5, 6], dtype=torch.long, device=self.golden_device
    )
    offsets_tpu = offsets_cpu.to("tpu")
    golden_offsets = torch.segment_reduce(
        data_cpu, reduce="sum", offsets=offsets_cpu
    )
    actual_offsets = torch.segment_reduce(
        data_tpu, reduce="sum", offsets=offsets_tpu
    )
    self.assert_close(
        golden_result=golden_offsets, torch_tpu_result=actual_offsets
    )

  def test_segment_reduce_axis(self):
    """Tests segment_reduce with axis parameter (axis=0 and axis=-1)."""
    data_cpu = torch.arange(
        24, dtype=torch.float32, device=self.golden_device
    ).reshape(6, 4)
    data_tpu = data_cpu.to("tpu")

    lengths0_cpu = torch.tensor(
        [2, 4], dtype=torch.long, device=self.golden_device
    )
    lengths0_tpu = lengths0_cpu.to("tpu")
    golden0 = torch.segment_reduce(
        data_cpu, reduce="sum", axis=0, lengths=lengths0_cpu
    )
    actual0 = torch.segment_reduce(
        data_tpu, reduce="sum", axis=0, lengths=lengths0_tpu
    )
    self.assert_close(golden_result=golden0, torch_tpu_result=actual0)

    data1_cpu = torch.arange(6, dtype=torch.float32, device=self.golden_device)
    data1_tpu = data1_cpu.to("tpu")
    lengths1_cpu = torch.tensor(
        [2, 4], dtype=torch.long, device=self.golden_device
    )
    lengths1_tpu = lengths1_cpu.to("tpu")
    golden1 = torch.segment_reduce(
        data1_cpu, reduce="sum", axis=0, lengths=lengths1_cpu
    )
    actual1 = torch.segment_reduce(
        data1_tpu, reduce="sum", axis=0, lengths=lengths1_tpu
    )
    self.assert_close(golden_result=golden1, torch_tpu_result=actual1)

  def test_segment_reduce_negative_axis(self):
    """Tests segment_reduce with negative axis (axis=-2)."""
    data_cpu = torch.arange(
        24, dtype=torch.float32, device=self.golden_device
    ).reshape(6, 4)
    lengths_cpu = torch.tensor(
        [2, 4], dtype=torch.long, device=self.golden_device
    )
    data_tpu = data_cpu.to("tpu")
    lengths_tpu = lengths_cpu.to("tpu")

    golden = torch.segment_reduce(
        data_cpu, reduce="sum", axis=-2, lengths=lengths_cpu
    )
    actual = torch.segment_reduce(
        data_tpu, reduce="sum", axis=-2, lengths=lengths_tpu
    )
    self.assert_close(golden_result=golden, torch_tpu_result=actual)

  def test_segment_reduce_initial(self):
    """Tests segment_reduce with initial scalar parameter across all modes."""
    data_cpu = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=self.golden_device
    )
    lengths_cpu = torch.tensor(
        [2, 3, 1], dtype=torch.long, device=self.golden_device
    )
    data_tpu = data_cpu.to("tpu")
    lengths_tpu = lengths_cpu.to("tpu")

    for reduce_mode in ("sum", "mean", "max", "min", "prod"):
      for init_val in (0.0, 10.0, -5.0):
        golden = torch.segment_reduce(
            data_cpu, reduce=reduce_mode, initial=init_val, lengths=lengths_cpu
        )
        actual = torch.segment_reduce(
            data_tpu, reduce=reduce_mode, initial=init_val, lengths=lengths_tpu
        )
        self.assert_close(golden_result=golden, torch_tpu_result=actual)

  def test_segment_reduce_empty_segments(self):
    """Tests segment_reduce with empty segments across all modes and initial values."""
    data_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.golden_device)
    lengths_cpu = torch.tensor(
        [2, 0, 2], dtype=torch.long, device=self.golden_device
    )
    data_tpu = data_cpu.to("tpu")
    lengths_tpu = lengths_cpu.to("tpu")

    for reduce_mode in ("sum", "mean", "max", "min", "prod"):
      for init_val in (None, 0.0, 10.0):
        golden = torch.segment_reduce(
            data_cpu, reduce=reduce_mode, initial=init_val, lengths=lengths_cpu
        )
        actual = torch.segment_reduce(
            data_tpu, reduce=reduce_mode, initial=init_val, lengths=lengths_tpu
        )
        self.assert_close(golden_result=golden, torch_tpu_result=actual)

  def test_segment_reduce_cache_key_lengths_then_offsets(self):
    """Tests that lengths vs offsets do not collide in compilation cache."""
    data_cpu = torch.tensor([10.0, 20.0, 30.0, 40.0], device=self.golden_device)
    data_tpu = data_cpu.to("tpu")

    # 1. First invocation using lengths: [2, 2] -> 2 segments of sizes 2 and 2
    lengths_cpu = torch.tensor(
        [2, 2], dtype=torch.long, device=self.golden_device
    )
    lengths_tpu = lengths_cpu.to("tpu")
    golden_lengths = torch.segment_reduce(
        data_cpu, reduce="sum", lengths=lengths_cpu
    )
    actual_lengths = torch.segment_reduce(
        data_tpu, reduce="sum", lengths=lengths_tpu
    )
    self.assert_close(
        golden_result=golden_lengths, torch_tpu_result=actual_lengths
    )

    # 2. Second invocation using offsets of SAME size (2 elements):
    # [0, 2] -> 1 segment of size 2
    offsets_cpu = torch.tensor(
        [0, 2], dtype=torch.long, device=self.golden_device
    )
    offsets_tpu = offsets_cpu.to("tpu")
    golden_offsets = torch.segment_reduce(
        data_cpu, reduce="sum", offsets=offsets_cpu
    )
    actual_offsets = torch.segment_reduce(
        data_tpu, reduce="sum", offsets=offsets_tpu
    )
    self.assert_close(
        golden_result=golden_offsets, torch_tpu_result=actual_offsets
    )

  def test_segment_reduce_float_dtypes(self):
    """Tests segment_reduce with floating-point data types (float32, bfloat16)."""
    for dtype in (torch.float32, torch.bfloat16):
      data_cpu = torch.tensor(
          [1.0, 5.0, 2.0, 8.0, 3.0, 9.0], dtype=dtype, device=self.golden_device
      )
      lengths_cpu = torch.tensor(
          [2, 4], dtype=torch.long, device=self.golden_device
      )
      data_tpu = data_cpu.to("tpu")
      lengths_tpu = lengths_cpu.to("tpu")

      for reduce_mode in ("sum", "max", "min", "prod", "mean"):
        golden = torch.segment_reduce(
            data_cpu, reduce=reduce_mode, lengths=lengths_cpu
        )
        actual = torch.segment_reduce(
            data_tpu, reduce=reduce_mode, lengths=lengths_tpu
        )
        self.assert_close(golden_result=golden, torch_tpu_result=actual)

  @parameterized.parameters(
      itertools.product(
          ("sum", "mean", "max", "min", "prod"),
          ("lengths", "offsets"),
      )
  )
  def test_segment_reduce_multidimensional_2d(self, reduce_mode, mode):
    """Tests 2D lengths/offsets on 2D data (5, 5), axis=1."""
    data_cpu = torch.randn(5, 5, dtype=torch.float32, device=self.golden_device)
    lengths_cpu = torch.tensor(
        [
            [1, 2, 0, 2],
            [2, 1, 1, 1],
            [0, 3, 2, 0],
            [1, 1, 1, 2],
            [5, 0, 0, 0],
        ],
        dtype=torch.long,
        device=self.golden_device,
    )
    data_tpu = data_cpu.to("tpu")
    lengths_tpu = lengths_cpu.to("tpu")

    if mode == "lengths":
      golden = torch.segment_reduce(
          data_cpu, reduce=reduce_mode, lengths=lengths_cpu, axis=1
      )
      actual = torch.segment_reduce(
          data_tpu, reduce=reduce_mode, lengths=lengths_tpu, axis=1
      )
    else:
      offsets_cpu = torch.zeros(
          5, 5, dtype=torch.long, device=self.golden_device
      )
      offsets_cpu[:, 1:] = torch.cumsum(lengths_cpu, dim=-1)
      offsets_tpu = offsets_cpu.to("tpu")
      golden = torch.segment_reduce(
          data_cpu, reduce=reduce_mode, offsets=offsets_cpu, axis=1
      )
      actual = torch.segment_reduce(
          data_tpu, reduce=reduce_mode, offsets=offsets_tpu, axis=1
      )

    self.assert_close(golden_result=golden, torch_tpu_result=actual)

  @parameterized.parameters(
      itertools.product(
          ("sum", "mean", "max", "min", "prod"),
          ("lengths", "offsets"),
      )
  )
  def test_segment_reduce_multidimensional_3d(self, reduce_mode, mode):
    """Tests 2D lengths/offsets on 3D data (2, 5, 4), axis=1."""
    data_cpu = torch.randn(
        2, 5, 4, dtype=torch.float32, device=self.golden_device
    )
    lengths_cpu = torch.tensor(
        [[2, 1, 2], [1, 3, 1]],
        dtype=torch.long,
        device=self.golden_device,
    )
    data_tpu = data_cpu.to("tpu")
    lengths_tpu = lengths_cpu.to("tpu")

    if mode == "lengths":
      golden = torch.segment_reduce(
          data_cpu, reduce=reduce_mode, lengths=lengths_cpu, axis=1
      )
      actual = torch.segment_reduce(
          data_tpu, reduce=reduce_mode, lengths=lengths_tpu, axis=1
      )
    else:
      offsets_cpu = torch.zeros(
          2, 4, dtype=torch.long, device=self.golden_device
      )
      offsets_cpu[:, 1:] = torch.cumsum(lengths_cpu, dim=-1)
      offsets_tpu = offsets_cpu.to("tpu")
      golden = torch.segment_reduce(
          data_cpu, reduce=reduce_mode, offsets=offsets_cpu, axis=1
      )
      actual = torch.segment_reduce(
          data_tpu, reduce=reduce_mode, offsets=offsets_tpu, axis=1
      )

    self.assert_close(golden_result=golden, torch_tpu_result=actual)

  @parameterized.parameters(
      itertools.product(
          ("sum", "mean", "max", "min", "prod"),
          ("lengths", "offsets"),
      )
  )
  def test_segment_reduce_multidimensional_4d(self, reduce_mode, mode):
    """Tests 3D lengths/offsets on 4D data (2, 3, 5, 4), axis=2."""
    data_cpu = torch.randn(
        2, 3, 5, 4, dtype=torch.float32, device=self.golden_device
    )
    lengths_cpu = torch.tensor(
        [[[2, 3], [1, 4], [3, 2]], [[4, 1], [2, 3], [0, 5]]],
        dtype=torch.long,
        device=self.golden_device,
    )
    data_tpu = data_cpu.to("tpu")
    lengths_tpu = lengths_cpu.to("tpu")

    if mode == "lengths":
      golden = torch.segment_reduce(
          data_cpu, reduce=reduce_mode, lengths=lengths_cpu, axis=2
      )
      actual = torch.segment_reduce(
          data_tpu, reduce=reduce_mode, lengths=lengths_tpu, axis=2
      )
    else:
      offsets_cpu = torch.zeros(
          2, 3, 3, dtype=torch.long, device=self.golden_device
      )
      offsets_cpu[..., 1:] = torch.cumsum(lengths_cpu, dim=-1)
      offsets_tpu = offsets_cpu.to("tpu")
      golden = torch.segment_reduce(
          data_cpu, reduce=reduce_mode, offsets=offsets_cpu, axis=2
      )
      actual = torch.segment_reduce(
          data_tpu, reduce=reduce_mode, offsets=offsets_tpu, axis=2
      )

    self.assert_close(golden_result=golden, torch_tpu_result=actual)

  def test_data_ptr_stable_after_inplace(self):
    x = torch.zeros(2, 3, 4, device="tpu")
    # torch.Tensor.data_ptr returns the C++ memory address of the tensor data,
    # expressed as Python int:
    # https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.data_ptr.html
    # For TorchTPU this is the heap address of a `new DeviceBufferRef`.
    x_addr = x.data_ptr()

    x.add_(1)

    # Verify that the data pointer address is unchanged.
    self.assertEqual(x_addr, x.data_ptr())

  @parameterized.product(
      shape=[(10,), (4, 5), (2, 3, 4)],
      return_inverse=[False, True],
      return_counts=[False, True],
      dim=[None, 0, -1],
      dtype=[
          torch.float32,
          torch.bfloat16,
          torch.float16,
          torch.float64,
          torch.int64,
          torch.int32,
          torch.int16,
          torch.int8,
          torch.uint8,
          torch.bool,
      ],
  )
  def test_unique_consecutive(
      self, shape, return_inverse, return_counts, dim, dtype
  ):
    if dim is not None and len(shape) <= 1 and dim != 0 and dim != -1:
      return

    numel = math.prod(shape)
    raw = (torch.arange(numel) // 2 % 5).reshape(shape)
    if dtype == torch.bool:
      input_value = (raw % 2).to(torch.bool)
    else:
      input_value = raw.to(dtype)

    def compute(device):
      return torch.unique_consecutive(
          input_value.to(device),
          return_inverse=return_inverse,
          return_counts=return_counts,
          dim=dim,
      )

    self.assert_close_tpu_vs_cpu(compute)

  @parameterized.product(
      shape=[(4, 5), (2, 3, 4)],
      return_inverse=[False, True],
      return_counts=[False, True],
      dim=[0, 1, -1],
      dtype=[
          torch.float32,
          torch.bfloat16,
          torch.float16,
          torch.float64,
          torch.int64,
          torch.int32,
          torch.int16,
          torch.int8,
          torch.uint8,
          torch.bool,
      ],
  )
  def test_unique_dim_consecutive(
      self, shape, return_inverse, return_counts, dim, dtype
  ):
    numel = math.prod(shape)
    raw = (torch.arange(numel) // 2 % 5).reshape(shape)
    if dtype == torch.bool:
      input_value = (raw % 2).to(torch.bool)
    else:
      input_value = raw.to(dtype)

    def compute(device):
      out = torch.ops.aten.unique_dim_consecutive(
          input_value.to(device),
          dim=dim,
          return_inverse=return_inverse,
          return_counts=return_counts,
      )
      result = [out[0]]
      if return_inverse:
        result.append(out[1])
      if return_counts:
        result.append(out[2])
      return tuple(result) if len(result) > 1 else result[0]

    self.assert_close_tpu_vs_cpu(compute)

    # Verify TPU returns 0-element empty tensors for disabled flags (matching
    # CUDA)
    tpu_res = torch.ops.aten.unique_dim_consecutive(
        input_value.to("tpu"),
        dim=dim,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )
    if not return_inverse:
      self.assertEqual(tpu_res[1].cpu().numel(), 0)
    if not return_counts:
      self.assertEqual(tpu_res[2].cpu().numel(), 0)

  @parameterized.product(
      return_inverse=[False, True],
      return_counts=[False, True],
      dtype=[torch.float32, torch.int64],
  )
  def test_unique_consecutive_scalar(
      self, return_inverse, return_counts, dtype
  ):
    input_value = torch.tensor(42, dtype=dtype)

    def compute(device):
      return torch.unique_consecutive(
          input_value.to(device),
          return_inverse=return_inverse,
          return_counts=return_counts,
      )

    self.assert_close_tpu_vs_cpu(compute)

  @parameterized.product(
      shape=[(0,), (0, 5), (4, 0, 3)],
      return_inverse=[False, True],
      return_counts=[False, True],
      dim=[None, 0, 1],
  )
  def test_unique_consecutive_empty(
      self, shape, return_inverse, return_counts, dim
  ):
    if dim is not None:
      if dim >= len(shape):
        return
      # Only test valid empty dim cases matching PyTorch rules (single zero dim
      # selected).
      num_zero_dims = sum(1 for s in shape if s == 0)
      if shape[dim] == 0 and num_zero_dims > 1:
        return
      if shape[dim] != 0 and num_zero_dims > 0:
        return

    input_value = torch.empty(shape, dtype=torch.float32)

    def compute(device):
      return torch.unique_consecutive(
          input_value.to(device),
          return_inverse=return_inverse,
          return_counts=return_counts,
          dim=dim,
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_unique_consecutive_nan(self):
    input_value = torch.tensor(
        [float("nan"), float("nan"), 1.0, 1.0, float("nan"), 2.0, 2.0]
    )

    def compute(device):
      return torch.unique_consecutive(
          input_value.to(device),
          return_inverse=True,
          return_counts=True,
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_unique_consecutive_infs(self):
    input_value = torch.tensor([
        float("inf"),
        float("inf"),
        float("-inf"),
        float("-inf"),
        1.0,
        1.0,
        float("inf"),
    ])

    def compute(device):
      return torch.unique_consecutive(
          input_value.to(device),
          return_inverse=True,
          return_counts=True,
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_unique_consecutive_distributions(self):
    for dtype in [torch.float32, torch.int64]:
      for distribution in [
          "all_same",
          "all_different",
          "runs",
      ]:
        if distribution == "all_same":
          input_value = torch.full((100,), 42, dtype=dtype)
        elif distribution == "all_different":
          input_value = torch.arange(100, dtype=dtype)
        elif distribution == "runs":
          input_value = torch.tensor(
              [1, 1, 1, 2, 2, 3, 3, 3, 3, 1, 1, 4, 4, 4, 4, 4], dtype=dtype
          )
        else:
          raise ValueError(f"Unknown distribution: {distribution}")

        def compute(device, input_value=input_value):
          return torch.unique_consecutive(
              input_value.to(device),
              return_inverse=True,
              return_counts=True,
          )

        self.assert_close_tpu_vs_cpu(compute)

  def test_unique_consecutive_non_contiguous(self):
    input_value = (torch.arange(25, dtype=torch.float32).reshape(5, 5) // 2)[
        ::2, ::2
    ]

    def compute(device):
      return torch.unique_consecutive(
          input_value.to(device),
          return_inverse=True,
          return_counts=True,
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_negative_indices_cpu_tensor(self):
    def test_fn(device):
      x = torch.arange(20, dtype=torch.float32, device=device).reshape(4, 5)
      cpu_index = torch.tensor([-1, 2], dtype=torch.int64, device="cpu")
      return x[cpu_index]

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_boolean_mask_cpu_tensor_index(self):
    # 1D CPU boolean mask with repeat_interleave + ~mask (LightGlue pattern).
    x_3d = torch.randn(4, 6, 8, dtype=torch.float32)
    prune_mask_cpu = torch.tensor([True, False], dtype=torch.bool, device="cpu")
    mask_1d_cpu = ~prune_mask_cpu.repeat_interleave(2)
    self.assert_close_tpu_vs_cpu(lambda device: x_3d.to(device)[mask_1d_cpu])

    # 2D CPU boolean mask with leading and trailing slices, plus non-contiguous.
    x_4d = torch.randn(2, 3, 4, 5, dtype=torch.bfloat16)
    mask_2d_cpu = torch.tensor(
        [
            [True, False, True],
            [False, True, False],
            [True, True, False],
            [False, False, True],
        ],
        dtype=torch.bool,
        device="cpu",
    ).t()
    self.assert_close_tpu_vs_cpu(
        lambda device: x_4d.to(device)[:, mask_2d_cpu, :]
    )

    # Zero-sized CPU boolean mask.
    x_empty = torch.randn(2, 0, 4, dtype=torch.float32)
    mask_empty_cpu = torch.zeros(0, 4, dtype=torch.bool, device="cpu")
    self.assert_close_tpu_vs_cpu(
        lambda device: x_empty.to(device)[:, mask_empty_cpu]
    )

    # 0D True and False CPU boolean masks.
    x_2d = torch.randn(2, 3, dtype=torch.float32)
    mask_true_0d = torch.tensor(True, dtype=torch.bool, device="cpu")
    mask_false_0d = torch.tensor(False, dtype=torch.bool, device="cpu")
    self.assert_close_tpu_vs_cpu(
        lambda device: x_2d.to(device)[:, mask_true_0d, :]
    )
    self.assert_close_tpu_vs_cpu(lambda device: x_2d.to(device)[mask_false_0d])

  def test_boolean_mask_cpu_tensor_index_put(self):
    mask_1d_cpu = torch.tensor(
        [False, True, False, True], dtype=torch.bool, device="cpu"
    )

    # Select fast path (broadcastable values, overwrite and accumulate).
    def select_overwrite(device):
      t = torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2)
      t[mask_1d_cpu] = torch.tensor(9.0, dtype=torch.float32, device=device)
      return t

    self.assert_close_tpu_vs_cpu(select_overwrite)

    def select_accumulate(device):
      t = torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2)
      t[mask_1d_cpu] += torch.tensor(
          [10.0, 20.0], dtype=torch.float32, device=device
      )
      return t

    self.assert_close_tpu_vs_cpu(select_accumulate)

    # 0D CPU boolean masks on select fast path.
    mask_true_0d = torch.tensor(True, dtype=torch.bool, device="cpu")
    mask_false_0d = torch.tensor(False, dtype=torch.bool, device="cpu")

    def select_0d(device):
      t1 = torch.ones(2, 3, dtype=torch.float32, device=device)
      t1[:, mask_true_0d] = torch.tensor(
          5.0, dtype=torch.float32, device=device
      )
      t2 = torch.ones(2, 3, dtype=torch.float32, device=device)
      t2[mask_false_0d] = torch.tensor(7.0, dtype=torch.float32, device=device)
      return t1, t2

    self.assert_close_tpu_vs_cpu(select_0d)

    # Scatter path (non-broadcastable values varying across masked dimension).
    def scatter_overwrite_and_accumulate(device):
      t = torch.zeros(4, 2, dtype=torch.float32, device=device)
      vals = torch.tensor(
          [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device
      )
      t[mask_1d_cpu] = vals
      t[mask_1d_cpu] += vals
      return t

    self.assert_close_tpu_vs_cpu(scatter_overwrite_and_accumulate)

    # 2D CPU boolean mask with leading slice on scatter path, plus all-False mask.
    mask_2d_cpu = torch.tensor(
        [[True, False, True], [False, True, False]],
        dtype=torch.bool,
        device="cpu",
    )
    mask_all_false_cpu = torch.zeros(3, dtype=torch.bool, device="cpu")

    def scatter_2d_and_empty(device):
      t = torch.zeros(2, 2, 3, dtype=torch.float32, device=device)
      vals = torch.tensor(
          [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
          dtype=torch.float32,
          device=device,
      )
      t[:, mask_2d_cpu] = vals
      t_empty = torch.tensor(
          [1.0, 2.0, 3.0], dtype=torch.float32, device=device
      )
      t_empty[mask_all_false_cpu] = torch.empty(
          0, dtype=torch.float32, device=device
      )
      return t, t_empty

    self.assert_close_tpu_vs_cpu(scatter_2d_and_empty)

  def test_masked_fill_inplace_tensor_value_different_dtype(self):
    input_value = torch.randn(2, 3, dtype=torch.float32)
    mask_value = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool
    )
    tensor_value = torch.tensor(5, dtype=torch.int32)

    def compute(device):
      inp = to(input_value, device).clone()
      mask = to(mask_value, device)
      val = to(tensor_value, device)
      inp.masked_fill_(mask, val)
      return inp

    self.assert_close_tpu_vs_cpu(compute)

  def test_masked_fill_inplace_cpu_tensor_value(self):
    input_value = torch.randn(2, 3, dtype=torch.float32)
    mask_value = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool
    )

    def compute(device):
      inp = to(input_value, device).clone()
      mask = to(mask_value, device)
      val = torch.tensor(torch.finfo(torch.float32).min, device="cpu")
      inp.masked_fill_(mask, val)
      return inp

    self.assert_close_tpu_vs_cpu(compute)

  def test_masked_fill_cpu_tensor_value(self):
    input_value = torch.randn(2, 3, dtype=torch.float32)
    mask_value = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool
    )

    def compute(device):
      inp = to(input_value, device)
      mask = to(mask_value, device)
      val = torch.tensor(torch.finfo(torch.float32).min, device="cpu")
      return torch.masked_fill(inp, mask, val)

    self.assert_close_tpu_vs_cpu(compute)

  def test_masked_fill_inplace_cpu_tensor_value_different_dtype(self):
    input_value = torch.randn(2, 3, dtype=torch.float32)
    mask_value = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool
    )

    def compute(device):
      inp = to(input_value, device).clone()
      mask = to(mask_value, device)
      val = torch.tensor(5, dtype=torch.int32, device="cpu")
      inp.masked_fill_(mask, val)
      return inp

    self.assert_close_tpu_vs_cpu(compute)

  def test_masked_fill_scalar_and_tensor_values(self):
    input_f32 = torch.randn(2, 3, dtype=torch.float32)
    input_bf16 = torch.randn(2, 3, dtype=torch.bfloat16)
    mask_value = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool
    )
    tpu_tensor_val = torch.tensor(5, dtype=torch.int32)

    def compute(device):
      inp_f32 = to(input_f32, device).clone()
      inp_bf16 = to(input_bf16, device).clone()
      mask = to(mask_value, device)
      val_tpu = to(tpu_tensor_val, device)
      val_cpu = torch.tensor(-7, dtype=torch.int32, device="cpu")

      out_neg_inf = torch.masked_fill(inp_f32, mask, -float("inf"))
      out_bf16_min = inp_bf16.masked_fill(mask, torch.finfo(torch.bfloat16).min)
      out_zero = inp_f32.masked_fill(mask, 0.0)
      out_tpu_tensor = torch.masked_fill(inp_f32, mask, val_tpu)
      out_cpu_tensor = inp_f32.masked_fill(mask, val_cpu)
      return (
          out_neg_inf,
          out_bf16_min,
          out_zero,
          out_tpu_tensor,
          out_cpu_tensor,
          inp_f32,
          inp_bf16,
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_masked_fill_out_of_place_mutual_broadcast(self):
    input_1d = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    input_2d = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32)
    mask_2d = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool
    )
    tensor_val = torch.tensor(-9.5, dtype=torch.float32)

    def compute(device):
      inp1 = to(input_1d, device).clone()
      inp2 = to(input_2d, device).clone()
      mask = to(mask_2d, device)
      val = to(tensor_val, device)

      out_1d_scalar = torch.masked_fill(inp1, mask, -1.0)
      out_1d_tensor = torch.masked_fill(inp1, mask, val)
      out_2d_scalar = inp2.masked_fill(mask, -2.0)
      out_2d_tensor = inp2.masked_fill(mask, val)
      return (
          out_1d_scalar,
          out_1d_tensor,
          out_2d_scalar,
          out_2d_tensor,
          inp1,
          inp2,
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_masked_fill_autograd_backward(self):
    input_value = torch.randn(2, 3, dtype=torch.float32)
    mask_value = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool
    )
    grad_output_value = torch.randn(2, 3, dtype=torch.float32)

    def compute(device):
      inp_scalar = to(input_value, device).clone().requires_grad_(True)
      mask = to(mask_value, device)
      grad_out = to(grad_output_value, device)

      out_scalar = inp_scalar.masked_fill(mask, -10.0)
      out_scalar.backward(grad_out)

      inp_tensor = to(input_value, device).clone().requires_grad_(True)
      val_tensor = torch.tensor(
          3.5, dtype=torch.float32, device=device, requires_grad=True
      )
      out_tensor = torch.masked_fill(inp_tensor, mask, val_tensor)
      out_tensor.backward(grad_out)

      return (
          out_scalar,
          inp_scalar.grad,
          out_tensor,
          inp_tensor.grad,
          val_tensor.grad,
      )

    self.assert_close_tpu_vs_cpu(compute)

  @parameterized.named_parameters(
      ("crop_left_and_pad_right", (-2, 3), False),
      ("pad_left_and_crop_right", (3, -2), False),
      ("crop_entire_input_and_pad_right", (-5, 4), False),
      ("pad_left_and_crop_entire_input", (4, -5), False),
      ("crop_left_exceeding_input", (-8, 10), True),
      ("crop_right_exceeding_input", (10, -8), True),
  )
  def test_constant_pad_nd_negative_padding_overflow(self, pad, allow_failure):
    x = torch.arange(1, 6, dtype=torch.float32)

    def compute(device):
      return torch.nn.functional.pad(to(x, device), pad, "constant", 0.0)

    self.assert_close_tpu_vs_cpu(compute, allow_failure=allow_failure)

  # ============================================================================
  # aten::silu_backward and aten::silu_backward.grad_input tests
  # ============================================================================

  @parameterized.product(
      dtype=[torch.float32, torch.float64, torch.bfloat16, torch.float16],
      shapes=[
          ((2, 3, 4), (2, 3, 4)),  # same shape
          ((2, 1, 4), (2, 3, 4)),  # grad_output broadcasts
          ((2, 3, 4), (1, 3, 1)),  # self broadcasts
          ((), ()),  # 0-d tensors
          ((), (2, 3)),  # 0-d grad_output broadcasts
          ((0, 4), (0, 4)),  # zero-sized
      ],
  )
  def test_silu_backward(self, dtype, shapes):
    grad_shape, self_shape = shapes
    grad_out = torch.randn(grad_shape, dtype=dtype)
    x = torch.randn(self_shape, dtype=dtype)

    def compute(device):
      g = to(grad_out, device)
      inp = to(x, device)
      res = torch.ops.aten.silu_backward(g, inp)
      out = torch.empty_like(res)
      torch.ops.aten.silu_backward.grad_input(g, inp, grad_input=out)
      return res, out

    self.assert_close_tpu_vs_cpu(compute)

  @parameterized.parameters(torch.float32, torch.bfloat16)
  def test_silu_backward_extreme_values(self, dtype):
    """Checks saturation of sigmoid and IEEE special value propagation."""
    x = torch.tensor(
        [
            -float("inf"),
            -1e4,
            -100.0,
            -20.0,
            -1e-3,
            -0.0,
            0.0,
            1e-3,
            20.0,
            100.0,
            1e4,
            float("inf"),
            float("nan"),
        ],
        dtype=dtype,
    )
    grad_out = torch.ones_like(x)

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ops.aten.silu_backward(
            to(grad_out, device), to(x, device)
        )
    )

  @parameterized.named_parameters(
      ("float32_grad_bfloat16_self", torch.float32, torch.bfloat16),
      ("bfloat16_grad_float32_self", torch.bfloat16, torch.float32),
      ("float16_grad_bfloat16_self", torch.float16, torch.bfloat16),
      ("int64_grad_float32_self", torch.int64, torch.float32),
      ("float32_grad_int32_self", torch.float32, torch.int32),
  )
  def test_silu_backward_mixed_dtypes(self, grad_dtype, self_dtype):
    """Inputs are promoted to a common dtype like PyTorch's TensorIterator."""
    grad_out = torch.randn(2, 3).mul(4).to(grad_dtype)
    x = torch.randn(2, 3).mul(4).to(self_dtype)

    def compute(device):
      g = to(grad_out, device)
      inp = to(x, device)
      res = torch.ops.aten.silu_backward(g, inp)
      out = torch.empty_like(res)
      torch.ops.aten.silu_backward.grad_input(g, inp, grad_input=out)
      return res, out

    self.assert_close_tpu_vs_cpu(compute, rtol=1e-4, atol=1e-4)

  def test_silu_backward_non_contiguous(self):
    grad_out = torch.randn(4, 6)
    x = torch.randn(6, 4)

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ops.aten.silu_backward(
            to(grad_out, device)[:, ::2], to(x, device).t()[:, ::2]
        )
    )

  def test_silu_backward_cpu_scalar(self):
    """A 0-d CPU tensor is accepted for either operand."""
    x = torch.randn(2, 3)

    def compute(device):
      return (
          torch.ops.aten.silu_backward(torch.tensor(2.0), to(x, device)),
          torch.ops.aten.silu_backward(to(x, device), torch.tensor(0.5)),
      )

    self.assert_close_tpu_vs_cpu(compute)

  @parameterized.named_parameters(
      ("empty_out_is_resized", (0,), torch.float32, False),
      ("wrong_shape_out_is_resized", (5,), torch.float32, False),
      ("downcast_out_dtype", (2, 3), torch.bfloat16, False),
      ("upcast_out_dtype", (2, 3), torch.float64, False),
      ("integral_out_dtype", (2, 3), torch.int32, True),
  )
  def test_silu_backward_grad_input_out(
      self, out_shape, out_dtype, allow_failure
  ):
    grad_out = torch.randn(2, 3)
    x = torch.randn(2, 3)

    def compute(device):
      out = torch.empty(out_shape, dtype=out_dtype, device=device)
      torch.ops.aten.silu_backward.grad_input(
          to(grad_out, device), to(x, device), grad_input=out
      )
      return out

    self.assert_close_tpu_vs_cpu(
        compute, allow_failure=allow_failure, rtol=1e-4, atol=1e-4
    )

  @parameterized.parameters(
      (torch.int32, torch.int32),
      (torch.int64, torch.int64),
      (torch.bool, torch.bool),
  )
  def test_silu_backward_unsupported_dtypes(self, grad_dtype, self_dtype):
    grad_out = torch.ones(2, 3, dtype=grad_dtype)
    x = torch.ones(2, 3, dtype=self_dtype)

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.ops.aten.silu_backward(
            to(grad_out, device), to(x, device)
        ),
        allow_failure=True,
    )

  @parameterized.parameters(torch.float32, torch.bfloat16)
  def test_silu_autograd_backward(self, dtype):
    """End-to-end `F.silu` backward, which dispatches to `silu_backward`."""
    x = torch.randn(3, 5, dtype=dtype)
    grad_out = torch.randn(3, 5, dtype=dtype)

    def compute(device):
      # Detach to get a fresh leaf per device: `to(x, "cpu")` returns `x`
      # itself, so otherwise the TPU run would accumulate into the CPU `.grad`.
      inp = to(x, device).detach().requires_grad_()
      torch.nn.functional.silu(inp).backward(to(grad_out, device))
      return inp.grad

    self.assert_close_tpu_vs_cpu(compute)

  def test_silu_backward_differentiable(self):
    """Differentiating through `silu_backward` uses the composite fallback.

    PyTorch's CPU backend has no derivative for `aten::silu_backward`, so the
    reference evaluates the same closed-form expression on CPU.
    """
    x = torch.randn(3, 5, dtype=torch.float32)
    grad_out = torch.randn(3, 5, dtype=torch.float32)

    def compute(device):
      inp = to(x, device).detach().requires_grad_()
      g = to(grad_out, device).detach().requires_grad_()
      if device == "cpu":
        sig = torch.sigmoid(inp)
        res = g * sig * (1 + inp * (1 - sig))
      else:
        res = torch.ops.aten.silu_backward(g, inp)
      res.sum().backward()
      return res.detach(), inp.grad, g.grad

    self.assert_close_tpu_vs_cpu(compute)

  def test_silu_double_backward(self):
    """Second-order gradients of `F.silu` through autograd."""
    x = torch.randn(3, 5, dtype=torch.float32)
    grad_out = torch.randn(3, 5, dtype=torch.float32)

    def compute(device):
      inp = to(x, device).detach().requires_grad_()
      g = to(grad_out, device).detach().requires_grad_()
      (dx,) = torch.autograd.grad(
          torch.nn.functional.silu(inp), inp, g, create_graph=True
      )
      ddx, dg = torch.autograd.grad(dx.sum(), (inp, g))
      return dx.detach(), ddx, dg

    self.assert_close_tpu_vs_cpu(compute)

  def test_silu_backward_grad_input_inplace(self):
    """Tests in-place `grad_input` aliasing (matching shape and invalid broadcast)."""
    grad_out = torch.randn(2, 3, dtype=torch.float32)
    x = torch.randn(2, 3, dtype=torch.float32)

    def compute_valid(device):
      g1 = to(grad_out, device).clone()
      x1 = to(x, device).clone()
      torch.ops.aten.silu_backward.grad_input(g1, x1, grad_input=g1)
      g2 = to(grad_out, device).clone()
      x2 = to(x, device).clone()
      torch.ops.aten.silu_backward.grad_input(g2, x2, grad_input=x2)
      return g1, x2

    self.assert_close_tpu_vs_cpu(compute_valid)

    small_grad = torch.randn(1, 3, dtype=torch.float32)

    def compute_invalid_broadcast(device):
      g_small = to(small_grad, device).clone()
      x_full = to(x, device).clone()
      torch.ops.aten.silu_backward.grad_input(
          g_small, x_full, grad_input=g_small
      )
      return g_small

    self.assert_close_tpu_vs_cpu(compute_invalid_broadcast, allow_failure=True)

  def test_silu_backward_fake_tensor_and_compile(self):
    """Tests FakeTensorMode metadata propagation, validation, and torch.compile."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
      fg = torch.randn(2, 1, 4, dtype=torch.bfloat16, device="tpu")
      fx = torch.randn(2, 3, 4, dtype=torch.bfloat16, device="tpu")
      f_out = torch.ops.aten.silu_backward(fg, fx)
      self.assertEqual(f_out.shape, (2, 3, 4))
      self.assertEqual(f_out.dtype, torch.bfloat16)

      f_buf = torch.empty(0, dtype=torch.bfloat16, device="tpu")
      torch.ops.aten.silu_backward.grad_input(fg, fx, grad_input=f_buf)
      self.assertEqual(f_buf.shape, (2, 3, 4))

      f_int = torch.ones(2, 3, dtype=torch.int32, device="tpu")
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing FakeTensorMode dtype validation.
          (RuntimeError, NotImplementedError), "not implemented for"
      ):
        torch.ops.aten.silu_backward(f_int, f_int)

    from torch.fx.experimental.proxy_tensor import make_fx

    x = torch.randn(3, 4, dtype=torch.bfloat16)
    grad_out = torch.randn(3, 4, dtype=torch.bfloat16)

    def silu_fwd_bwd(inp, g):
      out = torch.nn.functional.silu(inp)
      return torch.autograd.grad(out, inp, g)[0]

    bwd_fx = make_fx(silu_fwd_bwd, tracing_mode="fake")(
        to(x, "tpu").requires_grad_(True), to(grad_out, "tpu")
    )
    bwd_targets = [
        n.target for n in bwd_fx.graph.nodes if n.op == "call_function"
    ]
    self.assertIn(torch.ops.aten.silu_backward.default, bwd_targets)

    @torch.compile(backend="tpu")
    def compiled_silu_bwd(g, inp):
      return torch.ops.aten.silu_backward(g, inp)

    res_tpu = compiled_silu_bwd(to(grad_out, "tpu"), to(x, "tpu")).cpu()
    res_cpu = torch.ops.aten.silu_backward(grad_out, x)
    utils.assert_close(res_tpu, res_cpu)

  def test_min_max_rwb_fusion(self):
    # Lowering min/max value outputs to MaxOp/MinOp (instead of SelectOp)
    # ensures XLA DCE simplifies unused-index reductions to kMaximum/kMinimum
    # required by TPU Reduce-Broadcast-into-ReduceWindow (RWB) fusion.
    # Quantize float32 inputs to bfloat16 mantissas so TPU v5e MXU bmm matches CPU fp32.
    q_val = torch.randn(4, 1045, 64, dtype=torch.bfloat16).float() * 0.125
    k_val = torch.randn(4, 256, 64, dtype=torch.bfloat16).float() * 0.125

    def rwb_fn(device):
      q = to(q_val, device)
      k = to(k_val, device)
      attn = torch.clamp(
          torch.bmm(q, k.transpose(1, 2)), min=-50000.0, max=50000.0
      )
      attn_t = attn.transpose(1, 2)
      text_attn = (
          attn_t
          - torch.max(attn_t, dim=-1, keepdim=True)[0]
          + torch.min(attn_t, dim=-1, keepdim=True)[0]
      )
      return text_attn.softmax(dim=-1)

    self.assert_close_tpu_vs_cpu(rwb_fn, rtol=1e-4, atol=1e-4)

  def test_min_max_nan_propagation(self):
    # Verify torch.max and torch.min (which use BuildReduceBody) propagate NaNs
    # matching PyTorch CPU regardless of whether NaN is in lhs or rhs.
    lhs_cpu = torch.tensor([float("nan"), 5.0])
    rhs_cpu = torch.tensor([5.0, float("nan")])
    stacked_cpu = torch.stack([lhs_cpu, rhs_cpu], dim=0)
    stacked_tpu = to(stacked_cpu, "tpu")
    utils.assert_close(
        torch.max(stacked_tpu, dim=0)[0], torch.max(stacked_cpu, dim=0)[0]
    )
    utils.assert_close(
        torch.min(stacked_tpu, dim=0)[0], torch.min(stacked_cpu, dim=0)[0]
    )

  def test_max_pool2d_with_indices_nan_propagation(self):
    # Verify max_pool2d_with_indices (which uses buildMaxAndArgmaxBody) propagates
    # NaNs matching PyTorch CPU regardless of whether NaN is first or last in the window.
    pool_in = torch.tensor(
        [[[[float("nan"), 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, float("nan")]]]]
    )
    tpu_pool_in = to(pool_in, "tpu")
    utils.assert_close(
        torch.nn.functional.max_pool2d(
            tpu_pool_in, kernel_size=2, return_indices=True
        )[0],
        torch.nn.functional.max_pool2d(
            pool_in, kernel_size=2, return_indices=True
        )[0],
    )

  def test_adaptive_max_pool2d_nan_propagation(self):
    # Verify adaptive_max_pool2d (which uses buildMaxAndArgmaxBody) propagates
    # NaNs matching PyTorch CPU regardless of whether NaN is first or last in the window.
    pool_in = torch.tensor(
        [[[[float("nan"), 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, float("nan")]]]]
    )
    tpu_pool_in = to(pool_in, "tpu")
    utils.assert_close(
        torch.nn.functional.adaptive_max_pool2d(tpu_pool_in, (1, 1))[0],
        torch.nn.functional.adaptive_max_pool2d(pool_in, (1, 1))[0],
    )

  def test_min_max_nan_selected_index(self):
    # Verify selected_index (indices) and values for torch.max/min and argmax/argmin
    # when NaN is first, last, or tied (both NaN), matching CPU.
    x_cpu = torch.tensor(
        [[float("nan"), 5.0, float("nan")], [5.0, float("nan"), float("nan")]],
        dtype=torch.float32,
    )
    x_tpu = to(x_cpu, "tpu")
    utils.assert_close(torch.max(x_tpu, dim=0), torch.max(x_cpu, dim=0))
    utils.assert_close(torch.min(x_tpu, dim=0), torch.min(x_cpu, dim=0))
    utils.assert_close(torch.argmax(x_tpu, dim=0), torch.argmax(x_cpu, dim=0))
    utils.assert_close(torch.argmin(x_tpu, dim=0), torch.argmin(x_cpu, dim=0))

  def test_max_pool2d_with_indices_nan_selected_index(self):
    # Verify both values and selected_index for max_pool2d_with_indices on NaN windows.
    pool_in = torch.tensor(
        [[[[float("nan"), 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, float("nan")]]]]
    )
    tpu_pool_in = to(pool_in, "tpu")
    utils.assert_close(
        torch.nn.functional.max_pool2d(
            tpu_pool_in, kernel_size=2, return_indices=True
        ),
        torch.nn.functional.max_pool2d(
            pool_in, kernel_size=2, return_indices=True
        ),
    )

  def test_adaptive_max_pool2d_nan_selected_index(self):
    # Verify both values and selected_index for adaptive_max_pool2d on NaN windows.
    pool_in = torch.tensor(
        [[[[float("nan"), 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, float("nan")]]]]
    )
    tpu_pool_in = to(pool_in, "tpu")
    utils.assert_close(
        torch.nn.functional.adaptive_max_pool2d(
            tpu_pool_in, (1, 1), return_indices=True
        ),
        torch.nn.functional.adaptive_max_pool2d(
            pool_in, (1, 1), return_indices=True
        ),
    )

  def test_linear_shapes_and_bias(self):
    cases = [
        # 1D input x 2D weight
        ((4,), (3, 4), None),
        ((4,), (3, 4), (3,)),
        # 1D input x 1D weight
        ((4,), (4,), None),
        ((4,), (4,), ()),
        # 2D input x 2D weight
        ((2, 4), (3, 4), None),
        ((2, 4), (3, 4), (3,)),
        ((2, 4), (3, 4), (1, 3)),
        ((2, 4), (3, 4), (2, 3)),
        # 2D/3D input x 1D weight
        ((2, 4), (4,), None),
        ((2, 5, 4), (4,), None),
        ((2, 5, 4), (4,), (2, 5)),
        # 3D input x 2D weight
        ((2, 5, 4), (3, 4), None),
        ((2, 5, 4), (3, 4), (3,)),
        ((2, 5, 4), (3, 4), (1, 5, 3)),
        # 4D input x 2D weight
        ((2, 3, 4, 5), (6, 5), None),
        ((2, 3, 4, 5), (6, 5), (6,)),
        # Zero-dimension batch shapes
        ((0, 4), (3, 4), None),
        ((0, 4), (3, 4), (3,)),
        ((2, 0, 4), (3, 4), None),
        ((2, 0, 4), (3, 4), (3,)),
        # Zero-sized contracting dimension
        ((3, 4, 0), (5, 0), None),
        ((3, 4, 0), (5, 0), (5,)),
        ((3, 4, 0), (0,), None),
        # Single-element shape
        ((1, 1), (1, 1), (1,)),
    ]
    for x_shape, w_shape, b_shape in cases:
      with self.subTest(x_shape=x_shape, w_shape=w_shape, b_shape=b_shape):
        x = torch.randn(x_shape, dtype=torch.float32)
        w = torch.randn(w_shape, dtype=torch.float32)
        b = (
            torch.randn(b_shape, dtype=torch.float32)
            if b_shape is not None
            else None
        )
        self.assert_close_tpu_vs_cpu(
            lambda device, x=x, w=w, b=b: torch.nn.functional.linear(
                to(x, device=device),
                to(w, device=device),
                to(b, device=device) if b is not None else None,
            ),
            rtol=2e-2,
            atol=2e-2,
        )

  def test_linear_dtypes(self):
    dtypes = [
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.complex64,
        torch.int32,
        torch.int64,
    ]
    for dtype in dtypes:
      for x_shape in [(2, 4), (2, 5, 4)]:
        for use_bias in [False, True]:
          with self.subTest(dtype=dtype, x_shape=x_shape, use_bias=use_bias):
            if dtype in (torch.int32, torch.int64):
              x = torch.randint(-5, 6, x_shape, dtype=dtype)
              w = torch.randint(-5, 6, (3, 4), dtype=dtype)
              b = torch.randint(-5, 6, (3,), dtype=dtype) if use_bias else None
              rtol, atol = None, None
            else:
              x = torch.randn(x_shape, dtype=dtype)
              w = torch.randn((3, 4), dtype=dtype)
              b = torch.randn((3,), dtype=dtype) if use_bias else None
              rtol, atol = 2e-2, 2e-2

            self.assert_close_tpu_vs_cpu(
                lambda device, x=x, w=w, b=b: torch.nn.functional.linear(
                    to(x, device=device),
                    to(w, device=device),
                    to(b, device=device) if b is not None else None,
                ),
                rtol=rtol,
                atol=atol,
            )

    # Verify Precision::HIGHEST mode with tight tolerances for float32/complex64
    prev_prec = torch.get_float32_matmul_precision()
    try:
      torch.set_float32_matmul_precision("highest")
      for dtype in [torch.float32, torch.complex64]:
        with self.subTest(dtype=dtype, precision="highest"):
          x = torch.randn((2, 5, 4), dtype=dtype)
          w = torch.randn((3, 4), dtype=dtype)
          b = torch.randn((3,), dtype=dtype)
          self.assert_close_tpu_vs_cpu(
              lambda device, x=x, w=w, b=b: torch.nn.functional.linear(
                  to(x, device=device),
                  to(w, device=device),
                  to(b, device=device),
              ),
              rtol=1e-4,
              atol=1e-4,
          )
    finally:
      torch.set_float32_matmul_precision(prev_prec)

  def test_linear_non_contiguous_and_views(self):
    x_base = torch.randn(5, 2, 4, dtype=torch.float32)
    w_base = torch.randn(4, 3, dtype=torch.float32)
    b_base = torch.randn(6, dtype=torch.float32)

    def run_non_contiguous(device):
      x = to(x_base, device=device).transpose(0, 1)
      w = to(w_base, device=device).t()
      b = to(b_base, device=device)[::2]
      return (
          torch.nn.functional.linear(x, w, b),
          torch.nn.functional.linear(x, w, None),
          torch.nn.functional.linear(x[:, ::2, :], w, b),
      )

    self.assert_close_tpu_vs_cpu(run_non_contiguous, rtol=2e-2, atol=2e-2)

  def test_linear_out(self):
    x_2d = torch.randn(2, 4, dtype=torch.float32)
    x_3d = torch.randn(2, 5, 4, dtype=torch.float32)
    w = torch.randn(3, 4, dtype=torch.float32)
    b = torch.randn(3, dtype=torch.float32)

    def run_out(device):
      results = []
      for x, out_shape in [(x_2d, (2, 3)), (x_3d, (2, 5, 3))]:
        x_dev = to(x, device=device)
        w_dev = to(w, device=device)
        b_dev = to(b, device=device)

        # Pre-allocated out with bias
        out_exact_b = torch.empty(out_shape, dtype=torch.float32, device=device)
        res_exact_b = torch.nn.functional.linear(
            x_dev, w_dev, b_dev, out=out_exact_b
        )
        results.append(out_exact_b)
        results.append(res_exact_b)

        # Pre-allocated out without bias
        out_exact_nob = torch.empty(
            out_shape, dtype=torch.float32, device=device
        )
        res_exact_nob = torch.nn.functional.linear(
            x_dev, w_dev, None, out=out_exact_nob
        )
        results.append(out_exact_nob)
        results.append(res_exact_nob)

        # Empty 0-element out (auto-resize) with and without bias
        out_empty_b = torch.empty(0, dtype=torch.float32, device=device)
        torch.nn.functional.linear(x_dev, w_dev, b_dev, out=out_empty_b)
        results.append(out_empty_b)

        out_empty_nob = torch.empty(0, dtype=torch.float32, device=device)
        torch.nn.functional.linear(x_dev, w_dev, None, out=out_empty_nob)
        results.append(out_empty_nob)

      return tuple(results)

    self.assert_close_tpu_vs_cpu(run_out, rtol=2e-2, atol=2e-2)

  @parameterized.named_parameters(
      ("int8", torch.int8),
      ("uint8", torch.uint8),
      ("int16", torch.int16),
      ("int32", torch.int32),
      ("int64", torch.int64),
  )
  def test_lcm_zero_and_signs(self, dtype: torch.dtype):
    """Tests lcm with zero edge cases and negative inputs across integer dtypes."""

    def compute(device):
      if dtype == torch.uint8:
        a = torch.tensor([0, 0, 12, 12, 15, 7], dtype=dtype, device=device)
        b = torch.tensor([0, 18, 0, 18, 20, 13], dtype=dtype, device=device)
      else:
        a = torch.tensor(
            [0, 0, 12, -12, 12, -12, 15, -7], dtype=dtype, device=device
        )
        b = torch.tensor(
            [0, 18, 0, 18, -18, -18, 20, 13], dtype=dtype, device=device
        )
      return torch.lcm(a, b)

    self.assert_close_tpu_vs_cpu(compute)

  def test_lcm_narrow_overflow_and_int_min(self):
    """Tests narrow integer overflow, INT_MIN, and large coprime Fibonacci pairs."""

    def compute_int8(device):
      a = torch.tensor([-20, -128, -128, 100], dtype=torch.int8, device=device)
      b = torch.tensor([14, 1, 2, 120], dtype=torch.int8, device=device)
      return torch.lcm(a, b)

    def compute_int16(device):
      a = torch.tensor([-32768, 300, -250], dtype=torch.int16, device=device)
      b = torch.tensor([1, 400, 350], dtype=torch.int16, device=device)
      return torch.lcm(a, b)

    def compute_int32_fibonacci(device):
      a = torch.tensor(
          [1836311903, -2147483648, 46340], dtype=torch.int32, device=device
      )
      b = torch.tensor([1134903170, 1, 46341], dtype=torch.int32, device=device)
      return torch.lcm(a, b)

    def compute_int64(device):
      a = torch.tensor(
          [1836311903, -(2**63), 1000000007], dtype=torch.int64, device=device
      )
      b = torch.tensor(
          [1134903170, 1, 1000000009], dtype=torch.int64, device=device
      )
      return torch.lcm(a, b)

    self.assert_close_tpu_vs_cpu(compute_int8)
    self.assert_close_tpu_vs_cpu(compute_int16)
    self.assert_close_tpu_vs_cpu(compute_int32_fibonacci)
    self.assert_close_tpu_vs_cpu(compute_int64)

  def test_lcm_promotion_out_cast_and_shapes(self):
    """Tests mixed-dtype promotion, out casting, resizing, 0-D, and 0-element tensors."""

    def compute(device):
      a_i16 = torch.tensor(
          [[12, -18], [0, 7]], dtype=torch.int16, device=device
      )
      b_i32 = torch.tensor([18, 24], dtype=torch.int32, device=device)
      res_promoted = torch.lcm(a_i16, b_i32)

      b_bool = torch.tensor([True, False], dtype=torch.bool, device=device)
      res_bool_promo = torch.lcm(b_i32, b_bool)

      out_i32 = torch.empty((2, 2), dtype=torch.int32, device=device)
      torch.lcm(a_i16, a_i16, out=out_i32)

      out_f32 = torch.empty((2, 2), dtype=torch.float32, device=device)
      torch.lcm(a_i16, b_i32, out=out_f32)

      out_resized = torch.empty(0, dtype=torch.int32, device=device)
      torch.lcm(a_i16, b_i32, out=out_resized)

      s1 = torch.tensor(-12, dtype=torch.int32, device=device)
      s2 = torch.tensor(18, dtype=torch.int32, device=device)
      res_scalar = torch.lcm(s1, s2)

      empty_a = torch.empty((2, 0, 3), dtype=torch.int32, device=device)
      empty_b = torch.empty((1, 0, 3), dtype=torch.int32, device=device)
      res_empty = torch.lcm(empty_a, empty_b)

      return (
          res_promoted,
          res_bool_promo,
          out_i32,
          out_f32,
          out_resized,
          res_scalar,
          res_empty,
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_lcm_inplace_and_out_buffer_donation(self):
    """Tests in-place lcm_ and out=x buffer donation in DEFER_NEVER mode."""
    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_NEVER):

      def compute(device):
        x1 = torch.tensor([12, -18, 0, 21], dtype=torch.int32, device=device)
        y1 = torch.tensor([18, 24, 5, 14], dtype=torch.int32, device=device)
        torch.lcm(x1, y1, out=x1)

        x2 = torch.tensor([15, -20, 8, 0], dtype=torch.int32, device=device)
        y2 = torch.tensor([25, 30, 12, 0], dtype=torch.int32, device=device)
        x2.lcm_(y2)
        return x1, x2

      self.assert_close_tpu_vs_cpu(compute)

  def test_var_empty_reduction_dim(self):
    """Tests 0-sized reduction with negative correction returns NaN (b/568045113)."""

    def compute(device):
      x = torch.empty((0, 3), dtype=torch.float32, device=device)
      return torch.var(x, dim=0, correction=-1)

    self.assert_close_tpu_vs_cpu(compute)

  def test_var_mean_empty_reduction_dim(self):
    """Tests 0-sized reduction with negative correction returns NaN (b/568045113)."""

    def compute(device):
      x = torch.empty((0, 3), dtype=torch.float32, device=device)
      return torch.var_mean(x, dim=0, correction=-1)

    self.assert_close_tpu_vs_cpu(compute)

  def test_sdpa_overrideable_meta_checks_kv_seq_len(self):
    """Tests SDPA overrideable meta checks and guards key/value sequence lengths."""
    shape_env = symbolic_shapes.ShapeEnv(duck_shape=False)
    with fake_tensor.FakeTensorMode(shape_env=shape_env):
      s_k = shape_env.create_symintnode(
          shape_env.create_symbol(
              8, source=torch._dynamo.source.LocalSource("s_k")
          ),
          hint=8,
      )
      s_v = shape_env.create_symintnode(
          shape_env.create_symbol(
              8, source=torch._dynamo.source.LocalSource("s_v")
          ),
          hint=8,
      )
      q = torch.empty((2, 4, 8, 16), dtype=torch.bfloat16, device="tpu")
      k = torch.empty((2, 4, s_k, 16), dtype=torch.bfloat16, device="tpu")
      v = torch.empty((2, 4, s_v, 16), dtype=torch.bfloat16, device="tpu")
      res = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
          q, k, v
      )
      self.assertEqual(res[0].shape, (2, 4, 8, 16))
      self.assertTrue(
          any(
              "s_k == s_v" in str(g.expr) or "Eq(" in str(g.expr)
              for g in shape_env.guards
          )
      )

      v_mismatched = torch.empty(
          (2, 4, 10, 16), dtype=torch.bfloat16, device="tpu"
      )
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Meta shape validation.
          RuntimeError,
          r"key sequence length \(8\) must match value sequence length \(10\)",
      ):
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
            q,
            torch.empty((2, 4, 8, 16), dtype=torch.bfloat16, device="tpu"),
            v_mismatched,
        )


class OpsCustomOpUnitTest(TorchTpuVsCpuTestBase):
  """Tests for custom ops."""

  def test_ragged_dot_on_tpu(self):
    """Tests the tpu.ragged_dot custom op on TPU."""
    device = torch.device("tpu")
    m, k, n, g = 5, 4, 3, 2
    lhs = torch.arange(m * k, dtype=torch.float32, device=device).reshape(m, k)
    rhs = torch.arange(g * k * n, dtype=torch.float32, device=device).reshape(
        g, k, n
    )
    group_sizes = torch.tensor([1, 4], dtype=torch.int32, device=device)
    out = torch.tpu.ops.ragged_dot(lhs, rhs, group_sizes)
    expected = torch.asarray(
        [
            [42.0, 48.0, 54.0],
            [378.0, 400.0, 422.0],
            [642.0, 680.0, 718.0],
            [906.0, 960.0, 1014.0],
            [1170.0, 1240.0, 1310.0],
        ],
        dtype=torch.float32,
    )
    self.assert_close(golden_result=expected.cpu(), torch_tpu_result=out.cpu())

  def test_ragged_dot_out_on_tpu(self):
    """Tests the tpu.ragged_dot custom op with an out parameter on TPU."""
    device = torch.device("tpu")
    m, k, n, g = 4, 3, 2, 5
    lhs = torch.arange(m * k, dtype=torch.float32, device=device).reshape(m, k)
    rhs = torch.arange(g * k * n, dtype=torch.float32, device=device).reshape(
        g, k, n
    )
    group_sizes = torch.tensor(
        [1, 1, 0, 1, 1], dtype=torch.int32, device=device
    )
    out = torch.zeros(m, n, dtype=torch.float32, device=device)
    res = torch.tpu.ops.ragged_dot(lhs, rhs, group_sizes, out=out)
    expected = torch.asarray(
        [
            [10.0, 13.0],
            [100.0, 112.0],
            [424.0, 445.0],
            [784.0, 814.0],
        ],
        dtype=torch.float32,
    )
    self.assert_close(golden_result=expected.cpu(), torch_tpu_result=res.cpu())
    self.assert_close(golden_result=expected.cpu(), torch_tpu_result=out.cpu())

  def test_ragged_dot_weight_grad_on_tpu(self):
    """Tests the tpu.ragged_dot_weight_grad custom op on TPU.

    Verifies that torch.tpu.ops.ragged_dot_weight_grad computes the gradient
    with respect to the 3D weights tensor (grad_rhs) by contracting activations
    and output gradients along the ragged token dimension.

    Expected behavior:
      For each group i with token count gs = group_sizes[i], the weight slice
      is computed as:
        expected[i] = lhs[start:end].T @ grad_output[start:end]
      where [start:end] corresponds to the tokens assigned to group i.
      The TPU kernel result must match this CPU golden slice contraction.
    """
    device = torch.device("tpu")
    m, k, n, g = 5, 4, 3, 2
    lhs = torch.arange(m * k, dtype=torch.float32, device=device).reshape(m, k)
    grad_output = torch.ones(m, n, dtype=torch.float32, device=device)
    group_sizes = torch.tensor([2, 3], dtype=torch.int32, device=device)
    res = torch.tpu.ops.ragged_dot_weight_grad(lhs, grad_output, group_sizes)

    # Compute expected golden reference slice-by-slice on CPU.
    lhs_cpu = lhs.cpu()
    grad_output_cpu = grad_output.cpu()
    expected = torch.zeros(g, k, n, dtype=torch.float32)
    idx = 0
    for i, gs in enumerate([2, 3]):
      if gs > 0:
        expected[i] = (
            lhs_cpu[idx : idx + gs].T @ grad_output_cpu[idx : idx + gs]
        )
        idx += gs

    self.assert_close(golden_result=expected, torch_tpu_result=res.cpu())

  @parameterized.named_parameters(
      dict(testcase_name="eager", compile_test=False),
      dict(testcase_name="compiled", compile_test=True),
  )
  def test_ragged_dot_autograd_on_tpu(self, compile_test: bool = False):
    """Tests full autograd backward differentiation for tpu.ragged_dot on TPU.

    Verifies that calling .backward() on the output of torch.tpu.ops.ragged_dot
    (under both eager execution and torch.compile with TpuBackend):
      1. Accurately differentiates with respect to activations (lhs, shape [M,
      K]).
         Activations gradient (grad_lhs) is computed via ragged contracting
         matmul:
           grad_lhs[slice_i] = grad_output[slice_i] @ rhs[i].T
      2. Accurately differentiates with respect to weights (rhs, shape [G, K,
      N]).
         Weights gradient (grad_rhs) is computed via ragged_dot_weight_grad:
           grad_rhs[i] = lhs[slice_i].T @ grad_output[slice_i]
      3. Gracefully handles zero-sized groups (e.g. gs=[2, 0, 4] where group 1
         has 0 tokens assigned). For zero-sized groups, the weight gradient
         slice
         should remain zero and cause no indexing errors.

    Expected behavior:
      Both forward output and backward gradients (lhs.grad and rhs.grad) on TPU
      must match the analytical golden reference computed on CPU.
    """
    device = torch.device("tpu")
    m, k, n, g = 6, 4, 3, 3
    lhs_cpu = torch.randn(m, k, dtype=torch.float32, requires_grad=True)
    rhs_cpu = torch.randn(g, k, n, dtype=torch.float32, requires_grad=True)
    # Include an empty group (size 0) to test ragged boundary edge cases.
    gs = [2, 0, 4]
    group_sizes = torch.tensor(gs, dtype=torch.int32, device=device)

    lhs_tpu = lhs_cpu.detach().to(device).requires_grad_(True)
    rhs_tpu = rhs_cpu.detach().to(device).requires_grad_(True)

    def op_fn(x, w, s):
      return torch.tpu.ops.ragged_dot(x, w, s)

    # Execute forward pass (either via torch.compile or eager).
    if compile_test:
      compiled_op = torch.compile(op_fn, backend="tpu")
      out_tpu = compiled_op(lhs_tpu, rhs_tpu, group_sizes)
    else:
      out_tpu = op_fn(lhs_tpu, rhs_tpu, group_sizes)

    grad_out = torch.randn_like(out_tpu)
    # Backward pass on TPU via registered AutogradPrivateUse1 kernel.
    out_tpu.backward(grad_out)

    # Compute golden forward and backward on CPU using slice matmuls.
    out_slices = []
    idx = 0
    for i, size in enumerate(gs):
      if size > 0:
        out_slices.append(lhs_cpu[idx : idx + size] @ rhs_cpu[i])
        idx += size
    out_cpu = torch.cat(out_slices, dim=0)
    out_cpu.backward(grad_out.cpu())

    # Verify forward output matches CPU reference.
    self.assert_close(
        golden_result=out_cpu,
        torch_tpu_result=out_tpu.cpu(),
        atol=5e-2,
        rtol=1e-1,
    )
    # Verify activation gradient (grad_lhs) matches CPU reference.
    self.assert_close(
        golden_result=lhs_cpu.grad,
        torch_tpu_result=lhs_tpu.grad.cpu(),
        atol=5e-2,
        rtol=1e-1,
    )
    # Verify weight gradient (grad_rhs) matches CPU reference.
    self.assert_close(
        golden_result=rhs_cpu.grad,
        torch_tpu_result=rhs_tpu.grad.cpu(),
        atol=5e-2,
        rtol=1e-1,
    )

  @absltest.skip("b/498564738")
  def test_set_dimension_logical_size_on_tpu(self):
    """Tests the tpu._set_dimension_logical_size custom op on TPU."""
    device = torch.device("tpu")
    x = torch.arange(100, device=device, dtype=torch.int32).reshape(10, 10)
    size = torch.tensor(5, device=device, dtype=torch.int32)
    out = torch.ops.tpu._set_dimension_logical_size(x, 0, size)
    self.assert_close(golden_result=x[:5].cpu(), torch_tpu_result=out[:5].cpu())

  def test_set_dimension_logical_size_with_mlir_on_tpu(self):
    """Tests the tpu._set_dimension_logical_size custom op on TPU using MLIR."""
    device = torch.device("tpu")
    x = torch.arange(25, device=device, dtype=torch.float32).reshape(5, 5)
    y = torch.arange(25, device=device, dtype=torch.float32).reshape(5, 5)
    size = torch.tensor(1, device=device, dtype=torch.int32)
    golden_result = torch.matmul(x[:, : size.item()], y[: size.item(), :])
    mlir_program = """
module {
  func.func @main(%arg0: tensor<5x5xf32>, %arg1: tensor<5x5xf32>, %arg2: tensor<i32>) -> tensor<5x5xf32> {
    %0 = stablehlo.set_dimension_size %arg0, %arg2, dim=1 : (tensor<5x5xf32>, tensor<i32>) -> tensor<5x?xf32, #stablehlo.bounds<?, 5>>
    %1 = stablehlo.set_dimension_size %arg1, %arg2, dim=0 : (tensor<5x5xf32>, tensor<i32>) -> tensor<?x5xf32, #stablehlo.bounds<5, ?>>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<5x?xf32, #stablehlo.bounds<?, 5>>, tensor<?x5xf32, #stablehlo.bounds<5, ?>>) -> tensor<5x5xf32>
    return %2: tensor<5x5xf32>
  }
}
"""
    module = tpu_torch_compile.parse_mlir_text(mlir_program)
    executable_key = tpu_torch_compile.compile_mlir(module)
    result = tpu_torch_compile.execute(executable_key, [x, y, size])
    self.assert_close(
        golden_result=golden_result.cpu(), torch_tpu_result=result[0].cpu()
    )

  def test_dynamic_arange_int_on_tpu(self):
    """Tests the tpu._dynamic_arange custom op on TPU with integers."""
    device = torch.device("tpu")
    start = torch.tensor(1, device=device, dtype=torch.int32)
    end = torch.tensor(6, device=device, dtype=torch.int32)
    step = torch.tensor(2, device=device, dtype=torch.int32)
    max_length = 5
    out = torch.ops.tpu._dynamic_arange(
        start, end, step, max_length, torch.int32
    )
    expected = torch.arange(1, 6, step=2, dtype=torch.int32)
    self.assert_close(
        golden_result=expected, torch_tpu_result=out[: expected.size(0)].cpu()
    )

  def test_dynamic_arange_float_on_tpu(self):
    """Tests the tpu._dynamic_arange custom op on TPU with floats."""
    device = torch.device("tpu")
    start = torch.tensor(0.0, device=device, dtype=torch.float32)
    end = torch.tensor(2.5, device=device, dtype=torch.float32)
    step = torch.tensor(0.5, device=device, dtype=torch.float32)
    max_length = 6
    out = torch.ops.tpu._dynamic_arange(
        start, end, step, max_length, torch.float32
    )
    expected = torch.arange(0.0, 2.5, step=0.5, dtype=torch.float32)
    self.assert_close(
        golden_result=expected, torch_tpu_result=out[: expected.size(0)].cpu()
    )

  def test_dynamic_arange_empty_on_tpu(self):
    """Tests the tpu._dynamic_arange custom op on TPU with an empty range."""
    device = torch.device("tpu")
    start = torch.tensor(5, device=device, dtype=torch.int32)
    end = torch.tensor(2, device=device, dtype=torch.int32)
    step = torch.tensor(1, device=device, dtype=torch.int32)
    max_length = 5
    out = torch.ops.tpu._dynamic_arange(
        start, end, step, max_length, torch.int32
    )
    expected = torch.tensor([], dtype=torch.int32)
    self.assert_close(golden_result=expected, torch_tpu_result=out[:0].cpu())

  def test_dynamic_broadcast_output_matches_static_bound(self):
    """Tests dynamic_broadcast when runtime output shape matches static bounds."""
    device = torch.device("tpu")
    x = torch.tensor([1.0, 2.0], device=device, dtype=torch.float32)

    shape = [
        torch.tensor(3, device=device, dtype=torch.int32),
        torch.tensor(2, device=device, dtype=torch.int32),
    ]
    broadcast_dims = [1]
    static_shape = [3, 2]
    is_dynamic = [True, False]

    out = torch.ops.tpu._dynamic_broadcast(
        x, shape, broadcast_dims, static_shape, is_dynamic
    )
    expected = x.unsqueeze(0).expand(3, 2)
    self.assert_close(golden_result=expected.cpu(), torch_tpu_result=out.cpu())

  def test_dynamic_broadcast_output_below_static_bound(self):
    """Tests dynamic_broadcast when runtime output shape is below static bounds."""
    device = torch.device("tpu")
    x = torch.tensor([1.0, 2.0], device=device, dtype=torch.float32)

    shape = [
        torch.tensor(3, device=device, dtype=torch.int32),
        torch.tensor(2, device=device, dtype=torch.int32),
    ]
    broadcast_dims = [1]
    static_shape = [5, 2]
    is_dynamic = [True, False]

    out = torch.ops.tpu._dynamic_broadcast(
        x, shape, broadcast_dims, static_shape, is_dynamic
    )
    expected = x.unsqueeze(0).expand(3, 2)
    self.assert_close(
        golden_result=expected.cpu(),
        torch_tpu_result=out.cpu().flatten()[:6].reshape(3, 2),
    )

  def test_dynamic_broadcast_dynamic_input_mapped(self):
    """Tests dynamic_broadcast when dynamic input dim maps to dynamic output dim."""
    device = torch.device("tpu")
    x = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float32)
    size = torch.tensor(2, device=device, dtype=torch.int32)

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Make input dynamic: physical [3], logical [2] (dim 0 is dynamic)
      x_dynamic = torch.ops.tpu._set_dimension_logical_size(x, 0, size)

      # Broadcast to physical [3, 4], logical [2, 4]
      # Input dim 0 (dynamic) maps to output dim 0 (dynamic)
      shape = [
          torch.tensor(2, device=device, dtype=torch.int32),
          torch.tensor(4, device=device, dtype=torch.int32),
      ]
      broadcast_dims = [0]
      static_shape = [3, 4]
      is_dynamic = [True, False]

      out = torch.ops.tpu._dynamic_broadcast(
          x_dynamic, shape, broadcast_dims, static_shape, is_dynamic
      )

    # Expected: x_dynamic logical [2] expanded to [2, 4]
    x_cpu = x.cpu()
    x_logical_cpu = x_cpu[:2]
    expected = x_logical_cpu.unsqueeze(1).expand(2, 4)
    self.assert_close(
        golden_result=expected,
        torch_tpu_result=out.cpu().flatten()[:8].reshape(2, 4),
    )

  def test_dynamic_broadcast_dynamic_input_and_new_dim_dynamic(self):
    """Tests dynamic_broadcast with dynamic input and a new dynamic output dim."""
    device = torch.device("tpu")
    x = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float32)
    size = torch.tensor(2, device=device, dtype=torch.int32)

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Make input dynamic: physical [3], logical [2] (dim 0 is dynamic)
      x_dynamic = torch.ops.tpu._set_dimension_logical_size(x, 0, size)

      # Broadcast to physical [3, 3], logical [2, 2]
      # Input dim 0 (dynamic) maps to output dim 1 (dynamic)
      # Output dim 0 is new dim, dynamic
      shape = [
          torch.tensor(2, device=device, dtype=torch.int32),
          torch.tensor(2, device=device, dtype=torch.int32),
      ]
      broadcast_dims = [1]
      static_shape = [3, 3]
      is_dynamic = [True, True]

      out = torch.ops.tpu._dynamic_broadcast(
          x_dynamic, shape, broadcast_dims, static_shape, is_dynamic
      )

    # Expected: x_dynamic logical [2] expanded to [2, 2]
    x_cpu = x.cpu()
    x_logical_cpu = x_cpu[:2]
    expected = x_logical_cpu.unsqueeze(0).expand(2, 2)
    self.assert_close(
        golden_result=expected,
        torch_tpu_result=out.cpu().flatten()[:4].reshape(2, 2),
    )

  def test_dynamic_reshape_flatten(self):
    """Tests dynamic_reshape flattening a dynamic input."""
    device = torch.device("tpu")
    x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
    size = torch.tensor(3, device=device, dtype=torch.int32)
    # Target shape: physical [12], logical [9] (dim 0 is dynamic)
    shape = [torch.tensor(9, device=device, dtype=torch.int32)]
    static_shape = [12]
    is_dynamic = [True]

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Make input dynamic: physical [3, 4], logical [3, 3] (dim 1 is dynamic)
      x_dynamic = torch.ops.tpu._set_dimension_logical_size(x, 1, size)
      out = torch.ops.tpu._dynamic_reshape(
          x_dynamic, shape, static_shape, is_dynamic
      )

    # Expected: x_dynamic logical is [3, 3], flattening it should
    # give 9 elements.
    x_cpu = x.cpu()
    x_logical_cpu = x_cpu[:, :3]
    expected = x_logical_cpu.flatten()
    self.assert_close(
        golden_result=expected, torch_tpu_result=out.cpu().flatten()[:9]
    )

  def test_dynamic_reshape_change_dims(self):
    """Tests dynamic_reshape changing dimensions of a 2D dynamic input to another 2D dynamic shape."""
    device = torch.device("tpu")
    x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
    size = torch.tensor(2, device=device, dtype=torch.int32)
    # Target shape: physical [2, 6], logical [2, 3] (dim 1 is dynamic)
    shape = [
        torch.tensor(2, device=device, dtype=torch.int32),
        torch.tensor(3, device=device, dtype=torch.int32),
    ]
    static_shape = [2, 6]
    is_dynamic = [False, True]

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Make input dynamic: physical [3, 4], logical [3, 2] (dim 1 is dynamic)
      x_dynamic = torch.ops.tpu._set_dimension_logical_size(x, 1, size)
      out = torch.ops.tpu._dynamic_reshape(
          x_dynamic, shape, static_shape, is_dynamic
      )

    # Expected: x_dynamic logical [3, 2] reshaped to [2, 3]
    x_cpu = x.cpu()
    x_logical_cpu = x_cpu[:, :2]
    expected = x_logical_cpu.reshape(2, 3)
    self.assert_close(
        golden_result=expected,
        torch_tpu_result=out.cpu().flatten()[:6].reshape(2, 3),
    )

  def test_dynamic_reshape_add_singleton_dims(self):
    """Tests dynamic_reshape adding singleton dimensions to a dynamic input."""
    device = torch.device("tpu")
    x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
    size = torch.tensor(2, device=device, dtype=torch.int32)
    # Target shape: physical [1, 3, 1, 4], logical [1, 3, 1, 2]
    shape = [
        torch.tensor(1, device=device, dtype=torch.int32),
        torch.tensor(3, device=device, dtype=torch.int32),
        torch.tensor(1, device=device, dtype=torch.int32),
        torch.tensor(2, device=device, dtype=torch.int32),
    ]
    static_shape = [1, 3, 1, 4]
    is_dynamic = [False, False, False, True]

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Make input dynamic: physical [3, 4], logical [3, 2] (dim 1 is dynamic)
      x_dynamic = torch.ops.tpu._set_dimension_logical_size(x, 1, size)
      out = torch.ops.tpu._dynamic_reshape(
          x_dynamic, shape, static_shape, is_dynamic
      )

    # Expected: x_dynamic logical [3, 2] reshaped to [1, 3, 1, 2]
    x_cpu = x.cpu()
    x_logical_cpu = x_cpu[:, :2]
    expected = x_logical_cpu.reshape(1, 3, 1, 2)
    self.assert_close(
        golden_result=expected,
        torch_tpu_result=out.cpu().flatten()[:6].reshape(1, 3, 1, 2),
    )

  def test_dynamic_reshape_unflatten(self):
    """Tests dynamic_reshape unflattening a 1D dynamic input to a 2D dynamic output."""
    device = torch.device("tpu")
    x = torch.arange(12, device=device, dtype=torch.float32)
    size = torch.tensor(8, device=device, dtype=torch.int32)
    # Target shape: physical [2, 6], logical [2, 4] (dim 1 is dynamic)
    shape = [
        torch.tensor(2, device=device, dtype=torch.int32),
        torch.tensor(4, device=device, dtype=torch.int32),
    ]
    static_shape = [2, 6]
    is_dynamic = [False, True]

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Make input dynamic: physical [12], logical [8] (dim 0 is dynamic)
      x_dynamic = torch.ops.tpu._set_dimension_logical_size(x, 0, size)
      out = torch.ops.tpu._dynamic_reshape(
          x_dynamic, shape, static_shape, is_dynamic
      )

    expected = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    self.assert_close(
        golden_result=expected,
        torch_tpu_result=out.cpu().flatten()[:8].reshape(2, 4),
    )

  def test_dynamic_reshape_move_dynamic_dim(self):
    """Tests dynamic_reshape moving the dynamic property from one dim to another."""
    device = torch.device("tpu")
    x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
    size = torch.tensor(2, device=device, dtype=torch.int32)
    # Target shape: physical [2, 6], logical [2, 4] (dim 1 is dynamic)
    shape = [
        torch.tensor(2, device=device, dtype=torch.int32),
        torch.tensor(4, device=device, dtype=torch.int32),
    ]
    static_shape = [2, 6]
    is_dynamic = [False, True]

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Make input dynamic: physical [3, 4], logical [2, 4] (dim 0 is dynamic)
      x_dynamic = torch.ops.tpu._set_dimension_logical_size(x, 0, size)
      out = torch.ops.tpu._dynamic_reshape(
          x_dynamic, shape, static_shape, is_dynamic
      )

    # Expected: x_dynamic logical [2, 4] reshaped to [2, 4]
    x_cpu = x.cpu()
    x_logical_cpu = x_cpu[:2, :]
    expected = x_logical_cpu.reshape(2, 4)
    self.assert_close(
        golden_result=expected,
        torch_tpu_result=out.cpu().flatten()[:8].reshape(2, 4),
    )

  def test_dynamic_reshape_unflatten_dynamic_dim(self):
    """Tests dynamic_reshape unflattening (splitting) a dynamic dimension."""
    device = torch.device("tpu")
    x = torch.arange(12, device=device, dtype=torch.float32).reshape(2, 6)
    size = torch.tensor(4, device=device, dtype=torch.int32)
    # Target shape: physical [2, 2, 3], logical [2, 2, 2] (dim 2 is dynamic)
    shape = [
        torch.tensor(2, device=device, dtype=torch.int32),
        torch.tensor(2, device=device, dtype=torch.int32),
        torch.tensor(2, device=device, dtype=torch.int32),
    ]
    static_shape = [2, 2, 3]
    is_dynamic = [False, False, True]

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Make input dynamic: physical [2, 6], logical [2, 4] (dim 1 is dynamic)
      x_dynamic = torch.ops.tpu._set_dimension_logical_size(x, 1, size)
      out = torch.ops.tpu._dynamic_reshape(
          x_dynamic, shape, static_shape, is_dynamic
      )

    # Expected: x_dynamic logical [2, 4] reshaped to [2, 2, 2]
    x_cpu = x.cpu()
    x_logical_cpu = x_cpu[:, :4]
    expected = x_logical_cpu.reshape(2, 2, 2)
    self.assert_close(
        golden_result=expected,
        torch_tpu_result=out.cpu().flatten()[:8].reshape(2, 2, 2),
    )

  @parameterized.product(
      index_dtype=[torch.int32, torch.int64],
  )
  def test_dynamic_slice_1d(self, index_dtype):
    """Tests torch.ops.tpu._dynamic_slice on 1D tensor."""
    device = torch.device("tpu")
    x = torch.arange(10, device=device, dtype=torch.float32)
    start_indices = [torch.tensor(3, device=device, dtype=index_dtype)]
    slice_sizes = [4]

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      out = torch.ops.tpu._dynamic_slice(x, start_indices, slice_sizes)

    expected = x.cpu()[3:7]
    self.assert_close(golden_result=expected, torch_tpu_result=out.cpu())

  @parameterized.product(
      index_dtype=[torch.int32, torch.int64],
  )
  def test_dynamic_slice_2d(self, index_dtype):
    """Tests torch.ops.tpu._dynamic_slice on 2D tensor."""
    device = torch.device("tpu")
    x = torch.arange(20, device=device, dtype=torch.float32).reshape(4, 5)
    start_indices = [
        torch.tensor(1, device=device, dtype=index_dtype),
        torch.tensor(2, device=device, dtype=index_dtype),
    ]
    slice_sizes = [2, 3]

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      out = torch.ops.tpu._dynamic_slice(x, start_indices, slice_sizes)

    expected = x.cpu()[1:3, 2:5]
    self.assert_close(golden_result=expected, torch_tpu_result=out.cpu())

  def test_pad_real_to_static_with_mlir_on_tpu(self):
    """Tests the PadRealToStatic custom call on TPU using direct MLIR."""

    mlir_program = """
module {
  func.func @main(%input: tensor<2x4x?xi32, #stablehlo.bounds<?, ?, 1024>>, %d_dim2: tensor<i32>) -> tensor<2x4x?xi32, #stablehlo.bounds<?, ?, 1024>> {
    %c_dim0 = stablehlo.constant dense<2> : tensor<i32>
    %c_dim1 = stablehlo.constant dense<4> : tensor<i32>

    %0 = stablehlo.custom_call @PadRealToStatic(%input, %c_dim0, %c_dim1, %d_dim2) : (tensor<2x4x?xi32, #stablehlo.bounds<?, ?, 1024>>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x4x1024xi32>
    %1 = stablehlo.set_dimension_size %0, %d_dim2, dim=2 : (tensor<2x4x1024xi32>, tensor<i32>) -> tensor<2x4x?xi32, #stablehlo.bounds<?, ?, 1024>>
    return %1 : tensor<2x4x?xi32, #stablehlo.bounds<?, ?, 1024>>
  }
}
"""
    module = tpu_torch_compile.parse_mlir_text(mlir_program)
    executable_key = tpu_torch_compile.compile_mlir(module)

    device = torch.device("tpu")
    for dim2 in range(8, 1025, 8):
      x = (
          torch.arange(2 * 4 * dim2, device="cpu", dtype=torch.int32)
          .reshape(2, 4, dim2)
          .to(device)
      )
      x_d2 = torch.tensor(dim2, device="cpu", dtype=torch.int32).to(device)

      result = tpu_torch_compile.execute(
          executable_key,
          [x, x_d2],
          [tpu_torch_compile.OutputShape([2, 4, dim2])],
      )
      self.assert_close(
          golden_result=x.to("cpu"),
          torch_tpu_result=result[0].cpu(),
      )

    # Test with a larger dimension and static input shape
    x = (
        torch.arange(2 * 4 * 1021, device="cpu", dtype=torch.int32)
        .reshape(2, 4, 1021)
        .to(device)
    )
    x_d2 = torch.tensor(1021, device="cpu", dtype=torch.int32).to(device)

    result = tpu_torch_compile.execute(
        executable_key,
        [x, x_d2],
        [tpu_torch_compile.OutputShape([2, 4, 1021])],
    )
    self.assert_close(
        golden_result=x.to("cpu"),
        torch_tpu_result=result[0].cpu(),
    )

    # Test with a larger dimension and dynamic input (with prefix metadata)
    # Previous execution creates dynamic output with prefix metadata. Used that
    # as input to this execution to test dynamic input with prefix metadata.
    result_new = tpu_torch_compile.execute(
        executable_key,
        [result[0], x_d2],
        [tpu_torch_compile.OutputShape([2, 4, 1021])],
    )
    self.assert_close(
        golden_result=x.to("cpu"),
        torch_tpu_result=result_new[0].cpu(),
    )

  @parameterized.parameters(64, 128)
  def test_pad_real_to_static_kv_cache_multi_layout_execution_on_tpu(
      self, head_dim
  ):
    """Tests compiles executables with multiple layouts,

    caches them, selects the executable based on
    get_device_layout_if_materialized, and verifies accuracy against CPU.
    """
    mlir_program = f"""
module {{
  func.func @main(
      %key: tensor<1x8x?x{head_dim}xf32, #stablehlo.bounds<?, ?, 4096, ?>>,
      %d_seq: tensor<i32>,
      %new_key: tensor<1x8x1x{head_dim}xf32>
  ) -> tensor<1x8x?x{head_dim}xf32, #stablehlo.bounds<?, ?, 4097, ?>> {{
    %c_b = stablehlo.constant dense<1> : tensor<i32>
    %c_h = stablehlo.constant dense<8> : tensor<i32>
    %c_d = stablehlo.constant dense<{head_dim}> : tensor<i32>

    %0 = stablehlo.custom_call @PadRealToStatic(%key, %c_b, %c_h, %d_seq, %c_d)
        : (tensor<1x8x?x{head_dim}xf32, #stablehlo.bounds<?, ?, 4096, ?>>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x4096x{head_dim}xf32>
    %1 = stablehlo.set_dimension_size %0, %d_seq, dim=2
        : (tensor<1x8x4096x{head_dim}xf32>, tensor<i32>) -> tensor<1x8x?x{head_dim}xf32, #stablehlo.bounds<?, ?, 4096, ?>>
    %2 = stablehlo.concatenate %1, %new_key, dim=2
        : (tensor<1x8x?x{head_dim}xf32, #stablehlo.bounds<?, ?, 4096, ?>>, tensor<1x8x1x{head_dim}xf32>) -> tensor<1x8x?x{head_dim}xf32, #stablehlo.bounds<?, ?, 4097, ?>>
    return %2 : tensor<1x8x?x{head_dim}xf32, #stablehlo.bounds<?, ?, 4097, ?>>
  }}
}}
"""
    # Compile executables with layouts {3,2,1,0}, {2,3,1,0}, and {3,1,2,0} and cache them
    module_3210 = tpu_torch_compile.parse_mlir_text(mlir_program)
    executable_3210 = tpu_torch_compile.compile_mlir(
        module_3210, argument_layouts=[[3, 2, 1, 0], [], []]
    )

    module_2310 = tpu_torch_compile.parse_mlir_text(mlir_program)
    executable_2310 = tpu_torch_compile.compile_mlir(
        module_2310, argument_layouts=[[2, 3, 1, 0], [], []]
    )

    module_3120 = tpu_torch_compile.parse_mlir_text(mlir_program)
    executable_3120 = tpu_torch_compile.compile_mlir(
        module_3120, argument_layouts=[[3, 1, 2, 0], [], []]
    )

    cached_executables = {
        (3, 2, 1, 0): executable_3210,
        (2, 3, 1, 0): executable_2310,
        (3, 1, 2, 0): executable_3120,
    }

    device = torch.device("tpu")
    seq_len = 2048
    key_cpu = torch.randn(1, 8, seq_len, head_dim, dtype=torch.float32)
    key = key_cpu.clone().to(device)
    d_seq = torch.tensor(seq_len, device="cpu", dtype=torch.int32).to(device)
    new_key_cpu = torch.randn(1, 8, 1, head_dim, dtype=torch.float32)
    new_key = new_key_cpu.clone().to(device)

    # Prefill step on CPU
    golden_cpu_key = torch.cat([key_cpu, new_key_cpu], dim=2)

    # Prefill step on TPU: check input layout and select executable
    layout_info = tpu_torch_compile.get_device_layout_if_materialized(key)
    self.assertIsNotNone(layout_info)
    minor_to_major = tuple(layout_info[0])
    self.assertIn(minor_to_major, cached_executables)

    result = tpu_torch_compile.execute(
        cached_executables[minor_to_major],
        [key, d_seq, new_key],
        [tpu_torch_compile.OutputShape([1, 8, seq_len + 1, head_dim])],
    )
    current_key = result[0]
    self.assertEqual(list(current_key.shape), [1, 8, seq_len + 1, head_dim])
    self.assert_close(
        golden_result=golden_cpu_key,
        torch_tpu_result=current_key.cpu(),
    )

    # Decode steps: 10 decode iterations
    for _ in range(10):
      seq_len += 1
      d_seq = torch.tensor(seq_len, device="cpu", dtype=torch.int32).to(device)
      new_key_cpu = torch.randn(1, 8, 1, head_dim, dtype=torch.float32)
      new_key = new_key_cpu.clone().to(device)
      golden_cpu_key = torch.cat([golden_cpu_key, new_key_cpu], dim=2)

      # Check layout of the input to the decode step
      layout_info = tpu_torch_compile.get_device_layout_if_materialized(
          current_key
      )
      self.assertIsNotNone(layout_info)
      minor_to_major = tuple(layout_info[0])
      self.assertIn(minor_to_major, cached_executables)

      executable = cached_executables[minor_to_major]
      result = tpu_torch_compile.execute(
          executable,
          [current_key, d_seq, new_key],
          [tpu_torch_compile.OutputShape([1, 8, seq_len + 1, head_dim])],
      )
      current_key = result[0]
      self.assert_close(
          golden_result=golden_cpu_key,
          torch_tpu_result=current_key.cpu(),
      )

  def test_pad_real_to_static_dynamic_dimension(self):
    """Tests torch.ops.tpu._pad_real_to_static with a dynamic dimension."""
    device = torch.device("tpu")
    max_dim2 = 1024
    real_dim2 = 512

    x_ph = tpu_torch_compile.dynamic_placeholder(
        (2, 4, max_dim2),
        torch.int32,
        tpu_torch_compile.TensorBounds(([2], [max_dim2])),
    )
    dim0_ph = tpu_torch_compile.placeholder_like(
        torch.tensor(2, dtype=torch.int32)
    )
    dim1_ph = tpu_torch_compile.placeholder_like(
        torch.tensor(4, dtype=torch.int32)
    )
    dim2_ph = tpu_torch_compile.placeholder_like(
        torch.tensor(real_dim2, dtype=torch.int32)
    )

    with execution_mode.set_eager_mode(
        execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
    ):
      out_ph = torch.ops.tpu._pad_real_to_static(
          x_ph,
          [dim0_ph, dim1_ph, dim2_ph],
          [2, 4, max_dim2],
          [False, False, True],
      )

    compile_result = tpu_torch_compile.traverse_and_compile(
        [out_ph],
        [x_ph, dim0_ph, dim1_ph, dim2_ph],
        use_stablehlo_bounds=True,
    )

    x = (
        torch.arange(2 * 4 * real_dim2, device="cpu", dtype=torch.int32)
        .reshape(2, 4, real_dim2)
        .to(device)
    )
    dim0 = torch.tensor(2, device="cpu", dtype=torch.int32).to(device)
    dim1 = torch.tensor(4, device="cpu", dtype=torch.int32).to(device)
    dim2 = torch.tensor(real_dim2, device="cpu", dtype=torch.int32).to(device)

    result = tpu_torch_compile.execute(
        compile_result.executable,
        [x, dim0, dim1, dim2],
        [tpu_torch_compile.OutputShape([2, 4, real_dim2])],
    )

    self.assert_close(
        golden_result=x.cpu(),
        torch_tpu_result=result[0].cpu(),
    )

  @parameterized.product(
      dtype=[torch.float32, torch.bfloat16, torch.float16],
  )
  def test_grouped_mm_2d_3d(self, dtype):
    """Tests torch.ops.aten._grouped_mm: Case 1 (2D x 3D with offs)."""
    t1 = torch.randn(16, 8, dtype=dtype)
    t2 = torch.randn(2, 8, 8, dtype=dtype)
    offs = torch.tensor([8, 16], dtype=torch.int32)
    self.assert_close_tpu_vs_cpu(
        lambda device, t1=t1, t2=t2, offs=offs: torch.ops.aten._grouped_mm(
            t1.to(device), t2.to(device), offs=offs.to(device)
        ),
        rtol=2e-1,
        atol=2e-1,
    )

  @parameterized.product(
      dtype=[torch.float32, torch.bfloat16, torch.float16],
  )
  def test_grouped_mm_3d_2d(self, dtype):
    """Tests torch.ops.aten._grouped_mm: Case 2 (3D x 2D with offs)."""
    t1 = torch.randn(2, 8, 8, dtype=dtype)
    t2 = torch.randn(8, 16, dtype=dtype)
    offs = torch.tensor([8, 16], dtype=torch.int32)
    self.assert_close_tpu_vs_cpu(
        lambda device, t1=t1, t2=t2, offs=offs: torch.ops.aten._grouped_mm(
            t1.to(device), t2.to(device), offs=offs.to(device)
        ),
        rtol=2e-1,
        atol=2e-1,
    )

  @parameterized.product(
      dtype=[torch.float32, torch.bfloat16, torch.float16],
  )
  def test_grouped_mm_2d_2d(self, dtype):
    """Tests torch.ops.aten._grouped_mm: Case 3 (2D x 2D with offs)."""
    t1 = torch.randn(8, 16, dtype=dtype)
    t2 = torch.randn(16, 8, dtype=dtype)
    offs = torch.tensor([8, 16], dtype=torch.int32)
    self.assert_close_tpu_vs_cpu(
        lambda device, t1=t1, t2=t2, offs=offs: torch.ops.aten._grouped_mm(
            t1.to(device), t2.to(device), offs=offs.to(device)
        ),
        rtol=2e-1,
        atol=2e-1,
    )

  @parameterized.product(
      dtype=[torch.float32, torch.bfloat16, torch.float16],
  )
  def test_grouped_mm_3d_3d(self, dtype):
    """Tests torch.ops.aten._grouped_mm: Case 4 (3D x 3D without offs)."""
    t1 = torch.randn(2, 8, 8, dtype=dtype)
    t2 = torch.randn(2, 8, 8, dtype=dtype)
    self.assert_close_tpu_vs_cpu(
        lambda device, t1=t1, t2=t2: torch.ops.aten._grouped_mm(
            t1.to(device), t2.to(device), offs=None
        ),
        rtol=2e-1,
        atol=2e-1,
    )

  def test_aten_grouped_mm_int64_offs_eager_and_compiled(self):
    """Tests that eager and compiled execution of aten._grouped_mm on TPU handles int64 offsets."""
    device = torch.device("tpu")
    t1_cpu = torch.randn(16, 8, dtype=torch.bfloat16)
    t2_cpu = torch.randn(2, 8, 8, dtype=torch.bfloat16)
    offs_i32_cpu = torch.tensor([8, 16], dtype=torch.int32)
    offs_i64_cpu = torch.tensor([8, 16], dtype=torch.int64)

    out_i32 = torch.ops.aten._grouped_mm(
        t1_cpu.to(device), t2_cpu.to(device), offs_i32_cpu.to(device)
    )
    out_i64_eager = torch.ops.aten._grouped_mm(
        t1_cpu.to(device), t2_cpu.to(device), offs_i64_cpu.to(device)
    )
    utils.assert_close(out_i64_eager.cpu(), out_i32.cpu(), rtol=1e-3, atol=1e-3)

    compiled_fn = torch.compile(
        lambda a, b, offs: torch.ops.aten._grouped_mm(a, b, offs),
        backend="tpu",
        fullgraph=True,
    )
    out_i64_compiled = compiled_fn(
        t1_cpu.to(device), t2_cpu.to(device), offs_i64_cpu.to(device)
    )
    utils.assert_close(
        out_i64_compiled.cpu(), out_i32.cpu(), rtol=1e-3, atol=1e-3
    )

  def test_aten_grouped_mm_transposed_3d_rhs_eager_and_compiled(self):
    """Tests transposed [g, k, n] view over contiguous [g, n, k] expert weights."""
    device = torch.device("tpu")
    lhs_cpu = torch.randn(256, 128, dtype=torch.bfloat16)
    rhs_storage_cpu = torch.randn(4, 256, 128, dtype=torch.bfloat16)
    rhs_contig_cpu = rhs_storage_cpu.transpose(1, 2).contiguous()
    offs_cpu = torch.tensor([64, 128, 192, 256], dtype=torch.int32)

    expected = torch.ops.aten._grouped_mm(
        lhs_cpu.to(device), rhs_contig_cpu.to(device), offs_cpu.to(device)
    )
    actual_eager = torch.ops.aten._grouped_mm(
        lhs_cpu.to(device),
        rhs_storage_cpu.to(device).transpose(1, 2),
        offs_cpu.to(device),
    )
    utils.assert_close(actual_eager.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)

    compiled_fn = torch.compile(
        lambda x, w, o: torch.ops.aten._grouped_mm(x, w.transpose(1, 2), o),
        backend="tpu",
        fullgraph=True,
    )
    actual_compiled = compiled_fn(
        lhs_cpu.to(device), rhs_storage_cpu.to(device), offs_cpu.to(device)
    )
    utils.assert_close(
        actual_compiled.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2
    )

  def test_aten_grouped_mm_meta_symbolic_shapes(self):
    """Tests the TPU meta preserves SymInt shapes without guarding."""
    shape_env = symbolic_shapes.ShapeEnv()
    with fake_tensor.FakeTensorMode(shape_env=shape_env):
      s0 = shape_env.create_symintnode(
          shape_env.create_symbol(
              16, source=torch._dynamo.source.LocalSource("m")
          ),
          hint=16,
      )
      s1 = shape_env.create_symintnode(
          shape_env.create_symbol(
              2, source=torch._dynamo.source.LocalSource("g")
          ),
          hint=2,
      )
      # Symbolic M and group count; the batch check sees `s1 == s1`.
      a_sym = torch.empty((s0, 8), dtype=torch.bfloat16, device="tpu")
      b = torch.empty((s1, 8, 4), dtype=torch.bfloat16, device="tpu")
      offs = torch.empty((s1,), dtype=torch.int64, device="tpu")
      out = torch.ops.aten._grouped_mm(a_sym, b, offs)
      self.assertIsInstance(out.shape[0], torch.SymInt)
      self.assertEqual(out.shape[1], 4)
      self.assertEqual(len(shape_env.guards), 0)

  @parameterized.named_parameters(
      ("2d_3d", False, True),
      ("3d_2d", True, False),
      ("3d_3d", True, True),
  )
  def test_aten_grouped_mm_meta_unbacked_symbolic_shapes(
      self, a_is_3d, b_is_3d
  ):
    """Tests the TPU meta defers data-dependent batch checks, not guards."""
    shape_env = symbolic_shapes.ShapeEnv()
    with fake_tensor.FakeTensorMode(shape_env=shape_env):
      # Two distinct unbacked symints make the batch-size check data
      # dependent; guarding on it would raise GuardOnDataDependentSymNode.
      u0 = shape_env.create_unbacked_symint()
      u1 = shape_env.create_unbacked_symint()
      torch._check_is_size(u0)
      torch._check_is_size(u1)
      a_shape = (u0, 16, 8) if a_is_3d else (16, 8)
      b_shape = ((u1 if a_is_3d else u0), 8, 4) if b_is_3d else (8, 4)
      a = torch.empty(a_shape, dtype=torch.bfloat16, device="tpu")
      b = torch.empty(b_shape, dtype=torch.bfloat16, device="tpu")
      offs = None
      if not (a_is_3d and b_is_3d):
        offs = torch.empty((u1,), dtype=torch.int64, device="tpu")
      out = torch.ops.aten._grouped_mm(a, b, offs)
      self.assertEqual(tuple(out.shape[-2:]), (16, 4))
      self.assertEqual(out.dtype, torch.bfloat16)

  def test_aten_grouped_mm_meta_non_tpu_uses_upstream(self):
    """Tests the TPU Meta override leaves other devices on upstream's meta."""
    with fake_tensor.FakeTensorMode():
      a = torch.empty((16, 8), dtype=torch.bfloat16, device="cpu")
      # 16-byte aligned so upstream reaches its int32-only offs check.
      b = torch.empty((2, 8, 8), dtype=torch.bfloat16, device="cpu")
      offs = torch.empty((2,), dtype=torch.int64, device="cpu")
      # Upstream only accepts int32 offsets; the TPU override accepts int64.
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Upstream meta fallback.
          RuntimeError, "int32"
      ):
        torch.ops.aten._grouped_mm(a, b, offs)
      # Upstream's 16-byte stride check still applies off TPU.
      b_unaligned = torch.empty((2, 8, 4), dtype=torch.bfloat16, device="cpu")
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Upstream meta fallback.
          RuntimeError, "16 bytes"
      ):
        torch.ops.aten._grouped_mm(a, b_unaligned, offs.int())
      out = torch.ops.aten._grouped_mm(a.to("tpu"), b.to("tpu"), offs.to("tpu"))
      self.assertEqual(out.shape, (16, 8))
      self.assertEqual(out.dtype, torch.bfloat16)

  def test_aten_grouped_mm_compiled_rejects_invalid_inputs(self):
    """Tests compile raises the same contraction error as eager."""
    device = torch.device("tpu")
    a = torch.randn(2, 4, 8, device=device)
    b = torch.randn(5, 8, device=device)
    offs = torch.tensor([2, 4], dtype=torch.int32, device=device)
    compiled_fn = torch.compile(
        lambda a, b, offs: torch.ops.aten._grouped_mm(a, b, offs),
        backend="tpu",
        fullgraph=True,
    )
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing compile parity.
        RuntimeError,
        "expected the contraction dimension of self and mat2 to match, got 8"
        " and 5",
    ):
      compiled_fn(a, b, offs)

  def _ragged_k_group_sizes(
      self, num_tokens: int, num_groups: int, layout: str
  ) -> list[int]:
    """Returns per-group token counts for a ragged-K grouped_mm test."""
    if layout == "uneven":
      gen = torch.Generator().manual_seed(num_tokens + num_groups)
      weights = torch.rand(num_groups, generator=gen) + 0.05
      sizes = (weights / weights.sum() * num_tokens).floor().long().tolist()
      sizes[-1] += num_tokens - sum(sizes)
      return sizes
    # "tail" leaves trailing tokens outside every group.
    used = num_tokens - 13 if layout == "tail" else num_tokens
    base, rem = divmod(used, num_groups)
    sizes = [base + (1 if i < rem else 0) for i in range(num_groups)]
    if layout == "one_empty":
      empty = num_groups // 2
      sizes[empty + 1] += sizes[empty]
      sizes[empty] = 0
    elif layout == "trailing_empty":
      # The last group is empty: offs[-1] == offs[-2].
      sizes[-2] += sizes[-1]
      sizes[-1] = 0
    return sizes

  def _assert_ragged_k_close(
      self,
      out: torch.Tensor,
      lhs_ref: torch.Tensor,
      rhs_ref: torch.Tensor,
      group_sizes: list[int],
  ) -> None:
    """Compares a Case 3 result with a CPU fp32 per-group reference.

    Each group g uses rtol=2e-2 and atol=2e-2 * max|ref_g|, so an empty group
    must come back exactly zero.

    Args:
      out: The [g, m, n] TPU result.
      lhs_ref: The [m, T] fp32 CPU lhs (dequantized for FP8).
      rhs_ref: The [T, n] fp32 CPU rhs (dequantized for FP8).
      group_sizes: Tokens per group along T.
    """
    out = out.cpu().float()
    self.assertEqual(
        out.shape, (len(group_sizes), lhs_ref.shape[0], rhs_ref.shape[1])
    )
    start = 0
    for g, size in enumerate(group_sizes):
      end = start + size
      ref = lhs_ref[:, start:end] @ rhs_ref[start:end]
      ref_norm = ref.norm().item()
      err_norm = (out[g] - ref).norm().item()
      rel_fro = err_norm / ref_norm if ref_norm > 0 else err_norm
      utils.assert_close(
          out[g],
          ref,
          rtol=2e-2,
          atol=2e-2 * ref.abs().max().item(),
          preamble=(
              f"group {g} (size {size}, relative Frobenius error {rel_fro:.3g})"
          ),
      )
      start = end

  @parameterized.product(
      num_tokens=[4096, 4104],
      num_groups=[8, 128],
      layout=["even", "uneven", "one_empty", "trailing_empty", "tail"],
      mn=[(256, 128), (128, 256)],
  )
  def test_grouped_mm_2d_2d_ragged_k_wgrad(
      self, num_tokens, num_groups, layout, mn
  ):
    """Tests _grouped_mm Case 3 as a MoE weight gradient (x.t() @ dy).

    Regression test for github.com/google-pytorch/torch_tpu/issues/2668, where
    groups inside the first contracting tile came back wrong. 4104 is not a
    multiple of 256 and exercises the padding in the lowering.

    Args:
      num_tokens: Total tokens T (the ragged contracting dimension).
      num_groups: Number of groups (experts).
      layout: How tokens are distributed across groups.
      mn: The (m, n) output dimensions per group.
    """
    m, n = mn
    device = torch.device("tpu")
    group_sizes = self._ragged_k_group_sizes(num_tokens, num_groups, layout)
    offs = torch.tensor(group_sizes).cumsum(0).to(torch.int32)
    x = torch.randn(num_tokens, m, dtype=torch.bfloat16)
    dy = torch.randn(num_tokens, n, dtype=torch.bfloat16)

    with torch.tpu.precision(torch.tpu.Precision.DEFAULT):
      out = torch._grouped_mm(
          x.to(device).t(), dy.to(device), offs=offs.to(device)
      )

    self._assert_ragged_k_close(out, x.float().t(), dy.float(), group_sizes)

  @parameterized.product(
      num_tokens=[4096, 4104],
      num_groups=[8, 128],
      layout=["even", "uneven", "one_empty", "trailing_empty", "tail"],
      mn=[(256, 128), (128, 256)],
  )
  def test_scaled_grouped_mm_2d_2d_ragged_k_wgrad(
      self, num_tokens, num_groups, layout, mn
  ):
    """Tests _scaled_grouped_mm Case 3 in FP8 with tensorwise scales (#2668).

    Args:
      num_tokens: Total tokens T (the ragged contracting dimension).
      num_groups: Number of groups (experts).
      layout: How tokens are distributed across groups.
      mn: The (m, n) output dimensions per group.
    """
    m, n = mn
    device = torch.device("tpu")
    group_sizes = self._ragged_k_group_sizes(num_tokens, num_groups, layout)
    offs = torch.tensor(group_sizes).cumsum(0).to(torch.int32)
    fp8 = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8).max
    x = torch.randn(num_tokens, m)
    dy = torch.randn(num_tokens, n)
    scale_x = (x.abs().max() / fp8_max).reshape(1)
    scale_dy = (dy.abs().max() / fp8_max).reshape(1)
    x_fp8 = (x / scale_x).to(fp8)
    dy_fp8 = (dy / scale_dy).to(fp8)

    with torch.tpu.precision(torch.tpu.Precision.DEFAULT):
      out = torch._scaled_grouped_mm(
          x_fp8.to(device).t(),
          dy_fp8.to(device),
          scale_x.to(device),
          scale_dy.to(device),
          offs=offs.to(device),
          out_dtype=torch.bfloat16,
      )

    self._assert_ragged_k_close(
        out,
        (x_fp8.float() * scale_x).t(),
        dy_fp8.float() * scale_dy,
        group_sizes,
    )

  @parameterized.named_parameters(
      dict(testcase_name="aligned", num_tokens=4096, num_pad_rows=0),
      dict(testcase_name="padded", num_tokens=4104, num_pad_rows=248),
  )
  def test_grouped_mm_2d_2d_ragged_k_stablehlo(self, num_tokens, num_pad_rows):
    """Pins the Case 3 lowering to the form XLA:TPU maps to its Gmm kernel.

    The numeric ragged-K tests reproduce #2668 only on TPU v7x. This test
    checks the emitted StableHLO, so it guards the lowering on every chip: the
    ragged dimension must be dim 0 of both operands, and T must be zero-padded
    to a multiple of 256.

    Args:
      num_tokens: Total tokens T (the ragged contracting dimension).
      num_pad_rows: Expected zero rows appended to T.
    """
    m, n, num_groups = 256, 128, 8
    device = torch.device("tpu")
    group_sizes = self._ragged_k_group_sizes(num_tokens, num_groups, "even")
    offs = torch.tensor(group_sizes).cumsum(0).to(torch.int32).to(device)
    x = torch.randn(num_tokens, m, dtype=torch.bfloat16, device=device)
    dy = torch.randn(num_tokens, n, dtype=torch.bfloat16, device=device)

    torch.compiler.reset()
    debugs = []
    compiled = torch.compile(
        lambda x, dy, offs: torch._grouped_mm(x.t(), dy, offs=offs),
        backend="tpu",
        dynamic=False,
        options={"serializable": False, "debug_callback": debugs.append},
    )
    compiled(x, dy, offs)
    text = "\n".join(
        t for debug in debugs for t in debug.stablehlo_forward_text
    )

    self.assertIn("chlo.ragged_dot", text)
    for attr in (
        "lhs_contracting_dimensions = [0]",
        "rhs_contracting_dimensions = [0]",
        "lhs_ragged_dimensions = [0]",
    ):
      self.assertIn(attr, text)
    self.assertNotIn("rhs_group_dimensions", text)
    if num_pad_rows:
      self.assertIn("stablehlo.pad", text)
      self.assertIn(f"high = [{num_pad_rows}, 0]", text)
      self.assertIn(f"tensor<{num_tokens + num_pad_rows}x{m}xbf16>", text)
    else:
      self.assertNotIn("stablehlo.pad", text)

  def _apply_group_scales(
      self, tensor, scale_a, scale_b, offs=None, is_3d_2d=False
  ):
    """Applies per-group scale_a * scale_b to a grouped matmul result.

    Used by FP4 scaled_grouped_mm tests to construct the expected output.
    The golden is computed by dequantizing FP4→F32 on the CPU side and running
    _grouped_mm in F32, then scaling. The TPU path operates on raw FP4 tensors
    with internal dequantization. FP4's 2-bit mantissa introduces substantial
    quantization noise that compounds through dequantize→matmul→scale, so these
    tests use 20% tolerance (rtol=2e-1, atol=2e-1).

    Args:
      tensor: The unscaled matmul result tensor.
      scale_a: Scale factor for the first operand.
      scale_b: Scale factor for the second operand.
      offs: Optional cumulative group offsets for ragged dimensions.
      is_3d_2d: If True, scales along dim 1 instead of dim 0 for 2D results.

    Returns:
      A new tensor with per-group scaling applied.
    """
    out = tensor.clone()
    num_groups = offs.shape[0] if offs is not None else tensor.shape[0]
    sa = (
        scale_a.view(-1).expand(num_groups) if scale_a.numel() == 1 else scale_a
    )
    sb = (
        scale_b.view(-1).expand(num_groups) if scale_b.numel() == 1 else scale_b
    )
    for g in range(num_groups):
      factor = sa[g] * sb[g]
      if tensor.dim() == 2:
        start = 0 if g == 0 else offs[g - 1].item()
        end = offs[g].item()
        if is_3d_2d:
          out[:, start:end] *= factor
        else:
          out[start:end, :] *= factor
      else:
        out[g] *= factor
    return out

  @parameterized.product(
      out_dtype=[torch.bfloat16, torch.float32],
  )
  def test_scaled_grouped_mm_fp4_2d_3d(self, out_dtype):
    """Tests torch._scaled_grouped_mm: Case 1 (2D x 3D with offs) in FP4."""
    if "TPU_NAME" not in os.environ:
      self.skipTest("FP4 scaled_grouped_mm test requires TPU hardware.")

    device = "tpu"
    t1_float = torch.randn(16, 8, dtype=torch.float32).to(device)
    t2_float = torch.randn(2, 8, 8, dtype=torch.float32).to(device)
    t1_fp4 = t1_float.to(torch.float4_e2m1fn_x2)
    t2_fp4 = t2_float.to(torch.float4_e2m1fn_x2)
    offs = torch.tensor([8, 16], dtype=torch.int32).to(device)

    scale_a = torch.rand(1, dtype=torch.float32).to(device)
    scale_b = torch.rand(1, dtype=torch.float32).to(device)

    golden = torch.ops.aten._grouped_mm(
        t1_fp4.to(torch.float32), t2_fp4.to(torch.float32), offs=offs
    )
    expected = self._apply_group_scales(golden, scale_a, scale_b, offs)

    out = torch._scaled_grouped_mm(
        t1_fp4,
        t2_fp4,
        scale_a,
        scale_b,
        offs=offs,
        out_dtype=out_dtype,
    )

    utils.assert_close(
        out.to(torch.float32),
        expected.to(torch.float32),
        rtol=2e-1,
        atol=2e-1,
    )

  @parameterized.product(
      out_dtype=[torch.bfloat16, torch.float32],
  )
  def test_scaled_grouped_mm_fp4_3d_2d(self, out_dtype):
    """Tests torch._scaled_grouped_mm: Case 2 (3D x 2D with offs) in FP4."""
    if "TPU_NAME" not in os.environ:
      self.skipTest("FP4 scaled_grouped_mm test requires TPU hardware.")

    device = "tpu"
    t1_float = torch.randn(2, 8, 8, dtype=torch.float32).to(device)
    t2_float = torch.randn(8, 16, dtype=torch.float32).to(device)
    t1_fp4 = t1_float.to(torch.float4_e2m1fn_x2)
    t2_fp4 = t2_float.to(torch.float4_e2m1fn_x2)
    offs = torch.tensor([8, 16], dtype=torch.int32).to(device)

    scale_a = torch.rand(1, dtype=torch.float32).to(device)
    scale_b = torch.rand(1, dtype=torch.float32).to(device)

    golden = torch.ops.aten._grouped_mm(
        t1_fp4.to(torch.float32), t2_fp4.to(torch.float32), offs=offs
    )
    expected = self._apply_group_scales(
        golden, scale_a, scale_b, offs, is_3d_2d=True
    )

    out = torch._scaled_grouped_mm(
        t1_fp4,
        t2_fp4,
        scale_a,
        scale_b,
        offs=offs,
        out_dtype=out_dtype,
    )

    utils.assert_close(
        out.to(torch.float32),
        expected.to(torch.float32),
        rtol=2e-1,
        atol=2e-1,
    )

  @parameterized.product(
      out_dtype=[torch.bfloat16, torch.float32],
  )
  def test_scaled_grouped_mm_fp4_2d_2d(self, out_dtype):
    """Tests torch._scaled_grouped_mm: Case 3 (2D x 2D with offs) in FP4."""
    if "TPU_NAME" not in os.environ:
      self.skipTest("FP4 scaled_grouped_mm test requires TPU hardware.")

    device = "tpu"
    t1_float = torch.randn(16, 8, dtype=torch.float32).to(device)
    t2_float = torch.randn(8, 16, dtype=torch.float32).to(device)
    t1_fp4 = t1_float.to(torch.float4_e2m1fn_x2)
    t2_fp4 = t2_float.to(torch.float4_e2m1fn_x2)
    offs = torch.tensor([8, 16], dtype=torch.int32).to(device)

    scale_a = torch.rand(1, dtype=torch.float32).to(device)
    scale_b = torch.rand(1, dtype=torch.float32).to(device)

    golden = torch.ops.aten._grouped_mm(
        t1_fp4.to(torch.float32), t2_fp4.to(torch.float32), offs=offs
    )
    expected = self._apply_group_scales(golden, scale_a, scale_b, offs)

    out = torch._scaled_grouped_mm(
        t1_fp4,
        t2_fp4,
        scale_a,
        scale_b,
        offs=offs,
        out_dtype=out_dtype,
    )

    utils.assert_close(
        out.to(torch.float32),
        expected.to(torch.float32),
        rtol=2e-1,
        atol=2e-1,
    )

  @parameterized.product(
      out_dtype=[torch.bfloat16, torch.float32],
  )
  def test_scaled_grouped_mm_fp4_3d_3d(self, out_dtype):
    """Tests torch._scaled_grouped_mm: Case 4 (3D x 3D) in FP4."""
    if "TPU_NAME" not in os.environ:
      self.skipTest("FP4 scaled_grouped_mm test requires TPU hardware.")

    device = "tpu"
    t1_float = torch.randn(2, 8, 8, dtype=torch.float32).to(device)
    t2_float = torch.randn(2, 8, 8, dtype=torch.float32).to(device)
    t1_fp4 = t1_float.to(torch.float4_e2m1fn_x2)
    t2_fp4 = t2_float.to(torch.float4_e2m1fn_x2)

    scale_a = torch.rand(1, dtype=torch.float32).to(device)
    scale_b = torch.rand(1, dtype=torch.float32).to(device)

    golden = torch.ops.aten._grouped_mm(
        t1_fp4.to(torch.float32), t2_fp4.to(torch.float32)
    )
    expected = self._apply_group_scales(golden, scale_a, scale_b)

    out = torch._scaled_grouped_mm(
        t1_fp4,
        t2_fp4,
        scale_a,
        scale_b,
        out_dtype=out_dtype,
    )

    utils.assert_close(
        out.to(torch.float32),
        expected.to(torch.float32),
        rtol=2e-1,
        atol=2e-1,
    )

  @parameterized.product(
      out_dtype=[torch.bfloat16, torch.float32],
  )
  def test_scaled_grouped_mm_fp4_ragged(self, out_dtype):
    """Tests torch._scaled_grouped_mm with non-uniform group sizes."""
    if "TPU_NAME" not in os.environ:
      self.skipTest("FP4 scaled_grouped_mm test requires TPU hardware.")

    device = "tpu"
    t1_float = torch.randn(12, 8, dtype=torch.float32).to(device)
    t2_float = torch.randn(2, 8, 16, dtype=torch.float32).to(device)
    t1_fp4 = t1_float.to(torch.float4_e2m1fn_x2)
    t2_fp4 = t2_float.to(torch.float4_e2m1fn_x2)
    offs = torch.tensor([4, 12], dtype=torch.int32).to(device)

    scale_a = torch.rand(1, dtype=torch.float32).to(device)
    scale_b = torch.rand(1, dtype=torch.float32).to(device)

    golden = torch.ops.aten._grouped_mm(
        t1_fp4.to(torch.float32), t2_fp4.to(torch.float32), offs=offs
    )
    expected = self._apply_group_scales(golden, scale_a, scale_b, offs)

    out = torch._scaled_grouped_mm(
        t1_fp4,
        t2_fp4,
        scale_a,
        scale_b,
        offs=offs,
        out_dtype=out_dtype,
    )

    utils.assert_close(
        out.to(torch.float32),
        expected.to(torch.float32),
        rtol=2e-1,
        atol=2e-1,
    )

  def test_scaled_grouped_mm_use_fast_accum_matches_default(self):
    """Tests that use_fast_accum=True reuses the default's kernel and result."""
    device = "tpu"
    t1_fp8 = torch.randn(32, 16).to(torch.float8_e4m3fn).to(device)
    t2_fp8 = torch.randn(2, 16, 32).to(torch.float8_e4m3fn).to(device)
    offs = torch.tensor([16, 32], dtype=torch.int32).to(device)
    scale_a = torch.tensor(1.5, dtype=torch.float32).to(device)
    scale_b = torch.tensor(0.5, dtype=torch.float32).to(device)

    def run(use_fast_accum):
      return torch._scaled_grouped_mm(
          t1_fp8,
          t2_fp8,
          scale_a,
          scale_b,
          offs=offs,
          out_dtype=torch.bfloat16,
          use_fast_accum=use_fast_accum,
      ).cpu()

    expected = run(use_fast_accum=False)
    start_misses = torch.tpu._get_cache_misses()
    actual = run(use_fast_accum=True)
    # use_fast_accum doesn't change the lowering, so the kernel is reused.
    self.assertEqual(torch.tpu._get_cache_misses() - start_misses, 0)
    utils.assert_close(actual, expected, rtol=0, atol=0)

  @parameterized.product(
      layout=["2d_3d", "3d_2d", "2d_2d", "3d_3d"],
      variant=["default", "out", "out_resize", "no_swizzle", "contraction_dim"],
  )
  def test_scaled_grouped_mm_v2_tensorwise(self, layout, variant):
    """Tests torch.ops.aten._scaled_grouped_mm_v2 with FP8 TensorWise scales.

    v2 delegates to the v1 lowering, so its result must equal v1's exactly.

    Args:
      layout: Operand ranks, e.g. "2d_3d" is a 2D self (ragged via offs) and a
        3D mat2.
      variant: "out" and "out_resize" call the `.out` overload with a correctly
        sized or an empty `out`; "no_swizzle" passes empty swizzle lists (the
        torch.nn.functional.scaled_grouped_mm default); "contraction_dim" passes
        contraction_dim=[-1, -2].
    """
    if variant in ("out", "out_resize") and utils.parse_version(
        torch.__version__
    ) < (2, 14, 0):
      self.skipTest("_scaled_grouped_mm_v2.out requires PyTorch >= 2.14.")
    self_shape, mat2_shape, offs = {
        "2d_3d": ((16, 8), (2, 8, 8), [8, 16]),
        "3d_2d": ((2, 8, 8), (8, 16), [8, 16]),
        "2d_2d": ((8, 16), (16, 8), [8, 16]),
        "3d_3d": ((2, 8, 8), (2, 8, 8), None),
    }[layout]
    t1 = torch.randn(self_shape).to(torch.float8_e4m3fn)
    t2 = torch.randn(mat2_shape).to(torch.float8_e4m3fn)
    offs = torch.tensor(offs, dtype=torch.int32) if offs else None
    scale_a = torch.tensor([1.5])
    scale_b = torch.tensor([0.5])
    # at::blas::ScalingType::TensorWise and at::blas::SwizzleType::NO_SWIZZLE.
    tensorwise = 0
    swizzle = [] if variant == "no_swizzle" else [0]

    device = "tpu"
    t1_tpu, t2_tpu = t1.to(device), t2.to(device)
    sa_tpu, sb_tpu = scale_a.to(device), scale_b.to(device)
    offs_tpu = offs.to(device) if offs is not None else None
    v2_args = (
        t1_tpu,
        t2_tpu,
        [sa_tpu],
        [tensorwise],
        swizzle,
        [sb_tpu],
        [tensorwise],
        swizzle,
    )
    v2_kwargs = dict(offs=offs_tpu, out_dtype=torch.bfloat16)
    if variant == "contraction_dim":
      v2_kwargs["contraction_dim"] = [-1, -2]
    v1_out = torch._scaled_grouped_mm(
        t1_tpu, t2_tpu, sa_tpu, sb_tpu, offs=offs_tpu, out_dtype=torch.bfloat16
    )
    if variant in ("out", "out_resize"):
      out = (
          torch.empty_like(v1_out)
          if variant == "out"
          else torch.empty(0, dtype=torch.bfloat16, device=device)
      )
      v2_out = torch.ops.aten._scaled_grouped_mm_v2.out(
          *v2_args, **v2_kwargs, out=out
      )
      self.assertIs(v2_out, out)
    else:
      v2_out = torch.ops.aten._scaled_grouped_mm_v2(*v2_args, **v2_kwargs)

    self.assertEqual(v2_out.dtype, torch.bfloat16)
    self.assertTrue(torch.equal(v2_out.cpu(), v1_out.cpu()))
    golden = torch.ops.aten._grouped_mm(
        t1.to(torch.float32), t2.to(torch.float32), offs=offs
    ) * (scale_a * scale_b)
    utils.assert_close(
        v2_out.cpu().to(torch.float32), golden, rtol=2e-1, atol=2e-1
    )

  @parameterized.parameters(False, True)
  def test_scaled_grouped_mm_v2_use_fast_accum_matches_default(self, use_out):
    """Tests that use_fast_accum=True reuses the default's kernel and result."""
    if use_out and utils.parse_version(torch.__version__) < (2, 14, 0):
      self.skipTest("_scaled_grouped_mm_v2.out requires PyTorch >= 2.14.")
    device = "tpu"
    t1_fp8 = torch.randn(32, 16).to(torch.float8_e4m3fn).to(device)
    t2_fp8 = torch.randn(2, 16, 32).to(torch.float8_e4m3fn).to(device)
    offs = torch.tensor([16, 32], dtype=torch.int32).to(device)
    scale_a = torch.tensor([1.5]).to(device)
    scale_b = torch.tensor([0.5]).to(device)
    # at::blas::ScalingType::TensorWise and at::blas::SwizzleType::NO_SWIZZLE.
    args = (t1_fp8, t2_fp8, [scale_a], [0], [0], [scale_b], [0], [0])

    def run(use_fast_accum):
      kwargs = dict(
          offs=offs, out_dtype=torch.bfloat16, use_fast_accum=use_fast_accum
      )
      if use_out:
        out = torch.empty(32, 32, dtype=torch.bfloat16, device=device)
        torch.ops.aten._scaled_grouped_mm_v2.out(*args, **kwargs, out=out)
        return out.cpu()
      return torch.ops.aten._scaled_grouped_mm_v2(*args, **kwargs).cpu()

    expected = run(use_fast_accum=False)
    start_misses = torch.tpu._get_cache_misses()
    actual = run(use_fast_accum=True)
    # use_fast_accum doesn't change the lowering, so the kernel is reused.
    self.assertEqual(torch.tpu._get_cache_misses() - start_misses, 0)
    utils.assert_close(actual, expected, rtol=0, atol=0)

  def _make_rowwise_inputs(
      self,
      layout: str,
      group_sizes: list[int],
      ragged_size: int | None = None,
      m: int = 16,
      k: int = 32,
      n: int = 24,
      dtype_a: torch.dtype = torch.float8_e4m3fn,
  ) -> tuple[torch.Tensor, ...]:
    """Returns CPU (a, b, scale_a, scale_b, offs) for a rowwise test.

    Scales follow the upstream shapes (torch/_meta_registrations.py): a 2D
    operand takes a 1D scale over its non-contracting dim (times G for 2D x 2D)
    and a 3D operand a [G, non-contracting] scale.

    Args:
      layout: Operand ranks, e.g. "2d_3d" is a 2D self and a 3D mat2.
      group_sizes: Per-group sizes of the ragged dim: M for 2d_3d, N for 3d_2d
        and K for 2d_2d. 3d_3d only uses the group count.
      ragged_size: Size of the ragged dim. Defaults to sum(group_sizes); larger
        values leave a tail that belongs to no group.
      m: Rows of each group's output, if not ragged.
      k: Contracting size, if not ragged.
      n: Columns of each group's output, if not ragged.
      dtype_a: dtype of `a`; `b` is e4m3.
    """
    g = len(group_sizes)
    ragged = ragged_size or sum(group_sizes)
    a_shape, b_shape, sa_shape, sb_shape = {
        "2d_3d": ((ragged, k), (g, k, n), (ragged,), (g, n)),
        "3d_2d": ((g, m, k), (k, ragged), (g, m), (ragged,)),
        "2d_2d": ((m, ragged), (ragged, n), (g * m,), (g * n,)),
        "3d_3d": ((g, m, k), (g, k, n), (g, m), (g, n)),
    }[layout]
    a = torch.randn(a_shape).to(dtype_a)
    b = torch.randn(b_shape).to(torch.float8_e4m3fn)
    scale_a = torch.rand(sa_shape) + 0.5
    scale_b = torch.rand(sb_shape) + 0.5
    offs = torch.tensor(group_sizes).cumsum(0).to(torch.int32)
    return a, b, scale_a, scale_b, None if layout == "3d_3d" else offs

  def _rowwise_reference(
      self,
      a: torch.Tensor,
      b: torch.Tensor,
      scale_a: torch.Tensor,
      scale_b: torch.Tensor,
      offs: torch.Tensor | None,
  ) -> torch.Tensor:
    """Returns the CPU fp32 per-group reference for rowwise scales.

    Rows or columns past offs[-1] belong to no group and stay zero.

    Args:
      a: self, as made by _make_rowwise_inputs.
      b: mat2.
      scale_a: Rowwise scale of `a`.
      scale_b: Rowwise scale of `b`.
      offs: Cumulative group ends; None for 3d_3d.
    """
    a, b = a.float(), b.float()
    if offs is None:
      return a @ b * scale_a[:, :, None] * scale_b[:, None, :]
    ends = offs.tolist()
    bounds = list(zip([0] + ends[:-1], ends))
    if a.dim() == 2 and b.dim() == 3:
      out = torch.zeros(a.shape[0], b.shape[2])
      for g, (start, end) in enumerate(bounds):
        out[start:end] = (
            a[start:end] @ b[g] * scale_a[start:end, None] * scale_b[g]
        )
      return out
    if a.dim() == 3:
      out = torch.zeros(a.shape[1], b.shape[1])
      for g, (start, end) in enumerate(bounds):
        out[:, start:end] = (
            a[g] @ b[:, start:end] * scale_a[g][:, None] * scale_b[start:end]
        )
      return out
    scale_a = scale_a.view(len(ends), -1)
    scale_b = scale_b.view(len(ends), -1)
    out = torch.zeros(len(ends), a.shape[0], b.shape[1])
    for g, (start, end) in enumerate(bounds):
      out[g] = a[:, start:end] @ b[start:end] * scale_a[g][:, None] * scale_b[g]
    return out

  def _scaled_grouped_mm_tpu(
      self,
      a: torch.Tensor,
      b: torch.Tensor,
      scale_a: torch.Tensor,
      scale_b: torch.Tensor,
      offs: torch.Tensor | None,
      out_dtype: torch.dtype = torch.bfloat16,
  ) -> torch.Tensor:
    """Runs torch._scaled_grouped_mm on TPU and returns the CPU result."""
    device = "tpu"
    return torch._scaled_grouped_mm(
        a.to(device),
        b.to(device),
        scale_a.to(device),
        scale_b.to(device),
        offs=offs.to(device) if offs is not None else None,
        out_dtype=out_dtype,
    ).cpu()

  def _assert_rowwise_close(
      self, out: torch.Tensor, expected: torch.Tensor
  ) -> None:
    """Compares with the fp32 reference: 1e-2 for bf16 out, 1e-4 for fp32."""
    tol = 1e-2 if out.dtype == torch.bfloat16 else 1e-4
    utils.assert_close(
        out.float(),
        expected,
        rtol=tol,
        atol=tol * expected.abs().max().item(),
    )

  @parameterized.product(
      layout=["2d_3d", "3d_2d", "2d_2d", "3d_3d"],
      group_sizes=[[3, 0, 5, 8], [8, 8, 0, 0], [1, 0, 0, 0]],
      out_dtype=[torch.bfloat16, torch.float32],
  )
  def test_scaled_grouped_mm_rowwise(self, layout, group_sizes, out_dtype):
    """Tests rowwise FP8 scales against a CPU fp32 per-group reference.

    Args:
      layout: Operand ranks, e.g. "2d_3d" is a 2D self and a 3D mat2.
      group_sizes: Uneven with an empty group, trailing empty groups, or a
        single-token ragged dimension (M=1 / N=1 / K=1).
      out_dtype: The output dtype.
    """
    a, b, scale_a, scale_b, offs = self._make_rowwise_inputs(
        layout, group_sizes
    )

    out = self._scaled_grouped_mm_tpu(a, b, scale_a, scale_b, offs, out_dtype)

    self.assertEqual(out.dtype, out_dtype)
    self._assert_rowwise_close(
        out, self._rowwise_reference(a, b, scale_a, scale_b, offs)
    )

  @parameterized.parameters("2d_3d", "3d_2d", "2d_2d")
  def test_scaled_grouped_mm_rowwise_tail(self, layout):
    """Tests rows (2d_3d), columns (3d_2d), or contracting tokens (2d_2d) past offs[-1].

    For 2d_3d and 3d_2d, the lowering reads the last group's scale on tail
    tokens and masks the unpartitioned tail to zero. For 2d_2d, tokens past
    offs[-1] along the contracting dimension K belong to no group.

    Args:
      layout: Operand ranks.
    """
    a, b, scale_a, scale_b, offs = self._make_rowwise_inputs(
        layout, [3, 0, 5, 8], ragged_size=24
    )

    out = self._scaled_grouped_mm_tpu(a, b, scale_a, scale_b, offs)

    self._assert_rowwise_close(
        out, self._rowwise_reference(a, b, scale_a, scale_b, offs)
    )
    if layout != "2d_2d":
      tail = out[16:] if layout == "2d_3d" else out[:, 16:]
      self.assertTrue(torch.equal(tail, torch.zeros_like(tail)))

  def test_scaled_grouped_mm_rowwise_many_empty_groups(self):
    """Tests 2d_3d with 128 groups, 120 of them empty."""
    group_sizes = [0] * 128
    gen = torch.Generator().manual_seed(0)
    for g in torch.randperm(128, generator=gen)[:8].tolist():
      group_sizes[g] = 8
    a, b, scale_a, scale_b, offs = self._make_rowwise_inputs(
        "2d_3d", group_sizes
    )

    out = self._scaled_grouped_mm_tpu(a, b, scale_a, scale_b, offs)

    self._assert_rowwise_close(
        out, self._rowwise_reference(a, b, scale_a, scale_b, offs)
    )

  def test_scaled_grouped_mm_rowwise_e5m2(self):
    """Tests an e5m2 self (e.g. a gradient) with an e4m3 mat2."""
    a, b, scale_a, scale_b, offs = self._make_rowwise_inputs(
        "2d_3d", [3, 0, 5, 8], dtype_a=torch.float8_e5m2
    )

    out = self._scaled_grouped_mm_tpu(a, b, scale_a, scale_b, offs)

    self._assert_rowwise_close(
        out, self._rowwise_reference(a, b, scale_a, scale_b, offs)
    )

  @parameterized.product(
      matmul=["fwd", "dgrad", "wgrad"], dims=[(2048, 768), (768, 2048)]
  )
  def test_scaled_grouped_mm_rowwise_qwen3_shapes(self, matmul, dims):
    """Tests the torchao MoE matmuls at Qwen3-30B-A3B expert sizes.

    Args:
      matmul: "fwd" is x @ w, "dgrad" dy @ w.t() and "wgrad" x.t() @ dy, with
        512 tokens split unevenly over 8 experts.
      dims: The expert's (in, out) feature sizes.
    """
    d_in, d_out = dims
    layout, sizes = {
        "fwd": ("2d_3d", dict(k=d_in, n=d_out)),
        "dgrad": ("2d_3d", dict(k=d_out, n=d_in)),
        "wgrad": ("2d_2d", dict(m=d_in, n=d_out)),
    }[matmul]
    group_sizes = self._ragged_k_group_sizes(512, 8, "uneven")
    a, b, scale_a, scale_b, offs = self._make_rowwise_inputs(
        layout, group_sizes, **sizes
    )

    out = self._scaled_grouped_mm_tpu(a, b, scale_a, scale_b, offs)

    self._assert_rowwise_close(
        out, self._rowwise_reference(a, b, scale_a, scale_b, offs)
    )

  @parameterized.parameters("2d_3d", "3d_2d", "2d_2d", "3d_3d")
  def test_scaled_grouped_mm_v2_rowwise(self, layout):
    """Tests the v2 RowWise recipe. v2 delegates to v1, so results are equal.

    Args:
      layout: Operand ranks.
    """
    a, b, scale_a, scale_b, offs = self._make_rowwise_inputs(
        layout, [3, 0, 5, 8]
    )
    device = "tpu"
    a, b = a.to(device), b.to(device)
    scale_a, scale_b = scale_a.to(device), scale_b.to(device)
    offs = offs.to(device) if offs is not None else None
    rowwise = 1  # at::blas::ScalingType::RowWise.

    v1_out = torch._scaled_grouped_mm(
        a, b, scale_a, scale_b, offs=offs, out_dtype=torch.bfloat16
    )
    v2_out = torch.ops.aten._scaled_grouped_mm_v2(
        a,
        b,
        [scale_a],
        [rowwise],
        [],
        [scale_b],
        [rowwise],
        [],
        offs=offs,
        out_dtype=torch.bfloat16,
    )

    self.assertTrue(torch.equal(v2_out.cpu(), v1_out.cpu()))
    if utils.parse_version(torch.__version__) >= (2, 14, 0):
      v2_out_buf = torch.empty(0, dtype=torch.bfloat16, device=device)
      torch.ops.aten._scaled_grouped_mm_v2.out(
          a,
          b,
          [scale_a],
          [rowwise],
          [],
          [scale_b],
          [rowwise],
          [],
          offs=offs,
          out_dtype=torch.bfloat16,
          out=v2_out_buf,
      )
      self.assertTrue(torch.equal(v2_out_buf.cpu(), v1_out.cpu()))

  @parameterized.product(
      layout=["2d_3d", "3d_2d", "2d_2d"],
      dtype_a=[torch.float8_e4m3fn, torch.float8_e5m2],
  )
  def test_scaled_grouped_mm_fp8_operands_stablehlo(self, layout, dtype_a):
    """Verifies that FP8 _scaled_grouped_mm emits chlo.ragged_dot with f8 operands and DEFAULT precision."""
    if layout == "2d_3d":
      a = torch.randn(16, 16, dtype=torch.bfloat16).to(dtype_a)
      b = (
          torch.randn(2, 16, 16, dtype=torch.bfloat16)
          .transpose(-1, -2)
          .to(torch.float8_e4m3fn)
      )
      scale_a = torch.ones(16, dtype=torch.float32)
      scale_b = torch.ones(2, 16, dtype=torch.float32)
    elif layout == "3d_2d":
      a = torch.randn(2, 8, 16, dtype=torch.bfloat16).to(dtype_a)
      b = torch.randn(16, 16, dtype=torch.bfloat16).t().to(torch.float8_e4m3fn)
      scale_a = torch.ones(2, 8, dtype=torch.float32)
      scale_b = torch.ones(16, dtype=torch.float32)
    else:  # 2d_2d
      a = torch.randn(16, 16, dtype=torch.bfloat16).to(dtype_a)
      b = torch.randn(16, 16, dtype=torch.bfloat16).t().to(torch.float8_e4m3fn)
      scale_a = torch.ones(2 * 16, dtype=torch.float32)
      scale_b = torch.ones(2 * 16, dtype=torch.float32)
    offs = torch.tensor([8, 16], dtype=torch.int32)

    device = "tpu"
    a_tpu = a.to(device)
    b_tpu = b.to(device)
    offs_tpu = offs.to(device)
    sa_tpu = scale_a.to(device)
    sb_tpu = scale_b.to(device)

    debugs = []

    def fn(x, y, sa, sb, o):
      return torch._scaled_grouped_mm(
          x, y, sa, sb, offs=o, out_dtype=torch.bfloat16
      )

    compiled = torch.compile(
        fn,
        backend="tpu",
        options={"serializable": False, "debug_callback": debugs.append},
    )
    with torch.tpu.precision(torch.tpu.Precision.HIGHEST):
      _ = compiled(a_tpu, b_tpu, sa_tpu, sb_tpu, offs_tpu)

    self.assertNotEmpty(debugs)
    shlo = debugs[0].stablehlo_forward_text[0]
    self.assertIn("chlo.ragged_dot", shlo)
    self.assertNotIn("stablehlo.convert", shlo.split("chlo.ragged_dot")[0])
    self.assertIn("DEFAULT", shlo)
    self.assertNotIn("HIGHEST", shlo)
    f8_a = "f8E4M3FN" if dtype_a == torch.float8_e4m3fn else "f8E5M2"
    lhs_f8, rhs_f8 = (
        ("f8E4M3FN", f8_a) if layout == "3d_2d" else (f8_a, "f8E4M3FN")
    )
    self.assertRegex(
        shlo,
        rf"chlo\.ragged_dot.*tensor<[^>]*{lhs_f8}>.*tensor<[^>]*{rhs_f8}>",
    )

  @parameterized.parameters(
      "fwd_2d_3d",
      "fwd_2d_3d_contiguous_rhs",
      "fwd_2d_3d_in_graph_transpose",
      "wgrad_2d_2d",
      "wgrad_2d_2d_col_major_lhs",
  )
  def test_scaled_grouped_mm_torchao_patterns_compile(self, pattern):
    """Tests torchao.prototype.moe_training shapes under compile and StaticCompiler.

    Verifies torch._scaled_grouped_mm with exact shapes/strides used in torchao
    and torchtitan (including contiguous RHS from functionalize remove_views in
    scan layers, in-graph 3D transpose unwrapping, and col-major LHS in wgrad):
    - fwd/dgrad (2D x 3D): A e4m3 (M, K) row-major, B either (E, N,
      K).transpose(-2, -1) (col-major per expert), contiguous (E, K, N), or
      contiguous (E, N, K) transposed inside the compiled graph,
      sa fp32 [M], sb fp32 [E, N], offs int32 [E], out_dtype bf16,
      use_fast_accum=True.
    - wgrad (2D x 2D ragged-K): A [M, T] (row-major or (T, M).t() col-major),
      B [T, N] (col-major or row-major), sa fp32 [E * M], sb fp32 [E * N],
      offs int32 [E], out_dtype bf16, use_fast_accum=True.
    Checks:
    1. torch.compile(backend="tpu", fullgraph=True) has graph_count == 1 and
       graph_break_count == 0 via torch._dynamo.explain, matching CPU fp32
       reference.
    2. make_fx + StaticCompiler runs and matches eager result.
    """
    e = 8
    if pattern in (
        "fwd_2d_3d",
        "fwd_2d_3d_contiguous_rhs",
        "fwd_2d_3d_in_graph_transpose",
    ):
      m = 32
      k = 32
      n = 24
      group_sizes = [4, 0, 4, 8, 0, 4, 8, 4]
      offs = torch.tensor(group_sizes, dtype=torch.int32).cumsum(
          0, dtype=torch.int32
      )
      a = torch.randn(m, k).to(torch.float8_e4m3fn)
      if pattern == "fwd_2d_3d":
        b = torch.randn(e, n, k).transpose(-2, -1).to(torch.float8_e4m3fn)
        self.assertFalse(b.is_contiguous())
        b_ref = b
      elif pattern == "fwd_2d_3d_contiguous_rhs":
        b = torch.randn(e, k, n).to(torch.float8_e4m3fn).contiguous()
        self.assertTrue(b.is_contiguous())
        b_ref = b
      else:
        b = torch.randn(e, n, k).to(torch.float8_e4m3fn).contiguous()
        self.assertTrue(b.is_contiguous())
        b_ref = b.transpose(-2, -1)
      scale_a = (torch.rand(m, dtype=torch.float32) + 0.5).contiguous()
      scale_b = (torch.rand(e, n, dtype=torch.float32) + 0.5).contiguous()
    else:  # wgrad_2d_2d or wgrad_2d_2d_col_major_lhs
      t = 32
      m = 24
      n = 16
      group_sizes = [4, 0, 4, 8, 0, 4, 8, 4]
      offs = torch.tensor(group_sizes, dtype=torch.int32).cumsum(
          0, dtype=torch.int32
      )
      if pattern == "wgrad_2d_2d":
        a = torch.randn(m, t).to(torch.float8_e4m3fn).contiguous()
        b = torch.randn(n, t).to(torch.float8_e4m3fn).t()
        self.assertFalse(b.is_contiguous())
      else:
        a = torch.randn(t, m).to(torch.float8_e4m3fn).t()
        b = torch.randn(t, n).to(torch.float8_e4m3fn).contiguous()
        self.assertFalse(a.is_contiguous())
        self.assertTrue(b.is_contiguous())
      b_ref = b
      scale_a = (torch.rand(e * m, dtype=torch.float32) + 0.5).contiguous()
      scale_b = (torch.rand(e * n, dtype=torch.float32) + 0.5).contiguous()

    device = "tpu"
    a_tpu = a.to(device)
    b_tpu = b.to(device)
    sa_tpu = scale_a.to(device)
    sb_tpu = scale_b.to(device)
    offs_tpu = offs.to(device)

    def op(x, y, sa, sb, o):
      y_mm = (
          y.transpose(-2, -1)
          if pattern == "fwd_2d_3d_in_graph_transpose"
          else y
      )
      return torch._scaled_grouped_mm(
          x,
          y_mm,
          sa,
          sb,
          offs=o,
          out_dtype=torch.bfloat16,
          use_fast_accum=True,
      )

    # 1. Verify Dynamo explain: 1 graph, 0 breaks
    explanation = torch._dynamo.explain(op)(
        a_tpu, b_tpu, sa_tpu, sb_tpu, offs_tpu
    )
    self.assertEqual(explanation.graph_count, 1)
    self.assertEqual(explanation.graph_break_count, 0)

    # 2. Run under torch.compile(backend="tpu", fullgraph=True)
    compiled_op = torch.compile(op, backend="tpu", fullgraph=True)
    compile_out = compiled_op(a_tpu, b_tpu, sa_tpu, sb_tpu, offs_tpu).cpu()

    # Compare with reference
    expected = self._rowwise_reference(a, b_ref, scale_a, scale_b, offs)
    self._assert_rowwise_close(compile_out, expected)

    # 3. Run under make_fx + StaticCompiler
    fx_graph = make_fx(op)(a_tpu, b_tpu, sa_tpu, sb_tpu, offs_tpu)
    static_compiler = compile_compiler.StaticCompiler()
    executable = static_compiler(
        fx_graph, [a_tpu, b_tpu, sa_tpu, sb_tpu, offs_tpu]
    )
    static_out = executable(a_tpu, b_tpu, sa_tpu, sb_tpu, offs_tpu).cpu()
    self._assert_rowwise_close(static_out, expected)

  def test_aten_scaled_grouped_mm_meta_override(self):
    """Tests TPU Meta override for aten._scaled_grouped_mm (symbolic, scan/functionalize, CPU fallback)."""
    # 1. Non-TPU device keeps upstream meta (rejects contiguous mat2 / int64 offs),
    # while TPU device accepts both.
    with fake_tensor.FakeTensorMode():
      a_cpu = torch.empty((16, 16), dtype=torch.float8_e4m3fn, device="cpu")
      b_contig_cpu = torch.empty(
          (2, 16, 16), dtype=torch.float8_e4m3fn, device="cpu"
      )
      sa_cpu = torch.empty((16,), dtype=torch.float32, device="cpu")
      sb_cpu = torch.empty((2, 16), dtype=torch.float32, device="cpu")
      offs_i64_cpu = torch.empty((2,), dtype=torch.int64, device="cpu")
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Upstream meta fallback.
          RuntimeError, "column major"
      ):
        torch._scaled_grouped_mm(
            a_cpu,
            b_contig_cpu,
            sa_cpu,
            sb_cpu,
            offs=offs_i64_cpu.int(),
            out_dtype=torch.bfloat16,
        )
      out_tpu = torch._scaled_grouped_mm(
          a_cpu.to("tpu"),
          b_contig_cpu.to("tpu"),
          sa_cpu.to("tpu"),
          sb_cpu.to("tpu"),
          offs=offs_i64_cpu.to("tpu"),
          out_dtype=torch.bfloat16,
      )
      self.assertEqual(out_tpu.shape, (16, 16))
      self.assertEqual(out_tpu.dtype, torch.bfloat16)

    # 2. Symbolic shapes produce 0 guards on M and G.
    shape_env = symbolic_shapes.ShapeEnv()
    with fake_tensor.FakeTensorMode(shape_env=shape_env):
      s_m = shape_env.create_symintnode(
          shape_env.create_symbol(
              16, source=torch._dynamo.source.LocalSource("m")
          ),
          hint=16,
      )
      s_g = shape_env.create_symintnode(
          shape_env.create_symbol(
              2, source=torch._dynamo.source.LocalSource("g")
          ),
          hint=2,
      )
      a_sym = torch.empty((s_m, 16), dtype=torch.float8_e4m3fn, device="tpu")
      b_sym = torch.empty((s_g, 16, 8), dtype=torch.float8_e4m3fn, device="tpu")
      sa_sym = torch.empty((s_m,), dtype=torch.float32, device="tpu")
      sb_sym = torch.empty((s_g, 8), dtype=torch.float32, device="tpu")
      offs_sym = torch.empty((s_g,), dtype=torch.int32, device="tpu")
      out_sym = torch._scaled_grouped_mm(
          a_sym, b_sym, sa_sym, sb_sym, offs=offs_sym, out_dtype=torch.bfloat16
      )
      self.assertIsInstance(out_sym.shape[0], torch.SymInt)
      self.assertEqual(out_sym.shape[1], 8)
      self.assertEqual(len(shape_env.guards), 0)

    # 3. Functionalize with remove="mutations_and_views" (used by higher_order.scan)
    # turns transpose(-2, -1) into a contiguous transpose_copy.
    a, b_t, sa, sb, offs = self._make_rowwise_inputs("2d_3d", [8, 8])
    b_storage = b_t.transpose(-2, -1).contiguous()
    device = "tpu"
    a_tpu = a.to(device)
    b_storage_tpu = b_storage.to(device)
    sa_tpu = sa.to(device)
    sb_tpu = sb.to(device)
    offs_tpu = offs.to(device)

    def scan_body_fn(x, w_storage, s_a, s_b, o):
      w = w_storage.transpose(-2, -1)
      return torch._scaled_grouped_mm(
          x, w, s_a, s_b, offs=o, out_dtype=torch.bfloat16, use_fast_accum=True
      )

    functionalized_fn = torch.func.functionalize(
        scan_body_fn, remove="mutations_and_views"
    )
    fx_graph = make_fx(functionalized_fn)(
        a_tpu, b_storage_tpu, sa_tpu, sb_tpu, offs_tpu
    )
    self.assertIn("transpose_copy", fx_graph.code)
    static_compiler = compile_compiler.StaticCompiler()
    executable = static_compiler(
        fx_graph, [a_tpu, b_storage_tpu, sa_tpu, sb_tpu, offs_tpu]
    )
    out = executable(a_tpu, b_storage_tpu, sa_tpu, sb_tpu, offs_tpu).cpu()
    expected = self._rowwise_reference(a, b_t, sa, sb, offs)
    self._assert_rowwise_close(out, expected)

  def test_optimization_barrier_meta_non_contiguous_eager(self):
    x = torch.empty((16, 2, 256, 256), device="meta")
    x1 = x[:, 0, :, :]

    inputs = [x1]

    results = torch.tpu.ops.optimization_barrier(inputs)

    self.assertLen(results, 1)
    self.assertEqual(results[0].device.type, "meta")
    self.assertEqual(results[0].shape, x1.shape)
    self.assertEqual(results[0].dtype, x1.dtype)
    self.assertEqual(results[0].stride(), x1.stride())

  def test_optimization_barrier_meta_non_contiguous_compiled(self):
    def fn(x):
      x1 = x[:, 0, :, :]
      return torch.tpu.ops.optimization_barrier([x1])

    x = torch.ones((16, 2, 256, 256), device="tpu")
    x1 = x[:, 0, :, :]
    results = torch.compile(fn)(x)

    self.assertLen(results, 1)
    self.assertEqual(results[0].device.type, "tpu")
    self.assertEqual(results[0].shape, x1.shape)
    self.assertEqual(results[0].dtype, x1.dtype)
    self.assertEqual(results[0].stride(), x1.stride())

  def test_optimization_barrier_multiple_inputs(self):
    x1 = torch.empty((8,), dtype=torch.float32, device="meta")
    x2 = torch.empty((4, 4), dtype=torch.int32, device="meta")
    x3 = torch.empty((2, 2, 2), dtype=torch.bool, device="meta")

    inputs = [x1, x2, x3]
    results = torch.tpu.ops.optimization_barrier(inputs)

    self.assertLen(results, 3)
    for i in range(3):
      self.assertEqual(results[i].shape, inputs[i].shape)
      self.assertEqual(results[i].dtype, inputs[i].dtype)
      self.assertEqual(results[i].stride(), inputs[i].stride())
      self.assertEqual(results[i].device.type, "meta")

  def test_optimization_barrier_scalar(self):
    x = torch.tensor(1.0, device="meta")
    results = torch.tpu.ops.optimization_barrier([x])
    self.assertEqual(results[0].shape, torch.Size([]))


class OpsGradUnitTest(TorchTpuVsCpuTestBase):
  """Tests for backward ops."""

  def _nll_loss_grad(self, device, reduction, use_weight):
    # Prepare the data.
    n = 20
    c = 10
    ignore_index = -1
    g = torch.Generator().manual_seed(0)
    data = torch.randn(n, c, generator=g)
    log_probs_data = torch.nn.functional.log_softmax(data, dim=1)
    target_data = torch.randint(-1, c, (n,), generator=g)
    weight_data = torch.ones(c) if use_weight else None

    # Move the data to the device.
    log_probs = log_probs_data.to(device).requires_grad_(True)
    target = target_data.to(device)
    weight = weight_data.to(device) if use_weight else None

    # Compute the loss and its gradient.
    loss = torch.nn.functional.nll_loss(
        log_probs,
        target,
        weight=weight,
        reduction=reduction,
        ignore_index=ignore_index,
    )
    if reduction == "none":
      loss.sum().backward()
    else:
      loss.backward()

    return log_probs.grad

  def test_nll_loss_grad_mean_no_weight(self):
    self.assert_close_tpu_vs_cpu(
        functools.partial(
            self._nll_loss_grad, reduction="mean", use_weight=False
        )
    )

  def test_nll_loss_grad_sum_no_weight(self):
    self.assert_close_tpu_vs_cpu(
        functools.partial(
            self._nll_loss_grad, reduction="sum", use_weight=False
        )
    )

  def test_nll_loss_grad_none_no_weight(self):
    self.assert_close_tpu_vs_cpu(
        functools.partial(
            self._nll_loss_grad, reduction="none", use_weight=False
        )
    )

  def test_nll_loss_grad_mean_with_weight(self):
    self.assert_close_tpu_vs_cpu(
        functools.partial(
            self._nll_loss_grad, reduction="mean", use_weight=True
        )
    )

  def test_dot_complex(self):
    def test_fn(device):
      x = torch.tensor(
          [
              2.0151 + 4.9530j,
              -7.6987 - 8.7858j,
              4.1643 - 1.4473j,
              7.8339 + 1.1005j,
              1.8672 - 8.6496j,
          ],
          device=device,
      )
      y = torch.tensor(
          [
              -0.3849 - 6.4954j,
              -4.3033 - 6.3564j,
              -1.7299 - 3.8675j,
              4.3967 - 2.4084j,
              4.1380 - 3.4876j,
          ],
          device=device,
      )
      z = torch.zeros(0, dtype=torch.complex64, device=device)
      torch.dot(x, y, out=z)
      return z

    def test_fn2(device):
      x = torch.tensor(
          [
              12.0151 + 4.9530j,
              -7.6987 - 8.7858j,
              4.1643 - 1.4473j,
              7.8339 + 1.1005j,
              1.8672 - 8.6496j,
          ],
          device=device,
      )
      y = torch.tensor(
          [
              -0.3849 - 6.4954j,
              -4.3033 - 6.3564j,
              -1.7299 - 3.8675j,
              4.3967 - 2.4084j,
              4.1380 - 3.4876j,
          ],
          device=device,
      )
      z = torch.zeros(0, dtype=torch.complex64, device=device)
      torch.dot(x, y, out=z)
      return z

    self.assertNotEqual(
        test_fn(torch.device("tpu")).cpu(),
        test_fn2(torch.device("tpu")).cpu(),
    )
    self.assert_close_tpu_vs_cpu(test_fn)
    self.assert_close_tpu_vs_cpu(test_fn2)

  def test_fill_complex(self):
    def test_fn(device):
      x = torch.zeros(1, dtype=torch.complex64, device=device)
      torch.Tensor.fill_(x, 2.0 + 3.0j)
      return x

    self.assert_close_tpu_vs_cpu(test_fn)

    def test_fn2(device):
      x = torch.zeros(1, dtype=torch.complex64, device=device)
      torch.Tensor.fill_(x, torch.tensor(2.0 + 3.0j, device=device))
      return x

    self.assert_close_tpu_vs_cpu(test_fn2)

  def test_complex_scalar_add(self):

    x_cpu = torch.randn(4, 4, dtype=torch.complex64, device="cpu")

    def test_fn(device):
      x_device = x_cpu.detach().clone().to(device)
      return x_device + (4.0 + 5.0j)

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_nll_loss_grad_sum_with_weight(self):
    self.assert_close_tpu_vs_cpu(
        functools.partial(self._nll_loss_grad, reduction="sum", use_weight=True)
    )

  def test_nll_loss_grad_none_with_weight(self):
    self.assert_close_tpu_vs_cpu(
        functools.partial(
            self._nll_loss_grad, reduction="none", use_weight=True
        )
    )

  def test_nll_loss_ignore_all_targets(self):
    def test_fn(device):
      n = 5
      c = 3
      ignore_index = 0
      log_probs = torch.randn(n, c, device=device).log_softmax(dim=1)
      # All targets are the ignore_index.
      target = torch.full((n,), ignore_index, dtype=torch.long, device=device)
      loss = torch.nn.functional.nll_loss(
          log_probs, target, reduction="mean", ignore_index=ignore_index
      )
      return loss

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_nll_loss_with_all_zero_weights(self):
    def test_fn(device):
      n = 5
      c = 3
      log_probs = torch.randn(n, c, device=device).log_softmax(dim=1)
      target = torch.randint(0, c, (n,), dtype=torch.long, device=device)
      # weight is all zeros, so total weight is 0.
      weight = torch.zeros(c, device=device)
      loss = torch.nn.functional.nll_loss(
          log_probs, target, weight=weight, reduction="mean"
      )
      return loss

    self.assert_close_tpu_vs_cpu(test_fn)

  def _embedding_dense_backward(
      self, scale_grad_by_freq: bool, padding_idx: int
  ):
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
      with self.subTest(dtype=dtype):
        grad_output = torch.randn(4, 2, dtype=dtype)
        indices = torch.tensor([0, 1, 0, 2])
        num_weights = 5
        self.assert_close_tpu_vs_cpu(
            lambda device: torch.ops.aten.embedding_dense_backward(
                grad_output.to(device),
                indices.to(device),
                num_weights,
                padding_idx,
                scale_grad_by_freq,
            ),
        )

        grad_output_3d = torch.randn(2, 2, 2, dtype=dtype)
        indices_3d = torch.tensor([[0, 1], [0, 2]])
        self.assert_close_tpu_vs_cpu(
            lambda device: torch.ops.aten.embedding_dense_backward(
                grad_output_3d.to(device),
                indices_3d.to(device),
                num_weights,
                padding_idx,
                scale_grad_by_freq,
            ),
        )

  def test_signbit_float_zeros(self):
    t = torch.tensor([-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf")])
    t_tpu = t.to(torch.device("tpu"))
    expected = torch.tensor([True, True, True, False, False, False])
    self.assertEqual(torch.signbit(t), expected)
    self.assertEqual(torch.signbit(t_tpu).cpu(), expected)

  def test_embedding_dense_backward_scale_grad_true_padding_idx_neg_1(self):
    self._embedding_dense_backward(scale_grad_by_freq=True, padding_idx=-1)

  def test_embedding_dense_backward_scale_grad_true_padding_idx_0(self):
    self._embedding_dense_backward(scale_grad_by_freq=True, padding_idx=0)

  def test_embedding_dense_backward_scale_grad_true_padding_idx_1(self):
    self._embedding_dense_backward(scale_grad_by_freq=True, padding_idx=1)

  def test_embedding_dense_backward_scale_grad_true_padding_idx_2(self):
    self._embedding_dense_backward(scale_grad_by_freq=True, padding_idx=2)

  def test_embedding_dense_backward_scale_grad_true_padding_idx_3(self):
    self._embedding_dense_backward(scale_grad_by_freq=True, padding_idx=3)

  def test_embedding_dense_backward_scale_grad_false_padding_idx_neg_1(self):
    self._embedding_dense_backward(scale_grad_by_freq=False, padding_idx=-1)

  def test_embedding_dense_backward_scale_grad_false_padding_idx_0(self):
    self._embedding_dense_backward(scale_grad_by_freq=False, padding_idx=0)

  def test_embedding_dense_backward_scale_grad_false_padding_idx_1(self):
    self._embedding_dense_backward(scale_grad_by_freq=False, padding_idx=1)

  def test_embedding_dense_backward_scale_grad_false_padding_idx_2(self):
    self._embedding_dense_backward(scale_grad_by_freq=False, padding_idx=2)

  def test_embedding_dense_backward_scale_grad_false_padding_idx_3(self):
    self._embedding_dense_backward(scale_grad_by_freq=False, padding_idx=3)

  def test_addmm_zero_beta_ignores_nan_input(self):
    """torch.addmm() with zero beta should ignore NaN input."""
    nan_input_ = torch.ones(2, 2, dtype=torch.float32) * torch.nan
    mat1 = torch.ones(2, 2, dtype=torch.float32)
    mat2 = torch.ones(2, 2, dtype=torch.float32)
    beta = 0.0
    alpha = 0.0

    # Internal check.
    assert torch.all(torch.isnan(nan_input_))

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.addmm(
            nan_input_.to(device),
            mat1.to(device),
            mat2.to(device),
            beta=beta,
            alpha=alpha,
        )
    )

  def test_addmm_zero_beta_ignores_inf_input(self):
    """torch.addmm() with zero beta should ignore Inf input."""
    inf_input_ = torch.ones(2, 2, dtype=torch.float32) * torch.inf
    mat1 = torch.ones(2, 2, dtype=torch.float32)
    mat2 = torch.ones(2, 2, dtype=torch.float32)
    beta = 0.0
    alpha = 0.0

    # Internal check.
    assert torch.all(torch.isinf(inf_input_))

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.addmm(
            inf_input_.to(device),
            mat1.to(device),
            mat2.to(device),
            beta=beta,
            alpha=alpha,
        )
    )

  def test_linspace_steps_zero(self):
    def test_fn(device):
      return torch.linspace(0, 10, steps=0, device=device)

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_linspace_steps_one(self):
    def test_fn(device):
      return torch.linspace(0, 10, steps=1, device=device)

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_linalg_vector_norm_float_64(self):

    def fn(device, norm, dim=None, keepdim=False):
      x = torch.tensor(
          [[-4.0, -3.0, -2.0], [-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]],
          device=device,
          dtype=torch.float64,
      )
      return torch.linalg.vector_norm(x, ord=norm, dim=dim, keepdim=keepdim)

    assert_with_tol = functools.partial(
        self.assert_close_tpu_vs_cpu, rtol=1e-7, atol=1e-6
    )
    assert_with_tol(functools.partial(fn, norm=torch.inf))
    assert_with_tol(functools.partial(fn, norm=-torch.inf))
    assert_with_tol(functools.partial(fn, norm=0))
    assert_with_tol(functools.partial(fn, norm=1))
    assert_with_tol(functools.partial(fn, norm=2))
    assert_with_tol(functools.partial(fn, norm=3))
    assert_with_tol(functools.partial(fn, norm=1.0))
    assert_with_tol(functools.partial(fn, norm=2.0))
    assert_with_tol(functools.partial(fn, norm=3.0))
    assert_with_tol(functools.partial(fn, norm=3.456))
    assert_with_tol(functools.partial(fn, norm=3.0, keepdim=True))
    assert_with_tol(functools.partial(fn, norm=2.0, dim=1, keepdim=True))

  def test_linalg_vector_norm_empty_tensor_nan_ord(self):
    def fn_1d(device):
      x = torch.empty((0,), device=device)
      return torch.linalg.vector_norm(x, ord=float("nan"))

    def fn_1d_keepdim(device):
      x = torch.empty((0,), device=device)
      return torch.linalg.vector_norm(x, ord=float("nan"), keepdim=True)

    def fn_3d(device):
      x = torch.empty((2, 0, 3), device=device)
      return torch.linalg.vector_norm(x, ord=float("nan"), dim=1)

    def fn_3d_keepdim(device):
      x = torch.empty((2, 0, 3), device=device)
      return torch.linalg.vector_norm(x, ord=float("nan"), dim=1, keepdim=True)

    self.assert_close_tpu_vs_cpu(fn_1d)
    self.assert_close_tpu_vs_cpu(fn_1d_keepdim)
    self.assert_close_tpu_vs_cpu(fn_3d)
    self.assert_close_tpu_vs_cpu(fn_3d_keepdim)

  def test_linalg_vector_norm_single_element_reduction(self):
    # Tests single element reduction fast path for ord != 0 (b/568044911).
    def fn_1d_large_pos(device):
      x = torch.tensor([5.0], device=device)
      return torch.linalg.vector_norm(x, ord=100.0)

    def fn_1d_large_neg(device):
      x = torch.tensor([5.0], device=device)
      return torch.linalg.vector_norm(x, ord=-100.0)

    def fn_1d_neg_large_pos(device):
      x = torch.tensor([-5.0], device=device)
      return torch.linalg.vector_norm(x, ord=100.0)

    def fn_1d_neg_large_neg(device):
      x = torch.tensor([-5.0], device=device)
      return torch.linalg.vector_norm(x, ord=-100.0)

    def fn_0d_large_ord(device):
      x = torch.tensor(5.0, device=device)
      return torch.linalg.vector_norm(x, ord=100.0)

    def fn_0d_neg_large_ord(device):
      x = torch.tensor(-5.0, device=device)
      return torch.linalg.vector_norm(x, ord=100.0)

    def fn_2d_dim1(device):
      x = torch.tensor([[5.0], [3.0]], device=device)
      return torch.linalg.vector_norm(x, ord=100.0, dim=1)

    def fn_2d_dim1_keepdim(device):
      x = torch.tensor([[5.0], [3.0]], device=device)
      return torch.linalg.vector_norm(x, ord=100.0, dim=1, keepdim=True)

    def fn_2d_neg_dim1(device):
      x = torch.tensor([[-5.0], [3.0]], device=device)
      return torch.linalg.vector_norm(x, ord=100.0, dim=1)

    def fn_2d_neg_dim_neg1(device):
      x = torch.tensor([[-5.0], [3.0]], device=device)
      return torch.linalg.vector_norm(x, ord=100.0, dim=-1)

    def fn_2d_neg_dim1_keepdim(device):
      x = torch.tensor([[-5.0], [3.0]], device=device)
      return torch.linalg.vector_norm(x, ord=100.0, dim=1, keepdim=True)

    def fn_2d_multidim(device):
      x = torch.tensor([[-5.0]], device=device)
      return torch.linalg.vector_norm(x, ord=100.0, dim=(0, 1))

    def fn_2d_multidim_keepdim(device):
      x = torch.tensor([[-5.0]], device=device)
      return torch.linalg.vector_norm(x, ord=100.0, dim=(-2, -1), keepdim=True)

    def fn_complex64_large_components(device):
      x = torch.tensor([3e20 + 4e20j], dtype=torch.complex64, device=device)
      return torch.linalg.vector_norm(x, ord=2.0)

    def fn_1d_nan_ord(device):
      x = torch.tensor([5.0], device=device)
      return torch.linalg.vector_norm(x, ord=float("nan"))

    def fn_0d_nan_ord(device):
      x = torch.tensor(5.0, device=device)
      return torch.linalg.vector_norm(x, ord=float("nan"))

    self.assert_close_tpu_vs_cpu(fn_1d_large_pos)
    self.assert_close_tpu_vs_cpu(fn_1d_large_neg)
    self.assert_close_tpu_vs_cpu(fn_1d_neg_large_pos)
    self.assert_close_tpu_vs_cpu(fn_1d_neg_large_neg)
    self.assert_close_tpu_vs_cpu(fn_0d_large_ord)
    self.assert_close_tpu_vs_cpu(fn_0d_neg_large_ord)
    self.assert_close_tpu_vs_cpu(fn_2d_dim1)
    self.assert_close_tpu_vs_cpu(fn_2d_dim1_keepdim)
    self.assert_close_tpu_vs_cpu(fn_2d_neg_dim1)
    self.assert_close_tpu_vs_cpu(fn_2d_neg_dim_neg1)
    self.assert_close_tpu_vs_cpu(fn_2d_neg_dim1_keepdim)
    self.assert_close_tpu_vs_cpu(fn_2d_multidim)
    self.assert_close_tpu_vs_cpu(fn_2d_multidim_keepdim)
    self.assert_close_tpu_vs_cpu(fn_complex64_large_components)
    self.assert_close_tpu_vs_cpu(fn_1d_nan_ord)
    self.assert_close_tpu_vs_cpu(fn_0d_nan_ord)

  def test_norm_out(self):
    def test_fn(
        device, p=2.0, dim=None, keepdim=False, dtype=None, empty_out=False
    ):
      x = torch.tensor(
          [[-4.0, -3.0, -2.0], [-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]],
          device=device,
          dtype=torch.float32,
      )
      dim_list = (
          [] if dim is None else ([dim] if isinstance(dim, int) else list(dim))
      )
      if empty_out:
        out = torch.empty(0, device=device, dtype=dtype or torch.float32)
      else:
        out = torch.zeros(1, device=device, dtype=dtype or torch.float32)

      if dtype is not None:
        torch.ops.aten.norm.dtype_out(
            x, p, dim_list, keepdim, dtype=dtype, out=out
        )
      else:
        torch.ops.aten.norm.out(x, p, dim_list, keepdim, out=out)
      return out

    assert_with_tol = functools.partial(
        self.assert_close_tpu_vs_cpu, rtol=1e-5, atol=1e-5
    )
    # Default p (None / 2.0)
    assert_with_tol(functools.partial(test_fn, p=None))
    assert_with_tol(functools.partial(test_fn, p=None, dim=0))
    assert_with_tol(functools.partial(test_fn, p=None, dim=1, keepdim=True))
    assert_with_tol(functools.partial(test_fn, p=None, dim=(0, 1)))
    assert_with_tol(functools.partial(test_fn, p=None, empty_out=True))

    # Various p orders
    for p in [0, 1, 2, 3, 1.0, 2.0, 3.0, 1.5, float("inf"), float("-inf")]:
      assert_with_tol(functools.partial(test_fn, p=p))
      assert_with_tol(functools.partial(test_fn, p=p, dim=0))
      assert_with_tol(functools.partial(test_fn, p=p, dim=1, keepdim=True))

    # norm.dtype_out
    assert_with_tol(functools.partial(test_fn, p=2.0, dtype=torch.float32))
    assert_with_tol(
        functools.partial(test_fn, p=2.0, dim=0, dtype=torch.float64)
    )
    assert_with_tol(
        functools.partial(test_fn, p=2.0, dim=-1, dtype=torch.float32)
    )

    # Complex tensor norm
    def test_complex_fn(device, p=2.0, dim=0):
      x = torch.tensor(
          [[1.0 + 2.0j, -3.0 + 4.0j], [5.0 - 6.0j, 7.0 + 8.0j]],
          device=device,
          dtype=torch.complex64,
      )
      out = torch.empty(0, device=device, dtype=torch.float32)
      torch.ops.aten.norm.out(x, p, [dim], False, out=out)
      return out

    assert_with_tol(functools.partial(test_complex_fn, p=2.0, dim=0))
    assert_with_tol(functools.partial(test_complex_fn, p=1.0, dim=1))

    # 0-D scalar tensor
    def test_0d_fn(device, p=2.0):
      x = torch.tensor(3.5, device=device, dtype=torch.float32)
      out = torch.empty(0, device=device, dtype=torch.float32)
      torch.ops.aten.norm.out(x, p, [], False, out=out)
      return out

    assert_with_tol(functools.partial(test_0d_fn, p=2.0))
    assert_with_tol(functools.partial(test_0d_fn, p=None))

    # Zero-size empty tensor
    def test_empty_tensor_fn(device, p=2.0, dim=0):
      x = torch.empty((0, 4), device=device, dtype=torch.float32)
      out = torch.empty(0, device=device, dtype=torch.float32)
      torch.ops.aten.norm.out(x, p, [dim], False, out=out)
      return out

    assert_with_tol(functools.partial(test_empty_tensor_fn, p=2.0, dim=0))
    assert_with_tol(functools.partial(test_empty_tensor_fn, p=2.0, dim=1))

    # Non-contiguous tensor
    def test_noncontiguous_fn(device, p=2.0):
      x = torch.tensor(
          [[-4.0, -3.0, -2.0], [-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]],
          device=device,
          dtype=torch.float32,
      ).t()
      out = torch.empty(0, device=device, dtype=torch.float32)
      torch.ops.aten.norm.out(x, p, [0], False, out=out)
      return out

    assert_with_tol(functools.partial(test_noncontiguous_fn, p=2.0))
    assert_with_tol(functools.partial(test_noncontiguous_fn, p=1.0))

    # Float dtypes: float16, bfloat16, float64
    def test_dtypes_fn(device, in_dtype, p=2.0):
      x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=in_dtype)
      out = torch.empty(0, device=device, dtype=in_dtype)
      torch.ops.aten.norm.out(x, p, [0], False, out=out)
      return out

    assert_with_tol(functools.partial(test_dtypes_fn, in_dtype=torch.float16))
    assert_with_tol(functools.partial(test_dtypes_fn, in_dtype=torch.bfloat16))
    assert_with_tol(functools.partial(test_dtypes_fn, in_dtype=torch.float64))

  def test_layer_norm_backward(self):
    def fn(device):
      c, h, w = 2, 2, 4
      x = torch.tensor(
          [
              [
                  [
                      [-0.1117, -0.4966, 0.1631, -0.8817],
                      [0.0539, 0.6684, -0.0597, -0.4675],
                  ],
                  [
                      [-0.2153, 0.8840, -0.7584, -0.3689],
                      [-1.3424, -1.4020, 0.3206, -1.0219],
                  ],
              ],
              [
                  [
                      [0.7988, -0.0923, -0.7049, -1.6024],
                      [0.2891, 0.4899, -0.3853, -0.7120],
                  ],
                  [
                      [-0.1706, -1.4594, 0.2207, 0.2463],
                      [-1.3248, 0.6970, -0.6631, 1.2158],
                  ],
              ],
          ],
          device=device,
          requires_grad=True,
      )
      num_el = x.numel()
      g = (
          torch.arange(
              1, num_el + 1, dtype=torch.float32, device=device
          ).reshape_as(x)
          / num_el
      )
      layer_norm = torch.nn.LayerNorm([c, h, w], device=device)
      with torch.no_grad():
        layer_norm.weight.copy_(
            torch.linspace(0.5, 2.0, c * h * w, device=device).reshape(c, h, w)
        )
      out = layer_norm(x)
      out.backward(g)
      return x.grad, layer_norm.weight.grad, layer_norm.bias.grad

    assert_with_tol = functools.partial(
        self.assert_close_tpu_vs_cpu, rtol=1e-5, atol=1e-5
    )
    assert_with_tol(fn)

  # Regression test for b/503472873.
  def test_layer_norm_backward_no_affine(self):
    def fn(device):
      n, c, h, w = 2, 2, 2, 4
      num_elements = n * c * h * w

      x = (
          (
              torch.arange(num_elements, dtype=torch.float32, device=device)
              - num_elements / 2
          )
          .reshape(n, c, h, w)
          .requires_grad_()
      )
      g = (
          torch.arange(
              1, num_elements + 1, dtype=torch.float32, device=device
          ).reshape(n, c, h, w)
          * 0.1
      )

      layer_norm = torch.nn.LayerNorm(
          [c, h, w], elementwise_affine=False, device=device
      )

      out = layer_norm(x)
      out.backward(g)
      return x.grad

    self.assert_close_tpu_vs_cpu(fn, rtol=1e-5, atol=1e-5)

  def test_layer_norm_backward_comprehensive(self):
    """Tests layer_norm backward across dtypes, ranks, affine modes, and compile."""
    torch.manual_seed(42)
    configs = [
        ((8,), (8,), True, torch.float32, 1e-5, 1e-5),
        ((4, 16), (16,), True, torch.float32, 1e-5, 1e-5),
        ((2, 3, 12), (3, 12), False, torch.float32, 1e-5, 1e-5),
        ((2, 4, 32), (32,), True, torch.bfloat16, 2e-2, 2e-2),
        ((2, 4, 32), (32,), True, torch.float16, 5e-3, 5e-3),
        ((2, 0), (0,), False, torch.float32, 1e-5, 1e-5),
    ]
    for shape, norm_shape, affine, dtype, rtol, atol in configs:
      x_cpu = torch.randn(shape, dtype=dtype, requires_grad=True)
      dy_cpu = torch.randn(shape, dtype=dtype)
      w_cpu = (
          torch.randn(norm_shape, dtype=dtype, requires_grad=True)
          if affine
          else None
      )
      b_cpu = (
          torch.randn(norm_shape, dtype=dtype, requires_grad=True)
          if affine
          else None
      )

      def run_step(x_in, dy_in, w_in, b_in, ns=norm_shape):
        y = torch.nn.functional.layer_norm(x_in, ns, w_in, b_in, eps=1e-5)
        y.backward(dy_in)
        grads = [x_in.grad]
        if w_in is not None:
          grads.append(w_in.grad)
        if b_in is not None:
          grads.append(b_in.grad)
        return grads

      expected_grads = run_step(x_cpu, dy_cpu, w_cpu, b_cpu)

      x_tpu = x_cpu.detach().to("tpu").requires_grad_(True)
      dy_tpu = dy_cpu.to("tpu")
      w_tpu = (
          w_cpu.detach().to("tpu").requires_grad_(True)
          if w_cpu is not None
          else None
      )
      b_tpu = (
          b_cpu.detach().to("tpu").requires_grad_(True)
          if b_cpu is not None
          else None
      )
      actual_grads = run_step(x_tpu, dy_tpu, w_tpu, b_tpu)
      for actual, expected in zip(actual_grads, expected_grads):
        utils.assert_close(
            actual=actual.cpu(), expected=expected, rtol=rtol, atol=atol
        )

    # Also verify under torch.compile(backend="tpu")
    @torch.compile(backend="tpu")
    def compiled_ln_bwd(x_in, dy_in, w_in, b_in):
      y = torch.nn.functional.layer_norm(x_in, (16,), w_in, b_in, eps=1e-5)
      return torch.autograd.grad(
          y, (x_in, w_in, b_in), dy_in, create_graph=False
      )

    x_c = torch.randn((4, 16), dtype=torch.float32, requires_grad=True)
    dy_c = torch.randn((4, 16), dtype=torch.float32)
    w_c = torch.randn((16,), dtype=torch.float32, requires_grad=True)
    b_c = torch.randn((16,), dtype=torch.float32, requires_grad=True)
    y_ref = torch.nn.functional.layer_norm(x_c, (16,), w_c, b_c, eps=1e-5)
    ref_dx, ref_dw, ref_db = torch.autograd.grad(
        y_ref, (x_c, w_c, b_c), dy_c, create_graph=False
    )
    tpu_dx, tpu_dw, tpu_db = compiled_ln_bwd(
        x_c.detach().to("tpu").requires_grad_(True),
        dy_c.to("tpu"),
        w_c.detach().to("tpu").requires_grad_(True),
        b_c.detach().to("tpu").requires_grad_(True),
    )
    utils.assert_close(
        actual=tpu_dx.cpu(), expected=ref_dx, rtol=1e-5, atol=1e-5
    )
    utils.assert_close(
        actual=tpu_dw.cpu(), expected=ref_dw, rtol=1e-5, atol=1e-5
    )
    utils.assert_close(
        actual=tpu_db.cpu(), expected=ref_db, rtol=1e-5, atol=1e-5
    )

  def test_layer_norm_zero_variance(self):
    """Verifies that constant inputs (zero variance) do not produce NaN."""
    torch.manual_seed(42)
    x = torch.full((2, 3, 16), 5.0, dtype=torch.float32)
    weight = torch.randn(16, dtype=torch.float32)
    bias = torch.randn(16, dtype=torch.float32)

    def fn(device):
      return torch.ops.aten.native_layer_norm(
          x.to(device), [16], weight.to(device), bias.to(device), 1e-5
      )

    self.assert_close_tpu_vs_cpu(fn, rtol=1e-5, atol=1e-5)

  def test_layer_norm_large_mean(self):
    """Verifies numerical stability when mean is large relative to variance."""
    torch.manual_seed(42)
    x = 100.0 + torch.randn(2, 4, 32, dtype=torch.float32)

    def fn(device):
      return torch.ops.aten.native_layer_norm(
          x.to(device), [32], None, None, 1e-5
      )

    # Two-moment variance calculation (E[x^2] - E[x]^2) on TPU vs Welford on CPU
    # has expected floating point round-off differences when mean is large.
    self.assert_close_tpu_vs_cpu(fn, rtol=2e-3, atol=5e-3)

  def test_layer_norm_single_element_normalized_shape(self):
    """Verifies layer_norm when the normalized dimension size is 1."""
    torch.manual_seed(42)
    x = torch.randn(3, 4, 1, dtype=torch.float32)

    def fn(device):
      return torch.ops.aten.native_layer_norm(
          x.to(device), [1], None, None, 1e-5
      )

    self.assert_close_tpu_vs_cpu(fn, rtol=1e-5, atol=1e-5)

  def test_layer_norm_zero_sized_batch(self):
    """Verifies layer_norm with a zero-sized non-reduction dimension."""
    x = torch.empty(0, 16, dtype=torch.float32)

    def fn(device):
      return torch.ops.aten.native_layer_norm(
          x.to(device), [16], None, None, 1e-5
      )

    self.assert_close_tpu_vs_cpu(fn)

  def test_layer_norm_non_contiguous(self):
    """Verifies layer_norm with transposed non-contiguous inputs."""
    torch.manual_seed(42)
    x = torch.randn(2, 16, 8, dtype=torch.float32)

    def fn(device):
      x_transposed = x.to(device).transpose(1, 2)
      return torch.ops.aten.native_layer_norm(
          x_transposed, [16], None, None, 1e-5
      )

    self.assert_close_tpu_vs_cpu(fn, rtol=1e-5, atol=1e-5)

  @parameterized.parameters(
      (True, False),  # weight only
      (False, True),  # bias only
  )
  def test_layer_norm_partial_affine(self, has_weight, has_bias):
    """Verifies layer_norm with only weight or only bias provided."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 8, dtype=torch.float32)
    weight = torch.randn(8, dtype=torch.float32) if has_weight else None
    bias = torch.randn(8, dtype=torch.float32) if has_bias else None

    def fn(device):
      w = weight.to(device) if weight is not None else None
      b = bias.to(device) if bias is not None else None
      return torch.ops.aten.native_layer_norm(x.to(device), [8], w, b, 1e-5)

    self.assert_close_tpu_vs_cpu(fn, rtol=1e-5, atol=1e-5)

  def test_rms_norm_no_weight(self):
    """Tests nn.functional.rms_norm forward and backward without weight tensor."""

    def fn(device):
      x = torch.tensor(
          [[0.5, -1.2, 2.3, -0.8], [1.1, 0.4, -1.5, 0.9]],
          dtype=torch.float32,
          device=device,
      ).requires_grad_()
      g = torch.ones_like(x)
      out = torch.nn.functional.rms_norm(x, (4,), eps=1e-5)
      out.backward(g)
      return out, x.grad

    self.assert_close_tpu_vs_cpu(fn, rtol=1e-5, atol=1e-5)

  def test_rms_norm_multidim_normalized_shape(self):
    """Tests nn.functional.rms_norm with multi-dimensional normalized_shape."""

    def fn(device):
      x = (
          torch.arange(24, dtype=torch.float32, device=device).reshape(3, 2, 4)
          / 10.0
      ).requires_grad_()
      w = (
          torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4)
          / 5.0
      ).requires_grad_()
      g = torch.ones_like(x)
      out = torch.nn.functional.rms_norm(x, (2, 4), weight=w, eps=1e-5)
      out.backward(g)
      return out, x.grad, w.grad

    self.assert_close_tpu_vs_cpu(fn, rtol=1e-5, atol=1e-5)

  def test_rms_norm_1d_input(self):
    """Tests nn.functional.rms_norm when input rank equals normalized_shape rank."""

    def fn(device):
      x = (
          torch.arange(8, dtype=torch.float32, device=device) / 4.0
      ).requires_grad_()
      w = (
          torch.arange(8, dtype=torch.float32, device=device) / 8.0 + 1.0
      ).requires_grad_()
      g = torch.ones_like(x)
      out = torch.nn.functional.rms_norm(x, (8,), weight=w, eps=1e-5)
      out.backward(g)
      return out, x.grad, w.grad

    self.assert_close_tpu_vs_cpu(fn, rtol=1e-5, atol=1e-5)

  def test_rms_norm_zero_elements(self):
    """Tests nn.functional.rms_norm edge case with empty zero-element input."""

    def fn(device):
      x = torch.empty(0, 8, dtype=torch.float32, device=device).requires_grad_()
      w = torch.ones(8, dtype=torch.float32, device=device).requires_grad_()
      g = torch.empty(0, 8, dtype=torch.float32, device=device)
      out = torch.nn.functional.rms_norm(x, (8,), weight=w, eps=1e-5)
      out.backward(g)
      return out, x.grad, w.grad

    self.assert_close_tpu_vs_cpu(fn)

  def test_rms_norm_non_contiguous(self):
    """Tests nn.functional.rms_norm with non-contiguous strided inputs."""

    def fn(device):
      x_base = (
          torch.arange(32, dtype=torch.float32, device=device).reshape(4, 8)
          / 10.0
      ).requires_grad_()
      w_base = (
          torch.arange(8, dtype=torch.float32, device=device) / 4.0
      ).requires_grad_()
      x = x_base[:, ::2]
      w = w_base[::2]
      g = torch.ones_like(x)
      out = torch.nn.functional.rms_norm(x, (4,), weight=w, eps=1e-5)
      out.backward(g)
      return out, x_base.grad, w_base.grad

    self.assert_close_tpu_vs_cpu(fn, rtol=1e-5, atol=1e-5)

  def test_rms_norm_bfloat16(self):
    """Tests nn.functional.rms_norm mixed-precision and upcasting with bfloat16."""

    def fn(device):
      x = (
          torch.arange(32, dtype=torch.bfloat16, device=device).reshape(4, 8)
          / 10.0
      ).requires_grad_()
      w = (
          torch.arange(8, dtype=torch.bfloat16, device=device) / 4.0 + 0.5
      ).requires_grad_()
      g = torch.ones_like(x)
      out = torch.nn.functional.rms_norm(x, (8,), weight=w, eps=1e-5)
      out.backward(g)
      return out, x.grad, w.grad

    self.assert_close_tpu_vs_cpu(fn, rtol=1e-2, atol=1e-2)

  def test_rms_norm_grad_input_only(self):
    """Tests autograd backward when only input gradient is required."""

    def fn(device):
      x = (
          torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
          / 5.0
      ).requires_grad_()
      w = (
          torch.arange(4, dtype=torch.float32, device=device) / 2.0
      ).requires_grad_(False)
      g = torch.ones_like(x)
      out = torch.nn.functional.rms_norm(x, (4,), weight=w, eps=1e-5)
      out.backward(g)
      self.assertIsNone(w.grad)
      return out, x.grad

    self.assert_close_tpu_vs_cpu(fn, rtol=1e-5, atol=1e-5)

  def test_rms_norm_grad_weight_only(self):
    """Tests autograd backward when only weight gradient is required."""

    def fn(device):
      x = (
          torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
          / 5.0
      ).requires_grad_(False)
      w = (
          torch.arange(4, dtype=torch.float32, device=device) / 2.0
      ).requires_grad_()
      g = torch.ones_like(x)
      out = torch.nn.functional.rms_norm(x, (4,), weight=w, eps=1e-5)
      out.backward(g)
      self.assertIsNone(x.grad)
      return out, w.grad

    self.assert_close_tpu_vs_cpu(fn, rtol=1e-5, atol=1e-5)

  def test_rms_norm_inference_no_grad_and_default_eps(self):
    """Tests nn.functional.rms_norm 1-output inference path and default eps=None."""
    for dtype, tol in [
        (torch.float32, 1e-5),
        (torch.bfloat16, 1e-2),
        (torch.float16, 1e-2),
        (torch.float64, 1e-6),
    ]:
      for eps in [None, 1e-6]:

        def fn_no_grad(device, dtype=dtype, eps=eps):
          with torch.no_grad():
            x = (
                torch.arange(24, dtype=dtype, device=device).reshape(2, 3, 4)
                / 10.0
                + 0.1
            )
            w = torch.arange(4, dtype=dtype, device=device) / 4.0 + 0.5
            out_with_w = torch.nn.functional.rms_norm(
                x, (4,), weight=w, eps=eps
            )
            out_no_w = torch.nn.functional.rms_norm(
                x, (4,), weight=None, eps=eps
            )
            self.assertEqual(out_with_w.dtype, dtype)
            self.assertEqual(out_no_w.dtype, dtype)
            return out_with_w, out_no_w

        self.assert_close_tpu_vs_cpu(fn_no_grad, rtol=tol, atol=tol)

        def fn_requires_grad_false(device, dtype=dtype, eps=eps):
          x = (
              torch.arange(24, dtype=dtype, device=device).reshape(2, 3, 4)
              / 10.0
              + 0.1
          ).requires_grad_(False)
          w = (
              torch.arange(4, dtype=dtype, device=device) / 4.0 + 0.5
          ).requires_grad_(False)
          return torch.nn.functional.rms_norm(x, (4,), weight=w, eps=eps)

        self.assert_close_tpu_vs_cpu(fn_requires_grad_false, rtol=tol, atol=tol)

  def test_rms_norm_mixed_dtypes_forward_and_backward(self):
    """Tests nn.functional.rms_norm when input and weight have different dtypes."""
    dtype_pairs = [
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float16),
        (torch.float16, torch.float32),
    ]
    for input_dtype, weight_dtype in dtype_pairs:
      # 1. Eager inference (torch.no_grad())
      def fn_infer(device, in_dt=input_dtype, w_dt=weight_dtype):
        with torch.no_grad():
          x = (
              torch.arange(32, dtype=in_dt, device=device).reshape(4, 8) / 10.0
              + 0.2
          )
          w = torch.arange(8, dtype=w_dt, device=device) / 4.0 + 0.5
          out = torch.nn.functional.rms_norm(x, (8,), weight=w, eps=1e-5)
          self.assertEqual(out.dtype, in_dt)
          return out

      self.assert_close_tpu_vs_cpu(fn_infer, rtol=2e-2, atol=2e-2)

      # 2. Eager training (forward + backward)
      def fn_train(device, in_dt=input_dtype, w_dt=weight_dtype):
        x = (
            torch.arange(32, dtype=in_dt, device=device).reshape(4, 8) / 10.0
            + 0.2
        ).requires_grad_()
        w = (
            torch.arange(8, dtype=w_dt, device=device) / 4.0 + 0.5
        ).requires_grad_()
        g = (
            torch.arange(32, dtype=in_dt, device=device).reshape(4, 8) / 20.0
            - 0.3
        )
        out = torch.nn.functional.rms_norm(x, (8,), weight=w, eps=1e-5)
        out.backward(g)
        self.assertEqual(out.dtype, in_dt)
        self.assertEqual(x.grad.dtype, in_dt)
        self.assertEqual(w.grad.dtype, w_dt)
        return out, x.grad, w.grad

      self.assert_close_tpu_vs_cpu(fn_train, rtol=2e-2, atol=2e-2)

      # 3. 1D input (batch_dims.empty()) with mixed dtypes
      def fn_1d(device, in_dt=input_dtype, w_dt=weight_dtype):
        x = (
            torch.arange(8, dtype=in_dt, device=device) / 4.0 + 0.2
        ).requires_grad_()
        w = (
            torch.arange(8, dtype=w_dt, device=device) / 8.0 + 1.0
        ).requires_grad_()
        g = torch.ones_like(x)
        out = torch.nn.functional.rms_norm(x, (8,), weight=w, eps=1e-5)
        out.backward(g)
        self.assertEqual(out.dtype, in_dt)
        self.assertEqual(x.grad.dtype, in_dt)
        self.assertEqual(w.grad.dtype, w_dt)
        return out, x.grad, w.grad

      self.assert_close_tpu_vs_cpu(fn_1d, rtol=2e-2, atol=2e-2)

      # 4. torch.compile with mixed dtypes (forward + backward)
      def fn_compiled(device, in_dt=input_dtype, w_dt=weight_dtype):
        x = (
            torch.arange(32, dtype=in_dt, device=device).reshape(4, 8) / 10.0
            + 0.2
        ).requires_grad_()
        w = (
            torch.arange(8, dtype=w_dt, device=device) / 4.0 + 0.5
        ).requires_grad_()
        g = torch.ones_like(x)
        rms_fn = (
            torch.compile(torch.nn.functional.rms_norm, backend="tpu")
            if torch.device(device).type == "tpu"
            else torch.nn.functional.rms_norm
        )
        out = rms_fn(x, (8,), weight=w, eps=1e-5)
        out.backward(g)
        self.assertEqual(out.dtype, in_dt)
        self.assertEqual(x.grad.dtype, in_dt)
        self.assertEqual(w.grad.dtype, w_dt)
        return out, x.grad, w.grad

      self.assert_close_tpu_vs_cpu(fn_compiled, rtol=2e-2, atol=2e-2)

  def test_rms_norm_channels_last_and_transposed_views(self):
    """Tests nn.functional.rms_norm on 4D QK-norm transposed views and channels_last layouts."""

    # 1. 4D QK-norm transposed view (2, 16, 4, 8).transpose(1, 2) -> (2, 4, 16, 8)
    def fn_transposed_infer(device):
      with torch.no_grad():
        x_base = (
            torch.arange(1024, dtype=torch.float32, device=device).reshape(
                2, 16, 4, 8
            )
            / 100.0
            + 0.1
        )
        x = x_base.transpose(1, 2)
        w = torch.arange(8, dtype=torch.bfloat16, device=device) / 8.0 + 0.5
        return torch.nn.functional.rms_norm(x, (8,), weight=w, eps=1e-5)

    self.assert_close_tpu_vs_cpu(fn_transposed_infer, rtol=1e-2, atol=1e-2)

    def fn_transposed_train(device):
      x_base = (
          torch.arange(1024, dtype=torch.float32, device=device).reshape(
              2, 16, 4, 8
          )
          / 100.0
          + 0.1
      ).requires_grad_()
      x = x_base.transpose(1, 2)
      w = (
          torch.arange(8, dtype=torch.float32, device=device) / 8.0 + 0.5
      ).requires_grad_()
      g = torch.ones_like(x)
      out = torch.nn.functional.rms_norm(x, (8,), weight=w, eps=1e-5)
      out.backward(g)
      return out, x_base.grad, w.grad

    self.assert_close_tpu_vs_cpu(fn_transposed_train, rtol=1e-5, atol=1e-5)

    # 2. Explicit channels_last (4D) and channels_last_3d (5D) memory formats
    def fn_channels_last(device):
      x4 = (
          torch.arange(192, dtype=torch.float32, device=device)
          .reshape(2, 3, 4, 8)
          .to(memory_format=torch.channels_last)
          / 50.0
          + 0.1
      ).requires_grad_()
      w4 = (
          torch.arange(8, dtype=torch.float32, device=device) / 4.0 + 0.5
      ).requires_grad_()
      out4 = torch.nn.functional.rms_norm(x4, (8,), weight=w4, eps=1e-5)
      out4.backward(torch.ones_like(out4))

      x5 = (
          torch.arange(384, dtype=torch.float32, device=device)
          .reshape(2, 3, 2, 4, 8)
          .to(memory_format=torch.channels_last_3d)
          / 50.0
          + 0.1
      ).requires_grad_()
      w5 = (
          torch.arange(8, dtype=torch.float32, device=device) / 4.0 + 0.5
      ).requires_grad_()
      out5 = torch.nn.functional.rms_norm(x5, (8,), weight=w5, eps=1e-5)
      out5.backward(torch.ones_like(out5))
      return out4, x4.grad, w4.grad, out5, x5.grad, w5.grad

    self.assert_close_tpu_vs_cpu(fn_channels_last, rtol=1e-5, atol=1e-5)

  def test_rms_norm_zero_elements_inference_and_mixed_dtype(self):
    """Tests nn.functional.rms_norm with empty (0, 8) input in inference and mixed-dtype training."""

    def fn_infer(device):
      with torch.no_grad():
        x = torch.empty(0, 8, dtype=torch.float32, device=device)
        w = torch.ones(8, dtype=torch.bfloat16, device=device)
        out = torch.nn.functional.rms_norm(x, (8,), weight=w, eps=1e-5)
        self.assertEqual(out.dtype, torch.float32)
        return out

    self.assert_close_tpu_vs_cpu(fn_infer)

    def fn_train(device):
      x = torch.empty(0, 8, dtype=torch.float32, device=device).requires_grad_()
      w = torch.ones(8, dtype=torch.bfloat16, device=device).requires_grad_()
      g = torch.empty(0, 8, dtype=torch.float32, device=device)
      out = torch.nn.functional.rms_norm(x, (8,), weight=w, eps=1e-5)
      out.backward(g)
      self.assertEqual(out.dtype, torch.float32)
      self.assertEqual(x.grad.dtype, torch.float32)
      self.assertEqual(w.grad.dtype, torch.bfloat16)
      return out, x.grad, w.grad

    self.assert_close_tpu_vs_cpu(fn_train)

  @parameterized.product(
      device=["tpu", "cpu"],  # CPU is the upstream control.
      requires_grad=[False, True],
      has_weight=[False, True],
  )
  def test_rms_norm_fake_symbolic_shapes(
      self, device, requires_grad, has_weight
  ):
    """Tests rms_norm on symbolic-shape FakeTensors doesn't throw or guard."""
    x = torch.randn(3, 5, 16, device=device, requires_grad=requires_grad)
    inputs = [(x, (True, True, False))]
    if has_weight:
      weight = torch.randn(16, device=device, requires_grad=requires_grad)
      inputs.append((weight, (False,)))
      fn = lambda x_fake, w_fake: torch.rms_norm(x_fake, [16], w_fake, 1e-6)
    else:
      fn = lambda x_fake: torch.rms_norm(x_fake, [16], None, 1e-6)

    specialized = run_with_dynamic_dims(fn, inputs, no_grad=not requires_grad)
    self.assertEmpty(specialized)

  @parameterized.parameters("tpu", "cpu")  # CPU is the upstream control.
  def test_rms_norm_compile_dynamic_shapes(self, device):
    """Tests compiled rms_norm recompiles only once for dynamic batch/seq."""
    torch._dynamo.reset()
    self.addCleanup(torch._dynamo.reset)
    counter = dynamo_testing.CompileCounter()

    @torch.compile(backend=counter, fullgraph=True)
    def fn(x, w):
      return torch.nn.functional.rms_norm(x, (16,), w, 1e-6)

    torch.manual_seed(42)
    w = torch.randn(16)
    for n in (3, 5, 7):
      x = torch.randn(n, n + 2, 16)
      out = fn(x.to(device), w.to(device))
      utils.assert_close(
          out.cpu(),
          torch.nn.functional.rms_norm(x, (16,), w, 1e-6),
          rtol=1e-5,
          atol=1e-5,
      )
    # One static frame, then one dynamic frame that serves all later sizes.
    self.assertEqual(counter.frame_count, 2)

  @parameterized.product(
      device=["tpu", "cpu"],  # CPU is the upstream control.
      requires_grad=[False, True],
      bias_shape=[None, (7,), (5, 7), (1, 5, 7)],
  )
  def test_linear_fake_symbolic_shapes(self, device, requires_grad, bias_shape):
    """Tests linear on symbolic-shape FakeTensors doesn't specialize dims."""
    # Before PyTorch 2.13, upstream `at::native::matmul` called `numel()` on its
    # input, which throws on (or specializes) symbolic shapes. Upstream CPU
    # linear always uses it, and TPU linear falls back to it for a broadcast
    # bias that requires grad.
    uses_upstream_matmul = device == "cpu" or (
        requires_grad and bias_shape is not None and len(bias_shape) > 1
    )
    torch_version = utils.parse_version(torch.__version__)
    if uses_upstream_matmul and torch_version < (2, 13, 0):
      self.skipTest(
          "Upstream matmul needs PyTorch >= 2.13 for symbolic shapes."
      )
    # Batch, sequence and out_features are dynamic; in_features is static.
    x = torch.randn(3, 5, 16, device=device, requires_grad=requires_grad)
    weight = torch.randn(7, 16, device=device, requires_grad=requires_grad)
    inputs = [(x, (True, True, False)), (weight, (True, False))]
    if bias_shape is None:
      fn = lambda x_fake, w_fake: torch.nn.functional.linear(x_fake, w_fake)
    else:
      bias = torch.randn(bias_shape, device=device, requires_grad=requires_grad)
      inputs.append((bias, tuple(size != 1 for size in bias_shape)))
      fn = torch.nn.functional.linear

    specialized = run_with_dynamic_dims(fn, inputs, no_grad=not requires_grad)
    self.assertEmpty(specialized)

  @parameterized.product(
      device=["tpu", "cpu"],  # CPU is the upstream control.
      broadcast_bias=[False, True],
  )
  def test_linear_compile_dynamic_shapes(self, device, broadcast_bias):
    """Tests compiled linear recompiles only once for dynamic shapes."""
    torch._dynamo.reset()
    self.addCleanup(torch._dynamo.reset)
    counter = dynamo_testing.CompileCounter()

    @torch.compile(backend=counter, fullgraph=True)
    def fn(x, w, b):
      return torch.nn.functional.linear(x, w, b)

    torch.manual_seed(42)
    for n in (3, 5, 7):
      # Batch, sequence and out_features all change between calls.
      x = torch.randn(n, n + 2, 16)
      w = torch.randn(n + 1, 16)
      b = torch.randn((n + 2, n + 1) if broadcast_bias else (n + 1,))
      x_dev, w_dev, b_dev = (t.to(device).requires_grad_() for t in (x, w, b))
      # The compiled graph runs the same ops, so compare it to eager on the
      # same device.
      utils.assert_close(
          fn(x_dev, w_dev, b_dev),
          torch.nn.functional.linear(x_dev, w_dev, b_dev),
      )
    # One static frame, then one dynamic frame that serves all later sizes.
    self.assertEqual(counter.frame_count, 2)

  def test_max_pool2d_with_indices(self):
    """Tests nn.functional.max_pool2d_float32_sample54."""
    device = torch.device("tpu")
    maxpool_input = torch.tensor(
        [[
            [
                [-7.7435, -8.8254, 7.2097, 4.3371, 2.8040, -3.4491],
                [-8.5819, 3.9336, -6.2229, 1.1184, -6.0094, 7.3457],
                [-2.0333, 5.7398, 1.8601, 8.6590, 0.6541, 0.0145],
            ],
            [
                [2.8523, -5.7473, 2.1480, -0.3480, 2.5668, -8.3042],
                [-1.1508, -8.2351, 4.4935, -0.0096, -2.7059, -5.8874],
                [5.4567, 0.2254, -3.6194, -6.1967, 8.8962, -1.7928],
            ],
        ]],
        dtype=torch.float32,
        device=device,
    )

    expected_res = torch.tensor(
        [[
            [
                [4.3371, 7.2097, 7.3457, 7.2097],
                [8.6590, 1.8601, 8.6590, 1.8601],
            ],
            [
                [-0.0096, 4.4935, -0.0096, 4.4935],
                [0.2254, 8.8962, 0.2254, 8.8962],
            ],
        ]],
        dtype=torch.float32,
    )
    expected_indices = torch.tensor(
        [[[[3, 2, 11, 2], [15, 14, 15, 14]], [[9, 8, 9, 8], [13, 16, 13, 16]]]],
        dtype=torch.int64,
    )

    res, indices = torch.nn.functional.max_pool2d(
        maxpool_input,
        kernel_size=3,
        stride=(2, 1),
        padding=1,
        dilation=(1, 2),
        ceil_mode=True,
        return_indices=True,
    )
    self.assertEqual(res.to("cpu"), expected_res)
    self.assertEqual(indices.to("cpu"), expected_indices)

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nn.functional.max_pool2d(
            maxpool_input.to(device),
            kernel_size=3,
            stride=(2, 1),
            padding=1,
            dilation=(1, 2),
            ceil_mode=True,
            return_indices=True,
        )
    )

  def test_max_pool2d_with_indices_int(self):
    device = torch.device("tpu")
    maxpool_int8 = torch.tensor(
        [
            [
                [3, 8, 6, 2, 5, 3],
                [2, 2, 6, 1, 3, 4],
                [2, 1, 9, 1, 8, 2],
            ],
            [
                [1, 4, 0, 7, 8, 0],
                [4, 0, 5, 7, 9, 5],
                [8, 3, 7, 7, 3, 0],
            ],
        ],
        dtype=torch.int8,
        device=device,
    )

    maxpool_int16 = torch.tensor(
        [
            [
                [3, 8, 6, 2, 5, 3],
                [2, 2, 6, 1, 3, 4],
                [2, 1, 9, 1, 8, 2],
            ],
            [
                [1, 4, 0, 7, 8, 0],
                [4, 0, 5, 7, 9, 5],
                [8, 3, 7, 7, 3, 0],
            ],
        ],
        dtype=torch.int16,
        device=device,
    )

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nn.functional.max_pool2d(
            maxpool_int8.to(device),
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(0, 0),
            dilation=(1, 1),
            ceil_mode=True,
            return_indices=True,
        )
    )

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nn.functional.max_pool2d(
            maxpool_int16.to(device),
            kernel_size=(3, 3),
            padding=(0, 0),
            dilation=(1, 2),
            ceil_mode=True,
            return_indices=True,
        )
    )

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nn.functional.max_pool2d(
            maxpool_int16.to(device),
            kernel_size=(3, 2),
            stride=(2, 2),
            padding=(1, 1),
            dilation=(1, 1),
            ceil_mode=True,
            return_indices=True,
        )
    )

  def test_max_pool2d_no_indices_grad(self):
    """Tests nn.functional.max_pool2d backward without indices."""

    def get_grad(device):
      maxpool_input = torch.tensor(
          [[
              [
                  [-7.7435, -8.8254, 7.2097, 4.3371, 2.8040, -3.4491],
                  [-8.5819, 3.9336, -6.2229, 1.1184, -6.0094, 7.3457],
                  [-2.0333, 5.7398, 1.8601, 8.6590, 0.6541, 0.0145],
              ],
              [
                  [2.8523, -5.7473, 2.1480, -0.3480, 2.5668, -8.3042],
                  [-1.1508, -8.2351, 4.4935, -0.0096, -2.7059, -5.8874],
                  [5.4567, 0.2254, -3.6194, -6.1967, 8.8962, -1.7928],
              ],
          ]],
          dtype=torch.float32,
          device=device,
          requires_grad=True,
      )

      out = torch.nn.functional.max_pool2d(
          maxpool_input,
          kernel_size=3,
          stride=(2, 1),
          padding=1,
          dilation=(1, 2),
          ceil_mode=True,
          return_indices=False,
      )

      loss = out.sum()
      loss.backward()

      return maxpool_input.grad

    self.assert_close_tpu_vs_cpu(get_grad)

  def test_max_pool2d_with_indices_grad(self):
    """Tests nn.functional.max_pool2d backward with indices (overlapping).

    This serves as a regression test for a v6e-specific gradient mismatch.
    The underlying bug (violating the uniqueness promise in stablehlo.scatter)
    is masked on v5e due to serialized hardware execution naturally avoiding
    races. On highly concurrent v6e hardware, parallel out-of-order writes
    expose the data race, resulting in lost gradient updates unless
    unique_indices is set to False.
    """

    def get_grad(device):
      maxpool_input = torch.tensor(
          [[
              [
                  [-7.7435, -8.8254, 7.2097, 4.3371, 2.8040, -3.4491],
                  [-8.5819, 3.9336, -6.2229, 1.1184, -6.0094, 7.3457],
                  [-2.0333, 5.7398, 1.8601, 8.6590, 0.6541, 0.0145],
              ],
              [
                  [2.8523, -5.7473, 2.1480, -0.3480, 2.5668, -8.3042],
                  [-1.1508, -8.2351, 4.4935, -0.0096, -2.7059, -5.8874],
                  [5.4567, 0.2254, -3.6194, -6.1967, 8.8962, -1.7928],
              ],
          ]],
          dtype=torch.float32,
          device=device,
          requires_grad=True,
      )

      out, _ = torch.nn.functional.max_pool2d(
          maxpool_input,
          kernel_size=(3, 2),
          stride=2,
          padding=(1, 1),
          dilation=1,
          ceil_mode=True,
          return_indices=True,
      )

      loss = out.sum()
      loss.backward()

      return maxpool_input.grad

    self.assert_close_tpu_vs_cpu(get_grad)

  def test_max_pool2d_no_indices_grad_trivial_dilation(self):
    """Tests nn.functional.max_pool2d backward with trivial dilation (using SelectAndScatter)."""

    def get_grad(device):
      maxpool_input = torch.tensor(
          [[
              [
                  [-7.7435, -8.8254, 7.2097, 4.3371, 2.8040, -3.4491],
                  [-8.5819, 3.9336, -6.2229, 1.1184, -6.0094, 7.3457],
                  [-2.0333, 5.7398, 1.8601, 8.6590, 0.6541, 0.0145],
              ],
              [
                  [2.8523, -5.7473, 2.1480, -0.3480, 2.5668, -8.3042],
                  [-1.1508, -8.2351, 4.4935, -0.0096, -2.7059, -5.8874],
                  [5.4567, 0.2254, -3.6194, -6.1967, 8.8962, -1.7928],
              ],
          ]],
          dtype=torch.float32,
          device=device,
          requires_grad=True,
      )

      out = torch.nn.functional.max_pool2d(
          maxpool_input,
          kernel_size=3,
          stride=(2, 1),
          padding=1,
          dilation=(1, 1),
          ceil_mode=True,
          return_indices=False,
      )

      loss = out.sum()
      loss.backward()

      return maxpool_input.grad

    self.assert_close_tpu_vs_cpu(get_grad)

  def test_max_pool2d_backward_uncovered_spatial_tail(self):
    """Tests max_pool2d backward zeroes uncovered spatial tail elements."""
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
      # Case 1: H=W=5, kernel_size=2, stride=3, padding=1, ceil_mode=False.
      # Windows cover indices [-1, 1) and [2, 4), leaving index 4 uncovered.
      base_5x5 = torch.randn(1, 2, 5, 5, dtype=torch.float32)
      grad_out_5x5 = torch.randn(1, 2, 2, 2, dtype=torch.float32)

      def run_padded_tail(
          device, dtype=dtype, base=base_5x5, grad_out=grad_out_5x5
      ):
        x = (
            base.detach()
            .clone()
            .to(device=device, dtype=dtype)
            .requires_grad_(True)
        )
        g = grad_out.detach().clone().to(device=device, dtype=dtype)
        out = torch.nn.functional.max_pool2d(
            x, kernel_size=2, stride=3, padding=1, ceil_mode=False
        )
        out.backward(g)
        return x.grad

      self.assert_close_tpu_vs_cpu(run_padded_tail)
      tpu_grad_5x5 = run_padded_tail("tpu").cpu()
      self.assertTrue(torch.all(tpu_grad_5x5[:, :, 4:, :] == 0))
      self.assertTrue(torch.all(tpu_grad_5x5[:, :, :, 4:] == 0))

      # Case 2: H=W=3, kernel_size=2, stride=2, padding=0, ceil_mode=False.
      # Window covers [0, 2), leaving index 2 uncovered.
      base_3x3 = torch.randn(1, 2, 3, 3, dtype=torch.float32)
      grad_out_3x3 = torch.randn(1, 2, 1, 1, dtype=torch.float32)

      def run_unpadded_tail(
          device, dtype=dtype, base=base_3x3, grad_out=grad_out_3x3
      ):
        x = (
            base.detach()
            .clone()
            .to(device=device, dtype=dtype)
            .requires_grad_(True)
        )
        g = grad_out.detach().clone().to(device=device, dtype=dtype)
        out = torch.nn.functional.max_pool2d(
            x, kernel_size=2, stride=2, padding=0, ceil_mode=False
        )
        out.backward(g)
        return x.grad

      self.assert_close_tpu_vs_cpu(run_unpadded_tail)
      tpu_grad_3x3 = run_unpadded_tail("tpu").cpu()
      self.assertTrue(torch.all(tpu_grad_3x3[:, :, 2:, :] == 0))
      self.assertTrue(torch.all(tpu_grad_3x3[:, :, :, 2:] == 0))

  def test_max_pool2d_neg_inf_input(self):
    """Tests max_pool2d forward, indices, and backward with -inf inputs."""
    neg_inf = float("-inf")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
      base = torch.tensor(
          [[[
              [neg_inf, neg_inf, neg_inf, -2.0],
              [neg_inf, neg_inf, -5.0, neg_inf],
              [neg_inf, -1.0, neg_inf, neg_inf],
              [neg_inf, neg_inf, neg_inf, neg_inf],
          ]]],
          dtype=dtype,
      )
      grad_out = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=dtype)

      def run_with_indices(device, dtype=dtype, base=base, grad_out=grad_out):
        x = (
            base.detach()
            .clone()
            .to(device=device, dtype=dtype)
            .requires_grad_(True)
        )
        g = grad_out.detach().clone().to(device=device, dtype=dtype)
        out, indices = torch.nn.functional.max_pool2d(
            x, kernel_size=2, stride=2, return_indices=True
        )
        out.backward(g)
        return out, indices, x.grad

      self.assert_close_tpu_vs_cpu(run_with_indices)

      def run_no_indices(device, dtype=dtype, base=base, grad_out=grad_out):
        x = (
            base.detach()
            .clone()
            .to(device=device, dtype=dtype)
            .requires_grad_(True)
        )
        g = grad_out.detach().clone().to(device=device, dtype=dtype)
        out = torch.nn.functional.max_pool2d(
            x, kernel_size=2, stride=2, return_indices=False
        )
        out.backward(g)
        return out, x.grad

      self.assert_close_tpu_vs_cpu(run_no_indices)

  def test_max_pool1d_grad(self):
    """Tests nn.functional.max_pool1d backward without indices."""

    def get_grad(device):
      maxpool_input = torch.tensor(
          [[
              [
                  -7.7435,
                  -8.8254,
                  7.2097,
                  4.3371,
                  2.8040,
                  -3.4491,
                  5.1234,
                  -1.23,
              ],
              [2.8523, -5.7473, 2.1480, -0.3480, 2.5668, -8.3042, 3.456, 0.789],
          ]],
          dtype=torch.float32,
          device=device,
          requires_grad=True,
      )

      out = torch.nn.functional.max_pool1d(
          maxpool_input,
          kernel_size=3,
          stride=2,
          padding=1,
          dilation=1,
          ceil_mode=True,
          return_indices=False,
      )

      loss = out.sum()
      loss.backward()

      return maxpool_input.grad

    self.assert_close_tpu_vs_cpu(get_grad)

  def test_max_pool3d_grad(self):
    """Tests nn.functional.max_pool3d backward without indices."""
    input_data = torch.randn(
        (2, 2, 4, 6, 6),
        dtype=torch.float32,
        generator=torch.manual_seed(42),
    )

    def get_grad(device):
      maxpool_input = input_data.clone().to(device).requires_grad_(True)

      out = torch.nn.functional.max_pool3d(
          maxpool_input,
          kernel_size=3,
          stride=2,
          padding=1,
          dilation=1,
          ceil_mode=True,
          return_indices=False,
      )

      loss = out.sum()
      loss.backward()

      return maxpool_input.grad

    self.assert_close_tpu_vs_cpu(get_grad)

  def test_fmax_special_values(self):
    """Tests torch.fmax correctly handles special values."""
    values = [0.0, -float("inf"), float("inf"), float("nan")]

    for val1, val2 in itertools.product(values, repeat=2):

      def test_fn(device, v1=val1, v2=val2):
        t = torch.tensor([v1], device=device, dtype=torch.float32)
        other = torch.tensor([v2], device=device, dtype=torch.float32)
        return torch.fmax(t, other)

      cpu_res = test_fn("cpu")
      tpu_res = test_fn(torch.device("tpu"))
      self.assertEqual(cpu_res, tpu_res.cpu())

  def test_fmod_float_zero_division(self):
    # Return NaN for floating point division by zero on both CPU and TPU
    def test_fn(device):
      t = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float32)
      other = torch.zeros_like(t)
      return torch.fmod(t, other)

    cpu_res = test_fn("cpu")
    self.assertTrue(torch.isnan(cpu_res).all())

    tpu_res = test_fn(torch.device("tpu"))
    self.assertTrue(torch.isnan(tpu_res.cpu()).all())

  # TODO: Make fmod() consistent across CPU and TPU for integer zero division
  def test_fmod_int_zero_division(self):
    # CPU raises RuntimeError for integer division by zero
    t_cpu = torch.tensor([1, 2, 3], device="cpu", dtype=torch.int32)
    other_cpu = torch.zeros_like(t_cpu)
    with self.assertRaisesRegex(RuntimeError, "ZeroDivisionError"):
      torch.fmod(t_cpu, other_cpu)

    # TPU can return any value
    t_tpu = torch.tensor(
        [1, 2, 3],
        device=torch.device("tpu"),
        dtype=torch.int32,
    )
    other_tpu = torch.zeros_like(t_tpu)
    res_tpu = torch.fmod(t_tpu, other_tpu)
    self.assertEqual(res_tpu.dtype, torch.int32)
    self.assertEqual(res_tpu.shape, t_tpu.shape)

  def test_cdist_forward_empty_input(self):
    # Test that cdist_forward returns an empty tensor when the output shape
    # contains a 0 dimension.
    x1_cpu = torch.randn(0, 5, device="cpu", dtype=torch.float32)
    x2_cpu = torch.randn(4, 5, device="cpu", dtype=torch.float32)
    res_cpu = torch.cdist(x1_cpu, x2_cpu)

    x1_tpu = x1_cpu.to(torch.device("tpu"))
    x2_tpu = x2_cpu.to(torch.device("tpu"))
    res_tpu = torch.cdist(x1_tpu, x2_tpu)

    self.assertEqual(res_tpu.shape, res_cpu.shape)
    self.assertEqual(res_tpu.dtype, res_cpu.dtype)
    self.assertEqual(res_tpu.shape, (0, 4))

    # Test that BF16 support for empty inputs
    x1_bf16 = torch.randn(
        0,
        5,
        device=torch.device("tpu"),
        dtype=torch.bfloat16,
    )
    x2_bf16 = torch.randn(
        4,
        5,
        device=torch.device("tpu"),
        dtype=torch.bfloat16,
    )

    res_bf16 = torch.cdist(x1_bf16, x2_bf16)
    self.assertEqual(res_bf16.shape, (0, 4))
    self.assertEqual(res_bf16.dtype, torch.bfloat16)

  def test_cdist_with_different_p_values(self):
    p_values = [0.0, 1.0, 2.0, float("inf")]
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])

    for p in p_values:

      def test_fn(device, p=p):
        return torch.cdist(x1.to(device), x2.to(device), p=p)

      self.assert_close_tpu_vs_cpu(test_fn)

  @parameterized.parameters(0.0, 0.5, 1.0, 2.0, 3.0, float("inf"))
  def test_cdist_backward_with_different_p_values(self, p):
    x1_init = torch.tensor(
        [[-1.5, 0.25, 2.0], [0.5, -2.25, 1.0]], dtype=torch.float32
    )
    x2_init = torch.tensor(
        [[0.75, -1.25, -0.5], [-0.25, 1.5, 3.25], [2.5, 0.75, -1.5]],
        dtype=torch.float32,
    )
    grad_out = torch.tensor(
        [[0.5, -1.25, 0.75], [-0.5, 0.25, 1.5]], dtype=torch.float32
    )

    def test_fn(device):
      x1 = to(x1_init.clone(), device=device).requires_grad_(True)
      x2 = to(x2_init.clone(), device=device).requires_grad_(True)
      out = torch.cdist(
          x1, x2, p=p, compute_mode="donot_use_mm_for_euclid_dist"
      )
      out.backward(to(grad_out, device=device))
      return out, x1.grad, x2.grad

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_pdist_with_different_p_values(self):
    p_values = [0.0, 1.0, 2.0, float("inf")]
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = torch.randn(4, 5)

    for p in p_values:

      def test_fn(device, p=p):
        loss_x = torch.pdist(x.to(device), p=p)
        loss_y = torch.pdist(y.to(device), p=p)
        return loss_x, loss_y

      self.assert_close_tpu_vs_cpu(test_fn)

    for dtype in [torch.bfloat16, torch.float16]:
      x_zero_feat = torch.randn(3, 0, dtype=dtype)

      def test_zero_feat_fn(device, x_zero_feat=x_zero_feat):
        return torch.nn.functional.pdist(x_zero_feat.to(device), p=2.0)

      self.assert_close_tpu_vs_cpu(test_zero_feat_fn)

  @parameterized.product(
      stride=[2],
      padding=[1],
      groups=[1],
  )
  def test_convolution_backward_strides(self, stride, padding, groups):
    """Tests convolution backward with various strides."""
    batch = 1
    in_channels_per_group = 1
    out_channels_per_group = 2
    in_channels = groups * in_channels_per_group
    out_channels = groups * out_channels_per_group
    input_ = torch.randn(batch, in_channels, 4, 3)
    weight = torch.randn(out_channels, in_channels_per_group, 3, 4)
    bias = torch.randn(out_channels)
    grad_output = torch.randn(batch, out_channels, 2, 1)

    def test_fn(device):
      in_dev = input_.to(device).detach().requires_grad_(True)
      w_dev = weight.to(device).detach().requires_grad_(True)
      b_dev = bias.to(device).detach().requires_grad_(True)
      out = torch.nn.functional.conv2d(
          in_dev,
          w_dev,
          bias=b_dev,
          stride=stride,
          padding=padding,
          groups=groups,
      )
      out.backward(grad_output.to(device))

      return in_dev.grad, w_dev.grad, b_dev.grad

    self.assert_close_tpu_vs_cpu(test_fn, rtol=1e-2, atol=1e-2)

  @parameterized.named_parameters(
      (
          "conv1d_bfloat16",
          "conv1d",
          torch.bfloat16,
          (16, 8, 256),
          (8, 8, 3),
      ),
      (
          "conv2d_bfloat16",
          "conv2d",
          torch.bfloat16,
          (16, 8, 32, 32),
          (8, 8, 3, 3),
      ),
      (
          "conv3d_bfloat16",
          "conv3d",
          torch.bfloat16,
          (4, 8, 16, 16, 16),
          (8, 8, 3, 3, 3),
      ),
      (
          "conv_transpose1d_bfloat16",
          "conv_transpose1d",
          torch.bfloat16,
          (16, 8, 256),
          (8, 8, 3),
      ),
      (
          "conv_transpose2d_bfloat16",
          "conv_transpose2d",
          torch.bfloat16,
          (16, 8, 32, 32),
          (8, 8, 3, 3),
      ),
      (
          "conv1d_float16",
          "conv1d",
          torch.float16,
          (16, 8, 256),
          (8, 8, 3),
      ),
      (
          "conv2d_float16",
          "conv2d",
          torch.float16,
          (16, 8, 32, 32),
          (8, 8, 3, 3),
      ),
      (
          "conv3d_float16",
          "conv3d",
          torch.float16,
          (4, 8, 16, 16, 16),
          (8, 8, 3, 3, 3),
      ),
      (
          "conv_transpose1d_float16",
          "conv_transpose1d",
          torch.float16,
          (16, 8, 256),
          (8, 8, 3),
      ),
      (
          "conv_transpose2d_float16",
          "conv_transpose2d",
          torch.float16,
          (16, 8, 32, 32),
          (8, 8, 3, 3),
      ),
  )
  def test_convolution_backward_bias_precision(
      self,
      op_name: str,
      dtype: torch.dtype,
      x_shape: tuple[int, ...],
      w_shape: tuple[int, ...],
  ) -> None:
    """Verifies that convolution backward bias accumulates in float32."""
    fn = getattr(torch.nn.functional, op_name)
    torch.manual_seed(42)
    out_channels = w_shape[1] if "transpose" in op_name else w_shape[0]
    input_ = torch.randn(x_shape, dtype=dtype)
    weight = torch.randn(w_shape, dtype=dtype)
    bias = torch.randn(out_channels, dtype=dtype)
    dummy_out = fn(input_, weight, bias)
    grad_output = (torch.randn_like(dummy_out) + 0.5).to(dtype)

    def test_fn(device: torch.device) -> torch.Tensor:
      in_dev = input_.to(device).detach().requires_grad_(True)
      w_dev = weight.to(device).detach().requires_grad_(True)
      b_dev = bias.to(device).detach().requires_grad_(True)
      out = fn(in_dev, w_dev, bias=b_dev)
      out.backward(grad_output.to(device))
      return b_dev.grad

    self.assert_close_tpu_vs_cpu(test_fn, rtol=1e-4, atol=1e-4)

  @parameterized.product(
      kernel_size=[3],
      stride=[2],
      padding=[1],
      input_size=[28],
  )
  def test_convolution_backward_weight_overhang(
      self,
      kernel_size: int | tuple[int, int],
      stride: int | tuple[int, int],
      padding: int | tuple[int, int],
      input_size: int | tuple[int, int],
  ) -> None:
    """Tests convolution backward where stride causes an overhang."""

    # Explanation of the failure in symmetric padding case:
    # 1. Forward Pass:
    #    - Input 28 with Padding 1 becomes 30.
    #    - Output Size = floor((30 - 3) / 2) + 1 = 14.
    #    - The last window starts at index (13 * 2 - 1) = 25 and ends at 27.
    #    - Index 28 (the rightmost padding) is NOT reached by any window. This
    #      is called 'Overhang'.

    # 2. Backward Weight Gradient:
    #    - Effective Kernel Size (K_eff) = 2 * (14 - 1) + 1 = 27.
    #    - If we use SYMMETRIC padding (lo=1, hi=1):
    #      Result Size = (28 + 1 + 1 - 27) / 1 + 1 = 4.
    #    - This results in a 4x4 gradient, but the original weight is 3x3.

    # 3. Fix:
    #    - Use asymmetric padding (lo=0, hi=1) to make the output size 13.
    #    - Then, the weight gradient size is (28 + 0 + 1 - 26) / 1 + 1 = 3.
    batch = 1
    in_channels = 1
    out_channels = 1

    input_ = torch.randn(
        batch, in_channels, input_size, input_size, requires_grad=True
    )
    weight = torch.randn(
        out_channels, in_channels, kernel_size, kernel_size, requires_grad=True
    )
    bias = torch.randn(out_channels, requires_grad=True)

    def test_fn(
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      out = torch.nn.functional.conv2d(
          input_.to(device),
          weight.to(device),
          bias=bias.to(device),
          stride=stride,
          padding=padding,
      )
      grad_output = torch.randn_like(out, device=device)
      out.backward(grad_output)

      return input_.grad, weight.grad, bias.grad

    self.assert_close_tpu_vs_cpu(test_fn, rtol=1e-2, atol=1e-2)

  @parameterized.product(
      groups=[1, 2, 4],
      in_channels_per_group=[1, 2],
      out_channels_per_group=[1, 2],
  )
  def test_conv_transpose2d_backward_groups(
      self, groups, in_channels_per_group, out_channels_per_group
  ):
    """Tests conv_transpose2d backward with various group sizes."""
    in_channels = groups * in_channels_per_group
    out_channels = groups * out_channels_per_group
    input_ = torch.randn(
        2,
        in_channels,
        4,
        4,
        requires_grad=True,
    )
    weight = torch.randn(
        in_channels,
        out_channels_per_group,
        3,
        3,
        requires_grad=True,
    )
    bias = torch.randn(out_channels, requires_grad=True)

    def test_fn(device):
      out = torch.nn.functional.conv_transpose2d(
          input_.to(device),
          weight.to(device),
          bias=bias.to(device),
          groups=groups,
      )
      grad_output = torch.randn_like(out, device=device)
      out.backward(grad_output)

      return input_.grad, weight.grad, bias.grad

    self.assert_close_tpu_vs_cpu(test_fn, rtol=1e-2, atol=1e-2)

  @parameterized.product(
      batch_size=[1, 2],
      dilation=[2, 4],
      padding=[0, 2],
  )
  def test_conv1d_dilated_backward(self, batch_size, dilation, padding):
    """Tests 1D dilated convolution backward pass on TPU vs CPU golden.

    Dilated 1D convolutions (atrous convolutions) expand the receptive field
    exponentially across time steps without downsampling or increasing the
    parameter count. This is a foundational building block in Temporal
    Convolutional Networks (TCN architectures) used in quantitative finance for
    volatility forecasting, macro trend detection, and price prediction.

    Test Configuration:
      - Input tensor: (batch_size, in_channels=2, length=16)
      - Weight tensor: (out_channels=2, in_channels=2, kernel_size=3)
      - Bias tensor: (out_channels=2)
      - Dilation: 2 or 4 (expanding effective receptive field to 5 or 9 steps)
      - Padding: 0 or 2 (valid or symmetric padding)

    Expected Output:
      - Forward output tensor matches CPU golden within rtol=1e-2, atol=1e-2.
      - Input gradient (x.grad) matches CPU golden shape and values.
      - Weight gradient (w.grad) matches CPU golden shape and values.
      - Bias gradient (b.grad) matches CPU golden shape and values.
    """
    in_channels = 2
    out_channels = 2
    length = 16
    kernel_size = 3

    # Generate synthetic input and weight tensors on CPU as golden references.
    x_cpu = torch.randn(batch_size, in_channels, length, dtype=torch.float32)
    weight_cpu = torch.randn(
        out_channels, in_channels, kernel_size, dtype=torch.float32
    )
    bias_cpu = torch.randn(out_channels, dtype=torch.float32)

    def test_fn(device):
      # Attach gradients to test tensors transferred to the target device.
      x = x_cpu.to(device).detach().requires_grad_(True)
      w = weight_cpu.to(device).detach().requires_grad_(True)
      b = bias_cpu.to(device).detach().requires_grad_(True)

      # Forward pass: 1D convolution with dilated spacing between kernel taps.
      out = torch.nn.functional.conv1d(
          x, w, bias=b, dilation=dilation, padding=padding
      )

      # Backward pass: backpropagate unit gradients through all parameters.
      grad_output = torch.ones_like(out)
      out.backward(grad_output)

      # Return forward activation and all backward gradient tensors.
      return out, x.grad, w.grad, b.grad

    # Verify forward and backward numerical parity between TPU and CPU.
    self.assert_close_tpu_vs_cpu(test_fn, rtol=1e-2, atol=1e-2)

  @parameterized.product(
      batch_size=[1, 2],
      dilation=[1, 2, 4],
  )
  def test_conv1d_causal_backward(self, batch_size, dilation):
    """Tests causal 1D convolution backward pass with strict left-padding.

    In financial algorithmic trading, convolutions must be strictly causal:
    the output at timestamp t must only depend on inputs at timestamps <= t,
    strictly preventing lookahead bias and future information leakage.

    Causality is enforced by prepending (kernel_size - 1) * dilation zero-pads
    strictly to the left (past) side of the time series, followed by an
    unpadded (padding=0) convolution.

    Test Configuration:
      - Input tensor: (batch_size, in_channels=2, length=12)
      - Weight tensor: (out_channels=2, in_channels=2, kernel_size=3)
      - Left-padding: (kernel_size - 1) * dilation = 2 * dilation zeros
      - Dilation: 1, 2, or 4

    Expected Output:
      - Forward activation of shape (batch_size, out_channels=2, length=12),
        preserving sequence length without future data leakage.
      - Accurate backpropagation through both convolution and left-pad layers.
      - Exact numerical parity between TPU and CPU golden within tolerances.
    """
    in_channels = 2
    out_channels = 2
    length = 12
    kernel_size = 3
    left_pad = (kernel_size - 1) * dilation

    # Generate synthetic input and weight tensors on CPU.
    x_cpu = torch.randn(batch_size, in_channels, length, dtype=torch.float32)
    weight_cpu = torch.randn(
        out_channels, in_channels, kernel_size, dtype=torch.float32
    )
    bias_cpu = torch.randn(out_channels, dtype=torch.float32)

    def test_fn(device):
      # Attach autograd gradient trackers.
      x = x_cpu.to(device).detach().requires_grad_(True)
      w = weight_cpu.to(device).detach().requires_grad_(True)
      b = bias_cpu.to(device).detach().requires_grad_(True)

      # Apply asymmetric left-padding: pad format is (left, right).
      x_padded = torch.nn.functional.pad(x, (left_pad, 0))

      # Causal convolution with padding=0 on the padded time-series.
      out = torch.nn.functional.conv1d(
          x_padded, w, bias=b, dilation=dilation, padding=0
      )

      # Backpropagate unit gradients through convolution and left-pad op.
      grad_output = torch.ones_like(out)
      out.backward(grad_output)

      return out, x.grad, w.grad, b.grad

    # Verify TPU and CPU produce identical forward results and gradients.
    self.assert_close_tpu_vs_cpu(test_fn, rtol=1e-2, atol=1e-2)

  @parameterized.product(
      length=[4, 6],
      stride=[1, 2],
  )
  def test_conv1d_streaming_small_shape_backward(self, length, stride):
    """Tests low-latency streaming 1D convolution backward on minimal buffers.

    Ultra-low-latency financial trading systems process events (tick quotes,
    order updates) as individual streaming items (batch_size=1) with very
    short history windows (lengths 4 to 6).

    This test validates that TPU lowering, padding, and autograd kernels
    execute correctly on sub-tile shapes without tiling corruption or
    dimension alignment bugs on physical TPU vector hardware.

    Test Configuration:
      - Batch size: 1 (single-stream streaming inference / online adaptation)
      - Channels: in_channels=1, out_channels=2
      - Sequence length: 4 or 6 timesteps
      - Kernel size: 2, Stride: 1 or 2

    Expected Output:
      - Forward output tensor matching CPU reference shape and values.
      - Complete gradient propagation (x.grad, w.grad, b.grad) matching CPU
        golden within rtol=1e-2, atol=1e-2.
    """
    batch_size = 1
    in_channels = 1
    out_channels = 2
    kernel_size = 2

    # Initialize small-shape tensors for low-latency streaming test.
    x_cpu = torch.randn(batch_size, in_channels, length, dtype=torch.float32)
    weight_cpu = torch.randn(
        out_channels, in_channels, kernel_size, dtype=torch.float32
    )
    bias_cpu = torch.randn(out_channels, dtype=torch.float32)

    def test_fn(device):
      # Transfer tensors to device and enable gradient tracking.
      x = x_cpu.to(device).detach().requires_grad_(True)
      w = weight_cpu.to(device).detach().requires_grad_(True)
      b = bias_cpu.to(device).detach().requires_grad_(True)

      # Forward pass: 1D convolution across short streaming buffer.
      out = torch.nn.functional.conv1d(x, w, bias=b, stride=stride)

      # Backward pass: backpropagate gradients across minimal sequence lengths.
      grad_output = torch.ones_like(out)
      out.backward(grad_output)

      return out, x.grad, w.grad, b.grad

    # Verify TPU output and parameter gradients match CPU golden.
    self.assert_close_tpu_vs_cpu(test_fn, rtol=1e-2, atol=1e-2)

  @parameterized.product(
      batch_size=[1, 2],
      kernel_and_stride=[
          ((1, 2), (1, 2)),  # Spatial aggregation across price-volume depth
          ((3, 1), (1, 1)),  # Temporal aggregation across timestamps
      ],
  )
  def test_conv2d_anisotropic_lob_backward(self, batch_size, kernel_and_stride):
    """Tests 2D convolution backward with anisotropic kernels for LOB data.

    Deep Limit Order Book (DeepLOB) models represent order books as 2D grids
    where height represents time (consecutive event updates) and width
    represents order book depth (bid/ask price levels and queue volumes).
    Because the two spatial axes represent fundamentally different physics,
    anisotropic (non-square) kernels are required:
      1. Spatial kernel (1, 2) with stride (1, 2): captures interactions
         between adjacent bid/ask price and volume levels at the same timestamp.
      2. Temporal kernel (3, 1) with stride (1, 1): captures multi-step
         temporal momentum across consecutive events for a single price level.

    Test Configuration:
      - Input grid: (batch_size, in_channels=1, height=10, width=6)
        representing [time] x [depth].
      - Weight tensor: (out_channels=2, in_channels=1, k_h, k_w)
      - Kernel & stride pairs: ((1, 2), (1, 2)) and ((3, 1), (1, 1))

    Expected Output:
      - Forward feature map matching CPU golden reference.
      - Backpropagated input, weight, and bias gradients matching CPU reference
        within rtol=5e-2, atol=5e-2.
    """
    kernel_size, stride = kernel_and_stride
    in_channels = 1
    out_channels = 2
    height, width = 10, 6

    # Generate synthetic limit order book grid (time x depth).
    x_cpu = torch.randn(
        batch_size, in_channels, height, width, dtype=torch.float32
    )
    weight_cpu = torch.randn(
        out_channels,
        in_channels,
        kernel_size[0],
        kernel_size[1],
        dtype=torch.float32,
    )
    bias_cpu = torch.randn(out_channels, dtype=torch.float32)

    def test_fn(device):
      # Enable gradient tracking on device tensors.
      x = x_cpu.to(device).detach().requires_grad_(True)
      w = weight_cpu.to(device).detach().requires_grad_(True)
      b = bias_cpu.to(device).detach().requires_grad_(True)

      # Forward pass: 2D convolution using anisotropic rectangular kernel.
      out = torch.nn.functional.conv2d(x, w, bias=b, stride=stride)

      # Backward pass: compute gradients across anisotropic dimensions.
      grad_output = torch.ones_like(out)
      out.backward(grad_output)

      return out, x.grad, w.grad, b.grad

    # Verify forward and backward parity between TPU and CPU golden.
    self.assert_close_tpu_vs_cpu(test_fn, rtol=5e-2, atol=5e-2)

  @parameterized.product(
      batch_size=[1, 2],
      dilation=[1, 2],
  )
  def test_conv1d_depthwise_backward(self, batch_size, dilation):
    """Tests depthwise 1D convolution backward (groups = in_channels).

    Depthwise convolutions apply independent spatial filters to each input
    channel individually. In quantitative finance and multi-asset portfolio
    modeling, each channel corresponds to an independent financial instrument
    (e.g., FX pair, stock ticker, cryptocurrency) to learn asset-specific
    temporal dynamics prior to cross-asset aggregation.

    Test Configuration:
      - Input tensor: (batch_size, channels=2, length=12)
      - Weight tensor: (channels=2, 1, kernel_size=3)
      - Groups: channels (strictly depthwise)
      - Dilation: 1 or 2

    Expected Output:
      - Forward output tensor computed independently per channel.
      - Weight gradients of shape (channels, 1, kernel_size) isolated per group.
      - Full numerical parity between TPU and CPU reference within rtol=1e-2,
        atol=1e-2.
    """
    channels = 2
    groups = channels
    length = 12
    kernel_size = 3

    # Generate multi-channel input and per-channel independent weights.
    x_cpu = torch.randn(batch_size, channels, length, dtype=torch.float32)
    weight_cpu = torch.randn(channels, 1, kernel_size, dtype=torch.float32)
    bias_cpu = torch.randn(channels, dtype=torch.float32)

    def test_fn(device):
      # Enable autograd tracking for input, depthwise weight, and bias.
      x = x_cpu.to(device).detach().requires_grad_(True)
      w = weight_cpu.to(device).detach().requires_grad_(True)
      b = bias_cpu.to(device).detach().requires_grad_(True)

      # Forward pass: depthwise 1D conv where groups equals channel count.
      out = torch.nn.functional.conv1d(
          x, w, bias=b, dilation=dilation, groups=groups
      )

      # Backward pass: verify gradient routing to grouped weight channels.
      grad_output = torch.ones_like(out)
      out.backward(grad_output)

      return out, x.grad, w.grad, b.grad

    # Verify TPU execution matches CPU golden reference.
    self.assert_close_tpu_vs_cpu(test_fn, rtol=1e-2, atol=1e-2)

  @parameterized.product(
      batch_size=[1, 2],
      stride=[1, 2],
      padding=[0, 1],
  )
  def test_conv3d_backward(self, batch_size, stride, padding):
    """Tests standard 3D convolution backward on fast, bounded tensor shapes.

    Standard PyTorch OpInfo generates randomly sized 3D convolution inputs with
    large dimensions (e.g. 10x11x12) and large dilation/padding combinations,
    which cause XLA / StableHLO compilation timeouts during test execution.

    This unit test provides fast, regression-safe coverage for standard 3D
    convolution backward by constraining dimensions to 4x4x4 and kernel size
    to 2x2x2.

    Test Configuration:
      - Input tensor: (batch_size, in_channels=2, d=4, h=4, w=4)
      - Weight tensor: (out_channels=2, in_channels=2, k=2, k=2, k=2)
      - Stride: 1 or 2
      - Padding: 0 or 1

    Expected Output:
      - Forward 3D convolution output matching CPU reference.
      - Backpropagated input, weight, and bias gradients matching CPU reference
        within rtol=5e-2, atol=5e-2.
    """
    in_channels = 2
    out_channels = 2
    d, h, w = 4, 4, 4
    kernel_size = 2

    # Initialize compact 3D tensors.
    x_cpu = torch.randn(batch_size, in_channels, d, h, w, dtype=torch.float32)
    weight_cpu = torch.randn(
        out_channels,
        in_channels,
        kernel_size,
        kernel_size,
        kernel_size,
        dtype=torch.float32,
    )
    bias_cpu = torch.randn(out_channels, dtype=torch.float32)

    def test_fn(device):
      # Track gradients for input, kernel weights, and bias.
      x = x_cpu.to(device).detach().requires_grad_(True)
      w = weight_cpu.to(device).detach().requires_grad_(True)
      b = bias_cpu.to(device).detach().requires_grad_(True)

      # Forward pass: standard 3D convolution across spatial depth/height/width.
      out = torch.nn.functional.conv3d(
          x, w, bias=b, stride=stride, padding=padding
      )

      # Backward pass: compute adjoint gradients across 3D dimensions.
      grad_output = torch.ones_like(out)
      out.backward(grad_output)

      return out, x.grad, w.grad, b.grad

    # Verify numerical equivalence between TPU and CPU reference.
    self.assert_close_tpu_vs_cpu(test_fn, rtol=5e-2, atol=5e-2)

  @parameterized.product(
      batch_size=[1, 2],
      stride=[1, 2],
  )
  def test_conv_transpose3d_backward(self, batch_size, stride):
    """Tests 3D transposed convolution backward on fast, bounded tensor shapes.

    Transposed 3D convolution (fractionally strided convolution) is the inverse
    transformation of standard 3D convolution, commonly used in 3D generative
    models, volumetric segmentation decoders, and physics upsamplers.

    Test Configuration:
      - Input tensor: (batch_size, in_channels=2, d=3, h=3, w=3)
      - Weight tensor: (in_channels=2, out_channels=2, k=2, k=2, k=2)
      - Stride: 1 or 2

    Expected Output:
      - Upsampled forward 3D output matching CPU golden reference.
      - Correct backpropagation computing input, filter, and bias gradients
        matching CPU reference within rtol=5e-2, atol=5e-2.
    """
    in_channels = 2
    out_channels = 2
    d, h, w = 3, 3, 3
    kernel_size = 2

    # Initialize 3D tensors for transposed convolution.
    x_cpu = torch.randn(batch_size, in_channels, d, h, w, dtype=torch.float32)
    weight_cpu = torch.randn(
        in_channels,
        out_channels,
        kernel_size,
        kernel_size,
        kernel_size,
        dtype=torch.float32,
    )
    bias_cpu = torch.randn(out_channels, dtype=torch.float32)

    def test_fn(device):
      # Enable autograd tracking for input, transposed filter, and bias.
      x = x_cpu.to(device).detach().requires_grad_(True)
      w = weight_cpu.to(device).detach().requires_grad_(True)
      b = bias_cpu.to(device).detach().requires_grad_(True)

      # Forward pass: 3D transposed convolution upsampling spatial dimensions.
      out = torch.nn.functional.conv_transpose3d(x, w, bias=b, stride=stride)

      # Backward pass: backpropagate unit gradients through the transposed conv.
      grad_output = torch.ones_like(out)
      out.backward(grad_output)

      return out, x.grad, w.grad, b.grad

    # Verify TPU results match CPU golden reference.
    self.assert_close_tpu_vs_cpu(test_fn, rtol=5e-2, atol=5e-2)

  @parameterized.product(
      batch_size=[1, 2],
      dilation_and_padding=[
          ((2, 1), (1, 0)),
          ((1, 2), (0, 1)),
      ],
  )
  def test_conv2d_asymmetric_dilation_padding_backward(
      self, batch_size, dilation_and_padding
  ):
    """Tests 2D convolution backward with asymmetric dilation and padding pairs.

    Verifies that rectangular dilation rates (e.g., dilation=(2, 1) or (1, 2))
    paired with corresponding asymmetric padding (e.g., padding=(1, 0) or
    (0, 1)) lower correctly into TPU XLA HLO convolution operators and produce
    accurate forward activations and autograd gradients.

    Test Configuration:
      - Input tensor: (batch_size, in_channels=2, height=8, width=8)
      - Weight tensor: (out_channels=2, in_channels=2, 3, 3)
      - Dilation & padding pairs: ((2, 1), (1, 0)) and ((1, 2), (0, 1))

    Expected Output:
      - Forward 2D convolution output matching CPU reference with asymmetric
        receptive fields.
      - Input, weight, and bias gradients matching CPU golden reference within
        rtol=5e-2, atol=5e-2.
    """
    dilation, padding = dilation_and_padding
    in_channels = 2
    out_channels = 2
    height, width = 8, 8
    kernel_size = 3

    # Generate synthetic input and weight tensors on CPU.
    x_cpu = torch.randn(
        batch_size, in_channels, height, width, dtype=torch.float32
    )
    weight_cpu = torch.randn(
        out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32
    )
    bias_cpu = torch.randn(out_channels, dtype=torch.float32)

    def test_fn(device):
      # Transfer tensors to device and enable gradient computation.
      x = x_cpu.to(device).detach().requires_grad_(True)
      w = weight_cpu.to(device).detach().requires_grad_(True)
      b = bias_cpu.to(device).detach().requires_grad_(True)

      # Forward pass: 2D convolution with asymmetric dilation and padding.
      out = torch.nn.functional.conv2d(
          x, w, bias=b, dilation=dilation, padding=padding
      )

      # Backward pass: backpropagate unit gradients through asymmetric conv.
      grad_output = torch.ones_like(out)
      out.backward(grad_output)

      return out, x.grad, w.grad, b.grad

    # Verify TPU outputs and gradients match CPU golden reference.
    self.assert_close_tpu_vs_cpu(test_fn, rtol=5e-2, atol=5e-2)

  def test_linear_inference_mode(self):
    class LinearModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 10)

      def forward(self, x):
        return self.linear(x)

    model = LinearModel()
    input_tensor = torch.ones(2, 8, 100)

    def test_fn(device):
      model_device = model.to(device)
      input_tensor_device = input_tensor.to(device)
      with torch.inference_mode():
        output = model_device(input_tensor_device)
      return output

    self.assert_close_tpu_vs_cpu(test_fn, rtol=4e-2, atol=1e-2)

  def test_avg_pool3d_backward_padding(self):
    """Test for avg_pool3d_backward crashing when stride is large.

    - Input Dim: 2
    - Kernel: 2
    - Stride: 4
    - Padding: 1

    - Intermediate tensor dim: (Out-1)*Stride + Kernel = 2
    - Slice needed: pad_low=1, input_dim=2 -> slice[1:1+2] = slice[1:3]

    - Crash: limit index 3 is larger than dimension size 2 in dimension 2
    - Fix: Pad the reconstructed tensor to size 3.
    """
    input_val = torch.randn(1, 1, 2, 2, 2, dtype=torch.float64)
    kernel_size = (2, 2, 2)
    stride = 4
    padding = 1

    def test_fn(device):
      inp = input_val.clone().to(device).requires_grad_(True)
      output = torch.nn.functional.avg_pool3d(
          inp,
          kernel_size=kernel_size,
          stride=stride,
          padding=padding,
          divisor_override=8,
      )

      self.assertEqual(output.shape, (1, 1, 1, 1, 1))

      grad_output = torch.ones_like(output)
      output.backward(grad_output)
      return inp.grad

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_avg_pool2d_ceil_mode_count_include_pad(self):
    """Tests avg_pool2d with ceil_mode=True and count_include_pad=True.

    Demonstrates the bug where the TPU implementation divides by the full
    window size even when `ceil_mode=True` pushes the window out of the
    explicit padding bounds.
    """
    # - PyTorch sums them (400) and divides by the valid area (4) -> 100.0
    # - Buggy TPU sums them (400) and divides by k*k (16) -> 25.0
    input_value = torch.full((1, 1, 5, 5), 100.0, dtype=torch.float32)

    def compute(device):
      return torch.nn.functional.avg_pool2d(
          input_value.to(device),
          kernel_size=4,
          stride=3,
          padding=0,
          ceil_mode=True,
          count_include_pad=True,
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_avg_pool3d_ceil_mode_count_include_pad(self):
    """Tests avg_pool3d with ceil_mode=True and count_include_pad=True."""
    input_value = torch.full((1, 1, 5, 5, 5), 100.0, dtype=torch.float32)

    def compute(device):
      return torch.nn.functional.avg_pool3d(
          input_value.to(device),
          kernel_size=4,
          stride=3,
          padding=1,
          ceil_mode=True,
          count_include_pad=True,
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_avg_pool_low_precision_large_window(self):
    """Tests 16-bit avg_pool1d/2d with windows larger than 256 elements.

    Window sums (and, with count_include_pad=False, the element counts) must
    accumulate in float32 like ATen does. Accumulating in bfloat16 cannot
    represent counts above 256 exactly, which made the result up to 3x off.
    """
    torch.manual_seed(0)
    x2d = torch.randn(2, 4, 64, 64) + 3.0
    grad2d = torch.randn(2, 4, 4, 4)
    x1d = torch.randn(2, 4, 2048) + 3.0
    for dtype in (torch.bfloat16, torch.float16):

      def compute_2d(device, dtype=dtype):
        return torch.nn.functional.avg_pool2d(
            x2d.to(dtype).to(device),
            kernel_size=32,
            stride=16,
            padding=8,
            count_include_pad=False,
        )

      def compute_2d_grad(device, dtype=dtype):
        inp = x2d.to(dtype).to(device).requires_grad_()
        out = torch.nn.functional.avg_pool2d(
            inp, kernel_size=32, stride=16, padding=8, count_include_pad=False
        )
        out.backward(grad2d.to(dtype).to(device))
        return inp.grad

      def compute_1d(device, dtype=dtype):
        return torch.nn.functional.avg_pool1d(
            x1d.to(dtype).to(device),
            kernel_size=512,
            stride=256,
            padding=128,
            count_include_pad=False,
        )

      with self.subTest(dtype=dtype, op="avg_pool2d"):
        self.assert_close_tpu_vs_cpu(compute_2d)
      with self.subTest(dtype=dtype, op="avg_pool2d_backward"):
        self.assert_close_tpu_vs_cpu(compute_2d_grad)
      with self.subTest(dtype=dtype, op="avg_pool1d"):
        self.assert_close_tpu_vs_cpu(compute_1d)

  def test_bernoulli_out_dtypes(self):
    device = torch.device("tpu")
    p = torch.full((4, 4), 0.5, device=device)
    for out_dtype in [torch.int32, torch.int64, torch.float32, torch.bool]:
      out = torch.empty((0,), device=device, dtype=out_dtype)
      res = torch.bernoulli(p, out=out)
      self.assertEqual(res.shape, (4, 4))
      self.assertEqual(res.dtype, out_dtype)
      self.assertIs(res, out)

  def test_multinomial_out_resizing(self):
    device = torch.device("tpu")
    weights = torch.tensor([[0.1, 0.9], [0.8, 0.2]], device=device)
    out = torch.empty((0,), device=device, dtype=torch.int64)
    res = torch.multinomial(weights, num_samples=2, replacement=True, out=out)
    self.assertEqual(res.shape, (2, 2))
    self.assertEqual(res.dtype, torch.int64)
    self.assertIs(res, out)

  def test_histc_out_upcast_and_resizing(self):
    device = torch.device("tpu")
    x = torch.tensor([1.0, 2.0, 2.5, 3.0], dtype=torch.float32)
    bins = 4
    min_val = 0.0
    max_val = 4.0
    expected = torch.histc(x, bins=bins, min=min_val, max=max_val).to(
        torch.float64
    )
    inp = x.to(device)
    out = torch.empty((0,), device=device, dtype=torch.float64)
    res = torch.histc(inp, bins=bins, min=min_val, max=max_val, out=out)
    self.assertEqual(out.shape, (bins,))
    self.assertEqual(out.dtype, torch.float64)
    utils.assert_close(res.cpu(), expected)
    self.assertEqual(res.data_ptr(), out.data_ptr())

  def test_avg_pool2d_out_resizing(self):
    x = torch.randn(2, 3, 8, 8)

    def compute(device):
      inp = x.to(device)
      out = torch.empty((0,), device=device, dtype=torch.float32)
      return torch.ops.aten.avg_pool2d.out(
          inp, [2, 2], [2, 2], [0, 0], False, True, None, out=out
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_avg_pool3d_out_resizing(self):
    x = torch.randn(2, 3, 6, 6, 6)

    def compute(device):
      inp = x.to(device)
      out = torch.empty((0,), device=device, dtype=torch.float32)
      return torch.ops.aten.avg_pool3d.out(
          inp, [2, 2, 2], [2, 2, 2], [0, 0, 0], False, True, None, out=out
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_thnn_fused_gru_cell_out_resizing(self):
    device = torch.device("tpu")
    ig = torch.randn(2, 6, device=device)
    hg = torch.randn(2, 6, device=device)
    hx = torch.randn(2, 2, device=device)
    expected0, expected1 = torch.ops.aten._thnn_fused_gru_cell(ig, hg, hx)
    out0 = torch.empty((0,), device=device, dtype=torch.float32)
    out1 = torch.empty((0,), device=device, dtype=torch.float32)
    res0, res1 = torch.ops.aten._thnn_fused_gru_cell.out(
        ig, hg, hx, None, None, out0=out0, out1=out1
    )
    self.assertEqual(out0.shape, (2, 2))
    self.assertEqual(out1.shape, (2, 10))
    utils.assert_close(res0, expected0)
    utils.assert_close(res1, expected1)
    self.assertEqual(res0.data_ptr(), out0.data_ptr())
    self.assertEqual(res1.data_ptr(), out1.data_ptr())

  def test_thnn_fused_lstm_cell_out_resizing(self):
    device = torch.device("tpu")
    ig = torch.randn(2, 8, device=device)
    hg = torch.randn(2, 8, device=device)
    cx = torch.randn(2, 2, device=device)
    expected0, expected1, expected2 = torch.ops.aten._thnn_fused_lstm_cell(
        ig, hg, cx
    )
    out0 = torch.empty((0,), device=device, dtype=torch.float32)
    out1 = torch.empty((0,), device=device, dtype=torch.float32)
    out2 = torch.empty((0,), device=device, dtype=torch.float32)
    res0, res1, res2 = torch.ops.aten._thnn_fused_lstm_cell.out(
        ig, hg, cx, None, None, out0=out0, out1=out1, out2=out2
    )
    self.assertEqual(out0.shape, (2, 2))
    self.assertEqual(out1.shape, (2, 2))
    self.assertEqual(out2.shape, (2, 8))
    utils.assert_close(res0, expected0)
    utils.assert_close(res1, expected1)
    utils.assert_close(res2, expected2)
    self.assertEqual(res0.data_ptr(), out0.data_ptr())
    self.assertEqual(res1.data_ptr(), out1.data_ptr())
    self.assertEqual(res2.data_ptr(), out2.data_ptr())

  @parameterized.product(
      shape=[(2, 3, 16), (3, 20)],
      kernel_size=[2, 3],
      stride=[None, 1, 2],
      padding=[0, 1],
      ceil_mode=[False, True],
      count_include_pad=[False, True],
      dtype=[torch.float32, torch.bfloat16],
  )
  def test_avg_pool1d(
      self,
      shape,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      dtype,
  ):
    torch.manual_seed(42)
    input_val = torch.randn(shape, dtype=dtype)

    def compute(device):
      kwargs = {
          "kernel_size": kernel_size,
          "padding": padding,
          "ceil_mode": ceil_mode,
          "count_include_pad": count_include_pad,
      }
      if stride is not None:
        kwargs["stride"] = stride
      return torch.nn.functional.avg_pool1d(input_val.to(device), **kwargs)

    atol = 5e-3 if dtype == torch.bfloat16 else None
    self.assert_close_tpu_vs_cpu(compute, atol=atol)

  def test_avg_pool1d_backward_empty_batch(self):
    x_cpu = torch.randn(0, 3, 16, requires_grad=True)
    x_tpu = x_cpu.detach().to("tpu").requires_grad_(True)

    out_cpu = torch.nn.functional.avg_pool1d(x_cpu, 2)
    out_tpu = torch.nn.functional.avg_pool1d(x_tpu, 2)
    self.assertIsNotNone(out_tpu.grad_fn)
    self.assertEqual(out_tpu.shape, out_cpu.shape)

    out_cpu.sum().backward()
    out_tpu.sum().backward()
    self.assertEqual(x_tpu.grad.shape, x_cpu.grad.shape)

  def test_avg_pool1d_invalid_args_with_grad(self):
    x = torch.randn(2, 3, 16)

    def compute(device):
      x_device = x.detach().to(device).requires_grad_(True)
      # Padding larger than half the kernel size is rejected.
      return torch.nn.functional.avg_pool1d(x_device, 2, padding=2)

    self.assert_close_tpu_vs_cpu(compute, allow_failure=True)

  def test_avg_pool1d_vmap(self):
    torch.manual_seed(42)
    x = torch.randn(4, 3, 16)

    def compute(device):
      return torch.vmap(lambda t: torch.nn.functional.avg_pool1d(t, 2))(
          x.to(device)
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_avg_pool1d_double_backward(self):
    torch.manual_seed(42)
    x_cpu = torch.randn(2, 3, 16, requires_grad=True)
    x_tpu = x_cpu.detach().to("tpu").requires_grad_(True)

    def grad_of_grad(x):
      out = torch.nn.functional.avg_pool1d(x, 3, stride=2, padding=1)
      (grad,) = torch.autograd.grad(out.pow(2).sum(), x, create_graph=True)
      grad.pow(2).sum().backward()
      return x.grad

    utils.assert_close(
        actual=grad_of_grad(x_tpu).cpu(), expected=grad_of_grad(x_cpu)
    )

  def test_avg_pool1d_forward_ad(self):
    torch.manual_seed(42)
    x = torch.randn(2, 3, 16)
    tangent = torch.randn_like(x)

    def jvp(device):
      with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(
            x.to(device), tangent.to(device)
        )
        out = torch.nn.functional.avg_pool1d(dual, 3, stride=2, padding=1)
        return torch.autograd.forward_ad.unpack_dual(out).tangent

    utils.assert_close(actual=jvp("tpu").cpu(), expected=jvp("cpu"))

  def test_avg_pool1d_backward_compiled(self):
    def pool(x):
      return torch.nn.functional.avg_pool1d(x, 3, stride=2, padding=1)

    compiled_pool = torch.compile(
        pool, backend="tpu", fullgraph=True, dynamic=False
    )
    torch.manual_seed(42)
    for length in (16, 24):
      x_cpu = torch.randn(2, 3, length, requires_grad=True)
      x_tpu = x_cpu.detach().to("tpu").requires_grad_(True)

      out_cpu = pool(x_cpu)
      out_tpu = compiled_pool(x_tpu)
      self.assertIsNotNone(out_tpu.grad_fn)
      utils.assert_close(actual=out_tpu.cpu(), expected=out_cpu)

      grad_out = torch.randn_like(out_cpu)
      out_cpu.backward(grad_out)
      out_tpu.backward(grad_out.to("tpu"))
      utils.assert_close(actual=x_tpu.grad.cpu(), expected=x_cpu.grad)

  @parameterized.product(
      shape=[(2, 3, 16), (3, 20)],
      kernel_size=[2, 3],
      stride=[[], [1], [2]],
      padding=[[0], [1]],
      ceil_mode=[False, True],
      count_include_pad=[False, True],
      dtype=[torch.float32, torch.bfloat16],
  )
  def test_avg_pool1d_out(
      self,
      shape,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      dtype,
  ):
    torch.manual_seed(42)
    input_val = torch.randn(shape, dtype=dtype)

    def compute(device):
      x = input_val.to(device)
      out = torch.empty(0, dtype=dtype, device=device)
      res = torch.ops.aten.avg_pool1d.out(
          x,
          [kernel_size],
          stride,
          padding,
          ceil_mode,
          count_include_pad,
          out=out,
      )
      self.assertIs(res, out)
      return out

    atol = 5e-3 if dtype == torch.bfloat16 else None
    self.assert_close_tpu_vs_cpu(compute, atol=atol)

  def test_avg_pool_corner_cases(self):
    """Tests corner cases for avg_pool1d, avg_pool2d, and avg_pool3d:

    - kernel_size=1, stride=1, padding=0 (window_size=1 no-op scaling).
    - divisor_override=1 (no-op scaling with explicit divisor override).
    - divisor_override > 1 with low-precision dtype (reciprocal multiplication).
    - Unbatched 2D (C, H, W) and 3D (C, D, H, W) inputs.
    - count_include_pad=False with padding on borders.
    """
    torch.manual_seed(42)

    # 1. Window size == 1 (no-op scaling) on 2D bfloat16
    x_2d_bf16 = torch.randn(2, 4, 8, 8, dtype=torch.bfloat16)
    self.assert_close_tpu_vs_cpu(
        lambda dev: torch.nn.functional.avg_pool2d(
            x_2d_bf16.to(dev), kernel_size=1, stride=1, padding=0
        )
    )

    # 2. Divisor override == 1 (no-op scaling) on 2D float32
    x_2d_f32 = torch.randn(2, 4, 8, 8, dtype=torch.float32)
    self.assert_close_tpu_vs_cpu(
        lambda dev: torch.nn.functional.avg_pool2d(
            x_2d_f32.to(dev), kernel_size=2, stride=2, divisor_override=1
        )
    )

    # 3. Divisor override > 1 on 2D bfloat16 (reciprocal multiplication)
    self.assert_close_tpu_vs_cpu(
        lambda dev: torch.nn.functional.avg_pool2d(
            x_2d_bf16.to(dev), kernel_size=2, stride=2, divisor_override=5
        ),
        atol=5e-3,
    )

    # 4. Unbatched 2D input (C, H, W) and 3D input (C, D, H, W)
    x_unbatched_2d = torch.randn(3, 10, 10, dtype=torch.float32)
    self.assert_close_tpu_vs_cpu(
        lambda dev: torch.nn.functional.avg_pool2d(
            x_unbatched_2d.to(dev), kernel_size=3, stride=2, padding=1
        )
    )

    x_unbatched_3d = torch.randn(3, 6, 6, 6, dtype=torch.float32)
    self.assert_close_tpu_vs_cpu(
        lambda dev: torch.nn.functional.avg_pool3d(
            x_unbatched_3d.to(dev),
            kernel_size=2,
            stride=2,
            divisor_override=3,
        )
    )

    # 5. count_include_pad=False with padding on 2D and 3D
    self.assert_close_tpu_vs_cpu(
        lambda dev: torch.nn.functional.avg_pool2d(
            x_2d_f32.to(dev),
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
    )

  def test_adaptive_avg_pool_corner_cases(self):
    """Tests corner cases for adaptive_avg_pool2d and adaptive_avg_pool3d:

    - Divisible shapes triggering std_avg_pool fast path in forward and
    backward.
    - Non-divisible shapes triggering Gather/Scatter with variable window
    lengths.
    - Output size equal to input size (max_k == 1 no-op scaling fast path).
    - Low-precision bfloat16 and float16 dtypes.
    - 3D adaptive pooling divisible and non-divisible shapes.
    - Backward gradient checks for both fast-path and scatter paths.
    """
    torch.manual_seed(42)

    # 1. Forward + Backward: Divisible shapes (fast path std_avg_pool)
    x_2d_div = torch.randn(2, 3, 8, 8, dtype=torch.float32, requires_grad=True)

    def run_divisible_2d(dev):
      x = x_2d_div.detach().clone().to(dev).requires_grad_(True)
      out = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
      out.backward(torch.ones_like(out))
      return out, x.grad

    self.assert_close_tpu_vs_cpu(run_divisible_2d)

    # 2. Forward + Backward: max_k == 1 (out_size == in_size, identity scaling)
    x_2d_same = torch.randn(2, 3, 5, 5, dtype=torch.float32, requires_grad=True)

    def run_same_size_2d(dev):
      x = x_2d_same.detach().clone().to(dev).requires_grad_(True)
      out = torch.nn.functional.adaptive_avg_pool2d(x, (5, 5))
      out.backward(torch.ones_like(out))
      return out, x.grad

    self.assert_close_tpu_vs_cpu(run_same_size_2d)

    # 3. Forward + Backward: Non-divisible shapes (GatherPool / ScatterAdd path)
    x_2d_nondiv = torch.randn(
        2, 2, 7, 10, dtype=torch.float32, requires_grad=True
    )

    def run_nondivisible_2d(dev):
      x = x_2d_nondiv.detach().clone().to(dev).requires_grad_(True)
      out = torch.nn.functional.adaptive_avg_pool2d(x, (3, 4))
      out.backward(torch.ones_like(out))
      return out, x.grad

    self.assert_close_tpu_vs_cpu(run_nondivisible_2d)

    # 4. Native low-precision bfloat16 with divisible shape (fast path)
    x_2d_bf16 = torch.randn(
        2, 3, 6, 6, dtype=torch.bfloat16, requires_grad=True
    )

    def run_bf16_divisible_2d(dev):
      x = x_2d_bf16.detach().clone().to(dev).requires_grad_(True)
      out = torch.nn.functional.adaptive_avg_pool2d(x, (3, 3))
      out.backward(torch.ones_like(out))
      return out, x.grad

    self.assert_close_tpu_vs_cpu(run_bf16_divisible_2d, rtol=2e-2, atol=2e-2)

    # 5. 3D Adaptive pooling: Divisible shape (fast path std_avg_pool)
    x_3d_div = torch.randn(
        1, 2, 4, 6, 8, dtype=torch.float32, requires_grad=True
    )

    def run_divisible_3d(dev):
      x = x_3d_div.detach().clone().to(dev).requires_grad_(True)
      out = torch.nn.functional.adaptive_avg_pool3d(x, (2, 3, 4))
      out.backward(torch.ones_like(out))
      return out, x.grad

    self.assert_close_tpu_vs_cpu(run_divisible_3d)

    # 6. 3D Adaptive pooling: Non-divisible shape
    x_3d_nondiv = torch.randn(
        1, 2, 5, 5, 5, dtype=torch.float32, requires_grad=True
    )

    def run_nondivisible_3d(dev):
      x = x_3d_nondiv.detach().clone().to(dev).requires_grad_(True)
      out = torch.nn.functional.adaptive_avg_pool3d(x, (2, 2, 2))
      out.backward(torch.ones_like(out))
      return out, x.grad

    self.assert_close_tpu_vs_cpu(run_nondivisible_3d)

    # 7. 2D Adaptive pooling: Zero spatial output_size and zero batch size
    x_2d_zero = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    x_2d_zero_batch = torch.randn(0, 3, 4, 4, dtype=torch.float32)

    def run_zero_output_2d(dev):
      x = x_2d_zero.detach().clone().to(dev)
      out1 = torch.nn.functional.adaptive_avg_pool2d(x, (0, 0))
      out2 = torch.empty(0, dtype=torch.float32, device=dev)
      torch.ops.aten.adaptive_avg_pool2d.out(x, [0, 2], out=out2)
      out3 = torch.nn.functional.adaptive_avg_pool2d(x[0], (0, 0))
      x_zero_batch = (
          x_2d_zero_batch.detach().clone().to(dev).requires_grad_(True)
      )
      out4 = torch.nn.functional.adaptive_avg_pool2d(x_zero_batch, (2, 2))
      out4.backward(torch.ones_like(out4))
      return out1, out2, out3, out4, x_zero_batch.grad

    self.assert_close_tpu_vs_cpu(run_zero_output_2d)

    # 7. 3D Adaptive pooling: Zero spatial output_size and zero batch size
    x_3d_zero = torch.randn(2, 2, 4, 4, 4, dtype=torch.float32)
    x_3d_zero_batch = torch.randn(0, 2, 4, 4, 4, dtype=torch.float32)

    def run_zero_output_3d(dev):
      x = x_3d_zero.detach().clone().to(dev)
      out1 = torch.nn.functional.adaptive_avg_pool3d(x, (0, 0, 0))
      out2 = torch.empty(0, dtype=torch.float32, device=dev)
      torch.ops.aten.adaptive_avg_pool3d.out(x, [0, 2, 2], out=out2)
      x_zero_batch = (
          x_3d_zero_batch.detach().clone().to(dev).requires_grad_(True)
      )
      out3 = torch.nn.functional.adaptive_avg_pool3d(x_zero_batch, (2, 2, 2))
      out3.backward(torch.ones_like(out3))
      return out1, out2, out3, x_zero_batch.grad

    self.assert_close_tpu_vs_cpu(run_zero_output_3d)

  def test_ldexp_large_exponent(self):
    def compute(device):
      exponent = 127
      t_mant = torch.tensor([1.0], dtype=torch.float32, device=device)
      t_exp = torch.tensor([exponent], dtype=torch.int32, device=device)
      return torch.ldexp(t_mant, t_exp)

    self.assert_close_tpu_vs_cpu(compute)

  def test_pow_large_int(self):
    def compute(device):
      return torch.pow(
          2.0, torch.tensor([127], dtype=torch.int32, device=device)
      )

    self.assert_close_tpu_vs_cpu(compute)

  @parameterized.parameters([(True), (False)])
  def test_pow_bool_tensor_scalar(self, exponent):
    def compute(device):
      return torch.pow(
          torch.tensor([True, False], dtype=torch.bool, device=device), exponent
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_pow_bool_scalar_tensor(self):
    def compute(device):
      return torch.pow(
          True, torch.tensor([True, False], dtype=torch.bool, device=device)
      )

    self.assert_close_tpu_vs_cpu(
        compute, check_exception_type=False, allow_failure=True
    )

  def test_nll_loss_backward(self):
    inp = torch.randn(2, 2)
    target = torch.ones(2, dtype=torch.uint8)
    output = torch.empty(1)
    total_weight = torch.empty(1)

    weight = None
    reduction = 1
    ignore_index = -100

    def compute(device):
      return torch.ops.aten.nll_loss_forward(
          inp.to(device),
          target.to(device),
          weight,
          reduction,
          ignore_index,
          output=output.to(device),
          total_weight=total_weight.to(device),
      )

    self.assert_close_tpu_vs_cpu(compute)

  @parameterized.product(
      shape=[(4,), (3, 4), (2, 4, 3), (2, 4, 2, 1), (2, 3, 2, 1, 1)],
      dtype=[torch.float32, torch.bfloat16, torch.float16],
      target_dtype=[torch.int64, torch.uint8],
      reduction=["none", "mean", "sum"],
      has_weight=[False, True],
  )
  def test_nll_loss_comprehensive(
      self, shape, dtype, target_dtype, reduction, has_weight
  ):
    g = torch.Generator().manual_seed(42)
    c = shape[0] if len(shape) == 1 else shape[1]
    target_shape = () if len(shape) == 1 else (shape[0],) + shape[2:]
    inp = torch.randn(shape, generator=g, dtype=dtype).log_softmax(
        dim=0 if len(shape) == 1 else 1
    )
    target = torch.randint(0, c, target_shape, generator=g, dtype=target_dtype)
    weight = torch.rand(c, generator=g, dtype=dtype) if has_weight else None

    def compute(device):
      x = inp.to(device).detach().clone().requires_grad_(True)
      w = weight.to(device) if weight is not None else None
      # PyTorch CPU nll_loss2d (rank >= 3) only implements int64 targets,
      # whereas TPU supports both int64 and uint8 across all ranks.
      t_dtype = (
          torch.int64
          if torch.device(device).type == "cpu" and len(shape) >= 3
          else target_dtype
      )
      loss = torch.nn.functional.nll_loss(
          x, target.to(device, dtype=t_dtype), weight=w, reduction=reduction
      )
      if reduction == "none":
        loss.sum().backward()
      else:
        loss.backward()
      return loss, x.grad

    self.assert_close_tpu_vs_cpu(compute)

  def test_nll_loss_ignore_index_and_edge_cases(self):
    g = torch.Generator().manual_seed(123)

    # 1. Active class ignore_index across 1D..4D with weight and all reductions.
    for shape in [(4,), (3, 4), (2, 4, 3), (2, 4, 2, 2)]:
      c = shape[0] if len(shape) == 1 else shape[1]
      target_shape = () if len(shape) == 1 else (shape[0],) + shape[2:]
      inp = torch.randn(shape, generator=g)
      target = torch.randint(0, c, target_shape, generator=g, dtype=torch.int64)
      weight = torch.rand(c, generator=g)
      for reduction in ["none", "mean", "sum"]:
        with self.subTest(
            case="active_ignore", shape=shape, reduction=reduction
        ):

          def compute_active(
              device, inp=inp, target=target, weight=weight, reduction=reduction
          ):
            x = inp.to(device).detach().clone().requires_grad_(True)
            loss = torch.nn.functional.nll_loss(
                x,
                target.to(device),
                weight=weight.to(device),
                reduction=reduction,
                ignore_index=1,
            )
            (loss.sum() if reduction == "none" else loss).backward()
            return loss, x.grad

          self.assert_close_tpu_vs_cpu(compute_active)

    # 2. Out-of-bounds ignore_index when class 0 has -inf (tests Select masking).
    for shape in [(3, 4), (2, 4, 3, 2)]:
      target_shape = (shape[0],) + shape[2:]
      inp = torch.randn(shape, generator=g)
      inp[:, 0] = -float("inf")
      target = torch.ones(target_shape, dtype=torch.int64)
      target.view(-1)[0] = 999
      for reduction in ["none", "mean", "sum"]:
        with self.subTest(
            case="oob_ignore_neg_inf", shape=shape, reduction=reduction
        ):

          def compute_oob(device, inp=inp, target=target, reduction=reduction):
            x = inp.to(device).detach().clone().requires_grad_(True)
            loss = torch.nn.functional.nll_loss(
                x,
                target.to(device),
                reduction=reduction,
                ignore_index=999,
            )
            (loss.sum() if reduction == "none" else loss).backward()
            return loss, x.grad

          self.assert_close_tpu_vs_cpu(compute_oob)

    # 3. All targets ignored under reduction='mean' (loss is NaN, grad is 0).
    for shape in [(4,), (3, 4), (2, 4, 3), (2, 4, 2, 2)]:
      c = shape[0] if len(shape) == 1 else shape[1]
      target_shape = () if len(shape) == 1 else (shape[0],) + shape[2:]
      inp = torch.randn(shape, generator=g)
      target = torch.full(target_shape, -100, dtype=torch.int64)
      for has_weight in [False, True]:
        weight = torch.rand(c, generator=g) if has_weight else None
        with self.subTest(
            case="all_ignored_mean", shape=shape, has_weight=has_weight
        ):

          def compute_all_ignored(
              device, inp=inp, target=target, weight=weight
          ):
            x = inp.to(device).detach().clone().requires_grad_(True)
            w = weight.to(device) if weight is not None else None
            loss = torch.nn.functional.nll_loss(
                x,
                target.to(device),
                weight=w,
                reduction="mean",
                ignore_index=-100,
            )
            loss.backward()
            return loss, x.grad

          self.assert_close_tpu_vs_cpu(compute_all_ignored)

    # 4. Zero-element tensors across 2D, 3D, 4D and reductions.
    for shape, target_shape in [
        ((0, 4), (0,)),
        ((2, 4, 0), (2, 0)),
        ((2, 4, 0, 3), (2, 0, 3)),
    ]:
      inp = torch.empty(shape, dtype=torch.float32)
      target = torch.empty(target_shape, dtype=torch.int64)
      for reduction in ["none", "mean", "sum"]:
        with self.subTest(case="empty", shape=shape, reduction=reduction):

          def compute_empty(
              device, inp=inp, target=target, reduction=reduction
          ):
            x = inp.to(device).detach().clone().requires_grad_(True)
            loss = torch.nn.functional.nll_loss(
                x, target.to(device), reduction=reduction
            )
            (loss.sum() if reduction == "none" else loss).backward()
            return loss, x.grad

          self.assert_close_tpu_vs_cpu(compute_empty)

  def test_nll_loss_out_and_2d_variants(self):
    g = torch.Generator().manual_seed(7)
    # 1. nll_loss_forward.output and nll_loss_backward.grad_input (with resize)
    inp_2d = torch.randn(3, 4, generator=g).log_softmax(dim=1)
    target_2d = torch.tensor([0, 2, 1], dtype=torch.int64)
    weight_2d = torch.rand(4, generator=g)
    for reduction in [0, 1, 2]:
      with self.subTest(op="nll_loss_out", reduction=reduction):

        def compute_1d_out(device, reduction=reduction):
          out = torch.empty((9, 9), dtype=torch.float32, device=device)
          tw = torch.empty((5,), dtype=torch.float32, device=device)
          torch.ops.aten.nll_loss_forward.output(
              inp_2d.to(device),
              target_2d.to(device),
              weight_2d.to(device),
              reduction,
              -100,
              output=out,
              total_weight=tw,
          )
          grad_out = torch.ones_like(out)
          grad_in = torch.empty((1,), dtype=torch.float32, device=device)
          torch.ops.aten.nll_loss_backward.grad_input(
              grad_out,
              inp_2d.to(device),
              target_2d.to(device),
              weight_2d.to(device),
              reduction,
              -100,
              tw,
              grad_input=grad_in,
          )
          return out, tw, grad_in

        self.assert_close_tpu_vs_cpu(compute_1d_out)

    # 2. nll_loss2d_forward.output and nll_loss2d_backward.grad_input (with resize and buffer donation)
    inp_4d = torch.randn(2, 3, 4, 4, generator=g).log_softmax(dim=1)
    target_4d = torch.randint(0, 3, (2, 4, 4), generator=g, dtype=torch.int64)
    weight_4d = torch.rand(3, generator=g)
    for reduction in [0, 1, 2]:
      with self.subTest(op="nll_loss2d_out", reduction=reduction):

        def compute_2d_out(device, reduction=reduction):
          out = torch.empty((7,), dtype=torch.float32, device=device)
          tw = torch.empty((3, 3), dtype=torch.float32, device=device)
          torch.ops.aten.nll_loss2d_forward.output(
              inp_4d.to(device),
              target_4d.to(device),
              weight_4d.to(device),
              reduction,
              -100,
              output=out,
              total_weight=tw,
          )
          grad_out = torch.ones_like(out)
          # Also test buffer donation where grad_input aliases self
          x_donate = inp_4d.to(device).clone()
          torch.ops.aten.nll_loss2d_backward.grad_input(
              grad_out,
              x_donate,
              target_4d.to(device),
              weight_4d.to(device),
              reduction,
              -100,
              tw,
              grad_input=x_donate,
          )
          return out, tw, x_donate

        self.assert_close_tpu_vs_cpu(compute_2d_out)

  def _make_well_separated_logits(self, shape, dtype, generator):
    c = shape[0] if len(shape) == 1 else shape[1]
    class_shape = (c,) if len(shape) == 1 else (1, c) + (1,) * (len(shape) - 2)
    class_ramp = (
        torch.arange(c, dtype=torch.float32).view(class_shape) - (c - 1) / 2.0
    ) * 16.0
    noise = (
        torch.randint(-3, 4, shape, generator=generator, dtype=torch.int64).to(
            torch.float32
        )
        * 0.125
    )
    return (class_ramp + noise).to(dtype)

  @parameterized.product(
      shape=[(4,), (3, 4), (2, 4, 3), (2, 4, 2, 1), (2, 4, 2, 1, 1)],
      dtype=[torch.float32, torch.bfloat16, torch.float16],
      reduction=["none", "mean", "sum"],
      has_weight=[False, True],
      label_smoothing=[0.0, 0.25],
  )
  def test_cross_entropy_class_indices_comprehensive(
      self, shape, dtype, reduction, has_weight, label_smoothing
  ):
    g = torch.Generator().manual_seed(99)
    c = shape[0] if len(shape) == 1 else shape[1]
    target_shape = () if len(shape) == 1 else (shape[0],) + shape[2:]
    inp = self._make_well_separated_logits(shape, dtype, g)
    target = torch.randint(0, c, target_shape, generator=g, dtype=torch.int64)
    weight = (
        torch.tensor([0.5, 1.0, 1.5, 2.0][:c], dtype=dtype)
        if has_weight
        else None
    )

    def compute(device):
      x = inp.to(device).detach().clone().requires_grad_(True)
      w = weight.to(device) if weight is not None else None
      loss = torch.nn.functional.cross_entropy(
          x,
          target.to(device),
          weight=w,
          reduction=reduction,
          ignore_index=1 if len(shape) > 1 else -100,
          label_smoothing=label_smoothing,
      )
      (loss.sum() if reduction == "none" else loss).backward()
      return loss, x.grad

    self.assert_close_tpu_vs_cpu(compute)

  @parameterized.product(
      shape=[(4,), (3, 4), (2, 4, 3), (2, 4, 2, 1), (2, 4, 2, 1, 1)],
      dtype=[torch.float32, torch.bfloat16, torch.float16],
      reduction=["none", "mean", "sum"],
      has_weight=[False, True],
      label_smoothing=[0.0, 0.25],
  )
  def test_cross_entropy_prob_targets_comprehensive(
      self, shape, dtype, reduction, has_weight, label_smoothing
  ):
    g = torch.Generator().manual_seed(101)
    c = shape[0] if len(shape) == 1 else shape[1]
    inp = self._make_well_separated_logits(shape, dtype, g)
    target = (
        torch.randint(1, 4, shape, generator=g, dtype=torch.int64).to(dtype)
        * 0.25
    )
    weight = (
        torch.tensor([0.5, 1.0, 1.5, 2.0][:c], dtype=dtype)
        if has_weight
        else None
    )

    def compute(device):
      x = inp.to(device).detach().clone().requires_grad_(True)
      t = target.to(device).detach().clone().requires_grad_(True)
      w = weight.to(device) if weight is not None else None
      loss = torch.nn.functional.cross_entropy(
          x,
          t,
          weight=w,
          reduction=reduction,
          label_smoothing=label_smoothing,
      )
      (loss.sum() if reduction == "none" else loss).backward()
      return loss, x.grad, t.grad

    self.assert_close_tpu_vs_cpu(compute)

  def test_cross_entropy_edge_cases(self):
    g = torch.Generator().manual_seed(202)

    # 1. uint8 class index targets with label_smoothing and weight.
    inp_u8 = self._make_well_separated_logits((2, 4, 2), torch.float32, g)
    target_u8 = torch.randint(0, 4, (2, 2), generator=g, dtype=torch.uint8)
    weight_u8 = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32)
    with self.subTest(case="uint8_target"):

      def compute_u8(device):
        x = inp_u8.to(device).detach().clone().requires_grad_(True)
        # PyTorch CPU gather/nll_loss2d requires int64 targets for 3D + label_smoothing,
        # whereas TPU supports uint8 directly.
        t_dtype = (
            torch.int64 if torch.device(device).type == "cpu" else torch.uint8
        )
        loss = torch.nn.functional.cross_entropy(
            x,
            target_u8.to(device, dtype=t_dtype),
            weight=weight_u8.to(device),
            reduction="mean",
            ignore_index=2,
            label_smoothing=0.25,
        )
        loss.backward()
        return loss, x.grad

      self.assert_close_tpu_vs_cpu(compute_u8)

    # 2. Target-only requires_grad (self.requires_grad=False) in prob-target mode.
    inp_prob = self._make_well_separated_logits((2, 4, 2), torch.float32, g)
    target_prob = (
        torch.randint(1, 4, (2, 4, 2), generator=g, dtype=torch.int64).to(
            torch.float32
        )
        * 0.25
    )
    for reduction in ["none", "mean", "sum"]:
      with self.subTest(case="target_only_grad", reduction=reduction):

        def compute_target_grad(device, reduction=reduction):
          x = inp_prob.to(device).detach().clone()
          t = target_prob.to(device).detach().clone().requires_grad_(True)
          loss = torch.nn.functional.cross_entropy(
              x,
              t,
              weight=weight_u8.to(device),
              reduction=reduction,
              label_smoothing=0.25,
          )
          (loss.sum() if reduction == "none" else loss).backward()
          return loss, t.grad

        self.assert_close_tpu_vs_cpu(compute_target_grad)

    # 3. All targets ignored under reduction='mean' (with and without label_smoothing and weight).
    for shape in [(4,), (3, 4), (2, 4, 3)]:
      c = shape[0] if len(shape) == 1 else shape[1]
      target_shape = () if len(shape) == 1 else (shape[0],) + shape[2:]
      inp = self._make_well_separated_logits(shape, torch.float32, g)
      target = torch.full(target_shape, -100, dtype=torch.int64)
      for ls in [0.0, 0.25]:
        for has_weight in [False, True]:
          w = (
              torch.tensor([0.5, 1.0, 1.5, 2.0][:c], dtype=torch.float32)
              if has_weight
              else None
          )
          with self.subTest(
              case="all_ignored_mean",
              shape=shape,
              label_smoothing=ls,
              has_weight=has_weight,
          ):

            def compute_all_ignored(device, inp=inp, target=target, w=w, ls=ls):
              x = inp.to(device).detach().clone().requires_grad_(True)
              w_dev = w.to(device) if w is not None else None
              loss = torch.nn.functional.cross_entropy(
                  x,
                  target.to(device),
                  weight=w_dev,
                  reduction="mean",
                  ignore_index=-100,
                  label_smoothing=ls,
              )
              loss.backward()
              return loss, x.grad

            self.assert_close_tpu_vs_cpu(compute_all_ignored)

    # 4. Out-of-bounds ignore_index (e.g. 999) with label_smoothing.
    inp_oob = self._make_well_separated_logits((3, 4, 2), torch.float32, g)
    target_oob = torch.randint(0, 4, (3, 2), generator=g, dtype=torch.int64)
    target_oob[0, 0] = 999
    for reduction in ["none", "mean", "sum"]:
      with self.subTest(case="oob_ignore", reduction=reduction):

        def compute_oob(device, reduction=reduction):
          x = inp_oob.to(device).detach().clone().requires_grad_(True)
          loss = torch.nn.functional.cross_entropy(
              x,
              target_oob.to(device),
              reduction=reduction,
              ignore_index=999,
              label_smoothing=0.25,
          )
          (loss.sum() if reduction == "none" else loss).backward()
          return loss, x.grad

        self.assert_close_tpu_vs_cpu(compute_oob)

    # 5. Zero-element tensors for both class indices and probability targets.
    for shape, target_shape in [
        ((0, 4), (0,)),
        ((2, 4, 0), (2, 0)),
        ((2, 4, 0, 3), (2, 0, 3)),
    ]:
      inp_empty = torch.empty(shape, dtype=torch.float32)
      target_idx_empty = torch.empty(target_shape, dtype=torch.int64)
      target_prob_empty = torch.empty(shape, dtype=torch.float32)
      for reduction in ["none", "mean", "sum"]:
        with self.subTest(
            case="empty_class_indices", shape=shape, reduction=reduction
        ):

          def compute_empty_idx(
              device,
              inp_empty=inp_empty,
              target_idx_empty=target_idx_empty,
              reduction=reduction,
          ):
            x = inp_empty.to(device).detach().clone().requires_grad_(True)
            loss = torch.nn.functional.cross_entropy(
                x,
                target_idx_empty.to(device),
                reduction=reduction,
                label_smoothing=0.1,
            )
            (loss.sum() if reduction == "none" else loss).backward()
            return loss, x.grad

          self.assert_close_tpu_vs_cpu(compute_empty_idx)

        with self.subTest(
            case="empty_prob_targets", shape=shape, reduction=reduction
        ):

          def compute_empty_prob(
              device,
              inp_empty=inp_empty,
              target_prob_empty=target_prob_empty,
              reduction=reduction,
          ):
            x = inp_empty.to(device).detach().clone().requires_grad_(True)
            t = (
                target_prob_empty.to(device)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            loss = torch.nn.functional.cross_entropy(
                x, t, reduction=reduction, label_smoothing=0.1
            )
            (loss.sum() if reduction == "none" else loss).backward()
            return loss, x.grad, t.grad

          self.assert_close_tpu_vs_cpu(compute_empty_prob)

  @parameterized.product(
      shape=[(100,), (0,), (2, 10, 5), (2, 2, 2, 2), (1, 0, 2)],
      return_inverse=[True, False],
      return_counts=[True, False],
      dtype=[
          torch.float32,
          torch.float64,
          torch.bfloat16,
          torch.float16,
          torch.int64,
          torch.int32,
          torch.int16,
          torch.int8,
          torch.uint8,
          torch.bool,
      ],
  )
  def test_unique2(self, shape, return_inverse, return_counts, dtype):
    if dtype.is_floating_point:
      input_value = torch.randn(shape, dtype=dtype)
    elif dtype == torch.bool:
      input_value = torch.randint(0, 2, shape, dtype=torch.uint8).to(torch.bool)
    else:
      input_value = torch.randint(0, 10, shape, dtype=dtype)

    def compute(device):
      return torch._unique2(
          input_value.to(device),
          sorted=True,
          return_inverse=return_inverse,
          return_counts=return_counts,
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_unique2_distributions(self):
    for dtype in [torch.float32, torch.int64]:
      for distribution in [
          "all_same",
          "all_different",
          "sorted",
          "reverse_sorted",
      ]:
        if distribution == "all_same":
          input_value = torch.full((100,), 42, dtype=dtype)
        elif distribution == "all_different":
          input_value = torch.arange(100, dtype=dtype)
        elif distribution == "sorted":
          input_value = torch.sort(torch.randint(0, 50, (100,), dtype=dtype))[0]
        elif distribution == "reverse_sorted":
          input_value = torch.sort(
              torch.randint(0, 50, (100,), dtype=dtype), descending=True
          )[0]
        else:
          raise ValueError(f"Unknown distribution: {distribution}")

        def compute(device, input_value=input_value):
          return torch._unique2(
              input_value.to(device),
              sorted=True,
              return_inverse=True,
              return_counts=True,
          )

        with self.subTest(dtype=dtype, distribution=distribution):
          self.assert_close_tpu_vs_cpu(compute)

  def test_unique2_nan(self):
    # Multiple NaNs collapse into a single NaN, sorted at the end.
    # Inverse indices for NaNs are documented to be unpredictable,
    # so we omit them from the CPU vs TPU check.
    input_value = torch.tensor([1.0, float("nan"), 2.0, float("nan"), 1.0])

    def compute(device):
      res = torch._unique2(
          input_value.to(device),
          sorted=True,
          return_inverse=True,
          return_counts=True,
      )
      return res[0], res[2]

    self.assert_close_tpu_vs_cpu(compute)

  def test_unique2_infs(self):
    # Tests that +/- Infs are sorted correctly (-Inf < ... < +Inf < NaN)
    # Inverse indices are omitted due to NaN unpredictability.
    input_value = torch.tensor(
        [float("inf"), 1.0, float("-inf"), float("nan"), float("-inf"), 2.0]
    )

    def compute(device):
      res = torch._unique2(
          input_value.to(device),
          sorted=True,
          return_inverse=True,
          return_counts=True,
      )
      return res[0], res[2]

    self.assert_close_tpu_vs_cpu(compute)

  def test_unique2_zeros(self):
    # -0.0 and 0.0 are considered equivalent. The exact returned value
    # depends on the backend, so we check the absolute value instead.
    input_value = torch.tensor([-0.0, 0.0, -0.0, 0.0, 1.0])

    def compute(device):
      res = torch._unique2(
          input_value.to(device),
          sorted=True,
          return_inverse=True,
          return_counts=True,
      )
      # We check out.abs(), inverse_indices, and counts
      return res[0].abs(), res[1], res[2]

    self.assert_close_tpu_vs_cpu(compute)

  def test_unique2_non_contiguous(self):
    # (2, 3) -> (3, 2) but non-contiguous
    input_value = torch.tensor([[1, 2], [3, 4], [1, 2]]).t()

    def compute(device):
      return torch._unique2(
          input_value.to(device),
          sorted=True,
          return_inverse=True,
          return_counts=True,
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_unique2_large(self):
    input_value = torch.randint(0, 1000, (10000,), dtype=torch.int64)

    def compute(device):
      return torch._unique2(
          input_value.to(device),
          sorted=True,
          return_inverse=True,
          return_counts=True,
      )

    self.assert_close_tpu_vs_cpu(compute)

  def test_unique2_more_shapes(self):
    shapes = [(2, 0, 3), (1, 5, 1, 1), (0, 0)]
    for shape in shapes:
      input_value = torch.randint(0, 10, shape, dtype=torch.int32)

      def compute(device, input_value=input_value):
        return torch._unique2(
            input_value.to(device),
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )

      with self.subTest(shape=shape):
        self.assert_close_tpu_vs_cpu(compute)

  def test_unique2_empty_inverse_shape_parity(self):
    """Verifies that unique2 with empty inputs properly materializes inverse_indices.

    Prior to the fix, using return_inverse=True on multidimensional empty inputs
    (like shape [2, 0, 3]) would result in a mismatch between PyTorch's declared
    output shape (which expects rank-3, e.g. [2, 0, 3]) and the backend's
    internally generated zero-sized tensor (which was statically 1D [0]).
    Although empty tensors skip host copy-to-CPU (which hides runtime errors),
    enforcing synchronization in this test guarantees that the TPU backend
    materialization runs and fails without the fix.
    """
    shapes = [(2, 0, 3), (0, 0)]
    for shape in shapes:
      input_value = torch.randint(0, 10, shape, dtype=torch.int32)

      def compute(device, input_value=input_value):
        device = torch.device(device)
        res = torch._unique2(
            input_value.to(device),
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        if device.type == "tpu":
          # Force materialization and waiting of the unique2 output tensors
          # explicitly, which catches and propagates background execution
          # failures for empty/0-element outputs!
          sync.synchronize(list(res), wait=True)
        return res

      with self.subTest(shape=shape):
        self.assert_close_tpu_vs_cpu(compute)

  @parameterized.named_parameters(
      ops_test_data.generate_configs_for_parameterized([
          # Reduce the batch size to max 2 to avoid OOM on smaller devices.
          dataclasses.replace(c, batch_size=min(c.batch_size, 2))
          for c in ops_test_data.SDPA_CONFIGS
      ])
  )
  def test_scaled_dot_product_attention(self, config: ops_test_data.SdpaConfig):
    """Tests torch.nn.functional.scaled_dot_product_attention."""

    torch.manual_seed(5432)
    q = torch.randn(
        config.batch_size,
        config.q_num_heads,
        config.q_seq_len,
        config.qk_head_dim,
        dtype=config.dtype,
    )
    k = torch.randn(
        config.batch_size,
        config.kv_num_heads,
        config.kv_seq_len,
        config.qk_head_dim,
        dtype=config.dtype,
    )
    v = torch.randn(
        config.batch_size,
        config.kv_num_heads,
        config.kv_seq_len,
        config.v_head_dim,
        dtype=config.dtype,
    )

    if config.attn_bias_type is not None:
      bias_shape = (
          config.q_seq_len,
          config.kv_seq_len,
      )
      if config.attn_bias_type is torch.bool:
        attn_mask = torch.randint(
            0,
            2,
            bias_shape,
            dtype=torch.bool,
        )
      elif config.attn_bias_type.is_floating_point:
        attn_mask = torch.randn(*bias_shape, dtype=config.dtype)
    else:
      attn_mask = None

    def compute(device: torch.device):
      q_t = q.clone().detach().to(device)
      k_t = k.clone().detach().to(device)
      v_t = v.clone().detach().to(device)
      q_t.requires_grad_(config.requires_grad)
      k_t.requires_grad_(config.requires_grad)
      v_t.requires_grad_(config.requires_grad)

      with torch.nn.attention.sdpa_kernel(
          torch.nn.attention.SDPBackend.MATH
          if device == "cpu"
          else config.backend
      ):
        y = torch.nn.functional.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask.to(device) if attn_mask is not None else None,
            is_causal=config.is_causal,
            enable_gqa=config.enable_gqa,
            scale=config.scale,
        )

      results = {"y": y}
      if config.requires_grad:
        y.sum().backward()
        results.update({
            "q_grad": q_t.grad,
            "k_grad": k_t.grad,
            "v_grad": v_t.grad,
        })
      return results

    cpu_results = compute("cpu")
    tpu_results = compute("tpu")

    # The gradient of the first token is zero with causal attention.
    if config.is_causal and "q_grad" in cpu_results:
      with self.subTest("causal_query_first_token_grad_is_zero"):
        self.assertTrue((tpu_results["q_grad"][..., 0, :] == 0.0).all())
      cpu_results["q_grad"] = cpu_results["q_grad"][..., 1:, :]
      tpu_results["q_grad"] = tpu_results["q_grad"][..., 1:, :]

      # KV tokens beyond q_seq_len are completely masked out in causal
      # attention, so their gradients are zero.
      if config.kv_seq_len > config.q_seq_len:
        for key in ("k_grad", "v_grad"):
          with self.subTest(f"causal_{key}_masked_tokens_grad_is_zero"):
            self.assertTrue(
                (tpu_results[key][..., config.q_seq_len :, :] == 0.0).all()
            )
          cpu_results[key] = cpu_results[key][..., : config.q_seq_len, :]
          tpu_results[key] = tpu_results[key][..., : config.q_seq_len, :]

    def check_sim(cpu, tpu, key):
      # TODO(willfroom): We should also check the min similarity but currently
      # the flash implementation has a few outliers that we need to investigate
      # first.
      mean_sim_bound = 0.9999
      sim = torch.nn.functional.cosine_similarity(cpu, tpu, dim=-1)
      # Rows that are all zeros in both results (e.g. key/value positions that
      # are fully masked out by causal attention when q_seq_len < kv_seq_len)
      # have an undefined cosine similarity, so exclude them. assert_close
      # below still verifies them.
      nonzero_rows = (cpu != 0).any(dim=-1) | (tpu != 0).any(dim=-1)
      sim = sim[nonzero_rows]
      self.assertGreater(
          sim.mean().item(),
          mean_sim_bound,
          f"similarity check failed for {key}",
      )

    def assert_close(cpu, tpu, key):
      rtol, atol = 5e-2, 1e-1
      self.assert_close(
          golden_result=cpu,
          torch_tpu_result=tpu,
          rtol=rtol,
          atol=atol,
          preamble=f"close check failed for {key}",
      )

    for key in cpu_results:
      with self.subTest(key):
        cpu_tensor = cpu_results[key].cpu()
        tpu_tensor = tpu_results[key].cpu()
        check_sim(cpu_tensor, tpu_tensor, key)
        assert_close(cpu_tensor, tpu_tensor, key)

  @parameterized.named_parameters(
      ("flash", torch.nn.attention.SDPBackend.FLASH_ATTENTION),
      ("overrideable", torch.nn.attention.SDPBackend.OVERRIDEABLE),
  )
  @oss_utils.skip_if_libtpu_version(
      lambda v: v < (0, 0, 51),
      "Requires libtpu >= 0.0.51 for correct numerics.",
  )
  def test_sdpa_masked_out_row(self, backend: torch.nn.attention.SDPBackend):
    """Check that a fully masked query row produces zeros and does not contain NaNs."""
    query = torch.randn(1, 1, 50, 64, device="tpu")
    key = torch.randn(1, 1, 100, 64, device="tpu")
    # Non-zero entries across all keys
    value = torch.randn(1, 1, 100, 64, device="tpu") + 1.0

    # Boolean mask: the first query attends to nothing
    mask_bool = torch.ones(1, 1, 50, 100, dtype=torch.bool, device="tpu")
    mask_bool[:, :, 0, :] = False

    # Additive mask with -inf: the first query attends to nothing
    mask_ninf = torch.zeros(1, 1, 50, 100, dtype=torch.float32, device="tpu")
    mask_ninf[:, :, 0, :] = -float("inf")

    for mask in (mask_bool, mask_ninf):
      with torch.nn.attention.sdpa_kernel(backend):
        result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask
        ).cpu()
        self.assertFalse(torch.isnan(result).any())
        # First query attends to nothing, so output must be exactly zero.
        self.assertTrue((result[0, 0, 0] == 0.0).all())
        # Second query is unmasked, so output should be non-zero.
        self.assertFalse((result[0, 0, 1] == 0.0).all())

    # Test gradients when query attends to nothing
    q_grad = query.clone().requires_grad_(True)
    k_grad = key.clone().requires_grad_(True)
    v_grad = value.clone().requires_grad_(True)
    with torch.nn.attention.sdpa_kernel(backend):
      out = torch.nn.functional.scaled_dot_product_attention(
          q_grad, k_grad, v_grad, attn_mask=mask_bool
      )
      out.sum().backward()
      self.assertFalse(torch.isnan(q_grad.grad).any())
      self.assertFalse(torch.isnan(k_grad.grad).any())
      self.assertFalse(torch.isnan(v_grad.grad).any())
      # Query 0 attends to nothing, so its grad must be 0
      self.assertTrue((q_grad.grad[0, 0, 0] == 0.0).all())

  @parameterized.named_parameters(
      ("flash", torch.nn.attention.SDPBackend.FLASH_ATTENTION),
      ("overrideable", torch.nn.attention.SDPBackend.OVERRIDEABLE),
  )
  def test_sdpa_gqa_different_v_head_dim(
      self, backend: torch.nn.attention.SDPBackend
  ):
    """Test SDPA GQA with v_head_dim != qk_head_dim against CPU golden."""
    torch.manual_seed(42)
    q_cpu = torch.randn(2, 8, 16, 32, dtype=torch.float32, requires_grad=True)
    k_cpu = torch.randn(2, 2, 16, 32, dtype=torch.float32, requires_grad=True)
    v_cpu = torch.randn(2, 2, 16, 16, dtype=torch.float32, requires_grad=True)

    q_tpu = q_cpu.detach().clone().tpu().requires_grad_(True)
    k_tpu = k_cpu.detach().clone().tpu().requires_grad_(True)
    v_tpu = v_cpu.detach().clone().tpu().requires_grad_(True)

    with torch.nn.attention.sdpa_kernel(backend):
      out_tpu = torch.nn.functional.scaled_dot_product_attention(
          q_tpu, k_tpu, v_tpu, is_causal=True, enable_gqa=True
      )
      out_tpu.sum().backward()

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
      out_cpu = torch.nn.functional.scaled_dot_product_attention(
          q_cpu, k_cpu, v_cpu, is_causal=True, enable_gqa=True
      )
      out_cpu.sum().backward()

    for golden, actual in (
        (out_cpu, out_tpu),
        (q_cpu.grad, q_tpu.grad),
        (k_cpu.grad, k_tpu.grad),
        (v_cpu.grad, v_tpu.grad),
    ):
      self.assert_close(
          golden_result=golden,
          torch_tpu_result=actual.cpu(),
          rtol=2e-2,
          atol=2e-2,
      )

  @parameterized.product(
      broadcast_dim=["query", "key"],
      dtype=[torch.float32, torch.bfloat16],
  )
  def test_sdpa_flash_mask_broadcast(self, broadcast_dim: str, dtype):
    """Tests flash attention with a mask of size 1 along the query or key dim."""
    batch, heads, seq, head_dim = 2, 2, 256, 64
    torch.manual_seed(0)
    q, k, v, grad_out = (
        torch.randn(batch, heads, seq, head_dim, dtype=dtype) for _ in range(4)
    )
    if broadcast_dim == "query":
      # [B, 1, 1, S] key padding mask.
      key_lengths = torch.tensor([100, 200])
      mask = (torch.arange(seq)[None] < key_lengths[:, None])[:, None, None, :]
    else:
      # [B, 1, L, 1] additive bias (a bool mask here would mask whole rows).
      mask = torch.randn(batch, 1, seq, 1, dtype=dtype)

    def compute(device):
      q_d, k_d, v_d = (
          t.to(device).detach().requires_grad_(True) for t in (q, k, v)
      )
      active_backend = (
          torch.nn.attention.SDPBackend.MATH
          if device == "cpu"
          else torch.nn.attention.SDPBackend.FLASH_ATTENTION
      )
      with torch.nn.attention.sdpa_kernel(active_backend):
        out = torch.nn.functional.scaled_dot_product_attention(
            q_d, k_d, v_d, attn_mask=mask.to(device)
        )
      out.backward(grad_out.to(device))
      return out, q_d.grad, k_d.grad, v_d.grad

    tol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    prev_prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
      self.assert_close_tpu_vs_cpu(compute, rtol=tol, atol=tol)
    finally:
      torch.set_float32_matmul_precision(prev_prec)

  @parameterized.named_parameters(
      # (q_seq, kv_seq, head_dim) -> expected backend with default settings.
      (
          "short_q_and_kv",
          (50, 50, 64),
          torch.nn.attention.SDPBackend.OVERRIDEABLE,
      ),
      (
          "below_tile_boundary",
          (127, 127, 64),
          torch.nn.attention.SDPBackend.OVERRIDEABLE,
      ),
      (
          "short_q_only",
          (1, 1024, 128),
          torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
      ),
      (
          "short_kv_only",
          (256, 64, 64),
          torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
      ),
      (
          "at_tile_boundary",
          (128, 128, 64),
          torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
      ),
      (
          "long",
          (1024, 1024, 128),
          torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
      ),
  )
  def test_sdpa_backend_choice_short_sequences(
      self,
      shape: tuple[int, int, int],
      expected: torch.nn.attention.SDPBackend,
  ):
    """Test that short query and key sequences prefer the OVERRIDEABLE backend.

    TorchTPU reports its flash attention kernel as CUDNN_ATTENTION.
    """
    q_seq, kv_seq, head_dim = shape
    q = torch.randn(2, 4, q_seq, head_dim, dtype=torch.bfloat16, device="tpu")
    k = torch.randn(2, 4, kv_seq, head_dim, dtype=torch.bfloat16, device="tpu")
    v = torch.randn(2, 4, kv_seq, head_dim, dtype=torch.bfloat16, device="tpu")

    # Default: all backends enabled.
    self.assertEqual(torch._fused_sdp_choice(q, k, v), expected.value)

    # An explicit flash-only request must still be honored.
    with torch.nn.attention.sdpa_kernel(
        torch.nn.attention.SDPBackend.FLASH_ATTENTION
    ):
      self.assertEqual(
          torch._fused_sdp_choice(q, k, v),
          torch.nn.attention.SDPBackend.CUDNN_ATTENTION.value,
      )

  @parameterized.product(
      backend=[
          torch.nn.attention.SDPBackend.FLASH_ATTENTION,
          torch.nn.attention.SDPBackend.OVERRIDEABLE,
      ],
      contiguous=[True, False],
  )
  def test_sdpa_view(
      self, backend: torch.nn.attention.SDPBackend, contiguous: bool
  ):
    """Test that SDPA with view works.

    This is not trying to check numerics, just that it actually runs.
    """
    dh = 128
    h_q = 32
    h_kv = 2

    def run_sdpa(h, wq, wkv, wo):
      bsz, n, dm = h.shape

      def maybe_contiguous(t):
        return t.contiguous() if contiguous else t

      q = maybe_contiguous((h @ wq).view(bsz, n, h_q, dh).transpose(1, 2))
      k = maybe_contiguous((h @ wkv).view(bsz, n, h_kv, dh).transpose(1, 2))
      v = maybe_contiguous((h @ wkv).view(bsz, n, h_kv, dh).transpose(1, 2))
      with torch.nn.attention.sdpa_kernel(backend):
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True
        )
      attn_out = maybe_contiguous(attn_out.transpose(1, 2))
      attn_out = attn_out.view(bsz, n, h_q * dh)
      out = attn_out @ wo

      return out

    h = torch.randn(2, 128, 256, device="tpu", requires_grad=True)
    wq = torch.randn(256, h_q * dh, device="tpu", requires_grad=True)
    wkv = torch.randn(256, h_kv * dh, device="tpu", requires_grad=True)
    wo = torch.randn(h_q * dh, 256, device="tpu", requires_grad=True)

    compiled_sdpa = torch.compile(run_sdpa, dynamic=False, backend="tpu")
    out = compiled_sdpa(h, wq, wkv, wo)
    out.backward(gradient=torch.randn_like(out))
    torch.accelerator.synchronize()

  @parameterized.product(
      h_kv=[1, 2],
      backend=[
          torch.nn.attention.SDPBackend.FLASH_ATTENTION,
          torch.nn.attention.SDPBackend.OVERRIDEABLE,
      ],
  )
  def test_sdpa_view_sliced_and_size_one_head(
      self, h_kv: int, backend: torch.nn.attention.SDPBackend
  ):
    """Tests compiled SDPA views with sliced inputs, GQA/MQA and dv != dqk.

    torch.compile traces with the strides of the meta functions, so if the
    kernel returns different strides, views traced on the attention output (e.g.
    merging the heads) can fail at runtime.
    """
    dh = 128
    h_q = 4
    dv = 64

    def run_sdpa(h, wq_gate, wk, wv, wo):
      bsz, n, _ = h.shape
      q_raw, gate = torch.chunk(
          (h @ wq_gate).view(bsz, n, h_q, dh * 2), 2, dim=-1
      )
      q = q_raw.transpose(1, 2)
      k = (h @ wk).view(bsz, n, h_kv, dh).transpose(1, 2)
      v = (h @ wv).view(bsz, n, h_kv, dv).transpose(1, 2)
      with torch.nn.attention.sdpa_kernel(backend):
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True
        )
      attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, n, h_q * dv)
      gate_flat = gate.reshape(bsz, n, h_q * dh)[..., : h_q * dv]
      return (attn_out * torch.sigmoid(gate_flat)) @ wo

    h = torch.randn(1, 128, 256, device="tpu", requires_grad=True)
    wq_gate = torch.randn(256, h_q * dh * 2, device="tpu", requires_grad=True)
    wk = torch.randn(256, h_kv * dh, device="tpu", requires_grad=True)
    wv = torch.randn(256, h_kv * dv, device="tpu", requires_grad=True)
    wo = torch.randn(h_q * dv, 256, device="tpu", requires_grad=True)

    compiled_sdpa = torch.compile(run_sdpa, dynamic=False, backend="tpu")
    out = compiled_sdpa(h, wq_gate, wk, wv, wo)
    out.backward(gradient=torch.randn_like(out))
    torch.accelerator.synchronize()

    self.assertEqual(out.shape, (1, 128, 256))
    self.assertFalse(torch.isnan(out).any())
    self.assertIsNotNone(h.grad)
    self.assertFalse(torch.isnan(h.grad).any())

  @parameterized.product(
      backend=[
          torch.nn.attention.SDPBackend.FLASH_ATTENTION,
          torch.nn.attention.SDPBackend.OVERRIDEABLE,
      ],
      layout=["sliced", "size_one_head", "expanded_batch"],
  )
  def test_sdpa_strides_match_meta(
      self, backend: torch.nn.attention.SDPBackend, layout: str
  ):
    """Tests eager SDPA output and grad strides and values for strided inputs.

    torch.compile traces with the strides of the meta functions, so any
    mismatch can make views that are valid in compiled mode fail in eager mode.
    The values are also compared against CPU, since the numerical SDPA test
    only uses contiguous inputs, for which the results need no relayout.

    Args:
      backend: The SDPA backend to test (`FLASH_ATTENTION` or `OVERRIDEABLE`).
      layout: "sliced" slices q/k/v out of fused projections, so they have gaps;
        "size_one_head" has a single head, so the head and sequence dims have
        equal strides; "expanded_batch" broadcasts the query over the batch, so
        it has a zero stride.
    """
    bsz, seq, d_qk, d_v = 2, 128, 128, 64
    h_q, h_kv = (1, 1) if layout == "size_one_head" else (4, 2)

    torch.manual_seed(0)
    if layout == "sliced":
      leaves = {
          "query": torch.randn(bsz, seq, h_q, 2 * d_qk),
          "key": torch.randn(bsz, seq, h_kv, 2 * d_qk),
          "value": torch.randn(bsz, seq, h_kv, 2 * d_v),
      }
    else:
      leaves = {
          "query": torch.randn(
              *(
                  (1, h_q, seq)
                  if layout == "expanded_batch"
                  else (bsz, seq, h_q)
              ),
              d_qk,
          ),
          "key": torch.randn(bsz, seq, h_kv, d_qk),
          "value": torch.randn(bsz, seq, h_kv, d_v),
      }
    grad_out = torch.randn(bsz, h_q, seq, d_v)

    def make_inputs(device):
      inputs = {}
      for name, leaf in leaves.items():
        x = leaf.to(device).requires_grad_()
        if layout == "sliced":
          x = torch.chunk(x, 2, dim=-1)[0].transpose(1, 2)
        elif layout == "expanded_batch" and name == "query":
          x = x.expand(bsz, -1, -1, -1)
        else:
          x = x.transpose(1, 2)
        inputs[name] = x
      return inputs

    def run_sdpa(q, k, v):
      active_backend = (
          torch.nn.attention.SDPBackend.MATH
          if q.device.type == "cpu"
          else backend
      )
      with torch.nn.attention.sdpa_kernel(active_backend):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, enable_gqa=True
        )

    def compute(device):
      inputs = make_inputs(device)
      grads = {}
      for name, x in inputs.items():
        x.register_hook(functools.partial(grads.__setitem__, name))
      out = run_sdpa(*inputs.values())
      out.backward(grad_out.to(device))
      return inputs, out, grads

    inputs, out, grads = compute("tpu")
    with fake_tensor.FakeTensorMode(allow_non_fake_inputs=True):
      _, meta_out, meta_grads = compute("tpu")
    expected_out_stride = meta_out.stride()
    if layout != "sliced" and utils.parse_version(torch.__version__) < (
        2,
        15,
        0,
    ):
      # PyTorch PR #193039 (torch 2.15) changed how `alloc_with_matching_layout`
      # orders dims with equal or zero strides when the value head dim differs
      # from the query head dim. The kernels follow the new order, so compare
      # against the strides of the new meta function on older versions.
      expected_out_stride = {
          "size_one_head": (8192, 64, 64, 1),
          "expanded_batch": (32768, 8192, 64, 1),
      }[layout]
    self.assertEqual(out.stride(), expected_out_stride)
    for name in inputs:
      with self.subTest(f"{name}_grad_stride"):
        self.assertEqual(grads[name].stride(), meta_grads[name].stride())

    _, cpu_out, cpu_grads = compute("cpu")
    for name, cpu, tpu in [
        ("out", cpu_out, out),
        *((f"{n}_grad", cpu_grads[n], grads[n]) for n in inputs),
    ]:
      with self.subTest(f"{name}_values"):
        sim = torch.nn.functional.cosine_similarity(cpu, tpu.cpu(), dim=-1)
        self.assertGreater(sim.mean().item(), 0.9999)

  def test_scaled_dot_product_attention_dropout(self):
    """Verifies FlashAttention dropout rate and 1/(1-p) scaling."""
    batch_size, num_heads, seq_len, head_dim = 2, 4, 128, 128
    dropout_p = 0.25
    device = torch.device("tpu")

    q = torch.zeros(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        requires_grad=True,
    )
    k = torch.zeros(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        requires_grad=True,
    )
    v = (
        torch.eye(seq_len, device=device)
        .expand(batch_size, num_heads, seq_len, head_dim)
        .clone()
        .requires_grad_(True)
    )

    torch.manual_seed(42)
    with torch.nn.attention.sdpa_kernel(
        torch.nn.attention.SDPBackend.FLASH_ATTENTION
    ):
      out = torch.nn.functional.scaled_dot_product_attention(
          q, k, v, dropout_p=dropout_p
      )
      out.backward(torch.ones_like(out))

    # With Q=0, K=0, and V=I, out equals the dropped attention weight matrix P.
    out_cpu = out.detach().cpu()
    drop_rate = (out_cpu == 0).to(torch.float32).mean().item()
    row_sum_mean = out_cpu.sum(dim=-1).mean().item()

    self.assertAlmostEqual(drop_rate, dropout_p, delta=0.05)
    self.assertAlmostEqual(row_sum_mean, 1.0, delta=0.05)
    self.assertIsNotNone(v.grad)

  @parameterized.product(
      use_mask=[False, True],
      dropout_p=[0.0, 0.1],
  )
  def test_scaled_dot_product_attention_compiled_backward(
      self, use_mask: bool, dropout_p: float
  ):
    """Tests compiled FlashAttention forward and backward, with dropout.

    Under torch.compile, the meta function of
    `_scaled_dot_product_cudnn_attention` returns 0-d philox seed/offset
    tensors, which the backward kernel must accept.

    Args:
      use_mask: Whether to pass a constant (non-trainable) boolean attn_mask.
      dropout_p: The dropout probability.
    """
    batch_size, num_heads, seq_len, head_dim = 2, 4, 128, 64
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(batch_size, num_heads, seq_len, head_dim) for _ in range(3)
    )
    attn_mask = (
        torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
        if use_mask
        else None
    )
    grad_out = torch.randn(batch_size, num_heads, seq_len, head_dim)

    def run_sdpa(q, k, v, attn_mask):
      return torch.nn.functional.scaled_dot_product_attention(
          q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
      )

    def compute(device):
      inputs = [
          t.to(device=device, dtype=torch.bfloat16).requires_grad_()
          for t in (q, k, v)
      ]
      mask = attn_mask.to(device) if attn_mask is not None else None
      if device == "cpu":
        out = run_sdpa(*inputs, mask)
      else:
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
          out = torch.compile(run_sdpa, dynamic=False, backend="tpu")(
              *inputs, mask
          )
      out.backward(grad_out.to(device=device, dtype=torch.bfloat16))
      return [out.detach().cpu().float()] + [
          t.grad.cpu().float() for t in inputs
      ]

    names = ("out", "q_grad", "k_grad", "v_grad")
    tpu_results = compute("tpu")
    for name, tpu in zip(names, tpu_results):
      with self.subTest(f"{name}_finite"):
        self.assertTrue(torch.isfinite(tpu).all())

    if dropout_p == 0.0:
      cpu_results = compute("cpu")
      for name, cpu, tpu in zip(names, cpu_results, tpu_results):
        with self.subTest(f"{name}_values"):
          self.assert_close(
              golden_result=cpu,
              torch_tpu_result=tpu,
              rtol=5e-2,
              atol=1e-1,
              preamble=f"close check failed for {name}",
          )

  @parameterized.product(
      compiled=[False, True],
  )
  def test_scaled_dot_product_attention_with_mask_gradient(self, compiled):
    """Verifies that SDPA correctly handles requires_grad for attn_mask."""
    batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 64
    torch.manual_seed(1337)
    q = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        requires_grad=True,
    )
    k = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        requires_grad=True,
    )
    mask = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        seq_len,
        requires_grad=True,
    )

    def run_sdpa(q, k, v, mask, compile_fn=False):
      sdpa_fn = torch.nn.functional.scaled_dot_product_attention
      if compile_fn:
        sdpa_fn = torch.compile(sdpa_fn, dynamic=False, backend="tpu")
      out = sdpa_fn(q, k, v, attn_mask=mask)
      out.mean().backward()
      return q.grad, k.grad, v.grad, mask.grad

    q_grad_cpu, k_grad_cpu, v_grad_cpu, mask_grad_cpu = run_sdpa(q, k, v, mask)

    q_tpu = q.detach().tpu().requires_grad_(True)
    k_tpu = k.detach().tpu().requires_grad_(True)
    v_tpu = v.detach().tpu().requires_grad_(True)
    mask_tpu = mask.detach().tpu().requires_grad_(True)
    q_grad_tpu, k_grad_tpu, v_grad_tpu, mask_grad_tpu = run_sdpa(
        q_tpu, k_tpu, v_tpu, mask_tpu, compile_fn=compiled
    )

    def assert_close(cpu, tpu, name):
      self.assertIsNotNone(cpu)
      self.assertIsNotNone(tpu)
      self.assert_close(
          golden_result=cpu,
          torch_tpu_result=tpu,
          rtol=1e-4,
          atol=1e-3,
          preamble=f"close check failed for {name}",
      )

    assert_close(q_grad_cpu, q_grad_tpu, "q_grad")
    assert_close(k_grad_cpu, k_grad_tpu, "k_grad")
    assert_close(v_grad_cpu, v_grad_tpu, "v_grad")
    assert_close(mask_grad_cpu, mask_grad_tpu, "mask_grad")

  def test_scaled_dot_product_attention_flash_under_func_grad(self):
    """Tests FlashAttention gradients under the torch.func.grad transform.

    torch.func.grad passes functorch wrapper tensors, which have no storage, to
    the SDPA backend selection.
    """
    batch_size, num_heads, seq_len, head_dim = 2, 4, 128, 64
    torch.manual_seed(0)
    q, k, v, grad_out = (
        torch.randn(batch_size, num_heads, seq_len, head_dim) for _ in range(4)
    )

    def compute(device):
      cotangent = grad_out.to(device=device, dtype=torch.bfloat16)

      def loss(q, k, v):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return (out * cotangent).sum()

      inputs = [t.to(device=device, dtype=torch.bfloat16) for t in (q, k, v)]
      grad_fn = torch.func.grad(loss, argnums=(0, 1, 2))
      if device == "cpu":
        grads = grad_fn(*inputs)
      else:
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
          grads = grad_fn(*inputs)
      return [g.cpu().float() for g in grads]

    cpu_grads = compute("cpu")
    tpu_grads = compute("tpu")
    for name, cpu, tpu in zip(
        ("q_grad", "k_grad", "v_grad"), cpu_grads, tpu_grads
    ):
      with self.subTest(name):
        self.assert_close(
            golden_result=cpu,
            torch_tpu_result=tpu,
            rtol=5e-2,
            atol=1e-1,
            preamble=f"close check failed for {name}",
        )

  @parameterized.named_parameters(
      ("flash_bfloat16", 128, 4, 4, torch.bfloat16, False),
      ("math_float32_mha", 24, 4, 4, torch.float32, False),
      ("math_float32_gqa", 24, 4, 2, torch.float32, True),
  )
  def test_scaled_dot_product_attention_scaling_precision(
      self, head_dim, num_heads_q, num_heads_kv, dtype, enable_gqa
  ):
    """Tests FlashAttention LSE forward/backward consistency and F32 post-dot scaling."""
    batch_size, seq_len = 2, 128
    torch.manual_seed(42)
    q_init = torch.randn(
        batch_size, num_heads_q, seq_len, head_dim, dtype=dtype
    )
    k_init = torch.randn(
        batch_size, num_heads_kv, seq_len, head_dim, dtype=dtype
    )
    v_init = torch.randn(
        batch_size, num_heads_kv, seq_len, head_dim, dtype=dtype
    )
    grad_out = torch.randn(
        batch_size, num_heads_q, seq_len, head_dim, dtype=dtype
    )

    def compute(device):
      q = to(q_init.clone(), device=device).requires_grad_(True)
      k = to(k_init.clone(), device=device).requires_grad_(True)
      v = to(v_init.clone(), device=device).requires_grad_(True)
      out = torch.nn.functional.scaled_dot_product_attention(
          q, k, v, is_causal=True, enable_gqa=enable_gqa
      )
      out.backward(to(grad_out, device=device))
      return out, q.grad, k.grad, v.grad

    rtol = 2e-2 if dtype == torch.bfloat16 else 3e-4
    atol = 2e-2 if dtype == torch.bfloat16 else 3e-4
    prev_prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
      self.assert_close_tpu_vs_cpu(compute, rtol=rtol, atol=atol)
    finally:
      torch.set_float32_matmul_precision(prev_prec)

  @parameterized.named_parameters(
      ("causal", lambda b, h, q_i, kv_i: q_i >= kv_i, True),
      ("unmasked", flex_attention.noop_mask, False),
      ("prefix", lambda b, h, q_i, kv_i: (kv_i < 128) | (q_i >= kv_i), False),
  )
  def test_flex_attention_causal(self, mask_mod, is_causal: bool):
    q, k, v = (
        torch.randn(1, 2, 512, 128, device="tpu", dtype=torch.bfloat16)
        for _ in range(3)
    )
    bm = flex_attention.create_block_mask(
        mask_mod, None, None, 512, 512, device="tpu"
    )

    with mock.patch.object(
        torch.nn.functional,
        "scaled_dot_product_attention",
        wraps=torch.nn.functional.scaled_dot_product_attention,
    ) as sdpa_mock:
      out = flex_attention.flex_attention(q, k, v, block_mask=bm)
      self.assertEqual(sdpa_mock.call_args.kwargs["is_causal"], is_causal)
      if is_causal:
        self.assertIsNone(sdpa_mock.call_args.kwargs["attn_mask"])
      else:
        self.assertEqual(
            sdpa_mock.call_args.kwargs["attn_mask"].shape, (1, 1, 512, 512)
        )

    mask_2d = flex_attention.create_mask(mask_mod, 1, 1, 512, 512, device="tpu")
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask_2d
    )
    utils.assert_close(out, expected)

  def test_flex_attention_packed_documents(self):
    # Equal-length packed documents are folded into the batch dimension and
    # attended with the mask shared by all documents.
    # Converting that per-document mask to SDPA arguments is
    # covered by the causal test above.
    docs, doc_len, heads = 4, 512, 2
    seq = docs * doc_len
    q, k, v = (
        torch.randn(1, heads, seq, 128, device="tpu", dtype=torch.bfloat16)
        for _ in range(3)
    )

    def mask_mod(b, h, q_i, kv_i):
      del b, h
      # First part: token i belongs to document i // doc_len, so this checks
      # both tokens are from the same document. Second part: makes it causal.
      return (q_i // doc_len == kv_i // doc_len) & (q_i >= kv_i)

    bm = flex_attention.create_block_mask(
        mask_mod, None, None, seq, seq, device="tpu"
    )

    with mock.patch.object(
        torch.nn.functional,
        "scaled_dot_product_attention",
        wraps=torch.nn.functional.scaled_dot_product_attention,
    ) as sdpa_mock:
      out = flex_attention.flex_attention(q, k, v, block_mask=bm)

    for folded in sdpa_mock.call_args.args[:3]:
      self.assertEqual(folded.shape, (docs, heads, doc_len, 128))

    self.assertEqual(out.shape, (1, heads, seq, 128))
    mask_2d = flex_attention.create_mask(mask_mod, 1, 1, seq, seq, device="tpu")
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask_2d
    )
    utils.assert_close(out, expected)

  @staticmethod
  def _causal_mask(n: int) -> torch.Tensor:
    return torch.ones(n, n, dtype=torch.bool).tril()

  @staticmethod
  def _sliding_window_mask(n: int, window: int) -> torch.Tensor:
    # Causal band: True where 0 <= row - col < window.
    return torch.ones(n, n, dtype=torch.bool).tril().triu(1 - window)

  def test_equal_square_tile_size_equal_causal_documents(self):
    mask = torch.block_diag(*[self._causal_mask(4)] * 4)
    self.assertEqual(flex_attention_redirect.equal_square_tile_size(mask), 4)

  def test_equal_square_tile_size_equal_sliding_window_documents(self):
    mask = torch.block_diag(*[self._sliding_window_mask(4, 2)] * 4)
    self.assertEqual(flex_attention_redirect.equal_square_tile_size(mask), 4)

  def test_equal_square_tile_size_tile_not_power_of_two(self):
    mask = torch.block_diag(*[self._causal_mask(3)] * 4)
    self.assertEqual(flex_attention_redirect.equal_square_tile_size(mask), 3)

  def test_equal_square_tile_size_single_document(self):
    mask = self._causal_mask(16)
    self.assertIsNone(flex_attention_redirect.equal_square_tile_size(mask))

  def test_equal_square_tile_size_full_attention(self):
    mask = torch.ones(16, 16, dtype=torch.bool)
    self.assertIsNone(flex_attention_redirect.equal_square_tile_size(mask))

  def test_equal_square_tile_size_unequal_documents(self):
    mask = torch.block_diag(self._causal_mask(6), self._causal_mask(10))
    self.assertIsNone(flex_attention_redirect.equal_square_tile_size(mask))

  def test_equal_square_tile_size_documents_with_different_masks(self):
    # The second document's mask differs from the others (token 3 can't see
    # token 1), so the tiles are not identical.
    causal_with_gap = self._causal_mask(4)
    causal_with_gap[3, 1] = False
    mask = torch.block_diag(
        self._causal_mask(4),
        causal_with_gap,
        self._causal_mask(4),
        self._causal_mask(4),
    )
    self.assertIsNone(flex_attention_redirect.equal_square_tile_size(mask))

  def test_equal_square_tile_size_cross_document_attention(self):
    # Four equal causal documents, but token 5 (document 1) attends to token 2
    # (document 0), so the mask is not block-diagonal.
    mask = torch.block_diag(*[self._causal_mask(4)] * 4)
    mask[5, 2] = True
    self.assertIsNone(flex_attention_redirect.equal_square_tile_size(mask))

  def test_splash_sdpa_gqa_fwd_bwd(self):
    batch, n_heads, n_kv_heads, seq_len, head_dim = 2, 8, 2, 512, 128
    dtype = torch.bfloat16
    torch.manual_seed(42)
    q = torch.randn(
        batch, n_heads, seq_len, head_dim, device="tpu", dtype=dtype
    ).requires_grad_(True)
    k = torch.randn(
        batch, n_kv_heads, seq_len, head_dim, device="tpu", dtype=dtype
    ).requires_grad_(True)
    v = torch.randn(
        batch, n_kv_heads, seq_len, head_dim, device="tpu", dtype=dtype
    ).requires_grad_(True)
    dy = torch.randn(
        batch, n_heads, seq_len, head_dim, device="tpu", dtype=dtype
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, is_causal=True, enable_gqa=True
    )
    out_ref.backward(dy)

    out = torch.tpu.ops.splash_sdpa(q, k, v, is_causal=True, enable_gqa=True)
    out.backward(dy)

    self.assertEqual(k.grad.shape, (batch, n_kv_heads, seq_len, head_dim))
    self.assertEqual(v.grad.shape, (batch, n_kv_heads, seq_len, head_dim))
    utils.assert_close(out, out_ref, rtol=2e-2, atol=2e-2)
    utils.assert_close(q.grad, q_ref.grad, rtol=3e-2, atol=3e-2)
    utils.assert_close(k.grad, k_ref.grad, rtol=3e-2, atol=3e-2)
    utils.assert_close(v.grad, v_ref.grad, rtol=3e-2, atol=3e-2)

  @parameterized.product(
      seq_len=[384, 768],
      compiled=[False, True],
  )
  def test_splash_sdpa_non_512_seq_len(self, seq_len: int, compiled: bool):
    batch, n_heads, head_dim = 1, 4, 128
    dtype = torch.bfloat16
    torch.manual_seed(42)
    q = torch.randn(
        batch, n_heads, seq_len, head_dim, device="tpu", dtype=dtype
    ).requires_grad_(True)
    k = torch.randn(
        batch, n_heads, seq_len, head_dim, device="tpu", dtype=dtype
    ).requires_grad_(True)
    v = torch.randn(
        batch, n_heads, seq_len, head_dim, device="tpu", dtype=dtype
    ).requires_grad_(True)
    dy = torch.randn(
        batch, n_heads, seq_len, head_dim, device="tpu", dtype=dtype
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, is_causal=True
    )
    out_ref.backward(dy)

    fn = torch.tpu.ops.splash_sdpa
    if compiled:
      fn = torch.compile(fn, backend="tpu")
    out = fn(q, k, v, is_causal=True)
    out.backward(dy)

    utils.assert_close(out, out_ref, rtol=2e-2, atol=2e-2)
    utils.assert_close(q.grad, q_ref.grad, rtol=3e-2, atol=3e-2)
    utils.assert_close(k.grad, k_ref.grad, rtol=3e-2, atol=3e-2)
    utils.assert_close(v.grad, v_ref.grad, rtol=3e-2, atol=3e-2)

  @parameterized.parameters(
      {"block_dkv": 256},
      {"block_kv": 256},
  )
  def test_splash_sdpa_small_explicit_blocks(self, **block_kwargs):
    batch, n_heads, seq_len, head_dim = 1, 4, 1024, 128
    dtype = torch.bfloat16
    torch.manual_seed(42)
    q = torch.randn(
        batch, n_heads, seq_len, head_dim, device="tpu", dtype=dtype
    ).requires_grad_(True)
    k = torch.randn(
        batch, n_heads, seq_len, head_dim, device="tpu", dtype=dtype
    ).requires_grad_(True)
    v = torch.randn(
        batch, n_heads, seq_len, head_dim, device="tpu", dtype=dtype
    ).requires_grad_(True)
    dy = torch.randn(
        batch, n_heads, seq_len, head_dim, device="tpu", dtype=dtype
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, is_causal=True
    )
    out_ref.backward(dy)

    out = torch.tpu.ops.splash_sdpa(q, k, v, is_causal=True, **block_kwargs)
    out.backward(dy)

    utils.assert_close(out, out_ref, rtol=2e-2, atol=2e-2)
    utils.assert_close(q.grad, q_ref.grad, rtol=3e-2, atol=3e-2)
    utils.assert_close(k.grad, k_ref.grad, rtol=3e-2, atol=3e-2)
    utils.assert_close(v.grad, v_ref.grad, rtol=3e-2, atol=3e-2)

  @parameterized.product(
      seg_dtype=[torch.int32, torch.int64],
      compiled=[False, True],
  )
  def test_splash_sdpa_segment_ids_packed_documents(
      self, seg_dtype: torch.dtype, compiled: bool
  ):
    batch, n_heads, n_kv_heads, seq_len, head_dim = 2, 4, 2, 768, 128
    dtype = torch.bfloat16
    torch.manual_seed(42)
    q = torch.randn(
        batch, n_heads, seq_len, head_dim, device="tpu", dtype=dtype
    ).requires_grad_(True)
    k = torch.randn(
        batch, n_kv_heads, seq_len, head_dim, device="tpu", dtype=dtype
    ).requires_grad_(True)
    v = torch.randn(
        batch, n_kv_heads, seq_len, head_dim, device="tpu", dtype=dtype
    ).requires_grad_(True)
    dy = torch.randn(
        batch, n_heads, seq_len, head_dim, device="tpu", dtype=dtype
    )

    seg_row0 = [0] * 200 + [1] * 312 + [2] * 256
    seg_row1 = [0] * 384 + [1] * 128 + [2] * 256
    segment_ids = torch.tensor(
        [seg_row0, seg_row1], device="tpu", dtype=seg_dtype
    )

    causal = torch.ones(seq_len, seq_len, device="tpu", dtype=torch.bool).tril()
    same_seg = segment_ids[:, None, :, None] == segment_ids[:, None, None, :]
    attn_mask = causal[None, None, :, :] & same_seg

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, attn_mask=attn_mask, enable_gqa=True
    )
    out_ref.backward(dy)

    fn = torch.tpu.ops.splash_sdpa
    if compiled:
      fn = torch.compile(fn, backend="tpu")
    out = fn(q, k, v, is_causal=True, enable_gqa=True, segment_ids=segment_ids)
    out.backward(dy)

    utils.assert_close(out, out_ref, rtol=2e-2, atol=2e-2)
    utils.assert_close(q.grad, q_ref.grad, rtol=3e-2, atol=3e-2)
    utils.assert_close(k.grad, k_ref.grad, rtol=3e-2, atol=3e-2)
    utils.assert_close(v.grad, v_ref.grad, rtol=3e-2, atol=3e-2)

    # Verify zero cross-document gradient leakage: a loss depending only on
    # document 0 in batch 0 (positions < 200) must produce zero gradients on
    # documents 1 and 2 (positions >= 200).
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    out2 = torch.tpu.ops.splash_sdpa(
        q2, k2, v2, is_causal=True, enable_gqa=True, segment_ids=segment_ids
    )
    out2[0, :, :200, :].sum().backward()
    self.assertEqual(q2.grad[0, :, 200:, :].abs().max().item(), 0.0)
    self.assertEqual(k2.grad[0, :, 200:, :].abs().max().item(), 0.0)
    self.assertEqual(v2.grad[0, :, 200:, :].abs().max().item(), 0.0)

  def test_splash_sdpa_invalid_inputs(self):
    q = torch.randn(2, 8, 512, 128, device="tpu", dtype=torch.bfloat16)
    k_bad_gqa = torch.randn(2, 3, 512, 128, device="tpu", dtype=torch.bfloat16)
    v_bad_gqa = torch.randn(2, 3, 512, 128, device="tpu", dtype=torch.bfloat16)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=no jax in errors tests
        ValueError, r"n_heads \(8\) must be divisible by n_kv_heads \(3\)"
    ):
      torch.tpu.ops.splash_sdpa(q, k_bad_gqa, v_bad_gqa, enable_gqa=True)

    q_bad_seq = torch.randn(1, 4, 100, 128, device="tpu", dtype=torch.bfloat16)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=no jax in errors tests
        ValueError, r"seq_len \(100\) must be a positive multiple of 128"
    ):
      torch.tpu.ops.splash_sdpa(q_bad_seq, q_bad_seq, q_bad_seq)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=no jax in errors tests
        ValueError, r"block_kv \(200\) must be a positive multiple of 128"
    ):
      torch.tpu.ops.splash_sdpa(q, q, q, block_kv=200)

    bad_seg_1d = torch.zeros(512, device="tpu", dtype=torch.int32)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=no jax in errors tests
        ValueError, "segment_ids"
    ):
      torch.tpu.ops.splash_sdpa(q, q, q, segment_ids=bad_seg_1d)

    bad_seg_float = torch.zeros(2, 512, device="tpu", dtype=torch.float32)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=no jax in errors tests
        ValueError, "segment_ids"
    ):
      torch.tpu.ops.splash_sdpa(q, q, q, segment_ids=bad_seg_float)

  def test_pointwise_op_dtype_promotion(self):
    """Ensure that pointwise ops promote as expected.

    In particular, they promote in all cases except one: rank zero
    scalar tensors. This is the expected behavior of PyTorch.
    """
    # Sanity check
    self.assertEqual(
        torch.float32, torch.promote_types(torch.bfloat16, torch.float32)
    )
    self.assertEqual(
        torch.float32, torch.promote_types(torch.bfloat16, torch.float16)
    )

    # Arrange
    device = torch.device("tpu")
    dtypes = op_testing.FLOAT_DTYPES

    # TODO: Systematically enumerate all 2-ary, pointwise ops out of
    # native_functions.yaml. Note that matmul is not a pointwise op.
    ops = [
        torch.add,
        torch.mul,
        torch.sub,
        torch.div,
        torch.pow,
        torch.remainder,
        torch.fmod,
        torch.maximum,
        torch.minimum,
        # torch.hypot,  # TODO: Uncomment once operator is added.
        torch.atan2,
        # torch.nextafter,  # TODO: Uncomment once operator is added.
        # torch.copysign,  # TODO: Uncomment once operator is added.
        torch.ldexp,  # not in tpu_aten_kernels, but native_functions.yaml
        # indicates this is a decomposed op.
    ]

    # 1) Non-rank-zero tensors
    for op in ops:
      for left_dtype in dtypes:
        for right_dtype in dtypes:
          left = torch.tensor([0.0], dtype=left_dtype, device=device)
          right = torch.tensor([0.0], dtype=right_dtype, device=device)
          expected = torch.promote_types(left_dtype, right_dtype)
          actual = op(left, right).dtype

          self.assertEqual(
              expected,
              actual,
              msg=(
                  f"{op.__name__=} {left_dtype=} {right_dtype=} {expected=}"
                  f" {actual=}"
              ),
          )

    # 2) Mixed rank-zero and non-rank-zero tensors
    for op in ops:
      for left_dtype in dtypes:
        for right_dtype in dtypes:
          left = torch.tensor([0.0], dtype=left_dtype, device=device)
          right = torch.tensor(0.0, dtype=right_dtype, device=device)
          # Notice that the output type matches the type of the non-scalar
          # input, rather than the result of torch.promote_types.
          expected = left.dtype
          actual = op(left, right).dtype

          self.assertEqual(
              expected,
              actual,
              msg=(
                  f"{op.__name__=} {left_dtype=} {right_dtype=} {expected=}"
                  f" {actual=}"
              ),
          )

    # 3) Rank-zero tensors
    for op in ops:
      for left_dtype in dtypes:
        for right_dtype in dtypes:
          left = torch.tensor(0.0, dtype=left_dtype, device=device)
          right = torch.tensor(0.0, dtype=right_dtype, device=device)
          expected = torch.promote_types(left_dtype, right_dtype)
          actual = op(left, right).dtype

          self.assertEqual(
              expected,
              actual,
              msg=(
                  f"{op.__name__=} {left_dtype=} {right_dtype=} {expected=}"
                  f" {actual=}"
              ),
          )

  def test_index_backward_with_cpu_mask(self):
    # The backward of `index` scatters the gradient with
    # `_index_put_impl_(accumulate=True)` using the saved CPU mask.
    x = torch.randn(4, 3)
    mask = torch.tensor([True, False, True, True])

    def index_grad(device):
      x_device = x.to(device).detach().requires_grad_()
      x_device[mask].pow(2).sum().backward()
      return x_device.grad

    self.assert_close_tpu_vs_cpu(index_grad)

  def test_index_put_boolean(self):
    # Case 1: Standard 1D boolean advanced indexing
    def test_fn_1(device):
      t = to(torch.zeros(4), device=device)
      mask = to(
          torch.tensor([False, True, False, True], dtype=torch.bool),
          device=device,
      )
      values = to(torch.tensor([10.0, 20.0]), device=device)
      t[mask] = values
      return t

    self.assert_close_tpu_vs_cpu(test_fn_1)

    # Case 2: Prefix Boolean Indexing (Leading dimension mask check)
    # self has shape (3, 2), mask has shape (3,) - selects whole rows!
    # This verifies our C++ prefix-mask dimension unsqueeze alignment!
    def test_fn_2(device):
      t = to(torch.zeros(3, 2), device=device)
      mask = to(
          torch.tensor([True, False, True], dtype=torch.bool), device=device
      )
      values = to(torch.tensor([[10.0, 20.0], [30.0, 40.0]]), device=device)
      t[mask] = values
      return t

    self.assert_close_tpu_vs_cpu(test_fn_2)

    # Case 3: Dtype variations and automatic casting alignment
    # Assigning float32 values to an int32 self tensor
    # NOTE: Fails on CPU due to strict dtype matching requirement in PyTorch.
    # Removed from active tests but kept in backup for reference.
    # def test_fn_3(device):
    #   t = to(torch.zeros(3, dtype=torch.int32), device=device)
    #   mask = to(
    #       torch.tensor([True, False, True], dtype=torch.bool), device=device
    #   )
    #   values = to(
    #       torch.tensor([10.7, 20.3], dtype=torch.float32), device=device
    #   )
    #   t[mask] = values
    #   return t
    # self.assert_close_tpu_vs_cpu(test_fn_3)

    # Case 4: Empty Mask (0 active elements - self returned unmodified)
    def test_fn_4(device):
      t = to(torch.tensor([1.0, 2.0, 3.0]), device=device)
      mask = to(
          torch.tensor([False, False, False], dtype=torch.bool), device=device
      )
      values = to(torch.tensor([], dtype=torch.float32), device=device)
      t[mask] = values
      return t

    self.assert_close_tpu_vs_cpu(test_fn_4)

    # Case 5: Accumulation check (accumulate=True)
    def test_fn_5(device):
      t = to(torch.tensor([1.0, 2.0, 3.0]), device=device)
      mask = to(
          torch.tensor([True, False, True], dtype=torch.bool), device=device
      )
      values = to(torch.tensor([10.0, 20.0]), device=device)
      t[mask] += values
      return t

    self.assert_close_tpu_vs_cpu(test_fn_5)

  def test_masked_scatter(self):
    # Case 1: Standard 1D masked_scatter
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.masked_scatter(
            input=to(torch.zeros(4), device=device),
            mask=to(
                torch.tensor([False, True, False, True], dtype=torch.bool),
                device=device,
            ),
            source=to(torch.tensor([10.0, 20.0, 30.0, 40.0]), device=device),
        ),
    )

    # Case 2: Multi-dimensional broadcast masking
    t_2d = torch.zeros(2, 3)
    mask_2d = torch.tensor(
        [[False, True, False], [True, False, True]], dtype=torch.bool
    )
    source_2d = torch.tensor([10.0, 20.0, 30.0, 40.0])
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.masked_scatter(
            input=to(t_2d, device=device),
            mask=to(mask_2d, device=device),
            source=to(source_2d, device=device),
        ),
    )

    # Case 3: Empty Mask (0 active elements - self returned unmodified)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.masked_scatter(
            input=to(torch.tensor([1.0, 2.0, 3.0]), device=device),
            mask=to(
                torch.tensor([False, False, False], dtype=torch.bool),
                device=device,
            ),
            source=to(torch.tensor([10.0, 20.0]), device=device),
        ),
    )

    # Case 4: Full Mask (All active elements - completely overwritten)
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.masked_scatter(
            input=to(torch.zeros(3), device=device),
            mask=to(
                torch.tensor([True, True, True], dtype=torch.bool),
                device=device,
            ),
            source=to(torch.tensor([10.0, 20.0, 30.0]), device=device),
        ),
    )

    # Case 5: Dtype variations (bfloat16, int32)
    for dtype in [torch.bfloat16, torch.int32]:
      t_dtype = torch.zeros(3, dtype=dtype)
      mask_dtype = torch.tensor([True, False, True], dtype=torch.bool)
      source_dtype = torch.tensor([10, 20], dtype=dtype)
      self.assert_close_tpu_vs_cpu(
          lambda device, t_dtype=t_dtype, mask_dtype=mask_dtype, source_dtype=source_dtype: torch.masked_scatter(
              input=to(t_dtype, device=device),
              mask=to(mask_dtype, device=device),
              source=to(source_dtype, device=device),
          ),
      )

  def test_masked_scatter_compiled_broadcast_row_gather(self):
    # Exercises torch.compile on expand_as trailing broadcast mask with bounded
    # feature rows (row_size >= 128, num_source_rows <= 1024).
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
      inputs_cpu = torch.randn(1, 64, 256, dtype=dtype)
      token_mask_cpu = torch.zeros(1, 64, 1, dtype=torch.bool)
      token_mask_cpu[0, 5:13, 0] = True
      source_cpu = torch.randn(8, 256, dtype=dtype)

      def fn(x, m, s):
        return x.masked_scatter(m.expand_as(x), s)

      expected = fn(inputs_cpu, token_mask_cpu, source_cpu)
      compiled_fn = torch.compile(fn, backend="tpu")
      actual = compiled_fn(
          inputs_cpu.to("tpu"), token_mask_cpu.to("tpu"), source_cpu.to("tpu")
      ).cpu()
      utils.assert_close(actual=actual, expected=expected)

  def test_masked_scatter_broadcast_mask(self):
    # The mask is broadcast along the trailing dimensions of the input, so
    # every mask element selects a whole contiguous row.
    for input_shape, mask_shape, source_numel in [
        # Mask broadcast along the last dimension.
        ((2, 3), (2, 1), 6),
        # Mask with fewer dimensions than the input.
        ((2, 3, 4), (3, 1), 16),
        # Mask broadcast along several trailing dimensions.
        ((2, 3, 4), (2, 1, 1), 24),
        # Fully broadcast mask.
        ((2, 3), (1, 1), 6),
        # Source that is not a multiple of the row size.
        ((2, 4), (2, 1), 7),
        # Source that is larger than the number of masked elements.
        ((4, 5), (4, 1), 40),
        # Mask broadcast along a middle dimension only, so there is no row to
        # fold.
        ((2, 3, 4), (2, 1, 4), 24),
        # Source with a partial trailing row, which is dropped.
        ((2, 3, 4), (2, 1, 1), 16),
        # Trailing dimension of size one in the input, so the mask is not
        # broadcast at all.
        ((2, 3, 1), (2, 3, 1), 6),
        # Scalar mask: every dimension folds into a single row.
        ((2, 3, 4), (), 24),
        # Lower rank mask of all ones.
        ((2, 3, 4), (1,), 24),
        # Empty input.
        ((0, 4), (0, 1), 0),
    ]:
      input_tensor = torch.arange(
          math.prod(input_shape), dtype=torch.float32
      ).reshape(input_shape)
      mask = torch.arange(math.prod(mask_shape)).reshape(mask_shape) % 2 == 0
      source = torch.arange(source_numel, dtype=torch.float32) + 100.0
      self.assert_close_tpu_vs_cpu(
          lambda device, input_tensor=input_tensor, mask=mask, source=source: torch.masked_scatter(
              input=to(input_tensor, device=device),
              mask=to(mask, device=device),
              source=to(source, device=device),
          ),
      )

    # A mask without any set element leaves the input untouched, even when the
    # source is empty.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.masked_scatter(
            input=to(torch.ones(2, 3), device=device),
            mask=to(torch.zeros(2, 1, dtype=torch.bool), device=device),
            source=to(torch.zeros(0), device=device),
        ),
    )

    # The same, with a source that holds less than one whole row.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.masked_scatter(
            input=to(torch.ones(2, 4), device=device),
            mask=to(torch.zeros(2, 1, dtype=torch.bool), device=device),
            source=to(torch.arange(3, dtype=torch.float32), device=device),
        ),
    )

  def test_masked_scatter_broadcast_mask_lowers_to_row_gather(self):
    """A mask broadcast over the trailing dimension lowers to a row gather.

    The numerics tests above pass whether or not the row lowering fires, so
    this guards the lowering itself: a `{1, 2048, 1}` mask over a
    `{1, 2048, 64}` input must gather 2048 rows of 64 elements rather than
    131072 individual elements.
    """
    mask = (torch.arange(2048).reshape(1, 2048, 1) % 2 == 0).to("tpu")
    source = torch.ones(1024, 64, device="cpu").to("tpu")
    with execution_mode.set_eager_mode(
        execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
    ):
      base = torch.zeros(1, 2048, 64, device="tpu")
      result = torch.masked_scatter(base, mask, source)
    mlir_text = tpu_torch_compile.serialize_mlir_text(
        tpu_torch_compile.build_mlir([result], [mask, source])
    )
    self.assertIn("stablehlo.gather", mlir_text)
    self.assertIn("tensor<2048x64xf32>", mlir_text)
    self.assertNotIn("tensor<131072x1xf32>", mlir_text)

  def test_masked_scatter_elementwise_mask_lowers_to_element_gather(self):
    """A mask with no broadcast trailing dimension stays element-wise.

    The counterpart to the row gather test above. A fully materialized
    `{1, 2048, 64}` mask has no row structure to exploit, so the lowering falls
    back to the `{131072, 1}` element grid. Asserting this keeps the row
    detection from widening to masks it must not fold.
    """
    mask = (torch.arange(2048 * 64).reshape(1, 2048, 64) % 2 == 0).to("tpu")
    source = torch.ones(65536, device="cpu").to("tpu")
    with execution_mode.set_eager_mode(
        execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
    ):
      base = torch.zeros(1, 2048, 64, device="tpu")
      result = torch.masked_scatter(base, mask, source)
    mlir_text = tpu_torch_compile.serialize_mlir_text(
        tpu_torch_compile.build_mlir([result], [mask, source])
    )
    self.assertIn("stablehlo.gather", mlir_text)
    self.assertIn("tensor<131072x1xf32>", mlir_text)
    self.assertNotIn("tensor<2048x64xf32>", mlir_text)

  def test_slice_scatter(self):
    # Fast-path 1: Full replacement (step == 1 && slice_len == total_len)
    t = torch.randn(4, 8)
    src = torch.randn(4, 8) + 10.0
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t, src=src: torch.slice_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=0,
            start=0,
            end=4,
            step=1,
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t, src=src: torch.slice_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=1,
            start=None,
            end=None,
            step=1,
        )
    )

    # Fast-path 2: Boundary-anchored concatenate (start == 0, slice_len * 2 >= total_len)
    t_boundary_start = torch.randn(2, 10, 8)
    src_boundary_start = torch.randn(2, 7, 8) + 5.0
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_boundary_start, src=src_boundary_start: torch.slice_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=1,
            start=0,
            end=7,
            step=1,
        )
    )

    # Fast-path 2: Boundary-anchored concatenate (end == total_len, slice_len * 2 >= total_len)
    t_boundary_end = torch.randn(2, 10, 8)
    src_boundary_end = torch.randn(2, 6, 8) + 5.0
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_boundary_end, src=src_boundary_end: torch.slice_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=1,
            start=4,
            end=10,
            step=1,
        )
    )

    # Fallback: Interior slices (step == 1)
    t_interior = torch.randn(2, 10, 8)
    src_interior = torch.randn(2, 3, 8) + 5.0
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_interior, src=src_interior: torch.slice_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=1,
            start=2,
            end=5,
            step=1,
        )
    )

    # Boundary concatenate on last dim with negative dim
    t_neg_dim = torch.randn(3, 4, 12)
    src_neg_dim = torch.randn(3, 4, 8) + 2.0
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_neg_dim, src=src_neg_dim: torch.slice_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=-1,
            start=0,
            end=8,
            step=1,
        )
    )

    # Negative indices
    t_neg_idx = torch.randn(2, 10, 4)
    src_neg_idx = torch.randn(2, 6, 4) + 1.0
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_neg_idx, src=src_neg_idx: torch.slice_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=1,
            start=-8,
            end=-2,
            step=1,
        )
    )

    # Empty slice (start == end)
    t_empty = torch.randn(3, 5)
    src_empty = torch.randn(3, 0)
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_empty, src=src_empty: torch.slice_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=1,
            start=2,
            end=2,
            step=1,
        )
    )

    # Strided slice (step > 1): even/odd offsets, minor-most dim, and non-divisible length
    t_strided = torch.randn(10, 4)
    src_strided = torch.randn(5, 4) + 3.0
    for start in [0, 1]:
      self.assert_close_tpu_vs_cpu(
          lambda device, t=t_strided, src=src_strided, s=start: torch.slice_scatter(
              to(t, device=device),
              to(src, device=device),
              dim=0,
              start=s,
              end=10,
              step=2,
          )
      )
    t_strided_minor = torch.randn(2, 6, 16)
    src_strided_minor = torch.randn(2, 6, 8) + 3.0
    for start in [0, 1]:
      self.assert_close_tpu_vs_cpu(
          lambda device, t=t_strided_minor, src=src_strided_minor, s=start: torch.slice_scatter(
              to(t, device=device),
              to(src, device=device),
              dim=-1,
              start=s,
              end=16,
              step=2,
          )
      )
    t_strided_nondiv = torch.randn(3, 11, 5)
    src_strided_nondiv = torch.randn(3, 2, 5) + 4.0
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_strided_nondiv, src=src_strided_nondiv: torch.slice_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=1,
            start=4,
            end=10,
            step=3,
        )
    )

    # Different dtypes
    for dtype in [torch.bfloat16, torch.int32]:
      t_dt = torch.arange(24, dtype=dtype).reshape(4, 6)
      src_dt = torch.arange(12, dtype=dtype).reshape(2, 6) + 100
      self.assert_close_tpu_vs_cpu(
          lambda device, t=t_dt, src=src_dt: torch.slice_scatter(
              to(t, device=device),
              to(src, device=device),
              dim=0,
              start=0,
              end=2,
              step=1,
          )
      )

    # Zero-element tensor and mixed self/src dtypes
    t_empty = torch.zeros(0, 4, dtype=torch.float32)
    src_empty = torch.zeros(0, 2, dtype=torch.float32)
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_empty, src=src_empty: torch.slice_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=1,
            start=1,
            end=3,
            step=1,
        )
    )
    t_mixed = torch.ones(4, 6, dtype=torch.float32)
    src_mixed = torch.full((2, 6), 5.0, dtype=torch.bfloat16)
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_mixed, src=src_mixed: torch.slice_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=0,
            start=1,
            end=3,
            step=1,
        )
    )

  def test_select_scatter(self):
    # Fast-path: total_len == 1
    t_len1 = torch.randn(3, 1, 4)
    src_len1 = torch.randn(3, 4) + 10.0
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_len1, src=src_len1: torch.select_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=1,
            index=0,
        )
    )

    # 1D tensor
    t_1d = torch.randn(5)
    src_1d = torch.tensor(42.0)
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_1d, src=src_1d: torch.select_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=0,
            index=2,
        )
    )

    # Multi-dimensional tensor
    t_multi = torch.randn(3, 5, 4)
    src_multi = torch.randn(3, 4) + 7.0
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_multi, src=src_multi: torch.select_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=1,
            index=3,
        )
    )

    # Negative index
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_multi, src=src_multi: torch.select_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=1,
            index=-1,
        )
    )

    # Negative dim
    t_neg_dim = torch.randn(2, 3, 6)
    src_neg_dim = torch.randn(2, 3) + 9.0
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_neg_dim, src=src_neg_dim: torch.select_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=-1,
            index=4,
        )
    )

    # Different dtypes
    for dtype in [torch.bfloat16, torch.int32]:
      t_dt = torch.arange(24, dtype=dtype).reshape(4, 6)
      src_dt = torch.arange(6, dtype=dtype) + 100
      self.assert_close_tpu_vs_cpu(
          lambda device, t=t_dt, src=src_dt: torch.select_scatter(
              to(t, device=device),
              to(src, device=device),
              dim=0,
              index=1,
          )
      )

    # Zero-element tensor and mixed self/src dtypes
    t_empty = torch.zeros(0, 4, dtype=torch.float32)
    src_empty = torch.zeros(0, dtype=torch.float32)
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_empty, src=src_empty: torch.select_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=1,
            index=2,
        )
    )
    t_mixed = torch.ones(4, 6, dtype=torch.float32)
    src_mixed = torch.full((6,), 5.0, dtype=torch.bfloat16)
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_mixed, src=src_mixed: torch.select_scatter(
            to(t, device=device),
            to(src, device=device),
            dim=0,
            index=1,
        )
    )

  def test_slice_scatter_out(self):
    # Functional equivalence vs CPU with preallocated out
    t = torch.randn(4, 8)
    src = torch.randn(2, 8) + 10.0
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t, src=src: torch.ops.aten.slice_scatter.out(
            to(t, device=device),
            to(src, device=device),
            dim=0,
            start=1,
            end=3,
            step=1,
            out=to(torch.empty(4, 8), device=device),
        )
    )

    # Empty out tensor resizing
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t, src=src: torch.ops.aten.slice_scatter.out(
            to(t, device=device),
            to(src, device=device),
            dim=0,
            start=1,
            end=3,
            step=1,
            out=to(torch.empty(0), device=device),
        )
    )

    # Strided slice (step > 1) with out
    t_strided = torch.randn(10, 4)
    src_strided = torch.randn(5, 4) + 3.0
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_strided, src=src_strided: (
            torch.ops.aten.slice_scatter.out(
                to(t, device=device),
                to(src, device=device),
                dim=0,
                start=0,
                end=10,
                step=2,
                out=to(torch.empty(0), device=device),
            )
        )
    )

    # Empty slice (slice_len == 0) with out
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t: torch.ops.aten.slice_scatter.out(
            to(t, device=device),
            to(torch.empty(0, 8), device=device),
            dim=0,
            start=2,
            end=2,
            step=1,
            out=to(torch.empty(0), device=device),
        )
    )

    # Zero-element input tensor
    t_empty = torch.zeros(0, 4, dtype=torch.float32)
    src_empty = torch.zeros(0, 2, dtype=torch.float32)
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_empty, src=src_empty: (
            torch.ops.aten.slice_scatter.out(
                to(t, device=device),
                to(src, device=device),
                dim=1,
                start=1,
                end=3,
                step=1,
                out=to(torch.empty(0), device=device),
            )
        )
    )

    # In-place donation: out is self
    t_donate = torch.randn(4, 8)
    src_donate = torch.randn(2, 8) + 5.0

    def run_slice_inplace(device, t=t_donate, src=src_donate):
      inp = to(t.clone(), device=device)
      res = torch.ops.aten.slice_scatter.out(
          inp,
          to(src, device=device),
          dim=0,
          start=1,
          end=3,
          step=1,
          out=inp,
      )
      return res

    self.assert_close_tpu_vs_cpu(run_slice_inplace)

  def test_select_scatter_out(self):
    # Functional equivalence vs CPU with preallocated out
    t = torch.randn(3, 5, 4)
    src = torch.randn(3, 4) + 7.0
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t, src=src: torch.ops.aten.select_scatter.out(
            to(t, device=device),
            to(src, device=device),
            dim=1,
            index=3,
            out=to(torch.empty(3, 5, 4), device=device),
        )
    )

    # Empty out tensor resizing
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t, src=src: torch.ops.aten.select_scatter.out(
            to(t, device=device),
            to(src, device=device),
            dim=1,
            index=3,
            out=to(torch.empty(0), device=device),
        )
    )

    # Fast-path total_len == 1 with out
    t_len1 = torch.randn(3, 1, 4)
    src_len1 = torch.randn(3, 4) + 10.0
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_len1, src=src_len1: (
            torch.ops.aten.select_scatter.out(
                to(t, device=device),
                to(src, device=device),
                dim=1,
                index=0,
                out=to(torch.empty(0), device=device),
            )
        )
    )

    # Negative index and negative dim
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t, src=src: torch.ops.aten.select_scatter.out(
            to(t, device=device),
            to(src, device=device),
            dim=-2,
            index=-1,
            out=to(torch.empty(0), device=device),
        )
    )

    # Zero-element input tensor
    t_empty = torch.zeros(0, 4, dtype=torch.float32)
    src_empty = torch.zeros(0, dtype=torch.float32)
    self.assert_close_tpu_vs_cpu(
        lambda device, t=t_empty, src=src_empty: (
            torch.ops.aten.select_scatter.out(
                to(t, device=device),
                to(src, device=device),
                dim=1,
                index=2,
                out=to(torch.empty(0), device=device),
            )
        )
    )

    # In-place donation: out is self
    t_donate = torch.randn(3, 5, 4)
    src_donate = torch.randn(3, 4) + 5.0

    def run_select_inplace(device, t=t_donate, src=src_donate):
      inp = to(t.clone(), device=device)
      res = torch.ops.aten.select_scatter.out(
          inp,
          to(src, device=device),
          dim=1,
          index=2,
          out=inp,
      )
      return res

    self.assert_close_tpu_vs_cpu(run_select_inplace)

  def test_polygamma_n_0(self):
    # n = 0 is digamma.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.polygamma(
            0,
            to(
                torch.tensor(
                    [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=torch.float32
                ),
                device=device,
            ),
        ),
        rtol=5e-5,
        atol=5e-5,
    )

    # Test pole at 0 (should return -inf) and negative integer poles (should
    # return NaN). Since our assert_close has equal_nan=True by default,
    # we can safely compare NaNs.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.polygamma(
            0,
            to(
                torch.tensor([-2.0, -1.0, 0.0], dtype=torch.float32),
                device=device,
            ),
        )
    )

    # Test type promotion with float32 input and float64 output.
    # computation_type is float32, but it must be cast to float64 for output.
    def test_type_promotion(device):
      self_tensor = to(
          torch.tensor([1.0, 2.0], dtype=torch.float32), device=device
      )
      out_tensor = to(torch.empty(2, dtype=torch.float64), device=device)
      torch.polygamma(0, self_tensor, out=out_tensor)
      return out_tensor

    self.assert_close_tpu_vs_cpu(test_type_promotion, rtol=1.6e-5, atol=6.8e-6)

  def test_polygamma_pos_inf(self):
    # Test polygamma(n, +inf) returns 0.0 for n=1 and nan for n > 1 matching CPU (b/568040580).
    for n in (1, 2, 3):
      for dtype in (
          torch.float32,
          torch.float64,
          torch.bfloat16,
          torch.float16,
      ):
        for shape in ((), (1,)):
          self.assert_close_tpu_vs_cpu(
              lambda device, n=n, dtype=dtype, shape=shape: torch.polygamma(
                  n,
                  to(
                      torch.full(shape, float("inf"), dtype=dtype),
                      device=device,
                  ),
              )
          )

  def test_polygamma_integral_default_dtype(self):
    # Test integral inputs under default_dtype=float64 compute in float64
    # matching CPU (b/568040996).
    prev_default = torch.get_default_dtype()
    try:
      torch.set_default_dtype(torch.float64)
      for n in (0, 1):
        for dtype in (torch.int64, torch.int32):
          with self.subTest(n=n, dtype=dtype):
            self.assert_close_tpu_vs_cpu(
                lambda device, n=n, dtype=dtype: torch.polygamma(
                    n, to(torch.tensor([1, 2, 3], dtype=dtype), device=device)
                )
            )
    finally:
      torch.set_default_dtype(prev_default)

  def test_polygamma_out_dtype_rounding(self):
    # Test polygamma with out= preserves intermediate rounding matching CPU
    # (b/568040996).
    for in_dtype, out_dtype in (
        (torch.bfloat16, torch.float32),
        (torch.float64, torch.float32),
    ):
      with self.subTest(in_dtype=in_dtype, out_dtype=out_dtype):

        def test_fn(device, in_dtype=in_dtype, out_dtype=out_dtype):
          x = to(torch.tensor([1.5, 2.5], dtype=in_dtype), device=device)
          out = to(torch.zeros(2, dtype=out_dtype), device=device)
          torch.polygamma(1, x, out=out)
          return out

        self.assert_close_tpu_vs_cpu(test_fn)

  def test_masked_softmax(self):
    def test_fn(device):
      results = []
      for dtype in (torch.float32, torch.bfloat16):
        x = (
            torch.arange(2 * 4 * 16 * 16, dtype=dtype, device=device) / 1000.0
        ).reshape(2, 4, 16, 16)

        # 1. 2D padding mask (B=2, L=16) (mask_type=1)
        mask_padding = (torch.arange(2 * 16, device=device) % 2 == 0).reshape(
            2, 16
        )
        res1 = torch.ops.aten._masked_softmax(
            x, mask_padding, dim=-1, mask_type=1
        )
        results.append(res1)

        # 2. Cross-attention (B=2, H=4, L=8, S=16) and 2D padding mask
        # (B=2, S=16) (mask_type=1)
        x_cross = (
            torch.arange(2 * 4 * 8 * 16, dtype=dtype, device=device) / 1000.0
        ).reshape(2, 4, 8, 16)
        mask_padding_cross = (
            torch.arange(2 * 16, device=device) % 2 == 0
        ).reshape(2, 16)
        if device == "cpu":
          x_masked = x_cross.float().masked_fill(
              mask_padding_cross.view(2, 1, 1, 16), float("-inf")
          )
          res_cross = torch.softmax(x_masked, dim=-1).to(dtype)
        else:
          res_cross = torch.ops.aten._masked_softmax(
              x_cross, mask_padding_cross, dim=-1, mask_type=1
          )
        results.append(res_cross)

        # 3. 4D input and 2D attention mask (L=16, L=16) (mask_type=0)
        mask_attn = (torch.arange(16 * 16, device=device) % 3 == 0).reshape(
            16, 16
        )
        res3 = torch.ops.aten._masked_softmax(x, mask_attn, dim=-1, mask_type=0)
        results.append(res3)

        # 4. Generic mask (mask_type=2)
        mask_generic = (
            torch.arange(2 * 4 * 16 * 16, device=device) % 5 == 0
        ).reshape(2, 4, 16, 16)
        res4 = torch.ops.aten._masked_softmax(
            x, mask_generic, dim=-1, mask_type=2
        )
        results.append(res4)

        # 5. Out variant
        out = torch.empty_like(x)
        torch.ops.aten._masked_softmax.out(
            x, mask_generic, dim=-1, mask_type=2, out=out
        )
        results.append(out)

        # 6. Non-4D inputs with explicit mask_type=0 and mask_type=1
        x_2d = (
            torch.arange(4 * 5, dtype=dtype, device=device) / 100.0
        ).reshape(4, 5)
        mask_2d = (torch.arange(4 * 5, device=device) % 2 == 0).reshape(4, 5)
        for mt in (0, 1):
          res_non4d = torch.ops.aten._masked_softmax(
              x_2d, mask_2d, dim=-1, mask_type=mt
          )
          results.append(res_non4d)

        # 7. 0D scalar tensors (dim=0, dim=-1)
        x_0d = torch.tensor(3.14, dtype=dtype, device=device)
        for mask_val in (False, True):
          mask_0d = torch.tensor(mask_val, dtype=torch.bool, device=device)
          for dim_val in (0, -1):
            res_0d = torch.ops.aten._masked_softmax(x_0d, mask_0d, dim=dim_val)
            results.append(res_0d)

      return tuple(results)

    self.assert_close_tpu_vs_cpu(test_fn, rtol=1e-2, atol=1e-2)

  def test_masked_softmax_backward(self):
    def test_fn(device):
      results = []
      for dtype in (torch.float32, torch.bfloat16):
        x = (
            torch.arange(2 * 4 * 16 * 16, dtype=dtype, device=device) / 1000.0
        ).reshape(2, 4, 16, 16)
        grad = (
            (torch.arange(2 * 4 * 16 * 16, dtype=dtype, device=device) - 1000.0)
            / 1000.0
        ).reshape(2, 4, 16, 16)
        mask = (torch.arange(2 * 4 * 16 * 16, device=device) % 3 == 0).reshape(
            2, 4, 16, 16
        )
        output = torch.ops.aten._masked_softmax(x, mask, dim=-1, mask_type=2)

        # 1. Standard backward
        res1 = torch.ops.aten._masked_softmax_backward(
            grad, output, mask, dim=-1
        )
        results.append(res1)

        # 2. 2D padding mask (B=2, S=16)
        mask_padding_2d = (
            torch.arange(2 * 16, device=device) % 2 == 0
        ).reshape(2, 16)
        mask_padding_4d = mask_padding_2d.view(2, 1, 1, 16).expand(2, 4, 16, 16)
        output_pad = torch.ops.aten._masked_softmax(
            x, mask_padding_2d, dim=-1, mask_type=1
        )
        if device == "cpu":
          res_pad = torch.ops.aten._masked_softmax_backward(
              grad, output_pad, mask_padding_4d, dim=-1
          )
        else:
          res_pad = torch.ops.aten._masked_softmax_backward(
              grad, output_pad, mask_padding_2d, dim=-1
          )
        results.append(res_pad)

        # 3. 2D attention mask (L=16, S=16)
        mask_attn_2d = (torch.arange(16 * 16, device=device) % 3 == 0).reshape(
            16, 16
        )
        mask_attn_4d = mask_attn_2d.view(1, 1, 16, 16).expand(2, 4, 16, 16)
        output_attn = torch.ops.aten._masked_softmax(
            x, mask_attn_2d, dim=-1, mask_type=0
        )
        if device == "cpu":
          res_attn = torch.ops.aten._masked_softmax_backward(
              grad, output_attn, mask_attn_4d, dim=-1
          )
        else:
          res_attn = torch.ops.aten._masked_softmax_backward(
              grad, output_attn, mask_attn_2d, dim=-1
          )
        results.append(res_attn)

        # 4. Out variant
        out = torch.empty_like(grad)
        torch.ops.aten._masked_softmax_backward.out(
            grad, output, mask, dim=-1, out=out
        )
        results.append(out)

        # 5. 0D scalar tensors (dim=0, dim=-1)
        x_0d = torch.tensor(3.14, dtype=dtype, device=device)
        mask_0d = torch.tensor(False, dtype=torch.bool, device=device)
        grad_0d = torch.tensor(1.5, dtype=dtype, device=device)
        output_0d = torch.ops.aten._masked_softmax(x_0d, mask_0d, dim=-1)
        for dim_val in (0, -1):
          res_cpu_0d_bwd = torch.ops.aten._masked_softmax_backward(
              grad_0d, output_0d, mask_0d, dim=dim_val
          )
          results.append(res_cpu_0d_bwd.reshape(()))

      return tuple(results)

    self.assert_close_tpu_vs_cpu(test_fn, rtol=1e-2, atol=1e-2)

  def test_linear_grad(self):
    shape_cases = [
        # 1D, 2D, 3D, 4D input with 2D weight (with and without bias)
        ((4,), (3, 4), None),
        ((4,), (3, 4), (3,)),
        ((2, 4), (3, 4), None),
        ((2, 4), (3, 4), (3,)),
        ((2, 5, 4), (3, 4), None),
        ((2, 5, 4), (3, 4), (3,)),
        ((2, 3, 4, 5), (6, 5), None),
        ((2, 3, 4, 5), (6, 5), (6,)),
        # 1D and 2D input with 1D weight
        ((4,), (4,), None),
        ((4,), (4,), ()),
        ((2, 4), (4,), None),
        # Zero-sized batch and contracting dimensions
        ((0, 4), (3, 4), (3,)),
        ((3, 4, 0), (5, 0), (5,)),
    ]
    for dtype in [
        torch.float32,
        torch.bfloat16,
        torch.complex64,
        torch.float16,
    ]:
      rtol, atol = (
          (5e-2, 5e-2)
          if dtype in (torch.bfloat16, torch.float16)
          else (2e-2, 2e-2)
      )
      for x_shape, w_shape, b_shape in shape_cases:
        with self.subTest(
            dtype=dtype, x_shape=x_shape, w_shape=w_shape, b_shape=b_shape
        ):
          x_base = torch.randn(x_shape, dtype=dtype)
          w_base = torch.randn(w_shape, dtype=dtype)
          b_base = (
              torch.randn(b_shape, dtype=dtype) if b_shape is not None else None
          )
          out_shape = torch.nn.functional.linear(x_base, w_base, b_base).shape
          grad_out_base = torch.randn(out_shape, dtype=dtype)

          def run_grad(
              device,
              x_b=x_base,
              w_b=w_base,
              b_b=b_base,
              g_b=grad_out_base,
          ):
            x = to(x_b.clone(), device=device).requires_grad_(True)
            w = to(w_b.clone(), device=device).requires_grad_(True)
            b = (
                to(b_b.clone(), device=device).requires_grad_(True)
                if b_b is not None
                else None
            )
            grad_out = to(g_b.clone(), device=device)
            out = torch.nn.functional.linear(x, w, b)
            out.backward(grad_out)
            return (x.grad, w.grad, b.grad if b is not None else None)

          self.assert_close_tpu_vs_cpu(run_grad, rtol=rtol, atol=atol)

    # Non-contiguous input, weight, and grad_out
    with self.subTest("non_contiguous"):
      x_nc_base = torch.randn(5, 2, 4, dtype=torch.float32)
      w_nc_base = torch.randn(4, 3, dtype=torch.float32)
      b_nc_base = torch.randn(6, dtype=torch.float32)
      grad_nc_base = torch.randn(5, 2, 3, dtype=torch.float32)

      def run_nc_grad(device):
        x = (
            to(x_nc_base.clone(), device=device)
            .transpose(0, 1)
            .detach()
            .requires_grad_(True)
        )
        w = (
            to(w_nc_base.clone(), device=device)
            .t()
            .detach()
            .requires_grad_(True)
        )
        b = (
            to(b_nc_base.clone(), device=device)[::2]
            .detach()
            .requires_grad_(True)
        )
        grad_out = to(grad_nc_base.clone(), device=device).transpose(0, 1)
        out = torch.nn.functional.linear(x, w, b)
        out.backward(grad_out)
        return (x.grad, w.grad, b.grad)

      self.assert_close_tpu_vs_cpu(run_nc_grad, rtol=2e-2, atol=2e-2)

    # All combinations of requires_grad for (x, w, b)
    x_mask_base = torch.randn(2, 5, 4, dtype=torch.float32)
    w_mask_base = torch.randn(3, 4, dtype=torch.float32)
    b_mask_base = torch.randn(3, dtype=torch.float32)
    grad_mask_base = torch.randn(2, 5, 3, dtype=torch.float32)
    for mask in itertools.product([True, False], repeat=3):
      if not any(mask):
        continue
      with self.subTest(requires_grad_mask=mask):
        req_x, req_w, req_b = mask

        def run_mask_grad(device, rx=req_x, rw=req_w, rb=req_b):
          x = to(x_mask_base.clone(), device=device).requires_grad_(rx)
          w = to(w_mask_base.clone(), device=device).requires_grad_(rw)
          b = to(b_mask_base.clone(), device=device).requires_grad_(rb)
          grad_out = to(grad_mask_base.clone(), device=device)
          out = torch.nn.functional.linear(x, w, b)
          out.backward(grad_out)
          return (x.grad, w.grad, b.grad)

        self.assert_close_tpu_vs_cpu(run_mask_grad, rtol=2e-2, atol=2e-2)

    # Non-standard broadcastable bias shapes with requires_grad=True
    for b_bcast_shape in [(1,), (1, 3), (2, 1, 3), (2, 5, 3)]:
      with self.subTest(broadcastable_bias_shape=b_bcast_shape):
        x_bcast_base = torch.randn(2, 5, 4, dtype=torch.float32)
        w_bcast_base = torch.randn(3, 4, dtype=torch.float32)
        b_bcast_base = torch.randn(b_bcast_shape, dtype=torch.float32)
        grad_bcast_base = torch.randn(2, 5, 3, dtype=torch.float32)

        def run_bcast_bias_grad(
            device,
            x_b=x_bcast_base,
            w_b=w_bcast_base,
            b_b=b_bcast_base,
            g_b=grad_bcast_base,
        ):
          x = to(x_b.clone(), device=device).requires_grad_(True)
          w = to(w_b.clone(), device=device).requires_grad_(True)
          b = to(b_b.clone(), device=device).requires_grad_(True)
          grad_out = to(g_b.clone(), device=device)
          out = torch.nn.functional.linear(x, w, b)
          out.backward(grad_out)
          return (x.grad, w.grad, b.grad)

        self.assert_close_tpu_vs_cpu(run_bcast_bias_grad, rtol=2e-2, atol=4e-2)

    # Low-precision (bf16/f16) grad_bias accumulation over large batch*seq (8192)
    # to verify XLA:TPU reduction accumulation prevents stagnation at 256 (bf16)
    # or 2048 (f16).
    for lp_dtype in [torch.bfloat16, torch.float16]:
      with self.subTest(low_precision_grad_bias_accum=lp_dtype):
        x_lp_base = torch.ones(64, 128, 4, dtype=lp_dtype)
        w_lp_base = torch.ones(3, 4, dtype=lp_dtype)
        b_lp_base = torch.zeros(3, dtype=lp_dtype)
        grad_lp_base = torch.ones(64, 128, 3, dtype=lp_dtype)

        def run_lp_grad_bias(
            device,
            x_b=x_lp_base,
            w_b=w_lp_base,
            b_b=b_lp_base,
            g_b=grad_lp_base,
        ):
          x = to(x_b.clone(), device=device).requires_grad_(True)
          w = to(w_b.clone(), device=device).requires_grad_(True)
          b = to(b_b.clone(), device=device).requires_grad_(True)
          grad_out = to(g_b.clone(), device=device)
          out = torch.nn.functional.linear(x, w, b)
          out.backward(grad_out)
          return b.grad

        self.assert_close_tpu_vs_cpu(run_lp_grad_bias, rtol=1e-3, atol=1e-3)

    # ND input (2D and 3D) + 1D weight + non-scalar broadcastable bias with
    # requires_grad=True verified against CPU matmul + bias reference.
    for x_shape_1dw, b_shape_1dw in [
        ((2, 4), (2,)),
        ((2, 5, 4), (5,)),
    ]:
      with self.subTest(
          "1d_weight_bcast_bias", x_shape=x_shape_1dw, b_shape=b_shape_1dw
      ):
        x_cpu = torch.randn(
            x_shape_1dw, dtype=torch.float32, requires_grad=True
        )
        w_cpu = torch.randn(4, dtype=torch.float32, requires_grad=True)
        b_cpu = torch.randn(
            b_shape_1dw, dtype=torch.float32, requires_grad=True
        )
        out_cpu = torch.matmul(x_cpu, w_cpu) + b_cpu
        grad_cpu = torch.randn_like(out_cpu)
        out_cpu.backward(grad_cpu)

        x_1dw = to(x_cpu.detach().clone(), "tpu").requires_grad_(True)
        w_1dw = to(w_cpu.detach().clone(), "tpu").requires_grad_(True)
        b_1dw = to(b_cpu.detach().clone(), "tpu").requires_grad_(True)
        grad_1dw = to(grad_cpu.detach().clone(), "tpu")
        out_1dw = torch.nn.functional.linear(x_1dw, w_1dw, b_1dw)
        out_1dw.backward(grad_1dw)
        self.assert_close(
            golden_result=out_cpu.detach(),
            torch_tpu_result=out_1dw.detach().cpu(),
            rtol=1e-4,
            atol=1e-4,
        )
        self.assert_close(
            golden_result=x_cpu.grad,
            torch_tpu_result=x_1dw.grad.cpu(),
            rtol=1e-4,
            atol=1e-4,
        )
        self.assert_close(
            golden_result=w_cpu.grad,
            torch_tpu_result=w_1dw.grad.cpu(),
            rtol=1e-4,
            atol=1e-4,
        )
        self.assert_close(
            golden_result=b_cpu.grad,
            torch_tpu_result=b_1dw.grad.cpu(),
            rtol=1e-4,
            atol=1e-4,
        )

    # FakeTensorMode with 1D weight and requires_grad=True
    with self.subTest("fake_tensor_1d_weight_grad"):
      with fake_tensor.FakeTensorMode():
        fx = torch.randn(2, 4, device="tpu", requires_grad=True)
        fw = torch.randn(4, device="tpu", requires_grad=True)
        fb = torch.randn((), device="tpu", requires_grad=True)
        fout = torch.nn.functional.linear(fx, fw, fb)
        fout.sum().backward()
        self.assertEqual(fx.grad.shape, (2, 4))
        self.assertEqual(fw.grad.shape, (4,))
        self.assertEqual(fb.grad.shape, ())

  def test_linear_backward_direct_all_masks(self):
    cases = [
        ((2, 5, 4), (3, 4), (3,)),
        ((3, 4), (3, 4), (3,)),
        ((4,), (3, 4), (3,)),
        ((2, 4), (4,), ()),
        ((4,), (4,), ()),
    ]
    for x_shape, w_shape, b_shape in cases:
      x_cpu = torch.randn(x_shape, dtype=torch.float32, requires_grad=True)
      w_cpu = torch.randn(w_shape, dtype=torch.float32, requires_grad=True)
      b_cpu = torch.randn(b_shape, dtype=torch.float32, requires_grad=True)
      if len(x_shape) == 2 and len(w_shape) == 1:
        # CPU aten::linear dispatches 2D input with bias to addmm, which
        # rejects 1D weight; compute equivalent matmul + bias reference.
        out_cpu = torch.matmul(x_cpu, w_cpu.t()) + b_cpu
      else:
        out_cpu = torch.nn.functional.linear(x_cpu, w_cpu, b_cpu)
      grad_out_cpu = torch.randn_like(out_cpu)
      out_cpu.backward(grad_out_cpu)
      ref_grads = (x_cpu.grad, w_cpu.grad, b_cpu.grad)

      x_tpu = to(x_cpu.detach(), "tpu")
      w_tpu = to(w_cpu.detach(), "tpu")
      grad_out_tpu = to(grad_out_cpu.detach(), "tpu")

      for output_mask in itertools.product([True, False], repeat=3):
        with self.subTest(
            x_shape=x_shape, w_shape=w_shape, output_mask=output_mask
        ):
          tpu_grads = torch.ops.aten.linear_backward(
              x_tpu, grad_out_tpu, w_tpu, list(output_mask)
          )
          self.assertLen(tpu_grads, 3)
          for mask_bit, tpu_grad, ref_grad in zip(
              output_mask, tpu_grads, ref_grads, strict=True
          ):
            if mask_bit:
              self.assertIsNotNone(tpu_grad)
              self.assert_close(
                  golden_result=ref_grad,
                  torch_tpu_result=tpu_grad.cpu(),
                  rtol=2e-2,
                  atol=2e-2,
              )
            else:
              self.assertTrue(tpu_grad is None or not tpu_grad.defined())

  def test_materialize_non_leaf_tensor_does_not_warn(self):
    """Verifies copying a non-leaf tensor with requires_grad does not warn."""
    device = torch.device("tpu")
    x = torch.ones(1, requires_grad=True, device=device)
    y = x * 2
    y.backward()
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      _ = y.item()
      _ = y.cpu()
      grad_warnings = [
          warning
          for warning in w
          if issubclass(warning.category, UserWarning)
          and re.search(r"\bgrad\b", str(warning.message))
      ]
      self.assertEmpty(
          grad_warnings,
          msg=(
              "Materializing non-leaf tensor emitted spurious UserWarning about"
              " accessing .grad"
          ),
      )

  @parameterized.product(
      spatial_ndim=[2, 3],
      padding_mode=["zeros", "border", "reflection"],
      align_corners=[False, True],
      output_mask=[(False, True), (True, False)],
  )
  def test_grid_sample_backward_partial_output_mask(
      self,
      spatial_ndim: int,
      padding_mode: str,
      align_corners: bool,
      output_mask: tuple[bool, bool],
  ) -> None:
    torch.manual_seed(0)
    in_shape = (2, 2, 4, 5) if spatial_ndim == 2 else (1, 2, 3, 4, 4)
    grid_shape = (2, 3, 4, 2) if spatial_ndim == 2 else (1, 2, 3, 3, 3)
    modes = (
        ["bilinear", "nearest", "bicubic"]
        if spatial_ndim == 2
        else ["bilinear", "nearest"]
    )
    compute_grad_input, compute_grad_grid = output_mask
    x_base = torch.randn(in_shape, dtype=torch.float32)
    grid_base = torch.empty(grid_shape, dtype=torch.float32).uniform_(-1.5, 1.5)

    for mode in modes:
      with self.subTest(mode=mode):
        x_cpu = x_base.clone().detach().requires_grad_(compute_grad_input)
        grid_cpu = grid_base.clone().detach().requires_grad_(compute_grad_grid)
        x_tpu = x_base.detach().to("tpu").requires_grad_(compute_grad_input)
        grid_tpu = (
            grid_base.detach().to("tpu").requires_grad_(compute_grad_grid)
        )

        out_cpu = torch.nn.functional.grid_sample(
            x_cpu,
            grid_cpu,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        out_tpu = torch.nn.functional.grid_sample(
            x_tpu,
            grid_tpu,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_tpu.backward(grad_out.to("tpu"))

        if compute_grad_input:
          utils.assert_close(
              actual=x_tpu.grad.cpu(),
              expected=x_cpu.grad,
              rtol=1e-4,
              atol=1e-4,
          )
        else:
          self.assertIsNone(x_tpu.grad)

        if compute_grad_grid:
          utils.assert_close(
              actual=grid_tpu.grad.cpu(),
              expected=grid_cpu.grad,
              rtol=1e-4,
              atol=1e-4,
          )
        else:
          self.assertIsNone(grid_tpu.grad)

  @parameterized.product(
      in_shape=[(2, 2, 1, 4), (2, 2, 4, 1), (2, 2, 1, 1), (1, 2, 1, 3, 4)],
      align_corners=[False, True],
  )
  def test_grid_sample_reflection_unit_spatial_dim(
      self, in_shape: tuple[int, ...], align_corners: bool
  ) -> None:
    torch.manual_seed(0)
    grid_shape = (2, 3, 3, 2) if len(in_shape) == 4 else (1, 2, 2, 2, 3)
    modes = (
        ["bilinear", "nearest", "bicubic"]
        if len(in_shape) == 4
        else ["bilinear", "nearest"]
    )
    x_base = torch.randn(in_shape, dtype=torch.float32)
    grid_base = torch.empty(grid_shape, dtype=torch.float32).uniform_(-2.0, 2.0)

    for mode in modes:
      with self.subTest(mode=mode):
        x_cpu = x_base.clone().detach().requires_grad_(True)
        grid_cpu = grid_base.clone().detach().requires_grad_(True)
        x_tpu = x_base.detach().to("tpu").requires_grad_(True)
        grid_tpu = grid_base.detach().to("tpu").requires_grad_(True)

        out_cpu = torch.nn.functional.grid_sample(
            x_cpu,
            grid_cpu,
            mode=mode,
            padding_mode="reflection",
            align_corners=align_corners,
        )
        out_tpu = torch.nn.functional.grid_sample(
            x_tpu,
            grid_tpu,
            mode=mode,
            padding_mode="reflection",
            align_corners=align_corners,
        )
        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_tpu.backward(grad_out.to("tpu"))

        utils.assert_close(
            actual=out_tpu.cpu(),
            expected=out_cpu,
            rtol=1e-4,
            atol=1e-4,
        )
        utils.assert_close(
            actual=x_tpu.grad.cpu(),
            expected=x_cpu.grad,
            rtol=1e-4,
            atol=1e-4,
        )
        utils.assert_close(
            actual=grid_tpu.grad.cpu(),
            expected=grid_cpu.grad,
            rtol=1e-4,
            atol=1e-4,
        )


class OpTestingFrameworkTest(op_testing.OpInfoTestBase):
  """Tests for the op_testing framework itself."""

  def test_torch_tpu_vs_gpu_missing_golden_fails(self):
    op = next(op for op in op_db if op.name == "add")
    with (
        flagsaver.flagsaver(test_mode=op_testing.TestMode.TORCH_TPU_VS_GPU),
        mock.patch.object(
            op_testing, "_GOLDEN_GPU_DATA", op_testing.GoldenGpuData()
        ),
    ):
      with self.assertRaisesRegex(
          AssertionError,
          "No GPU golden samples found for add.*Please re-generate the GPU"
          " golden files",
      ):
        self._get_golden_input_output_pairs(
            op=op,
            dtype=torch.float32,
            variant=op_testing.OpVariant.BASE,
            max_samples=1,
        )

      # Ensure that if samples are present, no error is raised and samples are
      # returned.
      fake_input = op_testing.OpInput(FakeSample("s", torch.zeros(1), (), {}))
      fake_output = op_testing.OpOutput(torch.zeros(1))
      fake_sample = (fake_input, fake_output)
      op_testing._GOLDEN_GPU_DATA.add(
          self._testMethodName,
          op_testing.OpVariant.BASE,
          torch.float32,
          fake_input,
          fake_output,
      )
      res = self._get_golden_input_output_pairs(
          op=op,
          dtype=torch.float32,
          variant=op_testing.OpVariant.BASE,
          max_samples=1,
      )
      self.assertEqual(res, [fake_sample])

  def test_searchsorted_scalar_float_in_int(self):
    def test_fn(device):
      seq = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=device)
      res_left = torch.searchsorted(seq, 2.5, right=False)
      res_right = torch.searchsorted(seq, 2.5, right=True)
      return res_left, res_right

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_searchsorted_0d_query_1elem_sorter(self):
    def test_fn(device):
      seq = torch.tensor([10], dtype=torch.int64, device=device)
      sorter = torch.tensor([0], dtype=torch.int64, device=device)
      query = torch.tensor(10, dtype=torch.int64, device=device)
      res = torch.searchsorted(seq, query, sorter=sorter)
      return res

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_bucketize_out_resize(self):
    def test_fn(device):
      inp = torch.tensor([[0.0, 2.0, 3.0], [3.0, 4.0, 6.0]], device=device)
      boundaries = torch.tensor([1.0, 3.0, 5.0], device=device)
      out = torch.empty(0, dtype=torch.int64, device=device)
      torch.bucketize(inp, boundaries, out=out)
      return out

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_searchsorted_out_resize(self):
    def test_fn(device):
      seq = torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], device=device)
      values = torch.tensor([[0.0, 2.0, 4.0], [3.0, 5.0, 7.0]], device=device)
      out = torch.empty(0, dtype=torch.int64, device=device)
      torch.searchsorted(seq, values, out=out)
      return out

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_isin_out_resize(self):
    def test_fn(device):
      elements = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
      test_elements = torch.tensor([2, 4, 6], device=device)
      out = torch.empty(0, dtype=torch.bool, device=device)
      torch.isin(elements, test_elements, out=out)
      return out

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_sort_out_resize(self):
    def test_fn(device):
      t = torch.tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], device=device)
      values = torch.empty(0, dtype=torch.float32, device=device)
      indices = torch.empty(0, dtype=torch.int64, device=device)
      torch.sort(t, out=(values, indices))
      return values, indices

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_topk_out_resize(self):
    def test_fn(device):
      t = torch.tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], device=device)
      values = torch.empty(0, dtype=torch.float32, device=device)
      indices = torch.empty(0, dtype=torch.int64, device=device)
      torch.topk(t, 2, out=(values, indices))
      return values, indices

    self.assert_close_tpu_vs_cpu(test_fn)

  @parameterized.named_parameters(
      dict(
          testcase_name="float4_e2m1fn_x2",
          dtype=torch.float4_e2m1fn_x2,
      ),
      dict(
          testcase_name="int4",
          dtype=torch.int4,
      ),
  )
  def test_sub_byte_sample_generation(self, dtype: torch.dtype):
    abs_op = next(op for op in op_db if op.name == "abs")
    samples = op_testing._sample_inputs(abs_op, torch.device("cpu"), dtype)
    self.assertNotEmpty(samples)
    for sample in samples:
      self.assertEqual(op_testing.OpInput(sample).input_value.dtype, dtype)

  def test_plistlib_sub_byte_tensor_serialization(self):
    for dtype in quantize_utils.SUB_BYTE_DTYPES:
      with self.subTest(dtype=dtype):
        raw = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
        tensor = raw.view(dtype)
        encoded = op_testing._to_plistlib_compatible(tensor)
        decoded = op_testing._from_plistlib_compatible(encoded)
        self.assertEqual(decoded.dtype, dtype)
        self.assertEqual(decoded.shape, tensor.shape)
        utils.assert_close(decoded.view(torch.uint8), raw)

  def test_perf_mode_reads_gpu_golden_inputs(self):
    abs_op = next(op for op in op_db if op.name == "abs")
    fake_input = op_testing.OpInput(FakeSample("s", torch.zeros(1), (), {}))
    fake_output = op_testing.OpOutput(torch.zeros(1))
    with (
        flagsaver.flagsaver(test_mode=op_testing.TestMode.PERF),
        mock.patch.object(
            op_testing, "_GOLDEN_GPU_DATA", op_testing.GoldenGpuData()
        ),
    ):
      op_testing._GOLDEN_GPU_DATA.add(
          self._testMethodName,
          op_testing.OpVariant.BASE,
          torch.float32,
          fake_input,
          fake_output,
      )
      pairs = self._get_golden_input_output_pairs(
          op=abs_op,
          dtype=torch.float32,
          variant=op_testing.OpVariant.BASE,
          max_samples=2,
          verbose=False,
      )
    self.assertEqual(pairs, [(fake_input, fake_output)])

  def test_plistlib_enum_serialization(self):
    enum_val = getattr(
        torch.nn.functional,
        "_ScalingType",
        getattr(torch._C, "_ScalingType", None),
    )
    if enum_val is not None:
      test_enum = enum_val.RowWise
    else:

      class CustomEnum(enum.Enum):
        ROW_WISE = 1

      test_enum = CustomEnum.ROW_WISE

    data = {"enum_arg": test_enum, "tuple_arg": (test_enum, 42)}
    plist_compat = op_testing._to_plistlib_compatible(data)
    bin_data = plistlib.dumps(
        plist_compat,
        fmt=typing.cast(plistlib.PlistFormat, plistlib.FMT_BINARY),
    )
    self.assertIsInstance(bin_data, bytes)

    restored = op_testing._from_plistlib_compatible(plist_compat)
    self.assertEqual(restored["enum_arg"], test_enum)
    self.assertEqual(restored["tuple_arg"][0], test_enum)

  def test_plistlib_exception_serialization(self):
    """Tests that golden exceptions preserve their type and message."""
    output = op_testing.OpOutput(ValueError("bad value: 42"))
    restored = op_testing.OpOutput.from_plistlib_pytree(
        output.to_plistlib_pytree()
    )

    self.assertIsInstance(restored.output_value, ValueError)
    self.assertEqual(type(restored.output_value).__name__, "ValueError")
    self.assertEqual(str(restored.output_value), "bad value: 42")

  @mock.patch.dict(op_testing._DTYPE_EXCLUSIONS, clear=True)
  def test_record_gpu_dtype_exclusions_multiple_calls_intersects_exclusions(
      self,
  ):
    op_testing._record_gpu_dtype_exclusions(
        "add",
        exclude_dtypes=(torch.float32, torch.float64),
        exclude_inplace_dtypes=(torch.int16,),
    )
    op_testing._record_gpu_dtype_exclusions(
        "add",
        exclude_dtypes=(torch.float64, torch.bfloat16),
        exclude_inplace_dtypes=(torch.int32,),
    )
    self.assertEqual(
        op_testing._DTYPE_EXCLUSIONS["add"]["exclude_dtypes"],
        {"torch.float64"},
    )
    self.assertEqual(
        op_testing._DTYPE_EXCLUSIONS["add"]["exclude_inplace_dtypes"],
        set(),
    )

  @mock.patch.dict(op_testing._DTYPE_EXCLUSIONS, clear=True)
  def test_build_dtype_exclusions_yaml_returns_sorted_formatted_yaml(self):
    op_testing._record_gpu_dtype_exclusions(
        "add",
        exclude_dtypes=(torch.float64, torch.float32),
        exclude_inplace_dtypes=(torch.int32,),
    )
    expected_yaml = (
        "add:\n"
        "  exclude_dtypes:\n"
        "  - torch.float32\n"
        "  - torch.float64\n"
        "  exclude_inplace_dtypes:\n"
        "  - torch.int32\n"
    )
    self.assertEqual(op_testing._build_dtype_exclusions_yaml(), expected_yaml)

  def test_softmax_empty_tensor(self):
    for dtype in (torch.float32, torch.bfloat16):
      test_cases = [
          # Empty reduction dimension
          ((2, 0, 3), 1),
          ((0,), 0),
          ((0, 5), 0),
          # Empty non-reduction dimension
          ((0, 5), 1),
          ((2, 0, 3), 0),
          ((2, 0, 3), 2),
          ((2, 0, 3), -1),
      ]
      for shape, dim in test_cases:
        cpu_x = torch.empty(shape, dtype=dtype)
        tpu_x = cpu_x.to("tpu")

        # Softmax
        cpu_sm = torch.softmax(cpu_x, dim=dim)
        tpu_sm = torch.softmax(tpu_x, dim=dim)
        self.assertEqual(tpu_sm.shape, cpu_sm.shape)
        utils.assert_close(actual=tpu_sm.cpu(), expected=cpu_sm)

        # LogSoftmax
        cpu_lsm = torch.log_softmax(cpu_x, dim=dim)
        tpu_lsm = torch.log_softmax(tpu_x, dim=dim)
        self.assertEqual(tpu_lsm.shape, cpu_lsm.shape)
        utils.assert_close(actual=tpu_lsm.cpu(), expected=cpu_lsm)

        # Compiled
        @torch.compile(dynamic=False)
        def compiled_fn(x, d=dim):
          return torch.softmax(x, dim=d), torch.log_softmax(x, dim=d)

        tpu_comp_sm, tpu_comp_lsm = compiled_fn(tpu_x)
        self.assertEqual(tpu_comp_sm.shape, cpu_sm.shape)
        self.assertEqual(tpu_comp_lsm.shape, cpu_lsm.shape)
        utils.assert_close(actual=tpu_comp_sm.cpu(), expected=cpu_sm)
        utils.assert_close(actual=tpu_comp_lsm.cpu(), expected=cpu_lsm)

  def test_softmax_dynamic_reduction_dim(self):
    for dtype in (torch.float32, torch.bfloat16):
      for dim in (0, 1, -1):
        cpu_x = torch.randn(4, 6, dtype=dtype)
        tpu_x = cpu_x.to("tpu")
        dyn_dim = dim if dim >= 0 else cpu_x.ndim + dim
        dynamism.mark_dynamic(
            tpu_x, dimension=dyn_dim, lower_bound=2, upper_bound=10
        )

        rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (2e-2, 2e-2)
        cpu_sm = torch.softmax(cpu_x, dim=dim)
        tpu_sm = torch.softmax(tpu_x, dim=dim)
        utils.assert_close(
            actual=tpu_sm.cpu(), expected=cpu_sm, rtol=rtol, atol=atol
        )

        cpu_lsm = torch.log_softmax(cpu_x, dim=dim)
        tpu_lsm = torch.log_softmax(tpu_x, dim=dim)
        utils.assert_close(
            actual=tpu_lsm.cpu(), expected=cpu_lsm, rtol=rtol, atol=atol
        )

  def test_foreach_norm_and_clip_grad_norm(self):
    for ord_val in (0, 1, 2, 3.0, -1.0, -1.5, float("inf"), float("-inf")):
      for out_dtype in (None, torch.float32):

        def foreach_norm_fn(device, ord_arg=ord_val, dtype_arg=out_dtype):
          tensors = [
              torch.tensor(
                  [-3.0, 0.0, 4.0], dtype=torch.float32, device=device
              ),
              torch.tensor(
                  [[1.0, -2.0], [3.0, 4.0]], dtype=torch.bfloat16, device=device
              ),
              torch.zeros((2, 2), dtype=torch.float32, device=device),
              torch.tensor(5.0, dtype=torch.float32, device=device),
          ]
          if ord_arg >= 0 and ord_arg != float("inf"):
            tensors.append(
                torch.empty((0,), dtype=torch.float32, device=device)
            )
          return torch._foreach_norm(tensors, ord=ord_arg, dtype=dtype_arg)

        self.assert_close_tpu_vs_cpu(foreach_norm_fn, rtol=1e-2, atol=1e-2)

    def clip_grad_norm_fn(device):
      params = [
          torch.nn.Parameter(
              torch.arange(1, 7, dtype=torch.float32, device=device).reshape(
                  2, 3
              )
              * (i + 1)
          )
          for i in range(6)
      ]
      for i, p in enumerate(params):
        p.grad = torch.full_like(p, float(i + 1))
      total_norm = torch.nn.utils.clip_grad_norm_(
          params, max_norm=1.0, foreach=True
      )
      return [total_norm] + [p.grad for p in params]

    self.assert_close_tpu_vs_cpu(clip_grad_norm_fn, rtol=1e-5, atol=1e-5)

  def test_select_backward(self):
    for dtype in (torch.float32, torch.bfloat16):
      test_cases = [
          ((5,), 0, 0),
          ((5,), 0, 2),
          ((5,), 0, 4),
          ((4, 6), 0, 0),
          ((4, 6), 0, 3),
          ((4, 6), 1, 0),
          ((4, 6), 1, 2),
          ((4, 6), 1, 5),
          ((2, 3, 4), 1, 1),
          ((3, 142, 32, 512), 0, 0),
          ((3, 142, 32, 512), 0, 1),
          ((3, 142, 32, 512), 0, 2),
          ((2, 8, 16, 32), 2, 5),
          # Size-1 along selected dimension (exercises early return branch)
          ((1,), 0, 0),
          ((4, 1, 6), 1, 0),
          # Negative indexing (dim and index normalization)
          ((5,), -1, -1),
          ((4, 6), -2, -1),
          ((4, 6), -1, -4),
          ((2, 3, 4), -1, -2),
          # Zero-element tensor
          ((0, 5), 1, 2),
          ((4, 0, 6), 0, 1),
      ]
      for input_sizes, dim, index in test_cases:
        grad_shape = list(input_sizes)
        del grad_shape[dim]
        cpu_grad = torch.randn(grad_shape, dtype=dtype)
        tpu_grad = cpu_grad.to("tpu")

        cpu_out = torch.ops.aten.select_backward(
            cpu_grad, input_sizes, dim, index
        )
        tpu_out = torch.ops.aten.select_backward(
            tpu_grad, input_sizes, dim, index
        )
        self.assertEqual(tpu_out.shape, cpu_out.shape)
        utils.assert_close(actual=tpu_out.cpu(), expected=cpu_out)

        @torch.compile(dynamic=False)
        def compiled_fn(g, sz=input_sizes, d=dim, idx=index):
          return torch.ops.aten.select_backward(g, sz, d, idx)

        compiled_out = compiled_fn(tpu_grad)
        self.assertEqual(compiled_out.shape, cpu_out.shape)
        utils.assert_close(actual=compiled_out.cpu(), expected=cpu_out)

  @contextlib.contextmanager
  def _highest_matmul_precision(self):
    prev_prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
      yield
    finally:
      torch.set_float32_matmul_precision(prev_prec)

  @parameterized.named_parameters(
      ("identity", "ij->ij", (3, 4)),
      ("transpose_2d", "ij->ji", (3, 4)),
      ("permute_3d", "ijk->kij", (2, 3, 4)),
      ("trace", "ii->", (4, 4)),
      ("batched_trace", "bii->b", (2, 4, 4)),
      ("diagonal", "ii->i", (4, 4)),
      ("batched_diagonal", "bii->bi", (2, 4, 4)),
      ("three_way_diagonal", "iii->i", (4, 4, 4)),
      ("partial_reduction", "ijk->ik", (2, 3, 4)),
      ("full_sum", "ij->", (3, 4)),
      ("implicit_identity", "ij", (3, 4)),
      ("implicit_trace", "ii", (4, 4)),
      ("implicit_ellipsis_trace", "...ii", (2, 4, 4)),
  )
  def test_einsum_one_operand(self, equation, shape):
    with self._highest_matmul_precision():
      x_cpu = torch.randn(shape, dtype=torch.float32, requires_grad=True)
      x_tpu = x_cpu.detach().to("tpu").requires_grad_(True)

      out_cpu = torch.einsum(equation, x_cpu)
      out_tpu = torch.einsum(equation, x_tpu)
      self.assertEqual(out_tpu.grad_fn.name(), "EinsumBackward0")
      utils.assert_close(actual=out_tpu.cpu(), expected=out_cpu)

      grad_out_cpu = torch.randn_like(out_cpu)
      out_cpu.backward(grad_out_cpu)
      out_tpu.backward(grad_out_cpu.to("tpu"))
      utils.assert_close(actual=x_tpu.grad.cpu(), expected=x_cpu.grad)

  @parameterized.named_parameters(
      ("inner_product", "i,i->", (6,), (6,)),
      ("outer_product", "i,j->ij", (4,), (5,)),
      ("matvec", "ij,j->i", (3, 4), (4,)),
      ("matmul", "ij,jk->ik", (3, 4), (4, 5)),
      ("matmul_nt", "ij,kj->ik", (3, 4), (5, 4)),
      ("matmul_tn", "ji,jk->ik", (4, 3), (4, 5)),
      ("matmul_tt", "ji,kj->ki", (4, 3), (5, 4)),
      ("bmm", "bij,bjk->bik", (2, 3, 4), (2, 4, 5)),
      ("mha", "bhqd,bhkd->bhqk", (2, 4, 6, 8), (2, 4, 7, 8)),
      ("gqa_explicit_size1", "bhqd,bhkd->bhqk", (2, 4, 6, 8), (2, 1, 7, 8)),
      ("gqa_lhs_size1", "bhqd,bhkd->bhqk", (2, 1, 6, 8), (2, 4, 7, 8)),
      ("gqa_rank_broadcast", "bhqd,bkd->bhqk", (2, 4, 6, 8), (2, 7, 8)),
      (
          "mamba3_6d",
          "bctrhd,bcsqhd->bhctrsq",
          (2, 2, 3, 2, 2, 4),
          (2, 2, 3, 2, 2, 4),
      ),
      ("mamba3_proj", "blhp,hrp->blrhp", (2, 3, 4, 5), (4, 2, 5)),
      ("uncontracted_sum_rhs", "ij,jk->i", (3, 4), (4, 5)),
      ("uncontracted_sum_both", "ijk,jkl->il", (2, 3, 4), (3, 4, 5)),
      ("full_sum_disjoint", "ij,kl->", (2, 3), (4, 5)),
      ("diag_lhs", "ii,ij->j", (4, 4), (4, 5)),
      ("batched_diag_lhs", "bii,bij->bj", (2, 4, 4), (2, 4, 5)),
      ("ellipsis_hadamard", "...,...->...", (2, 3), (2, 3)),
      ("ellipsis_bmm", "...ij,...jk->...ik", (2, 3, 4), (2, 4, 5)),
      ("ellipsis_bcast_size1", "...ij,...jk->...ik", (1, 3, 4), (2, 4, 5)),
      ("ellipsis_bcast_rank", "...ij,...jk->...ik", (3, 4), (2, 4, 5)),
      ("ellipsis_sum_batch", "...ij,...jk->ik", (2, 3, 4), (2, 4, 5)),
      ("implicit_matmul", "ij,jk", (3, 4), (4, 5)),
      ("implicit_ellipsis_bmm", "...ij,...jk", (2, 3, 4), (2, 4, 5)),
  )
  def test_einsum_two_operands(self, equation, shape_a, shape_b):
    with self._highest_matmul_precision():
      a_cpu = torch.randn(shape_a, dtype=torch.float32, requires_grad=True)
      b_cpu = torch.randn(shape_b, dtype=torch.float32, requires_grad=True)
      a_tpu = a_cpu.detach().to("tpu").requires_grad_(True)
      b_tpu = b_cpu.detach().to("tpu").requires_grad_(True)

      out_cpu = torch.einsum(equation, a_cpu, b_cpu)
      out_tpu = torch.einsum(equation, a_tpu, b_tpu)
      self.assertEqual(out_tpu.grad_fn.name(), "EinsumBackward0")
      utils.assert_close(
          actual=out_tpu.cpu(), expected=out_cpu, rtol=1e-4, atol=1e-4
      )

      grad_out_cpu = torch.randn_like(out_cpu)
      out_cpu.backward(grad_out_cpu)
      out_tpu.backward(grad_out_cpu.to("tpu"))
      utils.assert_close(
          actual=a_tpu.grad.cpu(), expected=a_cpu.grad, rtol=1e-4, atol=1e-4
      )
      utils.assert_close(
          actual=b_tpu.grad.cpu(), expected=b_cpu.grad, rtol=1e-4, atol=1e-4
      )

  def test_einsum_multi_operands_and_path(self):
    with self._highest_matmul_precision():
      # 3-operand chain matmul
      a_cpu = torch.randn((3, 4), dtype=torch.float32, requires_grad=True)
      b_cpu = torch.randn((4, 5), dtype=torch.float32, requires_grad=True)
      c_cpu = torch.randn((5, 2), dtype=torch.float32, requires_grad=True)
      a_tpu = a_cpu.detach().to("tpu").requires_grad_(True)
      b_tpu = b_cpu.detach().to("tpu").requires_grad_(True)
      c_tpu = c_cpu.detach().to("tpu").requires_grad_(True)

      out_cpu = torch.einsum("ij,jk,kl->il", a_cpu, b_cpu, c_cpu)
      out_tpu = torch.einsum("ij,jk,kl->il", a_tpu, b_tpu, c_tpu)
      self.assertEqual(out_tpu.grad_fn.name(), "EinsumBackward0")
      utils.assert_close(
          actual=out_tpu.cpu(), expected=out_cpu, rtol=1e-4, atol=1e-4
      )

      grad_out_cpu = torch.randn_like(out_cpu)
      out_cpu.backward(grad_out_cpu)
      out_tpu.backward(grad_out_cpu.to("tpu"))
      utils.assert_close(
          actual=a_tpu.grad.cpu(), expected=a_cpu.grad, rtol=1e-4, atol=1e-4
      )
      utils.assert_close(
          actual=b_tpu.grad.cpu(), expected=b_cpu.grad, rtol=1e-4, atol=1e-4
      )
      utils.assert_close(
          actual=c_tpu.grad.cpu(), expected=c_cpu.grad, rtol=1e-4, atol=1e-4
      )

      # Explicit contraction paths via torch.ops.aten.einsum.default
      for path in ([0, 1, 0, 1], [1, 2, 0, 1]):
        out_path_cpu = torch.ops.aten.einsum.default(
            "ij,jk,kl->il",
            [a_cpu.detach(), b_cpu.detach(), c_cpu.detach()],
            path=path,
        )
        out_path_tpu = torch.ops.aten.einsum.default(
            "ij,jk,kl->il",
            [a_tpu.detach(), b_tpu.detach(), c_tpu.detach()],
            path=path,
        )
        utils.assert_close(
            actual=out_path_tpu.cpu(),
            expected=out_path_cpu,
            rtol=1e-4,
            atol=1e-4,
        )

      # 4-operand contraction
      ops_cpu = [
          torch.randn((2, 3), dtype=torch.float32, requires_grad=True),
          torch.randn((3, 4), dtype=torch.float32, requires_grad=True),
          torch.randn((4, 3), dtype=torch.float32, requires_grad=True),
          torch.randn((3, 2), dtype=torch.float32, requires_grad=True),
      ]
      ops_tpu = [t.detach().to("tpu").requires_grad_(True) for t in ops_cpu]
      out4_cpu = torch.einsum("ab,bc,cd,de->ae", *ops_cpu)
      out4_tpu = torch.einsum("ab,bc,cd,de->ae", *ops_tpu)
      grad4_cpu = torch.randn_like(out4_cpu)
      out4_cpu.backward(grad4_cpu)
      out4_tpu.backward(grad4_cpu.to("tpu"))
      utils.assert_close(
          actual=out4_tpu.cpu(), expected=out4_cpu, rtol=1e-4, atol=1e-4
      )
      for t_tpu, t_cpu in zip(ops_tpu, ops_cpu):
        utils.assert_close(
            actual=t_tpu.grad.cpu(), expected=t_cpu.grad, rtol=1e-4, atol=1e-4
        )

  def test_einsum_double_backward(self):
    with self._highest_matmul_precision():
      cases = [
          ("ij->ji", [(3, 4)]),
          ("bii->bi", [(2, 4, 4)]),
          ("ij,jk->ik", [(3, 4), (4, 5)]),
          ("bhqd,bhkd->bhqk", [(2, 4, 3, 4), (2, 1, 5, 4)]),
          (
              "bctrhd,bcsqhd->bhctrsq",
              [(1, 2, 2, 2, 2, 3), (1, 2, 2, 2, 2, 3)],
          ),
          ("ij,jk,kl->il", [(2, 3), (3, 4), (4, 2)]),
      ]
      for equation, shapes in cases:
        with self.subTest(equation=equation):
          ops_cpu = [
              torch.randn(s, dtype=torch.float32, requires_grad=True)
              for s in shapes
          ]
          ops_tpu = [t.detach().to("tpu").requires_grad_(True) for t in ops_cpu]

          out_cpu = torch.einsum(equation, *ops_cpu)
          out_tpu = torch.einsum(equation, *ops_tpu)
          v_cpu = torch.randn_like(out_cpu, requires_grad=True)
          v_tpu = v_cpu.detach().to("tpu").requires_grad_(True)

          grads_cpu = torch.autograd.grad(
              out_cpu, ops_cpu, grad_outputs=out_cpu * v_cpu, create_graph=True
          )
          grads_tpu = torch.autograd.grad(
              out_tpu, ops_tpu, grad_outputs=out_tpu * v_tpu, create_graph=True
          )

          loss_cpu = sum((g * g).sum() for g in grads_cpu)
          loss_tpu = sum((g * g).sum() for g in grads_tpu)
          loss_cpu.backward()
          loss_tpu.backward()

          utils.assert_close(
              actual=v_tpu.grad.cpu(),
              expected=v_cpu.grad,
              rtol=1e-4,
              atol=1e-4,
          )
          for t_tpu, t_cpu in zip(ops_tpu, ops_cpu):
            utils.assert_close(
                actual=t_tpu.grad.cpu(),
                expected=t_cpu.grad,
                rtol=1e-4,
                atol=1e-4,
            )

  def test_einsum_partial_requires_grad(self):
    with self._highest_matmul_precision():
      for req_a, req_b in ((True, False), (False, True)):
        a_cpu = torch.randn((3, 4), dtype=torch.float32, requires_grad=req_a)
        b_cpu = torch.randn((4, 5), dtype=torch.float32, requires_grad=req_b)
        a_tpu = a_cpu.detach().to("tpu").requires_grad_(req_a)
        b_tpu = b_cpu.detach().to("tpu").requires_grad_(req_b)

        out_cpu = torch.einsum("ij,jk->ik", a_cpu, b_cpu)
        out_tpu = torch.einsum("ij,jk->ik", a_tpu, b_tpu)
        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_tpu.backward(grad_out.to("tpu"))

        if req_a:
          utils.assert_close(
              actual=a_tpu.grad.cpu(), expected=a_cpu.grad, rtol=1e-4, atol=1e-4
          )
        else:
          self.assertIsNone(a_tpu.grad)
        if req_b:
          utils.assert_close(
              actual=b_tpu.grad.cpu(), expected=b_cpu.grad, rtol=1e-4, atol=1e-4
          )
        else:
          self.assertIsNone(b_tpu.grad)

  def test_einsum_views_and_conj(self):
    with self._highest_matmul_precision():
      # Non-contiguous transposed and sliced views
      base_a_cpu = torch.randn((4, 6), dtype=torch.float32, requires_grad=True)
      base_b_cpu = torch.randn((5, 8), dtype=torch.float32, requires_grad=True)
      base_a_tpu = base_a_cpu.detach().to("tpu").requires_grad_(True)
      base_b_tpu = base_b_cpu.detach().to("tpu").requires_grad_(True)

      out_cpu = torch.einsum(
          "ij,jk->ik", base_a_cpu.t()[::2, :], base_b_cpu[:, ::2].t()
      )
      out_tpu = torch.einsum(
          "ij,jk->ik", base_a_tpu.t()[::2, :], base_b_tpu[:, ::2].t()
      )
      grad_out = torch.randn_like(out_cpu)
      out_cpu.backward(grad_out)
      out_tpu.backward(grad_out.to("tpu"))
      utils.assert_close(
          actual=out_tpu.cpu(), expected=out_cpu, rtol=1e-4, atol=1e-4
      )
      utils.assert_close(
          actual=base_a_tpu.grad.cpu(),
          expected=base_a_cpu.grad,
          rtol=1e-4,
          atol=1e-4,
      )
      utils.assert_close(
          actual=base_b_tpu.grad.cpu(),
          expected=base_b_cpu.grad,
          rtol=1e-4,
          atol=1e-4,
      )

      # Complex .conj() views (Wirtinger derivatives)
      ca_cpu = torch.randn((3, 4), dtype=torch.complex64, requires_grad=True)
      cb_cpu = torch.randn((4, 5), dtype=torch.complex64, requires_grad=True)
      ca_tpu = ca_cpu.detach().to("tpu").requires_grad_(True)
      cb_tpu = cb_cpu.detach().to("tpu").requires_grad_(True)

      cout_cpu = torch.einsum("ij,jk->ik", ca_cpu.conj(), cb_cpu.conj())
      cout_tpu = torch.einsum("ij,jk->ik", ca_tpu.conj(), cb_tpu.conj())
      cgrad_out = torch.randn_like(cout_cpu)
      cout_cpu.backward(cgrad_out)
      cout_tpu.backward(cgrad_out.to("tpu"))
      utils.assert_close(
          actual=cout_tpu.cpu(), expected=cout_cpu, rtol=1e-4, atol=1e-4
      )
      utils.assert_close(
          actual=ca_tpu.grad.cpu(), expected=ca_cpu.grad, rtol=1e-4, atol=1e-4
      )
      utils.assert_close(
          actual=cb_tpu.grad.cpu(), expected=cb_cpu.grad, rtol=1e-4, atol=1e-4
      )

  def test_einsum_zero_elements(self):
    cases = [
        ("ij,jk->ik", [(3, 0), (0, 4)]),
        ("ij,jk->", [(3, 0), (0, 4)]),
        ("bij,bjk->bik", [(0, 3, 4), (0, 4, 5)]),
        ("ij->ji", [(0, 4)]),
        ("ij->", [(0, 4)]),
    ]
    for equation, shapes in cases:
      with self.subTest(equation=equation, shapes=shapes):
        ops_cpu = [
            torch.randn(s, dtype=torch.float32, requires_grad=True)
            for s in shapes
        ]
        ops_tpu = [t.detach().to("tpu").requires_grad_(True) for t in ops_cpu]
        out_cpu = torch.einsum(equation, *ops_cpu)
        out_tpu = torch.einsum(equation, *ops_tpu)
        self.assertEqual(out_tpu.shape, out_cpu.shape)
        utils.assert_close(actual=out_tpu.cpu(), expected=out_cpu)

        grad_out = torch.ones_like(out_cpu)
        out_cpu.backward(grad_out)
        out_tpu.backward(grad_out.to("tpu"))
        for t_tpu, t_cpu in zip(ops_tpu, ops_cpu):
          self.assertEqual(t_tpu.grad.shape, t_cpu.grad.shape)
          utils.assert_close(actual=t_tpu.grad.cpu(), expected=t_cpu.grad)

  @parameterized.named_parameters(
      ("float32", torch.float32),
      ("bfloat16", torch.bfloat16),
      ("float16", torch.float16),
      ("float64", torch.float64),
      ("int8", torch.int8),
      ("uint8", torch.uint8),
      ("int16", torch.int16),
      ("int32", torch.int32),
      ("int64", torch.int64),
      ("complex64", torch.complex64),
      ("complex128", torch.complex128),
  )
  def test_einsum_dtypes(self, dtype):
    with self._highest_matmul_precision():
      is_floating_or_complex = dtype.is_floating_point or dtype.is_complex
      if is_floating_or_complex:
        a_cpu = torch.randn((2, 3, 4), dtype=dtype, requires_grad=True)
        b_cpu = torch.randn((2, 4, 5), dtype=dtype, requires_grad=True)
      else:
        a_cpu = torch.randint(1, 5, (2, 3, 4), dtype=dtype)
        b_cpu = torch.randint(1, 5, (2, 4, 5), dtype=dtype)

      a_tpu = a_cpu.detach().to("tpu").requires_grad_(is_floating_or_complex)
      b_tpu = b_cpu.detach().to("tpu").requires_grad_(is_floating_or_complex)

      out_cpu = torch.einsum("bij,bjk->bik", a_cpu, b_cpu)
      out_tpu = torch.einsum("bij,bjk->bik", a_tpu, b_tpu)
      rtol = 1e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-4
      atol = 1e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-4
      utils.assert_close(
          actual=out_tpu.cpu(), expected=out_cpu, rtol=rtol, atol=atol
      )

      if is_floating_or_complex:
        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_tpu.backward(grad_out.to("tpu"))
        utils.assert_close(
            actual=a_tpu.grad.cpu(), expected=a_cpu.grad, rtol=rtol, atol=atol
        )
        utils.assert_close(
            actual=b_tpu.grad.cpu(), expected=b_cpu.grad, rtol=rtol, atol=atol
        )

  def test_einsum_int_and_float_outer_product(self):
    with self._highest_matmul_precision():
      pos_cpu = torch.randint(0, 10, (2, 4), dtype=torch.int64)
      omega_cpu = torch.randn((8,), dtype=torch.float32, requires_grad=True)
      out_cpu = torch.einsum("...n,d->...nd", pos_cpu, omega_cpu)
      grad_out = torch.randn_like(out_cpu)
      out_cpu.backward(grad_out)

      pos_tpu = pos_cpu.to("tpu")
      omega_tpu = omega_cpu.detach().to("tpu").requires_grad_(True)
      out_tpu = torch.einsum("...n,d->...nd", pos_tpu, omega_tpu)
      self.assertEqual(out_tpu.dtype, torch.float32)
      out_tpu.backward(grad_out.to("tpu"))
      self.assertEqual(omega_tpu.grad.dtype, torch.float32)
      utils.assert_close(
          actual=out_tpu.cpu(), expected=out_cpu, rtol=1e-4, atol=1e-4
      )
      utils.assert_close(
          actual=omega_tpu.grad.cpu(),
          expected=omega_cpu.grad,
          rtol=1e-4,
          atol=1e-4,
      )

  @parameterized.named_parameters(
      ("outer_bf16", "i,j->ij", (4,), (5,), torch.bfloat16),
      ("outer_fp16", "i,j->ij", (4,), (5,), torch.float16),
      ("hadamard_bf16", "ij,ij->ij", (3, 4), (3, 4), torch.bfloat16),
      ("hadamard_fp16", "ij,ij->ij", (3, 4), (3, 4), torch.float16),
      (
          "broadcast_bf16",
          "...ij,...j->...ij",
          (2, 3, 4),
          (2, 4),
          torch.bfloat16,
      ),
      (
          "broadcast_fp16",
          "...ij,...j->...ij",
          (2, 3, 4),
          (2, 4),
          torch.float16,
      ),
  )
  def test_einsum_mixed_float_uncontracted(
      self, equation, shape_a, shape_b, low_dtype
  ):
    with self._highest_matmul_precision():
      a_cpu = torch.randn(shape_a, dtype=low_dtype, requires_grad=True)
      b_cpu = torch.randn(shape_b, dtype=torch.float32, requires_grad=True)
      out_cpu = torch.einsum(equation, a_cpu, b_cpu)
      grad_out = torch.randn_like(out_cpu)
      out_cpu.backward(grad_out)

      a_tpu = a_cpu.detach().to("tpu").requires_grad_(True)
      b_tpu = b_cpu.detach().to("tpu").requires_grad_(True)
      out_tpu = torch.einsum(equation, a_tpu, b_tpu)
      self.assertEqual(out_tpu.dtype, torch.float32)
      out_tpu.backward(grad_out.to("tpu"))
      self.assertEqual(a_tpu.grad.dtype, low_dtype)
      self.assertEqual(b_tpu.grad.dtype, torch.float32)
      utils.assert_close(
          actual=out_tpu.cpu(), expected=out_cpu, rtol=1e-2, atol=1e-2
      )
      utils.assert_close(
          actual=a_tpu.grad.cpu(), expected=a_cpu.grad, rtol=1e-2, atol=1e-2
      )
      utils.assert_close(
          actual=b_tpu.grad.cpu(), expected=b_cpu.grad, rtol=1e-2, atol=1e-2
      )

  def test_einsum_mixed_dtype_multi_operand_chain(self):
    with self._highest_matmul_precision():
      x_cpu = torch.randint(1, 5, (3,), dtype=torch.int64)
      y_cpu = torch.randn((4,), dtype=torch.float32, requires_grad=True)
      z_cpu = torch.randn((4, 5), dtype=torch.float32, requires_grad=True)
      out_cpu = torch.einsum("i,j,jk->ik", x_cpu, y_cpu, z_cpu)
      grad_out = torch.randn_like(out_cpu)
      out_cpu.backward(grad_out)

      x_tpu = x_cpu.to("tpu")
      y_tpu = y_cpu.detach().to("tpu").requires_grad_(True)
      z_tpu = z_cpu.detach().to("tpu").requires_grad_(True)
      out_tpu = torch.einsum("i,j,jk->ik", x_tpu, y_tpu, z_tpu)
      self.assertEqual(out_tpu.dtype, torch.float32)
      out_tpu.backward(grad_out.to("tpu"))
      utils.assert_close(
          actual=out_tpu.cpu(), expected=out_cpu, rtol=1e-4, atol=1e-4
      )
      utils.assert_close(
          actual=y_tpu.grad.cpu(), expected=y_cpu.grad, rtol=1e-4, atol=1e-4
      )
      utils.assert_close(
          actual=z_tpu.grad.cpu(), expected=z_cpu.grad, rtol=1e-4, atol=1e-4
      )

  def test_einsum_direct_dot_general_ir_and_compile(self):
    # Verify that 6D Mamba-3 einsum lowers directly to a single
    # stablehlo.dot_general without intermediate 3D reshapes.
    q = torch.randn((2, 2, 4, 2, 2, 8), dtype=torch.bfloat16).to("tpu")
    k = torch.randn((2, 2, 4, 2, 2, 8), dtype=torch.bfloat16).to("tpu")

    with execution_mode.set_eager_mode(
        execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
    ):
      out = torch.einsum("bctrhd,bcsqhd->bhctrsq", q, k)

    mlir_module = tpu_torch_compile.build_mlir([out], [q, k])
    mlir_text = tpu_torch_compile.serialize_mlir_text(mlir_module)
    self.assertIn("stablehlo.dot_general", mlir_text)
    self.assertNotIn("stablehlo.reshape", mlir_text)

    # Also verify torch.compile forward + backward numerical accuracy.
    def mamba3_qk(x, y):
      return torch.einsum("bctrhd,bcsqhd->bhctrsq", x, y)

    with self._highest_matmul_precision():
      q_cpu = torch.randn(
          (2, 2, 3, 2, 2, 4), dtype=torch.float32, requires_grad=True
      )
      k_cpu = torch.randn(
          (2, 2, 3, 2, 2, 4), dtype=torch.float32, requires_grad=True
      )
      q_tpu = q_cpu.detach().to("tpu").requires_grad_(True)
      k_tpu = k_cpu.detach().to("tpu").requires_grad_(True)

      out_cpu = mamba3_qk(q_cpu, k_cpu)
      out_tpu = torch.compile(mamba3_qk, backend="tpu")(q_tpu, k_tpu)
      grad_out = torch.randn_like(out_cpu)
      out_cpu.backward(grad_out)
      out_tpu.backward(grad_out.to("tpu"))

      utils.assert_close(
          actual=out_tpu.cpu(), expected=out_cpu, rtol=1e-4, atol=1e-4
      )
      utils.assert_close(
          actual=q_tpu.grad.cpu(), expected=q_cpu.grad, rtol=1e-4, atol=1e-4
      )
      utils.assert_close(
          actual=k_tpu.grad.cpu(), expected=k_cpu.grad, rtol=1e-4, atol=1e-4
      )

  def test_linalg_backward_no_tf32_guard_precision(self):
    # Verify that linalg.solve_triangular and linalg.lu_factor_ex backward
    # passes (which use at::NoTF32Guard in ATen) automatically run internal
    # matmuls at HIGHEST precision when global precision is DEFAULT ("medium")
    # and restore DEFAULT precision on guard exit.
    torch.manual_seed(0)
    a_cpu = torch.randn(32, 32, dtype=torch.float32)
    a_cpu = a_cpu @ a_cpu.mT + 4.0 * torch.eye(32, dtype=torch.float32)
    b_cpu = torch.randn(32, 16, dtype=torch.float32)

    # 1. linalg.solve_triangular backward under default ("medium") precision
    a_tri_cpu = a_cpu.triu().detach().requires_grad_(True)
    b_tri_cpu = b_cpu.clone().detach().requires_grad_(True)
    out_tri_cpu = torch.linalg.solve_triangular(
        a_tri_cpu, b_tri_cpu, upper=True
    )
    grad_tri_out = torch.randn_like(out_tri_cpu)
    out_tri_cpu.backward(grad_tri_out)

    a_tri_tpu = a_tri_cpu.detach().to("tpu").requires_grad_(True)
    b_tri_tpu = b_tri_cpu.detach().to("tpu").requires_grad_(True)
    out_tri_tpu = torch.linalg.solve_triangular(
        a_tri_tpu, b_tri_tpu, upper=True
    )
    out_tri_tpu.backward(grad_tri_out.to("tpu"))

    utils.assert_close(
        actual=a_tri_tpu.grad.cpu(),
        expected=a_tri_cpu.grad,
        rtol=1.3e-6,
        atol=1e-5,
    )
    utils.assert_close(
        actual=b_tri_tpu.grad.cpu(),
        expected=b_tri_cpu.grad,
        rtol=1.3e-6,
        atol=1e-5,
    )

    # 2. linalg.lu_factor_ex backward under default ("medium") precision
    a_lu_cpu = a_cpu.clone().detach().requires_grad_(True)
    lu_cpu, _, _ = torch.linalg.lu_factor_ex(a_lu_cpu)
    grad_lu_out = torch.randn_like(lu_cpu)
    lu_cpu.backward(grad_lu_out)

    a_lu_tpu = a_cpu.detach().to("tpu").requires_grad_(True)
    lu_tpu, _, _ = torch.linalg.lu_factor_ex(a_lu_tpu)
    lu_tpu.backward(grad_lu_out.to("tpu"))

    utils.assert_close(
        actual=a_lu_tpu.grad.cpu(),
        expected=a_lu_cpu.grad,
        rtol=2.3e-5,
        atol=4.8e-4,
    )

    # 3. Verify prior precision (DEFAULT) is restored after NoTF32Guard exits
    with execution_mode.set_eager_mode(
        execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
    ):
      mm_out = torch.matmul(a_lu_tpu.detach(), b_tri_tpu.detach())
    mlir_module = tpu_torch_compile.build_mlir(
        [mm_out], [a_lu_tpu.detach(), b_tri_tpu.detach()]
    )
    mlir_text = tpu_torch_compile.serialize_mlir_text(mlir_module)
    self.assertIn("precision = [DEFAULT, DEFAULT]", mlir_text)
    self.assertNotIn("precision = [HIGHEST, HIGHEST]", mlir_text)

  def test_sum_unsigned_dtypes(self):
    for dtype in (torch.uint16, torch.uint32, torch.uint64):
      with self.subTest(dtype=dtype):
        x_cpu = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        x_tpu = x_cpu.to("tpu")
        self.assertEqual(
            torch.sum(x_tpu, dim=0, dtype=dtype).cpu().tolist(),
            torch.sum(x_cpu.to(torch.int64), dim=0).tolist(),
        )
        self.assertEqual(
            torch.sum(x_tpu, dtype=dtype).cpu().item(),
            torch.sum(x_cpu.to(torch.int64)).item(),
        )

  def test_addcdiv_mixed_dtypes(self):
    with self.subTest("int32_self_float32_tensors"):

      def compute_int_float(device):
        s = torch.tensor([1, 2], dtype=torch.int32, device=device)
        t1 = torch.tensor([3.0, 5.0], dtype=torch.float32, device=device)
        t2 = torch.tensor([2.0, 2.0], dtype=torch.float32, device=device)
        out = torch.empty(2, dtype=torch.float32, device=device)
        res_func = torch.addcdiv(s, t1, t2, value=2.5)
        torch.addcdiv(s, t1, t2, value=2.5, out=out)
        return res_func, out

      self.assert_close_tpu_vs_cpu(compute_int_float)

    with self.subTest("float32_self_float64_tensors"):

      def compute_f32_f64(device):
        s = torch.tensor(
            [-16777216.0, -33554432.0], dtype=torch.float32, device=device
        )
        t1 = torch.tensor(
            [16777217.0, 67108866.0], dtype=torch.float64, device=device
        )
        t2 = torch.tensor([1.0, 2.0], dtype=torch.float64, device=device)
        out_f32 = torch.empty(2, dtype=torch.float32, device=device)
        res_func = torch.addcdiv(s, t1, t2)
        torch.addcdiv(s, t1, t2, out=out_f32)
        return res_func, out_f32

      self.assert_close_tpu_vs_cpu(compute_f32_f64)

    with self.subTest("float32_self_complex64_tensors"):

      def compute_f32_c64(device):
        s = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
        t1 = torch.tensor(
            [2.0 + 4.0j, 6.0 - 2.0j], dtype=torch.complex64, device=device
        )
        t2 = torch.tensor(
            [1.0 + 1.0j, 2.0 + 0.0j], dtype=torch.complex64, device=device
        )
        out = torch.empty(2, dtype=torch.complex64, device=device)
        res_func = torch.addcdiv(s, t1, t2, value=1.0 + 2.0j)
        torch.addcdiv(s, t1, t2, value=1.0 + 2.0j, out=out)
        return res_func, out

      self.assert_close_tpu_vs_cpu(compute_f32_c64)

  def test_add_float16_opmath_alpha(self):
    a = torch.zeros(2, dtype=torch.float16, device="tpu")
    b = torch.full((2,), 1e-4, dtype=torch.float16, device="tpu")
    # PyTorch CUDA/TPU cast alpha to opmath_type (float32 for float16), whereas
    # PyTorch CPU casts alpha directly to float16 and overflows for 70000.0.
    expected = (a.cpu().float() + 70000.0 * b.cpu().float()).to(torch.float16)
    res = torch.add(a, b, alpha=70000.0)
    self.assertEqual(res.dtype, torch.float16)
    utils.assert_close(actual=res.cpu(), expected=expected)

    out = torch.empty_like(a)
    torch.add(a, b, alpha=70000.0, out=out)
    self.assertEqual(out.dtype, torch.float16)
    utils.assert_close(actual=out.cpu(), expected=expected)

  def test_addmm_bool_scalars_on_float(self):
    b = torch.ones(2, 2, dtype=torch.float32)
    m1 = torch.full((2, 2), 2.0, dtype=torch.float32)
    m2 = torch.full((2, 2), 3.0, dtype=torch.float32)
    for beta, alpha in ((False, True), (True, True), (True, False)):
      with self.subTest(beta=beta, alpha=alpha):
        self.assert_close_tpu_vs_cpu(
            lambda device, beta=beta, alpha=alpha: torch.addmm(
                b.to(device),
                m1.to(device),
                m2.to(device),
                beta=beta,
                alpha=alpha,
            )
        )

  def test_addmm_out_dtype_float32_from_low_precision(self):
    for mat_dtype in (torch.bfloat16, torch.float16):
      for self_dtype in (mat_dtype, torch.float32):
        for beta, alpha in ((1.0, 1.0), (0.0, 2.0), (2.0, 3.0)):
          with self.subTest(
              mat_dtype=mat_dtype, self_dtype=self_dtype, beta=beta, alpha=alpha
          ):
            b_cpu = torch.ones(3, 4, dtype=self_dtype)
            m1_cpu = torch.ones(3, 5, dtype=mat_dtype)
            m2_cpu = torch.ones(5, 4, dtype=mat_dtype)
            expected = beta * b_cpu.float() + alpha * (
                m1_cpu.float() @ m2_cpu.float()
            )

            b_tpu = b_cpu.to("tpu")
            m1_tpu = m1_cpu.to("tpu")
            m2_tpu = m2_cpu.to("tpu")
            res = torch.addmm(
                b_tpu,
                m1_tpu,
                m2_tpu,
                beta=beta,
                alpha=alpha,
                out_dtype=torch.float32,
            )
            self.assertEqual(res.dtype, torch.float32)
            utils.assert_close(res.cpu(), expected)

            out_tpu = torch.empty(3, 4, dtype=torch.float32, device="tpu")
            torch.addmm(
                b_tpu,
                m1_tpu,
                m2_tpu,
                beta=beta,
                alpha=alpha,
                out_dtype=torch.float32,
                out=out_tpu,
            )
            self.assertEqual(out_tpu.dtype, torch.float32)
            utils.assert_close(out_tpu.cpu(), expected)


if __name__ == "__main__":
  absltest.main()
