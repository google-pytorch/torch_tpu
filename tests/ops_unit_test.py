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

import concurrent
import contextlib
import dataclasses
import enum
import functools
import itertools
import math
import os
import plistlib
import sys
import tempfile
import threading
import time
import typing
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from scipy import stats
import torch
from torch.testing._internal import common_methods_invocations
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.utils import test_utils as utils
from tests import op_testing
from tests import ops_test_data

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


class OpsUnitTest(TorchTpuVsCpuTestBase, parameterized.TestCase):
  """Tests for ops using custom values.

  If a bug is found that's not covered by do_test_op() in ops_test.py, please
  add it here.
  """

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

      self.assert_close_tpu_vs_cpu(compute_bwd_direct)

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

        self.assert_close_tpu_vs_cpu(run, atol=0.0)

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

  @parameterized.product(
      batch_size=[1, 2],
      in_channels=[4],
      out_channels=[8],
      length=[16],
      kernel_size=[3],
      stride=[1, 2],
      padding=[0, 1],
      output_padding=[0],
      groups=[1, 2],
      bias=[True, False],
      dtype=[torch.float32, torch.bfloat16],
  )
  def test_conv_transpose1d(
      self,
      batch_size,
      in_channels,
      out_channels,
      length,
      kernel_size,
      stride,
      padding,
      output_padding,
      groups,
      bias,
      dtype,
  ):
    """Tests torch.nn.functional.conv_transpose1d."""
    input_value = torch.randn(batch_size, in_channels, length, dtype=dtype)
    weight_value = torch.randn(
        in_channels,
        out_channels // groups,
        kernel_size,
        dtype=dtype,
    )
    bias_value = torch.randn(out_channels, dtype=dtype)

    def compute(device):
      output = torch.nn.functional.conv_transpose1d(
          input_value.to(device),
          weight_value.to(device),
          bias=(bias_value.to(device) if bias else None),
          stride=stride,
          padding=padding,
          output_padding=output_padding,
          groups=groups,
      )
      return output

    self.assert_close_tpu_vs_cpu(compute, rtol=5e-1, atol=5e-1)

  @parameterized.product(
      batch_size=[1, 2],
      in_channels=[4],
      out_channels=[8],
      h=[16],
      w=[16],
      kernel_size=[3],
      stride=[1, 2],
      padding=[0, 1],
      output_padding=[0],
      groups=[1, 2],
      bias=[True, False],
      dtype=[torch.float32, torch.bfloat16],
  )
  def test_conv_transpose2d(
      self,
      batch_size,
      in_channels,
      out_channels,
      h,
      w,
      kernel_size,
      stride,
      padding,
      output_padding,
      groups,
      bias,
      dtype,
  ):
    """Tests torch.nn.functional.conv_transpose2d."""
    input_value = torch.randn(batch_size, in_channels, h, w, dtype=dtype)
    weight_value = torch.randn(
        in_channels,
        out_channels // groups,
        kernel_size,
        kernel_size,
        dtype=dtype,
    )
    bias_value = torch.randn(out_channels, dtype=dtype)

    def compute(device):
      output = torch.nn.functional.conv_transpose2d(
          input_value.to(device),
          weight_value.to(device),
          bias=(bias_value.to(device) if bias else None),
          stride=stride,
          padding=padding,
          output_padding=output_padding,
          groups=groups,
      )
      return output

    self.assert_close_tpu_vs_cpu(compute, rtol=5e-1, atol=5e-1)

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

  @absltest.skip("b/540887166")
  def test_torch_tpu_vs_gpu_missing_golden_fails(self):
    op = next(op for op in op_db if op.name == "add")
    with (
        flagsaver.flagsaver(test_mode=op_testing.TestMode.TORCH_TPU_VS_GPU),
        mock.patch.dict(op_testing._GOLDEN_GPU_DATA, clear=True),
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
      fake_sample = (
          op_testing.OpInput(FakeSample("s", torch.zeros(1), (), {})),
          op_testing.OpOutput(torch.zeros(1)),
      )
      op_testing._GOLDEN_GPU_DATA["add"] = {torch.float32: [fake_sample]}
      res = self._get_golden_input_output_pairs(
          op=op,
          dtype=torch.float32,
          variant=op_testing.OpVariant.BASE,
          max_samples=1,
      )
      self.assertEqual(res, [fake_sample])

  @parameterized.product(training=[True, False])
  def test_batch_norm_forward_mixed_dtype(self, training):
    """Tests batch_norm forward with mixed BF16 input and F32 stats."""
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
          training,
          0.1,  # momentum
          1e-5,  # eps
      )

    # Comparison results: (output, save_mean, save_invstd)
    # output should be BF16, stats should be F32
    # Relax tolerances because we are running in BF16 on TPU for performance.
    self.assert_close_tpu_vs_cpu(run_forward, rtol=3e-2, atol=2e-2)

  @parameterized.product(training=[True, False])
  def test_batch_norm_backward_mixed_dtype(self, training):
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
    weight = torch.randn(c, dtype=stats_dtype)
    running_mean = torch.randn(c, dtype=stats_dtype)
    running_var = torch.rand(c, dtype=stats_dtype)  # Positive variance
    save_mean = torch.randn(c, dtype=stats_dtype)
    save_invstd = torch.rand(c, dtype=stats_dtype)  # Positive invstd

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
        rtol=0.0,
        atol=0.0,
    )

  @parameterized.product(
      m=[32, 128],
      n=[32, 128],
      k=[128],
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

  def test_swizzle_math(self):
    # Test LHS
    logical_lhs = torch.randn(128, 8)
    swizzled_lhs = op_testing._swizzle_lhs_scale(logical_lhs)
    unswizzled_lhs = op_testing._unswizzle_lhs_scale(swizzled_lhs)
    self.assertTrue(torch.equal(logical_lhs, unswizzled_lhs))

    # Test RHS
    logical_rhs = torch.randn(8, 128)
    swizzled_rhs = op_testing._swizzle_rhs_scale(logical_rhs)
    unswizzled_rhs = op_testing._unswizzle_rhs_scale(swizzled_rhs)
    self.assertTrue(torch.equal(logical_rhs, unswizzled_rhs))

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
        rtol=0.0,
        atol=0.0,
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
    torch.testing.assert_close(  # TORCH_ASSERT_CLOSE_OK=fp4
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
      (torch.float16,),
      (torch.bfloat16,),
      (torch.int32,),
      (torch.int64,),
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
      return torch.nn.functional.col2im(
          col_dev, output_size, kernel_size, dilation, padding, stride
      )

    # TODO(b/489136147): Fix test case & disable allow_failure
    self.assert_close_tpu_vs_cpu(test_fn, allow_failure=True)

  @parameterized.parameters(
      # (kernel_size, dilation, padding, stride, output_size)
      ((1, 1), (1, 1), (0, 0), (1, 1), (4, 4)),  # 1x1
      ((2, 3), (1, 1), (0, 0), (1, 1), (4, 4)),  # asymmetric kernel
      ((2, 2), (2, 2), (0, 0), (1, 1), (8, 8)),  # dilation
      ((2, 2), (1, 1), (1, 1), (1, 1), (4, 4)),  # padding
      ((2, 2), (1, 1), (0, 0), (2, 2), (4, 4)),  # stride
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
      return torch.nn.functional.col2im(
          col_dev, output_size, kernel_size, dilation, padding, stride
      )

    # TODO(b/489136147): Fix test case & disable allow_failure
    self.assert_close_tpu_vs_cpu(test_fn, allow_failure=True)

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

    # Test with float16 and bfloat16 dtypes. These were excluded from the
    # general op tests due to expected binning errors.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.histc(
            torch.tensor(inputs, dtype=torch.float16).to(device=device),
            bins=bins,
        ),
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.histc(
            torch.tensor(inputs, dtype=torch.bfloat16).to(device=device),
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
        rtol=0.0,
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

    # Test properties (non-negative)
    self.assertGreaterEqual(t1.min(), 0)

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
          rtol=0.0,
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

  def test_boolean_indexing(self):
    """Test boolean indexing."""

    def single_tensor_bool_indexing_with(device):
      # Create sample tensors
      y = torch.tensor(
          [0, 1, 2, 5, 10, -1, 3, 255], dtype=torch.long, device=device
      )
      y_pred = torch.tensor(
          [0, 1, 2, 4, 8, 2, 3, 200], dtype=torch.long, device=device
      )

      num_classes = 10

      print(f"Original y: {y}")
      print(f"Original y_pred: {y_pred}")

      target_mask = (y >= 0) & (y < num_classes)
      print(f"\nBoolean mask: {target_mask}")

      y_filtered = y[target_mask]
      y_pred_filtered = y_pred[target_mask]

      print(f"\nFiltered y: {y_filtered}")
      print(f"Filtered y_pred: {y_pred_filtered}")
      print("\nSUCCESS: Boolean indexing works!")
      return y_filtered, y_pred_filtered

    self.assert_close_tpu_vs_cpu(single_tensor_bool_indexing_with)

  def test_boolean_indexing_with_same_rank_mask(self):
    x = torch.rand(8, 128, 128)
    mask = torch.rand(8, 128, 128) > 0.5

    def boolean_indexing_with_same_rank_mask(device):
      x_tpu = x.to(device)
      mask_tpu = mask.to(device)
      return x_tpu[mask_tpu]

    self.assert_close_tpu_vs_cpu(boolean_indexing_with_same_rank_mask)

  def test_boolean_indexing_with_smaller_rank_mask(self):
    x = torch.rand(8, 128, 128)
    mask = torch.rand(8, 128) > 0.5

    def boolean_indexing_with_smaller_rank_mask(device):
      x_tpu = x.to(device)
      mask_tpu = mask.to(device)
      return x_tpu[mask_tpu]

    self.assert_close_tpu_vs_cpu(boolean_indexing_with_smaller_rank_mask)

  def test_is_nonzero(self):
    def assert_is_nonzero_equal_on_cpu_vs_tpu(tensor: torch.Tensor):
      cpu_result = torch.is_nonzero(tensor.to("cpu"))
      tpu_result = torch.is_nonzero(tensor.to(torch.device("tpu")))
      self.assertEqual(cpu_result, tpu_result)
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
            repeats=0,
        ),
    )

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
    """Tests that embedding works with a scalar index."""
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
        rtol=0.0,
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
    src = torch.ones(t * k, feat, device="tpu")
    with execution_mode.set_eager_mode(
        execution_mode.EagerMode.INTERNAL_DEFER_ALL
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

  def test_arange_infinite_step(self):
    """Tests arange where step is infinite."""

    def _test_positive(device):
      return torch.arange(10, 20, float("inf"), device=device)

    def _test_negative(device):
      return torch.arange(20, 10, float("-inf"), device=device)

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
          torch.int32,
          torch.int64,
          torch.float32,
          torch.float64,
          torch.complex64,
      ],
  )
  def test_geqrf(self, shape, dtype):
    if dtype in (torch.int32, torch.int64):
      input_tensor = torch.randint(-10, 10, shape, dtype=dtype)
    else:
      input_tensor = torch.randn(shape, dtype=dtype)

    def run(device):
      # torch.geqrf on CPU does not support integer types, whereas the TPU
      # implementation automatically promotes them to floating point.
      if device == "cpu" and dtype in (torch.int32, torch.int64):
        return torch.geqrf(input_tensor.to(torch.float64).to(device))
      else:
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
    if dtype in (torch.int32, torch.int64):
      input_tensor = torch.randint(-10, 10, shape, dtype=dtype)
    else:
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

  def test_assert_async_stub(self):
    """Tests _assert_async and _assert_async.msg as no-ops."""
    device = torch.device("tpu")
    true_cond = torch.tensor(True, device=device)
    false_cond = torch.tensor(False, device=device)

    # Should not fail regardless of condition value.
    torch.ops.aten._assert_async(true_cond)
    torch.ops.aten._assert_async(false_cond)
    torch.ops.aten._assert_async.msg(true_cond, "should not fail")
    torch.ops.aten._assert_async.msg(false_cond, "intended failure but no-op")

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

  def test_ceil_integral(self):
    for dtype in [torch.int32, torch.uint8]:
      x = torch.tensor([-2, -1, 0, 1, 2], dtype=dtype)
      self.assert_close_tpu_vs_cpu(lambda device, x=x: torch.ceil(x.to(device)))
      self.assert_close_tpu_vs_cpu(
          lambda device, x=x: torch.ceil_(x.clone().to(device))
      )

    # For boolean, CPU doesn't support it, so test TPU execution alone.
    x_bool = torch.tensor([True, False, True], dtype=torch.bool)
    x_tpu = x_bool.to("tpu")
    self.assertEqual(torch.ceil(x_tpu).cpu(), x_bool)
    self.assertEqual(torch.ceil_(x_tpu.clone()).cpu(), x_bool)

  def test_floor_integral(self):
    for dtype in [torch.int32, torch.uint8]:
      x = torch.tensor([-2, -1, 0, 1, 2], dtype=dtype)
      self.assert_close_tpu_vs_cpu(
          lambda device, x=x: torch.floor(x.to(device))
      )
      self.assert_close_tpu_vs_cpu(
          lambda device, x=x: torch.floor_(x.clone().to(device))
      )

    # For boolean, CPU doesn't support it, so test TPU execution alone.
    x_bool = torch.tensor([True, False, True], dtype=torch.bool)
    x_tpu = x_bool.to("tpu")
    self.assertEqual(torch.floor(x_tpu).cpu(), x_bool)
    self.assertEqual(torch.floor_(x_tpu.clone()).cpu(), x_bool)

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

    # For boolean, CPU doesn't support it, so test TPU execution alone.
    x_bool = [
        torch.tensor([True, False, True], dtype=torch.bool),
        torch.tensor([False, True], dtype=torch.bool),
    ]
    x_tpu = [t.to("tpu") for t in x_bool]
    res_tpu = torch._foreach_ceil(x_tpu)
    for res, expected in zip(res_tpu, x_bool):
      self.assertEqual(res.cpu(), expected)

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

    # For boolean, CPU doesn't support it, so test TPU execution alone.
    x_bool = [
        torch.tensor([True, False, True], dtype=torch.bool),
        torch.tensor([False, True], dtype=torch.bool),
    ]
    x_tpu = [t.to("tpu") for t in x_bool]
    res_tpu = torch._foreach_floor(x_tpu)
    for res, expected in zip(res_tpu, x_bool):
      self.assertEqual(res.cpu(), expected)

  def test_sort_indices(self):
    """Tests that torch.sort returns correct indices (dtype and values)."""
    x = torch.tensor([5.0, 1.0, 3.0, 2.0, 4.0], dtype=torch.float32)

    def run(device):
      values, indices = torch.sort(x.to(device))
      return values, indices

    self.assert_close_tpu_vs_cpu(run)


class OpsCustomOpUnitTest(TorchTpuVsCpuTestBase, parameterized.TestCase):
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
    out = torch.ops.tpu.ragged_dot(lhs, rhs, group_sizes)
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
    res = torch.ops.tpu.ragged_dot(lhs, rhs, group_sizes, out=out)
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

  def test_experimental_op_warning_once(self):
    """Verifies that experimental ops warn exactly once per operator."""

    @contextlib.contextmanager
    def capture_c_stderr():
      sys.stdout.flush()
      sys.stderr.flush()
      original_stderr_fd = 2
      saved_stderr_fd = os.dup(original_stderr_fd)
      tfile = tempfile.TemporaryFile(mode="w+b")
      os.dup2(tfile.fileno(), original_stderr_fd)
      try:
        yield tfile
      finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stderr_fd, original_stderr_fd)
        os.close(saved_stderr_fd)

    device = torch.device("tpu")
    m, k, n, g = 5, 4, 3, 2
    lhs = torch.arange(m * k, dtype=torch.float32, device=device).reshape(m, k)
    rhs = torch.arange(g * k * n, dtype=torch.float32, device=device).reshape(
        g, k, n
    )
    group_sizes = torch.tensor([1, 4], dtype=torch.int32, device=device)

    # 1. First call to ragged_dot should warn.
    with capture_c_stderr() as f:
      _ = torch.ops.tpu.ragged_dot(lhs, rhs, group_sizes)
      f.seek(0)
      output = f.read().decode()
      self.assertIn("Warning: operator ragged_dot is experimental", output)

    # 2. Second call to ragged_dot should NOT warn.
    with capture_c_stderr() as f:
      _ = torch.ops.tpu.ragged_dot(lhs, rhs, group_sizes)
      f.seek(0)
      output = f.read().decode()
      self.assertNotIn("experimental", output)

    # 3. First call to ragged_dot_out (different op) should warn.
    out = torch.zeros(m, n, dtype=torch.float32, device=device)
    with capture_c_stderr() as f:
      _ = torch.ops.tpu.ragged_dot(lhs, rhs, group_sizes, out=out)
      f.seek(0)
      output = f.read().decode()
      self.assertIn("Warning: operator ragged_dot.out is experimental", output)

    # 4. Second call to ragged_dot_out should NOT warn.
    with capture_c_stderr() as f:
      _ = torch.ops.tpu.ragged_dot(lhs, rhs, group_sizes, out=out)
      f.seek(0)
      output = f.read().decode()
      self.assertNotIn("experimental", output)

  @absltest.skip("b/498564738")
  def test_set_dimension_logical_size_on_tpu(self):
    """Tests the tpu.set_dimension_logical_size custom op on TPU."""
    device = torch.device("tpu")
    x = torch.arange(100, device=device, dtype=torch.int32).reshape(10, 10)
    size = torch.tensor(5, device=device, dtype=torch.int32)
    out = torch.ops.tpu.set_dimension_logical_size(x, 0, size)
    self.assert_close(golden_result=x[:5].cpu(), torch_tpu_result=out[:5].cpu())

  def test_set_dimension_logical_size_with_mlir_on_tpu(self):
    """Tests the tpu.set_dimension_logical_size custom op on TPU using MLIR."""
    device = torch.device("tpu")
    x = torch.arange(25, device=device, dtype=torch.int32).reshape(5, 5)
    y = torch.arange(25, device=device, dtype=torch.int32).reshape(5, 5)
    size = torch.tensor(1, device=device, dtype=torch.int32)
    golden_result = torch.matmul(x[:, : size.item()], y[: size.item(), :])
    mlir_program = """
module {
  func.func @main(%arg0: tensor<5x5xi32>, %arg1: tensor<5x5xi32>, %arg2: tensor<i32>) -> tensor<5x5xi32> {
    %0 = stablehlo.set_dimension_size %arg0, %arg2, dim=1 : (tensor<5x5xi32>, tensor<i32>) -> tensor<5x?xi32, #stablehlo.bounds<?, 5>>
    %1 = stablehlo.set_dimension_size %arg1, %arg2, dim=0 : (tensor<5x5xi32>, tensor<i32>) -> tensor<?x5xi32, #stablehlo.bounds<5, ?>>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<5x?xi32, #stablehlo.bounds<?, 5>>, tensor<?x5xi32, #stablehlo.bounds<5, ?>>) -> tensor<5x5xi32>
    return %2: tensor<5x5xi32>
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
    """Tests the tpu.dynamic_arange custom op on TPU with integers."""
    device = torch.device("tpu")
    start = torch.tensor(1, device=device, dtype=torch.int32)
    end = torch.tensor(6, device=device, dtype=torch.int32)
    step = torch.tensor(2, device=device, dtype=torch.int32)
    max_length = 5
    out = torch.ops.tpu.dynamic_arange(
        start, end, step, max_length, torch.int32
    )
    expected = torch.arange(1, 6, step=2, dtype=torch.int32)
    self.assert_close(
        golden_result=expected, torch_tpu_result=out[: expected.size(0)].cpu()
    )

  def test_dynamic_arange_float_on_tpu(self):
    """Tests the tpu.dynamic_arange custom op on TPU with floats."""
    device = torch.device("tpu")
    start = torch.tensor(0.0, device=device, dtype=torch.float32)
    end = torch.tensor(2.5, device=device, dtype=torch.float32)
    step = torch.tensor(0.5, device=device, dtype=torch.float32)
    max_length = 6
    out = torch.ops.tpu.dynamic_arange(
        start, end, step, max_length, torch.float32
    )
    expected = torch.arange(0.0, 2.5, step=0.5, dtype=torch.float32)
    self.assert_close(
        golden_result=expected, torch_tpu_result=out[: expected.size(0)].cpu()
    )

  def test_dynamic_arange_empty_on_tpu(self):
    """Tests the tpu.dynamic_arange custom op on TPU with an empty range."""
    device = torch.device("tpu")
    start = torch.tensor(5, device=device, dtype=torch.int32)
    end = torch.tensor(2, device=device, dtype=torch.int32)
    step = torch.tensor(1, device=device, dtype=torch.int32)
    max_length = 5
    out = torch.ops.tpu.dynamic_arange(
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

    out = torch.ops.tpu.dynamic_broadcast(
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

    out = torch.ops.tpu.dynamic_broadcast(
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
      x_dynamic = torch.ops.tpu.set_dimension_logical_size(x, 0, size)

      # Broadcast to physical [3, 4], logical [2, 4]
      # Input dim 0 (dynamic) maps to output dim 0 (dynamic)
      shape = [
          torch.tensor(2, device=device, dtype=torch.int32),
          torch.tensor(4, device=device, dtype=torch.int32),
      ]
      broadcast_dims = [0]
      static_shape = [3, 4]
      is_dynamic = [True, False]

      out = torch.ops.tpu.dynamic_broadcast(
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
      x_dynamic = torch.ops.tpu.set_dimension_logical_size(x, 0, size)

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

      out = torch.ops.tpu.dynamic_broadcast(
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

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Make input dynamic: physical [3, 4], logical [3, 3] (dim 1 is dynamic)
      x_dynamic = torch.ops.tpu.set_dimension_logical_size(x, 1, size)

      # Target shape: physical [12], logical [9] (dim 0 is dynamic)
      shape = [torch.tensor(9, device=device, dtype=torch.int32)]
      static_shape = [12]
      is_dynamic = [True]

      out = torch.ops.tpu.dynamic_reshape(
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

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Make input dynamic: physical [3, 4], logical [3, 2] (dim 1 is dynamic)
      x_dynamic = torch.ops.tpu.set_dimension_logical_size(x, 1, size)

      # Target shape: physical [2, 6], logical [2, 3] (dim 1 is dynamic)
      shape = [
          torch.tensor(2, device=device, dtype=torch.int32),
          torch.tensor(3, device=device, dtype=torch.int32),
      ]
      static_shape = [2, 6]
      is_dynamic = [False, True]

      out = torch.ops.tpu.dynamic_reshape(
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

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Make input dynamic: physical [3, 4], logical [3, 2] (dim 1 is dynamic)
      x_dynamic = torch.ops.tpu.set_dimension_logical_size(x, 1, size)

      # Target shape: physical [1, 3, 1, 4], logical [1, 3, 1, 2]
      shape = [
          torch.tensor(1, device=device, dtype=torch.int32),
          torch.tensor(3, device=device, dtype=torch.int32),
          torch.tensor(1, device=device, dtype=torch.int32),
          torch.tensor(2, device=device, dtype=torch.int32),
      ]
      static_shape = [1, 3, 1, 4]
      is_dynamic = [False, False, False, True]
      out = torch.ops.tpu.dynamic_reshape(
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

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Make input dynamic: physical [12], logical [8] (dim 0 is dynamic)
      x_dynamic = torch.ops.tpu.set_dimension_logical_size(x, 0, size)

      # Target shape: physical [2, 6], logical [2, 4] (dim 1 is dynamic)
      shape = [
          torch.tensor(2, device=device, dtype=torch.int32),
          torch.tensor(4, device=device, dtype=torch.int32),
      ]
      static_shape = [2, 6]
      is_dynamic = [False, True]

      out = torch.ops.tpu.dynamic_reshape(
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

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Make input dynamic: physical [3, 4], logical [2, 4] (dim 0 is dynamic)
      x_dynamic = torch.ops.tpu.set_dimension_logical_size(x, 0, size)

      # Target shape: physical [2, 6], logical [2, 4] (dim 1 is dynamic)
      shape = [
          torch.tensor(2, device=device, dtype=torch.int32),
          torch.tensor(4, device=device, dtype=torch.int32),
      ]
      static_shape = [2, 6]
      is_dynamic = [False, True]

      out = torch.ops.tpu.dynamic_reshape(
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

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Make input dynamic: physical [2, 6], logical [2, 4] (dim 1 is dynamic)
      x_dynamic = torch.ops.tpu.set_dimension_logical_size(x, 1, size)

      # Target shape: physical [2, 2, 3], logical [2, 2, 2] (dim 2 is dynamic)
      shape = [
          torch.tensor(2, device=device, dtype=torch.int32),
          torch.tensor(2, device=device, dtype=torch.int32),
          torch.tensor(2, device=device, dtype=torch.int32),
      ]
      static_shape = [2, 2, 3]
      is_dynamic = [False, False, True]

      out = torch.ops.tpu.dynamic_reshape(
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
      dtype=[torch.float32, torch.bfloat16],
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
      dtype=[torch.float32, torch.bfloat16],
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
      dtype=[torch.float32, torch.bfloat16],
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
      dtype=[torch.float32, torch.bfloat16],
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


class OpsGradUnitTest(TorchTpuVsCpuTestBase, parameterized.TestCase):
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
    grad_output = torch.randn(4, 2)
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

    grad_output_3d = torch.randn(2, 2, 2)
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
                      [-0.3424, -1.4020, 0.3206, -1.0219],
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
      )
      g = torch.ones_like(x)
      layer_norm = torch.nn.LayerNorm([c, h, w], device=device)
      out = layer_norm(x)
      out.backward(g)
      return out.grad, layer_norm.weight.grad, layer_norm.bias.grad

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
      g = torch.ones_like(x)

      layer_norm = torch.nn.LayerNorm(
          [c, h, w], elementwise_affine=False, device=device
      )

      out = layer_norm(x)
      out.backward(g)
      return x.grad

    self.assert_close_tpu_vs_cpu(fn, rtol=1e-5, atol=1e-5)

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

    maxpool_int32 = torch.tensor(
        [
            [
                [-7, -3, 2, -7, -4, -7],
                [-4, 9, 1, 7, 8, -9],
                [9, -1, -8, 1, -6, 0],
                [-5, 7, 1, 9, 4, 9],
                [-2, -4, 2, 1, 2, 0],
                [-6, 0, 1, 0, -3, 6],
            ],
            [
                [8, 5, 2, 1, 5, 3],
                [4, -2, -2, -5, -8, -4],
                [-6, 2, 0, -8, -9, -2],
                [-8, -1, 4, 6, -8, 4],
                [-7, -3, 6, 8, 6, -7],
                [-3, -9, 7, 0, 4, -4],
            ],
        ],
        dtype=torch.int32,
        device=device,
    )

    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nn.functional.max_pool2d(
            maxpool_int32.to(device),
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dilation=(1, 2),
            ceil_mode=True,
            return_indices=True,
        )
    )
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.nn.functional.max_pool2d(
            maxpool_int32.to(device),
            kernel_size=(3, 2),
            stride=(2, 1),
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

  @parameterized.product(
      input_dtype=[torch.float32, torch.bfloat16],
      weight_dtype=[torch.float32, torch.bfloat16],
  )
  def test_convolution_backward_mixed_dtype(self, input_dtype, weight_dtype):
    """Tests convolution backward with mixed dtypes."""
    input_ = torch.randn(1, 1, 3, 3, dtype=input_dtype, requires_grad=True)
    weight = torch.randn(1, 1, 1, 1, dtype=weight_dtype, requires_grad=True)
    bias = torch.randn(1, dtype=weight_dtype, requires_grad=True)

    def test_fn(device):
      out = torch.nn.functional.conv2d(
          input_.to(device), weight.to(device), bias=bias.to(device)
      )
      grad_output = torch.randn_like(out, dtype=input_dtype, device=device)
      out.backward(grad_output)

      return input_.grad, weight.grad, bias.grad

    # TODO(b/489136147): Fix test case & disable allow_failure
    self.assert_close_tpu_vs_cpu(
        test_fn,
        rtol=1e-2,
        atol=1e-2,
        allow_failure=(input_dtype != weight_dtype),
    )

  @parameterized.product(
      groups=[1, 2, 4],
      in_channels_per_group=[1, 2],
      out_channels_per_group=[1, 2],
  )
  def test_convolution_backward_groups(
      self, groups, in_channels_per_group, out_channels_per_group
  ):
    """Tests convolution backward with various group sizes."""
    in_channels = groups * in_channels_per_group
    out_channels = groups * out_channels_per_group
    input_ = torch.randn(
        2,
        in_channels,
        4,
        4,
        dtype=torch.float32,
        requires_grad=True,
    )
    weight = torch.randn(
        out_channels,
        in_channels_per_group,
        3,
        3,
        dtype=torch.float32,
        requires_grad=True,
    )
    bias = torch.randn(out_channels, dtype=torch.float32, requires_grad=True)

    def test_fn(device):
      out = torch.nn.functional.conv2d(
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
    input_ = torch.randn(batch, in_channels, 4, 3, requires_grad=True)
    weight = torch.randn(
        out_channels, in_channels_per_group, 3, 4, requires_grad=True
    )
    bias = torch.randn(1, requires_grad=True)

    def test_fn(device):
      out = torch.nn.functional.conv2d(
          input_.to(device),
          weight.to(device),
          bias=bias.to(device),
          stride=stride,
          padding=padding,
          groups=groups,
      )
      grad_output = torch.randn_like(out, device=device)
      out.backward(grad_output)

      return input_.grad, weight.grad, bias.grad

    # TODO(b/489136147): Fix test case & disable allow_failure
    self.assert_close_tpu_vs_cpu(
        test_fn, rtol=1e-2, atol=1e-2, allow_failure=True
    )

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
      q_t.requires_grad_(True)
      k_t.requires_grad_(True)
      v_t.requires_grad_(True)

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

      y.sum().backward()
      return {
          "y": y,
          "q_grad": q_t.grad,
          "k_grad": k_t.grad,
          "v_grad": v_t.grad,
      }

    cpu_results = compute("cpu")
    tpu_results = compute("tpu")

    # The gradient of the first token is zero with causal attention.
    if config.is_causal:
      with self.subTest("causal_query_first_token_grad_is_zero"):
        self.assertTrue((tpu_results["q_grad"][..., 0, :] == 0.0).all())
      cpu_results["q_grad"] = cpu_results["q_grad"][..., 1:, :]
      tpu_results["q_grad"] = tpu_results["q_grad"][..., 1:, :]

    def check_sim(cpu, tpu):
      # TODO(willfroom): We should also check the min similarity but currently
      # the flash implementation has a few outliers that we need to investigate
      # first.
      mean_sim_bound = 0.9999
      sim = torch.nn.functional.cosine_similarity(cpu, tpu, dim=-1)
      self.assertGreater(sim.mean().item(), mean_sim_bound)

    def assert_close(cpu, tpu):
      rtol, atol = 5e-2, 1e-1
      self.assert_close(
          golden_result=cpu,
          torch_tpu_result=tpu,
          rtol=rtol,
          atol=atol,
      )

    for key in cpu_results:
      with self.subTest(key):
        cpu_tensor = cpu_results[key].cpu()
        tpu_tensor = tpu_results[key].cpu()
        check_sim(cpu_tensor, tpu_tensor)
        assert_close(cpu_tensor, tpu_tensor)

  def test_sdpa_masked_out_row(self):
    """Check that a masked out row does not contain NaNs."""
    q = torch.ones((1, 1, 2, 2), dtype=torch.float32).tpu()
    k = torch.ones((1, 1, 2, 2), dtype=torch.float32).tpu()
    v = torch.ones((1, 1, 2, 2), dtype=torch.float32).tpu()
    mask = torch.tensor([[0, 0], [1, 1]], dtype=torch.bool).tpu()
    with torch.nn.attention.sdpa_kernel(
        torch.nn.attention.SDPBackend.OVERRIDEABLE
    ):
      result = torch.nn.functional.scaled_dot_product_attention(
          q, k, v, attn_mask=mask
      ).cpu()
      self.assertFalse(torch.isnan(result).any())

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


class OpTestingFrameworkTest(TorchTpuVsCpuTestBase):
  """Tests for the op_testing framework itself."""

  def test_accuracy_runs_use_different_seeds(self):
    if self.golden_device_type != "cpu":
      self.skipTest(
          "Only runs in CPU mode where inputs are generated dynamically."
      )

    abs_op = next(op for op in op_db if op.name == "abs")

    # 1. Verify uniqueness within a single run (loop)
    # Seed once at the start of the "loop"
    op_testing._seed_rngs(1234)

    pairs1 = self._get_golden_input_output_pairs(
        op=abs_op,
        dtype=torch.float32,
        variant=op_testing.OpVariant.BASE,
        max_samples=2,
        verbose=False,
        set_seed=False,  # match new loop behavior
    )

    # Run 2 continues from where Run 1 left off
    pairs2 = self._get_golden_input_output_pairs(
        op=abs_op,
        dtype=torch.float32,
        variant=op_testing.OpVariant.BASE,
        max_samples=2,
        verbose=False,
        set_seed=False,  # match new loop behavior
    )

    inputs1 = [p[0].input_value for p in pairs1]
    inputs2 = [p[0].input_value for p in pairs2]

    all_equal = True
    for t1, t2 in zip(inputs1, inputs2):
      if not torch.allclose(t1, t2):
        all_equal = False
        break
    self.assertFalse(all_equal, "Inputs within the loop were identical!")

  def test_accuracy_runs_are_unique_across_runs(self):
    if self.golden_device_type != "cpu":
      self.skipTest(
          "Only runs in CPU mode where inputs are generated dynamically."
      )

    abs_op = next(op for op in op_db if op.name == "abs")

    # Run with base seed 1234
    op_testing._seed_rngs(1234)
    pairs_1234 = self._get_golden_input_output_pairs(
        op=abs_op,
        dtype=torch.float32,
        variant=op_testing.OpVariant.BASE,
        max_samples=2,
        verbose=False,
        set_seed=False,
    )

    # Run with base seed 5678
    op_testing._seed_rngs(5678)
    pairs_5678 = self._get_golden_input_output_pairs(
        op=abs_op,
        dtype=torch.float32,
        variant=op_testing.OpVariant.BASE,
        max_samples=2,
        verbose=False,
        set_seed=False,
    )

    inputs_1234 = [p[0].input_value for p in pairs_1234]
    inputs_5678 = [p[0].input_value for p in pairs_5678]

    all_equal = True
    for t1, t2 in zip(inputs_1234, inputs_5678):
      if not torch.allclose(t1, t2):
        all_equal = False
        break
    self.assertFalse(
        all_equal,
        "Inputs across runs with different base seeds were identical!",
    )

  def test_accuracy_runs_are_deterministic_with_same_seed(self):
    if self.golden_device_type != "cpu":
      self.skipTest(
          "Only runs in CPU mode where inputs are generated dynamically."
      )

    abs_op = next(op for op in op_db if op.name == "abs")

    # Run 1 with base seed 1234
    op_testing._seed_rngs(1234)
    pairs1 = self._get_golden_input_output_pairs(
        op=abs_op,
        dtype=torch.float32,
        variant=op_testing.OpVariant.BASE,
        max_samples=2,
        verbose=False,
        set_seed=False,
    )

    # Run 2 with same base seed 1234
    op_testing._seed_rngs(1234)
    pairs2 = self._get_golden_input_output_pairs(
        op=abs_op,
        dtype=torch.float32,
        variant=op_testing.OpVariant.BASE,
        max_samples=2,
        verbose=False,
        set_seed=False,
    )

    inputs1 = [p[0].input_value for p in pairs1]
    inputs2 = [p[0].input_value for p in pairs2]

    all_equal = True
    for t1, t2 in zip(inputs1, inputs2):
      if not torch.allclose(t1, t2):
        all_equal = False
        break
    self.assertTrue(
        all_equal,
        "Inputs across runs with same base seeds were different!",
    )

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

  def test_float4_e2m1fn_x2_sample_generation(self):
    if self.golden_device_type != "cpu":
      self.skipTest(
          "Only need to verify sample generation in one mode (e.g. CPU)."
      )

    abs_op = next(op for op in op_db if op.name == "abs")
    pairs = self._get_golden_input_output_pairs(
        op=abs_op,
        dtype=torch.float4_e2m1fn_x2,
        variant=op_testing.OpVariant.BASE,
        max_samples=2,
        verbose=False,
    )
    self.assertGreater(len(pairs), 0)
    for golden_input, _ in pairs:
      self.assertEqual(golden_input.input_value.dtype, torch.float4_e2m1fn_x2)

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


if __name__ == "__main__":
  absltest.main()
