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
import functools
import itertools
import math
import threading
import time
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from scipy import stats
import torch
from torch_tpu._internal import sync
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.utils import utils
from tests import op_testing
from tests import ops_test_data

# In this file, we use the following naming convention for variables:
# - golden_*: a value for the device used for computing the golden results
#   (either CPU or GPU)
# - tpu_*: a value for the TPU device


OpInput = op_testing.OpInput
TorchTpuVsCpuTestBase = op_testing.TorchTpuVsCpuTestBase
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

    self.assert_close_tpu_vs_cpu(
        compute, check_value=CheckValueMode.LOOSE, rtol=5e-1, atol=5e-1
    )

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

    self.assert_close_tpu_vs_cpu(
        compute, check_value=CheckValueMode.LOOSE, rtol=5e-1, atol=5e-1
    )

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
    self.assertEqual(self_tensor, self_tensor_tpu.cpu())

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
    self.assertEqual(out, out_tpu.cpu())

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
    self.assertEqual(golden_result, tpu_result.cpu())

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
        return torch.relu(t + y1 * 1.0)
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

  def test_copy(self):
    """Tests tensor.copy_() and torch.ops.aten.copy()."""
    # Test copy within the same device
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.empty(3, device=device).copy_(
            torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
        )
    )

    # Test copy with broadcasting
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.empty(3, device=device).copy_(
            torch.tensor(3.14, dtype=torch.float32, device=device)
        )
    )

    tpu_device = torch.device("tpu")

    # Test CPU -> TPU copy
    src_cpu = torch.tensor([1, 2, 3], dtype=torch.int32)
    dst_tpu = torch.empty(3, dtype=torch.int32, device=tpu_device)
    dst_tpu.copy_(src_cpu)
    self.assertEqual(dst_tpu.cpu(), src_cpu)

    # Test TPU -> CPU copy
    src_tpu = torch.tensor([4, 5, 6], dtype=torch.int32, device=tpu_device)
    dst_cpu = torch.empty(3, dtype=torch.int32)
    dst_cpu.copy_(src_tpu)
    self.assertEqual(dst_cpu, src_tpu.cpu())

    # Test casting
    src_cpu = torch.tensor([1.5, 2.5], dtype=torch.float32)
    dst_tpu = torch.empty(2, dtype=torch.int32, device=tpu_device)
    dst_tpu.copy_(src_cpu)
    self.assertEqual(dst_tpu.cpu(), src_cpu.int())

    # Test torch.ops.aten.copy
    src_cpu = torch.tensor([1, 2, 3], dtype=torch.int32)
    dst_tpu = torch.empty(3, dtype=torch.int32, device=tpu_device)
    dst_tpu = torch.ops.aten.copy(dst_tpu, src_cpu)
    self.assertEqual(dst_tpu.cpu(), src_cpu)

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
    N, C = 1, 1
    H_in, W_in = 4, 4
    # We use a large output size that triggers the precision issue.
    # We only need one dimension to trigger it, but we'll scale both.
    H_out, W_out = 22, 22

    # Create input with known values (e.g., indices) to easily check correctness
    x = torch.arange(H_in * W_in, dtype=torch.float32).reshape(N, C, H_in, W_in)
    x_tpu = x.to(tpu_device)

    # We expect the last pixel to be exactly the last pixel of the input
    # because align_corners=True maps corner to corner.
    out_tpu = torch.nn.functional.interpolate(
        x_tpu, size=(H_out, W_out), mode="bilinear", align_corners=True
    )
    out_cpu = torch.nn.functional.interpolate(
        x, size=(H_out, W_out), mode="bilinear", align_corners=True
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
    #   res = val[-1]*(1-0.75) + val[0]*0.75 = 20*0.25 + 10*0.75 = 5 + 7.5 = 12.5.
    # Expected (clamped): val[0] = 10.

    tpu_device = torch.device("tpu")
    N, C = 1, 1
    H_in, W_in = 1, 2
    H_out, W_out = 1, 4

    x = torch.tensor([10.0, 20.0], dtype=torch.float32).reshape(
        N, C, H_in, W_in
    )
    x_tpu = x.to(tpu_device)

    out_tpu = torch.nn.functional.interpolate(
        x_tpu, size=(H_out, W_out), mode="bilinear", align_corners=False
    )
    out_cpu = torch.nn.functional.interpolate(
        x, size=(H_out, W_out), mode="bilinear", align_corners=False
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
    except:  # pylint: disable=bare-except
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
        check_value=CheckValueMode.LOOSE,
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
          check_value=CheckValueMode.LOOSE,
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

  def test_tensor_to_tpu_with_default_device(self):
    """Tests that a tensor can be moved to TPU with a default device set."""
    device = torch.device("tpu")
    torch.set_default_device(device)
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
      self.assert_close_tpu_vs_cpu(
          compute, atol=2e-2, rtol=1e-2, check_value=CheckValueMode.LOOSE
      )
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
      self.assert_close_tpu_vs_cpu(
          compute, atol=2e-2, rtol=1e-2, check_value=CheckValueMode.LOOSE
      )
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
      self.assert_close_tpu_vs_cpu(
          compute, atol=2e-2, rtol=1e-2, check_value=CheckValueMode.LOOSE
      )
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
      self.assert_close_tpu_vs_cpu(
          compute, atol=2e-2, rtol=1e-2, check_value=CheckValueMode.LOOSE
      )
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
      self.assert_close_tpu_vs_cpu(
          compute, atol=2e-2, rtol=1e-2, check_value=CheckValueMode.LOOSE
      )
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
      self.assert_close_tpu_vs_cpu(
          compute, atol=2e-2, rtol=1e-2, check_value=CheckValueMode.LOOSE
      )
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
        check_value=CheckValueMode.LOOSE,
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
            check_value=CheckValueMode.LOOSE,
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
    self.assertFalse(sync.is_materialized(tensor_tpu))

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
        check_value=CheckValueMode.LOOSE,
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
        check_value=CheckValueMode.LOOSE,
        rtol=5.7e-05,
        atol=6.3e-05,
    )

    # Test broadcasting
    def test_broadcasting_fn(device):
      x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
      y = torch.tensor([1.0, 2.0], device=device)
      return torch.xlogy(x, y)

    self.assert_close_tpu_vs_cpu(test_broadcasting_fn)


class OpsCustomOpUnitTest(TorchTpuVsCpuTestBase, parameterized.TestCase):
  """Tests for custom ops."""

  def test_ragged_dot_on_tpu(self):
    """Tests the torch_tpu.ragged_dot custom op on TPU."""
    device = torch.device("tpu")
    m, k, n, g = 5, 4, 3, 2
    lhs = torch.arange(m * k, dtype=torch.float32, device=device).reshape(m, k)
    rhs = torch.arange(g * k * n, dtype=torch.float32, device=device).reshape(
        g, k, n
    )
    group_sizes = torch.tensor([1, 4], dtype=torch.int32, device=device)
    out = torch.ops.torch_tpu.ragged_dot(lhs, rhs, group_sizes)
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
    """Tests the torch_tpu.ragged_dot custom op with an out parameter on TPU."""
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
    res = torch.ops.torch_tpu.ragged_dot(lhs, rhs, group_sizes, out=out)
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

  @absltest.skip("b/498564738")
  def test_set_dimension_logical_size_on_tpu(self):
    """Tests the torch_tpu.set_dimension_logical_size custom op on TPU."""
    device = torch.device("tpu")
    x = torch.arange(100, device=device, dtype=torch.int32).reshape(10, 10)
    size = torch.tensor(5, device=device, dtype=torch.int32)
    out = torch.ops.torch_tpu.set_dimension_logical_size(x, 0, size)
    self.assert_close(golden_result=x[:5].cpu(), torch_tpu_result=out[:5].cpu())

  def test_set_dimension_logical_size_with_mlir_on_tpu(self):
    """Tests the torch_tpu.set_dimension_logical_size custom op on TPU using MLIR."""
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
        config.q_seq_len,  # Using q_seq_len as per ml_layers_test logic for K
        config.qk_head_dim,
        dtype=config.dtype,
    )
    v = torch.randn(
        config.batch_size,
        config.kv_num_heads,
        config.q_seq_len,
        config.v_head_dim,
        dtype=config.dtype,
    )

    def compute(device: torch.device):
      q_t = q.clone().detach().to(device)
      k_t = k.clone().detach().to(device)
      v_t = v.clone().detach().to(device)
      q_t.requires_grad_(True)
      k_t.requires_grad_(True)
      v_t.requires_grad_(True)

      with torch.nn.attention.sdpa_kernel(
          torch.nn.attention.SDPBackend.MATH
          if config.use_math_backend or device == "cpu"
          else torch.nn.attention.SDPBackend.FLASH_ATTENTION
      ):
        y = torch.nn.functional.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            is_causal=config.is_causal,
            enable_gqa=config.enable_gqa,
            scale=config.scale,
        )

      y.sum().backward()
      return y, q_t.grad, k_t.grad, v_t.grad

    # Use tolerances from ops_test.py for SDPA
    rtol, atol = None, None
    if config.dtype == torch.bfloat16:
      rtol, atol = 5e-2, 1e-1
    elif config.dtype == torch.float32:
      rtol, atol = 1e-2, 1e-2

    self.assert_close_tpu_vs_cpu(
        compute, rtol=rtol, atol=atol, check_value=utils.CheckValueMode.LOOSE
    )

  def test_stateless_dropout(self):
    torch.manual_seed(0)
    input_tensor = torch.randn(
        10,
        10,
        dtype=torch.float32,
        device=torch.device("tpu"),
    )

    rng_state = torch.get_device_module(torch.device("tpu")).get_rng_state()
    # rng_state by spec is on CPU, move it to TPU.
    rng_state = rng_state.to(torch.device("tpu"))

    expected_out, expected_mask = torch.ops.aten.native_dropout(
        input_tensor, 0.5, True
    )
    _, out, mask = torch.ops.torch_tpu.stateless_dropout(
        rng_state, input_tensor, 0.5, True
    )

    self.assertEqual(expected_out.cpu(), out.cpu())
    self.assertEqual(expected_mask.cpu(), mask.cpu())

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
    dtypes = [torch.float16, torch.float32, torch.bfloat16, torch.float64]

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


if __name__ == "__main__":
  absltest.main()
