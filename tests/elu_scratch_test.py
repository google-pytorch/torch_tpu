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

"""Functional and dynamic verification scratch tests for TorchTPU ELU kernels.

Compares standard CPU implementations vs TPU lowering patterns for various
floating and integer configurations.
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal.utils import utils


def _cpu_elu_custom(
    input_tensor: torch.Tensor,
    alpha: float,
    scale: float,
    input_scale: float,
) -> torch.Tensor:
  """Computes reference ELU forward pass on CPU.

  Args:
    input_tensor: The input tensor.
    alpha: Alpha coefficient for the negative path.
    scale: Global scale multiplier.
    input_scale: Input scale coefficient.

  Returns:
    The forward activation output tensor.
  """
  cond = input_tensor >= 0
  pos = input_tensor  # Correct positive path math (no input_scale!)
  neg = alpha * (torch.exp(input_tensor * input_scale) - 1)
  inner = torch.where(cond, pos, neg)
  return scale * inner


def _cpu_elu_backward_custom(
    grad_output: torch.Tensor,
    alpha: float,
    scale: float,
    input_scale: float,
    is_result: bool,
    self_or_result: torch.Tensor,
) -> torch.Tensor:
  """Computes reference ELU backward gradient input pass on CPU.

  Args:
    grad_output: The backpropagated gradient tensor.
    alpha: Alpha coefficient.
    scale: Global scale multiplier.
    input_scale: Input scale coefficient.
    is_result: True if self_or_result represents the forward output.
    self_or_result: Forward input (kSelf) or output result (kResult).

  Returns:
    The computed gradient input tensor.
  """
  cond = self_or_result > 0
  pos = grad_output * scale
  if is_result:
    neg = grad_output * input_scale * (self_or_result + alpha * scale)
  else:
    neg = (
        grad_output
        * input_scale
        * alpha
        * scale
        * torch.exp(self_or_result * input_scale)
    )
  return torch.where(cond, pos, neg)


class EluScratchTest(parameterized.TestCase):
  """Rigorously verifies ELU forward and backward TPU lowering correctness."""

  def setUp(self) -> None:
    super().setUp()
    self.device = "tpu"

  @parameterized.parameters(
      (torch.float32, 1e-5, 1e-5),
      (torch.bfloat16, 1e-3, 1e-3),
      (torch.float16, 1e-3, 1e-3),
  )
  def test_elu_forward_float_scalars(
      self, dtype: torch.dtype, rtol: float, atol: float
  ) -> None:
    """Verifies forward ELU on floating-point types with float parameters."""
    # Arrange
    input_cpu = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
    input_tpu = input_cpu.to(self.device)
    alpha, scale, input_scale = 1.5, 2.0, 0.5
    expected = _cpu_elu_custom(input_cpu.float(), alpha, scale, input_scale).to(
        dtype
    )
    out_tpu = torch.empty_like(input_tpu)

    # Act
    torch.ops.aten.elu.out(input_tpu, alpha, scale, input_scale, out=out_tpu)
    out_cpu = out_tpu.to("cpu")

    # Assert
    utils.assert_close(out_cpu, expected, rtol=rtol, atol=atol)

  def test_elu_forward_integer_scalars(self) -> None:
    """Verifies dynamic promotion of integer inputs/parameters to float."""
    # Arrange
    input_cpu = torch.tensor(
        [-2, -1, 0, 1, 2], dtype=torch.int32
    ).float()  # Integer operators promote to float
    input_tpu = input_cpu.to(self.device)
    alpha, scale, input_scale = 2, 3, 1
    expected = _cpu_elu_custom(
        input_cpu, float(alpha), float(scale), float(input_scale)
    )
    out_tpu = torch.empty_like(input_tpu)

    # Act
    torch.ops.aten.elu.out(input_tpu, alpha, scale, input_scale, out=out_tpu)
    out_cpu = out_tpu.to("cpu")

    # Assert
    utils.assert_close(out_cpu, expected, rtol=1e-5, atol=1e-5)

  def test_elu_backward_is_result_true(self) -> None:
    """Verifies gradient pass using forward output results cache (kResult)."""
    # Arrange
    grad_output_cpu = torch.tensor(
        [0.5, -0.5, 1.0, -1.0, 0.0], dtype=torch.float32
    )
    self_or_result_cpu = torch.tensor(
        [-1.5, -0.5, 0.0, 0.5, 1.5], dtype=torch.float32
    )
    grad_output_tpu = grad_output_cpu.to(self.device)
    self_or_result_tpu = self_or_result_cpu.to(self.device)
    alpha, scale, input_scale = 1.5, 2.0, 0.5
    is_result = True
    expected = _cpu_elu_backward_custom(
        grad_output_cpu,
        alpha,
        scale,
        input_scale,
        is_result,
        self_or_result_cpu,
    )

    # Act - Functional call
    out_functional_tpu = torch.ops.aten.elu_backward(
        grad_output_tpu,
        alpha,
        scale,
        input_scale,
        is_result,
        self_or_result_tpu,
    )
    out_functional_cpu = out_functional_tpu.to("cpu")

    # Assert Functional
    utils.assert_close(out_functional_cpu, expected, rtol=1e-5, atol=1e-5)

    # Act - Out-variant (grad_input) signature
    out_grad_input_tpu = torch.empty_like(grad_output_tpu)
    torch.ops.aten.elu_backward.grad_input(
        grad_output_tpu,
        alpha,
        scale,
        input_scale,
        is_result,
        self_or_result_tpu,
        grad_input=out_grad_input_tpu,
    )
    out_grad_input_cpu = out_grad_input_tpu.to("cpu")

    # Assert Out-variant
    utils.assert_close(out_grad_input_cpu, expected, rtol=1e-5, atol=1e-5)

  def test_elu_backward_is_result_false(self) -> None:
    """Verifies gradient pass using raw backward self inputs cache (kSelf)."""
    # Arrange
    grad_output_cpu = torch.tensor(
        [0.5, -0.5, 1.0, -1.0, 0.0], dtype=torch.float32
    )
    self_or_result_cpu = torch.tensor(
        [-1.5, -0.5, 0.0, 0.5, 1.5], dtype=torch.float32
    )
    grad_output_tpu = grad_output_cpu.to(self.device)
    self_or_result_tpu = self_or_result_cpu.to(self.device)
    alpha, scale, input_scale = 1.5, 2.0, 0.5
    is_result = False
    expected = _cpu_elu_backward_custom(
        grad_output_cpu,
        alpha,
        scale,
        input_scale,
        is_result,
        self_or_result_cpu,
    )

    # Act - Functional call
    out_functional_tpu = torch.ops.aten.elu_backward(
        grad_output_tpu,
        alpha,
        scale,
        input_scale,
        is_result,
        self_or_result_tpu,
    )
    out_functional_cpu = out_functional_tpu.to("cpu")

    # Assert Functional
    utils.assert_close(out_functional_cpu, expected, rtol=1e-5, atol=1e-5)

    # Act - Out-variant (grad_input) signature
    out_grad_input_tpu = torch.empty_like(grad_output_tpu)
    torch.ops.aten.elu_backward.grad_input(
        grad_output_tpu,
        alpha,
        scale,
        input_scale,
        is_result,
        self_or_result_tpu,
        grad_input=out_grad_input_tpu,
    )
    out_grad_input_cpu = out_grad_input_tpu.to("cpu")

    # Assert Out-variant
    utils.assert_close(out_grad_input_cpu, expected, rtol=1e-5, atol=1e-5)

  def test_elu_backward_integer_scalars(self) -> None:
    """Verifies backward integer scalars are promoted correctly to float."""
    # Arrange
    grad_output_cpu = torch.tensor(
        [0.5, -0.5, 1.0, -1.0, 0.0], dtype=torch.float32
    )
    self_or_result_cpu = torch.tensor(
        [-1.5, -0.5, 0.0, 0.5, 1.5], dtype=torch.float32
    )
    grad_output_tpu = grad_output_cpu.to(self.device)
    self_or_result_tpu = self_or_result_cpu.to(self.device)
    alpha, scale, input_scale = 2, 3, 1
    is_result = False
    expected = _cpu_elu_backward_custom(
        grad_output_cpu,
        float(alpha),
        float(scale),
        float(input_scale),
        is_result,
        self_or_result_cpu,
    )

    # Act - Functional call
    out_functional_tpu = torch.ops.aten.elu_backward(
        grad_output_tpu,
        alpha,
        scale,
        input_scale,
        is_result,
        self_or_result_tpu,
    )
    out_functional_cpu = out_functional_tpu.to("cpu")

    # Assert Functional
    utils.assert_close(out_functional_cpu, expected, rtol=1e-5, atol=1e-5)

    # Act - Out-variant (grad_input) signature
    out_grad_input_tpu = torch.empty_like(grad_output_tpu)
    torch.ops.aten.elu_backward.grad_input(
        grad_output_tpu,
        alpha,
        scale,
        input_scale,
        is_result,
        self_or_result_tpu,
        grad_input=out_grad_input_tpu,
    )
    out_grad_input_cpu = out_grad_input_tpu.to("cpu")

    # Assert Out-variant
    utils.assert_close(out_grad_input_cpu, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
