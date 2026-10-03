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

"""TorchTPU custom operators exposed publicly under `torch.tpu.ops`."""

from collections.abc import Sequence

import torch
from torch_tpu._internal.utils import annotations


@annotations.experimental(
    "torch.tpu.ops.optimization_barrier is experimental and subject to change."
)
def optimization_barrier(inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
  """Inserts an XLA optimization barrier around `inputs`.

  The barrier lowers to `stablehlo.optimization_barrier`
  (https://openxla.org/stablehlo/spec#optimization_barrier). It is an identity
  operation on its operands, but the XLA compiler treats it as opaque: no
  computation is moved across it and computations on either side of it are not
  fused together or deduplicated (e.g. by common subexpression elimination).

  This is useful to control scheduling and memory usage, for example to force
  a recomputation in activation checkpointing to happen where it is placed
  rather than being merged with the original computation, or to prevent the
  compiler from fusing a numerically sensitive region with its neighbors.

  Example:
    a, b = torch.tpu.ops.optimization_barrier([a, b])

  Args:
    inputs: The TPU tensors to pass through the barrier. Every computation
      producing these tensors happens before the barrier, and every computation
      consuming the returned tensors happens after it.

  Returns:
    A list of tensors with the same values, shapes, dtypes and strides as
    `inputs`, in the same order. Use these tensors, rather than `inputs`, in
    subsequent computations so that they are ordered after the barrier.
  """
  return torch.ops.tpu._optimization_barrier(inputs)  # pylint: disable=protected-access


# The max pooling ops below compute only the pooled values (no indices), which
# avoids materializing an index tensor. Their arguments have the same meaning as
# in `torch.nn.functional.max_pool{1,2,3}d`, except that every sequence must
# have one element per spatial dimension and there are no defaults.


@annotations.stable()
def max_pool1d(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    ceil_mode: bool,
) -> torch.Tensor:
  """Applies 1D max pooling to a TPU tensor without computing indices.

  This op isn't differentiable. Use `max_pool1d_backward` to compute its
  gradient.

  Args:
    input: The input tensor of shape `(N, C, L)` or `(C, L)`.
    kernel_size: The size of the pooling window, `[kL]`.
    stride: The stride of the pooling window, `[sL]`.
    padding: The implicit negative-infinity padding on both sides, `[pL]`.
    dilation: The spacing between elements in the pooling window, `[dL]`.
    ceil_mode: Whether to use ceil instead of floor to compute the output size.

  Returns:
    The pooled tensor.
  """
  return torch.ops.tpu._max_pool1d(  # pylint: disable=protected-access
      input, kernel_size, stride, padding, dilation, ceil_mode
  )


@annotations.stable()
def max_pool1d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    ceil_mode: bool,
) -> torch.Tensor:
  """Computes the gradient of `max_pool1d` with respect to its input.

  The positions of the maxima are recomputed from `input`, so no index tensor
  is needed.

  Args:
    grad_output: The gradient with respect to the output of `max_pool1d`.
    input: The input that was passed to `max_pool1d`.
    kernel_size: The `kernel_size` that was passed to `max_pool1d`.
    stride: The `stride` that was passed to `max_pool1d`.
    padding: The `padding` that was passed to `max_pool1d`.
    dilation: The `dilation` that was passed to `max_pool1d`.
    ceil_mode: The `ceil_mode` that was passed to `max_pool1d`.

  Returns:
    The gradient with respect to `input`, which has the same shape as `input`.
  """
  return torch.ops.tpu._max_pool1d_backward(  # pylint: disable=protected-access
      grad_output, input, kernel_size, stride, padding, dilation, ceil_mode
  )


@annotations.stable()
def max_pool2d(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    ceil_mode: bool,
) -> torch.Tensor:
  """Applies 2D max pooling to a TPU tensor without computing indices.

  This op isn't differentiable. Use `max_pool2d_backward` to compute its
  gradient.

  Args:
    input: The input tensor of shape `(N, C, H, W)` or `(C, H, W)`.
    kernel_size: The size of the pooling window, `[kH, kW]`.
    stride: The stride of the pooling window, `[sH, sW]`.
    padding: The implicit negative-infinity padding on both sides, `[pH, pW]`.
    dilation: The spacing between elements in the pooling window, `[dH, dW]`.
    ceil_mode: Whether to use ceil instead of floor to compute the output size.

  Returns:
    The pooled tensor.
  """
  return torch.ops.tpu._max_pool2d(  # pylint: disable=protected-access
      input, kernel_size, stride, padding, dilation, ceil_mode
  )


@annotations.stable()
def max_pool2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    ceil_mode: bool,
) -> torch.Tensor:
  """Computes the gradient of `max_pool2d` with respect to its input.

  The positions of the maxima are recomputed from `input`, so no index tensor
  is needed.

  Args:
    grad_output: The gradient with respect to the output of `max_pool2d`.
    input: The input that was passed to `max_pool2d`.
    kernel_size: The `kernel_size` that was passed to `max_pool2d`.
    stride: The `stride` that was passed to `max_pool2d`.
    padding: The `padding` that was passed to `max_pool2d`.
    dilation: The `dilation` that was passed to `max_pool2d`.
    ceil_mode: The `ceil_mode` that was passed to `max_pool2d`.

  Returns:
    The gradient with respect to `input`, which has the same shape as `input`.
  """
  return torch.ops.tpu._max_pool2d_backward(  # pylint: disable=protected-access
      grad_output, input, kernel_size, stride, padding, dilation, ceil_mode
  )


@annotations.stable()
def max_pool3d(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    ceil_mode: bool,
) -> torch.Tensor:
  """Applies 3D max pooling to a TPU tensor without computing indices.

  This op isn't differentiable. Use `max_pool3d_backward` to compute its
  gradient.

  Args:
    input: The input tensor of shape `(N, C, D, H, W)` or `(C, D, H, W)`.
    kernel_size: The size of the pooling window, `[kD, kH, kW]`.
    stride: The stride of the pooling window, `[sD, sH, sW]`.
    padding: The implicit negative-infinity padding on both sides, `[pD, pH,
      pW]`.
    dilation: The spacing between elements in the pooling window, `[dD, dH,
      dW]`.
    ceil_mode: Whether to use ceil instead of floor to compute the output size.

  Returns:
    The pooled tensor.
  """
  return torch.ops.tpu._max_pool3d(  # pylint: disable=protected-access
      input, kernel_size, stride, padding, dilation, ceil_mode
  )


@annotations.stable()
def max_pool3d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    ceil_mode: bool,
) -> torch.Tensor:
  """Computes the gradient of `max_pool3d` with respect to its input.

  The positions of the maxima are recomputed from `input`, so no index tensor
  is needed.

  Args:
    grad_output: The gradient with respect to the output of `max_pool3d`.
    input: The input that was passed to `max_pool3d`.
    kernel_size: The `kernel_size` that was passed to `max_pool3d`.
    stride: The `stride` that was passed to `max_pool3d`.
    padding: The `padding` that was passed to `max_pool3d`.
    dilation: The `dilation` that was passed to `max_pool3d`.
    ceil_mode: The `ceil_mode` that was passed to `max_pool3d`.

  Returns:
    The gradient with respect to `input`, which has the same shape as `input`.
  """
  return torch.ops.tpu._max_pool3d_backward(  # pylint: disable=protected-access
      grad_output, input, kernel_size, stride, padding, dilation, ceil_mode
  )


# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "max_pool1d",
    "max_pool1d_backward",
    "max_pool2d",
    "max_pool2d_backward",
    "max_pool3d",
    "max_pool3d_backward",
    "optimization_barrier",
    # go/keep-sorted end
]
