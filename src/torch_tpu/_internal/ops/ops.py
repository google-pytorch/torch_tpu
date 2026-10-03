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
from typing import Any

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


@annotations.experimental(
    "torch.tpu.ops.ragged_dot is experimental and subject to change."
)
def ragged_dot(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
  """Computes a ragged dot product, e.g. for Mixture-of-Experts models.

  The rows of `lhs` are split into `g` consecutive groups, where group `i` has
  `group_sizes[i]` rows, and each group is multiplied by its own matrix in
  `rhs`:

    result[start_i:end_i, :] = lhs[start_i:end_i, :] @ rhs[i, :, :]

  where `start_i = sum(group_sizes[:i])` and `end_i = start_i + group_sizes[i]`.

  This op is differentiable with respect to `lhs` and `rhs` when `out` is not
  given.

  Args:
    lhs: A 2D tensor of shape `(m, k)`.
    rhs: A 3D tensor of shape `(g, k, n)`.
    group_sizes: A 1D tensor of shape `(g,)` whose elements sum to `m`.
    out: If given, a 2D tensor of shape `(m, n)` to write the result to.

  Returns:
    A 2D tensor of shape `(m, n)`. If `out` is given, it's `out`.
  """
  if out is None:
    return torch.ops.tpu._ragged_dot(lhs, rhs, group_sizes)  # pylint: disable=protected-access
  return torch.ops.tpu._ragged_dot.out(lhs, rhs, group_sizes, out=out)  # pylint: disable=protected-access


@annotations.experimental(
    "torch.tpu.ops.ragged_dot_weight_grad is experimental and subject to"
    " change."
)
def ragged_dot_weight_grad(
    lhs: torch.Tensor,
    grad_output: torch.Tensor,
    group_sizes: torch.Tensor,
) -> torch.Tensor:
  """Computes the gradient of `ragged_dot` with respect to `rhs`.

  For each group `i`:

    result[i, :, :] = lhs[start_i:end_i, :].T @ grad_output[start_i:end_i, :]

  where `start_i` and `end_i` are defined as in `ragged_dot`.

  Args:
    lhs: The `lhs` that was passed to `ragged_dot`, of shape `(m, k)`.
    grad_output: The gradient with respect to the output of `ragged_dot`, of
      shape `(m, n)`.
    group_sizes: The `group_sizes` that was passed to `ragged_dot`, of shape
      `(g,)`.

  Returns:
    A 3D tensor of shape `(g, k, n)`.
  """
  return torch.ops.tpu._ragged_dot_weight_grad(lhs, grad_output, group_sizes)  # pylint: disable=protected-access


@annotations.experimental(
    "torch.tpu.ops.ragged_all_to_all is experimental and subject to change."
)
def ragged_all_to_all(
    operand: torch.Tensor,
    output: torch.Tensor,
    input_offsets: torch.Tensor,
    send_sizes: torch.Tensor,
    output_offsets: torch.Tensor,
    recv_sizes: torch.Tensor,
    process_group_name: str,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
  """Exchanges ragged slices of a tensor among the ranks of a process group.

  Each rank sends the slice of `operand` that starts at `input_offsets[j]` and
  has `send_sizes[j]` rows to rank `j`, which writes it to its output starting
  at row `output_offsets[j]` (as seen by the sender). `recv_sizes[j]` is the
  number of rows that this rank receives from rank `j`.

  This op is differentiable with respect to `operand` when `out` is not given.

  Args:
    operand: The tensor to send slices of.
    output: A tensor that provides the initial contents, shape and dtype of the
      result. Rows that aren't received keep their values from `output`.
    input_offsets: A 1D tensor of the starting row in `operand` of the slice to
      send to each rank.
    send_sizes: A 1D tensor of the number of rows to send to each rank.
    output_offsets: A 1D tensor of the starting row in each receiving rank's
      output at which to write the slice sent to it.
    recv_sizes: A 1D tensor of the number of rows to receive from each rank.
    process_group_name: The name of the process group, i.e.
      `torch.distributed.ProcessGroup.group_name`.
    out: If given, a tensor to write the result to.

  Returns:
    A tensor with the shape and dtype of `output`, containing the received
    slices. If `out` is given, it's `out`.
  """
  if out is None:
    return torch.ops.tpu._ragged_all_to_all(  # pylint: disable=protected-access
        operand,
        output,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        process_group_name,
    )
  return torch.ops.tpu._ragged_all_to_all.out(  # pylint: disable=protected-access
      operand,
      output,
      input_offsets,
      send_sizes,
      output_offsets,
      recv_sizes,
      process_group_name,
      out=out,
  )


@annotations.experimental(
    "torch.tpu.ops.send is experimental and subject to change."
)
def send(tensors: Sequence[torch.Tensor], dst: int, tag: int) -> Any:
  """Sends tensors to another rank of the default TPU process group.

  Args:
    tensors: The tensors to send.
    dst: The destination rank.
    tag: The tag to match this send with a `recv` on `dst`.

  Returns:
    A `torch.distributed.Work` object. Call its `wait()` method to wait for
    the send to complete.
  """
  return torch.ops.tpu._send(tensors, dst, tag)  # pylint: disable=protected-access


@annotations.experimental(
    "torch.tpu.ops.recv is experimental and subject to change."
)
def recv(tensors: Sequence[torch.Tensor], src: int, tag: int) -> Any:
  """Receives tensors from another rank of the default TPU process group.

  Args:
    tensors: The tensors to receive into. They must match the shapes and dtypes
      of the tensors sent by `src`.
    src: The source rank.
    tag: The tag to match this receive with a `send` on `src`.

  Returns:
    A `torch.distributed.Work` object. Call its `wait()` method to wait for
    the receive to complete before using `tensors`.
  """
  return torch.ops.tpu._recv(tensors, src, tag)  # pylint: disable=protected-access


@annotations.experimental(
    "torch.tpu.ops.sparse_dense_matmul is experimental and subject to change."
)
def sparse_dense_matmul(
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    device_batch_size: int,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
) -> torch.Tensor:
  """Looks up and combines embeddings on TPU SparseCore.

  The sparse input is given in the preprocessed CSR format produced by the
  SparseCore input preprocessing: `row_pointers`, `embedding_ids`, `sample_ids`
  and `gains`.

  Args:
    row_pointers: The CSR row pointers.
    embedding_ids: The IDs of the embedding rows to look up.
    sample_ids: The sample that each looked-up embedding row belongs to.
    gains: The weight of each looked-up embedding row.
    embedding_table: The embedding table.
    device_batch_size: The batch size on this device.
    max_ids_per_partition: The maximum number of IDs per partition.
    max_unique_ids_per_partition: The maximum number of unique IDs per
      partition.

  Returns:
    The activations, with one combined embedding per sample.
  """
  return torch.ops.tpu._sparse_dense_matmul(  # pylint: disable=protected-access
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      embedding_table,
      device_batch_size,
      max_ids_per_partition,
      max_unique_ids_per_partition,
  )


@annotations.experimental(
    "torch.tpu.ops.sparse_dense_matmul_grad_with_sgd is experimental and"
    " subject to change."
)
def sparse_dense_matmul_grad_with_sgd(
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    activations_grad: torch.Tensor,
    learning_rate: torch.Tensor,
    device_batch_size: int,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str,
) -> torch.Tensor:
  """Backpropagates through `sparse_dense_matmul` and applies an SGD update.

  Args:
    row_pointers: The `row_pointers` passed to `sparse_dense_matmul`.
    embedding_ids: The `embedding_ids` passed to `sparse_dense_matmul`.
    sample_ids: The `sample_ids` passed to `sparse_dense_matmul`.
    gains: The `gains` passed to `sparse_dense_matmul`.
    embedding_table: The embedding table to update.
    activations_grad: The gradient with respect to the activations.
    learning_rate: The learning rate.
    device_batch_size: The batch size on this device.
    max_ids_per_partition: The maximum number of IDs per partition.
    max_unique_ids_per_partition: The maximum number of unique IDs per
      partition.
    computation_name: A name for the computation.

  Returns:
    The updated embedding table.
  """
  return torch.ops.tpu._sparse_dense_matmul_grad_with_sgd(  # pylint: disable=protected-access
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      embedding_table,
      activations_grad,
      learning_rate,
      device_batch_size,
      max_ids_per_partition,
      max_unique_ids_per_partition,
      computation_name,
  )


@annotations.experimental(
    "torch.tpu.ops.sparse_dense_matmul_grad_with_adagrad is experimental and"
    " subject to change."
)
def sparse_dense_matmul_grad_with_adagrad(
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    accumulator: torch.Tensor,
    activations_grad: torch.Tensor,
    learning_rate: torch.Tensor,
    epsilon: float,
    device_batch_size: int,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Backpropagates through `sparse_dense_matmul` and applies an Adagrad update.

  Args:
    row_pointers: The `row_pointers` passed to `sparse_dense_matmul`.
    embedding_ids: The `embedding_ids` passed to `sparse_dense_matmul`.
    sample_ids: The `sample_ids` passed to `sparse_dense_matmul`.
    gains: The `gains` passed to `sparse_dense_matmul`.
    embedding_table: The embedding table to update.
    accumulator: The Adagrad accumulator, either row-wise (1D) or of the same
      shape as `embedding_table` (2D).
    activations_grad: The gradient with respect to the activations.
    learning_rate: The learning rate.
    epsilon: The Adagrad epsilon.
    device_batch_size: The batch size on this device.
    max_ids_per_partition: The maximum number of IDs per partition.
    max_unique_ids_per_partition: The maximum number of unique IDs per
      partition.
    computation_name: A name for the computation.

  Returns:
    A tuple of the updated embedding table and the updated accumulator.
  """
  return torch.ops.tpu._sparse_dense_matmul_grad_with_adagrad(  # pylint: disable=protected-access
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      embedding_table,
      accumulator,
      activations_grad,
      learning_rate,
      epsilon,
      device_batch_size,
      max_ids_per_partition,
      max_unique_ids_per_partition,
      computation_name,
  )


@annotations.experimental(
    "torch.tpu.ops.sparse_dense_matmul_grad_with_adam is experimental and"
    " subject to change."
)
def sparse_dense_matmul_grad_with_adam(
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    momentum: torch.Tensor,
    velocity: torch.Tensor,
    activations_grad: torch.Tensor,
    alpha_t: torch.Tensor,
    beta_1: float,
    beta_2: float,
    epsilon: float,
    device_batch_size: int,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Backpropagates through `sparse_dense_matmul` and applies an Adam update.

  Args:
    row_pointers: The `row_pointers` passed to `sparse_dense_matmul`.
    embedding_ids: The `embedding_ids` passed to `sparse_dense_matmul`.
    sample_ids: The `sample_ids` passed to `sparse_dense_matmul`.
    gains: The `gains` passed to `sparse_dense_matmul`.
    embedding_table: The embedding table to update.
    momentum: The Adam first moment.
    velocity: The Adam second moment.
    activations_grad: The gradient with respect to the activations.
    alpha_t: The bias-corrected learning rate for the current step.
    beta_1: The Adam beta_1.
    beta_2: The Adam beta_2.
    epsilon: The Adam epsilon.
    device_batch_size: The batch size on this device.
    max_ids_per_partition: The maximum number of IDs per partition.
    max_unique_ids_per_partition: The maximum number of unique IDs per
      partition.
    computation_name: A name for the computation.

  Returns:
    A tuple of the updated embedding table, momentum and velocity.
  """
  return torch.ops.tpu._sparse_dense_matmul_grad_with_adam(  # pylint: disable=protected-access
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      embedding_table,
      momentum,
      velocity,
      activations_grad,
      alpha_t,
      beta_1,
      beta_2,
      epsilon,
      device_batch_size,
      max_ids_per_partition,
      max_unique_ids_per_partition,
      computation_name,
  )


@annotations.experimental(
    "torch.tpu.ops.sparse_dense_matmul_activation_unstack is experimental and"
    " subject to change."
)
def sparse_dense_matmul_activation_unstack(
    stacked_activations: torch.Tensor,
    per_feature_batch_sizes: Sequence[int],
    per_feature_dims: Sequence[int],
) -> list[torch.Tensor]:
  """Splits the stacked activations of multiple features into one per feature.

  This op is differentiable. Its backward is
  `sparse_dense_matmul_gradient_stack`.

  Args:
    stacked_activations: The stacked activations of all features.
    per_feature_batch_sizes: The batch size of each feature.
    per_feature_dims: The embedding dimension of each feature.

  Returns:
    A list with the activations of each feature.
  """
  return torch.ops.tpu._sparse_dense_matmul_activation_unstack(  # pylint: disable=protected-access
      stacked_activations, per_feature_batch_sizes, per_feature_dims
  )


@annotations.experimental(
    "torch.tpu.ops.sparse_dense_matmul_gradient_stack is experimental and"
    " subject to change."
)
def sparse_dense_matmul_gradient_stack(
    unstacked_gradients: Sequence[torch.Tensor],
    stacked_batch_size: int,
    stacked_feature_dim: int,
) -> torch.Tensor:
  """Stacks the activation gradients of multiple features into one tensor.

  This is the inverse of `sparse_dense_matmul_activation_unstack`.

  Args:
    unstacked_gradients: The activation gradients of each feature.
    stacked_batch_size: The batch size of the stacked gradient.
    stacked_feature_dim: The feature dimension of the stacked gradient.

  Returns:
    The stacked gradient.
  """
  return torch.ops.tpu._sparse_dense_matmul_gradient_stack(  # pylint: disable=protected-access
      unstacked_gradients, stacked_batch_size, stacked_feature_dim
  )


@annotations.experimental(
    "torch.tpu.ops.sparse_gather is experimental and subject to change."
)
def sparse_gather(
    row_pointers: torch.Tensor,
    indices: torch.Tensor,
    operand: torch.Tensor,
    max_non_zeroes_per_row: int,
) -> torch.Tensor:
  """Gathers rows of `operand` on TPU SparseCore.

  This op is differentiable with respect to `operand`. Its backward is
  `sparse_gather_backward`.

  Args:
    row_pointers: A 1D tensor of CSR row pointers.
    indices: A 1D tensor of the indices of the rows of `operand` to gather. Its
      length must be `len(row_pointers) * max_non_zeroes_per_row`.
    operand: The 2D tensor to gather rows from.
    max_non_zeroes_per_row: The maximum number of non-zeroes per CSR row.

  Returns:
    A 2D tensor of the gathered rows.
  """
  return torch.ops.tpu._sparse_gather(  # pylint: disable=protected-access
      row_pointers, indices, operand, max_non_zeroes_per_row
  )


@annotations.experimental(
    "torch.tpu.ops.sparse_gather_backward is experimental and subject to"
    " change."
)
def sparse_gather_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    grad_operand: torch.Tensor,
) -> torch.Tensor:
  """Scatter-adds the gradient of `sparse_gather` into `grad_operand`.

  Args:
    grad_output: The 2D gradient with respect to the output of `sparse_gather`.
    indices: The `indices` that was passed to `sparse_gather`.
    grad_operand: A 2D tensor with the shape of the `operand` that was passed to
      `sparse_gather`, to accumulate the gradient into.

  Returns:
    The gradient with respect to the `operand` of `sparse_gather`.
  """
  return torch.ops.tpu._sparse_gather_backward(  # pylint: disable=protected-access
      grad_output, indices, grad_operand
  )


@annotations.experimental(
    "torch.tpu.ops.sparse_iota is experimental and subject to change."
)
def sparse_iota(
    row_pointers: torch.Tensor,
    max_non_zeroes: int,
    max_non_zeroes_per_row: int,
) -> torch.Tensor:
  """Generates the position of each non-zero within its CSR row on SparseCore.

  Args:
    row_pointers: A 1D int32 tensor of CSR row pointers.
    max_non_zeroes: The maximum total number of non-zeroes.
    max_non_zeroes_per_row: The maximum number of non-zeroes per CSR row.

  Returns:
    A 1D tensor of the generated positions.
  """
  return torch.ops.tpu._sparse_iota(  # pylint: disable=protected-access
      row_pointers, max_non_zeroes, max_non_zeroes_per_row
  )


@annotations.experimental(
    "torch.tpu.ops.sparse_map_row is experimental and subject to change."
)
def sparse_map_row(
    csr_tensors: Sequence[torch.Tensor],
    csr_lengths: Sequence[int],
    dense_args: Sequence[torch.Tensor],
    body_mlir: str,
    dummy_outputs: Sequence[torch.Tensor],
    max_non_zeroes_per_row: int,
    pad_value: int,
    max_pad_per_row: int,
    wait_threshold: float,
    result_row_pointer_op_index: int,
) -> list[torch.Tensor]:
  """Maps a row-wise computation over sparse CSR matrices on TPU SparseCore.

  Args:
    csr_tensors: The tensors of all the CSR matrices, concatenated.
    csr_lengths: The number of tensors in `csr_tensors` for each CSR matrix.
    dense_args: Dense tensors passed to the body computation.
    body_mlir: The MLIR module of the computation to apply to each row.
    dummy_outputs: Tensors with the shapes and dtypes of the outputs.
    max_non_zeroes_per_row: The maximum number of non-zeroes per row.
    pad_value: The value to pad rows with.
    max_pad_per_row: The maximum amount of padding per row.
    wait_threshold: The wait threshold for the SparseCore scheduling.
    result_row_pointer_op_index: The index of the CSR matrix in `csr_lengths`
      whose row pointers the result uses.

  Returns:
    The outputs of the computation.
  """
  return torch.ops.tpu._sparse_map_row(  # pylint: disable=protected-access
      csr_tensors,
      csr_lengths,
      dense_args,
      body_mlir,
      dummy_outputs,
      max_non_zeroes_per_row,
      pad_value,
      max_pad_per_row,
      wait_threshold,
      result_row_pointer_op_index,
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
    "ragged_all_to_all",
    "ragged_dot",
    "ragged_dot_weight_grad",
    "recv",
    "send",
    "sparse_dense_matmul",
    "sparse_dense_matmul_activation_unstack",
    "sparse_dense_matmul_grad_with_adagrad",
    "sparse_dense_matmul_grad_with_adam",
    "sparse_dense_matmul_grad_with_sgd",
    "sparse_dense_matmul_gradient_stack",
    "sparse_gather",
    "sparse_gather_backward",
    "sparse_iota",
    "sparse_map_row",
    # go/keep-sorted end
]
