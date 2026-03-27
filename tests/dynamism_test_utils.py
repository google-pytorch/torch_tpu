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

"""Testing utilities for bounded dynamism.

Handles the following details for per-op unit testing:
  - Valide if an op supports bounded dynamism.
  - Discover valid bounded dynamic candidates.
  - Mark input as dynamic
  - Mark dependent args as dynamic.
"""

from collections.abc import Sequence
import random
from typing import Any

from absl import logging
import torch
from torch.testing._internal import common_methods_invocations
from torch.testing._internal.opinfo import core
from torch_tpu._internal import dynamism
from torch_tpu._internal.utils import utils

# Static variables for op classification.
foreach_binops = frozenset(
    op.name for op in common_methods_invocations.foreach_binary_op_db
)
foreach_ternary_ops = frozenset(
    op.name for op in common_methods_invocations.foreach_pointwise_op_db
)


def _get_dynamic_dimension_candidates(
    input_value: torch.Tensor,
) -> Sequence[int]:
  """Returns a list of dynamic dimension candidates for the given tensor."""
  if isinstance(input_value, torch.Tensor):
    return [i for i, sz in enumerate(input_value.shape) if sz > 1]
  return []


def _get_canonical_input_value(
    op_info: core.OpInfo,
    input_value: torch.Tensor | Sequence[torch.Tensor],
) -> torch.Tensor:
  """Returns the canonical input value for the given input.

  For most ops, this is a noop, but for foreach ops, returns a "canonical" input
  tensor that is used to check if test case is valid for dynamism. This should
  be expanded in the future to handle other cases.

  Args:
    op_info: The op to be tested.
    input_value: The primary input value to for the op test.

  Returns:
    The canonical input value for the given input.
  """

  if isinstance(input_value, (list, tuple)):
    # Scan for valid input values
    valid_inputs = [
        v for v in input_value if not _should_skip_input_marking(op_info, v)
    ]
    if valid_inputs:
      return valid_inputs[0]
    if input_value:
      return input_value[0]
  return input_value


def _should_skip_input_marking(
    op_info: core.OpInfo, input_value: torch.Tensor | Any
) -> None | str:
  """Verifies that the input_value is a valid input for dynamism.

  Args:
    op_info: The op to be tested.
    input_value: The primary input value to for the op test.

  Returns:
    None if the op is safe to run, or a string containing the reason why
    the test is not safe to run.
  """

  if not isinstance(input_value, torch.Tensor):
    # Some ops take scalar inputs
    # empty, empty_strided, full, tril, ones, zeros, stack, tril_indices.
    return f"Input value is not a tensor, got {type(input_value)}."
  dtype = input_value.dtype
  candidates = _get_dynamic_dimension_candidates(input_value)
  if not candidates:
    return "No valid input dimensions to mark dynamic."
  if dtype in [torch.complex64, torch.complex128]:
    return f"Input value has dtype {dtype}."
  if not isinstance(input_value, torch.Tensor):
    return f"Input value is not a tensor, got {type(input_value)}."
  # TODO: b/449736443 - [XLA] Crash in complex rewriter.
  if dtype in [torch.complex64, torch.complex128]:
    return f"{op_info.name} op is not supported with dtype {dtype}."
  # TODO: b/449736443 - Bug in XLA:TPU handling of 64-bit types?
  is_64_bit = dtype in [torch.int64, torch.float64]
  if is_64_bit:
    return f"Dynamism is not well supported with 64-bit types: {dtype}"
  # TODO: b/449736443 - Empty tensor handling needs reworking.
  if input_value.numel() == 0:
    return "Empty tensors are currently not supported."

  return None


def verify_op_supports_dynamism(
    op_info: core.OpInfo,
    input_value: torch.Tensor,
    args: Sequence[Any],
) -> None | str:
  """Checks if an op+input is supported with dynamism.

  Most skip reasons are bugs in lowering or XLA that will be fixed over time,
  enabling more coverage as we go.

  Args:
    op_info: The op to be tested.
    input_value: The primary input value to for the op test.
    args: The additional args to the op test.

  Returns:
    None if the op+input is supported, or an skipTest message if not.
  """

  untriaged_ops_deny_list = [
      "_log_softmax_backward_data",
      "_native_batch_norm_legit",
      "_softmax_backward_data",
      "as_strided",
      "bincount",
      "cdist",
      "constant_pad_nd",
      "cummin",
      "cummax",
      "cumprod",
      "cumsum",
      "diagonal",
      "expand",  # view op - expand not yet supported
      "fft.rfft",
      "histc",
      "index_add",  # invalid bound length (likely bad tensor copy)
      "index_copy",  # invalid bound length (likely bad tensor copy)
      "index_put",  # garbage data
      "index_select",  # invalid gather
      "isin",
      "kron",
      "log_softmax",
      "linalg.solve_ex",
      "linalg.lu_factor_ex",
      "linalg.inv",
      "linalg.lu",
      "linalg.lu_solve",
      "linalg.solve_triangular",
      "masked_fill",
      "masked_scatter",
      "masked_select",
      "matmul",  # reshapes in the structured delegate
      "multinomial",
      "native_batch_norm",
      "native_dropout_backward",
      "nn.functional.adaptive_avg_pool2d",  # sort op fails to infer
      "nn.functional.adaptive_avg_pool3d",
      "nn.functional.avg_pool2d",
      "nn.functional.avg_pool3d",
      "nn.functional.batch_norm",
      "nn.functional.conv1d",
      "nn.functional.conv2d",
      "nn.functional.conv_transpose1d",
      "nn.functional.conv_transpose2d",
      "nn.functional.embedding",
      "nn.functional.embedding_bag",  # invalid gather
      "nn.functional.group_norm",
      "nn.functional.interpolate",  # gather with dynamic slice size
      "nn.functional.leaky_relu",  # binop LHS / RHS mismatch
      "nn.functional.max_pool2d",
      "nn.functional.max_pool3d",
      "nn.functional.mse_loss",  # binop LHS / RHS mismatch
      "nn.functional.nll_loss",
      "nn.functional.pad",
      "nn.functional.pdist",  # inlined vector size fail (likely using bound)
      "nn.functional.rms_norm",
      "nn.functional.scaled_dot_product_attention",
      "nn.functional.upsample_bilinear",
      "nn.functional.upsample_nearest",
      "torch.nn.functional.conv_transpose1d",
      "nonzero",
      "normal",  # OK
      "repeat",
      "resize_",
      "roll",
      "scatter",
      "scatter_add",
      "select",
      "select_scatter",
      "slice",
      "sort",
      "split",
      "split_with_sizes",
      "squeeze",  # unsupported view op - enhance view op to detect squeeze
      "squeeze_copy",  # unsupported view op
      "to",  # fails numerics (?)
      "take",
      "topk",
      "torch.ops.aten._unsafe_view",
      "torch.ops.aten._safe_softmax.default",
      "tril",  # bad broadcast
      "triu",  # bad broadcast, iota-like
      "unbind",
      "unfold",
      "unsqueeze",
      "unsqueeze_copy",
      "vdot",
      "view",
      "view_as_complex",
      "view_as_real",  # OK
  ]
  op = op_info
  if op.name in untriaged_ops_deny_list:
    return f"Op {op.name} is not supported."

  ### [TEST INFRA TASKS] ###
  # The following indicate changes needed in this file to increase coverage.
  input_value = _get_canonical_input_value(op, input_value)
  skip_input = _should_skip_input_marking(op, input_value)
  if skip_input:
    return skip_input
  require_mark_dynamic_support = [
      # Needs matmul-style marking
      "addmm",
      "addmv",
      "dot",
      "native_layer_norm",  # input/weight/bias have dim dependency
      "lu_unpack",  # data, pivots have dim dependency
      # Can only mark the concatenated dim dynamic for cat-like ops.
      "cat",
      "stack",
  ]
  if op.name in require_mark_dynamic_support:
    return f"{op.name} op may work, but requires mark_dynamic test support."

  ### [XLA BUGS] ###
  # TODO: b/449736443 - [XLA] Reduce op with dynamism gets flipped.
  if op.name in ["min", "max"] and args:
    return "min/max with dim flips the result"
  # TODO: b/449736443 - [XLA] Crash in complex rewriter.
  if op.name in ["polar"]:
    return f"{op.name} op is not supported."

  ### [VIEW OP BUGS] ###
  # TODO: b/449736443 - Reshape support for bounded dynamism needed.
  # transposes and permutes can lower to a reshape sometimes (1x3->3x1)
  if op.name in ["flatten", "t", "t_", "transpose", "reshape", "permute"]:
    return f"{op.name} op uses a reshape that is not supported with dynamism."

  ### [TORCHTPU INFRA BUGS] ###
  # TODO: b/449736443 - incompatible buffer shapes at dimension 0;
  reduction_deny_list = ["all", "any"]
  if op.name in reduction_deny_list:
    return f"reduction op {op.name} is not supported with dynamism."

  # Valid testcase!
  return None


def _mark_numpy_broadcastable_arg_dynamic(
    arg: torch.Tensor, like: torch.Tensor, idx: int, ub: int
):
  """Marks dependent dims of a bounded dimension as dynamic.

  XLA has preconditions on bound sizes for certain ops, for example:
    - Binary elementwise ops: LHS/RHS bound sizes must match.
    - Ternary elementwise ops: All operand bound sizes must match.
    - Matmuls: Contracting dimension bound sizes must match.

  This function handles calls mark_dynamic on `arg` for the required
  preconditions of `op_info` and the given `like` tensor having been
  marked as dynamic.

  Args:
    arg: The tensor to mark dynamic.
    like: The tensor that was marked dynamic.
    idx: The dimension of `like` that was marked dynamic.
    ub: The upper bound of the dynamic dimension.
  """

  logging.debug(
      "[_mark_numpy_broadcastable_arg_dynamic] arg=%s, like=%s, idx=%d, ub=%d",
      arg.shape,
      like.shape,
      idx,
      ub,
  )

  # Don't annotate scalar args
  if not arg.shape:
    return

  # Find related `arg` dim given numpy-broadcasting rules from `like`.
  # Don't mark size 1 dims that broadcast or OOB idx when LHS size differs.
  negative_idx = idx - len(like.shape)
  arg_idx = negative_idx + len(arg.shape)
  if arg_idx < 0 or arg_idx >= len(arg.shape) or arg.shape[arg_idx] == 1:
    return

  # Found corresponding arg dim to mark dynamic.
  assert arg.shape[arg_idx] == like.shape[idx], "bound dims mismatch"
  print(
      f">>>> Marking dependent arg dim {arg_idx} dynamic [<={ub}] in"
      f" {utils.InputMetadata(arg)}",
      flush=True,
  )
  dynamism.mark_dynamic(arg, dimension=arg_idx, lower_bound=2, upper_bound=ub)


class DynamicOpInfo:
  """A wrapper around OpInfo that supports bounded dynamism."""

  def __init__(self, op_info: core.OpInfo):
    self.op_info = op_info

  @staticmethod
  def get(op_info: core.OpInfo):
    """Returns a DynamicOpInfo for the given op."""
    adapters = [
        ForEachDynamicOpInfo,
        MatmulDynamicOpInfo,
        BinaryElementwiseDynamicOpInfo,
        TernaryElementwiseDynamicOpInfo,
    ]
    for adapter in adapters:
      if adapter.adapter_supports_op(op_info):
        return adapter(op_info)
    return DynamicOpInfo(op_info)

  def mark_dynamic(
      self, seed: int, input_value: torch.Tensor, args: Sequence[Any]
  ):
    """Mark the op as dynamic."""

    logging.debug(
        "[DynamicOpInfo] mark_dynamic(%s, %s, %s)",
        self.op_info.name,
        utils.InputMetadata(input_value),
        utils.InputMetadata(args),
    )

    # Allow picking a list value from input_value for ops like cat, etc.
    input_value = _get_canonical_input_value(self.op_info, input_value)
    candidates = _get_dynamic_dimension_candidates(input_value)
    random.seed(seed)
    idx = random.choice(candidates)
    input_value_str = utils.get_tensor_summary(input_value, data=False)
    ub = input_value.shape[idx] + 10
    print(
        f">>>> Marking dim {idx} dynamic [<={ub}] in {input_value_str}",
        flush=True,
    )

    # Mark input_value as dynamic.
    dynamism.mark_dynamic(
        input_value, dimension=idx, lower_bound=2, upper_bound=ub
    )

    # Mark any dependent args as dynamic.
    self._mark_dependent_arg_dynamic(args, input_value, idx, ub)

  def _mark_dependent_arg_dynamic(
      self, args: Sequence[Any], like: torch.Tensor, idx: int, ub: int
  ):
    """Marks dependent dims of a bounded dimension as dynamic."""
    # No dependent args to mark.
    del like, idx, ub
    logging.debug(
        "[DynamicOpInfo] _mark_dependent_arg_dynamic %s %s",
        self.op_info.name,
        utils.InputMetadata(args),
    )
    return


class BinaryElementwiseDynamicOpInfo(DynamicOpInfo):
  """A wrapper around OpInfo that supports bounded dynamism."""

  @staticmethod
  def adapter_supports_op(op_info: core.OpInfo):
    """Method to declare ops supported by this adapter."""

    additional_binary_ops = ["equal"]
    return (
        isinstance(op_info, core.BinaryUfuncInfo)
        or op_info.name in additional_binary_ops
    )

  def _mark_dependent_arg_dynamic(
      self,
      args: Sequence[Any],
      like: torch.Tensor,
      idx: int,
      ub: int,
  ):
    """Mark the args of a binary elementwise op as dynamic."""
    rhs = args[0]
    if isinstance(rhs, torch.Tensor):
      _mark_numpy_broadcastable_arg_dynamic(rhs, like, idx, ub)


class TernaryElementwiseDynamicOpInfo(DynamicOpInfo):
  """A wrapper around OpInfo that supports bounded dynamism."""

  ternary_elementwise_ops = set([
      "addcdiv",
      "addcmul",
      "clamp",
      "lerp",
      "where",
  ])

  @staticmethod
  def adapter_supports_op(op_info: core.OpInfo):
    """Method to declare ops supported by this adapter."""

    return (
        op_info.name in TernaryElementwiseDynamicOpInfo.ternary_elementwise_ops
    )

  def _mark_dependent_arg_dynamic(
      self,
      args: Sequence[Any],
      like: torch.Tensor,
      idx: int,
      ub: int,
  ):
    """Mark the args of a ternary elementwise op as dynamic."""
    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
      _mark_numpy_broadcastable_arg_dynamic(args[0], like, idx, ub)
    if len(args) >= 2 and isinstance(args[1], torch.Tensor):
      _mark_numpy_broadcastable_arg_dynamic(args[1], like, idx, ub)


class ForEachDynamicOpInfo(DynamicOpInfo):
  """A wrapper around OpInfo that supports bounded dynamism."""

  # Max number of inputs in list to mark as dynamic (slow to run)
  max_dynamic_args = 3

  @staticmethod
  def adapter_supports_op(op_info: core.OpInfo):
    """Method to declare ops supported by this adapter."""

    return isinstance(op_info, core.ForeachFuncInfo)

  def mark_dynamic(
      self,
      seed: int,
      input_value: torch.Tensor | Sequence[torch.Tensor],
      args: Sequence[Any],
  ):
    """Mark the op as dynamic, handling for list types."""
    # Split the inputs into their logical units for marking dynamic.
    # Options are:
    #   - foreach_add(list, [scalar|tensor])
    #   - foreach_add(list, [[arg0s], [args1s], ...])

    logging.debug(
        "[ForEachDynamicOpInfo] mark_dynamic(%s, %s, %s)",
        self.op_info.name,
        utils.InputMetadata(input_value),
        utils.InputMetadata(args),
    )
    n_dynamic = 0
    for idx, value in enumerate(input_value):
      # If args not a list, no need to mark anything dependent dynamic
      if _should_skip_input_marking(self.op_info, value):
        continue

      # For test speed, limit number of dynamic args.
      if n_dynamic > self.max_dynamic_args:
        break
      n_dynamic += 1

      # foreach_add(list, scalar)
      if (
          not args
          or not isinstance(args, (list, tuple))
          or not isinstance(args[0], (list, tuple))
      ):
        super().mark_dynamic(seed, value, [])
        continue

      # Binary ops expect len(args) == 1
      if self.op_info.name in foreach_binops:
        rhs_vals = args[0]
        BinaryElementwiseDynamicOpInfo(self.op_info).mark_dynamic(
            seed, value, (rhs_vals[idx],)
        )
        continue

      # Ternary ops expect len(args) == 2
      base_op_name = self.op_info.name.split("_foreach_")[1]
      if self.op_info.name in foreach_ternary_ops or (
          base_op_name
          in TernaryElementwiseDynamicOpInfo.ternary_elementwise_ops
      ):
        rhs_vals = args[0] if len(args) >= 1 else None
        ehs_vals = args[1] if len(args) >= 2 else None
        TernaryElementwiseDynamicOpInfo(self.op_info).mark_dynamic(
            seed, value, (rhs_vals[idx], ehs_vals[idx])
        )
        continue

      # Not binary or ternary, no dependent args.
      super().mark_dynamic(seed, value, [])
      logging.debug(
          "[ForEachDynamicOpInfo] no dependent args %s",
          utils.InputMetadata(args),
      )


class MatmulDynamicOpInfo(DynamicOpInfo):
  """A wrapper around OpInfo that supports bounded dynamism."""

  @staticmethod
  def adapter_supports_op(op_info: core.OpInfo):
    """Method to declare ops supported by this adapter."""

    matmul_ops = ["bmm", "mm"]
    return op_info.name in matmul_ops

  def _mark_dependent_arg_dynamic(
      self,
      args: Sequence[Any],
      like: torch.Tensor,
      idx: int,
      ub: int,
  ):
    """Mark the args of a binary elementwise op as dynamic."""
    rhs = args[0]
    if isinstance(rhs, torch.Tensor):
      lhs_ndim = like.ndim
      rhs_ndim = rhs.ndim
      # If input contracting dim is dynamic, mark other's contracting dim too.
      if idx == lhs_ndim - 1:
        rhs_cdim = 0 if rhs_ndim == 1 else rhs_ndim - 2
        if rhs.shape[rhs_cdim] > 1:
          rhs_value_str = utils.get_tensor_summary(rhs, data=False)
          print(
              f">>>> Marking dim {rhs_cdim} dynamic [<={ub}] in"
              f" {rhs_value_str}",
              flush=True,
          )
          dynamism.mark_dynamic(
              rhs, dimension=rhs_cdim, lower_bound=2, upper_bound=ub
          )
      # If a batch dim is dynamic, mark rhs's batch dim too.
      elif idx < lhs_ndim - 2:
        _mark_numpy_broadcastable_arg_dynamic(rhs, like, idx, ub)
    return


def mark_input_dynamic(
    seed: int,
    op_info: core.OpInfo,
    input_value: torch.Tensor | Sequence[torch.Tensor],
    args: Sequence[Any],
):
  """Mark an arg tensor as dynamic.

  Args:
    seed: The seed to use for random choice of the dimension to mark dynamic.
    op_info: The op to be tested.
    input_value: The primary input value to for the op test.
    args: The additional args to the op test.
  """

  dynamic_op_info = DynamicOpInfo.get(op_info)
  print(">>> Marking input dynamic using ", type(dynamic_op_info))
  dynamic_op_info.mark_dynamic(seed, input_value, args)
