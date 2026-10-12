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
  - Validate if an op supports bounded dynamism.
  - Discover valid bounded dynamic candidates.
  - Mark input as dynamic
  - Mark dependent args as dynamic.
"""

from collections.abc import Mapping, Sequence
import enum
import random
from typing import Any

from absl import logging
import torch
from torch.testing._internal import common_methods_invocations
from torch.testing._internal.opinfo import core
from torch.utils import _pytree
from torch_tpu._internal import dynamism
from torch_tpu._internal.utils import utils


class DynamismMode(enum.Enum):
  """Mode of dynamism testing."""

  EAGER = "eager"
  COMPILE = "compile"


EAGER = DynamismMode.EAGER
COMPILE = DynamismMode.COMPILE
ALL_MODES = frozenset({DynamismMode.EAGER, DynamismMode.COMPILE})


def _mark_tensor_dynamic(
    tensor: torch.Tensor,
    dimension: int,
    lower_bound: int,
    upper_bound: int,
    mode: DynamismMode = DynamismMode.EAGER,
) -> None:
  """Marks a dimension of a tensor as dynamic according to the specified mode."""
  if mode == DynamismMode.EAGER:
    dynamism.mark_dynamic(
        tensor,
        dimension=dimension,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
  elif mode == DynamismMode.COMPILE:
    torch._dynamo.mark_dynamic(
        tensor,
        dimension,
        min=lower_bound,
        max=upper_bound,
    )
  else:
    raise ValueError(f"Unsupported dynamism mode: {mode}")


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
  # TODO: b/449736443 - [XLA] Crash in complex rewriter.
  if dtype in [torch.complex64, torch.complex128]:
    return f"{op_info.name} op is not supported with dtype {dtype}."
  # _foreach_exp_ does not support integral dtypes on TPU.
  if op_info.name == "_foreach_exp_" and not dtype.is_floating_point:
    return f"{op_info.name} does not support integral dtypes."
  # ceil/floor do not support int64 under dynamism (sign-extension bugs).
  if (
      op_info.name in ["ceil", "floor", "_foreach_ceil", "_foreach_floor"]
      and dtype == torch.int64
  ):
    return f"{op_info.name} does not support int64 under dynamism."
  # TODO: b/449736443 - Empty tensor handling needs reworking.
  if input_value.numel() == 0:
    return "Empty tensors are currently not supported."

  return None


# Mode-aware deny list mapping op names to the set of DynamismMode(s) for which they are excluded.
untriaged_ops_deny_list: dict[
    str, set[DynamismMode] | frozenset[DynamismMode]
] = {
    "angle": {COMPILE},  # Dynamo error
    "bincount": ALL_MODES,  # MLIR assertion failure (bounds length vs rank)
    "bmm": {COMPILE},  # numerical mismatch under dynamic compile
    "clamp_min": ALL_MODES,  # dynamic triage (invalid result)
    "conj_physical": ALL_MODES,  # identity operation for i64 and f64 fails
    "diagonal": ALL_MODES,  # materialization failure (negative dimension size)
    "eq": ALL_MODES,  # dynamic triage (invalid result)
    "expand": ALL_MODES,  # view op - expand not yet supported
    "expm1": ALL_MODES,  # dynamic triage (result differ)
    "fft.rfft": ALL_MODES,  # expected all dimensions to be static
    "fft.irfft": ALL_MODES,  # expected all dimensions to be static
    "fft.ifft": ALL_MODES,  # expected all dimensions to be static
    "fft.fft": ALL_MODES,  # expected all dimensions to be static
    "flip": ALL_MODES,  # dynamic triage (invalid result)
    "floor_divide": ALL_MODES,  # currently failing dynamism tests
    "full_like": {COMPILE},  # buffer dimension mismatch
    "index_add": ALL_MODES,  # invalid bound length (likely bad tensor copy)
    "index_copy": ALL_MODES,  # invalid bound length (likely bad tensor copy)
    "index_put": ALL_MODES,  # garbage data
    "index_select": ALL_MODES,  # invalid gather
    "isinf": {COMPILE},  # buffer dimension mismatch
    "isin": ALL_MODES,  # materialization failure (negative dimension size)
    "kron": ALL_MODES,  # dynamic triage (invalid result)
    "ldexp": ALL_MODES,  # dynamic shape value permutation mismatch
    "linalg.lu_factor_ex": (
        ALL_MODES
    ),  # materialization failure (negative dim size)
    "logical_and": ALL_MODES,  # dynamic triage (invalid result)
    "masked_scatter": ALL_MODES,  # unflatten ambiguous error
    "matmul": ALL_MODES,  # reshapes in the structured delegate
    "mm": {COMPILE},  # numerical mismatch under dynamic compile
    "maximum": ALL_MODES,  # dynamic triage (invalid result)
    "ne": ALL_MODES,  # dynamic triage (invalid result)
    "neg": ALL_MODES,  # dynamic triage (invalid result)
    "nn.functional.adaptive_avg_pool2d": ALL_MODES,  # sort op fails to infer
    "nn.functional.batch_norm": ALL_MODES,  # dynamic triage (xla error)
    "nn.functional.embedding": ALL_MODES,  # crash in shard 27
    "nn.functional.embedding_bag": ALL_MODES,  # invalid gather
    "nn.functional.grid_sample": ALL_MODES,  #
    "nn.functional.hardswish": ALL_MODES,  # dynamic triage (adjust tolerance)
    "nn.functional.interpolate": ALL_MODES,  # gather with dynamic slice size
    "nn.functional.leaky_relu": ALL_MODES,  # dynamic triage (invalid result)
    "nn.functional.max_pool1d": (
        ALL_MODES
    ),  # MLIR assertion failure (bounds length vs rank)
    "nn.functional.max_pool2d": (
        ALL_MODES
    ),  # crash (negative dim size in compiled_mode.cc)
    "nn.functional.max_pool3d": ALL_MODES,  # crash (Aborted)
    "nn.functional.mse_loss": ALL_MODES,  # binop LHS / RHS mismatch
    "nn.functional.nll_loss": (
        ALL_MODES
    ),  # numerical mismatch or failure in some shards
    "nn.functional.pdist": (
        ALL_MODES
    ),  # inlined vector size fail (likely using bound)
    "nn.functional.relu": ALL_MODES,  # dynamic triage (tolerance)
    # Also on `compiled_deny_list` in op_testing.py, where compiled results
    # mismatch CPU even with static shapes.
    "nn.functional.scaled_dot_product_attention": {COMPILE},
    "nn.functional.upsample_bilinear": ALL_MODES,  # MLIR gather out of bounds
    "nn.functional.upsample_nearest": ALL_MODES,  # MLIR gather build failure
    "normal": ALL_MODES,  # OK
    "ones_like": {COMPILE},  # buffer dimension mismatch
    "pow": ALL_MODES,  # dynamic triage (invalid result)
    "remainder": ALL_MODES,  # dynamic triage (invalid result)
    "round": ALL_MODES,  # dynamic triage (invalid result)
    "sgn": ALL_MODES,  # dynamic triage (invalid result)
    "searchsorted": ALL_MODES,  # expected last dim to be static
    "sign": ALL_MODES,  # dynamic triage (invalid result)
    "sinh": ALL_MODES,  # dynamic triage (invalid result)
    "slice": ALL_MODES,  # reshape reassociation not supported
    "squeeze": (
        ALL_MODES
    ),  # unsupported view op - enhance view op to detect squeeze
    "squeeze_copy": ALL_MODES,  # unsupported view op
    "std": {COMPILE},  # shape mismatch on out under dynamic compile
    "sub": ALL_MODES,  # dynamic triage (invalid result)
    "sum": ALL_MODES,  # return i64, which has lowering issues
    "take": ALL_MODES,  # dynamic triage (invalid result)
    "tan": ALL_MODES,  # dynamic triage (invalid result)
    "to": ALL_MODES,  # fails numerics (?)
    "tril": ALL_MODES,  # bad broadcast
    "triu": ALL_MODES,  # bad broadcast, iota-like
    "trunc": ALL_MODES,  # dynamic triage (invalid result)
    "unique_consecutive": (
        ALL_MODES
    ),  # dynamic shape operator / invalid TPU shape
    "var": ALL_MODES,  # dynamic triage (invalid result)
    "var_mean": {COMPILE},  # reshape lowering failure in compile mode
    "vdot": ALL_MODES,  # MLIR assertion failure (bounds length vs rank)
    "view_as_complex": ALL_MODES,  # crash (Aborted)
    "view_as_real": ALL_MODES,  # OK
    "xlogy": ALL_MODES,  # dynamic triage (invalid result)
    "zeros_like": {COMPILE},  # buffer dimension mismatch
}

# Mode-aware deny lists of triaged ops, keyed by the reason for the skip. Add
# newly triaged ops here instead of to `untriaged_ops_deny_list`.
# TODO: b/449736443 - Support bounded dimensions in these lowerings.
triaged_ops_deny_lists: dict[
    str, dict[str, set[DynamismMode] | frozenset[DynamismMode]]
] = {
    "the lowering broadcasts a bounded input to a static shape": {
        "baddbmm": {EAGER},
        "bernoulli": {EAGER},
        # These abort the test process with a CHECK failure.
        "linalg.lu_solve": {EAGER},
        "linalg.qr": {EAGER},
        "linalg.solve_triangular": {EAGER},
        "linalg.svd": {EAGER},
        "linalg.svdvals": {EAGER},
    },
    "the lowering emits an op without bounded shape support": {
        # _upsample_bicubic2d_aa and _upsample_bilinear2d_aa lower via 1D gathers with static output sizes.
        "_upsample_bicubic2d_aa": ALL_MODES,
        "_upsample_bilinear2d_aa": ALL_MODES,
        # cross_entropy_loss lowers via gather/scatter and static element count
        # queries, and requires synchronized target/weight dynamic marking.
        "nn.functional.cross_entropy": ALL_MODES,
        "sort": {EAGER},  # stablehlo.iota must be static
        # stablehlo.reshape allows at most one bounded dimension. This fails
        # when dims 0 and 2 are both dynamic, which happens in compile mode
        # because dims marked in earlier subtests stay dynamic.
        # TODO: b/566327707 - Remove once marked dims no longer carry over.
        "torch.ops.aten._safe_softmax.default": {COMPILE},
    },
    "linalg decompositions do not support bounded dimensions": {
        "geqrf": {EAGER},  # Aborts the test process with a CHECK failure.
        "linalg.cholesky": {EAGER},
        "linalg.cholesky_ex": {EAGER},
        "linalg.inv": {EAGER},
        "linalg.lu": {EAGER},
        "linalg.solve_ex": {EAGER},
    },
    "the result has wrong values": {
        # TODO: b/566327914 - With dim 0 bounded, the result matches the
        # transposed golden output.
        "nn.functional.hardtanh": {EAGER},
    },
}


def verify_op_supports_dynamism(
    op_info: core.OpInfo,
    input_value: torch.Tensor,
    args: Sequence[Any],
    mode: DynamismMode = DynamismMode.EAGER,
) -> None | str:
  """Checks if an op+input is supported with dynamism.

  Most skip reasons are bugs in lowering or XLA that will be fixed over time,
  enabling more coverage as we go.

  Args:
    op_info: The op to be tested.
    input_value: The primary input value to for the op test.
    args: The additional args to the op test.
    mode: The active dynamism mode (EAGER or COMPILE).

  Returns:
    None if the op+input is supported, or an skipTest message if not.
  """
  op = op_info
  if op.name in untriaged_ops_deny_list:
    denied_modes = untriaged_ops_deny_list[op.name]
    if mode in denied_modes:
      return f"Op {op.name} is not supported in {mode.value} dynamism mode."
  for reason, deny_list in triaged_ops_deny_lists.items():
    if mode in deny_list.get(op.name, ()):
      return (
          f"Op {op.name} is not supported in {mode.value} dynamism mode:"
          f" {reason}."
      )

  ### [TEST INFRA TASKS] ###
  # The following indicate changes needed in this file to increase coverage.

  # Verify all input tensors are supported with dynamism. Here we use
  # _pytree.tree_flatten to recursively inspect the input_value and args,
  # ensuring that tensors inside lists, tuples, or other nested structures
  # (common in foreach ops) are properly vetted.
  values_to_check = (
      _pytree.tree_flatten(input_value)[0]
      if op.name == "einsum"
      else _pytree.tree_flatten((input_value, args))[0]
  )
  for value in values_to_check:
    skip_input = _should_skip_input_marking(op, value)
    if skip_input:
      return skip_input

  # The loop above sees no values in an empty input, e.g. the shape `()` passed
  # to `torch.empty`, so also check the input that `mark_dynamic` will mark.
  canonical_input = _get_canonical_input_value(op, input_value)
  skip_input = _should_skip_input_marking(op, canonical_input)
  if skip_input:
    return skip_input
  if not DynamicOpInfo.get(op).dynamic_dimension_candidates(
      input_value if op.name == "einsum" else canonical_input, args
  ):
    return "No valid input dimensions to mark dynamic."

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
  # TODO: b/449736443 - [XLA] Crash in complex rewriter.
  if op.name in ["polar"]:
    return f"{op.name} op is not supported."

  ### [VIEW OP BUGS] ###
  # TODO: b/449736443 - Reshape support for bounded dynamism needed.
  # transposes and permutes can lower to a reshape sometimes (1x3->3x1)
  if op.name in ["t", "t_"]:
    return f"{op.name} op uses a reshape that is not supported with dynamism."

  ### [TORCHTPU INFRA BUGS] ###
  # TODO: b/449736443 - incompatible buffer shapes at dimension 0;
  reduction_deny_list = ["all", "any"]
  if op.name in reduction_deny_list:
    return f"reduction op {op.name} is not supported with dynamism."

  # Valid testcase!
  return None


def _mark_numpy_broadcastable_arg_dynamic(
    arg: torch.Tensor,
    like: torch.Tensor,
    idx: int,
    ub: int,
    mode: DynamismMode = DynamismMode.EAGER,
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
    mode: The dynamism mode (EAGER or COMPILE).
  """

  logging.debug(
      "[_mark_numpy_broadcastable_arg_dynamic] arg=%s, like=%s, idx=%d, ub=%d,"
      " mode=%s",
      arg.shape,
      like.shape,
      idx,
      ub,
      mode,
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
      f" {utils.InputMetadata(arg)} (mode={mode.value})",
      flush=True,
  )
  _mark_tensor_dynamic(arg, arg_idx, 2, ub, mode=mode)


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
        EinsumDynamicOpInfo,
        SdpaDynamicOpInfo,
        LinearDynamicOpInfo,
        BinaryElementwiseDynamicOpInfo,
        TernaryElementwiseDynamicOpInfo,
        ConvolutionDynamicOpInfo,
    ]
    for adapter in adapters:
      if adapter.adapter_supports_op(op_info):
        return adapter(op_info)
    return DynamicOpInfo(op_info)

  def dynamic_dimension_candidates(
      self, input_value: torch.Tensor, args: Sequence[Any]
  ) -> Sequence[int]:
    """Returns the dimensions of input_value that can be marked dynamic."""
    del args  # Unused.
    return _get_dynamic_dimension_candidates(input_value)

  def mark_dynamic(
      self,
      seed: int,
      input_value: torch.Tensor,
      args: Sequence[Any],
      mode: DynamismMode = DynamismMode.EAGER,
      kwargs: Mapping[str, Any] | None = None,
  ):
    """Mark the op as dynamic."""

    logging.debug(
        "[DynamicOpInfo] mark_dynamic(%s, %s, %s, mode=%s)",
        self.op_info.name,
        utils.InputMetadata(input_value),
        utils.InputMetadata(args),
        mode,
    )

    # Allow picking a list value from input_value for ops like cat, etc.
    input_value = _get_canonical_input_value(self.op_info, input_value)
    candidates = self.dynamic_dimension_candidates(input_value, args)
    random.seed(seed)
    idx = random.choice(candidates)
    input_value_str = utils.get_tensor_summary(input_value, data=False)
    ub = input_value.shape[idx] + 10
    print(
        f">>>> Marking dim {idx} dynamic [<={ub}] in {input_value_str}"
        f" (mode={mode.value})",
        flush=True,
    )

    # Mark input_value as dynamic.
    _mark_tensor_dynamic(input_value, idx, 2, ub, mode=mode)

    # Mark any dependent args as dynamic.
    self._mark_dependent_arg_dynamic(
        args, input_value, idx, ub, mode=mode, kwargs=kwargs
    )

  def _mark_dependent_arg_dynamic(
      self,
      args: Sequence[Any],
      like: torch.Tensor,
      idx: int,
      ub: int,
      mode: DynamismMode = DynamismMode.EAGER,
      kwargs: Mapping[str, Any] | None = None,
  ):
    """Marks dependent dims of a bounded dimension as dynamic."""
    # No dependent args to mark.
    del like, idx, ub, mode, kwargs
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

    additional_binary_ops = [
        "_log_softmax_backward_data",
        "_softmax_backward_data",
        "equal",
    ]
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
      mode: DynamismMode = DynamismMode.EAGER,
      kwargs: Mapping[str, Any] | None = None,
  ):
    """Mark the args of a binary elementwise op as dynamic."""
    del kwargs
    rhs = args[0]
    if isinstance(rhs, torch.Tensor):
      _mark_numpy_broadcastable_arg_dynamic(rhs, like, idx, ub, mode=mode)


class TernaryElementwiseDynamicOpInfo(DynamicOpInfo):
  """A wrapper around OpInfo that supports bounded dynamism."""

  ternary_elementwise_ops = {
      "addcdiv",
      "addcmul",
      "clamp",
      "lerp",
      "masked_fill",
      "where",
  }

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
      mode: DynamismMode = DynamismMode.EAGER,
      kwargs: Mapping[str, Any] | None = None,
  ):
    """Mark the args of a ternary elementwise op as dynamic."""
    del kwargs
    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
      _mark_numpy_broadcastable_arg_dynamic(args[0], like, idx, ub, mode=mode)
    if len(args) >= 2 and isinstance(args[1], torch.Tensor):
      _mark_numpy_broadcastable_arg_dynamic(args[1], like, idx, ub, mode=mode)


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
      mode: DynamismMode = DynamismMode.EAGER,
      kwargs: Mapping[str, Any] | None = None,
  ):
    """Mark the op as dynamic, handling for list types."""
    del kwargs
    # Split the inputs into their logical units for marking dynamic.
    # Options are:
    #   - foreach_add(list, [scalar|tensor])
    #   - foreach_add(list, [[arg0s], [args1s], ...])

    logging.debug(
        "[ForEachDynamicOpInfo] mark_dynamic(%s, %s, %s, mode=%s)",
        self.op_info.name,
        utils.InputMetadata(input_value),
        utils.InputMetadata(args),
        mode,
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
        super().mark_dynamic(seed, value, [], mode=mode)
        continue

      # Binary ops expect len(args) == 1
      if self.op_info.name in foreach_binops:
        rhs_vals = args[0]
        BinaryElementwiseDynamicOpInfo(self.op_info).mark_dynamic(
            seed, value, (rhs_vals[idx],), mode=mode
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
            seed,
            value,
            (rhs_vals[idx], ehs_vals[idx]),  # pyrefly: ignore[unsupported-operation]
            mode=mode,
        )
        continue

      # Not binary or ternary, no dependent args.
      super().mark_dynamic(seed, value, [], mode=mode)
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
      mode: DynamismMode = DynamismMode.EAGER,
      kwargs: Mapping[str, Any] | None = None,
  ):
    """Mark the args of a binary elementwise op as dynamic."""
    del kwargs
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
              f" {rhs_value_str} (mode={mode.value})",
              flush=True,
          )
          _mark_tensor_dynamic(rhs, rhs_cdim, 2, ub, mode=mode)
      # If a batch dim is dynamic, mark rhs's batch dim too.
      elif idx < lhs_ndim - 2:
        _mark_numpy_broadcastable_arg_dynamic(rhs, like, idx, ub, mode=mode)
    return


class LinearDynamicOpInfo(DynamicOpInfo):
  """A wrapper around nn.functional.linear OpInfo that supports bounded dynamism."""

  @staticmethod
  def adapter_supports_op(op_info: core.OpInfo) -> bool:
    """Method to declare ops supported by this adapter."""
    return op_info.name == "nn.functional.linear"

  def _mark_dependent_arg_dynamic(
      self,
      args: Sequence[Any],
      like: torch.Tensor,
      idx: int,
      ub: int,
      mode: DynamismMode = DynamismMode.EAGER,
      kwargs: Mapping[str, Any] | None = None,
  ):
    """Marks weight's contracting dim dynamic when input's last dim is dynamic."""
    del kwargs
    weight = args[0]
    if isinstance(weight, torch.Tensor) and idx == like.ndim - 1:
      weight_cdim = weight.ndim - 1
      if weight.shape[weight_cdim] > 1:
        weight_value_str = utils.get_tensor_summary(weight, data=False)
        print(
            f">>>> Marking dim {weight_cdim} dynamic [<={ub}] in"
            f" {weight_value_str} (mode={mode.value})",
            flush=True,
        )
        _mark_tensor_dynamic(weight, weight_cdim, 2, ub, mode=mode)


class ConvolutionDynamicOpInfo(DynamicOpInfo):
  """A wrapper around convolution OpInfos that keeps input channels static."""

  convolution_ops = {
      "nn.functional.conv1d",
      "nn.functional.conv2d",
      "nn.functional.conv3d",
      "nn.functional.conv_transpose1d",
      "nn.functional.conv_transpose2d",
      "nn.functional.conv_transpose3d",
  }

  @staticmethod
  def adapter_supports_op(op_info: core.OpInfo) -> bool:
    """Method to declare ops supported by this adapter."""

    return op_info.name in ConvolutionDynamicOpInfo.convolution_ops

  def dynamic_dimension_candidates(
      self, input_value: torch.Tensor, args: Sequence[Any]
  ) -> Sequence[int]:
    """Returns the batch and spatial dimensions of the convolution input.

    The input channels must match the static weight, so convolutions don't
    support bounded channel dimensions.
    """
    # An unbatched input has one dimension less than the weight in args[0].
    channel_dim = 1 if input_value.ndim == args[0].ndim else 0
    return [
        dim
        for dim in super().dynamic_dimension_candidates(input_value, args)
        if dim != channel_dim
    ]


class SdpaDynamicOpInfo(DynamicOpInfo):
  """A wrapper around OpInfo for scaled_dot_product_attention."""

  @staticmethod
  def adapter_supports_op(op_info: core.OpInfo) -> bool:
    """Method to declare ops supported by this adapter."""

    return op_info.name == "nn.functional.scaled_dot_product_attention"

  def dynamic_dimension_candidates(
      self, input_value: torch.Tensor, args: Sequence[Any]
  ) -> Sequence[int]:
    """Returns the batch dimensions of the query (rank >= 4, dims 0..ndim-4)."""
    del args  # Unused.
    if isinstance(input_value, torch.Tensor) and input_value.ndim >= 4:
      return [
          i
          for i, sz in enumerate(input_value.shape)
          if sz > 1 and i <= input_value.ndim - 4
      ]
    return []

  def _mark_dependent_arg_dynamic(
      self,
      args: Sequence[Any],
      like: torch.Tensor,
      idx: int,
      ub: int,
      mode: DynamismMode = DynamismMode.EAGER,
      kwargs: Mapping[str, Any] | None = None,
  ):
    """Mark the key, value and attn_mask batch dims as dynamic."""
    lhs_ndim = like.ndim
    if idx > lhs_ndim - 4:
      return
    if len(args) >= 2:
      key = args[0]
      value = args[1]
      if (
          isinstance(key, torch.Tensor)
          and isinstance(value, torch.Tensor)
          and key.ndim == lhs_ndim
          and value.ndim == lhs_ndim
      ):
        _mark_numpy_broadcastable_arg_dynamic(key, like, idx, ub, mode=mode)
        _mark_numpy_broadcastable_arg_dynamic(value, like, idx, ub, mode=mode)
    if kwargs and "attn_mask" in kwargs:
      attn_mask = kwargs["attn_mask"]
      if isinstance(attn_mask, torch.Tensor):
        _mark_numpy_broadcastable_arg_dynamic(
            attn_mask, like, idx, ub, mode=mode
        )


class EinsumDynamicOpInfo(DynamicOpInfo):
  """A wrapper around OpInfo that supports bounded dynamism for einsum."""

  @staticmethod
  def adapter_supports_op(op_info: core.OpInfo) -> bool:
    """Method to declare ops supported by this adapter."""
    return op_info.name == "einsum"

  @staticmethod
  def _tokenize_subscript(sub: str) -> list[str]:
    tokens = []
    i = 0
    while i < len(sub):
      if sub[i : i + 3] == "...":
        tokens.append("...")
        i += 3
      elif sub[i] != " ":
        tokens.append(sub[i])
        i += 1
      else:
        i += 1
    return tokens

  @classmethod
  def _parse_einsum_labels(
      cls, tensors: Sequence[torch.Tensor], equation: str
  ) -> tuple[list[list[str]], list[str]]:
    parts = equation.split("->")
    lhs = parts[0]
    subs = lhs.split(",")
    operand_tokens = [cls._tokenize_subscript(s) for s in subs]

    max_ell_dims = 0
    for tensor, tokens in zip(tensors, operand_tokens, strict=True):
      if "..." in tokens:
        ell_dims = tensor.ndim - (len(tokens) - 1)
        max_ell_dims = max(max_ell_dims, ell_dims)

    operand_labels: list[list[str]] = []
    label_counts: dict[str, int] = {}
    for tensor, tokens in zip(tensors, operand_tokens, strict=True):
      labels: list[str] = []
      for tok in tokens:
        if tok == "...":
          ell_dims = tensor.ndim - (len(tokens) - 1)
          for k in range(max_ell_dims - ell_dims, max_ell_dims):
            labels.append(f"...{k}")
        else:
          labels.append(tok)
          label_counts[tok] = label_counts.get(tok, 0) + 1
      operand_labels.append(labels)

    ell_output_labels = [f"...{k}" for k in range(max_ell_dims)]
    if len(parts) == 1:
      rhs_labels = ell_output_labels + sorted(
          lbl for lbl, cnt in label_counts.items() if cnt == 1
      )
    else:
      rhs_tokens = cls._tokenize_subscript(parts[1])
      rhs_labels = []
      for tok in rhs_tokens:
        if tok == "...":
          rhs_labels.extend(ell_output_labels)
        else:
          rhs_labels.append(tok)

    return operand_labels, rhs_labels

  def dynamic_dimension_candidates(
      self,
      input_value: torch.Tensor | Sequence[torch.Tensor],
      args: Sequence[Any],
  ) -> Sequence[int]:
    """Returns the dimensions of the first operand that can be marked dynamic."""
    tensors: list[torch.Tensor] = (
        [input_value]
        if isinstance(input_value, torch.Tensor)
        else list(input_value)
    )
    if (
        not tensors
        or not isinstance(tensors[0], torch.Tensor)
        or not args
        or not isinstance(args[0], str)
    ):
      return []

    operand_labels, rhs_labels = self._parse_einsum_labels(tensors, args[0])
    if not operand_labels or len(operand_labels[0]) != tensors[0].ndim:
      return []

    any_requires_grad = any(t.requires_grad for t in tensors)
    if any_requires_grad:
      # Repeated labels within an operand use diagonal_scatter into a static
      # zeros tensor in backward, and uncontracted single-operand dimensions
      # summed out in forward use static expand in backward.
      if any(len(set(lbls)) != len(lbls) for lbls in operand_labels):
        return []
      for k, t_k in enumerate(tensors):
        if not t_k.requires_grad:
          continue
        for d_k, lbl in enumerate(operand_labels[k]):
          if t_k.shape[d_k] <= 1:
            continue
          in_bwd_lhs = lbl in rhs_labels or any(
              lbl in operand_labels[j] for j in range(len(tensors)) if j != k
          )
          if not in_bwd_lhs:
            return []

    first_tensor = tensors[0]
    first_labels = operand_labels[0]
    candidates = []
    for d, sz in enumerate(first_tensor.shape):
      if sz <= 1:
        continue
      lbl = first_labels[d]
      # Repeated labels with size > 1 use stablehlo.iota in forward.
      if any(lbls.count(lbl) > 1 for lbls in operand_labels):
        continue
      candidates.append(d)
    return candidates

  def mark_dynamic(
      self,
      seed: int,
      input_value: torch.Tensor | Sequence[torch.Tensor],
      args: Sequence[Any],
      mode: DynamismMode = DynamismMode.EAGER,
      kwargs: Mapping[str, Any] | None = None,
  ) -> None:
    """Marks a chosen dimension of the first operand and matching label dims dynamic."""
    del kwargs
    tensors: list[torch.Tensor] = (
        [input_value]
        if isinstance(input_value, torch.Tensor)
        else list(input_value)
    )
    candidates = self.dynamic_dimension_candidates(tensors, args)
    random.seed(seed)
    idx = random.choice(candidates)
    first_tensor = tensors[0]
    ub = first_tensor.shape[idx] + 10
    first_tensor_str = utils.get_tensor_summary(first_tensor, data=False)
    print(
        f">>>> Marking dim {idx} dynamic [<={ub}] in {first_tensor_str}"
        f" (mode={mode.value})",
        flush=True,
    )
    _mark_tensor_dynamic(first_tensor, idx, 2, ub, mode=mode)

    operand_labels, _ = self._parse_einsum_labels(tensors, args[0])
    target_label = operand_labels[0][idx]
    for t_k, lbls_k in zip(tensors[1:], operand_labels[1:], strict=True):
      if target_label in lbls_k:
        d_k = lbls_k.index(target_label)
        if t_k.shape[d_k] == first_tensor.shape[idx] and t_k.shape[d_k] > 1:
          print(
              f">>>> Marking dependent arg dim {d_k} dynamic [<={ub}] in"
              f" {utils.InputMetadata(t_k)} (mode={mode.value})",
              flush=True,
          )
          _mark_tensor_dynamic(t_k, d_k, 2, ub, mode=mode)


def mark_input_dynamic(
    seed: int,
    op_info: core.OpInfo,
    input_value: torch.Tensor | Sequence[torch.Tensor],
    args: Sequence[Any],
    mode: DynamismMode = DynamismMode.EAGER,
    kwargs: Mapping[str, Any] | None = None,
):
  """Mark an arg tensor as dynamic.

  Args:
    seed: The seed to use for random choice of the dimension to mark dynamic.
    op_info: The op to be tested.
    input_value: The primary input value to for the op test.
    args: The additional args to the op test.
    mode: The dynamism mode (EAGER or COMPILE).
    kwargs: The keyword arguments to the op test.
  """

  dynamic_op_info = DynamicOpInfo.get(op_info)
  print(
      ">>> Marking input dynamic using"
      f" {type(dynamic_op_info)} (mode={mode.value})"
  )
  dynamic_op_info.mark_dynamic(
      seed, input_value, args, mode=mode, kwargs=kwargs
  )
