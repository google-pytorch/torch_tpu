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
import dataclasses
import enum
import itertools
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


@dataclasses.dataclass(frozen=True, order=True)
class TensorDim:
  """Identifies a 0-based dimension on a specific tensor in `input_value`.

  Attributes:
    tensor_index: Non-negative 0-based index of the tensor in `input_value` (`0`
      when `input_value` is a single `Tensor`).
    dim: Non-negative 0-based dimension index on that tensor.
  """

  tensor_index: int
  dim: int


def _format_plan_label(tensor_dims: Sequence[TensorDim]) -> str:
  """Formats a deterministic `DynamicMarkingPlan.label` for `tensor_dims`."""
  if not tensor_dims:
    raise ValueError("tensor_dims must be non-empty to format a plan label.")
  return "_".join(
      f"t{target.tensor_index}d{target.dim}" for target in tensor_dims
  )


@dataclasses.dataclass(frozen=True)
class DynamicMarkingPlan:
  """A deterministic plan describing which tensor dimensions to mark dynamic.

  Attributes:
    label: Identifier used in subtest names and `--dynamism_plan_filter` (e.g.,
      `"t0d0"`, `"t0d0_t0d1"`).
    tensor_dims: Strictly ascending, duplicate-free `TensorDim` targets to mark
      dynamic.
  """

  label: str
  tensor_dims: tuple[TensorDim, ...] = ()

  def __post_init__(self) -> None:
    if any(td.tensor_index < 0 or td.dim < 0 for td in self.tensor_dims) or any(
        a >= b for a, b in itertools.pairwise(self.tensor_dims)
    ):
      raise ValueError(
          "tensor_dims must be non-negative and strictly ascending without"
          f" duplicates, got {self.tensor_dims}."
      )


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
    input_value: torch.Tensor | Sequence[Any],
) -> tuple[int, Any]:
  """Returns `(tensor_index, canonical_value)` for the given `input_value`.

  When `input_value` is a list or tuple, selects the first element that is
  eligible for dynamism (falling back to element `0` if non-empty) and returns
  its 0-based index alongside the value. When `input_value` is a single
  `Tensor` (or an empty/non-sequence input), returns `(0, input_value)`.

  Args:
    op_info: The op to be tested.
    input_value: The primary input value for the op test.

  Returns:
    A `(tensor_index, canonical_value)` tuple, where `tensor_index` is the
    0-based index into `input_value` (`0` when `input_value` is a single
    `Tensor`).
  """
  if isinstance(input_value, (list, tuple)):
    for i, v in enumerate(input_value):
      if not _should_skip_input_marking(op_info, v):
        return i, v
    if input_value:
      return 0, input_value[0]
  return 0, input_value


def _resolve_input_tensor(
    input_value: torch.Tensor | Sequence[torch.Tensor],
    tensor_index: int,
) -> torch.Tensor:
  """Resolves the target `Tensor` in `input_value` from `tensor_index`.

  Args:
    input_value: Single `Tensor` or sequence of `Tensor`s passed to the op.
    tensor_index: 0-based index into `input_value` (`0` when `input_value` is a
      single `Tensor`).

  Returns:
    The resolved `torch.Tensor`.

  Raises:
    IndexError: If `input_value` is a single `Tensor` and `tensor_index != 0`,
      or `tensor_index` is out of range for a sequence `input_value`.
    TypeError: If the resolved element is not a `torch.Tensor`.
  """
  if isinstance(input_value, torch.Tensor):
    if tensor_index != 0:
      raise IndexError(
          f"tensor_index must be 0 for a single Tensor, got {tensor_index}."
      )
    return input_value
  tensor = input_value[tensor_index]
  if not isinstance(tensor, torch.Tensor):
    raise TypeError(
        f"Expected torch.Tensor at input_value[{tensor_index}], got"
        f" {type(tensor).__name__}."
    )
  return tensor


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


DenyList = dict[str, dict[str, set[DynamismMode] | frozenset[DynamismMode]]]

# Mode-aware deny lists for bounded dynamism op tests, keyed by the reason for
# the skip. The reason is included in the skip message.
#
# Policy: every entry is a known bug in a lowering, XLA or TorchTPU, or a gap
# in the dynamism test harness. The goal is to drive this to empty, so remove
# entries as soon as the underlying issue is fixed. Add new entries under a
# diagnosed reason, with a bug where possible. Use an "untriaged: " prefix only
# when the root cause is unknown.
#
# Do not disable dynamism for an op in its test (ops_test.py); add it here.
# For ops where dynamism is out of scope by design, use
# `DYNAMISM_NOT_APPLICABLE_OPS` instead.
#
# TODO: b/449736443 - Fix the underlying issues and remove every entry, so that
# DYNAMISM_DENY_LISTS is empty.
DYNAMISM_DENY_LISTS: DenyList = {
    # TODO: b/449736443 - Support bounded dimensions in these lowerings.
    "the lowering broadcasts a bounded input to a static shape": {
        "baddbmm": {EAGER},
        "baddbmm.dtype": {EAGER},
        "bernoulli": {EAGER},
        # These abort the test process with a CHECK failure.
        "linalg.lu_solve": {EAGER},
        "linalg.qr": {EAGER},
        "linalg.solve_triangular": {EAGER},
        "linalg.svd": {EAGER},
        "linalg.svdvals": {EAGER},
    },
    # TODO: b/449736443 - Support bounded dimensions in these lowerings.
    "the lowering emits an op without bounded shape support": {
        # _upsample_bicubic2d_aa and _upsample_bilinear2d_aa lower via 1D gathers with static output sizes.
        "_upsample_bicubic2d_aa": ALL_MODES,
        "_upsample_bilinear2d_aa": ALL_MODES,
        # embedding_renorm_ uses static BroadcastInDim and Scatter shapes.
        "embedding_renorm_": ALL_MODES,
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
    # TODO: b/449736443 - Support bounded dimensions in these lowerings.
    "linalg decompositions do not support bounded dimensions": {
        "geqrf": {EAGER},  # Aborts the test process with a CHECK failure.
        "linalg.cholesky": {EAGER},
        "linalg.cholesky_ex": {EAGER},
        "linalg.inv": {EAGER},
        "linalg.lu": {EAGER},
        "linalg.solve_ex": {EAGER},
    },
    # TODO: b/449736443 - Support bounded dimensions in these lowerings.
    "the result has wrong values": {
        # TODO: b/566327914 - With dim 0 bounded, the result matches the
        # transposed golden output.
        "nn.functional.hardtanh": {EAGER},
    },
    "the op may work, but requires mark_dynamic test support": {
        # Needs matmul-style marking.
        "_addmm_activation": ALL_MODES,
        "addmm": ALL_MODES,
        "addmm.dtype": ALL_MODES,
        "addmv": ALL_MODES,
        "baddbmm.dtype": {COMPILE},
        "dot": ALL_MODES,
        "native_layer_norm": ALL_MODES,  # input/weight/bias have dim dependency
        "_weight_norm_interface": ALL_MODES,  # v and g have dim dependency
        "_weight_norm_interface_backward": (
            ALL_MODES
        ),  # grad_w, saved_v, saved_g, saved_norms have dim dependency
        "lu_unpack": ALL_MODES,  # data, pivots have dim dependency
        # Can only mark the concatenated dim dynamic for cat-like ops.
        "cat": ALL_MODES,
        "stack": ALL_MODES,
        # col2im column dimension L must match static output_size/kernel_size.
        "col2im": ALL_MODES,
        # Per-channel observer/fake-quant buffers must match channel axis size.
        "_fused_moving_avg_obs_fq_helper": ALL_MODES,
        # RNN input, hidden state, and flat weight list have coupled dimensions.
        "gru.input": ALL_MODES,
        "lstm.input": ALL_MODES,
        # TODO(amainz): Bounded dynamism requires synchronized scale/operand
        # dimension marking and dynamic scale reshape support.
        "torch._scaled_mm": ALL_MODES,
        # TODO(amainz): Bounded dynamism requires synchronized group offset,
        # scale, and operand dimension marking.
        "_grouped_mm": ALL_MODES,
        "torch._scaled_grouped_mm": ALL_MODES,
        # TODO: segment_reduce and jagged ops require length/offset dimension
        # matching under dynamism.
        "_jagged_to_padded_dense_forward": ALL_MODES,
        "_padded_dense_to_jagged_forward": ALL_MODES,
        "_segment_reduce": ALL_MODES,
    },
    # TODO: b/449736443 - [XLA] Crash in complex rewriter.
    "XLA crashes in the complex rewriter": {
        "polar": ALL_MODES,
    },
    # TODO: b/449736443 - Reshape support for bounded dynamism needed.
    "the op may lower to a reshape, which is not supported with dynamism": {
        # transposes and permutes can lower to a reshape sometimes (1x3->3x1)
        "t": ALL_MODES,
    },
    # TODO: b/449736443 - incompatible buffer shapes at dimension 0.
    "reductions hit incompatible buffer shapes at dimension 0": {
        "all": ALL_MODES,
        "any": ALL_MODES,
    },
    "sparse COO ops are not supported": {
        "sparse_coo_add": ALL_MODES,
        "sparse_coo_add_dense_sparse": ALL_MODES,
        "sparse_coo_clone": ALL_MODES,
        "sparse_coo_coalesce": ALL_MODES,
        "sparse_coo_copy_": ALL_MODES,
        "sparse_coo_empty_like": ALL_MODES,
        "sparse_coo_mul": ALL_MODES,
        "sparse_coo_neg": ALL_MODES,
        "sparse_coo_sub": ALL_MODES,
        "sparse_coo_sub_dense_sparse": ALL_MODES,
        "sparse_coo_to_dense": ALL_MODES,
        "sparse_coo_zero_": ALL_MODES,
    },
    "untriaged: invalid result": {
        # TODO: fix clamp() returning enormous errors or nans when dynamism is
        # enabled.
        "clamp": ALL_MODES,
        "clamp_min": ALL_MODES,
        "eq": ALL_MODES,
        "flip": ALL_MODES,
        "index_put": ALL_MODES,  # garbage data
        "kron": ALL_MODES,
        "logical_and": ALL_MODES,
        "maximum": ALL_MODES,
        "ne": ALL_MODES,
        "neg": ALL_MODES,
        "nn.functional.leaky_relu": ALL_MODES,
        "pow": ALL_MODES,
        "remainder": ALL_MODES,
        "round": ALL_MODES,
        "sgn": ALL_MODES,
        "sign": ALL_MODES,
        "sinh": ALL_MODES,
        "sub": ALL_MODES,
        "take": ALL_MODES,
        "tan": ALL_MODES,
        "trunc": ALL_MODES,
        "var": ALL_MODES,
        "xlogy": ALL_MODES,
    },
    "untriaged: numerical mismatch": {
        "bmm": {COMPILE},  # under dynamic compile
        "bmm.dtype": {COMPILE},  # under dynamic compile
        "conj_physical": ALL_MODES,  # identity operation for i64 and f64 fails
        "expm1": ALL_MODES,  # result differs
        "mm": {COMPILE},  # under dynamic compile
        "mm.dtype": {COMPILE},  # under dynamic compile
        "nn.functional.hardswish": ALL_MODES,  # adjust tolerance
        "nn.functional.nll_loss": ALL_MODES,  # or failure in some shards
        "nn.functional.relu": ALL_MODES,  # tolerance
        # Also on `compiled_deny_list` in op_testing.py, where compiled results
        # mismatch CPU even with static shapes.
        "nn.functional.scaled_dot_product_attention": {COMPILE},
        "to": ALL_MODES,  # fails numerics (?)
    },
    "untriaged: buffer dimension mismatch": {
        "full_like": {COMPILE},
        "isinf": {COMPILE},
        "ones_like": {COMPILE},
        "zeros_like": {COMPILE},
    },
    "untriaged: MLIR assertion failure (bounds length vs rank)": {
        "bincount": ALL_MODES,
        "nn.functional.max_pool1d": ALL_MODES,
        "vdot": ALL_MODES,
    },
    "untriaged: materialization failure (negative dimension size)": {
        "diagonal": ALL_MODES,
        "isin": ALL_MODES,
        "linalg.lu_factor_ex": ALL_MODES,
    },
    "untriaged: expected all dimensions to be static": {
        "fft.fft": ALL_MODES,
        "fft.ifft": ALL_MODES,
        "fft.irfft": ALL_MODES,
        "fft.rfft": ALL_MODES,
        "searchsorted": ALL_MODES,  # expected last dim to be static
    },
    "untriaged: invalid bound length": {
        "index_add": ALL_MODES,  # likely bad tensor copy
        "index_copy": ALL_MODES,  # likely bad tensor copy
        "nn.functional.pdist": ALL_MODES,  # inlined vector size fail
    },
    "untriaged: invalid gather": {
        "index_select": ALL_MODES,
        "nn.functional.embedding_bag": ALL_MODES,
        "nn.functional.interpolate": (
            ALL_MODES
        ),  # gather with dynamic slice size
        "nn.functional.upsample_bilinear": ALL_MODES,  # gather out of bounds
        "nn.functional.upsample_nearest": ALL_MODES,  # gather build failure
    },
    "untriaged: crash": {
        "nn.functional.embedding": ALL_MODES,  # crash in shard 27
        # negative dim size in compiled_mode.cc
        "nn.functional.max_pool2d": ALL_MODES,
        "nn.functional.max_pool3d": ALL_MODES,  # Aborted
        "view_as_complex": ALL_MODES,  # Aborted
    },
    "untriaged: unsupported view op or reshape": {
        "expand": ALL_MODES,  # view op - expand not yet supported
        "masked_scatter": ALL_MODES,  # unflatten ambiguous error
        "matmul": ALL_MODES,  # reshapes in the structured delegate
        "slice": ALL_MODES,  # reshape reassociation not supported
        # enhance view op to detect squeeze
        "squeeze": ALL_MODES,
        "squeeze_copy": ALL_MODES,
        "var_mean": {COMPILE},  # reshape lowering failure in compile mode
    },
    "untriaged: shape mismatch": {
        "ldexp": ALL_MODES,  # dynamic shape value permutation mismatch
        "nn.functional.mse_loss": ALL_MODES,  # binop LHS / RHS mismatch
        "std": {COMPILE},  # shape mismatch on out under dynamic compile
    },
    "untriaged: data-dependent output shape (invalid TPU shape)": {
        "unique": ALL_MODES,
        "unique_consecutive": ALL_MODES,
    },
    # TODO(b/488338235): dynamism is flaky.
    "untriaged: flaky": {
        "_foreach_addcdiv": ALL_MODES,
    },
    "untriaged: unknown failure": {
        "_native_multi_head_attention": ALL_MODES,
        "_thnn_fused_gru_cell": ALL_MODES,
        "_thnn_fused_lstm_cell": ALL_MODES,
        "_transformer_encoder_layer_fwd": ALL_MODES,
        "angle": {COMPILE},  # Dynamo error
        "floor_divide": ALL_MODES,
        # Eager mode is triaged under "the lowering broadcasts a bounded input
        # to a static shape".
        "linalg.svd": {COMPILE},
        "linalg.svdvals": {COMPILE},
        "nn.functional.adaptive_avg_pool2d": (
            ALL_MODES
        ),  # sort op fails to infer
        "nn.functional.batch_norm": ALL_MODES,  # xla error
        "nn.functional.grid_sample": ALL_MODES,
        "normal": ALL_MODES,
        "split_with_sizes_copy": ALL_MODES,  # b/483055194
        "sum": ALL_MODES,  # return i64, which has lowering issues
        "torch._scaled_grouped_mm_v2": ALL_MODES,
        "view_as_real": ALL_MODES,
    },
}

# Mode-aware lists of ops for which bounded dynamism is out of scope by design,
# keyed by the reason. The reason is included in the skip message.
#
# Policy: unlike `DYNAMISM_DENY_LISTS`, these entries are not bugs and are not
# expected to be removed. Each entry requires a justification of why dynamism
# does not apply to the op, so this should rarely change.
DYNAMISM_NOT_APPLICABLE_OPS: DenyList = {
    "fused optimizers update fixed-shape parameters and optimizer states in place": {
        "_fused_adagrad_": ALL_MODES,
        "_fused_adagrad_.tensor_lr": ALL_MODES,
        "_fused_adam_": ALL_MODES,
        "_fused_adam_.tensor_lr": ALL_MODES,
        "_fused_adamw_": ALL_MODES,
        "_fused_adamw_.tensor_lr": ALL_MODES,
        "_fused_sgd_": ALL_MODES,
        "_fused_sgd_.tensor_lr": ALL_MODES,
    },
}


def _find_reason(
    deny_lists: DenyList, op_name: str, mode: DynamismMode
) -> str | None:
  """Returns the first reason in `deny_lists` that excludes `op_name` in `mode`."""
  for reason, ops in deny_lists.items():
    if mode in ops.get(op_name, ()):
      return reason
  return None


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
  reason = _find_reason(DYNAMISM_NOT_APPLICABLE_OPS, op.name, mode)
  if reason is not None:
    return (
        f"Dynamism is not applicable to {op.name} in {mode.value} mode:"
        f" {reason}."
    )
  reason = _find_reason(DYNAMISM_DENY_LISTS, op.name, mode)
  if reason is not None:
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
  _, canonical_input = _get_canonical_input_value(op, input_value)
  skip_input = _should_skip_input_marking(op, canonical_input)
  if skip_input:
    return skip_input
  if not DynamicOpInfo.get(op).get_candidate_dims(input_value, args):
    return "No valid input dimensions to mark dynamic."

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

  def get_candidate_dims(
      self,
      input_value: torch.Tensor | Sequence[torch.Tensor],
      args: Sequence[Any],
  ) -> Sequence[TensorDim]:
    """Returns candidate `TensorDim`s on the canonical input tensor.

    Note that this method only discovers candidate dimensions on a single
    primary tensor, not independently across every tensor in a sequence
    or in `args`.

    Args:
      input_value: First positional argument passed to the op — either a single
        `Tensor` or, for ops whose first parameter is a tensor list (such as
        foreach ops or `einsum`), a `Sequence[Tensor]`. When `input_value` is a
        sequence, this base implementation selects the first dynamism-eligible
        tensor via `_get_canonical_input_value`.
      args: Remaining positional op arguments, including any secondary operand
        tensors and non-tensor parameters.

    Returns:
      Strictly ascending `TensorDim` candidates (each with dimension size >= 2)
      identifying both the selected tensor and its 0-based dimension index.
    """
    tensor_index, canonical_input = _get_canonical_input_value(
        self.op_info, input_value
    )
    return [
        TensorDim(tensor_index=tensor_index, dim=dim)
        for dim in self.dynamic_dimension_candidates(canonical_input, args)
    ]

  def generate_exhaustive_marking_plans(
      self,
      input_value: torch.Tensor | Sequence[torch.Tensor],
      args: Sequence[Any],
      mode: DynamismMode = DynamismMode.EAGER,
  ) -> Sequence[DynamicMarkingPlan]:
    """Generates all valid dynamic marking plans for the canonical input tensor.

    Plans record both the selected tensor index and 0-based dimension indices on
    that tensor. Dimensions on secondary tensors or subsequent tensors in a
    sequence are not swept as separate plans yet.

    Args:
      input_value: First positional argument passed to the op.
      args: Remaining positional arguments passed to the op.
      mode: Dynamism execution mode (`EAGER` or `COMPILE`). Simultaneous
        multi-dimension plans are only emitted in `COMPILE` mode.

    Returns:
      A sequence of `DynamicMarkingPlan` instances targeting the canonical
      primary input tensor: one single-dimension plan (`"t{i}d{d}"`) per
      `TensorDim` returned by `get_candidate_dims`, plus a combined
      multi-dimension plan (`"t{i}d{d0}_t{i}d{d1}..."`) when
      `mode == DynamismMode.COMPILE` and at least two candidate dimensions
      exist. Returns `()` if no dimensions on the canonical input tensor can be
      marked dynamic.
    """
    candidates = list(self.get_candidate_dims(input_value, args))
    if not candidates:
      return ()
    plans = [
        DynamicMarkingPlan(
            label=_format_plan_label((candidate,)),
            tensor_dims=(candidate,),
        )
        for candidate in candidates
    ]
    # Eager `dynamism.mark_dynamic` only supports one dynamic dimension per
    # tensor, so simultaneous multi-dimension plans are restricted to COMPILE.
    if mode == DynamismMode.COMPILE and len(candidates) >= 2:
      dims_tuple = tuple(candidates)
      plans.append(
          DynamicMarkingPlan(
              label=_format_plan_label(dims_tuple),
              tensor_dims=dims_tuple,
          )
      )
    return tuple(plans)

  def _mark_single_dim(
      self,
      target: TensorDim,
      input_value: torch.Tensor | Sequence[torch.Tensor],
      args: Sequence[Any],
      mode: DynamismMode = DynamismMode.EAGER,
      kwargs: Mapping[str, Any] | None = None,
  ) -> None:
    """Marks `target` in `input_value` and coupled dimensions on dependent args.

    Args:
      target: `TensorDim` identifying which tensor in `input_value` and which
        0-based dimension on that tensor to mark dynamic.
      input_value: Primary tensor or sequence of tensors passed to the op.
      args: Additional positional arguments whose corresponding dimensions
        should also be marked dynamic.
      mode: Dynamism execution mode (`EAGER` or `COMPILE`).
      kwargs: Optional keyword arguments passed to the op (used by adapters such
        as `SdpaDynamicOpInfo` to mark keyword tensor arguments).
    """
    target_tensor = _resolve_input_tensor(input_value, target.tensor_index)
    idx = target.dim
    input_value_str = utils.get_tensor_summary(target_tensor, data=False)
    ub = target_tensor.shape[idx] + 10
    print(
        f">>>> Marking dim {idx} dynamic [<={ub}] in {input_value_str}"
        f" (mode={mode.value})",
        flush=True,
    )
    _mark_tensor_dynamic(target_tensor, idx, 2, ub, mode=mode)
    self._mark_dependent_arg_dynamic(
        args, target_tensor, idx, ub, mode=mode, kwargs=kwargs
    )

  def mark_dynamic(
      self,
      input_value: torch.Tensor | Sequence[torch.Tensor],
      args: Sequence[Any],
      mode: DynamismMode = DynamismMode.EAGER,
      kwargs: Mapping[str, Any] | None = None,
      plan: DynamicMarkingPlan | None = None,
  ) -> None:
    """Marks the op dynamic per `plan`, or on the first candidate dim if `None`.

    Args:
      input_value: Primary tensor or sequence of tensors passed to the op.
      args: Additional positional arguments passed to the op.
      mode: Dynamism execution mode (`EAGER` or `COMPILE`).
      kwargs: Optional keyword arguments passed to the op.
      plan: Explicit `DynamicMarkingPlan` specifying which dimensions to mark
        dynamic. When `None`, marks the first candidate `TensorDim`.

    Raises:
      ValueError: If `plan is None` and no candidate dynamic dimensions exist.
    """
    if plan is None:
      candidates = list(self.get_candidate_dims(input_value, args))
      if not candidates:
        raise ValueError(
            f"No dynamic marking plans available for {self.op_info.name}."
        )
      chosen = candidates[0]
      plan = DynamicMarkingPlan(
          label=_format_plan_label((chosen,)),
          tensor_dims=(chosen,),
      )
    logging.debug(
        "[DynamicOpInfo] mark_dynamic(%s, plan=%s, %s, %s, mode=%s)",
        self.op_info.name,
        plan.label,
        utils.InputMetadata(input_value),
        utils.InputMetadata(args),
        mode,
    )
    for target in plan.tensor_dims:
      self._mark_single_dim(target, input_value, args, mode=mode, kwargs=kwargs)

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
        "__lshift__",
        "__rshift__",
        "_log_softmax_backward_data",
        "_masked_softmax",
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
      "_masked_softmax_backward",
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

  def generate_exhaustive_marking_plans(
      self,
      input_value: torch.Tensor | Sequence[torch.Tensor],
      args: Sequence[Any],
      mode: DynamismMode = DynamismMode.EAGER,
  ) -> Sequence[DynamicMarkingPlan]:
    """Overrides `DynamicOpInfo.generate_exhaustive_marking_plans` for foreach.

    Foreach ops operate on heterogeneous tensor lists and delegate per-element
    dimension selection to `ForEachDynamicOpInfo.mark_dynamic`, so this override
    emits at most one `"foreach"` sentinel plan while keeping the base class
    method name and `Sequence[DynamicMarkingPlan]` return type for polymorphic
    dispatch via `DynamicOpInfo.get(op_info)`.

    Args:
      input_value: Sequence of primary input tensors passed to the foreach op.
      args: Additional positional arguments (scalars or sequences of tensors).
      mode: Dynamism execution mode (`EAGER` or `COMPILE`), accepted to match
        the `DynamicOpInfo` base signature.

    Returns:
      `(DynamicMarkingPlan(label="foreach"),)` if `input_value` has at least one
      candidate dynamic dimension, or `()` otherwise.
    """
    del mode  # Unused; accepted to match DynamicOpInfo base signature.
    if not self.get_candidate_dims(input_value, args):
      return ()
    return (DynamicMarkingPlan(label="foreach"),)

  def mark_dynamic(
      self,
      input_value: torch.Tensor | Sequence[torch.Tensor],
      args: Sequence[Any],
      mode: DynamismMode = DynamismMode.EAGER,
      kwargs: Mapping[str, Any] | None = None,
      plan: DynamicMarkingPlan | None = None,
  ):
    """Overrides `DynamicOpInfo.mark_dynamic` for foreach tensor-list inputs.

    Args:
      input_value: Sequence of primary input tensors passed to the foreach op.
      args: Additional positional arguments (scalar or per-element tensor
        sequences).
      mode: Dynamism execution mode (`EAGER` or `COMPILE`).
      kwargs: Optional keyword arguments, accepted to match the `DynamicOpInfo`
        base signature for polymorphic dispatch via `DynamicOpInfo.get`.
      plan: Optional `DynamicMarkingPlan` (e.g., the `"foreach"` sentinel plan),
        accepted to match the `DynamicOpInfo` base signature.
    """
    del kwargs, plan  # Unused; accepted to match DynamicOpInfo base signature.
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
        super().mark_dynamic(value, [], mode=mode)
        continue

      # Binary ops expect len(args) == 1
      if self.op_info.name in foreach_binops:
        rhs_vals = args[0]
        BinaryElementwiseDynamicOpInfo(self.op_info).mark_dynamic(
            value, (rhs_vals[idx],), mode=mode
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
            value,
            (rhs_vals[idx], ehs_vals[idx]),  # pyrefly: ignore[unsupported-operation]
            mode=mode,
        )
        continue

      # Not binary or ternary, no dependent args.
      super().mark_dynamic(value, [], mode=mode)
      logging.debug(
          "[ForEachDynamicOpInfo] no dependent args %s",
          utils.InputMetadata(args),
      )


class MatmulDynamicOpInfo(DynamicOpInfo):
  """A wrapper around OpInfo that supports bounded dynamism."""

  @staticmethod
  def adapter_supports_op(op_info: core.OpInfo):
    """Method to declare ops supported by this adapter."""

    matmul_ops = ["bmm", "bmm.dtype", "mm", "mm.dtype"]
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

  def get_candidate_dims(
      self,
      input_value: torch.Tensor | Sequence[torch.Tensor],
      args: Sequence[Any],
  ) -> Sequence[TensorDim]:
    """Returns candidate `TensorDim`s on `einsum`'s first operand.

    Overrides `DynamicOpInfo.get_candidate_dims` to inspect the full operand
    sequence when checking Einstein-label constraints instead of collapsing
    `input_value` via `_get_canonical_input_value`.

    Args:
      input_value: Single operand tensor or sequence of operand tensors passed
        to `torch.einsum`.
      args: Positional arguments where `args[0]` is the einsum equation string.

    Returns:
      Strictly ascending `TensorDim` candidates on the first operand
      (`tensor_index=0`) that can be safely marked dynamic along with matching
      label dimensions on subsequent operands.
    """
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
      return ()

    operand_labels, rhs_labels = self._parse_einsum_labels(tensors, args[0])
    if not operand_labels or len(operand_labels[0]) != tensors[0].ndim:
      return ()

    any_requires_grad = any(t.requires_grad for t in tensors)
    if any_requires_grad:
      # Repeated labels within an operand use diagonal_scatter into a static
      # zeros tensor in backward, and uncontracted single-operand dimensions
      # summed out in forward use static expand in backward.
      if any(len(set(lbls)) != len(lbls) for lbls in operand_labels):
        return ()
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
            return ()

    first_tensor = tensors[0]
    first_labels = operand_labels[0]
    candidates: list[TensorDim] = []
    for d, sz in enumerate(first_tensor.shape):
      if sz <= 1:
        continue
      lbl = first_labels[d]
      # Repeated labels with size > 1 use stablehlo.iota in forward.
      if any(lbls.count(lbl) > 1 for lbls in operand_labels):
        continue
      candidates.append(TensorDim(tensor_index=0, dim=d))
    return candidates

  def _mark_single_dim(
      self,
      target: TensorDim,
      input_value: torch.Tensor | Sequence[torch.Tensor],
      args: Sequence[Any],
      mode: DynamismMode = DynamismMode.EAGER,
      kwargs: Mapping[str, Any] | None = None,
  ) -> None:
    """Marks `target` and matching Einstein-label dims on other operands.

    Args:
      target: `TensorDim` identifying which einsum operand in `input_value` and
        which 0-based dimension on that operand to mark dynamic.
      input_value: Single operand tensor or sequence of operand tensors passed
        to `torch.einsum`.
      args: Positional arguments where `args[0]` is the einsum equation string.
      mode: Dynamism execution mode (`EAGER` or `COMPILE`).
      kwargs: Optional keyword arguments, accepted to match the
        `DynamicOpInfo._mark_single_dim` signature.
    """
    del kwargs
    target_tensor = _resolve_input_tensor(input_value, target.tensor_index)
    target_tensor_idx = target.tensor_index
    idx = target.dim
    tensors: list[torch.Tensor] = (
        [input_value]
        if isinstance(input_value, torch.Tensor)
        else list(input_value)
    )
    ub = target_tensor.shape[idx] + 10
    target_tensor_str = utils.get_tensor_summary(target_tensor, data=False)
    print(
        f">>>> Marking dim {idx} dynamic [<={ub}] in {target_tensor_str}"
        f" (mode={mode.value})",
        flush=True,
    )
    _mark_tensor_dynamic(target_tensor, idx, 2, ub, mode=mode)

    operand_labels, _ = self._parse_einsum_labels(tensors, args[0])
    target_label = operand_labels[target_tensor_idx][idx]
    for k, (t_k, lbls_k) in enumerate(
        zip(tensors, operand_labels, strict=True)
    ):
      if k == target_tensor_idx:
        continue
      if target_label in lbls_k:
        d_k = lbls_k.index(target_label)
        if t_k.shape[d_k] == target_tensor.shape[idx] and t_k.shape[d_k] > 1:
          print(
              f">>>> Marking dependent arg dim {d_k} dynamic [<={ub}] in"
              f" {utils.InputMetadata(t_k)} (mode={mode.value})",
              flush=True,
          )
          _mark_tensor_dynamic(t_k, d_k, 2, ub, mode=mode)


def generate_exhaustive_input_marking_plans(
    op_info: core.OpInfo,
    input_value: torch.Tensor | Sequence[torch.Tensor],
    args: Sequence[Any],
    mode: DynamismMode = DynamismMode.EAGER,
) -> Sequence[DynamicMarkingPlan]:
  """Generates all valid dynamic marking plans for an op sample.

  Args:
    op_info: The `OpInfo` descriptor for the operator under test.
    input_value: Primary tensor or sequence of tensors passed to the op.
    args: Additional positional arguments passed to the op.
    mode: Dynamism execution mode (`EAGER` or `COMPILE`).

  Returns:
    A sequence of `DynamicMarkingPlan` instances for the sample, or `()` if no
    dimensions can be marked dynamic.
  """
  return DynamicOpInfo.get(op_info).generate_exhaustive_marking_plans(
      input_value,
      args,
      mode=mode,
  )


def mark_input_dynamic(
    op_info: core.OpInfo,
    input_value: torch.Tensor | Sequence[torch.Tensor],
    args: Sequence[Any],
    mode: DynamismMode = DynamismMode.EAGER,
    kwargs: Mapping[str, Any] | None = None,
    plan: DynamicMarkingPlan | None = None,
) -> None:
  """Mark an arg tensor as dynamic.

  Args:
    op_info: The op to be tested.
    input_value: The primary input value to for the op test.
    args: The additional args to the op test.
    mode: The dynamism mode (EAGER or COMPILE).
    kwargs: The keyword arguments to the op test.
    plan: Optional deterministic marking plan to apply. When `None`, marks the
      first candidate dimension.
  """

  dynamic_op_info = DynamicOpInfo.get(op_info)
  plan_suffix = f" (plan={plan.label})" if plan is not None else ""
  print(
      f">>> Marking input dynamic{plan_suffix} using"
      f" {type(dynamic_op_info)} (mode={mode.value})"
  )
  dynamic_op_info.mark_dynamic(
      input_value, args, mode=mode, kwargs=kwargs, plan=plan
  )
