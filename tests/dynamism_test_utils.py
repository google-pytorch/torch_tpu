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

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import enum
import itertools
from typing import Any

from absl import logging
import sympy
import torch
from torch._dynamo import source as dynamo_source
from torch._subclasses import fake_tensor
from torch.fx.experimental import symbolic_shapes
from torch.testing._internal import common_methods_invocations
from torch.testing._internal.opinfo import core
from torch.utils import _pytree
from torch.utils._sympy import numbers as sympy_numbers
from torch_tpu._internal import dynamism
from torch_tpu._internal.compile.dynamic import symbol_bounds
from torch_tpu._internal.utils import utils


class DynamismMode(enum.Enum):
  """Mode of dynamism testing."""

  EAGER = "eager"  # Dynamic backward is not supported in eager mode.
  COMPILE = "compile"
  COMPILE_GRAD = "compile_grad"


EAGER = DynamismMode.EAGER
COMPILE = DynamismMode.COMPILE
COMPILE_GRAD = DynamismMode.COMPILE_GRAD
ALL_MODES = frozenset(DynamismMode)


@dataclasses.dataclass(frozen=True, order=True)
class TensorDim:
  """Identifies a 0-based dimension on an input tensor.

  Attributes:
    tensor_index: Non-negative 0-based index of the tensor in
      `_collect_tensors`.
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
      dynamic on the primary input.
    coupled_dims: Validated dimensions across all input tensors to mark dynamic
      together when executing this plan. Empty when adapter-specific marking is
      used (e.g., foreach ops).
  """

  label: str
  tensor_dims: tuple[TensorDim, ...] = ()
  coupled_dims: tuple[TensorDim, ...] = ()

  def __post_init__(self) -> None:
    if any(td.tensor_index < 0 or td.dim < 0 for td in self.tensor_dims) or any(
        a >= b for a, b in itertools.pairwise(self.tensor_dims)
    ):
      raise ValueError(
          "tensor_dims must be non-negative and strictly ascending without"
          f" duplicates, got {self.tensor_dims}."
      )


@dataclasses.dataclass(frozen=True)
class _OpArgs:
  """Bundles an operator's inputs."""

  input_value: Any
  args: tuple[Any, ...] = ()
  kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def _normalize_kwargs(
    kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
  """Returns `kwargs` sorted by key without `out` for deterministic traversal."""
  if not kwargs:
    return {}
  return {k: kwargs[k] for k in sorted(kwargs) if k != "out"}


def _collect_tensors(op_args: _OpArgs) -> list[torch.Tensor]:
  """Returns all `torch.Tensor`s in `op_args` (excluding `out` in `kwargs`)."""
  norm_kwargs = _normalize_kwargs(op_args.kwargs)
  return [
      x
      for x in _pytree.tree_leaves(
          (op_args.input_value, op_args.args, norm_kwargs)
      )
      if isinstance(x, torch.Tensor)
  ]


# Exceptions indicating `FakeTensorMode` cannot trace the op at all.
_UNTRACEABLE_FAKE_OP_EXCEPTIONS = (
    fake_tensor.DataDependentOutputException,
    fake_tensor.UnsupportedFakeTensorException,
    symbolic_shapes.GuardOnDataDependentSymNode,
    NotImplementedError,
)


def _make_fake_inputs(
    fake_mode: fake_tensor.FakeTensorMode,
    op_args: _OpArgs,
    *,
    mode: DynamismMode,
    dynamic_dims: set[TensorDim],
) -> tuple[_OpArgs, dict[TensorDim, sympy.Basic]]:
  """Converts `op_args` to symbolic `FakeTensor`s.

  Args:
    fake_mode: Active `FakeTensorMode` bound to a `ShapeEnv`.
    op_args: Operator inputs.
    mode: Dynamism execution mode.
    dynamic_dims: Set of `TensorDim`s to mark dynamic.

  Returns:
    A `(fake_op_args, initial_syms)` pair with the converted `_OpArgs` and a
    map from each symbolic `TensorDim` to its initial symbol.
  """
  initial_syms: dict[TensorDim, sympy.Basic] = {}
  tensor_indices = itertools.count()

  def to_fake(obj: Any) -> Any:
    if not isinstance(obj, torch.Tensor):
      return obj
    tensor_idx = next(tensor_indices)
    # Match `op_testing.to(..., "tpu")`, which materializes each tensor into an
    # independent non-view copy while preserving dense non-overlapping strides.
    detached = obj.detach().to(device=obj.device, copy=True)

    # Mark requested dimensions with size >= 2 as dynamic.
    dyn_sizes = [
        symbolic_shapes.DimDynamic.DYNAMIC
        if sz >= 2 and TensorDim(tensor_idx, d) in dynamic_dims
        else symbolic_shapes.DimDynamic.STATIC
        for d, sz in enumerate(detached.shape)
    ]
    fake = fake_mode.from_tensor(
        detached,
        source=dynamo_source.LocalSource(f"t{tensor_idx}"),
        symbolic_context=symbolic_shapes.StatelessSymbolicContext(
            dynamic_sizes=dyn_sizes
        ),
    )
    if obj.requires_grad:
      fake.requires_grad_(True)

    # Record the initial symbol allocated for each dynamic dimension.
    for d, s in enumerate(fake.size()):
      if isinstance(s, torch.SymInt):
        initial_syms[TensorDim(tensor_idx, d)] = s.node.expr
    return fake

  # Traverse inputs in the exact order used by `_collect_tensors`.
  fake_input, fake_args, fake_kwargs = _pytree.tree_map(
      to_fake,
      (op_args.input_value, op_args.args, _normalize_kwargs(op_args.kwargs)),
  )

  # Enable gradients on differentiable primary inputs in `COMPILE_GRAD`.
  if mode == DynamismMode.COMPILE_GRAD:
    for t in _pytree.tree_leaves(fake_input):
      if (
          isinstance(t, torch.Tensor)
          and (t.is_floating_point() or t.is_complex())
          and t.numel() > 0
      ):
        t.requires_grad_(True)

  # Convert `out` buffers separately as static FakeTensors so `tensor_indices`
  # stays aligned with `_collect_tensors` (which excludes `out`).
  if "out" in op_args.kwargs:
    fake_kwargs["out"] = _pytree.tree_map(
        lambda x: fake_mode.from_tensor(
            x.detach().to(device=x.device, copy=True), static_shapes=True
        )
        if isinstance(x, torch.Tensor)
        else x,
        op_args.kwargs["out"],
    )
  return _OpArgs(fake_input, fake_args, fake_kwargs), initial_syms


def _run_fake_backward(result: Any, fake_input: Any) -> None:
  """Runs backward from differentiable `result` tensors to `fake_input`."""
  grad_inputs = [
      t
      for t in _pytree.tree_leaves(fake_input)
      if isinstance(t, torch.Tensor) and t.requires_grad
  ]
  outputs = [
      t
      for t in _pytree.tree_leaves(result)
      if isinstance(t, torch.Tensor) and t.requires_grad
  ]
  if grad_inputs and outputs:
    torch.autograd.grad(
        outputs,
        grad_inputs,
        grad_outputs=[torch.ones_like(t) for t in outputs],
        allow_unused=True,
    )


def _run_fake_op(
    op_func: Callable[..., Any],
    op_args: _OpArgs,
    *,
    mode: DynamismMode,
    dynamic_dims: set[TensorDim],
) -> tuple[symbolic_shapes.ShapeEnv, dict[TensorDim, sympy.Basic]]:
  """Traces `op_func` (and backward in `COMPILE_GRAD`) on `FakeTensor`s.

  Args:
    op_func: Callable for the active op variant.
    op_args: Operator inputs (`input_value`, `args`, `kwargs`).
    mode: Dynamism execution mode (`EAGER`, `COMPILE`, or `COMPILE_GRAD`).
    dynamic_dims: Set of `TensorDim`s to mark dynamic.

  Returns:
    A `(shape_env, initial_syms)` pair with the populated `ShapeEnv` and a map
    from each symbolic `TensorDim` to its initial symbol.
  """
  # Disable duck-shaping so equal-sized dims start with distinct symbols.
  shape_env = symbolic_shapes.ShapeEnv(duck_shape=False)
  fake_mode = fake_tensor.FakeTensorMode(
      shape_env=shape_env, allow_non_fake_inputs=True
  )

  # Convert inputs to FakeTensors with symbols on `dynamic_dims`.
  fake_op_args, initial_syms = _make_fake_inputs(
      fake_mode,
      op_args,
      mode=mode,
      dynamic_dims=dynamic_dims,
  )

  # Run the op symbolically to record unifications and guards on `shape_env`.
  with fake_mode:
    result = op_func(
        fake_op_args.input_value, *fake_op_args.args, **fake_op_args.kwargs
    )
    if mode in (DynamismMode.COMPILE, DynamismMode.COMPILE_GRAD):
      # Mirror AOTAutograd's output stride/contiguity guards.
      for out_t in _pytree.tree_leaves(
          (result, fake_op_args.kwargs.get("out"))
      ):
        if isinstance(out_t, torch.Tensor):
          torch._prims_common.suggest_memory_format(out_t)  # pylint: disable=protected-access
    if mode == DynamismMode.COMPILE_GRAD:
      # Also trace backward to capture autograd shape constraints.
      _run_fake_backward(result, fake_op_args.input_value)

  return shape_env, initial_syms


def _satisfies_upper_bound_guards(
    shape_env: symbolic_shapes.ShapeEnv,
    canonical_syms: set[sympy.Symbol],
    mode: DynamismMode,
) -> bool:
  """Returns True if mode-specific upper bounds satisfy `shape_env` guards."""
  if mode == DynamismMode.EAGER:
    eager_ub: dict[str, int] = {}
    for sym in canonical_syms:
      sz = int(shape_env.backed_var_to_val[sym])
      vr_upper = shape_env.var_to_range[sym].upper
      # Reject if eager's `sz + 10` bound exceeds the symbol's max range.
      if not sympy_numbers.is_infinite(vr_upper) and sz + 10 > int(vr_upper):
        return False
      eager_ub[str(sym)] = sz + 10
    return symbol_bounds.find_violated_guard(shape_env, eager_ub) is None

  bucket_ub: dict[str, int] = {}
  fallback_ub: dict[str, int] = {}
  for sym in canonical_syms:
    sz = int(shape_env.backed_var_to_val[sym])
    vr_upper = shape_env.var_to_range[sym].upper
    finite_ub = None if sympy_numbers.is_infinite(vr_upper) else int(vr_upper)
    bucket_ub[str(sym)] = (
        finite_ub if finite_ub is not None else -(-sz // 128) * 128
    )
    fallback_ub[str(sym)] = (
        finite_ub
        if finite_ub is not None
        else symbol_bounds.get_fallback_upper_bound(sz)
    )
  return (
      symbol_bounds.find_violated_guard(shape_env, bucket_ub) is None
      or symbol_bounds.find_violated_guard(shape_env, fallback_ub) is None
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
  elif mode in (DynamismMode.COMPILE, DynamismMode.COMPILE_GRAD):
    torch._dynamo.mark_dynamic(tensor, dimension)
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


def _check_tensor_supported_for_dynamism(
    op_info: core.OpInfo, tensor: torch.Tensor
) -> None | str:
  """Verifies that `tensor` has no dtype or shape hazards for dynamism.

  Args:
    op_info: The op to be tested.
    tensor: An input tensor passed to the op (either primary or secondary).

  Returns:
    None if `tensor` is safe for dynamism testing, or a skip reason string.
  """
  dtype = tensor.dtype
  # TODO: b/449736443 - [XLA] Crash in complex rewriter.
  if dtype in (torch.complex64, torch.complex128):
    return f"{op_info.name} op is not supported with dtype {dtype}."
  # _foreach_exp_ does not support integral dtypes on TPU.
  if op_info.name == "_foreach_exp_" and not dtype.is_floating_point:
    return f"{op_info.name} does not support integral dtypes."
  # ceil/floor do not support int64 under dynamism (sign-extension bugs).
  if (
      op_info.name in ("ceil", "floor", "_foreach_ceil", "_foreach_floor")
      and dtype == torch.int64
  ):
    return f"{op_info.name} does not support int64 under dynamism."
  # TODO: b/449736443 - Empty tensor handling needs reworking.
  if tensor.numel() == 0:
    return "Empty tensors are currently not supported."

  return None


def _should_skip_input_marking(
    op_info: core.OpInfo, input_value: torch.Tensor | Any
) -> None | str:
  """Verifies that the input_value is a valid primary input for dynamism.

  Args:
    op_info: The op to be tested.
    input_value: The primary input value for the op test.

  Returns:
    None if the op is safe to run, or a string containing the reason why
    the test is not safe to run.
  """

  if not isinstance(input_value, torch.Tensor):
    # Some ops take scalar inputs
    # empty, empty_strided, full, tril, ones, zeros, stack, tril_indices.
    return f"Input value is not a tensor, got {type(input_value)}."
  candidates = _get_dynamic_dimension_candidates(input_value)
  if not candidates:
    return "No valid input dimensions to mark dynamic."
  return _check_tensor_supported_for_dynamism(op_info, input_value)


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
        "_segment_reduce": {EAGER, COMPILE},
        "_weight_norm_interface_backward": {EAGER, COMPILE},
        "atanh": {COMPILE_GRAD},
        "bernoulli": {EAGER},
        "cdist": {EAGER},
        # These abort the test process with a CHECK failure.
        "linalg.lu_solve": {EAGER, COMPILE_GRAD},
        "linalg.qr": {EAGER, COMPILE_GRAD},
        "linalg.svd": {EAGER},
        "linalg.svdvals": {EAGER},
    },
    # TODO: b/449736443 - Support bounded dimensions in these lowerings.
    "the lowering emits an op without bounded shape support": {
        # Backward broadcast fails with mixed bounded and static dimensions.
        "_foreach_lerp": {COMPILE_GRAD},
        # BuildMlirModule fails to inline MLIR functions on bounded inputs.
        "_fused_moving_avg_obs_fq_helper": ALL_MODES,
        # XLA ragged-dot does not support propagating bounded dimensions.
        "_grouped_mm": {EAGER, COMPILE},
        # XLA Scatter updates bound mismatches static scatter_indices bound.
        "_jagged_to_padded_dense_forward": {EAGER},
        # _upsample_bicubic2d_aa and _upsample_bilinear2d_aa lower via 1D gathers with static output sizes.
        "_upsample_bicubic2d_aa": ALL_MODES,
        "_upsample_bilinear2d_aa": ALL_MODES,
        # Calls getNumElements() in BroadcastInDimOp_ReplaceWithReshape.
        "_weight_norm_interface": {COMPILE, COMPILE_GRAD},
        # stablehlo.broadcast_in_dim requires static or single-bounded shape.
        "constant_pad_nd": ALL_MODES,
        # cummax_helper and cummin_helper emit static stablehlo.iota and query
        # static dim sizes (-9223372036854775808).
        "cummax": ALL_MODES,
        "cummin": ALL_MODES,
        # BuildCumprodShlo passes INT64_MIN shapes into BroadcastIfNeeded
        # (eager), and chlo.ScanOp fails MLIR function inlining (compile).
        "cumprod": ALL_MODES,
        # cumsum.out hits a TypedAttr cast assertion (eager), and chlo.ScanOp
        # fails MLIR function inlining (compile).
        "cumsum": ALL_MODES,
        # Calls getDotResultType assertion on bounded 1D operands.
        "dot": ALL_MODES,
        # embedding_renorm_ uses static BroadcastInDim and Scatter shapes.
        "embedding_renorm_": ALL_MODES,
        # Checks index.size(d) <= input.size(d) against INT64_MIN.
        "gather": ALL_MODES,
        # Compiled backward loses grad_fn when hidden size is dynamic.
        "gru.input": {COMPILE_GRAD},
        # stablehlo.broadcast_in_dim allows at most one bounded dimension, which
        # fails when multiple dimensions are marked dynamic in compile mode.
        "lcm": {COMPILE, COMPILE_GRAD},
        # chlo.ScanOp fails MLIR function inlining.
        "logcumsumexp": ALL_MODES,
        # SliceOp::build fails return type inference on bounded hidden slices.
        "lstm.input": ALL_MODES,
        # Slicing a dynamic dimension is not supported in lu_unpack lowering.
        "lu_unpack": ALL_MODES,
        # Queries static dim size (slice size -9223372036854775808).
        "nn.functional.adaptive_max_pool2d": ALL_MODES,
        # avg_pool2d_backward SliceBackwardResult queries static input shape;
        # avg_pool1d compile is under "compiled as_strided view lowering fails
        # to inline MLIR functions".
        "nn.functional.avg_pool1d": {COMPILE_GRAD},
        "nn.functional.avg_pool2d": {COMPILE_GRAD},
        # PropagateBoundedDynamism fails MarkDynamic when pooling reduces a
        # bounded spatial dimension to size 1 (eager), dynamic_pad_module
        # exceeds the 3 GiB shape bound on 5D <=128 inputs (compile), and
        # avg_pool3d_backward queries static input shape (compile_grad).
        "nn.functional.avg_pool3d": ALL_MODES,
        # cross_entropy_loss lowers via gather/scatter and static element count
        # queries, and requires synchronized target/weight dynamic marking.
        "nn.functional.cross_entropy": ALL_MODES,
        # getNumElements() overflows on INT64_MIN.
        "nn.functional.ctc_loss": {EAGER},
        # Queries static split dim size (INT64_MIN / 2).
        "nn.functional.glu": {EAGER},
        # var_mean reshape reassociation fails (eager) and local_scalar_dense
        # cannot materialize a placeholder tensor (compile).
        "nn.functional.group_norm": ALL_MODES,
        # reflection_pad*d and replication_pad*d query static input shape.
        "nn.functional.pad": ALL_MODES,
        # Calls getNumElements() on a dynamic shaped type.
        "norm": {EAGER, COMPILE},
        # Queries static input shape (limit index -9223372036854775808).
        "scatter": {EAGER, COMPILE},
        # AtenSelectScatter checks static src shape against padded self slice.
        "select_scatter": {COMPILE, COMPILE_GRAD},
        "sort": {EAGER, COMPILE_GRAD},  # stablehlo.iota must be static
        # Lowers via sort with %iota, which lacks XLA dynamic_padder support.
        "topk": {EAGER},
        # Calls getNumElements() and static scale checks on bounded operands.
        "torch._scaled_mm": {EAGER, COMPILE},
        "torch._scaled_mm_v2": {EAGER, COMPILE},
        # stablehlo.reshape allows at most one bounded dimension. This fails
        # when dims 0 and 2 are both dynamic, which happens in compile mode
        # because dims marked in earlier subtests stay dynamic.
        # TODO: b/566327707 - Remove once marked dims no longer carry over.
        "torch.ops.aten._safe_softmax.default": {COMPILE, COMPILE_GRAD},
    },
    # TODO: b/449736443 - Avoid Tensor.resize_() graph break in _addmm_dtype_wrapper.
    "the test wrapper calls Tensor.resize_(), which graph-breaks in Dynamo": {
        "addmm.dtype": {COMPILE},
    },
    # TODO: b/449736443 - Support bounded dimensions in these lowerings.
    "linalg decompositions do not support bounded dimensions": {
        "geqrf": {EAGER},  # Aborts the test process with a CHECK failure.
        "linalg.cholesky": {EAGER},
        "linalg.cholesky_ex": {EAGER, COMPILE_GRAD},
        "linalg.inv": {EAGER, COMPILE_GRAD},
        "linalg.lu": {EAGER, COMPILE_GRAD},
        "linalg.solve_ex": {EAGER, COMPILE_GRAD},
    },
    # TODO: b/449736443 - Support bounded dimensions in these lowerings.
    "the result has wrong values": {
        # TODO: b/566327914 - With dim 0 bounded, the result matches the
        # transposed golden output. Compile mode is triaged under
        # "XLA fails layout verification with 'layout does not match any allowed
        # tile kind'".
        "nn.functional.hardtanh": {EAGER},
    },
    # Fix MLIR function inlining in compiled as_strided
    # view lowerings.
    "compiled as_strided view lowering fails to inline MLIR functions": {
        "max": {COMPILE_GRAD},
        "min": {COMPILE_GRAD},
        "nn.functional.avg_pool1d": {COMPILE},
        "permute": {COMPILE, COMPILE_GRAD},
        "stack": {COMPILE, COMPILE_GRAD},
        "transpose": {COMPILE, COMPILE_GRAD},
        "unsqueeze": {COMPILE, COMPILE_GRAD},
        "unsqueeze_copy": {COMPILE, COMPILE_GRAD},
    },
    # Fix XLA TPU tile layout verification for multi-dim
    # <=128 bounded 1-byte (pred/s8/u8/f8) shapes.
    "XLA fails layout verification with 'layout does not match any allowed tile kind'": {
        "__lshift__": {COMPILE},
        "__rshift__": {COMPILE},
        "abs": {COMPILE},
        "addcmul": {COMPILE},
        "cat": {COMPILE},
        "ceil": {COMPILE},
        "clone": {COMPILE},
        "floor": {COMPILE},
        "isnan": {COMPILE, COMPILE_GRAD},
        "isneginf": {COMPILE, COMPILE_GRAD},
        "isposinf": {COMPILE, COMPILE_GRAD},
        "le": {COMPILE},
        "lt": {COMPILE},
        "masked_fill": {COMPILE, COMPILE_GRAD},
        # COMPILE_GRAD is triaged under "compiled as_strided view lowering
        # fails to inline MLIR functions".
        "max": {COMPILE},
        "min": {COMPILE},
        "native_dropout_backward": {COMPILE, COMPILE_GRAD},
        # Eager mode is triaged under "the result has wrong values".
        "nn.functional.hardtanh": {COMPILE},
        "scatter_add": {COMPILE},
        "scatter_reduce": {COMPILE},
        # Eager mode is triaged under "untriaged: unsupported view op or
        # reshape".
        "select": {COMPILE},
        "signbit": {COMPILE, COMPILE_GRAD},
        # Eager and compile_grad modes are triaged under "untriaged:
        # unsupported view op or reshape".
        "split": {COMPILE},
        "tril": {COMPILE},
        "triu": {COMPILE},
        "where": {COMPILE, COMPILE_GRAD},
    },
    # Avoid HBM OOM when compiling 3D spatial <=128 buckets.
    "XLA runs out of HBM compiling multi-dim <=128 bounded shapes": {
        "nn.functional.conv_transpose3d": {COMPILE, COMPILE_GRAD},
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
        "_foreach_acos": {COMPILE_GRAD},  # float16 compiled backward tolerance
        "_foreach_asin": {COMPILE_GRAD},  # float16 compiled backward tolerance
        "_foreach_atan": {COMPILE_GRAD},  # float16 compiled backward tolerance
        "_foreach_erfc": {COMPILE_GRAD},  # float16 compiled backward tolerance
        "_foreach_pow": {COMPILE_GRAD},  # float16 compiled backward tolerance
        "_foreach_tanh": {COMPILE_GRAD},  # bfloat16/float16 backward tolerance
        "asin": {COMPILE_GRAD},  # float16 compiled backward tolerance
        "atan": {COMPILE_GRAD},  # float16 compiled backward tolerance
        "bmm": {COMPILE, COMPILE_GRAD},  # under dynamic compile
        "bmm.dtype": {COMPILE, COMPILE_GRAD},  # under dynamic compile
        "conj_physical": ALL_MODES,  # identity operation for i64 and f64 fails
        "erf": {COMPILE_GRAD},  # float16 compiled backward tolerance
        "erfinv": {COMPILE_GRAD},  # bfloat16 compiled backward tolerance
        "exp2": {COMPILE_GRAD},  # float16 compiled backward tolerance
        "expm1": ALL_MODES,  # result differs
        "mm": {COMPILE, COMPILE_GRAD},  # under dynamic compile
        "mm.dtype": {COMPILE, COMPILE_GRAD},  # under dynamic compile
        "nn.functional.conv3d": {COMPILE},  # bfloat16 tolerance
        "nn.functional.hardswish": ALL_MODES,  # adjust tolerance
        "nn.functional.nll_loss": ALL_MODES,  # or failure in some shards
        "nn.functional.relu": ALL_MODES,  # tolerance
        # Also on `compiled_deny_list` in op_testing.py, where compiled results
        # mismatch CPU even with static shapes.
        "nn.functional.scaled_dot_product_attention": {COMPILE, COMPILE_GRAD},
        "tanh": {COMPILE_GRAD},  # float16/bfloat16 compiled backward tolerance
        "to": ALL_MODES,  # fails numerics (?)
    },
    "untriaged: buffer dimension mismatch": {
        "fake_quantize_per_tensor_affine_cachemask": {EAGER},
        "fill": {COMPILE_GRAD},  # backward emits zeros_like
        "full_like": {COMPILE, COMPILE_GRAD},
        "isinf": {COMPILE, COMPILE_GRAD},
        "ones_like": {COMPILE, COMPILE_GRAD},
        "zeros_like": {COMPILE, COMPILE_GRAD},
    },
    "untriaged: MLIR assertion failure (bounds length vs rank)": {
        "bincount": ALL_MODES,
        "multinomial": {EAGER},
        "nn.functional.channel_shuffle": ALL_MODES,
        "nn.functional.max_pool1d": ALL_MODES,
        "put": ALL_MODES,
        "slice_scatter": ALL_MODES,
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
        "select": {EAGER},
        "slice": ALL_MODES,  # reshape reassociation not supported
        "split": {EAGER, COMPILE_GRAD},
        "split_with_sizes": {EAGER},
        # enhance view op to detect squeeze
        "squeeze": ALL_MODES,
        "squeeze_copy": ALL_MODES,
        "unbind": {EAGER},
        "var_mean": {
            COMPILE,
            COMPILE_GRAD,
        },  # reshape lowering failure in compile mode
    },
    "untriaged: shape mismatch": {
        "ldexp": ALL_MODES,  # dynamic shape value permutation mismatch
        "nn.functional.logsigmoid": {COMPILE_GRAD},  # inferred bound [0]
        "std": {
            COMPILE,
            COMPILE_GRAD,
        },  # shape mismatch on out under dynamic compile
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
        "angle": {COMPILE, COMPILE_GRAD},  # Dynamo error
        "bernoulli": {COMPILE_GRAD},  # Dynamo no_dispatch error
        "floor_divide": ALL_MODES,
        # Eager mode is triaged under "the lowering broadcasts a bounded input
        # to a static shape".
        "linalg.svd": {COMPILE, COMPILE_GRAD},
        "linalg.svdvals": {COMPILE, COMPILE_GRAD},
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
    kwargs: Mapping[str, Any] | None = None,
) -> None | str:
  """Checks if an op+input is supported with dynamism.

  Most skip reasons are bugs in lowering or XLA that will be fixed over time,
  enabling more coverage as we go.

  Args:
    op_info: The op to be tested.
    input_value: The primary input value to for the op test.
    args: The additional args to the op test.
    mode: The active dynamism mode (EAGER or COMPILE).
    kwargs: Optional keyword arguments passed to the op test.

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
  # _pytree.tree_flatten to recursively inspect input_value, args, and kwargs,
  # ensuring that tensors inside lists, tuples, or dicts (common in foreach ops
  # and keyword tensor args) are vetted for unsupported dtypes or empty shapes
  # while ignoring non-tensor parameters (e.g., kernel_size, stride, dim), `out`
  # placeholders, and broadcastable size-<=1 secondary tensors.
  op_args = _OpArgs(input_value, tuple(args), kwargs or {})
  for value in _collect_tensors(op_args):
    skip_tensor = _check_tensor_supported_for_dynamism(op, value)
    if skip_tensor:
      return skip_tensor

  # The loop above sees no tensors in a non-tensor or empty input (e.g., the
  # shape `()` passed to `torch.empty`), so also check the primary input that
  # `mark_dynamic` will mark.
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
        EinsumDynamicOpInfo,
        SdpaDynamicOpInfo,
        ConvolutionDynamicOpInfo,
        GluDynamicOpInfo,
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

  def _static_tensor_indices(
      self, primary_tensor_idx: int, num_tensors: int
  ) -> frozenset[int]:
    """Returns tensor indices in `_collect_tensors` that must stay static."""
    del primary_tensor_idx, num_tensors
    return frozenset()

  def _is_valid_coupled_plan(
      self,
      op_func: Callable[..., Any],
      op_args: _OpArgs,
      coupled: Sequence[TensorDim],
      mode: DynamismMode,
  ) -> bool:
    """Returns True if `coupled` stays dynamic and satisfies shape guards.

    Args:
      op_func: Callable for the active op variant.
      op_args: Operator inputs.
      coupled: `TensorDim`s to mark dynamic together for validation.
      mode: Dynamism execution mode.

    Returns:
      `True` if tracing succeeds with every dimension in `coupled` remaining
      unspecialized and satisfying mode-specific upper-bound guards.
    """
    # Eager mode only supports one dynamic dimension per tensor.
    if mode == DynamismMode.EAGER and len(
        {c.tensor_index for c in coupled}
    ) < len(coupled):
      logging.debug(
          "[DynamicOpInfo] Rejected %s (%s) in %s: >1 dynamic dim on tensor",
          self.op_info.name,
          coupled,
          mode,
      )
      return False

    # Trace the op with only `coupled` marked dynamic.
    try:
      shape_env, initial_syms = _run_fake_op(
          op_func,
          op_args,
          mode=mode,
          dynamic_dims=set(coupled),
      )
    except _UNTRACEABLE_FAKE_OP_EXCEPTIONS:
      raise
    except Exception as e:  # pylint: disable=broad-except
      # Reject if symbolic tracing fails on this dynamic marking.
      logging.debug(
          "[DynamicOpInfo] Rejected %s (%s) in %s: %s",
          self.op_info.name,
          coupled,
          mode,
          e,
      )
      return False

    # Verify every marked dimension remains an unspecialized symbol.
    canonical_syms: set[sympy.Symbol] = set()
    for c in coupled:
      canonical_sym = shape_env.replace(initial_syms[c])
      if (
          not isinstance(canonical_sym, sympy.Symbol)
          or shape_env.var_to_range[canonical_sym].is_singleton()
      ):
        logging.debug(
            "[DynamicOpInfo] Rejected %s (%s) in %s: %s specialized to %s",
            self.op_info.name,
            coupled,
            mode,
            c,
            canonical_sym,
        )
        return False
      canonical_syms.add(canonical_sym)

    # Verify mode-specific upper bounds satisfy all shape guards.
    if not _satisfies_upper_bound_guards(shape_env, canonical_syms, mode):
      logging.debug(
          "[DynamicOpInfo] Rejected %s (%s) in %s: upper-bound guards failed",
          self.op_info.name,
          coupled,
          mode,
      )
      return False

    return True

  def _discover_coupled_dims(
      self,
      candidates: Sequence[TensorDim],
      tensors: Sequence[torch.Tensor],
      op_args: _OpArgs,
      op_func: Callable[..., Any],
      mode: DynamismMode,
  ) -> dict[TensorDim, tuple[TensorDim, ...]]:
    """Discovers coupled tensor dimensions for each candidate.

    Uses `FakeTensorMode` and `ShapeEnv` symbolic tracing to deduce which input
    tensor dimensions must be marked dynamic together. It does this by:
      1. Assigns a distinct symbol to every eligible dimension (`size >= 2`)
         across all non-static input tensors.
      2. Traces the op symbolically via so `ShapeEnv` records
         equality constraints and unifies symbols across coupled dimensions.
      3. Groups dimensions whose initial symbols resolve to the same canonical
         symbol, skipping candidates that specialized to a constant or
         arithmetic expression.
      4. If joint tracing fails (e.g., when an op cannot have all dimensions
         dynamic simultaneously), falls back to proposing each candidate on its
         own for individual validation.

    Args:
      candidates: Candidate `TensorDim`s on the primary input tensor.
      tensors: All input tensors of the op.
      op_args: Operator inputs (`input_value`, `args`, `kwargs`).
      op_func: Callable for the active op variant.
      mode: Dynamism execution mode (`EAGER`, `COMPILE`, or `COMPILE_GRAD`).

    Returns:
      Mapping from each unspecialized candidate `TensorDim` to the sorted tuple
      of coupled `TensorDim`s across all input tensors.
    """
    # Identify secondary tensors that must stay static (e.g., index tensors).
    primary_tensor_idx = candidates[0].tensor_index
    static_tensor_indices = self._static_tensor_indices(
        primary_tensor_idx, len(tensors)
    )

    # Give every eligible dimension a distinct symbol so the op's meta kernel
    # reveals exact 1-to-1 coupled dimension equivalence classes.
    primary_dynamic = set(candidates)
    secondary_dynamic = {
        TensorDim(tensor_idx, d)
        for tensor_idx, t in enumerate(tensors)
        if tensor_idx != primary_tensor_idx
        and tensor_idx not in static_tensor_indices
        for d, sz in enumerate(t.shape)
        if sz >= 2
    }
    pass1_dynamic = primary_dynamic | secondary_dynamic

    proposed_by_candidate: dict[TensorDim, tuple[TensorDim, ...]] = {}
    try:
      shape_env, initial_syms = _run_fake_op(
          op_func,
          op_args,
          mode=mode,
          dynamic_dims=pass1_dynamic,
      )
      logging.debug(
          "[DynamicOpInfo] Discovery trace for %s in %s: initial_syms=%s,"
          " replacements=%s",
          self.op_info.name,
          mode,
          initial_syms,
          shape_env.replacements,
      )

      # Associate each candidate with the set of coupled dimensions.
      # TODO: Extend symbolic discovery to support algebraic
      # dimension constraints (e.g., s1 = s0 + k or s2 = s0 * s1) and symbolic
      # SymInt scalar arguments.
      for candidate in candidates:
        primary_sym = initial_syms[candidate]
        canonical_sym = shape_env.replace(primary_sym)
        # Skip candidates that specialized to a constant or arithmetic expr.
        if not isinstance(canonical_sym, sympy.Symbol):
          continue
        proposed_by_candidate[candidate] = tuple(
            td
            for td, sym in sorted(initial_syms.items())
            if shape_env.replace(sym) == canonical_sym
        )
    except _UNTRACEABLE_FAKE_OP_EXCEPTIONS:
      raise
    except Exception as e:  # pylint: disable=broad-except
      # Joint dynamic tracing failed (e.g., reshape s0 * s1 == K); fall back to
      # validating each candidate dimension individually.
      logging.debug(
          "[DynamicOpInfo] Joint discovery failed on %s in %s (%s); probing"
          " single dims",
          self.op_info.name,
          mode,
          e,
      )
      for candidate in candidates:
        proposed_by_candidate[candidate] = (candidate,)

    logging.debug(
        "[DynamicOpInfo] Discovered coupled dims for %s in %s: %s",
        self.op_info.name,
        mode,
        proposed_by_candidate,
    )
    return proposed_by_candidate

  def _validate_coupled_plans(
      self,
      candidates: Sequence[TensorDim],
      proposed_by_candidate: Mapping[TensorDim, tuple[TensorDim, ...]],
      op_args: _OpArgs,
      op_func: Callable[..., Any],
      mode: DynamismMode,
  ) -> tuple[DynamicMarkingPlan, ...]:
    """Validates coupled marking plans.

    Args:
      candidates: Candidate `TensorDim`s on the primary input tensor.
      proposed_by_candidate: Mapping from each candidate to the coupled
        `TensorDim` group.
      op_args: Operator inputs (`input_value`, `args`, `kwargs`).
      op_func: Callable for the active op variant.
      mode: Dynamism execution mode (`EAGER`, `COMPILE`, or `COMPILE_GRAD`).

    Returns:
      Validated single-dimension plans, plus a combined multi-dimension plan in
      compiled modes when valid.
    """
    valid_plans: list[DynamicMarkingPlan] = []
    seen_coupled: set[tuple[TensorDim, ...]] = set()

    # Validate each candidate's coupled group, skipping duplicate markings.
    for candidate in candidates:
      proposed = proposed_by_candidate.get(candidate)
      if not proposed or proposed in seen_coupled:
        continue
      if not self._is_valid_coupled_plan(op_func, op_args, proposed, mode):
        continue
      seen_coupled.add(proposed)
      valid_plans.append(
          DynamicMarkingPlan(
              label=_format_plan_label((candidate,)),
              tensor_dims=(candidate,),
              coupled_dims=proposed,
          )
      )

    # In compiled modes, also try marking all valid candidates together.
    if (
        mode in (DynamismMode.COMPILE, DynamismMode.COMPILE_GRAD)
        and len(valid_plans) >= 2
    ):
      merged_coupled = tuple(
          sorted({c for plan in valid_plans for c in plan.coupled_dims})
      )
      if merged_coupled not in seen_coupled and self._is_valid_coupled_plan(
          op_func, op_args, merged_coupled, mode
      ):
        merged_dims = tuple(plan.tensor_dims[0] for plan in valid_plans)
        valid_plans.append(
            DynamicMarkingPlan(
                label=_format_plan_label(merged_dims),
                tensor_dims=merged_dims,
                coupled_dims=merged_coupled,
            )
        )

    logging.debug(
        "[DynamicOpInfo] Validated plans for %s in %s: %s",
        self.op_info.name,
        mode,
        valid_plans,
    )
    return tuple(valid_plans)

  def generate_exhaustive_marking_plans(
      self,
      input_value: torch.Tensor | Sequence[torch.Tensor],
      args: Sequence[Any],
      mode: DynamismMode = DynamismMode.EAGER,
      kwargs: Mapping[str, Any] | None = None,
      op_func: Callable[..., Any] | None = None,
  ) -> Sequence[DynamicMarkingPlan]:
    """Generates all valid dynamic marking plans for the canonical input tensor.

    Uses `FakeTensorMode` and `ShapeEnv` to discover coupled dimensions across
    `(input_value, args, kwargs)` and filter out dimensions that PyTorch
    specializes or that violate shape constraints.

    Args:
      input_value: First positional argument passed to the op.
      args: Remaining positional arguments passed to the op.
      mode: Dynamism execution mode (`EAGER`, `COMPILE`, or `COMPILE_GRAD`).
        Simultaneous multi-dimension plans are only emitted in compiled modes.
      kwargs: Optional keyword arguments passed to the op.
      op_func: Optional callable for the active op variant (e.g., inplace);
        defaults to `self.op_info` when `None`.

    Returns:
      A sequence of validated `DynamicMarkingPlan` instances, or `()` if no
      dimensions on the canonical input tensor can be marked dynamic.
    """
    candidates = list(self.get_candidate_dims(input_value, args))
    if not candidates:
      return ()

    op_args = _OpArgs(input_value, tuple(args), kwargs or {})

    # Extract all tensors from the op's inputs.
    tensors = _collect_tensors(op_args)

    # Skip symbolic analysis if any tensor is not strided.
    if any(t.layout != torch.strided for t in tensors):
      return ()

    effective_op_func = self.op_info if op_func is None else op_func
    try:
      proposed_by_candidate = self._discover_coupled_dims(
          candidates, tensors, op_args, effective_op_func, mode
      )
      return self._validate_coupled_plans(
          candidates, proposed_by_candidate, op_args, effective_op_func, mode
      )
    except _UNTRACEABLE_FAKE_OP_EXCEPTIONS:
      return ()

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
    if plan.coupled_dims:
      op_args = _OpArgs(input_value, tuple(args), kwargs or {})
      tensors = _collect_tensors(op_args)
      for coupled in plan.coupled_dims:
        tensor = tensors[coupled.tensor_index]
        tensor_str = utils.get_tensor_summary(tensor, data=False)
        ub = tensor.shape[coupled.dim] + 10
        bound_str = f" [<={ub}]" if mode == DynamismMode.EAGER else ""
        print(
            f">>>> Marking tensor[{coupled.tensor_index}] dim"
            f" {coupled.dim} dynamic{bound_str} in {tensor_str}"
            f" (mode={mode.value})",
            flush=True,
        )
        _mark_tensor_dynamic(tensor, coupled.dim, 2, ub, mode=mode)
    else:
      for target in plan.tensor_dims:
        self._mark_single_dim(
            target, input_value, args, mode=mode, kwargs=kwargs
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
      kwargs: Mapping[str, Any] | None = None,
      op_func: Callable[..., Any] | None = None,
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
      kwargs: Optional keyword arguments, accepted to match the `DynamicOpInfo`
        base signature.
      op_func: Optional op variant callable, accepted to match the
        `DynamicOpInfo` base signature.

    Returns:
      `(DynamicMarkingPlan(label="foreach"),)` if `input_value` has at least one
      candidate dynamic dimension, or `()` otherwise.
    """
    del mode, kwargs, op_func  # Accepted to match DynamicOpInfo signature.
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
        rhs_arg = (
            rhs_vals[idx] if isinstance(rhs_vals, (list, tuple)) else rhs_vals
        )
        ehs_arg = (
            ehs_vals[idx] if isinstance(ehs_vals, (list, tuple)) else ehs_vals
        )
        TernaryElementwiseDynamicOpInfo(self.op_info).mark_dynamic(
            value,
            (rhs_arg, ehs_arg),
            mode=mode,
        )
        continue

      # Not binary or ternary, no dependent args.
      super().mark_dynamic(value, [], mode=mode)
      logging.debug(
          "[ForEachDynamicOpInfo] no dependent args %s",
          utils.InputMetadata(args),
      )


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

  def _static_tensor_indices(
      self, primary_tensor_idx: int, num_tensors: int
  ) -> frozenset[int]:
    """Keeps convolution weights and biases static as required by StableHLO."""
    return frozenset(i for i in range(num_tensors) if i != primary_tensor_idx)

  def dynamic_dimension_candidates(
      self, input_value: torch.Tensor, args: Sequence[Any]
  ) -> Sequence[int]:
    """Returns the batch and spatial dimensions of the convolution input.

    The input channels must match the static weight, so convolutions don't
    support bounded channel dimensions.
    """
    if not args or not isinstance(args[0], torch.Tensor):
      return ()
    # An unbatched input has one dimension less than the weight in args[0].
    channel_dim = 1 if input_value.ndim == args[0].ndim else 0
    return [
        dim
        for dim in super().dynamic_dimension_candidates(input_value, args)
        if dim != channel_dim
    ]


class GluDynamicOpInfo(DynamicOpInfo):
  """A wrapper around glu OpInfos that keeps the split dimension static."""

  @staticmethod
  def adapter_supports_op(op_info: core.OpInfo) -> bool:
    """Method to declare ops supported by this adapter."""
    return op_info.name == "nn.functional.glu"

  def dynamic_dimension_candidates(
      self, input_value: torch.Tensor, args: Sequence[Any]
  ) -> Sequence[int]:
    """Returns dimensions of input_value excluding the split dimension."""
    if input_value.ndim == 0:
      return []
    split_dim = (args[0] if args else -1) % input_value.ndim
    return [
        dim
        for dim in super().dynamic_dimension_candidates(input_value, args)
        if dim != split_dim
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


def generate_exhaustive_input_marking_plans(
    op_info: core.OpInfo,
    input_value: torch.Tensor | Sequence[torch.Tensor],
    args: Sequence[Any],
    mode: DynamismMode = DynamismMode.EAGER,
    kwargs: Mapping[str, Any] | None = None,
    op_func: Callable[..., Any] | None = None,
) -> Sequence[DynamicMarkingPlan]:
  """Generates all valid dynamic marking plans for an op sample.

  Args:
    op_info: The `OpInfo` descriptor for the operator under test.
    input_value: Primary tensor or sequence of tensors passed to the op.
    args: Additional positional arguments passed to the op.
    mode: Dynamism execution mode (`EAGER` or `COMPILE`).
    kwargs: Optional keyword arguments passed to the op.
    op_func: Optional callable for the active op variant.

  Returns:
    A sequence of `DynamicMarkingPlan` instances for the sample, or `()` if no
    dimensions can be marked dynamic.
  """
  return DynamicOpInfo.get(op_info).generate_exhaustive_marking_plans(
      input_value,
      args,
      mode=mode,
      kwargs=kwargs,
      op_func=op_func,
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
  print(f">>> Marking input dynamic{plan_suffix} (mode={mode.value})")
  dynamic_op_info.mark_dynamic(
      input_value, args, mode=mode, kwargs=kwargs, plan=plan
  )
