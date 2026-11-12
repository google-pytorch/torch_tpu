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
  - Discover input dimensions that must be marked dynamic together.
  - Mark inputs as dynamic.
"""

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
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
      together when executing this plan.
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
    dynamic_dims: Collection[TensorDim],
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
    dynamic_dims: Collection[TensorDim],
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
    canonical_syms: Iterable[sympy.Symbol],
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


def verify_dtype_supports_dynamism(
    op_info: core.OpInfo, dtype: torch.dtype
) -> None | str:
  """Verifies that `dtype` is supported for `op_info` under dynamism.

  Args:
    op_info: The op to be tested.
    dtype: The input dtype to check.

  Returns:
    None if `dtype` is supported for dynamism testing, or a skip reason string.
  """
  # TODO: b/449736443 - [XLA] Crash in complex rewriter.
  if dtype in (torch.complex64, torch.complex128):
    return f"{op_info.name} op is not supported with dtype {dtype}."
  return None


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
  skip_dtype = verify_dtype_supports_dynamism(op_info, tensor.dtype)
  if skip_dtype is not None:
    return skip_dtype
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
        # Aborts the test process with a CHECK failure in BroadcastIfNeeded
        # (op_builder_utils.cc).
        "_segment_reduce": {EAGER, COMPILE},
        "_weight_norm_interface_backward": {EAGER, COMPILE},
        "atanh": {COMPILE_GRAD},
        "bernoulli": {EAGER},
        "cdist": {EAGER},
        # These abort the test process with a CHECK failure in
        # BroadcastIfNeeded (op_builder_utils.cc).
        "linalg.qr": {EAGER},
        "linalg.svd": {EAGER},
        "linalg.svdvals": {EAGER, COMPILE_GRAD},
        "masked_scatter": {EAGER},
        "multinomial": {EAGER},
        "nn.functional.nll_loss": {COMPILE_GRAD},
    },
    # TODO: b/449736443 - Support bounded dimensions in these lowerings.
    "the lowering emits an op without bounded shape support": {
        # Backward broadcast fails with mixed bounded and static dimensions.
        "_foreach_lerp": {COMPILE_GRAD},
        # Calls getNumElements() on a dynamic shaped type, failing an MLIR
        # assertion that aborts the test process in builds with MLIR
        # assertions enabled.
        "_foreach_norm": {EAGER, COMPILE_GRAD},
        # BuildMlirModule fails to inline MLIR functions on bounded inputs.
        "_fused_moving_avg_obs_fq_helper": ALL_MODES,
        # XLA ragged-dot does not support propagating bounded dimensions.
        "_grouped_mm": {EAGER, COMPILE},
        # XLA Scatter updates bound mismatches static scatter_indices bound.
        "_jagged_to_padded_dense_forward": {EAGER},
        # Fails to inline MLIR functions on bounded inputs.
        "_native_multi_head_attention": {EAGER},
        # BuildMlirModule fails to inline MLIR functions on bounded inputs.
        "_weight_norm_interface": {COMPILE, COMPILE_GRAD},
        # ViewPrimitiveShlo does not support slicing a dynamic dimension.
        "as_strided": ALL_MODES,
        # Fails to inline MLIR functions on bounded inputs.
        "bincount": {EAGER},
        # BuildMlirModule fails to inline MLIR functions on bounded inputs;
        # EAGER mode is triaged under "the lowering broadcasts a bounded input
        # to a static shape".
        "cdist": {COMPILE_GRAD},
        # cummax_helper and cummin_helper emit static stablehlo.iota and query
        # static dim sizes (-9223372036854775808). Aborts the test process.
        "cummax": ALL_MODES,
        "cummin": ALL_MODES,
        # chlo.ScanOp fails MLIR function inlining.
        "cumprod": {EAGER, COMPILE_GRAD},
        # cumsum.out segfaults (eager), and chlo.ScanOp fails MLIR function
        # inlining (compile).
        "cumsum": ALL_MODES,
        # Fails the getDotResultType assertion on bounded 1D operands. Aborts
        # the test process in builds with MLIR assertions enabled.
        "dot": ALL_MODES,
        # embedding_renorm_ uses static BroadcastInDim and Scatter shapes.
        # Aborts the test process.
        "embedding_renorm_": {EAGER, COMPILE},
        # BuildFmaxShlo and BuildFminShlo emit NaN CompareOp on integer tensors,
        # failing an MLIR assertion in DenseElementsAttr. Aborts the test
        # process in builds with MLIR assertions enabled.
        "fmax": {EAGER},
        "fmin": {EAGER},
        # Checks index.size(d) <= input.size(d) against INT64_MIN.
        "gather": {EAGER, COMPILE_GRAD},
        # Compiled backward loses grad_fn when hidden size is dynamic.
        "gru.input": {COMPILE_GRAD},
        # Fails to inline MLIR functions on bounded inputs. COMPILE is triaged
        # under "XLA fails layout verification with 'layout does not match any
        # allowed tile kind'".
        "index_copy": {COMPILE_GRAD},
        # BuildMlirModule fails to inline MLIR functions on bounded inputs.
        "index_fill": ALL_MODES,
        # chlo.ScanOp fails MLIR function inlining.
        "logcumsumexp": ALL_MODES,
        # SliceOp::build fails return type inference on bounded hidden slices.
        # Aborts the test process.
        "lstm.input": ALL_MODES,
        # Slicing a dynamic dimension is not supported in ViewPrimitiveShlo
        # (eager), and LU_pivots shape check fails (compile_grad).
        "lu_unpack": {EAGER, COMPILE_GRAD},
        # Backward pass calls local_scalar_dense on a placeholder tensor.
        "mean": {COMPILE_GRAD},
        # Allocates a zero-sized shape with no zero dimension. Aborts the test
        # process.
        "nn.functional.adaptive_avg_pool3d": {EAGER, COMPILE_GRAD},
        # Queries static dim size (slice size -9223372036854775808). Aborts the
        # test process.
        "nn.functional.adaptive_max_pool2d": {EAGER, COMPILE},
        # Aborts the test process.
        "nn.functional.avg_pool1d": {COMPILE_GRAD},
        # SliceBackwardResult queries static input shape. Aborts the test
        # process.
        "nn.functional.avg_pool2d": {COMPILE_GRAD},
        # PropagateBoundedDynamism fails MarkDynamic when pooling reduces a
        # bounded spatial dimension to size 1 (eager), and avg_pool3d_backward
        # queries static input shape, aborting the test process (compile_grad).
        "nn.functional.avg_pool3d": {EAGER, COMPILE_GRAD},
        # Fails to inline MLIR functions on bounded inputs.
        "nn.functional.channel_shuffle": ALL_MODES,
        # cross_entropy_loss queries a dynamic dim size, failing an MLIR
        # assertion that aborts the test process in builds with MLIR assertions
        # enabled (eager), and its backward fails to inline MLIR functions
        # (compile_grad). COMPILE is triaged under "the result has wrong
        # values".
        "nn.functional.cross_entropy": {EAGER, COMPILE_GRAD},
        # var_mean reshape reassociation fails (eager) and local_scalar_dense
        # cannot materialize a placeholder tensor (compile_grad).
        "nn.functional.group_norm": {EAGER, COMPILE_GRAD},
        # Queries static dim size (dimension size -9223372036854775808). Eager
        # aborts the test process, and compile fails to inline MLIR functions.
        "nn.functional.max_pool1d": ALL_MODES,
        # Queries static dim size (dimension size -9223372036854775808). Aborts
        # the test process.
        "nn.functional.max_pool2d": ALL_MODES,
        "nn.functional.max_pool3d": ALL_MODES,
        # Queries a dynamic dim size, failing an MLIR assertion that aborts the
        # test process in builds with MLIR assertions enabled. COMPILE and
        # COMPILE_GRAD are triaged under other reasons.
        "nn.functional.nll_loss": {EAGER},
        # Calls getNumElements() on a dynamic shaped type, failing an MLIR
        # assertion that aborts the test process in builds with MLIR
        # assertions enabled.
        "norm": {EAGER, COMPILE},
        # reflection_pad*d and replication_pad*d query static input shape.
        # Aborts the test process.
        "nn.functional.pad": {EAGER, COMPILE_GRAD},
        # Variable repeats fail MLIR function inlining in Cumsum/ScanOp.
        "repeat_interleave": {EAGER},
        # Queries static input shape (limit index -9223372036854775808). Aborts
        # the test process.
        "scatter": {EAGER, COMPILE},
        # AtenSelectScatter checks static src shape against padded self slice.
        "select_scatter": {COMPILE, COMPILE_GRAD},
        # sort backward lowers via scatter, which fails MLIR function inlining
        # on multi-dim bounded shapes.
        "sort": {COMPILE_GRAD},
        # Fails to inline MLIR functions on bounded inputs.
        "squeeze": {COMPILE, COMPILE_GRAD},
        "squeeze_copy": {COMPILE, COMPILE_GRAD},
        # Lowers via sort with %iota, which lacks XLA dynamic_padder support.
        "topk": {EAGER},
        # Calls getNumElements() and static scale checks on bounded operands.
        # torch._scaled_mm_v2 aborts the test process.
        "torch._scaled_mm": {EAGER, COMPILE},
        "torch._scaled_mm_v2": {EAGER, COMPILE},
        # stablehlo.reshape allows at most one bounded dimension. This fails
        # when dims 0 and 2 are both dynamic, which happens in compile mode
        # because dims marked in earlier subtests stay dynamic.
        # TODO: b/566327707 - Remove once marked dims no longer carry over.
        "torch.ops.aten._safe_softmax.default": {COMPILE, COMPILE_GRAD},
        # View decomposition reshape reassociation fails on bounded dimensions.
        "unfold": ALL_MODES,
    },
    # TODO: b/449736443 - Avoid Tensor.resize_() graph break in the *.dtype
    # test wrappers in op_testing.py.
    "the test wrapper calls Tensor.resize_(), which graph-breaks in Dynamo": {
        "addmm.dtype": {COMPILE},
        "bmm.dtype": {COMPILE},
        "mm.dtype": {COMPILE},
    },
    # TODO: b/449736443 - Support bounded dimensions in these lowerings.
    "linalg decompositions do not support bounded dimensions": {
        # Slicing a dynamic dimension is not supported.
        "linalg.inv": {COMPILE_GRAD},
        "linalg.lu": {EAGER},
        # Slicing a dynamic dimension is not supported.
        "linalg.solve_ex": {COMPILE_GRAD},
    },
    # TODO: b/449736443 - Support bounded dimensions in these lowerings.
    "the result has wrong values": {
        "nn.functional.cross_entropy": {COMPILE},
        # TODO: b/566327914 - With dim 0 bounded, the result matches the
        # transposed golden output.
        "nn.functional.hardtanh": {EAGER},
    },
    # Fix MLIR function inlining in compiled as_strided
    # view lowerings.
    "compiled as_strided view lowering fails to inline MLIR functions": {
        "max": {COMPILE_GRAD},
        "min": {COMPILE_GRAD},
        "permute": {COMPILE, COMPILE_GRAD},
        "stack": {COMPILE_GRAD},
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
        "ceil": {COMPILE},
        # Eager mode is triaged under "untriaged: invalid result".
        "clamp": {COMPILE, COMPILE_GRAD},
        "clone": {COMPILE},
        "floor": {COMPILE},
        "index_copy": {COMPILE},
        # COMPILE is triaged under "untriaged: buffer dimension mismatch".
        "isinf": {COMPILE_GRAD},
        "isnan": {COMPILE, COMPILE_GRAD},
        "isneginf": {COMPILE, COMPILE_GRAD},
        "isposinf": {COMPILE, COMPILE_GRAD},
        "masked_fill": {COMPILE, COMPILE_GRAD},
        # COMPILE_GRAD is triaged under "compiled as_strided view lowering
        # fails to inline MLIR functions".
        "max": {COMPILE},
        "min": {COMPILE},
        "native_dropout_backward": {COMPILE, COMPILE_GRAD},
        "neg": {COMPILE},
        "round": {COMPILE},
        "scatter_add": {COMPILE},
        "scatter_reduce": {COMPILE},
        # Eager mode is triaged under "untriaged: unsupported view op or
        # reshape".
        "select": {COMPILE},
        "sgn": {COMPILE},
        "sign": {COMPILE},
        "signbit": {COMPILE, COMPILE_GRAD},
        # Eager mode is triaged under "untriaged: unsupported view op or
        # reshape", and compile_grad under "untriaged: XLA compile or runtime
        # failure".
        "split": {COMPILE},
        "split_with_sizes_copy": {COMPILE},
        "tril": {COMPILE},
        "triu": {COMPILE},
        "trunc": {COMPILE},
        "where": {COMPILE, COMPILE_GRAD},
        "zero_": {COMPILE},
    },
    # Avoid HBM OOM when compiling 3D spatial <=128 buckets.
    "XLA runs out of HBM compiling multi-dim <=128 bounded shapes": {
        "nn.functional.conv_transpose3d": {COMPILE, COMPILE_GRAD},
    },
    "untriaged: invalid result": {
        # TODO: fix clamp() returning enormous errors or nans when dynamism is
        # enabled.
        "clamp": {EAGER},
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
        "erf": {COMPILE_GRAD},  # float16 compiled backward tolerance
        "erfinv": {COMPILE_GRAD},  # bfloat16 compiled backward tolerance
        "nn.functional.nll_loss": {COMPILE},
        "tanh": {COMPILE_GRAD},  # float16/bfloat16 compiled backward tolerance
    },
    "untriaged: buffer dimension mismatch": {
        # These abort the test process with a CHECK failure in
        # device_buffer.cc.
        "fill": {COMPILE_GRAD},  # backward emits zeros_like
        "isinf": {COMPILE},
    },
    # These fail an MLIR RankedTensorType invariant check, which aborts the test
    # process in builds with MLIR assertions enabled (not in -c opt).
    "untriaged: MLIR assertion failure (bounds length vs rank)": {
        "matmul": {EAGER, COMPILE_GRAD},
        "vdot": ALL_MODES,
    },
    "untriaged: materialization failure (negative dimension size)": {
        # Aborts the test process.
        "isin": ALL_MODES,
    },
    "untriaged: expected all dimensions to be static": {
        "fft.ifft": {EAGER},
        "fft.rfft": {EAGER},
        # Expects a static weight tensor.
        "nn.functional.embedding": {EAGER},
        "nn.functional.embedding_bag": {EAGER},
    },
    "untriaged: invalid gather": {
        # XLA scatter does not support a dynamic update window size.
        "index_copy": {EAGER},
    },
    "untriaged: XLA compile or runtime failure": {
        "bmm": {COMPILE_GRAD},  # RuntimeProgramInputMismatch
        "linalg.lu_factor_ex": {EAGER},  # incompatible shapes RET_CHECK
        "mm": {COMPILE_GRAD},  # TPU core halted unexpectedly
        "split": {COMPILE_GRAD},  # RuntimeProgramInputMismatch
    },
    # These abort the test process.
    "untriaged: crash": {
        "_transformer_encoder_layer_fwd": {EAGER},
        "fft.irfft": {COMPILE_GRAD},  # buffer != nullptr CHECK in shape.h
        "index_select": ALL_MODES,
        # Allocates a zero-sized shape with no zero dimension.
        "nn.functional.adaptive_avg_pool2d": ALL_MODES,
        "nn.functional.embedding": {COMPILE, COMPILE_GRAD},
        "nn.functional.embedding_bag": {COMPILE, COMPILE_GRAD},
        "nn.functional.grid_sample": ALL_MODES,
        "searchsorted": ALL_MODES,  # buffer != nullptr CHECK in shape.h
        "slice_scatter": {EAGER, COMPILE},  # buffer != nullptr CHECK in shape.h
        "unique": {EAGER},
        "unique_consecutive": {EAGER},
        "view_as_complex": {COMPILE, COMPILE_GRAD},
    },
    "untriaged: unsupported view op or reshape": {
        # Slicing a dynamic dimension is not supported.
        "diagonal": ALL_MODES,
        # unflatten is ambiguous for a bounded dimension.
        "put": {EAGER, COMPILE_GRAD},
        # Slicing a dynamic dimension is not supported.
        "select": {EAGER},
        "split": {EAGER},
        "split_with_sizes": {EAGER},
        "split_with_sizes_copy": {EAGER},
        # Reshape has mismatched element counts (eager), and unflatten is
        # ambiguous for a bounded dimension (compile_grad).
        "take": {EAGER, COMPILE_GRAD},
    },
    "untriaged: shape mismatch": {
        # Compile marks self dim bounded but not source.
        "index_add": {COMPILE, COMPILE_GRAD},
        "nn.functional.logsigmoid": {COMPILE_GRAD},  # inferred bound [0]
        # src does not match the bounded self slice.
        "slice_scatter": {COMPILE_GRAD},
    },
    "untriaged: unknown failure": {
        # Input SymInt is not present in example_inputs.
        "_thnn_fused_gru_cell": {COMPILE_GRAD},
        "bernoulli": {COMPILE_GRAD},  # Dynamo no_dispatch error
        # Compiled backward graph codegen emits invalid Python.
        "std": {COMPILE_GRAD},
        "var": {COMPILE_GRAD},
        # Unhandled SymInt usage in the compiled FX graph.
        "var_mean": {COMPILE_GRAD},
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
    mode: DynamismMode = DynamismMode.EAGER,
) -> None | str:
  """Checks if an op is supported in `mode` with dynamism.

  Args:
    op_info: The op to be tested.
    mode: The active dynamism mode (EAGER, COMPILE, or COMPILE_GRAD).

  Returns:
    None if the op is supported in `mode`, or a skipTest message if not.
  """
  reason = _find_reason(DYNAMISM_NOT_APPLICABLE_OPS, op_info.name, mode)
  if reason is not None:
    return (
        f"Dynamism is not applicable to {op_info.name} in {mode.value} mode:"
        f" {reason}."
    )
  reason = _find_reason(DYNAMISM_DENY_LISTS, op_info.name, mode)
  if reason is not None:
    return (
        f"Op {op_info.name} is not supported in {mode.value} dynamism mode:"
        f" {reason}."
    )
  return None


def verify_sample_supports_dynamism(
    op_info: core.OpInfo,
    input_value: torch.Tensor,
    args: Sequence[Any],
    kwargs: Mapping[str, Any] | None = None,
) -> None | str:
  """Checks if a sample input is supported with dynamism.

  Args:
    op_info: The op to be tested.
    input_value: The primary input value for the op test.
    args: The additional args to the op test.
    kwargs: Optional keyword arguments passed to the op test.

  Returns:
    None if the sample is supported, or a skipTest message if not.
  """
  # Verify all input tensors are supported with dynamism. Here we use
  # _pytree.tree_flatten to recursively inspect input_value, args, and kwargs,
  # ensuring that tensors inside lists, tuples, or dicts (common in foreach ops
  # and keyword tensor args) are vetted for unsupported dtypes or empty shapes
  # while ignoring non-tensor parameters (e.g., kernel_size, stride, dim), `out`
  # placeholders, and broadcastable size-<=1 secondary tensors.
  op_args = _OpArgs(input_value, tuple(args), kwargs or {})
  for value in _collect_tensors(op_args):
    skip_tensor = _check_tensor_supported_for_dynamism(op_info, value)
    if skip_tensor:
      return skip_tensor

  # The loop above sees no tensors in a non-tensor or empty input (e.g., the
  # shape `()` passed to `torch.empty`), so also check the primary input that
  # `mark_dynamic` will mark.
  _, canonical_input = _get_canonical_input_value(op_info, input_value)
  skip_input = _should_skip_input_marking(op_info, canonical_input)
  if skip_input:
    return skip_input
  if not DynamicOpInfo.get(op_info).get_candidate_dims(input_value, args):
    return "No valid input dimensions to mark dynamic."

  # Valid testcase!
  return None


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
        dynamic. When `None`, marks only the first candidate `TensorDim`.

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
          coupled_dims=(chosen,),
      )
    logging.debug(
        "[DynamicOpInfo] mark_dynamic(%s, plan=%s, %s, %s, mode=%s)",
        self.op_info.name,
        plan.label,
        utils.InputMetadata(input_value),
        utils.InputMetadata(args),
        mode,
    )
    tensors = _collect_tensors(_OpArgs(input_value, tuple(args), kwargs or {}))
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


class ForEachDynamicOpInfo(DynamicOpInfo):
  """A wrapper around foreach OpInfos that marks several list elements dynamic.

  Foreach ops apply the same op to element `i` of every tensor-list argument,
  so the dimensions to mark together follow from the op's structure: a
  dimension of input element `i` is coupled with the matching (NumPy
  broadcasting) dimension of element `i` of every other tensor list. Plans are
  derived from this structure instead of `FakeTensor` tracing, because tracing
  foreach ops on CPU specializes all dimensions for some dtypes (e.g., float16
  or integer inputs), which the TPU kernels do not.
  """

  # Max number of list elements to mark dynamic in the plan, to bound test
  # time and keep the plan representative.
  max_dynamic_tensors = 3

  @staticmethod
  def adapter_supports_op(op_info: core.OpInfo):
    """Method to declare ops supported by this adapter."""

    return isinstance(op_info, core.ForeachFuncInfo)

  def get_candidate_dims(
      self,
      input_value: torch.Tensor | Sequence[torch.Tensor],
      args: Sequence[Any],
  ) -> Sequence[TensorDim]:
    """Returns the first candidate dimension of each eligible list element.

    Args:
      input_value: Sequence of primary input tensors passed to the foreach op.
      args: Additional positional arguments (scalars or tensor sequences).

    Returns:
      Strictly ascending `TensorDim` candidates, one per eligible element of
      `input_value`, for at most `max_dynamic_tensors` elements.
    """
    if isinstance(input_value, torch.Tensor):
      return super().get_candidate_dims(input_value, args)
    candidates: list[TensorDim] = []
    for tensor_index, value in enumerate(input_value):
      if len(candidates) >= self.max_dynamic_tensors:
        break
      if _should_skip_input_marking(self.op_info, value):
        continue
      dims = self.dynamic_dimension_candidates(value, args)
      candidates.append(TensorDim(tensor_index=tensor_index, dim=dims[0]))
    return candidates

  def _coupled_dims(
      self,
      candidate: TensorDim,
      op_args: _OpArgs,
  ) -> tuple[TensorDim, ...]:
    """Returns `candidate` and the matching dims of the other tensor lists.

    Args:
      candidate: A dimension of an element of the primary input list.
      op_args: Operator inputs.

    Returns:
      Sorted `TensorDim`s, indexed like `_collect_tensors`, that must be marked
      dynamic together with `candidate`.
    """
    # The input list's tensors come first in `_collect_tensors`, so the
    # candidate's tensor index is also its element index `i` in the input list,
    # which a foreach op pairs with element `i` of every other tensor list.
    element_index = candidate.tensor_index
    primary = op_args.input_value[element_index]
    num_elements = len(op_args.input_value)

    # Flatten the inputs in the same order as `_collect_tensors`, keeping each
    # leaf's path: the keys that index it from the root tuple. For example, the
    # path `(SequenceKey(1), SequenceKey(0), SequenceKey(2))` is `args[0][2]`:
    #   - `path[0]` selects the input part (`input_value`, `args`, or `kwargs`).
    #   - The middle keys select the argument (a `SequenceKey` position in
    #     `args` or a `MappingKey` name in `kwargs`).
    #   - `path[-1]` is the leaf's element index in its containing list.
    leaves, _ = _pytree.tree_flatten_with_path(
        (op_args.input_value, op_args.args, _normalize_kwargs(op_args.kwargs))
    )
    tensor_leaves = [
        (path, leaf) for path, leaf in leaves if isinstance(leaf, torch.Tensor)
    ]
    input_value_key = _pytree.SequenceKey(0)
    paired_element_key = _pytree.SequenceKey(element_index)

    coupled = [candidate]
    for tensor_index, (path, tensor) in enumerate(tensor_leaves):
      # Other elements of the input list are independent of the candidate.
      if path[0] == input_value_key:
        continue
      # Tensors passed directly in `args` or `kwargs` (path `(part, argument)`)
      # are broadcast to every element, so they stay static.
      if len(path) < 3:
        continue
      element_key, list_path = path[-1], path[:-1]
      # Keep only element `i` of lists with one element per input element.
      if (
          element_key != paired_element_key
          or _list_length(op_args, list_path) != num_elements
      ):
        continue
      # Find the matching dim using NumPy broadcasting rules (shapes align from
      # the right), skipping dims that broadcast (size 1) or don't exist.
      dim = candidate.dim - primary.ndim + tensor.ndim
      if 0 <= dim < tensor.ndim and (
          tensor.shape[dim] == primary.shape[candidate.dim]
      ):
        coupled.append(TensorDim(tensor_index=tensor_index, dim=dim))
    return tuple(sorted(coupled))

  def generate_exhaustive_marking_plans(
      self,
      input_value: torch.Tensor | Sequence[torch.Tensor],
      args: Sequence[Any],
      mode: DynamismMode = DynamismMode.EAGER,
      kwargs: Mapping[str, Any] | None = None,
      op_func: Callable[..., Any] | None = None,
  ) -> Sequence[DynamicMarkingPlan]:
    """Generates a single plan marking several list elements dynamic together.

    The per-element math of foreach ops is covered by the tests of the
    corresponding base ops, so a single plan suffices to exercise what is
    foreach-specific: one multi-output module over several independently
    bounded elements, their coupling to the other tensor lists, and the
    foreach dtype promotion and in-place dispatch paths.

    Args:
      input_value: Sequence of primary input tensors passed to the foreach op.
      args: Remaining positional arguments passed to the op.
      mode: Unused; the plan marks at most one dim per tensor, which every mode
        supports.
      kwargs: Optional keyword arguments passed to the op.
      op_func: Unused; plans are derived without tracing the op.

    Returns:
      A single plan marking all candidate dimensions and their coupled dims, or
      `()` if there are no candidates.
    """
    del mode, op_func  # Unused.
    if isinstance(input_value, torch.Tensor):
      return ()
    candidates = tuple(self.get_candidate_dims(input_value, args))
    if not candidates:
      return ()
    op_args = _OpArgs(input_value, tuple(args), kwargs or {})
    coupled_dims: set[TensorDim] = set()
    for candidate in candidates:
      coupled_dims.update(self._coupled_dims(candidate, op_args))
    return (
        DynamicMarkingPlan(
            label=_format_plan_label(candidates),
            tensor_dims=candidates,
            coupled_dims=tuple(sorted(coupled_dims)),
        ),
    )


def _list_length(op_args: _OpArgs, path: Sequence[Any]) -> int | None:
  """Returns the length of the sequence at `path` in `op_args`, if any."""
  node: Any = (
      op_args.input_value,
      op_args.args,
      _normalize_kwargs(op_args.kwargs),
  )
  for key in path:
    if isinstance(key, _pytree.SequenceKey):
      node = node[key.idx]
    elif isinstance(key, _pytree.MappingKey):
      node = node[key.key]
    else:
      return None
  return len(node) if isinstance(node, (list, tuple)) else None


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
  def _tokenize_subscript(sub: str) -> Sequence[str]:
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
  ) -> tuple[Sequence[Sequence[str]], Sequence[str]]:
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
