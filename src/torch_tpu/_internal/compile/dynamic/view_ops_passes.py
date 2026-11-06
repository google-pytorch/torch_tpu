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

"""View operations transformation pass for handling dynamic shapes."""

from __future__ import annotations
from collections.abc import Sequence
import operator
from typing import Any
from absl import logging
import torch
import torch.utils._pytree as pytree
from torch_tpu._internal.compile.dynamic import sym_utils
from torch_tpu._internal.compile.dynamic.graph_transform_context import GraphTransformContext


def _get_node_val(arg: Any) -> Any:
  """Unwraps an FX Node to its metadata value ('val') if present."""
  if isinstance(arg, torch.fx.Node) and "val" in arg.meta:
    return arg.meta["val"]
  return arg


def _extract_shape_tensors_and_bounds(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
    target_shape: Sequence[Any],
    transform_context: GraphTransformContext,
) -> tuple[list[torch.fx.Node], list[int], list[bool]]:
  """Extracts shape tensor nodes, static bounds, and dynamic flags for a target shape."""
  shape_tensors = []
  static_shape = []
  is_dynamic = []

  for arg in target_shape:
    if sym_utils.is_symint(arg):
      is_dynamic.append(True)
      static_shape.append(transform_context.get_upper_bound(arg))
      tensor_node = transform_context.ensure_tensor(
          graph_module, arg, node, dtype=torch.int32
      )
      shape_tensors.append(tensor_node)
    else:
      is_dynamic.append(False)
      val = (
          arg.meta["val"]
          if isinstance(arg, torch.fx.Node) and "val" in arg.meta
          else arg
      )
      assert isinstance(
          val, int
      ), f"Expected int for static shape arg, got {val}"
      static_shape.append(val)
      tensor_node = transform_context.ensure_tensor(
          graph_module, arg, node, dtype=torch.int32
      )
      shape_tensors.append(tensor_node)

  return shape_tensors, static_shape, is_dynamic


class HandleReshapeLikeOpsPass:
  """Reshape-like ops transformation pass.

  Detects reshape-like view operations (such as aten.view, aten.reshape,
  aten._unsafe_view) that have dynamic dimensions (SymInt or SymInt
  expressions) as input and replaces them with torch.ops.tpu._dynamic_reshape.
  """

  def __init__(self, transform_context: GraphTransformContext):
    self._transform_context = transform_context
    self._op_handlers = {
        torch.ops.aten.view.default: self._process_view_op,
        torch.ops.aten._unsafe_view.default: self._process_view_op,
        torch.ops.aten.reshape.default: self._process_view_op,
        torch.ops.aten.view_copy.default: self._process_view_op,
        torch.ops.aten._reshape_alias.default: self._process_view_op,
        torch.ops.aten._reshape_alias_copy.default: self._process_view_op,
    }

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    """Runs the reshape-like ops transformation pass."""
    for node in list(graph_module.graph.nodes):
      if node.op == "call_function":
        handler = self._op_handlers.get(node.target)
        if handler:
          handler(graph_module, node)

  def _process_view_op(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
  ) -> None:
    """Processes view/reshape op node and converts to dynamic_reshape if dynamic."""
    inp = node.args[0]
    shape = node.args[1]
    if not isinstance(shape, (list, tuple)):
      shape = [shape]

    target_shape = (
        node.meta["val"].shape
        if "val" in node.meta and hasattr(node.meta["val"], "shape")
        else shape
    )

    num_dynamic_dims = sum(
        1 for arg in target_shape if sym_utils.is_symint(arg)
    )
    inp_has_dynamic = sym_utils.has_dynamic_shape(inp)
    if num_dynamic_dims == 0 and not inp_has_dynamic:
      return

    shape_tensors, static_shape, is_dynamic = _extract_shape_tensors_and_bounds(
        graph_module, node, target_shape, self._transform_context
    )

    with graph_module.graph.inserting_after(node):
      dynamic_reshape_node = graph_module.graph.call_function(
          torch.ops.tpu._dynamic_reshape,
          args=(inp, shape_tensors, static_shape, is_dynamic),
      )
      dynamic_reshape_node.meta = node.meta.copy()

    node.replace_all_uses_with(dynamic_reshape_node)
    graph_module.graph.erase_node(node)


class HandleBroadcastLikeOpsPass:
  """Broadcast-like ops transformation pass.

  Detects broadcast operations (such as aten.expand, aten.expand_copy) that
  have dynamic dimensions (SymInt or SymInt expressions) as input and replaces
  them with torch.ops.tpu._dynamic_broadcast.
  """

  def __init__(self, transform_context: GraphTransformContext):
    self._transform_context = transform_context
    self._op_handlers = {
        torch.ops.aten.expand.default: self._process_broadcast_op,
        torch.ops.aten.expand_copy.default: self._process_broadcast_op,
        torch.ops.aten.broadcast_to.default: self._process_broadcast_op,
    }

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    """Runs the broadcast-like ops transformation pass."""
    for node in list(graph_module.graph.nodes):
      if node.op == "call_function":
        handler = self._op_handlers.get(node.target)
        if handler:
          handler(graph_module, node)

  def _process_broadcast_op(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
  ) -> None:
    """Processes broadcast op node and converts to dynamic_broadcast if dynamic."""
    inp = node.args[0]
    shape = node.args[1]
    if not isinstance(shape, (list, tuple)):
      shape = [shape]

    target_shape = (
        node.meta["val"].shape
        if "val" in node.meta and hasattr(node.meta["val"], "shape")
        else shape
    )

    num_dynamic_dims = sum(
        1 for arg in target_shape if sym_utils.is_symint(arg)
    )
    if num_dynamic_dims == 0:
      return

    if (
        isinstance(inp, torch.fx.Node)
        and "val" in inp.meta
        and hasattr(inp.meta["val"], "shape")
    ):
      inp_shape = inp.meta["val"].shape
    else:
      inp_shape = []
    r_in = len(inp_shape)
    r_out = len(target_shape)
    broadcast_dims = list(range(r_out - r_in, r_out))

    shape_tensors, static_shape, is_dynamic = _extract_shape_tensors_and_bounds(
        graph_module, node, target_shape, self._transform_context
    )

    with graph_module.graph.inserting_after(node):
      dynamic_broadcast_node = graph_module.graph.call_function(
          torch.ops.tpu._dynamic_broadcast,
          args=(inp, shape_tensors, broadcast_dims, static_shape, is_dynamic),
      )
      dynamic_broadcast_node.meta = node.meta.copy()

    node.replace_all_uses_with(dynamic_broadcast_node)
    graph_module.graph.erase_node(node)


class HandleSliceLikeOpsPass:
  """Slice-like ops transformation pass.

  Detects slice operations that operate on tensors with dynamic
  dimensions or have dynamic slice arguments (SymInt or SymInt expressions).

  If the sliced input dimension is dynamic, it bounds it to its static upper
  bound before slicing. It then lowers the slice op with static upper bounds and
  applies set_dimension_logical_size for all dynamic output dimensions based on
  the operation's output shape metadata.
  """

  def __init__(self, transform_context: GraphTransformContext):
    self._transform_context = transform_context
    self._op_handlers = {
        torch.ops.aten.slice.Tensor: self._process_slice_op,
        torch.ops.aten.slice_copy.Tensor: self._process_slice_op,
        torch.ops.aten.select.int: self._process_select_op,
        torch.ops.aten.select_copy.int: self._process_select_op,
        torch.ops.aten.slice_backward.default: self._process_slice_backward_op,
        torch.ops.aten.constant_pad_nd.default: (
            self._process_constant_pad_nd_op
        ),
    }

  def _is_negative(self, index: int | torch.SymInt) -> bool:
    """Returns whether an int or SymInt index is negative, without guards.

    PyTorch's view meta kernels decided the sign from the trace-time hint and
    guarded on it, so only the hint is consistent with the compiled graph
    (symbolic ranges are not refined for multi-symbol guards). Hint-less
    (unbacked) SymInts use their symbolic range and must not straddle zero.
    """
    hint = sym_utils.get_symint_hint(index)
    if hint is not None:
      return hint < 0
    if not isinstance(index, torch.SymInt):
      return False
    lower, upper = self._transform_context.get_symint_bounds(index)
    if lower < 0 <= upper:
      raise NotImplementedError(
          f"Slice index {index} is an unbacked SymInt whose sign is unknown"
          f" (range [{lower}, {upper}]), which is not supported on TPU."
      )
    return upper < 0

  def _normalize_index(
      self, index: int | torch.SymInt, dim_size: Any
  ) -> int | torch.SymInt:
    """Normalizes a negative int or SymInt index relative to dim_size.

    dim_size may be an int, a SymInt, or an FX node carrying one (e.g. the
    sym_size entries of slice_backward's input_sizes).
    """
    if self._is_negative(index):
      index = _get_node_val(dim_size) + index
    const_val = sym_utils.get_constant_symint_value(index)
    if const_val is not None:
      return const_val
    return index

  def _build_constant_tensor(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
      block_shape: Sequence[Any],
      value: Any,
      dtype: torch.dtype,
  ) -> torch.fx.Node:
    """Builds a constant tensor of block_shape using _dynamic_broadcast."""
    val_scalar = self._transform_context.ensure_tensor(
        graph_module, value, node, dtype=dtype
    )
    shape_tensors, static_shape, is_dynamic = _extract_shape_tensors_and_bounds(
        graph_module, node, block_shape, self._transform_context
    )
    with graph_module.graph.inserting_before(node):
      return graph_module.graph.call_function(
          torch.ops.tpu._dynamic_broadcast,
          args=(val_scalar, shape_tensors, [], static_shape, is_dynamic),
      )

  def _bound_dynamic_dims(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
      inp: torch.fx.Node,
      inp_val: Any,
  ) -> tuple[torch.fx.Node, list[int]]:
    """Bounds every dynamic dim of inp to its static upper bound.

    Returns the bounded node (inp itself when it has no dynamic dim) and the
    static upper-bound shape of inp.
    """
    curr = inp
    ub_shape = [
        self._transform_context.get_upper_bound(s) for s in inp_val.shape
    ]
    for d, s in enumerate(inp_val.shape):
      if not sym_utils.is_symint(s):
        continue
      ub_tensor = self._transform_context.ensure_tensor(
          graph_module,
          ub_shape[d],
          node,
          dtype=torch.int32,
      )
      with graph_module.graph.inserting_before(node):
        curr = graph_module.graph.call_function(
            torch.ops.tpu._set_dimension_logical_size,
            args=(curr, d, ub_tensor),
        )
        curr.meta = inp.meta.copy()
        if isinstance(inp_val, torch.Tensor):
          curr.meta["val"] = inp_val.new_empty(ub_shape)
    return curr, ub_shape

  def _prepare_slice_operand(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
      inp: torch.fx.Node,
      inp_val: Any,
      dim: int,
      start: int | torch.SymInt,
      slice_extent: int,
  ) -> torch.fx.Node:
    """Returns inp prepared as the operand of a bound-sized _dynamic_slice.

    Every dynamic dim of inp is first bounded to its static upper bound, so
    the slice window is taken from the padded buffer instead of being clamped
    against the runtime size. XLA still clamps dynamic_slice starts to
    [0, operand_dim - slice_size], so when upper_bound(start) + slice_extent
    exceeds upper_bound(inp_dim) inp is also right-padded along dim with a
    zero block of the missing extent. Only that case materializes a padded
    copy of inp (e.g. sliding-window masks `m[:, -q_len:]` with independent
    kv/q bounds); start-0 slices never pad.
    """
    curr, ub_shape = self._bound_dynamic_dims(graph_module, node, inp, inp_val)

    inp_dim_ub = ub_shape[dim]
    start_ub = max(0, self._transform_context.get_upper_bound(start))
    # `get_upper_bound` uses the aligned bucket capacity, which can exceed the
    # user-provided range in ShapeEnv and cause unnecessary padding. Clamp
    # `start_ub` back to the user-provided range when finite (`int_oo` when
    # unbounded).
    user_start_ub = self._transform_context.get_user_upper_bound(start)
    if isinstance(user_start_ub, int):
      start_ub = min(start_ub, max(0, user_start_ub))
    extra = start_ub + slice_extent - inp_dim_ub
    if slice_extent == 0 or extra <= 0:
      return curr
    # curr is already bounded to static upper bounds across all dims above.
    pad_shape = [extra if d == dim else ub for d, ub in enumerate(ub_shape)]
    pad_block = self._build_constant_tensor(
        graph_module, node, pad_shape, 0, inp_val.dtype
    )
    with graph_module.graph.inserting_before(node):
      padded = graph_module.graph.call_function(
          torch.ops.aten.cat.default,
          args=([curr, pad_block], dim),
      )
      padded.meta = inp.meta.copy()
      if isinstance(inp_val, torch.Tensor):
        padded.meta["val"] = inp_val.new_empty(
            [ub + extra if d == dim else ub for d, ub in enumerate(ub_shape)]
        )
    return padded

  def _prepare_slice_gradient(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
      grad: torch.fx.Node,
      grad_val: Any,
      dim: int,
      window: int,
  ) -> torch.fx.Node:
    """Returns grad prepared as the static gradient operand of slice_backward.

    The slice_backward lowering embeds its gradient with static geometry, so
    every dynamic dim of grad is bounded to its static upper bound like the
    operand of _prepare_slice_operand. The elements past the runtime size
    along dim are undefined and would leak into the gradient of the sliced
    input, so they are zeroed, and the bound along dim is cut to `window`
    elements when it exceeds the window (a bucket capacity above the size of
    the sliced input).
    """
    if not sym_utils.has_dynamic_shape(grad_val):
      return grad
    curr, ub_shape = self._bound_dynamic_dims(
        graph_module, node, grad, grad_val
    )
    if sym_utils.is_symint(grad_val.shape[dim]):
      size_tensor = self._transform_context.ensure_tensor(
          graph_module, grad_val.shape[dim], node, dtype=torch.int32
      )
      zero = self._transform_context.ensure_tensor(
          graph_module, 0, node, dtype=grad_val.dtype
      )
      mask_shape = [ub if d == dim else 1 for d, ub in enumerate(ub_shape)]
      # E.g. a [s, 8] grad with runtime s = 3 bounded to 4 rows: positions is
      # [0, 1, 2, 3], in_window is positions < 3 = [T, T, T, F] viewed as
      # [4, 1], so the where keeps rows 0-2 of grad and zeroes row 3.
      with graph_module.graph.inserting_before(node):
        positions = graph_module.graph.call_function(
            torch.ops.aten.arange.default,
            args=(ub_shape[dim],),
            kwargs={
                "dtype": torch.int32,
                "device": sym_utils.get_target_device(node),
            },
        )
        in_window = graph_module.graph.call_function(
            torch.ops.aten.lt.Tensor, args=(positions, size_tensor)
        )
        in_window = graph_module.graph.call_function(
            torch.ops.aten.view.default, args=(in_window, mask_shape)
        )
        curr = graph_module.graph.call_function(
            torch.ops.aten.where.self, args=(in_window, curr, zero)
        )
        curr.meta = grad.meta.copy()
        if isinstance(grad_val, torch.Tensor):
          curr.meta["val"] = grad_val.new_empty(ub_shape)
    if ub_shape[dim] > window:
      with graph_module.graph.inserting_before(node):
        curr = graph_module.graph.call_function(
            torch.ops.aten.slice.Tensor, args=(curr, dim, 0, window)
        )
        curr.meta = grad.meta.copy()
        if isinstance(grad_val, torch.Tensor):
          curr.meta["val"] = grad_val.new_empty(
              [window if d == dim else ub for d, ub in enumerate(ub_shape)]
          )
    return curr

  def _dynamic_output_dim_sizes(
      self, graph_module: torch.fx.GraphModule, node: torch.fx.Node
  ) -> list[tuple[int, torch.fx.Node]]:
    """Returns (dim, runtime size tensor) for every dynamic output dim."""
    out_val = node.meta.get("val")
    if not hasattr(out_val, "shape"):
      return []
    return [
        (
            d,
            self._transform_context.ensure_tensor(
                graph_module, s, node, dtype=torch.int32
            ),
        )
        for d, s in enumerate(out_val.shape)
        if sym_utils.is_symint(s)
    ]

  def _extract_select_args(
      self, node: torch.fx.Node, offset: int = 0
  ) -> tuple[int, Any]:
    """Extracts (dim, index) from select node args/kwargs."""
    dim = (
        node.args[1 + offset]
        if len(node.args) > 1 + offset
        else node.kwargs.get("dim", 0)
    )
    index = _get_node_val(
        node.args[2 + offset]
        if len(node.args) > 2 + offset
        else node.kwargs.get("index", 0)
    )
    return dim, index  # pyrefly: ignore[bad-return]

  def _extract_slice_args(
      self, node: torch.fx.Node, offset: int = 0
  ) -> tuple[int, Any, Any, Any]:
    """Extracts (dim, start, end, step) from slice or slice_backward node args/kwargs."""
    dim = (
        node.args[1 + offset]
        if len(node.args) > 1 + offset
        else node.kwargs.get("dim", 0)
    )
    start = _get_node_val(
        node.args[2 + offset]
        if len(node.args) > 2 + offset
        else node.kwargs.get("start", 0)
    )
    end = _get_node_val(
        node.args[3 + offset]
        if len(node.args) > 3 + offset
        else node.kwargs.get("end", None)
    )
    step = _get_node_val(
        node.args[4 + offset]
        if len(node.args) > 4 + offset
        else node.kwargs.get("step", 1)
    )
    return dim, start, end, step  # pyrefly: ignore[bad-return]

  def _replace_node_with_set_logical_sizes(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
      new_op_node: torch.fx.Node,
      dynamic_dim_sizes: Sequence[tuple[int, torch.fx.Node]],
  ) -> None:
    """Wraps new_op_node with set_dimension_logical_size for all dynamic dims, replaces all uses, and erases node."""
    curr_node = new_op_node
    with graph_module.graph.inserting_before(node):
      curr_node.meta = node.meta.copy()
      for dim_idx, size_tensor_node in dynamic_dim_sizes:
        curr_node = graph_module.graph.call_function(
            torch.ops.tpu._set_dimension_logical_size,
            args=(curr_node, dim_idx, size_tensor_node),
        )
        curr_node.meta = node.meta.copy()

    node.replace_all_uses_with(curr_node)
    graph_module.graph.erase_node(node)

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    """Runs the slice-like ops transformation pass."""
    for node in list(graph_module.graph.nodes):
      if node.op == "call_function":
        handler = self._op_handlers.get(node.target)
        if handler:
          handler(graph_module, node)

  def _process_slice_op(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
  ) -> None:
    """Lowers a slice on dynamic input / with dynamic parameters.

    The slice becomes tpu._dynamic_slice over the bound-sized operand (see
    _prepare_slice_operand) followed by _set_dimension_logical_size for every
    dynamic output dim. `end` only matters through the output shape metadata,
    so it is never normalized here.
    """
    inp = node.args[0]
    dim, start, end, step = self._extract_slice_args(node, offset=0)
    inp_val = _get_node_val(inp)
    out_val = _get_node_val(node)

    if start is None:
      start = 0

    if step is None:
      step = 1

    # Normalize negative dim and start indices to absolute indices.
    if hasattr(inp_val, "shape"):
      if isinstance(dim, int) and dim < 0:
        dim = dim + len(inp_val.shape)
      start = self._normalize_index(start, inp_val.shape[dim])
      # A start that is still negative after normalization (or can never be
      # positive) is clamped to 0, exactly like the slice meta kernel did.
      if (
          self._is_negative(start)
          or self._transform_context.get_upper_bound(start) <= 0
      ):
        start = 0

    start_is_dynamic = sym_utils.is_symint(start)
    end_is_dynamic = sym_utils.is_symint(end)
    step_is_dynamic = sym_utils.is_symint(step)
    inp_has_dynamic = sym_utils.has_dynamic_shape(inp_val)
    out_has_dynamic = sym_utils.has_dynamic_shape(out_val)

    if not (
        start_is_dynamic
        or end_is_dynamic
        or step_is_dynamic
        or inp_has_dynamic
        or out_has_dynamic
    ):
      return

    # Decide statically and never through bool(SymBool): this pass runs inside
    # the backend compiler, where evaluating a SymInt comparison installs a
    # Dynamo guard (unintended specialization and recompiles).
    if not sym_utils.is_statically_equal(step, 1):
      raise NotImplementedError(
          f"Slice operation '{node.name}' has non-unit step ({step}) on dynamic"
          " dimension which is not supported on TPU."
      )

    if (
        hasattr(inp_val, "shape")
        and hasattr(out_val, "shape")
        and isinstance(start, int)
        and start == 0
        and sym_utils.is_statically_equal(
            out_val.shape[dim], inp_val.shape[dim]
        )
    ):
      node.replace_all_uses_with(inp)
      graph_module.graph.erase_node(node)
      return

    rank = len(inp_val.shape)
    start_indices = [
        self._transform_context.ensure_tensor(
            graph_module, start if d == dim else 0, node, dtype=torch.int32
        )
        for d in range(rank)
    ]

    slice_sizes = [
        self._transform_context.get_upper_bound(out_val.shape[d])
        for d in range(rank)
    ]

    dynamic_dim_sizes = self._dynamic_output_dim_sizes(graph_module, node)
    fitted_inp = self._prepare_slice_operand(
        graph_module, node, inp, inp_val, dim, start, slice_sizes[dim]
    )

    with graph_module.graph.inserting_before(node):
      new_slice_node = graph_module.graph.call_function(
          torch.ops.tpu._dynamic_slice,
          args=(fitted_inp, start_indices, slice_sizes),
      )

    self._replace_node_with_set_logical_sizes(
        graph_module, node, new_slice_node, dynamic_dim_sizes
    )

  def _process_slice_backward_op(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
  ) -> None:
    """Lowers slice_backward with dynamic input sizes / parameters.

    Negative static starts and ends are normalized against the (possibly
    symbolic) input size; a start that stays symbolic is rejected because the
    lowering needs a static offset for the gradient window. The window ends
    at the user-provided/ShapeEnv upper bound of `end` (like the start in
    _prepare_slice_operand) and the gradient is embedded as a static operand
    of that window (see _prepare_slice_gradient).
    """
    grad_output, input_sizes = node.args[0], node.args[1]
    dim, start, end, step = self._extract_slice_args(node, offset=1)

    if isinstance(dim, int) and dim < 0 and isinstance(input_sizes, Sequence):
      dim = dim + len(input_sizes)
    if start is None:
      start = 0
    elif isinstance(input_sizes, Sequence):
      start = self._normalize_index(start, input_sizes[dim])

    if sym_utils.is_symint(start):
      raise NotImplementedError(
          f"Slice_backward operation '{node.name}' has dynamic start"
          f" ({start}) which is not yet supported on TPU."
      )

    has_dynamic = (
        sym_utils.is_symint(end)
        or sym_utils.is_symint(step)
        or any(sym_utils.is_symint(s) for s in input_sizes)  # pyrefly: ignore[not-iterable]
    )
    if not has_dynamic:
      return

    input_sizes_upper = [
        self._transform_context.get_upper_bound(s) for s in input_sizes  # pyrefly: ignore[not-iterable]
    ]
    step_upper = self._transform_context.get_upper_bound(step)
    if isinstance(input_sizes, Sequence):
      end = self._normalize_index(end, input_sizes[dim])
    end_upper = self._transform_context.get_upper_bound(end)
    # `get_upper_bound` uses the aligned bucket capacity, which can exceed the
    # user-provided range in ShapeEnv (`int_oo` when unbounded).
    user_end_ub = self._transform_context.get_user_upper_bound(end)
    if isinstance(user_end_ub, int):
      end_upper = min(end_upper, user_end_ub)
    window = max(0, min(end_upper, input_sizes_upper[dim]) - start)
    # Gradient elements covering the window along dim (ceil(window / step)).
    grad_window = -(-window // step_upper)
    grad_val = _get_node_val(grad_output)
    fitted_grad = self._prepare_slice_gradient(
        graph_module, node, grad_output, grad_val, dim, grad_window
    )

    dynamic_dim_sizes: list[tuple[int, torch.fx.Node]] = []
    for d, s in enumerate(input_sizes):  # pyrefly: ignore[not-iterable]
      if sym_utils.is_symint(s):
        size_tensor_node = self._transform_context.ensure_tensor(
            graph_module, s, node, dtype=torch.int32
        )
        dynamic_dim_sizes.append((d, size_tensor_node))

    with graph_module.graph.inserting_before(node):
      new_slice_bwd_node = graph_module.graph.call_function(
          torch.ops.aten.slice_backward.default,
          args=(
              fitted_grad,
              input_sizes_upper,
              dim,
              start,
              start + window,
              step_upper,
          ),
      )

    self._replace_node_with_set_logical_sizes(
        graph_module, node, new_slice_bwd_node, dynamic_dim_sizes
    )

  def _process_select_op(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
  ) -> None:
    """Processes select op node and converts to dynamic_slice + squeeze."""
    inp = node.args[0]
    dim, index = self._extract_select_args(node, offset=0)
    inp_val = _get_node_val(inp)
    out_val = _get_node_val(node)

    # Normalize negative dim and index to absolute coordinates.
    if hasattr(inp_val, "shape"):
      if isinstance(dim, int) and dim < 0:
        dim = dim + len(inp_val.shape)
      index = self._normalize_index(index, inp_val.shape[dim])

    inp_has_dynamic = sym_utils.has_dynamic_shape(inp_val)
    out_has_dynamic = sym_utils.has_dynamic_shape(out_val)
    index_is_dynamic = sym_utils.is_symint(index)

    if not (inp_has_dynamic or out_has_dynamic or index_is_dynamic):
      return

    rank = len(inp_val.shape)
    start_indices = [
        self._transform_context.ensure_tensor(
            graph_module, index if d == dim else 0, node, dtype=torch.int32
        )
        for d in range(rank)
    ]

    slice_sizes = [
        1
        if d == dim
        else self._transform_context.get_upper_bound(inp_val.shape[d])
        for d in range(rank)
    ]

    dynamic_dim_sizes = self._dynamic_output_dim_sizes(graph_module, node)
    fitted_inp = self._prepare_slice_operand(
        graph_module, node, inp, inp_val, dim, index, 1
    )

    with graph_module.graph.inserting_before(node):
      new_slice_node = graph_module.graph.call_function(
          torch.ops.tpu._dynamic_slice,
          args=(fitted_inp, start_indices, slice_sizes),
      )
      new_select_node = graph_module.graph.call_function(
          torch.ops.aten.squeeze.dim,
          args=(new_slice_node, dim),
      )

    self._replace_node_with_set_logical_sizes(
        graph_module, node, new_select_node, dynamic_dim_sizes
    )

  def _process_constant_pad_nd_op(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
  ) -> None:
    """Lowers constant_pad_nd on dynamic tensors to dynamic_broadcast + cat.

    Each padded dim is concatenated with constant blocks sized from the static
    upper bounds, and the dynamic output dims are re-bounded with
    set_dimension_logical_size. Only non-negative static pads are supported on
    dynamic tensors.
    """
    inp = node.args[0]
    pad = node.args[1] if len(node.args) > 1 else node.kwargs.get("pad", ())
    value = (
        node.args[2] if len(node.args) > 2 else node.kwargs.get("value", 0.0)
    )
    if value is None:
      value = 0.0
    inp_val = _get_node_val(inp)
    out_val = _get_node_val(node)

    if not hasattr(inp_val, "shape") or not isinstance(pad, Sequence):
      return
    if not (
        sym_utils.has_dynamic_shape(inp_val)
        or sym_utils.has_dynamic_shape(out_val)
    ):
      return

    pad_vals = [_get_node_val(p) for p in pad]
    if any(sym_utils.is_symint(p) or p < 0 for p in pad_vals):
      raise NotImplementedError(
          f"constant_pad_nd operation '{node.name}' has negative or dynamic"
          f" pads ({pad_vals}) on a tensor with dynamic dimensions, which is"
          " not supported on TPU. Slice the tensor instead of using negative"
          " pads, or keep the pad amounts static and non-negative."
      )

    cur = inp
    cur_shape = list(inp_val.shape)
    rank = len(cur_shape)
    num_pad_dims = len(pad_vals) // 2

    for i in range(num_pad_dims):
      dim = rank - 1 - i
      left_pad = int(pad_vals[2 * i])
      right_pad = int(pad_vals[2 * i + 1])
      if left_pad == 0 and right_pad == 0:
        continue

      parts: list[torch.fx.Node] = []
      if left_pad > 0:
        left_shape = list(cur_shape)
        left_shape[dim] = left_pad
        parts.append(
            self._build_constant_tensor(
                graph_module, node, left_shape, value, inp_val.dtype
            )
        )
      parts.append(cur)
      if right_pad > 0:
        right_shape = list(cur_shape)
        right_shape[dim] = right_pad
        parts.append(
            self._build_constant_tensor(
                graph_module, node, right_shape, value, inp_val.dtype
            )
        )

      with graph_module.graph.inserting_before(node):
        cur = graph_module.graph.call_function(
            torch.ops.aten.cat.default,
            args=(parts, dim),
        )
      cur_shape[dim] = cur_shape[dim] + left_pad + right_pad

    self._replace_node_with_set_logical_sizes(
        graph_module,
        node,
        cur,
        self._dynamic_output_dim_sizes(graph_module, node),
    )


_SLICE_LIKE_FUNCTIONS = (
    operator.getitem,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.slice,
    torch.ops.aten.select.int,
    torch.ops.aten.select,
    torch.select,
    torch.ops.aten.narrow.default,
    torch.ops.aten.narrow,
    torch.narrow,
)
_SLICE_LIKE_METHODS = ("slice", "select", "narrow")


def _has_symbolic_index_args(node: torch.fx.Node) -> bool:
  """Returns whether any slice/select/narrow index argument of node is dynamic.

  In the Dynamo graph dynamic indices are either SymInts or FX nodes producing
  them (e.g. the output of `sym_size`); static indices are plain ints/None.
  """
  flat_idx_args, _ = pytree.tree_flatten((node.args[1:], node.kwargs))
  for item in flat_idx_args:
    parts = (
        (item.start, item.stop, item.step)
        if isinstance(item, slice)
        else (item,)
    )
    if any(isinstance(p, (torch.fx.Node, torch.SymInt)) for p in parts):
      return True
  return False


class ReplaceDynamicOutputBroadcastOpsPreGradPass:
  """Pre-grad pass turning returned dynamic views into explicit copies.

  AOTAutograd regenerates graph outputs that alias an input or an intermediate
  from their base at runtime (`gen_alias_from_base`). That only works through
  view replay, which AOTAutograd skips as soon as a view in the chain has
  symbolic (SymInt) arguments; it then falls back to `as_strided` with the
  metadata of the tensor the compiled graph produced. On TPU that tensor is a
  fresh contiguous buffer, so the fallback either reads the wrong window of
  the base (dynamic slice/select/narrow indices) or fails the storage size
  check (expand/broadcast_to to a dynamic shape).

  Before AOTAutograd runs, this pass therefore rewrites every returned
  `expand`/`broadcast_to` with a dynamic target shape to `expand_copy`, and
  clones every returned slice/select/narrow view with dynamic index arguments
  right before the `output` node (so in-graph mutations of the base after the
  view are still reflected). Those outputs become copies instead of aliases
  of their base; all other nodes are left as they are.
  """

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    """Runs the pre-grad pass to replace returned dynamic broadcast ops."""
    logging.debug(
        "[PreGrad] ReplaceDynamicOutputBroadcastOpsPreGradPass running on"
        " graph module: %s",
        graph_module,
    )
    output_nodes = list(graph_module.graph.find_nodes(op="output"))
    if not output_nodes:
      logging.debug("[PreGrad] No output nodes found")
      return

    returned_nodes: set[torch.fx.Node] = set()
    for out_node in output_nodes:
      flat_args, _ = pytree.tree_flatten(out_node.args)
      for arg in flat_args:
        if isinstance(arg, torch.fx.Node):
          returned_nodes.add(arg)

    logging.debug("[PreGrad] returned_nodes: %s", returned_nodes)
    changed = False
    for node in list(graph_module.graph.nodes):
      is_expand = (
          node.op == "call_function"
          and node.target
          in (
              torch.ops.aten.expand.default,
              torch.ops.aten.expand,
              "expand",
              torch.ops.aten.broadcast_to.default,
              torch.ops.aten.broadcast_to,
              torch.broadcast_to,
              "broadcast_to",
          )
      ) or (
          node.op == "call_method" and node.target in ("expand", "broadcast_to")
      )
      is_slice_like = (
          node.op == "call_function" and node.target in _SLICE_LIKE_FUNCTIONS
      ) or (node.op == "call_method" and node.target in _SLICE_LIKE_METHODS)
      if not (is_expand or is_slice_like):
        continue

      is_returned = node in returned_nodes
      if is_slice_like:
        # Only views are regenerated by AOTAutograd; e.g. advanced indexing
        # already returns a copy and needs no clone.
        example_value = node.meta.get("example_value")
        if not (
            is_returned
            and isinstance(example_value, torch.Tensor)
            and example_value._is_view()  # pylint: disable=protected-access
            and _has_symbolic_index_args(node)
        ):
          continue
        for out_node in output_nodes:
          with graph_module.graph.inserting_before(out_node):
            # contiguous_format: a preserve_format clone would evaluate the
            # view's contiguity (size/stride relations of SymInts) in the meta
            # kernel and install guards that conflict with the upper bounds.
            clone_node = graph_module.graph.call_function(
                torch.ops.aten.clone.default,
                args=(node,),
                kwargs={"memory_format": torch.contiguous_format},
            )
            clone_node.meta = node.meta.copy()
          out_node.replace_input_with(node, clone_node)
        changed = True
        continue

      # Extract input tensor node (`inp`) and target shape dimensions (`shape`).
      # Method calls (e.g. `x.expand(d0, d1)` or `x.expand([d0, d1])`) pass
      # `self` as `node.args[0]`, followed by shape as varargs or sequence.
      if node.op == "call_method":
        inp = node.args[0]
        if len(node.args) == 2 and isinstance(node.args[1], (list, tuple)):
          shape = list(node.args[1])
        else:
          shape = list(node.args[1:])
      # Function calls (e.g. `torch.broadcast_to(x, size)` or
      # `aten.expand(x, size)`) pass input tensor `x` as `node.args[0]` and
      # shape via `node.args[1]` or `node.kwargs["size"]`.
      else:
        inp = node.args[0]
        shape = (
            node.args[1] if len(node.args) > 1 else node.kwargs.get("size", [])
        )
        if not isinstance(shape, (list, tuple)):
          shape = [shape]

      has_symint = any(
          isinstance(arg, torch.fx.Node)
          or isinstance(arg, torch.SymInt)
          or sym_utils.is_symint_node(arg)
          for arg in shape
      )

      if is_returned and has_symint:
        with graph_module.graph.inserting_after(node):
          expand_copy_node = graph_module.graph.call_function(
              torch.ops.aten.expand_copy.default,
              args=(inp, shape),
              kwargs=node.kwargs,
          )
          expand_copy_node.meta = node.meta.copy()

        node.replace_all_uses_with(expand_copy_node)
        # Erase the node directly: `eliminate_dead_code` would also delete
        # Dynamo `call_method` in-place mutations (e.g. `add_`) without users.
        graph_module.graph.erase_node(node)
        changed = True
        logging.debug(
            "[PreGrad] Successfully replaced %s with expand_copy_node %s",
            node,
            expand_copy_node,
        )

    if changed:
      graph_module.recompile()
