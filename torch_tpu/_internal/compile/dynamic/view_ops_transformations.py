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
from typing import Any
from absl import logging
import torch
import torch.utils._pytree as pytree
from torch_tpu._internal.compile.dynamic import sym_utils
from torch_tpu._internal.compile.dynamic import symbol_bounds
from torch_tpu._internal.compile.dynamic.sym_shape_manager import SymShapeManager


def _extract_shape_tensors_and_bounds(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
    target_shape: Sequence[Any],
    sym_shape_manager: SymShapeManager,
) -> tuple[list[torch.fx.Node], list[int], list[bool]]:
  """Extracts shape tensor nodes, static bounds, and dynamic flags for a target shape."""
  shape_tensors = []
  static_shape = []
  is_dynamic = []

  for arg in target_shape:
    if sym_utils.is_symint_node(arg) or isinstance(arg, torch.SymInt):
      is_dynamic.append(True)
      symint = arg.meta["val"] if sym_utils.is_symint_node(arg) else arg
      _, upper = symbol_bounds.get_symint_bounds(symint)
      static_shape.append(upper)
      tensor_node = sym_shape_manager.ensure_tensor(
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
      tensor_node = sym_shape_manager.ensure_tensor(
          graph_module, arg, node, dtype=torch.int32
      )
      shape_tensors.append(tensor_node)

  return shape_tensors, static_shape, is_dynamic


def _ensure_input_dynamic_sizes(
    graph_module: torch.fx.GraphModule,
    inp: torch.fx.Node,
    sym_shape_manager: SymShapeManager,
    insert_node: torch.fx.Node,
) -> torch.fx.Node:
  """Ensures all dynamic dimensions of input tensor node have explicit set_dimension_logical_size calls."""
  if not isinstance(inp, torch.fx.Node) or "val" not in inp.meta:
    return inp
  if (
      inp.op == "call_function"
      and inp.target == torch.ops.tpu.set_dimension_logical_size
  ):
    return inp
  val = inp.meta["val"]
  if not hasattr(val, "shape"):
    return inp

  current_inp = inp
  for dim, size_dim in enumerate(val.shape):
    if sym_utils.is_symint_node(size_dim) or isinstance(size_dim, torch.SymInt):
      size_tensor_node = sym_shape_manager.ensure_tensor(
          graph_module, size_dim, insert_node, dtype=torch.int32
      )
      with graph_module.graph.inserting_before(insert_node):
        current_inp = graph_module.graph.call_function(
            torch.ops.tpu.set_dimension_logical_size,
            args=(current_inp, dim, size_tensor_node),
        )
        current_inp.meta = inp.meta.copy()

  return current_inp


class HandleReshapeLikeOpsPass:
  """Reshape-like ops transformation pass.

  Detects reshape-like view operations (such as aten.view, aten.reshape,
  aten._unsafe_view) that have dynamic dimensions (SymInt or SymInt
  expressions) as input and replaces them with torch.ops.tpu.dynamic_reshape.
  """

  def __init__(self, sym_shape_manager: SymShapeManager):
    self._sym_shape_manager = sym_shape_manager
    self._op_handlers = {
        torch.ops.aten.view.default: self._process_view_op,
        torch.ops.aten._unsafe_view.default: self._process_view_op,
        torch.ops.aten.reshape.default: self._process_view_op,
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
        1
        for arg in target_shape
        if sym_utils.is_symint_node(arg) or isinstance(arg, torch.SymInt)
    )
    if num_dynamic_dims == 0:
      return

    # Skip dynamic_reshape if dynamic dimensions are in identical positions and expressions
    if (
        isinstance(inp, torch.fx.Node)
        and "val" in inp.meta
        and hasattr(inp.meta["val"], "shape")
    ):
      inp_shape = inp.meta["val"].shape
      inp_dyn_dims = [
          (i, s)
          for i, s in enumerate(inp_shape)
          if sym_utils.is_symint_node(s) or isinstance(s, torch.SymInt)
      ]
      target_dyn_dims = [
          (i, s)
          for i, s in enumerate(target_shape)
          if sym_utils.is_symint_node(s) or isinstance(s, torch.SymInt)
      ]
      if (
          len(inp_dyn_dims) == len(target_dyn_dims)
          and all(
              i_idx == t_idx and str(i_s) == str(t_s)
              for (i_idx, i_s), (t_idx, t_s) in zip(inp_dyn_dims, target_dyn_dims)
          )
      ):
        # Replace SymInt arguments in shape list with static upper bound integers
        new_shape_args = []
        for arg in target_shape:
          if sym_utils.is_symint_node(arg) or isinstance(arg, torch.SymInt):
            symint = arg.meta["val"] if sym_utils.is_symint_node(arg) else arg
            _, upper = symbol_bounds.get_symint_bounds(symint)
            new_shape_args.append(upper)
          elif isinstance(arg, torch.fx.Node) and "val" in arg.meta:
            new_shape_args.append(int(arg.meta["val"]))
          else:
            new_shape_args.append(int(arg))

        if len(node.args) > 1:
          node.args = (node.args[0], new_shape_args) + node.args[2:]
        return

    inp = _ensure_input_dynamic_sizes(
        graph_module, inp, self._sym_shape_manager, node
    )

    shape_tensors, static_shape, is_dynamic = _extract_shape_tensors_and_bounds(
        graph_module, node, target_shape, self._sym_shape_manager
    )

    with graph_module.graph.inserting_after(node):
      dynamic_reshape_node = graph_module.graph.call_function(
          torch.ops.tpu.dynamic_reshape,
          args=(inp, shape_tensors, static_shape, is_dynamic),
      )
      dynamic_reshape_node.meta = node.meta.copy()

    node.replace_all_uses_with(dynamic_reshape_node)
    graph_module.graph.erase_node(node)


class HandleBroadcastLikeOpsPass:
  """Broadcast-like ops transformation pass.

  Detects broadcast operations (such as aten.expand, aten.expand_copy) that
  have dynamic dimensions (SymInt or SymInt expressions) as input and replaces
  them with torch.ops.tpu.dynamic_broadcast.
  """

  def __init__(self, sym_shape_manager: SymShapeManager):
    self._sym_shape_manager = sym_shape_manager
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
        1
        for arg in target_shape
        if sym_utils.is_symint_node(arg) or isinstance(arg, torch.SymInt)
    )
    if num_dynamic_dims == 0:
      return

    inp = _ensure_input_dynamic_sizes(
        graph_module, inp, self._sym_shape_manager, node
    )

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
        graph_module, node, target_shape, self._sym_shape_manager
    )

    with graph_module.graph.inserting_after(node):
      dynamic_broadcast_node = graph_module.graph.call_function(
          torch.ops.tpu.dynamic_broadcast,
          args=(inp, shape_tensors, broadcast_dims, static_shape, is_dynamic),
      )
      dynamic_broadcast_node.meta = node.meta.copy()

    node.replace_all_uses_with(dynamic_broadcast_node)
    graph_module.graph.erase_node(node)


class ReplaceDynamicOutputBroadcastOpsPreGradPass:
  """Pre-grad pass to replace returned dynamic broadcast/expand ops with expand_copy.

  Why this pass is needed:
  When a model returns a broadcast or expand operation directly as a graph
  output (e.g. `return x.expand(dynamic_size, ...)` or `return
  torch.broadcast_to(...)`) and the target shape depends on dynamic SymInt
  expressions, returning a view op causes PyTorch's AOTAutograd alias
  resolution to fail. AOTAutograd attempts to generate an alias reconstruction
  function using `as_strided`, but the original input tensor storage size is
  smaller than the expanded output shape causing a runtime storage size
  validation error.

  By replacing returned dynamic broadcast/expand nodes with
  `aten.expand_copy.default` before AOTAutograd runs (Pre-Grad), the returned
  tensor becomes an explicit contiguous copy rather than an aliased view,
  allowing AOTAutograd to handle the output safely.
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
      if not is_expand:
        continue

      is_returned = node in returned_nodes
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
        changed = True
        logging.debug(
            "[PreGrad] Successfully replaced %s with expand_copy_node %s",
            node,
            expand_copy_node,
        )

    if changed:
      graph_module.graph.eliminate_dead_code()
      graph_module.recompile()
