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

"""Graph transformation passes for handling dynamic shapes."""

from __future__ import annotations
from collections.abc import Sequence
import operator
import re
from typing import Any
import torch
from torch.fx.passes import graph_transform_observer
from torch.fx.passes import tools_common
from torch.utils import _pytree
from torch_tpu._internal.compile.dynamic import generative_ops_transformation
from torch_tpu._internal.compile.dynamic import sym_utils
from torch_tpu._internal.compile.dynamic import symbol_bounds
from torch_tpu._internal.compile.dynamic import view_ops_transformations
from torch_tpu._internal.compile.dynamic.sym_shape_manager import SymShapeManager

GraphTransformObserver = graph_transform_observer.GraphTransformObserver
get_symint_bounds = symbol_bounds.get_symint_bounds
HandleGenerativeOpsPass = generative_ops_transformation.HandleGenerativeOpsPass


class HandleDynamicInputTensorPass:
  """Dynamic input tensor transformation pass.

  This pass does the following:
  - Identifies input tensors with dynamic dimensions (tensors with SymInt in
    their shape).
  - Replaces all usage of the input tensor with the output of a chain of
    `set_dimension_logical_size` ops, each operating on the output of the
    previous one, and taking the new placeholder as the size input.
  - Modifies the graph in-place.

  Note: set_dimension_logical_size lowers down to
    stablehlo.set_dimension_size.

  Example:
    Model:
      def forward(x: torch.Tensor):
          z = x + 10
          return z

    Input FX Graph:
      def forward(self, arg0_1: "Sym(s27)", s27_size: "i32[]",
                 arg1_1: "Sym(s53)", s53_size: "i32[]",
                 arg2_1: "i64[1, s27, s53]"):
          add: "f32[1, s27, s53]" = torch.ops.aten.add.Tensor(arg2_1, 10)
          return (add,)

    Modified FX Graph:
      def forward(
          self,
          arg0_1: "Sym(s27)",
          s27_size: "i32[]",
          arg1_1: "Sym(s53)",
          s53_size: "i32[]",
          arg2_1: "i64[1, s27, s53]",
      ):
        set_dimension_logical_size_1: "i64[1, s27, s53]" =
            torch.ops.tpu.set_dimension_logical_size(
                arg2_1, 1, s27_size)

        set_dimension_logical_size_2: "i64[1, s27, s53]" =
            torch.ops.tpu.set_dimension_logical_size(
                set_dimension_logical_size_1, 2, s53_size)

        add: "f32[1, s27, s53]" = torch.ops.aten.add.Tensor(
            set_dimension_logical_size_2, 10)

        return (add,)
  """

  def __init__(
      self,
      sym_shape_manager: SymShapeManager,
      placeholders: list[torch.fx.Node],
  ):
    self._sym_shape_manager = sym_shape_manager
    self._placeholders = placeholders

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    """Runs the op insertion pass."""

    for idx, node in enumerate(self._placeholders):
      tensor_metadata = self._sym_shape_manager.input_tensors_metadata.get(idx)
      if tensor_metadata is None:
        continue

      self._process_node(graph_module, node, tensor_metadata.dynamic_dims)

  def _process_node(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
      dynamic_dims: Sequence[int],
  ) -> None:
    """Processes a single placeholder node and inserts set_dimension_size chains."""
    original_users = set(node.users.keys())
    current_tensor_node = node

    for dim in dynamic_dims:
      symint = node.meta["val"].shape[dim]
      size_tensor_node = self._sym_shape_manager.ensure_tensor(
          graph_module, symint, node
      )
      set_dim_size_node = self._insert_set_dimension_logical_size_node(
          graph_module, current_tensor_node, size_tensor_node, dim, node.meta
      )
      current_tensor_node = set_dim_size_node

    node.replace_all_uses_with(
        current_tensor_node, delete_user_cb=lambda u: u in original_users
    )

  def _insert_set_dimension_logical_size_node(
      self,
      graph_module: torch.fx.GraphModule,
      current_tensor_node: torch.fx.Node,
      size_tensor_node: torch.fx.Node,
      dim: int,
      meta: dict[str, Any],
  ) -> torch.fx.Node:
    """Inserts a set_dimension_logical_size node after the current tensor node."""
    with graph_module.graph.inserting_after(current_tensor_node):
      set_dim_size_node = graph_module.graph.call_function(
          torch.ops.tpu.set_dimension_logical_size,
          args=(current_tensor_node, dim, size_tensor_node),
      )
      set_dim_size_node.meta = meta.copy()
    return set_dim_size_node


class ScanInputsCreatePlaceholdersPass:
  """Pass to scan inputs and create placeholders for SymInt inputs.

  This pass scans the inputs and if an input is a SymInt, it creates a new
  placeholder for it.
  """

  def __init__(
      self,
      sym_shape_manager: SymShapeManager,
      placeholders: Sequence[torch.fx.Node],
  ):
    self._sym_shape_manager = sym_shape_manager
    self._placeholders = placeholders

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    """Runs the create placeholders pass."""
    seen_counts: dict[str, int] = {}
    for node in self._placeholders:
      if sym_utils.is_symint_node(node):
        sym_str = str(node.meta["val"])
        sanitized_sym_str = re.sub(r"\W+", "_", sym_str).strip("_")
        count = seen_counts.get(sym_str, 0)
        seen_counts[sym_str] = count + 1
        ph_name = (
            f"{sanitized_sym_str}_size"
            if count == 0
            else f"{sanitized_sym_str}_size_{count}"
        )

        # Create a new placeholder next to it (always create to keep
        # signature match).
        with graph_module.graph.inserting_after(node):
          size_ph = graph_module.graph.placeholder(
              ph_name, type_expr=torch.Tensor
          )
          size_ph.meta["val"] = torch.empty((), dtype=torch.int32)

        # Store the first placeholder encountered for this sym_str.
        first_size_ph = (
            self._sym_shape_manager.symint_to_placeholder.setdefault(
                sym_str, size_ph
            )
        )
        self._sym_shape_manager.symint_node_to_tensor_node[node] = first_size_ph


class HandleSymIntUsagesPass:
  """Usages of SymInt nodes in tensor operations and graph outputs transformation pass.

  When a SymInt node (e.g., an input symbol representing a dynamic size) is
  consumed by a standard data computational operator (like `aten.add.Tensor`) or
  returned in the graph output node, we replace that usage with the
  corresponding runtime size placeholder tensor node or expression tensor node.
  """

  def __init__(self, sym_shape_manager: SymShapeManager):
    self._sym_shape_manager = sym_shape_manager

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    for node in list(graph_module.graph.nodes):
      if (
          node.op == "call_function"
          and isinstance(node.target, torch._ops.OpOverload)
          and "SymInt" not in str(node.target)
          and "sym_size" not in str(node.target)
      ):
        new_args = []
        changed = False
        for arg in node.args:
          if sym_utils.is_symint_node(arg):
            tensor_node = self._sym_shape_manager.ensure_tensor(
                graph_module, arg, node
            )
            new_args.append(tensor_node)
            changed = True
          else:
            new_args.append(arg)

        if changed:
          node.args = tuple(new_args)
      elif node.op == "output":
        if node.args and isinstance(node.args[0], (tuple, list)):
          new_ret_args = []
          changed = False
          symint_output_indices = []
          for idx, arg in enumerate(node.args[0]):
            if sym_utils.is_symint_node(arg):
              symint_output_indices.append(idx)
              tensor_node = self._sym_shape_manager.ensure_tensor(
                  graph_module, arg, node
              )
              # Determine original target dtype from node metadata
              val_meta = arg.meta.get("val")
              target_dtype = getattr(val_meta, "dtype", torch.int64)
              current_dtype = getattr(
                  tensor_node.meta.get("val"), "dtype", None
              )

              if target_dtype is not None and current_dtype != target_dtype:
                with graph_module.graph.inserting_before(node):
                  cast_node = graph_module.graph.call_method(
                      "to", args=(tensor_node,), kwargs={"dtype": target_dtype}
                  )
                  cast_node.meta = tensor_node.meta.copy()
                  cast_node.meta["val"] = torch.empty((), dtype=target_dtype)
                  tensor_node = cast_node

              new_ret_args.append(tensor_node)
              changed = True
            else:
              new_ret_args.append(arg)

          if changed:
            node.args = (type(node.args[0])(new_ret_args),)
            self._sym_shape_manager.set_symint_output_indices(
                symint_output_indices
            )


_PYTHON_COMPARISON_BUILTINS = {
    operator.eq,
    operator.ne,
    operator.lt,
    operator.gt,
    operator.le,
    operator.ge,
}


def _is_assertion_or_check_node(node: torch.fx.Node) -> bool:
  if node.op == "call_function":
    if node.target in _PYTHON_COMPARISON_BUILTINS:
      return True
    target_str = str(node.target)
    if "_assert" in target_str or "sym_constrain" in target_str:
      return True
  return False


class DetectSymIntUsagesPass:
  """Pass to detect any remaining SymInt usages in the graph after transformations.

  If any FX node in the graph still consumes a SymInt input argument, this pass
  returns a list of unhandled (node, symint_arg) usages. Assertion and check
  nodes (e.g., inserted by torch._check) are excluded.
  """

  def __call__(
      self, graph_module: torch.fx.GraphModule
  ) -> list[tuple[torch.fx.Node, Any]]:
    unhandled_usages = []
    for node in graph_module.graph.nodes:
      if _is_assertion_or_check_node(node):
        continue
      flat_args, _ = _pytree.tree_flatten((node.args, node.kwargs))
      for arg in flat_args:
        if sym_utils.is_symint(arg):
          unhandled_usages.append((node, arg))

    return unhandled_usages


def apply_dynamism_transformations(
    graph_module: torch.fx.GraphModule, sym_shape_manager: SymShapeManager
) -> None:
  """Runs all FX graph transforms for dynamic shapes."""

  # Fetch original placeholders once in argument order
  original_placeholders = list(
      graph_module.graph.find_nodes(op="placeholder", sort=True)
  )

  # Scan inputs and create placeholders for SymInt inputs.
  GraphTransformObserver(
      graph_module, "scan_inputs_create_placeholders"
  ).apply_gm_pass(
      ScanInputsCreatePlaceholdersPass(sym_shape_manager, original_placeholders)
  )

  # Updates ops that have input tensors with dynamic dimensions.
  GraphTransformObserver(graph_module, "handle_dynamic_inputs").apply_gm_pass(
      HandleDynamicInputTensorPass(sym_shape_manager, original_placeholders)
  )

  # Updates the generative ops that have dynamic scalar inputs.
  GraphTransformObserver(graph_module, "handle_generative_ops").apply_gm_pass(
      HandleGenerativeOpsPass(sym_shape_manager)
  )

  # Updates view ops that are reshape-like and have dynamic inputs.
  GraphTransformObserver(graph_module, "handle_reshape_like_ops").apply_gm_pass(
      view_ops_transformations.HandleReshapeLikeOpsPass(sym_shape_manager)
  )

  # Updates broadcast operations that have dynamic inputs.
  GraphTransformObserver(
      graph_module, "handle_broadcast_like_ops"
  ).apply_gm_pass(
      view_ops_transformations.HandleBroadcastLikeOpsPass(sym_shape_manager)
  )

  # Updates slice operations that have dynamic inputs.
  GraphTransformObserver(graph_module, "handle_slice_like_ops").apply_gm_pass(
      view_ops_transformations.HandleSliceLikeOpsPass(sym_shape_manager)
  )

  # Replaces remaining usages of SymInt nodes in standard tensor operations.
  GraphTransformObserver(graph_module, "handle_symint_usages").apply_gm_pass(
      HandleSymIntUsagesPass(sym_shape_manager)
  )

  tools_common.stable_topological_sort(graph_module)
  graph_module.graph.eliminate_dead_code()
  graph_module.recompile()
  graph_module.graph.lint()

  # Detect any remaining unhandled symint usages in the graph.
  unhandled_usages = GraphTransformObserver(
      graph_module, "detect_symint_usages"
  ).apply_gm_pass(DetectSymIntUsagesPass())

  if unhandled_usages:
    err_msg_lines = [
        "Detected following unhandled SymInt usage(s) in FX graph. Please set"
        " torch.compile(..., dynamic=False, ...) and try again."
    ]
    for node, sym_arg in unhandled_usages:
      err_msg_lines.append(
          f"  - Node '{node.name}' (op={node.op!r}, target={node.target})"
          f" uses SymInt input '{sym_arg}'"
      )
    raise NotImplementedError("\n".join(err_msg_lines))
