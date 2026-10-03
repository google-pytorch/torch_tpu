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
from torch._inductor.utils import InputType
from torch.fx.passes import graph_transform_observer
from torch.fx.passes import tools_common
from torch.utils import _pytree
from torch_tpu._internal.compile.dynamic import convert_constant_symints_pass
from torch_tpu._internal.compile.dynamic import dynamic_view_pass
from torch_tpu._internal.compile.dynamic import generative_ops_pass
from torch_tpu._internal.compile.dynamic import sym_utils
from torch_tpu._internal.compile.dynamic import view_ops_passes
from torch_tpu._internal.compile.dynamic.graph_transform_context import GraphTransformContext
from torch_tpu._internal.compile.dynamic.symbol_bounds import SymbolBounds

ConvertConstantSymIntsPass = (
    convert_constant_symints_pass.ConvertConstantSymIntsPass
)
GraphTransformObserver = graph_transform_observer.GraphTransformObserver
HandleGenerativeOpsPass = generative_ops_pass.HandleGenerativeOpsPass


class HandleDynamicInputTensorPass:
  """Dynamic input tensor transformation pass using pad_real_to_static.

  Identifies input tensors with dynamic dimensions (SymInts in shape) and
  replaces their usages with the output of `torch.ops.tpu._pad_real_to_static`.
  """

  def __init__(
      self,
      transform_context: GraphTransformContext,
      placeholders: list[torch.fx.Node],
  ):
    self._transform_context = transform_context
    self._placeholders = placeholders

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    """Runs the op insertion pass."""
    for node in self._placeholders:
      val = node.meta.get("val")
      if not isinstance(val, torch.Tensor):
        continue
      target_shape = val.shape
      is_dynamic = [sym_utils.is_symint(d) for d in target_shape]
      if not any(is_dynamic):
        continue

      bounded_shape = [
          self._transform_context.get_upper_bound(d) for d in target_shape
      ]
      self._process_node(
          graph_module,
          node,
          bounded_shape,
          is_dynamic,
      )

  def _process_node(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
      bounded_shape: list[int],
      is_dynamic: list[bool],
  ) -> None:
    """Processes a single placeholder node and inserts pad_real_to_static."""
    original_users = set(node.users.keys())
    target_shape = node.meta["val"].shape
    rank = len(target_shape)

    shape_tensors = []
    for dim in range(rank):
      dim_arg = target_shape[dim]
      size_tensor_node = self._transform_context.ensure_tensor(
          graph_module, dim_arg, node, dtype=torch.int32
      )
      shape_tensors.append(size_tensor_node)

    with graph_module.graph.inserting_after(node):
      pad_node = graph_module.graph.call_function(
          torch.ops.tpu._pad_real_to_static,
          args=(node, shape_tensors, list(bounded_shape), is_dynamic),
      )
      pad_node.meta = node.meta.copy()

    node.replace_all_uses_with(
        pad_node, delete_user_cb=lambda u: u in original_users
    )


class HandleDynamicInputTensorSetDimensionSizePass:
  """Dynamic input tensor transformation pass using set_dimension_logical_size.

  Used when PadRealToStatic cannot be used (e.g., libtpu < 0.0.49 or when any
  dynamic input tensor is 64-bit/128-bit, as XLA does not support rewriting
  64-bit/128-bit element types for PadRealToStatic).
  """

  def __init__(
      self,
      transform_context: GraphTransformContext,
      placeholders: list[torch.fx.Node],
  ):
    self._transform_context = transform_context
    self._placeholders = placeholders

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    """Runs the op insertion pass."""
    for node in self._placeholders:
      val = node.meta.get("val")
      if not isinstance(val, torch.Tensor):
        continue
      dynamic_dims = [
          dim for dim, d in enumerate(val.shape) if sym_utils.is_symint(d)
      ]
      if not dynamic_dims:
        continue

      self._process_node(graph_module, node, dynamic_dims)

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
      size_tensor_node = self._transform_context.ensure_tensor(
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
          torch.ops.tpu._set_dimension_logical_size,
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
      transform_context: GraphTransformContext,
      placeholders: Sequence[torch.fx.Node],
  ):
    self._transform_context = transform_context
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
          size_ph.meta["val"] = torch.empty(
              (), dtype=torch.int32, device=sym_utils.get_target_device(node)
          )

        # Store the first placeholder encountered for this sym_str.
        self._transform_context.register_symint_placeholder(
            sym_str, node, size_ph
        )


class HandleSymIntUsagesPass:
  """Usages of SymInt nodes in tensor operations and graph outputs transformation pass.

  When a SymInt node (e.g., an input symbol representing a dynamic size) is
  consumed by a standard data computational operator (like `aten.add.Tensor`) or
  returned in the graph output node, we replace that usage with the
  corresponding runtime size placeholder tensor node or expression tensor node.
  """

  def __init__(self, transform_context: GraphTransformContext):
    self._transform_context = transform_context

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
            tensor_node = self._transform_context.ensure_tensor(
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
          for arg in node.args[0]:
            if sym_utils.is_symint(arg) or sym_utils.is_scalar(arg):
              val_meta = (
                  arg.meta.get("val")
                  if isinstance(arg, torch.fx.Node) and hasattr(arg, "meta")
                  else arg
              )
              target_dtype = getattr(
                  val_meta,
                  "dtype",
                  torch.int64 if isinstance(val_meta, int) else None,
              )
              tensor_node = self._transform_context.ensure_tensor(
                  graph_module, arg, node, dtype=target_dtype
              )
              current_dtype = getattr(
                  tensor_node.meta.get("val"), "dtype", None
              )

              if target_dtype is not None and current_dtype != target_dtype:
                with graph_module.graph.inserting_before(node):
                  device = sym_utils.get_target_device(node)
                  cast_node = graph_module.graph.call_method(
                      "to", args=(tensor_node,), kwargs={"dtype": target_dtype}
                  )
                  cast_node.meta = tensor_node.meta.copy()
                  cast_node.meta["val"] = torch.empty(
                      (), dtype=target_dtype, device=device
                  )
                  tensor_node = cast_node

              new_ret_args.append(tensor_node)
              changed = True
            else:
              new_ret_args.append(arg)

          if changed:
            node.args = (type(node.args[0])(new_ret_args),)


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


def apply_pre_dynamism_transformations(
    graph_module: torch.fx.GraphModule,
    example_inputs: Sequence[InputType],
) -> tuple[set[int], Sequence[InputType]]:
  """Runs pre-dynamism passes including view transformations and constant SymInt conversions.

  Args:
    graph_module: The FX graph module to transform.
    example_inputs: Example inputs for the graph module.

  Returns:
    A tuple of (view_arg_indices, updated_example_inputs), where
    view_arg_indices is a set of argument indices for input placeholders that
    were decomposed, and updated_example_inputs is a sequence of example inputs
    with decomposed view tensors and constant SymInts updated.
  """
  # Fetch original placeholders once in argument order
  original_placeholders = list(
      graph_module.graph.find_nodes(op="placeholder", sort=True)
  )

  input_dynamic_view_p = dynamic_view_pass.InputDynamicViewPass(
      original_placeholders, example_inputs
  )
  GraphTransformObserver(graph_module, "input_dynamic_view_pass").apply_gm_pass(
      input_dynamic_view_p
  )

  view_arg_indices = input_dynamic_view_p.view_arg_indices
  example_inputs = input_dynamic_view_p.updated_example_inputs

  convert_symints_p = ConvertConstantSymIntsPass(
      original_placeholders, example_inputs
  )
  GraphTransformObserver(
      graph_module, "convert_constant_symints_pass"
  ).apply_gm_pass(convert_symints_p)
  example_inputs = convert_symints_p.updated_example_inputs

  tools_common.stable_topological_sort(graph_module)
  graph_module.graph.lint()
  graph_module.recompile()

  return view_arg_indices, example_inputs


def apply_dynamism_transformations(
    graph_module: torch.fx.GraphModule,
    example_inputs: Sequence[InputType],
    symbol_bounds: SymbolBounds,
    use_pad_real_to_static: bool = True,
) -> None:
  """Runs all FX graph transforms for dynamic shapes."""
  transform_context = GraphTransformContext(
      graph_module, example_inputs, symbol_bounds
  )

  # Fetch original placeholders once in argument order
  original_placeholders = list(
      graph_module.graph.find_nodes(op="placeholder", sort=True)
  )

  # Scan inputs and create placeholders for SymInt inputs.
  GraphTransformObserver(
      graph_module, "scan_inputs_create_placeholders"
  ).apply_gm_pass(
      ScanInputsCreatePlaceholdersPass(transform_context, original_placeholders)
  )

  # Updates ops that have input tensors with dynamic dimensions.
  if use_pad_real_to_static:
    GraphTransformObserver(graph_module, "handle_dynamic_inputs").apply_gm_pass(
        HandleDynamicInputTensorPass(transform_context, original_placeholders)
    )
  else:
    GraphTransformObserver(
        graph_module, "handle_dynamic_inputs_set_dimension_size"
    ).apply_gm_pass(
        HandleDynamicInputTensorSetDimensionSizePass(
            transform_context, original_placeholders
        )
    )

  # Updates the generative ops that have dynamic scalar inputs.
  GraphTransformObserver(graph_module, "handle_generative_ops").apply_gm_pass(
      HandleGenerativeOpsPass(transform_context)
  )

  # Updates view ops that are reshape-like and have dynamic inputs.
  GraphTransformObserver(graph_module, "handle_reshape_like_ops").apply_gm_pass(
      view_ops_passes.HandleReshapeLikeOpsPass(transform_context)
  )

  # Updates broadcast operations that have dynamic inputs.
  GraphTransformObserver(
      graph_module, "handle_broadcast_like_ops"
  ).apply_gm_pass(view_ops_passes.HandleBroadcastLikeOpsPass(transform_context))

  # Updates slice operations that have dynamic inputs.
  GraphTransformObserver(graph_module, "handle_slice_like_ops").apply_gm_pass(
      view_ops_passes.HandleSliceLikeOpsPass(transform_context)
  )

  # Replaces remaining usages of SymInt nodes in standard tensor operations.
  GraphTransformObserver(graph_module, "handle_symint_usages").apply_gm_pass(
      HandleSymIntUsagesPass(transform_context)
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
