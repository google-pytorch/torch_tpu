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

"""Context shared across FX graph transformation passes for dynamic shapes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch_tpu._internal.compile.dynamic import sym_utils
from torch_tpu._internal.compile.dynamic.symbol_bounds import SymbolBounds


class GraphTransformContext:
  """Context shared across FX graph transformation passes for dynamic shapes.

  Tracks and resolves symbolic integers (SymInts) during FX graph transformation
  passes, providing access to symbol bounds, example inputs, and promotion of
  SymInt/scalar expressions to FX tensor nodes.

  Attributes:
    _symint_to_placeholder: Maps the string representation of a SymInt to the FX
      placeholder node created to hold its runtime value (used by generative
      ops).
    _symint_node_to_tensor_node: Maps scalar nodes representing expressions to
      tensor nodes.
    _sym_str_to_tensor_node: Internal cache mapping symint string to the created
      tensor node.
  """

  # Mapping of SymInt string representation to its corresponding tensor
  # placeholder node that was created for it by generative ops.
  _symint_to_placeholder: dict[str, torch.fx.Node]

  # Mapping of scalar nodes representing expressions to tensor nodes.
  _symint_node_to_tensor_node: dict[torch.fx.Node, torch.fx.Node]

  # Mapping of SymInt string representation to the created tensor node.
  _sym_str_to_tensor_node: dict[str, torch.fx.Node]

  def __init__(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[torch.Tensor | int],
      symbol_bounds: SymbolBounds,
  ):
    self._graph_module = graph_module
    self._example_inputs = example_inputs
    self._symbol_bounds = symbol_bounds
    self._symint_to_placeholder = {}
    self._symint_node_to_tensor_node = {}
    self._sym_str_to_tensor_node = {}

  def register_symint_placeholder(
      self, sym_str: str, node: torch.fx.Node, size_ph: torch.fx.Node
  ) -> None:
    """Registers the runtime size placeholder tensor node for a SymInt node."""
    first_size_ph = self._symint_to_placeholder.setdefault(sym_str, size_ph)
    self._symint_node_to_tensor_node[node] = first_size_ph

  @property
  def example_inputs(self) -> Sequence[torch.Tensor | int]:
    """The example inputs used to create the GraphTransformContext."""
    return self._example_inputs

  def get_symint_bounds(self, sym_int: torch.SymInt) -> tuple[int, int]:
    """Gets lower and upper bounds for a given SymInt or symbolic expression."""
    return self._symbol_bounds.get_symint_bounds(sym_int)

  def get_upper_bound(self, val: Any) -> int:
    """Gets the upper bound for an int, SymInt, or SymInt FX node."""
    return self._symbol_bounds.get_upper_bound(val)

  def _get_or_create_tensor_node(
      self,
      scalar_node: torch.fx.Node | torch.SymInt,
      consumer_node: torch.fx.Node,
  ) -> torch.fx.Node | None:
    """Gets or creates a tensor node for a symbolic argument.

    Internal method: Transformation passes should not call this directly. Use
    `ensure_tensor` instead.

    If the argument is already mapped to a tensor node, returns it.
    If it's a simple symbol and has a placeholder, returns the placeholder.
    If it's an expression, uses the expression parser to create ops for it.

    Args:
      scalar_node: The scalar node or SymInt representing the symbolic argument.
      consumer_node: The node that will consume the resulting tensor node.

    Returns:
      The tensor node representing the argument, or None if it cannot be
      resolved or created.
    """
    if isinstance(scalar_node, torch.fx.Node):
      if not sym_utils.is_symint_node(scalar_node):
        return None
      if scalar_node in self._symint_node_to_tensor_node:
        return self._symint_node_to_tensor_node[scalar_node]
      symint_val = scalar_node.meta["val"]
    elif isinstance(scalar_node, torch.SymInt):
      symint_val = scalar_node
    else:
      return None

    sym_str = str(symint_val)

    # Check if a tensor node for this sym_str has already been created.
    if sym_str in self._sym_str_to_tensor_node:
      return self._sym_str_to_tensor_node[sym_str]

    symint_placeholder = self._symint_to_placeholder.get(sym_str)
    if symint_placeholder:
      return symint_placeholder

    expr = None
    if hasattr(symint_val, "node") and hasattr(symint_val.node, "expr"):
      expr = symint_val.node.expr

    if expr is not None:
      aten_node = sym_utils.symexpr_to_aten(
          self._graph_module, consumer_node, expr, self._symint_to_placeholder
      )
      if aten_node is not None:
        if isinstance(scalar_node, torch.fx.Node):
          self._symint_node_to_tensor_node[scalar_node] = aten_node
        self._sym_str_to_tensor_node[sym_str] = aten_node
        return aten_node

    return None

  def ensure_tensor(
      self,
      graph_module: torch.fx.GraphModule,
      val: Any,
      consumer_node: torch.fx.Node,
      dtype: torch.dtype | None = None,
  ) -> torch.fx.Node:
    """Ensures the value is a tensor node, promoting scalars if needed.

    Args:
      graph_module: The FX GraphModule.
      val: The value to ensure as a tensor. Can be a torch.fx.Node, a SymInt, or
        a Python constant (int, float, bool).
      consumer_node: The node that will consume the resulting tensor node.
      dtype: Optional desired dtype for the tensor.

    Returns:
      A torch.fx.Node representing a tensor.

    Raises:
      RuntimeError: If the value is not a tensor node and cannot be converted
        to one.
    """
    if isinstance(val, torch.fx.Node):
      if "val" in val.meta and isinstance(val.meta["val"], torch.Tensor):
        return val
      if sym_utils.is_symint_node(val):
        tensor_node = self._get_or_create_tensor_node(val, consumer_node)
        assert tensor_node is not None, f"tensor node for {val} not found"
        return tensor_node
      if sym_utils.is_scalar(val):
        val = val.meta["val"]
      else:
        raise RuntimeError(f"Unsupported node type for ensure_tensor: {val}")
    elif isinstance(val, torch.SymInt):
      tensor_node = self._get_or_create_tensor_node(val, consumer_node)
      assert tensor_node is not None, f"tensor node for {val} not found"
      return tensor_node

    kwargs = {"device": sym_utils.get_target_device(consumer_node)}
    if dtype is not None:
      kwargs["dtype"] = dtype
    elif isinstance(val, bool):
      kwargs["dtype"] = torch.bool
    elif isinstance(val, int):
      kwargs["dtype"] = torch.int32
    elif isinstance(val, float):
      kwargs["dtype"] = torch.float32
    else:
      raise RuntimeError(f"Unsupported type for ensure_tensor: {val}")

    with graph_module.graph.inserting_before(consumer_node):
      tensor_node = graph_module.graph.call_function(
          torch.ops.aten.scalar_tensor.default, args=(val,), kwargs=kwargs
      )
      return tensor_node
