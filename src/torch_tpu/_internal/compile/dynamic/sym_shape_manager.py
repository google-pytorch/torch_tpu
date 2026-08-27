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

"""Manage symbolic integers and their bounds for dynamic shape support."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any
from absl import logging
import sympy
import torch
from torch.utils import _pytree
from torch_tpu._internal.compile.dynamic import sym_utils
from torch_tpu._internal.compile.dynamic.symbol_bounds import get_symint_bounds


@dataclasses.dataclass
class _OutputSymShape:
  """Information needed to compute the runtime shape of a dynamic output tensor.

  Sym shape for a output tensor. Each dimension is either:
    (1) an integer for a static dimension, or
    (2) a dictionary for a dynamic dimension with the following keys:
        "expr" (sympy expression string),
        "deps" (dict mapping symbol strings to (tensor_idx, dim_idx)).

  Example:[
            1,  # static dim
            {"expr": "sym_int_1 + sym_int_2",
             "deps": {"sym_int_1": (tensor_idx, dim_idx),
                      "sym_int_2": (tensor_idx, dim_idx)}} # dynamic dim
          ]
  """

  output_sym_shape: list[int | dict[str, Any]]
  is_tensor: bool = True

  def get_output_runtime_shape_from_sym_values(
      self, sym_values: dict[str, int]
  ) -> list[int]:
    """Computes the runtime shape of this output tensor from symbol values.

    Args:
      sym_values: A dictionary mapping symbol names (e.g., 's0') to concrete
        integer values.

    Returns:
      The calculated runtime shape of the output tensor.
    """
    new_shape = []
    for dim in self.output_sym_shape:
      if isinstance(dim, int):
        new_shape.append(dim)
      else:
        kwargs = {s: sym_values[s] for s in dim["deps"]}
        new_shape.append(dim["compiled_fn"](**kwargs))
    return new_shape

  def get_output_runtime_shape(
      self, inputs: Sequence[torch.Tensor | int]
  ) -> list[int]:
    """Computes the runtime shape of the output tensor.

    Args:
      inputs: The runtime input args(tensors and/or scalar) to compute the
        runtime shape of the output tensor.

    Returns:
      The runtime shape of the output tensor.

    The function evaluates the output shape expression to determine the runtime
    shape, with the input tensor dimensions and integer inputs being substituted
    for the symbols in the expression.

    Example:
        output_tensor[s0*2, 5, 2]
        s0 := symbol(input0.size) := 4
        runtime_shape: [8, 5, 2]
    """
    new_shape = []
    for dim in self.output_sym_shape:
      if isinstance(dim, int):
        new_shape.append(dim)
        continue
      kwargs = {
          s: (
              inputs[idx].shape[dim]  # pyrefly: ignore[missing-attribute]
              if isinstance(inputs[idx], torch.Tensor)
              else inputs[idx]
          )
          for s, (idx, dim) in dim["deps"].items()
      }

      runtime_size = dim["compiled_fn"](**kwargs)
      new_shape.append(runtime_size)
    return new_shape

  @classmethod
  def from_output_node_meta_value(
      cls,
      output_node_meta_value: Any,
      symint_to_tensor_dim: Mapping[str, tuple[int, int]],
  ) -> _OutputSymShape:
    """Creates output sym shape from output node's meta value.

    Args:
      output_node_meta_value: The meta value of the output node.
      symint_to_tensor_dim: A mapping from symbol names to (tensor_idx, dim_idx)
        tuples, indicating which input tensor dimension each symbol represents.

    Returns:
      An _OutputSymShape object containing the sym shape for the output tensor.

    Raises:
      RuntimeError: If the output shape dimension depends on symbols not derived
        from input tensor dimensions.
    """
    output_sym_shape = []
    is_tensor = output_node_meta_value is not None
    if (
        is_tensor
        and hasattr(output_node_meta_value, "shape")
        and output_node_meta_value.shape is not None
    ):
      for dim in output_node_meta_value.shape:
        if isinstance(dim, torch.SymInt) and hasattr(dim, "node"):
          expr = dim.node.expr
          symbols = expr.free_symbols
          deps = {}
          can_evaluate = True
          for s in symbols:
            s_str = str(s)
            tensor_dim = symint_to_tensor_dim.get(s_str)
            if tensor_dim is not None:
              deps[s_str] = tensor_dim
            else:
              can_evaluate = False
              break
          if can_evaluate:
            # Pre-compile the sympy expression using sympy.lambdify.
            # Sort symbols to ensure consistent argument order.
            symbols = sorted(expr.free_symbols, key=str)
            param_names = [str(s) for s in symbols]
            base_fn = sympy.lambdify(
                symbols, expr, modules=[sym_utils.CUSTOM_SYMPY_FUNCS, "math"]
            )

            # Use a helper to avoid closure variables being overwritten in the
            # loop.
            def make_compiled_fn(b_fn, p_names):
              return lambda **kwargs: int(
                  b_fn(*[kwargs[name] for name in p_names])
              )

            compiled_fn = make_compiled_fn(base_fn, param_names)
            output_sym_shape.append({"deps": deps, "compiled_fn": compiled_fn})
          else:
            raise RuntimeError(
                f"Output shape dimension '{dim}' depends on symbols not derived"
                " from input tensor dimensions."
            )
        elif isinstance(dim, int):
          output_sym_shape.append(dim)
        else:
          output_sym_shape.append(-1)
    return cls(output_sym_shape, is_tensor=is_tensor)


@dataclasses.dataclass
class InputTensorMetadata:
  """Metadata for an input tensor."""

  static_shape: list[int]
  dynamic_bounds: list[tuple[int, int]]
  dynamic_dims: list[int]


class SymShapeManager:
  """Symbolic shape manager for dynamic shape compilation.

  This class acts as a central repository for tracking and resolving symbolic
  integers (SymInts) across input and output tensors during FX graph tracing and
  compilation. It provides access to static shapes, dynamic bounds, and runtime
  shape computation logic.

  What it manages:
    - Input Tensors Metadata: Tracks static shapes, dynamic dimensions, and
    their bounds.
    - Symbolic Bounds: Maintains lower and upper bounds for all SymInts present
    in the inputs.
    - Output Shapes: Stores formulas to compute runtime output shapes from input
    dimensions.
    - Generative Op Placeholders: Maps SymInts to FX placeholder nodes created
    for them
      to support generative operations (like `torch.arange`).

  Attributes:
    input_tensors_metadata: Maps input tensor indices to their metadata.
    symints_to_bounds: Collection of all SymInts found in inputs and their
      corresponding (lower, upper) bounds.
    outputs_sym_shape: Output sym shape needed to compute the runtime shape of
      the output tensor based on input shapes.
    symint_to_placeholder: Maps the string representation of a SymInt to the FX
      placeholder node created to hold its runtime value (used by generative
      ops).
    symint_node_to_tensor_node: Maps scalar nodes representing expressions to
      tensor nodes.
    _sym_str_to_tensor_node: Internal cache mapping symint string to the created
      tensor node.
    _symint_output_indices: List of output indices that were originally SymInt
      nodes and promoted to 0D tensors.
  """

  # Tensor idx is its position in the example inputs.

  # Mapping of tensor idx in example inputs to its metadata.
  input_tensors_metadata: dict[int, InputTensorMetadata]

  # Collection of SymInts and their (lower, upper) bounds.
  # [(symint1, (lb1, ub1)), (symint2, (lb2, ub2)), ...]
  symints_to_bounds: list[tuple[torch.SymInt, tuple[int, int]]]

  # List of output shape infos.
  outputs_sym_shape: list[_OutputSymShape]

  # Mapping of SymInt string representation to its corresponding tensor
  # placeholder node that was created for it by generative ops.
  symint_to_placeholder: dict[str, torch.fx.Node]

  # Mapping of scalar nodes representing expressions to tensor nodes.
  symint_node_to_tensor_node: dict[torch.fx.Node, torch.fx.Node]

  # Mapping of SymInt string representation to the created tensor node.
  _sym_str_to_tensor_node: dict[str, torch.fx.Node]

  # List of output indices that were originally SymInt nodes and
  # promoted to 0D tensors.
  _symint_output_indices: list[int]

  def __init__(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[torch.Tensor | int],
  ):
    self._graph_module = graph_module
    self._example_inputs = example_inputs
    self.symint_to_placeholder = {}
    self.symint_node_to_tensor_node = {}
    self._sym_str_to_tensor_node = {}
    self._symint_output_indices = []
    self._create_outputs_sym_shape()
    self._populate_input_tensors_metadata()

  def set_symint_output_indices(self, indices: Sequence[int]) -> None:
    """Sets the indices of graph outputs that were originally SymInt nodes."""
    self._symint_output_indices = list(indices)

  def get_symint_output_indices(self) -> list[int]:
    """Returns the indices of graph outputs that were originally SymInt nodes."""
    return self._symint_output_indices

  @property
  def example_inputs(self) -> Sequence[torch.Tensor | int]:
    """The example inputs used to create the SymShapeManager."""
    return self._example_inputs

  def _populate_input_tensors_metadata(self) -> None:
    """Populates metadata for input tensors."""
    self.input_tensors_metadata = {}
    self.symints_to_bounds = []
    for idx, arg in enumerate(self._example_inputs):
      if isinstance(arg, torch.Tensor):
        static_shape = []
        dynamic_bounds = []
        dynamic_dims = []
        for shape_idx, s in enumerate(arg.shape):
          if isinstance(s, torch.SymInt):
            lower, upper = get_symint_bounds(s)
            self.symints_to_bounds.append((s, (lower, upper)))
            if lower == upper:
              static_shape.append(lower)
            else:
              static_shape.append(-1)
              dynamic_dims.append(shape_idx)
              dynamic_bounds.append((lower, upper))
          else:
            static_shape.append(int(s))

        self.input_tensors_metadata[idx] = InputTensorMetadata(
            static_shape=static_shape,
            dynamic_bounds=dynamic_bounds,
            dynamic_dims=dynamic_dims,
        )
      elif isinstance(arg, torch.SymInt):
        lower, upper = get_symint_bounds(arg)
        self.symints_to_bounds.append((arg, (lower, upper)))

  def _get_output_node_meta_val(self) -> list[Any]:
    """Reads output node.meta['val'] from a graph.

    Returns:
      A list of output node meta values.

    Raises:
      ValueError: If the graph has no output node.
      RuntimeError: If the output node has no 'val' in meta.
    """
    # Output node is conventionally the last node
    for n in reversed(self._graph_module.graph.nodes):
      if n.op == "output":
        output_node = n
        break
    else:
      raise ValueError("Graph has no output node")

    # output_node.args[0] contains the actual return value(s) of the graph,
    # which can be a single node, or a list/tuple of nodes.
    flat_output_nodes, _ = _pytree.tree_flatten(output_node.args[0])
    metas = []
    for node in flat_output_nodes:
      if isinstance(node, torch.fx.Node) and "val" in node.meta:
        metas.append(node.meta["val"])
      elif isinstance(node, torch.fx.Node):
        raise RuntimeError(
            f"Node {node} is in output but missing 'val' in meta."
        )
      else:
        # It might be a literal return like 1 or None
        metas.append(node)
    return metas

  def get_output_dtypes(self) -> list[torch.dtype]:
    """Returns the data types (dtypes) of all output tensors in the graph."""
    return [meta.dtype for meta in self._get_output_node_meta_val()]

  def compute_output_shapes_from_sym_values(
      self, sym_values: dict[str, int]
  ) -> list[list[int]] | None:
    """Computes the runtime output shapes directly from a dict of symbol values.

    This is used to predict output slice shapes for future steps without
    re-evaluating graph-module runtime inputs.
    Args:
      sym_values: A dictionary mapping symbol names to their predicted integer
        values.

    Returns:
      A list of shapes (each a list of ints) for each output tensor, or None
      if the graph has no symbolic outputs.
    """
    if not self.outputs_sym_shape:
      return None
    return [
        output_sym.get_output_runtime_shape_from_sym_values(sym_values)
        for output_sym in self.outputs_sym_shape
    ]

  def _create_outputs_sym_shape(self) -> None:
    """Creates sym shape for the output tensors.

    For each dynamic dimension of output tensor, it stores the expression and
    the input tensor indices it depends on.
    For static dimensions, it stores the value.

    Example:
      input: tensor_idx_0[symint_0, 2], tensor_idx_1[3, symint_1]
      output: [2*symint_0 + symint_1, 10]
      output_sym_shape: [
        {
          "expr": "2*symint_0 + symint_1",
          "deps": {
            "symint_0": { 0 /* tensor_idx */, 0 /* dim_idx */},
            "symint_1": { 1 /* tensor_idx */, 1 /* dim_idx */},
          },
        },   # dynamic dim
        10,  # static dim
      ]
    """
    symint_to_tensor_dim: dict[str, tuple[int, int]] = {}
    for idx, input_arg in enumerate(self._example_inputs):
      if isinstance(input_arg, torch.Tensor):
        for dim_idx, dim in enumerate(input_arg.shape):
          if isinstance(dim, torch.SymInt):
            dim_str = str(dim)
            if dim_str not in symint_to_tensor_dim:
              symint_to_tensor_dim[dim_str] = (idx, dim_idx)
      elif isinstance(input_arg, torch.SymInt):
        dim_str = str(input_arg)
        if dim_str not in symint_to_tensor_dim:
          symint_to_tensor_dim[dim_str] = (idx, 0)

    logging.debug("symint_to_tensor_dim: %s", symint_to_tensor_dim)
    output_nodes_meta_values = self._get_output_node_meta_val()

    outputs_sym_shape = [
        _OutputSymShape.from_output_node_meta_value(
            output_node_meta_value, symint_to_tensor_dim
        )
        for output_node_meta_value in output_nodes_meta_values
    ]

    logging.debug(
        "outputs_sym_shape: %s",
        list(outputs_sym_shape),
    )
    self.outputs_sym_shape = outputs_sym_shape

  def get_symint_upper_bound(self, symint: torch.SymInt) -> int:
    """Returns the upper bound of a given SymInt symbol.

    Args:
      symint: The symbolic integer whose upper bound is requested.

    Returns:
      The concrete upper bound integer value.
    Raises:
      ValueError: If no bounds are found for the given SymInt.
    """
    sym_str = str(symint)
    for s, (_, upper) in self.symints_to_bounds:
      if s == symint:
        return upper
    raise ValueError(f"No bounds found for SymInt {sym_str}")

  def get_bounded_shape(self, tensor_pos: int) -> list[int]:
    """Returns the shape of the tensor with dynamic dimensions replaced by their bounds.

    Args:
      tensor_pos: The position of the tensor in the example inputs.

    Returns:
      The shape of the tensor with dynamic dimensions replaced by their bounds.

    Raises:
      ValueError: If the tensor position is not found in the input tensors
      metadata.
    """
    metadata = self.input_tensors_metadata.get(tensor_pos)
    if metadata is None:
      raise ValueError(
          f"Tensor position {tensor_pos} not in input tensors metadata."
      )

    if not metadata.dynamic_dims:
      return list(metadata.static_shape)

    bounded_shape = list(metadata.static_shape)
    for (_, upper_bound), dim in zip(
        metadata.dynamic_bounds, metadata.dynamic_dims
    ):
      if 0 <= dim < len(bounded_shape):
        bounded_shape[dim] = upper_bound
      else:
        logging.warning(
            "Dynamic dimension index %d is out of bounds for static"
            " shape %s of input index %d. Ignoring it.",
            dim,
            metadata.static_shape,
            tensor_pos,
        )
    return bounded_shape

  def get_num_dynamic_dims(self, tensor_pos: int) -> int:
    """Returns the number of dynamic dimensions for the tensor at tensor_pos.

    Args:
      tensor_pos: The position of the tensor in the example inputs.

    Returns:
      The number of dynamic dimensions for the tensor at tensor_pos.
    """
    metadata = self.input_tensors_metadata.get(tensor_pos)
    if metadata is None:
      return 0
    return len(metadata.dynamic_dims)

  def add_bound_checks(self) -> None:
    """Adds bound checks for all SymInts in the graph.

    This function adds bound checks for all SymInts in the graph to prevent
    unnecessary recompilation when dynamic shapes are within the bounds.
    """

    unique_symints = {str(s): (s, b) for s, b in self.symints_to_bounds}
    for _, (symint, (lower, upper)) in unique_symints.items():
      torch._check(symint >= lower, "Dynamic shape lower bound check failed")  # pylint: disable=protected-access
      torch._check(symint <= upper, "Dynamic shape upper bound check failed")  # pylint: disable=protected-access

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
      if scalar_node in self.symint_node_to_tensor_node:
        return self.symint_node_to_tensor_node[scalar_node]
      symint_val = scalar_node.meta["val"]
    elif isinstance(scalar_node, torch.SymInt):
      symint_val = scalar_node
    else:
      return None

    sym_str = str(symint_val)

    # Check if a tensor node for this sym_str has already been created.
    if sym_str in self._sym_str_to_tensor_node:
      return self._sym_str_to_tensor_node[sym_str]

    symint_placeholder = self.symint_to_placeholder.get(sym_str)
    if symint_placeholder:
      return symint_placeholder

    expr = None
    if hasattr(symint_val, "node") and hasattr(symint_val.node, "expr"):
      expr = symint_val.node.expr

    if expr is not None:
      aten_node = sym_utils.symexpr_to_aten(
          self._graph_module, consumer_node, expr, self.symint_to_placeholder
      )
      if aten_node is not None:
        if isinstance(scalar_node, torch.fx.Node):
          self.symint_node_to_tensor_node[scalar_node] = aten_node
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
      if "val" in val.meta and isinstance(val.meta["val"], (int, float, bool)):
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
