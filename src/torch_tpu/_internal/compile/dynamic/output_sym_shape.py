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

"""Symbolic output shape representation and evaluation for dynamic shapes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from absl import logging
import sympy
import torch
from torch.utils import _pytree
from torch_tpu._internal.compile.dynamic import sym_utils


@dataclasses.dataclass
class OutputSymShape:
  """Information needed to compute the runtime shape of a dynamic output tensor.

  Sym shape for an output tensor. Each dimension is either:
    (1) an integer for a static dimension, or
    (2) a dictionary for a dynamic dimension with the following keys:
        "deps" (dict mapping symbol strings to (tensor_idx, dim_idx)),
        "compiled_fn" (callable evaluating the symbolic dimension expression).

  Example:[
            1,  # static dim
            {"deps": {"sym_int_1": (tensor_idx, dim_idx),
                      "sym_int_2": (tensor_idx, dim_idx)},
             "compiled_fn": <callable>} # dynamic dim
          ]
  """

  output_sym_shape: list[int | dict[str, Any]]
  is_tensor: bool = True
  is_promoted_scalar: bool = False
  is_dynamic: bool = dataclasses.field(init=False)

  def __post_init__(self) -> None:
    self.is_dynamic = any(
        isinstance(dim, dict) for dim in self.output_sym_shape
    )

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
              inputs[tensor_idx].shape[dim_idx]  # pyrefly: ignore[missing-attribute]
              if isinstance(inputs[tensor_idx], torch.Tensor)
              else inputs[tensor_idx]
          )
          for s, (tensor_idx, dim_idx) in dim["deps"].items()
      }

      runtime_size = dim["compiled_fn"](**kwargs)
      new_shape.append(runtime_size)
    return new_shape

  @classmethod
  def from_output_node_meta_value(
      cls,
      output_node_meta_value: Any,
      symint_to_tensor_dim: Mapping[str, tuple[int, int]],
  ) -> OutputSymShape:
    """Creates output sym shape from output node's meta value.

    Args:
      output_node_meta_value: The meta value of the output node.
      symint_to_tensor_dim: A mapping from symbol names to (tensor_idx, dim_idx)
        tuples, indicating which input tensor dimension each symbol represents.

    Returns:
      An OutputSymShape object containing the sym shape for the output tensor.

    Raises:
      RuntimeError: If the output shape dimension depends on symbols not derived
        from input tensor dimensions.
    """
    output_sym_shape = []
    is_tensor = output_node_meta_value is not None
    is_promoted_scalar = sym_utils.is_symint(
        output_node_meta_value
    ) or sym_utils.is_scalar(output_node_meta_value)
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

            compiled_fn = _make_compiled_fn(base_fn, param_names)
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
    return cls(
        output_sym_shape,
        is_tensor=is_tensor,
        is_promoted_scalar=is_promoted_scalar,
    )


def _make_compiled_fn(base_fn: Any, param_names: Sequence[str]) -> Any:
  """Wraps a lambdified sympy expression to accept keyword arguments by name."""
  return lambda **kwargs: int(base_fn(*[kwargs[name] for name in param_names]))


def _get_output_node_meta_val(graph_module: torch.fx.GraphModule) -> list[Any]:
  """Reads output node.meta['val'] from a graph.

  Args:
    graph_module: The FX graph module whose output node metadata to read.

  Returns:
    A list of output node meta values.

  Raises:
    ValueError: If the graph has no output node.
    RuntimeError: If the output node has no 'val' in meta.
  """
  # Output node is conventionally the last node
  for n in reversed(graph_module.graph.nodes):
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
      raise RuntimeError(f"Node {node} is in output but missing 'val' in meta.")
    else:
      # It might be a literal return like 1 or None
      metas.append(node)
  return metas


def create_output_sym_shapes(
    graph_module: torch.fx.GraphModule,
    example_inputs: Sequence[torch.Tensor | int],
) -> list[OutputSymShape]:
  """Creates symbolic shape specifications for all output tensors of the graph.

  For each dynamic dimension of output tensor, it stores the expression and
  the input tensor indices it depends on. For static dimensions, it stores the
  value.

  Args:
    graph_module: The FX graph module.
    example_inputs: The example inputs to the graph module.

  Returns:
    A list of OutputSymShape objects, one per flattened output of the graph.
  """
  symint_to_tensor_dim: dict[str, tuple[int, int]] = {}
  for idx, input_arg in enumerate(example_inputs):
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
  output_nodes_meta_values = _get_output_node_meta_val(graph_module)

  output_sym_shapes = [
      OutputSymShape.from_output_node_meta_value(
          output_node_meta_value, symint_to_tensor_dim
      )
      for output_node_meta_value in output_nodes_meta_values
  ]

  logging.debug(
      "output_sym_shapes: %s",
      list(output_sym_shapes),
  )
  return output_sym_shapes
