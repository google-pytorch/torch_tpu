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

"""Utility functions for dynamic shape compilation."""

from collections.abc import Mapping
import operator
from typing import Any
from absl import logging
import sympy
import torch


def _get_tracer(*args) -> torch.fx.Tracer:
  for arg in args:
    if isinstance(arg, torch.fx.Proxy):
      return arg.tracer  # pyrefly: ignore[bad-return]
  raise ValueError("Expected at least one torch.fx.Proxy argument.")


def _get_target_dtype(*args) -> torch.dtype:
  for arg in args:
    if isinstance(arg, torch.fx.Proxy) and hasattr(arg.node, "meta"):
      val = arg.node.meta.get("val")
      if (
          val is not None
          and hasattr(val, "dtype")
          and isinstance(val.dtype, torch.dtype)
      ):
        return val.dtype
  return torch.int32


def _ensure_tensor_proxy(
    x: Any, tracer: torch.fx.Tracer, dtype: torch.dtype = torch.int32
) -> torch.fx.Proxy:
  if isinstance(x, torch.fx.Proxy):
    if hasattr(x.node, "meta"):
      val = x.node.meta.get("val")
      if val is not None and hasattr(val, "dtype") and val.dtype != dtype:
        return x.to(dtype=dtype)
    return x
  return tracer.create_proxy(
      "call_function",
      torch.ops.aten.scalar_tensor.default,
      (int(x),),
      {"dtype": dtype},
  )


def _sym_reduce(op: Any, *args):
  tracer = _get_tracer(*args)
  dtype = _get_target_dtype(*args)
  res = _ensure_tensor_proxy(args[0], tracer, dtype=dtype)
  for arg in args[1:]:
    res = tracer.create_proxy(
        "call_function",
        op,
        (res, _ensure_tensor_proxy(arg, tracer, dtype=dtype)),
        {},
    )
  return res


def _sym_max(*args):
  return _sym_reduce(torch.ops.aten.maximum.default, *args)


def _sym_min(*args):
  return _sym_reduce(torch.ops.aten.minimum.default, *args)


CUSTOM_SYMPY_FUNCS = {
    "FloorDiv": operator.floordiv,
    "CeilDiv": lambda a, b: -(-a // b),
    "Max": _sym_max,
    "Min": _sym_min,
}


def is_symint_node(node: Any) -> bool:
  """Checks if the FX node evaluates to a symbolic integer (SymInt).

  Args:
    node: The object or FX node to check.

  Returns:
    True if the node is an FX node representing a symbolic integer, False
    otherwise.
  """
  return (
      isinstance(node, torch.fx.Node)
      and hasattr(node, "meta")
      and "val" in node.meta
      and isinstance(node.meta["val"], torch.SymInt)
  )


def is_symint(val: Any) -> bool:
  """Checks if a value or FX node represents a symbolic integer (SymInt).

  Args:
    val: The value or FX node to check.

  Returns:
    True if val is a SymInt or an FX node whose 'val' meta is a SymInt.
  """
  if isinstance(val, torch.SymInt):
    return True
  return isinstance(val, torch.fx.Node) and is_symint_node(val)


def has_dynamic_shape(val: Any) -> bool:
  """Checks if a tensor, FX Node, or FakeTensor has any dynamic dimensions.

  Args:
    val: An FX Node, Tensor, FakeTensor, or any object with a 'shape' attribute
      or 'meta["val"]' tensor.

  Returns:
    True if the object has a shape with at least one SymInt dimension, False
    otherwise.
  """
  tensor_val = (
      val.meta.get("val", val)
      if isinstance(val, torch.fx.Node) and hasattr(val, "meta")
      else val
  )
  return hasattr(tensor_val, "shape") and any(
      is_symint(d) for d in tensor_val.shape
  )


def is_symexpr_node(node: torch.fx.Node) -> bool:
  """Checks if the FX node represents a complex math expression (e.g., s0 + 1).

  Args:
    node: The FX node to check.

  Returns:
    True if the node represents a complex symbolic expression, False if it is a
    concrete integer or a base symbol (e.g., s0).
  """
  if not is_symint_node(node):
    return False

  symint = node.meta["val"]
  if hasattr(symint, "node") and hasattr(symint.node, "expr"):
    return not isinstance(symint.node.expr, sympy.Symbol)

  return False


def is_base_symbol_node(node: torch.fx.Node) -> bool:
  """Checks if the FX node represents a raw base symbol variable token (e.g., s0).

  Args:
    node: The FX node to check.

  Returns:
    True if the node represents a base symbol variable, False otherwise.
  """
  return is_symint_node(node) and not is_symexpr_node(node)


def get_target_device(consumer_node: torch.fx.Node) -> torch.device:
  """Extracts target device from consumer node's input, graph, or kwargs."""
  if (
      "device" in consumer_node.kwargs
      and consumer_node.kwargs["device"] is not None
  ):
    return consumer_node.kwargs["device"]

  for arg in consumer_node.all_input_nodes:
    if "val" in arg.meta and hasattr(arg.meta["val"], "device"):
      return arg.meta["val"].device

  for node in consumer_node.graph.nodes:
    if "val" in node.meta and hasattr(node.meta["val"], "device"):
      return node.meta["val"].device

  try:
    return torch.device("tpu")
  except RuntimeError:
    return torch.device("cpu")


def symexpr_to_aten(
    graph_module: torch.fx.GraphModule,
    consumer_node: torch.fx.Node,
    expr: sympy.Expr,
    symint_to_placeholder: Mapping[str, torch.fx.Node],
) -> torch.fx.Node | None:
  """Creates a chain of aten ops for a symexpr using lambdify and inserts them into the graph before the consumer node.

  Args:
    graph_module: The FX GraphModule to insert operations into.
    consumer_node: The node that consumes the expression result.
    expr: The sympy expression to parse.
    symint_to_placeholder: A mapping from symbol strings to their placeholder
      nodes.

  Returns:
    The new tensor node, or None if not supported.
  """
  symbols = list(expr.free_symbols)

  # Check if we have placeholders for all symbols
  for sym in symbols:
    if str(sym) not in symint_to_placeholder:
      return None

  # Create a Python function from the sympy expression
  f = sympy.lambdify(symbols, expr, modules=[CUSTOM_SYMPY_FUNCS, "math"])

  # Create Proxies for the placeholders
  proxies = [torch.fx.Proxy(symint_to_placeholder[str(sym)]) for sym in symbols]

  # Call the function with proxies to record operations in the graph
  # Use inserting_before to maintain topological order
  with graph_module.graph.inserting_before(consumer_node):
    try:
      result_proxy = f(*proxies)
      # the result should be always be of int32 type as that is the data
      # type for dynamic dimension in stablehlo.
      result_proxy = result_proxy.to(dtype=torch.int32)
      return result_proxy.node
    except (TypeError, AttributeError, ValueError) as e:
      logging.exception(
          "Failed to create ops for expression %s: %s", expr, repr(e)
      )
      return None
