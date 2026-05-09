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
import sympy
import torch
from torch_tpu._internal.compile import logging

CUSTOM_SYMPY_FUNCS = {
    "FloorDiv": operator.floordiv,
    "CeilDiv": lambda a, b: -(-a // b),
}


def is_symint_node(node: torch.fx.Node) -> bool:
  """Checks if the FX node evaluates to a symbolic integer (SymInt).

  Args:
    node: The FX node to check.

  Returns:
    True if the node represents a symbolic integer, False otherwise.
  """
  return (
      hasattr(node, "meta")
      and "val" in node.meta
      and isinstance(node.meta["val"], torch.SymInt)
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
      return result_proxy.node
    except (TypeError, AttributeError, ValueError) as e:
      logging.exception(
          "Failed to create ops for expression %s: %s", expr, repr(e)
      )
      return None
