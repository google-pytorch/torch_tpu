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

"""FX pass to canonicalize clamp and activation patterns into unified ops.

PyTorch decomposition often decomposes activations in forward
(e.g., clamp_min(x, 0) for relu, clamp(x, 0, 6) for hardtanh/relu6) and in
backward (e.g., autograd decomposes relu backward into where(x > 0, grad, 0)
or mul(grad, x > 0) or clamp_min_backward).

This pass applies algebraic strength reduction to canonicalize these forward
and backward activation patterns into unified high-level ops:
  - clamp_min(x, 0) -> relu(x)
  - clamp(x, 0, None) -> relu(x)
  - clamp(x, 0, 6) / clamp(x, 0.0, 6.0) -> hardtanh(x, 0.0, 6.0)
  - where(x > 0, grad, 0) / mul(grad, x > 0) / clamp_min_backward
      -> threshold_backward(grad, x, 0)

This unlocks XLA loop fusion and reduces redundant memory traffic.
"""

import os
from typing import Any

import torch
import torch.fx


def _is_canonicalize_clamp_enabled() -> bool:
  """Returns whether the pass is enabled via environment variable."""
  return os.environ.get(  # PY_ENVIRON_OK=killswitch
      "TORCH_TPU_INTERNAL_ENABLE_CANONICALIZE_CLAMP", "1"
  ) not in ("0", "false", "False")


def _normalize_gm_and_graph(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> tuple[torch.fx.GraphModule | None, torch.fx.Graph]:
  """Normalizes input into a (GraphModule or None, Graph) tuple."""
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    return gm_or_graph, gm_or_graph.graph
  return getattr(gm_or_graph, "owning_module", None), gm_or_graph


def _is_zero(val: Any) -> bool:
  """Checks whether an argument represents a scalar zero."""
  if isinstance(val, (int, float)) and not isinstance(val, bool) and val == 0:
    return True
  if isinstance(val, torch.fx.Node):
    if val.op == "call_function":
      target = getattr(val.target, "overloadpacket", val.target)
      if target == torch.ops.aten.scalar_tensor:
        if (
            len(val.args) >= 1
            and isinstance(val.args[0], (int, float))
            and not isinstance(val.args[0], bool)
            and val.args[0] == 0
        ):
          return True
      elif target == torch.ops.aten.full:
        if (
            len(val.args) >= 2
            and isinstance(val.args[1], (int, float))
            and not isinstance(val.args[1], bool)
            and val.args[1] == 0
        ):
          return True
      elif target in (
          torch.ops.aten.zeros,
          torch.ops.aten.zeros_like,
      ):
        return True
    elif val.op == "get_attr":
      if (
          hasattr(val, "graph")
          and hasattr(val.graph, "owning_module")
          and val.graph.owning_module is not None
      ):
        try:
          attr_val = torch.fx.graph_module._get_attr(
              val.graph.owning_module, str(val.target)
          )
          if (
              isinstance(attr_val, (int, float))
              and not isinstance(attr_val, bool)
              and attr_val == 0
          ):
            return True
          if isinstance(attr_val, torch.Tensor) and torch.all(attr_val == 0):
            return True
        except Exception:  # pylint: disable=broad-except
          pass
  return False


def _is_six(val: Any) -> bool:
  """Checks whether an argument represents a scalar 6 or 6.0."""
  if isinstance(val, (int, float)) and not isinstance(val, bool) and val == 6:
    return True
  if isinstance(val, torch.fx.Node):
    if val.op == "call_function":
      target = getattr(val.target, "overloadpacket", val.target)
      if target == torch.ops.aten.scalar_tensor:
        if (
            len(val.args) >= 1
            and isinstance(val.args[0], (int, float))
            and not isinstance(val.args[0], bool)
            and val.args[0] == 6
        ):
          return True
      elif target == torch.ops.aten.full:
        if (
            len(val.args) >= 2
            and isinstance(val.args[1], (int, float))
            and not isinstance(val.args[1], bool)
            and val.args[1] == 6
        ):
          return True
    elif val.op == "get_attr":
      if (
          hasattr(val, "graph")
          and hasattr(val.graph, "owning_module")
          and val.graph.owning_module is not None
      ):
        try:
          attr_val = torch.fx.graph_module._get_attr(
              val.graph.owning_module, str(val.target)
          )
          if (
              isinstance(attr_val, (int, float))
              and not isinstance(attr_val, bool)
              and attr_val == 6
          ):
            return True
          if isinstance(attr_val, torch.Tensor) and torch.all(attr_val == 6):
            return True
        except Exception:  # pylint: disable=broad-except
          pass
  return False


def _unwrap_cast(node: Any) -> Any:
  """Unwraps type conversion / copy wrappers around a node."""
  while isinstance(node, torch.fx.Node) and node.op == "call_function":
    target = getattr(node.target, "overloadpacket", node.target)
    if target in (
        torch.ops.aten._to_copy,
        torch.ops.aten.clone,
        getattr(torch.ops.prims, "convert_element_type", None),
    ):
      if node.args:
        node = node.args[0]
      else:
        break
    else:
      break
  return node


def _match_comparison_zero(node: Any) -> tuple[torch.fx.Node, str] | None:
  """Matches comparison with zero, returning (tensor_node, op_str) or None."""
  if not isinstance(node, torch.fx.Node) or node.op != "call_function":
    return None
  target = getattr(node.target, "overloadpacket", node.target)
  if len(node.args) < 2:
    return None
  arg0, arg1 = node.args[0], node.args[1]

  if target == torch.ops.aten.gt:
    if _is_zero(arg1) and isinstance(arg0, torch.fx.Node):
      return (arg0, ">")
    if _is_zero(arg0) and isinstance(arg1, torch.fx.Node):
      return (arg1, "<")
  elif target == torch.ops.aten.ge:
    if _is_zero(arg1) and isinstance(arg0, torch.fx.Node):
      return (arg0, ">=")
    if _is_zero(arg0) and isinstance(arg1, torch.fx.Node):
      return (arg1, "<=")
  elif target == torch.ops.aten.lt:
    if _is_zero(arg1) and isinstance(arg0, torch.fx.Node):
      return (arg0, "<")
    if _is_zero(arg0) and isinstance(arg1, torch.fx.Node):
      return (arg1, ">")
  elif target == torch.ops.aten.le:
    if _is_zero(arg1) and isinstance(arg0, torch.fx.Node):
      return (arg0, "<=")
    if _is_zero(arg0) and isinstance(arg1, torch.fx.Node):
      return (arg1, ">=")
  return None


def _is_clamp_min_backward_op(target: Any) -> bool:
  op_name = getattr(target, "__name__", "") or str(target)
  return "clamp_min_backward" in op_name


def _is_clamp_backward_op(target: Any) -> bool:
  op_name = getattr(target, "__name__", "") or str(target)
  return "clamp_backward" in op_name and "min_max" not in op_name


def apply(gm_or_graph: torch.fx.GraphModule | torch.fx.Graph) -> None:
  """Canonicalizes clamp and activation patterns into unified ops.

  Args:
    gm_or_graph: The FX GraphModule or Graph to transform.
  """
  if not _is_canonicalize_clamp_enabled():
    return

  gm, graph = _normalize_gm_and_graph(gm_or_graph)

  for node in list(graph.nodes):
    if node.op != "call_function":
      continue

    target = getattr(node.target, "overloadpacket", node.target)

    # 1. Forward: clamp_min(x, 0) -> relu(x)
    if target == torch.ops.aten.clamp_min:
      x = (
          node.args[0]
          if node.args
          else (node.kwargs.get("input") or node.kwargs.get("self"))
      )
      min_val = node.args[1] if len(node.args) >= 2 else node.kwargs.get("min")
      if _is_zero(min_val) and isinstance(x, torch.fx.Node):
        with graph.inserting_before(node):
          new_node = graph.call_function(
              torch.ops.aten.relu.default,
              args=(x,),
          )
          if hasattr(node, "meta") and node.meta:
            new_node.meta = node.meta.copy()
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        continue

    # 2. Forward: clamp(x, 0, None) -> relu(x)
    #             clamp(x, 0, 6) -> hardtanh(x, 0.0, 6.0)
    if target == torch.ops.aten.clamp:
      x = (
          node.args[0]
          if node.args
          else (node.kwargs.get("input") or node.kwargs.get("self"))
      )
      min_val = (
          node.args[1] if len(node.args) >= 2 else node.kwargs.get("min", None)
      )
      max_val = (
          node.args[2] if len(node.args) >= 3 else node.kwargs.get("max", None)
      )

      if _is_zero(min_val) and max_val is None and isinstance(x, torch.fx.Node):
        with graph.inserting_before(node):
          new_node = graph.call_function(
              torch.ops.aten.relu.default,
              args=(x,),
          )
          if hasattr(node, "meta") and node.meta:
            new_node.meta = node.meta.copy()
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        continue

      if (
          _is_zero(min_val)
          and _is_six(max_val)
          and isinstance(x, torch.fx.Node)
      ):
        with graph.inserting_before(node):
          new_node = graph.call_function(
              torch.ops.aten.hardtanh.default,
              args=(x, 0.0, 6.0),
          )
          if hasattr(node, "meta") and node.meta:
            new_node.meta = node.meta.copy()
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        continue

    # 3. Backward: where(gt(x, 0), grad, 0) or where(le(x, 0), 0, grad)
    if target == torch.ops.aten.where:
      if len(node.args) >= 3:
        condition, self_val, other_val = (
            node.args[0],
            node.args[1],
            node.args[2],
        )
      elif len(node.args) == 2:
        condition, self_val = node.args[0], node.args[1]
        other_val = node.kwargs.get("other")
      else:
        condition = node.kwargs.get("condition")
        self_val = node.kwargs.get("self")
        other_val = node.kwargs.get("other")

      # Case A: where(x > 0, grad, 0)
      if _is_zero(other_val) and isinstance(self_val, torch.fx.Node):
        comp = _match_comparison_zero(_unwrap_cast(condition))
        if comp is not None:
          x, cmp_op = comp
          if cmp_op in (">", ">="):
            with graph.inserting_before(node):
              new_node = graph.call_function(
                  torch.ops.aten.threshold_backward.default,
                  args=(self_val, x, 0),
              )
              if hasattr(node, "meta") and node.meta:
                new_node.meta = node.meta.copy()
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
            continue

      # Case B: where(x <= 0, 0, grad)
      if _is_zero(self_val) and isinstance(other_val, torch.fx.Node):
        comp = _match_comparison_zero(_unwrap_cast(condition))
        if comp is not None:
          x, cmp_op = comp
          if cmp_op in ("<", "<="):
            with graph.inserting_before(node):
              new_node = graph.call_function(
                  torch.ops.aten.threshold_backward.default,
                  args=(other_val, x, 0),
              )
              if hasattr(node, "meta") and node.meta:
                new_node.meta = node.meta.copy()
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
            continue

    # 4. Backward: mul(grad, gt(x, 0)) or mul(gt(x, 0), grad)
    if target == torch.ops.aten.mul:
      if len(node.args) >= 2:
        arg0, arg1 = node.args[0], node.args[1]
      else:
        arg0 = node.args[0] if node.args else node.kwargs.get("input")
        arg1 = node.kwargs.get("other")

      # Subcase A: mul(grad, mask)
      comp1 = _match_comparison_zero(_unwrap_cast(arg1))
      if comp1 is not None and isinstance(arg0, torch.fx.Node):
        x, cmp_op = comp1
        if cmp_op in (">", ">="):
          with graph.inserting_before(node):
            new_node = graph.call_function(
                torch.ops.aten.threshold_backward.default,
                args=(arg0, x, 0),
            )
            if hasattr(node, "meta") and node.meta:
              new_node.meta = node.meta.copy()
          node.replace_all_uses_with(new_node)
          graph.erase_node(node)
          continue

      # Subcase B: mul(mask, grad)
      comp0 = _match_comparison_zero(_unwrap_cast(arg0))
      if comp0 is not None and isinstance(arg1, torch.fx.Node):
        x, cmp_op = comp0
        if cmp_op in (">", ">="):
          with graph.inserting_before(node):
            new_node = graph.call_function(
                torch.ops.aten.threshold_backward.default,
                args=(arg1, x, 0),
            )
            if hasattr(node, "meta") and node.meta:
              new_node.meta = node.meta.copy()
          node.replace_all_uses_with(new_node)
          graph.erase_node(node)
          continue

    # 5. Backward: clamp_min_backward(grad, x, 0)
    if _is_clamp_min_backward_op(node.target):
      grad = node.args[0] if len(node.args) >= 1 else node.kwargs.get("grad")
      x = node.args[1] if len(node.args) >= 2 else node.kwargs.get("self")
      min_val = (
          node.args[2] if len(node.args) >= 3 else node.kwargs.get("min", 0)
      )
      if (
          _is_zero(min_val)
          and isinstance(grad, torch.fx.Node)
          and isinstance(x, torch.fx.Node)
      ):
        with graph.inserting_before(node):
          new_node = graph.call_function(
              torch.ops.aten.threshold_backward.default,
              args=(grad, x, 0),
          )
          if hasattr(node, "meta") and node.meta:
            new_node.meta = node.meta.copy()
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        continue

    # 6. Backward: clamp_backward(grad, x, min=0, max=None)
    if _is_clamp_backward_op(node.target):
      grad = node.args[0] if len(node.args) >= 1 else node.kwargs.get("grad")
      x = node.args[1] if len(node.args) >= 2 else node.kwargs.get("self")
      min_val = (
          node.args[2] if len(node.args) >= 3 else node.kwargs.get("min", None)
      )
      max_val = (
          node.args[3] if len(node.args) >= 4 else node.kwargs.get("max", None)
      )
      if (
          _is_zero(min_val)
          and max_val is None
          and isinstance(grad, torch.fx.Node)
          and isinstance(x, torch.fx.Node)
      ):
        with graph.inserting_before(node):
          new_node = graph.call_function(
              torch.ops.aten.threshold_backward.default,
              args=(grad, x, 0),
          )
          if hasattr(node, "meta") and node.meta:
            new_node.meta = node.meta.copy()
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        continue

  graph.eliminate_dead_code()
  graph.lint()
  if gm is not None:
    for _, submod in gm.named_children():
      if isinstance(submod, torch.fx.GraphModule):
        apply(submod)
    gm.recompile()
