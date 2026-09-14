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

"""FX pass to optimize Rotary Position Embedding (RoPE) via memoized algebraic reduction.

Variant 3: Class C/E (Algebraic Strength Reduction + Cross-Projection
Memoization).
In transformer multi-head self-attention:
  q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

Both `q` and `k` share the exact same `cos` and `sin` tensors.
In addition to eliminating the pre-multiplication concatenation and reducing
to half-width FMA operations:
  out1 = (x1 * c) - (x2 * s)
  out2 = (x2 * c) + (x1 * s)
  out = cat((out1, out2), dim=-1)

This pass memoizes the half-width slices `(c, s)` keyed by `(cos_node, sin_node,
half_dim)`.
When `k` is transformed after `q`, the slices are reused directly rather than
regenerated. This halves the slice overhead for rotary embeddings across all
attention layers in the model.
"""

import operator
import torch
from torch import fx

_ADD_TARGETS = {
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add.default,
    operator.add,
}

_MUL_TARGETS = {
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.default,
    operator.mul,
}

_NEG_TARGETS = {
    torch.ops.aten.neg.default,
    operator.neg,
}

_CAT_TARGETS = {
    torch.ops.aten.cat.default,
}

_SLICE_TARGETS = {
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.slice_copy.Tensor,
}


def _match_rotate_half(
    rot_node: fx.Node,
) -> tuple[fx.Node, fx.Node, fx.Node, int, int, int | None] | None:
  """Matches rotate_half(x) = cat((-x2, x1), dim=-1)."""
  if not isinstance(rot_node, fx.Node) or rot_node.op != "call_function":
    return None

  if rot_node.target not in _CAT_TARGETS:
    return None

  if not rot_node.args:
    return None

  cat_list = rot_node.args[0]
  if not isinstance(cat_list, (list, tuple)) or len(cat_list) != 2:
    return None

  neg_node, x1_node = cat_list[0], cat_list[1]
  if (
      not isinstance(neg_node, fx.Node)
      or neg_node.op != "call_function"
      or neg_node.target not in _NEG_TARGETS
  ):
    return None

  if not neg_node.args or not isinstance(neg_node.args[0], fx.Node):
    return None

  x2_node = neg_node.args[0]
  if (
      not isinstance(x1_node, fx.Node)
      or x1_node.op != "call_function"
      or x1_node.target not in _SLICE_TARGETS
  ):
    return None

  if (
      not isinstance(x2_node, fx.Node)
      or x2_node.op != "call_function"
      or x2_node.target not in _SLICE_TARGETS
  ):
    return None

  if not x1_node.args or not x2_node.args:
    return None

  x_source = x1_node.args[0]
  if not isinstance(x_source, fx.Node) or x_source != x2_node.args[0]:
    return None

  x1_dim = x1_node.args[1] if len(x1_node.args) > 1 else 0
  x2_dim = x2_node.args[1] if len(x2_node.args) > 1 else 0
  if x1_dim != x2_dim or not isinstance(x1_dim, int):
    return None

  x1_start = x1_node.args[2] if len(x1_node.args) > 2 else 0
  x1_end = x1_node.args[3] if len(x1_node.args) > 3 else None
  x2_start = x2_node.args[2] if len(x2_node.args) > 2 else 0
  x2_end = x2_node.args[3] if len(x2_node.args) > 3 else None

  if x1_start != 0 or x1_end != x2_start:
    return None

  if not isinstance(x1_end, int):
    return None
  if x2_end is not None and not isinstance(x2_end, int):
    return None

  cat_dim = (
      rot_node.args[1]
      if len(rot_node.args) > 1
      else rot_node.kwargs.get("dim", -1)
  )
  if not isinstance(cat_dim, int):
    return None

  return (x_source, x1_node, x2_node, cat_dim, x1_end, x2_end)


def _match_rope_add(
    add_node: fx.Node,
) -> (
    tuple[fx.Node, fx.Node, fx.Node, fx.Node, fx.Node, int, int, int | None]
    | None
):
  """Matches add(x * cos, rotate_half(x) * sin)."""
  if not isinstance(add_node, fx.Node) or add_node.op != "call_function":
    return None

  if add_node.target not in _ADD_TARGETS:
    return None

  if len(add_node.args) < 2:
    return None

  op1, op2 = add_node.args[0], add_node.args[1]
  if (
      not isinstance(op1, fx.Node)
      or op1.op != "call_function"
      or op1.target not in _MUL_TARGETS
  ):
    return None
  if (
      not isinstance(op2, fx.Node)
      or op2.op != "call_function"
      or op2.target not in _MUL_TARGETS
  ):
    return None

  for mul_x, mul_rot in [(op1, op2), (op2, op1)]:
    if len(mul_rot.args) < 2:
      continue

    for rot_candidate, sin_node in [
        (mul_rot.args[0], mul_rot.args[1]),
        (mul_rot.args[1], mul_rot.args[0]),
    ]:
      if not isinstance(rot_candidate, fx.Node) or not isinstance(
          sin_node, fx.Node
      ):
        continue
      rot_info = _match_rotate_half(rot_candidate)
      if rot_info is None:
        continue

      x_source, x1_node, x2_node, dim, half_dim, full_dim = rot_info

      if len(mul_x.args) < 2:
        continue

      cos_node = None
      if mul_x.args[0] == x_source and isinstance(mul_x.args[1], fx.Node):
        cos_node = mul_x.args[1]
      elif mul_x.args[1] == x_source and isinstance(mul_x.args[0], fx.Node):
        cos_node = mul_x.args[0]

      if cos_node is not None:
        return (
            x_source,
            cos_node,
            sin_node,
            x1_node,
            x2_node,
            dim,
            half_dim,
            full_dim,
        )

  return None


def _erase_unused_chain(node: fx.Node, graph: fx.Graph) -> None:
  """Erases node and any of its input nodes that become dead."""
  to_check = [node]
  while to_check:
    n = to_check.pop(0)
    if n.graph is not None and not n.users:
      inputs = []

      def visit(arg):
        if isinstance(arg, fx.Node):
          inputs.append(arg)

      fx.node.map_arg(n.args, visit)
      fx.node.map_arg(n.kwargs, visit)
      graph.erase_node(n)
      to_check.extend(inputs)


def apply(gm_or_graph: torch.fx.GraphModule | torch.fx.Graph) -> None:
  """Optimizes RoPE using memoized half-width trigonometric tables."""
  graph = (
      gm_or_graph.graph
      if isinstance(gm_or_graph, torch.fx.GraphModule)
      else gm_or_graph
  )

  modified = False
  dead_nodes = []
  # Memoize half-width slices so q and k share the exact same table slices
  slice_cache: dict[tuple[fx.Node, fx.Node, int], tuple[fx.Node, fx.Node]] = {}

  for node in list(graph.nodes):
    rope_info = _match_rope_add(node)
    if rope_info is None:
      continue

    x_source, cos_node, sin_node, x1_node, x2_node, dim, half_dim, full_dim = (
        rope_info
    )

    cache_key = (cos_node, sin_node, half_dim)
    if cache_key in slice_cache:
      c, s = slice_cache[cache_key]
    else:
      with graph.inserting_before(node):
        c = graph.call_function(
            torch.ops.aten.slice.Tensor, (cos_node, dim, 0, half_dim)
        )
        s = graph.call_function(
            torch.ops.aten.slice.Tensor, (sin_node, dim, 0, half_dim)
        )
      slice_cache[cache_key] = (c, s)

    with graph.inserting_before(node):
      # out1 = (x1 * c) - (x2 * s)
      term1 = graph.call_function(torch.ops.aten.mul.Tensor, (x1_node, c))
      term2 = graph.call_function(torch.ops.aten.mul.Tensor, (x2_node, s))
      out1 = graph.call_function(torch.ops.aten.sub.Tensor, (term1, term2))

      # out2 = (x2 * c) + (x1 * s)
      term3 = graph.call_function(torch.ops.aten.mul.Tensor, (x2_node, c))
      term4 = graph.call_function(torch.ops.aten.mul.Tensor, (x1_node, s))
      out2 = graph.call_function(torch.ops.aten.add.Tensor, (term3, term4))

      # out = cat((out1, out2), dim=dim)
      new_out = graph.call_function(
          torch.ops.aten.cat.default, ([out1, out2], dim)
      )

    node.replace_all_uses_with(new_out)
    dead_nodes.append(node)
    modified = True

  if modified:
    for n in dead_nodes:
      _erase_unused_chain(n, graph)
    graph.lint()
    if isinstance(gm_or_graph, torch.fx.GraphModule):
      gm_or_graph.recompile()
