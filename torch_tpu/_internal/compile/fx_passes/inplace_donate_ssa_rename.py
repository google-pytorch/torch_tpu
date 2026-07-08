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

"""SSA-renames donated inputs for pallas inplace_donate=True ops.

Under torch.compile, the eager per-call rebind cannot fire between calls
fused into one executable, so the naive FX graph has every donating call
reading the same donated SSA value — XLA rejects the double donation.
:func:`apply` rewrites every later use of a donated input to the donating
call's aliased output, giving each donation a distinct SSA value.
:func:`find_donate_aliases` then returns each placeholder's chain
*terminus*, which the compile backend plumbs as an extra executable output
and rebinds the input tensor's storage to after execution.

Ops are identified by the ``_pallas_inplace_donate_argnums`` attribute
pallas stamps on the ``OpOverload`` at registration.
"""

import operator

import torch

# Keep in sync with pallas._annotate_inplace_donate.
_INPLACE_DONATE_ARGNUMS_ATTR = "_pallas_inplace_donate_argnums"

# Optional {donated_input_index: aliased_output_index} map for ops whose
# donated input index differs from the aliased output index; identity when
# absent.
_INPLACE_DONATE_OUTPUT_MAP_ATTR = "_pallas_inplace_donate_output_map"


def _output_index(node: torch.fx.Node, donate_idx: int) -> int:
  """Donated input index -> aliased output index (identity when unmapped)."""
  output_map = getattr(node.target, _INPLACE_DONATE_OUTPUT_MAP_ATTR, None)
  if output_map is None:
    return donate_idx
  return output_map.get(donate_idx, donate_idx)


def _node_position_map(graph: torch.fx.Graph) -> dict[torch.fx.Node, int]:
  # FX graphs stay topologically ordered, so enumeration gives "comes after".
  return {node: idx for idx, node in enumerate(graph.nodes)}


def _inplace_donate_argnums(
    node: torch.fx.Node,
) -> tuple[int, ...] | None:
  """Donated input indices for an inplace-donate call node, else None."""
  if node.op != "call_function":
    return None
  return getattr(node.target, _INPLACE_DONATE_ARGNUMS_ATTR, None)


def _get_arg_value(
    call_node: torch.fx.Node, idx: int,
) -> torch.fx.Node | None:
  """FX value at parameter index ``idx``: positional args first, then
  kwargs via the schema's parameter name; None when unbound."""
  if idx < len(call_node.args):
    return call_node.args[idx]
  schema = getattr(call_node.target, "_schema", None)
  if schema is None:
    return None
  schema_args = schema.arguments
  if idx >= len(schema_args):
    return None
  return call_node.kwargs.get(schema_args[idx].name, None)


def _select_output_node(
    graph: torch.fx.Graph,
    call_node: torch.fx.Node,
    donate_idx: int,
) -> torch.fx.Node:
  """FX node yielding the donated output: the call node itself for a
  single-output op (aliased output must be 0), else a getitem inserted or
  reused right after the call."""
  output_idx = _output_index(call_node, donate_idx)
  val = call_node.meta.get("val")
  if not isinstance(val, (tuple, list)):
    # Single-output kernel — the call node IS the donated output.
    if output_idx != 0:
      schema = getattr(call_node.target, "_schema", None)
      qualname = schema.name if schema is not None else str(call_node.target)
      raise RuntimeError(
          f"inplace_donate FX pass: single-output kernel "
          f"{qualname!r} cannot donate to output index {output_idx}; "
          "single-output kernels must alias to output 0."
      )
    return call_node

  # Reuse an existing getitem — keeps the rewrite idempotent.
  for user in call_node.users:
    if (
        user.op == "call_function"
        and user.target is operator.getitem
        and len(user.args) == 2
        and user.args[0] is call_node
        and user.args[1] == output_idx
    ):
      return user

  with graph.inserting_after(call_node):
    return graph.call_function(operator.getitem, (call_node, output_idx))


def apply(gm_or_graph: torch.fx.GraphModule | torch.fx.Graph) -> None:
  """Runs the SSA rename across `gm_or_graph`.

  Safe to call on a graph with no inplace-donate ops — it is a no-op
  in that case. Idempotent on a graph it has already processed.
  """
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    graph = gm_or_graph.graph
  else:
    graph = gm_or_graph

  position = _node_position_map(graph)

  changed = False
  for call_node in list(graph.nodes):
    donate_argnums = _inplace_donate_argnums(call_node)
    if donate_argnums is None:
      continue

    call_pos = position[call_node]
    for donate_idx in donate_argnums:
      donated_input = _get_arg_value(call_node, donate_idx)
      if not isinstance(donated_input, torch.fx.Node):
        # No SSA value bound at this parameter (defaulted / literal).
        continue

      output_node = _select_output_node(graph, call_node, donate_idx)
      if output_node is donated_input:
        continue

      if output_node not in position:
        position[output_node] = call_pos + 0.5  # type: ignore[assignment]

      for user in list(donated_input.users):
        if user is output_node:
          continue
        user_pos = position.get(user)
        if user_pos is None or user_pos <= call_pos:
          continue
        user.replace_input_with(donated_input, output_node)
        changed = True

  if changed:
    graph.lint()


def _follow_donate_chain_to_terminus(
    start: torch.fx.Node,
) -> torch.fx.Node | None:
  """Returns the last donating call's SSA value in the chain rooted at
  ``start``; None if ``start`` feeds no donating call or the chain is
  structurally broken (missing donated getitem)."""
  current = start
  advanced = False
  visited: set[torch.fx.Node] = {current}
  while True:
    next_output: torch.fx.Node | None = None
    for user in current.users:
      donate_argnums = _inplace_donate_argnums(user)
      if donate_argnums is None:
        continue
      donate_idx: int | None = None
      for didx in donate_argnums:
        if _get_arg_value(user, didx) is current:
          donate_idx = didx
          break
      if donate_idx is None:
        continue

      output_idx = _output_index(user, donate_idx)
      val = user.meta.get("val")
      if not isinstance(val, (tuple, list)):
        if output_idx != 0:
          return None
        candidate = user
      else:
        # Multi-output op: the donated output is `getitem(user, output_idx)`,
        # inserted by apply(); its absence means no meaningful alias.
        candidate = None
        for sub_user in user.users:
          if (
              sub_user.op == "call_function"
              and sub_user.target is operator.getitem
              and len(sub_user.args) == 2
              and sub_user.args[0] is user
              and sub_user.args[1] == output_idx
          ):
            candidate = sub_user
            break
        if candidate is None:
          return None
      next_output = candidate
      break

    if next_output is None or next_output is current:
      break
    if next_output in visited:
      return None
    visited.add(next_output)
    current = next_output
    advanced = True

  return current if advanced else None


def find_donate_aliases(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> list[tuple[int, torch.fx.Node]]:
  """Returns (placeholder_index, terminus_node) for each placeholder that
  feeds a donation chain. Must run after :func:`apply`. The compile
  backend plumbs each terminus as an extra executable output and rebinds
  the input tensor's storage to it after execution."""
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    graph = gm_or_graph.graph
  else:
    graph = gm_or_graph

  aliases: list[tuple[int, torch.fx.Node]] = []
  seen_termini: set[torch.fx.Node] = set()
  for ph_idx, placeholder in enumerate(graph.find_nodes(op="placeholder")):
    terminus = _follow_donate_chain_to_terminus(placeholder)
    if terminus is None:
      continue
    if terminus in seen_termini:
      # Don't rebind the same buffer twice.
      continue
    seen_termini.add(terminus)
    aliases.append((ph_idx, terminus))
  return aliases


def raise_if_present(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> None:
  """Raises if any inplace-donate op appears — for compile paths that do
  not run :func:`apply` (e.g. DynamicCompiler), where the graph would
  otherwise hit XLA's opaque double-donation error."""
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    graph = gm_or_graph.graph
  else:
    graph = gm_or_graph
  for node in graph.nodes:
    if _inplace_donate_argnums(node) is None:
      continue
    schema = getattr(node.target, "_schema", None)
    qualname = schema.name if schema is not None else str(node.target)
    raise NotImplementedError(
        f"Pallas op {qualname!r} is registered with inplace_donate=True but"
        " the current compile path does not run the inplace-donate FX"
        " transform (currently only StaticCompiler does). The chain would"
        " reach XLA with the same SSA value referenced by every donating"
        " call and XLA would reject it. Either compile with static shapes"
        " (e.g. ``torch.compile(dynamic=False)``) or drop"
        " inplace_donate=True from this op."
    )
