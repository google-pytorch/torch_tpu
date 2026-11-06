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

"""Clones placeholder nodes that the graph both mutates and returns."""

import operator
from typing import Any

import torch
from torch._functorch._aot_autograd import descriptors as _aot_descriptors
from torch._library import utils as library_utils

# Ops whose results can share storage with an input although their schemas do
# not annotate an alias.
_UNANNOTATED_ALIAS_OPS = frozenset({
    torch.ops.aten._unsafe_view.default,  # pylint: disable=protected-access
    torch.ops.aten.lift.default,
    torch.ops.aten.set.source_Tensor,
    torch.ops.aten.unsafe_chunk.default,
    torch.ops.aten.unsafe_split.Tensor,
    torch.ops.aten.unsafe_split_with_sizes.default,
})

# After `self.set_(source)`, `self` and the result share the storage of
# `source`, which the schema does not annotate.
_SET_SOURCE_OPS = frozenset({
    torch.ops.aten.set_.source_Tensor,
    torch.ops.aten.set_.source_Tensor_storage_offset,
})


def _normalize_gm_and_graph(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> tuple[torch.fx.GraphModule | None, torch.fx.Graph]:
  """Extracts the owning GraphModule (if any) and the Graph."""
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    return gm_or_graph, gm_or_graph.graph
  return getattr(gm_or_graph, "owning_module", None), gm_or_graph


def _get_output_node(graph: torch.fx.Graph) -> torch.fx.Node | None:
  """Finds and validates that the graph has at most one output node."""
  output_nodes = [node for node in graph.nodes if node.op == "output"]
  if not output_nodes:
    return None

  assert (
      len(output_nodes) == 1
  ), f"Expected exactly one output node, got {len(output_nodes)} output nodes."
  return output_nodes[0]


def _get_nodes(value: Any) -> set[torch.fx.Node]:
  """Extracts all FX nodes in a (possibly nested) argument value."""
  flat_args, _ = torch.utils._pytree.tree_flatten(value)
  return {arg for arg in flat_args if isinstance(arg, torch.fx.Node)}


def _root_descriptor(desc: Any) -> Any:
  """Unwraps `SubclassGetAttr*` descriptors to the outer tensor's descriptor."""
  while isinstance(
      desc,
      (
          _aot_descriptors.SubclassGetAttrAOTInput,
          _aot_descriptors.SubclassGetAttrAOTOutput,
      ),
  ):
    desc = desc.base
  return desc


def _aliased(
    aliases: dict[torch.fx.Node, set[torch.fx.Node]], value: Any
) -> set[torch.fx.Node]:
  """Returns the placeholders that the nodes in `value` may alias."""
  return set().union(*(aliases.get(n, ()) for n in _get_nodes(value)))


def _track_op_overload(
    node: torch.fx.Node,
    aliases: dict[torch.fx.Node, set[torch.fx.Node]],
    mutated: set[torch.fx.Node],
) -> None:
  """Records what an op with a schema may write to and what its result aliases.

  Args:
    node: A `call_function` node whose target is an `OpOverload`.
    aliases: Maps each node seen so far to the placeholders it may alias.
      Updated in place.
    mutated: The placeholders that may be written to. Updated in place.
  """
  op = node.target
  assert isinstance(
      op, torch._ops.OpOverload  # pylint: disable=protected-access
  )
  schema = op._schema  # pylint: disable=protected-access
  arguments = list(library_utils.zip_schema(schema, node.args, node.kwargs))
  # Arguments without alias_info are only read, and no result shares their
  # storage.
  annotated = [
      (argument.alias_info, _aliased(aliases, value))
      for argument, value in arguments
      if argument.alias_info is not None
  ]
  for alias_info, value_aliases in annotated:
    if alias_info.is_write:
      mutated.update(value_aliases)

  if any(r.alias_info is not None for r in schema.returns):
    aliases[node] = set().union(*(a for _, a in annotated))

  # `set_` writes to `self`, so unlike `_UNANNOTATED_ALIAS_OPS` it needs the
  # schema handling above. The schema leaves out that `self` and the result
  # share the storage of `source` after the call.
  if node.target in _SET_SOURCE_OPS:
    values = {argument.name: value for argument, value in arguments}
    source = _aliased(aliases, values["source"])
    for n in _get_nodes(values["self"]) | {node}:
      aliases[n] = aliases.get(n, set()) | source


def _find_mutated_placeholders(
    graph: torch.fx.Graph, output_node: torch.fx.Node
) -> set[torch.fx.Node]:
  """Returns the placeholders whose storage may be written to.

  A write to a `Tensor(a!)` argument counts as a write to every placeholder
  the argument may alias, through views, `getitem`, `_UNANNOTATED_ALIAS_OPS`
  or `_SET_SOURCE_OPS`. Targets without a schema, other than `getitem`, are
  assumed to write to all of their inputs and to return aliases of them.
  Inputs that AOTAutograd copies a new value into after the graph runs count
  as written too.
  """
  mutated_inputs = set()
  # Only AOTAutograd sets `desc`. Graphs from make_fx or symbolic_trace, or
  # built by hand, have none.
  for desc in output_node.meta.get("desc") or ():
    desc = _root_descriptor(desc)
    if isinstance(desc, _aot_descriptors.InputMutationAOTOutput):
      mutated_inputs.add(_root_descriptor(desc.mutated_input))

  # Maps each node to the placeholders whose storage it may share.
  aliases: dict[torch.fx.Node, set[torch.fx.Node]] = {}
  mutated: set[torch.fx.Node] = set()
  for node in graph.nodes:
    if node.op == "placeholder":
      aliases[node] = {node}
      if _root_descriptor(node.meta.get("desc")) in mutated_inputs:
        mutated.add(node)
    elif node.op in ("get_attr", "output"):
      continue
    elif node.op == "call_function" and node.target in _UNANNOTATED_ALIAS_OPS:
      # These ops write nothing, and their schemas leave out that the result
      # shares storage with the inputs.
      aliases[node] = _aliased(aliases, node.all_input_nodes)
    elif node.op == "call_function" and isinstance(
        node.target, torch._ops.OpOverload  # pylint: disable=protected-access
    ):
      # Only OpOverloads have a schema that says what they write and alias.
      _track_op_overload(node, aliases, mutated)
    elif node.op == "call_function" and node.target is operator.getitem:
      # `getitem` picks an element of a list or tuple, such as one result of
      # `split`. It shares that element's storage but writes nothing.
      aliases[node] = _aliased(aliases, node.all_input_nodes)
    else:
      # Other targets have no schema to read: higher-order ops such as
      # `auto_functionalized`, Python callables, `call_module` and `call_method`
      # nodes. For safety, assume they write to and alias all of their inputs.
      aliases[node] = _aliased(aliases, node.all_input_nodes)
      mutated |= aliases[node]
  return mutated


def _is_returned_compute_placeholder(
    node: torch.fx.Node,
    returned_nodes: set[torch.fx.Node],
) -> bool:
  """Checks if a node is a torch.Tensor output placeholder consumed by compute operations."""
  if node.op != "placeholder" or node not in returned_nodes:
    return False
  if "val" in node.meta and not isinstance(node.meta["val"], torch.Tensor):
    return False
  return any(user.op != "output" for user in node.users)


def _find_placeholders_to_clone(
    graph: torch.fx.Graph,
    output_node: torch.fx.Node,
) -> list[torch.fx.Node]:
  """Identifies all placeholder nodes that need cloning."""
  returned_nodes = _get_nodes(output_node.args)
  mutated = _find_mutated_placeholders(graph, output_node)
  return [
      node
      for node in graph.nodes
      if _is_returned_compute_placeholder(node, returned_nodes)
      and node in mutated
  ]


def _clone_and_replace_placeholder(
    graph: torch.fx.Graph,
    placeholder: torch.fx.Node,
) -> None:
  """Clones a placeholder and redirects all downstream uses to the clone."""
  with graph.inserting_after(placeholder):
    p_clone = graph.call_function(
        torch.ops.aten.clone.default, args=(placeholder,)
    )
    if hasattr(placeholder, "meta"):
      p_clone.meta = placeholder.meta.copy()

    placeholder.replace_all_uses_with(
        p_clone, delete_user_cb=lambda u, clone=p_clone: u != clone
    )


def apply(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> None:
  """Clones placeholder nodes that are returned in the output and mutated.

  When an FX graph contains in-place operations that mutate input placeholders,
  the graph's output node may reference the input placeholder directly.

  If a pass (e.g. `split_module`) alters the graph structure such that a mutated
  placeholder is no longer in the output, the output will be a passthrough
  meaning that the user won't receive the mutated value.

  A returned placeholder that is only read stays a passthrough. AOTAutograd
  forward graphs return every parameter that backward needs, and cloning them
  would keep a copy of each one alive until backward.

  Args:
    gm_or_graph: The FX GraphModule or Graph to modify.
  """
  gm, graph = _normalize_gm_and_graph(gm_or_graph)

  output_node = _get_output_node(graph)
  if output_node is None:
    return

  placeholders_to_clone = _find_placeholders_to_clone(graph, output_node)
  if not placeholders_to_clone:
    return

  for placeholder in placeholders_to_clone:
    _clone_and_replace_placeholder(graph, placeholder)

  graph.lint()
  if gm is not None:
    gm.recompile()
