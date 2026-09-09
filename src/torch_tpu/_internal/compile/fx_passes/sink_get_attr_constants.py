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

"""Sinks get_attr constant arguments from root module into child submodules."""

import torch
from torch.export.unflatten import _assign_attr
from torch.export.unflatten import _AttrKind


def _map_call_module_nodes(
    gm: torch.fx.GraphModule,
) -> dict[str, list[torch.fx.Node]]:
  """Maps each submodule name to the call_module nodes that invoke it.

  A submodule may be invoked from more than one call site. All of them must be
  tracked so that a change to the submodule signature can be reflected at every
  call site rather than just one.
  """
  mapping: dict[str, list[torch.fx.Node]] = {}
  for node in gm.graph.nodes:
    if node.op == "call_module":
      mapping.setdefault(node.target, []).append(node)
  return mapping


def _get_placeholders(graph: torch.fx.Graph) -> list[torch.fx.Node]:
  """Returns the placeholder (input) nodes of a graph, in order."""
  return [node for node in graph.nodes if node.op == "placeholder"]


def _get_sinkable_constant(
    gm: torch.fx.GraphModule,
    arg: object,
) -> tuple[str, torch.Tensor] | None:
  """Returns the (name, tensor) a call-site arg refers to, if it can be sunk.

  A call-site argument can be sunk when it is a get_attr node whose target
  resolves to a torch.Tensor on the root module. _assign_attr with
  _AttrKind.CONSTANT only accepts tensors/script objects, so anything
  else must be left as a normal runtime input.

  Args:
    gm: The root GraphModule that owns the attribute.
    arg: A single argument from a call_module node.

  Returns:
    A (attr_name, attr_val) tuple for the referenced constant tensor, or None if
    the arg is not a sinkable constant.
  """
  if not (isinstance(arg, torch.fx.Node) and arg.op == "get_attr"):
    return None
  attr_name = str(arg.target)
  # pylint: disable-next=protected-access
  attr_val = torch.fx.graph_module._get_attr(gm, attr_name)
  if not isinstance(attr_val, torch.Tensor):
    return None
  return attr_name, attr_val


def _replace_placeholder_with_get_attr(
    submod: torch.fx.GraphModule,
    placeholder: torch.fx.Node,
    attr_name: str,
    attr_val: torch.Tensor,
) -> None:
  """Registers a constant on a submodule and swaps its placeholder for get_attr."""
  # Register the constant directly on the submodule.
  # pylint: disable-next=protected-access
  _assign_attr(attr_val, submod, attr_name, _AttrKind.CONSTANT)

  with submod.graph.inserting_before(placeholder):
    get_attr_node = submod.graph.get_attr(attr_name)
    if hasattr(placeholder, "meta"):
      get_attr_node.meta = placeholder.meta.copy()
    placeholder.replace_all_uses_with(get_attr_node)
  submod.graph.erase_node(placeholder)


def _sink_constants_for_submodule(
    gm: torch.fx.GraphModule,
    submod: torch.fx.GraphModule,
    submod_name: str,
    call_site_nodes: list[torch.fx.Node],
) -> None:
  """Sinks the constant get_attr arguments shared by all calls to a submodule.

  A placeholder is only sunk when every call site passes the same constant
  get_attr at that position. When it is, the constant becomes an in-submodule
  get_attr node and the argument is removed from every call site so that all
  call-site signatures stay consistent with the modified submodule.

  Args:
    gm: The root GraphModule that owns the constants.
    submod: The child submodule whose placeholders may be sunk.
    submod_name: The attribute name of `submod` on `gm`.
    call_site_nodes: All call_module nodes that invoke `submod`.

  Raises:
    ValueError: If a call site's argument count does not match the number of
      submodule placeholders.
  """
  placeholders = _get_placeholders(submod.graph)
  if not placeholders:
    return

  for call_site_node in call_site_nodes:
    if len(placeholders) != len(call_site_node.args):
      raise ValueError(
          f"Mismatch between placeholders ({len(placeholders)}) and args"
          f" ({len(call_site_node.args)}) for {submod_name}"
      )

  sunk_positions = []
  for position, placeholder in enumerate(placeholders):
    first = _get_sinkable_constant(gm, call_site_nodes[0].args[position])
    if first is None:
      continue
    attr_name, attr_val = first

    # Only sink when every call site passes the same constant at this position.
    same_across_call_sites = all(
        (other := _get_sinkable_constant(gm, node.args[position])) is not None
        and other[0] == attr_name
        for node in call_site_nodes[1:]
    )
    if not same_across_call_sites:
      continue

    _replace_placeholder_with_get_attr(submod, placeholder, attr_name, attr_val)
    sunk_positions.append(position)

  if sunk_positions:
    sunk = set(sunk_positions)
    for call_site_node in call_site_nodes:
      call_site_node.args = tuple(
          arg
          for position, arg in enumerate(call_site_node.args)
          if position not in sunk
      )
    submod.graph.lint()
    submod.recompile()


def _erase_dead_get_attrs(graph: torch.fx.Graph) -> None:
  """Removes get_attr nodes that no longer have any users."""
  for node in list(graph.find_nodes(op="get_attr")):
    if not node.users:
      graph.erase_node(node)


def apply(gm: torch.fx.GraphModule) -> None:
  """Sinks get_attr arguments from root module into child submodules.

  When split_module partitions a graph, it hoists all get_attr nodes (e.g.
  constant tensors, parameters) to the root GraphModule and passes them as
  placeholder inputs into each submodule.

  This pass restores them as get_attr nodes inside each child submodule,
  removing them from the submodule signature and call site. This enables us to
  profit from optimizations such as constant folding.

  Args:
    gm: The split GraphModule whose submodules should have get_attr nodes sunk.
  """
  call_module_nodes = _map_call_module_nodes(gm)

  for submod_name, submod in gm.named_children():
    if not isinstance(submod, torch.fx.GraphModule):
      continue
    call_site_nodes = call_module_nodes.get(submod_name)
    if not call_site_nodes:
      continue
    _sink_constants_for_submodule(gm, submod, submod_name, call_site_nodes)

  _erase_dead_get_attrs(gm.graph)

  gm.graph.lint()
  gm.recompile()
