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

"""FX Graph transformation pass to roll sequential nested compile regions into a scan loop."""

import operator
from typing import Any
import torch
import torch.fx


class _RolledScanCombineModule(torch.nn.Module):
  """Wrapper module for the scan combine step function."""

  def __init__(self, submod: Any, num_carries: int = 1):
    super().__init__()
    self.submod = submod
    self.num_carries = num_carries

  def forward(self, *args: Any) -> Any:
    out = self.submod(*args)
    return out, out


def _is_invoke_subgraph_node(node: torch.fx.Node) -> bool:
  """Checks if an FX node is a call to torch.ops.higher_order.invoke_subgraph."""
  if node.op != "call_function":
    return False
  target = node.target
  if hasattr(torch.ops, "higher_order") and hasattr(
      torch.ops.higher_order, "invoke_subgraph"
  ):
    hop = torch.ops.higher_order.invoke_subgraph
    if target is hop or getattr(hop, "default", None) is target:
      return True
  target_name = getattr(target, "_name", "") or getattr(target, "__name__", "")
  return "invoke_subgraph" in target_name


def _get_submodule(graph_module: torch.fx.GraphModule, target: Any) -> Any:
  """Resolves a submodule by string, get_attr node, or returns the module directly."""
  if isinstance(target, str):
    try:
      return graph_module.get_submodule(target)
    except (AttributeError, KeyError):
      return getattr(graph_module, target, target)
  if isinstance(target, torch.fx.Node) and target.op == "get_attr":
    attr_name = str(target.target)
    try:
      return graph_module.get_submodule(attr_name)
    except (AttributeError, KeyError):
      return getattr(graph_module, attr_name, target)
  return target


def _get_nodes_list(
    gm: torch.fx.GraphModule | torch.fx.Graph,
) -> list[torch.fx.Node]:
  if isinstance(gm, torch.fx.GraphModule):
    return list(gm.graph.nodes)
  return list(gm.nodes)


def _are_graphs_isomorphic(gm1: Any, gm2: Any) -> bool:
  """Checks if two GraphModules (or Graphs) have isomorphic operation graphs."""
  if gm1 is gm2:
    return True
  if not isinstance(
      gm1, (torch.fx.GraphModule, torch.fx.Graph)
  ) or not isinstance(gm2, (torch.fx.GraphModule, torch.fx.Graph)):
    return False

  nodes1 = _get_nodes_list(gm1)
  nodes2 = _get_nodes_list(gm2)

  if len(nodes1) != len(nodes2):
    return False

  node_map: dict[torch.fx.Node, torch.fx.Node] = {}
  for n1, n2 in zip(nodes1, nodes2):
    if n1.op != n2.op:
      return False
    if n1.op == "placeholder":
      node_map[n1] = n2
      continue
    if n1.target != n2.target:
      return False
    if len(n1.args) != len(n2.args):
      return False
    for a1, a2 in zip(n1.args, n2.args):
      if isinstance(a1, torch.fx.Node):
        if not isinstance(a2, torch.fx.Node) or node_map.get(a1) is not a2:
          return False
      elif isinstance(a1, (list, tuple)):
        if not isinstance(a2, (list, tuple)) or len(a1) != len(a2):
          return False
        for elem1, elem2 in zip(a1, a2):
          if isinstance(elem1, torch.fx.Node):
            if (
                not isinstance(elem2, torch.fx.Node)
                or node_map.get(elem1) is not elem2
            ):
              return False
          elif elem1 != elem2:
            return False
      elif a1 != a2:
        return False
    node_map[n1] = n2
  return True


def _are_submodules_compatible(
    submod1: Any, submod2: Any, node1: torch.fx.Node, node2: torch.fx.Node
) -> bool:
  """Checks if two submodules referenced by invoke_subgraph nodes are isomorphic."""
  if submod1 is submod2:
    return True

  code1 = getattr(submod1, "code", None)
  code2 = getattr(submod2, "code", None)
  if code1 is not None and code2 is not None and code1 == code2:
    return True

  # Check exact identifier in args[1]
  id1 = node1.args[1] if len(node1.args) > 1 else None
  id2 = node2.args[1] if len(node2.args) > 1 else None
  if id1 and id2 and id1 == id2:
    return True

  if _are_graphs_isomorphic(submod1, submod2):
    return True

  return False


def _get_carry_source(node: torch.fx.Node) -> torch.fx.Node | None:
  """Returns the producer node if `node` is a direct node or a getitem output."""
  if node.op == "call_function" and node.target in (
      operator.getitem,
      getattr(operator, "__getitem__", None),
  ):
    return node.args[0]
  return node


def _find_sequential_chains(
    graph_module: torch.fx.GraphModule,
) -> list[list[torch.fx.Node]]:
  """Finds maximal sequences of consecutive, compatible invoke_subgraph nodes."""
  nodes = list(graph_module.graph.nodes)
  visited = set()
  chains = []

  for i, node in enumerate(nodes):
    if node in visited or not _is_invoke_subgraph_node(node):
      continue

    # A scan chain requires at least 1 parameter input in args[3:] to scan over.
    if len(node.args) <= 3:
      continue

    current_chain = [node]
    visited.add(node)
    curr_node = node

    for next_node in nodes[i + 1 :]:
      if next_node in visited:
        continue
      if not _is_invoke_subgraph_node(next_node):
        continue

      # Check argument length
      if len(next_node.args) != len(curr_node.args):
        break

      # Check subgraph compatibility
      submod_curr = _get_submodule(graph_module, curr_node.args[0])
      submod_next = _get_submodule(graph_module, next_node.args[0])
      if not _are_submodules_compatible(
          submod_curr, submod_next, curr_node, next_node
      ):
        break

      # Check carry connection: next_node.args[2] must come from curr_node
      carry_arg = next_node.args[2] if len(next_node.args) > 2 else None
      if carry_arg is None:
        break
      carry_source = _get_carry_source(carry_arg)
      if carry_source is not curr_node:
        break

      current_chain.append(next_node)
      visited.add(next_node)
      curr_node = next_node

    if len(current_chain) >= 2:
      chains.append(current_chain)

  return chains


def _collect_upstream_nodes(node: torch.fx.Node) -> list[torch.fx.Node]:
  """Returns all upstream FX nodes that `node` depends on in topological order."""
  deps = []
  visited = set()

  def dfs(curr: torch.fx.Node):
    if not isinstance(curr, torch.fx.Node) or curr in visited:
      return
    visited.add(curr)
    flat_args = []
    for a in curr.args:
      if isinstance(a, torch.fx.Node):
        flat_args.append(a)
      elif isinstance(a, (list, tuple)):
        flat_args.extend(x for x in a if isinstance(x, torch.fx.Node))
    for v in curr.kwargs.values():
      if isinstance(v, torch.fx.Node):
        flat_args.append(v)
      elif isinstance(v, (list, tuple)):
        flat_args.extend(x for x in v if isinstance(x, torch.fx.Node))
    for arg in flat_args:
      dfs(arg)
    deps.append(curr)

  dfs(node)
  return deps


def _hoist_param_dependencies_before(
    graph: torch.fx.Graph,
    chain: list[torch.fx.Node],
    first_node: torch.fx.Node,
) -> bool:
  """Hoists all dependencies of parameter arguments before first_node.

  Returns False if any parameter dependency depends on chain nodes (cannot
  hoist).
  """
  chain_nodes = set(chain)
  node_order = {node: i for i, node in enumerate(graph.nodes)}
  first_pos = node_order[first_node]

  nodes_to_hoist = []
  visited = set()

  for node in chain:
    for param_arg in node.args[3:]:
      if isinstance(param_arg, torch.fx.Node):
        deps = _collect_upstream_nodes(param_arg)
        for dep in deps:
          if dep in chain_nodes:
            return False
          if dep not in visited:
            visited.add(dep)
            nodes_to_hoist.append(dep)

  # Check that none of the nodes to hoist depend on any chain node or getitem of chain node
  for dep in nodes_to_hoist:
    flat_args = []
    for a in dep.args:
      if isinstance(a, torch.fx.Node):
        flat_args.append(a)
      elif isinstance(a, (list, tuple)):
        flat_args.extend(x for x in a if isinstance(x, torch.fx.Node))
    for arg in flat_args:
      if arg in chain_nodes:
        return False
      if (
          arg.op == "call_function"
          and arg.target
          in (operator.getitem, getattr(operator, "__getitem__", None))
          and arg.args[0] in chain_nodes
      ):
        return False

  # Hoist nodes that appear after first_node before first_node in topological order
  for dep in nodes_to_hoist:
    if node_order[dep] > first_pos:
      first_node.prepend(dep)

  return True


def apply(gm_or_graph: torch.fx.GraphModule | torch.fx.Graph) -> None:
  """Detects sequential invoke_subgraph calls and rolls them into a scan loop."""
  if isinstance(gm_or_graph, torch.fx.Graph):
    if gm_or_graph.owning_module is not None and isinstance(
        gm_or_graph.owning_module, torch.fx.GraphModule
    ):
      graph_module = gm_or_graph.owning_module
    else:
      return
  else:
    graph_module = gm_or_graph

  chains = _find_sequential_chains(graph_module)
  if not chains:
    return

  graph = graph_module.graph

  for chain in chains:
    first_node = chain[0]

    # 1. Hoist parameter dependencies before first_node
    if not _hoist_param_dependencies_before(graph, chain, first_node):
      continue

    # 2. Resolve subgraph
    submod = _get_submodule(graph_module, first_node.args[0])

    # 3. Initial carry (args[2])
    init_carry = first_node.args[2]
    if isinstance(init_carry, (list, tuple)):
      inits = list(init_carry)
    else:
      inits = [init_carry]
    num_carries = len(inits)

    # Determine number of submodule outputs
    num_submod_outputs = 1
    if isinstance(submod, torch.fx.GraphModule):
      output_nodes = [n for n in submod.graph.nodes if n.op == "output"]
      if output_nodes:
        out_args = output_nodes[0].args[0]
        if isinstance(out_args, (tuple, list)):
          num_submod_outputs = len(out_args)
        else:
          num_submod_outputs = 1
    num_ys = max(0, num_submod_outputs - num_carries)

    # 4. Collect and stack parameters across all steps in the chain (args[3:])
    stacked_params = []
    with graph.inserting_before(first_node):
      for param_idx in range(3, len(first_node.args)):
        step_params = [node.args[param_idx] for node in chain]
        stacked_node = graph.call_function(
            torch.stack,
            args=(step_params,),
            kwargs={"dim": 0},
        )
        stacked_params.append(stacked_node)

    # 5. Insert torch.ops.higher_order.scan with the submod GraphModule
    with graph.inserting_before(first_node):
      scan_submod_name = f"scan_body_{len(graph_module._modules)}"
      graph_module.register_module(scan_submod_name, submod)
      submod_get_attr = graph.get_attr(scan_submod_name)
      scan_target = torch.ops.higher_order.scan
      scan_node = graph.call_function(
          scan_target,
          args=(submod_get_attr, inits, stacked_params, []),
      )
      final_carry_nodes = [
          graph.call_function(
              operator.getitem,
              args=(scan_node, i),
          )
          for i in range(num_carries)
      ]
      stacked_ys_nodes = [
          graph.call_function(
              operator.getitem,
              args=(scan_node, num_carries + j),
          )
          for j in range(num_ys)
      ]

    # 6. Replace uses of chain nodes outside the chain
    num_steps = len(chain)
    for step_idx, node in enumerate(chain):
      for user in list(node.users):
        if user in chain:
          continue
        if user.op == "call_function" and user.target in (
            operator.getitem,
            getattr(operator, "__getitem__", None),
        ):
          out_idx = user.args[1] if len(user.args) > 1 else 0
          if isinstance(out_idx, int):
            if out_idx < num_carries:
              if step_idx == num_steps - 1:
                user.replace_all_uses_with(final_carry_nodes[out_idx])
            else:
              y_idx = out_idx - num_carries
              if y_idx < len(stacked_ys_nodes):
                with graph.inserting_before(first_node):
                  yk_stacked = stacked_ys_nodes[y_idx]
                  yk_step = graph.call_function(
                      operator.getitem,
                      args=(yk_stacked, step_idx),
                  )
                user.replace_all_uses_with(yk_step)
        else:
          if step_idx == num_steps - 1:
            if num_carries == 1:
              user.replace_input_with(node, final_carry_nodes[0])

    # 7. Erase all getitems of chain nodes, then chain nodes in reverse order
    for node in reversed(chain):
      for user in list(node.users):
        if user.op == "call_function" and user.target in (
            operator.getitem,
            getattr(operator, "__getitem__", None),
        ):
          if len(user.users) == 0:
            graph.erase_node(user)
      if len(node.users) == 0:
        graph.erase_node(node)

  graph.lint()
  graph_module.recompile()
