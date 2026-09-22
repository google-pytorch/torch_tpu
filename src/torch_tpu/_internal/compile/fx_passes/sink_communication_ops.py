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

"""Generic FX graph pass for collective communication pushdown and scheduling.

Implements collective sinking into higher-order op subgraphs as well as the trio
of collective scheduling algorithms (raise_comms, sink_waits, and
reorder_communication_preserving_peak_memory) to maximize compute/communication
overlap.
"""

import copy
import heapq
import sys
from typing import Any
import torch
import torch.fx


def _is_wait_tensor_node(node: Any) -> bool:
  """Checks if an FX node is a call to torch.ops._c10d_functional.wait_tensor."""
  if not isinstance(node, torch.fx.Node) or node.op != "call_function":
    return False
  target = node.target
  if hasattr(torch.ops, "_c10d_functional") and hasattr(
      torch.ops._c10d_functional, "wait_tensor"
  ):
    wt = torch.ops._c10d_functional.wait_tensor
    if target is wt or getattr(wt, "default", None) is target:
      return True
  target_name = getattr(target, "_name", "") or getattr(target, "__name__", "")
  return "wait_tensor" in target_name


def _is_c10d_comm_node(node: Any) -> bool:
  """Checks if an FX node is a functional collective communication op (excluding wait)."""
  if not isinstance(node, torch.fx.Node) or node.op != "call_function":
    return False
  if _is_wait_tensor_node(node):
    return False
  target = node.target
  target_name = str(target)
  return any(
      k in target_name
      for k in (
          "all_gather_into_tensor",
          "reduce_scatter_tensor",
          "all_to_all_single",
          "all_reduce",
          "broadcast",
      )
  )


def _extract_comm_chain(
    node: Any,
) -> (
    tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node, list[torch.fx.Node]]
    | None
):
  """If node is produced by comm -> wait -> [view_ops], returns (comm_node, wait_node, raw_input, view_nodes)."""
  if not isinstance(node, torch.fx.Node) or node.op != "call_function":
    return None

  view_nodes = []
  curr = node
  while isinstance(curr, torch.fx.Node) and curr.op == "call_function":
    if _is_wait_tensor_node(curr):
      break
    target_name = str(curr.target)
    if any(
        k in target_name
        for k in ("view", "reshape", "as_strided", "expand", "t", "transpose")
    ):
      view_nodes.append(curr)
      if curr.args and isinstance(curr.args[0], torch.fx.Node):
        curr = curr.args[0]
      else:
        return None
    else:
      return None

  if not _is_wait_tensor_node(curr):
    return None

  wait_node = curr
  if (
      not wait_node.args
      or not isinstance(wait_node.args[0], torch.fx.Node)
      or not _is_c10d_comm_node(wait_node.args[0])
  ):
    return None

  comm_node = wait_node.args[0]
  if not comm_node.args or not isinstance(comm_node.args[0], torch.fx.Node):
    return None

  raw_input = comm_node.args[0]
  view_nodes.reverse()  # Order from wait_node output to final opnd
  return (comm_node, wait_node, raw_input, view_nodes)


def _is_subgraph_hop_node(node: torch.fx.Node) -> bool:
  """Checks if an FX node invokes a subgraph or higher order operator."""
  if node.op != "call_function":
    return False
  target = node.target
  if hasattr(torch.ops, "higher_order"):
    for hop_name in ("invoke_subgraph", "scan", "cond", "map"):
      if hasattr(torch.ops.higher_order, hop_name):
        hop = getattr(torch.ops.higher_order, hop_name)
        if target is hop or getattr(hop, "default", None) is target:
          return True
  target_name = getattr(target, "_name", "") or getattr(target, "__name__", "")
  return any(
      k in target_name for k in ("invoke_subgraph", "scan", "cond", "map")
  )


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


def sink_comm_into_subgraphs(graph_module: torch.fx.GraphModule) -> None:
  """Sinks collective communication chains into higher-order op subgraphs."""
  transformed_submods_cache = {}
  nodes_to_erase = []
  all_comm_rewrites = []

  for node in list(graph_module.graph.nodes):
    if not _is_subgraph_hop_node(node):
      continue

    submod_target = node.args[0]
    submod = _get_submodule(graph_module, submod_target)
    if not isinstance(submod, torch.fx.GraphModule):
      continue

    identifier = node.args[1] if len(node.args) > 1 else None
    operands = list(node.args[2:])

    comm_rewrites = {}
    for idx, opnd in enumerate(operands):
      chain_info = _extract_comm_chain(opnd)
      if chain_info is not None:
        comm_node, wait_node, raw_input, view_nodes = chain_info
        comm_rewrites[idx] = (
            wait_node,
            comm_node,
            raw_input,
            comm_node.args[1:],
            comm_node.kwargs,
            view_nodes,
        )
        all_comm_rewrites.append((wait_node, comm_node, raw_input, view_nodes))

    if not comm_rewrites:
      continue

    cache_key = (
        getattr(submod, "code", id(submod)),
        tuple(sorted(comm_rewrites.keys())),
    )
    if cache_key in transformed_submods_cache:
      new_submod, qualname = transformed_submods_cache[cache_key]
    else:
      new_submod = copy.deepcopy(submod)
      placeholders = [
          n for n in new_submod.graph.nodes if n.op == "placeholder"
      ]
      for opnd_idx, (
          wait_node,
          comm_node,
          raw_input,
          comm_args_tail,
          comm_kwargs,
          view_nodes,
      ) in comm_rewrites.items():
        if opnd_idx >= len(placeholders):
          continue
        ph = placeholders[opnd_idx]
        with new_submod.graph.inserting_after(ph):
          new_comm = new_submod.graph.call_function(
              comm_node.target,
              (ph, *comm_args_tail),
              comm_kwargs,
          )
        with new_submod.graph.inserting_after(new_comm):
          new_wait = new_submod.graph.call_function(
              wait_node.target,
              (new_comm,),
          )
        prev_node = new_wait
        for v_node in view_nodes:
          with new_submod.graph.inserting_after(prev_node):
            v_args = (prev_node, *v_node.args[1:])
            prev_node = new_submod.graph.call_function(
                v_node.target,
                v_args,
                v_node.kwargs,
            )
        ph.replace_all_uses_with(
            prev_node,
            delete_user_cb=lambda u, nc=new_comm: u != nc,
        )

      new_submod.graph.lint()
      new_submod.recompile()

      qualname = f"subgraph_ingested_comm_{len(transformed_submods_cache)}"
      graph_module.register_module(qualname, new_submod)
      transformed_submods_cache[cache_key] = (new_submod, qualname)

    with graph_module.graph.inserting_before(node):
      submod_get_attr = graph_module.graph.get_attr(qualname)

    new_operands = []
    for idx, opnd in enumerate(operands):
      if idx in comm_rewrites:
        _, _, raw_input, _, _, _ = comm_rewrites[idx]
        new_operands.append(raw_input)
      else:
        new_operands.append(opnd)

    node.args = (submod_get_attr, identifier, *new_operands)

    for (
        wait_node,
        comm_node,
        raw_input,
        _,
        _,
        view_nodes,
    ) in comm_rewrites.values():
      for v in reversed(view_nodes):
        nodes_to_erase.append(v)
      nodes_to_erase.append(wait_node)
      nodes_to_erase.append(comm_node)

  # Replace any lingering uses of nodes to erase in the graph output tuple with raw_input
  output_nodes = [n for n in graph_module.graph.nodes if n.op == "output"]
  if output_nodes:
    output_node = output_nodes[0]
    if output_node.args and isinstance(output_node.args[0], (list, tuple)):
      out_list = list(output_node.args[0])
      replaced = False
      for idx, out_item in enumerate(out_list):
        if not isinstance(out_item, torch.fx.Node):
          continue
        for wait_node, comm_node, raw_input, view_nodes in all_comm_rewrites:
          if (
              out_item in view_nodes
              or out_item is wait_node
              or out_item is comm_node
          ):
            out_list[idx] = raw_input
            replaced = True
            break
      if replaced:
        output_node.args = (
            tuple(out_list)
            if isinstance(output_node.args[0], tuple)
            else out_list,
        )

  for n in nodes_to_erase:
    if len(n.users) == 0:
      graph_module.graph.erase_node(n)

  if transformed_submods_cache:
    graph_module.graph.lint()
    graph_module.recompile()


def _schedule_for_comm(
    graph: torch.fx.Graph,
    raise_comms_opt: bool = True,
    sink_waits_opt: bool = True,
    reorder_for_overlap_opt: bool = True,
) -> None:
  """Schedules graph nodes using priority topological sort to optimize comm/compute overlap."""
  nodes = list(graph.nodes)
  if not nodes:
    return

  placeholders = [n for n in nodes if n.op == "placeholder"]
  output_nodes = [n for n in nodes if n.op == "output"]
  schedulable_nodes = [
      n for n in nodes if n.op not in ("placeholder", "output")
  ]

  if not schedulable_nodes:
    return

  orig_indices = {n: i for i, n in enumerate(nodes)}
  sched_set = set(schedulable_nodes)

  node_deps: dict[torch.fx.Node, set[torch.fx.Node]] = {
      n: {inp for inp in n.all_input_nodes if inp in sched_set}
      for n in schedulable_nodes
  }
  node_users: dict[torch.fx.Node, list[torch.fx.Node]] = {
      n: [] for n in schedulable_nodes
  }
  for n in schedulable_nodes:
    for dep in node_deps[n]:
      node_users[dep].append(n)

  scores_0: dict[torch.fx.Node, int] = {
      n: sys.maxsize for n in schedulable_nodes
  }
  scores_1: dict[torch.fx.Node, int] = {n: 0 for n in schedulable_nodes}
  scores_2: dict[torch.fx.Node, int] = {
      n: orig_indices[n] for n in schedulable_nodes
  }

  if raise_comms_opt:
    comm_idx = 0
    for n in schedulable_nodes:
      if _is_c10d_comm_node(n):
        scores_0[n] = min(scores_0[n], comm_idx)
        visited = set()
        queue = list(node_deps[n])
        while queue:
          anc = queue.pop(0)
          if anc not in visited and anc in scores_0:
            visited.add(anc)
            scores_0[anc] = min(scores_0[anc], comm_idx)
            queue.extend(node_deps[anc])
        comm_idx += 1

  if sink_waits_opt:
    for n in schedulable_nodes:
      if _is_wait_tensor_node(n):
        scores_1[n] = 1
        curr = n
        while True:
          users = [u for u in curr.users if u in sched_set]
          if len(users) == 1:
            u = users[0]
            if u.op == "call_function":
              t_name = str(u.target)
              if any(
                  k in t_name
                  for k in (
                      "view",
                      "reshape",
                      "as_strided",
                      "expand",
                      "t",
                      "transpose",
                  )
              ):
                scores_1[u] = 1
                curr = u
                continue
          break

  unmet_deps: dict[torch.fx.Node, int] = {
      n: len(node_deps[n]) for n in schedulable_nodes
  }

  ready_heap: list[tuple[int, int, int, int, torch.fx.Node]] = []
  for n in schedulable_nodes:
    if unmet_deps[n] == 0:
      heapq.heappush(
          ready_heap,
          (scores_0[n], scores_1[n], scores_2[n], id(n), n),
      )

  scheduled_nodes: list[torch.fx.Node] = []
  while ready_heap:
    _, _, _, _, curr_node = heapq.heappop(ready_heap)
    scheduled_nodes.append(curr_node)

    for user in node_users[curr_node]:
      unmet_deps[user] -= 1
      if unmet_deps[user] == 0:
        heapq.heappush(
            ready_heap,
            (scores_0[user], scores_1[user], scores_2[user], id(user), user),
        )

  if len(scheduled_nodes) != len(schedulable_nodes):
    return

  prev = graph._root
  for ph in placeholders:
    prev.append(ph)
    prev = ph

  for n in scheduled_nodes:
    prev.append(n)
    prev = n

  for out in output_nodes:
    prev.append(out)
    prev = out


def raise_comms(graph: torch.fx.Graph) -> None:
  """Greedily schedules collective communication start operations as early as possible."""
  _schedule_for_comm(
      graph,
      raise_comms_opt=True,
      sink_waits_opt=False,
      reorder_for_overlap_opt=False,
  )


def sink_waits(graph: torch.fx.Graph) -> None:
  """Greedily schedules wait_tensor and intermediate view nodes as late as possible."""
  _schedule_for_comm(
      graph,
      raise_comms_opt=False,
      sink_waits_opt=True,
      reorder_for_overlap_opt=False,
  )


def reorder_communication_preserving_peak_memory(graph: torch.fx.Graph) -> None:
  """Reorders communication ops relative to computation ops to maximize overlap."""
  _schedule_for_comm(
      graph,
      raise_comms_opt=True,
      sink_waits_opt=True,
      reorder_for_overlap_opt=True,
  )


def apply(gm_or_graph: torch.fx.GraphModule | torch.fx.Graph) -> None:
  """Applies generic collective communication pushdown and scheduling to the FX graph."""
  if isinstance(gm_or_graph, torch.fx.Graph):
    if gm_or_graph.owning_module is not None and isinstance(
        gm_or_graph.owning_module, torch.fx.GraphModule
    ):
      graph_module = gm_or_graph.owning_module
    else:
      reorder_communication_preserving_peak_memory(gm_or_graph)
      return
  else:
    graph_module = gm_or_graph

  # 1. Sink collective operations into subgraphs / HOPs
  sink_comm_into_subgraphs(graph_module)

  # 2. Schedule subgraphs recursively if any
  for submod in graph_module.children():
    if isinstance(submod, torch.fx.GraphModule):
      reorder_communication_preserving_peak_memory(submod.graph)
      submod.graph.lint()
      submod.recompile()

  # 3. Schedule root graph
  reorder_communication_preserving_peak_memory(graph_module.graph)

  graph_module.graph.lint()
  graph_module.recompile()
