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

"""Finds common partitions between `torch.fx` graphs."""

import collections
from collections.abc import Iterable, Sequence
import graphlib
from typing import Any, TypeAlias

import torch
from torch import fx
from torch.utils import _pytree as pytree
from torch_tpu._internal import env

# Corresponding nodes, one per graph, where element `i` belongs to `graphs[i]`.
_NodePerGraph: TypeAlias = tuple[fx.Node, ...]
# Maps each partitioned node of one graph to its partition ID.
_PartitionMap: TypeAlias = dict[fx.Node, int]
# Maps each partition ID to its member tuples.
_Partitions: TypeAlias = dict[int, list[_NodePerGraph]]


def is_partitionable(node: fx.Node) -> bool:
  """Returns whether `split_module` puts `node` in a partition."""
  return node.op not in ("placeholder", "get_attr", "output")


def _tensor_specs(node: fx.Node) -> tuple[tuple[str, torch.dtype, str], ...]:
  """Returns (shape, dtype, device type) per tensor in `node.meta["val"]`."""
  return tuple(
      # `str` keeps symbolic sizes hashable without adding shape guards. Device
      # indices may differ between ranks, so compare only the type.
      (str(leaf.shape), leaf.dtype, leaf.device.type)
      for leaf in pytree.tree_leaves(node.meta.get("val"))
      if isinstance(leaf, torch.Tensor)
  )


def _canonicalize_arg(value: Any) -> Any:
  if isinstance(value, fx.Node):
    return (fx.Node, _tensor_specs(value))
  if isinstance(value, tuple):
    return ("tuple", tuple(_canonicalize_arg(item) for item in value))
  if isinstance(value, list):
    return ("list", tuple(_canonicalize_arg(item) for item in value))
  if isinstance(value, dict):
    # Keep key order: `split_module` numbers submodule inputs by first read.
    return (
        "dict",
        tuple((key, _canonicalize_arg(item)) for key, item in value.items()),
    )
  if isinstance(value, slice):
    return (
        "slice",
        _canonicalize_arg(value.start),
        _canonicalize_arg(value.stop),
        _canonicalize_arg(value.step),
    )
  if isinstance(value, torch.device):
    # Device indices may differ between ranks, so compare only the type.
    return ("device", value.type)
  return (type(value), value)


def _node_signature(node: fx.Node) -> tuple[Any, ...]:
  return (
      node.op,
      node.target,
      _canonicalize_arg(node.args),
      _canonicalize_arg(node.kwargs),
      _tensor_specs(node),
  )


def _input_nodes(node: fx.Node) -> list[fx.Node]:
  """Returns the input nodes of `node` in argument order, keeping duplicates."""
  inputs: list[fx.Node] = []
  fx.map_arg((node.args, node.kwargs), inputs.append)
  return inputs


def _debug_checks_enabled() -> bool:
  """Returns whether `TORCH_TPU_ENABLE_DEBUG_CHECKS` is turned on."""
  return env.get_bool_env_once(
      "TORCH_TPU_ENABLE_DEBUG_CHECKS", default_value=False
  )


def _validate_inputs(
    graphs: Sequence[fx.Graph],
    anchors: Sequence[Sequence[fx.Node]],
) -> None:
  num_graphs = len(graphs)
  anchored: list[set[fx.Node]] = [set() for _ in graphs]
  for anchor_index, anchor in enumerate(anchors):
    if len(anchor) != num_graphs:
      raise ValueError(
          f"Anchor at index {anchor_index} has length {len(anchor)}, expected"
          f" {num_graphs}."
      )
    for graph_index, (node, graph) in enumerate(
        zip(anchor, graphs, strict=True)
    ):
      if node.graph is not graph:
        raise ValueError(
            f"Anchor {anchor_index} element {graph_index} ({node.name!r}) does"
            f" not belong to graphs[{graph_index}]."
        )
      if not is_partitionable(node):
        raise ValueError(
            f"Anchor {anchor_index} element {graph_index} ({node.name!r}) has"
            f" op {node.op!r}, which `split_module` does not partition."
        )
      if node in anchored[graph_index]:
        raise ValueError(
            f"Anchor {anchor_index} element {graph_index} ({node.name!r}) is"
            " also in an earlier anchor; anchors must not share nodes."
        )
      anchored[graph_index].add(node)


def _nodes_match(nodes: Sequence[fx.Node]) -> bool:
  """Returns whether `nodes` are partitionable and equivalent across graphs.

  Nodes match when they have the same op, target, argument structure, constant
  types and values, and the shapes, dtypes, and device types of the tensors
  they read and produce. Node names and device indices may differ.
  """
  if not all(is_partitionable(node) for node in nodes):
    return False
  first_signature = _node_signature(nodes[0])
  return all(_node_signature(node) == first_signature for node in nodes[1:])


def _matching_predecessors(
    current: _NodePerGraph,
) -> Iterable[_NodePerGraph]:
  """Yields the inputs of `current` at each argument position, if they match."""
  # Nodes in `current` have equal signatures, so `_input_nodes` has the same
  # length on every graph.
  for candidate in zip(*(_input_nodes(node) for node in current), strict=True):
    if _nodes_match(candidate):
      yield candidate


def _users_by_edge(node: fx.Node) -> dict[Any, list[fx.Node]]:
  """Groups `node`'s partitionable users by signature and where they read it."""
  by_edge: dict[Any, list[fx.Node]] = collections.defaultdict(list)
  for user in node.users:
    if is_partitionable(user):
      slots = tuple(
          slot
          for slot, input_node in enumerate(_input_nodes(user))
          if input_node is node
      )
      by_edge[(_node_signature(user), slots)].append(user)
  return by_edge


def _matching_users(
    current: _NodePerGraph,
) -> Iterable[_NodePerGraph]:
  """Yields users of `current` that are alone in their group on every graph."""
  by_edge = [_users_by_edge(node) for node in current]
  for edge, users in by_edge[0].items():
    users_per_graph = [
        users,
        *(users_by_edge.get(edge, []) for users_by_edge in by_edge[1:]),
    ]
    if all(len(graph_users) == 1 for graph_users in users_per_graph):
      yield tuple(graph_users[0] for graph_users in users_per_graph)


def _outputs_match(
    members: Sequence[_NodePerGraph],
    node_positions_in_graph: Sequence[dict[fx.Node, int]],
    index_per_graph: Sequence[dict[fx.Node, int]],
) -> bool:
  """Returns whether the submodules for `members` have the same outputs.

  `split_module` makes each submodule return every member that a node outside
  the partition reads, in the order of each member's first outside read.
  Nothing makes those outside nodes match between graphs, so one graph might
  read an extra member, or read two members in the opposite order.
  """
  outputs_per_graph: list[list[int]] = []
  for graph_index, index_of in enumerate(index_per_graph):
    node_positions = node_positions_in_graph[graph_index]
    first_outside_reads: list[tuple[tuple[int, int], int]] = []
    for member_index, member in enumerate(members):
      node = member[graph_index]
      reads = [
          (node_positions[user], _input_nodes(user).index(node))
          for user in node.users
          if user not in index_of
      ]
      if reads:
        first_outside_reads.append((min(reads), member_index))
    outputs_per_graph.append(
        [member_index for _, member_index in sorted(first_outside_reads)]
    )
  return all(outputs == outputs_per_graph[0] for outputs in outputs_per_graph)


def _is_valid_partition(
    members: Sequence[_NodePerGraph],
    node_positions_in_graph: Sequence[dict[fx.Node, int]],
) -> bool:
  """Returns whether `split_module` builds identical submodules for `members`.

  Each tuple in `members` must already pass `_nodes_match`. Growth reaches each
  member through one input or user, so this checks that all graphs agree on:

  *   Node order, since `split_module` copies nodes in graph order. If `sin(x)`
      and `cos(y)` both feed `add(sin, cos)` but run in opposite orders, the
      submodules take `(x, y)` on one graph and `(y, x)` on the other.
  *   Wiring between members. `addcmul(sin, cos, add)` and
      `addcmul(cos, sin, add)` both read `add` the same way, but compute
      `sin + cos * add` and `cos + sin * add`.
  *   Inputs from outside, numbered by first read. `add(x, x)` reads one
      submodule input, while `add(x, y)` reads two.
  *   Outputs, which `_outputs_match` checks.
  """
  # TODO(b/553332536): Keep `members` sorted as they are built instead of
  # sorting on every call.
  members = sorted(
      members, key=lambda member: node_positions_in_graph[0][member[0]]
  )
  index_per_graph: list[dict[fx.Node, int]] = []
  reads_per_graph: list[list[tuple[bool, int]]] = []
  for graph_index, node_positions in enumerate(node_positions_in_graph):
    nodes = [member[graph_index] for member in members]
    order = [node_positions[node] for node in nodes]
    if order != sorted(order):
      return False
    index_of = {node: member_index for member_index, node in enumerate(nodes)}
    first_read: dict[fx.Node, int] = {}
    reads: list[tuple[bool, int]] = []
    for node in nodes:
      for input_node in _input_nodes(node):
        if input_node in index_of:
          reads.append((True, index_of[input_node]))
        else:
          reads.append(
              (False, first_read.setdefault(input_node, len(first_read)))
          )
    index_per_graph.append(index_of)
    reads_per_graph.append(reads)
  if any(reads != reads_per_graph[0] for reads in reads_per_graph):
    return False
  return _outputs_match(members, node_positions_in_graph, index_per_graph)


def _partition_order(
    graphs: Sequence[fx.Graph],
    partition_maps: Sequence[_PartitionMap],
) -> list[int] | None:
  """Returns a partition run order valid on every graph, if one exists.

  Each partition runs as one unit, in the same order on every graph, so no
  valid order exists when:

  *   A node outside a partition both reads from and feeds it. For
      `add(sin, mul(sin, 2))` with `mul` outside `{sin, add}`, the partition
      would have to run both before and after `mul`.
  *   Graphs need opposite orders. Partition 0 might feed partition 1 on one
      graph, while partition 1 feeds partition 0 on another.

  Merging each partition into one vertex shared by all graphs, while keeping
  unpartitioned nodes separate per graph, turns both cases into cycles caught
  by one topological sort, in which case this returns `None`.
  """
  sorter = graphlib.TopologicalSorter()
  for graph_index, (graph, partition_map) in enumerate(
      zip(graphs, partition_maps, strict=True)
  ):
    for node in graph.nodes:
      # The graph index keeps unpartitioned nodes separate per graph, even if
      # `graphs` holds the same graph object twice.
      vertex = partition_map.get(node, (graph_index, node))
      inputs = (
          partition_map.get(input_node, (graph_index, input_node))
          for input_node in node.all_input_nodes
      )
      sorter.add(
          vertex,
          *(input_vertex for input_vertex in inputs if input_vertex != vertex),
      )
  try:
    sorter.prepare()
  except graphlib.CycleError:
    return None
  order: list[int] = []
  while sorter.is_active():
    ready = sorter.get_ready()
    order.extend(sorted(vertex for vertex in ready if isinstance(vertex, int)))
    sorter.done(*ready)
  return order


def _assign_if_valid(
    graphs: Sequence[fx.Graph],
    node_positions_in_graph: Sequence[dict[fx.Node, int]],
    partition_maps: Sequence[_PartitionMap],
    partition_id: int,
    members: list[_NodePerGraph],
) -> list[_PartitionMap] | None:
  """Returns `partition_maps` with `members` in `partition_id`, if valid.

  Valid means `split_module` builds the same submodule for `members` on every
  graph, and one run order of all partitions still fits every graph. Returns
  `None` otherwise. Assumes `members[:-1]` are already in `partition_id` in
  `partition_maps`. Does not modify `partition_maps`.
  """
  # TODO(b/553332536): Re-checking the whole partition and re-sorting every
  # graph for each candidate makes growth quadratic. Only the new member's
  # edges can change the submodule or the run order, so check just those.
  if not _is_valid_partition(members, node_positions_in_graph):
    return None
  new_partition_maps = [dict(partition_map) for partition_map in partition_maps]
  for partition_map, node in zip(new_partition_maps, members[-1], strict=True):
    partition_map[node] = partition_id
  if _partition_order(graphs, new_partition_maps) is None:
    return None
  return new_partition_maps


def _grow_partitions(
    graphs: Sequence[fx.Graph],
    anchors: Sequence[Sequence[fx.Node]],
) -> tuple[list[_PartitionMap], _Partitions]:
  """Greedily grows partitions from `anchors` via BFS, one anchor at a time.

  Visits anchors in order. Each anchor that is not yet partitioned and forms a
  valid partition on its own starts a new partition, which grows until no more
  neighbors can join. Growth takes in other anchors like regular nodes, but
  only as a whole, so it never pairs an anchor's nodes with other nodes.

  Returns `(partition_maps, partitions)`, mapping each partitioned node per
  graph to its partition ID and each partition ID to its member tuples.
  """
  node_positions_in_graph = [
      {node: position for position, node in enumerate(graph.nodes)}
      for graph in graphs
  ]
  # `anchor_maps[i]` maps each anchor node of `graphs[i]` to its index in
  # `anchors`.
  anchor_maps: list[dict[fx.Node, int]] = [{} for _ in graphs]
  for anchor_index, anchor in enumerate(anchors):
    for anchor_map, node in zip(anchor_maps, anchor, strict=True):
      anchor_map[node] = anchor_index
    if not _nodes_match(anchor):
      raise ValueError(
          f"Anchor {anchor_index} nodes {[node.name for node in anchor]} do"
          " not match across graphs."
      )
  # `partition_maps[i]` maps each partitioned node of `graphs[i]` to its
  # partition ID.
  partition_maps: list[_PartitionMap] = [{} for _ in graphs]
  partitions: _Partitions = {}

  for partition_id, raw_anchor in enumerate(anchors):
    anchor = tuple(raw_anchor)
    # Skip an anchor that growth from an earlier one already took in.
    if any(
        node in partition_map
        for partition_map, node in zip(partition_maps, anchor, strict=True)
    ):
      continue
    new_partition_maps = _assign_if_valid(
        graphs,
        node_positions_in_graph,
        partition_maps,
        partition_id,
        [anchor],
    )
    if new_partition_maps is None:
      continue
    partition_maps = new_partition_maps
    partitions[partition_id] = [anchor]
    queue: collections.deque[_NodePerGraph] = collections.deque([anchor])
    while queue:
      current = queue.popleft()
      for candidate in (
          *_matching_predecessors(current),
          *_matching_users(current),
      ):
        if any(
            node in partition_map
            for partition_map, node in zip(
                partition_maps, candidate, strict=True
            )
        ):
          continue
        # Anchor nodes may only join as their whole anchor. Non-anchor nodes
        # map to `None`, so any other mix of nodes gives more than one value.
        candidate_anchor_indices = {
            anchor_map.get(node)
            for anchor_map, node in zip(anchor_maps, candidate, strict=True)
        }
        if len(candidate_anchor_indices) > 1:
          continue
        members = [*partitions[partition_id], candidate]
        new_partition_maps = _assign_if_valid(
            graphs,
            node_positions_in_graph,
            partition_maps,
            partition_id,
            members,
        )
        if new_partition_maps is not None:
          partition_maps = new_partition_maps
          partitions[partition_id] = members
          queue.append(candidate)
  return partition_maps, partitions


def largest_common_spanning_subgraph(
    graphs: Sequence[fx.Graph],
    anchors: Sequence[Sequence[fx.Node]],
) -> list[dict[str, int]]:
  """Finds common partitions between graphs, centered at `anchors`.

  Greedily grows connected subsets of matching nodes around `anchors` such that
  `torch.fx.passes.split_module` extracts an identical submodule for each
  partition on every graph.

  Args:
    graphs: Non-empty sequence of FX graphs.
    anchors: Disjoint tuples of matching partitionable nodes, where
      `anchors[k][i]` belongs to `graphs[i]`. Partitions grow from anchors in
      order, and growth from an earlier anchor can take in later ones. An
      anchor's nodes are only ever partitioned together, and an anchor that
      cannot join a valid shared partition is left out.

  Returns:
    One dict per graph mapping that graph's node names to partition IDs.
    Partition IDs count up from 0 in an execution order valid on every graph.

  Raises:
    ValueError: If `graphs` is empty or an anchor's nodes do not match.
  """
  if not graphs:
    raise ValueError("graphs must be a non-empty sequence of fx.Graph.")
  if _debug_checks_enabled():
    _validate_inputs(graphs, anchors)
  partition_maps, partitions = _grow_partitions(graphs, anchors)
  # `_grow_partitions` keeps a valid order at every step, so this is not `None`.
  order = _partition_order(graphs, partition_maps) or []
  result: list[dict[str, int]] = [{} for _ in graphs]
  for run_index, partition_id in enumerate(order):
    for member in partitions[partition_id]:
      for graph_result, node in zip(result, member, strict=True):
        graph_result[node.name] = run_index
  return result
