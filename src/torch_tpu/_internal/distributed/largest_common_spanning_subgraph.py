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

import bisect
import collections
from collections.abc import Iterable, MutableSequence, MutableSet, Sequence, Set
import dataclasses
import graphlib
import itertools
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
# A partition ID representing a partitioned vertex in the run-order graph.
_PartitionedVertex: TypeAlias = int
# `(graph_index, node)` for an unpartitioned vertex in the run-order graph.
_UnpartitionedVertex: TypeAlias = tuple[int, fx.Node]
_Vertex: TypeAlias = _PartitionedVertex | _UnpartitionedVertex
# Maps each node of one graph to `(read_order, user)` for every user that reads
# it, in node order.
_ReadOrders: TypeAlias = dict[fx.Node, list[tuple[int, fx.Node]]]
# The read order of one member's first outside read, per graph.
_OutputOrders: TypeAlias = tuple[int, ...]
# Maps `(graph_index, node)` to the `graphs[0]` node that `node` is paired with.
_PairedNodes: TypeAlias = dict[tuple[int, fx.Node], fx.Node]


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


@dataclasses.dataclass
class _SubmoduleState:
  """Incremental submodule validity state for one partition.

  Attributes:
    partition_id: ID of the partition.
    partition_maps: Maps each partitioned node, per graph, to its partition ID.
    node_positions_in_graph: Maps each node, per graph, to its position.
    read_orders_per_graph: The `_build_read_orders` result of each graph.
    members: Member tuples in node order, which is the same on every graph.
    paired_nodes: Pairs every member, and every node that a member reads from
      outside the partition, with a node of `graphs[0]`.
    output_orders: The `_OutputOrders` of every member that the submodule
      returns, sorted. Every graph returns those members in this order.
    first_read_index: Maps `(graph_index, node)` to how many of the first reads
      of `node` are known to be by members.
  """

  partition_id: int
  partition_maps: Sequence[_PartitionMap]
  node_positions_in_graph: Sequence[dict[fx.Node, int]]
  read_orders_per_graph: Sequence[_ReadOrders]
  members: list[_NodePerGraph] = dataclasses.field(default_factory=list)
  paired_nodes: _PairedNodes = dataclasses.field(default_factory=dict)
  output_orders: list[_OutputOrders] = dataclasses.field(default_factory=list)
  first_read_index: dict[tuple[int, fx.Node], int] = dataclasses.field(
      default_factory=dict
  )


def _build_read_orders(graph: fx.Graph) -> _ReadOrders:
  """Numbers every read in `graph` in the order `split_module` visits them.

  `split_module` makes each submodule return every member that a node outside
  the partition reads, in the order of each member's first outside read. It
  visits users in node order and the inputs of each user in argument order, so
  numbering reads the same way turns comparing first outside reads into
  comparing integers.
  """
  read_orders: _ReadOrders = collections.defaultdict(list)
  next_read_order = itertools.count()
  for user in graph.nodes:
    for input_node in dict.fromkeys(_input_nodes(user)):
      read_orders[input_node].append((next(next_read_order), user))
  return read_orders


def _insertion_index(
    state: _SubmoduleState, graph_index: int, node: fx.Node
) -> int:
  """Returns where `node` belongs among the members of `graphs[graph_index]`."""
  node_positions = state.node_positions_in_graph[graph_index]
  return bisect.bisect_left(
      state.members,
      node_positions[node],
      key=lambda member: node_positions[member[graph_index]],
  )


def _pair_nodes(
    state: _SubmoduleState, candidate: _NodePerGraph
) -> tuple[set[_NodePerGraph], _PairedNodes] | None:
  """Pairs `candidate` and each of its inputs across graphs.

  The nodes of `candidate` are paired with each other, and so are the nodes
  that they read in the same argument position. As long as no node is paired
  with two different nodes, every graph wires the members to each other, and
  to the inputs of the submodule, in the same way.

  Returns:
    The members that `candidate` reads and the pairs to add to
    `state.paired_nodes`, or `None` if a node is already paired differently.
  """
  new_paired_nodes: _PairedNodes = {}
  read_members: set[_NodePerGraph] = set()
  for nodes in (
      candidate,
      *zip(*(_input_nodes(node) for node in candidate), strict=True),
  ):
    keys = list(enumerate(nodes))
    first_graph_nodes = [
        new_paired_nodes.get(key, state.paired_nodes.get(key)) for key in keys
    ]
    if all(node is None for node in first_graph_nodes):
      new_paired_nodes.update(dict.fromkeys(keys, nodes[0]))
    elif any(node is not nodes[0] for node in first_graph_nodes):
      return None
    elif state.partition_maps[0].get(nodes[0]) == state.partition_id:
      read_members.add(nodes)
  return read_members, new_paired_nodes


def _outside_read_orders(
    state: _SubmoduleState,
    member: _NodePerGraph,
    joining: _NodePerGraph | None = None,
) -> _OutputOrders:
  """Returns the read order of `member`'s first outside read on each graph.

  Reads by `joining`, which is about to join the partition, do not count.
  Graphs on which no node outside the partition reads `member` are left out.
  """
  orders: list[int] = []
  for graph_index, node in enumerate(member):
    partition_map = state.partition_maps[graph_index]
    joining_node = None if joining is None else joining[graph_index]
    reads = state.read_orders_per_graph[graph_index].get(node, [])
    # Members never leave the partition, so their reads are skipped for good,
    # even if `joining` ends up rejected.
    index = state.first_read_index.get((graph_index, node), 0)
    while (
        index < len(reads)
        and partition_map.get(reads[index][1]) == state.partition_id
    ):
      index += 1
    state.first_read_index[(graph_index, node)] = index
    for read_index in range(index, len(reads)):
      order, user = reads[read_index]
      if (
          user is not joining_node
          and partition_map.get(user) != state.partition_id
      ):
        orders.append(order)
        break
  return tuple(orders)


def _replace_sorted(
    values: MutableSequence[_OutputOrders],
    old: Iterable[_OutputOrders],
    new: Iterable[_OutputOrders],
) -> None:
  """Removes `old` from the sorted `values` and inserts `new`."""
  for value in old:
    values.pop(bisect.bisect_left(values, value))
  for value in new:
    bisect.insort(values, value)


def _update_outputs(
    state: _SubmoduleState,
    candidate: _NodePerGraph,
    read_members: Set[_NodePerGraph],
) -> bool:
  """Updates submodule outputs if every graph still returns the same members.

  `split_module` makes each submodule return every member that a node outside
  the partition reads, in the order of each member's first outside read.
  Nothing makes those outside nodes match between graphs, so one graph might
  read an extra member, or read two members in the opposite order.

  Adding `candidate` changes this only for `candidate` and for `read_members`,
  the members it reads.
  """
  # Until it joins, `candidate` reads `read_members` from outside the
  # partition, so the submodule returns all of them.
  old_orders = [_outside_read_orders(state, member) for member in read_members]
  orders_after_joining = [
      _outside_read_orders(state, member, candidate)
      for member in (candidate, *read_members)
  ]
  # Some graphs would return a member that the others do not.
  if any(0 < len(orders) < len(candidate) for orders in orders_after_joining):
    return False
  new_orders = [orders for orders in orders_after_joining if orders]
  output_orders = state.output_orders
  _replace_sorted(output_orders, old_orders, new_orders)
  for orders in new_orders:
    index = bisect.bisect_left(output_orders, orders)
    # `output_orders` is in the read order of `graphs[0]`, so every other graph
    # must read `orders` and its neighbors in that order too.
    neighbors = output_orders[max(index - 1, 0) : index + 2]
    if any(
        list(graph_orders) != sorted(graph_orders)
        for graph_orders in zip(*neighbors, strict=True)
    ):
      # Undo, so that the rejected `candidate` leaves the outputs unchanged.
      _replace_sorted(output_orders, new_orders, old_orders)
      return False
  return True


def _try_add_member(state: _SubmoduleState, candidate: _NodePerGraph) -> bool:
  """Adds `candidate` if `split_module` still builds identical submodules.

  `candidate` must already pass `_nodes_match`. Growth reaches each member
  through one input or user, so this incrementally checks that all graphs
  agree on:

  *   Node order, since `split_module` copies nodes in graph order. If `sin(x)`
      and `cos(y)` both feed `add(sin, cos)` but run in opposite orders, the
      submodules take `(x, y)` on one graph and `(y, x)` on the other.
  *   Wiring between members. `addcmul(sin, cos, add)` and
      `addcmul(cos, sin, add)` both read `add` the same way, but compute
      `sin + cos * add` and `cos + sin * add`.
  *   Inputs from outside, numbered by first read. `add(x, x)` reads one
      submodule input, while `add(x, y)` reads two.
  *   Outputs, which `_update_outputs` checks.

  Only the inputs and users of `candidate` can change these, so the other
  members are not checked again.
  """
  # Members are in node order on every graph, so `candidate` keeps them in
  # order only if it belongs at the same index on every graph.
  insert_indices = {
      _insertion_index(state, graph_index, node)
      for graph_index, node in enumerate(candidate)
  }
  if len(insert_indices) > 1:
    return False
  pairing = _pair_nodes(state, candidate)
  if pairing is None:
    return False
  read_members, new_paired_nodes = pairing
  if not _update_outputs(state, candidate, read_members):
    return False
  state.members.insert(insert_indices.pop(), candidate)
  state.paired_nodes.update(new_paired_nodes)
  return True


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


def _vertex(
    partition_maps: Sequence[_PartitionMap], graph_index: int, node: fx.Node
) -> _Vertex:
  return partition_maps[graph_index].get(node, (graph_index, node))


def _extend_reachable(
    partition_maps: Sequence[_PartitionMap],
    partitions: _Partitions,
    partition_id: int,
    seeds: _NodePerGraph,
    reachable: MutableSet[_Vertex],
    *,
    forward: bool,
) -> None:
  """Adds vertices reachable from `seeds` outside `partition_id` to `reachable`.

  Walks forward along `node.users` when `forward` is true, or backward along
  `node.all_input_nodes` otherwise, crossing between graphs whenever traversal
  reaches an earlier partition.
  """
  stack = list(enumerate(seeds))
  while stack:
    graph_index, node = stack.pop()
    neighbors = node.users if forward else node.all_input_nodes
    for neighbor in neighbors:
      vertex = _vertex(partition_maps, graph_index, neighbor)
      if vertex != partition_id and vertex not in reachable:
        reachable.add(vertex)
        if isinstance(vertex, int):
          for member in partitions[vertex]:
            stack.extend(enumerate(member))
        else:
          stack.append(vertex)


def _preserves_run_order(
    partition_maps: Sequence[_PartitionMap],
    descendants: Set[_Vertex],
    ancestors: Set[_Vertex],
    candidate: _NodePerGraph,
) -> bool:
  """Returns whether adding `candidate` keeps one shared partition run order.

  `descendants` and `ancestors` are vertices downstream and upstream of the
  active partition. Assumes `candidate` is an input of the partition on every
  graph, or a user of it on every graph.
  """
  for graph_index, node in enumerate(candidate):
    if any(
        _vertex(partition_maps, graph_index, input_node) in descendants
        for input_node in node.all_input_nodes
    ) or any(
        _vertex(partition_maps, graph_index, user) in ancestors
        for user in node.users
    ):
      return False
  return True


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
  read_orders_per_graph = [_build_read_orders(graph) for graph in graphs]
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
    state = _SubmoduleState(
        partition_id=partition_id,
        partition_maps=partition_maps,
        node_positions_in_graph=node_positions_in_graph,
        read_orders_per_graph=read_orders_per_graph,
    )
    if not _try_add_member(state, anchor):
      continue
    # Check reachability before assigning `partition_id` so paths between
    # `anchor`'s nodes are not collapsed into self-loops.
    descendants: set[_Vertex] = set()
    _extend_reachable(
        partition_maps,
        partitions,
        partition_id,
        anchor,
        descendants,
        forward=True,
    )
    if any(
        (graph_index, node) in descendants
        for graph_index, node in enumerate(anchor)
    ):
      continue
    for partition_map, node in zip(partition_maps, anchor, strict=True):
      partition_map[node] = partition_id
    ancestors: set[_Vertex] = set()
    _extend_reachable(
        partition_maps,
        partitions,
        partition_id,
        anchor,
        ancestors,
        forward=False,
    )
    partitions[partition_id] = state.members
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
        if not _preserves_run_order(
            partition_maps, descendants, ancestors, candidate
        ):
          continue
        if not _try_add_member(state, candidate):
          continue
        for partition_map, node in zip(partition_maps, candidate, strict=True):
          partition_map[node] = partition_id
        _extend_reachable(
            partition_maps,
            partitions,
            partition_id,
            candidate,
            descendants,
            forward=True,
        )
        _extend_reachable(
            partition_maps,
            partitions,
            partition_id,
            candidate,
            ancestors,
            forward=False,
        )
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
  order = _partition_order(graphs, partition_maps)
  if order is None:
    raise RuntimeError(
        "Partition growth produced a cyclic partition run order."
    )
  result: list[dict[str, int]] = [{} for _ in graphs]
  for run_index, partition_id in enumerate(order):
    for member in partitions[partition_id]:
      for graph_result, node in zip(result, member, strict=True):
        graph_result[node.name] = run_index
  return result
