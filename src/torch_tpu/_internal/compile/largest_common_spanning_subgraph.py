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

"""Finds the largest common spanning subgraph (LCS) of torch.fx graphs.

Anchors are nodes that the caller knows to correspond across the graphs, such
as the same collective on every SPMD rank. Starting from the anchors, the
finder aligns matching inputs and users, one node per graph, until no more
match on every graph. The nodes aligned this way are the common nodes, and
together they form the common region. The region is cut into segments that
`split_module` would partition the same way on every graph; see
`largest_common_spanning_subgraph` for what matches and what a segment is.
"""

import collections
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import fx

_NODE_SENTINEL = ("__fx_node__",)


def _is_spannable(node: fx.Node) -> bool:
  """Returns whether `node` can be part of a segment.

  Placeholders, `get_attr` nodes and output nodes cannot, since `split_module`
  does not put them in partitions. They are never aligned, and a segment stays
  contiguous across them.
  """
  return node.op not in ("placeholder", "get_attr", "output")


def _canonicalize_arg(value: Any) -> Any:
  """Returns a hashable form of `value` for `_node_signature`.

  Constants keep their type, so `1`, `1.0` and `True` differ. Nodes become
  `_NODE_SENTINEL`. A device keeps only its type, since the device index can
  differ per rank, e.g. `cuda:{local_rank}`.
  """
  if isinstance(value, fx.Node):
    return _NODE_SENTINEL
  if isinstance(value, tuple):
    return ("tuple", tuple(_canonicalize_arg(v) for v in value))
  if isinstance(value, list):
    return ("list", tuple(_canonicalize_arg(v) for v in value))
  if isinstance(value, dict):
    return ("dict", tuple((k, _canonicalize_arg(v)) for k, v in value.items()))
  if isinstance(value, slice):
    return (
        "slice",
        _canonicalize_arg(value.start),
        _canonicalize_arg(value.stop),
        _canonicalize_arg(value.step),
    )
  if isinstance(value, torch.device):
    return ("device", value.type)
  return (type(value), value)


def _node_signature(node: fx.Node) -> tuple[Any, ...]:
  """Returns the op, target and canonicalized arguments of `node`.

  Nodes with equal signatures look the same in generated code, up to node names
  and device indices. The compilation handshake fingerprints generated code, so
  what it leaves out, such as tensor dtypes and shapes, is not a difference
  here either.
  """
  return (
      node.op,
      node.target,
      _canonicalize_arg(node.args),
      _canonicalize_arg(node.kwargs),
  )


def _input_nodes(node: fx.Node) -> list[fx.Node]:
  """Returns the nodes that `node` reads, once per read, in argument order.

  Unlike `node.all_input_nodes`, this repeats a node that is read twice, so
  nodes with equal signatures have their inputs in corresponding slots.
  """
  inputs: list[fx.Node] = []
  fx.map_arg((node.args, node.kwargs), inputs.append)
  return inputs


def _validate_inputs(
    graphs: Sequence[fx.Graph],
    anchors: Sequence[Sequence[fx.Node]],
) -> None:
  if not graphs:
    raise ValueError("graphs must be a non-empty sequence of fx.Graph.")

  num_graphs = len(graphs)
  for anchor_idx, anchor in enumerate(anchors):
    if len(anchor) != num_graphs:
      raise ValueError(
          f"Anchor at index {anchor_idx} has length {len(anchor)}, expected"
          f" {num_graphs}."
      )
    for graph_idx, (node, graph) in enumerate(zip(anchor, graphs)):
      if node.graph is not graph:
        raise ValueError(
            f"Anchor {anchor_idx} element {graph_idx} ({node.name!r}) does not"
            f" belong to graphs[{graph_idx}]."
        )


def _tuples_have_matching_signatures(nodes: Sequence[fx.Node]) -> bool:
  if not all(_is_spannable(n) for n in nodes):
    return False
  first_sig = _node_signature(nodes[0])
  return all(_node_signature(n) == first_sig for n in nodes[1:])


def _inputs_aligned_alike(
    nodes: Sequence[fx.Node],
    aligned_in_graph: Sequence[dict[fx.Node, fx.Node]],
) -> bool:
  """Returns whether the inputs of `nodes` are aligned alike on all graphs.

  Alike means that in every input slot, the inputs are all aligned to the same
  `graphs[0]` node or all unaligned. `nodes` must have matching signatures.
  """
  inputs = [_input_nodes(n) for n in nodes]
  return all(
      len({aligned_in_graph[g].get(inp) for g, inp in enumerate(slot)}) == 1
      for slot in zip(*inputs, strict=True)
  )


def _matching_predecessors(
    current: tuple[fx.Node, ...],
) -> Iterable[tuple[fx.Node, ...]]:
  for candidate in zip(*(_input_nodes(n) for n in current), strict=True):
    if _tuples_have_matching_signatures(candidate):
      yield candidate


def _users_by_edge(node: fx.Node) -> dict[Any, list[fx.Node]]:
  """Groups the spannable users of `node` by edge.

  The edge of a user is its signature together with the slots in which it reads
  `node`.
  """
  by_edge: dict[Any, list[fx.Node]] = collections.defaultdict(list)
  for user in node.users:
    if _is_spannable(user):
      slots = tuple(
          i for i, inp in enumerate(_input_nodes(user)) if inp is node
      )
      by_edge[(_node_signature(user), slots)].append(user)
  return by_edge


def _matching_users(
    current: tuple[fx.Node, ...],
) -> Iterable[tuple[fx.Node, ...]]:
  """Yields the users of `current` that correspond across graphs.

  Users correspond when exactly one user per graph has the same edge. Several
  users of one node with the same edge, twins, cannot be told apart, so none
  of them is yielded.
  """
  by_edge = [_users_by_edge(n) for n in current]
  # Iterating `by_edge[0]` visits users in `graphs[0]` order, which keeps the
  # traversal deterministic.
  for edge, users in by_edge[0].items():
    candidates = [users, *(b.get(edge, []) for b in by_edge[1:])]
    if all(len(c) == 1 for c in candidates):
      yield tuple(c[0] for c in candidates)


def _align(
    graphs: Sequence[fx.Graph],
    anchors: Sequence[Sequence[fx.Node]],
) -> list[dict[fx.Node, fx.Node]]:
  """Grows the common region of `graphs` from `anchors` by a cross-graph BFS.

  All anchors are aligned before any growth, so that a match found while
  growing from one anchor cannot take the place of another anchor. Growth is
  greedy: when several nodes of one graph match the same node of another, the
  first one reached is aligned, so the region can depend on the order of the
  anchors.

  Args:
    graphs: See `largest_common_spanning_subgraph`.
    anchors: See `largest_common_spanning_subgraph`.

  Returns:
    One `node -> graphs[0] node` mapping per graph, for the aligned nodes.
  """
  _validate_inputs(graphs, anchors)
  aligned_in_graph: list[dict[fx.Node, fx.Node]] = [{} for _ in graphs]

  def is_aligned(nodes: tuple[fx.Node, ...]) -> bool:
    return any(node in aligned_in_graph[g] for g, node in enumerate(nodes))

  def align(nodes: tuple[fx.Node, ...]) -> None:
    for g, node in enumerate(nodes):
      aligned_in_graph[g][node] = nodes[0]

  # Anchors are known to correspond, so unlike grown nodes they skip the input
  # check below. A node already aligned by an earlier anchor is not aligned
  # again.
  queue: collections.deque[tuple[fx.Node, ...]] = collections.deque()
  for raw_anchor in anchors:
    anchor = tuple(raw_anchor)
    if _tuples_have_matching_signatures(anchor) and not is_aligned(anchor):
      align(anchor)
      queue.append(anchor)

  while queue:
    current = queue.popleft()
    for candidate in (
        *_matching_predecessors(current),
        *_matching_users(current),
    ):
      # This check keeps the region from growing through a mismatch, but it is
      # not final: an input aligned later can still break it.
      # `_unalign_inconsistent_nodes` repeats it once the region is complete.
      if not is_aligned(candidate) and _inputs_aligned_alike(
          candidate, aligned_in_graph
      ):
        align(candidate)
        queue.append(candidate)
  return aligned_in_graph


def _unalign_inconsistent_nodes(
    graphs: Sequence[fx.Graph],
    aligned_in_graph: list[dict[fx.Node, fx.Node]],
    nodes_of: dict[fx.Node, list[fx.Node]],
) -> None:
  """Unaligns nodes whose inputs or outside uses differ across graphs."""

  def unalign(ref: fx.Node) -> None:
    for g, node in enumerate(nodes_of.pop(ref)):
      del aligned_in_graph[g][node]

  # One pass suffices: a node is unaligned on every graph at once, so its
  # users' inputs stay aligned alike.
  order = [n for n in graphs[0].nodes if n in nodes_of]
  for ref in order:
    if not _inputs_aligned_alike(nodes_of[ref], aligned_in_graph):
      unalign(ref)

  # A node that only some graphs use outside the aligned nodes would be a
  # partition output on those graphs only. Users come after their inputs, so a
  # reverse pass sees the final alignment of all users.
  for ref in reversed(order):
    if ref not in nodes_of:
      continue
    used_outside = {
        any(user not in aligned_in_graph[g] for user in node.users)
        for g, node in enumerate(nodes_of[ref])
    }
    if len(used_outside) > 1:
      unalign(ref)


def _cut_segments(
    graphs: Sequence[fx.Graph],
    nodes_of: dict[fx.Node, list[fx.Node]],
    order: Sequence[fx.Node],
) -> list[list[fx.Node]]:
  """Cuts `order` into segments that partition alike on every graph.

  A node joins the current segment if, on every graph, it directly follows the
  segment in node order and reads external inputs alike. Otherwise it starts a
  new segment. External inputs are the nodes that a segment reads from outside
  itself. The segment numbers them by first read, so `add(x, x)` reads
  `[0, 0]` and `add(x, y)` reads `[0, 1]`, and the numbering must be the same
  on every graph. A node whose own reads disagree, such as `add(x, x)` on one
  graph against `add(x, y)` on another, joins no segment.

  Args:
    graphs: See `largest_common_spanning_subgraph`.
    nodes_of: See `_rank_consistent_segments`.
    order: The aligned `graphs[0]` nodes, in node order.

  Returns:
    The segments, in order. Nodes that join no segment are left out.
  """
  # Positions among spannable nodes only; see `_is_spannable`.
  positions: list[dict[fx.Node, int]] = []
  for graph in graphs:
    nodes = [n for n in graph.nodes if _is_spannable(n)]
    positions.append({n: i for i, n in enumerate(nodes)})

  segments: list[list[fx.Node]] = [[]]
  # Per graph, the nodes of the current segment and the external inputs it
  # reads, numbered by first read.
  members: list[set[fx.Node]] = [set() for _ in graphs]
  first_read: list[dict[fx.Node, int]] = [{} for _ in graphs]

  def start_segment() -> None:
    if segments[-1]:
      segments.append([])
    for segment_members, segment_reads in zip(members, first_read):
      segment_members.clear()
      segment_reads.clear()

  def try_extend(ref: fx.Node) -> bool:
    nodes = nodes_of[ref]
    if segments[-1]:
      last = nodes_of[segments[-1][-1]]
      if any(
          positions[g][node] != positions[g][last[g]] + 1
          for g, node in enumerate(nodes)
      ):
        return False
    numberings: list[list[int]] = []
    new_reads: list[dict[fx.Node, int]] = []
    for g, node in enumerate(nodes):
      new: dict[fx.Node, int] = {}
      numbering = []
      # Inputs precede users, so an input is either a member of the current
      # segment or external to it.
      for inp in _input_nodes(node):
        if inp in members[g]:
          continue
        if inp in first_read[g]:
          numbering.append(first_read[g][inp])
        else:
          numbering.append(new.setdefault(inp, len(first_read[g]) + len(new)))
      numberings.append(numbering)
      new_reads.append(new)
    if any(n != numberings[0] for n in numberings[1:]):
      return False
    for g, node in enumerate(nodes):
      first_read[g].update(new_reads[g])
      members[g].add(node)
    segments[-1].append(ref)
    return True

  for ref in order:
    if not try_extend(ref):
      start_segment()
      try_extend(ref)  # Fails only if the reads of `ref` itself disagree.
  return [segment for segment in segments if segment]


def _rank_consistent_segments(
    graphs: Sequence[fx.Graph],
    aligned_in_graph: list[dict[fx.Node, fx.Node]],
) -> dict[fx.Node, int]:
  """Cuts the aligned nodes into segments that partition alike on every graph.

  Nodes whose inputs or outside uses differ across graphs are removed from
  `aligned_in_graph` first, since no segment could contain them.

  Returns:
    Mapping from `graphs[0]` nodes to segment IDs (`0..S-1`, in node order).
    Empty if the aligned nodes are not in the same order on all graphs.
  """
  # `nodes_of[ref][g]` is the node of `graphs[g]` aligned to `ref`, a node of
  # `graphs[0]`.
  nodes_of: dict[fx.Node, list[fx.Node]] = {r: [] for r in aligned_in_graph[0]}
  for aligned in aligned_in_graph:
    for node, ref in aligned.items():
      nodes_of[ref].append(node)
  _unalign_inconsistent_nodes(graphs, aligned_in_graph, nodes_of)

  order = [n for n in graphs[0].nodes if n in nodes_of]
  # No segments if a graph orders the aligned nodes differently. Each rank runs
  # its partitions in node order, so segments ordered differently across ranks
  # would run their collectives in a different order.
  for g, graph in enumerate(graphs):
    aligned = aligned_in_graph[g]
    if [aligned[n] for n in graph.nodes if n in aligned] != order:
      return {}

  segments = _cut_segments(graphs, nodes_of, order)
  segment_of: dict[fx.Node, int] = {}
  for segment_id, segment in enumerate(segments):
    for ref in segment:
      segment_of[ref] = segment_id
  return segment_of


def largest_common_spanning_subgraph(
    graphs: Sequence[fx.Graph],
    anchors: Sequence[Sequence[fx.Node]],
) -> list[dict[str, int]]:
  """Returns the segments of the common region of `graphs` around `anchors`.

  Nodes match when they have the same op, target and constant arguments, with
  the type and value of each constant; node names, tensor dtypes and shapes,
  and device indices do not matter. Segments are runs of common nodes,
  contiguous in node order on every graph, that `split_module` partitions
  alike: with the same inputs and outputs on every graph.

  Args:
    graphs: Non-empty sequence of graphs.
    anchors: Corresponding nodes, one per graph in the order of `graphs`, such
      as the same collective on every graph. Anchors whose nodes do not match,
      or that are placeholders, `get_attr` nodes or output nodes, are ignored.
      The region can depend on the order of the anchors when several nodes of
      one graph match the same node of another.

  Returns:
    One `node.name -> segment_id` mapping per graph, all with the same
    segments. Segment IDs are `0..S-1` in the node order of `graphs[0]`. All
    mappings are empty if the graphs order the common nodes differently, since
    the ranks would then run their collectives in different orders.

  Raises:
    ValueError: If `graphs` is empty, an anchor does not have one node per
      graph, or an anchor node does not belong to its graph.
  """
  aligned_in_graph = _align(graphs, anchors)
  segment_of = _rank_consistent_segments(graphs, aligned_in_graph)
  return [
      {
          node.name: segment_of[ref]
          for node, ref in aligned.items()
          if ref in segment_of
      }
      for aligned in aligned_in_graph
  ]
