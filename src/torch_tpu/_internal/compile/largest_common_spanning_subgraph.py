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

"""Utilities for finding common spanning subgraphs across FX graphs."""

import collections
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import fx

_NODE_SENTINEL = ("__fx_node__",)


def _is_spannable(node: fx.Node) -> bool:
  """Returns whether `node` can join a segment, i.e. is not a graph boundary."""
  return node.op not in ("placeholder", "output")


def _canonicalize_arg(value: Any) -> Any:
  """Returns a hashable form of `value` for `_node_signature`.

  Constants keep their type, so `1`, `1.0` and `True` differ. Nodes become
  `_NODE_SENTINEL`. A device keeps only its type, since ranks' devices can
  differ in index, e.g. `cuda:{local_rank}`.
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
  """Returns the op, target and canonicalized arguments of `node`."""
  return (
      node.op,
      node.target,
      _canonicalize_arg(node.args),
      _canonicalize_arg(node.kwargs),
  )


def _input_nodes(node: fx.Node) -> list[fx.Node]:
  """Returns the nodes that `node` reads, once per read, in argument order.

  Nodes with equal signatures read their inputs in corresponding slots.
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
  """Returns whether `nodes` are all spannable and have equal signatures."""
  if not all(_is_spannable(n) for n in nodes):
    return False
  first_sig = _node_signature(nodes[0])
  return all(_node_signature(n) == first_sig for n in nodes[1:])


def _inputs_aligned_alike(
    nodes: Sequence[fx.Node],
    aligned_in_graph: Sequence[dict[fx.Node, fx.Node]],
) -> bool:
  """Returns whether the inputs of `nodes` are aligned alike on all graphs.

  `nodes` have matching signatures. In each input slot, the inputs must all be
  aligned to the same `graphs[0]` node, or all be unaligned.
  """
  inputs = [_input_nodes(n) for n in nodes]
  return all(
      len({aligned_in_graph[g].get(inp) for g, inp in enumerate(slot)}) == 1
      for slot in zip(*inputs, strict=True)
  )


def _matching_predecessors(
    current: tuple[fx.Node, ...],
) -> Iterable[tuple[fx.Node, ...]]:
  """Yields, slot by slot, the inputs of `current` whose signatures match."""
  for candidate in zip(*(_input_nodes(n) for n in current), strict=True):
    if _tuples_have_matching_signatures(candidate):
      yield candidate


def _users_by_edge(node: fx.Node) -> dict[Any, list[fx.Node]]:
  """Groups the spannable users of `node` by signature and slots reading it."""
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

  An edge yields only if exactly one user has it on every graph; twins are
  ambiguous and skipped.
  """
  by_edge = [_users_by_edge(n) for n in current]
  # `by_edge[0]` keeps the user order of `graphs[0]`, which keeps the traversal
  # deterministic.
  for edge, users in by_edge[0].items():
    candidates = [users, *(b.get(edge, []) for b in by_edge[1:])]
    if all(len(c) == 1 for c in candidates):
      yield tuple(c[0] for c in candidates)


def _align(
    graphs: Sequence[fx.Graph],
    anchors: Sequence[Sequence[fx.Node]],
) -> list[dict[fx.Node, fx.Node]]:
  """Grows the common region of `graphs` from `anchors` by a cross-graph BFS.

  Returns one `node -> graphs[0] node` mapping per graph, for the aligned nodes.
  """
  _validate_inputs(graphs, anchors)
  aligned_in_graph: list[dict[fx.Node, fx.Node]] = [{} for _ in graphs]

  def try_align(nodes: tuple[fx.Node, ...]) -> bool:
    # Each node aligns at most once. Inputs can still align after their user;
    # the caller re-checks them once the region is complete.
    if any(node in aligned_in_graph[g] for g, node in enumerate(nodes)):
      return False
    if not _inputs_aligned_alike(nodes, aligned_in_graph):
      return False
    for g, node in enumerate(nodes):
      aligned_in_graph[g][node] = nodes[0]
    return True

  for raw_anchor in anchors:
    anchor = tuple(raw_anchor)
    if not _tuples_have_matching_signatures(anchor) or not try_align(anchor):
      continue
    queue: collections.deque[tuple[fx.Node, ...]] = collections.deque([anchor])
    while queue:
      current = queue.popleft()
      for candidate in (
          *_matching_predecessors(current),
          *_matching_users(current),
      ):
        if try_align(candidate):
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

  # One forward pass suffices: unaligning a node unaligns it on every graph,
  # so its users' inputs stay alike.
  order = [n for n in graphs[0].nodes if n in nodes_of]
  for ref in order:
    if not _inputs_aligned_alike(nodes_of[ref], aligned_in_graph):
      unalign(ref)

  # A node used outside the aligned nodes on some graphs only would be a
  # partition output there and not elsewhere. Users come after their inputs,
  # so a reverse pass sees the final alignment of all users.
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
  """Cuts `order` into runs that are contiguous in every graph's node order."""
  # Placeholders and the output are not partitioned, so they do not cut.
  positions: list[dict[fx.Node, int]] = []
  for graph in graphs:
    nodes = [n for n in graph.nodes if _is_spannable(n)]
    positions.append({n: i for i, n in enumerate(nodes)})
  segments: list[list[fx.Node]] = []
  previous: list[fx.Node] | None = None
  for ref in order:
    nodes = nodes_of[ref]
    if previous is None or any(
        positions[g][node] != positions[g][previous[g]] + 1
        for g, node in enumerate(nodes)
    ):
      segments.append([])
    segments[-1].append(ref)
    previous = nodes
  return segments


def _input_sharing(
    nodes_of: dict[fx.Node, list[fx.Node]],
    segment: Sequence[fx.Node],
    g: int,
) -> list[int]:
  """Numbers the external inputs of `segment` on graph `g` by first read.

  E.g. `[0, 0]` for `add(x, x)` but `[0, 1]` for `add(x, y)`.
  """
  members = {nodes_of[ref][g] for ref in segment}
  first_read: dict[fx.Node, int] = {}
  sharing = []
  for ref in segment:
    for inp in _input_nodes(nodes_of[ref][g]):
      if inp not in members:
        sharing.append(first_read.setdefault(inp, len(first_read)))
  return sharing


def _rank_consistent_segments(
    graphs: Sequence[fx.Graph],
    aligned_in_graph: list[dict[fx.Node, fx.Node]],
) -> dict[fx.Node, int]:
  """Splits aligned nodes into segments that partition alike on all graphs.

  Unaligns nodes whose inputs or outside uses differ across graphs, in place.

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

  segments = []
  for segment in _cut_segments(graphs, nodes_of, order):
    reference_sharing = _input_sharing(nodes_of, segment, 0)
    if all(
        _input_sharing(nodes_of, segment, g) == reference_sharing
        for g in range(1, len(graphs))
    ):
      segments.append(segment)
  segment_of: dict[fx.Node, int] = {}
  for segment_id, segment in enumerate(segments):
    for ref in segment:
      segment_of[ref] = segment_id
  return segment_of


def largest_common_spanning_subgraph(
    graphs: Sequence[fx.Graph],
    anchors: Sequence[Sequence[fx.Node]],
) -> list[dict[str, int]]:
  """Finds the region that `graphs` share around `anchors`, cut into segments.

  The region grows from each anchor through inputs and users. A neighbor joins
  if exactly one node per graph reads or feeds the region at the same position
  with the same op, target and constants, as generated code spells them out:
  `1` and `1.0` differ, node names and tensor dtypes and shapes do not count,
  a device counts only by type (ranks differ in `cuda:{local_rank}`), and twins
  (several candidates on one graph) never join. Each segment is then contiguous
  in every graph's node order, and its nodes have the same aligned inputs, the
  same uses outside the region and the same external input sharing (`add(x, x)`
  vs `add(x, y)`) on every graph.

  Args:
    graphs: Non-empty sequence of graphs. `graphs[0]` is the reference graph.
    anchors: Corresponding nodes, one per graph in the order of `graphs`, such
      as the same collective on every graph. Anchors whose nodes differ, or that
      are placeholders or outputs, are ignored.

  Returns:
    One `node.name -> segment_id` mapping per graph, all with the same
    segments. Segment IDs are `0..S-1`, in the node order of `graphs[0]`. All
    mappings are empty if the aligned nodes are not in the same order on all
    graphs.

  Raises:
    ValueError: If `graphs` is empty, an anchor tuple length does not match
      `len(graphs)`, or an anchor node does not belong to its corresponding
      graph.
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
