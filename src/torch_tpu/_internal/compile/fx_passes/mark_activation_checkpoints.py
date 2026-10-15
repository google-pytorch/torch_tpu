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

"""FX pass to insert variadic optimization barriers for activation checkpoints.

For each activation checkpoint region in the backward pass, this pass inserts a
single variadic optimization barrier immediately before the region's first
remat node, binding the saved forward activations and parameters consumed by
the region's remat ops together with the incoming gradients consumed by the
region's backward ops. In combined forward-loss-backward step graphs, gating
the backward remat inputs through this barrier prevents MLIR and XLA common
subexpression elimination from deduplicating the remat ops against the forward
pass.
"""

import collections
from collections.abc import Iterator
import operator
import torch
from torch import fx
from torch.utils import checkpoint

_OPTIMIZATION_BARRIER_OP = (
    torch.ops.tpu._optimization_barrier.default  # pylint: disable=protected-access
)

_RECOMPUTE_POLICIES = (
    checkpoint.CheckpointPolicy.MUST_RECOMPUTE,
    checkpoint.CheckpointPolicy.PREFER_RECOMPUTE,
)

# Non-compute ops (tuple unpacking, optimization barriers, and PyTorch's
# `SAC_IGNORED_OPS` view/alias/metadata ops) that are excluded when identifying
# a checkpoint region's rematerialized compute nodes.
_IGNORED_REMAT_TARGETS = frozenset({
    operator.getitem,
    _OPTIMIZATION_BARRIER_OP,
    *checkpoint.SAC_IGNORED_OPS,
})


def _is_getitem(node: fx.Node) -> bool:
  """Returns True if `node` is an `operator.getitem` call."""
  return node.op == "call_function" and node.target is operator.getitem


def _is_barrier_getitem(node: fx.Node) -> bool:
  """Returns True if `node` is an `operator.getitem` unpacking a barrier."""
  return (
      _is_getitem(node)
      and isinstance(node.args[0], fx.Node)
      and node.args[0].target is _OPTIMIZATION_BARRIER_OP
  )


def _is_remat_node(node: fx.Node) -> bool:
  """Returns True if `node` is a remat forward op in the backward pass."""
  return (
      node.op == "call_function"
      and node.target not in _IGNORED_REMAT_TARGETS
      and node.meta.get("autograd_backward", False)
      and node.meta.get("ac_graph_id", None) is not None
      and node.meta.get("recompute", None) in _RECOMPUTE_POLICIES
  )


def _is_bwd_op(node: fx.Node) -> bool:
  """Returns True if `node` is a non-remat backward compute op.

  In AOTAutograd-lowered FX graphs, all tensor operations are `call_function`
  nodes (as opposed to `placeholder` inputs, `get_attr` attributes, and the
  `output` return node). Tuple projections (`operator.getitem`) and optimization
  barriers are excluded because they are structural tuple-unpacking and barrier
  plumbing nodes rather than backward compute ops.

  Args:
    node: The FX Node to inspect.

  Returns:
    True if `node` is a non-remat backward compute op.
  """
  return (
      node.op == "call_function"
      and not _is_remat_node(node)
      and node.target not in (operator.getitem, _OPTIMIZATION_BARRIER_OP)
      and node.meta.get("autograd_backward", False)
  )


def _is_bwd_op_or_getitem(node: fx.Node) -> bool:
  """Returns True if `node` is a backward compute op or unpacks one."""
  while _is_getitem(node) and isinstance(node.args[0], fx.Node):
    node = node.args[0]
  return _is_bwd_op(node)


def _iter_with_tuple_getitems(node: fx.Node) -> Iterator[fx.Node]:
  """Yields `node` and any `operator.getitem` unpackings of its output.

  When an op inside a checkpoint region returns a tuple of tensors (such as
  attention or layer norm), FX unpacks each element via `operator.getitem`
  nodes. Including those `getitem` nodes in the region's node set ensures that
  downstream ops consuming the unpacked tensors are recognized as consumers of
  the region, and prevents the `getitem` nodes from being misclassified as
  external inputs to the region.

  Args:
    node: The root FX Node whose output (and any nested tuple unpackings) should
      be yielded.

  Yields:
    `node` followed by all transitive `operator.getitem` users of `node`.
  """
  yield node
  for user in node.users:
    if _is_getitem(user) and user.args[0] is node:
      yield from _iter_with_tuple_getitems(user)


def _is_barrier_input_tensor(node: fx.Node) -> bool:
  """Returns True if `node` produces a non-scalar tensor to gate by a barrier.

  Excludes module attributes, zero-input factory ops, existing barriers, and
  scalar tensors (such as the scalar `1.0` backward loss seed, which would
  become opaque to XLA constant folding if routed through a barrier).

  Args:
    node: The FX Node to inspect.

  Returns:
    True if `node` produces a non-scalar tensor that should be routed through
    an optimization barrier.

  Raises:
    ValueError: If `node` does not have `"val"` metadata populated.
  """
  # Of the six FX node opcodes (`placeholder`, `call_function`, `get_attr`,
  # `call_module`, `call_method`, `output`), only `placeholder` graph inputs and
  # `call_function` operator results can be runtime tensors entering a region in
  # an AOTAutograd graph (`get_attr` nodes are registered constant buffers).
  if node.op not in ("placeholder", "call_function"):
    return False
  # Exclude optimization barriers and their tuple unpackings so running the pass
  # multiple times does not wrap already-gated inputs in nested barriers.
  if node.target is _OPTIMIZATION_BARRIER_OP or _is_barrier_getitem(node):
    return False
  # Exclude 0-input factory ops (such as `aten.ones`, `aten.full`, or
  # `aten.arange`) whose outputs depend only on literal arguments.
  if node.op == "call_function" and not node.all_input_nodes:
    return False
  if "val" not in node.meta:
    raise ValueError(
        f"Expected FX node {node.name!r} to have 'val' metadata populated."
    )
  val = node.meta["val"]
  return isinstance(val, torch.Tensor) and val.ndim > 0


def _group_remat_nodes_by_region(
    graph: fx.Graph,
) -> tuple[dict[int, set[fx.Node]], list[int]]:
  """Groups backward remat nodes by `ac_graph_id` in backward execution order.

  Args:
    graph: The FX Graph to inspect.

  Returns:
    A tuple `(remat_by_region, ordered_regions)` mapping each active region's
    `ac_graph_id` to its backward remat nodes (and tuple unpackings), along with
    the list of `ac_graph_id`s ordered by their first remat node in `graph`.
  """
  remat_by_region: dict[int, set[fx.Node]] = collections.defaultdict(set)
  node_to_region: dict[fx.Node, int] = {}
  node_index: dict[fx.Node, int] = {}
  for idx, node in enumerate(graph.nodes):
    node_index[node] = idx
    if _is_remat_node(node):
      region_id = node.meta["ac_graph_id"]
      for n in _iter_with_tuple_getitems(node):
        remat_by_region[region_id].add(n)
        node_to_region[n] = region_id

  # Normally, when region A's output feeds into region B in the forward pass,
  # PyTorch saves region A's output tensor from the forward pass. However, if
  # that output was not saved from the forward pass, PyTorch rematerializes it
  # in the backward pass right before region B's remat ops. In that case, region
  # B's remat ops take region A's remat ops as inputs; we visit downstream
  # regions first (descending `ac_graph_id`) and walk backward along input edges
  # to merge those upstream remat ops into region B so a single barrier is
  # placed before the combined remat chain.
  assigned_nodes: set[fx.Node] = set()
  for region_id in sorted(remat_by_region.keys(), reverse=True):
    remat = remat_by_region[region_id]
    worklist = list(remat)
    while worklist:
      curr = worklist.pop()
      if curr in assigned_nodes:
        continue
      # Include `curr` along with any `getitem` nodes that unpack its output so
      # a tuple-producing remat op and its unpackings stay in the same region.
      curr_and_getitems = set(_iter_with_tuple_getitems(curr))
      remat.update(curr_and_getitems)
      assigned_nodes.update(curr_and_getitems)
      orig_region_id = node_to_region[curr]
      if orig_region_id != region_id:
        remat_by_region[orig_region_id] -= curr_and_getitems
      for inp in curr.all_input_nodes:
        if inp in node_to_region and inp not in assigned_nodes:
          worklist.append(inp)

  # Remove any regions whose remat nodes were all absorbed into another region,
  # and order the remaining regions by the graph position of their first remat
  # node so region intervals `[entry_idx, next_remat_min)` are always
  # monotonically increasing even when a graph contains multiple
  # forward-backward segments.
  remat_by_region = {
      region_id: nodes for region_id, nodes in remat_by_region.items() if nodes
  }
  ordered_regions = sorted(
      remat_by_region.keys(),
      key=lambda r: min(node_index[n] for n in remat_by_region[r]),
  )
  return remat_by_region, ordered_regions


def _collect_region_nodes(
    remat_nodes: set[fx.Node],
    entry_idx: int,
    next_remat_min: int,
    node_order: dict[fx.Node, int],
) -> set[fx.Node]:
  """Returns the set of FX nodes that make up a region's backward computation.

  In the linear FX instruction order, a checkpoint region's backward pass
  spans the interval from its first remat node (`entry_idx`, where the barrier
  will be inserted) up to the next region's first remat node (`next_remat_min`).
  Any tensor input to the returned set that comes from outside the set is a
  node before the barrier line (`< entry_idx`) and becomes a barrier input.

  Anchoring the traversal on `remat_nodes` is what separates the incoming
  activation gradient (which combines with the remat activations in this
  region's backward ops) from upstream weight gradients (which flow to the
  graph outputs and never connect to this region's remat nodes).

  Args:
    remat_nodes: The backward remat nodes (and tuple unpackings) for the region.
    entry_idx: Index in `node_order` of this region's first remat node.
    next_remat_min: Index in `node_order` of the next backward region's first
      remat node (or `len(node_order)` for the last region).
    node_order: Mapping from each FX Node to its linear graph execution index.

  Returns:
    The set of FX Nodes belonging to this region's remat and backward pass that
    should consume inputs routed through the region's optimization barrier.
  """
  # Step 1: Step forward from remat nodes to include backward views or
  # transposes in this region (< next_remat_min) whose inputs are entirely
  # remat nodes.
  remat_and_views = set(remat_nodes)
  worklist: list[fx.Node] = list(remat_nodes)
  while worklist:
    node = worklist.pop()
    for user in node.users:
      if (
          user not in remat_and_views
          and node_order[user] < next_remat_min
          and _is_bwd_op(user)
          and user.all_input_nodes
          and all(inp in remat_and_views for inp in user.all_input_nodes)
      ):
        for getitem_node in _iter_with_tuple_getitems(user):
          if getitem_node not in remat_and_views:
            remat_and_views.add(getitem_node)
            worklist.append(getitem_node)

  # Step 2: Step forward to add the backward ops in this region
  # (< next_remat_min) that directly consume the remat activations.
  region_nodes = set(remat_and_views)
  for node in remat_and_views:
    for user in node.users:
      if (
          user not in remat_and_views
          and node_order[user] < next_remat_min
          and _is_bwd_op(user)
      ):
        region_nodes.update(_iter_with_tuple_getitems(user))

  # Step 3: Walk backward along inputs through any intermediate non-remat
  # backward ops computed at or after the barrier line (>= entry_idx) until
  # reaching external inputs before the barrier (< entry_idx).
  for node in region_nodes:
    for inp in node.all_input_nodes:
      if (
          inp not in region_nodes
          and node_order[inp] >= entry_idx
          and _is_bwd_op_or_getitem(inp)
      ):
        worklist.append(inp)

  while worklist:
    curr = worklist.pop()
    if curr in region_nodes:
      continue
    curr_and_getitems = set(_iter_with_tuple_getitems(curr))
    region_nodes.update(curr_and_getitems)
    for inp in curr.all_input_nodes:
      if (
          inp not in region_nodes
          and node_order[inp] >= entry_idx
          and _is_bwd_op_or_getitem(inp)
      ):
        worklist.append(inp)
  return region_nodes


def _hoist_tuple_getitems(
    graph: fx.Graph,
    entry_indices: list[int],
    node_order: dict[fx.Node, int],
) -> None:
  """Hoists straddling `operator.getitem` nodes to follow their parent tuple.

  Because the barrier only accepts tensors (not tuples), any `getitem` unpacking
  a pre-region tuple that appears at or after `entry_node` is moved to
  immediately follow its parent tuple so it is captured as a barrier input.

  Args:
    graph: The FX Graph to modify in-place.
    entry_indices: Indices in `node_order` of each region's first remat node.
    node_order: Mapping from each FX Node in `graph` to its linear graph
      execution index. Updated in-place for hoisted `getitem` nodes.
  """
  for node in list(graph.nodes):
    if node.op == "placeholder":
      continue
    parent_idx = node_order[node]
    cursor = node
    for user in list(node.users):
      if (
          _is_getitem(user)
          and user.args[0] is node
          and user.all_input_nodes == [node]
          and any(
              parent_idx < entry_idx <= node_order[user]
              for entry_idx in entry_indices
          )
      ):
        cursor.append(user)
        # Update `node_order` in-place so nested `getitem` users of `user` see
        # its hoisted pre-region index in later iterations.
        node_order[user] = parent_idx
        cursor = user


def apply(gm_or_graph: fx.GraphModule | fx.Graph) -> None:
  """Inserts a variadic optimization barrier at each checkpoint region entry.

  Args:
    gm_or_graph: The FX GraphModule or Graph to modify in-place.

  Raises:
    ValueError: If checkpoint regions have interleaved remat nodes in the FX
      graph, or if a candidate barrier input node is missing `"val"` metadata.
  """
  graph = (
      gm_or_graph.graph
      if isinstance(gm_or_graph, fx.GraphModule)
      else gm_or_graph
  )
  remat_by_region, ordered_regions = _group_remat_nodes_by_region(graph)
  if not ordered_regions:
    return

  initial_order = {node: idx for idx, node in enumerate(graph.nodes)}
  entry_nodes = [
      min(remat_by_region[region_id], key=lambda n: initial_order[n])
      for region_id in ordered_regions
  ]
  entry_indices = [initial_order[node] for node in entry_nodes]
  for i, region_id in enumerate(ordered_regions[:-1]):
    max_remat_idx = max(
        initial_order[n]
        for n in remat_by_region[region_id]
        if _is_remat_node(n)
    )
    if max_remat_idx >= entry_indices[i + 1]:
      raise ValueError(
          f"Checkpoint regions {region_id} and {ordered_regions[i + 1]} have "
          "interleaved remat nodes in the FX graph."
      )

  _hoist_tuple_getitems(graph, entry_indices, initial_order)

  node_order = {node: idx for idx, node in enumerate(graph.nodes)}
  num_nodes = len(node_order)

  # `ordered_regions` is ordered by the graph position of each region's first
  # remat node (`entry_nodes[i]`).
  for i, region_id in enumerate(ordered_regions):
    remat_nodes = remat_by_region[region_id]
    # The barrier is inserted just before the first remat node.
    entry_node = entry_nodes[i]
    entry_idx = node_order[entry_node]

    # Location of the next barrier in the ordering.
    next_remat_min = (
        node_order[entry_nodes[i + 1]]
        if i + 1 < len(entry_nodes)
        else num_nodes
    )

    region_nodes = _collect_region_nodes(
        remat_nodes=remat_nodes,
        entry_idx=entry_idx,
        next_remat_min=next_remat_min,
        node_order=node_order,
    )

    barrier_inputs: list[fx.Node] = []
    seen_inputs: set[fx.Node] = set()
    for node in sorted(region_nodes, key=lambda n: node_order[n]):
      for inp in node.all_input_nodes:
        if (
            inp not in seen_inputs
            and node_order[inp] < entry_idx
            and _is_barrier_input_tensor(inp)
        ):
          seen_inputs.add(inp)
          barrier_inputs.append(inp)

    if not barrier_inputs:
      continue

    copied_meta = {
        k: entry_node.meta[k]
        for k in ("autograd_backward", "stack_trace", "nn_module_stack")
        if k in entry_node.meta
    }
    with graph.inserting_before(entry_node):
      barrier_node = graph.call_function(
          _OPTIMIZATION_BARRIER_OP,
          (barrier_inputs,),
      )
      barrier_node.meta.update(copied_meta)
      barrier_node.meta["val"] = [inp.meta["val"] for inp in barrier_inputs]
      node_order[barrier_node] = entry_idx
      getitem_nodes = []
      for idx, inp in enumerate(barrier_inputs):
        getitem_node = graph.call_function(
            operator.getitem, (barrier_node, idx)
        )
        getitem_node.meta.update(copied_meta)
        getitem_node.meta["val"] = inp.meta["val"]
        node_order[getitem_node] = entry_idx
        getitem_nodes.append(getitem_node)

    def _should_rewire_user(user: fx.Node) -> bool:
      # pylint: disable=cell-var-from-loop
      if user is barrier_node:
        return False
      if user in region_nodes:
        return True
      # Also rewire multi-hop backward ops in this region's interval (such as
      # `dW1`, `dX`, or `db2 = sum(dY)`) that consume `barrier_inputs` without
      # directly consuming a remat node.
      return entry_idx <= node_order[user] < next_remat_min and _is_bwd_op(user)

    for inp, getitem_node in zip(barrier_inputs, getitem_nodes, strict=True):
      inp.replace_all_uses_with(
          getitem_node,
          delete_user_cb=_should_rewire_user,
      )

  graph.lint()
  if isinstance(gm_or_graph, fx.GraphModule):
    gm_or_graph.recompile()
