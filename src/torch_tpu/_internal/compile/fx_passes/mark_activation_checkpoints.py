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
from collections.abc import Collection, Iterator, Mapping, MutableMapping, Sequence, Set
import math
import operator
from typing import Final
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

# Metadata key stamped on synthetic 2D flatten and unflatten `aten.view` nodes
# created by `apply()` so they are never confused with user or autograd
# `aten.view` ops that happen to share the same local graph topology.
_BARRIER_VIEW_META_KEY: Final[str] = "_tpu_barrier_view"


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


def _is_aten_view(node: fx.Node) -> bool:
  """Returns True if `node` is an `aten.view.default` call."""
  return (
      node.op == "call_function" and node.target is torch.ops.aten.view.default
  )


def _is_barrier_flatten_view(node: fx.Node) -> bool:
  """Returns True if `node` flattens an input solely for a barrier."""
  return (
      node.meta.get(_BARRIER_VIEW_META_KEY, False)
      and _is_aten_view(node)
      and bool(node.users)
      and all(user.target is _OPTIMIZATION_BARRIER_OP for user in node.users)
  )


def _is_barrier_unflatten_view(node: fx.Node) -> bool:
  """Returns True if `node` unflattens a 2D-flattened barrier `getitem`."""
  if not (
      node.meta.get(_BARRIER_VIEW_META_KEY, False)
      and _is_aten_view(node)
      and bool(node.args)
      and isinstance(node.args[0], fx.Node)
      and _is_barrier_getitem(node.args[0])
  ):
    return False
  getitem_node = node.args[0]
  barrier_node = getitem_node.args[0]
  slot = getitem_node.args[1]
  if not (
      isinstance(barrier_node, fx.Node)
      and bool(barrier_node.args)
      and isinstance(barrier_node.args[0], (list, tuple))
      and isinstance(slot, int)
      and 0 <= slot < len(barrier_node.args[0])
  ):
    return False
  gated_inp = barrier_node.args[0][slot]
  return (
      isinstance(gated_inp, fx.Node)
      and _is_barrier_flatten_view(gated_inp)
      and bool(gated_inp.args)
      and isinstance(gated_inp.args[0], fx.Node)
      and "val" in gated_inp.args[0].meta
      and "val" in node.meta
      and gated_inp.args[0].meta["val"].shape == node.meta["val"].shape
  )


def _can_flatten_for_barrier(val: torch.Tensor) -> bool:
  """Returns True if `val` can be zero-copy flattened to 2D across a barrier."""
  return (
      val.layout is torch.strided
      and val.ndim >= 3
      and all(isinstance(d, int) for d in val.shape)
      and all(isinstance(s, int) for s in val.stride())
      and val.numel() > 0
      and val.is_contiguous()
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
  `output` return node). Tuple projections (`operator.getitem`), optimization
  barriers, and barrier 2D flatten/unflatten views are excluded because they are
  structural tuple-unpacking and barrier plumbing nodes rather than backward
  compute ops.

  Args:
    node: The FX Node to inspect.

  Returns:
    True if `node` is a non-remat backward compute op.
  """
  return (
      node.op == "call_function"
      and not _is_remat_node(node)
      and node.target not in (operator.getitem, _OPTIMIZATION_BARRIER_OP)
      and not _is_barrier_flatten_view(node)
      and not _is_barrier_unflatten_view(node)
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
  """Returns True if `node` produces a non-empty, non-scalar tensor for barrier.

  Excludes module attributes, zero-input factory ops, existing barriers,
  zero-element tensors (which have 0-byte device buffers), and scalar tensors
  (such as the scalar `1.0` backward loss seed, which would become opaque to XLA
  constant folding if routed through a barrier).

  Args:
    node: The FX Node to inspect.

  Returns:
    True if `node` produces a non-empty, non-scalar tensor that should be routed
    through an optimization barrier.

  Raises:
    ValueError: If `node` does not have `"val"` metadata populated.
  """
  # Of the six FX node opcodes (`placeholder`, `call_function`, `get_attr`,
  # `call_module`, `call_method`, `output`), only `placeholder` graph inputs and
  # `call_function` operator results can be runtime tensors entering a region in
  # an AOTAutograd graph (`get_attr` nodes are registered constant buffers).
  if node.op not in ("placeholder", "call_function"):
    return False
  # Exclude optimization barriers, their tuple unpackings, and their 2D
  # flatten/unflatten views so running the pass multiple times does not wrap
  # already-gated inputs in nested barriers.
  if (
      node.target is _OPTIMIZATION_BARRIER_OP
      or _is_barrier_getitem(node)
      or _is_barrier_flatten_view(node)
      or _is_barrier_unflatten_view(node)
  ):
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
  return (
      isinstance(val, torch.Tensor)
      and val.ndim > 0
      and not any(isinstance(d, int) and d == 0 for d in val.shape)
  )


def _group_remat_nodes_by_region(
    graph: fx.Graph,
) -> tuple[Mapping[int, Set[fx.Node]], Sequence[int]]:
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
    remat_nodes: Collection[fx.Node],
    entry_idx: int,
    next_remat_min: int,
    node_order: Mapping[fx.Node, int],
) -> Set[fx.Node]:
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
    entry_indices: Sequence[int],
    node_order: MutableMapping[fx.Node, int],
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


def _merge_interleaved_regions(
    remat_by_region: Mapping[int, Set[fx.Node]],
    ordered_regions: Sequence[int],
    node_order: Mapping[fx.Node, int],
) -> tuple[Mapping[int, Set[fx.Node]], Sequence[int]]:
  """Merges each region whose remat nodes reach past the next region's entry.

  Some PyTorch versions (such as 2.11 and 2.14) schedule the remat ops of
  different checkpoint regions interleaved in the backward graph. The
  `[entry_idx, next_remat_min)` intervals of such regions would overlap, so the
  later region is merged into the earlier one, and a single barrier gates both.

  Args:
    remat_by_region: Mapping from each region's `ac_graph_id` to its remat
      nodes.
    ordered_regions: `ac_graph_id`s ordered by their first remat node.
    node_order: Mapping from each FX Node to its linear graph execution index.

  Returns:
    The merged `(remat_by_region, ordered_regions)`, with the same meaning.
  """
  merged: dict[int, set[fx.Node]] = {}
  merged_order: list[int] = []
  max_remat_idx = -1
  for region_id in ordered_regions:
    nodes = remat_by_region[region_id]
    entry_idx = min(node_order[n] for n in nodes)
    region_max_idx = max(
        (node_order[n] for n in nodes if _is_remat_node(n)), default=entry_idx
    )
    if merged_order and entry_idx <= max_remat_idx:
      merged[merged_order[-1]] |= nodes
    else:
      merged[region_id] = set(nodes)
      merged_order.append(region_id)
    max_remat_idx = max(max_remat_idx, region_max_idx)
  return merged, merged_order


def apply(gm_or_graph: fx.GraphModule | fx.Graph) -> None:
  """Inserts a variadic optimization barrier at each checkpoint region entry.

  Args:
    gm_or_graph: The FX GraphModule or Graph to modify in-place.

  Raises:
    ValueError: If a candidate barrier input node is missing `"val"` metadata.
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
  remat_by_region, ordered_regions = _merge_interleaved_regions(
      remat_by_region, ordered_regions, initial_order
  )
  entry_nodes = [
      min(remat_by_region[region_id], key=lambda n: initial_order[n])
      for region_id in ordered_regions
  ]
  entry_indices = [initial_order[node] for node in entry_nodes]

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
    # Flatten contiguous >= 3D inputs (such as `[B, S, D]` activations) to 2D
    # before the barrier and unflatten back after `getitem`. Keeping barrier
    # operands <= 2D allows XLA `AlgebraicSimplifier` to cancel adjacent
    # `unflatten -> flatten` reshapes before 2D matmuls instead of folding 3D
    # barrier outputs into multi-contracting 3D `dot`s (which trigger mismatched
    # `{1, 2, 0}` layout assignment and deferred SparseCore transposes on TPU
    # v7x), while 3D consumers still receive a zero-copy bitcast of the 2D
    # barrier output.
    plumbing_nodes: set[fx.Node] = set()
    with graph.inserting_before(entry_node):
      gated_inputs: list[fx.Node] = []
      for inp in barrier_inputs:
        val = inp.meta["val"]
        if _can_flatten_for_barrier(val):
          flat_shape = [math.prod(val.shape[:-1]), val.shape[-1]]
          flat_inp = graph.call_function(
              torch.ops.aten.view.default, (inp, flat_shape)
          )
          flat_inp.meta.update(copied_meta)
          flat_inp.meta["val"] = val.view(flat_shape)
          flat_inp.meta[_BARRIER_VIEW_META_KEY] = True
          node_order[flat_inp] = entry_idx
          plumbing_nodes.add(flat_inp)
          gated_inputs.append(flat_inp)
        else:
          gated_inputs.append(inp)

      barrier_node = graph.call_function(
          _OPTIMIZATION_BARRIER_OP,
          (gated_inputs,),
      )
      barrier_node.meta.update(copied_meta)
      barrier_node.meta["val"] = [inp.meta["val"] for inp in gated_inputs]
      node_order[barrier_node] = entry_idx
      plumbing_nodes.add(barrier_node)

      getitem_nodes: list[fx.Node] = []
      for idx, gated_inp in enumerate(gated_inputs):
        getitem_node = graph.call_function(
            operator.getitem, (barrier_node, idx)
        )
        getitem_node.meta.update(copied_meta)
        getitem_node.meta["val"] = gated_inp.meta["val"]
        node_order[getitem_node] = entry_idx
        plumbing_nodes.add(getitem_node)
        getitem_nodes.append(getitem_node)

      replacement_nodes: list[fx.Node] = []
      for inp, gated_inp, getitem_node in zip(
          barrier_inputs, gated_inputs, getitem_nodes, strict=True
      ):
        if gated_inp is not inp:
          orig_val = inp.meta["val"]
          unflat_node = graph.call_function(
              torch.ops.aten.view.default,
              (getitem_node, list(orig_val.shape)),
          )
          unflat_node.meta.update(copied_meta)
          unflat_node.meta["val"] = orig_val
          unflat_node.meta[_BARRIER_VIEW_META_KEY] = True
          node_order[unflat_node] = entry_idx
          plumbing_nodes.add(unflat_node)
          replacement_nodes.append(unflat_node)
        else:
          replacement_nodes.append(getitem_node)

    def _should_rewire_user(user: fx.Node) -> bool:
      # pylint: disable=cell-var-from-loop
      if user in plumbing_nodes:
        return False
      if user in region_nodes:
        return True
      # Also rewire multi-hop backward ops in this region's interval (such as
      # `dW1`, `dX`, or `db2 = sum(dY)`) that consume `barrier_inputs` without
      # directly consuming a remat node.
      return entry_idx <= node_order[user] < next_remat_min and _is_bwd_op(user)

    for inp, replacement_node in zip(
        barrier_inputs, replacement_nodes, strict=True
    ):
      inp.replace_all_uses_with(
          replacement_node,
          delete_user_cb=_should_rewire_user,
      )

  graph.lint()
  if isinstance(gm_or_graph, fx.GraphModule):
    gm_or_graph.recompile()
