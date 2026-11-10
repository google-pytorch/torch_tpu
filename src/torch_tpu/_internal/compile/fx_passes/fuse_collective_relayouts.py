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

"""FX pass to fold chunk/cat relayouts into collectives on a non-leading dim.

`torch.distributed._functional_collectives` only gathers and scatters along dim
0, so gathering along dim `d != 0` (e.g. DTensor `Shard(d) -> Replicate`) is
traced as

  w = wait_tensor(all_gather_into_tensor(x, n, group))
  out = cat(chunk(w, n, dim=0), dim=d)

and reduce-scattering along dim `d != 0` (e.g. `Partial -> Shard(d)`) as

  c = cat(chunk(y, n, dim=d), dim=0)
  out = wait_tensor(reduce_scatter_tensor(c, op, n, group))

XLA's all-gather and reduce-scatter accept any dimension, so the chunk/cat only
adds a relayout copy before or after the collective. This pass replaces each
pattern with a single `tpu::_all_gather_dim` or `tpu::_reduce_scatter_dim` on
dim `d`. Both ops order chunks by the process group's replica groups, like the
`_c10d_functional` ops.

A pattern is only rewritten if the chunks are the `n` equal, in-order slices
covering the chunked dim, and the intermediate results have no other users, so
that the original collective becomes dead and is erased.

In an ATen graph, `torch.chunk` appears as `split` + `getitem` and
`torch.narrow` as `slice`, since `aten.chunk` and `aten.narrow` are
CompositeImplicitAutograd ops.
"""

import dataclasses
import operator
from typing import Any

from absl import logging
import torch
from torch import fx
from torch.fx import operator_schemas

# pylint: disable=protected-access
_ALL_GATHER = torch.ops._c10d_functional.all_gather_into_tensor.default
_REDUCE_SCATTER = torch.ops._c10d_functional.reduce_scatter_tensor.default
_WAIT = torch.ops._c10d_functional.wait_tensor.default
_ALL_GATHER_DIM = torch.ops.tpu._all_gather_dim.default
_REDUCE_SCATTER_DIM = torch.ops.tpu._reduce_scatter_dim.default
# pylint: enable=protected-access

_CAT = torch.ops.aten.cat.default
_SPLIT = torch.ops.aten.split.Tensor
_SPLIT_WITH_SIZES = torch.ops.aten.split_with_sizes.default
_SLICE = torch.ops.aten.slice.Tensor

# The reduce ops `tpu::_reduce_scatter_dim` accepts.
_SUPPORTED_REDUCE_OPS = frozenset(
    {"sum", "avg", "product", "min", "max", "band", "bor", "bxor"}
)


def _kwargs(node: Any) -> dict[str, Any] | None:
  """Returns the arguments of an ATen op node by name, with defaults.

  As in `torch.fx.operator_schemas`, a `self` argument is named `input`.
  """
  if not isinstance(node, fx.Node) or not callable(node.target):
    return None
  normalized = operator_schemas.normalize_function(
      node.target, node.args, node.kwargs, normalize_to_only_use_kwargs=True
  )
  return normalized.kwargs if normalized is not None else None


def _static_shape(node: Any) -> tuple[int, ...] | None:
  """Returns the static shape of the tensor value of `node`, if known."""
  val = node.meta.get("val") if isinstance(node, fx.Node) else None
  if not isinstance(val, torch.Tensor):
    return None
  shape = tuple(val.shape)
  return shape if all(isinstance(size, int) for size in shape) else None


def _normalize_dim(dim: Any, rank: int) -> int | None:
  """Returns `dim` wrapped to [0, rank), or None if it is invalid."""
  if not isinstance(dim, int) or not -rank <= dim < rank:
    return None
  return dim % rank


@dataclasses.dataclass(frozen=True)
class _Chunk:
  """A contiguous slice `source[start:stop]` along `dim`."""

  source: fx.Node
  dim: int
  start: int
  stop: int
  producers: tuple[fx.Node, ...]  # The nodes that compute this chunk.


def _as_chunk(node: Any) -> _Chunk | None:
  """Returns the slice of its source that `node` computes, if it is one.

  Handles `getitem(split(source, ...), i)` and `slice(source, dim, start, end)`.

  Args:
    node: A candidate chunk node.
  """
  if not isinstance(node, fx.Node) or node.op != "call_function":
    return None
  if node.target is operator.getitem:
    split, index = node.args
    if not isinstance(split, fx.Node) or split.target not in (
        _SPLIT,
        _SPLIT_WITH_SIZES,
    ):
      return None
    kwargs = _kwargs(split)
    if kwargs is None:
      return None
    shape = _static_shape(kwargs["input"])
    if shape is None or not isinstance(index, int):
      return None
    dim = _normalize_dim(kwargs["dim"], len(shape))
    if dim is None:
      return None
    if split.target == _SPLIT:
      size = kwargs["split_size"]
      even = isinstance(size, int) and size > 0 and shape[dim] % size == 0
      sizes = [size] * (shape[dim] // size) if even else []
    else:
      sizes = list(kwargs["split_sizes"])
    if sum(sizes) != shape[dim] or not 0 <= index < len(sizes):
      return None
    start = sum(sizes[:index])
    return _Chunk(
        kwargs["input"], dim, start, start + sizes[index], (node, split)
    )
  if node.target == _SLICE:
    kwargs = _kwargs(node)
    shape = _static_shape(kwargs["input"]) if kwargs else None
    if kwargs is None or shape is None or kwargs["step"] != 1:
      return None
    dim = _normalize_dim(kwargs["dim"], len(shape))
    if dim is None:
      return None
    # `x[:, :h]` and `x[:, h:]` trace with a None start and a None or INT64_MAX
    # end.
    start = 0 if kwargs["start"] is None else kwargs["start"]
    end = shape[dim] if kwargs["end"] is None else kwargs["end"]
    if not isinstance(start, int) or not isinstance(end, int):
      return None
    end = min(end, shape[dim])
    if not 0 <= start < end:
      return None
    return _Chunk(kwargs["input"], dim, start, end, (node,))
  return None


def _match_chunks(
    nodes: Any, num_chunks: int
) -> tuple[fx.Node, int, list[fx.Node]] | None:
  """Matches the `num_chunks` equal, in-order slices that cover a tensor.

  Args:
    nodes: The candidate chunks, e.g. the tensors concatenated by a `cat`.
    num_chunks: The expected number of chunks.

  Returns:
    The chunked tensor, the (non-negative) chunked dim and the nodes computing
    the chunks, or None if `nodes` do not cover the tensor that way.
  """
  if not isinstance(nodes, (list, tuple)) or not nodes:
    return None
  if len(nodes) != num_chunks:
    return None
  first = _as_chunk(nodes[0])
  source_shape = _static_shape(first.source) if first else None
  if first is None or source_shape is None:
    return None
  size = source_shape[first.dim] // num_chunks
  producers: dict[fx.Node, None] = {}  # Ordered set.
  for i, node in enumerate(nodes):
    chunk = _as_chunk(node)
    if chunk is None or (chunk.source, chunk.dim, chunk.start, chunk.stop) != (
        first.source,
        first.dim,
        i * size,
        (i + 1) * size,
    ):
      return None
    producers.update(dict.fromkeys(chunk.producers))
  return first.source, first.dim, list(producers)


def _is_private(chain: list[fx.Node], output: fx.Node) -> bool:
  """Returns whether all users of `chain` are in `chain` or are `output`."""
  allowed = set(chain) | {output}
  return all(set(node.users) <= allowed for node in chain)


def _replace(
    old: fx.Node,
    chain: list[fx.Node],
    collective: fx.Node,
    target: Any,
    args: tuple[Any, ...],
    graph: fx.Graph,
) -> None:
  """Replaces `old` with `target(*args)` and erases the now unused `chain`.

  Args:
    old: The node whose value the new collective computes.
    chain: The nodes of the matched pattern, including `old`.
    collective: The `_c10d_functional` collective being replaced. The new
      collective is inserted at its position, to keep it where it was relative
      to other collectives, and inherits its meta, e.g. its stack trace and
      custom SPMD-region annotations.
    target: The new collective op.
    args: The arguments of the new collective.
    graph: The graph to transform.
  """
  with graph.inserting_before(collective):
    fused = graph.call_function(target, args)
  fused.meta = dict(collective.meta)
  for key in ("val", "tensor_meta"):
    fused.meta.pop(key, None)
    if key in old.meta:
      fused.meta[key] = old.meta[key]
  old.replace_all_uses_with(fused)
  # Erase users before their inputs.
  for node in sorted(chain, key=list(graph.nodes).index, reverse=True):
    if not node.users:
      graph.erase_node(node)


def _fuse_all_gather(cat: fx.Node, graph: fx.Graph) -> bool:
  """Rewrites `cat(chunk(wait(all_gather(x)), dim=0), dim=d)`, if matched."""
  cat_kwargs, out_shape = _kwargs(cat), _static_shape(cat)
  if cat_kwargs is None or out_shape is None:
    return False
  gather_dim = _normalize_dim(cat_kwargs["dim"], len(out_shape))
  first = _as_chunk(next(iter(cat_kwargs["tensors"]), None))
  if not gather_dim or first is None:  # Dim 0 needs no relayout.
    return False
  wait = first.source
  all_gather = wait.args[0] if wait.target == _WAIT else None
  if not isinstance(all_gather, fx.Node) or all_gather.target != _ALL_GATHER:
    return False
  ag_kwargs = _kwargs(all_gather)
  group_size = ag_kwargs["group_size"] if ag_kwargs else None
  if ag_kwargs is None or not isinstance(group_size, int):
    return False
  match = _match_chunks(cat_kwargs["tensors"], group_size)
  if match is None or match[1] != 0:
    return False
  chain = [all_gather, wait, *match[2], cat]
  if not _is_private(chain[:-1], cat):
    return False
  _replace(
      cat,
      chain,
      all_gather,
      _ALL_GATHER_DIM,
      (ag_kwargs["input"], gather_dim, group_size, ag_kwargs["group_name"]),
      graph,
  )
  return True


def _fuse_reduce_scatter(reduce_scatter: fx.Node, graph: fx.Graph) -> bool:
  """Rewrites `wait(reduce_scatter(cat(chunk(y, dim=d), dim=0)))`, if matched."""
  rs_kwargs = _kwargs(reduce_scatter)
  if rs_kwargs is None or rs_kwargs["reduce_op"] not in _SUPPORTED_REDUCE_OPS:
    return False
  cat = rs_kwargs["input"]
  cat_kwargs, cat_shape = _kwargs(cat), _static_shape(cat)
  if cat_kwargs is None or cat_shape is None or cat.target != _CAT:
    return False
  if _normalize_dim(cat_kwargs["dim"], len(cat_shape)) != 0:
    return False
  group_size = rs_kwargs["group_size"]
  if not isinstance(group_size, int):
    return False
  match = _match_chunks(cat_kwargs["tensors"], group_size)
  if match is None or match[1] == 0:  # A dim-0 chunk/cat is a no-op.
    return False
  y, scatter_dim, producers = match
  waits = list(reduce_scatter.users)
  if len(waits) != 1 or waits[0].target != _WAIT:
    return False
  wait = waits[0]
  # The chunks may be used elsewhere, in which case `_replace` keeps them.
  if not _is_private([cat, reduce_scatter], wait):
    return False
  _replace(
      wait,
      [*producers, cat, reduce_scatter, wait],
      reduce_scatter,
      _REDUCE_SCATTER_DIM,
      (
          y,
          rs_kwargs["reduce_op"],
          scatter_dim,
          group_size,
          rs_kwargs["group_name"],
      ),
      graph,
  )
  return True


def apply(gm_or_graph: fx.GraphModule | fx.Graph) -> None:
  """Folds chunk/cat relayouts into all-gathers and reduce-scatters.

  Args:
    gm_or_graph: The FX GraphModule or Graph to transform in place.
  """
  graph = (
      gm_or_graph.graph
      if isinstance(gm_or_graph, fx.GraphModule)
      else gm_or_graph
  )
  num_all_gathers = num_reduce_scatters = 0
  for node in list(graph.nodes):
    # `erase_node` keeps `node.graph`, so check the erased flag instead.
    if node._erased or node.op != "call_function":  # pylint: disable=protected-access
      continue  # Erased by an earlier rewrite.
    if node.target == _CAT and _fuse_all_gather(node, graph):
      num_all_gathers += 1
    elif node.target == _REDUCE_SCATTER and _fuse_reduce_scatter(node, graph):
      num_reduce_scatters += 1

  if not num_all_gathers and not num_reduce_scatters:
    return
  logging.info(
      "fuse_collective_relayouts: fused %d all-gathers and %d reduce-scatters"
      " with their chunk/cat relayouts",
      num_all_gathers,
      num_reduce_scatters,
  )
  graph.lint()
  if isinstance(gm_or_graph, fx.GraphModule):
    gm_or_graph.recompile()
