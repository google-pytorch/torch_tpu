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

"""Utilities for extracting and manipulating process group information from FX graphs."""

import bisect
import collections
from collections.abc import Iterator, Mapping, Sequence
import functools
from typing import Any

import torch
import torch.distributed as dist
from torch.utils import _pytree as pytree
from torch_tpu._internal.distributed import collective_ops


class ProcessGroupId:
  """Identifier for a process group consisting of sorted, unique ranks.

  Attributes:
    ranks: A tuple of non-negative rank integers representing the process group
      members, sorted in ascending order with no duplicates.
    world_size: Optional world size. If provided, all ranks must be in [0,
      world_size).
  """

  def __init__(
      self,
      ranks: Sequence[int],
      world_size: int | None = None,
  ) -> None:
    if not ranks:
      raise ValueError("ranks cannot be empty.")

    if any(r < 0 for r in ranks):
      raise ValueError(f"All ranks must be non-negative, got {ranks}.")

    if world_size is not None:
      if world_size <= 0:
        raise ValueError(
            f"world_size must be strictly positive, got {world_size}."
        )
      if any(r >= world_size for r in ranks):
        raise ValueError(
            f"All ranks must be in [0, {world_size}), got {ranks}."
        )

    if len(ranks) != len(set(ranks)):
      raise ValueError(f"ranks must not contain duplicates, got {ranks}.")

    if list(ranks) != sorted(ranks):
      raise ValueError(f"ranks must be sorted in ascending order, got {ranks}.")

    self.ranks = tuple(ranks)
    self.world_size = world_size

  def __hash__(self) -> int:
    return hash(self.ranks)

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    if not isinstance(other, ProcessGroupId):
      return False
    if len(self.ranks) != len(other.ranks):
      return False
    if self.is_contiguous() and other.is_contiguous():
      # Sorted, unique and contiguous: equal iff same start and same length.
      # The handshake coordinator compares one group per incoming request, so
      # an O(len(ranks)) comparison would make each round O(world_size^2).
      return self.ranks[0] == other.ranks[0]
    return self.ranks == other.ranks

  def __contains__(self, rank: object) -> bool:
    """Returns whether `rank` is a member, in O(log(len(ranks)))."""
    if not isinstance(rank, int):
      return False
    i = bisect.bisect_left(self.ranks, rank)
    return i < len(self.ranks) and self.ranks[i] == rank

  def __iter__(self) -> Iterator[int]:
    return iter(self.ranks)

  def __len__(self) -> int:
    return len(self.ranks)

  def __getitem__(self, index: int) -> int:
    return self.ranks[index]

  def __repr__(self) -> str:
    if self.is_contiguous() and len(self.ranks) > 1:
      ranks_repr = f"range({self.ranks[0]}, {self.ranks[-1] + 1})"
    else:
      ranks_repr = repr(self.ranks)
    if self.world_size is not None:
      return f"ProcessGroupId(ranks={ranks_repr}, world_size={self.world_size})"
    return f"ProcessGroupId(ranks={ranks_repr})"

  def is_contiguous(self) -> bool:
    """Returns whether ranks are exactly range(ranks[0], ranks[-1] + 1)."""
    # Ranks are sorted and unique, so this is an O(1) check.
    return self.ranks[-1] - self.ranks[0] + 1 == len(self.ranks)

  def to_wire(self) -> dict[str, Any]:
    """Returns a compact, JSON-serializable encoding of this process group.

    Contiguous groups (e.g. the global group) are encoded as a half-open
    `{"start": ..., "stop": ...}` range, so the encoding size does not grow with
    the number of ranks. Other groups are encoded as `{"ranks": [...]}`.

    Note: The encoding is not guaranteed to be stable across TorchTPU versions.
    """
    if self.is_contiguous():
      return {"start": self.ranks[0], "stop": self.ranks[-1] + 1}
    return {"ranks": list(self.ranks)}

  @classmethod
  def from_wire(cls, wire: dict[str, Any]) -> "ProcessGroupId":
    """Decodes a ProcessGroupId produced by `to_wire`.

    Decoded groups are cached and shared between calls, so the O(len(ranks))
    validation in `__init__` runs once per distinct group rather than once per
    decoded message.

    Args:
      wire: The encoding returned by `to_wire`.

    Returns:
      The (possibly shared) decoded ProcessGroupId. Callers must not mutate it.
    """
    if "ranks" in wire:
      return _cached_process_group_from_ranks(tuple(wire["ranks"]))
    return _cached_process_group_from_range(wire["start"], wire["stop"])


# Upper bound on the number of distinct decoded process groups kept alive by
# `ProcessGroupId.from_wire`. Each entry holds an O(len(ranks)) tuple, so the
# bound caps memory if a job keeps creating ad-hoc groups. 1024 covers every
# subgroup of a device-mesh axis whose groups have at least world_size / 1024
# ranks (e.g. all 512 pairs of a 1024-rank job). A cache miss only costs
# re-validating that group, never correctness.
_MAX_CACHED_WIRE_PROCESS_GROUPS = 1024


@functools.lru_cache(maxsize=_MAX_CACHED_WIRE_PROCESS_GROUPS)
def _cached_process_group_from_range(start: int, stop: int) -> ProcessGroupId:
  return ProcessGroupId(range(start, stop))


@functools.lru_cache(maxsize=_MAX_CACHED_WIRE_PROCESS_GROUPS)
def _cached_process_group_from_ranks(ranks: tuple[int, ...]) -> ProcessGroupId:
  return ProcessGroupId(ranks)


def _get_collective_op(node: torch.fx.Node) -> Any | None:
  """Returns the collective OpOverloadPacket if node is a collective call, else None.

  In FX graphs, node.target may be an OpOverload (e.g., all_reduce.default) or
  directly an OpOverloadPacket. Checking the `overloadpacket` attribute resolves
  specific overloads back to their base OpOverloadPacket for matching against
  TorchTPU's collective registry.

  Args:
    node: An FX Node to inspect.

  Returns:
    The collective OpOverloadPacket if the node is a collective operation, or
    None otherwise.
  """
  if node.op != "call_function":
    return None
  target = getattr(node.target, "overloadpacket", node.target)
  if collective_ops.is_collective_op(target):
    return target
  return None


def _extract_process_group_id_from_node(
    node: torch.fx.Node,
) -> ProcessGroupId:
  """Extracts the ProcessGroupId from a collective FX node.

  Args:
    node: An FX Node representing a collective operation.

  Returns:
    The ProcessGroupId associated with the collective operation.

  Raises:
    ValueError: If the node is not a recognized collective operation,
      if the group argument cannot be found, or if the process group cannot be
      resolved.
  """
  collective_op = _get_collective_op(node)
  if collective_op is None:
    raise ValueError(
        "expected node target to be in the collective registry, got"
        f" {node.target}."
        " This is a torch_tpu bug."
    )

  group: Any = None
  if "group_name" in node.kwargs:
    group = node.kwargs["group_name"]
  elif "group" in node.kwargs:
    group = node.kwargs["group"]
  else:
    # When arguments are passed positionally in node.args, their position
    # differs per collective op (e.g. index 2 for all_reduce, index 3 for
    # reduce_scatter_tensor). Inspecting schema.arguments allows dynamically
    # locating the 'group_name' or 'group' parameter for any collective op
    # overload.
    schema = getattr(node.target, "_schema", None)
    if schema is None and hasattr(collective_op, "default"):
      schema = getattr(collective_op.default, "_schema", None)

    if schema is not None:
      for idx, arg in enumerate(schema.arguments):
        if arg.name in ("group_name", "group") and idx < len(node.args):
          group = node.args[idx]
          break

  if group is None:
    raise ValueError(
        f"Could not infer process group argument from collective node {node}."
    )

  if isinstance(group, (list, tuple, range)):
    ranks = [int(r) for r in group]
  elif isinstance(group, (str, int)):
    # pylint: disable=protected-access
    pg = dist.distributed_c10d._resolve_process_group(
        dist.distributed_c10d.GroupName(str(group))
    )
    # pylint: enable=protected-access
    ranks = dist.get_process_group_ranks(pg)
  elif isinstance(group, dist.ProcessGroup):
    ranks = dist.get_process_group_ranks(group)
  else:
    raise ValueError(
        f"Unsupported process group type {type(group)} with value {group!r} in"
        f" node {node}."
    )

  if not ranks:
    raise ValueError(f"Process group ranks cannot be empty for node {node}.")
  return ProcessGroupId(sorted(ranks), dist.get_world_size())


def get_num_collectives_per_pg(
    graph_module: torch.fx.GraphModule,
) -> Mapping[ProcessGroupId, int]:
  """Returns the number of collective operations per process group in the given graph_module.

  Args:
    graph_module: An FX GraphModule to analyze.

  Returns:
    A mapping from ProcessGroupId to the count of collective operations
    targeting that group.
  """
  counts: dict[ProcessGroupId, int] = collections.defaultdict(int)
  for node in graph_module.graph.nodes:
    if _get_collective_op(node) is not None:
      pg_id = _extract_process_group_id_from_node(node)
      counts[pg_id] += 1
  return counts


def get_process_groups(
    graph_module: torch.fx.GraphModule,
) -> Mapping[str, dist.ProcessGroup]:
  """Returns the registered process groups `graph_module` refers to, by name.

  A group counts if it, or its name, is an argument of a node of the graph or of
  a nested graph such as a scan body. Matching arguments rather than known
  collective ops keeps groups of unrecognized ops from being left out; an
  argument that only happens to equal a group's name adds that group.
  """
  # pylint: disable=protected-access
  pg_names: Mapping[dist.ProcessGroup, str] = (
      dist.distributed_c10d._world.pg_names
  )
  # pylint: enable=protected-access
  groups_by_name = {name: pg for pg, name in pg_names.items()}
  groups = {}
  for module in graph_module.modules():
    if not isinstance(module, torch.fx.GraphModule):
      continue
    for node in module.graph.nodes:
      values = pytree.tree_leaves((node.args, node.kwargs))
      if node.op == "get_attr":
        # pylint: disable-next=protected-access
        values.append(torch.fx.graph_module._get_attr(module, node.target))
      for value in values:
        if isinstance(value, dist.ProcessGroup) and value in pg_names:
          groups[pg_names[value]] = value
        elif isinstance(value, str) and value in groups_by_name:
          groups[value] = groups_by_name[value]
  return groups
