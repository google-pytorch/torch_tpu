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

import collections
import collections.abc
from typing import Any

import torch
import torch.distributed as dist
from torch_tpu._internal.distributed import collective_ops

_COLLECTIVE_OPS = collective_ops.COLLECTIVE_OPS


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
      ranks: collections.abc.Sequence[int],
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
    if isinstance(other, ProcessGroupId):
      return self.ranks == other.ranks
    return False

  def __iter__(self) -> collections.abc.Iterator[int]:
    return iter(self.ranks)

  def __len__(self) -> int:
    return len(self.ranks)

  def __getitem__(self, index: int) -> int:
    return self.ranks[index]


def _get_collective_op(node: torch.fx.Node) -> Any | None:
  """Returns the collective OpOverloadPacket if node is a collective call, else None.

  In FX graphs, node.target may be an OpOverload (e.g., all_reduce.default) or
  directly an OpOverloadPacket. Checking the `overloadpacket` attribute resolves
  specific overloads back to their base OpOverloadPacket for matching against
  _COLLECTIVE_OPS.

  Args:
    node: An FX Node to inspect.

  Returns:
    The collective OpOverloadPacket if the node is a collective operation, or
    None otherwise.
  """
  if node.op != "call_function":
    return None
  target = getattr(node.target, "overloadpacket", node.target)
  if target in _COLLECTIVE_OPS:
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
    ValueError: If the node is not a collective operation in _COLLECTIVE_OPS,
      if the group argument cannot be found, or if the process group cannot be
      resolved.
  """
  collective_op = _get_collective_op(node)
  if collective_op is None:
    raise ValueError(
        f"expected node target to be in {_COLLECTIVE_OPS}, got {node.target}."
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
) -> dict[ProcessGroupId, int]:
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
