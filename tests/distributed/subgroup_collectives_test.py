# Copyright 2025 Google LLC
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

"""Tests PyTorch collectives on process group subgroups.

This represents the "manual" (before DTensor) way of doing 2D+ parallelism,
in PyTorch, which is still used in practice (e.g. Meta's fairscale library).

2D+ parallelism means that devices are organized into a mesh, and collectives
can run independently "along rows/columns" of the device mesh.
"""

import inspect
import os

from absl import logging
from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed import device_mesh
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils


def run_manual_2d_mesh() -> None:
  """Test parallel all-reduce on a simple [2, 4] device mesh."""
  dist.init_process_group(backend="tpu_dist")

  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  assert world_size == 8
  rank_to_dev = device_mesh.get_device_assignment()
  dev_to_rank = {dev: r for r, dev in enumerate(rank_to_dev)}
  mesh = [[dev_to_rank[d] for d in row] for row in [[0, 2, 4, 6], [1, 3, 5, 7]]]

  # PyTorch's new_group must be called by every rank for every group, even
  # groups this rank isn't a member of. torch_tpu only uses this rank's own
  # group, so remember the group we are a member of.
  my_subgroup = None
  for subgrp in mesh:
    new_group = torch.distributed.new_group(ranks=subgrp)
    if rank in subgrp:
      my_subgroup = new_group
    else:
      assert new_group == torch.distributed.GroupMember.NON_GROUP_MEMBER

  assert torch.distributed.get_world_size(group=my_subgroup) == 4
  assert torch.distributed.get_rank(group=my_subgroup) in [0, 1, 2, 3]

  # Global input view, used to compute per-device input and expected output
  all_inputs = torch.tensor([[0.0, 1.0, r, r**2] for r in range(world_size)])

  # Get per-device input, and run parallel all-reduce within device subgroups
  x = all_inputs[rank].to(device="tpu")
  torch.distributed.all_reduce(x, group=my_subgroup)
  out_cpu = x.cpu()

  logging.info(
      "all_reduce (world rank=%d, group rank=%d), out = %s",
      rank,
      torch.distributed.get_rank(group=my_subgroup),
      out_cpu,
  )

  # Every process can calculate the expected result for their subgroup
  if rank in mesh[0]:
    expected = all_inputs[mesh[0], :].sum(dim=0)
  else:
    expected = all_inputs[mesh[1], :].sum(dim=0)

  utils.assert_close(actual=out_cpu, expected=expected)


def run_manual_1d_as_2d_mesh() -> None:
  """Test special (redundant) case of [1, n] device mesh. Used in fairscale."""
  dist.init_process_group(backend="tpu_dist")

  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  assert world_size == 8
  mesh_axis_1 = [[0, 1, 2, 3, 4, 5, 6, 7]]
  mesh_axis_2 = [[0], [1], [2], [3], [4], [5], [6], [7]]

  my_subgroup_axis_1 = None  # Process group for device mesh row
  my_subgroup_axis_2 = None  # Process group for device mesh column

  # PyTorch's new_group must be called by every rank for every group;
  # torch_tpu only uses this rank's own groups.
  for subgrp in mesh_axis_1:
    new_group = torch.distributed.new_group(ranks=subgrp)
    if rank in subgrp:
      assert my_subgroup_axis_1 is None  # This should only be set once.
      my_subgroup_axis_1 = new_group

  for subgrp in mesh_axis_2:
    new_group = torch.distributed.new_group(ranks=subgrp)
    if rank in subgrp:
      assert my_subgroup_axis_2 is None  # This should only be set once.
      my_subgroup_axis_2 = new_group

  # All-reduce over rows of [1, N] mesh is equivalent to global all-reduce.
  x = torch.tensor([1.0, rank**2], device="tpu")
  torch.distributed.all_reduce(x, group=my_subgroup_axis_1)
  expected = torch.tensor([8.0, 140.0])
  utils.assert_close(x.cpu(), expected)

  # All-reduce over columns of [1, N] mesh is a no-op.
  x = torch.tensor([1.0, rank**2], device="tpu")
  torch.distributed.all_reduce(x, group=my_subgroup_axis_2)
  expected = torch.tensor([1.0, rank**2])
  utils.assert_close(x.cpu(), expected)


def run_manual_2d_all_gather_reduce_scatter() -> None:
  """Test 2D device mesh of [all-gather dim, reduce-scatter dim] shape."""
  dist.init_process_group(backend="tpu_dist")

  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  assert world_size == 8
  rank_to_dev = device_mesh.get_device_assignment()
  dev_to_rank = {dev: r for r, dev in enumerate(rank_to_dev)}
  mesh_rows = [
      [dev_to_rank[d] for d in row] for row in [[0, 2, 4, 6], [1, 3, 5, 7]]
  ]
  mesh_cols = [[a, b] for (a, b) in zip(*mesh_rows)]  # transpose

  subgroup_ag = None  # Process group for all_gather axis
  subgroup_rs = None  # Process group for reduce_scatter axis
  mesh_row = None
  mesh_col = None

  # Init device mesh (standard PyTorch, before DTensor). PyTorch's new_group
  # must be called by every rank for every group; torch_tpu only uses this
  # rank's own groups.
  for subgroup_ranks in mesh_rows:
    new_group = torch.distributed.new_group(ranks=subgroup_ranks)
    if rank in subgroup_ranks:
      assert subgroup_ag is None  # This should only be set once.
      subgroup_ag = new_group  # (Sub-)ProcessGroup we are a member of.
      mesh_row = subgroup_ranks  # Ranks that are memebrs of this subgroup

  for subgroup_ranks in mesh_cols:
    new_group = torch.distributed.new_group(ranks=subgroup_ranks)
    if rank in subgroup_ranks:
      assert subgroup_rs is None  # This should only be set once.
      subgroup_rs = new_group  # (Sub-)ProcessGroup we are a member of.
      mesh_col = subgroup_ranks  # Ranks that are memebrs of this subgroup

  # PyTorch internally sorts the ranks during subgroup creation. We sort
  # here too, in order to compute same exected result.
  mesh_row = sorted(mesh_row)
  mesh_col = sorted(mesh_col)

  dim0, dim1 = 6, 10  # Shape of the tensor on each device.
  ag_group_size = len(mesh_row)  # Size of subgroup for all-gather dimension
  rs_group_size = len(mesh_col)  # Size of subgroup for reduce-scatter dimension

  # Setup input for all ranks (for computing expected result)
  torch.manual_seed(42)
  all_inputs = torch.rand([world_size, dim0, dim1])
  x = all_inputs[rank].to(device="tpu")  # Input for this specific rank.

  # Perform all_gather on AG axis
  y = torch.empty(ag_group_size * dim0, dim1, device="tpu")  # AG concat mode
  torch.distributed.all_gather_into_tensor(y, x, group=subgroup_ag)
  expected = all_inputs[mesh_row].view(ag_group_size * dim0, dim1)
  utils.assert_close(y.cpu(), expected)

  # Perform reduce_scatter on RS rank
  assert dim0 % rs_group_size == 0, "Invalid test input"
  y = torch.empty(dim0 // rs_group_size, dim1, device="tpu")  # RS concat mode
  torch.distributed.reduce_scatter_tensor(y, x, group=subgroup_rs)
  expected = all_inputs[mesh_col].sum(dim=0)
  if rank == mesh_col[0]:
    expected = expected[0 : (dim0 // rs_group_size), :]
  else:
    expected = expected[(dim0 // rs_group_size) : dim0, :]
  utils.assert_close(y.cpu(), expected)


def _check_group_rank_order(
    group: dist.ProcessGroup, ranks: list[int], world_rank_to_dev: list[int]
) -> None:
  """Checks that `group` follows the (unsorted) order of world `ranks`."""
  rank = int(os.environ["RANK"])
  assert dist.get_rank(group=group) == ranks.index(rank)

  # Group rank 0 is world rank ranks[0].
  x = torch.tensor([float(rank)], device="tpu")
  dist.broadcast(x, group_src=0, group=group)
  utils.assert_close(x.cpu(), torch.tensor([float(ranks[0])]))

  # Output chunk i comes from group rank i, i.e. world rank ranks[i].
  x = torch.tensor([float(rank)], device="tpu")
  y = torch.empty(len(ranks), device="tpu")
  dist.all_gather_into_tensor(y, x, group=group)
  utils.assert_close(y.cpu(), torch.tensor([float(r) for r in ranks]))

  assert device_mesh.get_device_assignment(group) == [
      world_rank_to_dev[r] for r in ranks
  ]


def run_unsorted_subgroups() -> None:
  """Test process groups created with sort_ranks=False."""
  dist.init_process_group(backend="tpu_dist")

  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  assert world_size == 8
  world_rank_to_dev = device_mesh.get_device_assignment()

  # World-sized group with reversed rank order.
  reversed_ranks = list(reversed(range(world_size)))
  reversed_group = dist.new_group(ranks=reversed_ranks, sort_ranks=False)
  _check_group_rank_order(reversed_group, reversed_ranks, world_rank_to_dev)

  # Sub-world groups with non-sorted rank order. PyTorch's new_group must be
  # called by every rank for every group; torch_tpu only uses this rank's own
  # group.
  my_subgroup = None
  my_subgroup_ranks = None
  for subgroup_ranks in [[3, 1, 2, 0], [7, 5, 6, 4]]:
    new_group = dist.new_group(ranks=subgroup_ranks, sort_ranks=False)
    if rank in subgroup_ranks:
      my_subgroup = new_group
      my_subgroup_ranks = subgroup_ranks
  _check_group_rank_order(my_subgroup, my_subgroup_ranks, world_rank_to_dev)


def run_unaligned_subgroups() -> None:
  """Test subgroups not aligned to halves of the world."""
  dist.init_process_group("tpu")
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  half = world_size // 2

  unaligned_a_ranks = list(range(1, half + 1))
  unaligned_b_ranks = list(range(half + 1, world_size)) + [0]
  unaligned_a = dist.new_group(ranks=unaligned_a_ranks)
  unaligned_b = dist.new_group(ranks=unaligned_b_ranks)
  unaligned_group = unaligned_a if rank in unaligned_a_ranks else unaligned_b

  # Dispatched separately, each collective runs on its own group's devices.
  x = torch.ones(4, device="tpu")
  y = torch.ones(4, device="tpu")
  dist.all_reduce(x)
  dist.all_reduce(y, group=unaligned_group)
  res = (x + y).cpu()
  expected = torch.full((4,), float(world_size + half), dtype=torch.float32)
  utils.assert_close(res, expected)

  # Fused into one executable, the program runs on the world and the subgroup
  # collective is tiled over both groups.
  @torch.compile(backend="tpu", fullgraph=True)
  @torch.tpu.distributed.spmd_safe
  def fused_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(a)
    dist.all_reduce(b, group=unaligned_group)
    return a + b

  x = torch.ones(4, device="tpu")
  y = torch.ones(4, device="tpu")
  res = fused_fn(x, y).cpu()
  utils.assert_close(res, expected)


def _run_with_cleanup(test_fn) -> None:
  """Runs `test_fn` and cleanly tears down the default process group on exit.

  On normal completion, synchronizes all workers via `dist.barrier()` before
  destroying the process group so no worker exits while peers are still
  executing collectives. On exception, destroys the process group without a
  barrier to avoid deadlocking when only a subset of workers fail.
  """
  try:
    test_fn()
  except BaseException:
    if dist.is_initialized():
      dist.destroy_process_group()
    raise
  if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()


class SubgroupCollectivesTest(seed_test_utils.MultiProcessRepeatableTest):

  def test_manual_2d_mesh(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=_run_with_cleanup,
        test_fn=run_manual_2d_mesh,
    )

  def test_manual_1d_as_2d_mesh(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=_run_with_cleanup,
        test_fn=run_manual_1d_as_2d_mesh,
    )

  def test_manual_2d_all_gather_reduce_scatter(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=_run_with_cleanup,
        test_fn=run_manual_2d_all_gather_reduce_scatter,
    )

  def test_unsorted_subgroups(self):
    if "sort_ranks" not in inspect.signature(dist.new_group).parameters:
      self.skipTest("dist.new_group() does not support sort_ranks")
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=_run_with_cleanup,
        test_fn=run_unsorted_subgroups,
    )

  def test_unaligned_subgroups(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=_run_with_cleanup,
        test_fn=run_unaligned_subgroups,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
