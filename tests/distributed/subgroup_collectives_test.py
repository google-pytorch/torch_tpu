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

import os

from absl import logging
from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


def run_manual_2d_mesh() -> None:
  """Test parallel all-reduce on a simple [2, 4] device mesh."""
  dist.init_process_group(backend="tpu_dist")

  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  assert world_size == 8
  mesh = [[0, 5, 3, 4], [7, 6, 1, 2]]  # Subgroups can be arbitrary subsets

  # Initialize all subgroups (even the ones we aren't members of!),
  # and remember the (sub)group we are a member of.
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
  mesh_rows = [[0, 3, 5, 1], [2, 6, 4, 7]]
  mesh_cols = [[a, b] for (a, b) in zip(*mesh_rows)]  # transpose

  subgroup_ag = None  # Process group for all_gather axis
  subgroup_rs = None  # Process group for reduce_scatter axis
  mesh_row = None
  mesh_col = None

  # Init device mesh (standard PyTorch, before DTensor):
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


class SubgroupCollectivesTest(absltest.TestCase):

  def test_manual_2d_mesh(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(run_manual_2d_mesh),
    )

  def test_manual_1d_as_2d_mesh(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(run_manual_1d_as_2d_mesh),
    )

  def test_manual_2d_all_gather_reduce_scatter(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_manual_2d_all_gather_reduce_scatter
        ),
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
