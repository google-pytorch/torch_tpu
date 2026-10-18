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

"""Tests compiled programs with collectives over two kinds of process groups.

Model-parallel inference layouts compile one forward that runs collectives over
two kinds of groups, for example tensor-parallel all-reduces over pairs of ranks
next to context-parallel collectives over other ranks. Each case compiles one
such program and checks its results.
"""

import signal

from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils


def _my_group(rank: int, layout: list[list[int]]) -> dist.ProcessGroup:
  """Creates every group of `layout` and returns the one holding `rank`."""
  mine = None
  for ranks in layout:
    group = dist.new_group(ranks=ranks)
    if rank in ranks:
      mine = group
  return mine


def _group_sum(rank: int, layout: list[list[int]]) -> float:
  """Sum of `r + 1` over the ranks r of the group of `layout` holding `rank`."""
  return float(sum(r + 1 for r in next(g for g in layout if rank in g)))


def run_pcp_x_tp() -> None:
  """Context parallelism 4 x tensor parallelism 2 in one program.

  Tensor-parallel groups are pairs of consecutive ranks; context-parallel
  groups are the ranks with the same tensor-parallel rank.
  """
  dist.init_process_group("tpu")
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  assert world_size == 8
  tp_layout = [[r, r + 1] for r in range(0, world_size, 2)]
  pcp_layout = [list(range(t, world_size, 2)) for t in range(2)]
  tp_group = _my_group(rank, tp_layout)
  pcp_group = _my_group(rank, pcp_layout)

  @torch.compile(backend="tpu", fullgraph=True)
  @torch.tpu.distributed.spmd_safe
  def forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(a, group=tp_group)
    dist.all_reduce(b, group=pcp_group)
    return torch.stack([a, b])

  x = torch.full((4,), float(rank + 1), device="tpu")
  y = torch.full((4,), float(rank + 1), device="tpu")
  res = forward(x, y).cpu()
  expected = torch.stack([
      torch.full((4,), _group_sum(rank, tp_layout)),
      torch.full((4,), _group_sum(rank, pcp_layout)),
  ])
  utils.assert_close(res, expected)


def run_tp_x_dcp() -> None:
  """Tensor parallelism 8 x decode context parallelism 4 in one program.

  The tensor-parallel group is the world; decode context-parallel groups are
  consecutive halves of it.
  """
  dist.init_process_group("tpu")
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  assert world_size == 8
  dcp_layout = [list(range(0, 4)), list(range(4, 8))]
  dcp_group = _my_group(rank, dcp_layout)

  @torch.compile(backend="tpu", fullgraph=True)
  @torch.tpu.distributed.spmd_safe
  def forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(a)
    dist.all_reduce(b, group=dcp_group)
    return torch.stack([a, b])

  x = torch.full((4,), float(rank + 1), device="tpu")
  y = torch.full((4,), float(rank + 1), device="tpu")
  res = forward(x, y).cpu()
  expected = torch.stack([
      torch.full((4,), _group_sum(rank, [list(range(world_size))])),
      torch.full((4,), _group_sum(rank, dcp_layout)),
  ])
  utils.assert_close(res, expected)


def run_pp_x_pcp_x_tp() -> None:
  """Pipeline parallelism 2 x context parallelism 2 x tensor parallelism 2.

  Each stage holds four consecutive ranks: two tensor-parallel pairs and two
  context-parallel pairs. Each stage compiles a program that combines its
  tensor-parallel and context-parallel collectives and runs it a different
  number of times than the other stage, so the program must run on its stage
  alone.
  """
  dist.init_process_group("tpu")
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  assert world_size == 8
  signal.alarm(300)
  stage = rank // 4
  tp_layout = [[r, r + 1] for r in range(0, world_size, 2)]
  pcp_layout = [[s + t, s + t + 2] for s in (0, 4) for t in range(2)]
  tp_group = _my_group(rank, tp_layout)
  pcp_group = _my_group(rank, pcp_layout)

  @torch.compile(backend="tpu", fullgraph=True)
  @torch.tpu.distributed.spmd_safe
  def forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(a, group=tp_group)
    dist.all_reduce(b, group=pcp_group)
    return torch.stack([a, b])

  for _ in range(stage + 1):
    x = torch.full((4,), float(rank + 1), device="tpu")
    y = torch.full((4,), float(rank + 1), device="tpu")
    res = forward(x, y).cpu()
    expected = torch.stack([
        torch.full((4,), _group_sum(rank, tp_layout)),
        torch.full((4,), _group_sum(rank, pcp_layout)),
    ])
    utils.assert_close(res, expected)
  # A world collective after the stages' programs; it deadlocks if those
  # programs ran over both stages.
  dist.barrier()
  signal.alarm(0)


def _run_with_cleanup(test_fn) -> None:
  """Runs `test_fn`, then tears down the default process group."""
  try:
    test_fn()
  except BaseException:
    if dist.is_initialized():
      dist.destroy_process_group()
    raise
  if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()


class MixedGroupProgramsTest(seed_test_utils.MultiProcessRepeatableTest):

  def test_pcp_x_tp(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=_run_with_cleanup,
        test_fn=run_pcp_x_tp,
    )

  def test_tp_x_dcp(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=_run_with_cleanup,
        test_fn=run_tp_x_dcp,
    )

  def test_pp_x_pcp_x_tp(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=_run_with_cleanup,
        test_fn=run_pp_x_pcp_x_tp,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
