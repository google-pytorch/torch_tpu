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

"""Tests that the compile cache tells process groups apart by their ranks.

A graph names a process group by the group's name, which counts groups in
creation order. Two runs on one host that give the same names to groups of
other ranks must not share compiled programs; a run with the same groups reuses
them.
"""

import os

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch._dynamo.utils import counters
from torch._inductor.runtime.cache_dir_utils import temporary_cache_dir
import torch.multiprocessing as mp
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.distributed import multiprocessing
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils
from tests.distributed import distributed_utils

_WORLD = 8
# Both layouts name their groups "1" and "2".
_COLUMNS = [[0, 2, 4, 6], [1, 3, 5, 7]]
_HALVES = [[0, 1, 2, 3], [4, 5, 6, 7]]


def run_compiled_group_all_reduce(
    layout: list[list[int]], cache_hit: bool
) -> None:
  """Compiles an all-reduce over the group of this rank in `layout`."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  assert int(os.environ["WORLD_SIZE"]) == _WORLD
  my_group, my_ranks = None, None
  for ranks in layout:
    group = dist.new_group(ranks=ranks)
    if rank in ranks:
      my_group, my_ranks = group, ranks

  # A graph without process groups keeps the cache key tag.
  without_groups = torch.fx.symbolic_trace(lambda x: x * 2.0)
  assert (
      _backend._process_group_cache_key_tag(without_groups)  # pylint: disable=protected-access
      == torch.compiler.config.cache_key_tag
  )

  @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
  def all_reduce(x: torch.Tensor) -> torch.Tensor:
    y = x * 1.0
    dist.all_reduce(y, group=my_group)
    return y

  out = all_reduce(torch.full((4,), float(rank + 1), device="tpu")).cpu()
  utils.assert_close(
      out, torch.full((4,), float(sum(r + 1 for r in my_ranks)))
  )
  hits = counters["aot_autograd"]["autograd_cache_hit"]
  assert hits == (1 if cache_hit else 0), dict(counters["aot_autograd"])


class CompileCacheProcessGroupTest(seed_test_utils.MultiProcessRepeatableTest):

  def test_groups_of_other_ranks_do_not_share_programs(self):
    with temporary_cache_dir(self.create_tempdir().full_path):
      distributed_utils.dist_run(
          nproc_per_node=_WORLD,
          fn=run_compiled_group_all_reduce,
          layout=_COLUMNS,
          cache_hit=False,
      )
      # The halves have the names of the columns but other ranks.
      distributed_utils.dist_run(
          nproc_per_node=_WORLD,
          fn=run_compiled_group_all_reduce,
          layout=_HALVES,
          cache_hit=False,
      )
      # The columns again reuse the programs of the first run.
      distributed_utils.dist_run(
          nproc_per_node=_WORLD,
          fn=run_compiled_group_all_reduce,
          layout=_COLUMNS,
          cache_hit=True,
      )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
