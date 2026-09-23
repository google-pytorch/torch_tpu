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

"""Regression test for eager collectives when TPU is touched before process group creation."""

from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils


def _warmup_then_init_pg(test_fn, *args, **kwargs):
  warmup = torch.zeros(1, device="tpu")
  _ = warmup.cpu()
  torch.tpu.synchronize()

  dist.init_process_group(backend="tpu_dist")
  try:
    test_fn(*args, **kwargs)
  finally:
    if dist.is_initialized():
      dist.barrier()
      dist.destroy_process_group()


def _run_eager_collectives():
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  explicit_world_pg = dist.new_group(
      ranks=list(range(world_size)), backend="tpu_dist"
  )
  expected = torch.arange(world_size, dtype=torch.float32)

  # all_reduce
  summed = torch.full(
      (world_size,), float(rank), device="tpu", dtype=torch.float32
  )
  dist.all_reduce(summed)
  utils.assert_close(
      summed.cpu(),
      torch.full((world_size,), world_size * (world_size - 1) / 2.0),
  )

  # all_gather (default group)
  gathered_default = torch.empty(world_size, device="tpu", dtype=torch.float32)
  dist.all_gather_into_tensor(
      gathered_default,
      torch.full((1,), float(rank), device="tpu", dtype=torch.float32),
  )
  utils.assert_close(gathered_default.cpu(), expected)

  # all_gather (explicit world group)
  gathered_new_group = torch.empty(
      world_size, device="tpu", dtype=torch.float32
  )
  dist.all_gather_into_tensor(
      gathered_new_group,
      torch.full((1,), float(rank), device="tpu", dtype=torch.float32),
      group=explicit_world_pg,
  )
  utils.assert_close(gathered_new_group.cpu(), expected)

  # all_to_all_single
  send = torch.full(
      (world_size,), float(rank), device="tpu", dtype=torch.float32
  )
  recv = torch.empty(world_size, device="tpu", dtype=torch.float32)
  dist.all_to_all_single(recv, send)
  utils.assert_close(recv.cpu(), expected)


class CollectiveOpsDeviceOrderTest(seed_test_utils.MultiProcessRepeatableTest):
  _world_size = 8

  def test_collectives_with_pre_pg_warmup(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _warmup_then_init_pg, world_size=self._world_size
        ),
        test_fn=_run_eager_collectives,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
