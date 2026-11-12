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

"""Tests two jobs sharing one slice through TORCH_TPU_JOB_DEVICE_IDS.

Slice ranks 0-3 form job 0 and 4-7 job 1. Each job's world spans its own
devices, so its collectives and compiled programs run while the other job runs
nothing, and a slice-wide group carries send/recv between the jobs. This is how
an RL trainer and an inference engine on the two halves of a host exchange
weights over ICI.
"""

from datetime import timedelta
import os

from absl.testing import absltest
import portpicker
import torch
from torch import distributed as dist
from torch._C._distributed_c10d import PrefixStore
from torch._inductor.runtime.cache_dir_utils import temporary_cache_dir
import torch.multiprocessing as mp
from torch_tpu._internal.distributed import multiprocessing
from torch_tpu._internal.distributed import tpu_distributed
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils
from tests.distributed import distributed_utils

_SLICE_SIZE = 8
_JOB_SIZE = _SLICE_SIZE // 2
_TIMEOUT = timedelta(minutes=5)


def _run_two_jobs(store_port: int, cache_dir: str) -> None:
  slice_rank = int(os.environ["RANK"])
  job, job_rank = divmod(slice_rank, _JOB_SIZE)
  # Starts the runtime, which joins the whole slice; nothing is compiled yet.
  device_id = int(tpu_distributed.global_device_id())
  slice_store = dist.TCPStore(
      os.environ["MASTER_ADDR"],
      store_port,
      _SLICE_SIZE,
      slice_rank == 0,
      _TIMEOUT,
  )
  slice_store.set(f"device:{slice_rank}", str(device_id))
  job_devices = [
      int(slice_store.get(f"device:{job * _JOB_SIZE + r}"))
      for r in range(_JOB_SIZE)
  ]
  os.environ["TORCH_TPU_JOB_DEVICE_IDS"] = ",".join(map(str, job_devices))
  p2p = tpu_distributed.create_process_group(
      PrefixStore("p2p", slice_store), slice_rank, _SLICE_SIZE, _TIMEOUT
  )
  dist.init_process_group(
      "tpu_dist",
      store=PrefixStore(f"job{job}", slice_store),
      rank=job_rank,
      world_size=_JOB_SIZE,
  )
  world_backend = dist.group.WORLD._get_backend(torch.device("tpu"))
  assert sorted(tpu_distributed.rank_to_device_id(world_backend)) == sorted(
      job_devices
  )

  @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
  def scaled(x: torch.Tensor) -> torch.Tensor:
    return x * 2 + 1

  # Job 1 waits for job 0 to finish its collective and compiled program, so
  # job 0 cannot depend on job 1 launching anything. Both jobs compile the same
  # graph into one shared cache.
  if job == 1:
    slice_store.wait(["job0_done"])
  x = torch.full((1024,), float(job_rank + 1), device="tpu")
  dist.all_reduce(x)
  with torch._inductor.config.patch(fx_graph_cache=True):
    with temporary_cache_dir(cache_dir):
      y = scaled(x)
  utils.assert_close(y.cpu(), torch.full((1024,), 21.0))
  # The jobs lay out their subgroups differently, and both gathers meet in the
  # root store the jobs share.
  if job == 0:
    halves = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    group = halves[job_rank // 2]
  else:
    group = dist.new_group(list(range(_JOB_SIZE)))
  ones = torch.ones(8, device="tpu")
  dist.all_reduce(ones, group=group)
  utils.assert_close(
      ones.cpu(), torch.full((8,), float(dist.get_world_size(group)))
  )
  torch.accelerator.synchronize()
  dist.barrier()
  if job == 0 and job_rank == 0:
    slice_store.set("job0_done", "1")

  # Each job-0 rank sends to its job-1 peer through the slice-wide group.
  peer = slice_rank + _JOB_SIZE if job == 0 else slice_rank - _JOB_SIZE
  for tag in range(2):
    if job == 0:
      payload = torch.full(
          (4096,),
          float(slice_rank * 10 + tag),
          dtype=torch.bfloat16,
          device="tpu",
      )
      p2p.send([payload], peer, tag).wait()
    else:
      payload = torch.empty((4096,), dtype=torch.bfloat16, device="tpu")
      p2p.recv([payload], peer, tag).wait()
      utils.assert_close(
          payload.cpu(),
          torch.full((4096,), float(peer * 10 + tag), dtype=torch.bfloat16),
      )
  torch.accelerator.synchronize()
  dist.barrier()


class JobLocalWorldTest(seed_test_utils.MultiProcessRepeatableTest):

  def test_two_jobs_share_one_slice(self) -> None:
    distributed_utils.dist_run(
        nproc_per_node=_SLICE_SIZE,
        fn=_run_two_jobs,
        store_port=portpicker.pick_unused_port(),
        cache_dir=self.create_tempdir().full_path,
    )


if __name__ == "__main__":  # We are in the parent process.
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
