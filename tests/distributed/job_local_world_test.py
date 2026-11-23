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

"""End-to-end test for sharing one TPU slice across independent jobs."""

import datetime
import os
import time

from absl.testing import absltest
import portpicker
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed import multiprocessing
from torch_tpu._internal.distributed import spmd_util
from torch_tpu._internal.distributed import tpu_distributed
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils
from tests.distributed import distributed_utils
from tests.distributed import worker_supervisor

_SLICE_SIZE = 8
_JOB_SIZE = 4
_TIMEOUT = datetime.timedelta(seconds=120)
_WORKER_DEADLINE_SECONDS = 240.0


def _run_job_collectives_and_compile(
    job: int, job_rank: int
) -> dist.ProcessGroup | None:
  """Exercises job-local collectives, torch.compile, mesh, and subgroups."""
  assert dist.get_world_size() == _JOB_SIZE
  assert dist.get_rank() == job_rank
  assert len(tpu_distributed.global_device_attributes()) == _JOB_SIZE

  x = torch.full((4,), float(job_rank + 1), device="tpu")
  dist.all_reduce(x)
  utils.assert_close(x.cpu(), torch.full((4,), 10.0))

  @torch.compile(backend="tpu", fullgraph=True)
  @spmd_util.spmd_safe
  def compiled_all_reduce(t: torch.Tensor) -> torch.Tensor:
    out = t * 2.0
    dist.all_reduce(out)
    return out

  y = torch.full((4, 8), float(job_rank + 1), device="tpu")
  utils.assert_close(compiled_all_reduce(y).cpu(), torch.full((4, 8), 20.0))

  mesh = torch.tpu.distributed.topology_aware_mesh((2, 2))
  assert tuple(mesh.shape) == (2, 2), mesh.shape
  assert sorted(mesh.flatten().tolist()) == [0, 1, 2, 3], mesh

  uneven_pg = None
  if job == 0:
    pg_low = dist.new_group(ranks=[0, 1])
    pg_high = dist.new_group(ranks=[2, 3])
    my_subgroup = pg_low if job_rank < 2 else pg_high
    sub_t = torch.full((4,), float(job_rank + 1), device="tpu")
    dist.all_reduce(sub_t, group=my_subgroup)
    expected = 3.0 if job_rank < 2 else 7.0
    utils.assert_close(sub_t.cpu(), torch.full((4,), expected))

    if job_rank in (1, 2, 3):
      uneven_pg = dist.new_group(
          ranks=[1, 2, 3], use_local_synchronization=True
      )
      u = torch.full((4,), float(job_rank + 1), device="tpu")
      dist.all_reduce(u, group=uneven_pg)
      utils.assert_close(u.cpu(), torch.full((4,), 9.0))
  else:
    pg_all = dist.new_group(ranks=[0, 1, 2, 3])
    sub_t = torch.full((4,), float(job_rank + 1), device="tpu")
    dist.all_reduce(sub_t, group=pg_all)
    utils.assert_close(sub_t.cpu(), torch.full((4,), 10.0))

  return uneven_pg


def _worker_fn(
    slice_rank: int,
    slice_store_port: int,
    job_ports: tuple[int, int],
    base_seed: int,
) -> None:
  """Runs one slice worker belonging to one of two 4-rank jobs."""
  multiprocessing.parse_absl_flags()

  job = slice_rank // _JOB_SIZE
  job_rank = slice_rank % _JOB_SIZE

  # Configure job-local identity without spoofing TPU_VISIBLE_DEVICES; C++
  # environment initialization derives the local device from slice_rank.
  os.environ.pop("TPU_VISIBLE_DEVICES", None)
  os.environ.pop("TPU_VISIBLE_CHIPS", None)
  os.environ["TORCH_TPU_SLICE_RANK"] = str(slice_rank)
  os.environ["RANK"] = str(job_rank)
  os.environ["LOCAL_RANK"] = str(job_rank)
  os.environ["WORLD_SIZE"] = str(_JOB_SIZE)
  os.environ["LOCAL_WORLD_SIZE"] = str(_JOB_SIZE)
  os.environ["GROUP_RANK"] = "0"
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = str(job_ports[job])

  seed_test_utils.seed_rngs(base_seed + slice_rank)

  slice_store = dist.TCPStore(
      host_name="localhost",
      port=slice_store_port,
      world_size=_SLICE_SIZE,
      is_master=(slice_rank == 0),
      timeout=_TIMEOUT,
  )
  full_slice_pg = tpu_distributed.create_process_group(
      dist.PrefixStore("full_slice", slice_store),
      slice_rank,
      _SLICE_SIZE,
      _TIMEOUT,
  )

  dist.init_process_group(
      "tpu_dist",
      store=dist.PrefixStore(f"job{job}", slice_store),
      rank=job_rank,
      world_size=_JOB_SIZE,
      timeout=_TIMEOUT,
  )
  uneven_pg = None
  try:
    assert dist.get_world_size() == _JOB_SIZE
    assert len(tpu_distributed.global_device_attributes()) == _JOB_SIZE

    if job == 0:
      uneven_pg = _run_job_collectives_and_compile(job=0, job_rank=job_rank)
      dist.barrier()
      if job_rank == 0:
        slice_store.set("job0_done", "1")
    else:
      slice_store.wait(["job0_done"], _TIMEOUT)
      uneven_pg = _run_job_collectives_and_compile(job=1, job_rank=job_rank)

    # Full-slice collective across all 8 devices on the independent PG.
    full_tensor = torch.full((16,), float(slice_rank + 1), device="tpu")
    work = full_slice_pg.allreduce([full_tensor])
    if work is not None:
      work.wait()
    utils.assert_close(full_tensor.cpu(), torch.full((16,), 36.0))

    # Cross-job multi-tag bfloat16 point-to-point exchange.
    if job == 0:
      peer = slice_rank + _JOB_SIZE
      send_a = torch.full(
          (4, 8), float(slice_rank + 1), dtype=torch.bfloat16, device="tpu"
      )
      send_b = torch.full(
          (2, 6), float(slice_rank + 11), dtype=torch.bfloat16, device="tpu"
      )
      recv_reply = torch.empty((4, 8), dtype=torch.bfloat16, device="tpu")
      w_send_a = full_slice_pg.send([send_a], peer, 10)
      w_send_b = full_slice_pg.send([send_b], peer, 11)
      w_recv = full_slice_pg.recv([recv_reply], peer, 20)
      w_send_a.wait()
      w_send_b.wait()
      w_recv.wait()
      utils.assert_close(
          recv_reply.cpu(),
          torch.full((4, 8), float(peer + 100), dtype=torch.bfloat16),
      )
    else:
      peer = slice_rank - _JOB_SIZE
      recv_a = torch.empty((4, 8), dtype=torch.bfloat16, device="tpu")
      recv_b = torch.empty((2, 6), dtype=torch.bfloat16, device="tpu")
      send_reply = torch.full(
          (4, 8), float(slice_rank + 100), dtype=torch.bfloat16, device="tpu"
      )
      w_recv_a = full_slice_pg.recv([recv_a], peer, 10)
      w_recv_b = full_slice_pg.recv([recv_b], peer, 11)
      w_send = full_slice_pg.send([send_reply], peer, 20)
      w_recv_a.wait()
      w_recv_b.wait()
      w_send.wait()
      utils.assert_close(
          recv_a.cpu(),
          torch.full((4, 8), float(peer + 1), dtype=torch.bfloat16),
      )
      utils.assert_close(
          recv_b.cpu(),
          torch.full((2, 6), float(peer + 11), dtype=torch.bfloat16),
      )
  finally:
    if uneven_pg is not None:
      del uneven_pg
    dist.destroy_process_group()
    del full_slice_pg
    slice_store.set(f"worker_done:{slice_rank}", "1")
    if slice_rank == 0:
      slice_store.wait(
          [f"worker_done:{r}" for r in range(_SLICE_SIZE)], _TIMEOUT
      )


class JobLocalWorldTest(seed_test_utils.MultiProcessRepeatableTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    device_count = torch.accelerator.device_count()
    if device_count < _SLICE_SIZE:
      raise AssertionError(
          f"expected at least {_SLICE_SIZE} TPU devices, got {device_count}"
      )

  def test_two_job_local_worlds_share_slice_and_exchange_p2p(self) -> None:
    distributed_utils.set_launch_env(nproc_per_node=_SLICE_SIZE)
    slice_store_port = portpicker.pick_unused_port()
    job_ports = (portpicker.pick_unused_port(), portpicker.pick_unused_port())
    worker_supervisor.spawn_and_join(
        _worker_fn,
        args=(slice_store_port, job_ports, self.choose_seed()),
        nprocs=_SLICE_SIZE,
        deadline=time.monotonic() + _WORKER_DEADLINE_SECONDS,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
