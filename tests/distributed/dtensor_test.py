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

"""Tests DTensor APIs."""

import os

from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.distributed.tensor as dt
import torch.multiprocessing as mp
from torch_tpu._internal import sync
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


def run_dtensor_1d_apply_op() -> None:
  """Operations can be applied to a DTensor."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  shard = torch.randn((1, world_size))
  device_mesh = dt.init_device_mesh("tpu", (world_size,))
  placements = [dt.Shard(0)]

  dtensor = dt.DTensor.from_local(
      shard,
      device_mesh=device_mesh,
      placements=placements,
  )
  dtensor = rank * dtensor

  updated_shard = dtensor.to_local().to("cpu")
  updated_tensor = dtensor.full_tensor().to("cpu")
  utils.assert_close(updated_shard, rank * shard)
  utils.assert_close(updated_tensor[rank, :], rank * shard.squeeze())


def run_dtensor_1d_distribute_from_src() -> None:
  """DTensor can be initialized with data coming from only one process."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  device_mesh = dt.init_device_mesh("tpu", (world_size,))
  if rank == 0:
    full_tensor = torch.ones(world_size, dtype=torch.float32)
  else:
    full_tensor = torch.zeros(world_size, dtype=torch.float32)

  dtensor = dt.distribute_tensor(
      full_tensor,
      device_mesh=device_mesh,
      placements=[dt.Shard(0)],
      src_data_rank=0,
  )
  full_tensor = dtensor.full_tensor().to("cpu")
  expected = torch.ones(world_size, dtype=torch.float32)
  utils.assert_close(full_tensor, expected)


def _range_tensor_2d(n: int) -> torch.Tensor:
  """A simple 2 dimensional tensor to use for testing."""
  return torch.arange(n * n, dtype=torch.float32).reshape(n, n)


def _test_redistribute(
    dtensor: dt.DTensor, new_placements: list[dt.Placement]
) -> None:
  """Exercises DTensor.redistribute() and checks the results.

  Ground truth is determined by gathering the full tensor and calling
  distribute_tensor() with the new placements.

  Args:
    dtensor: The DTensor to be redistributed.
    new_placements: The new placements for the redistribution.
  """
  device_mesh = dtensor.device_mesh
  redistributed = dtensor.redistribute(device_mesh, new_placements)
  redistributed_local = redistributed.to_local().to("cpu")

  full_tensor = dtensor.full_tensor()
  expected_local = (
      dt.distribute_tensor(
          full_tensor, device_mesh, new_placements, src_data_rank=None
      )
      .to_local()
      .to("cpu")
  )
  utils.assert_close(redistributed_local, expected_local)


def run_dtensor_1d_shard_to_replicate() -> None:
  """Redistributing from shard to replicate triggers all_gather."""
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  device_mesh = dt.init_device_mesh("tpu", (world_size,))
  dtensor = dt.distribute_tensor(
      _range_tensor_2d(world_size),
      device_mesh=device_mesh,
      placements=[dt.Shard(0)],
      src_data_rank=None,
  )
  _test_redistribute(dtensor, [dt.Replicate()])


def run_dtensor_1d_replicate_to_shard() -> None:
  """Redistributing from replicate to shard triggers local chunking."""
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  device_mesh = dt.init_device_mesh("tpu", (world_size,))
  dtensor = dt.distribute_tensor(
      _range_tensor_2d(world_size),
      device_mesh=device_mesh,
      placements=[dt.Replicate()],
      src_data_rank=None,
  )
  _test_redistribute(dtensor, [dt.Shard(0)])


def run_dtensor_1d_shard_to_shard() -> None:
  """Redistributing from shard to shard triggers all_to_all."""
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  device_mesh = dt.init_device_mesh("tpu", (world_size,))
  dtensor = dt.distribute_tensor(
      _range_tensor_2d(world_size),
      device_mesh=device_mesh,
      placements=[dt.Shard(0)],
      src_data_rank=None,
  )
  _test_redistribute(dtensor, [dt.Shard(1)])


def run_dtensor_1d_partial_to_replicate() -> None:
  """Redistributing from partial to replicate triggers all_reduce."""
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  device_mesh = dt.init_device_mesh("tpu", (world_size,))
  dtensor = dt.DTensor.from_local(
      _range_tensor_2d(world_size).to("tpu"),
      device_mesh=device_mesh,
      placements=[dt.Partial("sum")],
  )
  _test_redistribute(dtensor, [dt.Replicate()])


def run_dtensor_1d_partial_to_shard() -> None:
  """Redistributing from partial to shard triggers reduce_scatter."""
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  device_mesh = dt.init_device_mesh("tpu", (world_size,))
  dtensor = dt.DTensor.from_local(
      _range_tensor_2d(world_size).to("tpu"),
      device_mesh=device_mesh,
      placements=[dt.Partial("sum")],
  )
  _test_redistribute(dtensor, [dt.Shard(0)])


class DTensor1DimTest(absltest.TestCase):

  def test_apply_op(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_dtensor_1d_apply_op, world_size=8
        ),
    )

  def test_dtensor_1d_distribute_from_src(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_dtensor_1d_distribute_from_src, world_size=8
        ),
    )

  def test_shard_to_replicate(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_dtensor_1d_shard_to_replicate, world_size=8
        ),
    )

  def test_replicate_to_shard(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_dtensor_1d_replicate_to_shard, world_size=8
        ),
    )

  def test_shard_to_shard(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_dtensor_1d_shard_to_shard, world_size=8
        ),
    )

  def test_partial_to_replicate(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_dtensor_1d_partial_to_replicate, world_size=8
        ),
    )

  def test_partial_to_shard(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_dtensor_1d_partial_to_shard, world_size=8
        ),
    )


def run_dtensor_2d_shard_to_shard() -> None:
  """Redistributing from shard to shard triggers all_to_all."""
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  full_tensor = _range_tensor_2d(world_size)
  device_mesh = dt.init_device_mesh("tpu", (world_size // 2, 2))
  dtensor = dt.distribute_tensor(
      full_tensor,
      device_mesh=device_mesh,
      placements=[dt.Shard(0), dt.Shard(1)],
      src_data_rank=None,
  )
  _test_redistribute(dtensor, [dt.Shard(1), dt.Shard(0)])


class DTensor2DimTest(absltest.TestCase):

  def test_shard_to_shard(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_dtensor_2d_shard_to_shard, world_size=8
        ),
    )


def run_sync_dtensor() -> None:
  """Syncing a DTensor materializes the underlying shard."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  device_mesh = dt.init_device_mesh("tpu", (world_size,))
  shard_cpu = torch.full((world_size,), fill_value=rank, device="cpu")
  shard = shard_cpu.to("tpu")
  dtensor = dt.DTensor.from_local(shard, device_mesh, [dt.Shard(0)])
  dtensor = dtensor + 1

  sync.synchronize(dtensor, wait=True)
  assert sync.is_materialized(dtensor), "DTensor not materialized after sync"
  assert sync.is_ready(dtensor), "Dtensor materialized but not ready"

  updated_shard = dtensor.to_local().to("cpu")
  utils.assert_close(updated_shard, shard_cpu + 1)


class DTensorSyncTest(absltest.TestCase):

  def test_sync_dtensor(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(run_sync_dtensor, world_size=8),
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
