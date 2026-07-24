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

"""Small graph test for TPU backend."""

import functools

from absl import logging
from absl.testing import absltest
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal import pallas
from torch_tpu._internal.distributed import tpu_distributed
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


P = jax.sharding.PartitionSpec


def jax_kernel(global_input: jax.Array, mesh: jax.sharding.Mesh) -> jax.Array:
  def pallas_kernel(x_tile, y_tile, send_sem, recv_sem):
    rank = jax.lax.axis_index("x")
    world_size = jax.lax.axis_size("x")
    destination_id = (rank + 1) % world_size

    desc = pltpu.make_async_remote_copy(
        src_ref=x_tile,
        dst_ref=y_tile,
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id=(destination_id,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    desc.start()
    desc.wait()

  def shard_kernel(x_local):
    bs = pl.BlockSpec(x_local.shape)
    dt = jax.ShapeDtypeStruct(x_local.shape, x_local.dtype)
    pallas_fn = pl.pallas_call(
        pallas_kernel,
        out_shape=dt,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[bs],
            out_specs=bs,
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=100 * 1024 * 1024,
        ),
    )
    return pallas_fn(x_local)

  shard_kernel = jax.shard_map(
      shard_kernel, out_specs=P("x"), mesh=mesh, check_vma=False
  )
  global_output = shard_kernel(global_input)
  return global_output


def create_roll_op(world_size, rank):
  del rank

  mesh = jax.make_mesh([world_size], ("x",))
  partition_spec = P("x")

  input_partition_specs = [partition_spec]

  torch_kernel = pallas.jax_op(
      "pallas::roll_vectors",
      functools.partial(jax_kernel, mesh=mesh),
      mesh=mesh,
      input_partition_specs=input_partition_specs,
  )

  return torch_kernel


def _run(do_compile):
  device = torch.device("tpu")
  dist.init_process_group(backend="tpu_dist")
  world_size = dist.get_world_size()
  rank = dist.get_rank()

  num_per_device = 8

  num_devices = len(tpu_distributed.all_global_device_ids())
  device_id = tpu_distributed.global_device_id()

  def generate_input(device_id, num_per_device):
    return torch.tensor(
        [device_id * num_per_device + i for i in range(num_per_device)],
        dtype=torch.float32,
        device=device,
    )

  roll_op = create_roll_op(world_size, rank)

  # Add a non-kernel local operation to ensure that it is correctly stitched
  # with the wider graph.
  def roll_op_plus_one(x):
    return roll_op(x) + 1

  if do_compile:
    roll_op_plus_one = torch.compile(
        roll_op_plus_one, fullgraph=True, dynamic=False
    )

  x = generate_input(device_id, num_per_device)
  y = roll_op_plus_one(x)

  src_device_id = (device_id + num_devices - 1) % num_devices
  y_expected = generate_input(src_device_id, num_per_device) + 1

  logging.info("device_id = %d, src_device_id = %d", device_id, src_device_id)
  logging.info("device_id = %d, x = %s", device_id, x.cpu())
  logging.info("device_id = %d, y = %s", device_id, y.cpu())
  logging.info("device_id = %d, y_expected = %s", device_id, y_expected.cpu())

  utils.assert_close(y, y_expected)


class TestPallasCommunicationKernels(absltest.TestCase):
  _world_size = 8

  def test_kernel_communication(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            functools.partial(_run, False), world_size=self._world_size
        ),
    )

  def test_kernel_communication_compiled(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            functools.partial(_run, True), world_size=self._world_size
        ),
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
