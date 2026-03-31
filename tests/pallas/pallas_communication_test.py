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
import jax.export
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu import api
from torch_tpu._internal import pallas
from torch_tpu._internal.distributed import tpu_distributed
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils
from tests.distributed import distributed_utils
from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

P = jax.sharding.PartitionSpec


def jax_kernel(global_input, mesh):
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
        device_id_type=pltpu.DeviceIdType.MESH,
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


def roll_torch_pallas(input_local):
  world_size = dist.get_world_size()

  mesh = jax.make_mesh([world_size], ("x",))
  partition_spec = P("x")

  input_partition_specs = [partition_spec]

  output_local = pallas.custom_jax_kernel(
      functools.partial(jax_kernel, mesh=mesh),
      mesh=mesh,
      input_partition_specs=input_partition_specs,
  )(input_local)

  return output_local


def run():
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")

  device = "tpu"
  num_per_device = 8

  num_devices = len(tpu_distributed.all_global_device_ids())
  device_id = tpu_distributed.global_device_id()

  def generate_input(device_id, num_per_device):
    return torch.tensor(
        [device_id * num_per_device + i for i in range(num_per_device)],
        dtype=torch.float32,
        device=device,
    )

  x = generate_input(device_id, num_per_device)

  src_device_id = (device_id + num_devices - 1) % num_devices
  y_expected = generate_input(src_device_id, num_per_device)

  y = roll_torch_pallas(x)

  logging.info("device_id = %d, src_device_id = %d", device_id, src_device_id)
  logging.info("device_id = %d, x = %s", device_id, x.cpu())
  logging.info("device_id = %d, y = %s", device_id, y.cpu())
  logging.info("device_id = %d, y_expected = %s", device_id, y_expected.cpu())

  utils.assert_close(y, y_expected)


# TODO(elliotenglish): Add test for compiled: cl/885571521
class TestPallasCommunicationKernels(absltest.TestCase):

  def test_kernel_communication(self):

    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(run, world_size=8),
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
