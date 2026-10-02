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
from torch_tpu._internal import pallas
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests.distributed import continuous_test_utils
from tests.distributed import distributed_utils

P = jax.sharding.PartitionSpec

EXPECTED_1D_ROLL_OUTPUT_BY_WORLD_SIZE_AND_RANK = {
    4: {
        0: [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
        1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        2: [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        3: [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
    },
    8: {
        0: [57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0],
        1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        2: [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        3: [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        4: [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
        5: [33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0],
        6: [41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
        7: [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
    },
}

EXPECTED_2D_RANK_DP_ROLL_OUTPUT_BY_WORLD_SIZE_AND_RANK = {
    4: {
        0: [[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]],
        1: [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]],
        2: [[25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0]],
        3: [[17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]],
    },
    8: {
        0: [[25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0]],
        1: [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]],
        2: [[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]],
        3: [[17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]],
        4: [[57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0]],
        5: [[33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0]],
        6: [[41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0]],
        7: [[49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0]],
    },
}

# ==============================================================================
# 2. 1D Pallas Kernel with Mesh Coordinates (DeviceIdType.MESH, axis='rank')
# ==============================================================================


def jax_1d_mesh_kernel(
    global_input: jax.Array, mesh: jax.sharding.Mesh
) -> jax.Array:
  """1D kernel using DeviceIdType.MESH addressing along 'rank' axis."""

  def pallas_kernel(x_tile, y_tile, send_sem, recv_sem):
    axis_name = "rank"
    my_rank = jax.lax.axis_index(axis_name)
    world_size = jax.lax.axis_size(axis_name)
    next_rank = (my_rank + 1) % world_size

    desc = pltpu.make_async_remote_copy(
        src_ref=x_tile,
        dst_ref=y_tile,
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id={axis_name: next_rank},
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
      shard_kernel,
      in_specs=(P("rank"),),
      out_specs=P("rank"),
      mesh=mesh,
      check_vma=False,
  )
  global_output = shard_kernel(global_input)
  return global_output


def create_1d_mesh_roll_op(mesh, op_name="pallas::mesh_1d_roll"):
  partition_spec = P("rank")
  input_partition_specs = [partition_spec]

  torch_kernel = pallas.jax_op(
      op_name,
      functools.partial(jax_1d_mesh_kernel, mesh=mesh),
      mesh=mesh,
      input_partition_specs=input_partition_specs,
      output_partition_specs=[partition_spec],
  )
  return torch_kernel


def _run_1d_mesh(do_compile):
  device = torch.device("tpu")
  world_size = dist.get_world_size()
  assert world_size in (4, 8), f"Test requires 4 or 8 ranks, got {world_size}"
  rank = dist.get_rank()

  num_per_rank = 8
  mesh = pallas.get_default_pallas_mesh()
  assert isinstance(mesh, jax.sharding.Mesh)

  def generate_input(r, num_per_rank):
    return torch.tensor(
        [r * num_per_rank + i for i in range(num_per_rank)],
        dtype=torch.float32,
        device=device,
    )

  op_name = (
      "pallas::world_1d_mesh_roll_compiled"
      if do_compile
      else "pallas::world_1d_mesh_roll_eager"
  )
  roll_op = create_1d_mesh_roll_op(mesh, op_name=op_name)

  def roll_op_plus_one(x):
    return roll_op(x) + 1

  if do_compile:
    roll_op_plus_one = torch.compile(
        roll_op_plus_one, fullgraph=True, dynamic=False
    )

  x = generate_input(rank, num_per_rank)
  y = roll_op_plus_one(x)

  y_expected = torch.tensor(
      EXPECTED_1D_ROLL_OUTPUT_BY_WORLD_SIZE_AND_RANK[world_size][rank],
      dtype=torch.float32,
      device=device,
  )

  utils.assert_close(y, y_expected)


# ==============================================================================
# 3. 2D Pallas Kernel with Mesh Coordinates (DeviceIdType.MESH, axes=('rank', 'dp'))
# ==============================================================================


def jax_2d_rank_dp_mesh_kernel(
    global_input: jax.Array, mesh: jax.sharding.Mesh
) -> jax.Array:
  """2D kernel using DeviceIdType.MESH addressing along 'dp' axis in ('rank', 'dp') mesh."""

  def pallas_kernel(x_tile, y_tile, send_sem, recv_sem):
    axis_name = "dp"
    my_dp = jax.lax.axis_index(axis_name)
    dp_size = jax.lax.axis_size(axis_name)
    # Absolute 0-based coordinate on the dp axis
    next_dp = (my_dp + 1) % dp_size

    desc = pltpu.make_async_remote_copy(
        src_ref=x_tile,
        dst_ref=y_tile,
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id={axis_name: next_dp},
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
      shard_kernel,
      in_specs=(P("rank", "dp"),),
      out_specs=P("rank", "dp"),
      mesh=mesh,
      check_vma=False,
  )
  global_output = shard_kernel(global_input)
  return global_output


def create_2d_rank_dp_mesh_roll_op(
    mesh, op_name="pallas::mesh_2d_rank_dp_roll"
):
  partition_spec = P("rank", "dp")
  input_partition_specs = [partition_spec]

  torch_kernel = pallas.jax_op(
      op_name,
      functools.partial(jax_2d_rank_dp_mesh_kernel, mesh=mesh),
      mesh=mesh,
      input_partition_specs=input_partition_specs,
      output_partition_specs=[partition_spec],
  )

  return torch_kernel


def _run_2d_rank_dp_mesh(do_compile):
  device = torch.device("tpu")
  world_size = dist.get_world_size()
  assert world_size in (4, 8), f"Test requires 4 or 8 ranks, got {world_size}"
  rank = dist.get_rank()

  num_per_rank = 8

  # Create 2D mesh: ('rank', 'dp') with shape (2, world_size // 2).
  mesh = torch.tpu.pallas.get_pallas_mesh(
      axis_names=("rank", "dp"), mesh_shape=(2, world_size // 2)
  )
  assert isinstance(mesh, jax.sharding.Mesh)

  def generate_input(r, num_per_rank):
    return torch.tensor(
        [r * num_per_rank + i for i in range(num_per_rank)],
        dtype=torch.float32,
        device=device,
    )

  op_name = (
      "pallas::mesh_2d_rank_dp_roll_compiled"
      if do_compile
      else "pallas::mesh_2d_rank_dp_roll_eager"
  )
  roll_op = create_2d_rank_dp_mesh_roll_op(mesh, op_name=op_name)

  def roll_op_plus_one(x):
    return roll_op(x) + 1

  if do_compile:
    roll_op_plus_one = torch.compile(
        roll_op_plus_one, fullgraph=True, dynamic=False
    )

  # Each rank holds a slice of shape (1, num_per_rank) partitioned along ('rank', 'dp').
  x = generate_input(rank, num_per_rank).unsqueeze(0)
  y = roll_op_plus_one(x)

  y_expected = torch.tensor(
      EXPECTED_2D_RANK_DP_ROLL_OUTPUT_BY_WORLD_SIZE_AND_RANK[world_size][rank],
      dtype=torch.float32,
      device=device,
  )

  logging.info("rank = %d, x = %s", rank, x.cpu())
  logging.info("rank = %d, y = %s", rank, y.cpu())
  logging.info("rank = %d, y_expected = %s", rank, y_expected.cpu())

  utils.assert_close(y, y_expected)


# ==============================================================================
# 4. Combined PyTorch Collective + 1D Logical Pallas Kernel
# ==============================================================================


def _run_combined_collective_and_pallas(do_compile: bool = True):
  device = torch.device("tpu")
  world_size = dist.get_world_size()
  assert world_size in (4, 8), f"Test requires 4 or 8 ranks, got {world_size}"
  rank = dist.get_rank()

  num_per_rank = 8

  # Create rank-aligned Mesh directly from the default world group.
  mesh = pallas.get_default_pallas_mesh()
  assert isinstance(mesh, jax.sharding.Mesh)

  def generate_input(r, num_per_rank):
    return torch.tensor(
        [r * num_per_rank + i for i in range(num_per_rank)],
        dtype=torch.float32,
        device=device,
    )

  op_name = (
      "pallas::collective_and_pallas_compiled"
      if do_compile
      else "pallas::collective_and_pallas_eager"
  )
  roll_op = create_1d_mesh_roll_op(mesh, op_name=op_name)

  def combined_fn(x):
    dist.all_reduce(x)
    y = roll_op(x) + 1
    return y

  if do_compile:
    combined_fn = torch.compile(combined_fn, fullgraph=True, dynamic=False)

  x = generate_input(rank, num_per_rank)
  y = combined_fn(x)

  # After all_reduce across world_size ranks, for each element i:
  # sum_r (r * num_per_rank + i) = sum(range(world_size)) * num_per_rank + world_size * i.
  # roll_op shifts within world mesh, where all ranks hold identical values after all_reduce.
  # Then + 1 is added.
  rank_sum = sum(range(world_size)) * num_per_rank
  y_expected = torch.tensor(
      [rank_sum + world_size * i + 1 for i in range(num_per_rank)],
      dtype=torch.float32,
      device=device,
  )

  logging.info("rank = %d, output = %s", rank, y.cpu())
  utils.assert_close(y, y_expected)


def _test_pallas_axis_index_rank_equivalence():
  device = torch.device("tpu")
  rank = dist.get_rank()

  mesh = pallas.get_default_pallas_mesh()

  def jax_kernel(global_input: jax.Array) -> jax.Array:
    def pallas_kernel(x_tile, y_tile):
      idx = jax.lax.axis_index("rank")
      y_tile[...] = jax.numpy.full(x_tile.shape, idx, dtype=jax.numpy.float32)

    def shard_kernel(x_local):
      bs = pl.BlockSpec(x_local.shape)
      dt = jax.ShapeDtypeStruct(x_local.shape, jax.numpy.float32)
      pallas_fn = pl.pallas_call(
          pallas_kernel,
          out_shape=dt,
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              in_specs=[bs],
              out_specs=bs,
          ),
          compiler_params=pltpu.CompilerParams(
              vmem_limit_bytes=100 * 1024 * 1024,
          ),
      )
      return pallas_fn(x_local)

    shard_kernel_fn = jax.shard_map(
        shard_kernel,
        in_specs=(P("rank"),),
        out_specs=P("rank"),
        mesh=mesh,
        check_vma=False,
    )
    return shard_kernel_fn(global_input)

  op = pallas.jax_op(
      "test_pallas_comm::check_axis_index",
      jax_kernel,
      mesh=mesh,
      input_partition_specs=[P("rank")],
      output_partition_specs=[P("rank")],
  )

  x = torch.zeros(8, dtype=torch.float32, device=device)
  y = op(x)
  utils.assert_close(y, torch.full_like(x, rank))


class TestPallasCommunicationKernels(
    continuous_test_utils.TpuMultiProcContinuousTest
):

  @property
  def world_size(self) -> int:
    return distributed_utils.get_tpu_device_count()

  def test_1d_mesh_kernel_communication_eager(self):
    _run_1d_mesh(False)

  def test_1d_mesh_kernel_communication_compiled(self):
    _run_1d_mesh(True)

  def test_2d_rank_dp_mesh_kernel_communication_eager(self):
    _run_2d_rank_dp_mesh(False)

  def test_2d_rank_dp_mesh_kernel_communication_compiled(self):
    _run_2d_rank_dp_mesh(True)

  def test_combined_collective_and_pallas_eager(self):
    _run_combined_collective_and_pallas(False)

  def test_combined_collective_and_pallas_compiled(self):
    _run_combined_collective_and_pallas(True)

  def test_pallas_axis_index_rank_equivalence(self):
    _test_pallas_axis_index_rank_equivalence()


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
