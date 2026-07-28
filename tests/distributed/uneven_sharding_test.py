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

import os

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import distributed as dist
import torch.distributed.tensor as dt
import torch.multiprocessing as mp
from torch_tpu._internal import compile as tt_compile
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


def _replicate_tensor(
    tensor_size: int,
    compiled: bool,
    ndims: int = 1,
    pass_shape: bool = False,
    trailing_dim_size: int = 4,
) -> None:
  """Redistributes a tensor from shard to replicate (AllGather)."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  device_mesh = dt.init_device_mesh("tpu", (world_size,))

  trailing_dims = [trailing_dim_size] * (ndims - 1)

  def shard_and_redistribute(x):
    if pass_shape:
      global_shape = torch.Size([tensor_size] + trailing_dims)
      global_stride = torch.empty(global_shape, device="meta").stride()
      dtensor = dt.DTensor.from_local(
          x,
          device_mesh=device_mesh,
          placements=[dt.Shard(0)],
          shape=global_shape,
          stride=global_stride,
      )
    else:
      dtensor = dt.DTensor.from_local(x, device_mesh, [dt.Shard(0)])
    return dtensor.redistribute(device_mesh, [dt.Replicate()]).to_local()

  chunk_size = (tensor_size + world_size - 1) // world_size
  start = min(rank * chunk_size, tensor_size)
  end = min((rank + 1) * chunk_size, tensor_size)
  local_size = end - start
  local_shape = [local_size] + trailing_dims
  local_tensor = torch.ones(*local_shape, dtype=torch.float32, device="tpu")

  if compiled:
    backend = tt_compile.TpuBackend(debug=True)
    func = torch.compile(shard_and_redistribute, backend=backend)
  else:
    func = shard_and_redistribute

  output = func(local_tensor)
  out = output.cpu()

  if pass_shape or rank == 0:
    expected_shape = [tensor_size] + trailing_dims
    expected = torch.ones(*expected_shape, dtype=torch.float32)
    utils.assert_close(out, expected)


def _reduce_scatter_tensor(
    tensor_size: int,
    compiled: bool,
    ndims: int = 2,
    pass_shape: bool = False,
    trailing_dim_size: int = 4,
) -> None:
  """Redistributes a tensor from replicate to shard (ReduceScatter)."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  device_mesh = dt.init_device_mesh("tpu", (world_size,))

  trailing_dims = [trailing_dim_size] * (ndims - 1)
  global_shape = torch.Size([tensor_size] + trailing_dims)

  def replicate_and_reduce_scatter(x):
    if pass_shape:
      global_stride = torch.empty(global_shape, device="meta").stride()
      dtensor = dt.DTensor.from_local(
          x,
          device_mesh=device_mesh,
          placements=[dt.Replicate()],
          shape=global_shape,
          stride=global_stride,
      )
    else:
      dtensor = dt.DTensor.from_local(x, device_mesh, [dt.Replicate()])
    return dtensor.redistribute(device_mesh, [dt.Shard(0)]).to_local()

  local_tensor = torch.ones(*global_shape, dtype=torch.float32, device="tpu")

  if compiled:
    backend = tt_compile.TpuBackend(debug=True)
    func = torch.compile(replicate_and_reduce_scatter, backend=backend)
  else:
    func = replicate_and_reduce_scatter

  output = func(local_tensor)
  out = output.cpu()

  chunk_size = (tensor_size + world_size - 1) // world_size
  start = min(rank * chunk_size, tensor_size)
  end = min((rank + 1) * chunk_size, tensor_size)
  local_size = end - start

  if pass_shape or rank == 0:
    expected_shape = [local_size] + trailing_dims
    expected = torch.ones(*expected_shape, dtype=torch.float32)
    utils.assert_close(out, expected)


def _alltoall_tensor(
    tensor_size: int,
    compiled: bool,
    pass_shape: bool = False,
    trailing_dim_size: int = 4,
) -> None:
  """Redistributes a tensor from Shard(0) to Shard(1) (AllToAll)."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  device_mesh = dt.init_device_mesh("tpu", (world_size,))

  trailing_dims = [trailing_dim_size]
  global_shape = torch.Size([tensor_size] + trailing_dims)

  def shard_and_alltoall(x):
    if pass_shape:
      global_stride = torch.empty(global_shape, device="meta").stride()
      dtensor = dt.DTensor.from_local(
          x,
          device_mesh=device_mesh,
          placements=[dt.Shard(0)],
          shape=global_shape,
          stride=global_stride,
      )
    else:
      dtensor = dt.DTensor.from_local(x, device_mesh, [dt.Shard(0)])
    return dtensor.redistribute(device_mesh, [dt.Shard(1)]).to_local()

  chunk_size = (tensor_size + world_size - 1) // world_size
  start = min(rank * chunk_size, tensor_size)
  end = min((rank + 1) * chunk_size, tensor_size)
  local_size = end - start
  local_shape = [local_size] + trailing_dims
  local_tensor = torch.ones(*local_shape, dtype=torch.float32, device="tpu")

  if compiled:
    backend = tt_compile.TpuBackend(debug=True)
    func = torch.compile(shard_and_alltoall, backend=backend)
  else:
    func = shard_and_alltoall

  output = func(local_tensor)
  _ = output.cpu()


class UnevenShardingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("1d_eager_small", 1, False, 4),
      ("1d_compiled_small", 1, True, 4),
      ("1d_eager_uneven", 1, False, 12),
      ("1d_compiled_uneven", 1, True, 12),
      ("2d_eager_small", 2, False, 4),
      ("2d_compiled_small", 2, True, 4),
      ("2d_eager_uneven", 2, False, 12),
      ("2d_compiled_uneven", 2, True, 12),
  )
  def test_replicate_tensor_with_shape(self, ndims, compiled, tensor_size):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(_replicate_tensor, world_size=8),
        tensor_size=tensor_size,
        compiled=compiled,
        ndims=ndims,
        pass_shape=True,
    )

  @parameterized.named_parameters(
      (
          "1d_compiled_fail_small",
          1,
          True,
          4,
          (
              "(Tensor shape mismatch|Wait timeout on"
              " sflag|RuntimeUnexpectedCoreHalt)"
          ),
      ),
      (
          "1d_eager_fail_small",
          1,
          False,
          4,
          (
              "(Tensor shape mismatch|Wait timeout on"
              " sflag|RuntimeUnexpectedCoreHalt)"
          ),
      ),
      (
          "1d_compiled_fail_uneven",
          1,
          True,
          12,
          (
              "(Tensor shape mismatch|Wait timeout on"
              " sflag|RuntimeUnexpectedCoreHalt)"
          ),
      ),
      (
          "1d_eager_fail_uneven",
          1,
          False,
          12,
          (
              "(Tensor shape mismatch|Wait timeout on"
              " sflag|RuntimeUnexpectedCoreHalt)"
          ),
      ),
      (
          "2d_compiled_fail_small",
          2,
          True,
          4,
          (
              "(Tensor shape mismatch|Wait timeout on"
              " sflag|RuntimeUnexpectedCoreHalt)"
          ),
      ),
      (
          "2d_eager_fail_small",
          2,
          False,
          4,
          (
              "(Tensor shape mismatch|Wait timeout on"
              " sflag|RuntimeUnexpectedCoreHalt)"
          ),
      ),
      (
          "2d_compiled_fail_uneven",
          2,
          True,
          12,
          (
              "(Tensor shape mismatch|Wait timeout on"
              " sflag|RuntimeUnexpectedCoreHalt)"
          ),
      ),
      (
          "2d_eager_fail_uneven",
          2,
          False,
          12,
          (
              "(Tensor shape mismatch|Wait timeout on"
              " sflag|RuntimeUnexpectedCoreHalt)"
          ),
      ),
  )
  def test_replicate_tensor_without_shape(
      self, ndims, compiled, tensor_size, expected_regex
  ):
    with self.assertRaisesRegex(Exception, expected_regex):
      distributed_utils.dist_run(
          nproc_per_node=8,
          fn=singlehost_wrapper.tpu_env_wrapper(
              _replicate_tensor, world_size=8
          ),
          tensor_size=tensor_size,
          compiled=compiled,
          ndims=ndims,
          pass_shape=False,
      )

  @parameterized.named_parameters(
      ("2d_eager_reduce_scatter", False, 4),
      ("2d_compiled_reduce_scatter", True, 4),
  )
  def test_reduce_scatter_tensor_with_shape(self, compiled, tensor_size):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _reduce_scatter_tensor, world_size=8
        ),
        tensor_size=tensor_size,
        compiled=compiled,
        pass_shape=True,
    )

  @parameterized.named_parameters(
      ("2d_eager_alltoall", False, 4),
      ("2d_compiled_alltoall", True, 4),
  )
  def test_alltoall_tensor_with_shape(self, compiled, tensor_size):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(_alltoall_tensor, world_size=8),
        tensor_size=tensor_size,
        compiled=compiled,
        pass_shape=True,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
