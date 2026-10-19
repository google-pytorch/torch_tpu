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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import distributed as dist
import torch.distributed.tensor as dt
import torch.multiprocessing as mp
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils


def _replicate_tensor(
    tensor_size: int,
    compiled: bool,
    ndims: int = 1,
    trailing_dim_size: int = 4,
) -> None:
  """Redistributes a tensor from shard to replicate."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  device_mesh = dt.init_device_mesh("tpu", (world_size,))

  trailing_dims = [trailing_dim_size] * (ndims - 1)
  global_shape = torch.Size([tensor_size] + trailing_dims)
  global_stride = torch.empty(global_shape, device="meta").stride()

  def shard_and_redistribute(x):
    dtensor = dt.DTensor.from_local(
        x,
        device_mesh=device_mesh,
        placements=[dt.Shard(0)],
        shape=global_shape,
        stride=global_stride,
    )
    return dtensor.redistribute(device_mesh, [dt.Replicate()]).to_local()

  chunk_size = (tensor_size + world_size - 1) // world_size
  start = min(rank * chunk_size, tensor_size)
  end = min((rank + 1) * chunk_size, tensor_size)
  local_size = end - start
  local_shape = [local_size] + trailing_dims
  local_tensor = torch.ones(*local_shape, dtype=torch.float32, device="tpu")

  if compiled:
    func = torch.compile(
        shard_and_redistribute, backend="tpu", options={"serializable": False}
    )
  else:
    func = shard_and_redistribute

  output = func(local_tensor)

  out = output.cpu()

  expected_shape = [tensor_size] + trailing_dims
  expected = torch.ones(*expected_shape, dtype=torch.float32)
  utils.assert_close(out, expected)


class UnevenShardingTest(seed_test_utils.MultiProcessRepeatableTest):
  _world_size = distributed_utils.get_tpu_device_count()
  # Fewer rows than ranks, so the upper half of the ranks get empty shards.
  _small_size = _world_size // 2
  # Not a multiple of the world size: with ceil-sized chunks of 2 rows, the
  # last quarter of the ranks get empty shards.
  _uneven_size = _world_size * 3 // 2

  @parameterized.named_parameters(
      ("1d_eager_small", 1, False, _small_size),
      ("1d_compiled_small", 1, True, _small_size),
      ("1d_eager_uneven", 1, False, _uneven_size),
      ("1d_compiled_uneven", 1, True, _uneven_size),
      ("2d_eager_small", 2, False, _small_size),
      ("2d_compiled_small", 2, True, _small_size),
      ("2d_eager_uneven", 2, False, _uneven_size),
      ("2d_compiled_uneven", 2, True, _uneven_size),
  )
  def test_replicate_tensor_with_shape(self, ndims, compiled, tensor_size):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=_replicate_tensor,
        tensor_size=tensor_size,
        compiled=compiled,
        ndims=ndims,
    )

  @parameterized.named_parameters(
      ("1d_with_shape_stride_small", 1, _small_size),
      ("1d_with_shape_stride_uneven", 1, _uneven_size),
      ("2d_with_shape_stride_small", 2, _small_size),
      ("2d_with_shape_stride_uneven", 2, _uneven_size),
  )
  def test_replicate_tensor_debug_eager(self, ndims, tensor_size):
    with mock.patch.dict(os.environ, {"TORCH_TPU_ENABLE_DEBUG_CHECKS": "1"}):
      distributed_utils.dist_run(
          nproc_per_node=self._world_size,
          fn=_replicate_tensor,
          tensor_size=tensor_size,
          compiled=False,
          ndims=ndims,
      )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
