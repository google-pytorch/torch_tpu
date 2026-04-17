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

import os
from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils
from tests.distributed import distributed_utils
from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


def run_scatter() -> None:
  """Tests scatter functionality."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  src_rank = 0
  size = 5
  inputs = None
  if rank == src_rank:
    inputs = [
        torch.full((size,), i, device="tpu", dtype=torch.float32)
        for i in range(world_size)
    ]
  output = torch.empty(size, device="tpu", dtype=torch.float32)

  torch.distributed.scatter(output, inputs, src=src_rank)
  expected = torch.full((size,), fill_value=rank, dtype=torch.float32)
  utils.assert_close(output.cpu(), expected)


def run_scatter_scalar() -> None:
  """Tests scatter on scalar input."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  src_rank = 0
  inputs = None
  if rank == src_rank:
    inputs = [
        torch.tensor(i, device="tpu", dtype=torch.float32)
        for i in range(world_size)
    ]
  output = torch.tensor(-1, device="tpu", dtype=torch.float32)

  torch.distributed.scatter(output, inputs, src=src_rank)
  expected = torch.tensor(rank, dtype=torch.float32)
  utils.assert_close(output.cpu(), expected)


class ScatterTest(absltest.TestCase):

  def test_scatter_tensor(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(run_scatter),
    )

  def test_scatter_scalar(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(run_scatter_scalar),
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
