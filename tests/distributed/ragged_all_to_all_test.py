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

"""Tests for ragged_all_to_all in a distributed environment on TPU."""

import os

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils


def run_ragged_all_to_all_test() -> None:
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  output_offsets = torch.tensor([rank] * world_size, dtype=torch.int32).tpu()

  send_sizes = torch.tensor([1] * world_size, dtype=torch.int32).tpu()
  input_offsets = torch.tensor(range(world_size), dtype=torch.int32).tpu()
  recv_sizes = torch.tensor([1] * world_size, dtype=torch.int32).tpu()

  offset = rank * world_size
  operand = torch.arange(offset, offset + world_size, dtype=torch.int32).tpu()
  output = torch.zeros(world_size, dtype=torch.int32).tpu()

  result = torch.ops.tpu.ragged_all_to_all(
      operand,
      output,
      input_offsets,
      send_sizes,
      output_offsets,
      recv_sizes,
      dist.group.WORLD.group_name,
  )

  # Simply a transpose.
  expected = torch.arange(
      rank,
      rank + world_size * world_size,
      world_size,
      dtype=torch.int32,
  )

  utils.assert_close(result.cpu(), expected)

  dist.barrier()
  dist.destroy_process_group()


class RaggedAllToAllTest(seed_test_utils.MultiProcessRepeatableTest):
  """Tests the ragged_all_to_all TPU collective operation.

  This test initializes a distributed environment and performs a
  ragged_all_to_all operation on a single group of 8 TPUs. Each TPU sends its
  rank to all other TPUs in the group.
  """

  def test_ragged_all_to_all(self):
    world_size = 8
    distributed_utils.dist_run(
        nproc_per_node=world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(run_ragged_all_to_all_test),
    )


if __name__ == "__main__":
  torch.multiprocessing.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
