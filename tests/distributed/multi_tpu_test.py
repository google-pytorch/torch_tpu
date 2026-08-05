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

"""Tests creation and properties of multi-tpu process group."""

import os

from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed import tpu_distributed
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils


def run_global_device_count() -> None:
  # Verify that rank-related environment variables are set by torchrun or
  # launcher.
  assert "RANK" in os.environ
  assert "LOCAL_RANK" in os.environ
  assert "GROUP_RANK" in os.environ
  assert "WORLD_SIZE" in os.environ

  world_size = int(os.environ["WORLD_SIZE"])
  assert world_size == 8

  dist.init_process_group(backend="tpu_dist")

  global_device_ids = tpu_distributed.all_global_device_ids()

  assert len(global_device_ids) == world_size
  this_device = torch.tpu.current_device()
  # torch.tpu.current_device() returns the id among all addressable devices,
  # not the global device id.
  assert 0 <= this_device < torch.tpu.device_count(), (
      f"Expected device id to be in range [0, {torch.tpu.device_count()}), got"
      f" {this_device}"
  )


class MultiTpuTest(seed_test_utils.MultiProcessRepeatableTest):

  def test_global_device_count(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_global_device_count, world_size=8
        ),
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
