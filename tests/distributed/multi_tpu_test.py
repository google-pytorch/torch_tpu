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
from torch_tpu._internal.distributed import tpu_distributed
from torch_tpu._internal.distributed import multiprocessing
from tests.distributed import continuous_test_utils


class MultiTpuTest(continuous_test_utils.TpuMultiProcContinuousTest):

  def test_global_device_count(self) -> None:
    # Verify that rank-related environment variables are set by launcher.
    self.assertIn("RANK", os.environ)
    self.assertIn("LOCAL_RANK", os.environ)
    self.assertIn("GROUP_RANK", os.environ)
    self.assertIn("WORLD_SIZE", os.environ)
    self.assertEqual(int(os.environ["WORLD_SIZE"]), self.world_size)
    self.assertEqual(int(os.environ["RANK"]), self.rank)

    self.assertEqual(dist.get_world_size(), self.world_size)
    self.assertEqual(dist.get_rank(), self.rank)

    global_device_ids = tpu_distributed.all_global_device_ids()
    self.assertLen(global_device_ids, self.world_size)

    this_device = torch.accelerator.current_device_index()
    # torch.accelerator.current_device_index() returns the id among all
    # addressable devices, not the global device id.
    self.assertGreaterEqual(this_device, 0)
    self.assertLess(
        this_device,
        torch.accelerator.device_count(),
        msg=(
            "Expected device id to be in range [0,"
            f" {torch.accelerator.device_count()}), got {this_device}"
        ),
    )


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
