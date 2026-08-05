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

"""Tests for TPU Visualizer on 8 chips."""

import re

from absl.testing import absltest
import torch
from examples.tpu_visualizer import tpu_visualizer
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils


def _cell_string(expected_rank: int, my_rank: int) -> str:
  if expected_rank == my_rank:
    return f"*{expected_rank}*"
  return f" {expected_rank} "


class TpuVisualizerTest(seed_test_utils.MultiProcessRepeatableTest):
  """Tests TPU Visualizer tool on an 8-chip v5e slice."""

  def test_visualize_topology_on_8_chips(self):
    topology_list = tpu_visualizer.visualize_topology()
    self.assertLen(topology_list, 8)

    # Extract physical mesh rank order from rank 0's topology string.
    # This makes the test robust against varying physical TPU device numbering
    # which can be variable.
    r000, r100, r010, r110, r001, r101, r011, r111 = [
        int(m.group(1))
        for m in re.finditer(r"\[\s*\*?(\d+)\*?\s*\]", topology_list[0])
    ]

    for rank, top_str in enumerate(topology_list):
      expected_str = "\n".join([
          "=" * 60,
          f"TPU Topology Visualization (My Rank: {rank})",
          "Bounds: X:[0..1], Y:[0..1], Z:[0..1], Core:[0..0]",
          "=" * 60,
          "\n--- Z-Slice: 0 ---",
          "Y \\ X |    X=0   |    X=1   | ",
          "------------------------------",
          (
              f" Y=0 | [{_cell_string(r000, rank)}] |"
              f" [{_cell_string(r100, rank)}] | "
          ),
          (
              f" Y=1 | [{_cell_string(r010, rank)}] |"
              f" [{_cell_string(r110, rank)}] | "
          ),
          "\n--- Z-Slice: 1 ---",
          "Y \\ X |    X=0   |    X=1   | ",
          "------------------------------",
          (
              f" Y=0 | [{_cell_string(r001, rank)}] |"
              f" [{_cell_string(r101, rank)}] | "
          ),
          (
              f" Y=1 | [{_cell_string(r011, rank)}] |"
              f" [{_cell_string(r111, rank)}] | "
          ),
          "=" * 60,
      ])
      self.assertEqual(top_str, expected_str)


if __name__ == "__main__":
  torch.multiprocessing.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
