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

from unittest import mock

from absl.testing import absltest
import torch
from torch_tpu import _loader
from torch_tpu._internal.distributed import device_mesh
from tests import seed_test_utils

_loader._init_device("tpu")


class DeviceMeshTest(seed_test_utils.MultiProcessRepeatableTest):

  def test_pre_initialization_errors(self):
    # If the process group is not initialized, topology_aware_mesh should raise a RuntimeError.
    with mock.patch("torch.distributed.is_initialized", return_value=False):
      with self.assertRaisesRegex(
          RuntimeError, "process group must be initialized"
      ):
        torch.tpu.topology_aware_mesh(
            mesh_shape=(8, 8),
            topology="single_slice",
        )

  def test_v6e_tiled(self):
    # An 8x8 physical grid where IDs (from 0 to 63) are assigned in 2x2 blocks
    # but the assignment is randomized to ensure logical coordinates map to
    # contiguous physical mesh elements regardless of ID ordering.
    grid = [
        [28, 29, 36, 37, 20, 21, 24, 25],
        [30, 31, 38, 39, 22, 23, 26, 27],
        [56, 57, 40, 41, 48, 49, 32, 33],
        [58, 59, 42, 43, 50, 51, 34, 35],
        [4, 5, 8, 9, 52, 53, 60, 61],
        [6, 7, 10, 11, 54, 55, 62, 63],
        [16, 17, 44, 45, 0, 1, 12, 13],
        [18, 19, 46, 47, 2, 3, 14, 15],
    ]

    # Create a list of _DeviceStub objects, one for each device ID in the grid
    # with coordinates corresponding to their position in the grid.
    infos = [None] * 64
    for y in range(8):
      for x in range(8):
        device_id = grid[y][x]
        infos[device_id] = device_mesh._DeviceStub(
            id=device_id,
            coords=(x, y, 0),
            core_on_chip=0,
            device_kind="TPU v6e",
            platform="tpu",
            process_index=device_id,
            slice_index=0,
        )

    # Call topology_aware_mesh with mocked torch.dist.
    with mock.patch("torch.distributed.is_initialized", return_value=True):
      with mock.patch("torch.distributed.get_world_size", return_value=64):
        with mock.patch(
            "torch_tpu._internal.distributed.device_mesh._gather_global_device_info",
            return_value=infos,
        ):
          rank_mesh = torch.tpu.topology_aware_mesh(
              mesh_shape=(8, 8),
              topology="single_slice",
          )

    # Verify that the rank mesh has the correct shape.
    self.assertEqual(rank_mesh.shape, (8, 8))

    # Verify that logically adjacent elements in rank_mesh correspond
    # to physically adjacent devices in the mesh (coords difference is 1,
    # or wrap-around if it's a torus jump of length N which distance is N - 1).
    dim_x, dim_y = rank_mesh.shape

    def get_physical_dist(rank1, rank2):
      return sum(
          abs(c1 - c2)
          for c1, c2 in zip(infos[rank1].coords, infos[rank2].coords)
      )

    # Check horizontally adjacent logical ranks.
    for y in range(dim_y):
      for x in range(dim_x - 1):
        r1 = rank_mesh[x, y].item()
        r2 = rank_mesh[x + 1, y].item()
        self.assertIn(
            get_physical_dist(r1, r2),
            [1, dim_x - 1],
            f"Horizontal neighbors {r1} and {r2} are not physically adjacent",
        )

    # Check vertically adjacent logical ranks.
    for x in range(dim_x):
      for y in range(dim_y - 1):
        r1 = rank_mesh[x, y].item()
        r2 = rank_mesh[x, y + 1].item()
        self.assertIn(
            get_physical_dist(r1, r2),
            [1, dim_y - 1],
            f"Vertical neighbors {r1} and {r2} are not physically adjacent",
        )


if __name__ == "__main__":
  absltest.main()
