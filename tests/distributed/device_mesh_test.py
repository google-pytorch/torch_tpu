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

from absl.testing import absltest
from torch_tpu import _loader
from torch_tpu._internal.distributed import device_mesh
from tests import seed_test_utils
from tests.distributed import device_mesh_test_utils


class DeviceMeshTest(seed_test_utils.MultiProcessRepeatableTest):

  def setUp(self):
    super().setUp()
    _loader._init_device("tpu")  # INIT_DEVICE_OK=CPU runner

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

    rank_mesh = device_mesh._rank_mesh_from_infos(infos, mesh_shape=(8, 8))

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

  def test_multi_slice_groups_rows_by_slice(self):
    infos = device_mesh_test_utils.v7x_multislice_infos(num_slices=2)

    rank_mesh = device_mesh._rank_mesh_from_infos(
        infos,
        mesh_shape=(2, 8),
        dcn_mesh_shape=(2, 1),
    )

    self.assertEqual(tuple(rank_mesh.shape), (2, 8))

    # Each row must be exactly one slice, so the DCN hop lands on axis 0.
    for row in range(2):
      slices_in_row = {
          infos[rank_mesh[row, col].item()].slice_index for col in range(8)
      }
      self.assertEqual(slices_in_row, {row})

    # Every rank appears exactly once.
    self.assertCountEqual(rank_mesh.flatten().tolist(), range(len(infos)))

  def test_multi_slice_infers_dcn_mesh_shape(self):
    infos = device_mesh_test_utils.v7x_multislice_infos(num_slices=2)

    inferred = device_mesh._rank_mesh_from_infos(infos, mesh_shape=(2, 8))
    explicit = device_mesh._rank_mesh_from_infos(
        infos,
        mesh_shape=(2, 8),
        dcn_mesh_shape=(2, 1),
    )

    # Axis 0 is the outermost axis divisible by the slice count, so inference
    # has to reproduce dcn_mesh_shape=(2, 1).
    self.assertEqual(inferred.tolist(), explicit.tolist())

  def test_multi_slice_keeps_cores_of_a_chip_adjacent(self):
    infos = device_mesh_test_utils.v7x_multislice_infos(num_slices=2)

    rank_mesh = device_mesh._rank_mesh_from_infos(
        infos,
        mesh_shape=(2, 8),
        dcn_mesh_shape=(2, 1),
    )

    # The core axis has the highest bandwidth, so the two cores of a chip
    # should land on adjacent positions of the innermost axis.
    for row in range(2):
      for col in range(0, 8, 2):
        first = infos[rank_mesh[row, col].item()]
        second = infos[rank_mesh[row, col + 1].item()]
        self.assertEqual(
            first.coords,
            second.coords,
            f"positions {col},{col+1} of row {row} are not the same chip",
        )

  def test_single_slice_accepts_all_ones_dcn_mesh_shape(self):
    # On one slice no DCN split is needed, so the all-ones shape is
    # satisfiable and has to keep behaving exactly like omitting it.
    infos = device_mesh_test_utils.v7x_multislice_infos(num_slices=1)

    explicit = device_mesh._rank_mesh_from_infos(
        infos,
        mesh_shape=(1, 8),
        dcn_mesh_shape=(1, 1),
    )
    omitted = device_mesh._rank_mesh_from_infos(infos, mesh_shape=(1, 8))

    self.assertEqual(explicit.tolist(), omitted.tolist())

  def test_infer_dcn_mesh_shape_picks_outermost_divisible_axis(self):
    # TorchTitan orders its axes (pp, dp_replicate, dp_shard, cp, tp), so the
    # split lands on dp_replicate and every shard group stays in one slice.
    self.assertEqual(
        device_mesh._infer_dcn_mesh_shape((1, 4, 32, 1, 1), 4),
        (1, 4, 1, 1, 1),
    )
    # Pure FSDP only has one axis to give.
    self.assertEqual(device_mesh._infer_dcn_mesh_shape((128,), 4), (4,))


if __name__ == "__main__":
  absltest.main()
