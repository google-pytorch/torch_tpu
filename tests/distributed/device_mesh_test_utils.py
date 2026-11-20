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

"""Shared fixtures for the topology-aware device mesh tests.

`topology_aware_mesh` reads the job layout off the PJRT devices and hands the
gathered device table to the pure `_rank_mesh_from_infos`, so exercising a
multi-slice layout only requires synthesizing that table. Both
`device_mesh_test` and `device_mesh_errors_test` drive the function that way,
which is why these helpers live here rather than in either test.
"""

# TODO(wan): revisit DECISION (b/568593385): the gloo fixture
# (distributed_utils.gloo_run) is not used here. topology_aware_mesh's only
# distributed step is the PJRT device gather, which gloo cannot serve, so the
# tests drive the pure _rank_mesh_from_infos directly instead.

from torch_tpu._internal.distributed import device_mesh


def v7x_multislice_infos(
    num_slices: int = 2,
    has_slice_index: bool = True,
) -> list[device_mesh._DeviceStub]:
  """Builds the device table observed on a real v7x-4 x N-slice job.

  Each slice is a 2x2x1 chip block with 2 TensorCores per chip (8 devices),
  coords are slice-local, and PJRT ids follow the MegaScale encoding
  `(slice_id + 1) * 100000 + per_slice_device_id`.

  Args:
    num_slices: Number of slices to synthesize.
    has_slice_index: When False, simulates MegaScale being inactive so no device
      reports a slice_index.

  Returns:
    Device stubs indexed by torch rank.
  """
  chip_coords = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
  infos = []
  for slice_index in range(num_slices):
    for local_device_id, (coords, core) in enumerate(
        (c, core) for c in chip_coords for core in (0, 1)
    ):
      infos.append(
          device_mesh._DeviceStub(
              id=(slice_index + 1) * 100000 + local_device_id,
              coords=coords,
              core_on_chip=core,
              device_kind="TPU7x",
              platform="tpu",
              process_index=len(infos),
              slice_index=slice_index if has_slice_index else 0,
              has_slice_index=has_slice_index,
          )
      )
  return infos
