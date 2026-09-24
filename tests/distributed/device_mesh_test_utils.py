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

`topology_aware_mesh` reads the job layout off the PJRT devices, so exercising
a multi-slice layout only requires synthesizing the gathered device table.
Both `device_mesh_test` and `device_mesh_errors_test` drive the function that
way, which is why these helpers live here rather than in either test.
"""

from collections.abc import Callable
from typing import TypeVar
from unittest import mock

from torch_tpu._internal.distributed import device_mesh

_T = TypeVar("_T")


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


def patched_call(
    infos: list[device_mesh._DeviceStub],
    fn: Callable[[], _T],
) -> _T:
  """Invokes fn with torch.distributed and the device gather stubbed out.

  Args:
    infos: Device stubs the patched gather hands back, indexed by torch rank.
    fn: Callable to invoke under the patches.

  Returns:
    Whatever `fn` returns.
  """
  with mock.patch("torch.distributed.is_initialized", return_value=True):
    with mock.patch(
        "torch.distributed.get_world_size", return_value=len(infos)
    ):
      with mock.patch(
          "torch_tpu._internal.distributed.device_mesh._gather_global_device_info",
          return_value=infos,
      ):
        return fn()
