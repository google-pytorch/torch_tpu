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

"""Topology-aware DeviceMesh construction for TorchTPU.

Launchers may assign assign global rank ids with no topology input, so we need
to assume that the rank -> chip mapping is arbitrary. Therefore, the standard
sequential assignment of torch ranks may place logically adjacent mesh
coordinates on physically distant devices.

This module serves two purposes:
1. Topology mapping: rank's physical device coordinates are collected via an
   all_gather, making the resulting mesh correct and independent of the
   spawner's rank assignment.
2. Assignment: on top of the recovered physical layout, a device mesh build
   algorithm is used to compute a suitable logical-to-physical assignment using
   per-generation placement handlers and a N-D torus axis-assignment logic.

Note: `mesh_shape` follows the JAX convention — axes ordered by increasing
network intensity, i.e. the highest-traffic axis last. This matches the usual
PyTorch convention of data-parallel outermost.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist


@dataclasses.dataclass(frozen=True)
class _DeviceStub:
  """Duck-types the subset of jax.Device that mesh_utils touches."""

  id: int
  coords: Tuple[int, int, int]
  core_on_chip: int
  device_kind: str
  platform: str
  process_index: int
  slice_index: int


def _gather_global_device_info(device_type: str) -> list[_DeviceStub]:
  """Each torch rank reports its PJRT device; returns list indexed by rank."""
  d = torch.get_device_module(device_type).get_local_device_attributes()

  payload = torch.tensor(
      [
          d["id"],
          *d.get("coords", (0, 0, 0)),
          d.get("core_on_chip", 0),
          d["process_index"],
          d.get("slice_index", 0),
      ],
      dtype=torch.int64,
      device=device_type,
  )

  world_size = dist.get_world_size()
  gathered = [torch.zeros_like(payload) for _ in range(world_size)]
  dist.all_gather(gathered, payload)

  infos = []
  for t in gathered:
    g_dev_id, x, y, z, core, proc, slc = t.tolist()
    infos.append(
        _DeviceStub(
            id=g_dev_id,
            coords=(x, y, z),
            core_on_chip=core,
            device_kind=d["device_kind"],
            platform=device_type,
            process_index=proc,
            slice_index=slc,
        )
    )
  return infos


__all__ = ["topology_aware_mesh"]


def topology_aware_mesh(
    device_type: str,
    mesh_shape: Tuple[int, ...],
    *,
    topology: str = "single_slice",
    dcn_mesh_shape: Optional[Tuple[int, ...]] = None,
    process_is_granule: bool = False,
    allow_split_physical_axes: bool = False,
    contiguous_submeshes: bool = False,
) -> torch.Tensor:
  """Computes a topology-aware rank layout tensor, without building a mesh.

  Returns an int64 tensor of global torch ranks with shape `mesh_shape` (for
  topology="multi_slice": elementwise `dcn_mesh_shape * mesh_shape`), suitable
  for passing directly to `DeviceMesh(device_type, mesh=...)` or any other
  consumer of rank layouts (custom mesh construction, layout benchmarking).

  Deterministic across ranks: every rank derives the layout from the same
  all-gathered device payloads, satisfying DeviceMesh's requirement that the
  mesh tensor is identical on all ranks. Requires an initialized process group.


  Args:
    device_type: Device type for the mesh. Non-TPU device types are delegated to
      `torch.distributed.init_device_mesh` unchanged.
    mesh_shape: Logical mesh shape, ordered by increasing network intensity.
    topology: "single_slice" (default) arranges a single slice over ICI via
      mesh_utils.create_device_mesh; "multi_slice" uses
      create_hybrid_device_mesh for meshes spanning a slower outer network —
      pass dcn_mesh_shape, and set process_is_granule=True to make *hosts* the
      outer granule.
    dcn_mesh_shape: Outer (slower network) mesh shape for "multi_slice". The
      returned mesh has shape `dcn_mesh_shape * mesh_shape` elementwise.
    process_is_granule: Treat processes (hosts) rather than slices as the
      outer-network granule in "multi_slice" mode.
    allow_split_physical_axes: Forwarded to mesh_utils; permits splitting a
      physical axis across logical axes when required to realize mesh_shape.
    contiguous_submeshes: Forwarded to mesh_utils.create_device_mesh.

  Returns:
    An int64 tensor of global torch ranks mapping the requested virtual layout
    coordinates.

  Raises:
    RuntimeError: If the PyTorch process group is not initialized, or if there
      are duplicate PJRT device IDs across torch ranks.
    ValueError: If an unknown topology string is provided or if multi_slice mode
      is missing dcn_mesh_shape.
  """
  if not dist.is_initialized():
    raise RuntimeError(
        "PyTorch distributed process group must be initialized before calling "
        "topology_aware_mesh."
    )
  world_size = dist.get_world_size()

  from torch_tpu._internal.distributed import device_utils as mesh_utils  # pylint: disable=g-import-not-at-top

  infos = _gather_global_device_info(device_type)
  if len({i.id for i in infos}) != world_size:
    raise RuntimeError(
        "Duplicate pjrt device ids across torch ranks — expected exactly one "
        "torch process per TPU device (chip in megacore mode). This usually "
        "means multiple processes resolved the same local device."
    )

  # infos is indexed by torch rank; invert to map PJRT device id -> torch rank.
  id_to_rank = {info.id: rank for rank, info in enumerate(infos)}

  # JAX, the original upstream consumer of device_utils, enumerates by device
  # id. Some device-kind handlers in mesh_utils (e.g. the v2/v3 tray reshape)
  # may be sensitive to input order, so replicate that canonical ordering.
  jax_ordered = sorted(infos, key=lambda i: i.id)

  if topology == "multi_slice":
    if dcn_mesh_shape is None:
      raise ValueError("topology='multi_slice' requires dcn_mesh_shape")
    arranged = mesh_utils.create_hybrid_device_mesh(
        mesh_shape,
        dcn_mesh_shape,
        devices=jax_ordered,
        process_is_granule=process_is_granule,
        allow_split_physical_axes=allow_split_physical_axes,
    )
  elif topology == "single_slice":
    arranged = mesh_utils.create_device_mesh(
        mesh_shape,
        devices=jax_ordered,
        contiguous_submeshes=contiguous_submeshes,
        allow_split_physical_axes=allow_split_physical_axes,
    )
  else:
    raise ValueError(f"Unknown topology: {topology!r}")

  # arranged is an ndarray of _DeviceStub in logical-mesh shape; translate each
  # device back to the torch rank driving it.
  rank_mesh = np.vectorize(lambda dev: id_to_rank[dev.id], otypes=[np.int64])(
      arranged
  )

  return torch.from_numpy(rank_mesh)
