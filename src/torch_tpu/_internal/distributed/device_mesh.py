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
PyTorch convention of data-parallel outermost. It always describes the full
job: on a multi-slice job the mesh is split across slices along one of its
axes rather than being repeated per slice.
"""

from __future__ import annotations

import collections
import dataclasses
import math
from typing import Optional, Tuple

from absl import logging
import numpy as np
import torch
import torch.distributed as dist
from torch_tpu._internal.distributed import tpu_distributed


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
  # False when the PJRT device did not report a `slice_index` attribute, in
  # which case `slice_index` above is a placeholder zero. Only the MegaScale
  # PJRT client publishes this attribute, so its absence means the process sees
  # a single slice no matter how many slices the job requested.
  has_slice_index: bool = True


def _gather_global_device_info(device_type: str) -> list[_DeviceStub]:
  """Each torch rank reports its PJRT device; returns list indexed by rank."""
  attrs = tpu_distributed.global_device_attributes()
  return [
      _DeviceStub(
          id=d.id,
          coords=(
              int(d.coords[0]) if d.coords and len(d.coords) > 0 else 0,
              int(d.coords[1]) if d.coords and len(d.coords) > 1 else 0,
              int(d.coords[2]) if d.coords and len(d.coords) > 2 else 0,
          ),
          core_on_chip=d.core_on_chip if d.core_on_chip is not None else 0,
          device_kind=d.device_kind,
          platform=device_type,
          process_index=d.process_index,
          slice_index=d.slice_index if d.slice_index is not None else 0,
          has_slice_index=d.slice_index is not None,
      )
      for d in attrs
  ]


def _infer_dcn_mesh_shape(
    mesh_shape: Tuple[int, ...], num_slices: int
) -> Tuple[int, ...]:
  """Picks the logical axis along which to split the mesh across slices.

  The split goes on the outermost non-trivial axis. `mesh_shape` is ordered by
  increasing network intensity, so that axis is the least communication-hungry
  one and the right place to pay the inter-slice cost. For the usual HSDP shape
  it is the replicate axis, keeping every shard group inside one slice.

  We deliberately do not fall through to an inner axis when the outermost one
  is indivisible: the next candidate is typically the shard axis, and splitting
  shard groups across DCN is the exact layout multi-slice HSDP exists to avoid.
  A caller who genuinely wants that must ask for it via `dcn_mesh_shape`.

  Args:
    mesh_shape: Full logical mesh shape, ordered by increasing network
      intensity.
    num_slices: Number of slices the job spans.

  Returns:
    A shape with the same rank as `mesh_shape`, holding `num_slices` on the
    chosen axis and 1 everywhere else.

  Raises:
    ValueError: If every axis is trivial, or if the outermost non-trivial axis
      cannot absorb the split.
  """
  axis = next((i for i, extent in enumerate(mesh_shape) if extent > 1), None)
  if axis is None:
    raise ValueError(
        f"Cannot spread mesh_shape={tuple(mesh_shape)} over {num_slices} "
        "slices: every axis has extent 1. Pass dcn_mesh_shape explicitly to "
        "choose the split yourself."
    )

  extent = mesh_shape[axis]
  if extent % num_slices:
    raise ValueError(
        f"Cannot spread mesh_shape={tuple(mesh_shape)} over {num_slices} "
        f"slices: the outermost non-trivial axis (axis {axis}, extent "
        f"{extent}) is not divisible by {num_slices}. Refusing to fall back "
        "to an inner axis, which would split shard groups across the slower "
        "inter-slice network. Reshape the mesh so the outermost axis absorbs "
        "the slice count, or pass dcn_mesh_shape explicitly to choose the "
        "split yourself."
    )

  logging.info(
      "topology_aware_mesh: splitting mesh_shape=%s across %d slices on "
      "axis %d; pass dcn_mesh_shape to override.",
      tuple(mesh_shape),
      num_slices,
      axis,
  )
  dcn_mesh_shape = [1] * len(mesh_shape)
  dcn_mesh_shape[axis] = num_slices
  return tuple(dcn_mesh_shape)


def _validate_multi_slice(
    infos: list[_DeviceStub],
    mesh_shape: Tuple[int, ...],
    dcn_mesh_shape: Tuple[int, ...],
) -> None:
  """Validates a multi-slice request against the observed device topology.

  Args:
    infos: Gathered device stubs, indexed by torch rank.
    mesh_shape: Full logical mesh shape, spanning every slice.
    dcn_mesh_shape: Inter-slice mesh shape. Dividing `mesh_shape` by it
      elementwise gives the per-slice (ICI) shape.

  Raises:
    ValueError: If the shapes are inconsistent with each other or with the
      observed slice layout.
  """
  if len(dcn_mesh_shape) != len(mesh_shape):
    raise ValueError(
        "mesh_shape and dcn_mesh_shape must have the same number of "
        f"dimensions, got mesh_shape={tuple(mesh_shape)} "
        f"({len(mesh_shape)}D) and dcn_mesh_shape={tuple(dcn_mesh_shape)} "
        f"({len(dcn_mesh_shape)}D)."
    )

  world_size = len(infos)
  per_slice_counts = collections.Counter(i.slice_index for i in infos)
  num_slices = len(per_slice_counts)

  if math.prod(dcn_mesh_shape) != num_slices:
    raise ValueError(
        f"dcn_mesh_shape={tuple(dcn_mesh_shape)} covers "
        f"{math.prod(dcn_mesh_shape)} slices, but the job has {num_slices} "
        f"(slice_index values {sorted(per_slice_counts)})."
    )

  indivisible = {
      axis: (extent, dcn_extent)
      for axis, (extent, dcn_extent) in enumerate(
          zip(mesh_shape, dcn_mesh_shape)
      )
      if extent % dcn_extent
  }
  if indivisible:
    raise ValueError(
        f"dcn_mesh_shape={tuple(dcn_mesh_shape)} must divide "
        f"mesh_shape={tuple(mesh_shape)} elementwise, since mesh_shape spans "
        "every slice and the per-slice shape is their elementwise quotient. "
        "These axes do not divide evenly, as axis: (mesh, dcn): "
        f"{indivisible}."
    )

  if math.prod(mesh_shape) != world_size:
    raise ValueError(
        f"mesh_shape={tuple(mesh_shape)} has {math.prod(mesh_shape)} elements "
        f"but the job has {world_size} devices across {num_slices} slices. "
        "mesh_shape describes the full mesh, so pass a shape whose product is "
        f"{world_size} and let dcn_mesh_shape carve out the slices."
    )

  devices_per_slice = world_size // num_slices
  ragged = {s: n for s, n in per_slice_counts.items() if n != devices_per_slice}
  if ragged:
    raise ValueError(
        "Every slice must contribute the same number of devices "
        f"({devices_per_slice}), but these do not: {ragged}. This usually "
        "means a slice lost tasks or the launcher assigned ranks unevenly."
    )


__all__ = ["topology_aware_mesh", "get_device_assignment"]


def topology_aware_mesh(
    device_type: str,
    mesh_shape: Tuple[int, ...],
    *,
    dcn_mesh_shape: Optional[Tuple[int, ...]] = None,
    allow_split_physical_axes: bool = False,
    contiguous_submeshes: bool = False,
) -> torch.Tensor:
  """Computes a topology-aware rank layout tensor, without building a mesh.

  Returns an int64 tensor of global torch ranks with shape `mesh_shape`,
  suitable for passing directly to `DeviceMesh(device_type, mesh=...)` or any
  other consumer of rank layouts (custom mesh construction, layout
  benchmarking).

  Whether the job spans one slice or several is read off the devices, so the
  same call covers both. On a multi-slice job the logical mesh is split across
  slices along a single axis picked by `_infer_dcn_mesh_shape`, which
  `dcn_mesh_shape` overrides.

  Deterministic across ranks: every rank derives the layout from the same
  all-gathered device payloads, satisfying DeviceMesh's requirement that the
  mesh tensor is identical on all ranks. Requires an initialized process group.

  Args:
    device_type: Device type for the mesh, e.g. "tpu".
    mesh_shape: Full logical mesh shape, ordered by increasing network
      intensity. Its product must equal the world size — on multi-slice jobs it
      describes the whole job, not one slice.
    dcn_mesh_shape: Which logical axes are split across slices. Same rank as
      `mesh_shape`, product equal to the number of slices, and must divide
      `mesh_shape` elementwise; the quotient is the per-slice (ICI) shape. None,
      the default, infers it from `mesh_shape` and the observed slice count, and
      is the only way to defer the choice: an all-ones shape asks for no
      inter-slice split at all, which a job spanning more than one slice cannot
      satisfy and which is rejected rather than inferred.
    allow_split_physical_axes: Forwarded to mesh_utils; permits splitting a
      physical axis across logical axes when required to realize mesh_shape.
    contiguous_submeshes: Forwarded to mesh_utils.create_device_mesh.

  Returns:
    An int64 tensor of global torch ranks mapping the requested virtual layout
    coordinates.

  Raises:
    RuntimeError: If the PyTorch process group is not initialized, if there are
      duplicate PJRT device IDs across torch ranks, if only some ranks report a
      slice index, or if a multi-slice layout is requested on devices that do
      not report a slice index.
    ValueError: If `mesh_shape` and `dcn_mesh_shape` are inconsistent with each
      other or with the observed topology.
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
        "means multiple processes resolved the same local device, or that a "
        "multi-slice job is running without the MegaScale PJRT client, which "
        "leaves every slice numbering its devices from zero."
    )

  # infos is indexed by torch rank; invert to map PJRT device id -> torch rank.
  id_to_rank = {info.id: rank for rank, info in enumerate(infos)}

  # JAX, the original upstream consumer of device_utils, enumerates by device
  # id. Some device-kind handlers in mesh_utils (e.g. the v2/v3 tray reshape)
  # may be sensitive to input order, so replicate that canonical ordering.
  jax_ordered = sorted(infos, key=lambda i: i.id)

  # slice_index is published only by the MegaScale PJRT client, which the TPU
  # plugin installs exactly on multi-slice jobs, so its absence means the
  # process sees a single slice.
  ranks_reporting_slice_index = sum(i.has_slice_index for i in infos)
  if ranks_reporting_slice_index not in (0, world_size):
    raise RuntimeError(
        f"{ranks_reporting_slice_index} of {world_size} torch ranks report a "
        "'slice_index' attribute, so the job mixes the MegaScale PJRT client "
        "with the plain one. The ranks that stay silent cannot be placed, "
        "since their slice is unknowable, so every rank has to agree. Check "
        "that every task gets the same MegaScale settings (the MEGASCALE_* "
        "environment variables, or --megascale_port)."
    )
  has_slice_index = ranks_reporting_slice_index == world_size
  num_slices = len({i.slice_index for i in infos}) if has_slice_index else 1

  # Branch on the observed slice count, not on dcn_mesh_shape: a job spanning
  # several slices needs the hybrid path whatever the caller passed, since one
  # ICI fabric cannot cover them. The second term catches the reverse case, a
  # split asked for on a runtime that reports no slices at all.
  requested_dcn_split = (
      dcn_mesh_shape is not None and math.prod(dcn_mesh_shape) > 1
  )
  if num_slices > 1 or requested_dcn_split:
    # Only reachable through requested_dcn_split, since num_slices > 1 means
    # the devices did report a slice index; the None test is for the checker.
    if dcn_mesh_shape is not None and not has_slice_index:
      raise RuntimeError(
          "A multi-slice layout was requested "
          f"(dcn_mesh_shape={tuple(dcn_mesh_shape)}), but the PJRT devices do "
          "not expose a 'slice_index' attribute. That attribute is published "
          "only by the MegaScale PJRT client, which TorchTPU skips when the "
          "world is a single slice, i.e. when WORLD_SIZE is no larger than "
          "the number of addresses in TORCH_TPU_SLICEBUILDER_ADDRESSES. "
          "Otherwise the TPU plugin installs it when --megascale_port >= 0 "
          "and SKIP_MEGASCALE_PJRT_CLIENT is unset; do not set it at launch, "
          "TorchTPU sets it itself after creating its client. libtpu sets "
          "--megascale_port to MEGASCALE_PORT once "
          "MEGASCALE_COORDINATOR_ADDRESS, MEGASCALE_SLICE_ID and "
          "MEGASCALE_NUM_SLICES are set. Give the coordinator address a port, "
          "e.g. <MASTER_ADDR>:8080, since libtpu otherwise appends each "
          "process's own MEGASCALE_PORT."
      )
    if dcn_mesh_shape is None:
      dcn_mesh_shape = _infer_dcn_mesh_shape(mesh_shape, num_slices)
    _validate_multi_slice(infos, mesh_shape, dcn_mesh_shape)
    arranged = mesh_utils.create_hybrid_device_mesh(
        tuple(m // d for m, d in zip(mesh_shape, dcn_mesh_shape)),
        dcn_mesh_shape,
        devices=jax_ordered,
        allow_split_physical_axes=allow_split_physical_axes,
    )
  else:
    arranged = mesh_utils.create_device_mesh(
        mesh_shape,
        devices=jax_ordered,
        contiguous_submeshes=contiguous_submeshes,
        allow_split_physical_axes=allow_split_physical_axes,
    )

  # arranged is an ndarray of _DeviceStub in logical-mesh shape; translate each
  # device back to the torch rank driving it.
  rank_mesh = np.vectorize(lambda dev: id_to_rank[dev.id], otypes=[np.int64])(
      arranged
  )

  return torch.from_numpy(rank_mesh)


def get_device_assignment(
    pg: Optional[dist.ProcessGroup] = None,
) -> list[int]:
  """Returns physical device assignment for the given process group.

  Args:
    pg: The process group to query. If None or dist.group.WORLD, queries the
      world process group's rank-to-device mapping.

  Returns:
    A list of global device IDs corresponding to ranks in the process group.
  """
  if not dist.is_initialized():
    return []
  pg = pg or dist.group.WORLD
  if pg is None:
    return []
  try:
    backend = pg._get_backend(torch.device("tpu"))
    return [int(x) for x in tpu_distributed.rank_to_device_id(backend)]
  except (RuntimeError, TypeError):
    return []
