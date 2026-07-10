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
"""Topology-aware device mesh creation for PyTorch on TPU."""

import collections
import itertools
import math
import os
import re
from typing import Any, Generator, List, Sequence, Tuple
from absl import logging
import numpy as np
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch_tpu._internal.distributed import tpu_distributed


class MockDevice:
  """Mock device class representing a physical TPU core."""

  def __init__(self, rank: int, coords: Tuple[int, ...], core_on_chip: int):
    self.id = rank
    self.coords = coords  # N-D physical coordinates (e.g., X, Y, Z)
    self.core_on_chip = core_on_chip

  def __repr__(self):
    return (
        f"Device(id={self.id}, coords={self.coords}, core={self.core_on_chip})"
    )


def parse_physical_coords(debug_str: str) -> Tuple[Tuple[int, ...], int] | None:
  """Parses physical coordinates and core index from PJRT debug string.

  Format: TPU_<global_id>(process=<rank>,(<coords...>,<core>))
  Supports arbitrary N-D coordinates.

  Args:
    debug_str: The PJRT debug string.

  Returns:
    A tuple of ((coords...), core_index) if parsed, or None.
  """
  match = re.search(r"process=\d+,\(([^)]+)\)", debug_str)
  if match:
    parts = [int(x) for x in match.group(1).split(",")]
    if len(parts) >= 2:
      return tuple(parts[:-1]), parts[-1]
    elif len(parts) == 1:
      return (0,), parts[0]
  return None


def _get_prime_factors(x: int) -> List[int]:
  assert x > 0
  factors = []
  for p in range(2, math.isqrt(x) + 2):
    while x % p == 0:
      factors.append(p)
      x //= p
    if x == 1:
      return factors
  else:
    return [x]


def _enumerate_feasible_logical_axis_assignments(
    physical_mesh_shape: Sequence[int],
    assignment: np.ndarray,
    logical_axis_size: int,
) -> Generator[np.ndarray, None, None]:
  """Enumerates feasible assignments of logical axes to physical axes.

  Args:
    physical_mesh_shape: Shape of physical mesh.
    assignment: Current assignment matrix.
    logical_axis_size: Size of logical axis to assign.

  Yields:
    Feasible assignment vectors.
  """
  logical_axis_factors = collections.defaultdict(int)
  for factor in _get_prime_factors(logical_axis_size):
    logical_axis_factors[factor] += 1

  available_physical_mesh_shape = np.array(physical_mesh_shape) // np.prod(
      assignment, axis=-1
  )

  physical_axes_by_factor = collections.defaultdict(list)
  for physical_axis, physical_axis_size in enumerate(
      available_physical_mesh_shape
  ):
    for factor in _get_prime_factors(physical_axis_size):
      if factor not in logical_axis_factors:
        continue
      physical_axes_by_factor[factor].append(physical_axis)

  factors = []
  assignments_by_factor = []
  for factor, multiplicity in logical_axis_factors.items():
    factors.append(factor)
    assignments_by_factor.append(
        set(
            itertools.combinations(
                physical_axes_by_factor[factor], multiplicity
            )
        )
    )

  for axis_assignment in itertools.product(*assignments_by_factor):
    result = np.ones([len(physical_mesh_shape)], dtype=np.int64)
    for factor_index, per_factor_assignment in enumerate(axis_assignment):
      for physical_axis in per_factor_assignment:
        result[physical_axis] *= factors[factor_index]
    yield result


def _prefer_first_logical_axis_assignment(
    x: np.ndarray,
    y: np.ndarray,
    *,
    physical_mesh_shape: Sequence[int],
    assignment: np.ndarray,
) -> bool:
  """Determines if assignment x is preferred over y.

  Args:
    x: First assignment.
    y: Second assignment.
    physical_mesh_shape: Shape of physical mesh.
    assignment: Current assignment matrix.

  Returns:
    True if x is preferred, False otherwise.
  """
  x_whole_axis_size = np.prod(
      [s for i, s in enumerate(x) if s == physical_mesh_shape[i]]
  )
  y_whole_axis_size = np.prod(
      [s for i, s in enumerate(y) if s == physical_mesh_shape[i]]
  )

  if x_whole_axis_size != y_whole_axis_size:
    return x_whole_axis_size > y_whole_axis_size

  x_num_whole_axes = len(
      [1 for i, s in enumerate(x) if s == physical_mesh_shape[i] and s > 1]
  )
  y_num_whole_axes = len(
      [1 for i, s in enumerate(y) if s == physical_mesh_shape[i] and s > 1]
  )

  if x_num_whole_axes != y_num_whole_axes:
    return x_num_whole_axes > y_num_whole_axes

  assigned_physical_mesh_shape = np.prod(assignment, axis=-1)

  x_non_overlapping_axis_size = np.prod(
      [s for i, s in enumerate(x) if assigned_physical_mesh_shape[i] > 1]
  )
  y_non_overlapping_axis_size = np.prod(
      [s for i, s in enumerate(y) if assigned_physical_mesh_shape[i] > 1]
  )

  if x_non_overlapping_axis_size != y_non_overlapping_axis_size:
    return x_non_overlapping_axis_size > y_non_overlapping_axis_size

  return tuple(x) > tuple(y)


def _generate_logical_mesh(
    physical_mesh: np.ndarray,
    logical_mesh_shape: Sequence[int],
    assignment: np.ndarray,
) -> np.ndarray:
  """Generates logical mesh from physical mesh and assignment.

  Args:
    physical_mesh: Physical mesh array.
    logical_mesh_shape: Shape of logical mesh.
    assignment: Assignment matrix.

  Returns:
    Logical mesh array.
  """
  physical_indices = np.broadcast_to(
      np.expand_dims(
          np.arange(len(physical_mesh.shape), dtype=np.int64), axis=-1
      ),
      assignment.shape,
  ).reshape([-1])

  logical_indices = np.broadcast_to(
      np.expand_dims(
          np.arange(len(logical_mesh_shape), dtype=np.int64), axis=0
      ),
      assignment.shape,
  ).reshape([-1])

  logical_mesh = np.reshape(physical_mesh, assignment.reshape([-1]))

  _, _, transpose_axes = zip(
      *sorted(
          zip(logical_indices, physical_indices, range(len(logical_indices)))
      )
  )
  logical_mesh = np.transpose(logical_mesh, transpose_axes)
  logical_mesh = np.reshape(logical_mesh, logical_mesh_shape)
  return logical_mesh


def _create_device_mesh_for_nd_torus(
    physical_mesh: np.ndarray,
    mesh_shape: Sequence[int],
) -> np.ndarray:
  """Assigns logical parallelism axes to physical axes of an N-D torus network.

  Always allows splitting physical axes for maximum flexibility across
  topologies.

  Args:
    physical_mesh: Physical mesh array.
    mesh_shape: Logical mesh shape.

  Returns:
    Logical mesh array.
  """
  if np.prod(physical_mesh.shape) != np.prod(mesh_shape):
    raise ValueError(
        f"The number of devices in physical mesh {physical_mesh.shape} "
        f"does not match logical mesh {mesh_shape}."
    )

  physical_mesh_shape = physical_mesh.shape
  logical_mesh_shape = tuple(mesh_shape)

  assignment = np.ones(
      [len(physical_mesh_shape), len(logical_mesh_shape)], dtype=np.int64
  )

  for logical_axis, logical_axis_size in reversed(
      list(enumerate(logical_mesh_shape))
  ):
    best_logical_axis_assignment = None
    for logical_axis_assignment in _enumerate_feasible_logical_axis_assignments(
        physical_mesh_shape, assignment, logical_axis_size
    ):
      if (
          best_logical_axis_assignment is None
          or _prefer_first_logical_axis_assignment(
              logical_axis_assignment,
              best_logical_axis_assignment,
              physical_mesh_shape=physical_mesh_shape,
              assignment=assignment,
          )
      ):
        best_logical_axis_assignment = logical_axis_assignment
    assignment[:, logical_axis] = best_logical_axis_assignment

  return _generate_logical_mesh(physical_mesh, logical_mesh_shape, assignment)


def _get_physical_tpu_mesh(devices: Sequence[MockDevice]) -> np.ndarray:
  """Rearrange TPU devices into an N-D physical mesh [Coords..., Core]."""
  device_coords = [d.coords for d in devices]
  num_dims = len(device_coords[0])

  max_coords = tuple(
      max(dc[i] for dc in device_coords) for i in range(num_dims)
  )
  min_coords = tuple(
      min(dc[i] for dc in device_coords) for i in range(num_dims)
  )
  dims = tuple(h - l + 1 for (h, l) in zip(max_coords, min_coords))

  max_cores_per_chip = max(d.core_on_chip for d in devices)
  min_cores_per_chip = min(d.core_on_chip for d in devices)
  cores_per_chip = max_cores_per_chip - min_cores_per_chip + 1

  # We construct an N+1D physical mesh: (Coords..., Core)
  out = np.empty(dims + (cores_per_chip,), dtype=object)
  for d in devices:
    idx = tuple(c - min_c for c, min_c in zip(d.coords, min_coords)) + (
        d.core_on_chip - min_cores_per_chip,
    )
    out[idx] = d

  if (out == None).any():  # pylint: disable=singleton-comparison
    raise AssertionError("Constructed physical mesh contains holes.")
  return out


_layout_priority: List[str] | None = None


def set_layout_priority(priority: Sequence[str] | None) -> None:
  """Sets the global layout priority for topology-aware DeviceMesh mapping.

  Args:
    priority: Sequence of dimension names ordered from highest to lowest
      communication intensity (e.g., ('tp', 'sp', 'dp')), or None to clear it.
  """
  global _layout_priority
  if priority is None:
    _layout_priority = None
  else:
    _layout_priority = list(priority)


def get_layout_priority() -> List[str] | None:
  """Returns the currently configured global layout priority."""
  return _layout_priority


DEFAULT_PRIORITY = ("tp", "sp", "dp")


def _reorder_for_priority(
    mesh_shape: Sequence[int],
    mesh_dim_names: Sequence[str],
    priority: Sequence[str],
) -> Tuple[Tuple[int, ...], Tuple[str, ...], Tuple[int, ...]]:
  """Reorders mesh_shape and mesh_dim_names based on priority."""
  full_priority = []
  for p in priority:
    if p in mesh_dim_names and p not in full_priority:
      full_priority.append(p)
  for name in mesh_dim_names:
    if name not in full_priority:
      full_priority.append(name)

  target_internal_names = tuple(reversed(full_priority))
  name_to_size = dict(zip(mesh_dim_names, mesh_shape))
  internal_shape = tuple(name_to_size[name] for name in target_internal_names)

  name_to_internal_idx = {
      name: i for i, name in enumerate(target_internal_names)
  }
  transpose_perm = tuple(name_to_internal_idx[name] for name in mesh_dim_names)

  return internal_shape, target_internal_names, transpose_perm


def get_topology_aware_mesh(
    mesh_shape: Sequence[int],
    mesh_dim_names: Sequence[str] | None = None,
) -> np.ndarray:
  """Computes a topology-aware mapping of ranks to the logical mesh shape."""
  if not dist.is_initialized():
    logging.info("Initializing default process group (tpu_dist)")
    dist.init_process_group("tpu_dist")

  rank = dist.get_rank()
  world_size = dist.get_world_size()

  # Get local device info
  debug_str = tpu_distributed.device_debug_string()
  parsed = parse_physical_coords(debug_str)
  if not parsed:
    raise RuntimeError(f"Failed to parse coordinates from: {debug_str}")

  coords, core = parsed
  local_info = {"rank": rank, "coords": coords, "core": core}

  # Gather from all ranks
  gathered: List[Any] = [None] * world_size
  dist.all_gather_object(gathered, local_info)

  # Reconstruct devices list
  devices = []
  for info in gathered:
    devices.append(
        MockDevice(
            rank=info["rank"],
            coords=info["coords"],
            core_on_chip=info["core"],
        )
    )

  # Build physical mesh
  physical_mesh = _get_physical_tpu_mesh(devices)

  # Apply priority reordering if names are available
  if mesh_dim_names:
    priority = _layout_priority
    if not priority:
      priority_env = os.environ.get("TORCH_TPU_LAYOUT_PRIORITY")
      if priority_env:
        priority = [p.strip() for p in priority_env.split(",")]
      else:
        priority = DEFAULT_PRIORITY

    logging.info("Applying topology-aware mapping with priority: %s", priority)
    internal_shape, _, transpose_perm = _reorder_for_priority(
        mesh_shape, mesh_dim_names, priority
    )

    # Map logical to physical using internal shape
    logical_device_mesh = _create_device_mesh_for_nd_torus(
        physical_mesh, internal_shape
    )

    # Transpose back to user shape
    logical_device_mesh = np.transpose(logical_device_mesh, transpose_perm)
  else:
    # Fallback to default behavior if no names
    logical_device_mesh = _create_device_mesh_for_nd_torus(
        physical_mesh, mesh_shape
    )

  rank_mesh = np.ascontiguousarray(
      np.vectorize(lambda d: d.id)(logical_device_mesh)
  )
  return rank_mesh


def init_topology_aware_device_mesh(
    device_type: str,
    mesh_shape: Tuple[int, ...],
    *,
    mesh_dim_names: Tuple[str, ...] | None = None,
    backend_override: Any = None,
) -> DeviceMesh:
  """Initializes a DeviceMesh with topology-aware rank mapping for TPU."""
  if device_type == "tpu":
    logging.info("Creating topology-aware DeviceMesh for shape %s", mesh_shape)
    rank_mesh = get_topology_aware_mesh(
        mesh_shape, mesh_dim_names=mesh_dim_names
    )
    # Instantiate DeviceMesh with the custom mesh array
    mesh = DeviceMesh(
        device_type="tpu",
        mesh=rank_mesh,
        mesh_dim_names=mesh_dim_names,
        backend_override=backend_override,
    )
    return mesh
  else:
    raise NotImplementedError("Only tpu device_type is supported by this hook.")
