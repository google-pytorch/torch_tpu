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

"""Helpers for exporting TPU Pallas programs with logical CPU devices.

This module intentionally does not import JAX at module import time. Call
`configure_jax_cpu_export_env` before importing JAX when the process should use
JAX only as a TPU export frontend.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
import math
import os
import re
import sys
import threading

_CPU_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"
_TPU_INFO_PATCH_LOCK = threading.RLock()


def configure_jax_cpu_export_platform() -> None:
  """Configure JAX to use only the CPU backend before backend startup."""
  configured = os.environ.get("JAX_PLATFORMS")
  if configured is not None and configured.strip().lower() != "cpu":
    raise RuntimeError(
        "TorchTPU Pallas CPU export requires JAX_PLATFORMS=cpu; got"
        f" {configured!r}."
    )
  if configured is None and "jax" in sys.modules:
    raise RuntimeError(
        "JAX was imported before the TorchTPU Pallas CPU export platform was"
        " configured. Set JAX_PLATFORMS=cpu before importing JAX."
    )
  os.environ["JAX_PLATFORMS"] = "cpu"


def _jax_backend_initialized() -> bool:
  bridge = sys.modules.get("jax._src.xla_bridge")
  return bool(getattr(bridge, "_backends", None))


def configure_jax_cpu_export_env(world_size: int) -> None:
  """Configure one logical CPU device per global Torch rank."""
  if world_size < 1:
    raise ValueError(f"world_size must be positive, got {world_size}")

  configure_jax_cpu_export_platform()
  xla_flags = os.environ.get("XLA_FLAGS", "")
  pattern = re.compile(
      rf"(?:^|\s){re.escape(_CPU_DEVICE_COUNT_FLAG)}(?:=|\s+)(\d+)(?=\s|$)"
  )
  configured_counts = [int(value) for value in pattern.findall(xla_flags)]
  if configured_counts:
    if len(configured_counts) != 1 or configured_counts[0] != world_size:
      raise RuntimeError(
          f"{_CPU_DEVICE_COUNT_FLAG} must equal world_size={world_size}; "
          f"XLA_FLAGS={xla_flags!r}."
      )
    return

  if _jax_backend_initialized():
    raise RuntimeError(
        "The JAX backend was initialized before TorchTPU configured its"
        f" {world_size} logical CPU export devices."
    )
  os.environ["XLA_FLAGS"] = " ".join(
      part
      for part in (xla_flags.strip(), f"{_CPU_DEVICE_COUNT_FLAG}={world_size}")
      if part
  )


def get_jax_cpu_export_devices(world_size: int):
  """Return distinct logical CPU devices for global partitioning."""
  configure_jax_cpu_export_env(world_size)

  import jax  # pylint: disable=g-import-not-at-top

  devices = tuple(jax.devices("cpu"))
  if len(devices) != world_size:
    raise RuntimeError(
        f"Expected {world_size} logical CPU export devices, got {len(devices)}."
        " JAX may have initialized before the export environment was configured."
    )
  if len(set(devices)) != world_size:
    raise RuntimeError("JAX returned duplicate logical CPU export devices.")
  return devices


def make_jax_cpu_export_mesh(
    mesh_shape: int | Sequence[int],
    axis_names: Sequence[str],
):
  """Build a `jax.sharding.Mesh` backed by logical CPU devices."""
  if isinstance(mesh_shape, int):
    mesh_shape = (mesh_shape,)
  mesh_shape = tuple(mesh_shape)
  if len(mesh_shape) != len(axis_names):
    raise ValueError(
        f"mesh_shape rank {len(mesh_shape)} must match axis_names rank"
        f" {len(axis_names)}."
    )

  world_size = math.prod(mesh_shape)
  devices = get_jax_cpu_export_devices(world_size)

  import numpy as np  # pylint: disable=g-import-not-at-top
  from jax.sharding import Mesh  # pylint: disable=g-import-not-at-top

  return Mesh(np.asarray(devices).reshape(mesh_shape), tuple(axis_names))


def get_torch_tpu_device_kind() -> str:
  """Return physical TPU hardware metadata without querying JAX devices."""
  from torch_tpu._internal.utils.hardware import (  # pylint: disable=g-import-not-at-top
      get_tpu_device_name,
  )

  return get_tpu_device_name()


def get_pallas_tpu_device_kind() -> str:
  """Return the physical device spelling expected by JAX Pallas."""
  device_kind = get_torch_tpu_device_kind()
  return "TPU7x" if device_kind == "TPU v7" else device_kind


@contextmanager
def tpu_info_for_cpu_export(device_kind: str | None = None):
  """Use physical TPU metadata while exporting with CPU mesh devices."""
  from jax._src.pallas.mosaic import tpu_info  # pylint: disable=g-import-not-at-top
  from jax.experimental.pallas import tpu as pltpu  # pylint: disable=g-import-not-at-top

  with _TPU_INFO_PATCH_LOCK:
    original_public = pltpu.get_tpu_info
    original_private = tpu_info.get_tpu_info
    original_device_kind = tpu_info.get_device_kind
    original_is_tpu_device = tpu_info.is_tpu_device
    device_kind = device_kind or get_pallas_tpu_device_kind()
    chip_version = tpu_info.chip_version_from_device_kind(device_kind)
    if chip_version is None:
      raise RuntimeError(
          f"Unsupported TPU device kind for Pallas CPU export: {device_kind}"
      )
    info = tpu_info.get_tpu_info_for_chip(chip_version, 1)
    pltpu.get_tpu_info = lambda: info
    tpu_info.get_tpu_info = lambda: info
    tpu_info.get_device_kind = lambda: device_kind
    tpu_info.is_tpu_device = lambda: True
    try:
      yield
    finally:
      pltpu.get_tpu_info = original_public
      tpu_info.get_tpu_info = original_private
      tpu_info.get_device_kind = original_device_kind
      tpu_info.is_tpu_device = original_is_tpu_device
