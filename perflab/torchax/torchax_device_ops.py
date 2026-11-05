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

"""TorchaxDeviceOps: the TorchAx (torch_xla2 / JAX) implementation of DeviceOps."""

from collections.abc import Callable
from typing import Any

import jax
from torch_tpu._internal.benchmarks import xprof_adapter
from perflab import device_utils
from perflab import target as target_lib


class DeviceCountMismatch(RuntimeError):
  """Declared platform expects N devices but the host has M."""


def _sync_jax_device(x: Any) -> None:
  """Recursively synchronizes JAX devices across any PyTree or nested output.

  Blocks until every ready device future in x has completed. Value-level syncing
  in TorchAx requires syncing every leaf (including modified parameters and
  optimizer state) to avoid under-reporting step execution time.
  """

  def sync_leaf(leaf: Any) -> Any:
    if hasattr(leaf, "jax"):
      leaf.jax().block_until_ready()
    elif hasattr(leaf, "block_until_ready"):
      leaf.block_until_ready()
    return leaf

  jax.tree_util.tree_map(sync_leaf, x)


class TorchaxDeviceOps:
  """DeviceOps for the TorchAx / JAX backend framework.

  On construction it verifies the target topology against host JAX devices
  and manages value-level asynchronous future synchronization.
  """

  def __init__(self, target: target_lib.Target) -> None:
    self.target = target
    self._validate_device_count()
    self._compile_counter = 0

  def _validate_device_count(self) -> None:
    expected_local_devices = self.target.platform_spec.topology.nprocs_per_node
    actual_local_devices = jax.device_count()
    if actual_local_devices < expected_local_devices:
      raise DeviceCountMismatch(
          f"platform {self.target.platform.value!r} expects"
          f" {expected_local_devices} device(s) but the host has"
          f" {actual_local_devices}. Set BENCHMARK_PLATFORM to match"
          " this machine."
      )

  def await_result(self, out: Any) -> None:
    """Block until the value-level futures reachable from out complete."""
    _sync_jax_device(out)

  def check_result(self, out: Any) -> None:
    """No-op: `await_result` already raises if the work for out failed."""
    del out

  def reset_peak_memory(self) -> None:
    """Reset peak memory stats if supported by the underlying JAX backend."""
    try:
      for device in jax.local_devices():
        if hasattr(device, "reset_memory_stats"):
          device.reset_memory_stats()
    except Exception:  # pylint: disable=broad-except
      pass

  def peak_memory_mb(self) -> float | None:
    """Return peak device memory usage in MB, or None if unsupported."""
    return device_utils.get_jax_peak_memory_mb()

  def get_device_time(self, session_id: str | None = None) -> float:
    """Total active device time in seconds across profiling steps from XSpace."""
    if not session_id:
      return -1.0
    try:
      client = xprof_adapter.XprofAnalysisClient()
      return device_utils.get_max_total_device_time(session_id, client)
    except Exception:  # pylint: disable=broad-except
      return -1.0

  def get_device_time_from_xspace(
      self,
      xspace,
      expected_steps: int | None = None,
  ) -> float:
    """Total active device time in seconds from a locally captured XSpace."""
    return device_utils.get_total_device_time_from_xspace(
        xspace, expected_steps=expected_steps
    )

  def supports_event_device_time(self) -> bool:
    """Returns whether this backend can measure per-step device time via device events."""
    return False

  def measure_step_device_time(self, step_fn: Callable[[], Any]) -> float:
    """Executes one step and returns its active device time in seconds, or -1.0."""
    out = step_fn()
    self.await_result(out)
    self.check_result(out)
    return -1.0

  def compile_count(self) -> int:
    """Monotonic count of compilations performed so far."""
    # TODO - b/534438865:  Check if there's a way to track JAX compilation count
    # without monkey-patching jax.jit with a counter.
    return 0
