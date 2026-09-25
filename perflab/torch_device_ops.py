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

"""TorchDeviceOps: the torch implementation of the DeviceOps protocol."""

from typing import Any

import torch
from torch_tpu._internal.benchmarks import xprof_adapter
from perflab import device_utils
from perflab import target as target_lib

_MB = 2**20


class DeviceCountMismatch(RuntimeError):
  """Declared platform wants N devices but the host has M.

  Raised at construction so a misconfigured run fails at startup.
  """


_TORCH_DTYPE_MAP = {
    target_lib.DType.BF16: torch.bfloat16,
    target_lib.DType.FP32: torch.float32,
}


def get_torch_dtype(dtype: target_lib.DType) -> torch.dtype:
  """Get the torch dtype for the given target dtype."""
  if dtype not in _TORCH_DTYPE_MAP:
    raise ValueError(f"Unsupported torch dtype: {dtype}")
  return _TORCH_DTYPE_MAP[dtype]


class TorchDeviceOps:
  """DeviceOps for the torch framework

  On construction it materialises the torch device/dtype from the
  Target and validates the declared device count against the real host.
  """

  def __init__(
      self,
      target: target_lib.Target,
  ) -> None:
    self.target = target

    self.device = self._materialise_device()
    self.dtype = get_torch_dtype(self.target.dtype)

    # In distributed mode, each worker process only sees its single assigned
    # addressable device (device_count() == 1), so skip the host-level check.
    if not torch.distributed.is_initialized():
      self._validate_device_count()

  def _materialise_device(self) -> torch.device:
    kind = self.target.device_kind.value
    return torch.device(kind)

  def _validate_device_count(self) -> None:
    """The declared-vs-actual check that replaces detection."""
    if self.target.device_kind == target_lib.DeviceKind.CPU:
      return
    expected_local_devices = self.target.platform_spec.topology.nprocs_per_node
    actual_local_devices = torch.accelerator.device_count()
    if actual_local_devices < expected_local_devices:
      raise DeviceCountMismatch(
          f"platform {self.target.platform.value!r} expects"
          f" {expected_local_devices} device(s) but the host has"
          f" {actual_local_devices}. Set BENCHMARK_PLATFORM to match"
          " this machine."
      )

  def await_result(self, out: Any) -> None:  # pylint: disable=unused-argument
    """Block until the work that produced out has completed."""
    if self.target.device_kind != target_lib.DeviceKind.CPU:
      torch.accelerator.synchronize()

  def reset_peak_memory(self) -> None:
    """Reset the peak-memory counter."""
    if self.target.device_kind != target_lib.DeviceKind.CPU:
      torch.accelerator.reset_peak_memory_stats(self.device)

  def peak_memory_mb(self) -> float:
    """Peak device memory in MB since the last reset."""
    if self.target.device_kind == target_lib.DeviceKind.CPU:
      return 0.0
    return torch.accelerator.max_memory_allocated(self.device) / _MB

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

  def compile_count(self) -> int:
    """Monotonic count of successfully compiled frames this process."""
    if self.target.platform_spec.kind == target_lib.DeviceKind.TPU:
      return getattr(torch, "tpu")._get_cache_misses()  # pylint: disable=protected-access
    counters = self._dynamo_counters()
    if counters is None:
      return 0
    return int(counters["frames"]["ok"])

  def _dynamo_counters(self):
    """The Dynamo frame counter dict, or None if unavailable."""
    try:
      from torch._dynamo import utils  # pylint: disable=g-import-not-at-top

      return utils.counters
    except Exception:
      return None
