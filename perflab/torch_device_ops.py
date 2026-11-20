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

import collections
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch_tpu._internal.benchmarks import xprof_adapter
from perflab import device_utils
from perflab import target as target_lib

_MB = 2**20
_NS_PER_SECOND = 1e9


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> int:
  """Returns the total duration of the union of (start, end) intervals."""
  if not intervals:
    return 0
  sorted_intervals = sorted(intervals)
  cur_start, cur_end = sorted_intervals[0]
  total = 0
  for start, end in sorted_intervals[1:]:
    if start <= cur_end:
      if end > cur_end:
        cur_end = end
    else:
      total += cur_end - cur_start
      cur_start, cur_end = start, end
  total += cur_end - cur_start
  return total


def _cuda_device_time_from_profile(prof: Any) -> float:
  """Computes active CUDA device time in seconds from a torch.profiler run.

  Sums the union of CUDA device execution intervals (excluding pre-launch CPU
  overhead and inter-kernel host idle gaps) per GPU device and returns the max
  across devices, or -1.0 if no CUDA execution events were captured.

  Args:
    prof: A completed `torch.profiler.profile` context.

  Returns:
    Active CUDA device execution time in seconds, or -1.0 if unavailable.
  """
  kineto_results = getattr(
      getattr(prof, "profiler", None), "kineto_results", None
  )
  if kineto_results is None:
    return -1.0

  intervals_by_device: dict[int, list[tuple[int, int]]] = (
      collections.defaultdict(list)
  )
  for event in kineto_results.events():
    if event.device_type() != torch.autograd.DeviceType.CUDA:
      continue
    if getattr(event, "activity_type", lambda: "")() == "gpu_user_annotation":
      continue
    if getattr(event, "is_user_annotation", lambda: False)():
      continue
    if getattr(event, "is_hidden_event", lambda: False)():
      continue
    if event.name() == "Context Sync":
      continue
    start_ns = int(event.start_ns())
    end_ns = int(event.end_ns())
    if end_ns > start_ns:
      device_index = int(getattr(event, "device_index", lambda: 0)())
      intervals_by_device[device_index].append((start_ns, end_ns))

  if not intervals_by_device:
    return -1.0
  max_ns = max(
      _merge_intervals(intervals) for intervals in intervals_by_device.values()
  )
  return float(max_ns) / _NS_PER_SECOND if max_ns > 0 else -1.0


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

  def check_result(self, out: Any) -> None:
    """Raise if the work that produced out failed.

    A failed TorchTPU execution (e.g. an HBM OOM) does not fail
    torch.accelerator.synchronize(); it only marks its outputs as failed.
    Synchronizing on the outputs raises that failure.

    On CPU (including FakeTensorMode smoke tests), this still validates that the
    step returned at least one CPU tensor so step return-contract regressions
    are caught before reaching TPU. On CUDA, `await_result` already raises on
    failure, so this is a no-op.

    Args:
      out: The step's output. On TPU and CPU it must contain a tensor on the
        target device.

    Raises:
      ValueError: If the target is TPU or CPU and out contains no tensors on the
        target device.
    """
    if self.target.device_kind == target_lib.DeviceKind.CUDA:
      return
    tensors = device_utils.collect_tensors_on(self.device.type, out)
    if not tensors:
      device_label = self.device.type.upper()
      if self.target.device_kind == target_lib.DeviceKind.TPU:
        reason = "a failed TorchTPU execution only marks its outputs as failed"
      else:
        reason = "the TPU run checks the step's outputs to detect failures"
      raise ValueError(
          f"The step returned no {device_label} tensors, so its success can't"
          f" be verified: {reason}. Return the step's {device_label} outputs"
          " (e.g. the logits or the loss) from the step function. Got:"
          f" {type(out).__name__}."
      )
    if self.target.device_kind == target_lib.DeviceKind.TPU:
      device_utils.synchronize("tpu", tensors)

  def reset_peak_memory(self) -> None:
    """Reset the peak-memory counter."""
    if self.target.device_kind != target_lib.DeviceKind.CPU:
      torch.accelerator.synchronize()
      torch.accelerator.reset_peak_memory_stats(self.device)

  def release_cached_memory(self) -> None:
    """Releases the device memory that the allocator caches but doesn't use."""
    if self.target.device_kind != target_lib.DeviceKind.CPU:
      torch.accelerator.synchronize()
      torch.accelerator.empty_cache()

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

  def supports_event_device_time(self) -> bool:
    """Returns whether this backend can measure per-step device time via device events."""
    return self.target.device_kind == target_lib.DeviceKind.CUDA

  def measure_step_device_time(self, step_fn: Callable[[], Any]) -> float:
    """Executes one step and returns its active device time in seconds, or -1.0."""
    if not self.supports_event_device_time():
      out = step_fn()
      self.await_result(out)
      self.check_result(out)
      return -1.0
    self.await_result(None)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
      out = step_fn()
      self.await_result(out)
    self.check_result(out)
    return _cuda_device_time_from_profile(prof)

  def agree_on_step_count(self, steps: int) -> int:
    """Returns the largest step count proposed by the benchmark's processes.

    The processes of a distributed benchmark must run the same number of steps,
    as each step runs collectives across them.

    Args:
      steps: The step count that this process proposes.
    """
    if not torch.distributed.is_initialized():
      return steps
    # int32, as TorchTPU's all_reduce only sums and averages 64-bit integers.
    proposal = torch.tensor([steps], dtype=torch.int32, device=self.device)
    torch.distributed.all_reduce(proposal, op=torch.distributed.ReduceOp.MAX)
    return int(proposal.item())

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
