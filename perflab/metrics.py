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

"""Metrics and result classes for E2E benchmarks."""

import abc
from collections.abc import Mapping
import dataclasses

from perflab import workload as workload_lib


def _seconds_to_ms(value: float) -> float:
  """Converts a duration in seconds to milliseconds."""
  return float(value) * 1000.0


@dataclasses.dataclass
class WarmupRunResult:
  """Result of the warmup run.

  Attributes:
    num_warmup_steps: The number of warmup steps taken for the cache misses to
      stabilize
    first_step_time_seconds: The time taken for the first step.
    warmup_overhead_seconds: The extra time taken by the warmup steps over as
      many warm steps (see `PerformanceMetrics.warmup_overhead_seconds`), here
      with the last warmup step standing in for a warm step.
    warmup_session_xprof_url: The URL of the xprof session for the warmup run.
  """

  num_warmup_steps: int = 0
  first_step_time_seconds: float = 0.0
  warmup_overhead_seconds: float = 0.0
  warmup_session_xprof_url: str | None = None


@dataclasses.dataclass
class PostWarmupRunResult:
  """Result of the post warmup run.

  Attributes:
    post_warmup_step_time_seconds: The step time in seconds after warmup is
      complete (median across timed steps in `perflab.measure`).
    peak_device_memory_mb: The peak device memory usage in MB for a benchmark
      step (median across isolated post-warmup probe steps excluding
      `post_warmup_hook()` in `perflab.measure`, when supported).
    post_warmup_run_session_xprof_url: The URL of the xprof session for the post
      warmup run.
    average_post_warmup_device_time_seconds: The average device execution time.
    num_post_warmup_steps: The number of timed post-warmup steps.
    pipelined_step_time_seconds: The time per step of the post-warmup steps
      dispatched back to back, or -1.0 if they did not run.
    num_pipelined_steps: The number of pipelined post-warmup steps.
    throughput_step_time_seconds: The step time that the throughput, achieved
      TFLOP/s and MFU come from, or None if they are not reported (see
      `PerformanceMetrics`).
    diagnostics: Optional step-timing, preemption, and memory diagnostics logged
      only in `PERFLAB_RESULT_JSON` records and excluded from `metric_map()`.
  """

  post_warmup_step_time_seconds: float = 0.0
  peak_device_memory_mb: float | None = None
  post_warmup_run_session_xprof_url: str | None = None
  average_post_warmup_device_time_seconds: float = -1.0
  num_post_warmup_steps: int = 0
  pipelined_step_time_seconds: float = -1.0
  num_pipelined_steps: int = 0
  throughput_step_time_seconds: float | None = None
  diagnostics: dict[str, float] | None = None


@dataclasses.dataclass
class MetricsInterface(abc.ABC):
  """Interface for benchmark metrics.

  Attributes:
    e2e_wall_time_seconds: The total wall time of the benchmark.
  """

  e2e_wall_time_seconds: float = 0.0

  @abc.abstractmethod
  def metric_map(self) -> Mapping[str, float]:
    """Returns a map of metrics to be exported to MLCompass."""
    raise NotImplementedError


@dataclasses.dataclass
class PerformanceMetrics(MetricsInterface):
  """Result of a performance benchmark run.

  Attributes:
    num_warmup_steps: The number of warmup steps taken for the cache misses to
      stabilize
    first_step_time_seconds: The time taken for the first step.
    warmup_overhead_seconds: The extra time taken by the warmup steps, which run
      until the cache misses stabilize, over as many warm steps: (wall time of
      the n warmup steps) - n * (wall time of a warm step), at least 0. A warm
      step takes `post_warmup_step_time_seconds` when post-warmup steps ran, or
      the time of the last warmup step if there are no post-warmup steps. For
      example, if warmup takes the 3 steps [15, 10, 10] and the post-warmup
      steps have median 2, the warmup overhead is (15 + 10 + 10) - 2 * 3 = 29
      seconds.
    post_warmup_step_time_seconds: The run time of a benchmark step after warmup
      is complete (median across timed steps in `perflab.measure`).
    num_post_warmup_steps: The number of timed post-warmup steps measured.
    pipelined_step_time_seconds: The time per step of the post-warmup steps
      dispatched back to back, without waiting for each step's result, or -1.0
      if they did not run. It overlaps the host's dispatch of a step with the
      device's execution of the previous ones, as a training or serving loop
      does, so it measures throughput rather than the latency of one step. Only
      exported if measured.
    num_pipelined_steps: The number of pipelined post-warmup steps.
    throughput_step_time_seconds: The step time that the throughput (samples and
      tokens per second), the achieved TFLOP/s and the MFU come from, or None if
      they are not reported. `measure()` sets it to the pipelined step time; to
      None if the stepper supports pipelined steps but they did not run
      (--pipelined_steps=false) or ran out of memory; and to the synchronized
      step time for steppers that do not support pipelined steps yet (decode and
      torchax). The `device_*` variants come from the device time instead.
    peak_device_memory_mb: The peak device memory usage in MB for a benchmark
      step (median across isolated post-warmup probe steps excluding
      `post_warmup_hook()` in `perflab.measure`, when supported).
    warmup_session_xprof_url: The URL of the xprof session for the warmup run.
    post_warmup_run_session_xprof_url: The URL of the xprof session for the post
      warmup run.
    average_post_warmup_device_time_seconds: The average device time of a
      profiled post-warmup step, or -1.0 if it is unavailable (e.g. without
      xprof). An unavailable device time is not exported.
    peak_host_compilation_memory_mb: The peak host memory used by compilation in
      the process, or None if it was not measured. Only exported if measured.
    model_flops_per_step: The model FLOPs of one benchmark step (see
      perflab/flops.py), or None if they were not counted. Only exported, with
      the achieved TFLOP/s derived from it, if counted; the TFLOP/s and MFU are
      not derived from a count of 0.
    peak_flops_per_second: The peak FLOP/s of the device for the benchmark's
      dtype, or None if unknown. The model FLOPs utilization (MFU) is only
      exported if both it and `model_flops_per_step` are known.
    step_work: The work of one step (see perflab/workload.py), or None if it is
      unknown. Only exported, with the throughput derived from it, if known.
    diagnostics: Optional step-timing, preemption, and memory diagnostics logged
      only in `PERFLAB_RESULT_JSON` records and excluded from `metric_map()`.
  """

  num_warmup_steps: int = 0
  first_step_time_seconds: float = 0.0
  warmup_overhead_seconds: float = 0.0
  post_warmup_step_time_seconds: float = 0.0
  num_post_warmup_steps: int = 0
  pipelined_step_time_seconds: float = -1.0
  num_pipelined_steps: int = 0
  throughput_step_time_seconds: float | None = None
  peak_device_memory_mb: float | None = None
  warmup_session_xprof_url: str | None = None
  post_warmup_run_session_xprof_url: str | None = None
  average_post_warmup_device_time_seconds: float = -1.0
  peak_host_compilation_memory_mb: float | None = None
  model_flops_per_step: float | None = None
  peak_flops_per_second: float | None = None
  step_work: workload_lib.StepWork | None = None
  diagnostics: dict[str, float] | None = None

  def _throughput_metrics(self) -> Mapping[str, float]:
    """Returns the workload and the throughput from the wall and device times.

    Throughput is per process: samples or tokens per second of one device.
    `batch_size` and `seq_len` are those of a model replica; the processes of a
    model-parallel replica each export their share of its samples and tokens
    per step, which can be fractional.
    """
    if self.step_work is None:
      return {}
    res = {"batch_size": float(self.step_work.workload.batch_size)}
    if self.step_work.workload.seq_len is not None:
      res["seq_len"] = float(self.step_work.workload.seq_len)
    for unit, per_step in (
        ("samples", self.step_work.samples_per_step),
        ("tokens", self.step_work.tokens_per_step),
    ):
      if per_step is None:
        continue
      res[f"{unit}_per_step"] = float(per_step)
      for prefix, step_time in (
          ("", self.throughput_step_time_seconds),
          ("device_", self.average_post_warmup_device_time_seconds),
      ):
        if step_time is not None and step_time > 0:
          res[f"{prefix}{unit}_per_second"] = per_step / step_time
    return res

  def _flops_metrics(self) -> Mapping[str, float]:
    """Returns the FLOPs metrics, from the wall and the device step times.

    The wall-time achieved TFLOP/s and MFU come from the same step time as the
    throughput (`throughput_step_time_seconds`).

    A count of 0 FLOPs is exported, as it tells that the step's ops have no
    `FlopCounterMode` formula, but no FLOP rates or MFU are derived from it:
    they would read as a real utilization of 0 rather than an unknown one.
    """
    if self.model_flops_per_step is None:
      return {}
    res = {"model_flops_per_step": self.model_flops_per_step}
    if self.model_flops_per_step <= 0:
      return res
    for prefix, step_time in (
        ("", self.throughput_step_time_seconds),
        ("device_", self.average_post_warmup_device_time_seconds),
    ):
      if step_time is None or step_time <= 0:
        continue
      flops_per_second = self.model_flops_per_step / step_time
      res[f"{prefix}achieved_tflops_per_second"] = flops_per_second / 1e12
      if self.peak_flops_per_second:
        res[f"{prefix}mfu"] = flops_per_second / self.peak_flops_per_second
    return res

  def metric_map(self) -> Mapping[str, float]:
    """Returns a map of metrics to be exported to MLCompass."""
    res = {
        "num_warmup_steps": float(self.num_warmup_steps),
        "first_step_time_seconds": self.first_step_time_seconds,
        "first_step_time_ms": _seconds_to_ms(self.first_step_time_seconds),
        "warmup_overhead_seconds": self.warmup_overhead_seconds,
        "warmup_overhead_ms": _seconds_to_ms(self.warmup_overhead_seconds),
        "post_warmup_step_time_seconds": self.post_warmup_step_time_seconds,
        "post_warmup_step_time_ms": _seconds_to_ms(
            self.post_warmup_step_time_seconds
        ),
        "num_post_warmup_steps": float(self.num_post_warmup_steps),
    }
    if self.average_post_warmup_device_time_seconds >= 0:
      res["average_post_warmup_device_time_seconds"] = (
          self.average_post_warmup_device_time_seconds
      )
      res["average_post_warmup_device_time_ms"] = _seconds_to_ms(
          self.average_post_warmup_device_time_seconds
      )
    if self.pipelined_step_time_seconds > 0:
      res["pipelined_step_time_seconds"] = self.pipelined_step_time_seconds
      res["pipelined_step_time_ms"] = _seconds_to_ms(
          self.pipelined_step_time_seconds
      )
      res["num_pipelined_steps"] = float(self.num_pipelined_steps)
    if self.peak_device_memory_mb is not None:
      res["peak_device_memory_mb"] = self.peak_device_memory_mb
    if self.peak_host_compilation_memory_mb is not None:
      res["peak_host_compilation_memory_mb"] = (
          self.peak_host_compilation_memory_mb
      )
    res |= self._flops_metrics()
    res |= self._throughput_metrics()
    return res


@dataclasses.dataclass
class QualityMetrics(MetricsInterface):
  """Metrics for a quality benchmark run.

  Attributes:
    metrics: A map of metrics to be exported to MLCompass.
  """

  metrics: Mapping[str, float] = dataclasses.field(default_factory=dict)

  def metric_map(self) -> Mapping[str, float]:
    """Returns a map of metrics to be exported to MLCompass."""
    return self.metrics
