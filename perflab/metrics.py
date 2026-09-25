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
import dataclasses
from typing import Mapping


def _seconds_to_ms(value: float) -> float:
  """Converts a non-negative duration in seconds to milliseconds."""
  if value >= 0:
    return float(value) * 1000.0
  return value


@dataclasses.dataclass
class WarmupRunResult:
  """Result of the warmup run.

  Attributes:
    num_warmup_steps: The number of warmup steps taken for the cache misses to
      stabilize
    first_step_time_seconds: The time taken for the first step.
    warmup_overhead_seconds: This is the extra time taken to run the benchmark
      to warmup the caches. If it takes n steps for cache misses to stabilize,
      then the warmup overhead is (wall time of n steps) - (wall time of 1 warm
      step) * n. For example, if the cache misses are [100, 120, 130, 130] and
      wall times are [15, 10, 10, 2], then the warmup overhead is (15 + 10 + 10)
      - 2*3 = 29 seconds.
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
    post_warmup_step_time_seconds: The average step time in seconds after the
      warmup is complete.
    peak_device_memory_mb: The peak device memory usage in MB for a benchmark
      step.
    post_warmup_run_session_xprof_url: The URL of the xprof session for the post
      warmup run.
    average_post_warmup_device_time_seconds: The average device execution time.
  """

  post_warmup_step_time_seconds: float = 0.0
  peak_device_memory_mb: float | None = None
  post_warmup_run_session_xprof_url: str | None = None
  average_post_warmup_device_time_seconds: float = -1.0


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
    warmup_overhead_seconds: This is the extra time taken to run the benchmark
      to warmup the caches. If it takes n steps for cache misses to stabilize,
      then the warmup overhead is (wall time of n steps) - (wall time of 1 warm
      step) * n. For example, if the cache misses are [100, 120, 130, 130] and
      wall times are [15, 10, 10, 2], then the warmup overhead is (15 + 10 + 10)
      - 2*3 = 29 seconds.
    post_warmup_step_time_seconds: The average run time of a benchmark step
      after the warmup is complete.
    peak_device_memory_mb: The peak device memory usage in MB for a benchmark
      step.
    warmup_session_xprof_url: The URL of the xprof session for the warmup run.
    post_warmup_run_session_xprof_url: The URL of the xprof session for the post
      warmup run.
  """

  num_warmup_steps: int = 0
  first_step_time_seconds: float = 0.0
  warmup_overhead_seconds: float = 0.0
  post_warmup_step_time_seconds: float = 0.0
  peak_device_memory_mb: float | None = None
  warmup_session_xprof_url: str | None = None
  post_warmup_run_session_xprof_url: str | None = None
  average_post_warmup_device_time_seconds: float = -1.0
  peak_host_compilation_memory_mb: float = 0.0

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
        "average_post_warmup_device_time_seconds": (
            self.average_post_warmup_device_time_seconds
        ),
        "average_post_warmup_device_time_ms": _seconds_to_ms(
            self.average_post_warmup_device_time_seconds
        ),
        "peak_host_compilation_memory_mb": self.peak_host_compilation_memory_mb,
    }
    if self.peak_device_memory_mb is not None:
      res["peak_device_memory_mb"] = self.peak_device_memory_mb
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
