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

"""Data structures for benchmark export."""

import dataclasses

from torch_tpu._internal.benchmarks import export_adapter
from perflab import common
from perflab import known_failures
from perflab import metrics as metrics_lib
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import target as target_lib

# The metric that a batch sweep case that ran out of device memory exports, with
# its `batch_size`, instead of its performance metrics.
OUT_OF_MEMORY_METRIC = "out_of_memory"

# The metric that a batch sweep case that was not run exports, with its
# `batch_size`, because a smaller batch size of its benchmark ran out of device
# memory. Like the skip records of `export_skip`, but keyed by the batch size.
SKIPPED_METRIC = "skipped"


def batch_size_key(batch_size: int) -> str:
  """Returns the microbenchmark name of a batch sweep case, e.g. `bs16`."""
  return f"bs{batch_size}"


@dataclasses.dataclass
class BenchmarkData:
  """Data container for benchmark results.

  Attributes:
    spec_name: The name of the benchmark spec.
    platform: The hardware platform the benchmark was run on.
    framework: The execution framework (e.g. Framework.TORCH,
      Framework.TORCHAX).
    run_mode: The execution run mode (e.g. RunMode.EAGER_DEFAULT,
      RunMode.COMPILED).
    succeeded: Whether the benchmark run succeeded.
    metrics: The benchmark metrics collected during the run.
    microbenchmark_name: The name of the microbenchmark if the benchmark is
      composed of multiple microbenchmarks. Batch sweep cases use their batch
      size (`batch_size_key`).
    batch_size: The batch size of a batch sweep case, or None.
    out_of_memory: Whether the batch sweep case ran out of device memory at
      `batch_size`. It then exports `OUT_OF_MEMORY_METRIC` and its batch size
      instead of its metrics, and counts as succeeded: running out of memory
      bounds the batch sizes that a benchmark fits, it is not a failure.
    skipped: Whether the batch sweep case was not run, because a smaller batch
      size of the benchmark ran out of device memory. It then exports
      `SKIPPED_METRIC` and its batch size instead of its metrics, so that an
      assumed out of memory is not exported as a measured one.
  """

  spec_name: str
  platform: target_lib.Platform
  framework: mode_lib.Framework
  run_mode: common.RunMode
  succeeded: bool
  metrics: metrics_lib.MetricsInterface
  microbenchmark_name: str | None = None
  batch_size: int | None = None
  out_of_memory: bool = False
  skipped: bool = False


def _microbenchmark_name(
    spec: registry_lib.BenchmarkSpec, batch_size: int | None
) -> str | None:
  """Returns the key that a case exports its results under, if any.

  Args:
    spec: The benchmark of the case; its `microbenchmark_name` keys all of its
      results.
    batch_size: The batch size of a batch sweep case, which also keys its
      results (`bs<N>`), or None.
  """
  if batch_size is None:
    return spec.microbenchmark_name
  key = batch_size_key(batch_size)
  return (
      f"{spec.microbenchmark_name}_{key}" if spec.microbenchmark_name else key
  )


def make_benchmark_data(
    spec: registry_lib.BenchmarkSpec,
    *,
    platform: target_lib.Platform,
    framework: mode_lib.Framework,
    run_mode: common.RunMode,
    succeeded: bool,
    metrics: metrics_lib.MetricsInterface | None = None,
    batch_size: int | None = None,
    out_of_memory: bool = False,
    skipped: bool = False,
) -> BenchmarkData:
  """Returns the record of a case of `spec`, keyed by its microbenchmark name.

  Args:
    spec: The benchmark of the case.
    platform: The hardware platform the case ran on.
    framework: The execution framework.
    run_mode: The execution run mode.
    succeeded: Whether the case succeeded.
    metrics: The metrics of the case, if it measured any.
    batch_size: The batch size of a batch sweep case, which extends the
      microbenchmark name of `spec` (`batch_size_key`), or None.
    out_of_memory: See `BenchmarkData.out_of_memory`.
    skipped: See `BenchmarkData.skipped`.
  """
  return BenchmarkData(
      spec_name=spec.name,
      platform=platform,
      framework=framework,
      run_mode=run_mode,
      succeeded=succeeded,
      metrics=metrics or metrics_lib.PerformanceMetrics(),
      microbenchmark_name=_microbenchmark_name(spec, batch_size),
      batch_size=batch_size,
      out_of_memory=out_of_memory,
      skipped=skipped,
  )


def export(data: BenchmarkData) -> None:
  """Exports benchmark results to MLCompass via export_adapter.

  Args:
    data: The benchmark data to export.
  """
  if data.out_of_memory or data.skipped:
    metrics = {
        OUT_OF_MEMORY_METRIC if data.out_of_memory else SKIPPED_METRIC: 1.0
    }
    if data.batch_size is not None:
      metrics["batch_size"] = float(data.batch_size)
    wall_time = None
  else:
    metrics = data.metrics.metric_map()
    wall_time = data.metrics.e2e_wall_time_seconds
  export_adapter.export_benchmark_results(
      spec_name=data.spec_name,
      platform=data.platform.value,
      framework=data.framework.value,
      run_mode=data.run_mode.value,
      metrics=metrics,
      succeeded=data.succeeded,
      wall_time=wall_time,
      microbenchmark_name=data.microbenchmark_name,
  )


def export_skip(
    *,
    spec_name: str,
    platform: target_lib.Platform,
    framework: mode_lib.Framework,
    run_mode: common.RunMode,
) -> None:
  """Exports that a benchmark case was skipped as a known failure.

  A skipped case reports no result, so without this record dashboards would
  keep showing its last failure until it goes stale.

  Args:
    spec_name: The name of the benchmark spec.
    platform: The hardware platform the benchmark was skipped on.
    framework: The execution framework.
    run_mode: The execution run mode.
  """
  export_adapter.export_benchmark_skip(
      spec_name=spec_name,
      platform=platform.value,
      framework=framework.value,
      run_mode=run_mode.value,
  )


def export_known_failure_skip(
    *,
    spec_name: str,
    platform: target_lib.Platform,
    framework: mode_lib.Framework,
    run_mode: common.RunMode,
) -> None:
  """Exports a skip record for a skipped case if it is a known failure.

  The record lets dashboards drop the last failure of a benchmark that became
  a known failure. Cases skipped for other reasons are not recorded, as most of
  them never ran.

  Args:
    spec_name: The name of the skipped benchmark spec.
    platform: The hardware platform the case was skipped on.
    framework: The execution framework.
    run_mode: The skipped run mode.
  """
  failure_reason = known_failures.get_known_failure_reason(
      spec_name,
      framework=framework,
      device_kind=target_lib.make_target(platform).device_kind,
      run_mode=run_mode,
  )
  if failure_reason is None:
    return
  export_skip(
      spec_name=spec_name,
      platform=platform,
      framework=framework,
      run_mode=run_mode,
  )
