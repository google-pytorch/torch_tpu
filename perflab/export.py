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

from collections.abc import Mapping
import dataclasses
import datetime
import json
import math
from typing import Any

from absl import logging
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


# Log-line prefix for the machine-readable JSON record that `export` and
# `export_skip` emit for every case, so results are structured and parseable
# from test and runner logs whether or not an external exporter is configured.
# If a log contains more than one record for a case, the last record per
# (test_name, microbenchmark_name) wins.
RESULT_LOG_PREFIX = "PERFLAB_RESULT_JSON"

# The version of the record format logged with `RESULT_LOG_PREFIX`. Bump it
# when a change to the format can break its readers.
_RESULTS_SCHEMA_VERSION = 1


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


def _finite_or_none(value: Any) -> Any:
  """Returns `value`, or None for a NaN or infinite float, which JSON lacks."""
  if isinstance(value, float) and not math.isfinite(value):
    return None
  return value


def _test_name(
    *,
    spec_name: str,
    platform: target_lib.Platform,
    framework: mode_lib.Framework,
    run_mode: common.RunMode,
) -> str:
  """Returns the test name that the export adapter exports a case under.

  The name, `torch_tpu/{platform}/{spec}/{run_mode}/{framework}`, identifies
  the platform, run mode and framework of a record.

  Args:
    spec_name: The name of the benchmark spec.
    platform: The hardware platform of the run.
    framework: The execution framework.
    run_mode: The execution run mode.
  """
  return (
      f"torch_tpu/{platform.value}/{spec_name}/{run_mode.value}/"
      f"{framework.value}"
  )


def _log_result(
    *,
    spec_name: str,
    microbenchmark_name: str | None,
    batch_size: int | None,
    status: str,
    succeeded: bool,
    metrics: Mapping[str, Any],
    test_name: str,
    warmup_xprof_url: str | None = None,
    post_warmup_xprof_url: str | None = None,
) -> None:
  """Logs a machine-readable JSON record prefixed with `RESULT_LOG_PREFIX`.

  Any failure to build or log the record is logged rather than raised: emitting
  the structured log line must never fail a benchmark.

  Args:
    spec_name: The name of the benchmark spec.
    microbenchmark_name: The name of the microbenchmark, if any.
    batch_size: The batch size of a batch sweep case, or None.
    status: One of "ok", "failed", "skipped" and "oom".
    succeeded: Whether the record counts as succeeded, as exported.
    metrics: The exported metrics. NaN and infinite values are written as null.
    test_name: The test name of the run, as exported by `_test_name`.
    warmup_xprof_url: The XProf URL of the warmup run, if profiled.
    post_warmup_xprof_url: The XProf URL of the post-warmup run, if profiled.
  """
  try:
    record = {
        "schema_version": _RESULTS_SCHEMA_VERSION,
        "spec_name": spec_name,
        "microbenchmark_name": microbenchmark_name,
        "batch_size": batch_size,
        "status": status,
        "succeeded": succeeded,
        "metrics": {
            name: _finite_or_none(value) for name, value in metrics.items()
        },
        "test_name": test_name,
        "warmup_xprof_url": warmup_xprof_url,
        "post_warmup_xprof_url": post_warmup_xprof_url,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    line = json.dumps(record, sort_keys=True, allow_nan=False)
    logging.info("%s %s", RESULT_LOG_PREFIX, line)
  except Exception:  # pylint: disable=broad-exception-caught
    logging.exception("Failed to log the result of %s", spec_name)


def _status(data: BenchmarkData) -> str:
  """Returns the `_log_result` status of `data`."""
  if data.out_of_memory:
    return "oom"
  if data.skipped:
    return "skipped"
  return "ok" if data.succeeded else "failed"


def export(data: BenchmarkData) -> None:
  """Exports benchmark results via export_adapter and logs a JSON record.

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
  # Logged first, so that a failed export still leaves a record in the log.
  _log_result(
      spec_name=data.spec_name,
      microbenchmark_name=data.microbenchmark_name,
      batch_size=data.batch_size,
      status=_status(data),
      succeeded=data.succeeded,
      metrics=metrics,
      test_name=_test_name(
          spec_name=data.spec_name,
          platform=data.platform,
          framework=data.framework,
          run_mode=data.run_mode,
      ),
      warmup_xprof_url=getattr(data.metrics, "warmup_session_xprof_url", None),
      post_warmup_xprof_url=getattr(
          data.metrics, "post_warmup_run_session_xprof_url", None
      ),
  )
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
  # Like the skip record of the export adapter, which counts as succeeded.
  _log_result(
      spec_name=spec_name,
      microbenchmark_name=None,
      batch_size=None,
      status="skipped",
      succeeded=True,
      metrics={SKIPPED_METRIC: 1.0},
      test_name=_test_name(
          spec_name=spec_name,
          platform=platform,
          framework=framework,
          run_mode=run_mode,
      ),
  )
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
