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

"""Utilities for exporting benchmark results to MLCompass."""

from absl import logging
from torch_tpu._internal.utils import log_utils
from examples.benchmarks.e2e import common
from examples.benchmarks.e2e.harness import metrics as metrics_lib
from torch_tpu._internal.benchmarks import benchmark_adapter


log_utils.log_to_stderr()


TEAM_NAME = "torch_tpu"


def get_mlcompass_test_name(
    platform: common.Platform,
    test_method_name: str,
    benchmark_name: str,
) -> str:
  """Constructs a MLCompass test target name."""
  return f"{TEAM_NAME}/{platform.value}/{test_method_name}/{benchmark_name}"


def export_to_mlcompass(
    platform: common.Platform,
    metrics: metrics_lib.MetricsInterface | None,
    base_cl: str | None,
    mlcompass_tracking_id: str,
    mlcompass_execution_mode: str,
    *,
    test_method_name: str,
    benchmark_name: str,
    microbenchmark_name: str | None = None,
    succeeded: bool = True,
    pending_cl: str | None = None,
    benchmark_group: str | None = None,
) -> None:
  """Exports benchmark results to mlcompass.

  Args:
    platform: The platform the benchmark was run on.
    metrics: The performance benchmark results.
    base_cl: The base CL used for the benchmark run.
    mlcompass_tracking_id: The UUID used to track all the generated metrics by a
      single invocation.
    mlcompass_execution_mode: The execution mode of the mlcompass run. This is
      used to distinguish between cbuild and oneshot runs.
    test_method_name: The name of the test method being benchmarked.
    benchmark_name: The name of the benchmark.
    microbenchmark_name: The name of the microbenchmark. This is used to export
      microbenchmark results to MLCompass when a benchmark test method is
      composed of multiple microbenchmarks. See
      go/mlcompass-microbenchmark-guide for more details.
    succeeded: Whether the benchmark succeeded or failed.
    pending_cl: The pending CL used for the benchmark run.
    benchmark_group: The benchmark group (control or experiment).
  """

  assert mlcompass_tracking_id is not None
  logging.info(
      "Exporting results to mlcompass for tracking-id: %s, test-name: %s,"
      " benchmark-name: %s, microbenchmark-name: %s",
      mlcompass_tracking_id,
      test_method_name,
      benchmark_name,
      microbenchmark_name,
  )
  # MLCompass expects metrics as mapping from string (metric name) -> float
  # (metric value). It displays a graph for each metric name with metric value
  # across time. We need to make sure that the strings are uniform across all
  # benchmarks for displaying on dashboards but there are no restrictions on
  # what the names should be.
  metric_map = metrics.metric_map() if metrics is not None else {}
  wall_time = metrics.e2e_wall_time_seconds if metrics is not None else None
  test_name = get_mlcompass_test_name(
      platform, test_method_name, benchmark_name
  )

  mlcompass_run_tags = None
  if benchmark_group:
    mlcompass_run_tags = (benchmark_group,)

  benchmark_data = benchmark_adapter.BenchmarkData(
      test_name=test_name,
      wall_time=wall_time,
      base_cl=base_cl,
      pending_cl=pending_cl,
      metrics=metric_map,
      mlcompass_tracking_id=mlcompass_tracking_id,
      mlcompass_execution_mode=mlcompass_execution_mode,
      team_name=TEAM_NAME,
      succeeded=succeeded,
      mlcompass_run_tags=mlcompass_run_tags,
  )
  if microbenchmark_name:
    benchmark_data.micro_result_key = microbenchmark_name
  benchmark_adapter.export_benchmark_results(data=benchmark_data)
