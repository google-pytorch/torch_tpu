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
from perflab import metrics as metrics_lib
from perflab import mode as mode_lib
from perflab import target as target_lib


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
      composed of multiple microbenchmarks.
  """

  spec_name: str
  platform: target_lib.Platform
  framework: mode_lib.Framework
  run_mode: common.RunMode
  succeeded: bool
  metrics: metrics_lib.MetricsInterface
  microbenchmark_name: str | None = None


def export(data: BenchmarkData) -> None:
  """Exports benchmark results to MLCompass via export_adapter.

  Args:
    data: The benchmark data to export.
  """
  export_adapter.export_benchmark_results(
      spec_name=data.spec_name,
      platform=data.platform.value,
      framework=data.framework.value,
      run_mode=data.run_mode.value,
      metrics=data.metrics.metric_map(),
      succeeded=data.succeeded,
      wall_time=data.metrics.e2e_wall_time_seconds,
      microbenchmark_name=data.microbenchmark_name,
  )
