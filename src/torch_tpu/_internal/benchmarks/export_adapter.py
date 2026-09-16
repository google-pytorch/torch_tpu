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

"""Export adapter for OSS."""

import os
from typing import Mapping

from absl import logging
from tensorboardX import writer

_TENSORBOARD_OUTPUT_ENV_VAR = "TENSORBOARD_OUTPUT_DIR"
_TEAM_NAME = "torch_tpu"


def export_benchmark_results(
    spec_name: str,
    platform: str,
    framework: str,
    run_mode: str,
    metrics: Mapping[str, float | None],
    *,
    succeeded: bool = True,
    wall_time: float | None = None,
    microbenchmark_name: str | None = None,
) -> None:
  """Exports benchmark results to TensorBoard.

  Args:
    spec_name: The name of the benchmark spec.
    platform: The hardware platform the benchmark was run on.
    framework: The execution framework.
    run_mode: The execution run mode.
    metrics: Mapping of metric name to metric value.
    succeeded: Whether the benchmark run succeeded.
    wall_time: Optional total wall time in seconds.
    microbenchmark_name: Optional name of the microbenchmark.
  """
  tblog_dir = os.environ.get(_TENSORBOARD_OUTPUT_ENV_VAR, None)
  if not tblog_dir:
    return

  test_name = f"{_TEAM_NAME}/{platform}/{spec_name}/{run_mode}/{framework}"

  if microbenchmark_name:
    test_name = f"{test_name}/{microbenchmark_name}"

  try:
    os.makedirs(tblog_dir, exist_ok=True)
    with writer.SummaryWriter(log_dir=tblog_dir) as tb_writer:
      if succeeded:
        # Only export metrics if the benchmark succeeded. Otherwise, they are
        # likely to trigger incorrect alerts.
        for metric_name, value in metrics.items():
          # Skip metrics that are None (e.g. peak_host_compilation_memory on non-TPU platforms).
          if value is not None:
            metric_full_name = f"{test_name}/{metric_name}"
            tb_writer.add_scalar(metric_full_name, value, global_step=0)
        if wall_time is not None:
          tb_writer.add_scalar(
              f"{test_name}/wall_time",
              wall_time,
              global_step=0,
          )
      tb_writer.add_scalar(
          f"{test_name}/succeeded",
          int(succeeded),
          global_step=0,
      )
  except (OSError, IOError):
    logging.exception("Error writing TensorBoard logs")
