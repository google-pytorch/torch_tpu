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

"""Utilities for running end-to-end quality benchmarks."""

import dataclasses
from typing import Sequence

from absl import logging
from torch import distributed as dist
from torch.google import distributed as g3_distributed
from torch_tpu._internal.distributed import gpu_env
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import log_utils
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import mlcompass_utils
from examples.benchmarks.quality_utils import quality_benchmark_model
from examples.benchmarks.quality_utils.metrics import data_loader


log_utils.log_to_stderr()


N_PROC_MAP = {
    benchmark_utils.Platform.B200_4: 4,
    benchmark_utils.Platform.B200_8: 8,
    benchmark_utils.Platform.GFC_2X2X1: 8,
}


@dataclasses.dataclass
class QualityBenchmarkConfig:
  """The config for a quality benchmark.

  Attributes:
    supported_platforms: The platforms the benchmark supports as defined in
      benchmark_utils.Platform. If the current platform is not in this Sequence,
      the test will be skipped.
    benchmark_category: The category of the benchmark as defined by
      benchmark_utils.BenchmarkCategory. This defines how to get the model and
      example inputs and the benchmark function to run.
    run_mode: The mode to run the benchmark in as defined by
      benchmark_utils.RunMode. This is used to set environment variables.
    benchmark_model: The quality benchmark model to use.
    metrics: The Sequence of metrics to use.
    dataset_type: The dataset type to use as defined by data_loader.DatasetType.
  """

  supported_platforms: Sequence[benchmark_utils.Platform]
  benchmark_category: benchmark_utils.BenchmarkCategory
  run_mode: benchmark_utils.RunMode
  benchmark_model: quality_benchmark_model.QualityBenchmarkModel
  # LINT.IfChange
  metrics: Sequence[quality_benchmark_model.MetricProducer]
  # LINT.ThenChange(../../../g3doc/benchmarking.md)
  dataset_type: data_loader.DatasetType


def _run_single_process_benchmark(
    rank: int,
    world_size: int,
    config: QualityBenchmarkConfig,
    test_method_name: str,
    benchmark_name: str,
) -> None:
  platform = benchmark_utils.PLATFORM.value
  device = benchmark_utils.get_torch_device(platform)
  # Seed random number generators for reproducibility. This should be done after
  # initializing the device.
  benchmark_utils.seed_rngs()
  data_iterator = data_loader.get_dataset_loader(config.dataset_type)

  benchmark_succeeded = True
  result = None
  benchmark_exception = None
  try:
    result = benchmark_utils.run_quality_benchmark(
        config.benchmark_model,
        benchmark_utils.is_torch_compile(config.run_mode),
        data_iterator,
        config.metrics,
        device,
    )

    logging.info(
        "Test: %s, benchmark: %s, platform: %s, rank(0-indexed): %s,"
        " world_size: %s, metrics: %s",
        test_method_name,
        benchmark_name,
        platform,
        rank,
        world_size,
        result.metrics,
    )
  except Exception as e:
    logging.exception("Benchmark failed: %s", e)
    benchmark_succeeded = False
    benchmark_exception = e

  # Only export results from the rank 0 process to avoid duplicate entries in
  # MLCompass.
  if benchmark_utils.MLCOMPASS_TRACKING_ID.value and rank == 0:
    mlcompass_utils.export_to_mlcompass(
        platform,
        result,
        benchmark_utils.BASE_CL.value,
        benchmark_utils.MLCOMPASS_TRACKING_ID.value,
        benchmark_utils.MLCOMPASS_EXECUTION_MODE.value,
        test_method_name=test_method_name,
        benchmark_name=benchmark_name,
        succeeded=benchmark_succeeded,
        pending_cl=benchmark_utils.PENDING_CL.value,
        benchmark_group=benchmark_utils.BENCHMARK_GROUP.value,
    )

  if not benchmark_succeeded and benchmark_exception is not None:
    raise benchmark_exception


def _run_torch_tpu_worker(
    config: QualityBenchmarkConfig, test_method_name: str, benchmark_name: str
):
  dist.init_process_group(backend="tpu_dist")
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  _run_single_process_benchmark(
      rank, world_size, config, test_method_name, benchmark_name
  )
  dist.destroy_process_group()


def _run_distributed_benchmark(
    config: QualityBenchmarkConfig,
    *,
    test_method_name: str,
    benchmark_name: str,
) -> None:
  """Runs the benchmark for the given config."""
  if benchmark_utils.PLATFORM.value == benchmark_utils.Platform.GFC_2X2X1:
    singlehost_wrapper.prepare_tpu_environment()
    run_worker = singlehost_wrapper.tpu_env_wrapper(
        _run_torch_tpu_worker,
        world_size=8,
    )
    g3_distributed.torchrun(
        run_worker,
        nproc_per_node=N_PROC_MAP[benchmark_utils.PLATFORM.value],
    )(
        config,
        test_method_name,
        benchmark_name,
    )
  elif (
      benchmark_utils.PLATFORM_DEVICE_MAP[benchmark_utils.PLATFORM.value]
      == "cuda"
  ):
    g3_distributed.torchrun(
        gpu_env.run_in_workers,
        nproc_per_node=N_PROC_MAP[benchmark_utils.PLATFORM.value],
    )(
        _run_single_process_benchmark,
        config,
        test_method_name,
        benchmark_name,
    )


def run_benchmark(
    config: QualityBenchmarkConfig,
    *,
    test_method_name: str,
    benchmark_name: str,
    distributed: bool,
) -> None:
  """Runs the benchmark for the given config."""
  if distributed:
    _run_distributed_benchmark(
        config,
        test_method_name=test_method_name,
        benchmark_name=benchmark_name,
    )
  else:
    _run_single_process_benchmark(
        0,
        1,
        config,
        test_method_name=test_method_name,
        benchmark_name=benchmark_name,
    )
