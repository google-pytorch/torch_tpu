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

"""Benchmarks for model quality."""

from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
import torch_tpu
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import quality_utils
from examples.benchmarks.quality_utils.metrics import data_loader
from examples.benchmarks.quality_utils.metrics import perplexity_metric
from examples.benchmarks.quality_utils.models import meta_llama3_quality_benchmark
from examples.benchmarks.quality_utils.models import qwen3_vl_quality_benchmark
from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

_QWEN3_VL_BENCHMARK_NAME = "qwen3_vl"
_QWEN3_VL_EMBEDDING_8B_BENCHMARK_NAME = "qwen3_vl_embedding_8b"


_META_LLAMA_3_2_8B_BENCHMARK_NAME = "meta_llama_3_2_8b"
_META_LLAMA_3_2_70B_BENCHMARK_NAME = "meta_llama_3_2_70b"

_WORLD_SIZE_MAP = {
    benchmark_utils.Platform.GFC_2X2X1: 8,
    benchmark_utils.Platform.GFC_2X2X2: 16,
    benchmark_utils.Platform.B200_4: 4,
    benchmark_utils.Platform.B200_8: 8,
}


def _distributed_meta_llama_3_benchmark_config(
    platform: benchmark_utils.Platform,
    model_config: str,
    run_mode: benchmark_utils.RunMode,
) -> quality_utils.QualityBenchmarkConfig:
  """Sets up the benchmark config for distributed Meta Llama-3.2.

  Distributed benchmarks are only supported on GFC_2X2X1 and B200_8 platforms.

  Args:
    platform: The platform to run the benchmark on.
    model_config: The model config to use.
    run_mode: The mode to run the benchmark in.
  """
  supported_platforms = [
      benchmark_utils.Platform.GFC_2X2X1,
      benchmark_utils.Platform.B200_8,
  ]
  config_platform = platform
  if platform not in supported_platforms:
    config_platform = benchmark_utils.Platform.GFC_2X2X1

  # Determine world size and device based on platform
  world_size = _WORLD_SIZE_MAP[config_platform]
  device = benchmark_utils.PLATFORM_DEVICE_MAP[config_platform]

  # Instantiate the benchmark model
  benchmark_model = (
      meta_llama3_quality_benchmark.DistributedMetaLlama3QualityBenchmarkModel(
          device, world_size, model_config, 2048
      )
  )

  # Metrics
  metrics = [perplexity_metric.PerplexityMetric()]

  config = quality_utils.QualityBenchmarkConfig(
      supported_platforms=supported_platforms,
      benchmark_category=benchmark_utils.BenchmarkCategory.META_LLAMA,
      run_mode=run_mode,
      benchmark_model=benchmark_model,
      metrics=metrics,
      dataset_type=data_loader.DatasetType.WIKITEXT,
  )
  return config


def _distributed_qwen3_vl_benchmark_config(
    platform: benchmark_utils.Platform,
    run_mode: benchmark_utils.RunMode,
    model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct",
) -> quality_utils.QualityBenchmarkConfig:
  """Sets up the benchmark config for distributed Qwen3-VL.

  Args:
    platform: The platform to run the benchmark on.
    run_mode: The mode to run the benchmark in.
    model_name: The model name to use.
  """
  device = benchmark_utils.PLATFORM_DEVICE_MAP[platform]

  benchmark_model = (
      qwen3_vl_quality_benchmark.DistributedQwen3VLQualityBenchmarkModel(
          model_name, device
      )
  )

  metrics = [perplexity_metric.VLPerplexityMetric()]

  config = quality_utils.QualityBenchmarkConfig(
      supported_platforms=[
          benchmark_utils.Platform.GFC_2X2X1,
          benchmark_utils.Platform.GFC_2X2X2,
          benchmark_utils.Platform.V6E_2X2,
          benchmark_utils.Platform.V6E_2X4,
          benchmark_utils.Platform.V6E_4X4,
          benchmark_utils.Platform.V6E_4X8,
      ],
      benchmark_category=benchmark_utils.BenchmarkCategory.QWEN,
      run_mode=run_mode,
      benchmark_model=benchmark_model,
      metrics=metrics,
      dataset_type=data_loader.DatasetType.VQAV2,
  )
  return config


class BenchmarkTest(parameterized.TestCase):
  """Tests for end-to-end quality benchmarks."""

  def run_benchmark_test(
      self,
      config: quality_utils.QualityBenchmarkConfig,
      benchmark_name: str,
      distributed: bool,
  ) -> None:
    """Runs a benchmark test.

    Args:
      config: The benchmark config.
      benchmark_name: The name of the benchmark.
      distributed: Whether the benchmark is distributed.
    """
    platform = benchmark_utils.PLATFORM.value
    if platform not in config.supported_platforms:
      self.skipTest(
          f"Platform {benchmark_utils.PLATFORM.value} not in"
          f" {config.supported_platforms}"
      )

    quality_utils.run_benchmark(
        config=config,
        test_method_name=self._testMethodName,
        benchmark_name=benchmark_name,
        distributed=distributed,
    )

  def test_quality_distributed_meta_llama_3_2_8b_eager_forward(self):
    """Tests the forward pass of Meta Llama-3.2-8B in eager mode."""
    platform = benchmark_utils.PLATFORM.value
    self.run_benchmark_test(
        _distributed_meta_llama_3_benchmark_config(
            platform, "8B", benchmark_utils.RunMode.EAGER_DEFAULT
        ),
        _META_LLAMA_3_2_8B_BENCHMARK_NAME,
        True,
    )

  def test_quality_distributed_meta_llama_3_2_70b_eager_forward(self):
    """Tests the forward pass of Meta Llama-3.2-70B in eager mode."""
    platform = benchmark_utils.PLATFORM.value
    self.run_benchmark_test(
        _distributed_meta_llama_3_benchmark_config(
            platform,
            "70B",
            benchmark_utils.RunMode.EAGER_DEFAULT,
        ),
        _META_LLAMA_3_2_70B_BENCHMARK_NAME,
        True,
    )

  def test_quality_distributed_meta_llama_3_2_8b_compiled_forward(self):
    """Tests the forward pass of Meta Llama-3.2-8B in compiled mode."""
    platform = benchmark_utils.PLATFORM.value
    self.run_benchmark_test(
        _distributed_meta_llama_3_benchmark_config(
            platform, "8B", benchmark_utils.RunMode.COMPILED
        ),
        _META_LLAMA_3_2_8B_BENCHMARK_NAME,
        True,
    )

  def test_quality_distributed_meta_llama_3_2_70b_compiled_forward(self):
    """Tests the forward pass of Meta Llama-3.2-70B in compiled mode."""
    platform = benchmark_utils.PLATFORM.value
    self.run_benchmark_test(
        _distributed_meta_llama_3_benchmark_config(
            platform, "70B", benchmark_utils.RunMode.COMPILED
        ),
        _META_LLAMA_3_2_70B_BENCHMARK_NAME,
        True,
    )

  def test_quality_qwen3_vl_eager_forward(self):
    """Tests the quality of Qwen3-VL in eager mode."""
    platform = benchmark_utils.PLATFORM.value
    device = benchmark_utils.PLATFORM_DEVICE_MAP[platform]

    benchmark_model = qwen3_vl_quality_benchmark.Qwen3VLQualityBenchmarkModel(
        "Qwen/Qwen3-VL-235B-A22B-Instruct", device
    )

    metrics = [perplexity_metric.PerplexityMetric()]

    config = quality_utils.QualityBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.META_LLAMA,
        run_mode=benchmark_utils.RunMode.EAGER_DEFAULT,
        benchmark_model=benchmark_model,
        metrics=metrics,
        dataset_type=data_loader.DatasetType.WIKITEXT,
    )

    self.run_benchmark_test(config, _QWEN3_VL_BENCHMARK_NAME, distributed=False)

  def test_quality_qwen3_vl_compiled_forward(self):
    """Tests the quality of Qwen3-VL in compiled mode."""
    platform = benchmark_utils.PLATFORM.value
    device = benchmark_utils.PLATFORM_DEVICE_MAP[platform]

    benchmark_model = qwen3_vl_quality_benchmark.Qwen3VLQualityBenchmarkModel(
        "Qwen/Qwen3-VL-235B-A22B-Instruct", device
    )

    metrics = [perplexity_metric.PerplexityMetric()]

    config = quality_utils.QualityBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.META_LLAMA,
        run_mode=benchmark_utils.RunMode.COMPILED,
        benchmark_model=benchmark_model,
        metrics=metrics,
        dataset_type=data_loader.DatasetType.WIKITEXT,
    )

    self.run_benchmark_test(config, _QWEN3_VL_BENCHMARK_NAME, distributed=False)

  @parameterized.parameters(
      benchmark_utils.RunMode.EAGER_DEFAULT,
      benchmark_utils.RunMode.COMPILED,
  )
  def test_quality_distributed_qwen3_vl(self, run_mode):
    """Tests the quality of Qwen3-VL in distributed environment."""
    platform = benchmark_utils.PLATFORM.value
    self.run_benchmark_test(
        _distributed_qwen3_vl_benchmark_config(
            platform, run_mode, "Qwen/Qwen3-VL-235B-A22B-Instruct"
        ),
        _QWEN3_VL_BENCHMARK_NAME,
        distributed=True,
    )

  def test_quality_qwen3_vl_embedding_8b_eager_forward(self):
    """Tests the quality of Qwen3-VL-Embedding-8B in eager mode."""
    platform = benchmark_utils.PLATFORM.value
    device = benchmark_utils.PLATFORM_DEVICE_MAP[platform]

    benchmark_model = qwen3_vl_quality_benchmark.Qwen3VLQualityBenchmarkModel(
        "Qwen/Qwen3-VL-Embedding-8B", device
    )

    metrics = [perplexity_metric.PerplexityMetric()]

    config = quality_utils.QualityBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
            benchmark_utils.Platform.TORCH_CPU,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.QWEN,
        run_mode=benchmark_utils.RunMode.EAGER_DEFAULT,
        benchmark_model=benchmark_model,
        metrics=metrics,
        dataset_type=data_loader.DatasetType.WIKITEXT,
    )

    self.run_benchmark_test(
        config, _QWEN3_VL_EMBEDDING_8B_BENCHMARK_NAME, distributed=False
    )

  def test_quality_qwen3_vl_embedding_8b_compiled_forward(self):
    """Tests the quality of Qwen3-VL-Embedding-8B in compiled mode."""
    platform = benchmark_utils.PLATFORM.value
    device = benchmark_utils.PLATFORM_DEVICE_MAP[platform]

    benchmark_model = qwen3_vl_quality_benchmark.Qwen3VLQualityBenchmarkModel(
        "Qwen/Qwen3-VL-Embedding-8B", device
    )

    metrics = [perplexity_metric.PerplexityMetric()]

    config = quality_utils.QualityBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
            benchmark_utils.Platform.TORCH_CPU,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.QWEN,
        run_mode=benchmark_utils.RunMode.COMPILED,
        benchmark_model=benchmark_model,
        metrics=metrics,
        dataset_type=data_loader.DatasetType.WIKITEXT,
    )

    self.run_benchmark_test(
        config, _QWEN3_VL_EMBEDDING_8B_BENCHMARK_NAME, distributed=False
    )

  @parameterized.parameters(
      benchmark_utils.RunMode.EAGER_DEFAULT,
      benchmark_utils.RunMode.COMPILED,
  )
  def test_quality_distributed_qwen3_vl_embedding_8b(self, run_mode):
    """Tests the quality of Qwen3-VL-Embedding-8B in distributed environment."""
    platform = benchmark_utils.PLATFORM.value
    self.run_benchmark_test(
        _distributed_qwen3_vl_benchmark_config(
            platform, run_mode, "Qwen/Qwen3-VL-Embedding-8B"
        ),
        _QWEN3_VL_EMBEDDING_8B_BENCHMARK_NAME,
        distributed=True,
    )


if __name__ == "__main__":
  import os
  import sys

  mp.set_start_method("spawn")

  # Set env var from argv if present to support absltest filtering
  for i, arg in enumerate(sys.argv):
    if arg.startswith("--test_filter="):
      os.environ["TESTBRIDGE_TEST_ONLY"] = arg.split("=", 1)[1]
      sys.argv.pop(i)
      break

  # g3_multiprocessing is required to run absltest.main() in a multiprocess
  # environment. It doesn't affect single process runs.
  # See: go/g3_multiprocessing#resolution.
  g3_multiprocessing.handle_test_main(absltest.main)
