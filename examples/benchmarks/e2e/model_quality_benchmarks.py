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
import torch.multiprocessing as mp
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import quality_utils
from examples.benchmarks.quality_utils.metrics import data_loader
from examples.benchmarks.quality_utils.metrics import perplexity_metric
from examples.benchmarks.quality_utils.models import meta_llama3_quality_benchmark

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


_META_LLAMA_3_2_8B_BENCHMARK_NAME = "meta_llama_3_2_8b"
_META_LLAMA_3_2_70B_BENCHMARK_NAME = "meta_llama_3_2_70b"

_WORLD_SIZE_MAP = {
    benchmark_utils.Platform.GFC_2X2X1: 8,
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
  # Determine world size and device based on platform
  world_size = _WORLD_SIZE_MAP[platform]
  device = benchmark_utils.PLATFORM_DEVICE_MAP[platform]

  # Instantiate the benchmark model
  benchmark_model = (
      meta_llama3_quality_benchmark.DistributedMetaLlama3QualityBenchmarkModel(
          device, world_size, model_config, 2048
      )
  )

  # Metrics
  metrics = [perplexity_metric.PerplexityMetric(max_text_chunk_size=1024)]

  config = quality_utils.QualityBenchmarkConfig(
      supported_platforms=[
          benchmark_utils.Platform.GFC_2X2X1,
          benchmark_utils.Platform.B200_8,
      ],
      benchmark_category=benchmark_utils.BenchmarkCategory.META_LLAMA,
      run_mode=run_mode,
      benchmark_model=benchmark_model,
      metrics=metrics,
      dataset_type=data_loader.DatasetType.WIKITEXT,
  )
  return config


class BenchmarkTest(absltest.TestCase):
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
            platform, "70B", benchmark_utils.RunMode.EAGER_DEFAULT
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


if __name__ == "__main__":
  mp.set_start_method("spawn")
  # g3_multiprocessing is required to run absltest.main() in a multiprocess
  # environment. It doesn't affect single process runs.
  # See: go/g3_multiprocessing#resolution.
  g3_multiprocessing.handle_test_main(absltest.main)
