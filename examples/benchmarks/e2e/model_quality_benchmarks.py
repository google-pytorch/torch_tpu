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

from typing import Any

from absl.testing import absltest
import torch
import torch.multiprocessing as mp
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import quality_utils
from examples.benchmarks.quality_utils import quality_benchmark_model
from examples.benchmarks.quality_utils.metrics import data_loader
from examples.benchmarks.quality_utils.metrics import perplexity_metric
from examples.benchmarks.quality_utils.models import llama3_2_1b_quality_benchmark
from examples.benchmarks.quality_utils.models import meta_llama3_quality_benchmark
from examples.benchmarks.quality_utils.models import qwen3_1_7b_quality_benchmark

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


_HF_LLAMA_3_2_1B_BENCHMARK_NAME = "hf_llama_3_2_1b"
_META_LLAMA_3_2_8B_BENCHMARK_NAME = "meta_llama_3_2_8b"
_META_LLAMA_3_2_70B_BENCHMARK_NAME = "meta_llama_3_2_70b"
_QWEN3_1_7B_BENCHMARK_NAME = "qwen3_1_7b"

_SINGLE_DEVICE_PLATFORMS = (
    benchmark_utils.Platform.GFC_1X1X1,
    benchmark_utils.Platform.B200_1,
)

_WORLD_SIZE_MAP = {
    benchmark_utils.Platform.GFC_2X2X1: 8,
    benchmark_utils.Platform.B200_4: 4,
    benchmark_utils.Platform.B200_8: 8,
}


class _DummyQualityBenchmarkModel(
    quality_benchmark_model.QualityBenchmarkModel
):
  """A dummy model for unsupported platforms."""

  def initialize(self) -> None:
    pass

  def get_model(self) -> torch.nn.Module:
    return torch.nn.Module()

  def _compile_model_once(self) -> None:
    pass

  def format(self, raw_input: Any) -> quality_benchmark_model.FormattedInput:
    return quality_benchmark_model.FormattedInput(
        input=torch.tensor([]), unpadded_length=0
    )

  @property
  def max_seq_len(self) -> int:
    return 2048

  def encode(self, text: str) -> list[int]:
    return [0] * len(text)


def _distributed_meta_llama_3_benchmark_config(
    platform: benchmark_utils.Platform,
    model_config: str,
    run_mode: benchmark_utils.RunMode,
    dataset_type: data_loader.DatasetType = data_loader.DatasetType.WIKITEXT,
) -> quality_utils.QualityBenchmarkConfig:
  """Sets up the benchmark config for distributed Meta Llama-3.2.

  Distributed benchmarks are only supported on GFC_2X2X1 and B200_8 platforms.

  Args:
    platform: The platform to run the benchmark on.
    model_config: The model config to use.
    run_mode: The mode to run the benchmark in.
    dataset_type: The dataset type to use.

  Returns:
    The quality benchmark configuration.
  """
  if platform not in _WORLD_SIZE_MAP:
    return quality_utils.QualityBenchmarkConfig(
        supported_platforms=list(_WORLD_SIZE_MAP.keys()),
        benchmark_category=benchmark_utils.BenchmarkCategory.META_LLAMA,
        run_mode=run_mode,
        benchmark_model=_DummyQualityBenchmarkModel(),
        metrics=[],
        dataset_type=dataset_type,
    )

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
  metrics = [perplexity_metric.PerplexityMetric()]

  config = quality_utils.QualityBenchmarkConfig(
      supported_platforms=list(_WORLD_SIZE_MAP.keys()),
      benchmark_category=benchmark_utils.BenchmarkCategory.META_LLAMA,
      run_mode=run_mode,
      benchmark_model=benchmark_model,
      metrics=metrics,
      dataset_type=dataset_type,
  )
  return config


def _llama_3_2_1b_benchmark_config(
    platform: benchmark_utils.Platform,
    run_mode: benchmark_utils.RunMode,
    dataset_type: data_loader.DatasetType = data_loader.DatasetType.WIKITEXT,
) -> quality_utils.QualityBenchmarkConfig:
  """Sets up the benchmark config for Llama 3.2 1B.

  Args:
    platform: The platform to run the benchmark on.
    run_mode: The mode to run the benchmark in.
    dataset_type: The dataset type to use.

  Returns:
    The quality benchmark configuration.
  """
  if platform not in _SINGLE_DEVICE_PLATFORMS:
    return quality_utils.QualityBenchmarkConfig(
        supported_platforms=_SINGLE_DEVICE_PLATFORMS,
        benchmark_category=benchmark_utils.BenchmarkCategory.META_LLAMA,
        run_mode=run_mode,
        benchmark_model=_DummyQualityBenchmarkModel(),
        metrics=[],
        dataset_type=dataset_type,
    )

  device = benchmark_utils.PLATFORM_DEVICE_MAP[platform]

  # Instantiate the benchmark model
  benchmark_model = (
      llama3_2_1b_quality_benchmark.Llama321BQualityBenchmarkModel(device, 2048)
  )

  # Metrics
  metrics = [perplexity_metric.PerplexityMetric()]

  return quality_utils.QualityBenchmarkConfig(
      supported_platforms=_SINGLE_DEVICE_PLATFORMS,
      benchmark_category=benchmark_utils.BenchmarkCategory.META_LLAMA,
      run_mode=run_mode,
      benchmark_model=benchmark_model,
      metrics=metrics,
      dataset_type=dataset_type,
  )


def _qwen3_1_7b_benchmark_config(
    platform: benchmark_utils.Platform,
    run_mode: benchmark_utils.RunMode,
    dataset_type: data_loader.DatasetType = data_loader.DatasetType.WIKITEXT,
) -> quality_utils.QualityBenchmarkConfig:
  """Sets up the benchmark config for Qwen 3 1.7B.

  Args:
    platform: The platform to run the benchmark on.
    run_mode: The mode to run the benchmark in.
    dataset_type: The dataset type to use.

  Returns:
    The quality benchmark configuration.
  """
  if platform not in _SINGLE_DEVICE_PLATFORMS:
    return quality_utils.QualityBenchmarkConfig(
        supported_platforms=_SINGLE_DEVICE_PLATFORMS,
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        benchmark_model=_DummyQualityBenchmarkModel(),
        metrics=[],
        dataset_type=dataset_type,
    )

  device = benchmark_utils.PLATFORM_DEVICE_MAP[platform]

  # Instantiate the benchmark model
  benchmark_model = qwen3_1_7b_quality_benchmark.Qwen317BQualityBenchmarkModel(
      device, 2048
  )

  # Metrics
  metrics = [perplexity_metric.PerplexityMetric()]

  return quality_utils.QualityBenchmarkConfig(
      supported_platforms=_SINGLE_DEVICE_PLATFORMS,
      benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
      run_mode=run_mode,
      benchmark_model=benchmark_model,
      metrics=metrics,
      dataset_type=dataset_type,
  )


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

  def test_quality_llama_3_2_1b_eager_forward(self):
    platform = benchmark_utils.PLATFORM.value
    self.run_benchmark_test(
        _llama_3_2_1b_benchmark_config(
            platform, benchmark_utils.RunMode.EAGER_DEFAULT
        ),
        _HF_LLAMA_3_2_1B_BENCHMARK_NAME,
        False,  # distributed
    )

  def test_quality_llama_3_2_1b_compiled_forward(self):
    platform = benchmark_utils.PLATFORM.value
    self.run_benchmark_test(
        _llama_3_2_1b_benchmark_config(
            platform, benchmark_utils.RunMode.COMPILED
        ),
        _HF_LLAMA_3_2_1B_BENCHMARK_NAME,
        False,  # distributed
    )

  def test_quality_qwen3_1_7b_eager_forward(self):
    platform = benchmark_utils.PLATFORM.value
    self.run_benchmark_test(
        _qwen3_1_7b_benchmark_config(
            platform, benchmark_utils.RunMode.EAGER_DEFAULT
        ),
        _QWEN3_1_7B_BENCHMARK_NAME,
        False,  # distributed
    )

  def test_quality_qwen3_1_7b_compiled_forward(self):
    platform = benchmark_utils.PLATFORM.value
    self.run_benchmark_test(
        _qwen3_1_7b_benchmark_config(
            platform, benchmark_utils.RunMode.COMPILED
        ),
        _QWEN3_1_7B_BENCHMARK_NAME,
        False,  # distributed
    )

  def test_quality_distributed_meta_llama_3_2_8b_eager_forward(self):
    platform = benchmark_utils.PLATFORM.value
    self.run_benchmark_test(
        _distributed_meta_llama_3_benchmark_config(
            platform, "8B", benchmark_utils.RunMode.EAGER_DEFAULT
        ),
        _META_LLAMA_3_2_8B_BENCHMARK_NAME,
        True,
    )

  def test_quality_distributed_meta_llama_3_2_8b_eager_forward_edge_cases(self):
    platform = benchmark_utils.PLATFORM.value
    self.run_benchmark_test(
        _distributed_meta_llama_3_benchmark_config(
            platform,
            "8B",
            benchmark_utils.RunMode.EAGER_DEFAULT,
            dataset_type=data_loader.DatasetType.EDGE_CASES,
        ),
        _META_LLAMA_3_2_8B_BENCHMARK_NAME,
        True,
    )

  def test_quality_distributed_meta_llama_3_2_70b_eager_forward(self):
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
    platform = benchmark_utils.PLATFORM.value
    self.run_benchmark_test(
        _distributed_meta_llama_3_benchmark_config(
            platform, "8B", benchmark_utils.RunMode.COMPILED
        ),
        _META_LLAMA_3_2_8B_BENCHMARK_NAME,
        True,
    )

  def test_quality_distributed_meta_llama_3_2_70b_compiled_forward(self):
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
