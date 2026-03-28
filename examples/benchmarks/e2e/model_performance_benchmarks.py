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

"""Benchmarks for model performance."""

from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import performance_utils
from examples.benchmarks.e2e import test_utils
from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

_HF_LLAMA_3_2_1B_BENCHMARK_NAME = "hf_llama_3_2_1b"
_HF_GEMMA_3_270M_BENCHMARK_NAME = "hf_gemma_3_270m"
_META_LLAMA_3_2_8B_BENCHMARK_NAME = "meta_llama_3_2_8b"


class BenchmarkTest(test_utils.BenchmarkTest):
  """Tests for end-to-end model performance benchmarks."""

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.DEFER_NEVER_ONLY,
          benchmark_utils.RunMode.DEFER_NEVER_AND_LAUNCH_BLOCKING,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_llama_3_2_1b_forward(self, run_mode):
    """Tests the forward pass of Llama-3.2-1B."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="meta-llama/Llama-3.2-1B",
            sequence_length=4096,
            batch_size=1,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_LLAMA_3_2_1B_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.DEFER_NEVER_ONLY,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_llama_3_2_1b_train_1_step(self, run_mode):
    """Tests the training of Llama-3.2-1B."""
    batch_size = 12
    if run_mode == benchmark_utils.RunMode.COMPILED:
      # Batch size 12 OOMs on TPU Ironwood 1x1x1 for torch compile. Using batch
      # size 8 instead.
      batch_size = 8

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="meta-llama/Llama-3.2-1B",
            sequence_length=1024,
            batch_size=batch_size,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_LLAMA_3_2_1B_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.DEFER_NEVER_ONLY,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_gemma_3_270m_train_1_step(self, run_mode):
    """Tests the training of Gemma-3-270m."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="google/gemma-3-270m",
            sequence_length=512,
            batch_size=1,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_GEMMA_3_270M_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.DEFER_NEVER_ONLY,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_gemma_3_270m_forward(self, run_mode):
    """Tests the forward pass of Gemma-3-270m."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="google/gemma-3-270m",
            sequence_length=512,
            batch_size=1,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_GEMMA_3_270M_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.DEFER_NEVER_ONLY,
      ])
  )
  def test_distributed_meta_llama_3_2_8b_forward(self, run_mode):
    """Tests the forward pass of Meta Llama-3.2-8B."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_2X2X1,
            benchmark_utils.Platform.B200_4,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.META_LLAMA,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Llama-3.2-8B",
            sequence_length=2048,
            batch_size=1,
        ),
    )
    self.run_performance_benchmark_test(
        config, _META_LLAMA_3_2_8B_BENCHMARK_NAME
    )

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.DEFER_NEVER_ONLY,
      ])
  )
  def test_ddp_llama_3_2_1b_train_1_step(self, run_mode):
    """Tests training Llama-3.2-1B distributed with DDP."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_2X2X1,
            benchmark_utils.Platform.B200_4,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="meta-llama/Llama-3.2-1B",
            sequence_length=2056,
            batch_size=8,
            custom_kwargs={"dist_strat": "ddp"},
        ),
        grad_accumulation_steps=10,
    )
    self.run_performance_benchmark_test(config, _HF_LLAMA_3_2_1B_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.DEFER_NEVER_ONLY,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_fsdp_llama_3_2_1b_train_1_step(self, run_mode):
    """Tests training Llama-3.2-1B distributed with FSDP."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_2X2X1,
            benchmark_utils.Platform.B200_4,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="meta-llama/Llama-3.2-1B",
            sequence_length=2056,
            batch_size=16,
            custom_kwargs={"dist_strat": "fsdp"},
        ),
        grad_accumulation_steps=10,
    )
    self.run_performance_benchmark_test(config, _HF_LLAMA_3_2_1B_BENCHMARK_NAME)


if __name__ == "__main__":
  mp.set_start_method("spawn")
  # g3_multiprocessing is required to run absltest.main() in a multiprocess
  # environment. It doesn't affect single process runs.
  # See: go/g3_multiprocessing#resolution.
  g3_multiprocessing.handle_test_main(absltest.main)
