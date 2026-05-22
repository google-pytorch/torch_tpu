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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
from examples.benchmarks.e2e import benchmark_function_db
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import model_utils
from examples.benchmarks.e2e import performance_utils
from examples.benchmarks.e2e import test_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


_HF_LLAMA_3_2_1B_BENCHMARK_NAME = "hf_llama_3_2_1b"
_HF_GEMMA_3_270M_BENCHMARK_NAME = "hf_gemma_3_270m"
_HF_GEMMA_4_31B_BENCHMARK_NAME = "hf_gemma_4_31b"
_META_LLAMA_3_2_8B_BENCHMARK_NAME = "meta_llama_3_2_8b"
_HF_QWEN3_1_7B_BENCHMARK_NAME = "hf_qwen3_1_7b"
_HF_GPT_OSS_20B_BENCHMARK_NAME = "hf_gpt_oss_20b"
_HF_GPT_OSS_120B_BENCHMARK_NAME = "hf_gpt_oss_120b"
_HF_QWEN3_CODER_30B_RAGGED_MOE_BENCHMARK_NAME = "hf_qwen3_30b_ragged_moe"
_HF_GEMMA_4_26B_A4B_RAGGED_MOE_BENCHMARK_NAME = "hf_gemma_4_26b_ragged_moe"
_TIMM_RESNET_50_BENCHMARK_NAME = "timm_resnet_50"
_WAN_2_2_TI2V_5B_BENCHMARK_NAME = "wan_2_2_ti2v_5b"
_HF_WHISPER_LARGE_V3_BENCHMARK_NAME = "hf_whisper_large_v3"


class BenchmarkTest(test_utils.BenchmarkTest):
  """Tests for end-to-end model performance benchmarks."""

  # ============================================================================
  # 1. Llama Architecture Family
  # ============================================================================

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING,
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
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_LLAMA_3_2_1B_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_llama_3_2_1b_train_1_step(self, run_mode):
    """Tests the training of Llama-3.2-1B."""

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
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=4,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_LLAMA_3_2_1B_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_gemma_4_31b_6_layers_forward(self, run_mode):

    def _modify_gemma4_config_to_small(config):
      if hasattr(config, "text_config"):
        config.text_config.num_hidden_layers = 6
      return config

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="google/gemma-4-31B",
            sequence_length=512,
            batch_size=1,
            custom_kwargs={
                "modify_config_hook": _modify_gemma4_config_to_small
            },
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_GEMMA_4_31B_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_gemma_4_31b_6_layers_train_1_step(self, run_mode):

    def _modify_gemma4_config_to_small(config):
      if hasattr(config, "text_config"):
        config.text_config.num_hidden_layers = 6
      return config

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="google/gemma-4-31B",
            sequence_length=512,
            batch_size=1,
            custom_kwargs={
                "modify_config_hook": _modify_gemma4_config_to_small
            },
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=4,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_GEMMA_4_31B_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
      ])
  )
  def test_distributed_meta_llama_3_2_8b_forward(self, run_mode):
    """Tests the forward pass of Meta Llama-3.2-8B."""
    if self._is_torchax_backend():
      self.skipTest("TorchAX does not support distributed tests yet.")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_2X2X1,
            benchmark_utils.Platform.GFC_2X2X2,
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
        model_and_input_factory=model_utils.meta_llama_model_builder,
        eval_factory=benchmark_function_db.meta_llama_eval_factory,
    )
    self.run_performance_benchmark_test(
        config, _META_LLAMA_3_2_8B_BENCHMARK_NAME
    )

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
      ])
  )
  def test_ddp_llama_3_2_1b_train_1_step(self, run_mode):
    """Tests training Llama-3.2-1B distributed with DDP."""
    if self._is_torchax_backend():
      self.skipTest("TorchAX does not support distributed tests yet.")
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
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=10,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_LLAMA_3_2_1B_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          # TODO(b/502598781): Re-enable the compiled mode.
          # benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_fsdp_llama_3_2_1b_train_1_step(self, run_mode):
    """Tests training Llama-3.2-1B distributed with FSDP."""
    if self._is_torchax_backend():
      self.skipTest("TorchAX does not support distributed tests yet.")
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
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=10,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_LLAMA_3_2_1B_BENCHMARK_NAME)

  # ============================================================================
  # 2. Gemma Architecture Family
  # ============================================================================

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_gemma_3_270m_forward(self, run_mode):
    """Tests the forward pass of Gemma-3-270m."""
    if run_mode == benchmark_utils.RunMode.COMPILED:
      self.skipTest("Compiled mode fails on view().")
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
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_GEMMA_3_270M_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_gemma_3_270m_train_1_step(self, run_mode):
    """Tests the training of Gemma-3-270m."""
    # TODO(b/512109815): Reenable after fix.
    if run_mode == benchmark_utils.RunMode.COMPILED:
      self.skipTest("Compiled mode fails on view().")

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
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=4,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_GEMMA_3_270M_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
      ])
  )
  def test_gemma_4_26b_a4b_ragged_moe_12_layers_forward(self, run_mode):
    """Tests the forward pass of Gemma-4-26B-A4B (12 Layers) with Ragged MoE."""
    if self._is_torchax_backend():
      self.skipTest("TorchAX does not support distributed tests yet.")

    def modify_config_hook(config):
      if hasattr(config, "text_config"):
        config.text_config.num_hidden_layers = 12
      config.num_hidden_layers = 12
      return config

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[benchmark_utils.Platform.GFC_1X1X1],
        benchmark_category=benchmark_utils.BenchmarkCategory.GEMMA_RAGGED_MOE,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="google/gemma-4-26B-A4B",
            sequence_length=2048,
            batch_size=8,
            custom_kwargs={"modify_config_hook": modify_config_hook},
        ),
        model_and_input_factory=model_utils.gemma_ragged_moe_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(
        config, _HF_GEMMA_4_26B_A4B_RAGGED_MOE_BENCHMARK_NAME
    )

  # ============================================================================
  # 3. Qwen Architecture Family
  # ============================================================================

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_qwen3_1_7b_forward(self, run_mode):
    """Tests the forward pass of Qwen3 1.7B."""

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Qwen/Qwen3-1.7B",
            sequence_length=4096,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_QWEN3_1_7B_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_qwen3_1_7b_train(self, run_mode):
    """Tests the train pass of Qwen3 1.7B."""
    if (
        self._is_torchax_backend()
        and run_mode != benchmark_utils.RunMode.COMPILED
    ):
      self.skipTest("Device OOM")

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Qwen/Qwen3-1.7B",
            sequence_length=4096,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=1,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_QWEN3_1_7B_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
      ])
  )
  def test_qwen3_coder_30b_a3b_ragged_moe_forward(self, run_mode):
    """Tests the forward pass of Qwen3-Coder-30B-A3B."""
    if self._is_torchax_backend():
      self.skipTest("TorchAX does not support distributed tests yet.")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_2X2X1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.QWEN_RAGGED_MOE,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            sequence_length=2048,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.qwen_ragged_moe_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(
        config, _HF_QWEN3_CODER_30B_RAGGED_MOE_BENCHMARK_NAME
    )

  # ============================================================================
  # 4. GPT-OSS Architecture Family
  # ============================================================================

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_gpt_oss_20b_forward(self, run_mode):
    """Tests the forward pass of GPT-OSS-20B."""
    if (
        self._is_torchax_backend()
        and run_mode != benchmark_utils.RunMode.COMPILED
    ):
      self.skipTest("Device OOM")

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="openai/gpt-oss-20b",
            sequence_length=4096,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_GPT_OSS_20B_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_gpt_oss_20b_train(self, run_mode):
    """Tests the train pass of GPT-OSS-20B."""
    # TODO(b/510886286): Reenable after fix.
    self.skipTest("Assert async not supported yet.")

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="openai/gpt-oss-20b",
            sequence_length=4096,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=1,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_GPT_OSS_20B_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_gpt_oss_120b_4_layers_forward(self, run_mode):
    """Tests the forward pass of GPT-OSS-20B."""

    def modify_config_hook(config):
      config.num_hidden_layers = 4
      return config

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="openai/gpt-oss-120b",
            sequence_length=512,
            batch_size=1,
            custom_kwargs={"modify_config_hook": modify_config_hook},
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_GPT_OSS_120B_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_gpt_oss_120b_4_layers_train(self, run_mode):
    """Tests the train pass of GPT-OSS-20B."""
    # TODO(b/510886286): Reenable after fix.
    self.skipTest("Assert async not supported yet.")

    def modify_config_hook(config):
      config.num_hidden_layers = 4
      return config

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="openai/gpt-oss-120b",
            sequence_length=512,
            batch_size=1,
            custom_kwargs={"modify_config_hook": modify_config_hook},
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=1,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_GPT_OSS_120B_BENCHMARK_NAME)

  # ============================================================================
  # 5. TIMM / Vision Models
  # ============================================================================

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_timm_resnet_50_forward(self, run_mode):
    """Tests the forward pass of resnet-50."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.TIMM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="timm/resnet50d",
            custom_kwargs={"input_shape": (16, 3, 224, 224)},
        ),
        model_and_input_factory=model_utils.timm_model_builder,
        eval_factory=benchmark_function_db.timm_eval_factory,
    )
    self.run_performance_benchmark_test(config, _TIMM_RESNET_50_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_timm_resnet_50_backward(self, run_mode):
    """Tests the backward pass of resnet-50."""

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.TIMM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="timm/resnet50d",
            custom_kwargs={"input_shape": (16, 3, 224, 224)},
        ),
        model_and_input_factory=model_utils.timm_model_builder,
        train_factory=benchmark_function_db.simple_train_factory,
    )
    self.run_performance_benchmark_test(config, _TIMM_RESNET_50_BENCHMARK_NAME)

  # ============================================================================
  # 6. Wan Diffuser Family
  # ============================================================================

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_wan_2_2_ti2v_5b_forward(self, run_mode):
    """Tests the forward pass of Wan-2.2-TI2V-5B."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_DIFFUSER,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        ),
        model_and_input_factory=model_utils.huggingface_diffuser_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _WAN_2_2_TI2V_5B_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_wan_2_2_ti2v_5b_backward(self, run_mode):
    """Tests the backward pass of Wan-2.2-TI2V-5B."""
    if (
        run_mode == benchmark_utils.RunMode.COMPILED
        and not self._is_torchax_backend()
    ):
      self.skipTest("Compiled mode fails for TorchTPU.")

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_DIFFUSER,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        ),
        model_and_input_factory=model_utils.huggingface_diffuser_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_diffuser_train_factory,
            grad_accumulation_steps=4,
        ),
    )
    self.run_performance_benchmark_test(config, _WAN_2_2_TI2V_5B_BENCHMARK_NAME)

  # ============================================================================
  # 7. Audio / Whisper Models
  # ============================================================================

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          benchmark_utils.RunMode.EAGER_DEFAULT,
          benchmark_utils.RunMode.EAGER_OPTIMIZED,
          benchmark_utils.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING,
          benchmark_utils.RunMode.COMPILED,
      ])
  )
  def test_whisper_large_v3_forward(self, run_mode):
    """Tests the forward pass of Whisper-Large-v3."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="openai/whisper-large-v3",
            sequence_length=448,
            batch_size=16,
        ),
        model_and_input_factory=model_utils.whisper_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(
        config, _HF_WHISPER_LARGE_V3_BENCHMARK_NAME
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  # g3_multiprocessing is required to run absltest.main() in a multiprocess
  # environment. It doesn't affect single process runs.
  # See: go/g3_multiprocessing#resolution.
  g3_multiprocessing.handle_test_main(absltest.main)
