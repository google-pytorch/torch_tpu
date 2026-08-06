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
import torch_tpu
from examples.benchmarks.e2e import benchmark_function_db
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import common
from examples.benchmarks.e2e import model_utils
from examples.benchmarks.e2e import performance_utils
from examples.benchmarks.e2e import test_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

_HF_LLAMA_3_2_1B_BENCHMARK_NAME = "hf_llama_3_2_1b"
_HF_GEMMA_3_270M_BENCHMARK_NAME = "hf_gemma_3_270m"
_HF_GEMMA_4_31B_BENCHMARK_NAME = "hf_gemma_4_31b"
_HF_GEMMA_4_E2B_BENCHMARK_NAME = "hf_gemma_4_e2b"
_GEMMA_4_CUSTOM_STANDALONE_BENCHMARK_NAME = "gemma_4_custom_standalone"
_META_LLAMA_3_2_8B_BENCHMARK_NAME = "meta_llama_3_2_8b"
_HF_QWEN3_1_7B_BENCHMARK_NAME = "hf_qwen3_1_7b"
_HF_GPT_OSS_20B_BENCHMARK_NAME = "hf_gpt_oss_20b"
_HF_GPT_OSS_120B_BENCHMARK_NAME = "hf_gpt_oss_120b"
_HF_QWEN3_CODER_30B_RAGGED_MOE_BENCHMARK_NAME = "hf_qwen3_30b_ragged_moe"
_HF_QWEN3_5_397B_A17B_4LAYER_MOE_BENCHMARK_NAME = (
    "hf_qwen3_5_397b_a17b_4layer_moe"
)
_HF_GEMMA_4_26B_A4B_RAGGED_MOE_BENCHMARK_NAME = "hf_gemma_4_26b_ragged_moe"
_TIMM_RESNET_50_BENCHMARK_NAME = "timm_resnet_50"
_WAN_2_2_TI2V_5B_BENCHMARK_NAME = "wan_2_2_ti2v_5b"
_HF_WHISPER_LARGE_V3_BENCHMARK_NAME = "hf_whisper_large_v3"
_HF_VJEPA2_VITL_BENCHMARK_NAME = "hf_vjepa2_vitl"
_DETR_RESNET_50_BENCHMARK_NAME = "detr_resnet_50"
_HF_VJEPA2_VITG_BENCHMARK_NAME = "hf_vjepa2_vitg"
_HF_GEMMA_2_2B_BENCHMARK_NAME = "hf_gemma_2_2b"
_HF_QWEN3_0_6B_BENCHMARK_NAME = "hf_qwen3_0_6b"
_HF_QWEN3_4B_BENCHMARK_NAME = "hf_qwen3_4b"
_HF_QWEN3_8B_BENCHMARK_NAME = "hf_qwen3_8b"
_HF_LLAMA_3_1_8B_BENCHMARK_NAME = "hf_llama_3_1_8b"
_HF_LLAMA_3_2_3B_BENCHMARK_NAME = "hf_llama_3_2_3b"
_HF_GPT2_BENCHMARK_NAME = "hf_gpt2"
_HF_PHI_3_MINI_4K_INSTRUCT_BENCHMARK_NAME = "hf_phi_3_mini_4k_instruct"
# Note: hf_resnet_50 is from Hugging Face transformers, which differs from timm_resnet_50.
_HF_RESNET_50_BENCHMARK_NAME = "hf_resnet_50"
_TIMM_VIT_SMALL_DINO_BENCHMARK_NAME = "timm_vit_small_dino"
_HF_NEMOTRON_3_NANO_30B_BENCHMARK_NAME = "hf_nemotron_3_nano_30b"
_HF_MLLAMA_11B_BENCHMARK_NAME = "hf_mllama_11b"
_HF_LLAVA_7B_BENCHMARK_NAME = "hf_llava_7b"
_HF_CLIP_BASE_BENCHMARK_NAME = "hf_clip_base"
_HF_VIT_BASE_BENCHMARK_NAME = "hf_vit_base"
_HF_SIGLIP_BASE_BENCHMARK_NAME = "hf_siglip_base"
_HF_DINOV2_BASE_BENCHMARK_NAME = "hf_dinov2_base"
_HF_DEIT_BASE_BENCHMARK_NAME = "hf_deit_base"
_HF_SWIN_BASE_BENCHMARK_NAME = "hf_swin_base"
_HF_SEGFORMER_B2_BENCHMARK_NAME = "hf_segformer_b2"
_HF_WAV2VEC2_BASE_BENCHMARK_NAME = "hf_wav2vec2_base"
_HF_WHISPER_BASE_BENCHMARK_NAME = "hf_whisper_base"
_HF_BERT_BASE_BENCHMARK_NAME = "hf_bert_base"
_HF_T5_BASE_BENCHMARK_NAME = "hf_t5_base"
_HF_BART_BASE_BENCHMARK_NAME = "hf_bart_base"
_HF_STARCODER2_3B_BENCHMARK_NAME = "hf_starcoder2_3b"


class BenchmarkTest(test_utils.BenchmarkTest):
  """Tests for end-to-end model performance benchmarks."""

  # ============================================================================
  # 1. Llama Architecture Family
  # ============================================================================

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_llama_3_2_1b_forward(self, run_mode):
    """Tests the forward pass of Llama-3.2-1B."""
    batch_size, sequence_length = (
        (1, 256) if benchmark_utils.SMOKE_TEST.value else (1, 4096)
    )
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.V5E_1X1,
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
            common.Platform.XLA_CPU,
            common.Platform.TORCH_CPU,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="meta-llama/Llama-3.2-1B",
            sequence_length=sequence_length,
            batch_size=batch_size,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_LLAMA_3_2_1B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_llama_3_2_1b_train_1_step(self, run_mode):
    """Tests the training of Llama-3.2-1B."""

    batch_size = 8

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
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

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_llama_3_2_1b_train_1_step_grad_acc_1(self, run_mode):
    """Tests the training of Llama-3.2-1B with grad accumulation steps set to 1."""

    batch_size = 8

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
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
            grad_accumulation_steps=1,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_LLAMA_3_2_1B_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          common.RunMode.EAGER_DEFAULT,
          common.RunMode.EAGER_OPTIMIZED,
          common.RunMode.COMPILED,
      ])
  )
  def test_distributed_meta_llama_3_2_8b_forward(self, run_mode):
    """Tests the forward pass of Meta Llama-3.2-8B."""
    if self._is_torchax_backend():
      self.skipTest("TorchAX does not support distributed tests yet.")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_2X2X1,
            common.Platform.GFC_2X2X2,
            common.Platform.B200_4,
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

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_ddp_llama_3_2_1b_train_1_step(self, run_mode):
    """Tests training Llama-3.2-1B distributed with DDP."""
    if self._is_torchax_backend():
      self.skipTest("TorchAX does not support distributed tests yet.")
    if (
        common.PLATFORM.value == common.Platform.GFC_2X2X1
        and run_mode == common.RunMode.COMPILED
    ):
      self.skipTest(
          "Skipping compiled mode on GFC 2x2x1. Cannot view non-contiguous"
          " attention output [8, 2056, 32, 64] as [8, 2056, 2048] without copy."
      )
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_2X2X1,
            common.Platform.B200_4,
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

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_fsdp_llama_3_2_1b_train_1_step(self, run_mode):
    """Tests training Llama-3.2-1B distributed with FSDP."""
    if self._is_torchax_backend():
      self.skipTest("TorchAX does not support distributed tests yet.")
    if (
        common.PLATFORM.value == common.Platform.GFC_2X2X1
        and run_mode == common.RunMode.COMPILED
    ):
      self.skipTest(
          "Skipping compiled mode on GFC 2x2x1. DTensor generates symints"
          " which are not supported by TorchTPU compiled backend."
      )
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_2X2X1,
            common.Platform.B200_4,
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

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_llama_3_1_8b_forward(self, run_mode):
    """Tests the forward pass of Llama-3.1-8B."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="meta-llama/Llama-3.1-8B",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_LLAMA_3_1_8B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_llama_3_2_3b_forward(self, run_mode):
    """Tests the forward pass of Llama-3.2-3B."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="meta-llama/Llama-3.2-3B",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_LLAMA_3_2_3B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_llama_3_2_3b_train_1_step(self, run_mode):
    """Tests training Llama-3.2-3B."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="meta-llama/Llama-3.2-3B",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=1,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_LLAMA_3_2_3B_BENCHMARK_NAME)

  # ============================================================================
  # 2. Gemma Architecture Family
  # ============================================================================

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gemma_3_270m_forward(self, run_mode):
    """Tests the forward pass of Gemma-3-270m."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.V5E_1X1,
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
            common.Platform.XLA_CPU,
            common.Platform.TORCH_CPU,
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

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gemma_3_270m_train_1_step(self, run_mode):
    """Tests the training of Gemma-3-270m."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.V5E_1X1,
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
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

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gemma_3_270m_train_1_step_grad_acc_1(self, run_mode):
    """Tests the training of Gemma-3-270m with grad accumulation steps set to 1."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
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
            grad_accumulation_steps=1,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_GEMMA_3_270M_BENCHMARK_NAME)

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          common.RunMode.EAGER_DEFAULT,
          common.RunMode.EAGER_OPTIMIZED,
          common.RunMode.COMPILED,
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
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.GEMMA_RAGGED_MOE,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="google/gemma-4-26B-A4B",
            sequence_length=2048,
            batch_size=8,
            custom_kwargs={
                "modify_config_hook": modify_config_hook,
                # FIXME: Disable vision inputs under COMPILED mode because static
                # XLA graph compilation (TpuBackend) does not yet support dynamic
                # pre-patchified image grid slicing (pixel_values and
                # image_position_ids). Re-enable once TpuBackend supports dynamic
                # multimodal patch slicing.
                "disable_vision_inputs": run_mode == common.RunMode.COMPILED,
            },
        ),
        model_and_input_factory=model_utils.gemma_ragged_moe_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(
        config, _HF_GEMMA_4_26B_A4B_RAGGED_MOE_BENCHMARK_NAME
    )

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gemma_2_2b_forward(self, run_mode):
    """Tests the forward pass of Gemma-2-2B."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="google/gemma-2-2b",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_GEMMA_2_2B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gemma_2_2b_train_1_step(self, run_mode):
    """Tests the training of Gemma-2-2B."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="google/gemma-2-2b",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=1,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_GEMMA_2_2B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gemma_4_31b_6_layers_forward(self, run_mode):

    def _modify_gemma4_config_to_small(config):
      if hasattr(config, "text_config"):
        config.text_config.num_hidden_layers = 6
      return config

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="google/gemma-4-31b",
            sequence_length=2048,
            batch_size=8,
            custom_kwargs={
                "modify_config_hook": _modify_gemma4_config_to_small,
                "disable_vision_inputs": run_mode == common.RunMode.COMPILED,
            },
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_GEMMA_4_31B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gemma_4_31b_6_layers_train_1_step(self, run_mode):

    def _modify_gemma4_config_to_small(config):
      if hasattr(config, "text_config"):
        config.text_config.num_hidden_layers = 6
      return config

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="google/gemma-4-31b",
            sequence_length=1024,
            batch_size=4,
            custom_kwargs={
                "modify_config_hook": _modify_gemma4_config_to_small,
                "disable_vision_inputs": run_mode == common.RunMode.COMPILED,
            },
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=4,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_GEMMA_4_31B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gemma_4_31b_6_layers_train_1_step_grad_acc_1(self, run_mode):

    def _modify_gemma4_config_to_small(config):
      if hasattr(config, "text_config"):
        config.text_config.num_hidden_layers = 6
      return config

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="google/gemma-4-31b",
            sequence_length=512,
            batch_size=1,
            custom_kwargs={
                "modify_config_hook": _modify_gemma4_config_to_small,
                "disable_vision_inputs": run_mode == common.RunMode.COMPILED,
            },
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=1,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_GEMMA_4_31B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gemma_4_e2b_forward(self, run_mode):
    """Tests the forward pass of Gemma-4-E2B."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="google/gemma-4-e2b",
            sequence_length=2048,
            batch_size=8,
            custom_kwargs={
                "disable_vision_inputs": run_mode == common.RunMode.COMPILED,
            },
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_GEMMA_4_E2B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gemma_4_custom_standalone_forward(self, run_mode):
    """Tests forward pass of custom standalone Gemma4 SWA model."""
    if self._is_torchax_backend():
      self.skipTest(
          "Gemma 4 custom standalone SWA is not supported on TorchAX."
      )
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.INTERNAL_MODEL,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="gemma-4-custom-standalone",
            sequence_length=2048,
            batch_size=8,
            custom_kwargs={
                "preset": "e2b",
            },
        ),
        model_and_input_factory=model_utils.gemma4_custom_standalone_model_builder,
        eval_factory=benchmark_function_db.simple_eval_factory,
    )
    self.run_performance_benchmark_test(
        config, _GEMMA_4_CUSTOM_STANDALONE_BENCHMARK_NAME
    )

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_mllama_11b_forward(self, run_mode):
    """Tests the forward pass of mLlama-11B-Vision."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="meta-llama/Llama-3.2-11B-Vision",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_MLLAMA_11B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_llava_7b_forward(self, run_mode):
    """Tests the forward pass of LLaVA-1.5-7B."""
    # FIXME: LLaVA requires multi-modal input dictionary key mapping (pixel_values and image_position_ids).
    self.skipTest("LLaVA requires specialized multimodal input builder.")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="liuhaotian/llava-v1.5-7b",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_LLAVA_7B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_clip_base_forward(self, run_mode):
    """Tests the forward pass of CLIP-ViT-Base."""
    # FIXME: CLIP dual-encoder architecture requires both text (input_ids) and vision (pixel_values) inputs.
    self.skipTest("CLIP requires dual-modality (text+vision) input generator.")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="openai/clip-vit-base-patch16",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_CLIP_BASE_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_vit_base_forward(self, run_mode):
    """Tests the forward pass of ViT-Base."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="google/vit-base-patch16-224",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_VIT_BASE_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_siglip_base_forward(self, run_mode):
    """Tests the forward pass of SigLIP-Base."""
    # FIXME: SigLIP vision-language model requires pixel_values tensor input.
    self.skipTest("SigLIP requires vision pixel_values input generator.")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="google/siglip-base-patch16-224",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_SIGLIP_BASE_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_dinov2_base_forward(self, run_mode):
    """Tests the forward pass of DINOv2-Base."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="facebook/dinov2-base",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_DINOV2_BASE_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_deit_base_forward(self, run_mode):
    """Tests the forward pass of DeiT-Base."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="facebook/deit-tiny-patch16-224",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_DEIT_BASE_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_swin_base_forward(self, run_mode):
    """Tests the forward pass of Swin-Base."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="microsoft/swinv2-tiny-patch4-window16-256",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_SWIN_BASE_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_segformer_b2_forward(self, run_mode):
    """Tests the forward pass of SegFormer-B2."""
    # FIXME: SegFormer vision encoder requires pixel_values image tensor input.
    self.skipTest("SegFormer requires vision pixel_values input generator.")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nvidia/segformer-b2-finetuned-ade-512-512",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_SEGFORMER_B2_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_wav2vec2_base_forward(self, run_mode):
    """Tests the forward pass of Wav2Vec2-Base."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="facebook/wav2vec2-base-960h",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(
        config, _HF_WAV2VEC2_BASE_BENCHMARK_NAME
    )

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_whisper_base_forward(self, run_mode):
    """Tests the forward pass of Whisper-Base."""
    # FIXME: Whisper speech-to-text encoder requires input_features audio spectrogram input.
    self.skipTest("Whisper requires audio input_features generator.")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="openai/whisper-base",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_WHISPER_BASE_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_bert_base_forward(self, run_mode):
    """Tests the forward pass of BERT-Base."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="EhsanAghazadeh/bert-base-uncased-random-weights",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_BERT_BASE_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_t5_base_forward(self, run_mode):
    """Tests the forward pass of T5-Base."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="google-t5/t5-base",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_T5_BASE_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_bart_base_forward(self, run_mode):
    """Tests the forward pass of BART-Base."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="facebook/bart-base",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_BART_BASE_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_starcoder2_3b_forward(self, run_mode):
    """Tests the forward pass of StarCoder2-3B."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="bigcode/starcoder2-3b",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(
        config, _HF_STARCODER2_3B_BENCHMARK_NAME
    )

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gemma_4_e2b_train_1_step(self, run_mode: common.RunMode) -> None:
    """Tests the training of Gemma-4-E2B."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="google/gemma-4-e2b",
            sequence_length=1024,
            batch_size=4,
            custom_kwargs={
                "disable_vision_inputs": run_mode == common.RunMode.COMPILED,
            },
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=4,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_GEMMA_4_E2B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gemma_4_e2b_train_1_step_grad_acc_1(
      self, run_mode: common.RunMode
  ) -> None:
    """Tests the training of Gemma-4-E2B with grad accumulation steps set to 1."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="google/gemma-4-e2b",
            sequence_length=512,
            batch_size=1,
            custom_kwargs={
                "disable_vision_inputs": run_mode == common.RunMode.COMPILED,
            },
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=1,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_GEMMA_4_E2B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gemma_4_custom_standalone_train_1_step(self, run_mode):
    """Tests training of custom standalone Gemma4 SWA model."""
    if self._is_torchax_backend():
      self.skipTest(
          "Gemma 4 custom standalone SWA is not supported on TorchAX."
      )
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.INTERNAL_MODEL,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="gemma-4-custom-standalone",
            sequence_length=256,
            batch_size=2,
            custom_kwargs={
                "preset": "e2b",
                "num_layers": 35,
            },
        ),
        model_and_input_factory=model_utils.gemma4_custom_standalone_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.generic_train_factory,
            grad_accumulation_steps=1,
        ),
    )
    self.run_performance_benchmark_test(
        config, _GEMMA_4_CUSTOM_STANDALONE_BENCHMARK_NAME
    )

  # ============================================================================

  # 3. Qwen Architecture Family
  # ============================================================================

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_qwen3_1_7b_forward(self, run_mode):
    """Tests the forward pass of Qwen3 1.7B."""

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
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

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_qwen3_1_7b_train(self, run_mode):
    """Tests the train pass of Qwen3 1.7B."""

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
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
          common.RunMode.EAGER_DEFAULT,
          common.RunMode.EAGER_OPTIMIZED,
          common.RunMode.COMPILED,
      ])
  )
  def test_qwen3_coder_30b_a3b_ragged_moe_forward(self, run_mode):
    """Tests the forward pass of Qwen3-Coder-30B-A3B."""
    if self._is_torchax_backend():
      self.skipTest("TorchAX does not support distributed tests yet.")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_2X2X1,
            common.Platform.B200_4,
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

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([
          common.RunMode.EAGER_DEFAULT,
          common.RunMode.EAGER_OPTIMIZED,
          common.RunMode.COMPILED,
      ])
  )
  def test_qwen3_5_397b_a17b_4layer_moe_forward(self, run_mode):
    """Tests the forward pass of Qwen3-5-397B-A17B."""
    if self._is_torchax_backend():
      self.skipTest(
          "Not supported on torchax. Missing grouped_mm op for torchax."
      )

    def modify_config_hook(base_config):
      if hasattr(base_config, "get_text_config"):
        default_config = base_config.get_text_config()
      else:
        default_config = base_config

      default_config.num_hidden_layers = 4
      if hasattr(default_config, "layer_types"):
        default_config.layer_types = default_config.layer_types[:4]
      base_config.num_hidden_layers = 4
      return base_config

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.QWEN3_5_MOE,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Qwen/Qwen3.5-397B-A17B",
            sequence_length=2048,
            batch_size=1,
            custom_kwargs={
                "modify_config_hook": modify_config_hook,
            },
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(
        config, _HF_QWEN3_5_397B_A17B_4LAYER_MOE_BENCHMARK_NAME
    )

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_qwen3_0_6b_forward(self, run_mode):
    """Tests the forward pass of Qwen3-0.6B."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Qwen/Qwen3-0.6B",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_QWEN3_0_6B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_qwen3_0_6b_train_1_step(self, run_mode):
    """Tests the training of Qwen3-0.6B."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Qwen/Qwen3-0.6B",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=1,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_QWEN3_0_6B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_qwen3_4b_forward(self, run_mode):
    """Tests the forward pass of Qwen3-4B."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Qwen/Qwen3-4B",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_QWEN3_4B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_qwen3_4b_train_1_step(self, run_mode):
    """Tests the training of Qwen3-4B."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Qwen/Qwen3-4B",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=1,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_QWEN3_4B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_qwen3_8b_forward(self, run_mode):
    """Tests the forward pass of Qwen3-8B."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Qwen/Qwen3-8B",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_QWEN3_8B_BENCHMARK_NAME)

  # ============================================================================
  # 4. GPT-OSS Architecture Family
  # ============================================================================

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gpt_oss_20b_forward(self, run_mode):
    """Tests the forward pass of GPT-OSS-20B."""
    if self._is_torchax_backend():
      self.skipTest("Not supported on TorchAX")

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
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

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gpt_oss_20b_train(self, run_mode):
    """Tests the train pass of GPT-OSS-20B."""
    # TODO(b/510886286): Reenable after fix.
    self.skipTest("Assert async not supported yet.")
    if self._is_torchax_backend():
      self.skipTest("Missing grouped_mm op for torchax backend.")

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
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

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gpt_oss_120b_4_layers_forward(self, run_mode):
    """Tests the forward pass of GPT-OSS-20B."""
    if self._is_torchax_backend():
      self.skipTest("Not supported on TorchAX")

    def modify_config_hook(config):
      config.num_hidden_layers = 4
      return config

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
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

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gpt_oss_120b_4_layers_train(self, run_mode):
    """Tests the train pass of GPT-OSS-20B."""
    # TODO(b/510886286): Reenable after fix.
    self.skipTest("Assert async not supported yet.")
    if self._is_torchax_backend():
      self.skipTest("Missing grouped_mm op for torchax backend.")

    def modify_config_hook(config):
      config.num_hidden_layers = 4
      return config

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
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

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gpt2_forward(self, run_mode):
    """Tests the forward pass of GPT2."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="openai-community/gpt2",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_GPT2_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_gpt2_train_1_step(self, run_mode):
    """Tests training GPT2."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="openai-community/gpt2",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=1,
            use_math_attention_fallback=True,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_GPT2_BENCHMARK_NAME)

  # ============================================================================
  # 5. Phi-3 Family
  # ============================================================================
  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_phi_3_mini_4k_instruct_forward(self, run_mode):
    """Tests the forward pass of Phi-3-mini-4k-instruct."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(
        config, _HF_PHI_3_MINI_4K_INSTRUCT_BENCHMARK_NAME
    )

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_phi_3_mini_4k_instruct_train_1_step(self, run_mode):
    """Tests training Phi-3-mini-4k-instruct."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_llm_train_factory,
            grad_accumulation_steps=1,
        ),
    )
    self.run_performance_benchmark_test(
        config, _HF_PHI_3_MINI_4K_INSTRUCT_BENCHMARK_NAME
    )

  # ============================================================================
  # 6. TIMM / Vision Models
  # ============================================================================
  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_timm_resnet_50_forward(self, run_mode):
    """Tests the forward pass of resnet-50."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
            common.Platform.XLA_CPU,
            common.Platform.TORCH_CPU,
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

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_timm_resnet_50_backward(self, run_mode):
    """Tests the backward pass of resnet-50."""

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
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

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_hf_resnet_50_forward(self, run_mode):
    """Tests the forward pass of HuggingFace ResNet-50."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="microsoft/resnet-50",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_resnet_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_RESNET_50_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_hf_resnet_50_train_1_step(self, run_mode):
    """Tests training HuggingFace ResNet-50."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="microsoft/resnet-50",
            sequence_length=512,
            batch_size=1,
        ),
        model_and_input_factory=model_utils.huggingface_resnet_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.generic_train_factory,
            grad_accumulation_steps=1,
        ),
    )
    self.run_performance_benchmark_test(config, _HF_RESNET_50_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_timm_vit_small_dino_forward(self, run_mode):
    """Tests the forward pass of TIMM ViT Small DINO."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.TIMM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="timm/vit_small_patch8_224.dino",
            custom_kwargs={"input_shape": (1, 3, 224, 224)},
        ),
        model_and_input_factory=model_utils.timm_model_builder,
        eval_factory=benchmark_function_db.timm_eval_factory,
    )
    self.run_performance_benchmark_test(
        config, _TIMM_VIT_SMALL_DINO_BENCHMARK_NAME
    )

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_timm_vit_small_dino_train_1_step(self, run_mode):
    """Tests training TIMM ViT Small DINO."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.TIMM,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="timm/vit_small_patch8_224.dino",
            custom_kwargs={"input_shape": (1, 3, 224, 224)},
        ),
        model_and_input_factory=model_utils.timm_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.generic_train_factory,
            grad_accumulation_steps=1,
        ),
    )
    self.run_performance_benchmark_test(
        config, _TIMM_VIT_SMALL_DINO_BENCHMARK_NAME
    )

  # ============================================================================
  # 7. Wan Diffuser Family
  # ============================================================================

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_wan_2_2_ti2v_5b_forward(self, run_mode):
    """Tests the forward pass of Wan-2.2-TI2V-5B."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
            common.Platform.XLA_CPU,
            common.Platform.TORCH_CPU,
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
      test_utils.generate_run_mode_configs([common.RunMode.COMPILED])
  )
  def test_wan_2_2_ti2v_5b_forward_256x256(self, run_mode):
    """Tests the forward pass of Wan-2.2-TI2V-5B with 256x256 resolution."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
            common.Platform.XLA_CPU,
            common.Platform.TORCH_CPU,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_DIFFUSER,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            custom_kwargs={
                "height": 256,
                "width": 256,
                "num_frames": 1,
            },
        ),
        model_and_input_factory=model_utils.huggingface_diffuser_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _WAN_2_2_TI2V_5B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_wan_2_2_ti2v_5b_backward(self, run_mode):
    """Tests the backward pass of Wan-2.2-TI2V-5B."""

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
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

  @parameterized.named_parameters(
      test_utils.generate_run_mode_configs([common.RunMode.COMPILED])
  )
  def test_wan_2_2_ti2v_5b_backward_256x256(self, run_mode):
    """Tests the backward pass of Wan-2.2-TI2V-5B with 256x256 resolution."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_DIFFUSER,
        run_mode=run_mode,
        is_training=True,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            custom_kwargs={
                "height": 256,
                "width": 256,
                "num_frames": 1,
            },
        ),
        model_and_input_factory=model_utils.huggingface_diffuser_model_builder,
        train_factory=functools.partial(
            benchmark_function_db.huggingface_diffuser_train_factory,
            grad_accumulation_steps=1,
        ),
    )
    self.run_performance_benchmark_test(config, _WAN_2_2_TI2V_5B_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_wan_2_2_ti2v_5b_backward_grad_acc_1(self, run_mode):
    """Tests the backward pass of Wan-2.2-TI2V-5B with grad accumulation steps set to 1."""

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
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
            grad_accumulation_steps=1,
        ),
    )
    self.run_performance_benchmark_test(config, _WAN_2_2_TI2V_5B_BENCHMARK_NAME)

  # ============================================================================
  # 8. Audio / Whisper Models
  # ============================================================================

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_whisper_large_v3_forward(self, run_mode):
    """Tests the forward pass of Whisper-Large-v3."""
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
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

  # ============================================================================
  # 9. DETR Object Detection Family
  # ============================================================================

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_detr_resnet_50_forward(self, run_mode):
    """Tests the forward pass of DETR ResNet-50."""
    if self._is_torchax_backend():
      self.skipTest("Not supported on TorchAX")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_VISION,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="facebook/detr-resnet-50",
            custom_kwargs={"input_shape": (8, 3, 800, 800)},
        ),
        model_and_input_factory=model_utils.huggingface_detr_resnet_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _DETR_RESNET_50_BENCHMARK_NAME)

  # ============================================================================
  # 10. Vision/Video Models
  # ============================================================================

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_vjepa2_vitl_forward(self, run_mode):
    """Tests the forward pass of VJEPA2-ViT-L."""
    if self._is_torchax_backend():
      self.skipTest("Not supported on TorchAX")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.V5E_1X1,
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="facebook/vjepa2-vitl-fpc64-256",
            batch_size=1,
            sequence_length=None,
        ),
        model_and_input_factory=model_utils.vjepa_2_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_VJEPA2_VITL_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_vjepa2_vitg_forward(self, run_mode):
    """Tests the forward pass of VJEPA2-ViT-G."""
    if self._is_torchax_backend():
      self.skipTest("Not supported on TorchAX")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="facebook/vjepa2-vitg-fpc64-384",
            batch_size=1,
            sequence_length=None,
        ),
        model_and_input_factory=model_utils.vjepa_2_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(config, _HF_VJEPA2_VITG_BENCHMARK_NAME)

  @parameterized.named_parameters(test_utils.generate_run_mode_configs())
  def test_nemotron_3_nano_30b_6_layers_forward(self, run_mode):
    """Tests the forward pass of NVIDIA-Nemotron-3-Nano-30B-A3B-BF16."""
    if self._is_torchax_backend():
      self.skipTest("Not supported on TorchAX")

    def modify_config_hook(config):
      config.layers_block_type = config.layers_block_type[:6]
      return config

    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
        run_mode=run_mode,
        is_training=False,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            sequence_length=512,
            batch_size=1,
            custom_kwargs={"modify_config_hook": modify_config_hook},
        ),
        model_and_input_factory=model_utils.huggingface_llm_model_builder,
        eval_factory=benchmark_function_db.huggingface_eval_factory,
    )
    self.run_performance_benchmark_test(
        config, _HF_NEMOTRON_3_NANO_30B_BENCHMARK_NAME
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  # g3_multiprocessing is required to run absltest.main() in a multiprocess
  # environment. It doesn't affect single process runs.
  # See: go/g3_multiprocessing#resolution.
  g3_multiprocessing.handle_test_main(absltest.main)
