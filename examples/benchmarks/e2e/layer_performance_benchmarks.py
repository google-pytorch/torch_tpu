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

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import layer_configs
from examples.benchmarks.e2e import performance_utils
from examples.benchmarks.e2e import test_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

_ALL_RUN_MODES = (
    benchmark_utils.RunMode.EAGER_DEFAULT,
    benchmark_utils.RunMode.EAGER_OPTIMIZED,
    benchmark_utils.RunMode.COMPILED,
)

_LINEAR_LAYER_BENCHMARK_NAME = "linear"
_BATCHNORM1D_LAYER_BENCHMARK_NAME = "batchnorm1d"
_LAYERNORM_LAYER_BENCHMARK_NAME = "layernorm"
_CONV2D_LAYER_BENCHMARK_NAME = "conv2d"
_RMSNORM_LAYER_BENCHMARK_NAME = "rmsnorm"
_EMBEDDING_LAYER_BENCHMARK_NAME = "embedding"
_DROPOUT_LAYER_BENCHMARK_NAME = "dropout"
_TANH_LAYER_BENCHMARK_NAME = "tanh"
_GELU_ACTIVATION_BENCHMARK_NAME = "gelu_activation"
_BERT_LAYER_BENCHMARK_NAME = "bert_layer"
_BERT_SELF_OUTPUT_BENCHMARK_NAME = "bert_self_output"
_BERT_INTERMEDIATE_BENCHMARK_NAME = "bert_intermediate"
_BERT_OUTPUT_BENCHMARK_NAME = "bert_output"
_BERT_POOLER_BENCHMARK_NAME = "bert_pooler"
_BERT_EMBEDDINGS_BENCHMARK_NAME = "bert_embeddings"
_SILU_ACTIVATION_BENCHMARK_NAME = "silu_activation"
_QWEN3_ATTENTION_BENCHMARK_NAME = "qwen3_attention"
_QWEN3_RMSNORM_BENCHMARK_NAME = "qwen3_rms_norm"
_QWEN3_MLP_BENCHMARK_NAME = "qwen3_mlp"
_QWEN3_ROTARY_EMBEDDING_BENCHMARK_NAME = "qwen3_rotary_embedding"
_DEEPSEEK_PARALLEL_EMBEDDING_BENCHMARK_NAME = "deepseek_parallel_embedding"
_DEEPSEEK_RMSNORM_BENCHMARK_NAME = "deepseek_rms_norm"
_DEEPSEEK_EXPERT_BENCHMARK_NAME = "deepseek_expert"


class LayerPerformanceBenchmarks(test_utils.BenchmarkTest):
  """Tests for end-to-end performance benchmarks."""

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.LINEAR_CONFIGS
      )
  )
  def test_linear_layer(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.Linear",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "in_features": layer_config.in_features,
                "out_features": layer_config.out_features,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _LINEAR_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.BATCH_NORM_CONFIGS
      )
  )
  def test_batchnorm1d(self, run_mode, is_training, layer_config):
    # TODO(b/486204316): Fix batchnorm1d training with compiled mode on TPU.
    if run_mode == benchmark_utils.RunMode.COMPILED and is_training:
      self.skipTest(
          "Batchnorm1d in compiled mode with training doesn't stablize in cache"
          " misses.."
      )
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.BatchNorm1d",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "num_features": layer_config.num_features,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _BATCHNORM1D_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.LAYER_NORM_CONFIGS
      )
  )
  def test_layernorm(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.LayerNorm",
            custom_kwargs={
                "normalized_shape": layer_config.normalized_shape,
                "shape": layer_config.shape,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _LAYERNORM_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.CONV2D_CONFIGS
      )
  )
  def test_conv2d(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.Conv2d",
            batch_size=layer_config.batch_size,
            custom_kwargs={
                "in_channels": layer_config.in_channels,
                "out_channels": layer_config.out_channels,
                "kernel_size": layer_config.kernel_size,
                "stride": layer_config.stride,
                "padding": layer_config.padding,
                "height": layer_config.height,
                "width": layer_config.width,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _CONV2D_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.EMBEDDING_CONFIGS
      )
  )
  def test_embedding(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.Embedding",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "num_embeddings": layer_config.num_embeddings,
                "embedding_dim": layer_config.embedding_dim,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _EMBEDDING_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.DROPOUT_CONFIGS
      )
  )
  def test_dropout(self, run_mode, is_training, layer_config):
    # TODO: b/494430218 - Fix dropout training.
    if is_training:
      self.skipTest("Dropout test fails in training mode.")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.Dropout",
            custom_kwargs={
                "p": layer_config.p,
                "shape": layer_config.shape,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _DROPOUT_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.TANH_CONFIGS
      )
  )
  def test_tanh(self, run_mode, is_training, layer_config):
    # TODO: b/494430218 - Fix tanh training.
    if is_training:
      self.skipTest("Tanh test fails in training mode.")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.Tanh",
            custom_kwargs={
                "shape": layer_config.shape,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _TANH_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (False, True), layer_configs.BERT_LAYER_CONFIGS
      )
  )
  def test_gelu_activation(self, run_mode, is_training, layer_config):
    # TODO: b/494430218 - Fix gelu_activation training.
    self.skipTest("Gelu activation test fails in training mode.")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="GELUActivation",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _GELU_ACTIVATION_BENCHMARK_NAME, microbenchmark_name
    )

  # TODO(b/484415655): Known bert training issue.
  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (False,), layer_configs.BERT_LAYER_CONFIGS
      )
  )
  def test_bert_layer(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="BertLayer",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _BERT_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (False,), layer_configs.BERT_LAYER_CONFIGS
      )
  )
  def test_bert_self_output(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="BertSelfOutput",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _BERT_SELF_OUTPUT_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (False,), layer_configs.BERT_LAYER_CONFIGS
      )
  )
  def test_bert_intermediate(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="BertIntermediate",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _BERT_INTERMEDIATE_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (False,), layer_configs.BERT_LAYER_CONFIGS
      )
  )
  def test_bert_output(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="BertOutput",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _BERT_OUTPUT_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (False,), layer_configs.BERT_LAYER_CONFIGS
      )
  )
  def test_bert_pooler(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="BertPooler",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _BERT_POOLER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (False,), layer_configs.BERT_LAYER_CONFIGS
      )
  )
  def test_bert_embeddings(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="BertEmbeddings",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _BERT_EMBEDDINGS_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.RMS_NORM_CONFIGS
      )
  )
  def test_rmsnorm(self, run_mode, is_training, layer_config):
    # TODO: b/494430218 - Fix RMSNorm training.
    if is_training and run_mode in (
        benchmark_utils.RunMode.EAGER_DEFAULT,
        benchmark_utils.RunMode.EAGER_OPTIMIZED,
    ):
      self.skipTest("RMSNorm training with eager mode fails.")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.RMSNorm",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "num_features": layer_config.num_features,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _RMSNORM_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.QWEN3_CONFIGS
      )
  )
  def test_qwen3_attention(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Qwen3Attention",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={},
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _QWEN3_ATTENTION_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.QWEN3_CONFIGS
      )
  )
  def test_qwen3_rms_norm(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Qwen3RMSNorm",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={"hidden_size": layer_config.hidden_size},
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _QWEN3_RMSNORM_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.QWEN3_CONFIGS
      )
  )
  def test_qwen3_mlp(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Qwen3MLP",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "hidden_size": layer_config.hidden_size,
                "intermediate_size": layer_config.intermediate_size,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _QWEN3_MLP_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (False,), layer_configs.SILU_CONFIGS
      )
  )
  def test_silu_activation(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="SiLUActivation",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={"shape": layer_config.shape},
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _SILU_ACTIVATION_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (False,), layer_configs.QWEN3_CONFIGS
      )
  )
  def test_qwen3_rotary_embedding(self, run_mode, is_training, layer_config):
    self.skipTest("TODO(b/484415655): Investigate cache miss.")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Qwen3RotaryEmbedding",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={"head_dim": layer_config.head_dim},
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _QWEN3_ROTARY_EMBEDDING_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.DEEPSEEK_CONFIGS
      )
  )
  def test_deepseek_parallel_embedding(
      self, run_mode, is_training, layer_config
  ):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="DeepSeekParallelEmbedding",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "vocab_size": layer_config.vocab_size,
                "dim": layer_config.dim,
                "inter_dim": layer_config.inter_dim,
                "moe_inter_dim": layer_config.moe_inter_dim,
                "n_layers": layer_config.n_layers,
                "n_dense_layers": layer_config.n_dense_layers,
                "n_heads": layer_config.n_heads,
                "n_routed_experts": layer_config.n_routed_experts,
                "n_shared_experts": layer_config.n_shared_experts,
                "n_activated_experts": layer_config.n_activated_experts,
                "in_features": layer_config.in_features,
                "out_features": layer_config.out_features,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config,
        _DEEPSEEK_PARALLEL_EMBEDDING_BENCHMARK_NAME,
        microbenchmark_name,
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.DEEPSEEK_CONFIGS
      )
  )
  def test_deepseek_rms_norm(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="DeepSeekRMSNorm",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "vocab_size": layer_config.vocab_size,
                "dim": layer_config.dim,
                "inter_dim": layer_config.inter_dim,
                "moe_inter_dim": layer_config.moe_inter_dim,
                "n_layers": layer_config.n_layers,
                "n_dense_layers": layer_config.n_dense_layers,
                "n_heads": layer_config.n_heads,
                "n_routed_experts": layer_config.n_routed_experts,
                "n_shared_experts": layer_config.n_shared_experts,
                "n_activated_experts": layer_config.n_activated_experts,
                "in_features": layer_config.in_features,
                "out_features": layer_config.out_features,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _DEEPSEEK_RMSNORM_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.DEEPSEEK_CONFIGS
      )
  )
  def test_deepseek_expert(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            benchmark_utils.Platform.GFC_1X1X1,
            benchmark_utils.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="DeepSeekExpert",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "vocab_size": layer_config.vocab_size,
                "dim": layer_config.dim,
                "inter_dim": layer_config.inter_dim,
                "moe_inter_dim": layer_config.moe_inter_dim,
                "n_layers": layer_config.n_layers,
                "n_dense_layers": layer_config.n_dense_layers,
                "n_heads": layer_config.n_heads,
                "n_routed_experts": layer_config.n_routed_experts,
                "n_shared_experts": layer_config.n_shared_experts,
                "n_activated_experts": layer_config.n_activated_experts,
                "in_features": layer_config.in_features,
                "out_features": layer_config.out_features,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _DEEPSEEK_EXPERT_BENCHMARK_NAME, microbenchmark_name
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  # g3_multiprocessing is required to run absltest.main() in a multiprocess
  # environment. It doesn't affect single process runs.
  # See: go/g3_multiprocessing#resolution.
  g3_multiprocessing.handle_test_main(absltest.main)
