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
import torch
import torch.multiprocessing as mp
from examples.benchmarks.e2e import benchmark_function_db
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import common
from examples.benchmarks.e2e import layer_configs
from examples.benchmarks.e2e import model_utils
from examples.benchmarks.e2e import performance_utils
from examples.benchmarks.e2e import test_utils
from torch_tpu._internal.distributed import multiprocessing

_ALL_RUN_MODES = (
    common.RunMode.EAGER_DEFAULT,
    common.RunMode.EAGER_OPTIMIZED,
    common.RunMode.COMPILED,
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
_GLU_LAYER_BENCHMARK_NAME = "glu"
_CONV1D_LAYER_BENCHMARK_NAME = "conv1d"
_MULTIHEAD_ATTENTION_LAYER_BENCHMARK_NAME = "multihead_attention"
_QWEN3_ATTENTION_BENCHMARK_NAME = "qwen3_attention"
_QWEN3_RMSNORM_BENCHMARK_NAME = "qwen3_rms_norm"
_QWEN3_MLP_BENCHMARK_NAME = "qwen3_mlp"
_QWEN3_ROTARY_EMBEDDING_BENCHMARK_NAME = "qwen3_rotary_embedding"
_SDPA_LAYER_BENCHMARK_NAME = "sdpa"
_LINEAR_TIMM_LAYER_BENCHMARK_NAME = "linear_timm"
_CONV2D_TIMM_LAYER_BENCHMARK_NAME = "conv2d_timm"
_BATCHNORM2D_TIMM_LAYER_BENCHMARK_NAME = "batchnorm2d_timm"
_AVGPOOL2D_TIMM_LAYER_BENCHMARK_NAME = "avgpool2d_timm"
_SELECT_ADAPTIVE_POOL2D_TIMM_LAYER_BENCHMARK_NAME = (
    "select_adaptive_pool2d_timm"
)
_ADAPTIVE_AVG_POOL2D_TIMM_LAYER_BENCHMARK_NAME = "adaptive_avg_pool2d_timm"
_FLATTEN_TIMM_LAYER_BENCHMARK_NAME = "flatten_timm"
_BOTTLENECK_TIMM_LAYER_BENCHMARK_NAME = "bottleneck_timm"
_MAXPOOL2D_TIMM_LAYER_BENCHMARK_NAME = "maxpool2d_timm"
_RELU_TIMM_LAYER_BENCHMARK_NAME = "relu_timm"
_PRELU_BENCHMARK_NAME = "prelu"
_FFT_LAYER_BENCHMARK_NAME = "fft"
_SLICE_SCATTER_BENCHMARK_NAME = "slice_scatter"
_MAMBA2_BLOCK_BENCHMARK_NAME = "mamba2_block"
_NEMOTRON_H_MAMBA2_BLOCK_BENCHMARK_NAME = "nemotron_h_mamba2_block"
_MASKED_SOFTMAX_BENCHMARK_NAME = "masked_softmax"
_NATIVE_MULTI_HEAD_ATTENTION_BENCHMARK_NAME = "native_multi_head_attention"
_TRANSFORMER_ENCODER_LAYER_FWD_BENCHMARK_NAME = "transformer_encoder_layer_fwd"
_TOPK_LAYER_BENCHMARK_NAME = "topk"
_SORT_LAYER_BENCHMARK_NAME = "sort"
_LSTM_LAYER_BENCHMARK_NAME = "lstm"

_GRU_LAYER_BENCHMARK_NAME = "gru"


_DYNAMIC_SKIPS = {
    "slice_scatter": "Slice scatter dynamic shape support not implemented.",
    # Blocked by remote TPU/MLIR/JAX C++ backend compiler bugs:
    "nn.Embedding": (
        "Blocked by RankedTensorType MLIR storage uniquer assertion crashes "
        "inside standard LLVM MLIR backend library."
    ),
    "Qwen3Attention": (
        "Blocked by slice_primitive assertion checks crashes inside JAX C++"
        " compiler."
    ),
    "nn.RMSNorm": (
        "Blocked by CPU constant mapping restrictions in dynamic shapes "
        "compilation grids."
    ),
    "Qwen3RMSNorm": (
        "Blocked by CPU constant mapping restrictions in dynamic shapes "
        "compilation grids."
    ),
    "BertLayer": (
        "Blocked by unflatten view shape collisions under dynamic batches "
        "inside view reassociation engine."
    ),
    "BertSelfOutput": (
        "Blocked by unflatten view shape collisions under dynamic batches "
        "inside view reassociation engine."
    ),
    "BertIntermediate": (
        "Blocked by unflatten view shape collisions under dynamic batches "
        "inside view reassociation engine."
    ),
    "BertOutput": (
        "Blocked by unflatten view shape collisions under dynamic batches "
        "inside view reassociation engine."
    ),
    "Qwen3MLP": (
        "Blocked by unflatten view shape collisions under dynamic batches "
        "inside view reassociation engine."
    ),
    "linear_timm": (
        "Blocked by unflatten view shape collisions under dynamic batches "
        "inside view reassociation engine."
    ),
    # Spatial / 2D Timm layers that do not utilize 1D sequence length dynamism:
    "nn.Dropout": (
        "Spatial layer does not utilize sequence length dynamic parameters "
        "under our 1D dynamism model."
    ),
    "batchnorm2d_timm": (
        "Spatial layer does not utilize sequence length dynamic parameters "
        "under our 1D dynamism model."
    ),
    "avgpool2d_timm": (
        "Spatial layer does not utilize sequence length dynamic parameters "
        "under our 1D dynamism model."
    ),
    "select_adaptive_pool2d_timm": (
        "Spatial layer does not utilize sequence length dynamic parameters "
        "under our 1D dynamism model."
    ),
    "adaptive_avg_pool2d_timm": (
        "Spatial layer does not utilize sequence length dynamic parameters "
        "under our 1D dynamism model."
    ),
    "flatten_timm": (
        "Spatial layer does not utilize sequence length dynamic parameters "
        "under our 1D dynamism model."
    ),
    "bottleneck_timm": (
        "Spatial layer does not utilize sequence length dynamic parameters "
        "under our 1D dynamism model."
    ),
    "maxpool2d_timm": (
        "Spatial layer does not utilize sequence length dynamic parameters "
        "under our 1D dynamism model."
    ),
    "relu_timm": (
        "Spatial layer does not utilize sequence length dynamic parameters "
        "under our 1D dynamism model."
    ),
    # Harness/Input generation setup skips:
    "nn.BatchNorm1d": "BatchNorm1d dynamic input generation not yet updated.",
    "nn.LayerNorm": "LayerNorm dynamic input generation not yet updated.",
    "nn.Conv2d": "Conv2d dynamic input generation not yet updated.",
    "nn.Tanh": "Tanh dynamic input generation not yet updated.",
    "nn.GLU": "GLU dynamic input generation not yet updated.",
    "BertPooler": "BertPooler dynamic input generation not yet updated.",
    "BertEmbeddings": (
        "BertEmbeddings dynamic input generation not yet updated."
    ),
    "SiLUActivation": (
        "SiLUActivation dynamic input generation not yet updated."
    ),
}


class LayerPerformanceBenchmarks(test_utils.BenchmarkTest):
  """Tests for end-to-end performance benchmarks."""

  def run_performance_benchmark_test(
      self, config, benchmark_name, microbenchmark_name=None
  ):
    if config.is_training:
      config.sync_params = True
      config.train_factory = benchmark_function_db.simple_train_factory
    else:
      config.eval_factory = benchmark_function_db.simple_eval_factory
    if performance_utils.BOUNDED_DYNAMIC.value:
      model_name = config.model_and_input_args.model_name
      if model_name in _DYNAMIC_SKIPS or benchmark_name in _DYNAMIC_SKIPS:
        skip_reason = _DYNAMIC_SKIPS.get(model_name) or _DYNAMIC_SKIPS.get(
            benchmark_name
        )
        self.skipTest(
            f"Layer {model_name} (benchmark {benchmark_name}) is currently"
            f" blocked in dynamic execution mode: {skip_reason}"
        )
    if self._is_torchax_backend():
      # Layer specific skips.
      if config.is_training:
        batch_size = config.model_and_input_args.batch_size or 1
        seq_len = config.model_and_input_args.sequence_length or 1
        effective_seq_len = batch_size * seq_len
        if effective_seq_len >= 262144:
          self.skipTest("Benchmark would likely OOM for TorchAX on Forge.")

    super().run_performance_benchmark_test(
        config, benchmark_name, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.LINEAR_CONFIGS
      )
  )
  def test_linear_layer(self, run_mode, is_training, layer_config):
    if performance_utils.BOUNDED_DYNAMIC.value and layer_config.batch_size != 1:
      self.skipTest(
          "batch_size > 1 produces an ambiguous expand reshape error."
      )
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
    if run_mode == common.RunMode.COMPILED and is_training:
      self.skipTest(
          "Batchnorm1d in compiled mode with training doesn't stablize in cache"
          " misses.."
      )
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.Conv2d",
            batch_size=layer_config.batch_size,
            custom_kwargs={
                "in_channels": layer_config.in_channels,
                "out_channels": layer_config.out_channels,
                "kernel_size": layer_config.kernel_size,
                "stride": layer_config.stride,
                "padding": layer_config.padding,
                "dilation": layer_config.dilation,
                "groups": layer_config.groups,
                "bias": layer_config.bias,
                "channels_last": layer_config.channels_last,
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
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
          _ALL_RUN_MODES, (False,), layer_configs.FFT_CONFIGS
      )
  )
  def test_fft(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="fft.fft",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "hidden_size": layer_config.hidden_size,
                "dim": layer_config.dim,
                "norm": layer_config.norm,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _FFT_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (False,), layer_configs.SLICE_SCATTER_CONFIGS
      )
  )
  def test_slice_scatter(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="slice_scatter",
            custom_kwargs={
                "input_shape": layer_config.input_shape,
                "src_shape": layer_config.src_shape,
                "dim": layer_config.dim,
                "start": layer_config.start,
                "end": layer_config.end,
                "step": layer_config.step,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _SLICE_SCATTER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.DROPOUT_CONFIGS
      )
  )
  def test_dropout(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
        common.RunMode.EAGER_DEFAULT,
        common.RunMode.EAGER_OPTIMIZED,
    ):
      self.skipTest("RMSNorm training with eager mode fails.")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
          _ALL_RUN_MODES, (True, False), layer_configs.GLU_CONFIGS
      )
  )
  def test_glu(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.GLU",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "shape": layer_config.shape,
                "dim": layer_config.dim,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _GLU_LAYER_BENCHMARK_NAME, microbenchmark_name
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
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
          _ALL_RUN_MODES,
          (True, False),
          list(
              layer_configs.SdpaConfig.configs_with_backends(
                  torch.nn.attention.SDPBackend.MATH,
                  # Allow fallback to math backend.
                  [
                      torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                      torch.nn.attention.SDPBackend.MATH,
                  ],
              )
          ),
      )
  )
  def test_sdpa_tpu(self, run_mode, is_training, layer_config):
    if run_mode == common.RunMode.COMPILED:
      self.skipTest("SDPA is broken in compiled mode")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.f.scaled_dot_product_attention",
            batch_size=layer_config.batch_size,
            custom_kwargs={
                "embed_dim": layer_config.embed_dim,
                "q_seq_len": layer_config.q_seq_len,
                "q_num_heads": layer_config.q_num_heads,
                "kv_num_heads": layer_config.kv_num_heads,
                "qk_head_dim": layer_config.qk_head_dim,
                "v_head_dim": layer_config.v_head_dim,
                "is_causal": layer_config.is_causal,
                "enable_gqa": layer_config.enable_gqa,
                "backend": layer_config.backend,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _SDPA_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      # TODO(b/431285931) - Training known issue.
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES,
          (False,),
          list(
              layer_configs.SdpaConfig.configs_with_backends(
                  #   torch.nn.attention.SDPBackend.FLASH_ATTENTION, TODO(b/431285931) - Known issue.
                  torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                  torch.nn.attention.SDPBackend.MATH,
                  torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
              )
          ),
      )
  )
  def test_sdpa_cuda(self, run_mode, is_training, layer_config):
    same_heads = layer_config.q_num_heads == layer_config.kv_num_heads
    if (
        layer_config.backend
        == torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
        and not same_heads
    ):
      self.skipTest("Efficient attention doesn't support non-equal head dims.")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.f.scaled_dot_product_attention",
            batch_size=layer_config.batch_size,
            custom_kwargs={
                "embed_dim": layer_config.embed_dim,
                "q_seq_len": layer_config.q_seq_len,
                "q_num_heads": layer_config.q_num_heads,
                "kv_num_heads": layer_config.kv_num_heads,
                "qk_head_dim": layer_config.qk_head_dim,
                "v_head_dim": layer_config.v_head_dim,
                "is_causal": layer_config.is_causal,
                "enable_gqa": layer_config.enable_gqa,
                "backend": layer_config.backend,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _SDPA_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          (common.RunMode.COMPILED,),
          (False,),
          layer_configs.LINEAR_TIMM_CONFIGS,
      )
  )
  def test_linear_timm(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
        config, _LINEAR_TIMM_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          (common.RunMode.COMPILED,),
          (False,),
          layer_configs.CONV2D_TIMM_CONFIGS,
      )
  )
  def test_conv2d_timm(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
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
        config, _CONV2D_TIMM_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          (common.RunMode.COMPILED,),
          (False,),
          layer_configs.BATCHNORM2D_TIMM_CONFIGS,
      )
  )
  def test_batchnorm2d_timm(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.BatchNorm2d",
            batch_size=layer_config.batch_size,
            custom_kwargs={
                "num_features": layer_config.num_features,
                "height": layer_config.height,
                "width": layer_config.width,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _BATCHNORM2D_TIMM_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          (common.RunMode.COMPILED,),
          (False,),
          layer_configs.AVGPOOL2D_TIMM_CONFIGS,
      )
  )
  def test_avgpool2d_timm(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.AvgPool2d",
            batch_size=layer_config.batch_size,
            custom_kwargs={
                "channels": layer_config.channels,
                "height": layer_config.height,
                "width": layer_config.width,
                "kernel_size": layer_config.kernel_size,
                "stride": layer_config.stride,
                "padding": layer_config.padding,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _AVGPOOL2D_TIMM_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          (common.RunMode.COMPILED,),
          (False,),
          layer_configs.SELECT_ADAPTIVE_POOL2D_TIMM_CONFIGS,
      )
  )
  def test_select_adaptive_pool2d_timm(
      self, run_mode, is_training, layer_config
  ):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="SelectAdaptivePool2d",
            batch_size=layer_config.batch_size,
            custom_kwargs={
                "channels": layer_config.channels,
                "height": layer_config.height,
                "width": layer_config.width,
                "output_size": layer_config.output_size,
                "pool_type": layer_config.pool_type,
                "flatten": layer_config.flatten,
                "input_fmt": layer_config.input_fmt,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config,
        _SELECT_ADAPTIVE_POOL2D_TIMM_LAYER_BENCHMARK_NAME,
        microbenchmark_name,
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          (common.RunMode.COMPILED,),
          (False,),
          layer_configs.ADAPTIVE_AVG_POOL2D_TIMM_CONFIGS,
      )
  )
  def test_adaptive_avg_pool2d_timm(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.AdaptiveAvgPool2d",
            batch_size=layer_config.batch_size,
            custom_kwargs={
                "channels": layer_config.channels,
                "height": layer_config.height,
                "width": layer_config.width,
                "output_size": layer_config.output_size,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config,
        _ADAPTIVE_AVG_POOL2D_TIMM_LAYER_BENCHMARK_NAME,
        microbenchmark_name,
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          (common.RunMode.COMPILED,),
          (False,),
          layer_configs.FLATTEN_TIMM_CONFIGS,
      )
  )
  def test_flatten_timm(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.Flatten",
            batch_size=layer_config.shape[0],
            custom_kwargs={
                "shape": layer_config.shape,
                "start_dim": layer_config.start_dim,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _FLATTEN_TIMM_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          (common.RunMode.COMPILED,),
          (False,),
          layer_configs.BOTTLENECK_TIMM_CONFIGS,
      )
  )
  def test_bottleneck_timm(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Bottleneck",
            batch_size=layer_config.batch_size,
            custom_kwargs={
                "height": layer_config.height,
                "width": layer_config.width,
                "inplanes": layer_config.inplanes,
                "planes": layer_config.planes,
                "stride": layer_config.stride,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _BOTTLENECK_TIMM_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          (common.RunMode.COMPILED,),
          (False,),
          layer_configs.MAXPOOL2D_TIMM_CONFIGS,
      )
  )
  def test_maxpool2d_timm(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.MaxPool2d",
            batch_size=layer_config.batch_size,
            custom_kwargs={
                "channels": layer_config.channels,
                "height": layer_config.height,
                "width": layer_config.width,
                "kernel_size": layer_config.kernel_size,
                "stride": layer_config.stride,
                "padding": layer_config.padding,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _MAXPOOL2D_TIMM_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          (common.RunMode.COMPILED,),
          (False,),
          layer_configs.RELU_TIMM_CONFIGS,
      )
  )
  def test_relu_timm(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.ReLU",
            batch_size=layer_config.shape[0],
            custom_kwargs={
                "shape": layer_config.shape,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _RELU_TIMM_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          (common.RunMode.COMPILED,),
          (True, False),
          layer_configs.CONV1D_CONFIGS,
      )
  )
  def test_conv1d(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.Conv1d",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "in_channels": layer_config.in_channels,
                "out_channels": layer_config.out_channels,
                "kernel_size": layer_config.kernel_size,
                "stride": layer_config.stride,
                "padding": layer_config.padding,
                "dilation": layer_config.dilation,
                "groups": layer_config.groups,
                "bias": layer_config.bias,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _CONV1D_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          (common.RunMode.COMPILED,),
          (False,),
          layer_configs.MULTIHEAD_ATTENTION_CONFIGS,
      )
  )
  def test_multihead_attention(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.MultiheadAttention",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "embed_dim": layer_config.embed_dim,
                "num_heads": layer_config.num_heads,
                "dropout": layer_config.dropout,
                "bias": layer_config.bias,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _MULTIHEAD_ATTENTION_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.MAMBA2_BLOCK_CONFIGS
      )
  )
  def test_mamba2_block(self, run_mode, is_training, layer_config):
    if run_mode == common.RunMode.EAGER_DEFAULT and layer_config.batch_size > 1:
      self.skipTest("Eager mode is expected to OOM on large configurations")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="Mamba2Block",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "hidden_size": layer_config.hidden_size,
                "state_size": layer_config.state_size,
                "conv_kernel": layer_config.conv_kernel,
                "expand": layer_config.expand,
                "num_heads": layer_config.num_heads,
                "head_dim": layer_config.head_dim,
                "n_groups": layer_config.n_groups,
                "chunk_size": layer_config.chunk_size,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _MAMBA2_BLOCK_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES,
          (True, False),
          layer_configs.NEMOTRON_H_MAMBA2_BLOCK_CONFIGS,
      )
  )
  def test_nemotron_h_mamba2_block(
      self,
      run_mode,
      is_training,
      layer_config: layer_configs.NemotronHMamba2BlockConfig,
  ):
    if run_mode == common.RunMode.EAGER_DEFAULT and layer_config.batch_size > 1:
      self.skipTest("Eager mode is expected to OOM on large configurations")
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="NemotronHMamba2Block",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "hidden_size": layer_config.hidden_size,
                "state_size": layer_config.state_size,
                "conv_kernel": layer_config.conv_kernel,
                "expand": layer_config.expand,
                "num_heads": layer_config.num_heads,
                "head_dim": layer_config.head_dim,
                "n_groups": layer_config.n_groups,
                "chunk_size": layer_config.chunk_size,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _NEMOTRON_H_MAMBA2_BLOCK_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.PRELU_CONFIGS
      )
  )
  def test_prelu(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.PReLU",
            batch_size=layer_config.batch_size,
            custom_kwargs={
                "channels": layer_config.channels,
                "height": layer_config.height,
                "width": layer_config.width,
                "channel_wise": layer_config.channel_wise,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _PRELU_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.MASKED_SOFTMAX_CONFIGS
      )
  )
  def test_masked_softmax(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="aten._masked_softmax",
            batch_size=layer_config.batch_size,
            custom_kwargs={
                "num_heads": layer_config.num_heads,
                "q_seq_len": layer_config.q_seq_len,
                "kv_seq_len": layer_config.kv_seq_len,
                "dim": layer_config.dim,
                "mask_type": layer_config.mask_type,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _MASKED_SOFTMAX_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES,
          (True, False),
          layer_configs.NATIVE_MULTI_HEAD_ATTENTION_CONFIGS,
      )
  )
  def test_native_multi_head_attention(
      self, run_mode, is_training, layer_config
  ):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="aten._native_multi_head_attention",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "embed_dim": layer_config.embed_dim,
                "num_heads": layer_config.num_heads,
                "need_weights": layer_config.need_weights,
                "average_attn_weights": layer_config.average_attn_weights,
                "mask_type": layer_config.mask_type,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config,
        _NATIVE_MULTI_HEAD_ATTENTION_BENCHMARK_NAME,
        microbenchmark_name,
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.TOPK_CONFIGS
      )
  )
  def test_topk(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="torch.topk",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "num_features": layer_config.num_features,
                "k": layer_config.k,
                "dim": layer_config.dim,
                "largest": layer_config.largest,
                "sorted": layer_config.sorted,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _TOPK_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.SORT_CONFIGS
      )
  )
  def test_sort(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="torch.sort",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "num_features": layer_config.num_features,
                "dim": layer_config.dim,
                "descending": layer_config.descending,
                "stable": layer_config.stable,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _SORT_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES,
          (True, False),
          layer_configs.TRANSFORMER_ENCODER_LAYER_FWD_CONFIGS,
      )
  )
  def test_transformer_encoder_layer_fwd(
      self, run_mode, is_training, layer_config
  ):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="aten._transformer_encoder_layer_fwd",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "embed_dim": layer_config.embed_dim,
                "num_heads": layer_config.num_heads,
                "dim_feedforward": layer_config.dim_feedforward,
                "use_gelu": layer_config.use_gelu,
                "norm_first": layer_config.norm_first,
                "mask_type": layer_config.mask_type,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config,
        _TRANSFORMER_ENCODER_LAYER_FWD_BENCHMARK_NAME,
        microbenchmark_name,
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.LSTM_CONFIGS
      )
  )
  def test_lstm(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.LSTM",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "input_size": layer_config.input_size,
                "hidden_size": layer_config.hidden_size,
                "num_layers": layer_config.num_layers,
                "bias": layer_config.bias,
                "batch_first": layer_config.batch_first,
                "bidirectional": layer_config.bidirectional,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _LSTM_LAYER_BENCHMARK_NAME, microbenchmark_name
    )

  @parameterized.named_parameters(
      test_utils.generate_layer_test_configs(
          _ALL_RUN_MODES, (True, False), layer_configs.GRU_CONFIGS
      )
  )
  def test_gru(self, run_mode, is_training, layer_config):
    config = performance_utils.PerformanceBenchmarkConfig(
        supported_platforms=[
            common.Platform.GFC_1X1X1,
            common.Platform.B200_1,
        ],
        benchmark_category=benchmark_utils.BenchmarkCategory.ML_LAYER,
        run_mode=run_mode,
        is_training=is_training,
        model_and_input_factory=model_utils.ml_layer_model_builder,
        model_and_input_args=performance_utils.ModelAndInputArgs(
            model_name="nn.GRU",
            batch_size=layer_config.batch_size,
            sequence_length=layer_config.seq_len,
            custom_kwargs={
                "input_size": layer_config.input_size,
                "hidden_size": layer_config.hidden_size,
                "num_layers": layer_config.num_layers,
                "bias": layer_config.bias,
                "batch_first": layer_config.batch_first,
                "bidirectional": layer_config.bidirectional,
            },
        ),
    )
    microbenchmark_name = test_utils.get_microbenchmark_name(layer_config)
    self.run_performance_benchmark_test(
        config, _GRU_LAYER_BENCHMARK_NAME, microbenchmark_name
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  # multiprocessing is required to run absltest.main() in a multiprocess
  # environment. It doesn't affect single process runs.
  # See: go/multiprocessing#resolution.
  multiprocessing.handle_test_main(absltest.main)
