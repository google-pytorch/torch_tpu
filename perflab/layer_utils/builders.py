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

"""Builders for the models and inputs of perflab layer benchmarks."""

from collections.abc import Mapping
import dataclasses
from typing import Any, Callable

import torch
from transformers import activations
from transformers.models.bert import modeling_bert
from transformers.models.mamba2 import configuration_mamba2
from transformers.models.mamba2 import modeling_mamba2
from transformers.models.nemotron_h import configuration_nemotron_h
from transformers.models.nemotron_h import modeling_nemotron_h
from transformers.models.qwen3 import configuration_qwen3
from transformers.models.qwen3 import modeling_qwen3


@dataclasses.dataclass(frozen=True)
class LayerArgs:
  """Args to build a layer model and its example inputs.

  Attributes:
    model_name: The name of the layer to build, e.g. `"nn.Linear"`.
    batch_size: The batch size of the input. Some layers take the input shape
      from `custom_kwargs` instead.
    sequence_length: The sequence length of the input, for layers that use it.
    custom_kwargs: Layer-specific kwargs to build the model and example inputs.
  """

  model_name: str
  batch_size: int | None
  sequence_length: int | None = None
  custom_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def _generate_inputs(
    batch_size: int,
    sequence_length: int,
    shape_fn: Callable[[int, int], Any],
) -> Any:
  """Returns the example inputs that `shape_fn` builds for the static shape."""
  return shape_fn(batch_size, sequence_length)


def _get_base_bert_config():
  cfg = modeling_bert.BertConfig(
      vocab_size=30522,
      hidden_size=768,
      num_hidden_layers=12,
      num_attention_heads=12,
      intermediate_size=3072,
      hidden_dropout_prob=0.1,
      attention_probs_dropout_prob=0.1,
      max_position_embeddings=512,
  )
  cfg._attn_implementation = "eager"  # pylint: disable=protected-access
  return cfg


def _get_base_qwen3_config():
  # https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
  config = configuration_qwen3.Qwen3Config.from_dict({
      "architectures": ["Qwen3ForCausalLM"],
      "attention_bias": False,
      "attention_dropout": 0.0,
      "bos_token_id": 151643,
      "eos_token_id": 151645,
      "head_dim": 128,
      "hidden_act": "silu",
      "hidden_size": 4096,
      "initializer_range": 0.02,
      "intermediate_size": 12288,
      "max_position_embeddings": 40960,
      "max_window_layers": 36,
      "model_type": "qwen3",
      "num_attention_heads": 32,
      "num_hidden_layers": 36,
      "num_key_value_heads": 8,
      "rms_norm_eps": 1e-06,
      "rope_scaling": None,
      "rope_theta": 1000000,
      "sliding_window": None,
      "tie_word_embeddings": False,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.51.0",
      "use_cache": True,
      "use_sliding_window": False,
      "vocab_size": 151936,
  })
  config._attn_implementation = "eager"  # pylint: disable=protected-access
  return config


@dataclasses.dataclass
class ModelAndInput:
  """Dataclass to hold the benchmarking model and its example inputs.

  Attributes:
      model: The model to run the benchmark on.
      example_inputs: The example inputs to run the model with. The model should
        be able to accept these inputs.
  """

  model: torch.nn.Module
  example_inputs: Any


def ml_layer_model_builder(
    layer_args: LayerArgs,
    device: str | torch.device,
    weights_dtype: torch.dtype,
    is_training: bool,
) -> ModelAndInput:
  """Returns a ModelAndInput for the specified ML layer.

  Args:
      layer_args: The layer and input args.
      device: The device to load the model and inputs on (e.g., 'tpu', 'cuda').
      weights_dtype: The data type for the model weights (e.g., torch.float32,
        torch.bfloat16).
      is_training: Whether the model is in training mode or eval mode.

  Returns:
      ModelAndInput dataclass.
  """
  model_name = layer_args.model_name
  # Typed as `Any`: each branch below reads only the dimensions its layer uses,
  # so the unused ones may be None.
  sequence_length: Any = layer_args.sequence_length
  batch_size: Any = layer_args.batch_size
  kwargs = layer_args.custom_kwargs

  if model_name == "nn.Linear":
    if "in_features" not in kwargs:
      raise ValueError("nn.Linear requires in_features to be specified.")
    if "out_features" not in kwargs:
      raise ValueError("nn.Linear requires out_features to be specified.")

    in_features = kwargs["in_features"]
    out_features = kwargs["out_features"]

    class LinearModel(torch.nn.Module):

      def __init__(self, in_features, out_features, dtype):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, dtype=dtype)

      def forward(self, x):
        return self.linear(x)

    model = LinearModel(in_features, out_features, dtype=weights_dtype)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            (bs, seq, in_features),
            dtype=weights_dtype,
            device=device,
        ),
    )
  elif model_name == "nn.Embedding":
    num_embeddings = kwargs["num_embeddings"]
    embedding_dim = kwargs["embedding_dim"]
    sparse = kwargs.get("sparse", False)

    class EmbeddingModel(torch.nn.Module):

      def __init__(self, num_embeddings, embedding_dim, sparse, dtype):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings, embedding_dim, sparse=sparse, dtype=dtype
        )

      def forward(self, x):
        return self.embedding(x)

    model = EmbeddingModel(
        num_embeddings, embedding_dim, sparse=sparse, dtype=weights_dtype
    )
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randint(
            0,
            num_embeddings,
            (bs, seq),
            dtype=torch.int32,  # Using int32 as per the reference
            device=device,
        ),
    )
  elif model_name == "nn.EmbeddingBag":
    num_embeddings = kwargs["num_embeddings"]
    embedding_dim = kwargs["embedding_dim"]
    mode = kwargs.get("mode", "sum")
    sparse = kwargs.get("sparse", False)

    class EmbeddingBagModel(torch.nn.Module):

      def __init__(self, num_embeddings, embedding_dim, mode, sparse, dtype):
        super().__init__()
        self.embedding_bag = torch.nn.EmbeddingBag(
            num_embeddings,
            embedding_dim,
            mode=mode,
            sparse=sparse,
            dtype=dtype,
        )

      def forward(self, x):
        return self.embedding_bag(x)

    model = EmbeddingBagModel(
        num_embeddings,
        embedding_dim,
        mode=mode,
        sparse=sparse,
        dtype=weights_dtype,
    )
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randint(
            0,
            num_embeddings,
            (bs, seq),
            dtype=torch.int32,
            device=device,
        ),
    )
  elif model_name == "nn.Dropout":
    p = kwargs["p"]
    input_shape = kwargs["shape"]

    class DropoutModel(torch.nn.Module):

      def __init__(self, p):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=p)

      def forward(self, x):
        return self.dropout(x)

    model = DropoutModel(p)
    example_inputs = torch.randn(
        input_shape,
        dtype=weights_dtype,
        device=device,
        requires_grad=is_training,
    )
  elif model_name == "nn.Tanh":
    input_shape = kwargs["shape"]

    class TanhModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()

      def forward(self, x):
        return self.tanh(x)

    model = TanhModel()
    example_inputs = torch.randn(
        input_shape,
        dtype=weights_dtype,
        device=device,
        requires_grad=is_training,
    )
  elif model_name == "nn.BatchNorm1d":
    num_features = kwargs["num_features"]

    class BatchNormModel(torch.nn.Module):

      def __init__(self, num_features, dtype):
        super().__init__()
        self.batchnorm = torch.nn.BatchNorm1d(num_features, dtype=dtype)

      def forward(self, x):
        return self.batchnorm(x)

    model = BatchNormModel(num_features, dtype=weights_dtype)
    example_inputs = torch.randn(
        (batch_size, num_features, sequence_length),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.BatchNorm2d":
    num_features = kwargs["num_features"]
    height = kwargs["height"]
    width = kwargs["width"]

    class BatchNorm2dModel(torch.nn.Module):

      def __init__(self, num_features, dtype):
        super().__init__()
        self.batchnorm = torch.nn.BatchNorm2d(num_features, dtype=dtype)

      def forward(self, x):
        return self.batchnorm(x)

    model = BatchNorm2dModel(num_features, dtype=weights_dtype)
    example_inputs = torch.randn(
        (batch_size, num_features, height, width),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.LayerNorm":
    normalized_shape = kwargs["normalized_shape"]
    # shape argument for input generation
    input_shape = kwargs["shape"]

    class LayerNormModel(torch.nn.Module):

      def __init__(self, normalized_shape, dtype):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(normalized_shape, dtype=dtype)

      def forward(self, x):
        return self.layernorm(x)

    model = LayerNormModel(normalized_shape, dtype=weights_dtype)
    example_inputs = torch.randn(
        input_shape,
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.Conv2d":
    in_channels = kwargs["in_channels"]
    out_channels = kwargs["out_channels"]
    kernel_size = kwargs["kernel_size"]
    stride = kwargs.get("stride", 1)
    padding = kwargs.get("padding", 0)
    dilation = kwargs.get("dilation", 1)
    groups = kwargs.get("groups", 1)
    bias = kwargs.get("bias", True)
    height = kwargs["height"]
    width = kwargs["width"]

    class Conv2dModel(torch.nn.Module):

      def __init__(
          self,
          in_channels,
          out_channels,
          kernel_size,
          stride,
          padding,
          dilation,
          groups,
          bias,
          dtype,
      ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            dtype=dtype,
        )

      def forward(self, x):
        return self.conv(x)

    model = Conv2dModel(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        dtype=weights_dtype,
    )
    example_inputs = torch.randn(
        (batch_size, in_channels, height, width),
        dtype=weights_dtype,
        device=device,
    )
    if kwargs.get("channels_last", False):
      model = model.to(memory_format=torch.channels_last)
      example_inputs = example_inputs.to(memory_format=torch.channels_last)
  elif model_name == "nn.RMSNorm":
    num_features = kwargs["num_features"]

    class RMSNormModel(torch.nn.Module):

      def __init__(self, num_features, dtype):
        super().__init__()
        self.rmsnorm = torch.nn.RMSNorm(num_features, dtype=dtype)

      def forward(self, x):
        return self.rmsnorm(x)

    model = RMSNormModel(num_features, dtype=weights_dtype)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            (bs, seq, num_features),
            dtype=weights_dtype,
            device=device,
        ),
    )
  elif model_name == "fft.fft":
    dim = kwargs.get("dim", -1)
    norm = kwargs.get("norm", "backward")
    hidden_size = kwargs["hidden_size"]

    class FftModel(torch.nn.Module):

      def __init__(self, dim, norm):
        super().__init__()
        self.dim = dim
        self.norm = norm

      def forward(self, x):
        return torch.fft.fft(x, dim=self.dim, norm=self.norm)

    model = FftModel(dim, norm)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.complex(
            torch.randn(
                (bs, seq, hidden_size), dtype=torch.float32, device=device
            ),
            torch.randn(
                (bs, seq, hidden_size), dtype=torch.float32, device=device
            ),
        ),
    )

  elif model_name == "slice_scatter":
    dim = kwargs["dim"]
    start = kwargs["start"]
    end = kwargs["end"]
    step = kwargs["step"]
    input_shape = kwargs["input_shape"]
    src_shape = kwargs["src_shape"]

    class SliceScatterModel(torch.nn.Module):

      def __init__(self, dim, start, end, step):
        super().__init__()
        self.dim = dim
        self.start = start
        self.end = end
        self.step = step

      def forward(self, x, src):
        return torch.slice_scatter(
            x,
            src,
            dim=self.dim,
            start=self.start,
            end=self.end,
            step=self.step,
        )

    model = SliceScatterModel(dim, start, end, step)
    example_inputs = (
        torch.randn(input_shape, dtype=weights_dtype, device=device),
        torch.randn(src_shape, dtype=weights_dtype, device=device),
    )

  elif model_name == "repeat_interleave":
    input_shape = kwargs["input_shape"]
    repeats = kwargs["repeats"]
    dim = kwargs["dim"]

    class RepeatInterleaveModel(torch.nn.Module):

      def __init__(self, repeats: int, dim: int):
        super().__init__()
        self.repeats = repeats
        self.dim = dim

      def forward(self, x):
        return torch.repeat_interleave(x, self.repeats, dim=self.dim)

    model = RepeatInterleaveModel(repeats, dim)
    example_inputs = (
        torch.randn(
            input_shape,
            dtype=weights_dtype,
            device=device,
            requires_grad=is_training,
        ),
    )

  elif model_name == "nn.AvgPool1d":
    kernel_size = kwargs["kernel_size"]
    stride = kwargs["stride"]
    padding = kwargs["padding"]
    channels = kwargs["channels"]
    length = kwargs["length"]

    class AvgPool1dModel(torch.nn.Module):

      def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.pool = torch.nn.AvgPool1d(
            kernel_size, stride=stride, padding=padding, ceil_mode=True
        )

      def forward(self, x):
        return self.pool(x)

    model = AvgPool1dModel(kernel_size, stride, padding)
    example_inputs = torch.randn(
        (batch_size, channels, length),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.AvgPool2d":
    kernel_size = kwargs["kernel_size"]
    stride = kwargs["stride"]
    padding = kwargs["padding"]
    channels = kwargs["channels"]
    height = kwargs["height"]
    width = kwargs["width"]

    class AvgPool2dModel(torch.nn.Module):

      def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(
            kernel_size, stride=stride, padding=padding, ceil_mode=True
        )

      def forward(self, x):
        return self.pool(x)

    model = AvgPool2dModel(kernel_size, stride, padding)
    example_inputs = torch.randn(
        (batch_size, channels, height, width),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.AvgPool3d":
    kernel_size = kwargs["kernel_size"]
    stride = kwargs["stride"]
    padding = kwargs["padding"]
    channels = kwargs["channels"]
    depth = kwargs["depth"]
    height = kwargs["height"]
    width = kwargs["width"]

    class AvgPool3dModel(torch.nn.Module):

      def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.pool = torch.nn.AvgPool3d(
            kernel_size, stride=stride, padding=padding, ceil_mode=True
        )

      def forward(self, x):
        return self.pool(x)

    model = AvgPool3dModel(kernel_size, stride, padding)
    example_inputs = torch.randn(
        (batch_size, channels, depth, height, width),
        dtype=torch.float32,
        device=device,
    )
  elif model_name == "nn.AdaptiveAvgPool2d":
    output_size = kwargs["output_size"]
    channels = kwargs["channels"]
    height = kwargs["height"]
    width = kwargs["width"]

    class AdaptiveAvgPool2dModel(torch.nn.Module):

      def __init__(self, output_size):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size)

      def forward(self, x):
        return self.pool(x)

    model = AdaptiveAvgPool2dModel(output_size)
    example_inputs = torch.randn(
        (batch_size, channels, height, width),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.AdaptiveAvgPool3d":
    output_size = kwargs["output_size"]
    channels = kwargs["channels"]
    depth = kwargs["depth"]
    height = kwargs["height"]
    width = kwargs["width"]

    class AdaptiveAvgPool3dModel(torch.nn.Module):

      def __init__(self, output_size):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool3d(output_size)

      def forward(self, x):
        return self.pool(x)

    model = AdaptiveAvgPool3dModel(output_size)
    example_inputs = torch.randn(
        (batch_size, channels, depth, height, width),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.Flatten":
    start_dim = kwargs["start_dim"]
    shape = kwargs["shape"]

    class FlattenModel(torch.nn.Module):

      def __init__(self, start_dim):
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=start_dim)

      def forward(self, x):
        return self.flatten(x)

    model = FlattenModel(start_dim)
    example_inputs = torch.randn(
        shape,
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.MaxPool2d":
    kernel_size = kwargs["kernel_size"]
    stride = kwargs["stride"]
    padding = kwargs["padding"]
    channels = kwargs["channels"]
    height = kwargs["height"]
    width = kwargs["width"]

    class MaxPool2dModel(torch.nn.Module):

      def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(
            kernel_size, stride=stride, padding=padding
        )

      def forward(self, x):
        return self.pool(x)

    model = MaxPool2dModel(kernel_size, stride, padding)
    example_inputs = torch.randn(
        (batch_size, channels, height, width),
        dtype=weights_dtype,
        device=device,
        requires_grad=is_training,
    )
  elif model_name == "nn.MaxPool1d":
    kernel_size = kwargs["kernel_size"]
    stride = kwargs["stride"]
    padding = kwargs["padding"]
    channels = kwargs["channels"]
    length = kwargs["length"]

    class MaxPool1dModel(torch.nn.Module):
      """1D max pooling benchmark wrapper module."""

      def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.pool = torch.nn.MaxPool1d(
            kernel_size, stride=stride, padding=padding
        )

      def forward(self, x):
        return self.pool(x)

    model = MaxPool1dModel(kernel_size, stride, padding)
    # Generate 1D input tensor of shape (batch_size, channels, length)
    example_inputs = torch.randn(
        (batch_size, channels, length),
        dtype=weights_dtype,
        device=device,
        requires_grad=is_training,
    )
  elif model_name == "nn.MaxPool3d":
    kernel_size = kwargs["kernel_size"]
    stride = kwargs["stride"]
    padding = kwargs["padding"]
    channels = kwargs["channels"]
    depth = kwargs["depth"]
    height = kwargs["height"]
    width = kwargs["width"]

    class MaxPool3dModel(torch.nn.Module):
      """3D max pooling benchmark wrapper module."""

      def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.pool = torch.nn.MaxPool3d(
            kernel_size, stride=stride, padding=padding
        )

      def forward(self, x):
        return self.pool(x)

    model = MaxPool3dModel(kernel_size, stride, padding)
    # Generate 3D input tensor of shape (batch_size, channels, depth, height, width)
    example_inputs = torch.randn(
        (batch_size, channels, depth, height, width),
        dtype=weights_dtype,
        device=device,
        requires_grad=is_training,
    )
  elif model_name == "nn.ReLU":
    shape = kwargs["shape"]

    class ReLUModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

      def forward(self, x):
        return self.relu(x)

    model = ReLUModel()
    example_inputs = torch.randn(
        shape,
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.PReLU":
    channels = kwargs["channels"]
    height = kwargs["height"]
    width = kwargs["width"]
    channel_wise = kwargs.get("channel_wise", True)

    class PreluModel(torch.nn.Module):

      def __init__(self, dtype):
        super().__init__()
        self.prelu = torch.nn.PReLU(
            num_parameters=channels if channel_wise else 1, dtype=dtype
        )

      def forward(self, x):
        return self.prelu(x)

    model = PreluModel(dtype=weights_dtype)

    def shape_fn(bs, seq):
      return torch.randn(
          (bs, channels, height, width),
          dtype=weights_dtype,
          device=device,
          requires_grad=is_training,
      )

    example_inputs = _generate_inputs(batch_size, sequence_length, shape_fn)
  elif model_name == "nn.GLU":
    dim = kwargs.get("dim", -1)
    shape = kwargs["shape"]
    num_features = shape[-1]

    class GLUModel(torch.nn.Module):

      def __init__(self, dim, features, is_training, dtype):
        super().__init__()
        self.dim = dim
        self.is_training = is_training
        if is_training:
          self.linear = torch.nn.Linear(features, features, dtype=dtype)

      def forward(self, x):
        if self.is_training:
          x = self.linear(x)
        return torch.nn.functional.glu(x, dim=self.dim)

    model = GLUModel(dim, num_features, is_training, dtype=weights_dtype)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            (bs, seq, num_features),
            dtype=weights_dtype,
            device=device,
        ),
    )
  elif model_name == "nn.Conv1d":
    in_channels = kwargs["in_channels"]
    out_channels = kwargs["out_channels"]
    kernel_size = kwargs["kernel_size"]
    stride = kwargs.get("stride", 1)
    padding = kwargs.get("padding", 0)
    dilation = kwargs.get("dilation", 1)
    groups = kwargs.get("groups", 1)
    bias = kwargs.get("bias", True)

    class Conv1dModel(torch.nn.Module):

      def __init__(
          self,
          in_channels,
          out_channels,
          kernel_size,
          stride,
          padding,
          dilation,
          groups,
          bias,
          dtype,
      ):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            dtype=dtype,
        )

      def forward(self, x):
        return self.conv(x)

    model = Conv1dModel(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        dtype=weights_dtype,
    )
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            (bs, in_channels, seq),
            dtype=weights_dtype,
            device=device,
        ),
    )
  elif model_name == "nn.MultiheadAttention":
    embed_dim = kwargs["embed_dim"]
    num_heads = kwargs["num_heads"]
    dropout = kwargs.get("dropout", 0.0)
    bias = kwargs.get("bias", True)

    class MultiheadAttentionModel(torch.nn.Module):

      def __init__(self, embed_dim, num_heads, dropout, bias, dtype):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            dtype=dtype,
        )

      def forward(self, query, key, value, key_padding_mask=None):
        return self.mha(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]

    model = MultiheadAttentionModel(
        embed_dim, num_heads, dropout, bias, dtype=weights_dtype
    )
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: (
            torch.randn(seq, bs, embed_dim, dtype=weights_dtype, device=device),
            torch.randn(seq, bs, embed_dim, dtype=weights_dtype, device=device),
            torch.randn(seq, bs, embed_dim, dtype=weights_dtype, device=device),
            None,
        ),
    )

  elif model_name == "SelectAdaptivePool2d":
    import timm.layers

    output_size = kwargs["output_size"]
    pool_type = kwargs["pool_type"]
    flatten = kwargs["flatten"]
    input_fmt = kwargs["input_fmt"]
    channels = kwargs["channels"]
    height = kwargs["height"]
    width = kwargs["width"]

    class SelectAdaptivePool2dModel(torch.nn.Module):

      def __init__(self, output_size, pool_type, flatten, input_fmt):
        super().__init__()
        self.pool = timm.layers.SelectAdaptivePool2d(
            output_size=output_size,
            pool_type=pool_type,
            flatten=flatten,
            input_fmt=input_fmt,
        )

      def forward(self, x):
        return self.pool(x)

    model = SelectAdaptivePool2dModel(
        output_size, pool_type, flatten, input_fmt
    )
    example_inputs = torch.randn(
        (batch_size, channels, height, width),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "Bottleneck":
    from timm.models import resnet

    inplanes = kwargs["inplanes"]
    planes = kwargs["planes"]
    stride = kwargs["stride"]
    height = kwargs["height"]
    width = kwargs["width"]

    class BottleneckModel(torch.nn.Module):

      def __init__(self, inplanes, planes, stride):
        super().__init__()
        self.bottleneck = resnet.Bottleneck(
            inplanes=inplanes, planes=planes, stride=stride
        )

      def forward(self, x):
        return self.bottleneck(x)

    model = BottleneckModel(inplanes, planes, stride).to(dtype=weights_dtype)
    example_inputs = torch.randn(
        (batch_size, inplanes, height, width),
        dtype=weights_dtype,
        device=device,
    )

  elif model_name == "BertLayer":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertLayer(cfg)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: (
            torch.randn(
                bs,
                seq,
                cfg.hidden_size,
                dtype=weights_dtype,
                device=device,
            ),
            None,
        ),
    )

    class BertLayerWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states, attention_mask):
        return self.m(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            output_attentions=False,
            cache_position=None,
        )[0]

    model = BertLayerWrapper(model).to(dtype=weights_dtype)

  elif model_name == "BertSelfOutput":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertSelfOutput(cfg)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: (
            torch.randn(
                bs,
                seq,
                cfg.hidden_size,
                dtype=weights_dtype,
                device=device,
            ),
            torch.randn(
                bs,
                seq,
                cfg.hidden_size,
                dtype=weights_dtype,
                device=device,
            ),
        ),
    )

    class BertSelfOutputWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states, input_tensor):
        return self.m(hidden_states=hidden_states, input_tensor=input_tensor)

    model = BertSelfOutputWrapper(model).to(dtype=weights_dtype)

  elif model_name == "BertIntermediate":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertIntermediate(cfg)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            bs,
            seq,
            cfg.hidden_size,
            dtype=weights_dtype,
            device=device,
        ),
    )

    class BertIntermediateWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states):
        return self.m(hidden_states)

    model = BertIntermediateWrapper(model).to(dtype=weights_dtype)

  elif model_name == "BertOutput":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertOutput(cfg)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: (
            torch.randn(
                bs,
                seq,
                cfg.intermediate_size,
                dtype=weights_dtype,
                device=device,
            ),
            torch.randn(
                bs,
                seq,
                cfg.hidden_size,
                dtype=weights_dtype,
                device=device,
            ),
        ),
    )

    class BertOutputWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states, input_tensor):
        return self.m(hidden_states=hidden_states, input_tensor=input_tensor)

    model = BertOutputWrapper(model).to(dtype=weights_dtype)

  elif model_name == "GELUActivation":
    cfg = _get_base_bert_config()
    model = activations.GELUActivation(use_gelu_python=False)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            bs,
            seq,
            cfg.intermediate_size,
            dtype=weights_dtype,
            device=device,
            requires_grad=is_training,
        ),
    )
  elif model_name == "BertPooler":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertPooler(cfg).to(dtype=weights_dtype)
    example_inputs = torch.randn(
        batch_size,
        sequence_length,
        cfg.hidden_size,
        dtype=weights_dtype,
        device=device,
    )

  elif model_name == "BertEmbeddings":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertEmbeddings(cfg).to(dtype=weights_dtype)
    input_ids = torch.randint(
        0,
        cfg.vocab_size,
        (batch_size, sequence_length),
        dtype=torch.long,
        device=device,
    )
    example_inputs = input_ids
  elif model_name == "Qwen3Attention":
    config = _get_base_qwen3_config()
    model = modeling_qwen3.Qwen3Attention(config, layer_idx=0)

    head_dim = config.hidden_size // config.num_attention_heads

    def _qwen3_shape_fn(bs, seq):
      cos = torch.randn(1, seq, head_dim, device=device, dtype=weights_dtype)
      sin = torch.randn(1, seq, head_dim, device=device, dtype=weights_dtype)
      return (
          torch.randn(
              bs,
              seq,
              config.hidden_size,
              dtype=weights_dtype,
              device=device,
          ),
          (cos, sin),
          None,
      )

    example_inputs = _generate_inputs(
        batch_size, sequence_length, _qwen3_shape_fn
    )

    class Qwen3AttentionWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states, position_embeddings, attention_mask):
        return self.m(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )[0]

    model = Qwen3AttentionWrapper(model)
    model = model.to(dtype=weights_dtype)

  elif model_name == "Qwen3RMSNorm":
    hidden_size = kwargs["hidden_size"]
    model = modeling_qwen3.Qwen3RMSNorm(hidden_size=hidden_size)
    model = model.to(dtype=weights_dtype)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            bs,
            seq,
            hidden_size,
            dtype=weights_dtype,
            device=device,
        ),
    )

  elif model_name == "Qwen3MLP":
    config = configuration_qwen3.Qwen3Config(
        hidden_size=kwargs["hidden_size"],
        intermediate_size=kwargs["intermediate_size"],
        hidden_act="silu",
    )
    model = modeling_qwen3.Qwen3MLP(config)
    model = model.to(dtype=weights_dtype)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            bs,
            seq,
            config.hidden_size,
            dtype=weights_dtype,
            device=device,
        ),
    )

  elif model_name == "SiLUActivation":
    model = activations.SiLUActivation()
    input_shape = kwargs["shape"]
    example_inputs = torch.randn(
        input_shape,
        dtype=weights_dtype,
        device=device,
    )

  elif model_name == "Qwen3RotaryEmbedding":
    config = configuration_qwen3.Qwen3Config(
        max_position_embeddings=kwargs["max_position_embeddings"],
        rope_theta=kwargs["rope_theta"],  # pyrefly: ignore[unexpected-keyword]
    )
    model = modeling_qwen3.Qwen3RotaryEmbedding(config)
    head_dim = kwargs["head_dim"]

    position_ids = (
        torch.arange(0, sequence_length, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    example_inputs = (
        torch.randn(
            batch_size,
            sequence_length,
            head_dim,
            dtype=weights_dtype,
            device=device,
        ),
        position_ids,
    )

    class Qwen3RotaryEmbeddingWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, x, position_ids):
        return self.m(x, position_ids)  # returns cos, sin

    model = Qwen3RotaryEmbeddingWrapper(model)

  elif model_name == "nn.f.scaled_dot_product_attention":
    embed_dim = kwargs["embed_dim"]
    q_num_heads = kwargs["q_num_heads"]
    kv_num_heads = kwargs["kv_num_heads"]
    qk_head_dim = kwargs["qk_head_dim"]
    v_head_dim = kwargs["v_head_dim"]
    is_causal = kwargs["is_causal"]
    enable_gqa = kwargs["enable_gqa"]
    backend = kwargs["backend"]

    class AttentionLayer(torch.nn.Module):

      def __init__(self, dtype):
        super().__init__()
        self.q_proj = torch.nn.Linear(
            embed_dim,
            q_num_heads * qk_head_dim,
            bias=False,
            dtype=dtype,
        )
        self.k_proj = torch.nn.Linear(
            embed_dim,
            kv_num_heads * qk_head_dim,
            bias=False,
            dtype=dtype,
        )
        self.v_proj = torch.nn.Linear(
            embed_dim,
            kv_num_heads * v_head_dim,
            bias=False,
            dtype=dtype,
        )
        self.out_proj = torch.nn.Linear(
            q_num_heads * v_head_dim,
            embed_dim,
            bias=False,
            dtype=dtype,
        )

      def forward(self, x):
        bsz, q_len, _ = x.size()
        q = (
            self.q_proj(x)
            .view(bsz, q_len, q_num_heads, qk_head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(bsz, q_len, kv_num_heads, qk_head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(bsz, q_len, kv_num_heads, v_head_dim)
            .transpose(1, 2)
        )

        with torch.nn.attention.sdpa_kernel(backend):
          attn_output = torch.nn.functional.scaled_dot_product_attention(
              q, k, v, is_causal=is_causal, enable_gqa=enable_gqa
          )
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(bsz, q_len, q_num_heads * v_head_dim)
        )
        return self.out_proj(attn_output)

    model = AttentionLayer(dtype=weights_dtype)
    example_inputs = torch.randn(
        (batch_size, kwargs["q_seq_len"], embed_dim),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "aten._masked_softmax":
    dim = kwargs.get("dim", -1)
    mask_type = kwargs.get("mask_type", 2)
    num_heads = kwargs["num_heads"]
    q_seq_len = kwargs["q_seq_len"]
    kv_seq_len = kwargs["kv_seq_len"]

    class MaskedSoftmaxModel(torch.nn.Module):

      def __init__(self, dim, mask_type):
        super().__init__()
        self.dim = dim
        self.mask_type = mask_type

      def forward(self, x, mask):
        return torch.ops.aten._masked_softmax(
            x, mask, dim=self.dim, mask_type=self.mask_type
        )

    model = MaskedSoftmaxModel(dim, mask_type)
    model = model.to(dtype=weights_dtype)

    def shape_fn(bs, seq):
      del seq  # Unused: the static shapes come from `kwargs`.
      cur_q_seq_len = q_seq_len
      cur_kv_seq_len = kv_seq_len

      x = torch.randn(
          (bs, num_heads, cur_q_seq_len, cur_kv_seq_len),
          dtype=weights_dtype,
          device=device,
          requires_grad=is_training,
      )

      if mask_type == 0:
        mask = torch.randint(
            0,
            2,
            (cur_q_seq_len, cur_kv_seq_len),
            dtype=torch.bool,
            device=device,
        )
      elif mask_type == 1:
        mask = torch.randint(
            0, 2, (bs, cur_kv_seq_len), dtype=torch.bool, device=device
        )
      else:
        mask = torch.randint(
            0,
            2,
            (bs, num_heads, cur_q_seq_len, cur_kv_seq_len),
            dtype=torch.bool,
            device=device,
        )
      return (x, mask)

    example_inputs = _generate_inputs(batch_size, sequence_length, shape_fn)
  elif model_name == "aten._native_multi_head_attention":
    embed_dim = kwargs["embed_dim"]
    num_heads = kwargs["num_heads"]
    need_weights = kwargs.get("need_weights", True)
    average_attn_weights = kwargs.get("average_attn_weights", False)
    mask_type = kwargs.get("mask_type", None)

    class NativeMultiHeadAttentionModel(torch.nn.Module):

      def __init__(
          self,
          embed_dim,
          num_heads,
          need_weights,
          average_attn_weights,
          mask_type,
      ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.need_weights = need_weights
        self.average_attn_weights = average_attn_weights
        self.mask_type = mask_type

        self.qkv_weight = torch.nn.Parameter(
            torch.empty((3 * embed_dim, embed_dim), dtype=weights_dtype)
        )
        self.qkv_bias = torch.nn.Parameter(
            torch.empty(3 * embed_dim, dtype=weights_dtype)
        )
        self.proj_weight = torch.nn.Parameter(
            torch.empty((embed_dim, embed_dim), dtype=weights_dtype)
        )
        self.proj_bias = torch.nn.Parameter(
            torch.empty(embed_dim, dtype=weights_dtype)
        )
        torch.nn.init.normal_(self.qkv_weight)
        torch.nn.init.zeros_(self.qkv_bias)
        torch.nn.init.normal_(self.proj_weight)
        torch.nn.init.zeros_(self.proj_bias)

      def forward(self, query, key, value, mask=None):
        out, attn_weights = torch.ops.aten._native_multi_head_attention(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.qkv_weight,
            self.qkv_bias,
            self.proj_weight,
            self.proj_bias,
            mask=mask,
            need_weights=self.need_weights,
            average_attn_weights=self.average_attn_weights,
            mask_type=self.mask_type,
        )
        if self.need_weights:
          return out, attn_weights
        # Workaround: with need_weights=False, both the TPU and CUDA kernels
        # compute only `out` and return an undefined `at::Tensor()` (`None` in
        # Python) for `attn_weights`. However, the meta kernel predicts an empty
        # tensor, so Inductor's compiled wrapper fails in `assert_alignment` if
        # the unused second output is returned.
        # TODO(b/571244299): Return both outputs once Inductor handles this.
        return out

    model = NativeMultiHeadAttentionModel(
        embed_dim,
        num_heads,
        need_weights,
        average_attn_weights,
        mask_type,
    )
    model = model.to(dtype=weights_dtype)

    def shape_fn(bs, seq):
      query = torch.randn(
          (bs, seq, embed_dim),
          dtype=weights_dtype,
          device=device,
          requires_grad=is_training,
      )
      key = torch.randn(
          (bs, seq, embed_dim),
          dtype=weights_dtype,
          device=device,
          requires_grad=is_training,
      )
      value = torch.randn(
          (bs, seq, embed_dim),
          dtype=weights_dtype,
          device=device,
          requires_grad=is_training,
      )

      if mask_type is None:
        return (query, key, value)
      elif mask_type == 0:
        mask = torch.randint(
            0,
            2,
            (seq, seq),
            dtype=torch.bool,
            device=device,
        )
        return (query, key, value, mask)
      elif mask_type == 1:
        mask = torch.randint(
            0,
            2,
            (bs, seq),
            dtype=torch.bool,
            device=device,
        )
        return (query, key, value, mask)
      else:
        mask = torch.randint(
            0,
            2,
            (bs, num_heads, seq, seq),
            dtype=torch.bool,
            device=device,
        )
        return (query, key, value, mask)

    example_inputs = _generate_inputs(batch_size, sequence_length, shape_fn)
  elif model_name == "aten._transformer_encoder_layer_fwd":
    embed_dim = kwargs["embed_dim"]
    num_heads = kwargs["num_heads"]
    dim_feedforward = kwargs.get("dim_feedforward", 2048)
    use_gelu = kwargs.get("use_gelu", False)
    norm_first = kwargs.get("norm_first", False)
    eps = kwargs.get("eps", 1e-5)
    mask_type = kwargs.get("mask_type", None)

    class TransformerEncoderLayerFwdModel(torch.nn.Module):

      def __init__(
          self,
          embed_dim,
          num_heads,
          dim_feedforward,
          use_gelu,
          norm_first,
          eps,
          mask_type,
      ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.use_gelu = use_gelu
        self.norm_first = norm_first
        self.eps = eps
        self.mask_type = mask_type

        self.qkv_weight = torch.nn.Parameter(
            torch.empty((3 * embed_dim, embed_dim), dtype=weights_dtype)
        )
        self.qkv_bias = torch.nn.Parameter(
            torch.empty(3 * embed_dim, dtype=weights_dtype)
        )
        self.proj_weight = torch.nn.Parameter(
            torch.empty((embed_dim, embed_dim), dtype=weights_dtype)
        )
        self.proj_bias = torch.nn.Parameter(
            torch.empty(embed_dim, dtype=weights_dtype)
        )
        self.norm_weight_1 = torch.nn.Parameter(
            torch.empty(embed_dim, dtype=weights_dtype)
        )
        self.norm_bias_1 = torch.nn.Parameter(
            torch.empty(embed_dim, dtype=weights_dtype)
        )
        self.norm_weight_2 = torch.nn.Parameter(
            torch.empty(embed_dim, dtype=weights_dtype)
        )
        self.norm_bias_2 = torch.nn.Parameter(
            torch.empty(embed_dim, dtype=weights_dtype)
        )
        self.ffn_weight_1 = torch.nn.Parameter(
            torch.empty((dim_feedforward, embed_dim), dtype=weights_dtype)
        )
        self.ffn_bias_1 = torch.nn.Parameter(
            torch.empty(dim_feedforward, dtype=weights_dtype)
        )
        self.ffn_weight_2 = torch.nn.Parameter(
            torch.empty((embed_dim, dim_feedforward), dtype=weights_dtype)
        )
        self.ffn_bias_2 = torch.nn.Parameter(
            torch.empty(embed_dim, dtype=weights_dtype)
        )

        torch.nn.init.normal_(self.qkv_weight)
        torch.nn.init.zeros_(self.qkv_bias)
        torch.nn.init.normal_(self.proj_weight)
        torch.nn.init.zeros_(self.proj_bias)
        torch.nn.init.ones_(self.norm_weight_1)
        torch.nn.init.zeros_(self.norm_bias_1)
        torch.nn.init.ones_(self.norm_weight_2)
        torch.nn.init.zeros_(self.norm_bias_2)
        torch.nn.init.normal_(self.ffn_weight_1)
        torch.nn.init.zeros_(self.ffn_bias_1)
        torch.nn.init.normal_(self.ffn_weight_2)
        torch.nn.init.zeros_(self.ffn_bias_2)

      def forward(self, src, mask=None):
        return torch.ops.aten._transformer_encoder_layer_fwd(
            src,
            self.embed_dim,
            self.num_heads,
            self.qkv_weight,
            self.qkv_bias,
            self.proj_weight,
            self.proj_bias,
            self.use_gelu,
            self.norm_first,
            self.eps,
            self.norm_weight_1,
            self.norm_bias_1,
            self.norm_weight_2,
            self.norm_bias_2,
            self.ffn_weight_1,
            self.ffn_bias_1,
            self.ffn_weight_2,
            self.ffn_bias_2,
            mask=mask,
            mask_type=self.mask_type,
        )

    model = TransformerEncoderLayerFwdModel(
        embed_dim,
        num_heads,
        dim_feedforward,
        use_gelu,
        norm_first,
        eps,
        mask_type,
    )
    model = model.to(dtype=weights_dtype)

    def shape_fn(bs, seq):
      src = torch.randn(
          (bs, seq, embed_dim),
          dtype=weights_dtype,
          device=device,
          requires_grad=is_training,
      )
      if mask_type is None:
        return (src,)
      elif mask_type == 0:
        mask = torch.randint(
            0,
            2,
            (seq, seq),
            dtype=torch.bool,
            device=device,
        )
        return (src, mask)
      elif mask_type == 1:
        mask = torch.randint(
            0,
            2,
            (bs, seq),
            dtype=torch.bool,
            device=device,
        )
        return (src, mask)
      else:
        mask = torch.randint(
            0,
            2,
            (bs, num_heads, seq, seq),
            dtype=torch.bool,
            device=device,
        )
        return (src, mask)

    example_inputs = _generate_inputs(batch_size, sequence_length, shape_fn)
  elif model_name == "Mamba2Block":
    config = configuration_mamba2.Mamba2Config(
        hidden_size=kwargs["hidden_size"],
        state_size=kwargs["state_size"],
        conv_kernel=kwargs["conv_kernel"],
        expand=kwargs["expand"],
        num_heads=kwargs["num_heads"],
        head_dim=kwargs["head_dim"],
        n_groups=kwargs["n_groups"],
        chunk_size=kwargs["chunk_size"],
    )
    model = modeling_mamba2.Mamba2Block(config, layer_idx=0)

    class Mamba2BlockWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states):
        return self.m(hidden_states)

    model = Mamba2BlockWrapper(model).to(dtype=weights_dtype)

    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            bs, seq, kwargs["hidden_size"], dtype=weights_dtype, device=device
        ),
    )

  elif model_name == "NemotronHMamba2Block":
    config = configuration_nemotron_h.NemotronHConfig(
        hidden_size=kwargs["hidden_size"],
        ssm_state_size=kwargs["state_size"],
        conv_kernel=kwargs["conv_kernel"],
        expand=kwargs["expand"],
        mamba_num_heads=kwargs["num_heads"],
        mamba_head_dim=kwargs["head_dim"],
        n_groups=kwargs["n_groups"],
        chunk_size=kwargs["chunk_size"],
    )
    config.use_mamba_kernels = False
    config.layers_block_type = ["linear_attention"]
    model = modeling_nemotron_h.NemotronHBlock(config, layer_idx=0)

    class NemotronHMamba2BlockWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states):
        return self.m(hidden_states)

    model = NemotronHMamba2BlockWrapper(model).to(dtype=weights_dtype)

    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            bs, seq, kwargs["hidden_size"], dtype=weights_dtype, device=device
        ),
    )

  elif model_name == "torch.topk":
    k = kwargs["k"]
    dim = kwargs.get("dim", None)
    largest = kwargs.get("largest", True)
    sorted_out = kwargs.get("sorted", True)
    num_features = kwargs["num_features"]

    class TopKModel(torch.nn.Module):

      def __init__(self, k, dim, largest, sorted_out):
        super().__init__()
        self.k = k
        self.dim = dim
        self.largest = largest
        self.sorted_out = sorted_out

      def forward(self, x):
        if self.dim is None:
          values, indices = torch.topk(
              x,
              self.k,
              largest=self.largest,
              sorted=self.sorted_out,
          )
        else:
          values, indices = torch.topk(
              x,
              self.k,
              dim=self.dim,
              largest=self.largest,
              sorted=self.sorted_out,
          )
        return values, indices

    model = TopKModel(k, dim, largest, sorted_out)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            (bs, seq, num_features),
            dtype=weights_dtype,
            device=device,
            requires_grad=is_training,
        ),
    )

  elif model_name == "torch.sort":
    dim = kwargs.get("dim", None)
    descending = kwargs.get("descending", False)
    stable = kwargs.get("stable", False)
    num_features = kwargs["num_features"]

    class SortModel(torch.nn.Module):
      """Sort op benchmark model wrapper.

      Wraps torch.sort behind a trainable nn.Linear so that in training mode the
      backward pass propagates a real gradient to a parameter (proj.weight).
      The benchmark harness only synchronizes model.parameters() gradients, so
      without a parameter on the differentiable path the backward would never be
      forced.
      """

      def __init__(self, num_features, dim, descending, stable, dtype):
        super().__init__()
        self.proj = torch.nn.Linear(num_features, num_features, dtype=dtype)
        self.dim = dim
        self.descending = descending
        self.stable = stable

      def forward(self, x):
        x = self.proj(x)
        # torch.sort requires an int dim; default to the last dimension (-1)
        # when dim is unset.
        dim = self.dim if self.dim is not None else -1
        values, indices = torch.sort(
            x,
            dim=dim,
            descending=self.descending,
            stable=self.stable,
        )
        return values, indices

    model = SortModel(
        num_features, dim, descending, stable, dtype=weights_dtype
    )
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            (bs, seq, num_features),
            dtype=weights_dtype,
            device=device,
        ),
    )

  elif model_name == "nn.LSTM":
    input_size = kwargs["input_size"]
    hidden_size = kwargs["hidden_size"]
    num_layers = kwargs.get("num_layers", 1)
    bias = kwargs.get("bias", True)
    batch_first = kwargs.get("batch_first", True)
    bidirectional = kwargs.get("bidirectional", False)

    class LstmModel(torch.nn.Module):

      def __init__(
          self,
          input_size,
          hidden_size,
          num_layers,
          bias,
          batch_first,
          bidirectional,
          dtype,
      ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional,
            dtype=dtype,
        )

      def forward(self, x):
        return self.lstm(x)[0]

    model = LstmModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        bidirectional=bidirectional,
        dtype=weights_dtype,
    )
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            (bs, seq, input_size) if batch_first else (seq, bs, input_size),
            dtype=weights_dtype,
            device=device,
            requires_grad=is_training,
        ),
    )

  elif model_name == "nn.GRU":
    input_size = kwargs["input_size"]
    hidden_size = kwargs["hidden_size"]
    num_layers = kwargs.get("num_layers", 1)
    bias = kwargs.get("bias", True)
    batch_first = kwargs.get("batch_first", True)
    bidirectional = kwargs.get("bidirectional", False)

    class GruModel(torch.nn.Module):

      def __init__(
          self,
          input_size,
          hidden_size,
          num_layers,
          bias,
          batch_first,
          bidirectional,
          dtype,
      ):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional,
            dtype=dtype,
        )

      def forward(self, x):
        return self.gru(x)[0]

    model = GruModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        bidirectional=bidirectional,
        dtype=weights_dtype,
    )
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            (bs, seq, input_size) if batch_first else (seq, bs, input_size),
            dtype=weights_dtype,
            device=device,
            requires_grad=is_training,
        ),
    )
  elif model_name == "associative_scan":
    channels = kwargs["channels"]
    dim = kwargs.get("dim", -1)
    mode = kwargs.get("mode", "add")

    def _add_combine(x, y):
      return x + y

    def _ssm_combine(left, right):
      a_l, b_l = left
      a_r, b_r = right
      return a_l * a_r, a_r * b_l + b_r

    class AssociativeScanModel(torch.nn.Module):

      def __init__(self, dim: int, mode: str):
        super().__init__()
        self.dim = dim
        self.mode = mode

      def forward(self, *xs):
        if self.mode == "ssm":
          a, b = xs
          return torch._higher_order_ops.associative_scan(
              _ssm_combine, (a, b), dim=self.dim, combine_mode="pointwise"
          )
        return torch._higher_order_ops.associative_scan(
            _add_combine, xs[0], dim=self.dim, combine_mode="pointwise"
        )

    model = AssociativeScanModel(dim=dim, mode=mode)
    shape_fn = (
        (lambda bs, seq: (bs, seq, channels))
        if dim in (1, -2)
        else (lambda bs, seq: (bs, channels, seq))
    )
    if mode == "ssm":
      example_inputs = _generate_inputs(
          batch_size,
          sequence_length,
          lambda bs, seq: (
              torch.sigmoid(
                  torch.randn(
                      shape_fn(bs, seq),
                      dtype=weights_dtype,
                      device=device,
                  )
              ),
              torch.randn(
                  shape_fn(bs, seq),
                  dtype=weights_dtype,
                  device=device,
              ),
          ),
      )
    else:
      example_inputs = _generate_inputs(
          batch_size,
          sequence_length,
          lambda bs, seq: torch.randn(
              shape_fn(bs, seq),
              dtype=weights_dtype,
              device=device,
          ),
      )
  elif model_name == "upsample_linear1d":
    channels = kwargs["channels"]
    input_length = kwargs["input_length"]
    output_size = kwargs.get("output_size", None)
    scale_factor = kwargs.get("scale_factor", None)
    align_corners = kwargs.get("align_corners", False)

    class UpsampleLinear1dModel(torch.nn.Module):

      def __init__(self, output_size, scale_factor, align_corners):
        super().__init__()
        self.upsample = torch.nn.Upsample(
            size=output_size,
            scale_factor=scale_factor,
            mode="linear",
            align_corners=align_corners,
        )

      def forward(self, x):
        return self.upsample(x)

    model = UpsampleLinear1dModel(output_size, scale_factor, align_corners)
    example_inputs = torch.randn(
        (batch_size, channels, input_length),
        dtype=weights_dtype,
        device=device,
        requires_grad=is_training,
    )
  elif model_name == "upsample_bicubic2d_aa":
    channels = kwargs["channels"]
    input_height = kwargs["input_height"]
    input_width = kwargs["input_width"]
    output_size = kwargs.get("output_size", None)
    scale_factor = kwargs.get("scale_factor", None)
    align_corners = kwargs.get("align_corners", False)

    class UpsampleBicubic2dAaModel(torch.nn.Module):

      def __init__(self, output_size, scale_factor, align_corners):
        super().__init__()
        self.output_size = output_size
        self.scale_factor = scale_factor
        self.align_corners = align_corners

      def forward(self, x):
        return torch.nn.functional.interpolate(
            x,
            size=self.output_size,
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=self.align_corners,
            antialias=True,
        )

    model = UpsampleBicubic2dAaModel(output_size, scale_factor, align_corners)
    example_inputs = torch.randn(
        (batch_size, channels, input_height, input_width),
        dtype=weights_dtype,
        device=device,
        requires_grad=is_training,
    )
  elif model_name == "upsample_bilinear2d_aa":
    channels = kwargs["channels"]
    input_height = kwargs["input_height"]
    input_width = kwargs["input_width"]
    output_size = kwargs.get("output_size", None)
    scale_factor = kwargs.get("scale_factor", None)
    align_corners = kwargs.get("align_corners", False)

    class UpsampleBilinear2dAaModel(torch.nn.Module):

      def __init__(self, output_size, scale_factor, align_corners):
        super().__init__()
        self.output_size = output_size
        self.scale_factor = scale_factor
        self.align_corners = align_corners

      def forward(self, x):
        return torch.nn.functional.interpolate(
            x,
            size=self.output_size,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=self.align_corners,
            antialias=True,
        )

    model = UpsampleBilinear2dAaModel(output_size, scale_factor, align_corners)
    example_inputs = torch.randn(
        (batch_size, channels, input_height, input_width),
        dtype=weights_dtype,
        device=device,
        requires_grad=is_training,
    )
  elif model_name == "unique_dim":
    input_shape = kwargs["input_shape"]
    dim = kwargs["dim"]
    return_inverse = kwargs.get("return_inverse", True)
    return_counts = kwargs.get("return_counts", True)

    class UniqueDimModel(torch.nn.Module):

      def __init__(self, dim: int, return_inverse: bool, return_counts: bool):
        super().__init__()
        self.dim = dim
        self.return_inverse = return_inverse
        self.return_counts = return_counts

      def forward(self, x):
        return torch.unique(
            x,
            sorted=True,
            return_inverse=self.return_inverse,
            return_counts=self.return_counts,
            dim=self.dim,
        )

    model = UniqueDimModel(dim, return_inverse, return_counts)
    # Create tensor with ~50% duplicate slices along `dim` so unique_dim does
    # non-trivial deduplication work.
    num_slices = input_shape[dim]
    half_slices = max(1, num_slices // 2)
    base_shape = list(input_shape)
    base_shape[dim] = half_slices
    if weights_dtype in (torch.int32, torch.int64):
      base = torch.randint(
          0, 100, tuple(base_shape), dtype=weights_dtype, device=device
      )
    else:
      base = torch.randn(tuple(base_shape), dtype=weights_dtype, device=device)
    idx = torch.randint(0, half_slices, (num_slices,), device=device)
    example_inputs = torch.index_select(base, dim, idx)
  elif model_name == "linalg_cholesky_ex":
    matrix_size = kwargs["matrix_size"]
    upper = kwargs.get("upper", False)
    input_dtype = (
        weights_dtype
        if weights_dtype in (
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        )
        else torch.float32
    )

    class LinalgCholeskyExModel(torch.nn.Module):

      def __init__(self, upper: bool) -> None:
        super().__init__()
        self.upper = upper

      def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        factor, info = torch.linalg.cholesky_ex(x, upper=self.upper)
        return factor, info

    model = LinalgCholeskyExModel(upper=upper)
    raw = torch.randn(
        (batch_size, matrix_size, matrix_size),
        dtype=input_dtype,
        device=device,
    )
    eye = torch.eye(matrix_size, dtype=input_dtype, device=device)
    example_inputs = raw @ raw.mH + matrix_size * eye
    if is_training:
      example_inputs = example_inputs.detach().requires_grad_(True)
  else:
    raise ValueError(f"Unknown ML layer model: {model_name}")

  model = model.to(device)
  if is_training:
    model.train()
  else:
    model.eval()

  return ModelAndInput(model=model, example_inputs=example_inputs)
