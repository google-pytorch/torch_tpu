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

"""Shared layer configurations for benchmarks."""

import dataclasses
from typing import Any
import torch


@dataclasses.dataclass
class LinearConfig:
  batch_size: int
  seq_len: int
  in_features: int
  out_features: int


LINEAR_CONFIGS = (
    # Default config for smoke test.
    LinearConfig(
        batch_size=1,
        seq_len=128,
        in_features=128,
        out_features=128,
    ),
    # Configs for Llama3 70B for MLP layers
    LinearConfig(
        batch_size=1,
        seq_len=8192,
        in_features=8192,
        out_features=28672,
    ),
    LinearConfig(
        batch_size=1,
        seq_len=8192,
        in_features=28672,
        out_features=8192,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=8192,
        in_features=8192,
        out_features=28672,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=8192,
        in_features=28672,
        out_features=8192,
    ),
    # Configs for Qwen3 480B MLP layers
    LinearConfig(
        batch_size=1,
        seq_len=8192,
        in_features=6144,
        out_features=2560,
    ),
    LinearConfig(
        batch_size=1,
        seq_len=8192,
        in_features=2560,
        out_features=6144,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=8192,
        in_features=6144,
        out_features=2560,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=8192,
        in_features=2560,
        out_features=6144,
    ),
    # Configs for Gemma3 27B MLP layers
    LinearConfig(
        batch_size=1,
        seq_len=8192,
        in_features=4608,
        out_features=36864,
    ),
    LinearConfig(
        batch_size=1,
        seq_len=8192,
        in_features=36864,
        out_features=4608,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=8192,
        in_features=4608,
        out_features=36864,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=8192,
        in_features=36864,
        out_features=4608,
    ),
    # Configs for BERT
    LinearConfig(
        batch_size=32,
        seq_len=128,
        in_features=768,
        out_features=768,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=128,
        in_features=768,
        out_features=3072,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=128,
        in_features=3072,
        out_features=768,
    ),
    # Conformer configs
    LinearConfig(
        batch_size=32,
        seq_len=128,
        in_features=512,
        out_features=2048,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=128,
        in_features=2048,
        out_features=512,
    ),
    LinearConfig(
        batch_size=32,
        seq_len=128,
        in_features=512,
        out_features=512,
    ),
)

LINEAR_TIMM_CONFIGS = (
    LinearConfig(batch_size=16, seq_len=1, in_features=2048, out_features=1000),
)


@dataclasses.dataclass
class BatchNormConfig:
  batch_size: int
  seq_len: int
  num_features: int


BATCH_NORM_CONFIGS = (
    # Default config for smoke test.
    BatchNormConfig(
        batch_size=1,
        seq_len=128,
        num_features=128,
    ),
    # Larger configs.
    BatchNormConfig(
        batch_size=32,
        seq_len=8192,
        num_features=8192,
    ),
    # High-batch, more TPU friendly shape.
    BatchNormConfig(
        batch_size=2056,
        seq_len=512,
        num_features=1024,
    ),
    # Conformer config
    BatchNormConfig(
        batch_size=32,
        seq_len=128,
        num_features=512,
    ),
)


@dataclasses.dataclass
class BatchNorm2dConfig:
  batch_size: int
  num_features: int
  height: int
  width: int


BATCHNORM2D_TIMM_CONFIGS = (
    BatchNorm2dConfig(batch_size=16, num_features=32, height=112, width=112),
    BatchNorm2dConfig(batch_size=16, num_features=64, height=56, width=56),
    BatchNorm2dConfig(batch_size=16, num_features=64, height=112, width=112),
    BatchNorm2dConfig(batch_size=16, num_features=128, height=28, width=28),
    BatchNorm2dConfig(batch_size=16, num_features=128, height=56, width=56),
    BatchNorm2dConfig(batch_size=16, num_features=256, height=14, width=14),
    BatchNorm2dConfig(batch_size=16, num_features=256, height=28, width=28),
    BatchNorm2dConfig(batch_size=16, num_features=256, height=56, width=56),
    BatchNorm2dConfig(batch_size=16, num_features=512, height=7, width=7),
    BatchNorm2dConfig(batch_size=16, num_features=512, height=14, width=14),
    BatchNorm2dConfig(batch_size=16, num_features=512, height=28, width=28),
    BatchNorm2dConfig(batch_size=16, num_features=1024, height=14, width=14),
    BatchNorm2dConfig(batch_size=16, num_features=2048, height=7, width=7),
)


@dataclasses.dataclass
class AvgPool2dConfig:
  batch_size: int
  channels: int
  height: int
  width: int
  kernel_size: int
  stride: int
  padding: int


AVGPOOL2D_TIMM_CONFIGS = (
    AvgPool2dConfig(
        batch_size=16,
        channels=256,
        height=56,
        width=56,
        kernel_size=2,
        stride=2,
        padding=0,
    ),
    AvgPool2dConfig(
        batch_size=16,
        channels=512,
        height=28,
        width=28,
        kernel_size=2,
        stride=2,
        padding=0,
    ),
    AvgPool2dConfig(
        batch_size=16,
        channels=1024,
        height=14,
        width=14,
        kernel_size=2,
        stride=2,
        padding=0,
    ),
)


@dataclasses.dataclass
class SelectAdaptivePool2dConfig:
  batch_size: int
  channels: int
  height: int
  width: int
  output_size: int = 1
  pool_type: str = "avg"
  flatten: bool = True
  input_fmt: str = "NCHW"


SELECT_ADAPTIVE_POOL2D_TIMM_CONFIGS = (
    SelectAdaptivePool2dConfig(
        batch_size=16,
        channels=2048,
        height=7,
        width=7,
        output_size=1,
        pool_type="avg",
        flatten=True,
        input_fmt="NCHW",
    ),
)


@dataclasses.dataclass
class AdaptiveAvgPool2dConfig:
  batch_size: int
  channels: int
  height: int
  width: int
  output_size: int = 1


ADAPTIVE_AVG_POOL2D_TIMM_CONFIGS = (
    AdaptiveAvgPool2dConfig(
        batch_size=16, channels=2048, height=7, width=7, output_size=1
    ),
)


@dataclasses.dataclass
class FlattenConfig:
  shape: tuple[int, ...]
  start_dim: int = 1


FLATTEN_TIMM_CONFIGS = (FlattenConfig(shape=(16, 2048, 1, 1), start_dim=1),)


@dataclasses.dataclass
class BottleneckConfig:
  batch_size: int
  height: int
  width: int
  inplanes: int
  planes: int
  stride: int = 1


BOTTLENECK_TIMM_CONFIGS = (
    BottleneckConfig(
        batch_size=16, height=7, width=7, inplanes=2048, planes=512, stride=1
    ),
    BottleneckConfig(
        batch_size=16, height=14, width=14, inplanes=1024, planes=256, stride=1
    ),
    BottleneckConfig(
        batch_size=16, height=28, width=28, inplanes=512, planes=128, stride=1
    ),
    BottleneckConfig(
        batch_size=16, height=56, width=56, inplanes=256, planes=64, stride=1
    ),
)


@dataclasses.dataclass
class MaxPool2dConfig:
  batch_size: int
  channels: int
  height: int
  width: int
  kernel_size: int
  stride: int
  padding: int


MAXPOOL2D_TIMM_CONFIGS = (
    MaxPool2dConfig(
        batch_size=16,
        channels=64,
        height=112,
        width=112,
        kernel_size=3,
        stride=2,
        padding=1,
    ),
)


@dataclasses.dataclass
class ReLUConfig:
  shape: tuple[int, ...]


RELU_TIMM_CONFIGS = (
    ReLUConfig(shape=(16, 32, 112, 112)),
    ReLUConfig(shape=(16, 64, 56, 56)),
    ReLUConfig(shape=(16, 64, 112, 112)),
    ReLUConfig(shape=(16, 128, 28, 28)),
    ReLUConfig(shape=(16, 128, 56, 56)),
    ReLUConfig(shape=(16, 256, 14, 14)),
    ReLUConfig(shape=(16, 256, 28, 28)),
    ReLUConfig(shape=(16, 256, 56, 56)),
    ReLUConfig(shape=(16, 512, 7, 7)),
    ReLUConfig(shape=(16, 512, 14, 14)),
    ReLUConfig(shape=(16, 512, 28, 28)),
    ReLUConfig(shape=(16, 1024, 14, 14)),
    ReLUConfig(shape=(16, 2048, 7, 7)),
)


@dataclasses.dataclass
class LayerNormConfig:
  batch_size: int
  seq_len: int
  num_features: int
  num_normalized_dims: int = 1

  @property
  def shape(self):
    return (self.batch_size, self.seq_len, self.num_features)

  @property
  def normalized_shape(self):
    return self.shape[-self.num_normalized_dims :]

  @property
  def reduction_axes(self):
    return range(len(self.shape) - self.num_normalized_dims, len(self.shape))


LAYER_NORM_CONFIGS = (
    # Default config for smoke test.
    LayerNormConfig(batch_size=1, seq_len=128, num_features=128),
    # Larger configs.
    LayerNormConfig(batch_size=32, seq_len=8192, num_features=8192),
    LayerNormConfig(
        batch_size=32,
        seq_len=8192,
        num_features=8192,
        num_normalized_dims=2,
    ),
    # BERT configs
    LayerNormConfig(
        batch_size=32,
        seq_len=128,
        num_features=768,
    ),
)


@dataclasses.dataclass
class Conv2dConfig:
  batch_size: int
  in_channels: int
  out_channels: int
  kernel_size: int
  stride: int
  padding: int
  height: int
  width: int


CONV2D_CONFIGS = (
    # Default config for smoke test.
    Conv2dConfig(
        batch_size=1,
        in_channels=2,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        height=128,
        width=128,
    ),
    # Larger configs.
    Conv2dConfig(
        batch_size=128,
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        height=256,
        width=256,
    ),
)

CONV2D_TIMM_CONFIGS = (
    Conv2dConfig(
        batch_size=16,
        in_channels=3,
        out_channels=32,
        kernel_size=3,
        stride=2,
        padding=1,
        height=224,
        width=224,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=1,
        height=112,
        width=112,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        height=112,
        width=112,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=64,
        out_channels=64,
        kernel_size=1,
        stride=1,
        padding=0,
        height=56,
        width=56,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        height=56,
        width=56,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=64,
        out_channels=256,
        kernel_size=1,
        stride=1,
        padding=0,
        height=56,
        width=56,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=128,
        out_channels=128,
        kernel_size=3,
        stride=1,
        padding=1,
        height=28,
        width=28,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=128,
        out_channels=128,
        kernel_size=3,
        stride=2,
        padding=1,
        height=56,
        width=56,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=128,
        out_channels=512,
        kernel_size=1,
        stride=1,
        padding=0,
        height=28,
        width=28,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=256,
        out_channels=64,
        kernel_size=1,
        stride=1,
        padding=0,
        height=56,
        width=56,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=256,
        out_channels=128,
        kernel_size=1,
        stride=1,
        padding=0,
        height=56,
        width=56,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=256,
        out_channels=256,
        kernel_size=3,
        stride=1,
        padding=1,
        height=14,
        width=14,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=256,
        out_channels=256,
        kernel_size=3,
        stride=2,
        padding=1,
        height=28,
        width=28,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=256,
        out_channels=512,
        kernel_size=1,
        stride=1,
        padding=0,
        height=28,
        width=28,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=256,
        out_channels=1024,
        kernel_size=1,
        stride=1,
        padding=0,
        height=14,
        width=14,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=512,
        out_channels=128,
        kernel_size=1,
        stride=1,
        padding=0,
        height=28,
        width=28,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=512,
        out_channels=256,
        kernel_size=1,
        stride=1,
        padding=0,
        height=28,
        width=28,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=512,
        out_channels=512,
        kernel_size=3,
        stride=1,
        padding=1,
        height=7,
        width=7,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=512,
        out_channels=512,
        kernel_size=3,
        stride=2,
        padding=1,
        height=14,
        width=14,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=512,
        out_channels=1024,
        kernel_size=1,
        stride=1,
        padding=0,
        height=14,
        width=14,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=512,
        out_channels=2048,
        kernel_size=1,
        stride=1,
        padding=0,
        height=7,
        width=7,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=1024,
        out_channels=256,
        kernel_size=1,
        stride=1,
        padding=0,
        height=14,
        width=14,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=1024,
        out_channels=512,
        kernel_size=1,
        stride=1,
        padding=0,
        height=14,
        width=14,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=1024,
        out_channels=2048,
        kernel_size=1,
        stride=1,
        padding=0,
        height=7,
        width=7,
    ),
    Conv2dConfig(
        batch_size=16,
        in_channels=2048,
        out_channels=512,
        kernel_size=1,
        stride=1,
        padding=0,
        height=7,
        width=7,
    ),
)


@dataclasses.dataclass
class RmsNormConfig:
  batch_size: int
  seq_len: int
  num_features: int


RMS_NORM_CONFIGS = (
    # Default config for smoke test.
    RmsNormConfig(
        batch_size=1,
        seq_len=128,
        num_features=128,
    ),
    # Configs for Llama3 70B for MLP layers
    RmsNormConfig(
        batch_size=1,
        seq_len=8192,
        num_features=8192,
    ),
    RmsNormConfig(
        batch_size=32,
        seq_len=8192,
        num_features=8192,
    ),
    # Configs for Qwen3 480B MLP layers
    RmsNormConfig(
        batch_size=1,
        seq_len=8192,
        num_features=6144,
    ),
    RmsNormConfig(
        batch_size=32,
        seq_len=8192,
        num_features=6144,
    ),
    # Configs for Gemma3 27B MLP layers
    RmsNormConfig(
        batch_size=1,
        seq_len=8192,
        num_features=4608,
    ),
    RmsNormConfig(
        batch_size=32,
        seq_len=8192,
        num_features=4608,
    ),
)


@dataclasses.dataclass(frozen=True)
class SdpaConfig:
  batch_size: int
  embed_dim: int
  q_seq_len: int
  q_num_heads: int
  kv_num_heads: int
  qk_head_dim: int
  v_head_dim: int
  is_causal: bool
  enable_gqa: bool
  dtype: torch.dtype
  backend: Any = torch.nn.attention.SDPBackend.MATH

  @classmethod
  def get_base_configs(cls):
    return [
        SdpaConfig(
            batch_size=1,
            embed_dim=4096,
            q_seq_len=128,
            q_num_heads=8,
            kv_num_heads=8,
            qk_head_dim=64,
            v_head_dim=64,
            is_causal=True,
            enable_gqa=True,
            dtype=torch.bfloat16,
        ),
        # Configs for Llama3 70B attention layers
        SdpaConfig(
            batch_size=1,
            embed_dim=4096,
            q_seq_len=2048,
            q_num_heads=64,
            kv_num_heads=64,
            qk_head_dim=128,
            v_head_dim=128,
            is_causal=True,
            enable_gqa=True,
            dtype=torch.bfloat16,
        ),
        SdpaConfig(
            batch_size=4,
            embed_dim=4096,
            q_seq_len=2048,
            q_num_heads=64,
            kv_num_heads=64,
            qk_head_dim=128,
            v_head_dim=128,
            is_causal=True,
            enable_gqa=True,
            dtype=torch.bfloat16,
        ),
        # Configs for Qwen3 480B attention layers
        SdpaConfig(
            batch_size=1,
            embed_dim=4096,
            q_seq_len=2048,
            q_num_heads=96,
            kv_num_heads=96,
            qk_head_dim=128,
            v_head_dim=128,
            is_causal=True,
            enable_gqa=True,
            dtype=torch.bfloat16,
        ),
        SdpaConfig(
            batch_size=4,
            embed_dim=4096,
            q_seq_len=2048,
            q_num_heads=96,
            kv_num_heads=96,
            qk_head_dim=128,
            v_head_dim=128,
            is_causal=True,
            enable_gqa=True,
            dtype=torch.bfloat16,
        ),
        # Configs for Gemma3 27B attention layers
        SdpaConfig(
            batch_size=1,
            embed_dim=4096,
            q_seq_len=2048,
            q_num_heads=32,
            kv_num_heads=32,
            qk_head_dim=128,
            v_head_dim=128,
            is_causal=True,
            enable_gqa=True,
            dtype=torch.bfloat16,
        ),
        SdpaConfig(
            batch_size=4,
            embed_dim=4096,
            q_seq_len=2048,
            q_num_heads=32,
            kv_num_heads=32,
            qk_head_dim=128,
            v_head_dim=128,
            is_causal=True,
            enable_gqa=True,
            dtype=torch.bfloat16,
        ),
        # AFM v7 configs
        SdpaConfig(
            batch_size=4,
            embed_dim=2048,
            q_seq_len=2048,
            q_num_heads=16,
            kv_num_heads=2,
            qk_head_dim=128,
            v_head_dim=128,
            is_causal=True,
            enable_gqa=True,
            dtype=torch.bfloat16,
        ),
    ]

  @classmethod
  def configs_with_backends(cls, *backends):
    for backend in backends:
      for config in cls.get_base_configs():
        yield dataclasses.replace(config, backend=backend)


@dataclasses.dataclass
class TopKConfig:
  batch_size: int
  seq_len: int
  num_features: int
  k: int
  dim: int


TOPK_CONFIGS = (
    TopKConfig(
        batch_size=1,
        seq_len=128,
        num_features=128,
        k=10,
        dim=-1,
    ),
    TopKConfig(
        batch_size=32,
        seq_len=8192,
        num_features=8192,
        k=100,
        dim=-1,
    ),
)


@dataclasses.dataclass
class NonZeroConfig:
  batch_size: int
  seq_len: int
  num_features: int


NONZERO_CONFIGS = (
    NonZeroConfig(
        batch_size=1,
        seq_len=128,
        num_features=128,
    ),
    NonZeroConfig(
        batch_size=32,
        seq_len=8192,
        num_features=8192,
    ),
)


@dataclasses.dataclass
class EmbeddingConfig:
  batch_size: int
  seq_len: int
  num_embeddings: int
  embedding_dim: int


EMBEDDING_CONFIGS = (
    # Default config for smoke test.
    EmbeddingConfig(
        batch_size=1,
        seq_len=128,
        num_embeddings=128,
        embedding_dim=128,
    ),
    # Configs for BERT
    EmbeddingConfig(
        batch_size=32,
        seq_len=128,
        num_embeddings=30522,
        embedding_dim=768,
    ),
    EmbeddingConfig(
        batch_size=1,
        seq_len=128,
        num_embeddings=512,
        embedding_dim=768,
    ),
    EmbeddingConfig(
        batch_size=32,
        seq_len=128,
        num_embeddings=2,
        embedding_dim=768,
    ),
)


@dataclasses.dataclass
class DropoutConfig:
  p: float
  shape: tuple[int, ...]


DROPOUT_CONFIGS = (
    # BERT configs
    DropoutConfig(
        p=0.1,
        shape=(32, 128, 768),
    ),
    # Conformer configs
    DropoutConfig(
        p=0.1,
        shape=(128, 32, 2048),
    ),
    DropoutConfig(
        p=0.1,
        shape=(128, 32, 512),
    ),
    DropoutConfig(
        p=0.1,
        shape=(32, 512, 128),
    ),
)


@dataclasses.dataclass
class TanhConfig:
  shape: tuple[int, ...]


TANH_CONFIGS = (
    # BERT configs
    TanhConfig(
        shape=(32, 768),
    ),
)


@dataclasses.dataclass
class BertLayerConfig:
  batch_size: int
  seq_len: int


BERT_LAYER_CONFIGS = (BertLayerConfig(batch_size=32, seq_len=128),)


@dataclasses.dataclass
class Qwen3Config:
  batch_size: int
  seq_len: int
  hidden_size: int = 128
  intermediate_size: int = 512
  head_dim: int = 128


QWEN3_CONFIGS = (Qwen3Config(batch_size=1, seq_len=128),)


@dataclasses.dataclass
class SiLUConfig:
  batch_size: int
  seq_len: int
  shape: tuple[int, ...] = (1, 128, 512)


SILU_CONFIGS = (
    SiLUConfig(batch_size=1, seq_len=128),
    # Conformer configs
    SiLUConfig(
        batch_size=32,
        seq_len=128,
        shape=(128, 32, 2048),
    ),
    SiLUConfig(
        batch_size=32,
        seq_len=128,
        shape=(32, 512, 128),
    ),
)


@dataclasses.dataclass
class GluConfig:
  batch_size: int
  seq_len: int
  shape: tuple[int, ...] = (1, 128, 512)
  dim: int = -1


GLU_CONFIGS = (
    GluConfig(batch_size=1, seq_len=128),
    # Conformer config
    GluConfig(
        batch_size=32,
        seq_len=128,
        shape=(32, 1024, 128),
        dim=1,
    ),
)


@dataclasses.dataclass
class DeepSeekConfig:
  batch_size: int
  seq_len: int
  vocab_size: int = 1024
  dim: int = 128
  inter_dim: int = 512
  moe_inter_dim: int = 64
  n_layers: int = 1
  n_dense_layers: int = 1
  n_heads: int = 4
  n_routed_experts: int = 4
  n_shared_experts: int = 2
  n_activated_experts: int = 2
  in_features: int = 128
  out_features: int = 128


DEEPSEEK_CONFIGS = (DeepSeekConfig(batch_size=1, seq_len=128),)


@dataclasses.dataclass
class Conv1dConfig:
  batch_size: int
  seq_len: int
  in_channels: int
  out_channels: int
  kernel_size: int
  stride: int = 1
  padding: int = 0
  dilation: int = 1
  groups: int = 1
  bias: bool = True


CONV1D_CONFIGS = (
    Conv1dConfig(
        batch_size=32,
        seq_len=128,
        in_channels=512,
        out_channels=1024,
        kernel_size=1,
    ),
    Conv1dConfig(
        batch_size=32,
        seq_len=128,
        in_channels=512,
        out_channels=512,
        kernel_size=31,
        padding=15,
        groups=512,
    ),
)


@dataclasses.dataclass
class MultiheadAttentionConfig:
  batch_size: int
  seq_len: int
  embed_dim: int
  num_heads: int
  dropout: float = 0.0
  bias: bool = True


MULTIHEAD_ATTENTION_CONFIGS = (
    MultiheadAttentionConfig(
        batch_size=32,
        seq_len=128,
        embed_dim=512,
        num_heads=8,
    ),
)


@dataclasses.dataclass
class FftConfig:
  batch_size: int
  seq_len: int
  hidden_size: int
  dim: int = -1
  norm: str = "backward"


FFT_CONFIGS = (
    # Default config for smoke test.
    FftConfig(batch_size=1, seq_len=128, hidden_size=128),
    # Standard intermediate ML shapes.
    FftConfig(batch_size=32, seq_len=128, hidden_size=768),
    FftConfig(batch_size=32, seq_len=1024, hidden_size=1024),
)


@dataclasses.dataclass
class Mamba2BlockConfig:
  batch_size: int
  seq_len: int
  hidden_size: int
  state_size: int = 128
  conv_kernel: int = 4
  expand: int = 2
  num_heads: int = 128
  head_dim: int = 64
  n_groups: int = 1
  chunk_size: int = 256


MAMBA2_BLOCK_CONFIGS = (
    Mamba2BlockConfig(
        batch_size=1,
        seq_len=128,
        hidden_size=256,
        num_heads=8,
    ),
    Mamba2BlockConfig(
        batch_size=8,
        seq_len=1024,
        hidden_size=2048,
        num_heads=64,
        n_groups=8,
        chunk_size=128,
    ),
)


@dataclasses.dataclass
class NemotronHMamba2BlockConfig:
  batch_size: int
  seq_len: int
  hidden_size: int
  state_size: int = 128
  conv_kernel: int = 4
  expand: int = 2
  num_heads: int = 128
  head_dim: int = 64
  n_groups: int = 8
  chunk_size: int = 128


NEMOTRON_H_MAMBA2_BLOCK_CONFIGS = (
    NemotronHMamba2BlockConfig(
        batch_size=1,
        seq_len=128,
        hidden_size=2688,
        num_heads=64,
        head_dim=64,
    ),
    NemotronHMamba2BlockConfig(
        batch_size=8,
        seq_len=1024,
        hidden_size=2688,
        num_heads=64,
        head_dim=64,
    ),
)
