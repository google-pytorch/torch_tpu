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
from typing import Any, Optional
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
  kernel_size: int | tuple[int, int]
  stride: int | tuple[int, int] = 1
  padding: int | tuple[int, int] = 0
  height: int = 128
  width: int = 128
  dilation: int | tuple[int, int] = 1
  groups: int = 1
  bias: bool = True
  channels_last: bool = False


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
    # DeepLOB / Limit Order Book (LOB) configs (Time x Depth/Price levels):
    # 1. Feature projection across depth levels (height=100, width=40 features).
    Conv2dConfig(
        batch_size=32,
        in_channels=1,
        out_channels=32,
        kernel_size=1,
        stride=1,
        padding=0,
        height=100,
        width=40,
    ),
    # 2. Spatial aggregation across order book depth (bid/ask, W=40 -> 20).
    Conv2dConfig(
        batch_size=32,
        in_channels=32,
        out_channels=32,
        kernel_size=(1, 2),
        stride=(1, 2),
        padding=0,
        height=100,
        width=40,
    ),
    # 3. Temporal aggregation across event updates (multi-step price momentum).
    Conv2dConfig(
        batch_size=32,
        in_channels=32,
        out_channels=32,
        kernel_size=(3, 1),
        stride=(1, 1),
        padding=(1, 0),
        height=100,
        width=20,
    ),
    # 4. Low-latency streaming LOB inference (B=1, sub-tile execution).
    Conv2dConfig(
        batch_size=1,
        in_channels=1,
        out_channels=32,
        kernel_size=(1, 2),
        stride=(1, 2),
        padding=0,
        height=100,
        width=40,
    ),
    # 5. TPU 128-tile-aligned DeepLOB configuration (H=128, W=40 -> 5120
    # elements, exactly 40 full 128-element MXU systolic tiles).
    Conv2dConfig(
        batch_size=32,
        in_channels=32,
        out_channels=64,
        kernel_size=(3, 1),
        stride=1,
        padding=(1, 0),
        height=128,
        width=40,
    ),
    # 6. Deep layer MXU saturation config (C_in=64, C_out=128, full MXU lanes).
    Conv2dConfig(
        batch_size=32,
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        stride=1,
        padding=1,
        height=128,
        width=40,
    ),
    # 7. Native TPU channels-last (NHWC) format to evaluate layout speedup.
    Conv2dConfig(
        batch_size=32,
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        stride=1,
        padding=1,
        height=128,
        width=40,
        channels_last=True,
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
  dim: int | None = None
  largest: bool = True
  sorted: bool = True


TOPK_CONFIGS = (
    # Default config for smoke test.
    TopKConfig(
        batch_size=1,
        seq_len=128,
        num_features=128,
        k=10,
    ),
    TopKConfig(
        batch_size=32,
        seq_len=8192,
        num_features=8192,
        k=100,
        dim=-1,
    ),
    TopKConfig(
        batch_size=32,
        seq_len=1024,
        num_features=4096,
        k=10,
    ),
    TopKConfig(
        batch_size=32,
        seq_len=1024,
        num_features=4096,
        k=100,
    ),
    TopKConfig(
        batch_size=32,
        seq_len=1024,
        num_features=4096,
        k=5,
        dim=1,
    ),
    TopKConfig(
        batch_size=32,
        seq_len=1024,
        num_features=4096,
        k=5,
        largest=False,
    ),
    TopKConfig(
        batch_size=32,
        seq_len=1024,
        num_features=4096,
        k=5,
        sorted=False,
    ),
    TopKConfig(
        batch_size=32,
        seq_len=1024,
        num_features=4096,
        k=5,
        largest=False,
        sorted=False,
    ),
)


@dataclasses.dataclass
class SortConfig:
  """Configuration parameters for sort benchmarks."""

  batch_size: int
  seq_len: int
  num_features: int
  dim: int | None = None
  descending: bool = False
  stable: bool = False


SORT_CONFIGS = (
    # Default config for smoke test.
    SortConfig(
        batch_size=1,
        seq_len=128,
        num_features=128,
    ),
    SortConfig(
        batch_size=32,
        seq_len=2048,
        num_features=2048,
        dim=-1,
    ),
    SortConfig(
        batch_size=32,
        seq_len=1024,
        num_features=4096,
    ),
    SortConfig(
        batch_size=32,
        seq_len=1024,
        num_features=4096,
        dim=1,
    ),
    SortConfig(
        batch_size=32,
        seq_len=1024,
        num_features=4096,
        descending=True,
    ),
    SortConfig(
        batch_size=32,
        seq_len=1024,
        num_features=4096,
        stable=True,
    ),
    SortConfig(
        batch_size=32,
        seq_len=1024,
        num_features=4096,
        descending=True,
        stable=True,
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
    # GPT2 configs
    DropoutConfig(
        p=0.1,
        shape=(1, 512, 768),
    ),
    DropoutConfig(
        p=0.1,
        shape=(1, 12, 512, 512),
    ),
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
    # Financial services / time-series configs (TCN & market data):
    # 1. HFT low-latency streaming inference (B=1, small channel count).
    Conv1dConfig(
        batch_size=1,
        seq_len=256,
        in_channels=32,
        out_channels=64,
        kernel_size=5,
    ),
    # 2. Dilated TCN layer for multi-horizon return / volatility forecasting.
    Conv1dConfig(
        batch_size=64,
        seq_len=1024,
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        dilation=8,
    ),
    # 3. Long-sequence high-dilation TCN layer (exponential receptive field).
    Conv1dConfig(
        batch_size=32,
        seq_len=2048,
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        dilation=32,
    ),
    # 4. Multi-asset depthwise 1D conv (128 assets, 128-lane VPU vectorization).
    Conv1dConfig(
        batch_size=32,
        seq_len=1024,
        in_channels=128,
        out_channels=128,
        kernel_size=5,
        padding=2,
        groups=128,
    ),
    # 5. MXU peak-compute financial backbone layer (C=256, saturating MXU).
    Conv1dConfig(
        batch_size=32,
        seq_len=1024,
        in_channels=256,
        out_channels=256,
        kernel_size=3,
        padding=1,
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


@dataclasses.dataclass
class SliceScatterConfig:
  input_shape: tuple[int, ...]
  src_shape: tuple[int, ...]
  dim: int
  start: int | None
  end: int | None
  step: int


SLICE_SCATTER_CONFIGS = (
    SliceScatterConfig(
        input_shape=(1, 128, 128),
        src_shape=(1, 10, 128),
        dim=1,
        start=10,
        end=20,
        step=1,
    ),
    SliceScatterConfig(
        input_shape=(32, 1024, 1024),
        src_shape=(32, 100, 1024),
        dim=1,
        start=100,
        end=200,
        step=1,
    ),
    SliceScatterConfig(
        input_shape=(1, 8, 2048, 128),
        src_shape=(1, 8, 1, 128),
        dim=2,
        start=1024,
        end=1025,
        step=1,
    ),
    SliceScatterConfig(
        input_shape=(32, 8, 2048, 128),
        src_shape=(32, 8, 1, 128),
        dim=2,
        start=1024,
        end=1025,
        step=1,
    ),
    SliceScatterConfig(
        input_shape=(1, 8, 8192, 128),
        src_shape=(1, 8, 512, 128),
        dim=2,
        start=4096,
        end=4608,
        step=1,
    ),
    SliceScatterConfig(
        input_shape=(32, 1024, 1024),
        src_shape=(32, 50, 1024),
        dim=1,
        start=100,
        end=200,
        step=2,
    ),
    SliceScatterConfig(
        input_shape=(1, 8, 8192, 128),
        src_shape=(1, 8, 256, 128),
        dim=2,
        start=4096,
        end=4608,
        step=2,
    ),
    SliceScatterConfig(
        input_shape=(32, 8, 1024, 128),
        src_shape=(32, 8, 1024, 64),
        dim=3,
        start=0,
        end=64,
        step=1,
    ),
    SliceScatterConfig(
        input_shape=(32, 8, 1024, 128),
        src_shape=(16, 8, 1024, 128),
        dim=0,
        start=0,
        end=16,
        step=1,
    ),
    SliceScatterConfig(
        input_shape=(4096, 4096),
        src_shape=(1024, 4096),
        dim=0,
        start=1024,
        end=2048,
        step=1,
    ),
    SliceScatterConfig(
        input_shape=(4096, 4096),
        src_shape=(256, 4096),
        dim=0,
        start=1024,
        end=2048,
        step=4,
    ),
)


@dataclasses.dataclass(frozen=True)
class PreluConfig:
  batch_size: Any  # Supports int or DynamicDimension
  channels: int
  height: int
  width: int
  channel_wise: bool
  dtype: torch.dtype = torch.bfloat16

  @classmethod
  def get_base_configs(cls):
    return [
        # 1. Channel-wise PReLU (typical CNN)
        PreluConfig(
            batch_size=1,
            channels=64,
            height=112,
            width=112,
            channel_wise=True,
        ),
        PreluConfig(
            batch_size=4,
            channels=128,
            height=56,
            width=56,
            channel_wise=True,
        ),
        # 2. Shared PReLU (scalar weight)
        PreluConfig(
            batch_size=1,
            channels=64,
            height=112,
            width=112,
            channel_wise=False,
        ),
    ]


PRELU_CONFIGS = PreluConfig.get_base_configs()


@dataclasses.dataclass(frozen=True)
class MaskedSoftmaxConfig:
  batch_size: Any  # Supports int or DynamicDimension
  num_heads: int
  q_seq_len: int
  kv_seq_len: int
  dim: int
  mask_type: int
  dtype: torch.dtype = torch.bfloat16

  @classmethod
  def get_base_configs(cls):
    return [
        # 1. Attention Mask (mask_type=0) - shape [L, S]
        MaskedSoftmaxConfig(
            batch_size=1,
            num_heads=8,
            q_seq_len=128,
            kv_seq_len=128,
            dim=-1,
            mask_type=0,
        ),
        # 2. Padding Mask (mask_type=1) - shape [B, S]
        MaskedSoftmaxConfig(
            batch_size=1,
            num_heads=8,
            q_seq_len=128,
            kv_seq_len=128,
            dim=-1,
            mask_type=1,
        ),
        # 3. Generic Mask (mask_type=2) - shape [B, H, L, S]
        MaskedSoftmaxConfig(
            batch_size=1,
            num_heads=8,
            q_seq_len=128,
            kv_seq_len=128,
            dim=-1,
            mask_type=2,
        ),
        # 4. Large Llama 3 8B Scale Attention Logit Dimensions
        MaskedSoftmaxConfig(
            batch_size=4,
            num_heads=32,
            q_seq_len=2048,
            kv_seq_len=2048,
            dim=-1,
            mask_type=0,
        ),
        MaskedSoftmaxConfig(
            batch_size=4,
            num_heads=32,
            q_seq_len=2048,
            kv_seq_len=2048,
            dim=-1,
            mask_type=1,
        ),
        MaskedSoftmaxConfig(
            batch_size=4,
            num_heads=32,
            q_seq_len=2048,
            kv_seq_len=2048,
            dim=-1,
            mask_type=2,
        ),
    ]


MASKED_SOFTMAX_CONFIGS = MaskedSoftmaxConfig.get_base_configs()


@dataclasses.dataclass(frozen=True)
class NativeMultiHeadAttentionConfig:
  batch_size: Any  # Supports int or DynamicDimension
  seq_len: int
  embed_dim: int
  num_heads: int
  need_weights: bool = True
  average_attn_weights: bool = False
  mask_type: Optional[int] = None
  dtype: torch.dtype = torch.bfloat16

  @classmethod
  def get_base_configs(cls):
    return [
        # 1. Standard MHA without mask
        NativeMultiHeadAttentionConfig(
            batch_size=1,
            seq_len=128,
            embed_dim=512,
            num_heads=8,
            mask_type=None,
        ),
        # 2. Attention Mask (mask_type=0) - shape [L, S]
        NativeMultiHeadAttentionConfig(
            batch_size=1,
            seq_len=128,
            embed_dim=512,
            num_heads=8,
            mask_type=0,
        ),
        # 3. Padding Mask (mask_type=1) - shape [B, S]
        NativeMultiHeadAttentionConfig(
            batch_size=1,
            seq_len=128,
            embed_dim=512,
            num_heads=8,
            mask_type=1,
        ),
        # 4. Generic Mask (mask_type=2) - shape [B, H, L, S]
        NativeMultiHeadAttentionConfig(
            batch_size=1,
            seq_len=128,
            embed_dim=512,
            num_heads=8,
            mask_type=2,
        ),
        # 5. need_weights=False
        NativeMultiHeadAttentionConfig(
            batch_size=1,
            seq_len=128,
            embed_dim=512,
            num_heads=8,
            need_weights=False,
        ),
        # 6. average_attn_weights=True
        NativeMultiHeadAttentionConfig(
            batch_size=1,
            seq_len=128,
            embed_dim=512,
            num_heads=8,
            need_weights=True,
            average_attn_weights=True,
        ),
        # 7. Larger Scale Attention Dimensions
        NativeMultiHeadAttentionConfig(
            batch_size=4,
            seq_len=1024,
            embed_dim=1024,
            num_heads=16,
            mask_type=None,
        ),
        NativeMultiHeadAttentionConfig(
            batch_size=4,
            seq_len=1024,
            embed_dim=1024,
            num_heads=16,
            mask_type=0,
        ),
        NativeMultiHeadAttentionConfig(
            batch_size=4,
            seq_len=1024,
            embed_dim=1024,
            num_heads=16,
            mask_type=1,
        ),
        NativeMultiHeadAttentionConfig(
            batch_size=4,
            seq_len=1024,
            embed_dim=1024,
            num_heads=16,
            mask_type=2,
        ),
    ]


NATIVE_MULTI_HEAD_ATTENTION_CONFIGS = (
    NativeMultiHeadAttentionConfig.get_base_configs()
)


@dataclasses.dataclass(frozen=True)
class TransformerEncoderLayerFwdConfig:
  batch_size: Any  # Supports int or DynamicDimension
  seq_len: int
  embed_dim: int
  num_heads: int
  dim_feedforward: int = 2048
  use_gelu: bool = False
  norm_first: bool = False
  mask_type: Optional[int] = None
  dtype: torch.dtype = torch.bfloat16

  @classmethod
  def get_base_configs(cls):
    return [
        # 1. Standard Post-LN ReLU
        TransformerEncoderLayerFwdConfig(
            batch_size=2,
            seq_len=128,
            embed_dim=512,
            num_heads=8,
            dim_feedforward=2048,
            use_gelu=False,
            norm_first=False,
            mask_type=None,
        ),
        # 2. Pre-LN GELU (e.g. modern Transformer architectures)
        TransformerEncoderLayerFwdConfig(
            batch_size=2,
            seq_len=128,
            embed_dim=512,
            num_heads=8,
            dim_feedforward=2048,
            use_gelu=True,
            norm_first=True,
            mask_type=None,
        ),
        # 3. Attention Mask (mask_type=0) - shape [T, T]
        TransformerEncoderLayerFwdConfig(
            batch_size=2,
            seq_len=128,
            embed_dim=512,
            num_heads=8,
            dim_feedforward=2048,
            use_gelu=False,
            norm_first=False,
            mask_type=0,
        ),
        # 4. Key-Padding Mask (mask_type=1) - shape [B, T]
        TransformerEncoderLayerFwdConfig(
            batch_size=2,
            seq_len=128,
            embed_dim=512,
            num_heads=8,
            dim_feedforward=2048,
            use_gelu=False,
            norm_first=False,
            mask_type=1,
        ),
        # 5. Large-Scale Transformer Layer (B=4, T=512, D=1024, H=16, d_ff=4096)
        TransformerEncoderLayerFwdConfig(
            batch_size=4,
            seq_len=512,
            embed_dim=1024,
            num_heads=16,
            dim_feedforward=4096,
            use_gelu=True,
            norm_first=True,
            mask_type=None,
        ),
    ]


TRANSFORMER_ENCODER_LAYER_FWD_CONFIGS = (
    TransformerEncoderLayerFwdConfig.get_base_configs()
)


@dataclasses.dataclass
class LstmConfig:
  batch_size: int
  seq_len: int
  input_size: int
  hidden_size: int
  num_layers: int = 1
  bias: bool = True
  batch_first: bool = True
  bidirectional: bool = False


LSTM_CONFIGS = (
    # Small / smoke test config
    LstmConfig(
        batch_size=32,
        seq_len=32,
        input_size=256,
        hidden_size=256,
        num_layers=1,
    ),
    # Standard recurrent workloads
    LstmConfig(
        batch_size=32,
        seq_len=64,
        input_size=512,
        hidden_size=512,
        num_layers=1,
    ),
    LstmConfig(
        batch_size=32,
        seq_len=128,
        input_size=512,
        hidden_size=512,
        num_layers=1,
    ),
    LstmConfig(
        batch_size=32,
        seq_len=256,
        input_size=512,
        hidden_size=512,
        num_layers=1,
    ),
    LstmConfig(
        batch_size=16,
        seq_len=512,
        input_size=512,
        hidden_size=512,
        num_layers=1,
    ),
    # Multi-layer stacked LSTM
    LstmConfig(
        batch_size=32,
        seq_len=64,
        input_size=512,
        hidden_size=512,
        num_layers=2,
    ),
    # Bidirectional LSTM workloads
    LstmConfig(
        batch_size=32,
        seq_len=64,
        input_size=512,
        hidden_size=512,
        num_layers=1,
        bidirectional=True,
    ),
    LstmConfig(
        batch_size=32,
        seq_len=128,
        input_size=512,
        hidden_size=512,
        num_layers=1,
        bidirectional=True,
    ),
    LstmConfig(
        batch_size=32,
        seq_len=256,
        input_size=512,
        hidden_size=512,
        num_layers=1,
        bidirectional=True,
    ),
)


@dataclasses.dataclass
class GruConfig:
  batch_size: int
  seq_len: int
  input_size: int
  hidden_size: int
  num_layers: int = 1
  bias: bool = True
  batch_first: bool = True
  bidirectional: bool = False


GRU_CONFIGS = (
    # Small / smoke test config
    GruConfig(
        batch_size=32,
        seq_len=32,
        input_size=256,
        hidden_size=256,
        num_layers=1,
    ),
    # Standard recurrent workloads
    GruConfig(
        batch_size=32,
        seq_len=64,
        input_size=512,
        hidden_size=512,
        num_layers=1,
    ),
    GruConfig(
        batch_size=32,
        seq_len=128,
        input_size=512,
        hidden_size=512,
        num_layers=1,
    ),
    GruConfig(
        batch_size=32,
        seq_len=256,
        input_size=512,
        hidden_size=512,
        num_layers=1,
    ),
    GruConfig(
        batch_size=16,
        seq_len=512,
        input_size=512,
        hidden_size=512,
        num_layers=1,
    ),
    # Multi-layer stacked GRU
    GruConfig(
        batch_size=32,
        seq_len=64,
        input_size=512,
        hidden_size=512,
        num_layers=2,
    ),
    # Bidirectional GRU workloads
    GruConfig(
        batch_size=32,
        seq_len=64,
        input_size=512,
        hidden_size=512,
        num_layers=1,
        bidirectional=True,
    ),
    GruConfig(
        batch_size=32,
        seq_len=128,
        input_size=512,
        hidden_size=512,
        num_layers=1,
        bidirectional=True,
    ),
    GruConfig(
        batch_size=32,
        seq_len=256,
        input_size=512,
        hidden_size=512,
        num_layers=1,
        bidirectional=True,
    ),
)
