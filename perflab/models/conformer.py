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

"""Conformer audio encoder benchmark definitions.

The `Conformer` module is adapted from `torchaudio.models.Conformer`
(https://github.com/pytorch/audio, BSD-2-Clause,
Copyright (c) 2017 Facebook Inc. (Soumith Chintala)); torchaudio itself is not
available as a dependency. The same model is built for every framework; the
legacy benchmark wrapped the normalization layers in TorchAX-only dtype casts,
which perflab factories must not do (see `context.Context`).
"""

import torch
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import torch_device_ops
from perflab import workload as workload_lib


class _ConformerConvModule(torch.nn.Module):
  """Depthwise separable convolution block for a Conformer layer."""

  def __init__(
      self,
      input_dim: int,
      num_channels: int,
      depthwise_kernel_size: int,
      dropout: float = 0.0,
      bias: bool = True,
      use_group_norm: bool = False,
  ) -> None:
    super().__init__()
    if (depthwise_kernel_size - 1) % 2 != 0:
      raise ValueError(
          "depthwise_kernel_size must be odd to preserve sequence length."
      )
    self.layer_norm = torch.nn.LayerNorm(input_dim)
    norm_layer = (
        torch.nn.GroupNorm(num_groups=1, num_channels=num_channels)
        if use_group_norm
        else torch.nn.BatchNorm1d(num_channels)
    )
    self.sequential = torch.nn.Sequential(
        torch.nn.Conv1d(
            input_dim,
            2 * num_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        ),
        torch.nn.GLU(dim=1),
        torch.nn.Conv1d(
            num_channels,
            num_channels,
            depthwise_kernel_size,
            stride=1,
            padding=(depthwise_kernel_size - 1) // 2,
            groups=num_channels,
            bias=bias,
        ),
        norm_layer,
        torch.nn.SiLU(),
        torch.nn.Conv1d(
            num_channels,
            input_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        ),
        torch.nn.Dropout(dropout),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.layer_norm(x)
    x = x.transpose(1, 2)
    x = self.sequential(x)
    return x.transpose(1, 2)


class _ConformerFeedForward(torch.nn.Module):
  """Macaron-style positionwise feed-forward network."""

  def __init__(
      self,
      input_dim: int,
      hidden_dim: int,
      dropout: float = 0.0,
  ) -> None:
    super().__init__()
    self.sequential = torch.nn.Sequential(
        torch.nn.LayerNorm(input_dim),
        torch.nn.Linear(input_dim, hidden_dim, bias=True),
        torch.nn.SiLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_dim, input_dim, bias=True),
        torch.nn.Dropout(dropout),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.sequential(x)


class _ConformerLayer(torch.nn.Module):
  """Single Conformer encoder block (FFN -> MHSA -> Conv -> FFN -> Norm)."""

  def __init__(
      self,
      input_dim: int,
      ffn_dim: int,
      num_attention_heads: int,
      depthwise_conv_kernel_size: int,
      dropout: float = 0.0,
      use_group_norm: bool = False,
      convolution_first: bool = False,
  ) -> None:
    super().__init__()
    self.ffn1 = _ConformerFeedForward(input_dim, ffn_dim, dropout=dropout)
    self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
    self.self_attn = torch.nn.MultiheadAttention(
        input_dim, num_attention_heads, dropout=dropout
    )
    self.self_attn_dropout = torch.nn.Dropout(dropout)
    self.conv_module = _ConformerConvModule(
        input_dim=input_dim,
        num_channels=input_dim,
        depthwise_kernel_size=depthwise_conv_kernel_size,
        dropout=dropout,
        bias=True,
        use_group_norm=use_group_norm,
    )
    self.ffn2 = _ConformerFeedForward(input_dim, ffn_dim, dropout=dropout)
    self.final_layer_norm = torch.nn.LayerNorm(input_dim)
    self.convolution_first = convolution_first

  def _apply_convolution(self, x: torch.Tensor) -> torch.Tensor:
    residual = x
    x = x.transpose(0, 1)
    x = self.conv_module(x)
    x = x.transpose(0, 1)
    return residual + x

  def forward(
      self, x: torch.Tensor, key_padding_mask: torch.Tensor | None
  ) -> torch.Tensor:
    residual = x
    x = self.ffn1(x) * 0.5 + residual

    if self.convolution_first:
      x = self._apply_convolution(x)

    residual = x
    x = self.self_attn_layer_norm(x)
    x, _ = self.self_attn(
        query=x,
        key=x,
        value=x,
        key_padding_mask=key_padding_mask,
        need_weights=False,
    )
    x = self.self_attn_dropout(x) + residual

    if not self.convolution_first:
      x = self._apply_convolution(x)

    residual = x
    x = self.ffn2(x) * 0.5 + residual
    return self.final_layer_norm(x)


class Conformer(torch.nn.Module):
  """Conformer speech recognition encoder."""

  @staticmethod
  def lengths_to_padding_mask(
      lengths: torch.Tensor,
      device: torch.device | str,
      max_length: int | None = None,
  ) -> torch.Tensor:
    """Computes static boolean key padding mask on CPU before moving to device.

    Args:
      lengths: 1D tensor of sequence lengths with shape `(batch_size,)`.
      device: Target device for the returned boolean mask.
      max_length: Static time dimension of the returned mask. If `None`, infers
        the maximum length from `lengths`.

    Returns:
      Boolean padding mask of shape `(batch_size, max_length)` on `device`,
      where `True` indicates padded positions (`index >= length`).
    """
    lengths_cpu = lengths.to("cpu")
    batch_size = lengths_cpu.shape[0]
    if max_length is None:
      max_length = int(torch.max(lengths_cpu).item())
    padding_mask = torch.arange(max_length, dtype=lengths_cpu.dtype).expand(
        batch_size, max_length
    ) >= lengths_cpu.unsqueeze(1)
    return padding_mask.to(device)

  def __init__(
      self,
      input_dim: int,
      num_heads: int,
      ffn_dim: int,
      num_layers: int,
      depthwise_conv_kernel_size: int,
      dropout: float = 0.0,
      use_group_norm: bool = False,
      convolution_first: bool = False,
  ) -> None:
    super().__init__()
    self.conformer_layers = torch.nn.ModuleList([
        _ConformerLayer(
            input_dim=input_dim,
            ffn_dim=ffn_dim,
            num_attention_heads=num_heads,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=use_group_norm,
            convolution_first=convolution_first,
        )
        for _ in range(num_layers)
    ])

  def forward(
      self, x: torch.Tensor, padding_mask: torch.Tensor
  ) -> torch.Tensor:
    x = x.transpose(0, 1)
    for layer in self.conformer_layers:
      x = layer(x, padding_mask)
    return x.transpose(0, 1)


# The workload of the Conformer benchmarks (built by `_load_conformer`): audio
# frames, not tokens.
_WORKLOAD = registry_lib.scoped_workload(
    full=workload_lib.Workload(32), presubmit=workload_lib.Workload(2)
)


def _load_conformer(
    ctx: context_lib.Context, is_training: bool
) -> tuple[torch.nn.Module, tuple[torch.Tensor, torch.Tensor]]:
  """Builds the Conformer and sample inputs of the legacy torchaudio benchmark.

  Full scope matches `torchaudio_conformer`: 17 layers, batch 32, up to 150
  frames of 512 features; presubmit shrinks it to 2 layers, batch 2, 32 frames.
  """
  batch_size = ctx.require_workload().batch_size
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    max_seq_len = 32
    num_layers = 2
  else:
    max_seq_len = 150
    num_layers = 17

  input_dim = 512
  dtype = torch_device_ops.get_torch_dtype(ctx.dtype)
  device_str = ctx.device_kind.value

  model = (
      Conformer(
          input_dim=input_dim,
          num_heads=8,
          ffn_dim=2048,
          num_layers=num_layers,
          depthwise_conv_kernel_size=31,
      )
      .to(dtype=dtype)
      .to(device=device_str)
  )

  if is_training:
    model.train()
  else:
    model.eval()

  lengths = torch.randint(1, max_seq_len + 1, (batch_size,))
  inputs = torch.rand(batch_size, max_seq_len, input_dim)
  padding_mask = Conformer.lengths_to_padding_mask(
      lengths, device_str, max_length=max_seq_len
  )
  example_inputs = (
      inputs.to(dtype=dtype, device=device_str),
      padding_mask,
  )
  return model, example_inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=_WORKLOAD,
)
def conformer_forward(ctx: context_lib.Context):
  """Benchmark factory for Conformer forward."""
  model, input_args = _load_conformer(ctx, is_training=False)
  return model, input_args, {}


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.TRAINING,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=_WORKLOAD,
)
def conformer_training(ctx: context_lib.Context):
  """Benchmark factory for Conformer training."""
  model, input_args = _load_conformer(ctx, is_training=True)
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, input_args, {}, opt
