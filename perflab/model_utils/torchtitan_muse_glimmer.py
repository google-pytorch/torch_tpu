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

"""TorchTitan Muse Glimmer (text decoder) for perflab benchmarks.

Muse Glimmer is a packed `[T]` decoder with GQA, QK-norm, a sigmoid output gate
and iRoPE: in each cycle of 4 layers, 3 use RoPE with a 2048-token sliding
window and the last is a global layer without RoPE. torchtitan's default
`flex` attention backend does not run on TPU, so this uses `sdpa`, which
ignores the sliding window. That is exact as long as `seq_len` is at most the
smallest window, which `text_config` enforces.

The RoPE cache is complex-valued, so `build_model` casts only floating-point
tensors to the benchmark dtype (`Module.to(dtype)` would drop its imaginary
part).
"""

from typing import Any

import torch
from torch import nn
from perflab.model_utils import torchtitan_batching
from torchtitan.models import muse_glimmer

use_batched_sdpa = torchtitan_batching.use_batched_sdpa
make_inputs = torchtitan_batching.make_inputs


def text_config(
    flavor: str, num_layers: int, seq_len: int
) -> muse_glimmer.MuseGlimmerModel.Config:
  """Returns a text-only torchtitan Muse Glimmer config using SDPA attention.

  Args:
    flavor: Key into `muse_glimmer.muse_glimmer_configs` (e.g. "30B").
    num_layers: Number of decoder layers to keep, counted from the last layer
      (Muse Glimmer counts its iRoPE / window cycle from the end, so the last 4
      layers are one full 3 sliding-window + 1 global cycle). Values <= 0 keep
      all layers.
    seq_len: Max context length used to size the RoPE cache.

  Returns:
    The model config.

  Raises:
    ValueError: If `seq_len` exceeds a sliding window (SDPA would ignore it).
  """
  config = muse_glimmer.model_registry(
      flavor, seq_len=seq_len, attn_backend="sdpa"
  ).model
  if not isinstance(config, muse_glimmer.MuseGlimmerModel.Config):
    raise TypeError(f"Unexpected config type {type(config).__name__}")

  if config.vision_encoder is not None:
    raise ValueError("vision_encoder must be None for text-only model")

  if num_layers > 0:
    config.layers = config.layers[-num_layers:]
  window_sizes = [
      layer.attention.window_size
      for layer in config.layers
      if getattr(layer.attention, "window_size", None) is not None
  ]
  if window_sizes and seq_len > min(window_sizes):
    raise ValueError(
        f"seq_len={seq_len} exceeds the sliding window {min(window_sizes)},"
        " which the sdpa attention backend ignores."
    )
  return config


def build_model(
    config: muse_glimmer.MuseGlimmerModel.Config,
    device: str,
    dtype: torch.dtype,
    *,
    seq_len: int | None = None,
) -> nn.Module:
  """Builds and initializes a torchtitan Muse Glimmer model directly on `device`.

  The model is built on meta under `dtype` as the default dtype and materialized
  directly on the device in `dtype`. Only floating-point parameters/buffers are
  cast to `dtype` (rather than `model.to(dtype)`) to avoid losing the complex
  RoPE cache, and SDPA submodules are swapped to support packed `(B * S,)`
  batches.

  Args:
    config: Model config.
    device: Device string (e.g. "tpu", "cuda" or "cpu").
    dtype: Parameter dtype.
    seq_len: Per-sequence token length `S` when inputs are packed `(B * S,)`.
      Defaults to the RoPE `max_context_length` on the first RoPE layer if
      present.

  Returns:
    The initialized model, in train mode.
  """
  if seq_len is None:
    seq_len = torchtitan_batching.infer_seq_len(config)
  prev_dtype = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    with torch.device("meta"):
      model = config.build()
    model.to_empty(device=device)
    with torch.no_grad():
      model.init_states(buffer_device=torch.device(device))
  finally:
    torch.set_default_dtype(prev_dtype)

  for p in model.parameters():
    if p.is_floating_point():
      p.data = p.data.to(dtype)
  for b in model.buffers():
    if b.is_floating_point():
      b.data = b.data.to(dtype)

  torchtitan_batching.use_batched_sdpa(model, seq_len=seq_len)
  return model


def cross_entropy_loss(
    model: nn.Module,
    input_args: tuple[Any, ...],
    input_kwargs: dict[str, Any],
) -> torch.Tensor:
  """Next-token cross entropy over fp32 logits (torchtitan returns logits)."""
  input_kwargs = dict(input_kwargs)
  labels = input_kwargs.pop("labels")
  logits = model(*input_args, **input_kwargs)
  return nn.functional.cross_entropy(logits.float(), labels)
