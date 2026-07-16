# Copyright 2025 Google LLC
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

"""Workarounds for llama_models.llama3.model."""

import functools
from typing import Tuple
from absl import logging
import llama_models.llama3.model as m
import torch
from torch_tpu._internal.utils import log_utils

log_utils.log_to_stderr()


# This signature matches the signature of the apply_rotary_emb function in
# llama_models.llama3.model.
def apply_rotary_emb_patch(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Apply rotary embeddings without the use of complex numbers.

  This function is intended to be monkey patched into
  llama_models.llama3.model.apply_rotary_emb.

  Args:
    xq: Query vector.
    xk: Key vector.
    freqs_cis: Frequencies for rotary embeddings.

  Returns:
    Rotated query and key vectors.
  """

  def rotate_half(x: torch.Tensor, freqs_cis: torch.Tensor):
    # Reshape for pairing dimensions.
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x_r, x_i = x_reshaped.unbind(dim=-1)

    # Get freqs as real and imag, and reshape for broadcast.
    freqs_cos = m.reshape_for_broadcast(freqs_cis.real, x_r)
    freqs_sin = m.reshape_for_broadcast(freqs_cis.imag, x_r)

    # Apply rotation
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    # Recombine and flatten
    x_out = torch.stack([x_out_r, x_out_i], dim=-1).flatten(3)

    return x_out.type_as(x)

  xq_out = rotate_half(xq, freqs_cis)
  xk_out = rotate_half(xk, freqs_cis)

  return xq_out, xk_out


def replace_rope_impl() -> None:
  """Patch llama_models.llama3.model module to avoid use of c64.

  This must be called after module import but before a forward pass or compile.

  It can be called before or after model instantiation.
  """
  m.apply_rotary_emb = apply_rotary_emb_patch


def undo_inference_mode(model: m.Transformer) -> None:
  """Bypass @torch.inference_mode decorator - needed for trainloop."""
  model.forward = functools.partial(model.forward.__wrapped__, model)  # pyrefly: ignore[missing-attribute]


def reset_kv_cache(model: m.Transformer) -> None:
  """Reset persistent KV cache - needed for trainloop."""
  for module in model.layers:
    module.attention.cache_k.detach_()
    module.attention.cache_v.detach_()


def init_model_weights(model):
  """Initializes model weights to small random values."""

  if hasattr(model, "weight") and model.weight is not None:
    tensor = model.weight.data
    logging.info(
        "Initializing model weight: %s %s",
        tensor.shape,
        tensor.dtype,
    )
    tensor.normal_(std=0.01)

  if hasattr(model, "bias") and model.bias is not None:
    tensor = model.bias.data
    logging.info(
        "Initializing model bias: %s %s",
        tensor.shape,
        tensor.dtype,
    )
    tensor.fill_(0.0)
