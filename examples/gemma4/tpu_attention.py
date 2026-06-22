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

"""TPU optimized attention implementation for Gemma 4."""

import os
from typing import Any, Optional
import torch
from torch import nn

DEBUG = os.environ.get("TORCH_TPU_DEBUG", "False").lower() == "true"

from examples.gemma4 import splash_attention
from transformers import modeling_utils
from transformers.models.gemma4 import modeling_gemma4

print("[DEBUG_IMPORT] Importing tpu_attention.py")


def get_stats_str(t: torch.Tensor) -> str:
  t_cpu = t.cpu().float()
  return (
      f"dtype={t.dtype}, mean={t_cpu.mean().item():.6f},"
      f" std={t_cpu.std().item():.6f}, max={t_cpu.abs().max().item():.6f}"
  )


def torch_tpu_gemma4_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float | int = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
  """Attention forward implementation optimized for TPU."""
  # Vision gemma4 doesn't support sliding window (it has use_bidirectional_attention == 'vision')
  use_bidirectional_attn = getattr(
      module.config, "use_bidirectional_attention", None
  )
  skip_sliding_mask = use_bidirectional_attn == "vision"
  is_sliding = getattr(module, "is_sliding", False)
  use_swa_env = os.environ.get("TORCH_TPU_USE_SWA", "True").lower() == "true"
  use_swa = is_sliding and not skip_sliding_mask and use_swa_env

  if DEBUG:
    print(
        f"[DEBUG] Device: {query.device}, type: {query.device.type},"
        f" is_sliding: {is_sliding}, use_swa: {use_swa}"
    )

  if query.device.type in ("xla", "tpu", "jax"):

    if use_swa:
      assert (
          sliding_window is not None
      ), "sliding_window must be provided for SWA"
      if scaling is None:
        scaling = module.head_dim**-0.5

      q_len = query.shape[-2]
      kv_len = key.shape[-2]

      # SWA is equivalent to Causal if kv_len <= sliding_window
      if kv_len <= sliding_window:
        if DEBUG:
          print(
              f"[DEBUG_ATTN] Layer {module.layer_idx} (SWA->Causal fallback) |"
              f" kv_len={kv_len} <= window={sliding_window}"
          )
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=module.is_causal,
            dropout_p=dropout if module.training else 0.0,
            scale=scaling,
        )
      # SWA is active, but sequence is too small for default Splash block size (512)
      elif q_len < 512:
        if DEBUG:
          print(
              f"[DEBUG_ATTN] Layer {module.layer_idx} (SWA->SDPA mask fallback)"
              f" | q_len={q_len} < 512, kv_len={kv_len}"
          )
        # Generate SWA mask (True means keep)
        kv_offset = kv_len - q_len
        causal_mask = torch.triu(
            torch.ones(q_len, kv_len, device=query.device),
            diagonal=1 + kv_offset,
        ).bool()
        window_mask = torch.tril(
            torch.ones(q_len, kv_len, device=query.device),
            diagonal=-sliding_window + kv_offset,
        ).bool()
        mask = ~(causal_mask | window_mask)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            is_causal=False,
            dropout_p=dropout if module.training else 0.0,
            scale=scaling,
        )
      else:
        # SWA using splash attention (safe because q_len >= 512)
        if DEBUG:
          print(
              f"[DEBUG_ATTN] Layer {module.layer_idx} (SWA Splash) |"
              f" q_len={q_len} >= 512"
          )
          print(
              f"[DEBUG_ATTN] Layer {module.layer_idx} (SWA) | device:"
              f" {query.device} | Q: {get_stats_str(query)}"
          )
          print(
              f"[DEBUG_ATTN] Layer {module.layer_idx} (SWA) | device:"
              f" {query.device} | K: {get_stats_str(key)}"
          )
          print(
              f"[DEBUG_ATTN] Layer {module.layer_idx} (SWA) | device:"
              f" {query.device} | V: {get_stats_str(value)}"
          )

        num_key_value_groups = getattr(module, "num_key_value_groups", 1)
        attn_output = splash_attention.splash_sdpa(
            query,
            key,
            value,
            scale=scaling,
            local_window_size=sliding_window - 1,
            enable_gqa=(num_key_value_groups > 1),
        )

      attn_output = attn_output.transpose(1, 2).contiguous()
      if DEBUG:
        print(
            f"[DEBUG_ATTN] Layer {module.layer_idx} (SWA) | device:"
            f" {query.device} | OUT: {get_stats_str(attn_output)}"
        )
      return attn_output, None
    else:
      # Non-SWA on TPU using standard SDPA
      if scaling is None:
        scaling = module.head_dim**-0.5

      if DEBUG:
        print(
            f"[DEBUG_ATTN] Layer {module.layer_idx} (Full) | device:"
            f" {query.device} | Q: {get_stats_str(query)}"
        )
        print(
            f"[DEBUG_ATTN] Layer {module.layer_idx} (Full) | device:"
            f" {query.device} | K: {get_stats_str(key)}"
        )
        print(
            f"[DEBUG_ATTN] Layer {module.layer_idx} (Full) | device:"
            f" {query.device} | V: {get_stats_str(value)}"
        )

      if attention_mask is not None:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=False,
            dropout_p=dropout if module.training else 0.0,
            scale=scaling,
        )
      else:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=module.is_causal,
            dropout_p=dropout if module.training else 0.0,
            scale=scaling,
        )
      attn_output = attn_output.transpose(1, 2).contiguous()
      if DEBUG:
        print(
            f"[DEBUG_ATTN] Layer {module.layer_idx} (Full) | device:"
            f" {query.device} | OUT: {get_stats_str(attn_output)}"
        )
      return attn_output, None
  else:
    # Fallback to eager implementation on CPU/GPU
    return modeling_gemma4.eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        **kwargs,
    )


# Register globally
print(
    "[DEBUG_REG] Registering torch_tpu_gemma4 on"
    f" {id(modeling_utils.ALL_ATTENTION_FUNCTIONS)}"
)
print(f"[DEBUG_REG] modeling_utils file: {modeling_utils.__file__}")
print(f"[DEBUG_REG] modeling_utils name: {modeling_utils.__name__}")
modeling_utils.ALL_ATTENTION_FUNCTIONS.register(
    "torch_tpu_gemma4", torch_tpu_gemma4_attention_forward
)
print(
    "[DEBUG_REG] Registered keys now:"
    f" {list(modeling_utils.ALL_ATTENTION_FUNCTIONS.valid_keys())}"
)
