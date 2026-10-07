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

"""TorchTitan Qwen3.5 / Qwen3.8 model with a pure-torch GatedDeltaNet kernel.

Qwen3.8 uses the Qwen3.5 architecture (torchtitan's `qwen3_8` package only adds
a registry on top of `qwen3_5`), so this builds torchtitan's `qwen3_5` model.

torchtitan's GatedDeltaNet (`qwen3_5/gdn.py`) imports the CUDA/Triton-only
`attn_gym` package at module load, which is not available in google3. This
module installs a stub `attn_gym` before importing torchtitan, and
`use_pure_torch_gated_delta` replaces the `GatedDeltaKernel.forward` of a built
model (the only `attn_gym` caller on the non-varlen path) with
`chunked_gated_delta`, a pure-torch port of HF transformers'
`torch_chunk_gated_delta_rule`. The stub raises if anything else in `attn_gym`
is called (e.g. the varlen / batch-invariant kernels).
"""

import importlib.util
import sys
import types
from typing import Any

import torch


def _install_attn_gym_stub() -> None:
  """Registers a stub `attn_gym` package if the real one is unavailable.

  Workaround until torchtitan's `qwen3_5/gdn.py` imports `attn_gym` lazily (only
  when its kernels are called); then this stub can be deleted.
  """
  if "attn_gym" in sys.modules or importlib.util.find_spec("attn_gym"):
    return

  def _unavailable(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise NotImplementedError(
        "attn_gym is not available in google3; only the pure-torch"
        " GatedDeltaNet path (use_pure_torch_gated_delta) is supported."
    )

  def _getattr(attr: str) -> Any:
    if attr.startswith("__"):
      raise AttributeError(attr)
    return _unavailable

  for name in ("attn_gym", "attn_gym.linear"):
    module = types.ModuleType(name)
    module.__path__ = []  # Makes it a package so submodules resolve.
    module.__getattr__ = _getattr  # pytype: disable=not-writable
    sys.modules[name] = module


_install_attn_gym_stub()

# pylint: disable=g-import-not-at-top,g-bad-import-order
from torchtitan.models import qwen3_5
from torchtitan.models.qwen3_5 import gdn as qwen3_5_gdn

# pylint: enable=g-import-not-at-top,g-bad-import-order


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def chunked_gated_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
  """Chunked gated delta rule in pure torch (FLA `chunk_gated_delta_rule` math).

  Ported from HF transformers' `torch_chunk_gated_delta_rule`. Work within a
  chunk is done with matmuls; only the inter-chunk state update is sequential,
  so the traced graph grows with seq_len / chunk_size instead of seq_len.
  Applies the q / k L2 normalization and the 1 / sqrt(key_head_dim) query scale
  itself, like torchtitan's `GatedDeltaKernel`.

  HF's function is not called directly: it is wrapped in
  `use_kernel_func_from_hub_with_fallback`, so it silently switches to the FLA
  Triton kernel whenever `fla` is importable, and it writes its outputs in
  place, which is unfriendly to tracing. The test checks this port against it.

  Args:
    q: (bs, seqlen, n_heads, key_head_dim) un-normalized queries.
    k: (bs, seqlen, n_heads, key_head_dim) un-normalized keys.
    v: (bs, seqlen, n_heads, value_head_dim).
    g: (bs, seqlen, n_heads) log-space decay (negative).
    beta: (bs, seqlen, n_heads) update gate in (0, 1).
    chunk_size: Tokens per chunk.

  Returns:
    (bs, seqlen, n_heads, value_head_dim) output in q's dtype.
  """
  out_dtype = q.dtype
  d_k = q.shape[-1]
  q = _l2norm(q.float()) * (d_k**-0.5)
  k = _l2norm(k.float())
  # (bs, heads, seqlen, dim) / (bs, heads, seqlen).
  q, k, v = [x.transpose(1, 2).float() for x in (q, k, v)]
  g, beta = [x.transpose(1, 2).float() for x in (g, beta)]

  bs, n_heads, seq_len, _ = k.shape
  d_v = v.shape[-1]
  pad = (chunk_size - seq_len % chunk_size) % chunk_size
  if pad:
    q, k, v = [torch.nn.functional.pad(x, (0, 0, 0, pad)) for x in (q, k, v)]
    g, beta = [torch.nn.functional.pad(x, (0, pad)) for x in (g, beta)]
  num_chunks = (seq_len + pad) // chunk_size

  v_beta = v * beta.unsqueeze(-1)
  k_beta = k * beta.unsqueeze(-1)
  q, k, v_beta, k_beta = [
      x.reshape(bs, n_heads, num_chunks, chunk_size, x.shape[-1])
      for x in (q, k, v_beta, k_beta)
  ]
  g = g.reshape(bs, n_heads, num_chunks, chunk_size).cumsum(dim=-1)

  incl_upper = torch.triu(
      torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device)
  )
  decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
  # Solve (I + strictly_lower(K_beta K^T * decay))^-1 by forward substitution.
  attn = -((k_beta @ k.transpose(-1, -2)) * decay_mask).masked_fill(
      incl_upper, 0
  )
  for i in range(1, chunk_size):
    row = attn[..., i, :i].clone()
    sub = attn[..., :i, :i].clone()
    attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
  attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
  u = attn @ v_beta
  k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

  state = torch.zeros(bs, n_heads, d_k, d_v, dtype=u.dtype, device=u.device)
  strict_upper = torch.triu(
      torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
      diagonal=1,
  )
  outs = []
  for i in range(num_chunks):
    q_i, k_i, g_i = q[:, :, i], k[:, :, i], g[:, :, i]
    intra = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill(
        strict_upper, 0
    )
    v_new = u[:, :, i] - k_cumdecay[:, :, i] @ state
    inter = (q_i * g_i.exp().unsqueeze(-1)) @ state
    outs.append(inter + intra @ v_new)
    g_last = g_i[..., -1]
    state = (
        state * g_last[..., None, None].exp()
        + (k_i * (g_last.unsqueeze(-1) - g_i).exp().unsqueeze(-1)).transpose(
            -1, -2
        )
        @ v_new
    )

  out = torch.stack(outs, dim=2).reshape(bs, n_heads, -1, d_v)[:, :, :seq_len]
  return out.transpose(1, 2).contiguous().to(out_dtype)


def _pure_torch_gated_delta_kernel_forward(
    self: qwen3_5_gdn.GatedDeltaKernel,
    xq_THK: torch.Tensor,  # pylint: disable=invalid-name
    xk_THK: torch.Tensor,  # pylint: disable=invalid-name
    xv_THV: torch.Tensor,  # pylint: disable=invalid-name
    g_TH: torch.Tensor,  # pylint: disable=invalid-name
    beta_TH: torch.Tensor,  # pylint: disable=invalid-name
    *,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
  """Drop-in `GatedDeltaKernel.forward` (single packed sequence only)."""
  del self  # Unused.
  if cu_seqlens is not None:
    raise NotImplementedError(
        "The pure-torch GatedDeltaNet kernel does not support varlen"
        " (multi-document) inputs."
    )
  # Grouped value heads: each key head serves n_value_heads / n_key_heads
  # consecutive value heads (HF `repeat_interleave` convention).
  repeats = xv_THV.shape[1] // xq_THK.shape[1]
  if repeats > 1:
    xq_THK = xq_THK.repeat_interleave(repeats, dim=1)
    xk_THK = xk_THK.repeat_interleave(repeats, dim=1)
  return chunked_gated_delta(
      xq_THK.unsqueeze(0),
      xk_THK.unsqueeze(0),
      xv_THV.unsqueeze(0),
      g_TH.unsqueeze(0),
      beta_TH.unsqueeze(0),
  ).squeeze(0)


def use_pure_torch_gated_delta(model: torch.nn.Module) -> int:
  """Swaps every `GatedDeltaKernel` in `model` to `chunked_gated_delta`.

  Args:
    model: A built torchtitan Qwen3.5 model.

  Returns:
    The number of kernels swapped.
  """
  count = 0
  for module in model.modules():
    if isinstance(module, qwen3_5_gdn.GatedDeltaKernel):
      module.forward = types.MethodType(
          _pure_torch_gated_delta_kernel_forward, module
      )
      count += 1
  return count


def text_config(
    flavor: str, num_layers: int, seq_len: int
) -> qwen3_5.Qwen35Model.Config:
  """Returns a text-only torchtitan Qwen3.5 config using SDPA attention.

  Args:
    flavor: Key into `qwen3_5.qwen3_5_configs` (e.g. "27B" or "debugmodel").
    num_layers: Number of hybrid decoder layers to keep (3 GatedDeltaNet : 1
      attention per 4 layers). Values <= 0 keep all layers.
    seq_len: Max context length used to size the RoPE cache.
  """
  config = qwen3_5.model_registry(
      flavor, seq_len=seq_len, attn_backend="sdpa"
  ).model
  if not isinstance(config, qwen3_5.Qwen35Model.Config):
    raise TypeError(f"Unexpected config type {type(config).__name__}")
  if num_layers > 0:
    config.layers = config.layers[:num_layers]
  # Text-only benchmark: drop the vision tower.
  config.vision_encoder = None
  return config


def build_model(
    config: qwen3_5.Qwen35Model.Config,
    device: str,
    dtype: torch.dtype,
) -> torch.nn.Module:
  """Builds and initializes a torchtitan Qwen3.5 model directly on `device`.

  The model is built on meta and materialized on the device (no full host copy),
  its parameters are cast to `dtype` and its GatedDeltaNet kernels are swapped
  to the pure-torch chunked rule. Seed torch beforehand for reproducible
  weights.

  Args:
    config: Model config, e.g. from `text_config`.
    device: Device string (e.g. "tpu", "cuda" or "cpu").
    dtype: Parameter dtype.

  Returns:
    The initialized model, in train mode.
  """
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
  model = model.to(dtype=dtype)
  use_pure_torch_gated_delta(model)
  return model
