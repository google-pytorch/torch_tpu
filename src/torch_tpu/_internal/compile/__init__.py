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

"""Public API for torch_tpu._internal.compile."""

import torch
from torch._dynamo.backends import registry
from torch._functorch._aot_autograd import utils as aot_utils
from torch_tpu._internal.compile import grouped_mm_meta
from torch_tpu._internal.compile._backend import async_compile
from torch_tpu._internal.compile._backend import AsyncCompilationSubmitted
from torch_tpu._internal.compile._backend import clear_cache
from torch_tpu._internal.compile._backend import resolve_compilations
from torch_tpu._internal.compile._backend import TpuBackend
from torch_tpu._internal.compile.debug import TpuCompileDebug

# Register "tpu" backend
registry.register_backend(compiler_fn=TpuBackend(), name="tpu")  # pyrefly: ignore[bad-argument-type]


def _initialize_graphsafe_rng():
  """Initializes graphsafe RNG operations for TPU.

  Registers 'privateuseone' (TPU) for graphsafe RNG and enables
  necessary config flags.
  """
  aot_utils.register_graphsafe_rng_device_type("privateuseone")
  # Add "tpu" to the supported device types for graphsafe RNG, as torch_tpu
  # renames "privateuseone" to "tpu".
  aot_utils._GRAPHSAFE_RNG_DEVICE_TYPES.add("tpu")  # pylint: disable=protected-access

  # pylint: disable=protected-access
  torch._functorch.config.graphsafe_rng_functionalization = True
  torch._functorch.config.functionalize_rng_ops = False
  # pylint: enable=protected-access


# TODO: b/496168350 - Remove if-condition once generic graphsafe RNG is
# available in OSS PyTorch release.
if hasattr(aot_utils, "register_graphsafe_rng_device_type"):
  _initialize_graphsafe_rng()


# New Dynamo flag that enables tracing through the `backward` function call.
# This enables the possibility of generating of a single fx graph containing
# both forward and backward subgraphs, rather than a discrete fx graph for each.
# See b/480979694 for details.
# pylint: disable=protected-access
if hasattr(torch._dynamo.config, "trace_autograd_ops"):
  torch._dynamo.config.trace_autograd_ops = True
# Disables PyTorch Dynamo's default graph break on RNN, LSTM, and GRU modules,
# allowing Dynamo to trace recurrent layers into aten::lstm and aten::gru nodes
# for lowering to TorchTPU's StableHLO kernels.
if hasattr(torch._dynamo.config, "allow_rnn"):
  torch._dynamo.config.allow_rnn = True
# pylint: enable=protected-access


def _register_scan_operator() -> None:
  """Registers the custom scan operator implementation for TPU."""
  from torch_tpu._internal.compile import scan as _  # pylint: disable=g-import-not-at-top  # noqa: F401


_register_scan_operator()


def _patch_flex_attention() -> None:
  """Patches PyTorch flex_attention to support TPU tensors."""
  # pylint: disable=g-import-not-at-top,protected-access
  from torch._higher_order_ops.flex_attention import flex_attention_backward
  from torch.nn.attention import flex_attention
  import torch.nn.functional as F

  original_validate = flex_attention._validate_device

  def _validate_device(query, key, value):
    if query.device.type != "tpu":
      original_validate(query, key, value)

  flex_attention._validate_device = _validate_device

  dispatch_key = torch._C.DispatchKey.CompositeExplicitAutograd
  original_bwd = flex_attention_backward.py_kernels.pop(dispatch_key)

  @flex_attention_backward.py_impl(dispatch_key)
  def _sdpa_dense_backward(query, key, value, *args, **kwargs):
    with query.device:
      return original_bwd(query, key, value, *args, **kwargs)

  original_flex_attention = flex_attention.flex_attention

  def _get_sdpa_attn_bias(
      block_mask: flex_attention.BlockMask,
      q_len: int,
      kv_len: int,
      dtype: torch.dtype,
      device: torch.device,
  ) -> torch.Tensor:
    if torch.compiler.is_compiling():
      # Build the bias inside the compiled graph rather than caching it on the
      # BlockMask, so the trace has no Python side effects.
      cache = {}
    else:
      cache = block_mask.__dict__.setdefault("_tpu_sdpa_mask_cache", {})
    key = (q_len, kv_len, dtype, device)
    if key not in cache:
      b_mask, h_mask = block_mask.kv_num_blocks.shape[:2]
      with torch.no_grad():
        bool_mask = flex_attention.create_mask(
            block_mask.mask_mod,
            b_mask,
            h_mask,
            q_len,
            kv_len,
            device=device,
        )
        attn_bias = torch.zeros(
            bool_mask.shape, dtype=dtype, device=device
        ).masked_fill_(~bool_mask, float("-inf"))
      cache[key] = attn_bias
    return cache[key]

  def _flex_attention_tpu(
      query: torch.Tensor,
      key: torch.Tensor,
      value: torch.Tensor,
      score_mod=None,
      block_mask=None,
      scale=None,
      enable_gqa: bool = False,
      return_lse: bool = False,
      kernel_options=None,
      *,
      return_aux=None,
  ):
    need_aux = return_lse or (
        return_aux is not None
        and (
            getattr(return_aux, "lse", False)
            or getattr(return_aux, "max_scores", False)
        )
    )
    if (
        query.device.type == "tpu"
        and score_mod is None
        and not need_aux
        and query.dtype in (torch.bfloat16, torch.float32)
    ):
      q_len = query.shape[-2]
      kv_len = key.shape[-2]
      attn_bias = (
          _get_sdpa_attn_bias(
              block_mask, q_len, kv_len, query.dtype, query.device
          )
          if block_mask is not None
          else None
      )
      out = F.scaled_dot_product_attention(
          query,
          key,
          value,
          attn_mask=attn_bias,
          dropout_p=0.0,
          is_causal=False,
          scale=scale,
          enable_gqa=enable_gqa,
      )
      if return_aux is not None:
        return out, flex_attention.AuxOutput()
      return out
    return original_flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_lse=return_lse,
        kernel_options=kernel_options,
        return_aux=return_aux,
    )

  flex_attention.flex_attention = _flex_attention_tpu

  # pylint: enable=g-import-not-at-top,protected-access


_patch_flex_attention()

grouped_mm_meta.register()

# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "AsyncCompilationSubmitted",
    "TpuBackend",
    "TpuCompileDebug",
    "async_compile",
    "clear_cache",
    "resolve_compilations",
    # go/keep-sorted end
]
