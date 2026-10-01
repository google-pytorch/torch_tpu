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

"""Overrides for upstream PyTorch Meta kernel registrations on TPU."""

import torch
from torch import _meta_registrations

_meta_lib: torch.library.Library | None = None


# TODO: Remove once PyTorch includes https://github.com/pytorch/pytorch/pull/198303.
def _patch_sdpa_meta(meta_lib: torch.library.Library) -> None:
  """Patches `aten._scaled_dot_product_fused_attention_overrideable` Meta kernel."""
  # pylint: disable=protected-access
  op = torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default
  meta_key = torch._C.DispatchKey.Meta
  upstream_meta = op.py_kernels.pop(meta_key, None)
  if upstream_meta is None:
    return

  def _sdpa_overrideable_meta(
      query: torch.Tensor,
      key: torch.Tensor,
      value: torch.Tensor,
      attn_bias: torch.Tensor | None = None,
      dropout_p: float = 0.0,
      is_causal: bool = False,
      return_debug_mask: bool = False,
      scale: float | None = None,
  ):
    if _meta_registrations.device_hint(query) == "tpu" and query.dim() in (
        3,
        4,
    ):
      s_kv = key.size(-2)
      torch._check(
          s_kv == value.size(-2),
          lambda: (
              f"key sequence length ({s_kv}) must match "
              f"value sequence length ({value.size(-2)})"
          ),
      )
    return upstream_meta(
        query,
        key,
        value,
        attn_bias,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale,
    )

  meta_lib.impl(op, _sdpa_overrideable_meta, allow_override=True)
  op.py_impl(meta_key)(_sdpa_overrideable_meta)
  # pylint: enable=protected-access


def register() -> None:
  """Registers TPU Meta kernel overrides."""
  global _meta_lib
  if _meta_lib is not None:
    return
  _meta_lib = torch.library.Library("aten", "IMPL", "Meta")
  _patch_sdpa_meta(_meta_lib)
