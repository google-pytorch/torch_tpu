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

"""Experimental Jagged Tensor utilities for TPU performance optimization.

These utilities optimize `torch.jagged` nested tensor execution on TPUs by
aligning flat values buffers to hardware microarchitectural boundaries:

1. **Hardware Systolic Tile Alignment**:
   TPU Matrix Multiply Units (MXUs) compute matrix multiplications in fixed
   128x128 systolic loop tiles. For ragged sequence batches where total token
   count `M` is not a multiple of 128, the hardware must execute a full
   128-token systolic pass for the remainder tokens. Pre-padding flat values
   buffers to multiples of 128 ensures full systolic compute lane utilization
   and aligns data with 128-byte Vector Processing Unit (VPU) vector registers
   (VREGs) without needing scalar peel loops.

2. **Compilation Cache Stability & Dynamic Shape Bucketing**:
   Irregular sequence lengths naturally vary per batch. Pre-padding to
   multiples of 128 collapses thousands of distinct irregular batch shapes
   into a small set of shared compilation cache buckets, avoiding expensive
   XLA graph recompilations.

3. **End-to-End Inter-Layer Graph Fusion**:
   Padding the jagged values buffer once at the input boundary allows the
   entire multi-layer Transformer pipeline (Linear1 -> GELU -> Linear2 ->
   LayerNorm -> Attention) to execute and fuse as a single dense graph without
   inserting intermediate `pad` <-> `slice` HLO roundtrips between layers.

4. **Zero-Sync Device-to-Host (D2H) Pipeline Dispatch**:
   `strip_jagged_padding` provides zero-copy slicing of trailing padding
   rows before loss computation or cross-node communication. Passing host-known
   `valid_tokens` eliminates the blocking `.item()` PCIe synchronization barrier
   that would otherwise stall host dispatch and create TPU execution bubbles.
"""

import typing
import torch
from torch.nested._internal import nested_tensor as nt_internal
from torch_tpu._internal.utils import annotations

experimental = annotations.experimental
__all__ = [
    "align_jagged_to_multiple",
    "strip_jagged_padding",
]


@experimental("align_jagged_to_multiple is experimental and subject to change.")
def align_jagged_to_multiple(
    nt: torch.Tensor,
    multiple: int = 128,
    padding_value: typing.Union[float, int, bool] = 0.0,
) -> torch.Tensor:
  """Pads a jagged tensor's flat values buffer to a multiple of `multiple`.

  This optimizes TPU matrix multiplication throughput by aligning sequence
  token dimensions to native 128x128 MXU systolic array tiles and 128-element
  vector registers. Sequence offsets and valid elements remain unchanged.

  Args:
    nt: A `torch.jagged` NestedTensor on TPU.
    multiple: Tile alignment size (default 128 for TPU MXU systolic arrays).
    padding_value: Fill value for the padded trailing tokens (default 0.0).

  Returns:
    A new `torch.jagged` NestedTensor with its values buffer padded to a
    multiple of `multiple`, sharing the original sequence offsets.
  """
  values = nt.values()
  total_l = values.size(0)
  remainder = total_l % multiple
  if remainder == 0:
    return nt

  pad_amount = multiple - remainder
  pad_shape = (pad_amount,) + tuple(values.shape[1:])
  padding = torch.full(
      pad_shape,
      padding_value,
      dtype=values.dtype,
      device=values.device,
  )
  padded_values = torch.cat([values, padding], dim=0)
  return nt_internal.nested_view_from_values_offsets(
      padded_values, nt.offsets()
  )


@experimental("strip_jagged_padding is experimental and subject to change.")
def strip_jagged_padding(
    nt: torch.Tensor,
    valid_tokens: typing.Optional[int] = None,
) -> torch.Tensor:
  """Removes trailing padding rows from a padded jagged values buffer.

  If `nt.values()` has trailing dummy tokens beyond `nt.offsets()[-1]` (e.g.
  from `align_jagged_to_multiple`), this function returns a zero-copy slice view
  matching the exact valid token count.

  Args:
    nt: A `torch.jagged` NestedTensor on TPU.
    valid_tokens: Optional host-known integer count of valid tokens in the
      batch. If omitted, computed from `nt.offsets()[-1].item()`. Note: passing
      a host-known integer avoids the blocking PCIe Device-to-Host (D2H)
      synchronization caused by `.item()`, eliminating TPU pipeline stalls.

  Returns:
    A `torch.jagged` NestedTensor containing only the valid sequence tokens.
  """
  offsets = nt.offsets()
  if valid_tokens is None:
    valid_tokens = int(offsets[-1].item()) if offsets.numel() > 0 else 0

  if nt.values().size(0) == valid_tokens:
    return nt

  compact_values = nt.values()[:valid_tokens]
  return nt_internal.nested_view_from_values_offsets(compact_values, offsets)
