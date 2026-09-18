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

"""Tests the contiguity guard raised by reading a preallocated KV cache.

`InPlaceDynamicLayer` hands the model a view covering the valid prefix of a
preallocated buffer. Attention then calls `repeat_kv()`, whose `reshape()`
chooses between a plain `view` and a `clone() + _unsafe_view` depending on
contiguity, installing the shape guard
`Ne(head_dim * length, head_dim * buffer_len)`.

Bounded dynamism validates that guard by substituting the *upper bound* of
`length`, and infers that bound from the length of the tensor being sliced.
Slicing the buffer directly couples the two, so the bound lands exactly on the
forbidden equality. Reserving a trailing slot and slicing through the usable
region first decouples them, which is what these tests pin down.
"""

from __future__ import annotations

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils

_BATCH = 1
_NUM_KV_HEADS = 2
_HEAD_DIM = 8
_N_REP = 3
_CAPACITY = 6


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
  """Mirrors `transformers.integrations.sdpa_attention.repeat_kv`."""
  batch, num_kv_heads, slen, head_dim = hidden_states.shape
  expanded = hidden_states[:, :, None, :, :].expand(
      batch, num_kv_heads, n_rep, slen, head_dim
  )
  return expanded.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class _PreallocatedKVCacheModel(torch.nn.Module):
  """Reads the valid prefix of a preallocated buffer, as the KV cache does.

  `usable_capacity` is the number of slots the cache may fill; the buffer it is
  given is allowed to be longer, and that extra length is what keeps the
  reshape guard satisfiable.
  """

  def __init__(self, usable_capacity: int):
    super().__init__()
    self.usable_capacity = usable_capacity

  def forward(self, buffer: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    usable = buffer[:, :, : self.usable_capacity, :]
    view = usable[:, :, : valid.shape[-2], :]
    return _repeat_kv(view, _N_REP)


class KVCacheReshapeGuardTest(seed_test_utils.RepeatableTest):

  def _make_inputs(
      self, headroom: int, valid_length: int
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds a buffer with `headroom` reserved slots and a length carrier."""
    buffer_length = _CAPACITY + headroom
    buffer = torch.arange(
        _BATCH * _NUM_KV_HEADS * buffer_length * _HEAD_DIM,
        dtype=torch.float32,
        device="tpu",
    ).reshape(_BATCH, _NUM_KV_HEADS, buffer_length, _HEAD_DIM)
    # Stands in for the previously exposed view, whose sequence dimension is
    # what makes the read length a shape symbol.
    valid = torch.zeros(
        _BATCH, _NUM_KV_HEADS, valid_length, _HEAD_DIM, device="tpu"
    )
    torch._dynamo.mark_dynamic(valid, 2)
    return buffer, valid

  def test_reserved_slot_keeps_reshape_guard_satisfiable(self):
    model = _PreallocatedKVCacheModel(_CAPACITY)
    compiled = torch.compile(
        model, backend="tpu", options={"bounded_dynamism": True}
    )

    # Reading right up to the usable capacity is the case that lands on the
    # forbidden equality when the buffer has no reserved slot.
    for valid_length in (_CAPACITY - 2, _CAPACITY - 1, _CAPACITY):
      with self.subTest(valid_length=valid_length):
        buffer, valid = self._make_inputs(headroom=1, valid_length=valid_length)
        utils.assert_close(compiled(buffer, valid), model(buffer, valid))


if __name__ == "__main__":
  absltest.main()
