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

"""Dynamic KV cache implementation optimized for PyTorch/TPU compilation.

Provides `InPlaceDynamicCache` and `InPlaceDynamicLayer` which manage
preallocated
and growable backing tensors in-place, avoiding `torch.cat` reallocations on
every decode step and eliminating stride gaps upon cropping.
"""

import functools
from typing import Any
import torch
from transformers import cache_utils

# Number of slots every KV buffer reserves past its usable capacity.
#
# `update()` exposes a view covering the valid slots. `repeat_kv()`'s
# `reshape()` picks between a `view` and a `clone()` based on contiguity, which
# installs the guard `Ne(head_dim * length, head_dim * buffer_len)`. Bounded
# dynamism validates it at `length`'s upper bound, and infers that bound from
# the tensor being sliced -- so slicing the buffer directly makes the bound
# saturate at `buffer_len` and land exactly on the forbidden equality.
#
# `_expose()` breaks that coupling by slicing the usable region first: the
# bound caps at `capacity` while the stride still reflects the longer buffer.
#
# The reserved slot must be a real dimension. Padding only the storage and
# exposing a shorter view does not survive functionalization, which rewrites
# the in-place write into an out-of-place op returning a densely packed tensor.
_CAPACITY_HEADROOM = 1


class InPlaceDynamicLayer(cache_utils.DynamicLayer):
  """A DynamicLayer backed by preallocated/growable tensors updated in-place.

  Eliminates `torch.cat` reallocations on every decode step, ensuring stable
  underlying tensor buffers and O(1) contiguous cropping.
  """

  def __init__(self, max_cache_len: int = 512, **kwargs):
    super().__init__()
    self.max_cache_len = max_cache_len
    self._keys_buf: torch.Tensor | None = None
    self._values_buf: torch.Tensor | None = None
    # Device-resident write offset. The in-place write below has to address the
    # buffer through a tensor index rather than a Python/symbolic scalar: under
    # bounded dynamism a scalar offset lowers to a `local_scalar_dense` on a
    # value derived from a graph placeholder, which the TPU backend cannot
    # materialize. This mirrors what `transformers.cache_utils.StaticLayer`
    # does for the same reason.
    self._write_pos: torch.Tensor | None = None

  def _allocate_buffer(
      self,
      batch_size: int,
      num_heads: int,
      capacity: int,
      head_dim: int,
  ) -> torch.Tensor:
    """Allocates a buffer holding `capacity` slots plus reserved headroom."""
    return torch.empty(
        (batch_size, num_heads, capacity + _CAPACITY_HEADROOM, head_dim),
        dtype=self.dtype,
        device=self.device,
    )

  def _usable_capacity(self) -> int:
    """Returns the largest sequence length the current buffers may hold."""
    assert self._keys_buf is not None
    return self._keys_buf.shape[-2] - _CAPACITY_HEADROOM

  def _expose(self, buffer: torch.Tensor, length: int) -> torch.Tensor:
    """Returns the view of `buffer` holding the `length` valid slots.

    The usable region is sliced off first, on purpose: the tracer bounds a
    symbolic slice end by the length of the tensor being sliced, so going
    through the intermediate caps `length` at `capacity` while the exposed view
    keeps the stride of the one-slot-longer buffer. See `_CAPACITY_HEADROOM`.
    """
    usable = buffer[:, :, : self._usable_capacity(), :]
    return usable[:, :, :length, :]

  def _current_length(self) -> int:
    """Returns the number of valid slots, possibly as a traced SymInt."""
    if not self.is_initialized or self.keys is None:
      return 0
    return self.keys.shape[-2]

  def lazy_initialization(
      self,
      key_states: torch.Tensor,
      value_states: torch.Tensor | None = None,
  ) -> None:
    self.dtype, self.device = key_states.dtype, key_states.device
    batch_size, num_heads, seq_len = key_states.shape[:3]
    k_head_dim = key_states.shape[-1]
    v_head_dim = (
        value_states.shape[-1] if value_states is not None else k_head_dim
    )

    capacity = max(self.max_cache_len, seq_len)
    self._keys_buf = self._allocate_buffer(
        batch_size, num_heads, capacity, k_head_dim
    )
    self._values_buf = self._allocate_buffer(
        batch_size, num_heads, capacity, v_head_dim
    )
    if seq_len > 0:
      self._keys_buf[:, :, :seq_len, :].copy_(key_states)
      if value_states is not None and value_states.numel() > 0:
        self._values_buf[:, :, :seq_len, :].copy_(value_states)
    self.keys = self._expose(self._keys_buf, seq_len)
    self.values = self._expose(self._values_buf, seq_len)

    self._write_pos = torch.tensor(
        seq_len, dtype=torch.long, device=self.device
    )
    self.is_initialized = True

  def update(
      self,
      key_states: torch.Tensor,
      value_states: torch.Tensor,
      *args,
      **kwargs,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    new_len = key_states.shape[-2]
    if (
        not self.is_initialized
        or self._keys_buf is None
        or self._values_buf is None
    ):
      self.lazy_initialization(key_states, value_states)
      return self.keys, self.values

    # Derive the current length from the exposed view rather than from a Python
    # attribute, so that the length stays a shape symbol the tracer can reason
    # about instead of a value it has to specialize on.
    start = self._current_length()
    end = start + new_len
    if end > self._usable_capacity():
      new_cap = max(int(end), self._usable_capacity() * 2)
      old_k, old_v = self._keys_buf, self._values_buf
      batch_size, num_heads = key_states.shape[:2]
      self._keys_buf = self._allocate_buffer(
          batch_size, num_heads, new_cap, key_states.shape[-1]
      )
      self._values_buf = self._allocate_buffer(
          batch_size, num_heads, new_cap, value_states.shape[-1]
      )
      if old_k is not None and old_v is not None and start > 0:
        self._keys_buf[:, :, :start, :].copy_(old_k[:, :, :start, :])
        self._values_buf[:, :, :start, :].copy_(old_v[:, :, :start, :])

    assert self._keys_buf is not None
    assert self._values_buf is not None
    assert self._write_pos is not None

    # Address the write through a tensor index, never a (possibly symbolic)
    # scalar offset. See the note on `_write_pos`.
    cache_position = torch.arange(new_len, device=self.device) + self._write_pos
    self._write_pos.add_(new_len)
    try:
      self._keys_buf.index_copy_(2, cache_position, key_states)
      self._values_buf.index_copy_(2, cache_position, value_states)
    except NotImplementedError:
      # Fallback for devices where `index_copy_` is unimplemented.
      self._keys_buf[:, :, cache_position] = key_states
      self._values_buf[:, :, cache_position] = value_states

    self.keys = self._expose(self._keys_buf, end)
    self.values = self._expose(self._values_buf, end)
    return self.keys, self.values

  def get_seq_length(self) -> int:
    return self._current_length()

  def crop(self, tokens_to_remove: int) -> None:
    if not self.is_initialized:
      return
    current_len = int(self._current_length())
    if current_len == 0:
      return
    if tokens_to_remove > 0:
      if tokens_to_remove >= current_len:
        return
      target_len = tokens_to_remove
    else:
      target_len = max(0, current_len - abs(tokens_to_remove))

    if self._write_pos is not None:
      # Mutate in place so the tensor keeps its stable data pointer.
      self._write_pos.fill_(target_len)
    if self._keys_buf is not None and self._values_buf is not None:
      self.keys = self._expose(self._keys_buf, target_len)
      self.values = self._expose(self._values_buf, target_len)

  def reset(self) -> None:
    if self._write_pos is not None:
      self._write_pos.fill_(0)
    if self._keys_buf is not None and self._values_buf is not None:
      self.keys = self._expose(self._keys_buf, 0)
      self.values = self._expose(self._values_buf, 0)

  def batch_repeat_interleave(self, repeats: int) -> None:
    if self._keys_buf is None or self._values_buf is None:
      return
    current_len = int(self._current_length())
    if current_len == 0:
      return
    # `repeat_interleave` returns a fresh, densely packed tensor covering the
    # whole buffer, so the reserved headroom is carried over unchanged.
    self._keys_buf = self._keys_buf.repeat_interleave(repeats, dim=0)
    self._values_buf = self._values_buf.repeat_interleave(repeats, dim=0)
    self.keys = self._expose(self._keys_buf, current_len)
    self.values = self._expose(self._values_buf, current_len)

  def batch_select_indices(self, indices: torch.Tensor) -> None:
    if self._keys_buf is None or self._values_buf is None:
      return
    current_len = int(self._current_length())
    if current_len == 0:
      return
    self._keys_buf = self._keys_buf[indices, ...]
    self._values_buf = self._values_buf[indices, ...]
    self.keys = self._expose(self._keys_buf, current_len)
    self.values = self._expose(self._values_buf, current_len)


class InPlaceDynamicCache(cache_utils.DynamicCache):
  """DynamicCache drop-in replacement that uses InPlaceDynamicLayer."""

  def __init__(
      self,
      config: Any = None,
      max_cache_len: int = 512,
      **kwargs,
  ):
    try:
      super().__init__(config=config, **kwargs)
    except TypeError:
      # Older `DynamicCache` signatures do not accept `config`.
      super().__init__(**kwargs)
    new_layers = []
    for layer in self.layers:
      if isinstance(layer, cache_utils.DynamicLayer) and not getattr(
          layer, "is_sliding", False
      ):
        new_layers.append(InPlaceDynamicLayer(max_cache_len=max_cache_len))
      else:
        new_layers.append(layer)
    self.layers = new_layers
    # `Cache.update()` lazily appends `layer_class_to_replicate()` (no args)
    # for layer indices it has not seen yet, so the capacity has to be bound
    # into the factory or those layers silently fall back to the default.
    # Transformers annotates this slot as a class but only ever calls it, so a
    # partial is substitutable here.
    self.layer_class_to_replicate = functools.partial(  # pyrefly: ignore[bad-assignment]
        InPlaceDynamicLayer, max_cache_len=max_cache_len
    )

  @classmethod
  def from_dynamic_cache(
      cls,
      dynamic_cache: Any,
      max_cache_len: int = 512,
  ) -> "InPlaceDynamicCache":
    """Converts a standard DynamicCache into an InPlaceDynamicCache."""
    new_cache = cls(max_cache_len=max_cache_len)
    new_layers = []
    for layer in getattr(dynamic_cache, "layers", []):
      if type(layer).__name__ == "DynamicLayer" and not getattr(
          layer, "is_sliding", False
      ):
        inplace_layer = InPlaceDynamicLayer(max_cache_len=max_cache_len)
        if getattr(layer, "is_initialized", False) and layer.keys is not None:
          inplace_layer.lazy_initialization(layer.keys, layer.values)
        new_layers.append(inplace_layer)
      else:
        new_layers.append(layer)
    new_cache.layers = new_layers
    return new_cache


def tpu_forward_hook(module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
  """Forward pre-hook upgrading a standard DynamicCache to InPlaceDynamicCache.

  Matching on the class name as well as by `isinstance` keeps this working for
  the transformers v4, v5 and duck-typed variants alike.
  """
  # TODO(lizliang): QuantizedLayer (and QuantizedCache) has the same problem
  # because its residual buffer dynamically calls `torch.cat` on every decode
  # step, causing continuous Dynamo retraces and recompilations on TPU.
  # Implement an in-place preallocated equivalent or converter for
  # QuantizedLayer.
  cache = kwargs.get("past_key_values", None)
  if (
      cache is not None
      and not isinstance(cache, InPlaceDynamicCache)
      and (
          type(cache).__name__ == "DynamicCache"
          or isinstance(cache, cache_utils.DynamicCache)
      )
  ):
    kwargs["past_key_values"] = InPlaceDynamicCache.from_dynamic_cache(cache)
  return args, kwargs


_CACHE_HOOK_HANDLE_ATTR = "_torch_tpu_cache_hook_handle"


def register_tpu_cache_hook(module: torch.nn.Module) -> Any:
  """Registers `tpu_forward_hook` as a forward pre-hook, at most once.

  Re-registering would stack duplicate hooks on the same module, each of which
  Dynamo traces as its own frame, so an existing registration is reused.

  Args:
    module: The module whose `past_key_values` kwarg should be upgraded.

  Returns:
    The hook handle, which callers may use to remove the hook.
  """
  handle = getattr(module, _CACHE_HOOK_HANDLE_ATTR, None)
  if handle is None:
    handle = module.register_forward_pre_hook(
        tpu_forward_hook, with_kwargs=True
    )
    setattr(module, _CACHE_HOOK_HANDLE_ATTR, handle)
  return handle


def patch_transformers_dynamic_cache() -> None:
  """Patches `transformers.cache_utils.DynamicCache` to use `InPlaceDynamicCache`.

  This provides a near-term solution for casual users so that any standard
  `DynamicCache` instantiated by user code or Hugging Face generation pipelines
  automatically uses in-place preallocated layers, preventing continuous
  recompilation on TPU backend.
  """
  cache_utils.DynamicCache = InPlaceDynamicCache  # pyrefly: ignore[bad-assignment]
  cache_utils.DynamicLayer = InPlaceDynamicLayer  # pyrefly: ignore[bad-assignment]
