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
# pylint: disable=protected-access
"""Tensor subclass for staging KV cache state at bucket boundaries."""

from absl import logging
import torch
from torch.utils._python_dispatch import return_and_correct_aliasing


class StateStagedTensor(torch.Tensor):
  """Subclass wrapping logical view but exposing padded base to compiler.

  This subclass allows PyTorch Dynamo to trace the model using the logical
  sequence length shape (S), while the compiler backend receives and executes
  on the contiguous padded base buffer (shape B). This avoids expensive eager
  TPU allocations and copies (.contiguous()) at every step.
  """

  @staticmethod
  def __new__(
      cls, logical_view, padded_base, logical_len, bucket_len, strides=None
  ):
    # Create the wrapper subclass with the shape of the logical view (S)
    if strides is None:
      strides = logical_view.stride()
    return torch.Tensor._make_wrapper_subclass(
        cls,
        logical_view.shape,
        strides=strides,
        storage_offset=logical_view.storage_offset(),
        dtype=logical_view.dtype,
        device=logical_view.device,
        requires_grad=logical_view.requires_grad,
    )

  def __init__(
      self, logical_view, padded_base, logical_len, bucket_len, strides=None
  ):
    self._logical_view = logical_view
    self._base_tensor = padded_base
    self.bucket_len = bucket_len

  @property
  def logical_len(self):
    return self.shape[2]

  def __repr__(self):
    return (
        f"StateStagedTensor(shape={list(self.shape)},"
        f" logical_len={self.logical_len}, bucket_len={self.bucket_len})"
    )

  def __tensor_flatten__(self):
    # Expose the padded base tensor to the compiler, hiding the sliced view
    return ["_base_tensor"], (self.logical_len, self.bucket_len)

  @staticmethod
  def __tensor_unflatten__(inner_tensors, meta, sizes, strides):
    _base_tensor = inner_tensors["_base_tensor"]
    logical_len, bucket_len = meta
    # _base_tensor shape is [1, batch, num_heads, bucket_len, head_dim]
    # (5D storage)
    # sizes is [batch, num_heads, logical_len, head_dim] (4D)
    # We construct 5D slices and squeeze the output to 4D
    slices = (slice(None),) + tuple(slice(0, sz) for sz in sizes)
    logical_view_5d = _base_tensor[slices]
    logical_view = logical_view_5d.squeeze(0)
    return StateStagedTensor(
        logical_view, _base_tensor, logical_len, bucket_len, strides=strides
    )

  @classmethod
  def __metadata_guard__(cls, orig_metadata, other_metadata):
    # orig_metadata = (logical_len_1, bucket_len_1)
    # other_metadata = (logical_len_2, bucket_len_2)
    # We only recompile if the physical bucket size changed.
    return orig_metadata[1] == other_metadata[1]

  @classmethod
  def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    if kwargs is None:
      kwargs = {}

    def unwrap(x):
      return x._logical_view if isinstance(x, StateStagedTensor) else x

    args_unwrapped = torch.utils._pytree.tree_map(unwrap, args)
    kwargs_unwrapped = torch.utils._pytree.tree_map(unwrap, kwargs)

    # Debug logging
    flat_args, _ = torch.utils._pytree.tree_flatten(args)
    flat_args_unwrapped, _ = torch.utils._pytree.tree_flatten(args_unwrapped)

    def arg_val(x):
      if isinstance(x, torch.Tensor):
        if isinstance(x, StateStagedTensor):
          return f"Subclass(shape={x.shape}, base={x._base_tensor.shape})"
        return f"Tensor(shape={x.shape})"
      return str(x)

    logging.debug("[__torch_dispatch__] func: %s", func)
    logging.debug("  args: %s", [arg_val(x) for x in flat_args])
    logging.debug("  unwrapped: %s", [arg_val(x) for x in flat_args_unwrapped])

    out_unwrapped = func(*args_unwrapped, **kwargs_unwrapped)

    if func == torch.ops.aten.cat.default:

      def wrap(x):
        if not isinstance(x, torch.Tensor) or isinstance(x, StateStagedTensor):
          return x

        # x is [batch, num_heads, logical_len, head_dim] (4D)
        logical_len = x.shape[2]

        # Extract hint for logical_len to compute bucket_len
        if isinstance(logical_len, torch.SymInt):
          concrete_logical_len = int(logical_len.node.hint)
        else:
          concrete_logical_len = logical_len

        # Compute new bucket_len
        bucket_len = 1 << (concrete_logical_len - 1).bit_length()
        if bucket_len < 4:
          bucket_len = 4

        # Allocate 5D padded base
        target_shape = [1] + list(x.shape)
        target_shape[3] = bucket_len

        padded_base = torch.zeros(target_shape, dtype=x.dtype, device=x.device)
        # Copy x into padded_base
        x_5d = x.unsqueeze(0)
        slices = [slice(None)] * len(target_shape)
        slices[3] = slice(0, logical_len)
        padded_base[slices] = x_5d

        # Construct logical_view as view of padded_base
        logical_view_5d = padded_base[:, :, :, :logical_len, :]
        logical_view = logical_view_5d.squeeze(0)

        return StateStagedTensor(
            logical_view,
            padded_base,
            logical_len,
            bucket_len,
        )

      out_wrapped = torch.utils._pytree.tree_map(wrap, out_unwrapped)

      return return_and_correct_aliasing(func, args, kwargs, out_wrapped)
    else:
      return out_unwrapped
