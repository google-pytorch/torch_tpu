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

"""Custom decompositions for TorchTPU compiler backend."""

from collections.abc import Sequence

import torch

# Dynamic Python decompositions to bypass eager C++ host-synchronization panics
# during JIT tracing.
TPU_DECOMPOSITIONS = {}


def _register_tpu_decomp(op):
  def decorator(fn):
    TPU_DECOMPOSITIONS[op] = fn
    return fn

  return decorator


@_register_tpu_decomp(torch.ops.aten.index_put.default)
@_register_tpu_decomp(torch.ops.aten.index_put_.default)
def _decomposed_index_put_tpu(
    self: torch.Tensor,
    indices: Sequence[torch.Tensor | None],
    values: torch.Tensor,
    accumulate: bool = False,
) -> torch.Tensor | object:
  """Decomposes advanced boolean indexing to a TPU-native static graph.

  Motivation (Why this decomposition exists):
  In eager PyTorch, advanced boolean indexing (`self[mask] = values`) produces
  a dynamically-shaped result of size `mask.sum()`. During JIT tracing in
  `torch.compile`, PyTorch's default behavior attempts to lower this by calling
  `nonzero()` on the mask to resolve the target indices.

  On TPU, `nonzero()` is highly problematic because:
  1.  It forces **host-synchronization** (the CPU must block and wait for the
      TPU to evaluate the mask, then the CPU allocates a tensor of that dynamic
      size).
  2.  Host-synchronization breaks the compiler JIT tracing pipeline, which
      requires a strictly static graph representation with known shapes.
  3.  It triggers compiler panics and crashes when attempting to materialize
      dynamic-shape placeholders on TPU HBM.

  This decomposition solves this by surgically replacing the dynamic
  `nonzero()`-based indexing with a sequence of strictly **static-shape,
  device-side operations** (broadcasting, cumsum, unsafe_index, and where).
  It executes the indexing entirely on the TPU without ever synchronizing
  with the host CPU, preserving a clean, compilation-friendly JIT graph.


  Args:
    self: The base tensor to be indexed.
    indices: A list of index tensors. This decomposition only supports a single
      boolean mask at `indices[0]` (i.e. `self[mask] = values`).
    values: The tensor containing the values to write.
    accumulate: If True, the values are added to `self` rather than overwriting
      them.

  Returns:
    The updated tensor with the values written or accumulated, or
    `NotImplemented` if the index pattern is not supported (so standard
    PyTorch fallback paths are used).
  """
  # Intercept advanced boolean indexing: self[mask] = values
  if not indices or indices[0] is None or indices[0].dtype != torch.bool:
    return NotImplemented

  # Only support a single boolean mask at the leading dimension, with
  # optional trailing slices (Nones)
  if not all(idx is None for idx in indices[1:]):
    return NotImplemented

  mask = indices[0]

  # 1. Align both device and dtype of values to self
  values = values.to(device=self.device, dtype=self.dtype)

  # Step 1. **Dimension Alignment**: Unsqueeze the mask to align dimensions
  # with `self` from the left (leading). PyTorch boolean indexing aligns
  # from the left, whereas standard broadcasting aligns from the right.
  if mask.ndim < self.ndim:
    mask = mask.view(mask.shape + (1,) * (self.ndim - mask.ndim))

  # Case A: values is a scalar or 0-D tensor
  if values.ndim == 0 or values.numel() == 1:
    if accumulate:
      # Use integer 0 to prevent allocating redundant zero-tensor on TPU HBM
      return self + torch.where(mask, values, 0)
    else:
      return torch.where(mask, values, self)

  # Case B: values is a tensor
  else:
    if self.numel() == 0 or values.numel() == 0:
      return self

    # Step 2. **Shape Broadcasting**: Broadcast `self`, `mask`, and
    # `values` to a common shape to handle all broadcasting cases safely
    # before flattening.
    self_broadcast, mask_broadcast = torch.ops.aten.broadcast_tensors(
        [self, mask]
    )

    # Broadcast values to target shape to handle all broadcasting cases safely
    # before flattening.
    self_shape = self_broadcast.shape
    vals_shape = values.shape
    offset = len(self_shape) - len(vals_shape)

    target_values_shape = [
        (
            vals_shape[i - offset]
            if i >= offset and vals_shape[i - offset] != 1
            else self_shape[i]
        )
        for i in range(len(self_shape))
    ]
    broadcasted_values = torch.broadcast_to(values, target_values_shape)

    # Step 3. **Flattening**: Flatten all tensors to 1D to simplify indexing.
    flat_self = self_broadcast.flatten()
    flat_mask = mask_broadcast.flatten()
    flat_source = broadcasted_values.flatten()

    # Step 4. **Prefix Sum (`cumsum`)**: Compute a device-side cumulative sum
    # of the flattened mask. This determines the target index in `flat_source`
    # for each `True` element in the mask without host-side synchronization.
    prefix_sum = torch.cumsum(flat_mask.to(torch.int64), dim=0)
    gather_indices = prefix_sum - 1

    # Step 5. **Clamping**: Clamp `gather_indices` to protect against
    # out-of-bounds panics at `False` mask positions on TPU during tracing.
    # `True` positions are guaranteed to be in-bounds after broadcasting.
    clamped_indices = torch.clamp(gather_indices, 0, flat_source.size(0) - 1)

    # Step 6. **Unsafe Indexing (`_unsafe_index`)**: Perform a dynamic gather
    # using the clamped indices. We use `_unsafe_index` because it lowers to a
    # native TPU gather without bounds checks, which traces stably.
    # pylint: disable=protected-access
    gathered = torch.ops.aten._unsafe_index(flat_source, [clamped_indices])

    # Step 7. **Elementwise Selection (`where`)**: Use the original mask to
    # select between `flat_self` (optionally accumulated with gathered values)
    # and `gathered` values.
    if accumulate:
      flat_output = torch.where(flat_mask, flat_self + gathered, flat_self)
    else:
      flat_output = torch.where(flat_mask, gathered, flat_self)

    return flat_output.view(self_broadcast.shape)
