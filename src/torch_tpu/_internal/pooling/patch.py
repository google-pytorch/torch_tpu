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

"""Patches PyTorch max pooling operations to route TPU tensors to native TPU ops.

### Problem Overview: The Dead Index Problem
In PyTorch core, `aten::max_pool1d`, `aten::max_pool2d`, and `aten::max_pool3d`
are defined as `CompositeImplicitAutograd` operators. Under the hood, PyTorch's
native composite implementation decomposes `max_pool*d` into its with-indices
variant (`max_pool*d_with_indices`), even when `return_indices=False`.

In standard PyTorch autograd:
1. The forward pass computes both the pooled output and an integer tensor of
   argmax indices.
2. The index tensor is saved in memory (`ctx.save_for_backward(indices)`) for
the
   entire forward duration until the backward pass executes.
3. The backward pass uses these indices to scatter gradients back to the input.

### Why This Hurts Performance on TPU:
1. **Compilation & Kernel Complexity**: Computing indices requires
`ReduceWindow`
   to emit a pair of outputs `(result, indices)` or build an integer index
   tracking subgraph, increasing HLO graph complexity and compilation time.
2. **Memory Bloat**: Holding large integer index tensors in device High
Bandwidth
   Memory (HBM) throughout forward and backward passes inflates peak memory
   usage
   by up to 66% during training.
3. **Inefficient Backward Lowering**: On TPUs, re-evaluating the argmax mask via
   StableHLO's `SelectAndScatter` (which compares the input against the output
   and
   scatters the gradient) is vastly faster (up to 50x faster) than gathering and
   scattering with an integer index tensor.

### Solution: Python-Level Autograd Interception
Because `aten::max_pool*d` is `CompositeImplicitAutograd`, attempting to
override
it in C++ (`PrivateUse1` or `AutogradPrivateUse1`) either bypasses autograd
graph
construction entirely (breaking training) or conflicts with AOTAutograd and
`functorch` tracing during `torch.compile`.

This module intercepts calls at the Python level
(`torch.nn.functional.max_pool*d`):
1. **Eval / No-Grad Mode**: When gradients are not required, routes directly to
   `torch.ops.tpu._max_pool*d`, invoking a single-operand `ReduceWindow` kernel
   with zero index materialization.
2. **Train / Autograd Mode**: When gradients are required, routes through a
   custom `torch.autograd.Function` (`_TpuMaxPool*Function`). The forward pass
   computes the pooled output without indices and saves *only* the input tensor.
   The backward pass calls `torch.ops.tpu._max_pool*d_backward`, lowering
   directly to StableHLO's `SelectAndScatterOp`.
3. **torch.compile Compatibility**: The autograd functions define
   `setup_context(ctx, inputs, output)` with `forward(*inputs)` (separate from
   `ctx`), adhering to the modern PyTorch autograd contract required by
   Dynamo, AOTAutograd, and `functorch`.
"""

import functools
from typing import Any
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _single, _triple

# Save references to original PyTorch functional pooling implementations
# so we can fall back when conditions are not met, or when unpatching.
_ORIG_F_MAX_POOL1D = F.max_pool1d
_ORIG_F_MAX_POOL2D = F.max_pool2d
_ORIG_F_MAX_POOL3D = F.max_pool3d

# Global state tracking whether the monkey patches are currently active.
_IS_PATCHED: bool = False


def _is_tpu_tensor(tensor: Any) -> bool:
  """Returns True if the object is a Tensor on a TPU device."""
  return isinstance(tensor, torch.Tensor) and tensor.device.type == "tpu"


# -----------------------------------------------------------------------------
# Differentiable Autograd Functions for TPU Max Pooling
# -----------------------------------------------------------------------------
# These classes inherit from `torch.autograd.Function` and implement the
# `setup_context` pattern. This pattern allows PyTorch's AOTAutograd / Dynamo
# compilers to trace the forward and backward graphs independently without
# tracing dead index computations into the graph.


class _TpuMaxPool1dFunction(torch.autograd.Function):
  """Differentiable TPU 1D max pooling using SelectAndScatter without indices.

  In the forward pass, computes the pooled output using
  `torch.ops.tpu._max_pool1d` without computing or materializing indices. In
  the backward pass, uses `torch.ops.tpu._max_pool1d_backward` to recompute the
  argmax positions on the fly via StableHLO `SelectAndScatterOp`.
  """

  @staticmethod
  def forward(
      input: torch.Tensor,
      kernel_size: list[int],
      stride: list[int],
      padding: list[int],
      dilation: list[int],
      ceil_mode: bool,
  ) -> torch.Tensor:
    """Executes native 1D max pooling without index materialization."""
    return torch.ops.tpu._max_pool1d(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )

  @staticmethod
  def setup_context(
      ctx: Any,
      inputs: tuple[Any, ...],
      output: Any,
  ) -> None:
    """Saves only the forward input tensor and pooling metadata for backward.

    Crucially, only the input tensor is saved; NO index tensor is materialized
    or retained in memory.
    """
    input, kernel_size, stride, padding, dilation, ceil_mode = inputs
    ctx.save_for_backward(input)
    ctx.kernel_size = kernel_size
    ctx.stride = stride
    ctx.padding = padding
    ctx.dilation = dilation
    ctx.ceil_mode = ceil_mode

  @staticmethod
  def backward(  # pytype: disable=bad-override  # pyrefly: ignore[bad-override]
      ctx: Any, *grad_outputs: Any
  ) -> tuple[Any, ...]:
    """Computes grad_input using SelectAndScatter from saved input tensor."""
    (grad_output,) = grad_outputs
    (input,) = ctx.saved_tensors
    grad_input = torch.ops.tpu._max_pool1d_backward(
        grad_output,
        input,
        ctx.kernel_size,
        ctx.stride,
        ctx.padding,
        ctx.dilation,
        ctx.ceil_mode,
    )
    # Return gradient for each forward input argument: input, kernel_size,
    # stride, padding, dilation, ceil_mode. Only input requires grad.
    return grad_input, None, None, None, None, None


class _TpuMaxPool2dFunction(torch.autograd.Function):
  """Differentiable TPU 2D max pooling using SelectAndScatter without indices.

  In the forward pass, computes the pooled output using
  `torch.ops.tpu._max_pool2d` without computing or materializing indices. In
  the backward pass, uses `torch.ops.tpu._max_pool2d_backward` to recompute the
  argmax positions on the fly via StableHLO `SelectAndScatterOp`.
  """

  @staticmethod
  def forward(
      input: torch.Tensor,
      kernel_size: list[int],
      stride: list[int],
      padding: list[int],
      dilation: list[int],
      ceil_mode: bool,
  ) -> torch.Tensor:
    """Executes native 2D max pooling without index materialization."""
    return torch.ops.tpu._max_pool2d(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )

  @staticmethod
  def setup_context(
      ctx: Any,
      inputs: tuple[Any, ...],
      output: Any,
  ) -> None:
    """Saves only the forward input tensor and pooling metadata for backward.

    Crucially, only the input tensor is saved; NO index tensor is materialized
    or retained in memory.
    """
    input, kernel_size, stride, padding, dilation, ceil_mode = inputs
    ctx.save_for_backward(input)
    ctx.kernel_size = kernel_size
    ctx.stride = stride
    ctx.padding = padding
    ctx.dilation = dilation
    ctx.ceil_mode = ceil_mode

  @staticmethod
  def backward(  # pytype: disable=bad-override  # pyrefly: ignore[bad-override]
      ctx: Any, *grad_outputs: Any
  ) -> tuple[Any, ...]:
    """Computes grad_input using SelectAndScatter from saved input tensor."""
    (grad_output,) = grad_outputs
    (input,) = ctx.saved_tensors
    grad_input = torch.ops.tpu._max_pool2d_backward(
        grad_output,
        input,
        ctx.kernel_size,
        ctx.stride,
        ctx.padding,
        ctx.dilation,
        ctx.ceil_mode,
    )
    # Return gradient for each forward input argument: input, kernel_size,
    # stride, padding, dilation, ceil_mode. Only input requires grad.
    return grad_input, None, None, None, None, None


class _TpuMaxPool3dFunction(torch.autograd.Function):
  """Differentiable TPU 3D max pooling using SelectAndScatter without indices.

  In the forward pass, computes the pooled output using
  `torch.ops.tpu._max_pool3d` without computing or materializing indices. In
  the backward pass, uses `torch.ops.tpu._max_pool3d_backward` to recompute the
  argmax positions on the fly via StableHLO `SelectAndScatterOp`.
  """

  @staticmethod
  def forward(
      input: torch.Tensor,
      kernel_size: list[int],
      stride: list[int],
      padding: list[int],
      dilation: list[int],
      ceil_mode: bool,
  ) -> torch.Tensor:
    """Executes native 3D max pooling without index materialization."""
    return torch.ops.tpu._max_pool3d(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )

  @staticmethod
  def setup_context(
      ctx: Any,
      inputs: tuple[Any, ...],
      output: Any,
  ) -> None:
    """Saves only the forward input tensor and pooling metadata for backward.

    Crucially, only the input tensor is saved; NO index tensor is materialized
    or retained in memory.
    """
    input, kernel_size, stride, padding, dilation, ceil_mode = inputs
    ctx.save_for_backward(input)
    ctx.kernel_size = kernel_size
    ctx.stride = stride
    ctx.padding = padding
    ctx.dilation = dilation
    ctx.ceil_mode = ceil_mode

  @staticmethod
  def backward(  # pytype: disable=bad-override  # pyrefly: ignore[bad-override]
      ctx: Any, *grad_outputs: Any
  ) -> tuple[Any, ...]:
    """Computes grad_input using SelectAndScatter from saved input tensor."""
    (grad_output,) = grad_outputs
    (input,) = ctx.saved_tensors
    grad_input = torch.ops.tpu._max_pool3d_backward(
        grad_output,
        input,
        ctx.kernel_size,
        ctx.stride,
        ctx.padding,
        ctx.dilation,
        ctx.ceil_mode,
    )
    # Return gradient for each forward input argument: input, kernel_size,
    # stride, padding, dilation, ceil_mode. Only input requires grad.
    return grad_input, None, None, None, None, None


# -----------------------------------------------------------------------------
# Functional Monkey Patches for torch.nn.functional
# -----------------------------------------------------------------------------


@functools.wraps(_ORIG_F_MAX_POOL1D)
def patched_max_pool1d(
    input: torch.Tensor,
    kernel_size: Any,
    stride: Any = None,
    padding: Any = 0,
    dilation: Any = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Any:
  """Optimized max_pool1d for TPU tensors avoiding unused index overhead.

  Routes TPU tensors to native TPU lowerings:
  - If `return_indices=True` or the tensor is not on a TPU device, falls back
    to original `F.max_pool1d`.
  - If dilation is non-trivial (dilation != 1), falls back to original
    `F.max_pool1d` as SelectAndScatter does not support dilated windows.
  - If `requires_grad=True` and grad is enabled, routes through
    `_TpuMaxPool1dFunction` for index-free autograd.
  - Otherwise (eval / inference), calls `torch.ops.tpu._max_pool1d` directly.
  """
  # Fall back to PyTorch core when caller explicitly requests indices or
  # when operating on non-TPU devices (e.g. CPU or CUDA tensors).
  if return_indices or not _is_tpu_tensor(input):
    return _ORIG_F_MAX_POOL1D(
        input,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )

  # Normalize parameters to lists of integers for 1D spatial dimension.
  k = list(_single(kernel_size))
  s = list(_single(stride)) if stride is not None else k
  p = list(_single(padding))
  d = list(_single(dilation))

  # StableHLO SelectAndScatterOp supports trivial dilation (all 1s). Fall back
  # to original PyTorch lowering if non-trivial dilation is specified.
  if any(dim != 1 for dim in d):
    return _ORIG_F_MAX_POOL1D(
        input,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )

  # In training mode with gradients enabled, route through the custom autograd
  # function so the backward graph can use SelectAndScatter without indices.
  if input.requires_grad and torch.is_grad_enabled():
    return _TpuMaxPool1dFunction.apply(input, k, s, p, d, bool(ceil_mode))

  # In eval or no-grad mode, call the TPU native operator directly.
  return torch.ops.tpu._max_pool1d(input, k, s, p, d, bool(ceil_mode))


@functools.wraps(_ORIG_F_MAX_POOL2D)
def patched_max_pool2d(
    input: torch.Tensor,
    kernel_size: Any,
    stride: Any = None,
    padding: Any = 0,
    dilation: Any = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Any:
  """Optimized max_pool2d for TPU tensors avoiding unused index overhead.

  Routes TPU tensors to native TPU lowerings:
  - If `return_indices=True` or the tensor is not on a TPU device, falls back
    to original `F.max_pool2d`.
  - If dilation is non-trivial (dilation != 1), falls back to original
    `F.max_pool2d` as SelectAndScatter does not support dilated windows.
  - If `requires_grad=True` and grad is enabled, routes through
    `_TpuMaxPool2dFunction` for index-free autograd.
  - Otherwise (eval / inference), calls `torch.ops.tpu._max_pool2d` directly.
  """
  # Fall back to PyTorch core when caller explicitly requests indices or
  # when operating on non-TPU devices (e.g. CPU or CUDA tensors).
  if return_indices or not _is_tpu_tensor(input):
    return _ORIG_F_MAX_POOL2D(
        input,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )

  # Normalize parameters to lists of integers for 2D spatial dimensions [H, W].
  k = list(_pair(kernel_size))
  s = list(_pair(stride)) if stride is not None else k
  p = list(_pair(padding))
  d = list(_pair(dilation))

  # StableHLO SelectAndScatterOp supports trivial dilation (all 1s). Fall back
  # to original PyTorch lowering if non-trivial dilation is specified.
  if any(dim != 1 for dim in d):
    return _ORIG_F_MAX_POOL2D(
        input,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )

  # In training mode with gradients enabled, route through the custom autograd
  # function so the backward graph can use SelectAndScatter without indices.
  if input.requires_grad and torch.is_grad_enabled():
    return _TpuMaxPool2dFunction.apply(input, k, s, p, d, bool(ceil_mode))

  # In eval or no-grad mode, call the TPU native operator directly.
  return torch.ops.tpu._max_pool2d(input, k, s, p, d, bool(ceil_mode))


@functools.wraps(_ORIG_F_MAX_POOL3D)
def patched_max_pool3d(
    input: torch.Tensor,
    kernel_size: Any,
    stride: Any = None,
    padding: Any = 0,
    dilation: Any = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Any:
  """Optimized max_pool3d for TPU tensors avoiding unused index overhead.

  Routes TPU tensors to native TPU lowerings:
  - If `return_indices=True` or the tensor is not on a TPU device, falls back
    to original `F.max_pool3d`.
  - If dilation is non-trivial (dilation != 1), falls back to original
    `F.max_pool3d` as SelectAndScatter does not support dilated windows.
  - If `requires_grad=True` and grad is enabled, routes through
    `_TpuMaxPool3dFunction` for index-free autograd.
  - Otherwise (eval / inference), calls `torch.ops.tpu._max_pool3d` directly.
  """
  # Fall back to PyTorch core when caller explicitly requests indices or
  # when operating on non-TPU devices (e.g. CPU or CUDA tensors).
  if return_indices or not _is_tpu_tensor(input):
    return _ORIG_F_MAX_POOL3D(
        input,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )

  # Normalize parameters to lists of integers for 3D spatial dimensions [D, H, W].
  k = list(_triple(kernel_size))
  s = list(_triple(stride)) if stride is not None else k
  p = list(_triple(padding))
  d = list(_triple(dilation))

  # StableHLO SelectAndScatterOp supports trivial dilation (all 1s). Fall back
  # to original PyTorch lowering if non-trivial dilation is specified.
  if any(dim != 1 for dim in d):
    return _ORIG_F_MAX_POOL3D(
        input,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )

  # In training mode with gradients enabled, route through the custom autograd
  # function so the backward graph can use SelectAndScatter without indices.
  if input.requires_grad and torch.is_grad_enabled():
    return _TpuMaxPool3dFunction.apply(input, k, s, p, d, bool(ceil_mode))

  # In eval or no-grad mode, call the TPU native operator directly.
  return torch.ops.tpu._max_pool3d(input, k, s, p, d, bool(ceil_mode))


# -----------------------------------------------------------------------------
# Patching and Unpatching Lifecycle Management
# -----------------------------------------------------------------------------


def patch_max_pool() -> None:
  """Patches PyTorch functional max pooling operations with TPU-optimized variants.

  Replaces `F.max_pool1d`, `F.max_pool2d`, and `F.max_pool3d` with TPU-aware
  alternatives. This operation is idempotent; multiple calls have no additional
  effect.
  """
  global _IS_PATCHED
  if _IS_PATCHED:
    return
  _IS_PATCHED = True
  F.max_pool1d = patched_max_pool1d
  F.max_pool2d = patched_max_pool2d
  F.max_pool3d = patched_max_pool3d


def unpatch_max_pool() -> None:
  """Restores original PyTorch functional max pooling operations.

  Reverts `F.max_pool1d`, `F.max_pool2d`, and `F.max_pool3d` to the native
  PyTorch implementations saved during module import. This operation is
  idempotent.
  """
  global _IS_PATCHED
  if not _IS_PATCHED:
    return
  _IS_PATCHED = False
  F.max_pool1d = _ORIG_F_MAX_POOL1D
  F.max_pool2d = _ORIG_F_MAX_POOL2D
  F.max_pool3d = _ORIG_F_MAX_POOL3D
