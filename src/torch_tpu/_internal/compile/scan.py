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

"""TPU implementation for torch.ops.higher_order.scan."""

import torch
import torch.export
import torch.utils._pytree as pytree
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.export import export


def _to_placeholder(t, shape=None):
  """Returns `t` as a like-shaped placeholder, or `t` itself if not a tensor.

  Args:
    t: A body operand.
    shape: Placeholder shape, defaulting to `t`'s own.
  """
  if not isinstance(t, torch.Tensor):
    return t
  if shape is None:
    shape = list(t.shape)
  return tpu_torch_compile.placeholder(shape, t.dtype, t.requires_grad)


def _handle_scan_impl(combine_fn, init, xs, additional_inputs, reverse=False):
  """TPU implementation for scan higher-order op.

  This implementation traces the combine_fn and lowers the scan to a deferred
  C++ op, which builds a `stablehlo.while` loop.

  Args:
    combine_fn: The combine function.
    init: The initial carry state.
    xs: The input sequence.
    additional_inputs: Additional inputs to the combine function.
    reverse: Whether to scan in reverse.

  Returns:
    A tuple of flat tensors representing the final carry and the stacked
    outputs of the scan operation.
  """
  # Normalize inputs to lists.
  init_list, _ = pytree.tree_flatten(init)
  xs_list, _ = pytree.tree_flatten(xs)
  extra_list, _ = pytree.tree_flatten(additional_inputs)

  # `fx_to_mlir`'s args become the body module's block arguments, which have to
  # be graph leaves, so every body operand has to be a placeholder.
  init_args = [_to_placeholder(t) for t in init_list]
  # The body sees one slice of each `xs` along the scan dimension.
  x_list = [_to_placeholder(t, list(t.shape)[1:]) for t in xs_list]
  extra_args = [_to_placeholder(t) for t in extra_list]

  # Resolve the actual callable body function.
  actual_combine_fn = combine_fn
  if isinstance(combine_fn, torch.export.ExportedProgram):
    actual_combine_fn = combine_fn.module

  # Perform a meta-run to infer the output shapes and dtypes.
  # pylint: disable=protected-access
  with torch._subclasses.fake_tensor.FakeTensorMode(allow_non_fake_inputs=True):

    def to_fake(tensors):
      return [
          tpu_torch_compile.placeholder_like(t)
          if isinstance(t, torch.Tensor)
          else t
          for t in tensors
      ]

    fake_inits = to_fake(init_args)
    fake_xs_heads = to_fake(x_list)
    fake_extras = to_fake(extra_args)

    fake_result = actual_combine_fn(*fake_inits, *fake_xs_heads, *fake_extras)
    fake_result_flat, _ = pytree.tree_flatten(fake_result)

    dummy_ys = []
    seq_len = xs_list[0].shape[0]
    num_carries = len(init_list)
    for y in fake_result_flat[num_carries:]:
      ys_shape = [seq_len] + list(y.shape)
      dummy_ys.append(torch.zeros(ys_shape, dtype=y.dtype, device="meta"))

  # Compile the body sub-graph to MLIR.
  body_args = init_args + x_list + extra_args
  body_mlir = export.fx_to_mlir(actual_combine_fn, args=body_args)

  # Call the C++ binding to create a deferred scan operation.
  scan_direction = (
      tpu_torch_compile.ScanDirection.kReverse
      if reverse
      else tpu_torch_compile.ScanDirection.kForward
  )
  num_scan_inputs = len(xs_list)
  inputs = xs_list + extra_list
  return tpu_torch_compile.create_scan_op(
      inits=init_list,
      inputs=inputs,
      body_module=body_mlir.module,
      scan_direction=scan_direction,
      dummy_ys=dummy_ys,
      num_scan_inputs=num_scan_inputs,
  )


try:
  # pylint: disable=g-import-not-at-top
  # pylint: disable=protected-access
  import torch._C

  for key in (
      torch._C.DispatchKey.XLA,
      torch._C.DispatchKey.PrivateUse1,
      torch._C.DispatchKey.Meta,
  ):
    torch.ops.higher_order.scan.py_impl(key)(_handle_scan_impl)
except (AttributeError, RuntimeError):
  # AttributeError: Safe to ignore if the environment lacks XLA/TPU support.
  # RuntimeError: Safe to ignore if the implementation is already registered.
  pass
