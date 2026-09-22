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
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.compile import compiler
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
  return tpu_torch_compile.placeholder(
      shape,
      t.dtype,
      t.requires_grad,
      is_pinned_host=compiler.is_pinned_host_tensor(t),
  )


def _pin_offloaded_body_args(combine_fn, body_args):
  """Returns `body_args` with the offloaded operands marked as host memory.

  The scan body is compiled as a standalone program, so every operand is
  allocated HBM. That includes activations already offloaded to host, which can
  be large enough to exhaust HBM on their own. Marking them as host memory keeps
  them out of the HBM total.

  An operand counts as offloaded if `ao.reload` copies it back from the host, or
  if the offload pass tagged its placeholder `saved_for_offloading`.

  `_backend._mark_pinned_host_inputs` does the same for the top-level graph, but
  does not see this compile.

  Args:
    combine_fn: The body function. Only a `GraphModule` can be inspected; any
      other callable is returned unchanged.
    body_args: The body operands, in the same order as the body placeholders.
  """
  if not isinstance(combine_fn, torch.fx.GraphModule):
    return body_args

  # pylint: disable=protected-access
  reloaded = set()
  for node in combine_fn.graph.nodes:
    if _backend._is_ao_op(node, "reload"):
      source = _backend._get_reload_source_placeholder(
          _backend._extract_tensor_node(node)
      )
      if source is not None:
        reloaded.add(source)
  # pylint: enable=protected-access

  placeholders = [n for n in combine_fn.graph.nodes if n.op == "placeholder"]
  pinned = list(body_args)
  for i, (node, arg) in enumerate(zip(placeholders, body_args, strict=True)):
    if (
        isinstance(arg, torch.Tensor)
        and not compiler.is_pinned_host_tensor(arg)
        and (node in reloaded or node.meta.get("saved_for_offloading", False))
    ):
      pinned[i] = tpu_torch_compile.placeholder_like(arg, is_pinned_host=True)
  return pinned


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

  # Compile the body sub-graph to MLIR.
  body_args = init_args + x_list + extra_args
  body_args = _pin_offloaded_body_args(actual_combine_fn, body_args)
  body_mlir = export.fx_to_mlir(actual_combine_fn, args=body_args)

  # Infer dummy_ys directly from MLIR result tensors
  seq_len = xs_list[0].shape[0] if xs_list else 1
  num_carries = len(init_list)
  dummy_ys = []
  for out_t in body_mlir.mlir_result_tensors[num_carries:]:
    ys_shape = [seq_len] + list(out_t.shape)
    dummy_ys.append(torch.zeros(ys_shape, dtype=out_t.dtype, device="meta"))

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
