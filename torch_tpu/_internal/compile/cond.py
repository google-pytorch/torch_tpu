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

"""TPU implementation for torch.ops.higher_order.cond."""

import torch
import torch.export
import torch.utils._pytree as pytree
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.export import export


def _actual_callable(fn):
  if isinstance(fn, torch.export.ExportedProgram):
    return fn.module
  return fn


def _num_graph_placeholders(fn):
  graph = getattr(fn, "graph", None)
  if graph is None:
    return None
  return sum(1 for node in graph.nodes if node.op == "placeholder")


def _branch_args(fn, flat_args, operands_spec):
  """Returns args using the argument convention the branch was traced with."""
  expected_args = _num_graph_placeholders(fn)
  if expected_args == len(flat_args):
    return tuple(flat_args)
  operands = pytree.tree_unflatten(flat_args, operands_spec)
  return tuple(operands)


def _call_branch(fn, flat_args, operands_spec):
  return fn(*_branch_args(fn, flat_args, operands_spec))


def _validate_pred(pred):
  if not isinstance(pred, torch.Tensor):
    return
  if pred.dtype != torch.bool:
    raise RuntimeError(
        "torch.cond TPU predicate must be a bool tensor, "
        f"but got dtype {pred.dtype}."
    )
  if pred.numel() != 1:
    raise RuntimeError(
        "torch.cond TPU predicate must contain exactly one element, "
        f"but got shape {tuple(pred.shape)}."
    )


def _validate_branch_outputs(true_flat, false_flat, true_spec, false_spec):
  if true_spec != false_spec:
    raise RuntimeError(
        "torch.cond branches must return the same pytree structure: "
        f"true branch tree_spec {true_spec} vs false branch tree_spec "
        f"{false_spec}."
    )
  if len(true_flat) != len(false_flat):
    raise RuntimeError(
        "torch.cond branches must return the same number of leaves: "
        f"true branch returned {len(true_flat)} leaves, false branch returned "
        f"{len(false_flat)} leaves."
    )
  if not true_flat:
    raise RuntimeError("torch.cond TPU branches must return at least one tensor.")

  for idx, (true_out, false_out) in enumerate(zip(true_flat, false_flat)):
    if not isinstance(true_out, torch.Tensor) or not isinstance(
        false_out, torch.Tensor
    ):
      raise RuntimeError(
          "torch.cond TPU branch outputs must be tensors. "
          f"Output leaf {idx} has true type {type(true_out).__name__} and "
          f"false type {type(false_out).__name__}."
      )
    if true_out.dtype != false_out.dtype:
      raise RuntimeError(
          f"torch.cond branch output {idx} dtype mismatch: true branch has "
          f"{true_out.dtype}, false branch has {false_out.dtype}."
      )
    if tuple(true_out.shape) != tuple(false_out.shape):
      raise RuntimeError(
          f"torch.cond branch output {idx} shape mismatch: true branch has "
          f"{tuple(true_out.shape)}, false branch has {tuple(false_out.shape)}."
      )


def _validate_operand_leaves(flat_operands):
  for idx, operand in enumerate(flat_operands):
    if not isinstance(operand, torch.Tensor):
      raise RuntimeError(
          "torch.cond TPU operand leaves must be tensors. "
          f"Operand leaf {idx} has type {type(operand).__name__}."
      )


def _make_fake_args(flat_args):
  return [
      tpu_torch_compile.placeholder_like(arg)
      if isinstance(arg, torch.Tensor)
      else arg
      for arg in flat_args
  ]


def _handle_cond_impl(pred, true_fn, false_fn, operands):
  """TPU implementation for the cond higher-order op."""
  if not isinstance(operands, (tuple, list)):
    raise RuntimeError(
        "torch.cond TPU operands must be a tuple or list, "
        f"but got {type(operands).__name__}."
    )

  _validate_pred(pred)
  actual_true_fn = _actual_callable(true_fn)
  actual_false_fn = _actual_callable(false_fn)

  flat_operands, operands_spec = pytree.tree_flatten(operands)
  tensor_operands = [
      operand for operand in flat_operands if isinstance(operand, torch.Tensor)
  ]

  if not isinstance(pred, torch.Tensor):
    if not tensor_operands:
      return _call_branch(
          actual_true_fn if pred else actual_false_fn,
          flat_operands,
          operands_spec,
      )
    pred = torch.tensor(
        bool(pred), dtype=torch.bool, device=tensor_operands[0].device
    )

  _validate_operand_leaves(flat_operands)

  # Perform a meta-run to validate branch output structure and infer output
  # shapes/dtypes without executing real TPU work.
  # pylint: disable=protected-access
  with torch._subclasses.fake_tensor.FakeTensorMode(allow_non_fake_inputs=True):
    fake_flat_operands = _make_fake_args(flat_operands)
    true_result = _call_branch(
        actual_true_fn, fake_flat_operands, operands_spec
    )
    false_result = _call_branch(
        actual_false_fn, fake_flat_operands, operands_spec
    )

  true_flat, true_spec = pytree.tree_flatten(true_result)
  false_flat, false_spec = pytree.tree_flatten(false_result)
  _validate_branch_outputs(true_flat, false_flat, true_spec, false_spec)

  dummy_outputs = [
      torch.zeros(tuple(out.shape), dtype=out.dtype, device="meta")
      for out in true_flat
  ]

  true_args = _branch_args(actual_true_fn, flat_operands, operands_spec)
  false_args = _branch_args(actual_false_fn, flat_operands, operands_spec)
  true_mlir = export.fx_to_mlir(actual_true_fn, args=true_args)
  false_mlir = export.fx_to_mlir(actual_false_fn, args=false_args)

  flat_results = tpu_torch_compile.create_cond_op(
      pred=pred,
      operands=tensor_operands,
      true_module=true_mlir.module,
      false_module=false_mlir.module,
      dummy_outputs=dummy_outputs,
  )
  return pytree.tree_unflatten(flat_results, true_spec)


try:
  # pylint: disable=g-import-not-at-top
  # pylint: disable=protected-access
  import torch._C

  for key in (
      torch._C.DispatchKey.XLA,
      torch._C.DispatchKey.PrivateUse1,
      torch._C.DispatchKey.Meta,
  ):
    torch.ops.higher_order.cond.py_impl(key)(_handle_cond_impl)
except (AttributeError, RuntimeError):
  # AttributeError: Safe to ignore if the environment lacks XLA/TPU support.
  # RuntimeError: Safe to ignore if the implementation is already registered.
  pass
