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

"""TPU implementation for torch.ops.higher_order.invoke_subgraph."""

import threading

import torch
import torch.export
import torch.utils._pytree as pytree
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.export import export

try:
  from torch._higher_order_ops.utils import FunctionalizeCtxWrapper
except ImportError:
  FunctionalizeCtxWrapper = None


_BODY_MODULE_CACHE = {}
_BODY_MODULE_CACHE_LOCK = threading.Lock()


def _unwrap_subgraph(subgraph):
  if FunctionalizeCtxWrapper is not None and isinstance(
      subgraph, FunctionalizeCtxWrapper
  ):
    return subgraph.subgraph
  if isinstance(subgraph, torch.export.ExportedProgram):
    return subgraph.module()
  return subgraph


def _handle_invoke_subgraph_impl(subgraph, identifier, *operands):
  """Lowers an invoke_subgraph body to a late-compiled StableHLO call."""
  if not all(isinstance(operand, torch.Tensor) for operand in operands):
    raise NotImplementedError(
        "TorchTPU invoke_subgraph currently requires tensor-only operands"
    )

  actual_subgraph = _unwrap_subgraph(subgraph)
  if getattr(actual_subgraph, "_boxed_call", False):
    raise NotImplementedError(
        "TorchTPU invoke_subgraph does not yet support boxed subgraphs"
    )

  with torch._subclasses.fake_tensor.FakeTensorMode(
      allow_non_fake_inputs=True
  ):
    fake_operands = [
        tpu_torch_compile.placeholder_like(operand) for operand in operands
    ]
    fake_result = actual_subgraph(*fake_operands)
    fake_outputs, output_spec = pytree.tree_flatten(fake_result)

  if not fake_outputs or not all(
      isinstance(output, torch.Tensor) for output in fake_outputs
  ):
    raise NotImplementedError(
      "TorchTPU invoke_subgraph currently requires tensor-only outputs"
    )

  body_args = [
      tpu_torch_compile.placeholder_like(operand) for operand in operands
  ]

  body_module = None
  if identifier:
    with _BODY_MODULE_CACHE_LOCK:
      body_module = _BODY_MODULE_CACHE.get(identifier)
      if body_module is None:
        body_module = export.fx_to_mlir(
            actual_subgraph, args=body_args
        ).module
        _BODY_MODULE_CACHE[identifier] = body_module
  else:
    body_module = export.fx_to_mlir(actual_subgraph, args=body_args).module
  flat_outputs = tpu_torch_compile.create_invoke_subgraph_op(
      inputs=list(operands),
      body_module=body_module,
      dummy_outputs=fake_outputs,
      identifier=identifier or "invoke_subgraph",
  )
  return pytree.tree_unflatten(flat_outputs, output_spec)


try:
  # pylint: disable=protected-access
  import torch._C

  for key in (
      torch._C.DispatchKey.XLA,
      torch._C.DispatchKey.PrivateUse1,
  ):
    torch.ops.higher_order.invoke_subgraph.py_impl(key)(
        _handle_invoke_subgraph_impl
    )
except (AttributeError, RuntimeError):
  pass
