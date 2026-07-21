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

"""In-graph custom call lowering for torch.compiler.nested_compile_region."""

import dataclasses
import hashlib
from typing import Any
import torch
import torch.utils._pytree as pytree
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.export import export
from torch_tpu._internal.pallas import tpu_torch_pallas


@dataclasses.dataclass(frozen=True)
class _CachedNestedRegion:
  """Cached metadata and fingerprint for a compiled nested region."""

  kernel_name: str
  mlir_fingerprint: str
  output_shapes: Any
  reconstruct_fn: Any


_NESTED_REGION_COMPILE_CACHE: dict[str, _CachedNestedRegion] = {}


def handle_invoke_subgraph(
    interpreter: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> Any:
  """Handles torch.ops.higher_order.invoke_subgraph during FX graph translation to MLIR."""
  if not args:
    raise ValueError(
        "invoke_subgraph expects at least 2 arguments (subgraph, identifier)"
    )

  kwargs = kwargs or {}
  subgraph_arg = args[0]
  if isinstance(subgraph_arg, str):
    if interpreter is not None:
      subgraph = interpreter.fetch_attr(subgraph_arg)
    else:
      raise ValueError(
          "Cannot resolve string subgraph name without interpreter:"
          f" {subgraph_arg}"
      )
  else:
    subgraph = subgraph_arg

  identifier = args[1] if len(args) > 1 else "subgraph"
  operands = list(args[2:])

  if not isinstance(subgraph, torch.fx.GraphModule):
    raise TypeError(f"Expected GraphModule, got {type(subgraph)}")

  # Extract tensor operands
  flat_operands, _ = pytree.tree_flatten((operands, kwargs))
  tensor_operands = [x for x in flat_operands if isinstance(x, torch.Tensor)]

  # Check compile cache
  cache_key = None
  if hasattr(subgraph, "code"):
    hasher = hashlib.blake2b(subgraph.code.encode("utf-8"), digest_size=16)
    for op in tensor_operands:
      hasher.update(f"{op.shape}_{op.dtype}_{op.stride()}".encode("utf-8"))
    cache_key = hasher.hexdigest()

  if cache_key and cache_key in _NESTED_REGION_COMPILE_CACHE:
    cached = _NESTED_REGION_COMPILE_CACHE[cache_key]
    kernel_name = cached.kernel_name
    mlir_fingerprint = cached.mlir_fingerprint
    output_shapes = cached.output_shapes
    reconstruct_fn = cached.reconstruct_fn
  else:
    # Construct clean placeholder sample arguments for subgraph compilation
    subgraph_sample_args = [
        tpu_torch_compile.placeholder(
            list(op.shape), op.dtype, op.requires_grad
        )
        if isinstance(op, torch.Tensor)
        else op
        for op in operands
    ]
    # Recursively compile subgraph to MLIR
    subgraph_exported_mlir = export.fx_to_mlir(
        subgraph, args=subgraph_sample_args
    )
    mlir_text = subgraph_exported_mlir.serialize_text()

    mlir_fingerprint = hashlib.blake2b(
        mlir_text.encode("utf-8"), digest_size=16
    ).hexdigest()
    kernel_name = f"nested_region_{identifier or 'subgraph'}"

    # Register kernel with the custom kernel registry (tpu_torch_pallas pybind binding)
    tpu_torch_pallas.register_custom_kernel(
        kernel_name,
        mlir_fingerprint,
        serialized_mlir_module=mlir_text,
    )

    output_shapes = subgraph_exported_mlir.mlir_result_tensors
    reconstruct_fn = subgraph_exported_mlir.reconstruct_fx_outputs_fn

    if cache_key:
      _NESTED_REGION_COMPILE_CACHE[cache_key] = _CachedNestedRegion(
          kernel_name=kernel_name,
          mlir_fingerprint=mlir_fingerprint,
          output_shapes=output_shapes,
          reconstruct_fn=reconstruct_fn,
      )

  # Call custom kernel via CustomKernelRegistry
  results = tpu_torch_pallas.call_custom_kernel(
      kernel_name,
      mlir_fingerprint,
      inputs=tensor_operands,
      output_shapes=output_shapes,
  )

  # Reconstruct the original output structure
  return reconstruct_fn(operands, results)


def _handle_invoke_subgraph_dispatch(
    subgraph: Any,
    identifier: str | None,
    *operands: Any,
    **kwargs: Any,
) -> Any:
  return handle_invoke_subgraph(None, (subgraph, identifier, *operands), kwargs)


try:
  import torch._C  # pylint: disable=g-import-not-at-top

  for key in (
      torch._C.DispatchKey.XLA,
      torch._C.DispatchKey.PrivateUse1,
      torch._C.DispatchKey.Meta,
  ):
    if hasattr(torch.ops, "higher_order") and hasattr(
        torch.ops.higher_order, "invoke_subgraph"
    ):
      torch.ops.higher_order.invoke_subgraph.py_impl(key)(
          _handle_invoke_subgraph_dispatch
      )
except (AttributeError, RuntimeError):
  pass
