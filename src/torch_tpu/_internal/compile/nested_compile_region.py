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

from collections.abc import Sequence
import ctypes
import dataclasses
import hashlib
from typing import Any
import torch
import torch.utils._pytree as pytree
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.fx_passes import mark_embedded_constants
from torch_tpu._internal.export import export
from torch_tpu._internal.pallas import tpu_torch_pallas


@dataclasses.dataclass(frozen=True)
class _CachedNestedRegion:
  """Cached metadata and fingerprint for a compiled nested region."""

  kernel_name: str
  mlir_fingerprint: str
  output_shapes: Any
  reconstruct_fn: Any

  def __repr__(self) -> str:
    return (
        f"_CachedNestedRegion(kernel_name={self.kernel_name!r},"
        f" mlir_fingerprint={self.mlir_fingerprint!r})"
    )


_NESTED_REGION_COMPILE_CACHE: dict[str, _CachedNestedRegion] = {}

__all__ = [
    "clear_cache",
    "handle_invoke_subgraph",
]


def clear_cache() -> None:
  """Clears the nested compile region compilation cache."""
  _NESTED_REGION_COMPILE_CACHE.clear()


def _hash_tensor_bytes(hasher: Any, tensor: torch.Tensor) -> None:
  """Hashes raw tensor bytes using a zero-copy ctypes buffer over CPU memory."""
  t_cpu = tensor.detach().cpu().contiguous()
  nbytes = t_cpu.numel() * t_cpu.element_size()
  if nbytes > 0:
    buf = (ctypes.c_char * nbytes).from_address(t_cpu.data_ptr())
    hasher.update(buf)


def _get_target_attr(module: torch.nn.Module, target: str) -> Any:
  """Safely retrieves a potentially nested attribute target from an FX module."""
  try:
    return torch.fx.graph_module._get_attr(module, target)
  except (AttributeError, AssertionError):
    return None


def _compute_cache_key(
    subgraph: torch.fx.GraphModule,
    identifier: str,
    bound_operands: Sequence[Any],
) -> str | None:
  """Computes a robust cache key including subgraph code, constants, and bound operands."""
  if not hasattr(subgraph, "code"):
    return None

  hasher = hashlib.blake2b(digest_size=16)
  hasher.update(str(identifier).encode("utf-8"))
  hasher.update(subgraph.code.encode("utf-8"))

  # Include constants, parameters, and buffers
  for name, param in subgraph.named_parameters():
    hasher.update(f"param_{name}_{param.shape}_{param.dtype}".encode("utf-8"))
    if param.device.type == "cpu" and param.numel() <= 1024:
      _hash_tensor_bytes(hasher, param)
    else:
      hasher.update(
          f"dev_{param.device}_{param.data_ptr()}_{param.shape}_{param.dtype}"
          .encode("utf-8")
      )
  for name, buf in subgraph.named_buffers():
    hasher.update(f"buf_{name}_{buf.shape}_{buf.dtype}".encode("utf-8"))
    if buf.device.type == "cpu" and buf.numel() <= 1024:
      _hash_tensor_bytes(hasher, buf)
    else:
      hasher.update(
          f"dev_{buf.device}_{buf.data_ptr()}_{buf.shape}_{buf.dtype}".encode(
              "utf-8"
          )
      )
  for node in subgraph.graph.nodes:
    if node.op == "get_attr":
      val = _get_target_attr(subgraph, str(node.target))
      if isinstance(val, torch.Tensor):
        hasher.update(
            f"attr_{node.target}_{val.shape}_{val.dtype}".encode("utf-8")
        )
        if val.device.type == "cpu" and val.numel() <= 1024:
          _hash_tensor_bytes(hasher, val)
        else:
          hasher.update(
              f"dev_{val.device}_{val.data_ptr()}_{val.shape}_{val.dtype}"
              .encode("utf-8")
          )

  # Include all bound operands (already ordered and defaults applied)
  flat_items, _ = pytree.tree_flatten(bound_operands)
  for item in flat_items:
    if isinstance(item, torch.Tensor):
      hasher.update(
          f"tensor_{item.shape}_{item.dtype}_{item.stride()}_{item.requires_grad}"
          .encode("utf-8")
      )
    else:
      hasher.update(f"prim_{type(item)}_{item}".encode("utf-8"))

  return hasher.hexdigest()


def handle_invoke_subgraph(
    subgraph: Any,
    identifier: str | None,
    *operands: Any,
    **kwargs: Any,
) -> Any:
  """Handles torch.ops.higher_order.invoke_subgraph during FX graph translation to MLIR."""
  if not isinstance(subgraph, torch.fx.GraphModule):
    raise TypeError(f"Expected GraphModule, got {type(subgraph)}")

  identifier = identifier or "subgraph"

  # Bind operands and kwargs to placeholder arguments
  placeholders = [n for n in subgraph.graph.nodes if n.op == "placeholder"]
  bound_operands = list(operands)
  for p in placeholders[len(bound_operands) :]:
    if p.target in kwargs:
      bound_operands.append(kwargs[p.target])
    elif len(p.args) > 0:
      bound_operands.append(p.args[0])
    else:
      raise TypeError(f"Missing required argument: {p.target}")

  flat_bound, _ = pytree.tree_flatten(bound_operands)
  tensor_operands = [x for x in flat_bound if isinstance(x, torch.Tensor)]

  # Ensure any lifted constants in the subgraph are converted to constant ops
  # BEFORE computing the cache key so the graph is in its canonical state.
  mark_embedded_constants.apply(subgraph)

  # Check compile cache
  cache_key = _compute_cache_key(subgraph, identifier, bound_operands)

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
        for op in bound_operands
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
  return reconstruct_fn(bound_operands, results)

