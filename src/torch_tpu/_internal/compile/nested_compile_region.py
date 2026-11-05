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

"""In-graph invoke_subgraph lowering for torch.compiler.nested_compile_region."""

from collections.abc import Callable, Sequence
import dataclasses
import hashlib
from typing import Any
import torch
import torch.utils._pytree as pytree
from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.fx_passes import mark_embedded_constants
from torch_tpu._internal.export import export


@dataclasses.dataclass(frozen=True)
class _CachedNestedRegion:
  """Cached exported MLIR module and output reconstructor for a nested region."""

  module: tpu_torch_compile.ContextedModule | None
  reconstruct_fn: Callable[[Sequence[Any], Sequence[torch.Tensor]], Any] = (
      dataclasses.field(repr=False)
  )
  updates_default_generator_state: bool = False


_NESTED_REGION_COMPILE_CACHE: dict[str, _CachedNestedRegion] = {}

__all__ = [
    "clear_cache",
    "handle_invoke_subgraph",
]


def clear_cache() -> None:
  """Clears the nested compile region cache."""
  _NESTED_REGION_COMPILE_CACHE.clear()


def _update_str(hasher: Any, value: str) -> None:
  encoded = value.encode("utf-8")
  hasher.update(f"{len(encoded)}:".encode("utf-8"))
  hasher.update(encoded)


def _get_target_attr(module: torch.nn.Module, target: str) -> Any:
  """Safely retrieves a potentially nested attribute target from an FX module."""
  try:
    return torch.fx.graph_module._get_attr(module, target)
  except (AttributeError, AssertionError):
    return None


def _hash_module_tensor(
    hasher: Any, kind: str, name: str, val: torch.Tensor
) -> None:
  fp = tpu_torch_compile.constant_tensor_fingerprint(val)
  if fp is not None:
    _update_str(
        hasher,
        f"{kind}:{name}:const:{fp}:{tuple(val.shape)}:{val.dtype}",
    )
    return
  is_pinned = compiler.is_pinned_host_tensor(val)
  _update_str(
      hasher,
      f"{kind}:{name}:dev:{val.device}:{val.data_ptr()}:{tuple(val.shape)}:"
      f"{val.dtype}:{is_pinned}",
  )


def _get_default_generator() -> tuple[torch.Generator, int]:
  device = torch.accelerator.current_accelerator()
  device_module = getattr(torch, device.type)  # pyrefly: ignore[missing-attribute]
  default_device_index = device.index or 0  # pyrefly: ignore[missing-attribute]
  return (
      device_module.default_generators[default_device_index],
      default_device_index,
  )


def _compute_cache_key(
    subgraph: torch.fx.GraphModule,
    identifier: str,
    flat_bound: Sequence[Any],
    spec: pytree.TreeSpec,
) -> str:
  """Computes a robust cache key including subgraph code, constants, and bound operands."""
  hasher = hashlib.blake2b(digest_size=16)
  _update_str(hasher, identifier)
  _update_str(hasher, subgraph.code)

  for name, param in subgraph.named_parameters():
    _hash_module_tensor(hasher, "param", name, param)
  for name, buf in subgraph.named_buffers():
    _hash_module_tensor(hasher, "buf", name, buf)
  for node in subgraph.graph.nodes:
    if node.op == "get_attr":
      val = _get_target_attr(subgraph, str(node.target))
      if isinstance(val, torch.Tensor):
        _hash_module_tensor(hasher, "attr", str(node.target), val)

  _update_str(hasher, f"spec:{spec!r}")
  for item in flat_bound:
    if isinstance(item, torch.Tensor):
      is_pinned = compiler.is_pinned_host_tensor(item)
      _update_str(
          hasher,
          f"tensor:{item.device}:{tuple(item.shape)}:{item.dtype}:"
          f"{item.requires_grad}:{is_pinned}",
      )
    elif isinstance(item, torch.Generator):
      _update_str(hasher, f"gen:{item.device}")
    else:
      _update_str(hasher, f"prim:{type(item).__qualname__}:{item!r}")

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

  flat_bound, spec = pytree.tree_flatten(bound_operands)
  tensor_operands = [x for x in flat_bound if isinstance(x, torch.Tensor)]
  explicit_generators = [
      x for x in flat_bound if isinstance(x, torch.Generator)
  ]

  # Ensure any lifted constants in the subgraph are converted to constant ops
  # BEFORE computing the cache key so the graph is in its canonical state.
  mark_embedded_constants.apply(subgraph)
  cache_key = _compute_cache_key(subgraph, identifier, flat_bound, spec)

  if (cached := _NESTED_REGION_COMPILE_CACHE.get(cache_key)) is not None:
    subgraph_module = cached.module
    reconstruct_fn = cached.reconstruct_fn
    updates_default_generator_state = cached.updates_default_generator_state
  else:

    def _make_sample_arg(op: Any) -> Any:
      if not isinstance(op, torch.Tensor):
        return op
      return tpu_torch_compile.placeholder(
          list(op.shape),
          op.dtype,
          op.requires_grad,
          is_pinned_host=compiler.is_pinned_host_tensor(op),
      )

    subgraph_sample_args = pytree.tree_map(_make_sample_arg, bound_operands)

    # export.fx_to_mlir clears global export-mode default generator overrides in
    # its finally block; preserve the outer override across nested exports.
    in_export_mode = (
        execution_mode.eager_mode
        == execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH  # pylint: disable=protected-access
    )
    saved_default_gen = (
        _get_default_generator()
        if in_export_mode and torch.accelerator.is_available()
        else None
    )
    try:
      subgraph_exported_mlir = export.fx_to_mlir(
          subgraph, args=subgraph_sample_args
      )
    finally:
      if saved_default_gen is not None:
        tpu_torch_compile.set_export_mode_default_device_generator(
            *saved_default_gen
        )

    subgraph_module = subgraph_exported_mlir.module
    reconstruct_fn = subgraph_exported_mlir.reconstruct_fx_outputs_fn
    updates_default_generator_state = (
        subgraph_exported_mlir.updates_default_generator_state
    )

    _NESTED_REGION_COMPILE_CACHE[cache_key] = _CachedNestedRegion(
        module=subgraph_module,
        reconstruct_fn=reconstruct_fn,
        updates_default_generator_state=updates_default_generator_state,
    )

  if subgraph_module is None:
    return reconstruct_fn(bound_operands, [])

  generators = list(explicit_generators)
  if updates_default_generator_state:
    default_gen, _ = _get_default_generator()
    if default_gen not in generators:
      generators.append(default_gen)

  rng_inputs = [
      tpu_torch_compile.get_device_state_tensor(g) for g in generators
  ]
  results = tpu_torch_compile.create_invoke_subgraph_op(
      inputs=tensor_operands + rng_inputs,
      subgraph_module=subgraph_module,
      identifier=identifier,
  )
  if generators:
    num_gens = len(generators)
    updated_rng_states = results[-num_gens:]
    results = results[:-num_gens]
    for gen, state in zip(generators, updated_rng_states, strict=True):
      tpu_torch_compile.set_device_state_tensor(gen, state)

  return reconstruct_fn(bound_operands, results)


for _key in (
    torch._C.DispatchKey.XLA,
    torch._C.DispatchKey.PrivateUse1,
):
  torch.ops.higher_order.invoke_subgraph.py_impl(_key)(handle_invoke_subgraph)
