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

"""Per-call buffer aliasing for ``inplace_donate=True`` pallas kernels.

XLA treats the pallas ``stablehlo.custom_call`` as a black box and
allocates a fresh output buffer per call. ``inject_inplace_donate_aliases``
post-processes the exported module: for each donated ``@main`` arg it walks
the in-place chain forward and adds ``output_operand_aliases`` to every
pallas custom_call and ``dynamic_update_slice`` (the slice→pallas→DUS
pattern) along the way, so buffer assignment reuses the donated buffer.
No-copy view ops and ``func.call`` are followed transparently.

Caveat: the StableHLO→HLO converter currently honors
``output_operand_aliases`` only on custom_calls; on a DUS the attribute is
structurally accurate but discardable, and the executable-level donation
plus BufferAssignment dataflow carry the DUS chain instead.
"""

from collections.abc import Sequence

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo

# JAX records donate_argnums (post static_argnums filtering) as this attr on
# @main args; the value is the aliased output index.
_TF_ALIASING_OUTPUT_ATTR = "tf.aliasing_output"

_PALLAS_CUSTOM_CALL_TARGET = "tpu_custom_call"
_OUTPUT_OPERAND_ALIASES_ATTR = "output_operand_aliases"
_CUSTOM_CALL_OP = "stablehlo.custom_call"
_DUS_OP = "stablehlo.dynamic_update_slice"
_FUNC_CALL_OP = "func.call"

# No-copy view ops the chain walker treats as transparent (same underlying
# buffer). broadcast_in_dim appears in chains only in its degenerate,
# typecast-like form.
_VIEW_OPS = frozenset(
    [
        "stablehlo.reshape",
        "stablehlo.transpose",
        "stablehlo.broadcast_in_dim",
        "stablehlo.bitcast_convert_type",
    ]
)


def _find_main_function(module: ir.Module) -> ir.Operation:
  for op in module.operation.regions[0].blocks[0].operations:
    sym_name = op.attributes.get("sym_name", None)
    if sym_name is not None and sym_name.value == "main":
      return op
  raise RuntimeError(
      "pallas inplace_donate alias injection: exported MLIR has no @main "
      "function."
  )


def _find_function_by_name(
    module: ir.Module, name: str,
) -> ir.Operation | None:
  for op in module.operation.regions[0].blocks[0].operations:
    sym_name = op.attributes.get("sym_name", None)
    if sym_name is not None and sym_name.value == name:
      return op
  return None


def _read_jit_alias_pairs(
    main_func: ir.Operation,
) -> list[tuple[int, int]]:
  """Returns the [(arg_idx, output_idx), ...] JIT donation pairs."""
  arg_attrs = main_func.attributes.get("arg_attrs", None)
  if arg_attrs is None:
    return []
  pairs: list[tuple[int, int]] = []
  for arg_idx, dict_attr in enumerate(arg_attrs):
    for entry in dict_attr:
      if entry.name == _TF_ALIASING_OUTPUT_ATTR:
        pairs.append((arg_idx, int(entry.attr.value)))
        break
  return pairs


def _is_pallas_custom_call(op: ir.OpView) -> bool:
  if op.operation.name != _CUSTOM_CALL_OP:
    return False
  target = op.attributes.get("call_target_name", None)
  if target is None:
    return False
  return target.value == _PALLAS_CUSTOM_CALL_TARGET


def _build_output_operand_aliases(
    context: ir.Context,
    operand_to_output: Sequence[tuple[int, int]],
    num_results: int,
) -> ir.ArrayAttr:
  is_multi_result = num_results > 1
  with ir.Location.unknown(context):
    entries = []
    for operand_idx, output_idx in operand_to_output:
      output_tuple_indices = [output_idx] if is_multi_result else []
      entry = stablehlo.OutputOperandAlias.get(
          output_tuple_indices=output_tuple_indices,
          operand_index=operand_idx,
          operand_tuple_indices=[],
      )
      entries.append(entry)
    return ir.ArrayAttr.get(entries)


def _resolve_pallas_output_index(
    op: ir.OpView, operand_idx: int,
) -> int | None:
  """Result index aliased to ``operand_idx``: the op's own
  ``output_operand_aliases`` when present, else identity; None if out of
  range (lets the walker fall through to another consumer)."""
  aliases_attr = op.attributes.get(_OUTPUT_OPERAND_ALIASES_ATTR, None)
  if aliases_attr is not None:
    for entry in aliases_attr:
      typed = stablehlo.OutputOperandAlias(entry)
      if int(typed.operand_index) != operand_idx:
        continue
      indices = list(typed.output_tuple_indices)
      candidate = int(indices[0]) if indices else 0
      if candidate < len(op.results):
        return candidate
      return None
  if operand_idx < len(op.results):
    return operand_idx
  return None


def _resolve_func_call_into_callee(
    use: "ir.OpOperand",
    module: ir.Module,
) -> ir.Value | None:
  """Follows a func.call use into the callee's matching block arg."""
  op = use.owner
  if op.operation.name != _FUNC_CALL_OP:
    return None
  callee_attr = op.attributes.get("callee", None)
  if callee_attr is None:
    return None
  callee_name = callee_attr.value
  callee = _find_function_by_name(module, callee_name)
  if callee is None:
    return None
  callee_block = callee.regions[0].blocks[0]
  if use.operand_number >= len(callee_block.arguments):
    return None
  return callee_block.arguments[use.operand_number]


def _first_chain_step(
    arg_value: ir.Value,
    module: ir.Module,
    visited_block_args: set[int] | None = None,
) -> tuple[ir.OpView, int, int] | None:
  """First chain step from a donated arg: may match at any operand
  position. Prefers a pallas custom_call, then a DUS at operand 0, then
  recurses through view ops and func.calls. Returns
  (op, operand_idx, output_idx) or None."""
  if visited_block_args is None:
    visited_block_args = set()
  for use in arg_value.uses:
    op = use.owner
    if not _is_pallas_custom_call(op):
      continue
    output_idx = _resolve_pallas_output_index(op, use.operand_number)
    if output_idx is None:
      continue
    return op, use.operand_number, output_idx
  for use in arg_value.uses:
    op = use.owner
    if op.operation.name == _DUS_OP and use.operand_number == 0:
      return op, 0, 0
  for use in arg_value.uses:
    op = use.owner
    if op.operation.name not in _VIEW_OPS:
      continue
    inner = _first_chain_step(op.results[0], module, visited_block_args)
    if inner is not None:
      return inner
  for use in arg_value.uses:
    callee_arg = _resolve_func_call_into_callee(use, module)
    if callee_arg is None:
      continue
    callee_arg_id = id(callee_arg)
    if callee_arg_id in visited_block_args:
      continue
    visited_block_args.add(callee_arg_id)
    inner = _first_chain_step(callee_arg, module, visited_block_args)
    if inner is not None:
      return inner
  return None


def _continuation_chain_step(
    value: ir.Value,
    module: ir.Module,
    visited_block_args: set[int] | None = None,
) -> tuple[ir.OpView, int, int] | None:
  """Next chain step from a carried value: the value must feed operand 0
  (the buffer-being-updated slot); views and func.calls are transparent."""
  if visited_block_args is None:
    visited_block_args = set()
  for use in value.uses:
    if use.operand_number != 0:
      continue
    op = use.owner
    if op.operation.name == _DUS_OP:
      return op, 0, 0
    if _is_pallas_custom_call(op):
      output_idx = _resolve_pallas_output_index(op, 0)
      if output_idx is None:
        continue
      return op, 0, output_idx
  for use in value.uses:
    op = use.owner
    if op.operation.name not in _VIEW_OPS:
      continue
    inner = _continuation_chain_step(
        op.results[0], module, visited_block_args
    )
    if inner is not None:
      return inner
  for use in value.uses:
    callee_arg = _resolve_func_call_into_callee(use, module)
    if callee_arg is None:
      continue
    callee_arg_id = id(callee_arg)
    if callee_arg_id in visited_block_args:
      continue
    visited_block_args.add(callee_arg_id)
    inner = _first_chain_step(callee_arg, module, visited_block_args)
    if inner is not None:
      return inner
  return None


def _walk_donation_chain(
    main_func: ir.Operation,
    arg_idx: int,
    module: ir.Module,
) -> list[tuple[ir.OpView, int, int]]:
  """Returns [(op, operand_idx, within-op output_idx), ...] along the
  chain rooted at @main arg ``arg_idx``; empty if no consumer matches."""
  current = main_func.regions[0].blocks[0].arguments[arg_idx]
  steps: list[tuple[ir.OpView, int, int]] = []
  visited: set[int] = set()
  visited_block_args: set[int] = set()
  is_first = True
  while True:
    next_step = (
        _first_chain_step(current, module, visited_block_args)
        if is_first
        else _continuation_chain_step(current, module, visited_block_args)
    )
    if next_step is None:
      break
    op, operand_idx_in_op, output_idx_in_op = next_step
    next_value = op.results[output_idx_in_op]
    next_id = id(next_value)
    if next_id in visited:
      break
    visited.add(next_id)
    steps.append((op, operand_idx_in_op, output_idx_in_op))
    current = next_value
    is_first = False
  return steps


def _operation_key(op: ir.OpView) -> ir.Operation:
  # Distinct OpView wrappers of one operation hash/compare equal.
  return op.operation


def read_inplace_donate_alias_pairs(
    serialized_mlir: bytes,
) -> list[tuple[int, int]]:
  """Returns the [(arg_idx, output_idx), ...] donation pairs of a module.

  Drives the eager storage rebind; explicit pairs because a donated
  input's aliased output generally sits at a different tuple position.
  """
  context = ir.Context()
  context.allow_unregistered_dialects = True
  stablehlo.register_dialect(context)
  module = stablehlo.deserialize_portable_artifact(context, serialized_mlir)
  return _read_jit_alias_pairs(_find_main_function(module))


def inject_inplace_donate_aliases(
    serialized_mlir: bytes,
    donate_argnums: Sequence[int],
) -> bytes:
  """Returns ``serialized_mlir`` with per-call buffer aliasing applied.

  Idempotent: ops that already carry ``output_operand_aliases`` are
  skipped. No-op when nothing is donated. Raises when the recorded
  donation set disagrees with ``donate_argnums``, or when a donated arg
  has no consumer that can honor the in-place contract.
  """
  if not donate_argnums:
    return serialized_mlir

  context = ir.Context()
  context.allow_unregistered_dialects = True
  stablehlo.register_dialect(context)
  module = stablehlo.deserialize_portable_artifact(context, serialized_mlir)

  main_func = _find_main_function(module)
  jit_pairs = _read_jit_alias_pairs(main_func)
  if not jit_pairs:
    return serialized_mlir

  expected_donations = set(int(i) for i in donate_argnums)
  recorded_donations = set(arg_idx for arg_idx, _ in jit_pairs)
  if expected_donations != recorded_donations:
    raise RuntimeError(
        "pallas inplace_donate alias injection: caller-supplied "
        f"donate_argnums={sorted(expected_donations)} disagrees with the "
        f"tf.aliasing_output set recorded on @main args="
        f"{sorted(recorded_donations)}."
    )

  # Donations feeding the same op share one attribute, one entry each.
  op_entries: dict[ir.Operation, tuple[ir.OpView, list[tuple[int, int]]]] = {}
  for arg_idx, _ in jit_pairs:
    chain = _walk_donation_chain(main_func, arg_idx, module)
    if not chain:
      raise RuntimeError(
          f"pallas inplace_donate alias injection: @main arg {arg_idx} "
          "is donated but no pallas custom_call or dynamic_update_slice "
          "consumes it in place."
      )
    for op, operand_idx_in_op, output_idx_in_op in chain:
      key = _operation_key(op)
      if key not in op_entries:
        op_entries[key] = (op, [])
      op_entries[key][1].append((operand_idx_in_op, output_idx_in_op))

  changed = False
  for op, entries in op_entries.values():
    if _OUTPUT_OPERAND_ALIASES_ATTR in op.attributes:
      continue
    num_results = len(op.results)
    aliases_attr = _build_output_operand_aliases(
        context, entries, num_results
    )
    op.attributes[_OUTPUT_OPERAND_ALIASES_ATTR] = aliases_attr
    changed = True

  if not changed:
    return serialized_mlir

  target_version = stablehlo.get_current_version()
  serialized = stablehlo.serialize_portable_artifact_str(
      str(module), target_version
  )
  if isinstance(serialized, str):
    serialized = serialized.encode("latin-1")
  return serialized
