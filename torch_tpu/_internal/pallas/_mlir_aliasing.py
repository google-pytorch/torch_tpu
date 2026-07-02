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

"""Inject per-call buffer-aliasing onto a pallas-kernel's exported StableHLO.

Pallas kernels registered with ``inplace_donate=True`` participate in
chains where the same large tensor (e.g. a KV-cache bundle) is threaded
through N sequential calls. The compile-backend's
``inplace_donate_ssa_rename`` FX pass already SSA-renames the chain so
that each call references a distinct value, and the executable-level
storage rebind in :class:`_TorchTpuCompiledExecutable` makes the host-side
tensor observe the final output buffer of the chain.

What that does NOT change is what XLA's buffer-assignment sees inside
the executable. Each per-layer kernel call lowers to a
``stablehlo.custom_call`` that wraps the Pallas kernel; XLA treats the
custom call as a black box and conservatively allocates a fresh output
buffer per call. With 28 attention layers and a ~37 GB bundle, that
fresh-buffer-per-call cost dominates wall-clock (~25× regression in the
vLLM block-major flow).

This module solves that by post-processing the JAX-exported kernel MLIR
to add ``output_operand_aliases`` attributes along the donation chain:

* For a kernel whose donated input flows DIRECTLY into the pallas
  ``stablehlo.custom_call`` (the simple full-bundle pattern, e.g.
  ``out = pl.pallas_call(...)(bundle, x)``), the attribute lands on the
  custom_call itself, mapping the donated operand position to the
  matching output position.

* For a kernel whose donated input is sliced before being passed to
  pallas and the per-layer result is then scattered back via a
  ``stablehlo.dynamic_update_slice`` (the slice→pallas→DUS pattern
  produced by ``jax.lax.dynamic_update_slice_in_dim`` wrappers around a
  per-layer kernel), the attribute lands on the DUS instead — DUS is
  the op that owns the bundle-shape output, so aliasing it is what
  conveys the in-place contract for the bundle.

The walk follows the chain forward from each donated ``@main`` arg.
The FIRST step may match the arg at any operand position (a multi-arg
full-bundle pallas kernel donates `arg i → operand i → result i`).
SUBSEQUENT steps require the carried value to feed operand 0 of the
next op (the standard "buffer being updated" slot for both pallas
custom_calls and DUSes). A small set of no-copy SHAPE-VIEW ops
(``stablehlo.reshape``, ``stablehlo.transpose``,
``stablehlo.broadcast_in_dim``, ``stablehlo.bitcast_convert_type``)
are TRANSPARENT to the walk: the carried buffer-view is the same
underlying buffer, so we recurse on the view's result without adding
a chain step for the view itself. A ``func.call`` use is likewise
transparent: the chain continues from the callee block argument at
the matching operand position (XLA's CallInliner preserves op
attributes when it folds the callee body back into @main).

For a pallas ``stablehlo.custom_call``, the within-op output index
that aliases a given operand position is read from the call's own
pre-existing ``output_operand_aliases`` attribute when present (this
is JAX's translation of ``input_output_aliases`` on ``pl.pallas_call``,
preserved across ``jax.export.export``). The default "operand i →
result i" mapping is used as a fallback only when that index is
in range; a chain step that would otherwise reference a non-existent
result is dropped so the walker can fall through to a different
consumer of the donated value rather than crashing.

The rewrite is idempotent and respectful: it skips any op that already
carries ``output_operand_aliases`` (e.g. the user set
``input_output_aliases`` on ``pl.pallas_call``, or an earlier
injection already ran).

Caveat on DUS-side aliasing: at the time of writing, the StableHLO→HLO
converter only recognizes ``output_operand_aliases`` on
``stablehlo.custom_call`` (its TableGen-declared owner); other ops
carry it as a discardable attribute. We still annotate the DUS because
the attribute is structurally accurate, survives MLIR-level passes,
documents the donation contract precisely where it applies, and tracks
the right shape so that any future converter improvement picks it up
without us needing to change this module. In the meantime, XLA's
executable-level donation (the JAX ``input_output_alias`` config) and
BufferAssignment's data-flow analysis together still thread the DUS
chain through a single buffer at the allocation level.
"""

from collections.abc import Sequence

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo

# JAX uses this attribute name on ``@main`` arg attrs to mark which
# function argument is donated. The value is the integer index of the
# matching output. This is JAX's translation of ``jax.jit``'s
# ``donate_argnums`` after ``static_argnums`` filtering, so reading it
# back lets us avoid re-doing that translation here.
_TF_ALIASING_OUTPUT_ATTR = "tf.aliasing_output"

# ``stablehlo.custom_call`` target name emitted by ``pl.pallas_call``.
# Pallas always uses this exact target string when emitting MLIR for
# the TPU backend.
_PALLAS_CUSTOM_CALL_TARGET = "tpu_custom_call"

# Attribute name carrying per-operand→result aliasing on the
# ``stablehlo.custom_call`` op (consumed by XLA's buffer assignment).
_OUTPUT_OPERAND_ALIASES_ATTR = "output_operand_aliases"

# Op names we recognize as carriers of a donation-chain step. For DUSes
# the donated value must be at operand 0 (the "buffer being updated"
# slot); for pallas custom_calls it can be at any operand position
# (multi-arg donation maps operand i → result i).
_CUSTOM_CALL_OP = "stablehlo.custom_call"
_DUS_OP = "stablehlo.dynamic_update_slice"

# A ``func.call`` use of a donated value is a transparent step: the
# chain continues from the matching block argument of the callee
# function. JAX export of a jit-inside-jit kernel wraps the body in a
# private callee whose @main bridges arg → call(callee, ..., arg, ...).
# Annotating ops inside the callee is correct because XLA's CallInliner
# preserves op attributes when it folds the callee body back into
# @main.
_FUNC_CALL_OP = "func.call"

# No-copy shape-view ops that the walker treats as transparent. The
# carried buffer survives through each of these as the same physical
# allocation under XLA's BufferAssignment, so the donation contract
# continues from ``op.results[0]``. ``broadcast_in_dim`` is included
# because it appears in chains where the broadcast is degenerate
# (matches input shape and dimensions) and acts as a typecast — when
# this is not the case the downstream in-place op simply does not
# fire and the walker terminates without an alias.
_VIEW_OPS = frozenset(
    [
        "stablehlo.reshape",
        "stablehlo.transpose",
        "stablehlo.broadcast_in_dim",
        "stablehlo.bitcast_convert_type",
    ]
)


def _find_main_function(module: ir.Module) -> ir.Operation:
  """Returns the @main ``func.func`` op of `module`.

  Raises ``RuntimeError`` if not present. The exported pallas kernel
  module is always produced by ``jax.export.export`` and contains a
  single public ``@main`` function.
  """
  for op in module.operation.regions[0].blocks[0].operations:
    sym_name = op.attributes.get("sym_name", None)
    if sym_name is not None and sym_name.value == "main":
      return op
  raise RuntimeError(
      "pallas inplace_donate alias injection: exported MLIR has no @main "
      "function. JAX export must produce a module with a single public "
      "@main; this module is malformed."
  )


def _find_function_by_name(
    module: ir.Module, name: str,
) -> ir.Operation | None:
  """Returns the ``func.func`` op with ``sym_name == name``, or None.

  Used to resolve a ``func.call``'s callee symbol within the same
  exported module. Returns ``None`` (rather than raising) so the chain
  walker can degrade gracefully when a call target lives outside the
  module — that case is structurally identical to a dead end for our
  purposes.
  """
  for op in module.operation.regions[0].blocks[0].operations:
    sym_name = op.attributes.get("sym_name", None)
    if sym_name is not None and sym_name.value == name:
      return op
  return None


def _read_jit_alias_pairs(
    main_func: ir.Operation,
) -> list[tuple[int, int]]:
  """Returns the ``[(arg_idx, output_idx), ...]`` JIT donation pairs.

  Reads each ``@main`` argument's ``tf.aliasing_output`` attribute,
  which JAX sets from ``donate_argnums`` on the JIT (post
  ``static_argnums`` filtering). The output index is the matching
  position in the function's flat return tuple.
  """
  arg_attrs = main_func.attributes.get("arg_attrs", None)
  if arg_attrs is None:
    return []
  pairs: list[tuple[int, int]] = []
  for arg_idx, dict_attr in enumerate(arg_attrs):
    for entry in dict_attr:
      if entry.name == _TF_ALIASING_OUTPUT_ATTR:
        # The value is an IntegerAttr ("0 : i32"); pull the int via .value.
        pairs.append((arg_idx, int(entry.attr.value)))
        break
  return pairs


def _is_pallas_custom_call(op: ir.OpView) -> bool:
  """Returns True if ``op`` is a pallas ``stablehlo.custom_call``."""
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
  """Builds the ``output_operand_aliases`` attribute.

  Each ``(operand_idx, output_idx)`` pair becomes a single
  ``OutputOperandAlias`` entry. For a single-result op,
  ``output_tuple_indices`` is empty; for a multi-result op, it carries
  the result position.
  """
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
  """Returns the within-op result index aliased to ``operand_idx``.

  Two paths, in priority order:

  1. If ``op`` already carries an ``output_operand_aliases`` attribute
     (e.g. JAX export translated ``input_output_aliases`` declared on
     ``pl.pallas_call``), look up the entry whose ``operand_index``
     matches ``operand_idx`` and return its
     ``output_tuple_indices[0]`` (or 0 for a single-result op where
     the indices list is empty).
  2. Otherwise, default to ``operand_idx`` itself — the standard
     "operand i → result i" assumption for full-bundle pallas kernels
     where the donated input flows directly into a same-shape result.

  Returns ``None`` if neither path produces an index that is in range
  for ``op.results``. This lets the chain walker treat a pallas
  custom_call whose declared aliasing does not match the current
  operand as a non-step (so it can fall through to another consumer
  of the donated value) rather than indexing past the end of
  ``op.results``.
  """
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
  """Follows a ``func.call`` use into the callee's matching block arg.

  When the donated value enters a ``func.call`` at some operand
  position, the chain continues from the callee function's block
  argument at the same position. Returns that block argument, or
  ``None`` if the use is not a ``func.call`` we can resolve.
  """
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
  """Returns the first step of the donation chain rooted at ``arg_value``.

  The first step may match the donated arg at any operand position. We
  prefer a pallas custom_call (the most direct in-place semantics),
  fall back to a DUS at operand 0, then recurse through a no-copy
  view op (``stablehlo.reshape``, ``transpose``, ``broadcast_in_dim``,
  ``bitcast_convert_type``), and finally follow a ``func.call`` into
  its callee's matching block argument. Returns
  ``(op, operand_idx_in_op, output_idx_in_op)`` or ``None`` if no
  matching consumer exists.

  ``visited_block_args`` carries the set of block-argument ids we have
  already entered, to defend against ill-formed modules with cyclic
  call graphs.
  """
  if visited_block_args is None:
    visited_block_args = set()
  # Pass 1: prefer a pallas custom_call. For pallas, the within-op
  # result index aliased to ``use.operand_number`` is resolved from
  # the op's existing ``output_operand_aliases`` attribute (if set by
  # the kernel via ``input_output_aliases``), falling back to
  # ``operand_idx`` when in range — see ``_resolve_pallas_output_index``.
  for use in arg_value.uses:
    op = use.owner
    if not _is_pallas_custom_call(op):
      continue
    output_idx = _resolve_pallas_output_index(op, use.operand_number)
    if output_idx is None:
      continue
    return op, use.operand_number, output_idx
  # Pass 2: a DUS at operand 0 — the "buffer being updated" slot.
  # DUSes always have a single result, so the output index is 0.
  for use in arg_value.uses:
    op = use.owner
    if op.operation.name == _DUS_OP and use.operand_number == 0:
      return op, 0, 0
  # Pass 3: recurse through a no-copy view op. View ops are
  # transparent — the carried buffer-view is the same allocation, so
  # the donation chain continues from ``op.results[0]`` with
  # first-step semantics (any operand position).
  for use in arg_value.uses:
    op = use.owner
    if op.operation.name not in _VIEW_OPS:
      continue
    inner = _first_chain_step(op.results[0], module, visited_block_args)
    if inner is not None:
      return inner
  # Pass 4: follow a func.call into the callee. JAX export wraps a
  # jit-inside-jit kernel in @main → call @callee(..., donated_arg, ...);
  # the actual chain lives in the callee.
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
  """Returns the next chain step from a carried value.

  After the first step, the donation value flows into operand 0 of the
  next op (the "buffer being updated" slot for both DUSes and pallas
  custom_calls in a continuation). View ops and ``func.call`` are
  followed transparently, same as for the first step. Returns the
  matched op or ``None``.
  """
  if visited_block_args is None:
    visited_block_args = set()
  # Direct in-place op at operand 0.
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
  # Recurse through a no-copy view op. The carried value's only use
  # of a unary view op is at operand 0, so no filter needed.
  for use in value.uses:
    op = use.owner
    if op.operation.name not in _VIEW_OPS:
      continue
    inner = _continuation_chain_step(
        op.results[0], module, visited_block_args
    )
    if inner is not None:
      return inner
  # No direct in-place op consumed `value` at operand 0 — try
  # threading through a func.call at any operand position.
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
  """Walks the donation chain from ``@main`` arg ``arg_idx``.

  Returns ``[(op, operand_idx, output_idx_in_op), ...]`` for each op
  along the chain. Each ``output_idx_in_op`` is the WITHIN-OP result
  index that aliases with the matched operand — the value
  ``_build_output_operand_aliases`` consumes for
  ``output_tuple_indices``. It is intentionally independent of the
  @main output index; XLA's BufferAssignment plumbs each instruction-
  level alias to @main's outputs via dataflow regardless of how the
  return tuple is ordered.

  Returns an empty list when the donated arg has no consumer matching
  the chain pattern.
  """
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
  """Returns a stable dict key for an MLIR op.

  Distinct ``OpView`` wrappers for the same underlying MLIR operation
  compare equal and hash to the same value via the C++ binding's
  ``__eq__``/``__hash__``, so we can use the unwrapped ``Operation``
  directly as a dict key.
  """
  return op.operation


def inject_inplace_donate_aliases(
    serialized_mlir: bytes,
    donate_argnums: Sequence[int],
) -> bytes:
  """Returns ``serialized_mlir`` with per-call buffer-aliasing applied.

  Walks the exported pallas kernel module and, for each ``@main``
  argument that JAX recorded as donated, traces the in-place chain
  forward — adding ``output_operand_aliases`` to every pallas
  ``stablehlo.custom_call`` and every ``stablehlo.dynamic_update_slice``
  along the way. The returned bytes are in StableHLO portable-artifact
  format, ready to be passed to ``tpu_torch_pallas.register_custom_kernel``.

  No-ops in the following safe cases:

  * ``donate_argnums`` is empty.
  * JAX did not record any donation on ``@main`` arg attrs (no
    ``tf.aliasing_output``).
  * Every op along every donation chain already carries
    ``output_operand_aliases`` (the user has set ``input_output_aliases``
    on ``pl.pallas_call`` or a previous injection already ran).

  Raises ``RuntimeError`` when:

  * The ``tf.aliasing_output`` mapping read from ``@main`` arg attrs
    disagrees with the caller-supplied ``donate_argnums`` (defensive:
    catches a stale ``donate_argnums`` from the caller).
  * A donated ``@main`` arg has no consumer that can honor the
    contract — i.e. there is no pallas custom_call that takes it as
    an operand AND no DUS that takes it as operand 0. The buffer-
    sharing assumption would be silently violated, so we refuse.
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
    # JAX did not record any donation; nothing to alias.
    return serialized_mlir

  expected_donations = set(int(i) for i in donate_argnums)
  recorded_donations = set(arg_idx for arg_idx, _ in jit_pairs)
  if expected_donations != recorded_donations:
    raise RuntimeError(
        "pallas inplace_donate alias injection: caller-supplied "
        f"donate_argnums={sorted(expected_donations)} disagrees with the "
        f"tf.aliasing_output set recorded on @main args="
        f"{sorted(recorded_donations)}. The JIT-level donation contract "
        "must match the inplace_donate contract."
    )

  # Per-op accumulator: when multiple donated args feed the same op
  # (e.g. a multi-output pallas kernel donating arg 0 → out 0 AND
  # arg 1 → out 1), they share one ``output_operand_aliases`` attribute
  # with one entry per donation. We key the dict on the op itself —
  # MLIR Python op wrappers hash on the underlying operation, so
  # distinct wrappers for the same op collide correctly.
  op_entries: dict[ir.Operation, tuple[ir.OpView, list[tuple[int, int]]]] = {}
  for arg_idx, _ in jit_pairs:
    chain = _walk_donation_chain(main_func, arg_idx, module)
    if not chain:
      raise RuntimeError(
          f"pallas inplace_donate alias injection: @main arg {arg_idx} "
          "is recorded as donated but has no consumer that can honor "
          "the contract — neither a pallas custom_call nor a "
          "dynamic_update_slice takes it as operand 0. Drop "
          "inplace_donate=True or rewrite the kernel so the donated "
          "input flows into an in-place op."
      )
    for op, operand_idx_in_op, output_idx_in_op in chain:
      key = _operation_key(op)
      if key not in op_entries:
        op_entries[key] = (op, [])
      op_entries[key][1].append((operand_idx_in_op, output_idx_in_op))

  changed = False
  for op, entries in op_entries.values():
    if _OUTPUT_OPERAND_ALIASES_ATTR in op.attributes:
      # Respect any pre-existing aliasing (user-set, or a previous
      # injection).
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
