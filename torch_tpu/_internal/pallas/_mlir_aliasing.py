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
to add an ``output_operand_aliases`` attribute to the kernel's
``stablehlo.custom_call``, matching ``donate_argnums`` 1:1 against the
kernel's outputs. After this attribute is set, XLA's buffer assignment
honors the per-call alias and reuses the input buffer for the output —
so the 28 chained calls share a single buffer instead of allocating 28
new buffers and copying through them.

The rewrite only fires when the user has not already set
``input_output_aliases`` on ``pl.pallas_call`` (the attribute is
present on the custom_call); if they have, we treat that as the source
of truth and leave it alone.
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


def _find_pallas_custom_calls(
    main_func: ir.Operation,
) -> list[ir.Operation]:
  """Returns every pallas custom_call op directly in ``@main``'s body."""
  found: list[ir.Operation] = []
  body = main_func.regions[0].blocks[0]
  for op in body.operations:
    if op.operation.name != "stablehlo.custom_call":
      continue
    target = op.attributes.get("call_target_name", None)
    if target is None:
      continue
    if target.value == _PALLAS_CUSTOM_CALL_TARGET:
      found.append(op)
  return found


def _operand_index_for_main_arg(
    custom_call: ir.Operation,
    main_func: ir.Operation,
    main_arg_idx: int,
) -> int | None:
  """Maps a ``@main`` arg index to a custom_call operand index.

  Returns the position in ``custom_call.operands`` whose underlying
  Value is the block argument ``main_arg_idx``, or ``None`` if no
  direct mapping exists (e.g. there's an intervening reshape/cast).
  For ``inplace_donate=True`` kernels we require this direct mapping
  to honor the buffer-sharing contract.
  """
  main_arg_value = main_func.regions[0].blocks[0].arguments[main_arg_idx]
  for operand_idx, operand in enumerate(custom_call.operands):
    if operand == main_arg_value:
      return operand_idx
  return None


def _build_output_operand_aliases(
    context: ir.Context,
    operand_to_output: Sequence[tuple[int, int]],
    num_results: int,
) -> ir.ArrayAttr:
  """Builds the ``output_operand_aliases`` attribute.

  Each ``(operand_idx, output_idx)`` pair becomes a single
  ``OutputOperandAlias`` entry. For a single-result custom_call,
  ``output_tuple_indices`` is empty; for a multi-result custom_call,
  it carries the result position.
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


def inject_inplace_donate_aliases(
    serialized_mlir: bytes,
    donate_argnums: Sequence[int],
) -> bytes:
  """Returns ``serialized_mlir`` with per-call buffer-aliasing applied.

  Walks the exported pallas kernel module, finds the inner
  ``stablehlo.custom_call`` for the pallas op, and adds
  ``output_operand_aliases`` matching the donation contract — donated
  input index ``i`` aliases output index ``i``. The returned bytes are
  in StableHLO portable-artifact format, ready to be passed to
  ``tpu_torch_pallas.register_custom_kernel``.

  No-ops in the following safe cases:
  * ``donate_argnums`` is empty.
  * The exported module has no pallas custom_call (e.g. a pure-JAX
    kernel with no ``pl.pallas_call``).
  * The custom_call already carries ``output_operand_aliases`` (user
    set ``input_output_aliases`` directly on ``pl.pallas_call`` — we
    treat that as the source of truth and leave it alone).

  Raises ``RuntimeError`` when:
  * The module contains more than one pallas custom_call. The
    donation contract for ``inplace_donate=True`` requires an
    unambiguous mapping from a single donated input to a single
    output buffer, and multiple custom_calls leave that ambiguous.
  * A donated ``@main`` argument does not directly feed the
    custom_call (e.g. there's an intervening reshape). That would
    break the buffer-sharing assumption: the reshape's output is a
    distinct buffer from the function argument, so aliasing the
    custom_call's output to its operand would not reuse the donated
    input's buffer.
  * The ``tf.aliasing_output`` mapping read from ``@main`` arg attrs
    disagrees with the caller-supplied ``donate_argnums`` (defensive:
    catches a stale ``donate_argnums`` from the caller).
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

  custom_calls = _find_pallas_custom_calls(main_func)
  if not custom_calls:
    # No pallas custom_call to annotate. This is the case for pure-JAX
    # kernels that don't use pl.pallas_call. Leave the module alone.
    return serialized_mlir
  if len(custom_calls) > 1:
    raise RuntimeError(
        f"pallas inplace_donate alias injection: @main contains "
        f"{len(custom_calls)} pallas custom_call ops, but "
        "inplace_donate=True requires exactly one so the donation "
        "contract maps unambiguously. Split the kernel or stop using "
        "inplace_donate=True for this op."
    )
  custom_call = custom_calls[0]

  if _OUTPUT_OPERAND_ALIASES_ATTR in custom_call.attributes:
    # User already set input_output_aliases on pl.pallas_call; respect
    # their choice rather than overwriting with potentially-different
    # semantics.
    return serialized_mlir

  operand_to_output: list[tuple[int, int]] = []
  for arg_idx, output_idx in jit_pairs:
    operand_idx = _operand_index_for_main_arg(
        custom_call, main_func, arg_idx
    )
    if operand_idx is None:
      raise RuntimeError(
          f"pallas inplace_donate alias injection: @main arg {arg_idx} "
          "is donated but does not directly feed the pallas "
          "custom_call (likely an intervening reshape/cast). "
          "inplace_donate=True requires the donated input to flow "
          "directly into the kernel so its buffer can be reused."
      )
    operand_to_output.append((operand_idx, output_idx))

  num_results = len(custom_call.results)
  aliases_attr = _build_output_operand_aliases(
      context, operand_to_output, num_results
  )
  custom_call.attributes[_OUTPUT_OPERAND_ALIASES_ATTR] = aliases_attr

  target_version = stablehlo.get_current_version()
  serialized = stablehlo.serialize_portable_artifact_str(
      str(module), target_version
  )
  if isinstance(serialized, str):
    serialized = serialized.encode("latin-1")
  return serialized
