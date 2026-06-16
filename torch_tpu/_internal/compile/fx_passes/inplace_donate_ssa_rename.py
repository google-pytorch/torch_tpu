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

"""SSA-renames donated inputs for pallas ops registered with inplace_donate=True.

The pallas eager wrapper at `pallas_py.cc::PyCallCustomKernel` implements the
``inplace_donate=True`` contract by rebinding each donated input tensor's
``c10::StorageImpl`` to the matching output buffer at the C++ binding layer.
That host-side rebind is the mechanism that lets callers in eager mode drop
the ``input.copy_(output)`` republish between sequential donating kernel
calls.

Under ``torch.compile`` with the TPU backend, sequential calls to the same
pallas op are fused into a single PJRT executable, and
``PyCallCustomKernel`` runs at most once during the placeholder trace.
The host-side rebind is therefore a no-op for runtime correctness, and the
naive FX graph has every donating call referencing the same donated SSA
value — which XLA rejects because the same buffer is donated more than once
(``"Donation requested for invalid buffer"``).

This module exposes two coordinated helpers that together let compile-mode
callers also drop the ``input.copy_(output)`` republish:

* :func:`apply` walks every call to a pallas op registered with
  ``inplace_donate=True``. For each donated input index it inserts a
  ``getitem(call, donate_idx)`` immediately after the call (or reuses the
  call node itself when the op has a single tensor output) and rewrites
  every later use of the donated input to read the corresponding output
  instead. After the pass each donation references a distinct SSA value
  and the chain runs cleanly through XLA, with the final value of the
  chain naturally flowing into whatever consumes the bundle next.

* :func:`find_donate_aliases` identifies, for each graph placeholder that
  feeds a donation chain, the *terminus* of that chain — the final SSA
  value produced by the last donating call. The compile backend uses these
  termini to add executable-level outputs and rebind each placeholder
  tensor's storage to the corresponding output buffer after the executable
  runs. That executable-level rebind makes the explicit
  ``ctx.bundle.copy_(new_bundle)`` redundant: subsequent calls observe the
  updated content because the tensor's storage now points at the
  executable's final output buffer.

Op identification is attribute-based, not registry-based: pallas stamps
``_pallas_inplace_donate_argnums`` onto the ``OpOverload`` at registration
time (see ``pallas._annotate_inplace_donate``). Both helpers read that
attribute directly off ``node.target``. This makes them agnostic to when
the op was registered relative to the FX trace, and lets them pick up
ops produced by both ``pallas.jax_op(..., inplace_donate=True)`` and
``pallas.declare_jax_op(..., inplace_donate=True)`` (the deferred-binding
flavor) without coupling the pass module to the pallas package.
"""

import operator

import torch

# Name of the attribute set on a pallas `OpOverload` when it was registered
# with ``inplace_donate=True``. Value is a tuple of donated input indices.
# This is the cross-module contract between pallas registration
# (``_annotate_inplace_donate`` in ``torch_tpu/_internal/pallas/pallas.py``)
# and this pass. Keep these strings in sync.
_INPLACE_DONATE_ARGNUMS_ATTR = "_pallas_inplace_donate_argnums"


def _node_position_map(graph: torch.fx.Graph) -> dict[torch.fx.Node, int]:
  """Returns a topological index for every node in `graph`.

  FX graphs are kept in topological order, so a simple enumeration is
  sufficient for "comes after" comparisons within the pass.
  """
  return {node: idx for idx, node in enumerate(graph.nodes)}


def _inplace_donate_argnums(
    node: torch.fx.Node,
) -> tuple[int, ...] | None:
  """Returns donated input indices for a pallas-inplace-donate node.

  Reads the ``_pallas_inplace_donate_argnums`` attribute that pallas
  stamps onto an ``OpOverload`` at registration time. Returns ``None``
  for any node that is not a call to such an op.
  """
  if node.op != "call_function":
    return None
  return getattr(node.target, _INPLACE_DONATE_ARGNUMS_ATTR, None)


def _get_arg_value(
    call_node: torch.fx.Node, idx: int,
) -> torch.fx.Node | None:
  """Returns the FX value bound to parameter index ``idx`` on ``call_node``.

  Pallas custom ops are registered with `torch.library.custom_op`, which
  produces a named schema for the wrapped JAX function. FX represents
  positional arguments under ``call_node.args`` and keyword arguments
  under ``call_node.kwargs`` (keyed by the schema's parameter name).

  This helper checks the positional path first; if ``idx`` is past the
  positional tail it falls back to the schema's ``arguments[idx].name``
  and reads ``call_node.kwargs``. Returns ``None`` if no value is bound
  at that parameter position (caller treats that as a non-applicable
  donation step rather than crashing).
  """
  if idx < len(call_node.args):
    return call_node.args[idx]
  schema = getattr(call_node.target, "_schema", None)
  if schema is None:
    return None
  schema_args = schema.arguments
  if idx >= len(schema_args):
    return None
  return call_node.kwargs.get(schema_args[idx].name, None)


def _select_output_node(
    graph: torch.fx.Graph,
    call_node: torch.fx.Node,
    donate_idx: int,
) -> torch.fx.Node:
  """Returns an FX node that yields ``call_node[donate_idx]``.

  When the op returns a single tensor the call node itself is the donated
  output (``donate_idx`` is then required to be ``0``). When it returns a
  tuple a ``getitem`` is inserted (or reused) immediately after the call.
  """
  val = call_node.meta.get("val")
  if not isinstance(val, (tuple, list)):
    # Single-output kernel — the call node IS the donated output.
    if donate_idx != 0:
      schema = getattr(call_node.target, "_schema", None)
      qualname = schema.name if schema is not None else str(call_node.target)
      raise RuntimeError(
          f"inplace_donate FX pass: single-output kernel "
          f"{qualname!r} cannot donate index {donate_idx}; "
          "donate_argnums for single-output kernels must be (0,)."
      )
    return call_node

  # Reuse an existing getitem(call_node, donate_idx) if one already exists
  # immediately after the call — keeps the rewrite idempotent and avoids
  # cluttering the graph when one of the kernel's outputs was already being
  # unpacked by user code.
  for user in call_node.users:
    if (
        user.op == "call_function"
        and user.target is operator.getitem
        and len(user.args) == 2
        and user.args[0] is call_node
        and user.args[1] == donate_idx
    ):
      return user

  with graph.inserting_after(call_node):
    return graph.call_function(operator.getitem, (call_node, donate_idx))


def apply(gm_or_graph: torch.fx.GraphModule | torch.fx.Graph) -> None:
  """Runs the SSA rename across `gm_or_graph`.

  Safe to call on a graph with no inplace-donate ops — it is a no-op
  in that case. Idempotent on a graph it has already processed.
  """
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    graph = gm_or_graph.graph
  else:
    graph = gm_or_graph

  position = _node_position_map(graph)

  changed = False
  for call_node in list(graph.nodes):
    donate_argnums = _inplace_donate_argnums(call_node)
    if donate_argnums is None:
      continue

    call_pos = position[call_node]
    for donate_idx in donate_argnums:
      donated_input = _get_arg_value(call_node, donate_idx)
      if not isinstance(donated_input, torch.fx.Node):
        # The donated parameter wasn't passed at this call site (e.g. it
        # has a default and the user omitted it), or the binding resolves
        # to a non-Node literal. Either way there's no SSA value to
        # rewrite; skip rather than crash.
        continue

      output_node = _select_output_node(graph, call_node, donate_idx)
      # `output_node` is either `call_node` itself or a getitem we just
      # inserted — either way it must not feed back into itself.
      if output_node is donated_input:
        continue

      # Track the getitem node's position so newly created users don't get
      # rewritten in this same pass iteration.
      if output_node not in position:
        position[output_node] = call_pos + 0.5  # type: ignore[assignment]

      for user in list(donated_input.users):
        if user is output_node:
          continue
        user_pos = position.get(user)
        if user_pos is None or user_pos <= call_pos:
          continue
        user.replace_input_with(donated_input, output_node)
        changed = True

  if changed:
    graph.lint()


def _follow_donate_chain_to_terminus(
    start: torch.fx.Node,
) -> torch.fx.Node | None:
  """Walks forward from ``start`` through chained inplace-donate calls.

  Returns the SSA value produced by the *last* donating call in the chain,
  or ``None`` if ``start`` is not threaded through any inplace-donate
  call. ``None`` is also returned when the chain is structurally broken
  (e.g. a multi-output op's donated ``getitem`` is missing), because in
  that case there is no single value that semantically represents the
  donated content at the executable boundary.
  """
  current = start
  advanced = False
  visited: set[torch.fx.Node] = {current}
  while True:
    next_output: torch.fx.Node | None = None
    for user in current.users:
      donate_argnums = _inplace_donate_argnums(user)
      if donate_argnums is None:
        continue
      # Find which donated index of `user` points at `current`. Checks
      # positional args first and falls back to kwargs via the op's
      # schema — mirrors ``_get_arg_value`` in ``apply``.
      donate_idx: int | None = None
      for didx in donate_argnums:
        if _get_arg_value(user, didx) is current:
          donate_idx = didx
          break
      if donate_idx is None:
        continue

      val = user.meta.get("val")
      if not isinstance(val, (tuple, list)):
        if donate_idx != 0:
          return None
        candidate = user
      else:
        # Multi-output op: the donated output is `getitem(user, donate_idx)`.
        candidate = None
        for sub_user in user.users:
          if (
              sub_user.op == "call_function"
              and sub_user.target is operator.getitem
              and len(sub_user.args) == 2
              and sub_user.args[0] is user
              and sub_user.args[1] == donate_idx
          ):
            candidate = sub_user
            break
        if candidate is None:
          # `apply()` always inserts this getitem when it rewrites a chain.
          # Its absence means either (a) the rewrite hasn't run, or (b) the
          # donated output is genuinely unused — in either case we cannot
          # produce a meaningful executable-level alias.
          return None
      next_output = candidate
      break

    if next_output is None or next_output is current:
      break
    if next_output in visited:
      # Defensive: prevents an infinite loop if the graph contains a cycle
      # through donated users (which would be ill-formed but we still want
      # to terminate).
      return None
    visited.add(next_output)
    current = next_output
    advanced = True

  return current if advanced else None


def find_donate_aliases(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> list[tuple[int, torch.fx.Node]]:
  """Identifies executable-level input→output donation aliases.

  For each placeholder of ``gm_or_graph`` that feeds a chain of pallas
  inplace-donate calls (after :func:`apply` has SSA-renamed the chain),
  returns a ``(placeholder_index, terminus_node)`` pair where
  ``terminus_node`` is the final SSA value of the chain. The compile
  backend appends each ``terminus_node`` as an additional executable
  output and, after the executable runs, rebinds the corresponding input
  tensor's storage to that output's buffer.

  Placeholders that are not threaded through any inplace-donate chain
  are skipped. Placeholder indices follow the order in which placeholders
  appear in the FX graph, matching the convention that the
  ``i``-th element of the runtime args list is the ``i``-th placeholder.

  Must be called *after* :func:`apply` has rewritten the chain, so that
  the per-call ``getitem`` nodes for multi-output donated outputs are
  present and the chain can be followed unambiguously.
  """
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    graph = gm_or_graph.graph
  else:
    graph = gm_or_graph

  aliases: list[tuple[int, torch.fx.Node]] = []
  seen_termini: set[torch.fx.Node] = set()
  for ph_idx, placeholder in enumerate(graph.find_nodes(op="placeholder")):
    terminus = _follow_donate_chain_to_terminus(placeholder)
    if terminus is None:
      continue
    if terminus in seen_termini:
      # Two placeholders converging on the same terminus would mean the
      # same buffer would be rebound twice — defensively skip the duplicate
      # rather than produce ambiguous storage semantics.
      continue
    seen_termini.add(terminus)
    aliases.append((ph_idx, terminus))
  return aliases


def raise_if_present(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> None:
  """Raises ``NotImplementedError`` if any inplace-donate op call appears in
  ``gm_or_graph``.

  Compile paths that do not run :func:`apply` (e.g. the dynamic-shapes
  ``DynamicCompiler``) call this to surface a clear "not supported" error
  at the entry of compilation. Without it, the same FX graph would reach
  XLA with the donated arg referenced by every chained call as the same
  SSA value, and XLA would reject the program with the much less actionable
  ``"Donation requested for invalid buffer"``.
  """
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    graph = gm_or_graph.graph
  else:
    graph = gm_or_graph
  for node in graph.nodes:
    if _inplace_donate_argnums(node) is None:
      continue
    schema = getattr(node.target, "_schema", None)
    qualname = schema.name if schema is not None else str(node.target)
    raise NotImplementedError(
        f"Pallas op {qualname!r} is registered with inplace_donate=True but"
        " the current compile path does not run the inplace-donate FX"
        " transform (currently only StaticCompiler does). The chain would"
        " reach XLA with the same SSA value referenced by every donating"
        " call and XLA would reject it. Either compile with static shapes"
        " (e.g. ``torch.compile(dynamic=False)``) or drop"
        " inplace_donate=True from this op."
    )
