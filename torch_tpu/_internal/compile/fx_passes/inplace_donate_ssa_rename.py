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

This pass takes that naive FX graph and walks every call to a pallas op
registered with ``inplace_donate=True``. For each donated input index it
inserts a ``getitem(call, donate_idx)`` immediately after the call (or
reuses the call node itself when the op has a single tensor output) and
rewrites every later use of the donated input to read the corresponding
output instead. After the pass each donation references a distinct SSA
value and the chain runs cleanly through XLA, with the final value of the
chain naturally flowing into whatever consumes the bundle next.
"""

import operator

import torch

from torch_tpu._internal.pallas import pallas as _pallas


def _node_position_map(graph: torch.fx.Graph) -> dict[torch.fx.Node, int]:
  """Returns a topological index for every node in `graph`.

  FX graphs are kept in topological order, so a simple enumeration is
  sufficient for "comes after" comparisons within the pass.
  """
  return {node: idx for idx, node in enumerate(graph.nodes)}


def _op_qualname(node: torch.fx.Node) -> str | None:
  """Returns ``namespace::opname`` for an FX op node, or ``None`` if N/A."""
  if node.op != "call_function":
    return None
  target = node.target
  schema = getattr(target, "_schema", None)
  if schema is None:
    return None
  return schema.name


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
      raise RuntimeError(
          f"inplace_donate FX pass: single-output kernel "
          f"{_op_qualname(call_node)!r} cannot donate index {donate_idx}; "
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
    op_name = _op_qualname(call_node)
    if op_name is None:
      continue
    donate_argnums = _pallas.get_inplace_donate_op_donate_argnums(op_name)
    if donate_argnums is None:
      continue

    call_pos = position[call_node]
    for donate_idx in donate_argnums:
      if donate_idx >= len(call_node.args):
        # Mismatched op — skip rather than crash; the eager-mode C++ check
        # will produce a clearer error if this op is actually invoked.
        continue
      donated_input = call_node.args[donate_idx]
      if not isinstance(donated_input, torch.fx.Node):
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
