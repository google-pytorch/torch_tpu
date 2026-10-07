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

"""Forces all collective ops to be used by an output of the submodule."""

import operator

import torch
from torch_tpu._internal.distributed import collective_ops


def _is_collective_or_wait(node: torch.fx.Node) -> bool:
  """Returns True if the node is a collective or a wait_tensor."""
  if node.op != "call_function":
    return False
  maybe_collective = getattr(node.target, "overloadpacket", None)
  return (
      collective_ops.is_collective_op(maybe_collective)
      or maybe_collective == torch.ops._c10d_functional.wait_tensor  # pylint: disable=protected-access
  )


def _mark_live(node: torch.fx.Node, live: set[torch.fx.Node]) -> None:
  """Adds the node and everything it transitively depends on to `live`."""
  stack = [node]
  while stack:
    current = stack.pop()
    if current in live:
      continue
    live.add(current)
    stack.extend(current.all_input_nodes)


def apply(gm: torch.fx.GraphModule) -> None:
  """Forces all is_collective submodules to output their collective results."""
  # Filter to only call_module nodes with a submodule target.
  for call_submodule_node in gm.graph.find_nodes(op="call_module"):
    submod = getattr(gm, call_submodule_node.target)
    if not isinstance(submod, torch.fx.GraphModule):
      continue

    # Get the original outputs (arguments to the output node).
    output_node = submod.graph.find_nodes(op="output")[0]
    output_arg = output_node.args[0] if output_node.args else ()
    if isinstance(output_arg, tuple):
      existing_outputs, was_tuple = list(output_arg), True
    else:
      existing_outputs, was_tuple = [output_arg], False

    # A node is live if it (transitively) feeds an existing output.
    live = set()
    for existing_output in existing_outputs:
      if isinstance(existing_output, torch.fx.Node):
        _mark_live(existing_output, live)

    # Walk the graph backwards so that forcing a node also keeps every
    # collective it depends on alive. This way a chain of dead collectives only
    # needs its last node to be returned.
    forced_nodes = []
    for node in reversed(submod.graph.nodes):
      if node in live or not _is_collective_or_wait(node):
        continue
      forced_nodes.append(node)
      _mark_live(node, live)

    if not forced_nodes:
      # Every collective and wait_tensor already reaches an output.
      # No changes needed to this submodule.
      continue

    # Append the new outputs to the existing outputs, in graph order.
    new_outputs = existing_outputs + forced_nodes[::-1]
    # We always want to output a tuple here:
    #   - If the original output node had no outputs, it would have been an
    #     empty tuple; we replace this with a non-empty tuple.
    #   - If the original output node had 1 or more outputs, then we're
    #     replacing it at least 2 outputs, which must be a tuple.
    output_node.args = (tuple(new_outputs),)
    submod.recompile()

    # If the submodule used to return a single non-tuple value, we might have
    # just changed it to be a tuple. In this case we need to insert a getitem
    # node to extract the original single output, and redirect all later users
    # to the getitem node.
    if not was_tuple and call_submodule_node.users:
      with gm.graph.inserting_after(call_submodule_node):
        getitem_node = gm.graph.call_function(
            operator.getitem, (call_submodule_node, 0)
        )
      for user in list(call_submodule_node.users):
        if user != getitem_node:
          user.replace_input_with(call_submodule_node, getitem_node)

  gm.recompile()
