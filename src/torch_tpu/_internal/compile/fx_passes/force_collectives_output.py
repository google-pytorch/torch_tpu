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
from torch_tpu._internal.compile import collective_ops


def apply(gm: torch.fx.GraphModule) -> None:
  """Forces all is_collective submodules to output their collective results."""
  # Filter to only call_module nodes with a submodule target.
  for call_submodule_node in gm.graph.find_nodes(op="call_module"):
    submod = getattr(gm, call_submodule_node.target)
    if not isinstance(submod, torch.fx.GraphModule):
      continue

    # Keep track of nodes that must be outputs.
    # Every collective and wait_tensor will be inserted to the list and set
    # initially, and then we'll remove already-forced nodes later.
    forced_nodes_set = set()
    forced_nodes_list = []

    # Get all of the collective ops and wait ops.
    # Find all nodes with "overloadpacket"; torch uses this to switch between
    # implementations, like "all_reduce_default" may have an overloadpacket of
    # all_reduce, all_reduce_, all_reduce_coalesced, or all_reduce_coalesced_.
    for node in submod.graph.nodes:
      if node.op != "call_function":
        continue
      maybe_collective = getattr(node.target, "overloadpacket", None)
      if maybe_collective in collective_ops.COLLECTIVE_OPS:
        # All collectives must be outputs unless used by a wait_tensor.
        forced_nodes_set.add(node)
        forced_nodes_list.append(node)
      elif maybe_collective == torch.ops._c10d_functional.wait_tensor:  # pylint: disable=protected-access
        # All wait_tensors must be outputs.
        forced_nodes_set.add(node)
        forced_nodes_list.append(node)
        for wait_arg in node.args:
          # If a wait_tensor node uses a collective, we don't need to force the
          # collective to be an output directly.
          forced_nodes_set.discard(wait_arg)

    # Get the original outputs (arguments to the output node).
    output_node = submod.graph.find_nodes(op="output")[0]
    output_arg = output_node.args[0] if output_node.args else ()
    if isinstance(output_arg, tuple):
      existing_outputs, was_tuple = list(output_arg), True
    else:
      existing_outputs, was_tuple = [output_arg], False

    # Don't need to force anything that was already an existing output.
    for e in existing_outputs:
      forced_nodes_set.discard(e)

    if not forced_nodes_set:
      # No new collectives or wait_tensors to promote to outputs.
      # No changes needed to this submodule.
      continue

    # Filter down to the nodes that need to be additional outputs.
    # Append the new outputs to the existing outputs.
    new_outputs = existing_outputs + [
        sn for sn in forced_nodes_list if sn in forced_nodes_set
    ]
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
