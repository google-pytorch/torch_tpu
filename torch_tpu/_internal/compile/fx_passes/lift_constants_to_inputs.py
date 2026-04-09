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

"""FX pass for extracting embedded constants to be passed in as graph inputs instead.

This pass is needed to consolidate FX graphs with what the Deferred op graph
expects: which is that all non-deferred ops are treated as inputs. Without this
pass `Traversal::ValidateAndReorderInputs` would fail.
"""

from collections.abc import Sequence

import torch


def lift_constants_to_inputs(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> Sequence[torch.Tensor]:
  """Lifts embedded constants (get_attr nodes) in an FX graph to graph inputs.

  This pass finds all `get_attr` nodes that reference torch.Tensor attributes
  on the module, replaces them with new `placeholder` (input) nodes, and
  returns the original constant tensors.

  Args:
    gm_or_graph: The FX GraphModule or Graph to modify.

  Returns:
    A sequence of the torch.Tensor constants that were lifted from
    attributes to inputs.
  """
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    graph = gm_or_graph.graph
  else:
    graph = gm_or_graph
  gm = graph.owning_module

  constants = []

  placeholders = graph.find_nodes(op="placeholder")
  last_placeholder = placeholders[-1] if len(placeholders) > 0 else None

  for node in graph.find_nodes(op="get_attr"):
    # pylint: disable=protected-access
    target = torch.fx.graph_module._get_attr(gm, node.target)
    # pylint: enable=protected-access

    if isinstance(target, torch.Tensor):
      constants.append(target)

      if last_placeholder is not None:
        with graph.inserting_after(last_placeholder):
          new_input = graph.placeholder(
              f"const_{node.target.replace('.', '_')}"
          )
          last_placeholder = new_input
      else:
        with graph.inserting_before(next(iter(graph.nodes))):
          new_input = graph.placeholder(
              f"const_{node.target.replace('.', '_')}"
          )
          last_placeholder = new_input

      node.replace_all_uses_with(new_input)
      graph.erase_node(node)

  graph.lint()
  gm.recompile()

  return constants
