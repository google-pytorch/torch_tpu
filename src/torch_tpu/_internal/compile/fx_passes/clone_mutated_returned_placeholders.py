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

"""Clones placeholder nodes that are returned in the graph output and used in computation."""

import torch


def _normalize_gm_and_graph(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> tuple[torch.fx.GraphModule | None, torch.fx.Graph]:
  """Extracts the owning GraphModule (if any) and the Graph."""
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    return gm_or_graph, gm_or_graph.graph
  return getattr(gm_or_graph, "owning_module", None), gm_or_graph


def _get_output_node(graph: torch.fx.Graph) -> torch.fx.Node | None:
  """Finds and validates that the graph has at most one output node."""
  output_nodes = [node for node in graph.nodes if node.op == "output"]
  if not output_nodes:
    return None

  assert (
      len(output_nodes) == 1
  ), f"Expected exactly one output node, got {len(output_nodes)} output nodes."
  return output_nodes[0]


def _get_returned_nodes(output_node: torch.fx.Node) -> set[torch.fx.Node]:
  """Extracts all FX nodes returned by the output node."""
  flat_output_args, _ = torch.utils._pytree.tree_flatten(output_node.args)
  return {arg for arg in flat_output_args if isinstance(arg, torch.fx.Node)}


def _is_returned_compute_placeholder(
    node: torch.fx.Node,
    returned_nodes: set[torch.fx.Node],
) -> bool:
  """Checks if a node is a torch.Tensor output placeholder consumed by compute operations."""
  if node.op != "placeholder" or node not in returned_nodes:
    return False
  if "val" in node.meta and not isinstance(node.meta["val"], torch.Tensor):
    return False
  return any(user.op != "output" for user in node.users)


def _find_placeholders_to_clone(
    graph: torch.fx.Graph,
    returned_nodes: set[torch.fx.Node],
) -> list[torch.fx.Node]:
  """Identifies all placeholder nodes that need cloning."""
  return [
      node
      for node in graph.nodes
      if _is_returned_compute_placeholder(node, returned_nodes)
  ]


def _clone_and_replace_placeholder(
    graph: torch.fx.Graph,
    placeholder: torch.fx.Node,
) -> None:
  """Clones a placeholder and redirects all downstream uses to the clone."""
  with graph.inserting_after(placeholder):
    p_clone = graph.call_function(
        torch.ops.aten.clone.default, args=(placeholder,)
    )
    if hasattr(placeholder, "meta"):
      p_clone.meta = placeholder.meta.copy()

    placeholder.replace_all_uses_with(
        p_clone, delete_user_cb=lambda u, clone=p_clone: u != clone
    )


def apply(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> None:
  """Clones placeholder nodes that are returned in the output and used in compute.

  When an FX graph contains in-place operations that mutate input placeholders,
  the graph's output node may reference the input placeholder directly.

  If a pass (e.g. `split_module`) alters the graph structure such that a mutated
  placeholder is no longer in the output, the output will be a passthrough
  meaning that the user won't receive the mutated value.

  Args:
    gm_or_graph: The FX GraphModule or Graph to modify.
  """
  gm, graph = _normalize_gm_and_graph(gm_or_graph)

  output_node = _get_output_node(graph)
  if output_node is None:
    return

  returned_nodes = _get_returned_nodes(output_node)
  placeholders_to_clone = _find_placeholders_to_clone(graph, returned_nodes)
  if not placeholders_to_clone:
    return

  for placeholder in placeholders_to_clone:
    _clone_and_replace_placeholder(graph, placeholder)

  graph.lint()
  if gm is not None:
    gm.recompile()
