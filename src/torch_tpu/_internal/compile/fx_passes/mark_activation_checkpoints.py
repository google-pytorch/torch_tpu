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

"""FX pass to insert optimization barriers for recomputed nodes."""

import operator
import torch
from torch.utils.checkpoint import CheckpointPolicy


def apply(gm_or_graph: torch.fx.GraphModule | torch.fx.Graph) -> None:
  """Inserts optimization barriers for recomputed leaf nodes.

  This pass finds all nodes marked with `recompute` meta set to
  `MUST_RECOMPUTE` or `PREFER_RECOMPUTE`. It then identifies the "leaf" nodes
  among them (nodes that do not have any users that are also marked for
  recomputation) and inserts a `torch.ops.tpu.optimization_barrier`
  after them.

  Args:
    gm_or_graph: The FX GraphModule or Graph to modify.
  """
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    graph = gm_or_graph.graph
  else:
    graph = gm_or_graph

  recompute_nodes = set()

  for node in graph.nodes:
    if node.meta.get("recompute", None) in [
        CheckpointPolicy.MUST_RECOMPUTE,
        CheckpointPolicy.PREFER_RECOMPUTE,
    ]:
      recompute_nodes.add(node)

  if not recompute_nodes:
    return

  leaf_nodes = set()
  for node in recompute_nodes:
    is_leaf = True
    for user in node.users:
      if user in recompute_nodes:
        is_leaf = False
        break
    if is_leaf:
      leaf_nodes.add(node)

  for node in leaf_nodes:
    users = list(node.users)

    with graph.inserting_after(node):
      # optimization_barrier accepts variadic input.
      barrier_node = graph.call_function(
          torch.ops.tpu.optimization_barrier.default,
          ([node],),
      )

    with graph.inserting_after(barrier_node):
      # optimization_barrier returns variadic output so need to extract the
      # first item.
      getitem_node = graph.create_node(
          "call_function",
          operator.getitem,
          args=(barrier_node, 0),
      )

    for user in users:
      user.replace_input_with(node, getitem_node)

  graph.lint()
