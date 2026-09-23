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

"""FX pass to canonicalize clamp_min(x, 0) and clamp(x, 0, None) to relu(x)."""

import torch


def apply(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> None:
  """Canonicalizes clamp operations with zero lower bound to aten.relu.

  PyTorch core decomposes torch.relu into torch.ops.aten.clamp_min.default(x,
  0).
  In TorchTPU, clamp lowerings traditionally promote scalar bounds into device
  tensors, leading to full-tensor memory materialization and runtime dynamic
  broadcasting. Rewriting clamp_min(x, 0) and clamp(x, 0, None) back to
  aten.relu
  enables direct routing to optimized 0D scalar constant ReLU lowering in MLIR.

  Args:
    gm_or_graph: The FX GraphModule or Graph to modify.
  """
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    graph = gm_or_graph.graph
  else:
    graph = gm_or_graph

  clamp_min_targets = {
      torch.ops.aten.clamp_min.default,
      torch.ops.aten.clamp_min,
  }
  clamp_targets = {
      torch.ops.aten.clamp.default,
      torch.ops.aten.clamp,
  }

  nodes_to_replace = []
  for node in graph.nodes:
    if node.op != "call_function":
      continue

    if node.target in clamp_min_targets:
      min_val = node.args[1] if len(node.args) > 1 else node.kwargs.get("min")
      if isinstance(min_val, (int, float)) and min_val == 0:
        nodes_to_replace.append((node, node.args[0]))
    elif node.target in clamp_targets:
      min_val = node.args[1] if len(node.args) > 1 else node.kwargs.get("min")
      max_val = node.args[2] if len(node.args) > 2 else node.kwargs.get("max")
      if isinstance(min_val, (int, float)) and min_val == 0 and max_val is None:
        nodes_to_replace.append((node, node.args[0]))

  for node, input_tensor in nodes_to_replace:
    with graph.inserting_after(node):
      relu_node = graph.call_function(
          torch.ops.aten.relu.default,
          args=(input_tensor,),
      )
      relu_node.meta.update(node.meta)
    node.replace_all_uses_with(relu_node)
    graph.erase_node(node)
