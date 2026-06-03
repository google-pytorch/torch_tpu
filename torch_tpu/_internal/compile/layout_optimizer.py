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
"""Compiler pass to optimize layouts of custom sentinels in the FX graph."""

import torch


def eliminate_back_to_back_sentinels(graph_module: torch.fx.GraphModule) -> None:
  """Merges back-to-back spmd_metadata sentinel ops of kind 'intermediate'.

  This optimizes chains like:
    x -> Sentinel A (Partial -> Partial) -> Sentinel B (Partial -> Shard(0)) -> y
  into:
    x -> Sentinel C (Partial -> Shard(0)) -> y

  This is a critical correctness fix to avoid intermediate '{unreduced}'
  shardings on copy boundaries, which trigger unwanted XLA-inserted logical
  All-Reduces.
  """
  g = graph_module.graph

  merged_any = True
  while merged_any:
    merged_any = False
    for node in list(g.nodes):
      if node.op != "call_function":
        continue
      if node.target != torch.ops.tpu.spmd_metadata.default:
        continue

      # node is node_B
      kind_b = node.args[5]
      if kind_b not in ("intermediate", "output"):
        continue

      parent = node.args[0]
      if not isinstance(parent, torch.fx.Node):
        continue
      if (
          parent.op != "call_function"
          or parent.target != torch.ops.tpu.spmd_metadata.default
      ):
        continue

      # parent is node_A
      kind_a = parent.args[5]
      if kind_a != "intermediate":
        continue

      if kind_b == "output" and parent.args[2] != parent.args[3]:
        continue

      # We can merge!
      # x -> node_A (parent) -> node_B (node) -> users
      # We create node_C (new_node) that merges their specs.
      x = parent.args[0]
      mesh_spec = parent.args[1]
      sharding_spec_in = parent.args[2]  # from node_A
      sharding_spec_out = node.args[3]  # from node_B
      global_shape = node.args[4]

      with g.inserting_before(node):
        new_node = g.create_node(
            "call_function",
            torch.ops.tpu.spmd_metadata.default,
            args=(
                x,
                mesh_spec,
                sharding_spec_in,
                sharding_spec_out,
                global_shape,
                kind_b,
            ),
        )
        new_node.meta = node.meta.copy()

      # Replace all uses of both sentinel nodes with the new merged sentinel node
      node.replace_all_uses_with(new_node)
      parent.replace_all_uses_with(new_node)
      g.erase_node(node)
      g.erase_node(parent)

      merged_any = True
      break

  graph_module.recompile()
