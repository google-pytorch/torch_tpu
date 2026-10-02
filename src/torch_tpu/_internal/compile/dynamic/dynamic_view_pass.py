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

"""Pass to convert input placeholders with dynamic stride and static size."""

from __future__ import annotations

from collections.abc import Sequence

from absl import logging
import torch
from torch._inductor.utils import InputType
from torch.fx.passes import shape_prop
from torch_tpu._internal.compile.dynamic import view_decomposition as decompose


class InputDynamicViewPass:
  """Pass to convert non-contiguous view placeholders into contiguous base buffer

  placeholders, and insert canonical FX view operations to represent the
  original view tensor.
  """

  def __init__(
      self,
      placeholders: list[torch.fx.Node],
      example_inputs: Sequence[InputType],
  ):
    self._placeholders = placeholders
    self._example_inputs = list(example_inputs)
    self.view_arg_indices: set[int] = set()

  @property
  def updated_example_inputs(self) -> Sequence[InputType]:
    return self._example_inputs

  def __call__(
      self,
      graph_module: torch.fx.GraphModule,
  ) -> None:
    """Runs the dynamic view transformation pass on graph_module."""
    for idx, node in enumerate(self._placeholders):
      if node.op != "placeholder" or not isinstance(
          node.meta.get("val"), torch.Tensor
      ):
        continue

      decomp = decompose.decompose_into_view_sequence(node)
      if decomp is None:
        continue
      base_shape, view_ops = decomp
      self.view_arg_indices.add(idx)

      val = node.meta["val"]
      logging.debug(
          "[InputDynamicViewPass] Converting view placeholder '%s':"
          " orig_shape=%s, strides=%s -> base_shape=%s, view_ops=%s",
          node.name,
          list(val.shape),
          list(val.stride()),
          base_shape,
          view_ops,
      )

      # Update placeholder metadata to represent contiguous base buffer
      new_val = val.new_empty(tuple(base_shape))
      if "val" in node.meta:
        node.meta["val"] = new_val
      if "tensor_meta" in node.meta:
        node.meta["tensor_meta"] = shape_prop._extract_tensor_metadata(new_val)

      # Update example inputs
      example_input = self._example_inputs[idx]
      if isinstance(example_input, torch.Tensor):
        if idx < len(self._example_inputs):
          self._example_inputs[idx] = example_input.new_empty(tuple(base_shape))

      # Insert canonical view operation nodes immediately after placeholder.
      # Every inserted node carries FakeTensor metadata, computed by applying
      # the same view op to the running FakeTensor, because the downstream
      # dynamism passes (e.g. HandleSliceLikeOpsPass) read ``meta["val"]`` of
      # each slice/permute node, not only of the final one.
      orig_val = val
      curr_node = node
      curr_val = new_val
      inserted_nodes = set()

      for op_type, op_args in view_ops:
        if op_type == "permute":
          target = torch.ops.aten.permute.default
          args = (list(op_args[0]),)
        elif op_type == "slice":
          dim_idx, start, end, step = op_args
          target = torch.ops.aten.slice.Tensor
          args = (int(dim_idx), start, end, step)
        else:
          raise ValueError(f"Unsupported view operation: {op_type}")

        with graph_module.graph.inserting_after(curr_node):
          view_node = graph_module.graph.call_function(
              target, args=(curr_node, *args)
          )
        curr_val = target(curr_val, *args)
        view_node.meta["val"] = curr_val
        if "tensor_meta" in node.meta:
          view_node.meta["tensor_meta"] = shape_prop._extract_tensor_metadata(
              curr_val
          )

        inserted_nodes.add(view_node)
        curr_node = view_node

      final_view_node = curr_node
      final_view_node.meta = node.meta.copy()
      final_view_node.meta["val"] = orig_val
      if "tensor_meta" in final_view_node.meta:
        final_view_node.meta["tensor_meta"] = (
            shape_prop._extract_tensor_metadata(orig_val)
        )
      for user in list(node.users.keys()):
        if user not in inserted_nodes:
          user.replace_input_with(node, final_view_node)
