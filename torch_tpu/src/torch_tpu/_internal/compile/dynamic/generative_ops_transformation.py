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

"""Generative operations transformation pass for handling dynamic shapes."""

from typing import Any
import torch
from torch_tpu._internal.compile.dynamic import sym_utils
from torch_tpu._internal.compile.dynamic import symbol_bounds
from torch_tpu._internal.compile.dynamic.sym_shape_manager import SymShapeManager

get_symint_bounds = symbol_bounds.get_symint_bounds


class HandleGenerativeOpsPass:
  """Generative ops transformation pass.

  Generative operations (like torch.arange, torch.ones, etc.) construct new
  tensors without taking a tensor as input. When their shape or size arguments
  are dynamic (represented by SymInts), they cannot be statically compiled with
  fixed shapes. This pass handles them by applying Bounded Dynamism.

  Depending on the operator and its parameter profile, this pass applies one of
  three strategies:
  1. Dedicated Custom Operators: For operations like `torch.arange` with truly
     dynamic lengths, it replaces the original operation directly with a custom
     operator (`dynamic_arange`) that computes the dynamic sequence natively.
  2. Constant Length Sequence Shifting: For operations like `torch.arange` where
     boundaries are dynamic but the sequence length evaluates to a static
     constant, it handles by creating static `torch.arange`, multiplying it by
     the step and adding the start.
  3. Generic Dimension Bounding: For operations like `torch.ones`, it replaces
     dynamic shape arguments with their static upper bounds to allow
     compilation, and appends a `set_dimension_logical_size` operation to
     truncate the tensor at runtime.

  Examples:

    --- Strategy 1: Dedicated Custom Operator (torch.arange) ---
    Model:
      def forward(x: torch.Tensor):
          z = torch.arange(0, x.shape[0])
          return z

    Tracing Context:
      Input `x` has shape `[4]`. The dynamic dimension `x.shape[0]` is assigned
      symbol `s77` with a runtime trace hint of `4`. Our bounds engine estimates
      the static upper bound as `hint * 2 = 8`.

    Input FX Graph:
      def forward(self, arg0_1: "Sym(s77)"):
          arange: "i64[s77]" = arange.start(0, arg0_1)
          return (arange,)

    Modified FX Graph:
      def forward(self, arg0_1: "Sym(s77)", s77_size: "i32[]"):
          c0 = scalar_tensor(0, dtype=int32)
          c1 = scalar_tensor(1, dtype=int32)
          arange_custom: "i64[s77]" = dynamic_arange(c0, s77_size, c1, 8, int64)
          return (arange_custom,)

    --- Strategy 2: Constant Length Sequence Shifting ---
    Model:
      def forward(x: torch.Tensor):
          z = torch.arange(x.shape[0], x.shape[0] + 5)
          return z

    Tracing Context:
      Even though the boundaries depend on the dynamic symbol `s77`, the
      sequence length evaluates to a static constant `5`.

    Input FX Graph:
      def forward(self, arg0_1: "Sym(s77)"):
          add: "Sym(s77 + 5)" = arg0_1 + 5
          arange: "i64[5]" = arange.start(arg0_1, add)
          return (arange,)

    Modified FX Graph:
      def forward(self, arg0_1: "Sym(s77)", s77_size: "i32[]"):
          new_arange: "i64[5]" = arange.default(5, dtype=int64)
          step_tensor = scalar_tensor(1, dtype=int64)
          mul: "i64[5]" = mul(new_arange, step_tensor)
          arange_shifted: "i64[5]" = add(mul, s77_size)
          return (arange_shifted,)

    --- Strategy 3: Generic Dimension Bounding (torch.ones) ---
    Model:
      def forward(x: torch.Tensor):
          z = torch.ones(x.shape[0])
          return z

    Tracing Context:
      Input `x` has shape `[4]`. The dynamic dimension `x.shape[0]` is assigned
      symbol `s77` with a runtime trace hint of `4`. The upper bound is `8`.

    Input FX Graph:
      def forward(self, arg0_1: "Sym(s77)"):
          ones: "f32[s77]" = ones(arg0_1)
          return (ones,)

    Modified FX Graph:
      def forward(self, arg0_1: "Sym(s77)", s77_size: "i32[]"):
          ones: "f32[8]" = ones(8)
          ones_bounded: "f32[s77]" = set_dimension_logical_size(ones, 0,
          s77_size)
          return (ones_bounded,)
  """

  def __init__(self, sym_shape_manager: SymShapeManager):
    self._sym_shape_manager = sym_shape_manager
    self._op_handlers = {
        torch.ops.aten.arange.default: self._process_arange_op,
        torch.ops.aten.arange.start: self._process_arange_op,
        torch.ops.aten.arange.start_step: self._process_arange_op,
        torch.ops.aten.ones.default: self._process_ones_op,
        # TODO (mkkhanna): Add support for other generative ops.
    }

  def __call__(self, graph_module: torch.fx.GraphModule) -> None:
    """Runs the op insertion pass."""
    for node in list(graph_module.graph.nodes):
      if node.op == "call_function":
        handler = self._op_handlers.get(node.target)
        if handler:
          handler(graph_module, node)

  def _process_arange_op(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
  ) -> None:
    """Unified pass to replace dynamic arange with custom op."""
    start = None
    end = None
    step = None

    if node.target == torch.ops.aten.arange.default:
      start = 0
      end = node.args[0]
      step = 1
    elif node.target in [
        torch.ops.aten.arange.start,
        torch.ops.aten.arange.start_step,
    ]:
      start = node.args[0]
      end = node.args[1]
      step = node.args[2] if len(node.args) > 2 else 1
      if "step" in node.kwargs:
        step = node.kwargs["step"]

    assert (
        start is not None
    ), "Failed to extract start parameter from arange node context"
    assert (
        end is not None
    ), "Failed to extract end parameter from arange node context"
    assert (
        step is not None
    ), "Failed to extract step parameter from arange node context"

    is_dynamic = (
        sym_utils.is_symint(start)
        or sym_utils.is_symint(end)
        or sym_utils.is_symint(step)
    )
    if not is_dynamic:
      return

    length = node.meta["val"].shape[0]
    expected_dtype = node.meta["val"].dtype

    if isinstance(length, int):
      return self._process_arange_static_length(
          graph_module, node, length, start, step, expected_dtype
      )

    return self._process_arange_dynamic_length(
        graph_module, node, length, start, end, step, expected_dtype
    )

  def _process_arange_static_length(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
      length: int,
      start: Any,
      step: Any,
      dtype: torch.dtype,
  ) -> None:
    """Handles arange with dynamic boundaries but constant sequence length."""
    kwargs = node.kwargs.copy()
    kwargs["dtype"] = dtype

    start_tensor = self._sym_shape_manager.ensure_tensor(
        graph_module, start, node, dtype=dtype
    )
    step_tensor = self._sym_shape_manager.ensure_tensor(
        graph_module, step, node, dtype=dtype
    )

    # Create arange with static length
    with graph_module.graph.inserting_after(node):
      new_arange = graph_module.graph.call_function(
          torch.ops.aten.arange.default,
          args=(length,),
          kwargs=kwargs,
      )
      new_arange.meta = node.meta.copy()

    # Create (arange * step)
    with graph_module.graph.inserting_after(new_arange):
      mul_node = graph_module.graph.call_function(
          torch.ops.aten.mul.Tensor,
          args=(new_arange, step_tensor),
      )
      mul_node.meta = node.meta.copy()

    # Create ((arange * step) + start)
    with graph_module.graph.inserting_after(mul_node):
      add_node = graph_module.graph.call_function(
          torch.ops.aten.add.Tensor,
          args=(mul_node, start_tensor),
      )
      add_node.meta = node.meta.copy()

    # Replace the original arange node with the new final node
    node.replace_all_uses_with(add_node)
    graph_module.graph.erase_node(node)

  def _process_arange_dynamic_length(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
      length_symint: torch.SymInt,
      start: Any,
      end: Any,
      step: Any,
      dtype: torch.dtype,
  ) -> None:
    """Handles arange with dynamic sequence length."""
    # Generate max_length from the dynamic length
    _, upper = get_symint_bounds(length_symint)
    max_length = upper
    assert max_length is not None, (
        "Failed to extract valid upper bound constraint size for dynamic"
        f" length expression: {length_symint}"
    )

    # Convert inputs to tensor nodes if needed
    start_tensor = self._sym_shape_manager.ensure_tensor(
        graph_module, start, node, dtype=dtype
    )
    end_tensor = self._sym_shape_manager.ensure_tensor(
        graph_module, end, node, dtype=dtype
    )
    step_tensor = self._sym_shape_manager.ensure_tensor(
        graph_module, step, node, dtype=dtype
    )

    # Create dynamic_arange op
    with graph_module.graph.inserting_after(node):
      dynamic_arange_node = graph_module.graph.call_function(
          torch.ops.tpu.dynamic_arange,
          args=(
              start_tensor,
              end_tensor,
              step_tensor,
              max_length,
              dtype,
          ),
      )
      dynamic_arange_node.meta = node.meta.copy()

    # Replace the original arange node with the new dynamic_arange node
    node.replace_all_uses_with(dynamic_arange_node)
    graph_module.graph.erase_node(node)

  def _process_ones_op(
      self,
      graph_module: torch.fx.GraphModule,
      node: torch.fx.Node,
  ) -> None:
    """Processes ones op node."""
    sizes = node.args[0]
    is_container = isinstance(sizes, (list, tuple))
    sizes_list = list(sizes) if is_container else [sizes]

    original_users = set(node.users.keys())
    current_node = node
    new_sizes = []
    has_symint = False

    for dim, arg in enumerate(sizes_list):
      if sym_utils.is_symint(arg):
        has_symint = True
        upper = symbol_bounds.get_upper_bound(arg)
        new_sizes.append(upper)

        tensor_node = self._sym_shape_manager.ensure_tensor(
            graph_module, arg, node
        )
        current_node = self._insert_set_dimension_logical_size(
            graph_module, current_node, tensor_node, dim, node
        )
      else:
        new_sizes.append(arg)

    if has_symint:
      new_size_arg = type(sizes)(new_sizes) if is_container else new_sizes[0]
      node.args = (new_size_arg, *node.args[1:])

    if current_node != node:
      node.replace_all_uses_with(
          current_node, delete_user_cb=lambda u: u in original_users
      )

  def _insert_set_dimension_logical_size(
      self,
      graph_module: torch.fx.GraphModule,
      current_node: torch.fx.Node,
      tensor_size_node: torch.fx.Node,
      dim: int,
      original_node: torch.fx.Node,
  ) -> torch.fx.Node:
    """Inserts a set_dimension_logical_size node after the current node."""
    with graph_module.graph.inserting_after(current_node):
      set_dim_size_node = graph_module.graph.call_function(
          torch.ops.tpu.set_dimension_logical_size,
          args=(current_node, dim, tensor_size_node),
      )
      set_dim_size_node.meta = original_node.meta.copy()
      set_dim_size_node.name = f"{original_node.name}_bounded_{dim}"
    return set_dim_size_node
