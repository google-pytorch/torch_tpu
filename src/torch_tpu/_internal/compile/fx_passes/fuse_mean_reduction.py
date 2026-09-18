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

"""FX pass to decompose static-dimension aten.mean into sum and reciprocal multiplication.

Rewrites:
  `mean(x, dim, keepdim)` -> `mul(sum(x, dim, keepdim), 1.0 / numel)`
  `mean(x)` -> `mul(sum(x), 1.0 / numel)`
when the reduced dimension(s) are static.

This allows whole-graph epilogue loop fusion before export, eliminating
separate divide_subtract_fusion and copy operations in XLA/StableHLO.
"""

import math
from typing import Any

import torch
import torch.fx


def _normalize_gm_and_graph(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> tuple[torch.fx.GraphModule | None, torch.fx.Graph]:
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    return gm_or_graph, gm_or_graph.graph
  return getattr(gm_or_graph, "owning_module", None), gm_or_graph


def _get_tensor_shape(node: torch.fx.Node) -> tuple[Any, ...] | None:
  """Extracts the tensor shape from node metadata if available."""
  for meta_key in ("val", "example_value", "tensor_meta"):
    meta_val = node.meta.get(meta_key, None)
    if meta_val is not None and hasattr(meta_val, "shape"):
      return tuple(meta_val.shape)
  return None


def _is_static_int(val: Any) -> bool:
  """Checks whether a dimension value is a static integer (not a SymInt)."""
  return isinstance(val, int) and not isinstance(val, torch.SymInt)


def _set_node_meta(new_node: torch.fx.Node, source_node: torch.fx.Node) -> None:
  """Copies metadata from source_node to new_node."""
  for k, v in source_node.meta.items():
    new_node.meta[k] = v
  if source_node.stack_trace:
    new_node.stack_trace = source_node.stack_trace


def apply(gm_or_graph: torch.fx.GraphModule | torch.fx.Graph) -> None:
  """Decomposes static-dimension aten.mean into sum and reciprocal multiplication.

  Args:
    gm_or_graph: The FX GraphModule or Graph to transform.
  """
  gm, graph = _normalize_gm_and_graph(gm_or_graph)

  nodes_to_rewrite: list[
      tuple[
          torch.fx.Node,  # mean_node
          torch.fx.Node,  # x_node
          bool,  # is_default
          Any,  # dim_for_sum
          bool,  # keepdim
          Any,  # dtype
          float,  # reciprocal
      ]
  ] = []

  for node in graph.nodes:
    if node.op != "call_function":
      continue

    is_dim = node.target == torch.ops.aten.mean.dim
    is_default = node.target == torch.ops.aten.mean.default
    if not (is_dim or is_default):
      continue

    if not node.args:
      continue

    x_node = node.args[0]
    if not isinstance(x_node, torch.fx.Node):
      continue

    shape = _get_tensor_shape(x_node)
    if shape is None:
      continue

    rank = len(shape)

    if is_default:
      if any(not _is_static_int(d) for d in shape):
        continue
      numel = math.prod(shape) if shape else 1
      if numel <= 0:
        continue
      dtype = node.kwargs.get("dtype", None)
      nodes_to_rewrite.append(
          (node, x_node, True, None, False, dtype, 1.0 / float(numel))
      )
    else:
      # is_dim
      dim = (
          node.args[1] if len(node.args) >= 2 else node.kwargs.get("dim", None)
      )
      keepdim = (
          node.args[2]
          if len(node.args) >= 3
          else node.kwargs.get("keepdim", False)
      )
      dtype = (
          node.args[3]
          if len(node.args) >= 4
          else node.kwargs.get("dtype", None)
      )

      if dim is None:
        dims_to_reduce = list(range(rank))
        dim_for_sum = None
      elif isinstance(dim, int):
        dims_to_reduce = [dim]
        dim_for_sum = [dim]
      elif isinstance(dim, (list, tuple)):
        dims_to_reduce = list(dim)
        dim_for_sum = list(dim)
      else:
        continue

      canonical_dims = []
      valid_dims = True
      for d in dims_to_reduce:
        if not _is_static_int(d):
          valid_dims = False
          break
        if d < 0:
          d += rank
        if d < 0 or d >= rank:
          valid_dims = False
          break
        canonical_dims.append(d)

      if not valid_dims:
        continue

      unique_dims = set(canonical_dims)
      if any(not _is_static_int(shape[d]) for d in unique_dims):
        continue

      numel = 1
      for d in unique_dims:
        numel *= shape[d]

      if numel <= 0:
        continue

      nodes_to_rewrite.append((
          node,
          x_node,
          False,
          dim_for_sum,
          bool(keepdim),
          dtype,
          1.0 / float(numel),
      ))

  for (
      mean_node,
      x_node,
      is_default,
      dim_for_sum,
      keepdim,
      dtype,
      reciprocal,
  ) in nodes_to_rewrite:
    with graph.inserting_before(mean_node):
      sum_kwargs = {}
      if dtype is not None:
        sum_kwargs["dtype"] = dtype

      if is_default:
        sum_node = graph.call_function(
            torch.ops.aten.sum.default,
            args=(x_node,),
            kwargs=sum_kwargs,
        )
      else:
        sum_args = (
            (x_node, dim_for_sum, keepdim)
            if dim_for_sum is not None
            else (x_node, None, keepdim)
        )
        sum_node = graph.call_function(
            torch.ops.aten.sum.dim_IntList,
            args=sum_args,
            kwargs=sum_kwargs,
        )
      _set_node_meta(sum_node, mean_node)

      mul_node = graph.call_function(
          torch.ops.aten.mul.Scalar,
          args=(sum_node, reciprocal),
      )
      _set_node_meta(mul_node, mean_node)

      mean_node.replace_all_uses_with(mul_node)
      graph.erase_node(mean_node)

  graph.eliminate_dead_code()
  graph.lint()
  if gm is not None:
    for _, submod in gm.named_children():
      if isinstance(submod, torch.fx.GraphModule):
        apply(submod)
    gm.recompile()
