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

"""FX pass to fuse RoPE and minor-dimension slice-concat pairs into outer-dimension slicing.

In transformer models (e.g. Gemma 2, Gemma 4, LLaMA 3), rotary position
embeddings (RoPE) use rotate_half:
  x1 = x[..., :half]
  x2 = x[..., half:]
  return torch.cat((-x2, x1), dim=-1)

On multi-dimensional attention tensors (e.g. [batch, num_heads, seq_len,
head_dim]), slicing and concatenating across the minor dimension (dim=3 or
dim=-1) forces unaligned sublane tiles on TPU and causes XLA to emit standalone
formatting copies (e.g. copy.1746) and excessive HBM materialization.

This pass recognizes adjacent slice pairs along the minor dimension (dim in
(-1, rank - 1, 3)) and rewrites them:
1. Identity elimination: If slices are adjacent and in order [0:half] and
   [half:end] with no negation, replace cat directly with the base tensor.
2. Outer-dimension slicing: For RoPE rotate_half or swapped slice pairs,
   reshape the base tensor to [..., 2, half], slice along dimension rank - 1
   (the outer dimension of size 2), concatenate along that dimension, and view
   back to the original shape. This keeps the minor elements contiguous in
   registers/tiles and eliminates strided formatting copies in HBM.
"""

import os
from typing import Any
import torch
import torch.fx


def _is_fuse_slice_concat_enabled() -> bool:
  """Returns whether the pass is enabled via environment variable."""
  val = os.environ.get(  # PY_ENVIRON_OK=internal pass toggle.
      "TORCH_TPU_INTERNAL_ENABLE_FUSE_SLICE_CONCAT", "1"
  )
  return val.lower() not in ("0", "false", "no")


def _normalize_gm_and_graph(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> tuple[torch.fx.GraphModule | None, torch.fx.Graph]:
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    return gm_or_graph, gm_or_graph.graph
  return getattr(gm_or_graph, "owning_module", None), gm_or_graph


def _unwrap_identity_views(node: Any) -> Any:
  curr = node
  while (
      isinstance(curr, torch.fx.Node)
      and curr.op == "call_function"
      and curr.target
      in (
          torch.ops.aten.clone.default,
          torch.ops.aten.contiguous.default,
      )
      and curr.args
      and isinstance(curr.args[0], torch.fx.Node)
  ):
    curr = curr.args[0]
  return curr


def _extract_slice_info(node: torch.fx.Node):
  """Extracts slice parameters and unwraps neg/clone/contiguous wrappers."""
  is_neg = False
  curr = node
  while isinstance(curr, torch.fx.Node) and curr.op == "call_function":
    if (
        curr.target in (torch.ops.aten.neg.default, torch.ops.aten.neg_.default)
        and curr.args
        and isinstance(curr.args[0], torch.fx.Node)
    ):
      curr = curr.args[0]
      is_neg = not is_neg
    elif (
        curr.target
        in (
            torch.ops.aten.clone.default,
            torch.ops.aten.contiguous.default,
            torch.ops.aten.copy.default,
        )
        and curr.args
        and isinstance(curr.args[0], torch.fx.Node)
    ):
      curr = curr.args[0]
    else:
      break

  if not isinstance(curr, torch.fx.Node) or curr.op != "call_function":
    return None

  if curr.target not in (
      torch.ops.aten.slice.Tensor,
      torch.ops.aten.slice_copy.Tensor,
  ):
    return None

  if not curr.args:
    return None

  base = curr.args[0]
  dim = curr.args[1] if len(curr.args) > 1 else curr.kwargs.get("dim", 0)
  start = curr.args[2] if len(curr.args) > 2 else curr.kwargs.get("start", 0)
  end = (
      curr.args[3]
      if len(curr.args) > 3
      else curr.kwargs.get("end", 9223372036854775807)
  )
  step = curr.args[4] if len(curr.args) > 4 else curr.kwargs.get("step", 1)

  return (curr, base, dim, start, end, step, is_neg)


def _get_node_shape(node: torch.fx.Node) -> tuple[int, ...] | None:
  for meta_key in ("example_value", "val", "tensor_meta"):
    meta_val = node.meta.get(meta_key)
    if hasattr(meta_val, "shape"):
      return tuple(meta_val.shape)
    elif isinstance(meta_val, (list, tuple)):
      return tuple(meta_val)
  return None


def _populate_val_meta(
    target_node: torch.fx.Node,
    ref_tensor: Any,
) -> None:
  if isinstance(ref_tensor, torch.Tensor):
    target_node.meta["val"] = ref_tensor
    target_node.meta["example_value"] = ref_tensor


def apply(gm_or_graph: torch.fx.GraphModule | torch.fx.Graph) -> bool:
  """Fuses adjacent slice-concat pairs along the minor dimension in FX graphs.

  Args:
    gm_or_graph: The FX GraphModule or Graph to transform.

  Returns:
    True if the graph was modified, False otherwise.
  """
  if not _is_fuse_slice_concat_enabled():
    return False

  gm, graph = _normalize_gm_and_graph(gm_or_graph)
  modified = False

  for node in list(graph.nodes):
    if node.op != "call_function" or node.target not in (
        torch.ops.aten.cat.default,
        torch.ops.aten.concat.default,
    ):
      continue

    if not node.args or not isinstance(node.args[0], (list, tuple)):
      continue

    tensor_list = node.args[0]
    if len(tensor_list) != 2:
      continue

    t0, t1 = tensor_list
    if not isinstance(t0, torch.fx.Node) or not isinstance(t1, torch.fx.Node):
      continue

    info0 = _extract_slice_info(t0)
    info1 = _extract_slice_info(t1)
    if not info0 or not info1:
      continue

    _, base0, dim0, start0, end0, step0, neg0 = info0
    _, base1, dim1, start1, end1, step1, neg1 = info1

    effective_base0 = _unwrap_identity_views(base0)
    effective_base1 = _unwrap_identity_views(base1)
    if effective_base0 is not effective_base1:
      continue

    if step0 != 1 or step1 != 1:
      continue

    shape = _get_node_shape(base0) or _get_node_shape(effective_base0)
    if shape is None:
      shape = _get_node_shape(node)

    if shape is None:
      continue

    rank = len(shape)
    if rank < 1:
      continue

    raw_cat_dim = (
        node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
    )
    if not isinstance(raw_cat_dim, int):
      continue

    cat_dim = raw_cat_dim + rank if raw_cat_dim < 0 else raw_cat_dim
    norm_dim0 = dim0 + rank if dim0 < 0 else dim0
    norm_dim1 = dim1 + rank if dim1 < 0 else dim1

    if cat_dim != norm_dim0 or cat_dim != norm_dim1:
      continue

    # Match minor dimension: rank - 1 (including canonical dim 3 on 4D tensors)
    if cat_dim != rank - 1 and not (rank == 4 and cat_dim == 3):
      continue

    d_size = shape[cat_dim]
    if not isinstance(d_size, int) or d_size % 2 != 0 or d_size <= 0:
      continue

    half = d_size // 2

    is_identity = (
        (start0 in (0, None) and end0 == half)
        and (start1 == half and end1 in (d_size, None, 9223372036854775807))
        and not neg0
        and not neg1
    )
    if is_identity:
      node.replace_all_uses_with(effective_base0)
      graph.erase_node(node)
      modified = True
      continue

    is_rope = (
        (start0 == half and end0 in (d_size, None, 9223372036854775807))
        and (start1 in (0, None) and end1 == half)
        and neg0
        and not neg1
    )

    is_adjacent_swapped = (
        start0 == half and end0 in (d_size, None, 9223372036854775807)
    ) and (start1 in (0, None) and end1 == half)

    is_ordered_negated = (
        (start0 in (0, None) and end0 == half)
        and (start1 == half and end1 in (d_size, None, 9223372036854775807))
        and (neg0 or neg1)
    )

    if not is_rope and not is_adjacent_swapped and not is_ordered_negated:
      continue

    reshaped_shape = list(shape[:-1]) + [2, half]
    with graph.inserting_before(node):
      reshaped = graph.call_function(
          torch.ops.aten.view.default, args=(effective_base0, reshaped_shape)
      )

      idx0 = 0 if (start0 in (0, None)) else 1
      idx1 = 0 if (start1 in (0, None)) else 1

      s0 = graph.call_function(
          torch.ops.aten.slice.Tensor,
          args=(reshaped, rank - 1, idx0, idx0 + 1, 1),
      )
      if neg0:
        s0 = graph.call_function(torch.ops.aten.neg.default, args=(s0,))

      s1 = graph.call_function(
          torch.ops.aten.slice.Tensor,
          args=(reshaped, rank - 1, idx1, idx1 + 1, 1),
      )
      if neg1:
        s1 = graph.call_function(torch.ops.aten.neg.default, args=(s1,))

      fused_cat = graph.call_function(
          torch.ops.aten.cat.default, args=([s0, s1], rank - 1)
      )
      out = graph.call_function(
          torch.ops.aten.view.default, args=(fused_cat, list(shape))
      )

      # Propagate node metadata
      if hasattr(node, "meta"):
        out.meta = node.meta.copy()

      ref_meta = effective_base0.meta.get("val")
      if ref_meta is None:
        ref_meta = effective_base0.meta.get("example_value")
      if isinstance(ref_meta, torch.Tensor):
        try:
          reshaped_val = ref_meta.view(reshaped_shape)
          _populate_val_meta(reshaped, reshaped_val)
          s0_val = reshaped_val.narrow(rank - 1, idx0, 1)
          if neg0:
            s0_val = -s0_val
          _populate_val_meta(s0, s0_val)
          s1_val = reshaped_val.narrow(rank - 1, idx1, 1)
          if neg1:
            s1_val = -s1_val
          _populate_val_meta(s1, s1_val)
          fused_cat_val = torch.cat([s0_val, s1_val], dim=rank - 1)
          _populate_val_meta(fused_cat, fused_cat_val)
        except Exception:  # pylint: disable=broad-exception-caught
          pass

      node.replace_all_uses_with(out)
      graph.erase_node(node)
      modified = True

  if modified:
    graph.eliminate_dead_code()
    graph.lint()
    if gm is not None:
      for _, submod in gm.named_children():
        if isinstance(submod, torch.fx.GraphModule):
          apply(submod)
      gm.recompile()

  return modified
