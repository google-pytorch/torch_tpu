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

"""FX pass to fuse Rotary Position Embedding (RoPE) into distributive split form.

In standard Hugging Face transformer models (LLaMA, Gemma, Qwen, Mistral),
`apply_rotary_pos_emb` computes RoPE via:
  `x1 = slice(x, -1, 0, D/2)`
  `x2 = slice(x, -1, D/2, D)`
  `neg_x2 = neg(x2)`
  `rot_x = cat([neg_x2, x1], -1)`
  `out = (x * cos) + (rot_x * sin)`

This materializes a full-size `rotate_half` tensor allocation and intermediate
stride manipulation, which incurs unnecessary memory bandwidth and impedes
loop fusion.

This pass rewrites the pattern into:
  `x1 = slice(x, -1, 0, D/2)`
  `x2 = slice(x, -1, D/2, D)`
  `c1 = slice(cos, -1, 0, D/2)`
  `s1 = slice(sin, -1, 0, D/2)`
  `y1 = (x1 * c1) - (x2 * s1)`
  `y2 = (x2 * c1) + (x1 * s1)`
  `out = cat([y1, y2], -1)`

Eliminating `rotate_half` avoids the intermediate full-size tensor allocation
and enables XLA loop fusion across the elementwise expressions.
"""

import dataclasses
import os
from typing import Any, Callable, cast
import torch
import torch.fx

_SLICE_TARGETS = frozenset({
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.slice_copy.Tensor,
})

_MUL_TARGETS = frozenset({
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.mul.default,
})

_ADD_TARGETS = frozenset({
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add.Scalar,
    torch.ops.aten.add.default,
})

_SUB_TARGETS = frozenset({
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.sub.Scalar,
    torch.ops.aten.sub.default,
})

_CAT_TARGETS = frozenset({
    torch.ops.aten.cat.default,
})

_NEG_TARGETS = frozenset({
    torch.ops.aten.neg.default,
})


def _is_fuse_rope_enabled() -> bool:
  """Returns whether the pass is enabled via environment variable."""
  return os.environ.get(
      "TORCH_TPU_INTERNAL_ENABLE_FUSE_ROPE", "1"
  ).lower() not in ("0", "false", "no")


def _normalize_gm_and_graph(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> tuple[torch.fx.GraphModule | None, torch.fx.Graph]:
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    return gm_or_graph, gm_or_graph.graph
  return getattr(gm_or_graph, "owning_module", None), gm_or_graph


def _is_op(
    node: Any, targets: frozenset[Any], overloadpacket: Any | None = None
) -> bool:
  if not isinstance(node, torch.fx.Node) or node.op != "call_function":
    return False
  if node.target in targets:
    return True
  if overloadpacket is not None:
    return getattr(node.target, "overloadpacket", None) == overloadpacket
  return False


def _is_slice(node: Any) -> bool:
  return _is_op(node, _SLICE_TARGETS) or getattr(
      node.target, "overloadpacket", None
  ) in (
      torch.ops.aten.slice,
      torch.ops.aten.slice_copy,
  )


def _is_mul(node: Any) -> bool:
  return _is_op(node, _MUL_TARGETS, torch.ops.aten.mul)


def _is_add(node: Any) -> bool:
  return _is_op(node, _ADD_TARGETS, torch.ops.aten.add)


def _is_sub(node: Any) -> bool:
  return _is_op(node, _SUB_TARGETS, torch.ops.aten.sub)


def _is_cat(node: Any) -> bool:
  if _is_op(node, _CAT_TARGETS, torch.ops.aten.cat):
    return True
  if isinstance(node, torch.fx.Node) and node.target == torch.cat:
    return True
  return False


def _is_neg(node: Any) -> bool:
  return _is_op(node, _NEG_TARGETS, torch.ops.aten.neg)


def _extract_slice_info(
    node: torch.fx.Node,
) -> tuple[torch.fx.Node, int | Any, Any, Any, Any] | None:
  """Extracts (input_node, dim, start, end, step) from a slice node."""
  if not _is_slice(node):
    return None

  args = node.args
  kwargs = node.kwargs

  inp = (
      args[0]
      if len(args) > 0
      else kwargs.get("input", kwargs.get("self", None))
  )
  if not isinstance(inp, torch.fx.Node):
    return None

  dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
  start = args[2] if len(args) > 2 else kwargs.get("start", 0)
  end = args[3] if len(args) > 3 else kwargs.get("end", None)
  step = args[4] if len(args) > 4 else kwargs.get("step", 1)

  if start is None:
    start = 0
  if step is None:
    step = 1

  return inp, dim, start, end, step


def _is_neg_of(node: Any, target_x: torch.fx.Node) -> bool:
  """Checks whether node represents the negation of target_x."""
  if not isinstance(node, torch.fx.Node):
    return False
  if _is_neg(node):
    return bool(len(node.args) >= 1 and node.args[0] == target_x)
  if _is_mul(node):
    if len(node.args) >= 2:
      a0, a1 = node.args[0], node.args[1]
      if a0 == target_x and isinstance(a1, (int, float)) and a1 == -1:
        return True
      if a1 == target_x and isinstance(a0, (int, float)) and a0 == -1:
        return True
  if _is_sub(node):
    if len(node.args) >= 2:
      a0, a1 = node.args[0], node.args[1]
      if a1 == target_x and isinstance(a0, (int, float)) and a0 == 0:
        return True
  return False


def _extract_cat_info(
    node: torch.fx.Node,
) -> tuple[list[torch.fx.Node], int | Any] | None:
  """Extracts (tensor_list, dim) from a cat node."""
  if not _is_cat(node):
    return None

  args = node.args
  kwargs = node.kwargs

  tensors = args[0] if len(args) > 0 else kwargs.get("tensors", None)
  if not isinstance(tensors, (list, tuple)):
    return None

  dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
  return list(tensors), dim


def _dims_match(
    dim1: Any, dim2: Any, ref_node: torch.fx.Node | None = None
) -> bool:
  """Checks whether two dimension indices refer to the same dimension."""
  if dim1 == dim2:
    return True
  if isinstance(dim1, int) and isinstance(dim2, int):
    rank = None
    if ref_node is not None and hasattr(ref_node, "meta"):
      val = ref_node.meta.get("val", ref_node.meta.get("example_value", None))
      if isinstance(val, torch.Tensor):
        rank = val.ndim
    if rank is not None:
      d1_norm = dim1 + rank if dim1 < 0 else dim1
      d2_norm = dim2 + rank if dim2 < 0 else dim2
      return d1_norm == d2_norm
    if (dim1 == -1 and dim2 >= 0) or (dim2 == -1 and dim1 >= 0):
      # Without rank information, assume -1 is the last dimension.
      return False
  return False


def _get_dim_size(node: torch.fx.Node, dim: Any) -> int | None:
  """Returns dimension size from node metadata if available."""
  if hasattr(node, "meta") and isinstance(dim, int):
    val = node.meta.get("val", node.meta.get("example_value", None))
    if isinstance(val, torch.Tensor):
      d = dim if dim >= 0 else dim + val.ndim
      if 0 <= d < val.ndim:
        size = val.shape[d]
        if isinstance(size, int):
          return size
  return None


def _is_equal_halves(
    x: torch.fx.Node,
    x1: torch.fx.Node,
    x2: torch.fx.Node,
    dim: Any,
    end1: Any,
    start2: Any,
    end2: Any,
) -> bool:
  """Verifies that x1 and x2 represent equal halves of x along dim."""
  # 1. If tensor shapes are known in metadata, verify lengths match
  x1_len = _get_dim_size(x1, dim)
  x2_len = _get_dim_size(x2, dim)
  if x1_len is not None and x2_len is not None:
    if x1_len != x2_len:
      return False

  x_len = _get_dim_size(x, dim)
  if x_len is not None and isinstance(end1, int):
    if end1 * 2 != x_len:
      return False

  # 2. If end2 is an explicit finite integer bound
  if (
      isinstance(end1, int)
      and isinstance(start2, int)
      and isinstance(end2, int)
      and end2 not in (9223372036854775807, 0x7FFFFFFFFFFFFFFF)
  ):
    if end2 - start2 != end1:
      return False

  return True


@dataclasses.dataclass(frozen=True)
class RotateHalfMatch:
  rot_node: torch.fx.Node
  x: torch.fx.Node
  x1: torch.fx.Node
  x2: torch.fx.Node
  dim: Any
  half_dim: Any
  is_standard: bool


def _match_rotate_half(node: torch.fx.Node) -> RotateHalfMatch | None:
  """Matches rotate_half(x) = cat([-x2, x1], dim=-1) or cat([x2, -x1], dim=-1)."""
  cat_info = _extract_cat_info(node)
  if cat_info is None:
    return None

  tensors, cat_dim = cat_info
  if len(tensors) != 2:
    return None

  elem0, elem1 = tensors[0], tensors[1]

  # Case 1: Standard RoPE: elem0 = -x2, elem1 = x1
  if isinstance(elem1, torch.fx.Node):
    slice1_info = _extract_slice_info(elem1)
    if slice1_info is not None:
      x, dim1, start1, end1, step1 = slice1_info
      if (
          start1 == 0
          and step1 == 1
          and end1 is not None
          and _dims_match(cat_dim, dim1, x)
      ):
        # Find x2 candidate: elem0 is neg of x2
        for x2_candidate in node.graph.nodes:
          if not _is_slice(x2_candidate):
            continue
          if not _is_neg_of(elem0, x2_candidate):
            continue
          slice2_info = _extract_slice_info(x2_candidate)
          if slice2_info is None:
            continue
          x_2, dim2, start2, end2, step2 = slice2_info
          if (
              x_2 == x
              and _dims_match(dim1, dim2, x)
              and step2 == 1
              and start2 == end1
              and _is_equal_halves(
                  x, elem1, x2_candidate, dim1, end1, start2, end2
              )
          ):
            return RotateHalfMatch(
                rot_node=node,
                x=x,
                x1=elem1,
                x2=x2_candidate,
                dim=cat_dim,
                half_dim=end1,
                is_standard=True,
            )

  # Case 2: Alternative RoPE: elem0 = x2, elem1 = -x1
  if isinstance(elem0, torch.fx.Node):
    slice2_info = _extract_slice_info(elem0)
    if slice2_info is not None:
      x_2, dim2, start2, end2, step2 = slice2_info
      if step2 == 1 and start2 is not None and _dims_match(cat_dim, dim2, x_2):
        for x1_candidate in node.graph.nodes:
          if not _is_slice(x1_candidate):
            continue
          if not _is_neg_of(elem1, x1_candidate):
            continue
          slice1_info = _extract_slice_info(x1_candidate)
          if slice1_info is None:
            continue
          x, dim1, start1, end1, step1 = slice1_info
          if (
              x == x_2
              and _dims_match(dim1, dim2, x)
              and step1 == 1
              and start1 == 0
              and end1 == start2
              and _is_equal_halves(
                  x, x1_candidate, elem0, dim1, end1, start2, end2
              )
          ):
            return RotateHalfMatch(
                rot_node=node,
                x=x,
                x1=x1_candidate,
                x2=elem0,
                dim=cat_dim,
                half_dim=end1,
                is_standard=False,
            )

  return None


@dataclasses.dataclass(frozen=True)
class RoPEMatch:
  out_node: torch.fx.Node
  term_x: torch.fx.Node
  term_rot: torch.fx.Node
  rot_match: RotateHalfMatch
  cos: torch.fx.Node
  sin: torch.fx.Node


def _match_rope(node: torch.fx.Node) -> RoPEMatch | None:
  """Matches out = (x * cos) + (rot_x * sin) or equivalent."""
  if not _is_add(node) and not _is_sub(node):
    return None

  if len(node.args) < 2:
    return None

  arg0, arg1 = node.args[0], node.args[1]
  if not isinstance(arg0, torch.fx.Node) or not isinstance(arg1, torch.fx.Node):
    return None

  # For add, (term0 + term1) or (term1 + term0)
  # For sub, (term0 - term1)
  candidate_pairs = [(arg0, arg1)]
  if _is_add(node):
    candidate_pairs.append((arg1, arg0))

  for term_x_cand, term_rot_cand in candidate_pairs:
    if not _is_mul(term_x_cand) or not _is_mul(term_rot_cand):
      continue

    if len(term_rot_cand.args) < 2 or len(term_x_cand.args) < 2:
      continue

    # Identify rot_x and sin from term_rot
    rot_rot_args = [
        (term_rot_cand.args[0], term_rot_cand.args[1]),
        (term_rot_cand.args[1], term_rot_cand.args[0]),
    ]
    for rot_cand, sin_cand in rot_rot_args:
      if not isinstance(rot_cand, torch.fx.Node) or not isinstance(
          sin_cand, torch.fx.Node
      ):
        continue

      rot_match = _match_rotate_half(rot_cand)
      if rot_match is None:
        continue

      # If node is add, require standard rotate_half: cat([-x2, x1])
      # If node is sub, require alternative rotate_half: cat([x2, -x1])
      if _is_add(node) and not rot_match.is_standard:
        continue
      if _is_sub(node) and rot_match.is_standard:
        continue

      # Now verify term_x_cand multiplies rot_match.x and cos
      x_node = rot_match.x
      x_mul_args = [
          (term_x_cand.args[0], term_x_cand.args[1]),
          (term_x_cand.args[1], term_x_cand.args[0]),
      ]
      for x_cand, cos_cand in x_mul_args:
        if x_cand == x_node and isinstance(cos_cand, torch.fx.Node):
          return RoPEMatch(
              out_node=node,
              term_x=term_x_cand,
              term_rot=term_rot_cand,
              rot_match=rot_match,
              cos=cos_cand,
              sin=sin_cand,
          )

  return None


def _copy_meta(new_node: torch.fx.Node, source_node: torch.fx.Node) -> None:
  """Copies metadata from source_node to new_node."""
  if hasattr(source_node, "meta"):
    new_node.meta = source_node.meta.copy()


def _set_slice_meta(
    new_node: torch.fx.Node,
    source_node: torch.fx.Node,
    dim: Any,
    start: Any,
    end: Any,
) -> None:
  """Copies metadata and slices FakeTensor/Tensor values along dim."""
  if not hasattr(source_node, "meta"):
    return
  new_node.meta = source_node.meta.copy()
  for key in ("val", "example_value"):
    val = source_node.meta.get(key)
    if isinstance(val, torch.Tensor):
      try:
        dim_idx = dim if isinstance(dim, int) else -1
        slices = [slice(None)] * val.ndim
        slices[dim_idx] = slice(start, end)
        new_node.meta[key] = val[tuple(slices)]
      except Exception:
        pass


def apply(gm_or_graph: torch.fx.GraphModule | torch.fx.Graph) -> None:
  """Rewrites Hugging Face RoPE pattern into fused distributive split form.

  Replaces:
    `out = (x * cos) + (rot_x * sin)` where `rot_x = cat([-x2, x1], dim)`
  with:
    `c1 = slice(cos, dim, 0, half_dim)`
    `s1 = slice(sin, dim, 0, half_dim)`
    `y1 = (x1 * c1) - (x2 * s1)`
    `y2 = (x2 * c1) + (x1 * s1)`
    `out = cat([y1, y2], dim)`

  Args:
    gm_or_graph: The FX GraphModule or Graph to transform.
  """
  if not _is_fuse_rope_enabled():
    return

  gm, graph = _normalize_gm_and_graph(gm_or_graph)

  matches: list[RoPEMatch] = []
  for node in graph.nodes:
    if node.op != "call_function":
      continue
    match = _match_rope(node)
    if match is not None:
      matches.append(match)

  if not matches:
    return

  # Cache sliced cos and sin to avoid redundant slices when multiple heads or
  # queries/keys share cos/sin.
  slice_cache: dict[tuple[torch.fx.Node, Any, Any], torch.fx.Node] = {}

  for match in matches:
    out_node = match.out_node
    if out_node.graph is None:
      continue

    rot_match = match.rot_match
    x1 = rot_match.x1
    x2 = rot_match.x2
    dim = rot_match.dim
    half_dim = rot_match.half_dim
    cos = match.cos
    sin = match.sin
    term_x = match.term_x

    # Select appropriate op overloads matching the existing graph style.
    if (
        x1.target == torch.ops.aten.slice_copy.Tensor
        or getattr(x1.target, "overloadpacket", None)
        == torch.ops.aten.slice_copy
    ):
      slice_op = torch.ops.aten.slice_copy.Tensor
    else:
      slice_op = torch.ops.aten.slice.Tensor

    if term_x.target in (
        torch.ops.aten.mul.default,
        torch.ops.aten.mul.Tensor,
    ):
      mul_op = term_x.target
    else:
      mul_op = torch.ops.aten.mul.Tensor

    if mul_op == torch.ops.aten.mul.default:
      sub_op = torch.ops.aten.sub.default
      add_op = torch.ops.aten.add.default
    else:
      sub_op = torch.ops.aten.sub.Tensor
      add_op = torch.ops.aten.add.Tensor

    cat_op = getattr(rot_match.rot_node, "target", torch.ops.aten.cat.default)

    with graph.inserting_before(out_node):
      # Slices for cos and sin
      cos_key = (cos, dim, half_dim)
      if cos_key in slice_cache:
        c1 = slice_cache[cos_key]
      else:
        c1 = graph.call_function(
            cast(Callable[..., Any], slice_op), args=(cos, dim, 0, half_dim)
        )
        _set_slice_meta(c1, cos, dim, 0, half_dim)
        slice_cache[cos_key] = c1

      sin_key = (sin, dim, half_dim)
      if sin_key in slice_cache:
        s1 = slice_cache[sin_key]
      else:
        s1 = graph.call_function(
            cast(Callable[..., Any], slice_op), args=(sin, dim, 0, half_dim)
        )
        _set_slice_meta(s1, sin, dim, 0, half_dim)
        slice_cache[sin_key] = s1

      # y1 = (x1 * c1) - (x2 * s1)
      x1_c1 = graph.call_function(
          cast(Callable[..., Any], mul_op), args=(x1, c1)
      )
      _copy_meta(x1_c1, x1)

      x2_s1 = graph.call_function(
          cast(Callable[..., Any], mul_op), args=(x2, s1)
      )
      _copy_meta(x2_s1, x2)

      y1 = graph.call_function(
          cast(Callable[..., Any], sub_op), args=(x1_c1, x2_s1)
      )
      _copy_meta(y1, x1)

      # y2 = (x2 * c1) + (x1 * s1)
      x2_c1 = graph.call_function(
          cast(Callable[..., Any], mul_op), args=(x2, c1)
      )
      _copy_meta(x2_c1, x2)

      x1_s1 = graph.call_function(
          cast(Callable[..., Any], mul_op), args=(x1, s1)
      )
      _copy_meta(x1_s1, x1)

      y2 = graph.call_function(
          cast(Callable[..., Any], add_op), args=(x2_c1, x1_s1)
      )
      _copy_meta(y2, x2)

      # out = cat([y1, y2], dim)
      new_out = graph.call_function(
          cast(Callable[..., Any], cat_op), args=([y1, y2], dim)
      )
      _copy_meta(new_out, out_node)

    out_node.replace_all_uses_with(new_out)
    graph.erase_node(out_node)

  graph.eliminate_dead_code()
  graph.lint()
  if gm is not None:
    for _, submod in gm.named_children():
      if isinstance(submod, torch.fx.GraphModule):
        apply(submod)
    gm.recompile()
