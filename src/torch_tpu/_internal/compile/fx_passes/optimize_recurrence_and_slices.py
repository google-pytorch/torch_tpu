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

"""FX optimization pass for linear recurrences, slice/select backward ops, and activation rematerialization.

This pass performs three complementary graph optimizations on FX graphs:
1. Fuses `aten.select_backward` and `aten.slice_backward` addition trees into
   batched `aten.stack`, `aten.cat`, or strided `stack + flatten` operations,
   and simplifies standalone `select_backward`/`slice_backward` ops into
   views (`unsqueeze`) or `cat` with zero padding. This eliminates O(L)
   full-tensor zero allocations and scatter operations in backward passes.
2. Vectorizes unrolled first-order linear recurrences (both forward and reverse
   scans of the form `h_t = a_t * h_{t-1} + b_t`) into O(log L) parallel prefix
   scans (Kogge-Stone doubling) with step-level `optimization_barrier`s to
   enforce ping-pong buffer reuse across doubling steps, and vectorizes
   cross-correlation decay gradient stacks into batched `mul + sum` reductions.
3. Rematerializes high-rank broadcast intermediates computed in the forward pass
   and saved for the backward pass when their producer subgraph consists of
   cheap elementwise/view/padding operations on much smaller frontier tensors,
   anchored on a 4-byte scalar dependency token from the backward pass via
   `torch.ops.tpu.optimization_barrier`.
"""

import collections
import operator
from typing import Any, cast
import torch
from torch import fx


def _register_cpu_optimization_barrier() -> None:
  if not hasattr(torch.ops, "tpu") or not hasattr(
      torch.ops.tpu, "optimization_barrier"
  ):
    return
  try:

    @torch.library.impl("tpu::optimization_barrier", "CPU")
    def _optimization_barrier_cpu(
        inputs: list[torch.Tensor],
    ) -> list[torch.Tensor]:
      return list(inputs)

  except Exception:  # pylint: disable=broad-exception-caught
    pass


_register_cpu_optimization_barrier()

_CHEAP_REMAT_OPS = {
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add.Scalar,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.sub.Scalar,
    torch.ops.aten.rsub.Scalar,
    torch.ops.aten.rsub.Tensor,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.div.Scalar,
    torch.ops.aten.neg.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.constant_pad_nd.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.clone.default,
}


def _emit_node(
    graph: fx.Graph,
    target: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> fx.Node:
  """Emits a call_function node and propagates FakeTensor metadata."""
  kwargs = kwargs or {}
  node = graph.call_function(target, args, kwargs)
  try:

    def _get_val(x: Any) -> Any:
      if isinstance(x, fx.Node):
        return x.meta.get("val", None)
      if isinstance(x, (list, tuple)):
        vals = [_get_val(i) for i in x]
        return type(x)(vals) if all(v is not None for v in vals) else None
      return x

    fake_args = [_get_val(a) for a in args]
    fake_kwargs = {k: _get_val(v) for k, v in kwargs.items()}
    if all(
        a is not None or not isinstance(orig, fx.Node)
        for a, orig in zip(fake_args, args)
    ):
      node.meta["val"] = target(*fake_args, **fake_kwargs)
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  return node


def _is_zeros_node(node: Any) -> bool:
  """Returns True if node is an aten.zeros or aten.zeros_like call."""
  return isinstance(node, fx.Node) and node.target in (
      torch.ops.aten.zeros.default,
      torch.ops.aten.zeros_like.default,
      torch.ops.aten.new_zeros.default,
  )


def _collect_add_tree_leaves(
    root: fx.Node,
) -> tuple[list[fx.Node], list[fx.Node]]:
  """Collects additive leaves and internal add nodes from an addition tree."""
  leaves: list[fx.Node] = []
  internal_nodes: list[fx.Node] = []
  stack = [root]

  while stack:
    n = stack.pop()
    n_args: Any = n.args
    if (
        n.op == "call_function"
        and n.target == torch.ops.aten.add.Tensor
        and (n is root or len(n.users) == 1)
        and len(n_args) >= 2
        and isinstance(n_args[0], fx.Node)
        and isinstance(n_args[1], fx.Node)
    ):
      internal_nodes.append(n)
      stack.append(n_args[1])
      stack.append(n_args[0])
    else:
      leaves.append(n)

  return leaves, internal_nodes


def _fuse_select_and_slice_backward(graph: fx.Graph) -> bool:
  """Fuses select_backward and slice_backward trees and simplifies standalone ops."""
  modified = False

  # 1. Process addition trees containing select_backward or slice_backward leaves.
  for node in list(graph.nodes):
    if node.op != "call_function" or node.target != torch.ops.aten.add.Tensor:
      continue
    # Skip internal nodes of a larger single-use addition tree.
    if (
        len(node.users) == 1
        and next(iter(node.users)).target == torch.ops.aten.add.Tensor
    ):
      continue

    leaves, _ = _collect_add_tree_leaves(node)
    if len(leaves) < 2:
      continue

    select_groups: dict[
        tuple[tuple[int, ...], int], list[tuple[int, fx.Node, fx.Node]]
    ] = collections.defaultdict(list)
    slice_groups: dict[
        tuple[tuple[int, ...], int, int],
        list[tuple[int, int, fx.Node, fx.Node]],
    ] = collections.defaultdict(list)
    other_leaves: list[fx.Node] = []

    for leaf in leaves:
      leaf_args: Any = leaf.args
      if (
          leaf.op == "call_function"
          and leaf.target == torch.ops.aten.select_backward.default
          and len(leaf_args) >= 4
      ):
        grad, input_sizes, dim, index = leaf_args[:4]
        input_sizes_list = [int(s) for s in input_sizes]
        ndim = len(input_sizes_list)
        dim_int = int(dim)
        idx_int = int(index)
        norm_dim = dim_int if dim_int >= 0 else dim_int + ndim
        dim_size = input_sizes_list[norm_dim]
        norm_idx = idx_int if idx_int >= 0 else idx_int + dim_size
        select_groups[(tuple(input_sizes_list), norm_dim)].append(
            (norm_idx, cast(fx.Node, grad), leaf)
        )
      elif (
          leaf.op == "call_function"
          and leaf.target == torch.ops.aten.slice_backward.default
          and len(leaf_args) >= 6
      ):
        grad, input_sizes, dim, start, end, step = leaf_args[:6]
        input_sizes_list = [int(s) for s in input_sizes]
        ndim = len(input_sizes_list)
        dim_int = int(dim)
        start_int = int(start)
        end_int = int(end)
        step_int = int(step)
        norm_dim = dim_int if dim_int >= 0 else dim_int + ndim
        dim_size = input_sizes_list[norm_dim]
        norm_start = max(
            0, start_int if start_int >= 0 else start_int + dim_size
        )
        norm_end = min(
            dim_size, end_int if end_int >= 0 else end_int + dim_size
        )
        slice_groups[(tuple(input_sizes_list), norm_dim, step_int)].append(
            (norm_start, norm_end, cast(fx.Node, grad), leaf)
        )
      else:
        other_leaves.append(leaf)

    replacement_terms: list[fx.Node] = []
    tree_changed = False

    with graph.inserting_before(node):
      # Handle select_backward groups
      for (input_sizes, norm_dim), items in select_groups.items():
        if len(items) < 2:
          for _, _, leaf in items:
            other_leaves.append(leaf)
          continue

        dim_size = input_sizes[norm_dim]
        slots: list[fx.Node | None] = [None] * dim_size
        ref_grad = items[0][1]
        for norm_idx, grad, _ in items:
          if 0 <= norm_idx < dim_size:
            if slots[norm_idx] is None:
              slots[norm_idx] = grad
            else:
              slots[norm_idx] = _emit_node(
                  graph, torch.ops.aten.add.Tensor, (slots[norm_idx], grad)
              )

        slot_shape = list(input_sizes[:norm_dim]) + list(
            input_sizes[norm_dim + 1 :]
        )
        zero_slot = None
        for i in range(dim_size):
          if slots[i] is None:
            if zero_slot is None:
              zero_slot = _emit_node(
                  graph,
                  torch.ops.aten.new_zeros.default,
                  (ref_grad, slot_shape),
              )
            slots[i] = zero_slot

        stacked = _emit_node(
            graph, torch.ops.aten.stack.default, (slots, norm_dim)
        )
        replacement_terms.append(stacked)
        tree_changed = True

      # Handle slice_backward groups
      for (input_sizes, norm_dim, step), slice_items in slice_groups.items():
        if len(slice_items) < 2:
          for _, _, _, leaf in slice_items:
            other_leaves.append(leaf)
          continue

        dim_size = input_sizes[norm_dim]
        slice_items.sort(key=lambda x: x[0])

        # Case A: Strided interleaved slices (e.g. step > 1 covering all mod offsets)
        if (
            step > 1
            and len(slice_items) == step
            and dim_size % step == 0
            and all(slice_items[r][0] == r for r in range(step))
        ):
          grads = [it[2] for it in slice_items]
          stacked = _emit_node(
              graph, torch.ops.aten.stack.default, (grads, norm_dim + 1)
          )
          flattened = _emit_node(
              graph,
              torch.ops.aten.flatten.using_ints,
              (stacked, norm_dim, norm_dim + 1),
          )
          replacement_terms.append(flattened)
          tree_changed = True
          continue

        # Case B: Contiguous non-overlapping slices (step == 1)
        if step == 1:
          pieces: list[fx.Node] = []
          curr_pos = 0
          ref_grad = slice_items[0][2]
          valid_partition = True
          for norm_start, norm_end, grad, _ in slice_items:
            if norm_start < curr_pos:
              valid_partition = False
              break
            if norm_start > curr_pos:
              pad_shape = list(input_sizes)
              pad_shape[norm_dim] = norm_start - curr_pos
              pieces.append(
                  _emit_node(
                      graph,
                      torch.ops.aten.new_zeros.default,
                      (ref_grad, pad_shape),
                  )
              )
            pieces.append(grad)
            curr_pos = norm_end

          if valid_partition:
            if curr_pos < dim_size:
              pad_shape = list(input_sizes)
              pad_shape[norm_dim] = dim_size - curr_pos
              pieces.append(
                  _emit_node(
                      graph,
                      torch.ops.aten.new_zeros.default,
                      (ref_grad, pad_shape),
                  )
              )
            cat_node = _emit_node(
                graph, torch.ops.aten.cat.default, (pieces, norm_dim)
            )
            replacement_terms.append(cat_node)
            tree_changed = True
            continue

        for _, _, _, leaf in slice_items:
          other_leaves.append(leaf)

      if tree_changed:
        all_terms = replacement_terms + other_leaves
        acc = all_terms[0]
        for term in all_terms[1:]:
          acc = _emit_node(graph, torch.ops.aten.add.Tensor, (acc, term))
        node.replace_all_uses_with(acc)
        modified = True

  if modified:
    graph.eliminate_dead_code()

  # 2. Simplify any remaining standalone select_backward and slice_backward nodes.
  for node in list(graph.nodes):
    if node.op != "call_function":
      continue

    n_args: Any = node.args
    if (
        node.target == torch.ops.aten.select_backward.default
        and len(n_args) >= 4
    ):
      grad, input_sizes, dim, index = n_args[:4]
      input_sizes_list = [int(s) for s in input_sizes]
      ndim = len(input_sizes_list)
      dim_int = int(dim)
      idx_int = int(index)
      norm_dim = dim_int if dim_int >= 0 else dim_int + ndim
      dim_size = input_sizes_list[norm_dim]
      norm_idx = idx_int if idx_int >= 0 else idx_int + dim_size

      with graph.inserting_before(node):
        unsq = _emit_node(
            graph, torch.ops.aten.unsqueeze.default, (grad, norm_dim)
        )
        if dim_size == 1:
          node.replace_all_uses_with(unsq)
        else:
          pieces_sel: list[fx.Node] = []
          if norm_idx > 0:
            pre_shape = list(input_sizes_list)
            pre_shape[norm_dim] = norm_idx
            pieces_sel.append(
                _emit_node(
                    graph, torch.ops.aten.new_zeros.default, (grad, pre_shape)
                )
            )
          pieces_sel.append(unsq)
          if norm_idx + 1 < dim_size:
            post_shape = list(input_sizes_list)
            post_shape[norm_dim] = dim_size - 1 - norm_idx
            pieces_sel.append(
                _emit_node(
                    graph, torch.ops.aten.new_zeros.default, (grad, post_shape)
                )
            )
          cat_node = _emit_node(
              graph, torch.ops.aten.cat.default, (pieces_sel, norm_dim)
          )
          node.replace_all_uses_with(cat_node)
      modified = True

    elif (
        node.target == torch.ops.aten.slice_backward.default
        and len(n_args) >= 6
    ):
      grad, input_sizes, dim, start, end, step = n_args[:6]
      input_sizes_list = [int(s) for s in input_sizes]
      ndim = len(input_sizes_list)
      dim_int = int(dim)
      start_int = int(start)
      end_int = int(end)
      step_int = int(step)
      norm_dim = dim_int if dim_int >= 0 else dim_int + ndim
      dim_size = input_sizes_list[norm_dim]
      norm_start = max(0, start_int if start_int >= 0 else start_int + dim_size)
      norm_end = min(dim_size, end_int if end_int >= 0 else end_int + dim_size)

      if step_int == 1:
        with graph.inserting_before(node):
          pieces_slc: list[fx.Node] = []
          if norm_start > 0:
            pre_shape = list(input_sizes_list)
            pre_shape[norm_dim] = norm_start
            pieces_slc.append(
                _emit_node(
                    graph, torch.ops.aten.new_zeros.default, (grad, pre_shape)
                )
            )
          pieces_slc.append(cast(fx.Node, grad))
          if norm_end < dim_size:
            post_shape = list(input_sizes_list)
            post_shape[norm_dim] = dim_size - norm_end
            pieces_slc.append(
                _emit_node(
                    graph, torch.ops.aten.new_zeros.default, (grad, post_shape)
                )
            )
          if len(pieces_slc) == 1:
            node.replace_all_uses_with(cast(fx.Node, grad))
          else:
            cat_node = _emit_node(
                graph, torch.ops.aten.cat.default, (pieces_slc, norm_dim)
            )
            node.replace_all_uses_with(cat_node)
        modified = True

  if modified:
    graph.eliminate_dead_code()
  return modified


def _match_select(
    node: Any, expected_dim: int | None, expected_idx: int
) -> tuple[fx.Node, int] | None:
  """Matches aten.select.int(src, dim, idx) and returns (src, norm_dim)."""
  if not isinstance(node, fx.Node) or node.op != "call_function":
    return None
  n_args: Any = node.args
  if node.target != torch.ops.aten.select.int or len(n_args) < 3:
    return None
  src, dim, idx = n_args[0], n_args[1], n_args[2]
  if (
      not isinstance(src, fx.Node)
      or not isinstance(dim, int)
      or not isinstance(idx, int)
  ):
    return None
  val = src.meta.get("val", None)
  if val is not None and hasattr(val, "ndim"):
    ndim = int(val.ndim)
    dim = dim if dim >= 0 else dim + ndim
    dim_size = int(val.shape[dim])
    idx = idx if idx >= 0 else idx + dim_size
  if expected_dim is not None and dim != expected_dim:
    return None
  if idx != expected_idx:
    return None
  return src, dim


def _emit_kogge_stone_fwd(
    graph: fx.Graph,
    a_tensor: fx.Node,
    b_tensor: fx.Node,
    h_init: fx.Node | None,
    dim: int,
    length: int,
) -> fx.Node:
  """Emits a forward Kogge-Stone parallel prefix scan along `dim`."""
  curr_a = a_tensor
  if h_init is not None and not _is_zeros_node(h_init):
    b_0 = _emit_node(graph, torch.ops.aten.select.int, (b_tensor, dim, 0))
    a_0 = _emit_node(graph, torch.ops.aten.select.int, (a_tensor, dim, 0))
    prod_0 = _emit_node(graph, torch.ops.aten.mul.Tensor, (a_0, h_init))
    b_0_new = _emit_node(graph, torch.ops.aten.add.Tensor, (prod_0, b_0))
    b_0_unsq = _emit_node(
        graph, torch.ops.aten.unsqueeze.default, (b_0_new, dim)
    )
    b_rest = _emit_node(
        graph, torch.ops.aten.slice.Tensor, (b_tensor, dim, 1, length)
    )
    curr_b = _emit_node(
        graph, torch.ops.aten.cat.default, ([b_0_unsq, b_rest], dim)
    )
  else:
    curr_b = b_tensor

  d = 1
  while d < length:
    a_hi = _emit_node(
        graph, torch.ops.aten.slice.Tensor, (curr_a, dim, d, length)
    )
    b_lo = _emit_node(
        graph, torch.ops.aten.slice.Tensor, (curr_b, dim, 0, length - d)
    )
    b_hi = _emit_node(
        graph, torch.ops.aten.slice.Tensor, (curr_b, dim, d, length)
    )
    prod = _emit_node(graph, torch.ops.aten.mul.Tensor, (a_hi, b_lo))
    new_b_hi = _emit_node(graph, torch.ops.aten.add.Tensor, (prod, b_hi))
    b_prefix = _emit_node(
        graph, torch.ops.aten.slice.Tensor, (curr_b, dim, 0, d)
    )
    curr_b = _emit_node(
        graph, torch.ops.aten.cat.default, ([b_prefix, new_b_hi], dim)
    )

    if 2 * d < length:
      a_lo = _emit_node(
          graph, torch.ops.aten.slice.Tensor, (curr_a, dim, 0, length - d)
      )
      new_a_hi = _emit_node(graph, torch.ops.aten.mul.Tensor, (a_hi, a_lo))
      a_prefix = _emit_node(
          graph, torch.ops.aten.slice.Tensor, (curr_a, dim, 0, d)
      )
      curr_a = _emit_node(
          graph, torch.ops.aten.cat.default, ([a_prefix, new_a_hi], dim)
      )
    d *= 2

  return curr_b


def _emit_kogge_stone_bwd(
    graph: fx.Graph,
    a_tensor: fx.Node,
    g_tensor: fx.Node,
    dim: int,
    length: int,
) -> fx.Node:
  """Emits a reverse Kogge-Stone parallel prefix scan along `dim`."""
  a_shifted = _emit_node(
      graph, torch.ops.aten.slice.Tensor, (a_tensor, dim, 1, length)
  )
  a_first = _emit_node(
      graph, torch.ops.aten.slice.Tensor, (a_tensor, dim, 0, 1)
  )
  zero_last = _emit_node(graph, torch.ops.aten.zeros_like.default, (a_first,))
  curr_a = _emit_node(
      graph, torch.ops.aten.cat.default, ([a_shifted, zero_last], dim)
  )
  curr_b = g_tensor

  d = 1
  while d < length:
    a_lo = _emit_node(
        graph, torch.ops.aten.slice.Tensor, (curr_a, dim, 0, length - d)
    )
    b_hi = _emit_node(
        graph, torch.ops.aten.slice.Tensor, (curr_b, dim, d, length)
    )
    b_lo = _emit_node(
        graph, torch.ops.aten.slice.Tensor, (curr_b, dim, 0, length - d)
    )
    prod = _emit_node(graph, torch.ops.aten.mul.Tensor, (a_lo, b_hi))
    new_b_lo = _emit_node(graph, torch.ops.aten.add.Tensor, (prod, b_lo))
    b_suffix = _emit_node(
        graph, torch.ops.aten.slice.Tensor, (curr_b, dim, length - d, length)
    )
    curr_b = _emit_node(
        graph, torch.ops.aten.cat.default, ([new_b_lo, b_suffix], dim)
    )

    if 2 * d < length:
      a_hi = _emit_node(
          graph, torch.ops.aten.slice.Tensor, (curr_a, dim, d, length)
      )
      new_a_lo = _emit_node(graph, torch.ops.aten.mul.Tensor, (a_lo, a_hi))
      a_suffix = _emit_node(
          graph, torch.ops.aten.slice.Tensor, (curr_a, dim, length - d, length)
      )
      curr_a = _emit_node(
          graph, torch.ops.aten.cat.default, ([new_a_lo, a_suffix], dim)
      )
    d *= 2

  return curr_b


def _vectorize_linear_recurrence(graph: fx.Graph) -> bool:
  """Vectorizes forward/reverse linear recurrences and decay gradient stacks."""
  modified = False
  fwd_scans: dict[
      fx.Node,
      tuple[fx.Node, fx.Node, fx.Node, fx.Node | None, int, list[fx.Node]],
  ] = {}
  bwd_scans: dict[
      fx.Node, tuple[fx.Node, fx.Node, fx.Node, int, list[fx.Node]]
  ] = {}

  # 1. Detect and vectorize forward and reverse linear recurrence stacks.
  for node in list(graph.nodes):
    n_args: Any = node.args
    if (
        node.op != "call_function"
        or node.target != torch.ops.aten.stack.default
        or len(n_args) < 1
    ):
      continue
    elements: Any = n_args[0]
    dim = int(n_args[1]) if len(n_args) >= 2 else 0
    if not isinstance(elements, (list, tuple)) or len(elements) < 4:
      continue
    if not all(isinstance(el, fx.Node) for el in elements):
      continue
    el_nodes: list[fx.Node] = [cast(fx.Node, el) for el in elements]

    length = len(el_nodes)
    val = node.meta.get("val", None)
    ndim = int(val.ndim) if val is not None and hasattr(val, "ndim") else None
    norm_dim = dim if (dim >= 0 or ndim is None) else dim + ndim

    # Try matching Forward Recurrence: h_t = a_t * h_{t-1} + b_t
    fwd_matched = True
    a_src: fx.Node | None = None
    b_src: fx.Node | None = None
    h_init: fx.Node | None = None

    h_1 = el_nodes[1]
    h_1_args: Any = h_1.args
    if (
        h_1.op == "call_function"
        and h_1.target == torch.ops.aten.add.Tensor
        and len(h_1_args) == 2
    ):
      for mul_cand, b_cand in [
          (h_1_args[0], h_1_args[1]),
          (h_1_args[1], h_1_args[0]),
      ]:
        b_match = _match_select(b_cand, norm_dim, 1)
        if b_match is None:
          continue
        if (
            isinstance(mul_cand, fx.Node)
            and mul_cand.op == "call_function"
            and mul_cand.target == torch.ops.aten.mul.Tensor
            and len(mul_cand.args) == 2
        ):
          mc_args: Any = mul_cand.args
          for a_cand, prev_cand in [
              (mc_args[0], mc_args[1]),
              (mc_args[1], mc_args[0]),
          ]:
            if prev_cand is el_nodes[0]:
              a_match = _match_select(a_cand, norm_dim, 1)
              if a_match is not None:
                a_src, norm_dim = a_match
                b_src, _ = b_match
                break
        if a_src is not None:
          break

    if a_src is not None and b_src is not None:
      h_0 = el_nodes[0]
      h_0_args: Any = h_0.args
      h0_ok = False
      b0_direct = _match_select(h_0, norm_dim, 0)
      if b0_direct is not None and b0_direct[0] is b_src:
        h0_ok = True
        h_init = None
      elif (
          h_0.op == "call_function"
          and h_0.target == torch.ops.aten.add.Tensor
          and len(h_0_args) == 2
      ):
        for mul_cand, b_cand in [
            (h_0_args[0], h_0_args[1]),
            (h_0_args[1], h_0_args[0]),
        ]:
          b0_match = _match_select(b_cand, norm_dim, 0)
          if b0_match is not None and b0_match[0] is b_src:
            if (
                isinstance(mul_cand, fx.Node)
                and mul_cand.op == "call_function"
                and mul_cand.target == torch.ops.aten.mul.Tensor
                and len(mul_cand.args) == 2
            ):
              mc0_args: Any = mul_cand.args
              for a_cand, init_cand in [
                  (mc0_args[0], mc0_args[1]),
                  (mc0_args[1], mc0_args[0]),
              ]:
                a0_match = _match_select(a_cand, norm_dim, 0)
                if a0_match is not None and a0_match[0] is a_src:
                  h0_ok = True
                  h_init = cast(fx.Node, init_cand)
                  break
          if h0_ok:
            break

      if h0_ok:
        for t in range(2, length):
          h_t = el_nodes[t]
          h_t_args: Any = h_t.args
          step_ok = False
          if (
              h_t.op == "call_function"
              and h_t.target == torch.ops.aten.add.Tensor
              and len(h_t_args) == 2
          ):
            for mul_cand, b_cand in [
                (h_t_args[0], h_t_args[1]),
                (h_t_args[1], h_t_args[0]),
            ]:
              bt_match = _match_select(b_cand, norm_dim, t)
              if bt_match is not None and bt_match[0] is b_src:
                if (
                    isinstance(mul_cand, fx.Node)
                    and mul_cand.op == "call_function"
                    and mul_cand.target == torch.ops.aten.mul.Tensor
                    and len(mul_cand.args) == 2
                ):
                  mct_args: Any = mul_cand.args
                  for a_cand, prev_cand in [
                      (mct_args[0], mct_args[1]),
                      (mct_args[1], mct_args[0]),
                  ]:
                    if prev_cand is el_nodes[t - 1]:
                      at_match = _match_select(a_cand, norm_dim, t)
                      if at_match is not None and at_match[0] is a_src:
                        step_ok = True
                        break
              if step_ok:
                break
          if not step_ok:
            fwd_matched = False
            break
      else:
        fwd_matched = False
    else:
      fwd_matched = False

    if fwd_matched and a_src is not None and b_src is not None:
      b_val = b_src.meta.get("val")
      b_bytes = (
          b_val.numel() * b_val.element_size()
          if isinstance(b_val, torch.Tensor)
          else 0
      )
      if b_bytes <= 128 * 1024 * 1024:
        with graph.inserting_before(node):
          h_all = _emit_kogge_stone_fwd(
              graph, a_src, b_src, h_init, norm_dim, length
          )
        node.replace_all_uses_with(h_all)
        modified = True
      else:
        h_all = node
      fwd_scans[node] = (h_all, a_src, b_src, h_init, norm_dim, el_nodes)
      continue

    # Try matching Reverse Recurrence: h_bar_t = a_{t+1} * h_bar_{t+1} + g_t
    bwd_matched = True
    a_bwd_src: fx.Node | None = None
    g_bwd_src: fx.Node | None = None
    last_el = el_nodes[length - 1]
    last_match = _match_select(last_el, norm_dim, length - 1)
    if last_match is not None:
      g_bwd_src, norm_dim = last_match
      for t in range(length - 2, -1, -1):
        el_t = el_nodes[t]
        el_t_args: Any = el_t.args
        step_ok = False
        if (
            el_t.op == "call_function"
            and el_t.target == torch.ops.aten.add.Tensor
            and len(el_t_args) == 2
        ):
          for mul_cand, g_cand in [
              (el_t_args[0], el_t_args[1]),
              (el_t_args[1], el_t_args[0]),
          ]:
            gt_match = _match_select(g_cand, norm_dim, t)
            if gt_match is not None and gt_match[0] is g_bwd_src:
              if (
                  isinstance(mul_cand, fx.Node)
                  and mul_cand.op == "call_function"
                  and mul_cand.target == torch.ops.aten.mul.Tensor
                  and len(mul_cand.args) == 2
              ):
                mcb_args: Any = mul_cand.args
                for a_cand, next_cand in [
                    (mcb_args[0], mcb_args[1]),
                    (mcb_args[1], mcb_args[0]),
                ]:
                  if next_cand is el_nodes[t + 1]:
                    at_match = _match_select(a_cand, norm_dim, t + 1)
                    if at_match is not None:
                      if a_bwd_src is None or at_match[0] is a_bwd_src:
                        a_bwd_src = at_match[0]
                        step_ok = True
                        break
            if step_ok:
              break
        if not step_ok:
          bwd_matched = False
          break
    else:
      bwd_matched = False

    if bwd_matched and a_bwd_src is not None and g_bwd_src is not None:
      g_val = g_bwd_src.meta.get("val")
      g_bytes = (
          g_val.numel() * g_val.element_size()
          if isinstance(g_val, torch.Tensor)
          else 0
      )
      if g_bytes <= 128 * 1024 * 1024:
        with graph.inserting_before(node):
          h_bar_all = _emit_kogge_stone_bwd(
              graph, a_bwd_src, g_bwd_src, norm_dim, length
          )
        node.replace_all_uses_with(h_bar_all)
        modified = True
      else:
        h_bar_all = node
      bwd_scans[node] = (
          h_bar_all,
          a_bwd_src,
          g_bwd_src,
          norm_dim,
          el_nodes,
      )

  # 2. Detect and vectorize decay gradient stacks: stack([sum(h_bar_t * h_{t-1})], dim)
  if fwd_scans and bwd_scans:
    for node in list(graph.nodes):
      n_args: Any = node.args
      if (
          node.op != "call_function"
          or node.target != torch.ops.aten.stack.default
          or len(n_args) < 1
      ):
        continue
      if node in fwd_scans or node in bwd_scans:
        continue

      elements_grad: Any = n_args[0]
      if not isinstance(elements_grad, (list, tuple)) or len(elements_grad) < 4:
        continue

      length = len(elements_grad)
      matched_fwd = None
      matched_bwd = None
      reduce_dims: list[int] = []
      keepdim = False

      for fwd_info in fwd_scans.values():
        h_all, _, _, h_init, norm_dim, fwd_els = fwd_info
        if len(fwd_els) != length:
          continue
        for bwd_info in bwd_scans.values():
          h_bar_all, _, _, bwd_dim, bwd_els = bwd_info
          if len(bwd_els) != length or bwd_dim != norm_dim:
            continue

          all_ok = True
          for t in range(length):
            el = elements_grad[t]
            if (
                not isinstance(el, fx.Node)
                or el.op != "call_function"
                or el.target != torch.ops.aten.sum.dim_IntList
                or len(el.args) < 2
            ):
              all_ok = False
              break
            el_args: Any = el.args
            mul_node = el_args[0]
            r_dims = [int(x) for x in el_args[1]]
            kd = bool(el_args[2]) if len(el_args) >= 3 else False
            if t == 0:
              reduce_dims = r_dims
              keepdim = kd
            elif r_dims != reduce_dims or kd != keepdim:
              all_ok = False
              break

            if (
                not isinstance(mul_node, fx.Node)
                or mul_node.op != "call_function"
                or mul_node.target != torch.ops.aten.mul.Tensor
                or len(mul_node.args) != 2
            ):
              all_ok = False
              break

            mn_args: Any = mul_node.args
            u, v = mn_args[0], mn_args[1]
            expected_bwd = bwd_els[t]
            if t > 0:
              expected_fwd = fwd_els[t - 1]
              pair_ok = (u is expected_bwd and v is expected_fwd) or (
                  v is expected_bwd and u is expected_fwd
              )
            else:
              pair_ok = (
                  u is expected_bwd and (v is h_init or _is_zeros_node(v))
              ) or (v is expected_bwd and (u is h_init or _is_zeros_node(u)))
            if not pair_ok:
              all_ok = False
              break

          if all_ok:
            matched_fwd = fwd_info
            matched_bwd = bwd_info
            break
        if matched_fwd is not None:
          break

      if matched_fwd is not None and matched_bwd is not None:
        h_all, _, _, h_init, norm_dim, _ = matched_fwd
        h_bar_all, _, _, _, _ = matched_bwd

        with graph.inserting_before(node):
          if h_init is None or _is_zeros_node(h_init):
            first_slice = _emit_node(
                graph, torch.ops.aten.slice.Tensor, (h_all, norm_dim, 0, 1)
            )
            zero_first = _emit_node(
                graph, torch.ops.aten.zeros_like.default, (first_slice,)
            )
          else:
            zero_first = _emit_node(
                graph, torch.ops.aten.unsqueeze.default, (h_init, norm_dim)
            )
          h_prefix = _emit_node(
              graph,
              torch.ops.aten.slice.Tensor,
              (h_all, norm_dim, 0, length - 1),
          )
          h_prev_all = _emit_node(
              graph,
              torch.ops.aten.cat.default,
              ([zero_first, h_prefix], norm_dim),
          )
          prod_all = _emit_node(
              graph, torch.ops.aten.mul.Tensor, (h_bar_all, h_prev_all)
          )

          val_5d = h_all.meta.get("val", None)
          ndim_4d = (
              int(val_5d.ndim) - 1
              if val_5d is not None and hasattr(val_5d, "ndim")
              else 4
          )
          norm_rdims = [(d if d >= 0 else d + ndim_4d) for d in reduce_dims]
          shifted_dims = [(d if d < norm_dim else d + 1) for d in norm_rdims]
          batched_sum = _emit_node(
              graph,
              torch.ops.aten.sum.dim_IntList,
              (prod_all, shifted_dims, keepdim),
          )
        node.replace_all_uses_with(batched_sum)
        modified = True

  if modified:
    graph.eliminate_dead_code()
  return modified


def _get_tensor_bytes(node: fx.Node) -> int:
  """Returns tensor size in bytes if FakeTensor metadata is present, else 0."""
  val = node.meta.get("val", None)
  if isinstance(val, torch.Tensor):
    return int(val.numel()) * int(val.element_size())
  return 0


def _rematerialize_large_fwd_intermediates(graph: fx.Graph) -> bool:
  """Rematerializes large forward broadcast intermediates in the backward pass.

  Uses a 4-byte scalar dependency token from the backward pass passed through
  `torch.ops.tpu.optimization_barrier` together with small frontier inputs to
  prevent XLA HLO CSE from merging recomputed tensors back into the forward pass
  without keeping any large 5D backward tensors live across the barrier.
  """
  nodes_list = list(graph.nodes)
  node_idx = {n: i for i, n in enumerate(nodes_list)}

  # Identify backward boundary (e.g. loss grad seed ones_like)
  bwd_start_idx = None
  for i, n in enumerate(nodes_list):
    if n.op == "call_function" and n.target == torch.ops.aten.ones_like.default:
      bwd_start_idx = i
      break
  if bwd_start_idx is None:
    return False

  # Find candidate forward nodes used in backward
  candidates = []
  for n in nodes_list[:bwd_start_idx]:
    if n.op != "call_function" or n.target not in _CHEAP_REMAT_OPS:
      continue
    n_bytes = _get_tensor_bytes(n)
    if n_bytes < 16384:  # Minimum size threshold (16 KB)
      continue
    if any(node_idx.get(u, -1) >= bwd_start_idx for u in n.users):
      candidates.append(n)

  if not candidates:
    return False

  combined_subgraph: set[fx.Node] = set()
  combined_frontier: set[fx.Node] = set()
  targets_to_remat: list[fx.Node] = []

  for root_n in candidates:
    root_bytes = _get_tensor_bytes(root_n)
    worklist = [root_n]
    subgraph: set[fx.Node] = set()
    frontier: set[fx.Node] = set()
    valid = True

    while worklist:
      curr = worklist.pop(0)
      if curr in subgraph or curr in frontier:
        continue
      if curr.op == "placeholder" or node_idx.get(curr, -1) >= bwd_start_idx:
        frontier.add(curr)
        continue
      curr_bytes = _get_tensor_bytes(curr)
      if curr is not root_n and curr_bytes <= 0.25 * root_bytes:
        frontier.add(curr)
        continue
      if (
          curr.op != "call_function"
          or curr.target not in _CHEAP_REMAT_OPS
          or len(subgraph) >= 16
      ):
        valid = False
        break
      subgraph.add(curr)
      curr_args: Any = curr.args
      for arg in curr_args:
        if isinstance(arg, fx.Node):
          worklist.append(arg)
        elif isinstance(arg, (list, tuple)):
          for item in arg:
            if isinstance(item, fx.Node):
              worklist.append(item)

    if valid and subgraph:
      frontier_bytes = sum(_get_tensor_bytes(f) for f in frontier)
      if frontier_bytes < 0.25 * root_bytes:
        combined_subgraph.update(subgraph)
        combined_frontier.update(frontier)
        targets_to_remat.append(root_n)

  if not targets_to_remat:
    return False

  bwd_users = [
      u
      for t in targets_to_remat
      for u in t.users
      if node_idx.get(u, -1) >= bwd_start_idx
  ]
  if not bwd_users:
    return False

  first_bwd_user = min(bwd_users, key=lambda u: node_idx[u])
  topo_subgraph = sorted(combined_subgraph, key=lambda n: node_idx[n])
  topo_frontier = sorted(combined_frontier, key=lambda n: node_idx[n])

  has_tpu_barrier = hasattr(torch.ops, "tpu") and hasattr(
      torch.ops.tpu, "optimization_barrier"
  )

  # Find a backward tensor argument of first_bwd_user to extract a 4-byte scalar token
  bwd_tensor_arg = None
  u_args: Any = first_bwd_user.args
  for arg in u_args:
    if (
        isinstance(arg, fx.Node)
        and node_idx.get(arg, -1) >= bwd_start_idx
        and isinstance(arg.meta.get("val", None), torch.Tensor)
    ):
      bwd_tensor_arg = arg
      break

  clone_map: dict[fx.Node, fx.Node] = {}
  with graph.inserting_before(first_bwd_user):
    if has_tpu_barrier and topo_frontier:
      barrier_inputs: list[fx.Node] = []
      offset = 0
      if bwd_tensor_arg is not None:
        flat_anchor = _emit_node(
            graph, torch.ops.aten.view.default, (bwd_tensor_arg, [-1])
        )
        scalar_anchor = _emit_node(
            graph, torch.ops.aten.slice.Tensor, (flat_anchor, 0, 0, 1)
        )
        barrier_inputs.append(scalar_anchor)
        offset = 1
      barrier_inputs.extend(topo_frontier)
      barrier_node = _emit_node(
          graph,
          torch.ops.tpu.optimization_barrier.default,
          (barrier_inputs,),
      )
      for i, f_node in enumerate(topo_frontier):
        clone_map[f_node] = _emit_node(
            graph,
            operator.getitem,
            (barrier_node, i + offset),
        )

    for orig_node in topo_subgraph:

      def _remap(x: Any) -> Any:
        if isinstance(x, fx.Node):
          return clone_map.get(x, x)
        if isinstance(x, list):
          return [_remap(i) for i in x]
        if isinstance(x, tuple):
          return tuple(_remap(i) for i in x)
        return x

      orig_args: Any = orig_node.args
      new_args = tuple(_remap(a) for a in orig_args)
      new_kwargs = {k: _remap(v) for k, v in orig_node.kwargs.items()}
      clone_node = _emit_node(graph, orig_node.target, new_args, new_kwargs)
      clone_map[orig_node] = clone_node

  clone_set = set(clone_map.values())
  for t in targets_to_remat:
    remat_node = clone_map[t]
    for u in list(t.users):
      if u not in clone_set and node_idx.get(u, -1) >= bwd_start_idx:
        u.replace_input_with(t, remat_node)

  graph.eliminate_dead_code()
  return True


def _replace_constant_pad_with_cat(graph: fx.Graph) -> bool:
  """Replaces single-dim aten.constant_pad_nd with aten.cat to enable XLA fusion."""
  modified = False
  for node in list(graph.nodes):
    if (
        node.op != "call_function"
        or node.target != torch.ops.aten.constant_pad_nd.default
        or len(node.args) < 2
    ):
      continue
    x_node = node.args[0]
    pad_spec = node.args[1]
    fill_val = node.args[2] if len(node.args) >= 3 else 0.0
    if not isinstance(x_node, fx.Node) or not isinstance(
        pad_spec, (list, tuple)
    ):
      continue
    x_val = x_node.meta.get("val")
    if not isinstance(x_val, torch.Tensor):
      continue
    ndim = x_val.ndim
    padded_dims = []
    for i in range(len(pad_spec) // 2):
      lo = int(pad_spec[2 * i])
      hi = int(pad_spec[2 * i + 1])
      if lo != 0 or hi != 0:
        dim = ndim - 1 - i
        padded_dims.append((dim, lo, hi))
    if len(padded_dims) != 1:
      continue
    dim, lo, hi = padded_dims[0]
    if lo < 0 or hi < 0 or x_val.shape[dim] < max(lo, hi, 1):
      continue

    with graph.inserting_before(node):
      cat_parts = []
      if lo > 0:
        slice_lo = _emit_node(
            graph, torch.ops.aten.slice.Tensor, (x_node, dim, 0, lo)
        )
        if fill_val == 0 or fill_val == 0.0:
          pad_lo = _emit_node(
              graph, torch.ops.aten.zeros_like.default, (slice_lo,)
          )
        else:
          pad_lo = _emit_node(
              graph, torch.ops.aten.full_like.default, (slice_lo, fill_val)
          )
        cat_parts.append(pad_lo)
      cat_parts.append(x_node)
      if hi > 0:
        slice_hi = _emit_node(
            graph, torch.ops.aten.slice.Tensor, (x_node, dim, 0, hi)
        )
        if fill_val == 0 or fill_val == 0.0:
          pad_hi = _emit_node(
              graph, torch.ops.aten.zeros_like.default, (slice_hi,)
          )
        else:
          pad_hi = _emit_node(
              graph, torch.ops.aten.full_like.default, (slice_hi, fill_val)
          )
        cat_parts.append(pad_hi)
      cat_node = _emit_node(graph, torch.ops.aten.cat.default, (cat_parts, dim))
    node.replace_all_uses_with(cat_node)
    modified = True

  if modified:
    graph.eliminate_dead_code()
  return modified


def _optimize_stack_contraction(graph: fx.Graph) -> bool:
  """Replaces bmm contraction over transposed 5D state stacks with per-step or native-layout reduction."""
  modified = False
  for node in list(graph.nodes):
    if (
        node.op != "call_function"
        or node.target != torch.ops.aten.bmm.default
        or len(node.args) != 2
    ):
      continue
    lhs, rhs = node.args[0], node.args[1]
    if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
      continue
    lhs_val = lhs.meta.get("val")
    rhs_val = rhs.meta.get("val")
    if (
        not isinstance(lhs_val, torch.Tensor)
        or not isinstance(rhs_val, torch.Tensor)
        or lhs_val.ndim != 3
        or rhs_val.ndim != 3
        or lhs_val.shape[1] != 1
    ):
      continue
    n_batch, _, k_dim = lhs_val.shape
    _, _, m_dim = rhs_val.shape

    if rhs.op != "call_function" or rhs.target not in (
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
    ):
      continue
    p_swap = rhs.args[0]
    if (
        not isinstance(p_swap, fx.Node)
        or p_swap.op != "call_function"
        or p_swap.target != torch.ops.aten.permute.default
        or len(p_swap.args) < 2
        or not isinstance(p_swap.args[1], (list, tuple))
    ):
      continue
    swap_perm = [int(cast(Any, x)) for x in p_swap.args[1]]
    rank = len(swap_perm)
    if rank < 3 or swap_perm != list(range(rank - 2)) + [rank - 1, rank - 2]:
      continue

    p_prev = p_swap.args[0]
    chain_nodes = {rhs, p_swap}
    if (
        isinstance(p_prev, fx.Node)
        and p_prev.op == "call_function"
        and p_prev.target == torch.ops.aten.permute.default
        and len(p_prev.args) >= 2
        and isinstance(p_prev.args[1], (list, tuple))
        and [int(cast(Any, x)) for x in p_prev.args[1]] == list(range(rank))
    ):
      chain_nodes.add(p_prev)
      src = p_prev.args[0]
    else:
      src = p_prev

    if not isinstance(src, fx.Node):
      continue
    src_val = src.meta.get("val")
    if (
        not isinstance(src_val, torch.Tensor)
        or src_val.ndim != rank
        or src_val.shape[-2] != m_dim
        or src_val.shape[-1] != k_dim
    ):
      continue

    exclusive_to_bmm = all(u in chain_nodes for u in src.users) and all(
        u in chain_nodes or u is node for n in chain_nodes for u in n.users
    )

    with graph.inserting_before(node):
      c_5d = _emit_node(
          graph,
          torch.ops.aten.view.default,
          (lhs, list(src_val.shape[:-2]) + [1, k_dim]),
      )
      if (
          exclusive_to_bmm
          and src.op == "call_function"
          and src.target == torch.ops.aten.stack.default
          and len(src.args) >= 1
          and isinstance(src.args[0], (list, tuple))
      ):
        elements = src.args[0]
        dim = int(cast(Any, src.args[1])) if len(src.args) >= 2 else 0
        if dim < 0:
          dim += rank
        if dim < rank - 2:
          y_elements = []
          for t, h_t in enumerate(elements):
            c_t = _emit_node(graph, torch.ops.aten.select.int, (c_5d, dim, t))
            prod_t = _emit_node(graph, torch.ops.aten.mul.Tensor, (h_t, c_t))
            y_t = _emit_node(
                graph, torch.ops.aten.sum.dim_IntList, (prod_t, [-1])
            )
            y_elements.append(y_t)
          summed = _emit_node(
              graph, torch.ops.aten.stack.default, (y_elements, dim)
          )
        else:
          prod = _emit_node(graph, torch.ops.aten.mul.Tensor, (src, c_5d))
          summed = _emit_node(
              graph, torch.ops.aten.sum.dim_IntList, (prod, [-1])
          )
      else:
        prod = _emit_node(graph, torch.ops.aten.mul.Tensor, (src, c_5d))
        summed = _emit_node(graph, torch.ops.aten.sum.dim_IntList, (prod, [-1]))
      bmm_repl = _emit_node(
          graph,
          torch.ops.aten.view.default,
          (summed, [n_batch, 1, m_dim]),
      )
    node.replace_all_uses_with(bmm_repl)
    modified = True

  if modified:
    graph.eliminate_dead_code()
  return modified


def apply(gm_or_graph: torch.fx.GraphModule | torch.fx.Graph) -> None:
  """Applies recurrence vectorization, slice/select backward fusion, and remat.

  Args:
    gm_or_graph: The FX GraphModule or Graph to optimize.
  """
  graph = (
      gm_or_graph.graph
      if isinstance(gm_or_graph, torch.fx.GraphModule)
      else gm_or_graph
  )

  m0 = _replace_constant_pad_with_cat(graph)
  m1 = _fuse_select_and_slice_backward(graph)
  m2 = _vectorize_linear_recurrence(graph)
  m3 = _optimize_stack_contraction(graph)
  m4 = _rematerialize_large_fwd_intermediates(graph)

  if m0 or m1 or m2 or m3 or m4:
    graph.lint()
    if isinstance(gm_or_graph, torch.fx.GraphModule):
      gm_or_graph.recompile()
