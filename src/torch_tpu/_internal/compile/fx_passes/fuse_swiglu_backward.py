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

"""FX pass to optimize compound SwiGLU backward computation.

In SwiGLU forward:
  `out = silu(gate) * up`

In SwiGLU backward:
  `d_up = dy * silu(gate)`
  `d_gate = (dy * up) * silu'(gate) = silu_backward(dy * up, gate)`

Notice that `dy` is shared between `d_up` and `d_gate`, and `dy * up` is an
intermediate tensor materialized strictly to feed `silu_backward`.

This pass detects the SwiGLU backward subgraph where `dy` multiplies both
`silu(gate)` and `up`, and the result `dy * up` is passed to `silu_backward`
(or decomposed silu backward ops). It rewrites the computation so that:
  `s = sigmoid(gate)`
  `silu_gate = gate * s`
  `d_up = dy * silu_gate`
  `d_gate = dy * (up * (s * (1 + gate * (1 - s))))`
which shares `s = sigmoid(gate)`, factors out `dy`, and eliminates the separate
materialization of `dy * up`, allowing TPU memory traffic and kernel fusion
to compute both gradients collaboratively in a single fused cluster.
"""

import os
import torch
import torch.fx
from torch_tpu._internal import env

_MUL_TARGETS = {
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.mul.default,
}

_SILU_TARGETS = {
    torch.ops.aten.silu.default,
    torch.ops.aten.silu.out,
}

_SILU_BACKWARD_TARGETS = {
    torch.ops.aten.silu_backward.default,
    torch.ops.aten.silu_backward.grad_input,
}

_SIGMOID_TARGETS = {
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.sigmoid.out,
}


def _is_fuse_swiglu_backward_enabled() -> bool:
  """Returns whether the pass is enabled via environment variable."""
  raw = os.environ.get("TORCH_TPU_INTERNAL_ENABLE_FUSE_SWIGLU_BACKWARD", None)
  if raw is not None:
    return raw.lower() not in ("0", "false", "off", "no")
  return env.get_bool_env_once(
      "TORCH_TPU_INTERNAL_ENABLE_FUSE_SWIGLU_BACKWARD", default_value=True
  )


def _normalize_gm_and_graph(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> tuple[torch.fx.GraphModule | None, torch.fx.Graph]:
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    return gm_or_graph, gm_or_graph.graph
  return getattr(gm_or_graph, "owning_module", None), gm_or_graph


def _is_mul(node: torch.fx.Node) -> bool:
  return node.op == "call_function" and (
      node.target in _MUL_TARGETS
      or getattr(node.target, "overloadpacket", None) == torch.ops.aten.mul
  )


def _is_silu(node: torch.fx.Node) -> bool:
  return node.op == "call_function" and (
      node.target in _SILU_TARGETS
      or getattr(node.target, "overloadpacket", None) == torch.ops.aten.silu
  )


def _is_silu_backward(node: torch.fx.Node) -> bool:
  return node.op == "call_function" and (
      node.target in _SILU_BACKWARD_TARGETS
      or getattr(node.target, "overloadpacket", None)
      == torch.ops.aten.silu_backward
  )


def _copy_meta(src: torch.fx.Node, dst: torch.fx.Node) -> None:
  if hasattr(src, "meta"):
    dst.meta = src.meta.copy()


def _try_fuse_swiglu_backward(graph: torch.fx.Graph) -> bool:
  """Matches SwiGLU backward pattern and fuses it."""
  modified = False

  silu_bw_nodes = [n for n in graph.nodes if _is_silu_backward(n)]

  for sbw_node in silu_bw_nodes:
    if len(sbw_node.args) < 2:
      continue
    grad_output = sbw_node.args[0]
    gate = sbw_node.args[1]

    if not isinstance(grad_output, torch.fx.Node) or not _is_mul(grad_output):
      continue

    if len(grad_output.args) < 2:
      continue
    arg0, arg1 = grad_output.args[0], grad_output.args[1]
    if not isinstance(arg0, torch.fx.Node) or not isinstance(
        arg1, torch.fx.Node
    ):
      continue

    candidate_dy_up_pairs = [(arg0, arg1), (arg1, arg0)]

    dy = None
    up = None
    d_up_node = None
    silu_node = None

    for dy_candidate, up_candidate in candidate_dy_up_pairs:
      for user in dy_candidate.users:
        if user == grad_output:
          continue
        if _is_mul(user) and len(user.args) >= 2:
          u0, u1 = user.args[0], user.args[1]
          other = u1 if u0 == dy_candidate else u0
          if isinstance(other, torch.fx.Node) and _is_silu(other):
            if len(other.args) >= 1 and other.args[0] == gate:
              d_up_node = user
              silu_node = other
              break
      if d_up_node is not None:
        dy = dy_candidate
        up = up_candidate
        break

    if d_up_node is None or dy is None or up is None:
      continue

    # Determine safe insertion point:
    # We can insert s and silu_gate before silu_node if silu_node only has 1 user.
    if silu_node is not None and len(silu_node.users) == 1:
      insert_pt = silu_node
    else:
      insert_pt = sbw_node

    with graph.inserting_before(insert_pt):
      # s = sigmoid(gate)
      s = graph.call_function(torch.ops.aten.sigmoid.default, args=(gate,))
      _copy_meta(gate, s)

      if silu_node is not None and len(silu_node.users) == 1:
        silu_gate = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(gate, s)
        )
        _copy_meta(silu_node, silu_gate)
        silu_node.replace_all_uses_with(silu_gate)
        graph.erase_node(silu_node)

    with graph.inserting_before(sbw_node):
      # one_minus_s = 1.0 - s:
      # In PyTorch ATen, 1.0 - s can be written as:
      # sub.Scalar(s, 1.0, alpha=-1) or sub.Tensor with scalar_tensor or neg + add
      neg_s = graph.call_function(torch.ops.aten.neg.default, args=(s,))
      _copy_meta(gate, neg_s)
      one_minus_s = graph.call_function(
          torch.ops.aten.add.Scalar, args=(neg_s, 1.0)
      )
      _copy_meta(gate, one_minus_s)

      # gate_times_one_minus_s = gate * (1.0 - s)
      gate_times_one_minus_s = graph.call_function(
          torch.ops.aten.mul.Tensor, args=(gate, one_minus_s)
      )
      _copy_meta(gate, gate_times_one_minus_s)

      # inner = 1.0 + gate * (1.0 - s)
      inner = graph.call_function(
          torch.ops.aten.add.Scalar, args=(gate_times_one_minus_s, 1.0)
      )
      _copy_meta(gate, inner)

      # dsilu = s * inner
      dsilu = graph.call_function(torch.ops.aten.mul.Tensor, args=(s, inner))
      _copy_meta(gate, dsilu)

      # up_dsilu = up * dsilu
      up_dsilu = graph.call_function(
          torch.ops.aten.mul.Tensor, args=(up, dsilu)
      )
      _copy_meta(sbw_node, up_dsilu)

      # new_d_gate = dy * (up * dsilu)
      new_d_gate = graph.call_function(
          torch.ops.aten.mul.Tensor, args=(dy, up_dsilu)
      )
      _copy_meta(sbw_node, new_d_gate)

      sbw_node.replace_all_uses_with(new_d_gate)
      graph.erase_node(sbw_node)

      if len(grad_output.users) == 0:
        graph.erase_node(grad_output)

      modified = True

  return modified


def apply(gm_or_graph: torch.fx.GraphModule | torch.fx.Graph) -> None:
  """Applies compound SwiGLU backward fusion to the FX graph."""
  if not _is_fuse_swiglu_backward_enabled():
    return

  gm, graph = _normalize_gm_and_graph(gm_or_graph)
  if _try_fuse_swiglu_backward(graph):
    graph.eliminate_dead_code()
    graph.lint()
    if gm is not None:
      for _, submod in gm.named_children():
        if isinstance(submod, torch.fx.GraphModule):
          apply(submod)
      gm.recompile()
