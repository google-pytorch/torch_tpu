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

"""FX pass to fold Grouped-Query Attention (GQA) head repetition into SDPA.

In Grouped-Query Attention (GQA), H_q query heads share H_kv key/value heads,
with group ratio G = H_q // H_kv. In standard Hugging Face transformer models,
`repeat_kv` explicitly expands key and value heads in Python/PyTorch via:
  `hidden_states[:, :, None, :, :].expand(B, H_kv, G, S, D).reshape(B, H_q, S,
  D)`

This materializes 4D expanded tensors in memory and forces autograd to trace
a backward `.sum(dim=2)` reduction pass across query groups.

TorchTPU natively supports 5D folded GQA contractions directly in its
StableHLO SDPA lowering. This pass detects the `unsqueeze -> expand -> reshape`
pattern on key and value inputs to SDPA, replacing them with the original
unexpanded key and value tensors so that head broadcast and backward reduction
are folded directly on device MXU without memory materialization.
"""

import operator
import torch
from torch import fx

_SDPA_FORWARD_TARGETS = {
    torch.ops.aten.scaled_dot_product_attention.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
}

_SDPA_BACKWARD_TARGETS = {
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward.default,
    torch.ops.aten._scaled_dot_product_efficient_attention_backward.default,
}

_RESHAPE_TARGETS = {
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
}

_EXPAND_TARGETS = {
    torch.ops.aten.expand.default,
    torch.ops.aten.expand_copy.default,
}

_UNSQUEEZE_TARGETS = {
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.unsqueeze_copy.default,
}

_SUM_TARGETS = {
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.sum.default,
}

_SQUEEZE_TARGETS = {
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze_copy.dim,
}


def _extract_repeat_kv_source(node: fx.Node) -> tuple[fx.Node, int] | None:
  """Extracts (orig_tensor_node, group_size) if node is a repeat_kv expansion.

  Detects:
    orig -> unsqueeze(dim=2) -> expand(..., H_kv, G, S, D) -> reshape(..., H_q,
    S, D)
  """
  if not isinstance(node, fx.Node) or node.op != "call_function":
    return None

  curr = node
  # Unwrap optional clone or contiguous operations
  while curr.target in (
      torch.ops.aten.clone.default,
      torch.ops.aten.contiguous.default,
  ):
    if curr.args and isinstance(curr.args[0], fx.Node):
      curr = curr.args[0]
    else:
      break

  if curr.target not in _RESHAPE_TARGETS:
    return None

  if not curr.args or not isinstance(curr.args[0], fx.Node):
    return None

  expand_node = curr.args[0]
  if (
      not isinstance(expand_node, fx.Node)
      or expand_node.target not in _EXPAND_TARGETS
  ):
    return None

  if len(expand_node.args) < 2:
    return None

  expand_shape = expand_node.args[1]
  if not isinstance(expand_shape, (list, tuple)) or len(expand_shape) != 5:
    return None

  group_size = expand_shape[2]
  if not isinstance(group_size, int) or group_size <= 1:
    return None

  unsqueeze_node = expand_node.args[0]
  if not isinstance(unsqueeze_node, fx.Node):
    return None

  if unsqueeze_node.target in _UNSQUEEZE_TARGETS:
    if len(unsqueeze_node.args) < 2:
      return None
    dim = unsqueeze_node.args[1]
    if dim != 2 and dim != -3:
      return None
    orig_node = unsqueeze_node.args[0]
    if isinstance(orig_node, fx.Node):
      return orig_node, group_size

  # Alternatively, unsqueeze might have been captured as view to 5D: [B, H_kv, 1, S, D]
  if unsqueeze_node.target in _RESHAPE_TARGETS:
    if len(unsqueeze_node.args) >= 2:
      view_shape = unsqueeze_node.args[1]
      if (
          isinstance(view_shape, (list, tuple))
          and len(view_shape) == 5
          and view_shape[2] == 1
      ):
        orig_node = unsqueeze_node.args[0]
        if isinstance(orig_node, fx.Node):
          return orig_node, group_size

  return None


def _detect_repeat_kv_backward_reduction(
    grad_consumer: fx.Node,
    expected_group_size: int | None = None,
) -> tuple[fx.Node, list[fx.Node]] | None:
  """Detects autograd backward reduction of repeat_kv.

  Forward repeat_kv pattern:
    hidden_states[:, :, None, :, :].expand(B, H_kv, G, S, D).reshape(B, H_q, S,
    D)

  Autograd backward reduction pattern:
    grad (B, H_q, S, D)
      -> reshape(..., H_kv, G, S, D)
      -> sum(dim=2, keepdim=False)  [or sum(dim=2, keepdim=True) -> squeeze(2)]
      -> output (B, H_kv, S, D)

  Returns (final_reduced_node, list_of_dead_nodes_to_erase) if matched.
  """
  if (
      not isinstance(grad_consumer, fx.Node)
      or grad_consumer.op != "call_function"
  ):
    return None

  # 1. Check if grad_consumer is a reshape to 5D [B, H_kv, G, S, D]
  if grad_consumer.target not in _RESHAPE_TARGETS:
    return None

  if len(grad_consumer.args) < 2:
    return None

  reshape_shape = grad_consumer.args[1]
  if not isinstance(reshape_shape, (list, tuple)) or len(reshape_shape) != 5:
    return None

  group_size = reshape_shape[2]
  if not isinstance(group_size, int) or group_size <= 1:
    return None

  if expected_group_size is not None and group_size != expected_group_size:
    return None

  # 2. Check consumers of the reshape node for a sum along dim 2
  for sum_node in list(grad_consumer.users):
    if sum_node.op != "call_function" or sum_node.target not in _SUM_TARGETS:
      continue

    if len(sum_node.args) < 2:
      continue

    sum_dim = sum_node.args[1]
    if isinstance(sum_dim, (list, tuple)):
      if len(sum_dim) != 1 or (sum_dim[0] != 2 and sum_dim[0] != -3):
        continue
    elif sum_dim != 2 and sum_dim != -3:
      continue

    keepdim = sum_node.kwargs.get("keepdim", False)
    if (
        not keepdim
        and len(sum_node.args) >= 3
        and isinstance(sum_node.args[2], bool)
    ):
      keepdim = sum_node.args[2]

    if not keepdim:
      return sum_node, [grad_consumer, sum_node]

    # If keepdim was True, look for squeeze(2) or reshape to 4D
    for squeeze_node in list(sum_node.users):
      if (
          squeeze_node.op == "call_function"
          and squeeze_node.target in _SQUEEZE_TARGETS
      ):
        sq_dim = 2
        if len(squeeze_node.args) >= 2:
          sq_dim = squeeze_node.args[1]
        if sq_dim == 2 or sq_dim == -3:
          return squeeze_node, [grad_consumer, sum_node, squeeze_node]
      if (
          squeeze_node.op == "call_function"
          and squeeze_node.target in _RESHAPE_TARGETS
          and len(squeeze_node.args) >= 2
      ):
        view_shape = squeeze_node.args[1]
        if isinstance(view_shape, (list, tuple)) and len(view_shape) == 4:
          return squeeze_node, [grad_consumer, sum_node, squeeze_node]

  return None


def _erase_unused_chain(node: fx.Node, graph: fx.Graph) -> None:
  """Erases node and any of its inputs that become dead."""
  to_check = [node]
  while to_check:
    n = to_check.pop(0)
    if n.graph is not None and not n.users:
      inputs = [arg for arg in n.args if isinstance(arg, fx.Node)]
      graph.erase_node(n)
      to_check.extend(inputs)


def apply(gm_or_graph: torch.fx.GraphModule | torch.fx.Graph) -> None:
  """Folds repeat_kv head expansion into SDPA forward and backward passes.

  Args:
    gm_or_graph: The FX GraphModule or Graph to transform.
  """
  graph = (
      gm_or_graph.graph
      if isinstance(gm_or_graph, torch.fx.GraphModule)
      else gm_or_graph
  )

  modified = False
  dead_nodes = []

  for node in list(graph.nodes):
    if node.op != "call_function":
      continue

    is_fwd = node.target in _SDPA_FORWARD_TARGETS
    is_bwd = node.target in _SDPA_BACKWARD_TARGETS
    if not is_fwd and not is_bwd:
      continue

    # In forward SDPA, key is args[1] and value is args[2].
    # In backward SDPA, args are (grad_out, query, key, value, ...), so key is
    # args[2] and value is args[3].
    key_idx, val_idx = (1, 2) if is_fwd else (2, 3)
    if len(node.args) <= val_idx:
      continue

    key_node = node.args[key_idx]
    value_node = node.args[val_idx]
    if not isinstance(key_node, fx.Node) or not isinstance(value_node, fx.Node):
      continue

    key_info = _extract_repeat_kv_source(key_node)
    value_info = _extract_repeat_kv_source(value_node)
    if key_info is None or value_info is None:
      continue

    orig_key, key_group = key_info
    orig_value, val_group = value_info
    if key_group != val_group:
      continue

    new_args = list(node.args)
    new_args[key_idx] = orig_key
    new_args[val_idx] = orig_value
    node.args = tuple(new_args)

    dead_nodes.extend([key_node, value_node])
    modified = True

    # In backward pass, output 1 (grad_key) and output 2 (grad_value) are
    # already reduced across the group dimension by the native 5D lowering.
    # Eliminate downstream autograd sum reductions on these outputs.
    if is_bwd:
      for user in list(node.users):
        if (
            user.op == "call_function"
            and user.target == operator.getitem
            and len(user.args) >= 2
            and user.args[1] in (1, 2)
        ):
          for grad_consumer in list(user.users):
            reduction_info = _detect_repeat_kv_backward_reduction(
                grad_consumer, expected_group_size=key_group
            )
            if reduction_info is not None:
              final_reduced_node, reduction_chain = reduction_info
              final_reduced_node.replace_all_uses_with(user)
              dead_nodes.extend(reduction_chain)
              modified = True

  if modified:
    for n in dead_nodes:
      _erase_unused_chain(n, graph)
    graph.lint()
    if isinstance(gm_or_graph, torch.fx.GraphModule):
      gm_or_graph.recompile()
