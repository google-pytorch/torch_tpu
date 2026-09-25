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

"""Detects and eliminates redundant causal attention masks in SDPA ops."""

from typing import Any, Sequence
import torch
import torch.fx
from torch.utils import _pytree


def _is_causal_mask(tensor: Any) -> bool:
  """Returns True if the tensor represents a lower-triangular causal mask."""
  # ---------------------------------------------------------------------------
  # Mathematical Structure of Lower-Triangular Causal Attention Mask:
  # ---------------------------------------------------------------------------
  # For attention queries Q with sequence length S_q and keys K with length S_kv:
  # In autoregressive models where S_q == S_kv = S:
  #   1. Additive attention bias:
  #        Bias[..., i, j] = 0.0      for j <= i (allowed attention: past & current tokens)
  #        Bias[..., i, j] = -\infty   for j > i  (masked attention: future tokens)
  #
  #   2. Boolean attention mask:
  #        Mask[..., i, j] = True     for j <= i
  #        Mask[..., i, j] = False    for j > i
  #
  # When an explicit mask tensor matches this structure, materializing it in HBM
  # across all transformer layers is redundant. We eliminate the argument and set
  # `is_causal = True`, triggering native hardware-optimized causal attention lowerings.
  # ---------------------------------------------------------------------------
  if not isinstance(tensor, torch.Tensor):
    return False
  if tensor.ndim < 2:
    return False

  seq_len_q, seq_len_kv = tensor.shape[-2], tensor.shape[-1]
  if seq_len_q != seq_len_kv or seq_len_q <= 1:
    return False

  try:
    if (
        getattr(tensor, "is_meta", False)
        or type(tensor).__name__ == "FakeTensor"
    ):
      return False

    tensor_cpu = tensor.detach().to("cpu")

    if tensor_cpu.dtype == torch.bool:
      expected_causal_2d = torch.tril(
          torch.ones((seq_len_q, seq_len_kv), dtype=torch.bool, device="cpu")
      )
      shape = [1] * (tensor_cpu.ndim - 2) + [seq_len_q, seq_len_kv]
      expected_causal = expected_causal_2d.view(shape).expand_as(tensor_cpu)
      return torch.equal(tensor_cpu, expected_causal)

    if tensor_cpu.is_floating_point():
      lower_triangular_2d = torch.tril(
          torch.ones((seq_len_q, seq_len_kv), dtype=torch.bool, device="cpu")
      )
      shape = [1] * (tensor_cpu.ndim - 2) + [seq_len_q, seq_len_kv]
      lower_triangular = lower_triangular_2d.view(shape).expand_as(tensor_cpu)
      upper_triangular = ~lower_triangular

      lower_values = tensor_cpu[lower_triangular]
      upper_values = tensor_cpu[upper_triangular]

      is_lower_tri_zero = torch.all(torch.abs(lower_values) < 1e-4)
      is_upper_tri_neginf = torch.all(
          (upper_values <= -1000.0) | torch.isneginf(upper_values)
      )
      return bool(is_lower_tri_zero and is_upper_tri_neginf)
  except Exception:
    return False

  return False


# Upper bounds on constant folding work done while evaluating a mask producer.
# Masks are at most (B, 1, S, S); expand() returns a view so B is free.
_MAX_FOLD_NUMEL = 1 << 26
_MAX_FOLD_DEPTH = 64


def _is_foldable(target: torch._ops.OpOverload) -> bool:  # pylint: disable=protected-access
  """Returns True if `target` is a deterministic, non-mutating op."""
  if torch.Tag.nondeterministic_seeded in target.tags:
    return False
  return not target._schema.is_mutable  # pylint: disable=protected-access


def _evaluate_node_on_inputs(
    node: Any,
    placeholder_to_input: dict[torch.fx.Node, Any],
    gm: torch.fx.GraphModule | None,
    cache: dict[torch.fx.Node, Any],
    depth: int = 0,
) -> Any:
  """Evaluates a node producing a mask tensor.

  Placeholders resolve to their example inputs and `get_attr` nodes to module
  attributes. Any deterministic, non-mutating aten op is then constant folded,
  including factory ops such as `arange`. This lets masks that are built inside
  the graph from positions alone (as Hugging Face does when it detects tracing)
  be proven causal without depending on example input values. Returns None if
  the value cannot be computed.
  """
  if not isinstance(node, torch.fx.Node):
    return node

  if node in cache:
    return cache[node]

  val = None
  target = node.target
  if node.op == "placeholder":
    val = placeholder_to_input.get(node)
  elif node.op == "get_attr":
    if gm is not None and isinstance(target, str):
      try:
        val = torch.fx.graph_module._get_attr(gm, target)  # pylint: disable=protected-access
      except AttributeError:
        val = None
  elif (
      node.op == "call_function"
      and depth < _MAX_FOLD_DEPTH
      and isinstance(target, torch._ops.OpOverload)  # pylint: disable=protected-access
      and _is_foldable(target)
  ):
    evaluate = lambda a: _evaluate_node_on_inputs(
        a, placeholder_to_input, gm, cache, depth + 1
    )
    args = _pytree.tree_map_only(torch.fx.Node, evaluate, node.args)
    kwargs = _pytree.tree_map_only(torch.fx.Node, evaluate, node.kwargs)
    if all(cache.get(n) is not None for n in node.all_input_nodes):
      if "device" in kwargs:
        kwargs = {**kwargs, "device": "cpu"}
      try:
        val = target(*args, **kwargs)
      except Exception:  # pylint: disable=broad-except
        val = None
      if isinstance(val, torch.Tensor) and val.numel() > _MAX_FOLD_NUMEL:
        val = None

  cache[node] = val
  return val


# Target specifications for SDPA variants: (bias_arg_idx, is_causal_arg_idx, is_backward)
_SDPA_SPECS: dict[Any, tuple[int, int, bool]] = {
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default: (
        3,
        5,
        False,
    ),
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward.default: (
        4,
        13,
        True,
    ),
    torch.ops.aten._scaled_dot_product_efficient_attention.default: (
        3,
        6,
        False,
    ),
    torch.ops.aten._scaled_dot_product_efficient_attention_backward.default: (
        4,
        11,
        True,
    ),
    torch.ops.aten.scaled_dot_product_attention.default: (3, 5, False),
}


def _try_eliminate_causal_mask(
    node: torch.fx.Node,
    bias_idx: int,
    is_causal_idx: int,
    is_bwd: bool,
    graph: torch.fx.Graph,
    placeholder_to_input: dict[torch.fx.Node, Any],
    gm: torch.fx.GraphModule | None,
    cache: dict[torch.fx.Node, Any],
) -> bool:
  """Inspects attn_bias in node and eliminates it if it is a causal mask."""
  is_causal = (
      node.args[is_causal_idx]
      if len(node.args) > is_causal_idx
      else node.kwargs.get("is_causal", False)
  )
  if is_causal:
    return False

  attn_bias_node = (
      node.args[bias_idx]
      if len(node.args) > bias_idx
      else node.kwargs.get("attn_bias", node.kwargs.get("attn_mask"))
  )
  if attn_bias_node is None:
    return False

  tensor = _evaluate_node_on_inputs(
      attn_bias_node, placeholder_to_input, gm, cache
  )
  if tensor is None or not _is_causal_mask(tensor):
    return False

  replacement_bias = None
  if is_bwd:
    with graph.inserting_before(node):
      replacement_bias = graph.call_function(
          torch.ops.aten.empty.memory_format, ([0],)
      )

  new_args = list(node.args)
  new_kwargs = dict(node.kwargs)
  if len(new_args) > bias_idx:
    new_args[bias_idx] = replacement_bias
  if len(new_args) > is_causal_idx:
    new_args[is_causal_idx] = True
  else:
    new_kwargs["is_causal"] = True
  node.args = tuple(new_args)

  for kw in ("attn_bias", "attn_mask"):
    if kw in new_kwargs:
      new_kwargs[kw] = replacement_bias
  node.kwargs = new_kwargs

  return True


def apply(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
    example_inputs: Sequence[Any] | None = None,
) -> None:
  """Replaces redundant causal attention masks in SDPA with is_causal=True."""
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    gm = gm_or_graph
    graph = gm_or_graph.graph
  elif isinstance(gm_or_graph, torch.fx.Graph) and isinstance(
      gm_or_graph.owning_module, torch.fx.GraphModule
  ):
    gm = gm_or_graph.owning_module
    graph = gm_or_graph
  else:
    gm = None
    graph = gm_or_graph

  placeholder_to_input = {}
  if example_inputs is not None:
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    for i, p in enumerate(placeholders):
      if i < len(example_inputs):
        placeholder_to_input[p] = example_inputs[i]

  cache: dict[torch.fx.Node, Any] = {}
  modified = False

  for node in list(graph.nodes):
    if node.op != "call_function":
      continue
    spec = _SDPA_SPECS.get(node.target)
    if spec is not None:
      bias_idx, is_causal_idx, is_bwd = spec
      if _try_eliminate_causal_mask(
          node,
          bias_idx,
          is_causal_idx,
          is_bwd,
          graph,
          placeholder_to_input,
          gm,
          cache,
      ):
        modified = True

  if modified:
    graph.eliminate_dead_code()
    if gm is not None:
      gm.recompile()
