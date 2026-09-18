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

"""FX pass to eliminate redundant clone/contiguous operations feeding SDPA.

In Transformer models, attention inputs are frequently prepared with:
  query = proj(...).view(...).transpose(1, 2).contiguous()
  key = proj(...).view(...).transpose(1, 2).contiguous()
  value = proj(...).view(...).transpose(1, 2).contiguous()
  out = F.scaled_dot_product_attention(query, key, value, ...)

In FX graph tracing, `.contiguous()` produces `torch.ops.aten.clone.default`,
`torch.ops.aten.copy.default`, or `torch.ops.aten._to_copy.default`.
TorchTPU's StableHLO compiler lowers SDPA by accepting transposed strides
directly via dimension mappings in dot_general without requiring physically
contiguous memory.

This pass inspects inputs to SDPA nodes (query, key, value) and bypasses
redundant clone/copy/contiguous operations, connecting the original source
tensor directly to SDPA and eliminating unnecessary HBM allocations and copies.
"""

import os
from typing import Any
import torch
from torch import fx


def _is_eliminate_sdpa_contiguous_enabled() -> bool:
  """Returns whether the pass is enabled via environment variable."""
  return os.environ.get(  # PY_ENVIRON_OK=internal compiler pass knob
      "TORCH_TPU_INTERNAL_ENABLE_ELIMINATE_SDPA_CONTIGUOUS", "1"
  ).lower() not in ("0", "false", "no")


def _normalize_gm_and_graph(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> tuple[torch.fx.GraphModule | None, torch.fx.Graph]:
  """Extracts the owning GraphModule (if any) and the Graph."""
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    return gm_or_graph, gm_or_graph.graph
  return getattr(gm_or_graph, "owning_module", None), gm_or_graph


_SDPA_FORWARD_TARGETS: set[Any] = {
    torch.ops.aten.scaled_dot_product_attention.default,
    torch.ops.aten.scaled_dot_product_attention,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_efficient_attention,
}

_SDPA_BACKWARD_TARGETS: set[Any] = {
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward,
    torch.ops.aten._scaled_dot_product_efficient_attention_backward.default,
    torch.ops.aten._scaled_dot_product_efficient_attention_backward,
}

for _name in (
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_flash_attention_for_cpu",
):
  _op = getattr(torch.ops.aten, _name, None)
  if _op is not None:
    if hasattr(_op, "default"):
      _SDPA_FORWARD_TARGETS.add(_op.default)
    _SDPA_FORWARD_TARGETS.add(_op)

for _name in (
    "_scaled_dot_product_flash_attention_backward",
    "_scaled_dot_product_flash_attention_for_cpu_backward",
):
  _op = getattr(torch.ops.aten, _name, None)
  if _op is not None:
    if hasattr(_op, "default"):
      _SDPA_BACKWARD_TARGETS.add(_op.default)
    _SDPA_BACKWARD_TARGETS.add(_op)

_CLONE_TARGETS: set[Any] = {
    torch.ops.aten.clone.default,
    torch.ops.aten.clone,
}
if hasattr(torch.ops.aten, "contiguous"):
  _CLONE_TARGETS.add(torch.ops.aten.contiguous)
  if hasattr(torch.ops.aten.contiguous, "default"):
    _CLONE_TARGETS.add(torch.ops.aten.contiguous.default)

_TO_COPY_TARGETS: set[Any] = {
    torch.ops.aten._to_copy.default,
    torch.ops.aten._to_copy,
}

_COPY_TARGETS: set[Any] = {
    torch.ops.aten.copy.default,
    torch.ops.aten.copy,
}


def _is_sdpa_fwd(target: Any) -> bool:
  if target in _SDPA_FORWARD_TARGETS:
    return True
  name = getattr(target, "__name__", str(target))
  return "scaled_dot_product" in name and "backward" not in name


def _is_sdpa_bwd(target: Any) -> bool:
  if target in _SDPA_BACKWARD_TARGETS:
    return True
  name = getattr(target, "__name__", str(target))
  return "scaled_dot_product" in name and "backward" in name


def _is_clone_node(node: fx.Node) -> bool:
  if node.op != "call_function":
    return False
  if node.target in _CLONE_TARGETS:
    return True
  name = getattr(node.target, "__name__", str(node.target))
  return name in (
      "aten.clone.default",
      "aten.contiguous.default",
      "clone",
      "contiguous",
  )


def _is_to_copy_node(node: fx.Node) -> bool:
  if node.op != "call_function":
    return False
  if node.target in _TO_COPY_TARGETS:
    return True
  name = getattr(node.target, "__name__", str(node.target))
  return name in ("aten._to_copy.default", "_to_copy")


def _is_copy_node(node: fx.Node) -> bool:
  if node.op != "call_function":
    return False
  if node.target in _COPY_TARGETS:
    return True
  name = getattr(node.target, "__name__", str(node.target))
  return name in ("aten.copy.default", "copy")


def _is_mutating_node(node: fx.Node) -> bool:
  """Checks if an FX node represents an in-place mutation."""
  if node.op == "call_function":
    target = node.target
    schema = getattr(target, "_schema", None)
    if schema is not None and getattr(schema, "is_mutable", False):
      return True
    name = getattr(target, "__name__", str(target))
    if name.endswith("_") and not name.endswith("__"):
      return True
  elif node.op == "call_method":
    name = str(node.target)
    if name.endswith("_") and not name.endswith("__"):
      return True
  return False


def _is_same_dtype_and_device(
    copy_node: fx.Node,
    source_node: fx.Node,
) -> bool:
  """Checks if copy_node preserves the dtype and device of source_node."""
  req_dtype = copy_node.kwargs.get("dtype")
  req_device = copy_node.kwargs.get("device")
  req_layout = copy_node.kwargs.get("layout")
  if req_layout is not None and req_layout != torch.strided:
    return False

  source_val = (
      source_node.meta.get("val") if hasattr(source_node, "meta") else None
  )
  copy_val = copy_node.meta.get("val") if hasattr(copy_node, "meta") else None

  source_dtype = None
  source_device = None
  if isinstance(source_val, torch.Tensor):
    source_dtype = source_val.dtype
    source_device = source_val.device
  elif hasattr(source_node, "meta") and "tensor_meta" in source_node.meta:
    tm = source_node.meta["tensor_meta"]
    source_dtype = getattr(tm, "dtype", None)
    source_device = getattr(tm, "device", None)

  copy_dtype = None
  copy_device = None
  if isinstance(copy_val, torch.Tensor):
    copy_dtype = copy_val.dtype
    copy_device = copy_val.device
  elif hasattr(copy_node, "meta") and "tensor_meta" in copy_node.meta:
    tm = copy_node.meta["tensor_meta"]
    copy_dtype = getattr(tm, "dtype", None)
    copy_device = getattr(tm, "device", None)

  if req_dtype is not None:
    if source_dtype is not None and req_dtype != source_dtype:
      return False
    if copy_dtype is not None and req_dtype != copy_dtype:
      return False
  if source_dtype is not None and copy_dtype is not None:
    if source_dtype != copy_dtype:
      return False

  if req_device is not None:
    req_dev_obj = torch.device(req_device)
    if source_device is not None and torch.device(source_device) != req_dev_obj:
      return False
    if copy_device is not None and torch.device(copy_device) != req_dev_obj:
      return False
  if source_device is not None and copy_device is not None:
    if torch.device(source_device) != torch.device(copy_device):
      return False

  return True


def _extract_redundant_source(
    candidate: fx.Node,
    sdpa_node: fx.Node,
    bypassed_nodes: set[fx.Node],
) -> fx.Node | None:
  """Extracts source node if candidate is a redundant clone/copy for SDPA."""
  if not isinstance(candidate, fx.Node) or candidate.op != "call_function":
    return None

  source_node: fx.Node | None = None

  if _is_clone_node(candidate):
    if candidate.args and isinstance(candidate.args[0], fx.Node):
      source_node = candidate.args[0]
  elif _is_to_copy_node(candidate):
    if candidate.args and isinstance(candidate.args[0], fx.Node):
      src = candidate.args[0]
      if _is_same_dtype_and_device(candidate, src):
        source_node = src
  elif _is_copy_node(candidate):
    src = None
    if len(candidate.args) >= 2 and isinstance(candidate.args[1], fx.Node):
      src = candidate.args[1]
    elif "src" in candidate.kwargs and isinstance(
        candidate.kwargs["src"], fx.Node
    ):
      src = candidate.kwargs["src"]
    elif candidate.args and isinstance(candidate.args[0], fx.Node):
      src = candidate.args[0]
    if src is not None and _is_same_dtype_and_device(candidate, src):
      source_node = src

  if source_node is None:
    return None

  # Check safety: candidate users must be allowed (sdpa_node or already
  # bypassed), or non-mutating users.
  candidate_users = set(candidate.users)
  allowed_users = {sdpa_node} | bypassed_nodes
  if not candidate_users.issubset(allowed_users):
    for user in candidate_users:
      if user in allowed_users:
        continue
      if _is_mutating_node(user):
        return None

  # Source node must not have mutating users.
  for user in source_node.users:
    if user in allowed_users:
      continue
    if _is_mutating_node(user):
      return None

  return source_node


def _unwrap_redundant_clones(
    node: fx.Node,
    sdpa_node: fx.Node,
) -> tuple[fx.Node, list[fx.Node]]:
  """Unwraps a chain of redundant clone/copy operations feeding sdpa_node.

  Returns:
    Tuple of (root_source_node, list_of_bypassed_nodes).
  """
  curr = node
  bypassed: list[fx.Node] = []
  bypassed_set: set[fx.Node] = set()

  while isinstance(curr, fx.Node):
    source = _extract_redundant_source(curr, sdpa_node, bypassed_set)
    if source is None or source is curr:
      break
    bypassed.append(curr)
    bypassed_set.add(curr)
    curr = source

  return curr, bypassed


def _erase_unused_chain(node: fx.Node, graph: fx.Graph) -> None:
  """Erases node and any of its inputs that become dead."""
  to_check = [node]
  while to_check:
    n = to_check.pop(0)
    if n.graph is not None and not n.users:
      inputs = [arg for arg in n.args if isinstance(arg, fx.Node)]
      for kw_val in n.kwargs.values():
        if isinstance(kw_val, fx.Node):
          inputs.append(kw_val)
      graph.erase_node(n)
      to_check.extend(inputs)


def apply(gm_or_graph: torch.fx.GraphModule | torch.fx.Graph) -> None:
  """Bypasses redundant clone/copy/contiguous operations feeding SDPA.

  Args:
    gm_or_graph: The FX GraphModule or Graph to transform.
  """
  if not _is_eliminate_sdpa_contiguous_enabled():
    return

  gm, graph = _normalize_gm_and_graph(gm_or_graph)
  modified = False
  dead_nodes: list[fx.Node] = []

  for node in list(graph.nodes):
    if node.op != "call_function":
      continue

    is_fwd = _is_sdpa_fwd(node.target)
    is_bwd = _is_sdpa_bwd(node.target)
    if not is_fwd and not is_bwd:
      continue

    # Forward: query=0, key=1, value=2
    # Backward: grad_out=0, query=1, key=2, value=3
    arg_indices = (0, 1, 2) if is_fwd else (0, 1, 2, 3)

    # Check positional args
    new_args = list(node.args)
    for idx in arg_indices:
      if idx < len(new_args):
        arg = new_args[idx]
        if isinstance(arg, fx.Node):
          orig, bypassed = _unwrap_redundant_clones(arg, node)
          if orig is not arg:
            new_args[idx] = orig
            dead_nodes.extend(bypassed)
            modified = True
    node.args = tuple(new_args)

    # Check kwargs
    new_kwargs = dict(node.kwargs)
    for kw in ("query", "key", "value", "grad_out", "grad_output"):
      if kw in new_kwargs:
        arg = new_kwargs[kw]
        if isinstance(arg, fx.Node):
          orig, bypassed = _unwrap_redundant_clones(arg, node)
          if orig is not arg:
            new_kwargs[kw] = orig
            dead_nodes.extend(bypassed)
            modified = True
    node.kwargs = new_kwargs

  if modified:
    for n in dead_nodes:
      _erase_unused_chain(n, graph)
    graph.lint()
    if gm is not None:
      gm.recompile()
