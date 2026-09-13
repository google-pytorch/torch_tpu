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

"""FX pass to optimize and canonicalize attention calls.

Specifically, this pass:
1. Detects SDPA patterns where attn_mask is None and is_causal is False,
   canonicalizing unmasked attention calls.
2. Detects and eliminates redundant mask allocations (e.g., empty tensors,
   all-zero additive biases, or all-True boolean masks) passed to SDPA.
3. Folds constant scale multiplications and divisions on query and key tensors
   into the SDPA `scale` argument, eliminating redundant tensor multiplication
   kernels and intermediate allocations before SDPA.
4. Canonicalizes scale computations (e.g. constant scale subgraphs) into
   explicit float constants in SDPA arguments and removes redundant scalar
   computation nodes.
"""

import dataclasses
import math
import operator
from typing import Any, Sequence
import torch
import torch.fx as fx


@dataclasses.dataclass(frozen=True)
class _SdpaOpSpec:
  q_idx: int
  k_idx: int
  v_idx: int
  mask_idx: int
  is_causal_idx: int
  scale_idx: int
  is_bwd: bool
  mask_kwarg: str


_SDPA_SPECS: dict[Any, _SdpaOpSpec] = {
    torch.ops.aten.scaled_dot_product_attention.default: _SdpaOpSpec(
        q_idx=0,
        k_idx=1,
        v_idx=2,
        mask_idx=3,
        is_causal_idx=5,
        scale_idx=6,
        is_bwd=False,
        mask_kwarg="attn_mask",
    ),
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default: (
        _SdpaOpSpec(
            q_idx=0,
            k_idx=1,
            v_idx=2,
            mask_idx=3,
            is_causal_idx=5,
            scale_idx=7,
            is_bwd=False,
            mask_kwarg="attn_bias",
        )
    ),
    torch.ops.aten._scaled_dot_product_efficient_attention.default: _SdpaOpSpec(
        q_idx=0,
        k_idx=1,
        v_idx=2,
        mask_idx=3,
        is_causal_idx=6,
        scale_idx=7,
        is_bwd=False,
        mask_kwarg="attn_bias",
    ),
    torch.ops.aten._scaled_dot_product_flash_attention.default: _SdpaOpSpec(
        q_idx=0,
        k_idx=1,
        v_idx=2,
        mask_idx=5,
        is_causal_idx=4,
        scale_idx=6,
        is_bwd=False,
        mask_kwarg="attn_mask",
    ),
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward.default: _SdpaOpSpec(
        q_idx=1,
        k_idx=2,
        v_idx=3,
        mask_idx=4,
        is_causal_idx=13,
        scale_idx=16,
        is_bwd=True,
        mask_kwarg="attn_bias",
    ),
    torch.ops.aten._scaled_dot_product_efficient_attention_backward.default: (
        _SdpaOpSpec(
            q_idx=1,
            k_idx=2,
            v_idx=3,
            mask_idx=4,
            is_causal_idx=11,
            scale_idx=12,
            is_bwd=True,
            mask_kwarg="attn_bias",
        )
    ),
    torch.ops.aten._scaled_dot_product_flash_attention_backward.default: (
        _SdpaOpSpec(
            q_idx=1,
            k_idx=2,
            v_idx=3,
            mask_idx=8,
            is_causal_idx=7,
            scale_idx=9,
            is_bwd=True,
            mask_kwarg="attn_mask",
        )
    ),
}


def _matches_op_name(target: Any, name_prefix: str) -> bool:
  """Checks if target string or op name matches a prefix like 'aten.empty'."""
  target_str = str(target)
  return target_str.startswith(name_prefix) or (
      hasattr(target, "name") and target.name().startswith(name_prefix)
  )


_EVALUATABLE_OPS = frozenset({
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.slice_copy.Tensor,
    torch.ops.aten.view.default,
    torch.ops.aten.view_copy.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.expand_copy.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze_copy.dim,
    torch.ops.aten.squeeze_copy.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.unsqueeze_copy.default,
    torch.ops.aten.to.dtype,
    torch.ops.aten._to_copy.default,
})


def _evaluate_node_on_inputs(
    node: Any,
    placeholder_to_input: dict[torch.fx.Node, Any],
    gm: torch.fx.GraphModule | None,
    cache: dict[torch.fx.Node, Any],
) -> Any:
  """Evaluates a node using example inputs or attributes."""
  if not isinstance(node, torch.fx.Node):
    return node

  if node in cache:
    return cache[node]

  if node.op == "placeholder":
    val = placeholder_to_input.get(node)
    cache[node] = val
    return val

  if node.op == "get_attr":
    if gm is not None and isinstance(node.target, str):
      try:
        val = torch.fx.graph_module._get_attr(gm, node.target)
        cache[node] = val
        return val
      except (AttributeError, KeyError):
        cache[node] = None
        return None
    cache[node] = None
    return None

  if node.op == "call_function" and node.target in _EVALUATABLE_OPS:
    args = [
        _evaluate_node_on_inputs(a, placeholder_to_input, gm, cache)
        for a in node.args
    ]
    kwargs = {
        k: _evaluate_node_on_inputs(v, placeholder_to_input, gm, cache)
        for k, v in node.kwargs.items()
    }
    if not args or args[0] is None or not isinstance(args[0], torch.Tensor):
      cache[node] = None
      return None
    if callable(node.target):
      try:
        res = node.target(*args, **kwargs)
        cache[node] = res
        return res
      except Exception:
        cache[node] = None
        return None

  cache[node] = None
  return None


def _extract_constant_scalar(
    val: Any,
    gm: torch.fx.GraphModule | None = None,
    placeholder_to_input: dict[torch.fx.Node, Any] | None = None,
    cache: dict[torch.fx.Node, Any] | None = None,
) -> float | None:
  """Extracts a constant scalar float value from a literal, Tensor, or fx.Node."""
  if val is None:
    return None

  if isinstance(val, (int, float, bool)):
    return float(val)

  if isinstance(val, torch.Tensor):
    if val.numel() == 1:
      try:
        return float(val.detach().cpu().item())
      except Exception:
        return None
    return None

  if not isinstance(val, fx.Node):
    return None

  if cache is not None and val in cache:
    cached = cache[val]
    if isinstance(cached, (int, float)):
      return float(cached)

  if val.op == "placeholder" and placeholder_to_input:
    inp = placeholder_to_input.get(val)
    res = _extract_constant_scalar(inp, gm, placeholder_to_input, cache)
    if cache is not None:
      cache[val] = res
    return res

  if val.op == "get_attr":
    if gm is not None and isinstance(val.target, str):
      try:
        attr = torch.fx.graph_module._get_attr(gm, val.target)
        res = _extract_constant_scalar(attr, gm, placeholder_to_input, cache)
        if cache is not None:
          cache[val] = res
        return res
      except (AttributeError, KeyError):
        pass
    return None

  if val.op == "call_function":
    target = val.target
    if _matches_op_name(target, "aten.scalar_tensor"):
      if val.args:
        res = _extract_constant_scalar(
            val.args[0], gm, placeholder_to_input, cache
        )
        if cache is not None:
          cache[val] = res
        return res
    elif _matches_op_name(target, "aten.full"):
      fill_val = (
          val.args[1]
          if len(val.args) >= 2
          else val.kwargs.get("fill_value", None)
      )
      res = _extract_constant_scalar(fill_val, gm, placeholder_to_input, cache)
      if cache is not None:
        cache[val] = res
      return res
    elif _matches_op_name(target, "aten.ones"):
      shape = val.args[0] if val.args else val.kwargs.get("size", ())
      if shape in ((), [], (1,), [1]):
        return 1.0
    elif _matches_op_name(target, "aten.zeros"):
      shape = val.args[0] if val.args else val.kwargs.get("size", ())
      if shape in ((), [], (1,), [1]):
        return 0.0
    elif _matches_op_name(target, "aten.mul") or target == operator.mul:
      if len(val.args) >= 2:
        a = _extract_constant_scalar(
            val.args[0], gm, placeholder_to_input, cache
        )
        b = _extract_constant_scalar(
            val.args[1], gm, placeholder_to_input, cache
        )
        if a is not None and b is not None:
          res = float(a * b)
          if cache is not None:
            cache[val] = res
          return res
    elif _matches_op_name(target, "aten.div") or target == operator.truediv:
      if len(val.args) >= 2:
        a = _extract_constant_scalar(
            val.args[0], gm, placeholder_to_input, cache
        )
        b = _extract_constant_scalar(
            val.args[1], gm, placeholder_to_input, cache
        )
        if a is not None and b is not None and b != 0:
          res = float(a / b)
          if cache is not None:
            cache[val] = res
          return res
    elif _matches_op_name(target, "aten.pow") or target == operator.pow:
      if len(val.args) >= 2:
        base = _extract_constant_scalar(
            val.args[0], gm, placeholder_to_input, cache
        )
        exp = _extract_constant_scalar(
            val.args[1], gm, placeholder_to_input, cache
        )
        if base is not None and exp is not None:
          try:
            res = float(base**exp)
            if cache is not None:
              cache[val] = res
            return res
          except Exception:
            pass
    elif _matches_op_name(target, "aten.sqrt") or target == math.sqrt:
      if val.args:
        x = _extract_constant_scalar(
            val.args[0], gm, placeholder_to_input, cache
        )
        if x is not None and x >= 0:
          res = float(math.sqrt(x))
          if cache is not None:
            cache[val] = res
          return res
    elif _matches_op_name(target, "aten.rsqrt"):
      if val.args:
        x = _extract_constant_scalar(
            val.args[0], gm, placeholder_to_input, cache
        )
        if x is not None and x > 0:
          res = float(1.0 / math.sqrt(x))
          if cache is not None:
            cache[val] = res
          return res
    elif _matches_op_name(target, "aten.reciprocal"):
      if val.args:
        x = _extract_constant_scalar(
            val.args[0], gm, placeholder_to_input, cache
        )
        if x is not None and x != 0:
          res = float(1.0 / x)
          if cache is not None:
            cache[val] = res
          return res
    elif _matches_op_name(target, "aten.to") or _matches_op_name(
        target, "aten._to_copy"
    ):
      if val.args:
        res = _extract_constant_scalar(
            val.args[0], gm, placeholder_to_input, cache
        )
        if cache is not None:
          cache[val] = res
        return res
    elif val.target in (float, int, torch.sym_float, torch.sym_sqrt):
      if val.args:
        x = _extract_constant_scalar(
            val.args[0], gm, placeholder_to_input, cache
        )
        if x is not None:
          res = float(x)
          if cache is not None:
            cache[val] = res
          return res

  # Check metadata val if available
  meta_val = val.meta.get("val")
  if meta_val is not None:
    if isinstance(meta_val, (int, float)):
      return float(meta_val)
    if hasattr(meta_val, "numel") and meta_val.numel() == 1:
      try:
        if hasattr(meta_val, "item"):
          return float(meta_val.item())
      except Exception:
        pass

  return None


def _extract_scaled_tensor(
    node: Any,
    gm: torch.fx.GraphModule | None = None,
    placeholder_to_input: dict[torch.fx.Node, Any] | None = None,
    cache: dict[torch.fx.Node, Any] | None = None,
) -> tuple[torch.fx.Node, float] | None:
  """Extracts (orig_tensor_node, scale_multiplier) if node is scaled."""
  if not isinstance(node, fx.Node) or node.op != "call_function":
    return None

  curr = node
  # Unwrap optional clone or contiguous operations
  while curr.op == "call_function" and (
      _matches_op_name(curr.target, "aten.clone")
      or _matches_op_name(curr.target, "aten.contiguous")
  ):
    if curr.args and isinstance(curr.args[0], fx.Node):
      curr = curr.args[0]
    else:
      break

  if curr.op != "call_function" or len(curr.args) < 2:
    return None

  target = curr.target

  if _matches_op_name(target, "aten.mul") or target == operator.mul:
    arg0, arg1 = curr.args[0], curr.args[1]
    s = _extract_constant_scalar(arg1, gm, placeholder_to_input, cache)
    if s is not None and math.isfinite(s) and s != 0:
      if isinstance(arg0, fx.Node):
        sub = _extract_scaled_tensor(arg0, gm, placeholder_to_input, cache)
        if sub is not None:
          return sub[0], sub[1] * s
        return arg0, s
    s = _extract_constant_scalar(arg0, gm, placeholder_to_input, cache)
    if s is not None and math.isfinite(s) and s != 0:
      if isinstance(arg1, fx.Node):
        sub = _extract_scaled_tensor(arg1, gm, placeholder_to_input, cache)
        if sub is not None:
          return sub[0], sub[1] * s
        return arg1, s

  elif _matches_op_name(target, "aten.div") or target == operator.truediv:
    arg0, arg1 = curr.args[0], curr.args[1]
    s = _extract_constant_scalar(arg1, gm, placeholder_to_input, cache)
    if s is not None and math.isfinite(s) and s != 0:
      factor = 1.0 / s
      if isinstance(arg0, fx.Node):
        sub = _extract_scaled_tensor(arg0, gm, placeholder_to_input, cache)
        if sub is not None:
          return sub[0], sub[1] * factor
        return arg0, factor

  return None


def _is_redundant_mask(
    mask_node: Any,
    placeholder_to_input: dict[torch.fx.Node, Any],
    gm: torch.fx.GraphModule | None,
    cache: dict[torch.fx.Node, Any],
) -> bool:
  """Returns True if mask is redundant (empty, all-zero, or all-True)."""
  if mask_node is None:
    return True

  if not isinstance(mask_node, fx.Node):
    if isinstance(mask_node, torch.Tensor):
      if mask_node.numel() == 0:
        return True
      if mask_node.dtype == torch.bool and bool(torch.all(mask_node).item()):
        return True
      if mask_node.is_floating_point() and bool(
          torch.all(mask_node == 0).item()
      ):
        return True
    return False

  val = _evaluate_node_on_inputs(mask_node, placeholder_to_input, gm, cache)
  if val is not None and isinstance(val, torch.Tensor):
    try:
      if val.numel() == 0:
        return True
      if val.dtype == torch.bool and bool(torch.all(val).item()):
        return True
      if val.is_floating_point() and bool(torch.all(val == 0).item()):
        return True
    except Exception:
      pass

  if mask_node.op == "call_function":
    target = mask_node.target
    if _matches_op_name(target, "aten.empty"):
      shape = (
          mask_node.args[0]
          if mask_node.args
          else mask_node.kwargs.get("size", ())
      )
      if shape in ([0], (0,), [], (), [0, 0], (0, 0)):
        return True
    elif _matches_op_name(target, "aten.zeros"):
      return True
    elif _matches_op_name(target, "aten.full"):
      fill_val = (
          mask_node.args[1]
          if len(mask_node.args) >= 2
          else mask_node.kwargs.get("fill_value")
      )
      if fill_val in (0, 0.0):
        return True
    elif _matches_op_name(target, "aten.ones"):
      dtype = mask_node.kwargs.get("dtype") or (
          mask_node.args[1] if len(mask_node.args) >= 2 else None
      )
      if dtype == torch.bool:
        return True
    elif _matches_op_name(target, "aten.to") or _matches_op_name(
        target, "aten._to_copy"
    ):
      if mask_node.args and isinstance(mask_node.args[0], fx.Node):
        return _is_redundant_mask(
            mask_node.args[0], placeholder_to_input, gm, cache
        )

  return False


def _get_head_dim(
    node: fx.Node,
    placeholder_to_input: dict[torch.fx.Node, Any] | None = None,
) -> int | None:
  """Infers the head dimension (last dimension) of query or key."""
  if placeholder_to_input and node in placeholder_to_input:
    inp = placeholder_to_input[node]
    if isinstance(inp, torch.Tensor) and inp.ndim >= 1:
      return inp.shape[-1]

  meta_val = node.meta.get("val")
  if (
      meta_val is not None
      and hasattr(meta_val, "shape")
      and len(meta_val.shape) >= 1
  ):
    dim = meta_val.shape[-1]
    if isinstance(dim, int):
      return dim
    try:
      return int(dim)
    except Exception:
      pass

  tensor_meta = node.meta.get("tensor_meta")
  if (
      tensor_meta is not None
      and hasattr(tensor_meta, "shape")
      and len(tensor_meta.shape) >= 1
  ):
    dim = tensor_meta.shape[-1]
    if isinstance(dim, int):
      return dim
    try:
      return int(dim)
    except Exception:
      pass

  if node.op == "call_function":
    target = node.target
    if (
        _matches_op_name(target, "aten.view")
        or _matches_op_name(target, "aten.reshape")
        or _matches_op_name(target, "aten._unsafe_view")
    ) and len(node.args) >= 2:
      shape = node.args[1]
      if isinstance(shape, (list, tuple)) and shape:
        last = shape[-1]
        if isinstance(last, int) and last > 0:
          return last
    elif (
        (
            _matches_op_name(target, "aten.clone")
            or _matches_op_name(target, "aten.contiguous")
            or _matches_op_name(target, "aten.mul")
            or _matches_op_name(target, "aten.div")
        )
        and node.args
        and isinstance(node.args[0], fx.Node)
    ):
      return _get_head_dim(node.args[0], placeholder_to_input)

  return None


def _get_arg_or_kwarg(
    node: fx.Node, arg_idx: int, kwarg_names: tuple[str, ...]
) -> Any:
  """Retrieves an argument by positional index or keyword name."""
  if len(node.args) > arg_idx:
    return node.args[arg_idx]
  for kw in kwarg_names:
    if kw in node.kwargs:
      return node.kwargs[kw]
  return None


def _set_arg_or_kwarg(
    node: fx.Node, arg_idx: int, kwarg_name: str, new_val: Any
) -> None:
  """Sets an argument by keyword name or positional index."""
  if kwarg_name in node.kwargs:
    new_kwargs = dict(node.kwargs)
    new_kwargs[kwarg_name] = new_val
    node.kwargs = new_kwargs
  elif len(node.args) > arg_idx:
    args_list = list(node.args)
    args_list[arg_idx] = new_val
    node.args = tuple(args_list)
  else:
    new_kwargs = dict(node.kwargs)
    new_kwargs[kwarg_name] = new_val
    node.kwargs = new_kwargs


def _set_arg(node: fx.Node, arg_idx: int, new_val: Any) -> None:
  """Replaces positional argument at arg_idx."""
  args_list = list(node.args)
  args_list[arg_idx] = new_val
  node.args = tuple(args_list)


def _set_scale(node: fx.Node, scale_idx: int, new_scale: float | None) -> None:
  """Sets the scale argument on an SDPA node."""
  if new_scale is None:
    if "scale" in node.kwargs:
      new_kwargs = dict(node.kwargs)
      del new_kwargs["scale"]
      node.kwargs = new_kwargs
    if len(node.args) > scale_idx:
      args_list = list(node.args)
      args_list[scale_idx] = None
      node.args = tuple(args_list)
  else:
    if len(node.args) > scale_idx:
      args_list = list(node.args)
      args_list[scale_idx] = new_scale
      node.args = tuple(args_list)
    else:
      new_kwargs = dict(node.kwargs)
      new_kwargs["scale"] = new_scale
      node.kwargs = new_kwargs


def _erase_unused_chain(node: fx.Node, graph: fx.Graph) -> None:
  """Erases node and any of its inputs that become dead."""
  to_check = [node]
  while to_check:
    n = to_check.pop(0)
    if (
        n.graph is not None
        and not n.users
        and n.op not in ("placeholder", "output")
    ):
      inputs = [arg for arg in n.args if isinstance(arg, fx.Node)]
      inputs.extend([v for v in n.kwargs.values() if isinstance(v, fx.Node)])
      graph.erase_node(n)
      to_check.extend(inputs)


def apply(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
    example_inputs: Sequence[Any] | None = None,
) -> None:
  """Optimizes attention calls and folds constant scaling into SDPA."""
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

  placeholder_to_input: dict[torch.fx.Node, Any] = {}
  if example_inputs is not None:
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    for i, p in enumerate(placeholders):
      if i < len(example_inputs):
        placeholder_to_input[p] = example_inputs[i]

  cache: dict[torch.fx.Node, Any] = {}
  dead_nodes: list[torch.fx.Node] = []
  modified = False

  for node in list(graph.nodes):
    if node.op != "call_function":
      continue

    spec = _SDPA_SPECS.get(node.target)
    if spec is None:
      continue

    query_node = node.args[spec.q_idx] if len(node.args) > spec.q_idx else None
    key_node = node.args[spec.k_idx] if len(node.args) > spec.k_idx else None
    mask_node = _get_arg_or_kwarg(
        node, spec.mask_idx, (spec.mask_kwarg, "attn_mask", "attn_bias")
    )
    is_causal = (
        _get_arg_or_kwarg(node, spec.is_causal_idx, ("is_causal",)) or False
    )
    curr_scale = _get_arg_or_kwarg(node, spec.scale_idx, ("scale",))
    orig_scale_node = curr_scale if isinstance(curr_scale, fx.Node) else None

    # 1. Optimize mask: eliminate redundant mask allocations
    if mask_node is not None and _is_redundant_mask(
        mask_node, placeholder_to_input, gm, cache
    ):
      if not spec.is_bwd:
        _set_arg_or_kwarg(node, spec.mask_idx, spec.mask_kwarg, None)
        new_kwargs = dict(node.kwargs)
        for kw in ("attn_mask", "attn_bias"):
          if kw in new_kwargs:
            new_kwargs[kw] = None
        node.kwargs = new_kwargs
        if isinstance(mask_node, fx.Node):
          dead_nodes.append(mask_node)
        modified = True
        mask_node = None
      else:
        # For backward pass, replace with an empty 1D tensor
        if not (
            isinstance(mask_node, fx.Node)
            and mask_node.target == torch.ops.aten.empty.memory_format
        ):
          with graph.inserting_before(node):
            empty_mask = graph.call_function(
                torch.ops.aten.empty.memory_format, ([0],)
            )
          _set_arg_or_kwarg(node, spec.mask_idx, spec.mask_kwarg, empty_mask)
          if isinstance(mask_node, fx.Node):
            dead_nodes.append(mask_node)
          modified = True
          mask_node = empty_mask

    # 2. Canonicalize unmasked attention pattern: attn_mask is None, is_causal is False
    if mask_node is None and not is_causal:
      if _get_arg_or_kwarg(node, spec.is_causal_idx, ("is_causal",)) is None:
        _set_arg_or_kwarg(node, spec.is_causal_idx, "is_causal", False)
        modified = True

    # 3. Canonicalize and fold scale computations
    scale_val = _extract_constant_scalar(
        curr_scale, gm, placeholder_to_input, cache
    )

    # Check for constant pre-scaling on Q and K
    q_scale_info = (
        _extract_scaled_tensor(query_node, gm, placeholder_to_input, cache)
        if isinstance(query_node, fx.Node)
        else None
    )
    k_scale_info = (
        _extract_scaled_tensor(key_node, gm, placeholder_to_input, cache)
        if isinstance(key_node, fx.Node)
        else None
    )

    s_q = q_scale_info[1] if q_scale_info else 1.0
    s_k = k_scale_info[1] if k_scale_info else 1.0
    total_factor = s_q * s_k

    if q_scale_info is not None or k_scale_info is not None:
      can_fold = False
      new_scale = None

      if scale_val is not None:
        new_scale = float(scale_val * total_factor)
        can_fold = True
      else:
        # Default scale is 1.0 / sqrt(head_dim)
        if math.isclose(total_factor, 1.0, rel_tol=1e-6):
          new_scale = None
          can_fold = True
        else:
          head_dim = _get_head_dim(
              q_scale_info[0] if q_scale_info else query_node,
              placeholder_to_input,
          )
          if head_dim is None and k_scale_info:
            head_dim = _get_head_dim(k_scale_info[0], placeholder_to_input)
          if head_dim is not None and head_dim > 0:
            default_scale = 1.0 / math.sqrt(head_dim)
            new_scale = float(default_scale * total_factor)
            can_fold = True

      if can_fold or math.isclose(total_factor, 1.0, rel_tol=1e-6):
        if q_scale_info is not None:
          _set_arg(node, spec.q_idx, q_scale_info[0])
          dead_nodes.append(query_node)
          modified = True
        if k_scale_info is not None:
          _set_arg(node, spec.k_idx, k_scale_info[0])
          dead_nodes.append(key_node)
          modified = True
        _set_scale(node, spec.scale_idx, new_scale)
        if orig_scale_node is not None:
          dead_nodes.append(orig_scale_node)
        modified = True

    elif orig_scale_node is not None and scale_val is not None:
      # Scale was computed as an FX graph node and can be canonicalized to float
      _set_scale(node, spec.scale_idx, scale_val)
      dead_nodes.append(orig_scale_node)
      modified = True

  if modified:
    for n in dead_nodes:
      _erase_unused_chain(n, graph)
    graph.eliminate_dead_code()
    graph.lint()
    if gm is not None:
      gm.recompile()
