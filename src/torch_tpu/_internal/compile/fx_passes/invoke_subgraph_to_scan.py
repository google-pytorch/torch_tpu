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

"""FX pass rewriting chains of identical `invoke_subgraph` calls into a scan."""

import dataclasses
import operator
from typing import Any, cast

from absl import logging
import torch
from torch import fx
from torch.fx.experimental import symbolic_shapes

# Minimum number of calls in a chain for it to be rewritten.
_MIN_CHAIN_LENGTH = 2

_INVOKE_SUBGRAPH = torch.ops.higher_order.invoke_subgraph
_AUTO_FUNCTIONALIZED_V2 = torch.ops.higher_order.auto_functionalized_v2

# `invoke_subgraph(subgraph, identifier, *operands)`: index of the first
# operand in the call's args.
_FIRST_OPERAND = 2

# `node.meta["custom"]` key of `invoke_subgraph` calls whose chain should be
# rolled into a scan; its value is the number of calls per loop iteration.
SCAN_UNROLL_ANNOTATION = "torch_tpu_scan_unroll"


def _is_call(node: Any, target: Any) -> bool:
  return (
      isinstance(node, fx.Node)
      and node.op == "call_function"
      and node.target is target
  )


def _is_invoke_subgraph(node: Any) -> bool:
  return _is_call(node, _INVOKE_SUBGRAPH)


def _invoke_subgraph_operands(call: fx.Node) -> tuple[Any, ...]:
  """Returns the operands that an `invoke_subgraph` call passes its subgraph."""
  assert _is_invoke_subgraph(call)
  return call.args[_FIRST_OPERAND:]


def _invoke_subgraph_identifier(call: fx.Node) -> str:
  """Returns the identifier of the region an `invoke_subgraph` call runs."""
  assert _is_invoke_subgraph(call)
  identifier = call.args[1]
  assert isinstance(identifier, str)
  return identifier


def _invoke_subgraph_attr(call: fx.Node) -> fx.Node:
  """Returns the `get_attr` node of the subgraph an `invoke_subgraph` runs.

  In FX graphs, `invoke_subgraph`'s subgraph argument is always a `get_attr`
  of a submodule of the graph module.
  """
  assert _is_invoke_subgraph(call)
  attr = call.args[0]
  assert isinstance(attr, fx.Node) and attr.op == "get_attr"
  return attr


def _invoke_subgraph_module(
    gm: fx.GraphModule, call: fx.Node
) -> fx.GraphModule:
  """Returns the subgraph module an `invoke_subgraph` call runs."""
  subgraph = gm.get_submodule(str(_invoke_subgraph_attr(call).target))
  assert isinstance(subgraph, fx.GraphModule)
  return subgraph


def _tensor_val(value: Any) -> torch.Tensor | None:
  """Returns the (fake) tensor value of a node, or None if it isn't one."""
  val = value.meta.get("val") if isinstance(value, fx.Node) else None
  return val if isinstance(val, torch.Tensor) else None


def _num_subgraph_outputs(subgraph: fx.GraphModule) -> int:
  """Returns the number of outputs of an `invoke_subgraph` subgraph."""
  # The output node's only argument is the returned value, which is always a
  # tuple (or list) for `invoke_subgraph` subgraphs.
  (output,) = subgraph.graph.find_nodes(op="output")
  (values,) = output.args
  assert isinstance(values, (tuple, list))
  return len(values)


def _invoke_subgraph_getitems(call: fx.Node) -> dict[int, fx.Node] | None:
  """Returns `{output index: getitem node}` for the users of a call.

  Returns None unless every user of the `invoke_subgraph` call is a `getitem`
  of a different output.
  """
  assert _is_invoke_subgraph(call)
  getitems: dict[int, fx.Node] = {}
  for user in call.users:
    if not _is_call(user, operator.getitem):
      return None
    index = user.args[1]
    if not isinstance(index, int) or index < 0 or index in getitems:
      return None
    getitems[index] = user
  return getitems


def _same_invoke_subgraph(gm: fx.GraphModule, a: fx.Node, b: fx.Node) -> bool:
  """Whether two `invoke_subgraph` calls run the same subgraph.

  Calls of the same region share its identifier and subgraph module.
  """
  return _invoke_subgraph_identifier(a) == _invoke_subgraph_identifier(
      b
  ) and _invoke_subgraph_module(gm, a) is _invoke_subgraph_module(gm, b)


def _functional_subgraph(
    gm: fx.GraphModule,
    subgraph: fx.GraphModule,
    mutated: list[int],
    cache: dict[Any, str],
) -> str | None:
  """Returns the attr name of `subgraph` with its input mutations as outputs.

  The copy drops each trailing `copy_(placeholder, value)` of a mutated input
  and returns `value` after the original outputs, in `mutated` order. Returns
  None if an input is mutated in any other way.
  """
  key = (id(subgraph), tuple(mutated))
  if key in cache:
    return cache[key]
  graph = fx.Graph()
  out = graph.graph_copy(subgraph.graph, {})
  outs = list(out) if isinstance(out, (tuple, list)) else [out]
  placeholders = graph.find_nodes(op="placeholder")
  new_values = []
  for p in mutated:
    copies = [
        user
        for user in placeholders[p].users
        if _is_call(user, torch.ops.aten.copy_.default)
        and user.args[0] is placeholders[p]
    ]
    if len(copies) != 1 or copies[0].users:
      return None
    new_values.append(copies[0].args[1])
    graph.erase_node(copies[0])
  graph.output(tuple(outs + new_values))
  name = f"_functional_subgraph_{len(cache)}"
  gm.add_submodule(name, fx.GraphModule(subgraph, graph))
  cache[key] = name
  return name


def _auto_functionalized_operands(
    node: fx.Node,
) -> tuple[list[Any], list[int]] | None:
  """Returns the operands of an `auto_functionalized_v2(invoke_subgraph)`.

  Returns:
    The operands of the call, and for each mutated base (in output order) the
    operand position it is passed at; or None if an operand is a view of a
    base, which is not supported.
  """
  kwargs = dict(node.kwargs)
  bases = cast(list[fx.Node], kwargs.pop("_all_bases"))
  for key in ("subgraph", "identifier", "_op_schema"):
    kwargs.pop(key, None)
  base_index = {}
  for key in list(kwargs):
    if key.startswith("_arg") and key.endswith("_base_index"):
      base_index[int(key[len("_arg") : -len("_base_index")])] = kwargs.pop(key)
  if not all(key.startswith("arg") for key in kwargs):
    return None  # Views of bases (size, stride, offset).
  num = 1 + max([int(key[len("arg") :]) for key in kwargs] + list(base_index))
  operands = []
  for i in range(num):
    if i in base_index:
      operands.append(bases[base_index[i]])
    elif f"arg{i}" in kwargs:
      operands.append(kwargs[f"arg{i}"])
    else:
      return None
  # The node's outputs are the subgraph outputs followed by the new value of
  # each base.
  mutated = [
      next(i for i, b in base_index.items() if b == j)
      for j in range(len(bases))
  ]
  return operands, mutated


def _functionalize_calls(gm: fx.GraphModule) -> set[fx.Node]:
  """Turns `auto_functionalized_v2(invoke_subgraph)` into functional calls.

  On PyTorch 2.11, AOTAutograd calls a subgraph that mutates one of its inputs
  (e.g. a KV cache updated in place) as `auto_functionalized_v2(invoke_subgraph,
  ...)`. Such calls are rewritten into plain `invoke_subgraph` calls of a copy
  of
  the subgraph that returns the new value of each mutated input as an extra
  output, so that the mutated input can become part of a scan carry. Newer
  versions already call such subgraphs that way when compiling with
  `keep_inference_input_mutations=False`, as TorchTPU does.

  Returns:
    The new `invoke_subgraph` nodes.
  """
  cache: dict[Any, str] = {}
  created = set()
  for node in list(gm.graph.nodes):
    if not (
        _is_call(node, _AUTO_FUNCTIONALIZED_V2)
        and node.args[0] is _INVOKE_SUBGRAPH
    ):
      continue
    found = _auto_functionalized_operands(node)
    if found is None:
      continue
    operands, mutated = found
    subgraph_attr = cast(fx.Node, node.kwargs["subgraph"])
    subgraph = cast(fx.GraphModule, gm.get_submodule(str(subgraph_attr.target)))
    name = _functional_subgraph(gm, subgraph, mutated, cache)
    if name is None:
      continue
    with gm.graph.inserting_before(node):
      call = gm.graph.call_function(
          _INVOKE_SUBGRAPH,
          (
              gm.graph.get_attr(name),
              f"{node.kwargs['identifier']}_functional",
              *operands,
          ),
      )
    call.meta = dict(node.meta)
    node.replace_all_uses_with(call)
    gm.graph.erase_node(node)
    if not subgraph_attr.users:
      gm.graph.erase_node(subgraph_attr)
    created.add(call)
  return created


def _inline(gm: fx.GraphModule, call: fx.Node) -> None:
  """Inlines an `invoke_subgraph` call."""
  subgraph = _invoke_subgraph_module(gm, call)
  env = dict(
      zip(
          subgraph.graph.find_nodes(op="placeholder"),
          _invoke_subgraph_operands(call),
      )
  )
  with gm.graph.inserting_before(call):
    out = gm.graph.graph_copy(subgraph.graph, env)
  outs = list(out) if isinstance(out, (tuple, list)) else [out]
  for getitem in list(call.users):
    getitem.replace_all_uses_with(outs[cast(int, getitem.args[1])])
    gm.graph.erase_node(getitem)
  subgraph_attr = _invoke_subgraph_attr(call)
  gm.graph.erase_node(call)
  if not subgraph_attr.users:
    gm.graph.erase_node(subgraph_attr)


def _annotated_unroll(call: fx.Node) -> int | None:
  return call.meta.get("custom", {}).get(SCAN_UNROLL_ANNOTATION)


def cache_key_tag(gm: fx.GraphModule) -> str:
  """Returns a compile cache key tag for the scan annotations in `gm`.

  The annotations decide how the pass rewrites `gm`, but they are node
  metadata, which compile caches do not hash. Tagging the cache key with them
  keeps a program compiled with one `scan_unroll` from being reused for
  another.
  """
  values = {
      value
      for n in gm.graph.nodes
      if _is_invoke_subgraph(n) and (value := _annotated_unroll(n)) is not None
  }
  if not values:
    return ""
  return f"{SCAN_UNROLL_ANNOTATION}={','.join(map(str, sorted(values)))}"


@dataclasses.dataclass
class _Chain:
  """A maximal chain of calls to one subgraph."""

  calls: list[fx.Node]
  # Operand position -> subgraph output index feeding it in the next call.
  carry_pos_to_out: dict[int, int]


def _carry_link(
    gm: fx.GraphModule, call: fx.Node, next_call: fx.Node
) -> dict[int, int] | None:
  """Returns how the outputs of `call` feed `next_call`, if they can be chained.

  Returns:
    `{operand position in next_call: output index of call}`, if every output
    of `call` feeds exactly one operand of `next_call` and nothing else, and
    both run the same subgraph. Otherwise None.
  """
  if not _same_invoke_subgraph(gm, call, next_call):
    return None
  getitems = _invoke_subgraph_getitems(call)
  if getitems is None or len(getitems) != _num_subgraph_outputs(
      _invoke_subgraph_module(gm, call)
  ):
    return None
  link: dict[int, int] = {}
  next_operands = _invoke_subgraph_operands(next_call)
  for out_index, getitem in getitems.items():
    if list(getitem.users) != [next_call]:
      return None
    positions = [p for p, op in enumerate(next_operands) if op is getitem]
    if len(positions) != 1:
      return None
    link[positions[0]] = out_index
  return link


def _next_call(call: fx.Node) -> fx.Node | None:
  """Returns the single `invoke_subgraph` call that uses the outputs of `call`.

  A call in the middle of a chain hands all its outputs to the next call, so
  the chain ends at a call whose outputs are used by several nodes, by none,
  or by something other than an `invoke_subgraph` call.
  """
  getitems = _invoke_subgraph_getitems(call)
  if not getitems:
    return None
  users = {user for getitem in getitems.values() for user in getitem.users}
  if len(users) != 1:
    return None
  (next_call,) = users
  return next_call if _is_invoke_subgraph(next_call) else None


def _find_chains(gm: fx.GraphModule) -> list[_Chain]:
  """Returns the maximal chains of calls to the same subgraph in `gm`."""
  chains: list[_Chain] = []
  used: set[fx.Node] = set()
  for start in gm.graph.nodes:
    if not _is_invoke_subgraph(start) or start in used:
      continue
    calls = [start]
    chain_link = None
    while (next_call := _next_call(calls[-1])) is not None:
      if next_call in used:
        break
      link = _carry_link(gm, calls[-1], next_call)
      if link is None:
        break
      if chain_link is None:
        chain_link = link
      elif link != chain_link:
        # One loop body serves every call, so all calls must pass the carried
        # values through the same operand positions.
        break
      calls.append(next_call)
    # The scan replaces the last call's outputs, so they too must be read only
    # through one `getitem` per output. (Earlier calls were checked by
    # `_carry_link`.)
    if _invoke_subgraph_getitems(calls[-1]) is None:
      calls.pop()
    if chain_link is not None and len(calls) >= _MIN_CHAIN_LENGTH:
      chains.append(_Chain(calls=calls, carry_pos_to_out=chain_link))
      used.update(calls)
  return chains


@dataclasses.dataclass
class _Operands:
  """How each operand position of a chain's calls maps to the scan."""

  # Positions of operands fed by the previous call's outputs.
  carry: list[int] = dataclasses.field(default_factory=list)
  # Positions of operands that differ per call, stacked and scanned over.
  xs: list[int] = dataclasses.field(default_factory=list)
  # Positions of operands shared by all calls, the scan's `additional_inputs`.
  extra: list[int] = dataclasses.field(default_factory=list)


def _shared(values: list[Any]) -> bool:
  """Whether `values` are all the same node, or all the same constant."""
  first = values[0]
  if isinstance(first, fx.Node):
    return all(v is first for v in values)
  return all(type(v) is type(first) and v == first for v in values)


def _stackable(values: list[Any]) -> bool:
  """Whether `values` are tensors of one shape, dtype and device.

  Symbolic sizes count as equal only if they are known to be, so that the
  check adds no guards.
  """
  vals = [_tensor_val(v) for v in values]
  first = vals[0]
  return first is not None and all(
      val is not None
      and val.dtype == first.dtype
      and val.device == first.device
      and symbolic_shapes.statically_known_true(
          symbolic_shapes.sym_eq(val.shape, first.shape)
      )
      for val in vals
  )


def _classify_operands(chain: _Chain) -> _Operands | None:
  """Returns how the chain's operands map to the scan, or None if they can't.

  The carries must be tensors that keep their shape from call to call, operands
  that differ per call must be tensors of one shape, dtype and device so that
  they can be stacked (not, e.g., different Python ints), and shared operands
  become `additional_inputs` whatever their type. Whether the scan supports an
  operand (e.g. `validate_subgraph_args_types` only takes tensors, ints and
  SymInts) is left to the scan to check.
  """
  first_operands = _invoke_subgraph_operands(chain.calls[0])
  operands = _Operands(carry=sorted(chain.carry_pos_to_out))
  for p in operands.carry:
    if not _stackable([_invoke_subgraph_operands(c)[p] for c in chain.calls]):
      return None
  for p in range(len(first_operands)):
    if p in chain.carry_pos_to_out:
      continue
    values = [_invoke_subgraph_operands(c)[p] for c in chain.calls]
    if _shared(values):
      operands.extra.append(p)
    elif _stackable(values):
      operands.xs.append(p)
    else:
      return None
  return operands


def _build_body(
    subgraph: fx.GraphModule, chain: _Chain, operands: _Operands, unroll: int
) -> fx.GraphModule:
  """Builds the scan body `(carries..., xs..., extras...) -> carries`.

  The body runs the subgraph `unroll` times, threading the carries through. If
  `unroll > 1`, each `xs` slice has a leading dimension of `unroll`.
  """
  placeholders = subgraph.graph.find_nodes(op="placeholder")
  graph = fx.Graph()

  def placeholder(name: str, p: int, leading: tuple[int, ...] = ()) -> fx.Node:
    node = graph.placeholder(name)
    node.meta = dict(placeholders[p].meta)
    val = node.meta.get("val")
    if leading and isinstance(val, torch.Tensor):
      node.meta["val"] = val.new_empty((*leading, *val.shape))
    return node

  carries = {p: placeholder(f"carry_{p}", p) for p in operands.carry}
  xs = {
      p: placeholder(f"x_{p}", p, (unroll,) if unroll > 1 else ())
      for p in operands.xs
  }
  if not operands.xs:
    # A slice of the trip-count input (see `_scan_inputs`), never read.
    trip_count = graph.placeholder("unused_trip_count")
    carry = _tensor_val(
        _invoke_subgraph_operands(chain.calls[0])[operands.carry[0]]
    )
    trip_count.meta["val"] = cast(torch.Tensor, carry).new_zeros(
        (), dtype=torch.int64
    )
  extras = {p: placeholder(f"extra_{p}", p) for p in operands.extra}
  for j in range(unroll):
    env: dict[fx.Node, Any] = {}
    for p, node in {**carries, **extras}.items():
      env[placeholders[p]] = node
    for p, node in xs.items():
      env[placeholders[p]] = (
          node
          if unroll == 1
          else graph.call_function(torch.ops.aten.select.int, (node, 0, j))
      )
    out = graph.graph_copy(subgraph.graph, env)
    assert isinstance(out, (tuple, list))  # See `_num_subgraph_outputs`.
    carries = {p: out[chain.carry_pos_to_out[p]] for p in operands.carry}
  graph.output(tuple(carries[p] for p in operands.carry))
  return fx.GraphModule(subgraph, graph)


def _stack(graph: fx.Graph, values: list[fx.Node]) -> fx.Node:
  """Stacks `values` (see `_stackable`) along a new leading dimension."""
  stacked = graph.call_function(torch.ops.aten.stack.default, (values, 0))
  val = cast(torch.Tensor, _tensor_val(values[0]))
  stacked.meta["val"] = val.new_empty((len(values), *val.shape))
  return stacked


def _views_of_one_tensor(values: list[Any]) -> tuple[fx.Node, int] | None:
  """Returns `(base, start)` if `values` are `base[start]`, `base[start + 1]`..."""
  found = None
  for i, v in enumerate(values):
    if not (
        _is_call(v, torch.ops.aten.select.int)
        and v.args[1] == 0
        and isinstance(v.args[2], int)
    ):
      return None
    base, index = v.args[0], v.args[2]
    if found is None:
      found = (base, index)
    elif base is not found[0] or index != found[1] + i:
      return None
  return found


def _stacked(graph: fx.Graph, values: list[fx.Node]) -> fx.Node:
  """Returns `values` stacked along a new leading dimension.

  If they are consecutive slices of one tensor (e.g. a model holding each
  weight for all layers as one `[num_layers, ...]` tensor and passing layer
  `i` its view `w[i]`), that tensor, or a slice of it, already is the stack and
  no copy is needed.
  """
  found = _views_of_one_tensor(values)
  if found is None:
    return _stack(graph, values)
  base, start = found
  val = base.meta.get("val")
  if not isinstance(val, torch.Tensor):
    return _stack(graph, values)
  if start == 0 and val.shape[0] == len(values):
    return base
  sliced = graph.call_function(
      torch.ops.aten.slice.Tensor, (base, 0, start, start + len(values))
  )
  sliced.meta["val"] = val.new_empty((len(values), *val.shape[1:]))
  return sliced


def _group(graph: fx.Graph, stacked: fx.Node, unroll: int) -> fx.Node:
  """Views a `[n, ...]` stack as `[n / unroll, unroll, ...]`."""
  val = cast(torch.Tensor, _tensor_val(stacked))
  shape = (val.shape[0] // unroll, unroll, *val.shape[1:])
  grouped = graph.call_function(torch.ops.aten.view.default, (stacked, shape))
  grouped.meta["val"] = val.new_empty(shape)
  return grouped


def _scan_inputs(
    graph: fx.Graph, chain: _Chain, operands: _Operands, unroll: int
) -> tuple[list[fx.Node], list[fx.Node], list[Any]]:
  """Inserts the scan's `init`, `xs` and `additional_inputs` into `graph`."""
  first_operands = _invoke_subgraph_operands(chain.calls[0])
  init = [cast(fx.Node, first_operands[p]) for p in operands.carry]
  xs = [
      _stacked(graph, [_invoke_subgraph_operands(c)[p] for c in chain.calls])
      for p in operands.xs
  ]
  if unroll > 1:
    xs = [_group(graph, x, unroll) for x in xs]
  num_steps = len(chain.calls) // unroll
  if not xs:
    # The scan needs at least one scanned input to define the trip count.
    carry = cast(torch.Tensor, _tensor_val(init[0]))
    trip_count = graph.call_function(
        torch.ops.aten.zeros.default,
        ([num_steps],),
        {"dtype": torch.int64, "device": carry.device},
    )
    trip_count.meta["val"] = carry.new_zeros((num_steps,), dtype=torch.int64)
    xs.append(trip_count)
  extras = [first_operands[p] for p in operands.extra]
  return init, xs, extras


def _erase_chain(graph: fx.Graph, chain: _Chain) -> None:
  """Erases the calls of `chain`, their `getitem`s and subgraph `get_attr`s."""
  for call in reversed(chain.calls):
    for getitem in list(call.users):
      graph.erase_node(getitem)
    subgraph_attr = _invoke_subgraph_attr(call)
    graph.erase_node(call)
    if not subgraph_attr.users:
      graph.erase_node(subgraph_attr)


def _free_attr_name(gm: fx.GraphModule, prefix: str) -> str:
  """Returns `prefix` followed by the lowest index not yet an attr of `gm`."""
  index = 0
  while hasattr(gm, f"{prefix}{index}"):
    index += 1
  return f"{prefix}{index}"


def _rewrite(
    gm: fx.GraphModule, chain: _Chain, operands: _Operands, unroll: int
) -> None:
  """Replaces `chain` in `gm` with a scan running `unroll` calls per step."""
  first, last = chain.calls[0], chain.calls[-1]
  body_name = _free_attr_name(gm, "_scan_body_")
  gm.add_submodule(
      body_name,
      _build_body(_invoke_subgraph_module(gm, first), chain, operands, unroll),
  )
  graph = gm.graph
  # Checked by `_find_chains`.
  last_getitems = cast(dict[int, fx.Node], _invoke_subgraph_getitems(last))
  # Before the last call: operands of later calls may be computed after the
  # first one, and nothing outside the chain uses its intermediate results.
  with graph.inserting_before(last):
    init, xs, extras = _scan_inputs(graph, chain, operands, unroll)
    scan = graph.call_function(
        torch.ops.higher_order.scan,
        (graph.get_attr(body_name), init, xs, extras),
    )
    # The scan returns the last call's outputs.
    results = [
        last_getitems.get(chain.carry_pos_to_out[p]) for p in operands.carry
    ]
    carry_vals = [
        _tensor_val(result) if result is not None else _tensor_val(init[i])
        for i, result in enumerate(results)
    ]
    scan.meta["val"] = carry_vals
    for i, old_result in enumerate(results):
      if old_result is None:
        continue
      result = graph.call_function(operator.getitem, (scan, i))
      result.meta["val"] = carry_vals[i]
      old_result.replace_all_uses_with(result)
  _erase_chain(graph, chain)


def _truncate(chain: _Chain, unroll: int) -> _Chain:
  """Drops trailing calls so the chain length is a multiple of `unroll`."""
  keep = len(chain.calls) - len(chain.calls) % unroll
  return _Chain(
      calls=chain.calls[:keep], carry_pos_to_out=chain.carry_pos_to_out
  )


def _chain_unroll(chain: _Chain, unroll: int | None) -> int | None:
  """Returns the calls per iteration for `chain`, or None to leave it."""
  if unroll is not None:
    return unroll
  values = {_annotated_unroll(c) for c in chain.calls}
  if len(values) != 1:
    return None
  (value,) = values
  return value if isinstance(value, int) and value >= 1 else None


def apply(gm: fx.GraphModule, unroll: int | None = 1) -> None:
  """Rewrites chains of identical `invoke_subgraph` calls into scans in place.

  A stack of identical layers becomes a chain of `invoke_subgraph` calls to the
  same subgraph, where the outputs of call `i` feed call `i + 1`:

    y1 = invoke_subgraph(subgraph, "subgraph_0", x0, w0_a, w0_b)[0]
    y2 = invoke_subgraph(subgraph, "subgraph_0", y1, w1_a, w1_b)[0]
    ...

  Such a chain is rewritten into one `torch.ops.higher_order.scan`, which
  TorchTPU lowers to a `stablehlo.while` loop, so that XLA compiles the
  subgraph once:

    * Operands fed by the previous call's outputs become the scan carry.
    * Operands that are the same in every call become `additional_inputs`.
    * Operands that differ per call (e.g. per-layer weights) are stacked along
      a new leading dimension and become the scanned `xs`. The stack is a copy
      made on every run of the graph.

  Only chains whose outputs are all carried (the subgraph returns exactly the
  carried state), and whose intermediate results are only used by the next
  call, are rewritten. Calls wrapped in `auto_functionalized_v2` (subgraphs
  that mutate their inputs) are first made functional (see
  `_functionalize_calls`); those that end up outside any scan are inlined.

  The rewrite only follows the contract of the scan op. Whether the backend's
  scan lowering supports the result (e.g. SymInt inputs, or random ops in the
  body) is for the lowering to check and report.

  Args:
    gm: The graph module to rewrite.
    unroll: Number of consecutive calls per loop iteration (partial unrolling),
      or None to roll only chains whose calls carry a `SCAN_UNROLL_ANNOTATION`,
      with its value. Calls left over when the chain length is not a multiple of
      it stay as `invoke_subgraph` calls.
  """
  functional = _functionalize_calls(gm)
  rewritten = 0
  for chain in _find_chains(gm):
    chain_unroll = _chain_unroll(chain, unroll)
    if chain_unroll is None:
      continue
    if chain_unroll > 1:
      chain = _truncate(chain, chain_unroll)
      if len(chain.calls) < max(_MIN_CHAIN_LENGTH, chain_unroll):
        continue
    operands = _classify_operands(chain)
    if operands is None:
      logging.info(
          "invoke_subgraph_to_scan: not rolling %d calls: an operand is"
          " neither a carry of fixed shape, a value shared by all calls, nor"
          " a tensor of one shape in every call",
          len(chain.calls),
      )
      continue
    _rewrite(gm, chain, operands, chain_unroll)
    logging.info(
        "invoke_subgraph_to_scan: rolled %d calls into a scan (unroll %d)",
        len(chain.calls),
        chain_unroll,
    )
    rewritten += 1
  # In graph order, so that the result is deterministic.
  for node in [n for n in gm.graph.nodes if n in functional]:
    _inline(gm, node)
  if rewritten or functional:
    gm.delete_all_unused_submodules()
    gm.graph.lint()
    gm.recompile()
