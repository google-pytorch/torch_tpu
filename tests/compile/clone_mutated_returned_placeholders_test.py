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

"""Tests for clone_mutated_returned_placeholders FX pass."""

import operator
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import fx
from torch._functorch._aot_autograd import descriptors
from torch_tpu._internal.compile.fx_passes import clone_mutated_returned_placeholders
from tests import seed_test_utils

GraphModule = fx.GraphModule
Graph = fx.Graph
aten = torch.ops.aten
# pylint: disable=protected-access
_UNANNOTATED_ALIAS_OPS = (
    clone_mutated_returned_placeholders._UNANNOTATED_ALIAS_OPS
)
_SET_SOURCE_OPS = clone_mutated_returned_placeholders._SET_SOURCE_OPS

# The required non-tensor arguments of each op in `_UNANNOTATED_ALIAS_OPS`
# and `_SET_SOURCE_OPS`, with values valid for 4-element tensors. Every op
# has an entry, empty if its required arguments are all tensors, so a new op
# without one fails its tests with a KeyError.
_NON_TENSOR_ARGS: dict[torch._ops.OpOverload, dict[str, Any]] = {
    aten._unsafe_view.default: {"size": [2, 2]},
    aten.lift.default: {},
    aten.set.source_Tensor: {},
    aten.set_.source_Tensor: {},
    aten.set_.source_Tensor_storage_offset: {"storage_offset": 0, "size": [4]},
    aten.unsafe_chunk.default: {"chunks": 2},
    aten.unsafe_split.Tensor: {"split_size": 2},
    aten.unsafe_split_with_sizes.default: {"split_sizes": [2, 2]},
}
# pylint: enable=protected-access


def _cloned_placeholders(graph: Graph) -> set[fx.Node]:
  """Returns the placeholders that `graph` clones."""
  return {
      n.args[0]
      for n in graph.nodes
      if n.op == "call_function"
      and n.target == aten.clone.default
      and n.args[0].op == "placeholder"
  }


def _get_readable_op_names(
    ops: frozenset[torch._ops.OpOverload],
) -> list[tuple[str, torch._ops.OpOverload]]:
  """Returns `(name, op)` for each op, sorted by name.

  The name is the op's qualified name with dots replaced by underscores.
  """
  return [(str(op).replace(".", "_"), op) for op in sorted(ops, key=str)]


def _call_on_tensors(
    graph: Graph, op: torch._ops.OpOverload, *tensors: fx.Node
) -> fx.Node:
  """Adds a call of `op` with `tensors` as its tensor arguments, in order.

  The other required arguments come from `_NON_TENSOR_ARGS`.

  Args:
    graph: The graph to add the call to.
    op: The op to call.
    *tensors: One node for each tensor argument of `op`.

  Returns:
    The node of the call.
  """
  remaining = iter(tensors)
  non_tensor_args = _NON_TENSOR_ARGS[op]
  args: list[Any] = []
  for argument in op._schema.arguments:  # pylint: disable=protected-access
    if argument.kwarg_only or argument.has_default_value():
      break
    if isinstance(argument.type, torch.TensorType):
      args.append(next(remaining))
    else:
      args.append(non_tensor_args[argument.name])
  return graph.call_function(op, tuple(args))


def _tensor_placeholders(
    graph: Graph, op: torch._ops.OpOverload
) -> list[fx.Node]:
  """Adds a placeholder for each tensor argument of `op`."""
  return [
      graph.placeholder(argument.name)
      for argument in op._schema.arguments  # pylint: disable=protected-access
      if isinstance(argument.type, torch.TensorType)
  ]


def _add_one_in_place(graph: Graph, node: fx.Node) -> None:
  """Adds `node.add_(1.0)`."""
  graph.call_function(aten.add_.Scalar, (node, 1.0))


def _add_one_in_place_to_result(
    graph: Graph, op: torch._ops.OpOverload, result: fx.Node
) -> None:
  """Adds 1 in place to `result`, or to its first element if it is a list."""
  returns = op._schema.returns  # pylint: disable=protected-access
  if isinstance(returns[0].type, torch.ListType):
    result = graph.call_function(operator.getitem, (result, 0))
  _add_one_in_place(graph, result)


def _empty(graph: Graph) -> fx.Node:
  return graph.call_function(aten.empty.memory_format, ([4],))


def _view(graph: Graph, node: fx.Node) -> fx.Node:
  return graph.call_function(aten.view.default, (node, [2, 2]))


def _split_and_get_first_piece(graph: Graph, node: fx.Node) -> fx.Node:
  """Adds `node.split(2)[0]`."""
  pieces = graph.call_function(aten.split.Tensor, (node, 2))
  return graph.call_function(operator.getitem, (pieces, 0))


def _mm_with_transpose(graph: Graph, p: fx.Node) -> None:
  """Adds `p @ p.t()`."""
  transpose = graph.call_function(aten.t.default, (p,))
  graph.call_function(aten.mm.default, (p, transpose))


def _add_one_in_place_to_view_of_new_tensor(graph: Graph, p: fx.Node) -> None:
  """Adds `(p * 2.0).view(2, 2).add_(1.0)`."""
  new_tensor = graph.call_function(aten.mul.Scalar, (p, 2.0))
  _add_one_in_place(graph, _view(graph, new_tensor))


def _add_into_new_tensor(graph: Graph, p: fx.Node) -> None:
  """Adds `torch.add(p, p, out=torch.empty(4))`."""
  graph.call_function(aten.add.out, (p, p), {"out": _empty(graph)})


def _add_one_in_place_to_view(graph: Graph, p: fx.Node) -> None:
  """Adds `p.view(2, 2).add_(1.0)`."""
  _add_one_in_place(graph, _view(graph, p))


def _zero_first_split_piece(graph: Graph, p: fx.Node) -> None:
  """Adds `p.split(2)[0].zero_()`."""
  piece = _split_and_get_first_piece(graph, p)
  graph.call_function(aten.zero_.default, (piece,))


def _add_new_tensors_into(graph: Graph, p: fx.Node) -> None:
  """Adds `torch.add(a, b, out=p)` for new tensors `a` and `b`."""
  new_tensors = (_empty(graph), _empty(graph))
  graph.call_function(aten.add.out, new_tensors, {"out": p})


def _call_python_function(graph: Graph, p: fx.Node) -> None:
  """Adds a call of a Python function, which has no schema, on `p`."""
  graph.call_function(lambda t: t, (p,))


class CloneMutatedReturnedPlaceholdersTest(seed_test_utils.RepeatableTest):

  def test_clones_mutated_returned_placeholder(self):
    """Verifies that mutated placeholders returned in the output are cloned."""
    graph = Graph()
    p_weight = graph.placeholder("weight")
    p_weight.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_grad = graph.placeholder("grad")
    p_grad.meta["val"] = torch.empty((4, 4), dtype=torch.float32)

    # In-place addition mutating p_weight
    add_node = graph.call_function(
        torch.ops.aten.add_.Tensor, (p_weight, p_grad)
    )
    # The output node returns the mutated placeholder directly
    graph.output((add_node, p_weight))
    gm = GraphModule(torch.nn.Module(), graph)

    clone_mutated_returned_placeholders.apply(gm)

    # Check that a clone node was created right after p_weight
    clone_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertLen(clone_nodes, 1)
    clone_node = clone_nodes[0]
    self.assertEqual(clone_node.args, (p_weight,))
    self.assertEqual(clone_node.meta["val"].shape, p_weight.meta["val"].shape)

    # p_weight's only user should be the clone node
    self.assertEqual(list(p_weight.users.keys()), [clone_node])

    # Downstream compute and output should now use the clone node
    self.assertIn(clone_node, add_node.args)
    output_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    self.assertIn(clone_node, output_node.args[0])
    self.assertNotIn(p_weight, output_node.args[0])

    # Verify that the generated code was recompiled to invoke clone
    self.assertIn("clone", gm.code)

  def test_apply_on_graph_directly(self):
    """Verifies that apply() works directly on a raw torch.fx.Graph instance."""
    graph = Graph()
    p_weight = graph.placeholder("weight")
    p_weight.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_grad = graph.placeholder("grad")
    p_grad.meta["val"] = torch.empty((4, 4), dtype=torch.float32)

    add_node = graph.call_function(
        torch.ops.aten.add_.Tensor, (p_weight, p_grad)
    )
    graph.output((add_node, p_weight))

    # Apply directly on graph, where gm is None
    clone_mutated_returned_placeholders.apply(graph)

    clone_nodes = [
        n
        for n in graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertLen(clone_nodes, 1)
    self.assertEqual(list(p_weight.users.keys()), [clone_nodes[0]])

  def test_recompiles_graph_module(self):
    """Verifies that apply() triggers gm.recompile(), updating gm.code."""
    graph = Graph()
    p_weight = graph.placeholder("weight")
    p_grad = graph.placeholder("grad")
    add_node = graph.call_function(
        torch.ops.aten.add_.Tensor, (p_weight, p_grad)
    )
    graph.output((add_node, p_weight))
    gm = GraphModule(torch.nn.Module(), graph)

    self.assertNotIn("clone", gm.code)
    clone_mutated_returned_placeholders.apply(gm)
    self.assertIn("clone", gm.code)

  def test_passthrough_placeholder_not_cloned(self):
    """Verifies that placeholders only returned without compute uses are not cloned."""
    graph = Graph()
    p_in = graph.placeholder("in_tensor")
    p_in.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    graph.output((p_in,))
    gm = GraphModule(torch.nn.Module(), graph)

    clone_mutated_returned_placeholders.apply(gm)

    clone_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertEmpty(clone_nodes)

  def test_compute_placeholder_not_returned_not_cloned(self):
    """Verifies that compute placeholders not present in the output are not cloned."""
    graph = Graph()
    p_x = graph.placeholder("x")
    p_x.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_y = graph.placeholder("y")
    p_y.meta["val"] = torch.empty((4, 4), dtype=torch.float32)

    out = graph.call_function(torch.ops.aten.add.Tensor, (p_x, p_y))
    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    clone_mutated_returned_placeholders.apply(gm)

    clone_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertEmpty(clone_nodes)

  def test_multiple_mutated_returned_placeholders(self):
    """Verifies multiple placeholders are all cloned."""
    graph = Graph()
    p_weight = graph.placeholder("weight")
    p_weight.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_m = graph.placeholder("exp_avg")
    p_m.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_v = graph.placeholder("exp_avg_sq")
    p_v.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_step = graph.placeholder("step")
    p_step.meta["val"] = torch.empty((), dtype=torch.int32)
    p_grad = graph.placeholder("grad")
    p_grad.meta["val"] = torch.empty((4, 4), dtype=torch.float32)

    # In-place ops mutating weight, m, v, step
    graph.call_function(torch.ops.aten.add_.Tensor, (p_step, 1))
    graph.call_function(torch.ops.aten.add_.Tensor, (p_m, p_grad))
    graph.call_function(torch.ops.aten.add_.Tensor, (p_v, p_grad))
    graph.call_function(torch.ops.aten.add_.Tensor, (p_weight, p_m))

    graph.output((p_weight, p_m, p_v, p_step))
    gm = GraphModule(torch.nn.Module(), graph)

    clone_mutated_returned_placeholders.apply(gm)

    clone_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertLen(clone_nodes, 4)
    for p in (p_weight, p_m, p_v, p_step):
      self.assertLen(p.users, 1)
      cloned_user = list(p.users.keys())[0]
      self.assertEqual(cloned_user.target, torch.ops.aten.clone.default)

  def test_symint_placeholder_not_cloned(self):
    """Verifies that SymInt placeholders in compute/output are not cloned."""
    shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
    symint_s0 = shape_env.create_symintnode(
        shape_env.create_symbol(4, torch._dynamo.source.ConstantSource("s0")),
        hint=4,
    )

    fake_mode = torch._subclasses.fake_tensor.FakeTensorMode(
        shape_env=shape_env
    )
    with fake_mode:
      fake_tensor = torch.empty((symint_s0, 4), dtype=torch.float32)

    graph = Graph()
    p_s0 = graph.placeholder("s0")
    p_s0.meta["val"] = symint_s0
    p_x = graph.placeholder("x")
    p_x.meta["val"] = fake_tensor

    # p_s0 is used in compute (slice) and also returned in output
    slice_node = graph.call_function(
        torch.ops.aten.slice.Tensor, (p_x, 0, 0, p_s0)
    )
    graph.output((slice_node, p_s0))
    gm = GraphModule(torch.nn.Module(), graph)

    clone_mutated_returned_placeholders.apply(gm)

    clone_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertEmpty(clone_nodes)

  @parameterized.named_parameters(
      ("mm", _mm_with_transpose),
      ("split", _split_and_get_first_piece),
      ("write_to_new_tensor", _add_one_in_place_to_view_of_new_tensor),
      ("out_source", _add_into_new_tensor),
  )
  def test_read_only_returned_placeholder_not_cloned(self, read):
    """Verifies that a returned placeholder that is only read is not cloned."""
    graph = Graph()
    p = graph.placeholder("p")
    read(graph, p)
    graph.output((p,))

    clone_mutated_returned_placeholders.apply(graph)

    self.assertEmpty(_cloned_placeholders(graph))

  @parameterized.named_parameters(
      ("in_place", _add_one_in_place),
      ("through_view", _add_one_in_place_to_view),
      ("through_split", _zero_first_split_piece),
      ("out_kwarg", _add_new_tensors_into),
      ("schemaless_target", _call_python_function),
  )
  def test_written_returned_placeholder_is_cloned(self, write):
    """Verifies writes in place, through aliases, and by unknown targets."""
    graph = Graph()
    p = graph.placeholder("p")
    write(graph, p)
    graph.output((p,))

    clone_mutated_returned_placeholders.apply(graph)

    self.assertEqual(_cloned_placeholders(graph), {p})

  @parameterized.named_parameters(
      *_get_readable_op_names(_UNANNOTATED_ALIAS_OPS)
  )
  def test_write_through_unannotated_alias_is_cloned(self, op):
    """Verifies that a write to the result of `op` writes to its inputs."""
    graph = Graph()
    inputs = _tensor_placeholders(graph, op)
    result = _call_on_tensors(graph, op, *inputs)
    _add_one_in_place_to_result(graph, op, result)
    graph.output(tuple(inputs))

    clone_mutated_returned_placeholders.apply(graph)

    self.assertEqual(_cloned_placeholders(graph), set(inputs))

  @parameterized.named_parameters(
      *_get_readable_op_names(_UNANNOTATED_ALIAS_OPS)
  )
  def test_unannotated_alias_op_does_not_write(self, op):
    """Verifies that `op` alone does not count as a write to its inputs."""
    graph = Graph()
    inputs = _tensor_placeholders(graph, op)
    graph.output((*inputs, _call_on_tensors(graph, op, *inputs)))

    clone_mutated_returned_placeholders.apply(graph)

    self.assertEmpty(_cloned_placeholders(graph))

  @parameterized.product(
      op=sorted(_SET_SOURCE_OPS, key=str),
      write_self=[True, False],
  )
  def test_write_after_set_source_is_cloned(self, op, write_self):
    """Verifies that `self` and the result of `self.set_(source)` alias it."""
    graph = Graph()
    source = graph.placeholder("source")
    self_ = _empty(graph)
    result = _call_on_tensors(graph, op, self_, source)
    _add_one_in_place(graph, self_ if write_self else result)
    graph.output((source,))

    clone_mutated_returned_placeholders.apply(graph)

    self.assertEqual(_cloned_placeholders(graph), {source})

  @parameterized.named_parameters(*_get_readable_op_names(_SET_SOURCE_OPS))
  def test_set_source_is_not_written(self, op):
    """Verifies that `self.set_(source)` alone does not write to `source`."""
    graph = Graph()
    source = graph.placeholder("source")
    _call_on_tensors(graph, op, _empty(graph), source)
    graph.output((source,))

    clone_mutated_returned_placeholders.apply(graph)

    self.assertEmpty(_cloned_placeholders(graph))

  @parameterized.named_parameters(*_get_readable_op_names(_SET_SOURCE_OPS))
  def test_write_to_set_placeholder_writes_new_source(self, op):
    """Verifies that a placeholder rebound by `set_` aliases its new source."""
    graph = Graph()
    p = graph.placeholder("p")
    source = graph.placeholder("source")
    _call_on_tensors(graph, op, p, source)
    _add_one_in_place(graph, p)
    graph.output((p, source))

    clone_mutated_returned_placeholders.apply(graph)

    self.assertEqual(_cloned_placeholders(graph), {p, source})

  @parameterized.named_parameters(
      ("plain_tensor", [descriptors.PlainAOTInput(2)]),
      (
          "tensor_subclass",
          [
              descriptors.SubclassGetAttrAOTInput(
                  descriptors.PlainAOTInput(2), attr
              )
              for attr in ("a", "b")
          ],
      ),
  )
  def test_input_that_aot_autograd_updates_is_cloned(self, buffer_descs):
    """Verifies that inputs AOTAutograd copies new values into are cloned.

    The graph and its descriptors follow an AOTAutograd forward graph. It
    returns the buffer's new value, which AOTAutograd copies into the buffer
    after the graph runs, and the buffer, the weight and `x` for backward. A
    tensor subclass buffer has a placeholder for each inner tensor.

    Args:
      buffer_descs: The descriptors of the buffer's placeholders.
    """
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["desc"] = descriptors.PlainAOTInput(0)
    weight = graph.placeholder("weight")
    weight.meta["desc"] = descriptors.PlainAOTInput(1)
    buffers = [graph.placeholder("buffer") for _ in buffer_descs]
    for buffer, desc in zip(buffers, buffer_descs):
      buffer.meta["desc"] = desc
    y = graph.call_function(
        aten.mm.default, (x, graph.call_function(aten.t.default, (weight,)))
    )
    updates = [graph.call_function(aten.add.Tensor, (b, y)) for b in buffers]
    output = graph.output((*updates, y, x, weight, *buffers))
    mutation = descriptors.InputMutationAOTOutput(descriptors.PlainAOTInput(2))
    output.meta["desc"] = [
        *(
            descriptors.SubclassGetAttrAOTOutput(mutation, desc.attr)
            if isinstance(desc, descriptors.SubclassGetAttrAOTInput)
            else mutation
            for desc in buffer_descs
        ),
        descriptors.PlainAOTOutput(0),
        *(
            descriptors.SavedForBackwardsAOTOutput(i)
            for i in range(2 + len(buffers))
        ),
    ]

    clone_mutated_returned_placeholders.apply(graph)

    self.assertEqual(_cloned_placeholders(graph), set(buffers))


if __name__ == "__main__":
  absltest.main()
