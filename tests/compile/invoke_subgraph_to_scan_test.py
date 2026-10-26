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

"""Tests for the invoke_subgraph_to_scan FX pass."""

import contextlib
import enum
import operator
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import fx
from torch._dynamo.backends import common
from torch._functorch import aot_autograd
from torch_tpu._internal.compile.fx_passes import invoke_subgraph_to_scan
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils

_SCAN = torch.ops.higher_order.scan
_INVOKE_SUBGRAPH = torch.ops.higher_order.invoke_subgraph
_STACK = torch.ops.aten.stack.default


def _count(gm: fx.GraphModule, target: Any) -> int:
  return sum(
      n.op == "call_function" and n.target is target for n in gm.graph.nodes
  )


def _scans(gm: fx.GraphModule) -> list[fx.Node]:
  return [
      n for n in gm.graph.nodes if n.op == "call_function" and n.target is _SCAN
  ]


def _keep_invoke_subgraph_calls() -> contextlib.AbstractContextManager[None]:
  """Keeps Dynamo from inlining `invoke_subgraph` calls into the graph.

  PyTorch 2.11 (used in OSS) has neither option and does not inline them.
  """
  # TODO: b/576283069 - Patch both unconditionally once minimum OSS PyTorch is
  # >= 2.12.0.
  config = torch._dynamo.config  # pylint: disable=protected-access
  options = {
      name: False
      for name in (
          "inline_invoke_subgraph",
          "inline_single_use_invoke_subgraph",
      )
      if hasattr(config, name)
  }
  return config.patch(options) if options else contextlib.nullcontext()


class _LastOutput(enum.Enum):
  """How a hand-built chain's last call output is used."""

  GETITEM = enum.auto()  # Read via `getitem`, like the other calls.
  TUPLE = enum.auto()  # The output tuple itself is returned.


def _hand_built_chain(
    ks: list[Any], last_output: _LastOutput = _LastOutput.GETITEM
) -> fx.GraphModule:
  """Chains calls of `x -> x * k` with the given `k` operands, by hand.

  Args:
    ks: The `k` operand of each call. For a `str`, a tensor placeholder of that
      name is created: of shape [8] if the name starts with "w", else [4].
    last_output: How the last call's output is used.

  Returns:
    The graph module, taking `x` of shape [8].
  """
  subgraph = fx.Graph()
  x, k_placeholder = subgraph.placeholder("x"), subgraph.placeholder("k")
  subgraph.output((subgraph.call_function(torch.mul, (x, k_placeholder)),))
  graph = fx.Graph()
  value = graph.placeholder("x")
  value.meta["val"] = torch.empty(8)
  call = None
  for k in ks:
    if isinstance(k, str):
      k_shape = (8,) if k.startswith("w") else (4,)
      k = graph.placeholder(k)
      k.meta["val"] = torch.empty(k_shape)
    call = graph.call_function(
        _INVOKE_SUBGRAPH, (graph.get_attr("subgraph"), "subgraph_0", value, k)
    )
    value = graph.call_function(operator.getitem, (call, 0))
    value.meta["val"] = torch.empty(8)
  if last_output is _LastOutput.TUPLE:
    graph.erase_node(value)
    value = call
  graph.output((value,))
  root = torch.nn.Module()
  root.subgraph = fx.GraphModule(torch.nn.Module(), subgraph)
  return fx.GraphModule(root, graph)


class InvokeSubgraphToScanTest(seed_test_utils.RepeatableTest):

  def _compile_with_pass(
      self,
      model: torch.nn.Module,
      *args: torch.Tensor,
      dynamic: bool = False,
      grad: bool = False,
      unroll: int = 1,
  ) -> tuple[torch.Tensor, fx.GraphModule]:
    """Compiles `model` on CPU, applying the pass to the forward graph."""
    graphs: list[fx.GraphModule] = []

    def fw_compiler(gm: fx.GraphModule, example_inputs: Any) -> Any:
      del example_inputs  # Unused.
      invoke_subgraph_to_scan.apply(gm, unroll=unroll)
      graphs.append(gm)
      return aot_autograd.make_boxed_func(gm.forward)

    torch._dynamo.reset()  # pylint: disable=protected-access
    backend = common.aot_autograd(
        fw_compiler=fw_compiler,
        bw_compiler=lambda gm, _: aot_autograd.make_boxed_func(gm.forward),
    )
    with (
        torch.enable_grad() if grad else torch.no_grad(),
        # Keep the `invoke_subgraph` calls for the pass.
        _keep_invoke_subgraph_calls(),
    ):
      compiled = torch.compile(
          model, backend=backend, fullgraph=True, dynamic=dynamic
      )
      out = compiled(*args)
    self.assertLen(graphs, 1)
    return out, graphs[0]

  def _assert_matches_eager(
      self, model: torch.nn.Module, out: Any, *args: torch.Tensor
  ) -> None:
    with torch.no_grad():
      utils.assert_close(out, model(*args), rtol=1e-5, atol=1e-5)

  def test_per_layer_weights_are_scanned_over(self):
    class Layer(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(16, 16)

      @torch.compiler.nested_compile_region
      def forward(self, x):
        return x + torch.tanh(self.lin(x))

    class Stack(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer() for _ in range(4)])

      def forward(self, x):
        for layer in self.layers:
          x = layer(x)
        return x * 2.0

    model = Stack().eval()
    x = torch.randn(2, 16)
    out, gm = self._compile_with_pass(model, x)

    self.assertLen(_scans(gm), 1)
    self.assertEqual(_count(gm, _INVOKE_SUBGRAPH), 0)
    # The per-layer weights and biases are stacked into the scanned inputs.
    self.assertEqual(_count(gm, _STACK), 2)
    self._assert_matches_eager(model, out, x)

  def test_operands_computed_between_calls(self):
    @torch.compiler.nested_compile_region
    def body(x, w):
      return x + torch.tanh(x @ w)

    class Stack(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4, 16, 16) * 0.1)

      def forward(self, x):
        for i in range(4):
          # Computed in the outer graph right before each call.
          x = body(x, self.w[i] * 2.0)
        return x

    model = Stack().eval()
    x = torch.randn(2, 16)
    out, gm = self._compile_with_pass(model, x)

    self.assertLen(_scans(gm), 1)
    self._assert_matches_eager(model, out, x)

  def test_multiple_carries_in_permuted_order(self):
    @torch.compiler.nested_compile_region
    def body(a, b, w):
      return torch.tanh(b @ w), a + 1.0

    class Swapping(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.ws = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(16, 16) * 0.1) for _ in range(3)]
        )

      def forward(self, a, b):
        for w in self.ws:
          # Output 1 feeds operand 0 and output 0 feeds operand 1.
          out0, out1 = body(a, b, w)
          a, b = out1, out0
        return a * 2.0 + b

    model = Swapping().eval()
    args = (torch.randn(2, 16), torch.randn(2, 16))
    out, gm = self._compile_with_pass(model, *args)

    (scan,) = _scans(gm)
    self.assertLen(scan.args[1], 2)  # Both carries.
    self._assert_matches_eager(model, out, *args)

  def test_chain_ends_where_the_carry_routing_changes(self):
    @torch.compiler.nested_compile_region
    def body(a, b):
      return torch.sin(a), torch.cos(b)

    class Rerouting(torch.nn.Module):

      def forward(self, a, b):
        a, b = body(a, b)
        a, b = body(a, b)
        # The third call gets the outputs as the other operands.
        a, b = body(b, a)
        return a * 2.0 + b

    model = Rerouting()
    args = (torch.randn(8), torch.randn(8))
    out, gm = self._compile_with_pass(model, *args)

    self.assertLen(_scans(gm), 1)
    self.assertEqual(_count(gm, _INVOKE_SUBGRAPH), 1)
    self._assert_matches_eager(model, out, *args)

  def test_shared_operands_become_additional_inputs(self):
    @torch.compiler.nested_compile_region
    def body(x, scale):
      return torch.sin(x) * scale

    class Repeat(torch.nn.Module):

      def forward(self, x, scale):
        for _ in range(5):
          x = body(x, scale)
        return x

    model = Repeat()
    args = (torch.randn(8), torch.randn(8))
    out, gm = self._compile_with_pass(model, *args)

    (scan,) = _scans(gm)
    _, init, xs, additional_inputs = scan.args
    self.assertLen(init, 1)
    # Nothing differs per call, so a dummy input defines the trip count.
    self.assertLen(xs, 1)
    self.assertIs(xs[0].target, torch.ops.aten.zeros.default)
    self.assertLen(additional_inputs, 1)
    self.assertEqual(_count(gm, _INVOKE_SUBGRAPH), 0)
    self._assert_matches_eager(model, out, *args)

  def test_shared_symint_operand_becomes_an_additional_input(self):
    @torch.compiler.nested_compile_region
    def body(x, n):
      return x + n

    class Repeat(torch.nn.Module):

      def forward(self, x):
        n = x.shape[0]
        for _ in range(3):
          x = body(x, n)
        return x

    model = Repeat()
    x = torch.randn(8)
    torch._dynamo.mark_dynamic(x, 0)  # pylint: disable=protected-access
    out, gm = self._compile_with_pass(model, x, dynamic=True)

    (scan,) = _scans(gm)
    additional_inputs = scan.args[3]
    self.assertLen(additional_inputs, 1)
    self.assertIsInstance(additional_inputs[0].meta["val"], torch.SymInt)
    self.assertEqual(_count(gm, _INVOKE_SUBGRAPH), 0)
    self._assert_matches_eager(model, out, x)

  def test_subgraph_drawing_random_numbers_is_rolled(self):
    @torch.compiler.nested_compile_region
    def body(x):
      return x + torch.rand_like(x)

    class Repeat(torch.nn.Module):

      def forward(self, x):
        for _ in range(3):
          x = body(x)
        return x

    _, gm = self._compile_with_pass(Repeat(), torch.randn(8))

    self.assertLen(_scans(gm), 1)
    self.assertEqual(_count(gm, _INVOKE_SUBGRAPH), 0)

  def test_graph_saving_activations_for_backward_is_left_alone(self):
    @torch.compiler.nested_compile_region
    def body(x):
      return torch.sin(x) * 2.0

    class Repeat(torch.nn.Module):

      def forward(self, x):
        for _ in range(3):
          x = body(x)
        return x

    # The forward graph's calls also return activations for the backward.
    _, gm = self._compile_with_pass(
        Repeat(), torch.randn(8, requires_grad=True), grad=True
    )

    self.assertEmpty(_scans(gm))
    self.assertEqual(_count(gm, _INVOKE_SUBGRAPH), 3)

  def test_two_chains_get_their_own_scans(self):
    @torch.compiler.nested_compile_region
    def f(x):
      return torch.sin(x)

    @torch.compiler.nested_compile_region
    def g(x):
      return torch.cos(x) * 2.0

    class TwoStacks(torch.nn.Module):

      def forward(self, x):
        for _ in range(3):
          x = f(x)
        for _ in range(2):
          x = g(x)
        return x

    model = TwoStacks()
    x = torch.randn(8)
    out, gm = self._compile_with_pass(model, x)

    bodies = [scan.args[0].target for scan in _scans(gm)]
    self.assertLen(set(bodies), 2)
    self.assertEqual(_count(gm, _INVOKE_SUBGRAPH), 0)
    self._assert_matches_eager(model, out, x)

  def test_chain_with_escaping_intermediate_is_left_alone(self):
    @torch.compiler.nested_compile_region
    def body(x):
      return torch.sin(x) * 2.0

    class Escaping(torch.nn.Module):

      def forward(self, x):
        a = body(x)
        b = body(a)
        # `a` is used outside the chain, so it cannot become a scan carry.
        return b + a

    model = Escaping()
    x = torch.randn(8)
    out, gm = self._compile_with_pass(model, x)

    self.assertEmpty(_scans(gm))
    self.assertEqual(_count(gm, _INVOKE_SUBGRAPH), 2)
    self._assert_matches_eager(model, out, x)

  def test_calls_of_different_regions_are_not_chained(self):
    @torch.compiler.nested_compile_region
    def f(x):
      return torch.sin(x)

    @torch.compiler.nested_compile_region
    def g(x):
      return torch.cos(x)

    class Alternating(torch.nn.Module):

      def forward(self, x):
        for _ in range(3):
          x = g(f(x))
        return x

    model = Alternating()
    x = torch.randn(8)
    out, gm = self._compile_with_pass(model, x)

    self.assertEmpty(_scans(gm))
    self.assertEqual(_count(gm, _INVOKE_SUBGRAPH), 6)
    self._assert_matches_eager(model, out, x)

  def test_partial_unroll_runs_k_calls_per_iteration(self):
    @torch.compiler.nested_compile_region
    def body(x, w):
      return x + torch.tanh(x @ w)

    class Stack(torch.nn.Module):

      def __init__(self):
        super().__init__()
        # Two calls per iteration; the fifth stays an `invoke_subgraph`.
        self.ws = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(16, 16) * 0.1) for _ in range(5)]
        )

      def forward(self, x):
        for w in self.ws:
          x = body(x, w)
        return x

    model = Stack().eval()
    x = torch.randn(2, 16)
    out, gm = self._compile_with_pass(model, x, unroll=2)

    (scan,) = _scans(gm)
    body_gm = getattr(gm, scan.args[0].target)
    # The body runs the subgraph twice, on two slices of each scanned input.
    self.assertEqual(_count(body_gm, torch.ops.aten.select.int), 2)
    self.assertEqual(tuple(scan.args[2][0].meta["val"].shape), (2, 2, 16, 16))
    self.assertEqual(_count(gm, _INVOKE_SUBGRAPH), 1)
    self._assert_matches_eager(model, out, x)

  def test_mutated_input_becomes_a_carry(self):
    class Layer(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(16, 16)

      @torch.compiler.nested_compile_region
      def forward(self, x, cache):
        # Updated in place, like a KV cache.
        cache.mul_(0.5).add_(x)
        return x + torch.tanh(self.lin(cache))

    class Stack(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer() for _ in range(4)])

      def forward(self, x, cache):
        for layer in self.layers:
          x = layer(x, cache)
        return x

    model = Stack().eval()
    x = torch.randn(2, 16)
    cache, expected_cache = torch.zeros(2, 16), torch.zeros(2, 16)
    out, gm = self._compile_with_pass(model, x, cache)

    (scan,) = _scans(gm)
    self.assertLen(scan.args[1], 2)  # The hidden state and the cache.
    with torch.no_grad():
      utils.assert_close(out, model(x, expected_cache))
    # The final cache is written back to the input.
    utils.assert_close(cache, expected_cache)

  # TODO: b/576283069 - Remove once minimum OSS PyTorch is >= 2.12.0.
  @absltest.skipIf(
      utils.get_torch_version() < (2, 12),
      "PyTorch < 2.12 does not reuse a region called on views of one tensor.",
  )
  def test_stacked_views_are_read_in_place(self):
    @torch.compiler.nested_compile_region
    def body(x, w):
      return x + torch.tanh(x @ w)

    class Stack(torch.nn.Module):
      """Holds the weights of all layers as one tensor, like `nn.scan` models."""

      def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4, 16, 16) * 0.1)

      def forward(self, x):
        for i in range(self.w.shape[0]):
          x = body(x, self.w[i])
        return x

    model = Stack().eval()
    x = torch.randn(2, 16)
    out, gm = self._compile_with_pass(model, x)

    self.assertLen(_scans(gm), 1)
    # The scan reads `w` itself instead of a per-step copy of its slices.
    self.assertEqual(_count(gm, _STACK), 0)
    self._assert_matches_eager(model, out, x)

  # TODO: b/576283069 - Remove once minimum OSS PyTorch is >= 2.12.0.
  @absltest.skipIf(
      utils.get_torch_version() < (2, 12),
      "PyTorch < 2.12 does not reuse a region called on views of one tensor.",
  )
  def test_partial_views_are_sliced(self):
    @torch.compiler.nested_compile_region
    def body(x, w):
      return x + torch.tanh(x @ w)

    class Stack(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(6, 16, 16) * 0.1)

      def forward(self, x):
        # Only layers 1 to 4 form the chain.
        x = x * 2.0
        for i in range(1, 5):
          x = body(x, self.w[i])
        return x

    model = Stack().eval()
    x = torch.randn(2, 16)
    out, gm = self._compile_with_pass(model, x)

    self.assertLen(_scans(gm), 1)
    self.assertEqual(_count(gm, _STACK), 0)
    self.assertEqual(_count(gm, torch.ops.aten.slice.Tensor), 1)
    self._assert_matches_eager(model, out, x)

  def test_shared_int_becomes_an_additional_input(self):
    gm = _hand_built_chain([3, 3, 3])

    invoke_subgraph_to_scan.apply(gm)

    (scan,) = _scans(gm)
    self.assertEqual(scan.args[3], [3])
    x = torch.randn(8)
    utils.assert_close(gm(x)[0], x * 27)

  @parameterized.named_parameters(("float", 2.0), ("none", None))
  def test_other_shared_constants_become_additional_inputs(self, k):
    # Whether the scan supports them is for the scan to check.
    gm = _hand_built_chain([k, k, k])

    invoke_subgraph_to_scan.apply(gm)

    (scan,) = _scans(gm)
    self.assertEqual(scan.args[3], [k])

  @parameterized.named_parameters(
      ("different_ints", [2, 3, 4]),
      ("tensors_of_different_shapes", ["w0", "w1", "b2"]),
  )
  def test_operands_that_cannot_be_scanned_are_left_alone(self, ks):
    gm = _hand_built_chain(ks)

    invoke_subgraph_to_scan.apply(gm)

    self.assertEmpty(_scans(gm))
    self.assertEqual(_count(gm, _INVOKE_SUBGRAPH), 3)

  def test_last_call_whose_tuple_is_used_directly_stays_out_of_the_scan(self):
    gm = _hand_built_chain([3, 3, 3], last_output=_LastOutput.TUPLE)

    invoke_subgraph_to_scan.apply(gm)

    self.assertLen(_scans(gm), 1)
    self.assertEqual(_count(gm, _INVOKE_SUBGRAPH), 1)
    x = torch.randn(8)
    utils.assert_close(gm(x)[0][0], x * 27)


if __name__ == "__main__":
  absltest.main()
