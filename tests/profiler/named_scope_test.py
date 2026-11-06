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

"""Tests for `named_scope`."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.export
import torch.fx
from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.export import export as torch_tpu_export
from torch_tpu._internal.profiler import named_scope
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils


class _Experts(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.fc = torch.nn.Linear(8, 8)

  def forward(self, x):
    return self.fc(x)


class _MoE(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.gate = torch.nn.Linear(8, 8)
    self.experts = _Experts()
    self.shared_experts = _Experts()

  def forward(self, x):
    with named_scope.named_scope("route"):
      weights = torch.softmax(self.gate(x), dim=-1)
    with named_scope.named_scope("expert_compute"):
      routed = self.experts(x * weights)
      shared = self.shared_experts(x)
    return routed + shared


@named_scope.named_scope("rope")
def _rope(x):
  return torch.sin(x)


class _Block(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.moe = _MoE()

  def forward(self, x):
    return self.moe(_rope(x))


class _Model(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.layers = torch.nn.ModuleList([_Block(), _Block()])

  def forward(self, x):
    with named_scope.named_scope("decoder"):
      for layer in self.layers:
        x = layer(x)
    return torch.relu(x)


def _layer_name_scopes(layer: int) -> tuple[str, ...]:
  """Returns the "/"-joined name scopes of the ops of a `_Model` layer."""
  block = f"decoder/layers.{layer}[_Block]"
  moe = f"{block}/moe[_MoE]"
  return (
      f"{block}/rope",
      f"{moe}/route/gate[Linear]",
      f"{moe}/route",
      f"{moe}/expert_compute",
      f"{moe}/expert_compute/experts[_Experts]/fc[Linear]",
      f"{moe}/expert_compute/shared_experts[_Experts]/fc[Linear]",
      moe,
  )


# Name scopes of the ops of `_Model` that are in a scope, joined by "/".
_MODEL_NAME_SCOPES = _layer_name_scopes(0) + _layer_name_scopes(1)


def _capturing_backend(graphs: list[torch.fx.GraphModule]):
  """Returns a Dynamo backend that runs and appends its graphs to `graphs`."""

  def backend(gm, example_inputs):
    del example_inputs  # Unused.
    graphs.append(gm)
    return gm.forward

  return backend


# Dynamo resumes `torch.fx.traceback.annotate` after a graph break only on
# PyTorch builds that have `_dynamo_annotate` (nightly 2.16.0a0 and later).
_requires_annotate_resume = absltest.skipIf(
    not hasattr(torch.fx.traceback, "_dynamo_annotate"),
    "This PyTorch build cannot resume torch.fx.traceback.annotate after a"
    " graph break.",
)


def _graph_name_scopes(gm: torch.fx.GraphModule) -> set[str]:
  """Returns the "/"-joined name scopes of the ops of a graph that have any."""
  node_scopes = named_scope.compute_graph_name_scopes(
      gm.graph, torch_tpu_export._module_scope_entries
  )
  return {"/".join(scopes) for scopes in node_scopes.values() if scopes}


class NamedScopeTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    torch._dynamo.reset()

  def _skip_without_tpu(self):
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")

  def test_eager_scopes_run_their_bodies(self):
    x = torch.ones(2, 8)
    with named_scope.named_scope("outer"):
      out = _rope(x)
    test_utils.assert_close(out, torch.sin(x))

  def test_eager_name_scopes_in_forward_and_backward(self):
    self._skip_without_tpu()
    x = torch.ones(2, 8, device="tpu", requires_grad=True)
    torch.accelerator.synchronize()
    with execution_mode.set_eager_mode(
        execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
    ):
      with named_scope.named_scope("outer"):
        with named_scope.named_scope("outer"):
          y = _rope(x)
      with named_scope.named_scope("bwd"):
        y.sum().backward()
      self.assertIsNotNone(x.grad)
      mlir_module = tpu_torch_compile.build_mlir([y, x.grad], [x], False)
      mlir_text = tpu_torch_compile.serialize_mlir_text(
          mlir_module, enable_debug_info=True
      )

    self.assertIn('loc("outer/rope/sin/sin"', mlir_text)
    self.assertIn('loc("bwd/cos/cos"', mlir_text)
    self.assertNotIn("outer/outer", mlir_text)

  def test_ambient_eager_scope_does_not_leak_into_compiled_graph(self):
    self._skip_without_tpu()
    debugs = []
    lifted_const = torch.ones(2, 8, device="tpu")

    @torch.compile(backend="tpu", options={"debug_callback": debugs.append})
    def compiled_fn(t):
      with named_scope.named_scope("compiled_inner"):
        inline_const = torch.tensor([1.0] * 8, device=t.device)
        return t + lifted_const + inline_const

    x = torch.ones(2, 8, device="tpu")
    with named_scope.named_scope("ambient_eager"):
      out = compiled_fn(x)
      torch.accelerator.synchronize()
      with execution_mode.set_eager_mode(
          execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
      ):
        eager_after = torch.neg(out)
        eager_mlir = tpu_torch_compile.serialize_mlir_text(
            tpu_torch_compile.build_mlir([eager_after], [out], False),
            enable_debug_info=True,
        )

    self.assertLen(debugs, 1)
    forward_mlir = "\n".join(debugs[0].stablehlo_forward_text)
    self.assertIn('loc("compiled_inner/add/add"', forward_mlir)
    self.assertIn(
        'loc("make_constant_tensor/torchtpu_internal_constant"',
        forward_mlir,
    )
    self.assertNotIn("ambient_eager/", forward_mlir)
    self.assertIn('loc("ambient_eager/neg/neg"', eager_mlir)

  def test_unscoped_op_in_plain_function_separates_scope_runs(self):
    class SubModule(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 8)
        self.fc2 = torch.nn.Linear(8, 8)

      def forward(self, x):
        x = self.fc1(x)
        with named_scope.named_scope("tail"):
          return self.fc2(x)

    sub = SubModule()

    def fn(x):
      x = sub(x)
      x = x + 1
      with named_scope.named_scope("tail"):
        return x * 2

    graphs = []
    torch.compile(fn, backend=_capturing_backend(graphs), fullgraph=True)(
        torch.ones(2, 8)
    )
    self.assertLen(graphs, 1)
    self.assertEqual(
        _graph_name_scopes(graphs[0]),
        {
            "sub[SubModule]/fc1[Linear]",
            "sub[SubModule]/tail/fc2[Linear]",
            "tail",
        },
    )

  def test_dynamo_graph_name_scopes(self):
    graphs = []
    # `fullgraph=True` checks that `named_scope` causes no graph breaks.
    torch.compile(_Model(), backend=_capturing_backend(graphs), fullgraph=True)(
        torch.ones(2, 8)
    )

    self.assertLen(graphs, 1)
    self.assertEqual(_graph_name_scopes(graphs[0]), set(_MODEL_NAME_SCOPES))
    # `relu` runs outside of all scopes.
    node_scopes = named_scope.compute_graph_name_scopes(
        graphs[0].graph, torch_tpu_export._module_scope_entries
    )
    (relu,) = graphs[0].graph.find_nodes(op="call_function", target=torch.relu)
    self.assertEqual(node_scopes.get(relu, ()), ())

  @_requires_annotate_resume
  def test_graph_break_inside_scope(self):
    def fn(x):
      with named_scope.named_scope("phase"):
        x = x * 2
        torch._dynamo.graph_break()
        return x - 3

    graphs = []
    x = torch.ones(2, 8)
    out = torch.compile(fn, backend=_capturing_backend(graphs))(x)

    test_utils.assert_close(out, x * 2 - 3)
    # The graph break splits the graph, and both graphs are in the scope.
    self.assertSequenceEqual(
        [_graph_name_scopes(gm) for gm in graphs], [{"phase"}, {"phase"}]
    )

  @_requires_annotate_resume
  def test_graph_break_inside_decorated_function(self):
    @named_scope.named_scope("phase")
    def fn(x):
      x = x * 2
      torch._dynamo.graph_break()
      return x - 3

    graphs = []
    x = torch.ones(2, 8)
    out = torch.compile(fn, backend=_capturing_backend(graphs))(x)

    test_utils.assert_close(out, x * 2 - 3)
    self.assertSequenceEqual(
        [_graph_name_scopes(gm) for gm in graphs], [{"phase"}, {"phase"}]
    )

  @_requires_annotate_resume
  def test_graph_break_in_callee_inside_scope(self):
    class Inner(torch.nn.Module):

      def forward(self, x):
        x = x * 2
        torch._dynamo.graph_break()
        return x - 3

    class Outer(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.inner = Inner()

      def forward(self, x):
        with named_scope.named_scope("phase"):
          return self.inner(x + 1)

    graphs = []
    x = torch.ones(2, 8)
    out = torch.compile(Outer(), backend=_capturing_backend(graphs))(x)

    test_utils.assert_close(out, (x + 1) * 2 - 3)
    # The outer frame up to the call, the callee up to the break, and the
    # callee's continuation are compiled separately; all are in the scope.
    self.assertSequenceEqual(
        [_graph_name_scopes(gm) for gm in graphs],
        [{"phase"}, {"phase"}, {"phase"}],
    )

  def test_compiled_name_scopes_in_forward_and_backward(self):
    self._skip_without_tpu()
    debugs = []
    compiled = torch.compile(
        _Model().to("tpu"),
        backend="tpu",
        options={"debug_callback": debugs.append},
    )

    x = torch.ones(2, 8, device="tpu", requires_grad=True)
    compiled(x).sum().backward()

    self.assertLen(debugs, 1)
    forward_mlir = "\n".join(debugs[0].stablehlo_forward_text)
    backward_mlir = "\n".join(debugs[0].stablehlo_backward_text)
    for mlir_text in (forward_mlir, backward_mlir):
      for scopes in _MODEL_NAME_SCOPES:
        self.assertIn(f'loc("{scopes}/', mlir_text)

  def test_exported_name_scopes(self):
    self._skip_without_tpu()
    exported = torch.export.export(_Model(), args=(torch.ones(2, 8),))
    mlir_text = torch_tpu_export.exported_to_mlir(exported).serialize_text(
        enable_debug_info=True
    )
    for scopes in _MODEL_NAME_SCOPES:
      self.assertIn(f'loc("{scopes}/', mlir_text)

  def test_non_strict_export_records_scopes_without_preserve_node_meta(self):
    exported = torch.export.export(
        _Model(), args=(torch.ones(2, 8),), strict=False
    )
    self.assertEqual(
        _graph_name_scopes(exported.graph_module), set(_MODEL_NAME_SCOPES)
    )

  def test_strict_export_records_scopes(self):
    exported = torch.export.export(
        _Model(), args=(torch.ones(2, 8),), strict=True
    )
    self.assertEqual(
        _graph_name_scopes(exported.graph_module), set(_MODEL_NAME_SCOPES)
    )

  def test_non_strict_export_inner_scope_does_not_leak(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        with named_scope.named_scope("outer"):
          a = x + 1
          with named_scope.named_scope("inner"):
            b = a * 2
          c = b - 3
        return c / 4

    exported = torch.export.export(Model(), args=(torch.ones(2, 8),))
    node_scopes = named_scope.compute_graph_name_scopes(
        exported.graph_module.graph, torch_tpu_export._module_scope_entries
    )
    by_target = {
        node.target: node_scopes.get(node, ())
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function"
    }
    self.assertEqual(by_target[torch.ops.aten.add.Tensor], ("outer",))
    self.assertEqual(by_target[torch.ops.aten.mul.Tensor], ("outer", "inner"))
    self.assertEqual(by_target[torch.ops.aten.sub.Tensor], ("outer",))
    self.assertEqual(by_target[torch.ops.aten.div.Tensor], ())

  def test_decorated_submodule_forward(self):
    class Expert(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 8)

      @named_scope.named_scope("expert")
      def forward(self, x):
        return self.fc(x) * 2

    class MoE(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.expert = Expert()

      def forward(self, x):
        return self.expert(x + 1)

    graphs = []
    torch.compile(MoE(), backend=_capturing_backend(graphs), fullgraph=True)(
        torch.ones(2, 8)
    )
    self.assertLen(graphs, 1)
    # The scope covers the whole `expert` call, so it is placed outside it.
    self.assertEqual(
        _graph_name_scopes(graphs[0]),
        {"expert/expert[Expert]/fc[Linear]", "expert/expert[Expert]"},
    )

  def test_slash_name_in_compiled_graph_creates_hierarchy(self):
    def fn(x):
      with named_scope.named_scope("outer/inner"):
        return x + 1

    graphs = []
    torch.compile(fn, backend=_capturing_backend(graphs), fullgraph=True)(
        torch.ones(2, 8)
    )
    self.assertLen(graphs, 1)
    self.assertEqual(_graph_name_scopes(graphs[0]), {"outer/inner"})

  @parameterized.named_parameters(
      ("nested", "outer", "inner"),
      ("duplicate", "outer/inner", "outer/inner"),
  )
  def test_nested_and_duplicate_scopes_collapse_in_compiled_graph(
      self, outer_name, inner_name
  ):
    def fn(x):
      with named_scope.named_scope(outer_name):
        with named_scope.named_scope(inner_name):
          return x + 1

    graphs = []
    torch.compile(fn, backend=_capturing_backend(graphs), fullgraph=True)(
        torch.ones(2, 8)
    )
    self.assertLen(graphs, 1)
    self.assertEqual(_graph_name_scopes(graphs[0]), {"outer/inner"})

  def test_reentrant_export_scope_collapses_duplicates(self):
    class ReentrantExportMod(torch.nn.Module):

      def forward(self, x):
        scope = named_scope.named_scope("outer/inner")
        with scope:
          with scope:
            return x + 1

    ep = torch.export.export(
        ReentrantExportMod(), (torch.ones(2, 8),), strict=False
    )
    self.assertEqual(_graph_name_scopes(ep.graph_module), {"outer/inner"})


if __name__ == "__main__":
  absltest.main()
