# Copyright 2025 Google LLC
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

"""Small graph test for TPU backend."""

import os
from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.export
from torch_tpu._internal.export import export as torch_tpu_export
from tests import seed_test_utils


class SimpleModule(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.a = torch.nn.Parameter(torch.Tensor(5))
    self.b = torch.nn.Parameter(torch.Tensor(5))
    self.c = torch.nn.Parameter(torch.Tensor(5))

  def forward(self, x, y):
    out = self.a + 0.8 * self.b + x
    out = out + 0.7 * self.c + y
    return out


class SimpleNN(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.linear1 = torch.nn.Linear(8, 8)

  def forward(self, x):
    x = self.linear1(x)
    x = x + torch.ones_like(x)
    return x


class TestExportLinearMode(seed_test_utils.RepeatableTest):

  def test_export_linear(self):
    sample_input = (torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),)
    exported = torch.export.export(SimpleNN(), args=sample_input)
    exported_mlir = torch_tpu_export.exported_to_mlir(exported)
    mlir_text = exported_mlir.serialize_text(enable_debug_info=True)

    # Check for module name and main func
    self.assertRegex(
        mlir_text,
        r"module @tt_jit_export_test_L[0-9]+C[0-9]+_forward_add",
    )
    self.assertIn(
        "func.func @main(%arg0: tensor<8x8xf32> loc(unknown), %arg1:"
        " tensor<8xf32> loc(unknown), %arg2: tensor<8xf32> loc(unknown)) ->"
        " tensor<8xf32>",
        mlir_text,
    )

    # Check for StableHLO ops
    self.assertIn("stablehlo.add", mlir_text)

    # Check for namescope / opname
    self.assertIn('loc("linear1[Linear]/linear/linear"', mlir_text)
    # Ops directly in the root module have no module scope.
    self.assertIn('loc("add/add"', mlir_text)

    # Check for file info
    current_filename = os.path.basename(__file__)
    self.assertIn(current_filename, mlir_text)

  def test_export_nested_module_name_scopes(self):
    class SimpleAttention(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)

      def forward(self, x):
        return self.q_proj(x)

    class SimpleTransformerBlock(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.attn = SimpleAttention()

      def forward(self, x):
        return self.attn(x)

    class SimpleModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([SimpleTransformerBlock()])

      def forward(self, x):
        return self.layers[0](x)

    sample_input = (torch.ones(8),)
    exported = torch.export.export(SimpleModel(), args=sample_input)
    exported_mlir = torch_tpu_export.exported_to_mlir(exported)
    mlir_text = exported_mlir.serialize_text(enable_debug_info=True)
    self.assertIn(
        'loc("layers.0[SimpleTransformerBlock]/'
        "attn[SimpleAttention]/"
        'q_proj[Linear]/linear/linear"',
        mlir_text,
    )


class _Block(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.proj = torch.nn.Linear(8, 8)

  def forward(self, x):
    return self.proj(x)


class _Stack(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.layers = torch.nn.ModuleList([_Block(), _Block()])

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


def _capture_dynamo_graphs(fn, *args):
  """Compiles `fn` with Dynamo and returns the captured graph modules."""
  graphs = []

  def capture_backend(gm, example_inputs):
    del example_inputs  # Unused.
    graphs.append(gm)
    return gm.forward

  torch._dynamo.reset()
  torch.compile(fn, backend=capture_backend, fullgraph=True)(*args)
  return graphs


def _call_model(model, x):
  return model(x)


class ModuleNameScopeTest(seed_test_utils.RepeatableTest):

  @parameterized.named_parameters(
      ("export_root", "", []),
      ("export_submodule", "layers.0", ["layers.0[_Block]"]),
      ("export_child_named_self", "self", ["self[_Block]"]),
      ("export_slash_in_name", "submodules/a/b", ["submodules_a_b[_Block]"]),
      ("dynamo_root", "L['self']", []),
      ("dynamo_double_quote_root", 'L["self"]', []),
      ("dynamo_global_self_root", "G['self']", []),
      ("dynamo_submodule", "L['self'].layers.0", ["layers.0[_Block]"]),
      ("dynamo_child_named_self", "L['self'].self", ["self[_Block]"]),
      (
          "dynamo_subscript_submodule",
          "L['self'].layers[0]",
          ["layers.0[_Block]"],
      ),
      ("dynamo_root_subscript", "L['self'][0]", ["0[_Block]"]),
      (
          "dynamo_dict_subscript",
          "L['self'].submodules['block']",
          ["submodules.block[_Block]"],
      ),
      (
          "dynamo_double_quote_submodule",
          'L["self"].layers.0',
          ["layers.0[_Block]"],
      ),
      ("dynamo_local_root", "L['model']", ["model[_Block]"]),
      (
          "dynamo_local_submodule",
          "L['model'].layers.0",
          ["model.layers.0[_Block]"],
      ),
      (
          "dynamo_global_submodule",
          "G['model'].layers.0",
          ["model.layers.0[_Block]"],
      ),
      (
          "dynamo_double_quote_global_submodule",
          'G["model"].layers.0',
          ["model.layers.0[_Block]"],
      ),
  )
  def test_extract_node_name_scopes_from_module_fqn(self, fqn, expected):
    node = torch.fx.Graph().placeholder("x")
    node.meta["nn_module_stack"] = {"module": (fqn, _Block)}
    self.assertEqual(torch_tpu_export._extract_node_name_scopes(node), expected)

  def test_extract_node_name_scopes_deduplicates_repeated_fqn_but_keeps_nested_containers(
      self,
  ):
    # A module FQN repeated on the stack (e.g. `@1` keys) is collapsed.
    node = torch.fx.Graph().placeholder("x")
    node.meta["nn_module_stack"] = {
        "L__self__": ("L['self']", _Stack),
        "L__self___layers_0": ("L['self'].layers[0]", _Block),
        "L__self___layers_0@1": ("L['self'].layers[0]", _Block),
        "L__self___layers_0_proj": (
            "L['self'].layers[0].proj",
            torch.nn.Linear,
        ),
    }
    self.assertEqual(
        torch_tpu_export._extract_node_name_scopes(node),
        ["layers.0[_Block]", "proj[Linear]"],
    )

    # Distinct nested modules with identical relative names (e.g. nested
    # Sequential containers at `0`, `0.0`, `0.0.0`) are preserved.
    nested_seq_node = torch.fx.Graph().placeholder("x")
    nested_seq_node.meta["nn_module_stack"] = {
        "root": ("", torch.nn.Sequential),
        "seq_0": ("0", torch.nn.Sequential),
        "seq_0_0": ("0.0", torch.nn.Sequential),
        "seq_0_0_0": ("0.0.0", torch.nn.Linear),
    }
    self.assertEqual(
        torch_tpu_export._extract_node_name_scopes(nested_seq_node),
        ["0[Sequential]", "0[Sequential]", "0[Linear]"],
    )

  def test_extract_node_name_scopes_from_nested_non_self_module_stack(self):
    node = torch.fx.Graph().placeholder("x")
    node.meta["nn_module_stack"] = {
        "L__model__": ("L['model']", _Stack),
        "L__model___layers_0": ("L['model'].layers.0", _Block),
        "L__model___layers_0_proj": (
            "L['model'].layers.0.proj",
            "torch.nn.modules.linear.Linear",
        ),
    }
    self.assertEqual(
        torch_tpu_export._extract_node_name_scopes(node),
        ["model[_Stack]", "layers.0[_Block]", "proj[Linear]"],
    )

  def test_extract_node_name_scopes_from_fwd_nn_module_stack(self):
    node = torch.fx.Graph().placeholder("x")
    node.meta["fwd_nn_module_stack"] = {
        "L__self__": ("L['self']", _Stack),
        "L__self___layers_1": ("L['self'].layers.1", _Block),
        "L__self___layers_1_proj": (
            "L['self'].layers.1.proj",
            "torch.nn.modules.linear.Linear",
        ),
    }
    self.assertEqual(
        torch_tpu_export._extract_node_name_scopes(node),
        ["layers.1[_Block]", "proj[Linear]"],
    )

  def test_extract_node_name_scopes_without_module_stack(self):
    node = torch.fx.Graph().placeholder("x")
    self.assertEqual(torch_tpu_export._extract_node_name_scopes(node), [])

  def test_extract_node_name_scopes_with_malformed_metadata(self):
    node = torch.fx.Graph().placeholder("x")
    node.meta["nn_module_stack"] = ["not_a_mapping"]
    self.assertEqual(torch_tpu_export._extract_node_name_scopes(node), [])

    node.meta["nn_module_stack"] = {
        "not_a_tuple": "layers.0",
        "too_short": ("layers.0",),
        "valid": ("layers.0", _Block),
    }
    self.assertEqual(
        torch_tpu_export._extract_node_name_scopes(node),
        ["layers.0[_Block]"],
    )

  def test_dynamo_name_scopes_of_compiled_module(self):
    (gm,) = _capture_dynamo_graphs(_Stack(), torch.ones(2, 8))
    scopes = [
        torch_tpu_export._extract_node_name_scopes(node)
        for node in gm.graph.nodes
        if node.op == "call_function"
    ]
    self.assertIn(["layers.0[_Block]", "proj[Linear]"], scopes)
    self.assertIn(["layers.1[_Block]", "proj[Linear]"], scopes)

  def test_dynamo_name_scopes_of_function_with_module_argument(self):
    # Dynamo roots the module FQNs at the `model` argument, i.e. `L['model']`.
    (gm,) = _capture_dynamo_graphs(_call_model, _Stack(), torch.ones(2, 8))
    scopes = [
        torch_tpu_export._extract_node_name_scopes(node)
        for node in gm.graph.nodes
        if node.op == "call_function"
    ]
    self.assertIn(["model[_Stack]", "layers.0[_Block]", "proj[Linear]"], scopes)
    self.assertIn(["model[_Stack]", "layers.1[_Block]", "proj[Linear]"], scopes)

  def test_compiled_module_name_scopes_in_forward_and_backward(self):
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    torch._dynamo.reset()
    debugs = []
    compiled = torch.compile(
        _Stack().to("tpu"),
        backend="tpu",
        options={"debug_callback": debugs.append},
    )

    x = torch.ones(2, 8, device="tpu", requires_grad=True)
    compiled(x).sum().backward()

    self.assertLen(debugs, 1)
    forward_mlir = "\n".join(debugs[0].stablehlo_forward_text)
    backward_mlir = "\n".join(debugs[0].stablehlo_backward_text)
    for mlir_text in (forward_mlir, backward_mlir):
      self.assertIn('loc("layers.0[_Block]/proj[Linear]/', mlir_text)
      self.assertIn('loc("layers.1[_Block]/proj[Linear]/', mlir_text)


class ExportTest(seed_test_utils.RepeatableTest):

  def test_module(self):
    sample_input = (
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.6]),
    )
    exported = torch.export.export(SimpleModule(), args=sample_input)
    with torch_tpu_export.enable_tracebacks(None):  # override with nullopt
      exported_mlir = torch_tpu_export.exported_to_mlir(exported)
    mlir_text = exported_mlir.serialize_text(enable_debug_info=True)

    # Check for module name and main func
    self.assertRegex(
        mlir_text,
        r"module @tt_jit_export_test_L[0-9]+C[0-9]+_forward_add",
    )
    self.assertIn(
        "func.func @main(%arg0: tensor<5xf32> loc(unknown), %arg1:"
        " tensor<5xf32> loc(unknown), %arg2: tensor<5xf32> loc(unknown), %arg3:"
        " tensor<5xf32> loc(unknown), %arg4: tensor<5xf32> loc(unknown)) ->"
        " tensor<5xf32>",
        mlir_text,
    )

    # Default export should include debuginfo.
    # Check that the current filename is in the MLIR text with debuginfo.
    # Currently checking for #loc1 so we don't match this filename higher up in
    # the stack trace.
    current_filename = os.path.basename(__file__)
    current_location_regex = f"#loc1 = loc.*{current_filename}"
    self.assertRegex(mlir_text, current_location_regex)

  def test_export_deduplication_different_views(self):
    """Tests that export deduplication handles different views of the same memory correctly."""

    class MyModule(torch.nn.Module):

      def forward(self, x):
        return x, x.view(torch.int32)

    mod = MyModule()
    x = torch.randn(5)

    exported = torch.export.export(mod, args=(x,))
    mlir_text = torch_tpu_export.exported_to_mlir(exported).serialize_text()
    self.assertIn("-> (tensor<5xf32>, tensor<5xi32>)", mlir_text)

  def test_export_deduplication_different_slices(self):
    """Tests that export deduplication does not deduplicate different slices of the same buffer."""

    class MyModule(torch.nn.Module):

      def forward(self, x):
        return x[0:2], x[1:3]

    mod = MyModule()
    x = torch.randn(4)

    exported = torch.export.export(mod, args=(x,))
    mlir_text = torch_tpu_export.exported_to_mlir(exported).serialize_text()

    self.assertIn("-> (tensor<2xf32>, tensor<2xf32>)", mlir_text)

  def test_module_traceback_disabled(self):
    sample_input = (
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.6]),
    )
    exported = torch.export.export(SimpleModule(), args=sample_input)
    with torch_tpu_export.enable_tracebacks(False):
      exported_mlir = torch_tpu_export.exported_to_mlir(exported)
    mlir_text = exported_mlir.serialize_text(enable_debug_info=True)

    # Check for module name and main func
    self.assertIn("module @tt_jit_add", mlir_text)
    self.assertIn(
        "func.func @main(%arg0: tensor<5xf32> loc(unknown), %arg1:"
        " tensor<5xf32> loc(unknown), %arg2: tensor<5xf32> loc(unknown), %arg3:"
        " tensor<5xf32> loc(unknown), %arg4: tensor<5xf32> loc(unknown)) ->"
        " tensor<5xf32>",
        mlir_text,
    )

    # If tracebacks are explicitly disabled, we should not see any location
    # information in the MLIR text.
    # Expect to see op names and redispatches even if tracebacks are disabled.
    self.assertIn('loc("add/add"', mlir_text)
    self.assertIn('loc("mul/mul"', mlir_text)

    # Expect no python filenames if tracebacks are disabled.
    current_filename = os.path.basename(__file__)
    self.assertNotIn(current_filename, mlir_text)
    self.assertNotRegex(mlir_text, r"loc.*\.py")

  def test_serialize_text_no_debug_info(self):
    sample_input = (
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.6]),
    )
    exported = torch.export.export(SimpleModule(), args=sample_input)
    exported_mlir = torch_tpu_export.exported_to_mlir(exported)

    # Check for the absence of location #loc(...) in mlir text
    mlir_text = exported_mlir.serialize_text(enable_debug_info=False)

    # Check for module name and main func
    self.assertIn("module @tt_jit", mlir_text)
    self.assertIn("func @main", mlir_text)
    self.assertNotIn("loc(", mlir_text)

  def test_serialize_text_debug_info(self):
    sample_input = (
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.6]),
    )
    exported = torch.export.export(SimpleModule(), args=sample_input)
    exported_mlir = torch_tpu_export.exported_to_mlir(exported)
    mlir_text = exported_mlir.serialize_text(enable_debug_info=True)

    # Check for the presence of location #loc(...) in mlir text
    self.assertIn("loc(", mlir_text)

  def test_serialize_bytecode(self):
    sample_input = (
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.6]),
    )
    exported = torch.export.export(SimpleModule(), args=sample_input)
    exported_mlir = torch_tpu_export.exported_to_mlir(exported)
    mlir_bytecode = exported_mlir.serialize_bytecode()
    # Check for the MLïR magic string that denotes bytecode.
    self.assertIn(b"ML\xefR", mlir_bytecode)

  def test_serialize_portable_artifact(self):
    sample_input = (
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.6]),
    )
    exported = torch.export.export(SimpleModule(), args=sample_input)
    exported_mlir = torch_tpu_export.exported_to_mlir(exported)
    mlir_portable_artifact = exported_mlir.serialize_portable_artifact()
    # Check for the MLïR magic string that denotes bytecode.
    # Check for the StableHLO_v1.X.Y producer string to indicate versioned
    # StableHLO.
    self.assertIn(b"ML\xefR", mlir_portable_artifact)
    self.assertIn(b"StableHLO_v1.", mlir_portable_artifact)


if __name__ == "__main__":
  absltest.main()
