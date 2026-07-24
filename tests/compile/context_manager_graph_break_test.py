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

"""Tests for TorchTPU context managers under `torch.compile`.

TorchTPU context managers (e.g. `precision`) are backed by opaque pybind11 calls
that TorchDynamo cannot symbolically trace. To avoid a graph break, `precision`
delegates to `torch.fx.traceback.annotate` while compiling, stamping the region
precision onto `node.meta["custom"]`; the TorchTPU lowering interpreter reads it
back so compiled precision matches eager. These tests pin that behavior: no
graph breaks, and the annotation surviving AOTAutograd into the forward and
backward graphs the backend lowers.

The test target sets `TORCH_LOGS=graph_breaks`, so every invocation's test log
carries per-graph-break diagnostics from PyTorch. The diagnostics info is meant
for human inspection only.
"""

from absl.testing import absltest
import torch
from torch_tpu._internal import precision
from torch_tpu._internal.utils import utils


def _torch_tpu_ctx_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Matmul within a TorchTPU context manager."""
  with precision.precision(precision.Precision.HIGHEST):
    return a @ b


def _torch_native_ctx_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Matmul within a PyTorch-native context manager."""
  with torch.no_grad():
    return a @ b


class ContextManagerGraphBreakTest(absltest.TestCase):
  """Tests for TorchTPU context manager graph breaks under `torch.compile`."""

  def setUp(self):
    super().setUp()
    # Drop cached compilations so each test observes a fresh trace.
    torch._dynamo.reset()

  def test_torch_tpu_ctx_compiles_without_graph_breaks(self):
    """Test that a TorchTPU context manager now compiles without graph breaks."""
    explanation = torch._dynamo.explain(_torch_tpu_ctx_matmul)(
        torch.randn(2, 2), torch.randn(2, 2)
    )
    # The context manager delegates to `torch.fx.traceback.annotate` under
    # `torch.compile`, so the opaque pybind11 push/pop is never traced and the
    # whole region is captured in a single graph.
    self.assertEqual(explanation.graph_count, 1)
    self.assertEqual(explanation.graph_break_count, 0)

  def test_torch_tpu_ctx_trace_fullgraph_succeeds(self):
    """Test that a TorchTPU context manager compiles under `fullgraph=True`."""
    compiled = torch.compile(
        _torch_tpu_ctx_matmul, backend="eager", fullgraph=True
    )
    x, y = torch.ones(2, 2), torch.ones(2, 2)
    utils.assert_close(compiled(x, y), x @ y)

  def test_torch_native_ctx_compiles_without_graph_breaks(self):
    """Test that a PyTorch-native context manager compiles cleanly."""
    compiled = torch.compile(
        _torch_native_ctx_matmul, backend="eager", fullgraph=True
    )
    x, y = torch.ones(2, 2), torch.ones(2, 2)
    utils.assert_close(compiled(x, y), x @ y)

    explanation = torch._dynamo.explain(_torch_native_ctx_matmul)(
        torch.randn(2, 2), torch.randn(2, 2)
    )
    self.assertEqual(explanation.graph_count, 1)
    self.assertEqual(explanation.graph_break_count, 0)


def _matmul_precision_annotations(gm: torch.fx.GraphModule) -> list[int]:
  """Precision values stamped on the matmul nodes of an FX graph."""
  values = []
  for node in gm.graph.nodes:
    target = str(node.target)
    if "mm" not in target and "matmul" not in target:
      continue
    custom = node.meta.get("custom")
    if custom and precision.TPU_PRECISION_META_KEY in custom:
      values.append(custom[precision.TPU_PRECISION_META_KEY])
  return values


def _capture_aot_graphs(
    fn, *args, run_backward: bool = False
) -> tuple[list[torch.fx.GraphModule], list[torch.fx.GraphModule]]:
  """Compiles `fn` and captures the post-AOTAutograd forward/backward graphs.

  This is the level at which the TorchTPU backend lowers, so it is where the
  precision annotation must survive for compiled precision to match eager. The
  backward graph is compiled lazily, so `run_backward` invokes `.backward()` on
  the (scalar) result to force it.

  Args:
    fn: The function to compile and capture graphs for.
    *args: Arguments to pass to `fn`.
    run_backward: Whether to invoke `.backward()` to capture the backward graph.

  Returns:
    A tuple containing (forward_graphs, backward_graphs).
  """
  from torch._functorch.aot_autograd import aot_module_simplified  # pylint: disable=g-import-not-at-top
  from torch._functorch.aot_autograd import make_boxed_func  # pylint: disable=g-import-not-at-top

  forward_graphs = []
  backward_graphs = []

  def backend(gm, example_inputs):
    def fw_compiler(g, _):
      forward_graphs.append(g)
      return make_boxed_func(g.forward)

    def bw_compiler(g, _):
      backward_graphs.append(g)
      return make_boxed_func(g.forward)

    return aot_module_simplified(
        gm, example_inputs, fw_compiler=fw_compiler, bw_compiler=bw_compiler
    )

  result = torch.compile(fn, backend=backend, fullgraph=True)(*args)
  if run_backward:
    result.backward()
  return forward_graphs, backward_graphs


class PrecisionAnnotationTest(absltest.TestCase):
  """Tests that `with precision(...)` regions annotate the compiled graph."""

  def setUp(self):
    super().setUp()
    torch._dynamo.reset()

  def test_annotation_reaches_aot_forward_graph(self):
    """The region precision lands on the matmul node of the forward graph."""
    forward_graphs, _ = _capture_aot_graphs(
        _torch_tpu_ctx_matmul, torch.randn(2, 2), torch.randn(2, 2)
    )
    self.assertLen(forward_graphs, 1)
    self.assertEqual(
        _matmul_precision_annotations(forward_graphs[0]),
        [int(precision.Precision.HIGHEST)],
    )

  def test_nested_regions_annotate_each_matmul(self):
    """Nested regions stamp each matmul with its innermost precision."""

    def fn(a, b):
      with precision.precision(precision.Precision.HIGH):
        x = a @ b
        with precision.precision(precision.Precision.HIGHEST):
          y = a @ b
      return x + y

    forward_graphs, _ = _capture_aot_graphs(
        fn, torch.randn(2, 2), torch.randn(2, 2)
    )
    self.assertLen(forward_graphs, 1)
    self.assertCountEqual(
        _matmul_precision_annotations(forward_graphs[0]),
        [int(precision.Precision.HIGH), int(precision.Precision.HIGHEST)],
    )

  def test_annotation_survives_into_backward_graph(self):
    """The precision annotation propagates through AOTAutograd to backward."""

    def fn(a, b):
      with precision.precision(precision.Precision.HIGHEST):
        return (a @ b).sum()

    a = torch.randn(2, 2, requires_grad=True)
    b = torch.randn(2, 2, requires_grad=True)
    _, backward_graphs = _capture_aot_graphs(fn, a, b, run_backward=True)
    self.assertLen(backward_graphs, 1)
    # `custom` is a copied meta field, so the matmul(s) in the backward pass are
    # stamped with the region precision too.
    self.assertIn(
        int(precision.Precision.HIGHEST),
        _matmul_precision_annotations(backward_graphs[0]),
    )

  def test_compiled_matches_eager(self):
    """The annotate bridge does not alter numerics versus eager."""
    a, b = torch.randn(4, 4), torch.randn(4, 4)
    compiled = torch.compile(
        _torch_tpu_ctx_matmul, backend="eager", fullgraph=True
    )
    utils.assert_close(compiled(a, b), _torch_tpu_ctx_matmul(a, b))


if __name__ == "__main__":
  absltest.main()
