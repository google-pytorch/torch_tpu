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

"""Tests for TorchTPU's `nested_compile_region`."""

from collections.abc import Callable
import contextlib
from typing import Any

from absl.testing import absltest
import torch
from torch import fx
from torch._dynamo.backends import common
from torch._functorch import aot_autograd
from torch_tpu._internal.compile import nested_compile_region as tt_ncr
from torch_tpu._internal.compile.fx_passes import invoke_subgraph_to_scan
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils

_INVOKE_SUBGRAPH = torch.ops.higher_order.invoke_subgraph


def _scan_unroll_annotations(gm: fx.GraphModule) -> list[Any]:
  """Returns the scan unroll annotation of each `invoke_subgraph` call."""
  return [
      n.meta.get("custom", {}).get(
          invoke_subgraph_to_scan.SCAN_UNROLL_ANNOTATION
      )
      for n in gm.graph.nodes
      if n.op == "call_function" and n.target is _INVOKE_SUBGRAPH
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


class NestedCompileRegionTest(seed_test_utils.RepeatableTest):

  def _forward_graph(
      self, model: Callable[..., Any], *args: torch.Tensor
  ) -> tuple[torch.Tensor, fx.GraphModule]:
    """Compiles `model` on CPU and returns its output and forward graph."""
    graphs: list[fx.GraphModule] = []

    def fw_compiler(gm: fx.GraphModule, example_inputs: Any) -> Any:
      del example_inputs  # Unused.
      graphs.append(gm)
      return aot_autograd.make_boxed_func(gm.forward)

    torch._dynamo.reset()  # pylint: disable=protected-access
    with (
        torch.no_grad(),
        # Keep the `invoke_subgraph` calls.
        _keep_invoke_subgraph_calls(),
    ):
      compiled = torch.compile(
          model,
          backend=common.aot_autograd(fw_compiler=fw_compiler),
          fullgraph=True,
      )
      out = compiled(*args)
    self.assertLen(graphs, 1)
    return out, graphs[0]

  def _make_stack(self, decorator: Any) -> torch.nn.Module:
    class Layer(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

      @decorator
      def forward(self, x):
        return x + torch.tanh(self.lin(x))

    class Stack(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer() for _ in range(3)])

      def forward(self, x):
        for layer in self.layers:
          x = layer(x)
        return x

    return Stack().eval()

  def test_scan_unroll_annotates_the_calls(self):
    model = self._make_stack(tt_ncr.nested_compile_region(scan_unroll=2))
    x = torch.randn(2, 8)

    out, gm = self._forward_graph(model, x)

    self.assertEqual(_scan_unroll_annotations(gm), [2, 2, 2])
    with torch.no_grad():
      utils.assert_close(out, model(x))

  def test_without_scan_unroll_the_calls_are_not_annotated(self):
    for decorator in (
        tt_ncr.nested_compile_region,
        tt_ncr.nested_compile_region(),
    ):
      with self.subTest(decorator=decorator):
        _, gm = self._forward_graph(
            self._make_stack(decorator), torch.randn(2, 8)
        )

        self.assertEqual(_scan_unroll_annotations(gm), [None, None, None])

  def test_function_region(self):
    @tt_ncr.nested_compile_region(scan_unroll=1)
    def body(x):
      return torch.sin(x)

    def f(x):
      return body(body(x))

    x = torch.randn(4)
    out, gm = self._forward_graph(f, x)

    self.assertEqual(_scan_unroll_annotations(gm), [1, 1])
    utils.assert_close(out, torch.sin(torch.sin(x)))


if __name__ == "__main__":
  absltest.main()
