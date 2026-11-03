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

"""Runs graphs rewritten by invoke_subgraph_to_scan with TorchTPU's scan."""

import contextlib
from typing import Any

from absl.testing import absltest
import torch
from torch import fx
from torch._dynamo.backends import common
from torch._functorch import aot_autograd
import torch_tpu  # pylint: disable=unused-import  # noqa: F401
from torch_tpu._internal.compile.fx_passes import invoke_subgraph_to_scan
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils

_SCAN = torch.ops.higher_order.scan


def _num_scans(gm: fx.GraphModule) -> int:
  return sum(
      n.op == "call_function" and n.target is _SCAN for n in gm.graph.nodes
  )


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


class InvokeSubgraphToScanTpuTest(seed_test_utils.RepeatableTest):

  def _run_rewritten(
      self, model: torch.nn.Module, *args: torch.Tensor
  ) -> tuple[torch.Tensor, fx.GraphModule]:
    """Runs `model` on TPU through its forward graph rewritten by the pass.

    The graph runs op by op, so its scan goes through TorchTPU's scan
    lowering.
    """
    graphs: list[fx.GraphModule] = []

    def fw_compiler(gm: fx.GraphModule, example_inputs: Any) -> Any:
      del example_inputs  # Unused.
      invoke_subgraph_to_scan.apply(gm)
      graphs.append(gm)
      return aot_autograd.make_boxed_func(gm.forward)

    torch._dynamo.reset()  # pylint: disable=protected-access
    with (
        torch.no_grad(),
        # Keep the `invoke_subgraph` calls for the pass.
        _keep_invoke_subgraph_calls(),
    ):
      compiled = torch.compile(
          model.to("tpu"),
          backend=common.aot_autograd(fw_compiler=fw_compiler),
          fullgraph=True,
      )
      out = compiled(*[a.to("tpu") for a in args]).to("cpu")
    self.assertLen(graphs, 1)
    return out, graphs[0]

  def test_per_layer_weights(self):
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
        return x

    model = Stack().eval()
    x = torch.randn(2, 16)
    with torch.no_grad():
      expected = model(x)
    out, gm = self._run_rewritten(model, x)

    self.assertEqual(_num_scans(gm), 1)
    # TPU matmuls use reduced precision by default.
    utils.assert_close(out, expected, rtol=2e-2, atol=2e-2)

  def test_shared_tensor_operand_and_no_scanned_inputs(self):
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
    expected = model(*args)
    out, gm = self._run_rewritten(model, *args)

    self.assertEqual(_num_scans(gm), 1)
    utils.assert_close(out, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
