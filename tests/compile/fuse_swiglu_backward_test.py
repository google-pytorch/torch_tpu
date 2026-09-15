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

"""Unit tests for fuse_swiglu_backward FX pass."""

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import fx
from torch_tpu._internal.compile.fx_passes import fuse_swiglu_backward
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class FuseSwigluBackwardTest(seed_test_utils.RepeatableTest):

  @parameterized.named_parameters(
      ("dy_first", True),
      ("up_first", False),
  )
  def test_fuse_swiglu_backward_basic(self, dy_first):
    """Verifies that SwiGLU backward pattern is fused and dy * up is eliminated."""
    graph = fx.Graph()
    dy = graph.placeholder("dy")
    gate = graph.placeholder("gate")
    up = graph.placeholder("up")

    shape = (2, 4)
    dy.meta["val"] = torch.empty(shape, dtype=torch.float32)
    gate.meta["val"] = torch.empty(shape, dtype=torch.float32)
    up.meta["val"] = torch.empty(shape, dtype=torch.float32)

    # Forward silu(gate)
    silu_gate = graph.call_function(torch.ops.aten.silu.default, (gate,))
    silu_gate.meta["val"] = torch.empty(shape, dtype=torch.float32)

    # d_up = dy * silu(gate)
    d_up = graph.call_function(torch.ops.aten.mul.Tensor, (dy, silu_gate))
    d_up.meta["val"] = torch.empty(shape, dtype=torch.float32)

    # dy_up = dy * up (or up * dy)
    mul_args = (dy, up) if dy_first else (up, dy)
    dy_up = graph.call_function(torch.ops.aten.mul.Tensor, mul_args)
    dy_up.meta["val"] = torch.empty(shape, dtype=torch.float32)

    # d_gate = silu_backward(dy_up, gate)
    d_gate = graph.call_function(
        torch.ops.aten.silu_backward.default, (dy_up, gate)
    )
    d_gate.meta["val"] = torch.empty(shape, dtype=torch.float32)

    graph.output((d_up, d_gate))
    gm = fx.GraphModule(torch.nn.Module(), graph)

    fuse_swiglu_backward.apply(gm)

    # 1. Check that silu_backward is eliminated
    sbw_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target)
        == torch.ops.aten.silu_backward
    ]
    self.assertEmpty(sbw_nodes)

    # 2. Check that sigmoid is introduced
    sigmoid_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target)
        == torch.ops.aten.sigmoid
    ]
    self.assertLen(sigmoid_nodes, 1)

    # 3. Check that the original silu node was eliminated because it had only 1 user
    silu_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.silu
    ]
    self.assertEmpty(silu_nodes)

    # 4. Check numerical equivalence with golden PyTorch execution
    gate_val = torch.randn(shape, dtype=torch.float32)
    up_val = torch.randn(shape, dtype=torch.float32)
    dy_val = torch.randn(shape, dtype=torch.float32)

    # Golden calculation
    golden_s = torch.sigmoid(gate_val)
    golden_d_up = dy_val * (gate_val * golden_s)
    golden_dsilu = golden_s * (1.0 + gate_val * (1.0 - golden_s))
    golden_d_gate = (dy_val * up_val) * golden_dsilu

    actual_d_up, actual_d_gate = gm(dy_val, gate_val, up_val)
    utils.assert_close(actual_d_up, golden_d_up)
    utils.assert_close(actual_d_gate, golden_d_gate)

  def test_fuse_swiglu_backward_preserves_silu_if_other_users(self):
    """If silu(gate) is used elsewhere, keep silu(gate) node."""
    graph = fx.Graph()
    dy = graph.placeholder("dy")
    gate = graph.placeholder("gate")
    up = graph.placeholder("up")

    shape = (2, 4)
    dy.meta["val"] = torch.empty(shape, dtype=torch.float32)
    gate.meta["val"] = torch.empty(shape, dtype=torch.float32)
    up.meta["val"] = torch.empty(shape, dtype=torch.float32)

    silu_gate = graph.call_function(torch.ops.aten.silu.default, (gate,))
    silu_gate.meta["val"] = torch.empty(shape, dtype=torch.float32)

    d_up = graph.call_function(torch.ops.aten.mul.Tensor, (dy, silu_gate))
    d_up.meta["val"] = torch.empty(shape, dtype=torch.float32)

    dy_up = graph.call_function(torch.ops.aten.mul.Tensor, (dy, up))
    dy_up.meta["val"] = torch.empty(shape, dtype=torch.float32)

    d_gate = graph.call_function(
        torch.ops.aten.silu_backward.default, (dy_up, gate)
    )
    d_gate.meta["val"] = torch.empty(shape, dtype=torch.float32)

    # Extra user of silu_gate
    extra = graph.call_function(torch.ops.aten.neg.default, (silu_gate,))

    graph.output((d_up, d_gate, extra))
    gm = fx.GraphModule(torch.nn.Module(), graph)

    fuse_swiglu_backward.apply(gm)

    # silu_gate should still exist because it has another user
    silu_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.silu
    ]
    self.assertLen(silu_nodes, 1)

  @mock.patch.dict(
      "os.environ",
      {"TORCH_TPU_INTERNAL_ENABLE_FUSE_SWIGLU_BACKWARD": "0"},
  )
  def test_fuse_swiglu_backward_disabled_by_env(self):
    graph = fx.Graph()
    dy = graph.placeholder("dy")
    gate = graph.placeholder("gate")
    up = graph.placeholder("up")

    silu_gate = graph.call_function(torch.ops.aten.silu.default, (gate,))
    d_up = graph.call_function(torch.ops.aten.mul.Tensor, (dy, silu_gate))
    dy_up = graph.call_function(torch.ops.aten.mul.Tensor, (dy, up))
    d_gate = graph.call_function(
        torch.ops.aten.silu_backward.default, (dy_up, gate)
    )
    graph.output((d_up, d_gate))
    gm = fx.GraphModule(torch.nn.Module(), graph)

    fuse_swiglu_backward.apply(gm)

    # When disabled, silu_backward should still be present
    sbw_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target)
        == torch.ops.aten.silu_backward
    ]
    self.assertLen(sbw_nodes, 1)


if __name__ == "__main__":
  absltest.main()
