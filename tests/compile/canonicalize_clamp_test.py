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

"""Tests for canonicalize_clamp FX pass."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import fx
from torch_tpu._internal.compile.fx_passes import canonicalize_clamp
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils

GraphModule = fx.GraphModule
Graph = fx.Graph


class CanonicalizeClampTest(seed_test_utils.RepeatableTest):

  @parameterized.named_parameters(
      ("int_zero", 0),
      ("float_zero", 0.0),
  )
  def test_forward_clamp_min_to_relu(self, min_val):
    """Verifies clamp_min(x, 0) canonicalizes to relu(x)."""
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, 4), dtype=torch.float32)
    out = graph.call_function(torch.ops.aten.clamp_min.default, (x, min_val))
    out.meta["val"] = torch.empty((2, 4), dtype=torch.float32)
    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    canonicalize_clamp.apply(gm)

    call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    self.assertLen(call_nodes, 1)
    self.assertEqual(call_nodes[0].target, torch.ops.aten.relu.default)
    self.assertEqual(call_nodes[0].args, (x,))

  def test_forward_clamp_min_scalar_tensor_zero(self):
    """Verifies clamp_min(x, scalar_tensor(0)) canonicalizes to relu(x)."""
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, 4), dtype=torch.float32)
    zero = graph.call_function(torch.ops.aten.scalar_tensor.default, (0,))
    out = graph.call_function(torch.ops.aten.clamp_min.Tensor, (x, zero))
    out.meta["val"] = torch.empty((2, 4), dtype=torch.float32)
    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    canonicalize_clamp.apply(gm)

    call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    self.assertLen(call_nodes, 1)
    self.assertEqual(call_nodes[0].target, torch.ops.aten.relu.default)
    self.assertEqual(call_nodes[0].args, (x,))

  def test_forward_clamp_min_nonzero_untouched(self):
    """Verifies clamp_min(x, 1) is not modified."""
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, 4), dtype=torch.float32)
    out = graph.call_function(torch.ops.aten.clamp_min.default, (x, 1))
    out.meta["val"] = torch.empty((2, 4), dtype=torch.float32)
    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    canonicalize_clamp.apply(gm)

    call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    self.assertLen(call_nodes, 1)
    self.assertEqual(call_nodes[0].target, torch.ops.aten.clamp_min.default)

  @parameterized.named_parameters(
      ("int_zero", 0),
      ("float_zero", 0.0),
  )
  def test_forward_clamp_zero_none_to_relu(self, min_val):
    """Verifies clamp(x, 0, None) canonicalizes to relu(x)."""
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, 4), dtype=torch.float32)
    out = graph.call_function(torch.ops.aten.clamp.default, (x, min_val, None))
    out.meta["val"] = torch.empty((2, 4), dtype=torch.float32)
    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    canonicalize_clamp.apply(gm)

    call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    self.assertLen(call_nodes, 1)
    self.assertEqual(call_nodes[0].target, torch.ops.aten.relu.default)
    self.assertEqual(call_nodes[0].args, (x,))

  @parameterized.named_parameters(
      ("int_bounds", 0, 6),
      ("float_bounds", 0.0, 6.0),
  )
  def test_forward_clamp_zero_six_to_hardtanh(self, min_val, max_val):
    """Verifies clamp(x, 0, 6) canonicalizes to hardtanh(x, 0.0, 6.0)."""
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, 4), dtype=torch.float32)
    out = graph.call_function(
        torch.ops.aten.clamp.default, (x, min_val, max_val)
    )
    out.meta["val"] = torch.empty((2, 4), dtype=torch.float32)
    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    canonicalize_clamp.apply(gm)

    call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    self.assertLen(call_nodes, 1)
    self.assertEqual(call_nodes[0].target, torch.ops.aten.hardtanh.default)
    self.assertEqual(call_nodes[0].args, (x, 0.0, 6.0))

  def test_forward_clamp_other_bounds_untouched(self):
    """Verifies clamp with bounds other than (0, None) or (0, 6) is untouched."""
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, 4), dtype=torch.float32)
    out = graph.call_function(torch.ops.aten.clamp.default, (x, 1, 5))
    out.meta["val"] = torch.empty((2, 4), dtype=torch.float32)
    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    canonicalize_clamp.apply(gm)

    call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    self.assertLen(call_nodes, 1)
    self.assertEqual(call_nodes[0].target, torch.ops.aten.clamp.default)

  @parameterized.named_parameters(
      ("gt_int_zero", torch.ops.aten.gt.Scalar, 0),
      ("gt_float_zero", torch.ops.aten.gt.Scalar, 0.0),
      ("ge_int_zero", torch.ops.aten.ge.Scalar, 0),
  )
  def test_backward_where_gt_zero(self, cmp_op, zero_val):
    """Verifies where(gt(x, 0), grad, 0) canonicalizes to threshold_backward."""
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((4, 8), dtype=torch.float32)
    grad = graph.placeholder("grad")
    grad.meta["val"] = torch.empty((4, 8), dtype=torch.float32)

    cond = graph.call_function(cmp_op, (x, zero_val))
    cond.meta["val"] = torch.empty((4, 8), dtype=torch.bool)

    out = graph.call_function(torch.ops.aten.where.self, (cond, grad, zero_val))
    out.meta["val"] = torch.empty((4, 8), dtype=torch.float32)
    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    canonicalize_clamp.apply(gm)

    call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    self.assertLen(call_nodes, 1)
    self.assertEqual(
        call_nodes[0].target, torch.ops.aten.threshold_backward.default
    )
    self.assertEqual(call_nodes[0].args, (grad, x, 0))

  def test_backward_where_inverted_condition(self):
    """Verifies where(le(x, 0), 0, grad) canonicalizes to threshold_backward."""
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((4, 8), dtype=torch.float32)
    grad = graph.placeholder("grad")
    grad.meta["val"] = torch.empty((4, 8), dtype=torch.float32)

    cond = graph.call_function(torch.ops.aten.le.Scalar, (x, 0))
    cond.meta["val"] = torch.empty((4, 8), dtype=torch.bool)

    out = graph.call_function(torch.ops.aten.where.self, (cond, 0, grad))
    out.meta["val"] = torch.empty((4, 8), dtype=torch.float32)
    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    canonicalize_clamp.apply(gm)

    call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    self.assertLen(call_nodes, 1)
    self.assertEqual(
        call_nodes[0].target, torch.ops.aten.threshold_backward.default
    )
    self.assertEqual(call_nodes[0].args, (grad, x, 0))

  def test_backward_where_nonzero_untouched(self):
    """Verifies where(gt(x, 1), grad, 0) is not modified."""
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((4, 8), dtype=torch.float32)
    grad = graph.placeholder("grad")
    grad.meta["val"] = torch.empty((4, 8), dtype=torch.float32)

    cond = graph.call_function(torch.ops.aten.gt.Scalar, (x, 1))
    cond.meta["val"] = torch.empty((4, 8), dtype=torch.bool)

    out = graph.call_function(torch.ops.aten.where.self, (cond, grad, 0))
    out.meta["val"] = torch.empty((4, 8), dtype=torch.float32)
    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    canonicalize_clamp.apply(gm)

    where_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target)
        == torch.ops.aten.where
    ]
    self.assertLen(where_nodes, 1)

  @parameterized.named_parameters(
      ("grad_first", True),
      ("mask_first", False),
  )
  def test_backward_mul_grad_gt_zero(self, grad_first):
    """Verifies mul(grad, gt(x, 0)) canonicalizes to threshold_backward."""
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((4, 8), dtype=torch.float32)
    grad = graph.placeholder("grad")
    grad.meta["val"] = torch.empty((4, 8), dtype=torch.float32)

    cond = graph.call_function(torch.ops.aten.gt.Scalar, (x, 0))
    cond.meta["val"] = torch.empty((4, 8), dtype=torch.bool)

    to_copy = graph.call_function(
        torch.ops.aten._to_copy.default,
        (cond,),
        {"dtype": torch.float32},
    )
    to_copy.meta["val"] = torch.empty((4, 8), dtype=torch.float32)

    mul_args = (grad, to_copy) if grad_first else (to_copy, grad)
    out = graph.call_function(torch.ops.aten.mul.Tensor, mul_args)
    out.meta["val"] = torch.empty((4, 8), dtype=torch.float32)
    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    canonicalize_clamp.apply(gm)

    call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    self.assertLen(call_nodes, 1)
    self.assertEqual(
        call_nodes[0].target, torch.ops.aten.threshold_backward.default
    )
    self.assertEqual(call_nodes[0].args, (grad, x, 0))

  def test_backward_mul_other_untouched(self):
    """Verifies regular tensor multiplication is untouched."""
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((4, 8), dtype=torch.float32)
    grad = graph.placeholder("grad")
    grad.meta["val"] = torch.empty((4, 8), dtype=torch.float32)

    out = graph.call_function(torch.ops.aten.mul.Tensor, (grad, x))
    out.meta["val"] = torch.empty((4, 8), dtype=torch.float32)
    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    canonicalize_clamp.apply(gm)

    call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    self.assertLen(call_nodes, 1)
    self.assertEqual(call_nodes[0].target, torch.ops.aten.mul.Tensor)

  def test_backward_clamp_backward(self):
    """Verifies clamp_backward(grad, x, 0, None) -> threshold_backward."""
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((4, 8), dtype=torch.float32)
    grad = graph.placeholder("grad")
    grad.meta["val"] = torch.empty((4, 8), dtype=torch.float32)

    # Use a dummy function named clamp_backward to simulate backward op
    def clamp_backward(grad, self, min=None, max=None):
      return grad

    out = graph.call_function(clamp_backward, (grad, x, 0, None))
    out.meta["val"] = torch.empty((4, 8), dtype=torch.float32)
    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    canonicalize_clamp.apply(gm)

    call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    self.assertLen(call_nodes, 1)
    self.assertEqual(
        call_nodes[0].target, torch.ops.aten.threshold_backward.default
    )
    self.assertEqual(call_nodes[0].args, (grad, x, 0))

  def test_backward_clamp_min_backward(self):
    """Verifies clamp_min_backward(grad, x, 0) -> threshold_backward."""
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((4, 8), dtype=torch.float32)
    grad = graph.placeholder("grad")
    grad.meta["val"] = torch.empty((4, 8), dtype=torch.float32)

    def clamp_min_backward(grad, self, min=0):
      return grad

    out = graph.call_function(clamp_min_backward, (grad, x, 0))
    out.meta["val"] = torch.empty((4, 8), dtype=torch.float32)
    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    canonicalize_clamp.apply(gm)

    call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    self.assertLen(call_nodes, 1)
    self.assertEqual(
        call_nodes[0].target, torch.ops.aten.threshold_backward.default
    )
    self.assertEqual(call_nodes[0].args, (grad, x, 0))

  def test_bare_graph(self):
    """Verifies apply works when passed a bare fx.Graph directly."""
    graph = Graph()
    x = graph.placeholder("x")
    out = graph.call_function(torch.ops.aten.clamp_min.default, (x, 0))
    graph.output((out,))

    canonicalize_clamp.apply(graph)

    call_nodes = [n for n in graph.nodes if n.op == "call_function"]
    self.assertLen(call_nodes, 1)
    self.assertEqual(call_nodes[0].target, torch.ops.aten.relu.default)

  def test_disabled_by_env_var(self):
    """Verifies pass is a no-op when disabled via environment variable."""
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, 4), dtype=torch.float32)
    out = graph.call_function(torch.ops.aten.clamp_min.default, (x, 0))
    out.meta["val"] = torch.empty((2, 4), dtype=torch.float32)
    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    orig_nodes = list(gm.graph.nodes)
    with mock.patch.object(
        canonicalize_clamp,
        "_is_canonicalize_clamp_enabled",
        return_value=False,
    ):
      canonicalize_clamp.apply(gm)

    self.assertEqual(list(gm.graph.nodes), orig_nodes)

  def test_numerical_equivalence_forward_and_backward(self):
    """Verifies outputs of canonicalized graph match original ops numerically."""
    x_val = torch.tensor([-2.0, -0.5, 0.0, 1.5, 5.0, 8.0], dtype=torch.float32)
    grad_val = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32)

    # 1. Forward relu vs clamp_min(0)
    graph_fwd = Graph()
    x_ph = graph_fwd.placeholder("x")
    out_fwd = graph_fwd.call_function(
        torch.ops.aten.clamp_min.default, (x_ph, 0.0)
    )
    graph_fwd.output(out_fwd)
    gm_fwd = GraphModule(torch.nn.Module(), graph_fwd)

    res_before = gm_fwd(x_val)
    canonicalize_clamp.apply(gm_fwd)
    res_after = gm_fwd(x_val)
    utils.assert_close(res_after, res_before)
    utils.assert_close(res_after, torch.relu(x_val))

    # 2. Forward hardtanh vs clamp(0, 6)
    graph_ht = Graph()
    x_ph2 = graph_ht.placeholder("x")
    out_ht = graph_ht.call_function(
        torch.ops.aten.clamp.default, (x_ph2, 0.0, 6.0)
    )
    graph_ht.output(out_ht)
    gm_ht = GraphModule(torch.nn.Module(), graph_ht)

    res_ht_before = gm_ht(x_val)
    canonicalize_clamp.apply(gm_ht)
    res_ht_after = gm_ht(x_val)
    utils.assert_close(res_ht_after, res_ht_before)
    utils.assert_close(res_ht_after, torch.clamp(x_val, min=0.0, max=6.0))

    # 3. Backward where.self vs threshold_backward
    graph_bwd = Graph()
    x_ph3 = graph_bwd.placeholder("x")
    grad_ph = graph_bwd.placeholder("grad")
    cond = graph_bwd.call_function(torch.ops.aten.gt.Scalar, (x_ph3, 0.0))
    zero_tensor = graph_bwd.call_function(
        torch.ops.aten.zeros_like.default, (grad_ph,)
    )
    out_bwd = graph_bwd.call_function(
        torch.ops.aten.where.self, (cond, grad_ph, zero_tensor)
    )
    graph_bwd.output(out_bwd)
    gm_bwd = GraphModule(torch.nn.Module(), graph_bwd)

    res_bwd_before = gm_bwd(x_val, grad_val)
    canonicalize_clamp.apply(gm_bwd)
    res_bwd_after = gm_bwd(x_val, grad_val)
    utils.assert_close(res_bwd_after, res_bwd_before)
    utils.assert_close(
        res_bwd_after, torch.ops.aten.threshold_backward(grad_val, x_val, 0)
    )

    # 4. Backward where.ScalarOther vs threshold_backward
    graph_bwd_scalar = Graph()
    x_ph4 = graph_bwd_scalar.placeholder("x")
    grad_ph4 = graph_bwd_scalar.placeholder("grad")
    cond4 = graph_bwd_scalar.call_function(
        torch.ops.aten.gt.Scalar, (x_ph4, 0.0)
    )
    out_bwd_scalar = graph_bwd_scalar.call_function(
        torch.ops.aten.where.ScalarOther, (cond4, grad_ph4, 0.0)
    )
    graph_bwd_scalar.output(out_bwd_scalar)
    gm_bwd_scalar = GraphModule(torch.nn.Module(), graph_bwd_scalar)

    res_bwd_scalar_before = gm_bwd_scalar(x_val, grad_val)
    canonicalize_clamp.apply(gm_bwd_scalar)
    res_bwd_scalar_after = gm_bwd_scalar(x_val, grad_val)
    utils.assert_close(res_bwd_scalar_after, res_bwd_scalar_before)
    utils.assert_close(
        res_bwd_scalar_after,
        torch.ops.aten.threshold_backward(grad_val, x_val, 0),
    )

  def test_forward_clamp_kwargs(self):
    """Verifies clamp with kwargs input and min=0 rewrites to relu."""
    graph = fx.Graph()
    x = graph.placeholder("x")
    out = graph.call_function(
        torch.ops.aten.clamp.default,
        kwargs={"input": x, "min": 0, "max": None},
    )
    graph.output(out)
    gm = GraphModule(torch.nn.Module(), graph)
    canonicalize_clamp.apply(gm)

    ops = [n.target for n in gm.graph.nodes if n.op == "call_function"]
    self.assertIn(torch.ops.aten.relu.default, ops)
    self.assertNotIn(torch.ops.aten.clamp.default, ops)

  def test_nested_graph_module_traversal(self):
    """Verifies recursive traversal processes nested GraphModules."""
    sub_graph = fx.Graph()
    sub_x = sub_graph.placeholder("x")
    sub_out = sub_graph.call_function(
        torch.ops.aten.clamp_min.default, (sub_x, 0)
    )
    sub_graph.output(sub_out)
    sub_gm = GraphModule(torch.nn.Module(), sub_graph)

    root_module = torch.nn.Module()
    root_module.submod = sub_gm
    root_graph = fx.Graph()
    root_x = root_graph.placeholder("x")
    root_out = root_graph.call_module("submod", (root_x,))
    root_graph.output(root_out)
    root_gm = GraphModule(root_module, root_graph)

    canonicalize_clamp.apply(root_gm)

    sub_ops = [
        n.target for n in root_gm.submod.graph.nodes if n.op == "call_function"
    ]
    self.assertIn(torch.ops.aten.relu.default, sub_ops)
    self.assertNotIn(torch.ops.aten.clamp_min.default, sub_ops)


if __name__ == "__main__":
  absltest.main()
