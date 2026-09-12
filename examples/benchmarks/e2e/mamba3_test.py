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

"""Tests for mamba3."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from examples.benchmarks.e2e import mamba3
from tests import seed_test_utils


class Mamba3Test(seed_test_utils.RepeatableTest):

  @parameterized.parameters(
      {"is_mimo": False, "mimo_rank": 1, "ngroups": 1},
      {"is_mimo": False, "mimo_rank": 1, "ngroups": 2},
      {"is_mimo": True, "mimo_rank": 4, "ngroups": 1},
      {"is_mimo": True, "mimo_rank": 4, "ngroups": 2},
  )
  def test_forward_eval(self, is_mimo: bool, mimo_rank: int, ngroups: int = 1):
    layer = mamba3.Mamba3(
        d_model=64,
        d_state=32,
        expand=2,
        headdim=32,
        ngroups=ngroups,
        is_mimo=is_mimo,
        mimo_rank=mimo_rank,
    )
    layer.eval()
    u = torch.randn(2, 16, 64)
    with torch.no_grad():
      out = layer(u)
    self.assertEqual(out.shape, (2, 16, 64))
    self.assertFalse(torch.isnan(out).any())

  @parameterized.parameters(
      {"is_mimo": False, "mimo_rank": 1, "ngroups": 1},
      {"is_mimo": False, "mimo_rank": 1, "ngroups": 2},
      {"is_mimo": True, "mimo_rank": 4, "ngroups": 1},
      {"is_mimo": True, "mimo_rank": 4, "ngroups": 2},
  )
  def test_forward_backward_train(
      self, is_mimo: bool, mimo_rank: int, ngroups: int = 1
  ):
    layer = mamba3.Mamba3(
        d_model=64,
        d_state=32,
        expand=2,
        headdim=32,
        ngroups=ngroups,
        is_mimo=is_mimo,
        mimo_rank=mimo_rank,
    )
    layer.train()
    u = torch.randn(2, 16, 64, requires_grad=True)
    out = layer(u)
    loss = out.sum()
    loss.backward()
    self.assertIsNotNone(u.grad)
    self.assertEqual(u.grad.shape, (2, 16, 64))
    self.assertFalse(torch.isnan(u.grad).any())

    for name, param in layer.named_parameters():
      if param.requires_grad:
        self.assertIsNotNone(param.grad, f"Gradient is None for {name}")
        self.assertFalse(
            torch.isnan(param.grad).any(), f"NaN gradient in {name}"
        )

  def test_invalid_ngroups_raises(self):
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates ngroups divisibility.
        ValueError, "nheads must be divisible by ngroups"
    ):
      mamba3.Mamba3(
          d_model=64,
          d_state=32,
          expand=2,
          headdim=32,  # nheads = (2 * 64) // 32 = 4
          ngroups=3,  # 4 % 3 != 0
      )

  def test_fx_optimization_pass(self):
    from torch.fx.experimental.proxy_tensor import make_fx
    import torch.func as func
    from torch_tpu._internal.compile.fx_passes import optimize_recurrence_and_slices

    layer = mamba3.Mamba3(
        d_model=64,
        d_state=32,
        expand=2,
        headdim=32,
        ngroups=2,
        is_mimo=False,
        mimo_rank=1,
    )
    u = torch.randn(2, 8, 64)
    params = dict(layer.named_parameters())
    buffers = dict(layer.named_buffers())

    def step(p, b, x):
      def loss_fn(p_arg):
        out = func.functional_call(layer, (p_arg, b), (x,))
        return out.mean()

      grads, loss = func.grad_and_value(loss_fn)(p)
      return loss, grads

    ref_loss, ref_grads = step(params, buffers, u)
    gm = make_fx(step, tracing_mode="fake")(params, buffers, u)
    optimize_recurrence_and_slices.apply(gm)

    select_bwd_count = sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == torch.ops.aten.select_backward.default
    )
    slice_bwd_count = sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == torch.ops.aten.slice_backward.default
    )
    seq_select_count = sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == torch.ops.aten.select.int
        and len(n.args) >= 2
        and n.args[1] == 1
    )
    self.assertEqual(select_bwd_count, 0)
    self.assertEqual(slice_bwd_count, 0)
    self.assertEqual(seq_select_count, 0)

    opt_loss, opt_grads = gm(params, buffers, u)
    torch.testing.assert_close(opt_loss, ref_loss, rtol=1e-4, atol=1e-4)
    for k in ref_grads:
      torch.testing.assert_close(
          opt_grads[k], ref_grads[k], rtol=1e-4, atol=1e-4
      )

  def test_config2_train_fx_optimization_pass(self):
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch._subclasses.fake_tensor import FakeTensorMode
    import torch.func as func
    from torch_tpu._internal.compile.fx_passes import optimize_recurrence_and_slices

    with FakeTensorMode():
      layer = mamba3.Mamba3(
          d_model=2048,
          d_state=128,
          expand=2,
          headdim=64,
          ngroups=8,
          is_mimo=False,
          mimo_rank=4,
      )
      u = torch.randn(8, 1024, 2048)
      params = dict(layer.named_parameters())
      buffers = dict(layer.named_buffers())

      def step(p, b, x):
        def loss_fn(p_arg):
          out = func.functional_call(layer, (p_arg, b), (x,))
          return out.mean()

        grads, loss = func.grad_and_value(loss_fn)(p)
        return loss, grads

      gm = make_fx(step)(params, buffers, u)
      optimize_recurrence_and_slices.apply(gm)

    select_bwd_count = sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == torch.ops.aten.select_backward.default
    )
    slice_bwd_count = sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == torch.ops.aten.slice_backward.default
    )
    const_pad_count = sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == torch.ops.aten.constant_pad_nd.default
    )
    self.assertEqual(select_bwd_count, 0)
    self.assertEqual(slice_bwd_count, 0)
    self.assertEqual(const_pad_count, 1)


if __name__ == "__main__":
  absltest.main()
