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

"""Unit tests for optimize_recurrence_and_slices pass."""

from absl.testing import absltest
import torch
import torch.func as func
from torch.fx.experimental.proxy_tensor import make_fx
from torch_tpu._internal.compile.fx_passes import optimize_recurrence_and_slices
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class OptimizeRecurrenceAndSlicesTest(seed_test_utils.RepeatableTest):

  def test_linear_recurrence_forward_and_backward_vectorization(self):
    batch, seq_len, heads, d_head, d_state = 2, 8, 4, 16, 16
    a = torch.sigmoid(torch.randn(batch, seq_len, heads, 1, 1))
    b = torch.randn(batch, seq_len, heads, d_head, d_state)

    def scan_fn(a_in, b_in):
      h = torch.zeros(
          batch, heads, d_head, d_state, dtype=b_in.dtype, device=b_in.device
      )
      hs = []
      for t in range(seq_len):
        h = a_in[:, t] * h + b_in[:, t]
        hs.append(h)
      out = torch.stack(hs, dim=1)
      return out.sum()

    def step(a_arg, b_arg):
      grads, val = func.grad_and_value(scan_fn, argnums=(0, 1))(a_arg, b_arg)
      return val, grads

    ref_val, (ref_grad_a, ref_grad_b) = step(a, b)
    gm = make_fx(step, tracing_mode="fake")(a, b)
    optimize_recurrence_and_slices.apply(gm)

    # All select_backward and unrolled sequence select(dim=1) ops must be eliminated
    select_bwd_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == torch.ops.aten.select_backward.default
    ]
    seq_select_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == torch.ops.aten.select.int
        and len(n.args) >= 2
        and n.args[1] == 1
    ]
    self.assertEmpty(select_bwd_nodes)
    self.assertEmpty(seq_select_nodes)

    opt_val, (opt_grad_a, opt_grad_b) = gm(a, b)
    utils.assert_close(opt_val, ref_val, rtol=1e-4, atol=1e-4)
    utils.assert_close(opt_grad_a, ref_grad_a, rtol=1e-4, atol=1e-4)
    utils.assert_close(opt_grad_b, ref_grad_b, rtol=1e-4, atol=1e-4)

  def test_strided_and_contiguous_slice_backward_fusion(self):
    x = torch.randn(2, 4, 16)
    cos = torch.randn(2, 4, 4)
    sin = torch.randn(2, 4, 4)

    def rope_slice_fn(x_in, cos_in, sin_in):
      x_part = x_in[..., :8]
      x_pass = x_in[..., 8:]
      x1 = x_part[..., 0::2]
      x2 = x_part[..., 1::2]
      out_part = torch.stack(
          [x1 * cos_in - x2 * sin_in, x1 * sin_in + x2 * cos_in], dim=-1
      )
      out_part = out_part.flatten(-2, -1)
      out = torch.cat([out_part, x_pass], dim=-1)
      return (out**2).sum()

    def step(x_arg, cos_arg, sin_arg):
      grad, val = func.grad_and_value(rope_slice_fn, argnums=0)(
          x_arg, cos_arg, sin_arg
      )
      return val, grad

    ref_val, ref_grad = step(x, cos, sin)
    gm = make_fx(step, tracing_mode="fake")(x, cos, sin)
    optimize_recurrence_and_slices.apply(gm)

    slice_bwd_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == torch.ops.aten.slice_backward.default
    ]
    self.assertEmpty(slice_bwd_nodes)

    opt_val, opt_grad = gm(x, cos, sin)
    utils.assert_close(opt_val, ref_val, rtol=1e-5, atol=1e-5)
    utils.assert_close(opt_grad, ref_grad, rtol=1e-5, atol=1e-5)

  def test_eager_and_mlir_linear_recurrence_vectorization(self):
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator required for eager MLIR execution.")
    device = torch.device("tpu")
    batch, seq_len, heads, d_head, d_state = 2, 16, 2, 8, 8
    a_cpu = torch.sigmoid(torch.randn(batch, seq_len, heads, 1, 1))
    b_cpu = torch.randn(batch, seq_len, heads, d_head, d_state)

    def run_recurrence(a_in, b_in):
      h = torch.zeros(
          batch, heads, d_head, d_state, dtype=b_in.dtype, device=b_in.device
      )
      hs = []
      for t in range(seq_len):
        h = a_in[:, t] * h + b_in[:, t]
        hs.append(h)
      return torch.stack(hs, dim=1), h

    ref_out, ref_h = run_recurrence(a_cpu, b_cpu)
    a_tpu = a_cpu.to(device).requires_grad_(True)
    b_tpu = b_cpu.to(device).requires_grad_(True)
    out_tpu, h_tpu = run_recurrence(a_tpu, b_tpu)
    loss_tpu = out_tpu.sum() + h_tpu.sum()
    loss_tpu.backward()

    a_ref = a_cpu.clone().requires_grad_(True)
    b_ref = b_cpu.clone().requires_grad_(True)
    out_ref, h_ref = run_recurrence(a_ref, b_ref)
    (out_ref.sum() + h_ref.sum()).backward()

    utils.assert_close(out_tpu.cpu(), ref_out, rtol=1e-4, atol=1e-4)
    utils.assert_close(h_tpu.cpu(), ref_h, rtol=1e-4, atol=1e-4)
    utils.assert_close(a_tpu.grad.cpu(), a_ref.grad, rtol=1e-4, atol=1e-4)
    utils.assert_close(b_tpu.grad.cpu(), b_ref.grad, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  absltest.main()
