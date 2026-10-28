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

"""Unit tests for _fused_adamw_ and _fused_adamw_.tensor_lr in TorchTPU."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal import compile as tpu_compile
from tests import op_testing

TorchTpuVsCpuTestBase = op_testing.TorchTpuVsCpuTestBase


class FusedAdamWTest(TorchTpuVsCpuTestBase):
  """Tests for aten::_fused_adamw_ and aten::_fused_adamw_.tensor_lr."""

  @parameterized.product(
      maximize=[False, True],
      amsgrad=[False, True],
      weight_decay=[0.0, 0.01],
  )
  def test_fused_adamw_default(self, maximize, amsgrad, weight_decay):
    """Tests torch.ops.aten._fused_adamw_.default with scalar lr."""

    def run_fn(device):
      p1 = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32, device=device)
      p2 = torch.tensor(
          [[0.5, -0.5], [1.5, 0.0]], dtype=torch.float32, device=device
      )
      g1 = torch.tensor([0.1, -0.2, 0.5], dtype=torch.float32, device=device)
      g2 = torch.tensor(
          [[-0.1, 0.2], [0.3, -0.4]], dtype=torch.float32, device=device
      )
      ea1 = torch.tensor([0.01, 0.02, 0.0], dtype=torch.float32, device=device)
      ea2 = torch.tensor(
          [[0.0, 0.01], [-0.01, 0.02]], dtype=torch.float32, device=device
      )
      eas1 = torch.tensor(
          [0.001, 0.002, 0.001], dtype=torch.float32, device=device
      )
      eas2 = torch.tensor(
          [[0.001, 0.001], [0.002, 0.002]], dtype=torch.float32, device=device
      )

      if amsgrad:
        meas1 = torch.tensor(
            [0.002, 0.002, 0.002], dtype=torch.float32, device=device
        )
        meas2 = torch.tensor(
            [[0.002, 0.002], [0.002, 0.002]], dtype=torch.float32, device=device
        )
        meas = [meas1, meas2]
      else:
        meas = []

      step1 = torch.tensor(1.0, dtype=torch.float32, device=device)
      step2 = torch.tensor(1.0, dtype=torch.float32, device=device)

      torch.ops.aten._fused_adamw_.default(
          [p1, p2],
          [g1, g2],
          [ea1, ea2],
          [eas1, eas2],
          meas,
          [step1, step2],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=weight_decay,
          eps=1e-8,
          amsgrad=amsgrad,
          maximize=maximize,
      )
      return p1, p2, ea1, ea2, eas1, eas2, meas

    self.assert_close_tpu_vs_cpu(run_fn)

  @parameterized.product(
      maximize=[False, True],
      amsgrad=[False, True],
      weight_decay=[0.0, 0.01],
  )
  def test_fused_adamw_tensor_lr(self, maximize, amsgrad, weight_decay):
    """Tests torch.ops.aten._fused_adamw_.tensor_lr with tensor lr."""

    def run_fn(device):
      p1 = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32, device=device)
      p2 = torch.tensor(
          [[0.5, -0.5], [1.5, 0.0]], dtype=torch.float32, device=device
      )
      g1 = torch.tensor([0.1, -0.2, 0.5], dtype=torch.float32, device=device)
      g2 = torch.tensor(
          [[-0.1, 0.2], [0.3, -0.4]], dtype=torch.float32, device=device
      )
      ea1 = torch.tensor([0.01, 0.02, 0.0], dtype=torch.float32, device=device)
      ea2 = torch.tensor(
          [[0.0, 0.01], [-0.01, 0.02]], dtype=torch.float32, device=device
      )
      eas1 = torch.tensor(
          [0.001, 0.002, 0.001], dtype=torch.float32, device=device
      )
      eas2 = torch.tensor(
          [[0.001, 0.001], [0.002, 0.002]], dtype=torch.float32, device=device
      )

      if amsgrad:
        meas1 = torch.tensor(
            [0.002, 0.002, 0.002], dtype=torch.float32, device=device
        )
        meas2 = torch.tensor(
            [[0.002, 0.002], [0.002, 0.002]], dtype=torch.float32, device=device
        )
        meas = [meas1, meas2]
      else:
        meas = []

      step1 = torch.tensor(1.0, dtype=torch.float32, device=device)
      step2 = torch.tensor(1.0, dtype=torch.float32, device=device)
      lr_t = torch.tensor(0.001, dtype=torch.float32, device=device)

      torch.ops.aten._fused_adamw_.tensor_lr(
          [p1, p2],
          [g1, g2],
          [ea1, ea2],
          [eas1, eas2],
          meas,
          [step1, step2],
          lr=lr_t,
          beta1=0.9,
          beta2=0.999,
          weight_decay=weight_decay,
          eps=1e-8,
          amsgrad=amsgrad,
          maximize=maximize,
      )
      return p1, p2, ea1, ea2, eas1, eas2, meas

    self.assert_close_tpu_vs_cpu(run_fn)

  @parameterized.product(
      maximize=[False, True],
      amsgrad=[False, True],
      weight_decay=[0.0, 0.01],
  )
  def test_optim_adamw_fused(self, maximize, amsgrad, weight_decay):
    """Tests torch.optim.AdamW with fused=True."""

    def run_fn(device):
      p1 = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32, device=device)
      p2 = torch.tensor(
          [[0.5, -0.5], [1.5, 0.0]], dtype=torch.float32, device=device
      )
      g1 = torch.tensor([0.1, -0.2, 0.5], dtype=torch.float32, device=device)
      g2 = torch.tensor(
          [[-0.1, 0.2], [0.3, -0.4]], dtype=torch.float32, device=device
      )

      p1.grad = g1
      p2.grad = g2

      optimizer = torch.optim.AdamW(
          [p1, p2],
          lr=0.001,
          betas=(0.9, 0.999),
          eps=1e-8,
          weight_decay=weight_decay,
          amsgrad=amsgrad,
          maximize=maximize,
          fused=True,
      )
      optimizer.step()

      state1 = optimizer.state[p1]
      state2 = optimizer.state[p2]
      empty_t = torch.tensor([], device=device)
      return (
          p1,
          p2,
          state1["exp_avg"],
          state2["exp_avg"],
          state1["exp_avg_sq"],
          state2["exp_avg_sq"],
          state1.get("max_exp_avg_sq", empty_t),
          state2.get("max_exp_avg_sq", empty_t),
      )

    self.assert_close_tpu_vs_cpu(run_fn)

  @parameterized.product(
      found_inf_val=[0.0, 1.0],
      grad_scale_val=[1.0, 2.0],
  )
  def test_fused_adamw_grad_scale_and_found_inf(
      self, found_inf_val, grad_scale_val
  ):
    """Tests gradient scaling and found_inf conditional revert behavior."""

    def run_fn(device):
      p = torch.tensor([1.0, -2.0], dtype=torch.float32, device=device)
      g = torch.tensor([0.4, 0.8], dtype=torch.float32, device=device)
      ea = torch.tensor([0.1, 0.2], dtype=torch.float32, device=device)
      eas = torch.tensor([0.01, 0.02], dtype=torch.float32, device=device)
      step = torch.tensor(1.0, dtype=torch.float32, device=device)
      grad_scale = torch.tensor(
          grad_scale_val, dtype=torch.float32, device=device
      )
      found_inf = torch.tensor(
          found_inf_val, dtype=torch.float32, device=device
      )

      torch.ops.aten._fused_adamw_.default(
          [p],
          [g],
          [ea],
          [eas],
          [],
          [step],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
          grad_scale=grad_scale,
          found_inf=found_inf,
      )
      return p, ea, eas

    self.assert_close_tpu_vs_cpu(run_fn)

  def test_fused_adamw_multi_step(self):
    """Tests accumulation over multiple iterations (step > 1)."""

    def run_fn(device):
      p = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
      g = torch.tensor([0.1, -0.1], dtype=torch.float32, device=device)
      ea = torch.zeros_like(p)
      eas = torch.zeros_like(p)

      for step_val in range(1, 4):
        step = torch.tensor(float(step_val), dtype=torch.float32, device=device)
        torch.ops.aten._fused_adamw_.default(
            [p],
            [g],
            [ea],
            [eas],
            [],
            [step],
            lr=0.01,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.01,
            eps=1e-8,
            amsgrad=False,
            maximize=False,
        )
      return p, ea, eas

    self.assert_close_tpu_vs_cpu(run_fn)

  def test_fused_adamw_bfloat16(self):
    """Tests _fused_adamw_ on bfloat16 tensors."""

    def run_fn(device):
      p = torch.tensor([1.0, -1.0], dtype=torch.bfloat16, device=device)
      g = torch.tensor([0.2, 0.4], dtype=torch.bfloat16, device=device)
      ea = torch.tensor([0.01, 0.02], dtype=torch.bfloat16, device=device)
      eas = torch.tensor([0.001, 0.002], dtype=torch.bfloat16, device=device)
      step = torch.tensor(1.0, dtype=torch.float32, device=device)

      torch.ops.aten._fused_adamw_.default(
          [p],
          [g],
          [ea],
          [eas],
          [],
          [step],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )
      return p, ea, eas

    self.assert_close_tpu_vs_cpu(run_fn, atol=1e-3, rtol=1e-1)

  def test_fused_adamw_mixed_precision(self):
    """Tests _fused_adamw_ where params/grads are bfloat16 and states are float32."""

    def run_fn(device):
      p = torch.tensor([1.0, -1.0], dtype=torch.bfloat16, device=device)
      g = torch.tensor([0.2, 0.4], dtype=torch.bfloat16, device=device)
      ea = torch.tensor([0.01, 0.02], dtype=torch.float32, device=device)
      eas = torch.tensor([0.001, 0.002], dtype=torch.float32, device=device)
      step = torch.tensor(1.0, dtype=torch.float32, device=device)

      if str(device) == "cpu":
        p_fp32 = p.float()
        g_fp32 = g.float()
        torch.ops.aten._fused_adamw_.default(
            [p_fp32],
            [g_fp32],
            [ea],
            [eas],
            [],
            [step],
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.01,
            eps=1e-8,
            amsgrad=False,
            maximize=False,
        )
        p.copy_(p_fp32.to(torch.bfloat16))
      else:
        torch.ops.aten._fused_adamw_.default(
            [p],
            [g],
            [ea],
            [eas],
            [],
            [step],
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.01,
            eps=1e-8,
            amsgrad=False,
            maximize=False,
        )
      return p, ea, eas

    self.assert_close_tpu_vs_cpu(run_fn, atol=1e-3, rtol=1e-1)

  def test_fused_adamw_grad_unscaling_inplace(self):
    """Tests that grads are unscaled in-place when grad_scale is provided."""

    def run_fn(device):
      p = torch.tensor([1.0, -2.0], dtype=torch.float32, device=device)
      g = torch.tensor([0.4, 0.8], dtype=torch.float32, device=device)
      ea = torch.tensor([0.1, 0.2], dtype=torch.float32, device=device)
      eas = torch.tensor([0.01, 0.02], dtype=torch.float32, device=device)
      step = torch.tensor(1.0, dtype=torch.float32, device=device)
      grad_scale = torch.tensor(2.0, dtype=torch.float32, device=device)

      torch.ops.aten._fused_adamw_.default(
          [p],
          [g],
          [ea],
          [eas],
          [],
          [step],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
          grad_scale=grad_scale,
      )
      return p, g, ea, eas

    self.assert_close_tpu_vs_cpu(run_fn)

  def test_fused_adamw_found_inf_mixed_precision(self):
    """Tests that found_inf revert compiles and runs in bfloat16 mixed precision."""

    def run_fn(device):
      p = torch.tensor([1.0, -2.0], dtype=torch.bfloat16, device=device)
      g = torch.tensor([0.4, 0.8], dtype=torch.bfloat16, device=device)
      ea = torch.tensor([0.1, 0.2], dtype=torch.float32, device=device)
      eas = torch.tensor([0.01, 0.02], dtype=torch.float32, device=device)
      step = torch.tensor(1.0, dtype=torch.float32, device=device)
      found_inf = torch.tensor(1.0, dtype=torch.float32, device=device)

      torch.ops.aten._fused_adamw_.default(
          [p],
          [g],
          [ea],
          [eas],
          [],
          [step],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
          grad_scale=None,
          found_inf=found_inf,
      )
      return p, g, ea, eas

    self.assert_close_tpu_vs_cpu(run_fn)

  def test_fused_adamw_non_contiguous_grads(self):
    """Tests that _fused_adamw_ handles non-contiguous sliced gradient views."""

    def run_fn(device):
      p = torch.zeros((2, 2), dtype=torch.float32, device=device)
      g_full = torch.tensor(
          [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
          dtype=torch.float32,
          device=device,
      )
      g = g_full[:, ::2]
      ea = torch.zeros((2, 2), dtype=torch.float32, device=device)
      eas = torch.zeros((2, 2), dtype=torch.float32, device=device)
      step = torch.tensor(1.0, dtype=torch.float32, device=device)
      if str(device) == "cpu" or getattr(device, "type", "") == "cpu":
        g_arg = g.contiguous()
      else:
        assert not g.is_contiguous()
        g_arg = g

      torch.ops.aten._fused_adamw_.default(
          [p],
          [g_arg],
          [ea],
          [eas],
          [],
          [step],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )
      return p, g, ea, eas

    self.assert_close_tpu_vs_cpu(run_fn)

  def test_fused_adamw_0d_scalar_tensor(self):
    """Tests _fused_adamw_ on 0-D scalar parameters."""

    def run_fn(device):
      p = torch.tensor(1.5, dtype=torch.float32, device=device)
      g = torch.tensor(0.1, dtype=torch.float32, device=device)
      ea = torch.tensor(0.01, dtype=torch.float32, device=device)
      eas = torch.tensor(0.001, dtype=torch.float32, device=device)
      step = torch.tensor(1.0, dtype=torch.float32, device=device)

      torch.ops.aten._fused_adamw_.default(
          [p],
          [g],
          [ea],
          [eas],
          [],
          [step],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )
      return p, ea, eas

    self.assert_close_tpu_vs_cpu(run_fn)

  def test_fused_adamw_0_element_tensor(self):
    """Tests _fused_adamw_ with 0-element (empty) tensors in parameter list."""

    def run_fn(device):
      p = torch.empty((0, 5), dtype=torch.float32, device=device)
      g = torch.empty((0, 5), dtype=torch.float32, device=device)
      ea = torch.empty((0, 5), dtype=torch.float32, device=device)
      eas = torch.empty((0, 5), dtype=torch.float32, device=device)
      step = torch.tensor(1.0, dtype=torch.float32, device=device)

      torch.ops.aten._fused_adamw_.default(
          [p],
          [g],
          [ea],
          [eas],
          [],
          [step],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )
      return p, ea, eas

    self.assert_close_tpu_vs_cpu(run_fn)

  @parameterized.parameters(torch.float16, torch.float64)
  def test_fused_adamw_precision_dtypes(self, dtype):
    """Tests _fused_adamw_ on float16 and float64 dtypes."""

    def run_fn(device):
      p = torch.tensor([1.0, -1.0], dtype=dtype, device=device)
      g = torch.tensor([0.2, 0.4], dtype=dtype, device=device)
      ea = torch.tensor([0.01, 0.02], dtype=dtype, device=device)
      eas = torch.tensor([0.001, 0.002], dtype=dtype, device=device)
      step = torch.tensor(1.0, dtype=torch.float32, device=device)

      torch.ops.aten._fused_adamw_.default(
          [p],
          [g],
          [ea],
          [eas],
          [],
          [step],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )
      return p, ea, eas

    self.assert_close_tpu_vs_cpu(
        run_fn, atol=1e-3 if dtype == torch.float16 else 1e-5, rtol=1e-2
    )

  def test_fused_adamw_buffer_donation(self):
    """Tests MLIR buffer donation annotations and runtime buffer invalidation."""
    # TODO: b/576318333 - Re-enable once torch.compile of _fused_adamw_ passes
    # functionalization.
    self.skipTest("b/576318333: functionalization fails for _fused_adamw_.")
    debugs: list[tpu_compile.TpuCompileDebug] = []

    @torch.compile(
        fullgraph=True,
        backend="tpu",
        options={"serializable": False, "debug_callback": debugs.append},
    )
    def step_fn(params, grads, exp_avgs, exp_avg_sqs, state_steps, lr):
      torch.ops.aten._fused_adamw_.default(
          params,
          grads,
          exp_avgs,
          exp_avg_sqs,
          [],
          state_steps,
          lr=lr,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )

    p = torch.tensor([1.0, 2.0], dtype=torch.float32, device="tpu")
    g = torch.tensor([0.1, 0.2], dtype=torch.float32, device="tpu")
    ea = torch.tensor([0.01, 0.02], dtype=torch.float32, device="tpu")
    eas = torch.tensor([0.001, 0.002], dtype=torch.float32, device="tpu")
    step = torch.tensor([1], dtype=torch.int64, device="tpu")

    step_fn([p], [g], [ea], [eas], [step], 0.001)

    self.assertLen(debugs, 1)
    executables = debugs[0].compiled_executables
    self.assertLen(executables, 1)
    self.assertIn("jax.buffer_donor", debugs[0].stablehlo_forward_text[0])

    self.assertIsNotNone(p.cpu())


if __name__ == "__main__":
  absltest.main()
