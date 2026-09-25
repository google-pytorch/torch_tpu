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

"""Unit tests for _fused_adagrad_ and _fused_adagrad_.tensor_lr in TorchTPU."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from tests import op_testing

TorchTpuVsCpuTestBase = op_testing.TorchTpuVsCpuTestBase


class FusedAdagradTest(TorchTpuVsCpuTestBase):
  """Tests for aten::_fused_adagrad_ and aten::_fused_adagrad_.tensor_lr."""

  def _get_tols(self, dtype):
    if dtype == torch.bfloat16:
      return 1e-2, 1e-2
    elif dtype == torch.float64:
      return 1e-5, 1e-5
    else:
      return 1e-4, 1e-4

  @parameterized.product(
      use_tensor_lr=[False, True],
      dtype=[torch.float32, torch.bfloat16, torch.float64],
      maximize=[False, True],
      lr_decay=[0.0, 0.1],
      weight_decay=[0.0, 0.01],
      eps=[1e-10, 1e-5],
  )
  def test_fused_adagrad_basic(
      self, use_tensor_lr, dtype, maximize, lr_decay, weight_decay, eps
  ):
    def run_fn(device):
      p1 = torch.tensor([1.0, 2.0, -1.0], dtype=dtype, device=device)
      p2 = torch.tensor([[0.5, -0.5], [1.5, 0.0]], dtype=dtype, device=device)
      g1 = torch.tensor([0.1, -0.2, 0.5], dtype=dtype, device=device)
      g2 = torch.tensor([[-0.1, 0.2], [0.3, -0.4]], dtype=dtype, device=device)
      v1 = torch.tensor([0.01, 0.02, 0.0], dtype=dtype, device=device)
      v2 = torch.tensor(
          [[0.0, 0.01], [-0.01, 0.02]], dtype=dtype, device=device
      )
      s1 = torch.tensor(1.0, dtype=torch.float32, device=device)
      s2 = torch.tensor(5.0, dtype=torch.float32, device=device)

      if use_tensor_lr:
        lr = torch.tensor(0.01, dtype=torch.float32, device=device)
        op = torch.ops.aten._fused_adagrad_.tensor_lr
      else:
        lr = 0.01
        op = torch.ops.aten._fused_adagrad_.default

      op(
          [p1, p2],
          [g1, g2],
          [v1, v2],
          [s1, s2],
          lr=lr,
          lr_decay=lr_decay,
          weight_decay=weight_decay,
          eps=eps,
          maximize=maximize,
      )
      return p1, p2, v1, v2, s1, s2

    rtol, atol = self._get_tols(dtype)
    self.assert_close_tpu_vs_cpu(run_fn, rtol=rtol, atol=atol)

  @parameterized.product(
      use_tensor_lr=[False, True],
      dtype=[torch.float32, torch.bfloat16, torch.float64],
      maximize=[False, True],
      grad_scale_val=[2.0, 0.5],
  )
  def test_fused_adagrad_grad_scale(
      self, use_tensor_lr, dtype, maximize, grad_scale_val
  ):
    def run_fn(device):
      p1 = torch.tensor([1.0, 2.0, -1.0], dtype=dtype, device=device)
      p2 = torch.tensor([[0.5, -0.5], [1.5, 0.0]], dtype=dtype, device=device)
      g1 = torch.tensor([0.2, -0.4, 0.8], dtype=dtype, device=device)
      g2 = torch.tensor([[-0.2, 0.6], [0.4, -0.8]], dtype=dtype, device=device)
      v1 = torch.tensor([0.01, 0.02, 0.0], dtype=dtype, device=device)
      v2 = torch.tensor(
          [[0.0, 0.01], [-0.01, 0.02]], dtype=dtype, device=device
      )
      s1 = torch.tensor(1.0, dtype=torch.float32, device=device)
      s2 = torch.tensor(2.0, dtype=torch.float32, device=device)

      if use_tensor_lr:
        lr = torch.tensor(0.01, dtype=torch.float32, device=device)
        op = torch.ops.aten._fused_adagrad_.tensor_lr
      else:
        lr = 0.01
        op = torch.ops.aten._fused_adagrad_.default

      grad_scale = torch.tensor(
          grad_scale_val, dtype=torch.float32, device=device
      )

      op(
          [p1, p2],
          [g1, g2],
          [v1, v2],
          [s1, s2],
          lr=lr,
          lr_decay=0.1,
          weight_decay=0.01,
          eps=1e-10,
          maximize=maximize,
          grad_scale=grad_scale,
      )
      return p1, p2, g1, g2, v1, v2

    rtol, atol = self._get_tols(dtype)
    self.assert_close_tpu_vs_cpu(run_fn, rtol=rtol, atol=atol)

  @parameterized.product(
      use_tensor_lr=[False, True],
      dtype=[torch.float32, torch.bfloat16, torch.float64],
      maximize=[False, True],
      found_inf_val=[0.0, 1.0],
      has_grad_scale=[False, True],
  )
  def test_fused_adagrad_found_inf_revert(
      self, use_tensor_lr, dtype, maximize, found_inf_val, has_grad_scale
  ):
    def run_fn(device):
      p1 = torch.tensor([1.0, 2.0, -1.0], dtype=dtype, device=device)
      p2 = torch.tensor([[0.5, -0.5], [1.5, 0.0]], dtype=dtype, device=device)
      g1 = torch.tensor([0.2, -0.4, 0.8], dtype=dtype, device=device)
      g2 = torch.tensor([[-0.2, 0.6], [0.4, -0.8]], dtype=dtype, device=device)
      v1 = torch.tensor([0.01, 0.02, 0.0], dtype=dtype, device=device)
      v2 = torch.tensor(
          [[0.0, 0.01], [-0.01, 0.02]], dtype=dtype, device=device
      )
      s1 = torch.tensor(1.0, dtype=torch.float32, device=device)
      s2 = torch.tensor(2.0, dtype=torch.float32, device=device)

      if use_tensor_lr:
        lr = torch.tensor(0.01, dtype=torch.float32, device=device)
        op = torch.ops.aten._fused_adagrad_.tensor_lr
      else:
        lr = 0.01
        op = torch.ops.aten._fused_adagrad_.default

      found_inf = torch.tensor(
          found_inf_val, dtype=torch.float32, device=device
      )
      grad_scale = (
          torch.tensor(2.0, dtype=torch.float32, device=device)
          if has_grad_scale
          else None
      )

      op(
          [p1, p2],
          [g1, g2],
          [v1, v2],
          [s1, s2],
          lr=lr,
          lr_decay=0.0,
          weight_decay=0.01,
          eps=1e-10,
          maximize=maximize,
          grad_scale=grad_scale,
          found_inf=found_inf,
      )
      return p1, p2, g1, g2, v1, v2

    rtol, atol = self._get_tols(dtype)
    self.assert_close_tpu_vs_cpu(run_fn, rtol=rtol, atol=atol)

  @parameterized.product(
      use_tensor_lr=[False, True],
      dtype=[torch.float32, torch.bfloat16, torch.float64],
      maximize=[False, True],
  )
  def test_fused_adagrad_found_inf_with_nans_reverts_to_initial(
      self, use_tensor_lr, dtype, maximize
  ):
    device = "tpu"
    p1_init = torch.tensor([1.0, 2.0, -1.0], dtype=dtype, device=device)
    p1 = p1_init.clone()
    g1_init = torch.tensor(
        [float("inf"), float("nan"), 0.5], dtype=dtype, device=device
    )
    g1 = g1_init.clone()
    v1_init = torch.tensor([0.01, 0.02, 0.0], dtype=dtype, device=device)
    v1 = v1_init.clone()
    s1 = torch.tensor(1.0, dtype=torch.float32, device=device)

    found_inf = torch.tensor(1.0, dtype=torch.float32, device=device)

    if use_tensor_lr:
      lr = torch.tensor(0.01, dtype=torch.float32, device=device)
      op = torch.ops.aten._fused_adagrad_.tensor_lr
    else:
      lr = 0.01
      op = torch.ops.aten._fused_adagrad_.default

    op(
        [p1],
        [g1],
        [v1],
        [s1],
        lr=lr,
        lr_decay=0.1,
        weight_decay=0.01,
        eps=1e-10,
        maximize=maximize,
        found_inf=found_inf,
    )

    self.assert_close(golden_result=p1_init, torch_tpu_result=p1)
    self.assert_close(golden_result=g1_init, torch_tpu_result=g1)
    self.assert_close(golden_result=v1_init, torch_tpu_result=v1)


if __name__ == "__main__":
  absltest.main()
