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

"""Unit tests for _fused_sgd_ and _fused_sgd_.tensor_lr in TorchTPU."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from tests import op_testing

TorchTpuVsCpuTestBase = op_testing.TorchTpuVsCpuTestBase


class FusedSgdTest(TorchTpuVsCpuTestBase):
  """Tests for aten::_fused_sgd_ and aten::_fused_sgd_.tensor_lr."""

  @parameterized.product(
      maximize=[False, True],
      nesterov=[False, True],
      weight_decay=[0.0, 0.01],
      momentum=[0.0, 0.9],
      dampening=[0.0, 0.1],
      is_first_step=[False, True],
  )
  def test_fused_sgd_default(
      self, maximize, nesterov, weight_decay, momentum, dampening, is_first_step
  ):
    if nesterov and (momentum <= 0 or dampening != 0):
      return  # Nesterov requires momentum > 0 and zero dampening

    def run_fn(device):
      p1 = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32, device=device)
      p2 = torch.tensor(
          [[0.5, -0.5], [1.5, 0.0]], dtype=torch.float32, device=device
      )
      g1 = torch.tensor([0.1, -0.2, 0.5], dtype=torch.float32, device=device)
      g2 = torch.tensor(
          [[-0.1, 0.2], [0.3, -0.4]], dtype=torch.float32, device=device
      )

      mb_list = []
      if momentum != 0.0:
        mb1 = torch.tensor(
            [0.01, 0.02, 0.0], dtype=torch.float32, device=device
        )
        mb2 = torch.tensor(
            [[0.0, 0.01], [-0.01, 0.02]], dtype=torch.float32, device=device
        )
        mb_list = [mb1, mb2]

      lr = 0.01

      torch.ops.aten._fused_sgd_.default(
          [p1, p2],
          [g1, g2],
          mb_list,
          weight_decay=weight_decay,
          momentum=momentum,
          lr=lr,
          dampening=dampening,
          nesterov=nesterov,
          maximize=maximize,
          is_first_step=is_first_step,
      )

      if momentum != 0.0:
        return p1, p2, mb_list[0], mb_list[1]
      return p1, p2

    self.assert_close_tpu_vs_cpu(run_fn)

  @parameterized.product(
      maximize=[False, True],
      nesterov=[False, True],
      weight_decay=[0.0, 0.01],
      momentum=[0.0, 0.9],
      dampening=[0.0, 0.1],
      is_first_step=[False, True],
  )
  def test_fused_sgd_tensor_lr(
      self, maximize, nesterov, weight_decay, momentum, dampening, is_first_step
  ):
    if nesterov and (momentum <= 0 or dampening != 0):
      return  # Nesterov requires momentum > 0 and zero dampening

    def run_fn(device):
      p1 = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32, device=device)
      p2 = torch.tensor(
          [[0.5, -0.5], [1.5, 0.0]], dtype=torch.float32, device=device
      )
      g1 = torch.tensor([0.1, -0.2, 0.5], dtype=torch.float32, device=device)
      g2 = torch.tensor(
          [[-0.1, 0.2], [0.3, -0.4]], dtype=torch.float32, device=device
      )

      mb_list = []
      if momentum != 0.0:
        mb1 = torch.tensor(
            [0.01, 0.02, 0.0], dtype=torch.float32, device=device
        )
        mb2 = torch.tensor(
            [[0.0, 0.01], [-0.01, 0.02]], dtype=torch.float32, device=device
        )
        mb_list = [mb1, mb2]

      lr = torch.tensor(0.01, dtype=torch.float32, device=device)

      torch.ops.aten._fused_sgd_.tensor_lr(
          [p1, p2],
          [g1, g2],
          mb_list,
          weight_decay=weight_decay,
          momentum=momentum,
          lr=lr,
          dampening=dampening,
          nesterov=nesterov,
          maximize=maximize,
          is_first_step=is_first_step,
      )

      if momentum != 0.0:
        return p1, p2, mb_list[0], mb_list[1]
      return p1, p2

    self.assert_close_tpu_vs_cpu(run_fn)

  @parameterized.product(
      maximize=[False, True],
      momentum=[0.0, 0.9],
  )
  def test_fused_sgd_grad_scale(self, maximize, momentum):
    def run_fn(device):
      p1 = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32, device=device)
      p2 = torch.tensor(
          [[0.5, -0.5], [1.5, 0.0]], dtype=torch.float32, device=device
      )
      g1 = torch.tensor([0.2, -0.4, 0.8], dtype=torch.float32, device=device)
      g2 = torch.tensor(
          [[-0.2, 0.6], [0.4, -0.8]], dtype=torch.float32, device=device
      )

      mb_list = []
      if momentum != 0.0:
        mb1 = torch.tensor(
            [0.01, 0.02, 0.0], dtype=torch.float32, device=device
        )
        mb2 = torch.tensor(
            [[0.0, 0.01], [-0.01, 0.02]], dtype=torch.float32, device=device
        )
        mb_list = [mb1, mb2]

      lr = 0.01
      grad_scale = torch.tensor(2.0, dtype=torch.float32, device=device)

      torch.ops.aten._fused_sgd_.default(
          [p1, p2],
          [g1, g2],
          mb_list,
          weight_decay=0.01,
          momentum=momentum,
          lr=lr,
          dampening=0.0,
          nesterov=False,
          maximize=maximize,
          is_first_step=False,
          grad_scale=grad_scale,
      )

      if momentum != 0.0:
        return p1, p2, g1, g2, mb_list[0], mb_list[1]
      return p1, p2, g1, g2

    self.assert_close_tpu_vs_cpu(run_fn)

  @parameterized.product(
      found_inf_val=[0.0, 1.0],
      momentum=[0.0, 0.9],
  )
  def test_fused_sgd_found_inf_revert(self, found_inf_val, momentum):
    def run_fn(device):
      p1 = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32, device=device)
      p2 = torch.tensor(
          [[0.5, -0.5], [1.5, 0.0]], dtype=torch.float32, device=device
      )
      g1 = torch.tensor([0.2, -0.4, 0.8], dtype=torch.float32, device=device)
      g2 = torch.tensor(
          [[-0.2, 0.6], [0.4, -0.8]], dtype=torch.float32, device=device
      )

      mb_list = []
      if momentum != 0.0:
        mb1 = torch.tensor(
            [0.01, 0.02, 0.0], dtype=torch.float32, device=device
        )
        mb2 = torch.tensor(
            [[0.0, 0.01], [-0.01, 0.02]], dtype=torch.float32, device=device
        )
        mb_list = [mb1, mb2]

      lr = 0.01
      found_inf = torch.tensor(
          found_inf_val, dtype=torch.float32, device=device
      )

      torch.ops.aten._fused_sgd_.default(
          [p1, p2],
          [g1, g2],
          mb_list,
          weight_decay=0.01,
          momentum=momentum,
          lr=lr,
          dampening=0.0,
          nesterov=False,
          maximize=False,
          is_first_step=False,
          found_inf=found_inf,
      )

      if momentum != 0.0:
        return p1, p2, g1, g2, mb_list[0], mb_list[1]
      return p1, p2, g1, g2

    self.assert_close_tpu_vs_cpu(run_fn)

  @parameterized.product(momentum=[0.0, 0.9])
  def test_fused_sgd_found_inf_with_nans_reverts_to_initial(self, momentum):
    device = "tpu"
    p1_init = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32, device=device)
    p1 = p1_init.clone()
    g1_init = torch.tensor(
        [float("inf"), float("nan"), 0.5], dtype=torch.float32, device=device
    )
    g1 = g1_init.clone()

    mb_list = []
    mb_list_init = []
    if momentum != 0.0:
      mb1_init = torch.tensor(
          [0.01, 0.02, 0.0], dtype=torch.float32, device=device
      )
      mb_list_init = [mb1_init]
      mb_list = [mb1_init.clone()]

    found_inf = torch.tensor(1.0, dtype=torch.float32, device=device)

    torch.ops.aten._fused_sgd_.default(
        [p1],
        [g1],
        mb_list,
        weight_decay=0.01,
        momentum=momentum,
        lr=0.01,
        dampening=0.0,
        nesterov=False,
        maximize=False,
        is_first_step=False,
        found_inf=found_inf,
    )

    self.assert_close(golden_result=p1_init, torch_tpu_result=p1)
    self.assert_close(golden_result=g1_init, torch_tpu_result=g1)
    if momentum != 0.0:
      self.assert_close(
          golden_result=mb_list_init[0], torch_tpu_result=mb_list[0]
      )

  @parameterized.product(
      dtype=[torch.float32, torch.bfloat16, torch.float64],
      momentum=[0.0, 0.9],
      maximize=[False, True],
  )
  def test_fused_sgd_multi_precision(self, dtype, momentum, maximize):
    def run_fn(device):
      p1 = torch.tensor([1.0, 2.0, -1.0], dtype=dtype, device=device)
      p2 = torch.tensor([[0.5, -0.5], [1.5, 0.0]], dtype=dtype, device=device)
      g1 = torch.tensor([0.1, -0.2, 0.5], dtype=dtype, device=device)
      g2 = torch.tensor([[-0.1, 0.2], [0.3, -0.4]], dtype=dtype, device=device)

      mb_list = []
      if momentum != 0.0:
        mb1 = torch.tensor([0.01, 0.02, 0.0], dtype=dtype, device=device)
        mb2 = torch.tensor(
            [[0.0, 0.01], [-0.01, 0.02]], dtype=dtype, device=device
        )
        mb_list = [mb1, mb2]

      lr = 0.01

      torch.ops.aten._fused_sgd_.default(
          [p1, p2],
          [g1, g2],
          mb_list,
          weight_decay=0.01,
          momentum=momentum,
          lr=lr,
          dampening=0.0,
          nesterov=False,
          maximize=maximize,
          is_first_step=False,
      )

      if momentum != 0.0:
        return p1, p2, mb_list[0], mb_list[1]
      return p1, p2

    rtol = (
        1e-2
        if dtype == torch.bfloat16
        else (1e-5 if dtype == torch.float64 else 1e-4)
    )
    atol = (
        1e-2
        if dtype == torch.bfloat16
        else (1e-5 if dtype == torch.float64 else 1e-4)
    )
    self.assert_close_tpu_vs_cpu(run_fn, rtol=rtol, atol=atol)


if __name__ == "__main__":
  absltest.main()
