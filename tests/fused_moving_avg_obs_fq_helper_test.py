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

"""Unit tests for _fused_moving_avg_obs_fq_helper in TorchTPU."""

from absl.testing import absltest
import torch
from tests import op_testing

TorchTpuVsCpuTestBase = op_testing.TorchTpuVsCpuTestBase


def _fused_moving_avg_obs_fq_helper_device_wrapper(
    device,
    self_t,
    observer_on,
    fake_quant_on,
    rmin,
    rmax,
    sc,
    zp,
    averaging_const,
    quant_min,
    quant_max,
    ch_axis,
    per_row,
    symmetric,
):
  s_t = self_t.clone().to(device)
  obs = observer_on.clone().to(device)
  fq = fake_quant_on.clone().to(device)
  rmin_dev = rmin.clone().to(device)
  rmax_dev = rmax.clone().to(device)
  sc_dev = sc.clone().to(device)
  zp_dev = zp.clone().to(device)

  if per_row and ch_axis != 0 and str(device).startswith("cpu"):
    # PyTorch CPU only supports ch_axis == 0 for per-channel fake quant.
    # We movedim ch_axis to 0, run with ch_axis=0, and movedim back.
    s_t_perm = s_t.movedim(ch_axis, 0).contiguous()
    out_p, mask_p = torch.ops.aten._fused_moving_avg_obs_fq_helper(
        s_t_perm,
        obs,
        fq,
        rmin_dev,
        rmax_dev,
        sc_dev,
        zp_dev,
        averaging_const,
        quant_min,
        quant_max,
        0,
        per_row,
        symmetric,
    )
    out = out_p.movedim(0, ch_axis).contiguous()
    mask = mask_p.movedim(0, ch_axis).contiguous()
  else:
    out, mask = torch.ops.aten._fused_moving_avg_obs_fq_helper(
        s_t,
        obs,
        fq,
        rmin_dev,
        rmax_dev,
        sc_dev,
        zp_dev,
        averaging_const,
        quant_min,
        quant_max,
        ch_axis,
        per_row,
        symmetric,
    )
  return out, mask, rmin_dev, rmax_dev, sc_dev, zp_dev


class FusedMovingAvgObsFqHelperTest(TorchTpuVsCpuTestBase):
  """Tests for aten::_fused_moving_avg_obs_fq_helper."""

  def assert_close_helper(self, run_fn, atol=1e-5, rtol=1e-5, zp_atol=1.0):
    cpu_res = run_fn("cpu")
    tpu_res = run_fn("tpu")
    self.assert_close(
        golden_result=cpu_res[0].cpu(),
        torch_tpu_result=tpu_res[0].cpu(),
        atol=atol,
        rtol=rtol,
    )
    self.assert_close(
        golden_result=cpu_res[1].cpu(),
        torch_tpu_result=tpu_res[1].cpu(),
    )
    self.assert_close(
        golden_result=cpu_res[2].cpu(),
        torch_tpu_result=tpu_res[2].cpu(),
        atol=atol,
        rtol=rtol,
    )
    self.assert_close(
        golden_result=cpu_res[3].cpu(),
        torch_tpu_result=tpu_res[3].cpu(),
        atol=atol,
        rtol=rtol,
    )
    self.assert_close(
        golden_result=cpu_res[4].cpu(),
        torch_tpu_result=tpu_res[4].cpu(),
        atol=atol,
        rtol=rtol,
    )
    self.assert_close(
        golden_result=cpu_res[5].cpu(),
        torch_tpu_result=tpu_res[5].cpu(),
        atol=zp_atol,
    )

  def test_fused_moving_avg_obs_fq_helper_case1_per_tensor_asymmetric(self):
    # Case 1: per-tensor, asymmetric, observer_on=1, fake_quant_on=1
    def run_fn(device):
      self_t = torch.ones((2, 3), dtype=torch.float32)
      observer_on = torch.tensor([1], dtype=torch.int32)
      fake_quant_on = torch.tensor([1], dtype=torch.int32)
      rmin = torch.tensor([-1.0], dtype=torch.float32)
      rmax = torch.tensor([1.0], dtype=torch.float32)
      sc = torch.tensor([0.01], dtype=torch.float32)
      zp = torch.tensor([0], dtype=torch.int32)
      return _fused_moving_avg_obs_fq_helper_device_wrapper(
          device,
          self_t,
          observer_on,
          fake_quant_on,
          rmin,
          rmax,
          sc,
          zp,
          0.01,
          0,
          255,
          0,
          False,
          False,
      )

    self.assert_close_helper(run_fn)

  def test_fused_moving_avg_obs_fq_helper_case2_per_tensor_symmetric(self):
    # Case 2: per-tensor, symmetric, observer_on=1, fake_quant_on=0
    def run_fn(device):
      self_t = torch.ones((2, 3), dtype=torch.float32)
      observer_on = torch.tensor([1], dtype=torch.int32)
      fake_quant_on = torch.tensor([0], dtype=torch.int32)
      rmin = torch.tensor([-1.0], dtype=torch.float32)
      rmax = torch.tensor([1.0], dtype=torch.float32)
      sc = torch.tensor([0.01], dtype=torch.float32)
      zp = torch.tensor([0], dtype=torch.int32)
      return _fused_moving_avg_obs_fq_helper_device_wrapper(
          device,
          self_t,
          observer_on,
          fake_quant_on,
          rmin,
          rmax,
          sc,
          zp,
          0.01,
          0,
          255,
          0,
          False,
          True,
      )

    self.assert_close_helper(run_fn)

  def test_fused_moving_avg_obs_fq_helper_case3_per_tensor_initialized_stats(
      self,
  ):
    # Case 3: per-tensor, initialized stats, observer_on=0, fake_quant_on=1
    def run_fn(device):
      self_t = torch.ones((2, 3), dtype=torch.float32)
      observer_on = torch.tensor([0], dtype=torch.int32)
      fake_quant_on = torch.tensor([1], dtype=torch.int32)
      rmin = torch.tensor([-5.1], dtype=torch.float32)
      rmax = torch.tensor([5.1], dtype=torch.float32)
      sc = torch.tensor([0.04], dtype=torch.float32)
      zp = torch.tensor([128], dtype=torch.int32)
      return _fused_moving_avg_obs_fq_helper_device_wrapper(
          device,
          self_t,
          observer_on,
          fake_quant_on,
          rmin,
          rmax,
          sc,
          zp,
          0.01,
          0,
          255,
          0,
          False,
          False,
      )

    self.assert_close_helper(run_fn)

  def test_fused_moving_avg_obs_fq_helper_case4_per_channel_axis_0(self):
    # Case 4: per-channel (ch_axis=0), asymmetric, observer=1, fake_quant=1
    def run_fn(device):
      self_t = torch.ones((2, 3), dtype=torch.float32)
      observer_on = torch.tensor([1], dtype=torch.int32)
      fake_quant_on = torch.tensor([1], dtype=torch.int32)
      rmin = torch.full((2,), float("inf"), dtype=torch.float32)
      rmax = torch.full((2,), float("-inf"), dtype=torch.float32)
      sc = torch.ones((2,), dtype=torch.float32)
      zp = torch.zeros((2,), dtype=torch.int32)
      return _fused_moving_avg_obs_fq_helper_device_wrapper(
          device,
          self_t,
          observer_on,
          fake_quant_on,
          rmin,
          rmax,
          sc,
          zp,
          0.01,
          0,
          255,
          0,
          True,
          False,
      )

    self.assert_close_helper(run_fn)

  def test_fused_moving_avg_obs_fq_helper_case5_per_channel_axis_1(self):
    # Case 5: per-channel (ch_axis=1), symmetric, observer=1, fake_quant=1
    def run_fn(device):
      self_t = torch.ones((2, 3), dtype=torch.float32)
      observer_on = torch.tensor([1], dtype=torch.int32)
      fake_quant_on = torch.tensor([1], dtype=torch.int32)
      rmin = torch.full((3,), float("inf"), dtype=torch.float32)
      rmax = torch.full((3,), float("-inf"), dtype=torch.float32)
      sc = torch.ones((3,), dtype=torch.float32)
      zp = torch.zeros((3,), dtype=torch.int32)
      return _fused_moving_avg_obs_fq_helper_device_wrapper(
          device,
          self_t,
          observer_on,
          fake_quant_on,
          rmin,
          rmax,
          sc,
          zp,
          0.01,
          -128,
          127,
          1,
          True,
          True,
      )

    self.assert_close_helper(run_fn)

  def test_fused_moving_avg_obs_fq_helper_case6_per_channel_axis_negative_1(
      self,
  ):
    # Case 6: per-channel (ch_axis=-1), asymmetric, observer=0, fake_quant=0
    def run_fn(device):
      self_t = torch.ones((2, 3), dtype=torch.float32)
      observer_on = torch.tensor([0], dtype=torch.int32)
      fake_quant_on = torch.tensor([0], dtype=torch.int32)
      rmin = torch.ones((3,), dtype=torch.float32)
      rmax = torch.ones((3,), dtype=torch.float32)
      sc = torch.ones((3,), dtype=torch.float32)
      zp = torch.zeros((3,), dtype=torch.int32)
      return _fused_moving_avg_obs_fq_helper_device_wrapper(
          device,
          self_t,
          observer_on,
          fake_quant_on,
          rmin,
          rmax,
          sc,
          zp,
          0.01,
          0,
          255,
          -1,
          True,
          False,
      )

    self.assert_close_helper(run_fn)

  def test_fused_moving_avg_obs_fq_helper_case7_empty_tensor(self):
    # Case 7: empty tensor
    def run_fn(device):
      empty_t = torch.empty((0, 3), dtype=torch.float32)
      observer_on = torch.tensor([0], dtype=torch.int32)
      fake_quant_on = torch.tensor([1], dtype=torch.int32)
      rmin = torch.tensor([-1.0], dtype=torch.float32)
      rmax = torch.tensor([1.0], dtype=torch.float32)
      sc = torch.tensor([0.01], dtype=torch.float32)
      zp = torch.tensor([0], dtype=torch.int32)
      return _fused_moving_avg_obs_fq_helper_device_wrapper(
          device,
          empty_t,
          observer_on,
          fake_quant_on,
          rmin,
          rmax,
          sc,
          zp,
          0.01,
          0,
          255,
          0,
          False,
          False,
      )

    self.assert_close_helper(run_fn)

  def test_fused_moving_avg_obs_fq_helper_empty_stats(self):
    # Case 8: uninitialized empty (0,) stats on TPU (testing TPU resize)
    # We run on TPU with empty stats, and compare with CPU run with inf/nan-like
    # stats.
    self_t = torch.ones((2, 3), dtype=torch.float32)

    # Golden run (on CPU) with uninitialized stats
    rmin_golden = torch.tensor(
        [float("inf")], dtype=torch.float32, device="cpu"
    )
    rmax_golden = torch.tensor(
        [float("-inf")], dtype=torch.float32, device="cpu"
    )
    sc_golden = torch.tensor([1.0], dtype=torch.float32, device="cpu")
    zp_golden = torch.tensor([0], dtype=torch.int32, device="cpu")

    out_g, mask_g = torch.ops.aten._fused_moving_avg_obs_fq_helper(
        self_t.to("cpu"),
        torch.tensor([1], dtype=torch.int32, device="cpu"),
        torch.tensor([1], dtype=torch.int32, device="cpu"),
        rmin_golden,
        rmax_golden,
        sc_golden,
        zp_golden,
        0.01,
        0,
        255,
        0,
        False,
        False,
    )

    # TPU run with empty stats
    tpu_dev = torch.device("tpu")
    rmin_tpu = torch.empty((0,), dtype=torch.float32, device=tpu_dev)
    rmax_tpu = torch.empty((0,), dtype=torch.float32, device=tpu_dev)
    sc_tpu = torch.empty((0,), dtype=torch.float32, device=tpu_dev)
    zp_tpu = torch.empty((0,), dtype=torch.int32, device=tpu_dev)

    out_t, mask_t = torch.ops.aten._fused_moving_avg_obs_fq_helper(
        self_t.to(tpu_dev),
        torch.tensor([1], dtype=torch.int32, device=tpu_dev),
        torch.tensor([1], dtype=torch.int32, device=tpu_dev),
        rmin_tpu,
        rmax_tpu,
        sc_tpu,
        zp_tpu,
        0.01,
        0,
        255,
        0,
        False,
        False,
    )

    # Compare
    self.assert_close(
        golden_result=out_g,
        torch_tpu_result=out_t.to("cpu"),
        atol=1e-5,
        rtol=1e-5,
    )
    self.assert_close(
        golden_result=mask_g,
        torch_tpu_result=mask_t.to("cpu"),
        atol=1e-5,
        rtol=1e-5,
    )
    self.assert_close(
        golden_result=rmin_golden,
        torch_tpu_result=rmin_tpu.to("cpu"),
        atol=1e-5,
        rtol=1e-5,
    )
    self.assert_close(
        golden_result=rmax_golden,
        torch_tpu_result=rmax_tpu.to("cpu"),
        atol=1e-5,
        rtol=1e-5,
    )
    self.assert_close(
        golden_result=sc_golden,
        torch_tpu_result=sc_tpu.to("cpu"),
        atol=1e-5,
        rtol=1e-5,
    )
    self.assert_close(
        golden_result=zp_golden,
        torch_tpu_result=zp_tpu.to("cpu"),
        atol=1e-5,
        rtol=1e-5,
    )


if __name__ == "__main__":
  absltest.main()
