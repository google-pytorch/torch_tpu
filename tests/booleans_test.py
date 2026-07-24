# Copyright 2025 Google LLC
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

"""Tests on boolean-valued tensors, not suitable for fuzzing."""


from absl.testing import absltest
import torch


class AllTest(absltest.TestCase):
  """Tests torch.all(), including when using an out tensor."""

  def _run_test(self, bools: list[bool], out=False) -> torch.Tensor:
    in_cpu_tensor = torch.tensor(bools, dtype=torch.bool)
    in_tpu_tensor = in_cpu_tensor.to(torch.device("tpu"))
    if not out:
      out_tpu_tensor = torch.all(in_tpu_tensor)
    else:
      out_tpu_tensor = torch.zeros(
          (), dtype=torch.bool, device=torch.device("tpu")
      )
      torch.all(in_tpu_tensor, out=out_tpu_tensor)
    return out_tpu_tensor.to("cpu")

  def test_all_returns_true(self):
    bools = [True] * 128
    out_tpu_tensor = self._run_test(bools)
    self.assertEqual(out_tpu_tensor.shape, ())
    out_cpu_tensor = out_tpu_tensor.to("cpu")
    self.assertEqual(out_cpu_tensor.dtype, torch.bool)
    self.assertEqual(out_cpu_tensor.item(), True)

  def test_all_returns_true_empty(self):
    bools = []
    out_tpu_tensor = self._run_test(bools)
    self.assertEqual(out_tpu_tensor.shape, ())
    out_cpu_tensor = out_tpu_tensor.to("cpu")
    self.assertEqual(out_cpu_tensor.dtype, torch.bool)
    self.assertEqual(out_cpu_tensor.item(), True)

  def test_all_returns_false(self):
    bools = [True] * 128
    bools[42] = False
    out_tpu_tensor = self._run_test(bools)
    self.assertEqual(out_tpu_tensor.shape, ())
    out_cpu_tensor = out_tpu_tensor.to("cpu")
    self.assertEqual(out_cpu_tensor.dtype, torch.bool)
    self.assertEqual(out_cpu_tensor.item(), False)

  def test_all_all_out_returns_true(self):
    bools = [True] * 128
    out_tpu_tensor = self._run_test(bools, out=True)
    self.assertEqual(out_tpu_tensor.shape, ())
    out_cpu_tensor = out_tpu_tensor.to("cpu")
    self.assertEqual(out_cpu_tensor.dtype, torch.bool)
    self.assertEqual(out_cpu_tensor.item(), True)

  def test_all_all_out_returns_false(self):
    bools = [True] * 128
    bools[42] = False
    out_tpu_tensor = self._run_test(bools, out=True)
    self.assertEqual(out_tpu_tensor.shape, ())
    out_cpu_tensor = out_tpu_tensor.to("cpu")
    self.assertEqual(out_cpu_tensor.dtype, torch.bool)
    self.assertEqual(out_cpu_tensor.item(), False)

  def test_all_all_out_returns_true_empty(self):
    bools = []
    out_tpu_tensor = self._run_test(bools, out=True)
    self.assertEqual(out_tpu_tensor.shape, ())
    out_cpu_tensor = out_tpu_tensor.to("cpu")
    self.assertEqual(out_cpu_tensor.dtype, torch.bool)
    self.assertEqual(out_cpu_tensor.item(), True)


if __name__ == "__main__":
  absltest.main()
