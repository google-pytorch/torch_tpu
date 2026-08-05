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

"""Tests that autoload is working for the tpu backend."""

from absl.testing import absltest
import torch
import torch.accelerator


class AutoloadTpuTest(absltest.TestCase):  # ABSLTEST_OK=Autoload test

  def test_current_accelerator(self):
    self.assertEqual(
        torch.device("tpu"), torch.accelerator.current_accelerator()
    )

  def test_torch_dot_tpu(self):
    import torch.tpu as module  # pylint: disable=g-import-not-at-top

    self.assertTrue(module.is_available())

  def test_torch_tpu_device(self):
    device = torch.device("tpu")
    self.assertEqual(device.type, "tpu")

  def test_tensor_to_tpu(self):
    tensor = torch.empty(1, 1)
    tensor.to("tpu")


if __name__ == "__main__":
  absltest.main()
