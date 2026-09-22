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

"""Unit tests for max pooling monkey patch module."""

from unittest import mock
from absl.testing import absltest
import torch
import torch.nn.functional as F
from torch_tpu._internal.pooling import patch as pooling_patch


class PatchTest(absltest.TestCase):  # ABSLTEST_OK=deterministic mock test

  def tearDown(self):
    super().tearDown()
    pooling_patch.unpatch_max_pool()

  def _create_mock_tensor(self, device_type: str, requires_grad: bool = False):
    t = mock.MagicMock(spec=torch.Tensor)
    t.device = mock.MagicMock()
    t.device.type = device_type
    t.requires_grad = requires_grad
    return t

  def test_is_tpu_tensor(self):
    self.assertTrue(
        pooling_patch._is_tpu_tensor(self._create_mock_tensor("tpu"))
    )
    self.assertTrue(
        pooling_patch._is_tpu_tensor(self._create_mock_tensor("xla_cpu"))
    )
    self.assertFalse(
        pooling_patch._is_tpu_tensor(self._create_mock_tensor("cpu"))
    )
    self.assertFalse(
        pooling_patch._is_tpu_tensor(self._create_mock_tensor("cuda"))
    )
    self.assertFalse(pooling_patch._is_tpu_tensor(123))

  def test_patch_and_unpatch_lifecycle(self):
    orig_f1 = F.max_pool1d
    orig_f2 = F.max_pool2d
    orig_f3 = F.max_pool3d

    pooling_patch.patch_max_pool()
    self.assertIs(F.max_pool1d, pooling_patch.patched_max_pool1d)
    self.assertIs(F.max_pool2d, pooling_patch.patched_max_pool2d)
    self.assertIs(F.max_pool3d, pooling_patch.patched_max_pool3d)

    pooling_patch.unpatch_max_pool()
    self.assertIs(F.max_pool1d, orig_f1)
    self.assertIs(F.max_pool2d, orig_f2)
    self.assertIs(F.max_pool3d, orig_f3)

  def test_patched_max_pool2d_cpu_fallback(self):
    cpu_tensor = torch.randn(2, 4, 8, 8)
    out = pooling_patch.patched_max_pool2d(cpu_tensor, kernel_size=2, stride=2)
    self.assertEqual(out.shape, (2, 4, 4, 4))
    self.assertEqual(out.device.type, "cpu")

  def test_patched_max_pool2d_return_indices_fallback(self):
    cpu_tensor = torch.randn(2, 4, 8, 8)
    out, idx = pooling_patch.patched_max_pool2d(
        cpu_tensor, kernel_size=2, stride=2, return_indices=True
    )
    self.assertEqual(out.shape, (2, 4, 4, 4))
    self.assertEqual(idx.shape, (2, 4, 4, 4))

  def test_patched_max_pool2d_tpu_eval_routes_to_tpu_op(self):
    tpu_tensor = self._create_mock_tensor("tpu", requires_grad=False)
    with mock.patch.object(torch.ops.tpu, "max_pool2d") as mock_op:
      mock_op.return_value = "mock_tpu_result"
      res = pooling_patch.patched_max_pool2d(
          tpu_tensor, kernel_size=3, stride=2, padding=1
      )
      self.assertEqual(res, "mock_tpu_result")
      mock_op.assert_called_once_with(
          tpu_tensor, [3, 3], [2, 2], [1, 1], [1, 1], False
      )

  def test_patched_max_pool2d_tpu_train_routes_to_autograd_function(self):
    tpu_tensor = self._create_mock_tensor("tpu", requires_grad=True)
    with torch.enable_grad():
      with mock.patch.object(
          pooling_patch._TpuMaxPool2dFunction, "apply"
      ) as mock_fn:
        mock_fn.return_value = "mock_autograd_result"
        res = pooling_patch.patched_max_pool2d(
            tpu_tensor, kernel_size=3, stride=2, padding=1
        )
        self.assertEqual(res, "mock_autograd_result")
        mock_fn.assert_called_once_with(
            tpu_tensor, [3, 3], [2, 2], [1, 1], [1, 1], False
        )

  def test_patched_max_pool1d_tpu_routes_to_tpu_op(self):
    tpu_tensor = self._create_mock_tensor("tpu", requires_grad=False)
    with mock.patch.object(torch.ops.tpu, "max_pool1d") as mock_op:
      mock_op.return_value = "mock_tpu_result"
      res = pooling_patch.patched_max_pool1d(
          tpu_tensor, kernel_size=3, stride=2, padding=1
      )
      self.assertEqual(res, "mock_tpu_result")
      mock_op.assert_called_once_with(tpu_tensor, [3], [2], [1], [1], False)

  def test_patched_max_pool3d_tpu_routes_to_tpu_op(self):
    tpu_tensor = self._create_mock_tensor("tpu", requires_grad=False)
    with mock.patch.object(torch.ops.tpu, "max_pool3d") as mock_op:
      mock_op.return_value = "mock_tpu_result"
      res = pooling_patch.patched_max_pool3d(
          tpu_tensor, kernel_size=3, stride=2, padding=1
      )
      self.assertEqual(res, "mock_tpu_result")
      mock_op.assert_called_once_with(
          tpu_tensor, [3, 3, 3], [2, 2, 2], [1, 1, 1], [1, 1, 1], False
      )


if __name__ == "__main__":
  absltest.main()
