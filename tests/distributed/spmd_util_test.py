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

from unittest import mock
from absl.testing import absltest
import torch
from torch_tpu._internal.distributed import spmd_util


class SpmdSafeTest(absltest.TestCase):

  @mock.patch("torch_tpu._internal.sync.sync.synchronize")
  def test_spmd_safe_decorator(self, mock_sync):
    @spmd_util.spmd_safe
    def my_func(a, b):
      return a + b

    t1 = torch.randn(2, 2).to("tpu")
    t2 = torch.randn(2, 2).to("tpu")

    res = my_func(t1, t2)

    # Should be called twice: once for inputs, once for results.
    self.assertEqual(mock_sync.call_count, 2)

    # Check first call (inputs)
    # The first call should be with the list of input tensors and wait=False.
    args, kwargs = mock_sync.call_args_list[0]
    self.assertTrue(any(t is t1 for t in args[0]))
    self.assertTrue(any(t is t2 for t in args[0]))
    self.assertFalse(kwargs.get("wait", True))

    # Check second call (outputs)
    # The second call should be with the list containing the result tensor and
    # wait=False.
    args, kwargs = mock_sync.call_args_list[1]
    self.assertTrue(any(t is res for t in args[0]))
    self.assertFalse(kwargs.get("wait", True))


if __name__ == "__main__":
  absltest.main()
