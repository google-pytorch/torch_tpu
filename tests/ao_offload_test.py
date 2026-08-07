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

from absl.testing import absltest
import torch
from torch._functorch._activation_offloading import offload_ops
from torch_tpu._internal.testing import get_memory_kind
from torch_tpu._internal.utils import test_utils

# Reference offload_ops to prevent a lint warning. The import is required to
# register the offload custom ops.
_ = offload_ops


class AoOffloadTest(absltest.TestCase):

  def test_offload(self):
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = torch.tensor(data, device="tpu", dtype=torch.float32)
    self.assertEqual(get_memory_kind(x), "device")

    y = torch.ops.ao.offload(x)
    self.assertEqual(get_memory_kind(y), "pinned_host")

    expected = torch.tensor(data, dtype=torch.float32)
    test_utils.assert_close(y.cpu(), expected)

  def test_reload(self):
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = torch.tensor(data, device="tpu", dtype=torch.float32).flatten()
    offloaded = torch.ops.ao.offload(x)
    self.assertEqual(get_memory_kind(offloaded), "pinned_host")

    y = torch.ops.ao.reload(
        offloaded,
        torch.device("tpu"),
        [2, 3],  # original_size
        [1, 2],  # original_stride
    )
    self.assertEqual(list(y.shape), [2, 3])
    self.assertEqual(list(y.stride()), [1, 2])
    self.assertEqual(get_memory_kind(y), "device")

    expected = torch.tensor(data, dtype=torch.float32)
    test_utils.assert_close(y.cpu(), expected)

  def test_wait_tensor(self):
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = torch.tensor(data, device="tpu", dtype=torch.float32)
    self.assertEqual(get_memory_kind(x), "device")

    y = torch.ops.ao.wait_tensor(x)
    self.assertEqual(x.data_ptr(), y.data_ptr())
    self.assertEqual(get_memory_kind(y), "device")

    expected = torch.tensor(data, dtype=torch.float32)
    test_utils.assert_close(y.cpu(), expected)


if __name__ == "__main__":
  absltest.main()
