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

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class SetTest(seed_test_utils.RepeatableTest):
  """Tests the Tensor.set_() operator."""

  def test_set_none(self):
    """Setting None assigns an empty storage."""
    tensor = torch.arange(16, dtype=torch.float32, device='tpu').reshape(8, 2)
    tensor.set_()
    expected = torch.tensor([], dtype=torch.float32)
    utils.assert_close(tensor.to('cpu'), expected)
    self.assertEqual(tensor.size(), expected.size())

  def test_set_tensor(self):
    """Setting a tensor assigns that tensor's storage and metadata."""
    tensor = torch.empty(1, dtype=torch.float32, device='tpu')
    source = torch.arange(16, dtype=torch.float32, device='tpu').reshape(8, 2)
    tensor.set_(source)
    utils.assert_close(tensor.to('cpu'), source.to('cpu'))

  def test_set_tensor_self_assignment(self):
    base = torch.arange(16, dtype=torch.float32, device='tpu').reshape(4, 4)
    view = base[1:, ::2]
    expected_ptr = view.data_ptr()
    result = view.set_(view)
    self.assertIs(result, view)
    self.assertEqual(view.data_ptr(), expected_ptr)
    self.assertEqual(view.size(), (3, 2))
    self.assertEqual(view.stride(), (4, 2))
    self.assertEqual(view.storage_offset(), 4)
    utils.assert_close(view.to('cpu'), base.to('cpu')[1:, ::2])

  def test_set_tensor_shared_storage_view(self):
    base = torch.arange(16, dtype=torch.float32, device='tpu').reshape(4, 4)
    tensor = base[:2, :2]
    source = base[1:, ::2]
    tensor.set_(source)
    self.assertEqual(tensor.size(), source.size())
    self.assertEqual(tensor.stride(), source.stride())
    self.assertEqual(tensor.storage_offset(), source.storage_offset())
    utils.assert_close(tensor.to('cpu'), source.to('cpu'))

  def test_set_tensor_from_zero_sized_self_storage(self):
    tensor = torch.zeros(8, dtype=torch.float32, device='tpu')
    tensor.untyped_storage().resize_(0)
    source = torch.arange(8, dtype=torch.float32, device='tpu')
    tensor.set_(source)
    utils.assert_close(tensor.to('cpu'), source.to('cpu'))

  def test_set_tensor_fp4(self):
    source_cpu = (
        torch.arange(12, dtype=torch.uint8)
        .reshape(3, 4)
        .view(torch.float4_e2m1fn_x2)
    )
    source_tpu = source_cpu.to('tpu')

    dst_contiguous = torch.empty(
        (1, 2), dtype=torch.float4_e2m1fn_x2, device='tpu'
    )
    dst_contiguous.set_(source_tpu)
    self.assertEqual(dst_contiguous.size(), (3, 4))
    self.assertEqual(dst_contiguous.stride(), (4, 1))
    self.assertEqual(dst_contiguous.storage_offset(), 0)
    utils.assert_close(dst_contiguous.to('cpu'), source_cpu)

    source_view = source_tpu[1:]
    dst_view = torch.empty((1, 2), dtype=torch.float4_e2m1fn_x2, device='tpu')
    dst_view.set_(source_view)
    self.assertEqual(dst_view.size(), (2, 4))
    self.assertEqual(dst_view.stride(), (4, 1))
    self.assertEqual(dst_view.storage_offset(), 4)
    utils.assert_close(dst_view.to('cpu'), source_cpu[1:])

  def test_set_storage(self):
    """Setting storage defaults to contiguous row-major metadata."""
    tensor = torch.empty(1, dtype=torch.float32, device='tpu')
    source_flat = torch.arange(16, dtype=torch.float32, device='tpu')
    source = source_flat.reshape(8, 2)
    tensor.set_(source.untyped_storage())
    utils.assert_close(tensor.to('cpu'), source_flat.to('cpu'))

  def test_set_storage_with_offset_and_stride(self):
    """If specified, offset/size/stride are copied over to the new tensor."""
    tensor = torch.empty(1, dtype=torch.float32, device='tpu')
    source_flat = torch.arange(16, dtype=torch.float32, device='tpu')
    source = source_flat.reshape((4, 4))[1:, ::2]
    tensor.set_(
        source_flat.untyped_storage(),
        storage_offset=source.storage_offset(),
        size=source.size(),
        stride=source.stride(),
    )
    utils.assert_close(tensor.to('cpu'), source.to('cpu'))


if __name__ == '__main__':
  absltest.main()
