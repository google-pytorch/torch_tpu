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
from torch_tpu._internal.utils import utils


class SetTest(absltest.TestCase):
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
