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

"""Test that untyped storage can be accessed and used as expected."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.accelerator
from torch_tpu._internal.utils import utils


class UntypedStorageTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tpu = torch.accelerator.current_accelerator()

  @parameterized.named_parameters(
      ('float32', torch.float32),
      ('bfloat16', torch.bfloat16),
      ('int64', torch.int64),
  )
  def test_resize_to_zero_and_back(self, dtype):
    """Checks behavior of untyped_storage().resize_() going to 0 and back.

    This effectively erases the data but preserve view relationships.
    """
    tensor = torch.arange(8, device=self.tpu, dtype=dtype)
    view = tensor.reshape(4, 2)
    initial_numel = tensor.numel()
    initial_bytes = tensor.untyped_storage().nbytes()
    tensor.untyped_storage().resize_(0)
    self.assertEqual(tensor.untyped_storage().size(), 0)
    self.assertEqual(view.untyped_storage().size(), 0)

    tensor.untyped_storage().resize_(initial_bytes)
    self.assertEqual(tensor.untyped_storage().size(), initial_bytes)
    self.assertEqual(view.untyped_storage().size(), initial_bytes)
    self.assertEqual(tensor.numel(), initial_numel)
    self.assertEqual(view.numel(), initial_numel)

    # Checks that the tensor can be materialized.
    # It's placeholder memory so the values themselves are unimportant.
    tensor.to('cpu')
    view.to('cpu')

    # The view relationship is preserved -- writing to one tensor writes
    # to the view as well.
    fill_value = 17
    tensor.fill_(fill_value)
    expected = torch.full_like(tensor, fill_value, device='cpu')
    expected_view = torch.full_like(view, fill_value, device='cpu')
    utils.assert_close(tensor.to('cpu'), expected)
    utils.assert_close(view.to('cpu'), expected_view)

  @parameterized.named_parameters(
      ('float32', torch.float32),
      ('bfloat16', torch.bfloat16),
      ('int64', torch.int64),
  )
  def test_resize_smaller_and_back(self, dtype):
    """Checks behavior of untyped_storage().resize_() going to a nonzero value and back.

    This erases part of the data while preserving view relationships.
    """
    init_tensor_cpu = torch.arange(8, dtype=dtype)
    init_view_cpu = init_tensor_cpu.reshape(4, 2)
    tensor = init_tensor_cpu.to('tpu')
    view = tensor.reshape(4, 2)
    initial_numel = tensor.numel()
    initial_bytes = tensor.untyped_storage().nbytes()

    # Removes half of the memory. The unaffected data is in the first half
    # of memory: tensor[:4] and view[:2, :].
    tensor.untyped_storage().resize_(initial_bytes // 2)
    self.assertEqual(tensor.untyped_storage().size(), initial_bytes // 2)
    self.assertEqual(view.untyped_storage().size(), initial_bytes // 2)
    # The remaining half of the data can be materialized.
    utils.assert_close(tensor[:4].to('cpu'), init_tensor_cpu[:4])
    utils.assert_close(view[:2, :].to('cpu'), init_view_cpu[:2, :])

    # Resizes to the original size.
    tensor.untyped_storage().resize_(initial_bytes)
    self.assertEqual(tensor.untyped_storage().size(), initial_bytes)
    self.assertEqual(view.untyped_storage().size(), initial_bytes)
    self.assertEqual(tensor.numel(), initial_numel)
    self.assertEqual(view.numel(), initial_numel)

    # The full tensor can now be materialized but this only checks the data
    # in the first half. The rest has arbitrary placeholder values.
    utils.assert_close(tensor.to('cpu')[:4], init_tensor_cpu[:4])
    self.assertNotEqual(tensor.to('cpu')[5], init_tensor_cpu[5])
    utils.assert_close(view.to('cpu')[:2, :], init_view_cpu[:2, :])
    self.assertNotEqual(view.to('cpu')[3, 1], init_view_cpu[3, 1])

    # Writes a new value to the erased half and checks that the view
    # relationship is preserved.
    fill_value = 17
    tensor[4:].fill_(fill_value)
    expected_final_tensor = torch.concat(
        [init_tensor_cpu[:4], torch.full((4,), fill_value, dtype=dtype)]
    )
    utils.assert_close(tensor.to('cpu'), expected_final_tensor)
    utils.assert_close(view.to('cpu'), expected_final_tensor.reshape(4, 2))


if __name__ == '__main__':
  absltest.main()
