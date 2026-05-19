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
from examples.benchmarks.ops.op_input_loader import deserialize_args


class OpInputLoaderTest(absltest.TestCase):

  def test_deserialize_args(self):
    inputs_str = (
        "([T([2, 3], torch.float32), 5], {'alpha': T([4], torch.int64)})"
    )

    # Use 'cpu' for unit testing to avoid requiring TPU hardware
    args, kwargs = deserialize_args(inputs_str, device="cpu")

    self.assertEqual(len(args), 2)
    self.assertTrue(isinstance(args[0], torch.Tensor))
    self.assertEqual(args[0].shape, torch.Size([2, 3]))
    self.assertEqual(args[0].dtype, torch.float32)
    self.assertEqual(args[0].device.type, "cpu")
    self.assertEqual(args[1], 5)

    self.assertEqual(len(kwargs), 1)
    self.assertTrue(isinstance(kwargs["alpha"], torch.Tensor))
    self.assertEqual(kwargs["alpha"].shape, torch.Size([4]))
    self.assertEqual(kwargs["alpha"].dtype, torch.int64)
    self.assertEqual(kwargs["alpha"].device.type, "cpu")

  def test_random_data(self):
    inputs_str = "([T([100], torch.float32)], {})"
    args, _ = deserialize_args(inputs_str, device="cpu")
    t = args[0]
    # Verify it's not all zeros or ones (implying random data was generated)
    self.assertFalse(torch.all(t == 0))
    self.assertFalse(torch.all(t == 1))

  def test_different_dtypes(self):
    inputs_str = "([T([2], torch.int32), T([2], torch.bool)], {})"
    args, _ = deserialize_args(inputs_str, device="cpu")

    self.assertEqual(args[0].dtype, torch.int32)
    self.assertEqual(args[1].dtype, torch.bool)

  def test_invalid_input_not_expression(self):
    with self.assertRaises(ValueError):
      deserialize_args("import os", device="cpu")

  def test_invalid_input_not_tuple(self):
    with self.assertRaises(ValueError):
      deserialize_args("[1, 2]", device="cpu")

  def test_invalid_input_wrong_tuple_len(self):
    with self.assertRaises(ValueError):
      deserialize_args("([1], {}, 3)", device="cpu")

  def test_invalid_t_args_count(self):
    with self.assertRaises(ValueError):
      deserialize_args("([T([2])], {})", device="cpu")

  def test_invalid_t_shape_not_list(self):
    with self.assertRaises(ValueError):
      deserialize_args("([T(2, torch.float32)], {})", device="cpu")

  def test_invalid_t_shape_elements_not_int(self):
    with self.assertRaises(ValueError):
      deserialize_args("([T(['2'], torch.float32)], {})", device="cpu")

  def test_invalid_t_dtype_not_attribute(self):
    with self.assertRaises(ValueError):
      deserialize_args("([T([2], 'float32')], {})", device="cpu")

  def test_invalid_t_dtype_not_torch(self):
    with self.assertRaises(ValueError):
      deserialize_args("([T([2], np.float32)], {})", device="cpu")

  def test_invalid_t_dtype_unknown(self):
    with self.assertRaises(ValueError):
      deserialize_args("([T([2], torch.invalid_dtype)], {})", device="cpu")

  def test_unsupported_function(self):
    with self.assertRaises(ValueError):
      deserialize_args("([some_func()], {})", device="cpu")

  def test_unsupported_ast_node(self):
    with self.assertRaises(ValueError):
      deserialize_args("({1, 2}, {})", device="cpu")


if __name__ == "__main__":
  absltest.main()
