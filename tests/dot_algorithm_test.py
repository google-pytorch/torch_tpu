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
from absl.testing import parameterized
import torch
import torch_tpu._internal.precision as p
from torch_tpu._internal.utils import utils

dot_algorithm = p.dot_algorithm
p_impl = p.precision_impl


class DotAlgorithmTest(parameterized.TestCase):

  def test_dot_algorithm_exports_to_torch_module(self):
    self.assertTrue(torch.tpu.dot_algorithm)

  def test_context_manager(self):
    config = p_impl._get_dot_algorithm()
    self.assertIsNone(config)

    with dot_algorithm(
        accumulation_type=torch.float32, num_primitive_operations=3
    ):
      config = p_impl._get_dot_algorithm()
      self.assertIsNotNone(config)
      self.assertEqual(config.accumulation_type, torch.float32)
      self.assertEqual(config.num_primitive_operations, 3)

    config = p_impl._get_dot_algorithm()
    self.assertIsNone(config)

  @parameterized.parameters(
      (torch.dot,),
      (torch.vdot,),
      (torch.matmul,),
  )
  def test_vector_ops_in_shlo(self, op):
    device = torch.device("tpu")
    a = torch.randn(10, device=device, dtype=torch.bfloat16)
    b = torch.randn(10, device=device, dtype=torch.bfloat16)

    with dot_algorithm(
        accumulation_type=torch.float32, num_primitive_operations=3
    ):
      shlo_str = utils.format_model(op, a, b, shlo=True)
      self.assertIn("algorithm = <", shlo_str)
      self.assertIn("lhs_precision_type = bf16", shlo_str)
      self.assertIn("rhs_precision_type = bf16", shlo_str)
      self.assertIn("accumulation_type = f32", shlo_str)
      self.assertIn("num_primitive_operations = 3", shlo_str)

      self.assertEqual(op(a, b).shape, ())

  @parameterized.parameters(
      (torch.mm,),
      (torch.matmul,),
      (lambda a, b: a @ b,),
  )
  def test_matrix_ops_in_shlo(self, op):
    device = torch.device("tpu")
    a = torch.randn(10, 10, device=device, dtype=torch.bfloat16)
    b = torch.randn(10, 10, device=device, dtype=torch.bfloat16)

    with dot_algorithm(
        accumulation_type=torch.float32, num_primitive_operations=3
    ):
      shlo_str = utils.format_model(op, a, b, shlo=True)
      self.assertIn("algorithm = <", shlo_str)
      self.assertIn("lhs_precision_type = bf16", shlo_str)
      self.assertIn("rhs_precision_type = bf16", shlo_str)
      self.assertIn("accumulation_type = f32", shlo_str)
      self.assertIn("num_primitive_operations = 3", shlo_str)


if __name__ == "__main__":
  absltest.main()
