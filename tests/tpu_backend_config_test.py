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

"""Tests for torch.backends.tpu.allow_excess_precision.

CRITICAL REQUIREMENT:
Since torch.backends.tpu.allow_excess_precision is stateful and can only be set
ONCE per process, each test method in this class must run in a completely
isolated process.

When executing these tests, the build target must be sharded such that
`shard_count` matches the number of test methods (currently 4), guaranteeing
exactly one test method executes per sharded process.
"""

import os
from typing import Final

os.environ["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = "disabled"

# pylint: disable=g-import-not-at-top
from absl.testing import absltest
import torch
# pylint: enable=g-import-not-at-top


def _run_compiled_func_tensor_square_bf16_then_mul_f32() -> float:
  device = torch.device("tpu")
  eps = torch.finfo(torch.bfloat16).eps
  x_element = 1.0 + eps  # 1.0000_001 in binary
  x = torch.tensor(
      [[x_element, 0.0], [0.0, x_element]],
      device=device,
      dtype=torch.bfloat16,
  )
  one = torch.tensor([1.0], dtype=torch.float32, device=device)
  del x_element

  def fn(a, b):
    return torch.mm(a, a) * b

  fn_compiled = torch.compile(fn, backend="tpu")
  output = fn_compiled(x, one)
  return output[0, 0].cpu().item()


# pytype: disable=module-attr
class TpuBackendConfigTest(absltest.TestCase):
  """Tests for torch.backends.tpu.allow_excess_precision under strict set-once constraints."""

  def test_default_value_is_false(self):
    # TODO: b/502610173 - Set to False when XLA_FLAGS is not set.
    self.assertTrue(torch.backends.tpu.allow_excess_precision)

  def test_error_when_set_twice(self):
    # Arrange
    torch.backends.tpu.allow_excess_precision = False

    # ACT
    expected_msg: Final[str] = (
        "torch.backends.tpu.allow_excess_precision can only be set once."
    )
    with self.assertRaisesRegex(RuntimeError, expected_msg):
      torch.backends.tpu.allow_excess_precision = False

    with self.assertRaisesRegex(RuntimeError, expected_msg):
      torch.backends.tpu.allow_excess_precision = True

  def test_functional_allow_excess_precision_true(self):
    val = _run_compiled_func_tensor_square_bf16_then_mul_f32()
    self.assertAlmostEqual(val, 1.01568603515625, delta=1e-6)

  def test_functional_allow_excess_precision_false(self):
    torch.backends.tpu.allow_excess_precision = False
    val = _run_compiled_func_tensor_square_bf16_then_mul_f32()
    self.assertAlmostEqual(val, 1.015625, delta=1e-6)


# pytype: enable=module-attr


if __name__ == "__main__":
  absltest.main()
