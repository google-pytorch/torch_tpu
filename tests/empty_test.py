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

"""Tests that torch.empty() materializes correctly.

We treat torch.empty() as a special case which is not materialized even in
defer_never mode unless explicitly requested.
"""

from absl.testing import absltest
import torch
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync
from tests import seed_test_utils


def is_materialized_or_materializing(tensor: torch.Tensor) -> bool:
  return sync.is_materialized(tensor) or sync.is_materializing(tensor)


class EmptyTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.blocking_mode = (
        execution_mode.EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING
    )

  def test_explicit_materialization(self):
    """Tests that empty tensors are materialized when explicitly requested."""
    with execution_mode.set_eager_mode(self.blocking_mode):
      empty = torch.empty(128, device="tpu")
    self.assertFalse(is_materialized_or_materializing(empty))
    sync.synchronize(empty)
    self.assertTrue(is_materialized_or_materializing(empty))

  def test_not_implicitly_materialized(self):
    """Tests that empty tensors are not materialized unless explicitly requested."""
    with execution_mode.set_eager_mode(self.blocking_mode):
      empty = torch.empty(128, device="tpu")
      empty_plus_one = empty + 1
    self.assertFalse(is_materialized_or_materializing(empty))
    self.assertTrue(is_materialized_or_materializing(empty_plus_one))

  def test_empty_view_not_implicitly_materialized(self):
    """Tests that transitive views of empty tensors are not materialized."""
    with execution_mode.set_eager_mode(self.blocking_mode):
      empty = torch.empty(128, device="tpu")
      empty_view = empty.view(2, 64)
    self.assertFalse(is_materialized_or_materializing(empty))
    self.assertFalse(is_materialized_or_materializing(empty_view))

    with execution_mode.set_eager_mode(self.blocking_mode):
      empty_view_plus_one = empty_view + 1
    self.assertFalse(is_materialized_or_materializing(empty))
    self.assertFalse(is_materialized_or_materializing(empty_view))
    self.assertTrue(is_materialized_or_materializing(empty_view_plus_one))

  def test_inplace_has_single_result(self):
    """Tests that inplace ops on empty tensors have a single result.

    This is to test that the empty tensor is not materialized.
    """
    with execution_mode.set_eager_mode(self.blocking_mode):
      empty = torch.empty(128, device="tpu")
      empty[0:64] = 1
    self.assertFalse(is_materialized_or_materializing(empty))
    mlir_str = sync.computation_mlir([empty])
    # Check it returns only a single result, i.e. not the empty tensor.
    self.assertRegex(mlir_str, "return %[a-zA-Z0-9_]+ : tensor<128xf32>")


if __name__ == "__main__":
  absltest.main()
