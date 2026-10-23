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

"""Ensure error is raised at torch.compile() when dynamic=True is applied."""

import os

from absl.testing import absltest
import torch
from torch_tpu._internal import testing as tt_testing
from tests import seed_test_utils


def simple(x):
  return 0.3 * x


class EnsureNoDynamicTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    tt_testing.reset_eager_state()
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["TORCH_LOGS"] = "+dynamo"
    torch.compiler.reset()

  def test_dynamic_true_not_supported(self):
    input_1 = torch.tensor([0.1, 0.2], device=torch.device("tpu"))

    compiled = torch.compile(simple, dynamic=True, backend="tpu")
    with self.assertRaises(Exception) as err:
      compiled(input_1).to("cpu")
    self.assertIn("dynamic=True", str(err.exception))


if __name__ == "__main__":
  absltest.main()
