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

"""Ensure we error out at torch.compile() when "dynamic" is applyed"""

import os

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import utils

def simple(x):
  a = 0.3 * x
  return a


class EnsureNoDynamicTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["TORCH_LOGS"] = "+dynamo"
    torch.compiler.reset()

  def test_dynamic_none_and_not_size_change(self):
    # As long as it runs without raising error, we are good.
    input_1 = torch.tensor(
        [0.1, 0.2], device=torch.accelerator.current_accelerator()
    )
    compiled = torch.compile(simple, backend="tpu")
    compiled(input_1).to("cpu")

  def test_dynamic_true_not_supported(self):
    input_1 = torch.tensor(
        [0.1, 0.2], device=torch.accelerator.current_accelerator()
    )

    compiled = torch.compile(simple, dynamic=True, backend="tpu")
    with self.assertRaises(Exception) as err:
      compiled(input_1).to("cpu")
    self.assertIn("torch.compile(..., dynamic=False, ...)", str(err.exception))

  def test_dynamic_none_and_size_change_recompile_not_supported(self):
    input_1 = torch.tensor(
        [0.1, 0.2], device=torch.accelerator.current_accelerator()
    )
    input_2 = torch.tensor(
        [0.1, 0.2, 0.3, 0.4], device=torch.accelerator.current_accelerator()
    )

    compiled = torch.compile(simple, backend="tpu")
    compiled(input_1).to("cpu")
    with self.assertRaises(Exception) as err:
      # Shape change and recompile
      compiled(input_2).to("cpu")
    self.assertIn("torch.compile(..., dynamic=False, ...)", str(err.exception))

  def test_dynamic_false_and_size_change_recompile_supported(self):
    input_1 = torch.tensor([0.1, 0.2])
    input_1_tpu = input_1.to(torch.accelerator.current_accelerator())

    input_2 = torch.tensor([0.1, 0.2, 0.3, 0.4])
    input_2_tpu = input_2.to(torch.accelerator.current_accelerator())

    compiled = torch.compile(simple, dynamic=False, backend="tpu")
    utils.assert_close(compiled(input_1_tpu).to("cpu"), simple(input_1))

    ## Shape change and recompile
    utils.assert_close(compiled(input_2_tpu).to("cpu"), simple(input_2))


if __name__ == "__main__":
  absltest.main()
